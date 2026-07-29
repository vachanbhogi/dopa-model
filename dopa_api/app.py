"""FastAPI application for authenticated Dopa video-ad scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from dopa_api.artifacts import (
    ArtifactExpired,
    ArtifactNotFound,
    ArtifactStore,
)
from dopa_api.auth import (
    AuthenticatedUser,
    AuthenticationError,
    SupabaseJwtVerifier,
)
from dopa_api.scoring import BrainRegionResponse, ScoringResult, VideoAdScorer
from dopa_api.visualization import BrainAnimationRenderer

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_CHUNK_BYTES = 1024 * 1024
HEMODYNAMIC_LAG_SECONDS = 5.0
ALLOWED_CONTENT_TYPES = {
    "application/octet-stream",
    "video/mp4",
    "video/quicktime",
}
bearer_scheme = HTTPBearer(auto_error=False)


class Scorer(Protocol):
    def load(self) -> None: ...

    def score(self, video_path: str | Path) -> ScoringResult: ...


class Renderer(Protocol):
    def render(self, predictions: Any, output_path: str | Path) -> None: ...


class TokenVerifier(Protocol):
    def verify(self, token: str) -> AuthenticatedUser: ...


class HealthResponse(BaseModel):
    status: Literal["ready"]


class BrainRegionResponseModel(BaseModel):
    region_id: str
    name: str
    hemisphere: Literal["left", "right"]
    relative_response: float = Field(ge=0.0, le=100.0)
    peak_second: float = Field(ge=0.0)
    description: str


class BrainResponseModel(BaseModel):
    status: Literal["ready", "unavailable"]
    animation_path: str | None
    expires_at: datetime | None
    duration_seconds: float = Field(gt=0.0)
    hemodynamic_lag_seconds: float = Field(ge=0.0)
    top_regions: list[BrainRegionResponseModel]


class ScoreResponse(BaseModel):
    metric: Literal["predicted_average_ctr"]
    score_percent: float = Field(ge=0.0, le=100.0)
    raw_mean_ictr: float = Field(ge=0.0)
    processing_seconds: float = Field(ge=0.0)
    model_load_seconds: float = Field(ge=0.0)
    peak_vram_mib: float = Field(ge=0.0)
    model_version: str
    brain_response: BrainResponseModel


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _allowed_origins() -> list[str]:
    raw = os.environ.get(
        "DOPA_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [
        origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()
    ]
    if not origins or any(origin == "*" for origin in origins):
        raise RuntimeError("DOPA_ALLOWED_ORIGINS must contain explicit origins")
    return origins


MODEL_PATH = Path(
    os.environ.get(
        "DOPA_MODEL_PATH",
        REPO_ROOT
        / "benchmark"
        / "saved_models"
        / "mean_ictr_brain_video_ensemble.joblib",
    )
)
CACHE_DIR = Path(os.environ.get("DOPA_CACHE_DIR", REPO_ROOT / "cache" / "tribev2"))
UPLOAD_DIR = (
    Path(os.environ.get("DOPA_UPLOAD_DIR", tempfile.gettempdir())) / "dopa-uploads"
)
RESULT_DIR = (
    Path(os.environ.get("DOPA_RESULT_DIR", tempfile.gettempdir())) / "dopa-results"
)
MAX_UPLOAD_BYTES = _positive_int_env("DOPA_MAX_UPLOAD_MIB", 250) * 1024 * 1024
MAX_VIDEO_SECONDS = _positive_float_env("DOPA_MAX_VIDEO_SECONDS", 60.0)
RESULT_TTL_SECONDS = _positive_int_env("DOPA_RESULT_TTL_SECONDS", 3600)
MODEL_VERSION = "tribev2-f894e783-video8+vjepa2-875c192b+mean-ictr-video-ensemble-v1"


def _probe_video(path: Path, max_video_seconds: float) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is unavailable")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    try:
        payload = json.loads(result.stdout)
        format_data = payload["format"]
        format_names = str(format_data["format_name"]).split(",")
        duration_seconds = float(format_data["duration"])
        has_video = any(
            stream.get("codec_type") == "video" for stream in payload["streams"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The upload is not a readable MP4 or QuickTime video.",
        ) from error
    if (
        result.returncode != 0
        or not has_video
        or not {"mov", "mp4"}.intersection(format_names)
        or duration_seconds <= 0
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The upload is not a readable MP4 or QuickTime video.",
        )
    if duration_seconds > max_video_seconds:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Upload an ad that is {max_video_seconds:g} seconds or shorter.",
        )
    return duration_seconds


def _region_model(region: BrainRegionResponse) -> BrainRegionResponseModel:
    return BrainRegionResponseModel(
        region_id=region.region_id,
        name=region.name,
        hemisphere=region.hemisphere,
        relative_response=region.relative_response,
        peak_second=region.peak_second,
        description=region.description,
    )


def create_app(
    *,
    scorer_instance: Scorer,
    renderer: Renderer,
    artifact_store: ArtifactStore,
    token_verifier: TokenVerifier,
    upload_dir: Path = UPLOAD_DIR,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_video_seconds: float = MAX_VIDEO_SECONDS,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    analysis_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(upload_dir, 0o700)
        artifact_store.initialize()
        await asyncio.to_thread(scorer_instance.load)
        yield

    application = FastAPI(
        title="Dopa Ad Score API",
        version="1.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or _allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Retry-After"],
        max_age=600,
    )

    def require_user(
        credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    ) -> AuthenticatedUser:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sign in to score an ad.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            return token_verifier.verify(credentials.credentials)
        except AuthenticationError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Your session is invalid or expired.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

    @application.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ready")

    @application.get("/v1/results/{artifact_id}/brain.mp4")
    async def brain_animation(
        artifact_id: str,
        user: AuthenticatedUser = Security(require_user),
    ) -> FileResponse:
        try:
            record = artifact_store.get(artifact_id, user.subject)
        except ArtifactExpired as error:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="This cortical response has expired. Score the ad again.",
            ) from error
        except ArtifactNotFound as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cortical response not found.",
            ) from error
        return FileResponse(
            record.animation_path,
            media_type="video/mp4",
            filename="dopa-cortical-response.mp4",
            content_disposition_type="inline",
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.post("/v1/score", response_model=ScoreResponse)
    async def score_ad(
        file: Annotated[
            UploadFile,
            File(description="MP4 or QuickTime ad video"),
        ],
        user: AuthenticatedUser = Security(require_user),
    ) -> ScoreResponse:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload an MP4 or QuickTime video.",
            )
        if analysis_lock.locked():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The GPU is analyzing another ad. Try again shortly.",
                headers={"Retry-After": "30"},
            )

        temporary_path: Path | None = None
        reservation = None
        try:
            suffix = (
                ".mov"
                if file.filename and Path(file.filename).suffix.lower() == ".mov"
                else ".mp4"
            )
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="ad-",
                suffix=suffix,
                dir=upload_dir,
                delete=False,
            ) as handle:
                os.chmod(handle.name, 0o600)
                temporary_path = Path(handle.name)
                total = 0
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    total += len(chunk)
                    if total > max_upload_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            detail="The uploaded video is too large.",
                        )
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded video is empty.",
                )

            duration_seconds = await asyncio.to_thread(
                _probe_video, temporary_path, max_video_seconds
            )
            try:
                await asyncio.wait_for(analysis_lock.acquire(), timeout=0.05)
            except TimeoutError as error:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="The GPU is analyzing another ad. Try again shortly.",
                    headers={"Retry-After": "30"},
                ) from error

            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(scorer_instance.score, temporary_path)
                reservation = artifact_store.reserve()
                animation_path: str | None = None
                expires_at: datetime | None = None
                brain_status: Literal["ready", "unavailable"] = "ready"
                try:
                    await asyncio.to_thread(
                        renderer.render,
                        result.predictions,
                        reservation.animation_path,
                    )
                    record = artifact_store.register(
                        reservation,
                        owner_subject=user.subject,
                        duration_seconds=duration_seconds,
                    )
                    animation_path = f"/v1/results/{record.artifact_id}/brain.mp4"
                    expires_at = record.expires_at
                except Exception:
                    LOGGER.exception("Cortical response rendering failed")
                    artifact_store.discard(reservation)
                    reservation = None
                    brain_status = "unavailable"
            finally:
                analysis_lock.release()

            return ScoreResponse(
                metric="predicted_average_ctr",
                score_percent=result.percentage,
                raw_mean_ictr=result.raw_mean_ictr,
                processing_seconds=time.perf_counter() - started,
                model_load_seconds=result.model_load_seconds,
                peak_vram_mib=result.peak_vram_mib,
                model_version=MODEL_VERSION,
                brain_response=BrainResponseModel(
                    status=brain_status,
                    animation_path=animation_path,
                    expires_at=expires_at,
                    duration_seconds=duration_seconds,
                    hemodynamic_lag_seconds=HEMODYNAMIC_LAG_SECONDS,
                    top_regions=[
                        _region_model(region) for region in result.top_regions
                    ],
                ),
            )
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("Ad scoring failed")
            if reservation is not None:
                artifact_store.discard(reservation)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ad scoring failed.",
            ) from error
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return application


scorer = VideoAdScorer(
    model_path=MODEL_PATH,
    cache_dir=CACHE_DIR,
    video_frames=8,
)
app = create_app(
    scorer_instance=scorer,
    renderer=BrainAnimationRenderer(),
    artifact_store=ArtifactStore(RESULT_DIR, ttl_seconds=RESULT_TTL_SECONDS),
    token_verifier=SupabaseJwtVerifier(),
)
