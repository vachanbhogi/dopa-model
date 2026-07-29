"""FastAPI application for authenticated Dopa video-ad scoring."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Response,
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
from dopa_api.visualization import BrainDataRenderer

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
    model_path: str | None
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


ScoreJobState = Literal["queued", "processing", "succeeded", "failed"]


class ScoreJobResponse(BaseModel):
    job_id: str
    status: ScoreJobState
    position: int | None = Field(default=None, ge=1)
    queued_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    poll_after_seconds: int = Field(default=2, ge=1, le=10)
    result: ScoreResponse | None = None
    error: str | None = None


@dataclass(slots=True)
class ScoreJob:
    job_id: str
    owner_subject: str
    temporary_path: Path
    duration_seconds: float
    queued_at: datetime
    status: ScoreJobState = "queued"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: ScoreResponse | None = None
    error: str | None = None


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
MAX_SCORE_QUEUE_JOBS = _positive_int_env("DOPA_MAX_SCORE_QUEUE_JOBS", 20)
SCORE_JOB_TTL_SECONDS = _positive_int_env("DOPA_SCORE_JOB_TTL_SECONDS", 3600)
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
    max_score_queue_jobs: int = MAX_SCORE_QUEUE_JOBS,
    score_job_ttl_seconds: int = SCORE_JOB_TTL_SECONDS,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    if max_score_queue_jobs <= 0:
        raise ValueError("max_score_queue_jobs must be positive")
    if score_job_ttl_seconds <= 0:
        raise ValueError("score_job_ttl_seconds must be positive")

    analysis_lock = asyncio.Lock()
    score_jobs: dict[str, ScoreJob] = {}
    pending_job_ids: list[str] = []
    score_queue: asyncio.Queue[str] = asyncio.Queue()
    score_jobs_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(upload_dir, 0o700)
        artifact_store.initialize()
        await asyncio.to_thread(scorer_instance.load)
        worker = asyncio.create_task(score_worker(), name="dopa-score-worker")
        try:
            yield
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
            async with score_jobs_lock:
                unfinished_paths = [
                    job.temporary_path
                    for job in score_jobs.values()
                    if job.status in {"queued", "processing"}
                ]
            for path in unfinished_paths:
                path.unlink(missing_ok=True)

    application = FastAPI(
        title="Dopa Ad Score API",
        version="1.3.0",
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

    async def persist_upload(file: UploadFile) -> tuple[Path, float]:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload an MP4 or QuickTime video.",
            )

        temporary_path: Path | None = None
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
            return temporary_path, duration_seconds
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    async def score_uploaded_video(
        temporary_path: Path,
        *,
        duration_seconds: float,
        owner_subject: str,
    ) -> ScoreResponse:
        reservation = None
        started = time.perf_counter()
        result = await asyncio.to_thread(scorer_instance.score, temporary_path)
        model_path: str | None = None
        expires_at: datetime | None = None
        brain_status: Literal["ready", "unavailable"] = "ready"
        try:
            reservation = artifact_store.reserve()
            await asyncio.to_thread(
                renderer.render,
                result.predictions,
                reservation.model_path,
            )
            record = artifact_store.register(
                reservation,
                owner_subject=owner_subject,
                duration_seconds=duration_seconds,
            )
            model_path = f"/v1/results/{record.artifact_id}/brain.json"
            expires_at = record.expires_at
        except Exception:
            LOGGER.exception("Cortical model artifact generation failed")
            if reservation is not None:
                artifact_store.discard(reservation)
            brain_status = "unavailable"

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
                model_path=model_path,
                expires_at=expires_at,
                duration_seconds=duration_seconds,
                hemodynamic_lag_seconds=HEMODYNAMIC_LAG_SECONDS,
                top_regions=[_region_model(region) for region in result.top_regions],
            ),
        )

    def cleanup_finished_jobs_locked() -> None:
        cutoff = time.time() - score_job_ttl_seconds
        expired_ids = [
            job_id
            for job_id, job in score_jobs.items()
            if job.completed_at is not None
            and job.completed_at.timestamp() <= cutoff
        ]
        for job_id in expired_ids:
            score_jobs.pop(job_id, None)

    def job_response_locked(job: ScoreJob) -> ScoreJobResponse:
        position = (
            pending_job_ids.index(job.job_id) + 1
            if job.status == "queued"
            else None
        )
        return ScoreJobResponse(
            job_id=job.job_id,
            status=job.status,
            position=position,
            queued_at=job.queued_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result=job.result,
            error=job.error,
        )

    async def score_worker() -> None:
        while True:
            job_id = await score_queue.get()
            job: ScoreJob | None = None
            acquired_lock = False
            try:
                async with score_jobs_lock:
                    job = score_jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue

                await analysis_lock.acquire()
                acquired_lock = True
                async with score_jobs_lock:
                    if job.status != "queued":
                        continue
                    pending_job_ids.remove(job_id)
                    job.status = "processing"
                    job.started_at = datetime.now(UTC)

                result = await score_uploaded_video(
                    job.temporary_path,
                    duration_seconds=job.duration_seconds,
                    owner_subject=job.owner_subject,
                )
                async with score_jobs_lock:
                    job.status = "succeeded"
                    job.result = result
                    job.completed_at = datetime.now(UTC)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Queued ad scoring failed")
                if job is not None:
                    async with score_jobs_lock:
                        if job.job_id in pending_job_ids:
                            pending_job_ids.remove(job.job_id)
                        job.status = "failed"
                        job.error = "Ad scoring failed. Please try again."
                        job.completed_at = datetime.now(UTC)
            finally:
                if acquired_lock:
                    analysis_lock.release()
                if job is not None:
                    job.temporary_path.unlink(missing_ok=True)
                score_queue.task_done()

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

    @application.get("/v1/results/{artifact_id}/brain.json")
    async def brain_model(
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
            record.model_path,
            media_type="application/json",
            filename="dopa-cortical-response.json",
            content_disposition_type="inline",
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "Content-Encoding": "gzip",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.post(
        "/v1/score/jobs",
        response_model=ScoreJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def enqueue_score_ad(
        response: Response,
        file: Annotated[
            UploadFile,
            File(description="MP4 or QuickTime ad video"),
        ],
        user: AuthenticatedUser = Security(require_user),
    ) -> ScoreJobResponse:
        async with score_jobs_lock:
            cleanup_finished_jobs_locked()
            active_jobs = sum(
                job.status in {"queued", "processing"}
                for job in score_jobs.values()
            )
            if active_jobs >= max_score_queue_jobs:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The analysis queue is full. Please try again shortly.",
                    headers={"Retry-After": "30"},
                )

        temporary_path: Path | None = None
        try:
            temporary_path, duration_seconds = await persist_upload(file)
        finally:
            await file.close()

        try:
            async with score_jobs_lock:
                cleanup_finished_jobs_locked()
                active_jobs = sum(
                    job.status in {"queued", "processing"}
                    for job in score_jobs.values()
                )
                if active_jobs >= max_score_queue_jobs:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="The analysis queue filled while your ad uploaded. "
                        "Please try again shortly.",
                        headers={"Retry-After": "30"},
                    )

                job_id = secrets.token_urlsafe(24)
                job = ScoreJob(
                    job_id=job_id,
                    owner_subject=user.subject,
                    temporary_path=temporary_path,
                    duration_seconds=duration_seconds,
                    queued_at=datetime.now(UTC),
                )
                score_jobs[job_id] = job
                pending_job_ids.append(job_id)
                score_queue.put_nowait(job_id)
                queued_response = job_response_locked(job)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        return queued_response

    @application.get(
        "/v1/score/jobs/{job_id}",
        response_model=ScoreJobResponse,
    )
    async def score_job(
        job_id: str,
        response: Response,
        user: AuthenticatedUser = Security(require_user),
    ) -> ScoreJobResponse:
        async with score_jobs_lock:
            cleanup_finished_jobs_locked()
            job = score_jobs.get(job_id)
            if job is None or not secrets.compare_digest(
                job.owner_subject, user.subject
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Analysis job not found.",
                )
            current_response = job_response_locked(job)

        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        return current_response

    @application.post("/v1/score", response_model=ScoreResponse)
    async def score_ad(
        file: Annotated[
            UploadFile,
            File(description="MP4 or QuickTime ad video"),
        ],
        user: AuthenticatedUser = Security(require_user),
    ) -> ScoreResponse:
        if analysis_lock.locked():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The GPU is analyzing another ad. Try again shortly.",
                headers={"Retry-After": "30"},
            )

        temporary_path: Path | None = None
        try:
            temporary_path, duration_seconds = await persist_upload(file)
            try:
                await asyncio.wait_for(analysis_lock.acquire(), timeout=0.05)
            except TimeoutError as error:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="The GPU is analyzing another ad. Try again shortly.",
                    headers={"Retry-After": "30"},
                ) from error

            try:
                return await score_uploaded_video(
                    temporary_path,
                    duration_seconds=duration_seconds,
                    owner_subject=user.subject,
                )
            finally:
                analysis_lock.release()
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("Ad scoring failed")
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
    renderer=BrainDataRenderer(),
    artifact_store=ArtifactStore(RESULT_DIR, ttl_seconds=RESULT_TTL_SECONDS),
    token_verifier=SupabaseJwtVerifier(),
)
