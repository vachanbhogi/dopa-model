"""Short-lived, user-owned brain-animation artifacts."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,64}$")


class ArtifactNotFound(Exception):
    """Raised when an artifact is absent or belongs to a different user."""


class ArtifactExpired(Exception):
    """Raised after an expired artifact has been removed."""


@dataclass(frozen=True)
class ArtifactReservation:
    artifact_id: str
    directory: Path
    animation_path: Path


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    owner_subject: str
    animation_path: Path
    expires_at: datetime
    duration_seconds: float


class ArtifactStore:
    """Filesystem store with ownership checks and bounded retention."""

    def __init__(self, root: str | Path, ttl_seconds: int = 3600) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.root = Path(root).resolve()
        self.ttl_seconds = ttl_seconds

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.cleanup_expired()

    def reserve(self) -> ArtifactReservation:
        self.initialize()
        artifact_id = secrets.token_urlsafe(32)
        directory = self.root / artifact_id
        directory.mkdir(mode=0o700)
        return ArtifactReservation(
            artifact_id=artifact_id,
            directory=directory,
            animation_path=directory / "brain-response.mp4",
        )

    def register(
        self,
        reservation: ArtifactReservation,
        *,
        owner_subject: str,
        duration_seconds: float,
    ) -> ArtifactRecord:
        if not reservation.animation_path.is_file():
            raise FileNotFoundError("Brain animation was not generated.")
        os.chmod(reservation.animation_path, 0o600)
        expires_timestamp = time.time() + self.ttl_seconds
        metadata = {
            "artifact_id": reservation.artifact_id,
            "owner_subject": owner_subject,
            "expires_at": expires_timestamp,
            "duration_seconds": duration_seconds,
        }
        metadata_path = reservation.directory / "metadata.json"
        temporary_path = reservation.directory / "metadata.json.tmp"
        temporary_path.write_text(json.dumps(metadata), encoding="utf-8")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, metadata_path)
        return ArtifactRecord(
            artifact_id=reservation.artifact_id,
            owner_subject=owner_subject,
            animation_path=reservation.animation_path,
            expires_at=datetime.fromtimestamp(expires_timestamp, tz=UTC),
            duration_seconds=duration_seconds,
        )

    def discard(self, reservation: ArtifactReservation) -> None:
        shutil.rmtree(reservation.directory, ignore_errors=True)

    def get(self, artifact_id: str, owner_subject: str) -> ArtifactRecord:
        directory = self._artifact_directory(artifact_id)
        metadata_path = directory / "metadata.json"
        animation_path = directory / "brain-response.mp4"
        if not metadata_path.is_file() or not animation_path.is_file():
            raise ArtifactNotFound
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expires_timestamp = float(metadata["expires_at"])
            duration_seconds = float(metadata["duration_seconds"])
            stored_owner = str(metadata["owner_subject"])
            stored_id = str(metadata["artifact_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ArtifactNotFound from error
        if stored_id != artifact_id or not secrets.compare_digest(
            stored_owner, owner_subject
        ):
            raise ArtifactNotFound
        if expires_timestamp <= time.time():
            shutil.rmtree(directory, ignore_errors=True)
            raise ArtifactExpired
        return ArtifactRecord(
            artifact_id=artifact_id,
            owner_subject=stored_owner,
            animation_path=animation_path,
            expires_at=datetime.fromtimestamp(expires_timestamp, tz=UTC),
            duration_seconds=duration_seconds,
        )

    def cleanup_expired(self) -> None:
        if not self.root.is_dir():
            return
        now = time.time()
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            metadata_path = directory / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expired = float(metadata["expires_at"]) <= now
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                expired = now - directory.stat().st_mtime > self.ttl_seconds
            if expired:
                shutil.rmtree(directory, ignore_errors=True)

    def _artifact_directory(self, artifact_id: str) -> Path:
        if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            raise ArtifactNotFound
        directory = (self.root / artifact_id).resolve()
        if directory.parent != self.root:
            raise ArtifactNotFound
        return directory
