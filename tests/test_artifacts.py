from __future__ import annotations

from pathlib import Path

import pytest

import dopa_api.artifacts as artifact_module
from dopa_api.artifacts import ArtifactExpired, ArtifactNotFound, ArtifactStore


def _registered(store: ArtifactStore, owner: str = "user-1"):
    reservation = store.reserve()
    reservation.animation_path.write_bytes(b"mp4")
    return store.register(
        reservation,
        owner_subject=owner,
        duration_seconds=12.0,
    )


def test_artifact_is_limited_to_its_owner(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, ttl_seconds=60)
    record = _registered(store)

    assert store.get(record.artifact_id, "user-1").animation_path.read_bytes() == b"mp4"
    with pytest.raises(ArtifactNotFound):
        store.get(record.artifact_id, "user-2")


def test_artifact_expires_and_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_000.0
    monkeypatch.setattr(artifact_module.time, "time", lambda: now)
    store = ArtifactStore(tmp_path, ttl_seconds=60)
    record = _registered(store)
    monkeypatch.setattr(artifact_module.time, "time", lambda: now + 61)

    with pytest.raises(ArtifactExpired):
        store.get(record.artifact_id, "user-1")
    assert not record.animation_path.parent.exists()


def test_cleanup_removes_expired_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_000.0
    monkeypatch.setattr(artifact_module.time, "time", lambda: now)
    store = ArtifactStore(tmp_path, ttl_seconds=60)
    record = _registered(store)
    monkeypatch.setattr(artifact_module.time, "time", lambda: now + 61)

    store.cleanup_expired()

    assert not record.animation_path.parent.exists()


@pytest.mark.parametrize("artifact_id", ["../secret", "not valid", "a" * 80])
def test_rejects_unsafe_artifact_ids(tmp_path: Path, artifact_id: str) -> None:
    store = ArtifactStore(tmp_path, ttl_seconds=60)

    with pytest.raises(ArtifactNotFound):
        store.get(artifact_id, "user-1")
