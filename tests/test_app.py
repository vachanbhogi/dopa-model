from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import dopa_api.app as app_module
from dopa_api.artifacts import ArtifactStore
from dopa_api.auth import AuthenticatedUser, AuthenticationError
from dopa_api.scoring import BrainRegionResponse, ScoringResult


class FakeVerifier:
    def verify(self, token: str) -> AuthenticatedUser:
        if token == "valid":
            return AuthenticatedUser(subject="user-1")
        if token == "other":
            return AuthenticatedUser(subject="user-2")
        raise AuthenticationError


class FakeRenderer:
    def render(self, _predictions: np.ndarray, output_path: str | Path) -> None:
        Path(output_path).write_bytes(gzip.compress(b'{"version":1}', mtime=0))


class FailingRenderer:
    def render(self, _predictions: np.ndarray, _output_path: str | Path) -> None:
        raise RuntimeError("render failed")


class FakeScorer:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.started = threading.Event()

    def load(self) -> None:
        return

    def score(self, _video_path: str | Path) -> ScoringResult:
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=5)
        return ScoringResult(
            percentage=2.75,
            raw_mean_ictr=0.0275,
            predictions=np.zeros((2, 20_484), dtype=np.float32),
            top_regions=(
                BrainRegionResponse(
                    region_id="lh_S_calcarine",
                    name="Sulcus calcarine",
                    hemisphere="left",
                    relative_response=100,
                    peak_second=1,
                    description="Visual response.",
                ),
            ),
            brain_timesteps=2,
            compact_features=900,
            model_load_seconds=1.5,
            inference_seconds=2.0,
            peak_vram_mib=512,
            process_rss_mib=1024,
        )


class SequencedScorer(FakeScorer):
    def __init__(self, calls: int) -> None:
        super().__init__()
        self.call_started = [threading.Event() for _ in range(calls)]
        self.call_released = [threading.Event() for _ in range(calls)]
        self.call_count = 0
        self.call_count_lock = threading.Lock()

    def score(self, video_path: str | Path) -> ScoringResult:
        with self.call_count_lock:
            call_index = self.call_count
            self.call_count += 1
        self.call_started[call_index].set()
        self.call_released[call_index].wait(timeout=5)
        return super().score(video_path)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_probe_video", lambda _path, _maximum: 12.0)


def _client(
    tmp_path: Path,
    scorer: FakeScorer | None = None,
    renderer: FakeRenderer | FailingRenderer | None = None,
) -> TestClient:
    application = app_module.create_app(
        scorer_instance=scorer or FakeScorer(),
        renderer=renderer or FakeRenderer(),
        artifact_store=ArtifactStore(tmp_path / "results", ttl_seconds=60),
        token_verifier=FakeVerifier(),
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=1024,
        max_video_seconds=60,
        allowed_origins=["http://localhost:3000"],
    )
    return TestClient(application)


def _post(client: TestClient, token: str = "valid"):
    return client.post(
        "/v1/score",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("ad.mp4", b"video", "video/mp4")},
    )


def _enqueue(client: TestClient, token: str = "valid"):
    return client.post(
        "/v1/score/jobs",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("ad.mp4", b"video", "video/mp4")},
    )


def _job(
    client: TestClient,
    job_id: str,
    token: str = "valid",
):
    return client.get(
        f"/v1/score/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )


def _wait_for_job_status(
    client: TestClient,
    job_id: str,
    expected: set[str],
    timeout: float = 2,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = _job(client, job_id)
        if response.json()["status"] in expected:
            return response
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not reach {expected}")


def test_scores_and_serves_owned_brain_model(tmp_path: Path, probe: None) -> None:
    with _client(tmp_path) as client:
        response = _post(client)
        assert response.status_code == 200
        payload = response.json()
        assert payload["score_percent"] == 2.75
        assert payload["brain_response"]["status"] == "ready"
        assert (
            payload["brain_response"]["top_regions"][0]["region_id"] == "lh_S_calcarine"
        )

        model_path = payload["brain_response"]["model_path"]
        model = client.get(
            model_path,
            headers={"Authorization": "Bearer valid"},
        )
        assert model.status_code == 200
        assert model.json() == {"version": 1}
        assert model.headers["content-encoding"] == "gzip"

        hidden = client.get(
            model_path,
            headers={"Authorization": "Bearer other"},
        )
        assert hidden.status_code == 404
        assert not list((tmp_path / "uploads").iterdir())


def test_returns_score_when_brain_model_generation_fails(
    tmp_path: Path, probe: None
) -> None:
    with _client(tmp_path, renderer=FailingRenderer()) as client:
        response = _post(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["score_percent"] == 2.75
    assert payload["brain_response"]["status"] == "unavailable"
    assert payload["brain_response"]["model_path"] is None
    assert not list((tmp_path / "results").iterdir())


def test_expired_brain_model_returns_gone(tmp_path: Path, probe: None) -> None:
    with _client(tmp_path) as client:
        response = _post(client)
        model_path = response.json()["brain_response"]["model_path"]
        artifact_id = model_path.split("/")[-2]
        metadata_path = tmp_path / "results" / artifact_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["expires_at"] = 0
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        expired = client.get(
            model_path,
            headers={"Authorization": "Bearer valid"},
        )

    assert expired.status_code == 410
    assert "expired" in expired.json()["detail"]
    assert not metadata_path.parent.exists()


def test_allows_localhost_cors(tmp_path: Path, probe: None) -> None:
    with _client(tmp_path) as client:
        response = client.options(
            "/v1/score",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ("http://localhost:3000")


def test_requires_valid_session(tmp_path: Path, probe: None) -> None:
    with _client(tmp_path) as client:
        missing = client.post(
            "/v1/score",
            files={"file": ("ad.mp4", b"video", "video/mp4")},
        )
        invalid = _post(client, token="invalid")

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_rejects_invalid_type_and_large_upload(tmp_path: Path, probe: None) -> None:
    with _client(tmp_path) as client:
        invalid = client.post(
            "/v1/score",
            headers={"Authorization": "Bearer valid"},
            files={"file": ("ad.webm", b"video", "video/webm")},
        )
    small_limit_app = app_module.create_app(
        scorer_instance=FakeScorer(),
        renderer=FakeRenderer(),
        artifact_store=ArtifactStore(tmp_path / "other-results", ttl_seconds=60),
        token_verifier=FakeVerifier(),
        upload_dir=tmp_path / "other-uploads",
        max_upload_bytes=3,
        max_video_seconds=60,
        allowed_origins=["http://localhost:3000"],
    )
    with TestClient(small_limit_app) as client:
        large = _post(client)

    assert invalid.status_code == 415
    assert large.status_code == 413


def test_rejects_concurrent_analysis(tmp_path: Path, probe: None) -> None:
    gate = threading.Event()
    scorer = FakeScorer(gate=gate)
    with _client(tmp_path, scorer=scorer) as client:
        responses: list[object] = []

        def first_request() -> None:
            responses.append(_post(client))

        thread = threading.Thread(target=first_request)
        thread.start()
        assert scorer.started.wait(timeout=2)
        busy = _post(client)
        gate.set()
        thread.join(timeout=5)

    assert busy.status_code == 429
    assert busy.headers["retry-after"] == "30"
    assert responses and responses[0].status_code == 200


def test_queues_ads_fifo_and_reports_live_position(
    tmp_path: Path, probe: None
) -> None:
    scorer = SequencedScorer(calls=3)
    with _client(tmp_path, scorer=scorer) as client:
        first = _enqueue(client)
        assert first.status_code == 202
        first_id = first.json()["job_id"]
        assert scorer.call_started[0].wait(timeout=2)

        second = _enqueue(client)
        third = _enqueue(client)
        second_id = second.json()["job_id"]
        third_id = third.json()["job_id"]

        assert second.json()["position"] == 1
        assert third.json()["position"] == 2
        assert _job(client, second_id, token="other").status_code == 404

        scorer.call_released[0].set()
        assert scorer.call_started[1].wait(timeout=2)
        assert _wait_for_job_status(
            client, second_id, {"processing"}
        ).json()["position"] is None
        assert _job(client, third_id).json()["position"] == 1

        scorer.call_released[1].set()
        assert scorer.call_started[2].wait(timeout=2)
        scorer.call_released[2].set()

        for job_id in (first_id, second_id, third_id):
            completed = _wait_for_job_status(client, job_id, {"succeeded"})
            payload = completed.json()
            assert payload["result"]["score_percent"] == 2.75
            assert payload["position"] is None
            assert payload["error"] is None

    assert not list((tmp_path / "uploads").iterdir())


def test_probe_enforces_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "ad.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(app_module.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        app_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "video"}],
                    "format": {"duration": "60.1", "format_name": "mov,mp4"},
                }
            ),
        ),
    )

    with pytest.raises(app_module.HTTPException) as caught:
        app_module._probe_video(video, 60)

    assert caught.value.status_code == 422
    assert "60 seconds or shorter" in caught.value.detail
