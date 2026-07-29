from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dopa_api.visualization import (
    _verify_encoded_animation,
    normalize_response_magnitude,
)


def test_normalizes_absolute_cortical_response() -> None:
    predictions = np.array([[-2.0, 0.0, 1.0], [4.0, -1.0, 0.0]])

    normalized = normalize_response_magnitude(predictions)

    assert normalized.shape == predictions.shape
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalized[0, 0] > normalized[0, 2]


def test_flat_cortical_response_normalizes_to_zero() -> None:
    normalized = normalize_response_magnitude(np.zeros((2, 20_484)))

    assert not normalized.any()


def test_rejects_animation_without_video_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    animation_path = tmp_path / "brain.mp4"
    animation_path.write_bytes(b"empty-container")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"streams": [], "format": {"duration": "0.000000"}}),
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="no playable video"):
        _verify_encoded_animation(animation_path, "ffprobe")


def test_accepts_animation_with_positive_video_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    animation_path = tmp_path / "brain.mp4"
    animation_path.write_bytes(b"mp4")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(
            {
                "streams": [{"codec_type": "video"}],
                "format": {"duration": "1.000000"},
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    _verify_encoded_animation(animation_path, "ffprobe")
