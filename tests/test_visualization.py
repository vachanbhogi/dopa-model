from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from dopa_api.visualization import (
    BrainDataRenderer,
    SurfaceMesh,
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


def test_serializes_compact_interactive_brain_data(tmp_path: Path) -> None:
    coordinates = np.arange(10_242 * 3, dtype=np.float32).reshape(10_242, 3)
    faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uint32)
    renderer = BrainDataRenderer(
        mesh_loader=lambda: (
            SurfaceMesh(coordinates=coordinates, faces=faces),
            SurfaceMesh(coordinates=-coordinates, faces=faces),
        )
    )
    predictions = np.zeros((2, 20_484), dtype=np.float32)
    predictions[1, 0] = -4
    predictions[1, 10_242] = 2
    output = tmp_path / "brain-response.json.gz"

    renderer.render(predictions, output)

    payload = json.loads(gzip.decompress(output.read_bytes()))
    assert payload["version"] == 1
    assert payload["frame_count"] == 2
    assert payload["response_encoding"] == "uint8-absolute-p99"
    assert [item["hemisphere"] for item in payload["hemispheres"]] == [
        "left",
        "right",
    ]

    left = payload["hemispheres"][0]
    decoded_positions = np.frombuffer(
        base64.b64decode(left["positions_f32"]),
        dtype="<f4",
    )
    decoded_indices = np.frombuffer(
        base64.b64decode(left["indices_u32"]),
        dtype="<u4",
    )
    decoded_responses = np.frombuffer(
        base64.b64decode(left["responses_u8"]),
        dtype=np.uint8,
    )
    assert decoded_positions.shape == (10_242 * 3,)
    assert decoded_indices.tolist() == faces.ravel().tolist()
    assert decoded_responses.shape == (2 * 10_242,)
    assert decoded_responses[10_242] == 255
    assert output.stat().st_mode & 0o777 == 0o600


def test_rejects_invalid_cortical_shape(tmp_path: Path) -> None:
    renderer = BrainDataRenderer(mesh_loader=lambda: pytest.fail("not called"))

    with pytest.raises(ValueError, match="Unexpected cortical prediction shape"):
        renderer.render(np.zeros((2, 3)), tmp_path / "brain.json.gz")
