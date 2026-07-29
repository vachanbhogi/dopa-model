"""Compact browser-ready cortical surface data from real TRIBE predictions."""

from __future__ import annotations

import base64
import gzip
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPECTED_VERTICES = 20_484
HEMISPHERE_VERTICES = EXPECTED_VERTICES // 2
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class SurfaceMesh:
    """One hemisphere of the fsaverage5 pial surface."""

    coordinates: np.ndarray
    faces: np.ndarray


MeshLoader = Callable[[], tuple[SurfaceMesh, SurfaceMesh]]


def normalize_response_magnitude(predictions: np.ndarray) -> np.ndarray:
    """Scale response magnitudes to [0, 1] without failing on flat output."""
    magnitudes = np.abs(np.asarray(predictions, dtype=np.float32))
    ceiling = float(np.percentile(magnitudes, 99))
    if np.isfinite(ceiling) and ceiling <= np.finfo(np.float32).eps:
        ceiling = float(np.max(magnitudes))
    if not np.isfinite(ceiling) or ceiling <= np.finfo(np.float32).eps:
        return np.zeros_like(magnitudes)
    return np.clip(magnitudes / ceiling, 0.0, 1.0)


def _load_fsaverage5_pial() -> tuple[SurfaceMesh, SurfaceMesh]:
    from nilearn.datasets import load_fsaverage

    fsaverage = load_fsaverage("fsaverage5")
    meshes = tuple(
        SurfaceMesh(
            coordinates=np.asarray(
                fsaverage.pial.parts[hemisphere].coordinates,
                dtype=np.float32,
            ),
            faces=np.asarray(
                fsaverage.pial.parts[hemisphere].faces,
                dtype=np.uint32,
            ),
        )
        for hemisphere in ("left", "right")
    )
    return meshes[0], meshes[1]


def _encode_array(values: np.ndarray, dtype: str) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


class BrainDataRenderer:
    """Serialize the pial mesh and time-varying cortical response for WebGL."""

    def __init__(self, mesh_loader: MeshLoader | None = None) -> None:
        self.mesh_loader = mesh_loader or _load_fsaverage5_pial

    def render(self, predictions: np.ndarray, output_path: str | Path) -> None:
        prediction_array = np.asarray(predictions, dtype=np.float32)
        if (
            prediction_array.ndim != 2
            or prediction_array.shape[1] != EXPECTED_VERTICES
        ):
            raise ValueError(
                f"Unexpected cortical prediction shape: {prediction_array.shape}"
            )
        if len(prediction_array) == 0 or not np.isfinite(prediction_array).all():
            raise ValueError("Cortical predictions must be finite and non-empty.")

        left_mesh, right_mesh = self.mesh_loader()
        meshes = (left_mesh, right_mesh)
        for mesh in meshes:
            if mesh.coordinates.shape != (HEMISPHERE_VERTICES, 3):
                raise ValueError(
                    f"Unexpected cortical mesh shape: {mesh.coordinates.shape}"
                )
            if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
                raise ValueError(f"Unexpected cortical face shape: {mesh.faces.shape}")
            if not np.isfinite(mesh.coordinates).all():
                raise ValueError("Cortical mesh contains non-finite coordinates.")

        quantized = np.rint(
            normalize_response_magnitude(prediction_array) * 255.0
        ).astype(np.uint8)
        payload = {
            "version": ARTIFACT_VERSION,
            "frame_count": int(len(prediction_array)),
            "frame_interval_seconds": 1.0,
            "response_encoding": "uint8-absolute-p99",
            "hemispheres": [
                {
                    "hemisphere": hemisphere,
                    "vertex_count": HEMISPHERE_VERTICES,
                    "positions_f32": _encode_array(mesh.coordinates, "<f4"),
                    "indices_u32": _encode_array(mesh.faces, "<u4"),
                    "responses_u8": _encode_array(
                        quantized[
                            :,
                            index * HEMISPHERE_VERTICES : (index + 1)
                            * HEMISPHERE_VERTICES,
                        ],
                        "u1",
                    ),
                }
                for index, (hemisphere, mesh) in enumerate(
                    zip(("left", "right"), meshes, strict=True)
                )
            ],
        }

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        destination.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
        os.chmod(destination, 0o600)
