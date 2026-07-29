"""Headless rendering of real TRIBE cortical predictions."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def normalize_response_magnitude(predictions: np.ndarray) -> np.ndarray:
    """Scale response magnitudes to [0, 1] without failing on flat output."""
    magnitudes = np.abs(np.asarray(predictions, dtype=np.float32))
    ceiling = float(np.percentile(magnitudes, 99))
    if not np.isfinite(ceiling) or ceiling <= np.finfo(np.float32).eps:
        return np.zeros_like(magnitudes)
    return np.clip(magnitudes / ceiling, 0.0, 1.0)


class BrainAnimationRenderer:
    """Render paired cortical views and encode them as a browser-safe MP4."""

    def __init__(self, output_fps: int = 12) -> None:
        if output_fps <= 0:
            raise ValueError("output_fps must be positive")
        self.output_fps = output_fps

    def render(self, predictions: np.ndarray, output_path: str | Path) -> None:
        if predictions.ndim != 2 or predictions.shape[1] != 20_484:
            raise ValueError(
                f"Unexpected cortical prediction shape: {predictions.shape}"
            )
        if len(predictions) == 0 or not np.isfinite(predictions).all():
            raise ValueError("Cortical predictions must be finite and non-empty.")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is unavailable")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tribev2.plotting import PlotBrainNilearn

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        frames_directory = Path(
            tempfile.mkdtemp(prefix="frames-", dir=destination.parent)
        )
        os.chmod(frames_directory, 0o700)

        normalized = normalize_response_magnitude(predictions)
        plotter = PlotBrainNilearn(mesh="fsaverage5", inflate="half")
        try:
            for index, frame in enumerate(normalized):
                figure, axes = plt.subplots(
                    1,
                    2,
                    figsize=(8, 4.5),
                    facecolor="#08090a",
                    subplot_kw={"projection": "3d"},
                    gridspec_kw={"wspace": -0.12},
                )
                for axis in axes:
                    axis.set_facecolor("#08090a")
                plotter.plot_surf(
                    frame,
                    axes=axes,
                    views=["left", "right"],
                    cmap="fire",
                    vmin=0.55,
                    vmax=1.0,
                    alpha_cmap=(0.0, 0.18),
                )
                figure.text(
                    0.5,
                    0.055,
                    f"Predicted cortical response  ·  {index}s",
                    color="#c9cbd1",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                figure.savefig(
                    frames_directory / f"frame_{index:05d}.png",
                    dpi=160,
                    facecolor=figure.get_facecolor(),
                    bbox_inches="tight",
                    pad_inches=0.08,
                )
                plt.close(figure)
            if len(normalized) == 1:
                shutil.copy2(
                    frames_directory / "frame_00000.png",
                    frames_directory / "frame_00001.png",
                )

            command = [
                ffmpeg,
                "-y",
                "-framerate",
                "1",
                "-i",
                str(frames_directory / "frame_%05d.png"),
                "-vf",
                (
                    f"minterpolate=fps={self.output_fps}:mi_mode=blend,"
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
                ),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(destination),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=300,
            )
            if result.returncode != 0 or not destination.is_file():
                message = result.stderr.strip().splitlines()[-1:] or ["unknown error"]
                raise RuntimeError(f"Brain animation encoding failed: {message[0]}")
            os.chmod(destination, 0o600)
        finally:
            shutil.rmtree(frames_directory, ignore_errors=True)
