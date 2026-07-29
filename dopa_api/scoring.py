"""Video-only TRIBE v2 inference followed by average-iCTR prediction."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import psutil

TRIBE_MODEL_ID = "facebook/tribev2"
TRIBE_MODEL_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
VJEPA_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
VJEPA_MODEL_REVISION = "875c192b7b704b87d1e1d99345769632dd5f739a"
EXPECTED_VERTICES = 20_484
EXPECTED_VIDEO_FEATURES = 900
TOP_REGION_COUNT = 5


@dataclass(frozen=True)
class ParcelDefinition:
    """One Destrieux atlas parcel on the fsaverage5 cortical surface."""

    feature_prefix: str
    label: str
    hemisphere: Literal["left", "right"]
    vertex_slice: slice
    mask: np.ndarray


@dataclass(frozen=True)
class BrainRegionResponse:
    """A clip-relative summary of one predicted cortical parcel response."""

    region_id: str
    name: str
    hemisphere: Literal["left", "right"]
    relative_response: float
    peak_second: float
    description: str


@dataclass(frozen=True)
class ScoringResult:
    """Real model output retained long enough to render the cortical response."""

    percentage: float
    raw_mean_ictr: float
    predictions: np.ndarray
    top_regions: tuple[BrainRegionResponse, ...]
    brain_timesteps: int
    compact_features: int
    model_load_seconds: float
    inference_seconds: float
    peak_vram_mib: float
    process_rss_mib: float


def _clean_feature_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "unknown"


def _display_region_name(label: str) -> str:
    replacements = {
        "G": "Gyrus",
        "S": "Sulcus",
        "Lat Fis": "Lateral fissure",
    }
    words = label.replace("-", " ").replace("_", " ").split()
    if words and words[0] in replacements:
        words[0] = replacements[words[0]]
    display = " ".join(words).strip()
    return display[:1].upper() + display[1:] if display else "Cortical region"


def _describe_region(label: str) -> str:
    """Return a cautious, broad description without implying causal effect."""
    normalized = label.lower()
    descriptions = (
        (
            ("calcar", "occip", "cuneus", "lingual"),
            "A visual-system parcel often involved in processing shapes, contrast, and motion.",
        ),
        (
            ("temporal", "temp_", "planum", "transvers"),
            "A temporal parcel often involved in combining sound, language, and visual meaning.",
        ),
        (
            ("intrapariet", "pariet", "precune"),
            "A parietal parcel often involved in spatial attention and integrating sensory information.",
        ),
        (
            ("front", "precentral", "orbital"),
            "A frontal parcel often involved in attention, planning, and evaluating information.",
        ),
        (
            ("cingul",),
            "A cingulate parcel often involved in attention, salience, and value-related processing.",
        ),
        (
            ("insula", "insular"),
            "An insular parcel often involved in salience and integrating internal and external signals.",
        ),
        (
            ("postcentral", "central"),
            "A sensorimotor parcel often involved in representing movement or bodily sensation.",
        ),
    )
    for tokens, description in descriptions:
        if any(token in normalized for token in tokens):
            return description
    return (
        "A parcel from the Destrieux cortical atlas; the model predicts a comparatively "
        "strong response here for this clip."
    )


class VideoAdScorer:
    """Load the pinned video models once and score uploaded ad videos."""

    def __init__(
        self,
        model_path: str | Path,
        cache_dir: str | Path,
        video_frames: int = 8,
    ) -> None:
        if video_frames not in (8, 64):
            raise ValueError("video_frames must be 8 or 64")
        self.model_path = Path(model_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.video_frames = video_frames
        self.regressor: Any | None = None
        self.tribe_model: Any | None = None
        self.expected_columns: list[str] = []
        self.parcels: list[ParcelDefinition] = []
        self.model_load_seconds = 0.0

    def load(self) -> None:
        """Load the saved regressor, atlas, TRIBE, and V-JEPA checkpoints."""
        if self.tribe_model is not None:
            return

        import torch
        from huggingface_hub import snapshot_download
        from nilearn.datasets import fetch_atlas_surf_destrieux
        from tribev2.demo_utils import TribeModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {self.model_path}")

        started = time.perf_counter()
        self.regressor = joblib.load(self.model_path)
        columns = getattr(self.regressor.scaler_, "feature_names_in_", None)
        if columns is None:
            raise RuntimeError(
                "The saved regressor does not contain its feature schema"
            )
        self.expected_columns = [str(column) for column in columns]
        if len(self.expected_columns) != EXPECTED_VIDEO_FEATURES:
            raise RuntimeError(
                f"Expected {EXPECTED_VIDEO_FEATURES} video features, "
                f"model requires {len(self.expected_columns)}"
            )
        if any(not column.endswith("_vid") for column in self.expected_columns):
            raise RuntimeError("The saved regressor is not a video-only checkpoint")

        atlas = fetch_atlas_surf_destrieux(
            data_dir=str(self.cache_dir / "nilearn"),
            verbose=0,
        )
        labels = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in atlas.labels
        ]
        self.parcels = []
        hemispheres = (
            ("lh", "left", slice(0, 10_242), np.asarray(atlas.map_left)),
            ("rh", "right", slice(10_242, 20_484), np.asarray(atlas.map_right)),
        )
        for prefix, hemisphere, vertex_slice, mapping in hemispheres:
            for label_index, label in enumerate(labels):
                if label_index == 0 or "unknown" in label.lower():
                    continue
                mask = mapping == label_index
                if mask.any():
                    self.parcels.append(
                        ParcelDefinition(
                            feature_prefix=f"{prefix}_{_clean_feature_name(label)}",
                            label=label,
                            hemisphere=hemisphere,
                            vertex_slice=vertex_slice,
                            mask=mask,
                        )
                    )

        tribe_snapshot = snapshot_download(
            repo_id=TRIBE_MODEL_ID,
            revision=TRIBE_MODEL_REVISION,
            allow_patterns=("config.yaml", "best.ckpt"),
        )
        vjepa_snapshot = snapshot_download(
            repo_id=VJEPA_MODEL_ID,
            revision=VJEPA_MODEL_REVISION,
            allow_patterns=(
                "config.json",
                "model.safetensors",
                "video_preprocessor_config.json",
            ),
        )
        # Neuralset validates model names against Hub repository IDs before
        # Transformers sees them. Register this verified local snapshot so the
        # pinned revision can be loaded without falling back to a moving branch.
        from neuralset.extractors.base import HuggingFaceMixin

        if vjepa_snapshot not in HuggingFaceMixin._REPOS:
            HuggingFaceMixin._REPOS.append(vjepa_snapshot)
        config_update = {
            "data.features_to_use": ["video"],
            "data.batch_size": 1,
            "data.num_workers": 0,
            "data.shuffle_train": False,
            "data.shuffle_val": False,
            "data.video_feature.use_audio": False,
            "data.video_feature.num_frames": self.video_frames,
            "data.video_feature.image.model_name": vjepa_snapshot,
            "data.video_feature.image.batch_size": 1,
            "data.video_feature.image.device": "cuda",
            "data.video_feature.infra.max_jobs": 1,
            "data.video_feature.infra.min_samples_per_job": 1,
        }
        self.tribe_model = TribeModel.from_pretrained(
            tribe_snapshot,
            cache_folder=self.cache_dir / "features" / "video",
            cluster=None,
            device="cuda",
            config_update=config_update,
        )
        self.model_load_seconds = time.perf_counter() - started

    def _parcel_features(
        self, predictions: np.ndarray
    ) -> tuple[pd.DataFrame, tuple[BrainRegionResponse, ...]]:
        if predictions.ndim != 2 or predictions.shape[1] != EXPECTED_VERTICES:
            raise RuntimeError(f"Unexpected TRIBE output shape: {predictions.shape}")
        if not np.isfinite(predictions).all():
            raise RuntimeError("TRIBE output contains non-finite values")

        features: dict[str, float] = {}
        region_magnitudes: list[tuple[ParcelDefinition, float, float]] = []
        for parcel in self.parcels:
            series = predictions[:, parcel.vertex_slice][:, parcel.mask].mean(axis=1)
            time_axis = np.arange(len(series), dtype=np.float32)
            slope = (
                float(np.polyfit(time_axis, series, 1)[0]) if len(series) >= 2 else 0.0
            )
            peak_fraction = (
                float(np.argmax(series) / max(len(series) - 1, 1))
                if len(series)
                else 0.0
            )
            prefix = parcel.feature_prefix
            features[f"{prefix}__mean_vid"] = float(np.mean(series))
            features[f"{prefix}__std_vid"] = float(np.std(series))
            features[f"{prefix}__max_vid"] = float(np.max(series))
            features[f"{prefix}__p95_vid"] = float(np.percentile(series, 95))
            features[f"{prefix}__slope_vid"] = slope
            features[f"{prefix}__peak_fraction_vid"] = peak_fraction
            magnitude = float(np.mean(np.abs(series)))
            peak_second = float(np.argmax(np.abs(series))) if len(series) else 0.0
            region_magnitudes.append((parcel, magnitude, peak_second))

        missing = sorted(set(self.expected_columns) - set(features))
        unexpected = sorted(set(features) - set(self.expected_columns))
        if missing or unexpected:
            raise RuntimeError(
                "TRIBE feature schema does not match the saved model: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        compact = pd.DataFrame(
            [[features[column] for column in self.expected_columns]],
            columns=self.expected_columns,
        )
        ranked = sorted(region_magnitudes, key=lambda item: item[1], reverse=True)
        maximum = ranked[0][1] if ranked else 0.0
        regions = tuple(
            BrainRegionResponse(
                region_id=parcel.feature_prefix,
                name=_display_region_name(parcel.label),
                hemisphere=parcel.hemisphere,
                relative_response=(
                    float(np.clip(magnitude / maximum * 100.0, 0.0, 100.0))
                    if maximum > 0
                    else 0.0
                ),
                peak_second=peak_second,
                description=_describe_region(parcel.label),
            )
            for parcel, magnitude, peak_second in ranked[:TOP_REGION_COUNT]
        )
        return compact, regions

    def score(self, video_path: str | Path) -> ScoringResult:
        """Return predicted average CTR and runtime measurements for one MP4."""
        import torch
        from neuralset.events.utils import standardize_events

        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        self.load()
        assert self.tribe_model is not None
        assert self.regressor is not None

        events = standardize_events(
            pd.DataFrame(
                [
                    {
                        "type": "Video",
                        "filepath": str(path),
                        "start": 0.0,
                        "timeline": "default",
                        "subject": "default",
                    }
                ]
            )
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            predictions, _segments = self.tribe_model.predict(
                events=events,
                verbose=False,
            )
        inference_seconds = time.perf_counter() - started
        prediction_array = np.asarray(predictions, dtype=np.float32)
        compact, top_regions = self._parcel_features(prediction_array)
        mean_ictr = float(self.regressor.predict(compact)[0])
        mean_ictr = max(mean_ictr, 0.0)

        return ScoringResult(
            percentage=float(np.clip(mean_ictr * 100.0, 0.0, 100.0)),
            raw_mean_ictr=mean_ictr,
            predictions=prediction_array,
            top_regions=top_regions,
            brain_timesteps=int(len(predictions)),
            compact_features=int(compact.shape[1]),
            model_load_seconds=self.model_load_seconds,
            inference_seconds=inference_seconds,
            peak_vram_mib=float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            process_rss_mib=float(psutil.Process().memory_info().rss / (1024 * 1024)),
        )
