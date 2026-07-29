from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TRIBE_CODE_REVISION = "af58661791a351a448a489042a28f6c37e1c14b7"
TRIBE_MODEL_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
VJEPA_MODEL_REVISION = "875c192b7b704b87d1e1d99345769632dd5f739a"
LLAMA_MODEL_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
W2V_BERT_MODEL_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
TRIBE_MODEL_ID = "facebook/tribev2"
VJEPA_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B"
W2V_BERT_MODEL_ID = "facebook/w2v-bert-2.0"
EXPECTED_VERTICES = 20_484
DEFAULT_SPLIT_COUNTS = {"train": 1_200, "val": 150, "test": 150}


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    vjepa_frames: int
    reconstruction_fps: int


PROFILES = {
    "lowmem8": RuntimeProfile("lowmem8", vjepa_frames=8, reconstruction_fps=16),
    "fidelity64": RuntimeProfile(
        "fidelity64", vjepa_frames=64, reconstruction_fps=16
    ),
}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def read_source_table(data_dir: Path) -> pd.DataFrame:
    tags = pd.read_csv(data_dir / "tags_cn.csv", encoding="utf-8-sig")
    tags["ID"] = tags["ID"].astype(str)
    tags = tags.rename(
        columns={
            "ID": "ad_id",
            "视频特点": "ad_type",
            "商品名字": "product_name",
            "主播": "presenter",
            "脚本": "script_tag",
            "剪辑技术": "editing_technique",
            "促销活动": "promotion",
            "件数": "item_count",
            "ROI": "roi",
            "CVR": "cvr",
        }
    )

    split_data = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    split_lookup = {
        str(ad_id): split for split, values in split_data.items() for ad_id in values
    }
    tags["split"] = tags["ad_id"].map(split_lookup)
    tags["duration_seconds"] = tags["ad_id"].map(
        lambda ad_id: max(
            0, sum(1 for _ in (data_dir / "ictr" / f"{ad_id}.csv").open("rb")) - 1
        )
    )
    tags["duration_bin"] = pd.qcut(
        tags["duration_seconds"],
        q=3,
        labels=["short", "medium", "long"],
        duplicates="drop",
    ).astype(str)
    if tags["split"].isna().any():
        raise RuntimeError("Some AdsTrace IDs are missing from split.json")
    return tags


def _allocate_group_counts(
    sizes: pd.Series, target: int, ensure_coverage: bool = True
) -> dict[object, int]:
    if target > int(sizes.sum()):
        raise ValueError("Target exceeds the available rows")
    raw = sizes / sizes.sum() * target
    allocated = np.floor(raw).astype(int)
    if ensure_coverage and target >= len(sizes):
        allocated = allocated.clip(lower=1)

    while int(allocated.sum()) > target:
        candidates = [
            key
            for key in allocated.index
            if allocated[key] > (1 if ensure_coverage and target >= len(sizes) else 0)
        ]
        key = min(candidates, key=lambda item: (raw[item] - allocated[item], str(item)))
        allocated[key] -= 1
    while int(allocated.sum()) < target:
        candidates = [key for key in allocated.index if allocated[key] < sizes[key]]
        key = max(candidates, key=lambda item: (raw[item] - allocated[item], str(item)))
        allocated[key] += 1
    return {key: int(value) for key, value in allocated.items()}


def stratified_sample(
    table: pd.DataFrame,
    split_counts: dict[str, int],
    seed: int = 33,
) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for split, target in split_counts.items():
        candidates = table[table["split"] == split].copy()
        candidates["_stratum"] = list(
            zip(candidates["ad_type"].fillna(""), candidates["duration_bin"])
        )
        sizes = candidates.groupby("_stratum", sort=True).size()
        allocations = _allocate_group_counts(sizes, target)
        chosen_parts = []
        for index, (stratum, count) in enumerate(sorted(allocations.items(), key=str)):
            group = candidates[candidates["_stratum"] == stratum]
            chosen_parts.append(
                group.sample(n=count, random_state=seed + index, replace=False)
            )
        chosen = pd.concat(chosen_parts, ignore_index=True).drop(columns="_stratum")
        if len(chosen) != target:
            raise RuntimeError(
                f"Sampling {split} produced {len(chosen)} rows; expected {target}"
            )
        selections.append(chosen)
    result = pd.concat(selections, ignore_index=True)
    return result.sort_values(["split", "ad_id"]).reset_index(drop=True)


def split_counts_for_sample(sample_size: int) -> dict[str, int]:
    if sample_size == 1_500:
        return dict(DEFAULT_SPLIT_COUNTS)
    train = round(sample_size * 0.8)
    val = round(sample_size * 0.1)
    return {"train": train, "val": val, "test": sample_size - train - val}


def select_benchmark(selection: pd.DataFrame, count: int = 8) -> pd.DataFrame:
    if count < 3:
        raise ValueError("Benchmark count must be at least 3")
    frequency = (
        selection.groupby("ad_type").size().sort_values(ascending=False, kind="stable")
    )
    common_types = list(frequency.index[: max(1, count - 2)])
    chosen: list[pd.Series] = []
    used_ids: set[str] = set()

    for ad_type in common_types:
        group = selection[selection["ad_type"] == ad_type]
        median = group["duration_seconds"].median()
        row = group.iloc[(group["duration_seconds"] - median).abs().argsort().iloc[0]]
        chosen.append(row)
        used_ids.add(str(row["ad_id"]))

    remaining = selection[~selection["ad_id"].isin(used_ids)]
    for quantile in (0.1, 0.9):
        target = selection["duration_seconds"].quantile(quantile)
        ordered = remaining.assign(
            _distance=(remaining["duration_seconds"] - target).abs()
        ).sort_values(["_distance", "ad_type", "ad_id"])
        seen_types = {str(row["ad_type"]) for row in chosen}
        distinct = ordered[~ordered["ad_type"].astype(str).isin(seen_types)]
        row = (distinct if not distinct.empty else ordered).iloc[0].drop(
            labels=["_distance"]
        )
        chosen.append(row)
        used_ids.add(str(row["ad_id"]))
        remaining = remaining[remaining["ad_id"] != row["ad_id"]]

    if len(chosen) < count:
        fill = remaining.sample(n=count - len(chosen), random_state=33)
        chosen.extend(row for _, row in fill.iterrows())
    return pd.DataFrame(chosen[:count]).reset_index(drop=True)


def locate_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        from shutil import which

        executable = which("ffmpeg")
        if executable:
            return executable
        raise RuntimeError(
            "ffmpeg was not found. Install imageio-ffmpeg from requirements.txt."
        )


def reconstruct_video(
    data_dir: Path,
    ad_id: str,
    destination: Path,
    profile: RuntimeProfile,
) -> None:
    frames_dir = data_dir / "frames" / ad_id
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found for ad {ad_id}")
    pattern = str(frames_dir / "%02d.jpg")
    ffmpeg = locate_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "1",
        "-start_number",
        "1",
        "-i",
        pattern,
        "-vf",
        f"fps={profile.reconstruction_fps}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    for attempt in range(1, 3):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise RuntimeError(
                    f"ffmpeg timed out twice while reconstructing {ad_id}"
                )
            continue
        if result.returncode == 0:
            return
        if attempt == 2:
            raise RuntimeError(
                f"ffmpeg failed for {ad_id}: {result.stderr.strip()}"
            )


def features_for_mode(modalities: str) -> list[str]:
    if modalities == "video":
        return ["video"]
    if modalities == "text-audio":
        return ["text", "audio"]
    raise ValueError(f"Unsupported modality mode: {modalities}")


def check_model_revisions(
    modalities: str, allow_change: bool = False
) -> dict[str, str]:
    from huggingface_hub import model_info

    expected = {TRIBE_MODEL_ID: TRIBE_MODEL_REVISION}
    if modalities == "video":
        expected[VJEPA_MODEL_ID] = VJEPA_MODEL_REVISION
    else:
        expected[LLAMA_MODEL_ID] = LLAMA_MODEL_REVISION
        expected[W2V_BERT_MODEL_ID] = W2V_BERT_MODEL_REVISION
    actual = {}
    for repo, pinned_revision in expected.items():
        try:
            actual[repo] = model_info(repo).sha
        except Exception as error:
            print(
                f"WARNING: Could not check the current revision for {repo}: "
                f"{type(error).__name__}: {error}. Using the pinned cached "
                f"revision {pinned_revision}.",
                file=sys.stderr,
                flush=True,
            )
            actual[repo] = pinned_revision
    mismatches = {
        repo: (expected[repo], actual[repo])
        for repo in expected
        if expected[repo] != actual[repo]
    }
    if mismatches and not allow_change:
        raise RuntimeError(
            f"Model revisions changed upstream: {mismatches}. "
            "Review before using --allow-revision-change."
        )
    return actual


def model_artifacts_for_mode(
    modalities: str,
) -> dict[tuple[str, str], tuple[str, ...]]:
    if modalities == "video":
        return {
            (VJEPA_MODEL_ID, VJEPA_MODEL_REVISION): (
                "config.json",
                "model.safetensors",
                "video_preprocessor_config.json",
            )
        }
    if modalities == "text-audio":
        return {
            (LLAMA_MODEL_ID, LLAMA_MODEL_REVISION): (
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
            (W2V_BERT_MODEL_ID, W2V_BERT_MODEL_REVISION): (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            ),
        }
    raise ValueError(f"Unsupported modality mode: {modalities}")


def ensure_model_artifacts_cached(modalities: str) -> None:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    for (repo_id, revision), filenames in model_artifacts_for_mode(
        modalities
    ).items():
        for filename in filenames:
            cached = try_to_load_from_cache(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
            if isinstance(cached, str) and Path(cached).is_file():
                continue
            print(f"Downloading required model file {repo_id}/{filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )


def enable_hugging_face_offline_mode() -> None:
    # Transformers checks this module-level flag on every from_pretrained call.
    # This prevents its lazily-created per-ad extractors from making redundant
    # HEAD requests after all pinned artifacts have been cached.
    from huggingface_hub import constants

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    constants.HF_HUB_OFFLINE = True


def load_tribe_model(
    cache_dir: Path,
    profile: RuntimeProfile,
    modalities: str,
    text_device: str = "cpu",
    allow_revision_change: bool = False,
):
    # The Xet transfer backend is extremely slow on this Windows connection.
    # Standard resumable HTTPS is substantially faster and is also used for
    # the source dataset downloader.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    from huggingface_hub import hf_hub_download
    from tribev2.demo_utils import TribeModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install the PyTorch 2.6 CUDA build before benchmarking."
        )
    revisions = check_model_revisions(modalities, allow_revision_change)
    try:
        ensure_model_artifacts_cached(modalities)
    except Exception as error:
        if modalities == "text-audio":
            raise RuntimeError(
                "Text mode requires approved access to "
                "meta-llama/Llama-3.2-3B and a Hugging Face login. "
                "Accept the model terms, then run '.\\.venv\\Scripts\\hf.exe "
                "auth login' before retrying."
            ) from error
        raise
    # TribeModel.from_pretrained converts its argument through pathlib.Path.
    # On Windows that changes "facebook/tribev2" to "facebook\\tribev2", which
    # Hugging Face rejects as a repository ID. Download the two pinned files
    # explicitly and pass a real local directory instead.
    checkpoint_dir = cache_dir / "models" / "tribev2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.yaml", "best.ckpt"):
        if not (checkpoint_dir / filename).is_file():
            hf_hub_download(
                repo_id=TRIBE_MODEL_ID,
                filename=filename,
                revision=TRIBE_MODEL_REVISION,
                local_dir=checkpoint_dir,
            )
    config_path = checkpoint_dir / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    windows_config = config_text.replace(
        "!!python/object/apply:pathlib.PosixPath",
        "!!python/object/apply:pathlib.WindowsPath",
    )
    if windows_config != config_text:
        config_path.write_text(windows_config, encoding="utf-8")
    config_update = {
        "data.features_to_use": features_for_mode(modalities),
        "data.batch_size": 1,
        "data.num_workers": 0,
        "data.shuffle_train": False,
        "data.shuffle_val": False,
    }
    if modalities == "video":
        config_update.update(
            {
                "data.video_feature.use_audio": False,
                "data.video_feature.num_frames": profile.vjepa_frames,
                "data.video_feature.image.batch_size": 1,
                "data.video_feature.image.device": "cuda",
                "data.video_feature.infra.max_jobs": 1,
                "data.video_feature.infra.min_samples_per_job": 1,
            }
        )
    else:
        config_update.update(
            {
                "data.text_feature.batch_size": 1,
                "data.text_feature.device": text_device,
                "data.text_feature.infra.max_jobs": 1,
                "data.text_feature.infra.min_samples_per_job": 1,
                "data.audio_feature.device": "cuda",
                "data.audio_feature.infra.max_jobs": 1,
                "data.audio_feature.infra.min_samples_per_job": 1,
            }
        )
    started = time.perf_counter()
    model = TribeModel.from_pretrained(
        checkpoint_dir,
        cache_folder=cache_dir,
        cluster=None,
        device="cuda",
        config_update=config_update,
    )
    enable_hugging_face_offline_mode()
    print(
        "INFO - All pinned model artifacts are cached; Hugging Face offline "
        "mode enabled for inference.",
        flush=True,
    )
    return model, revisions, time.perf_counter() - started


def make_video_events(video_path: Path) -> pd.DataFrame:
    from neuralset.events.utils import standardize_events

    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    return standardize_events(pd.DataFrame([event]))


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def transcript_word_events(transcript_path: Path) -> list[dict[str, object]]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    events: list[dict[str, object]] = []
    history: deque[str] = deque(maxlen=1_024)
    for sequence_id, segment in enumerate(payload.get("segments", [])):
        start = float(segment["start"])
        end = float(segment["end"])
        characters = [
            character
            for character in str(segment.get("text", ""))
            if not character.isspace()
        ]
        if not characters or end <= start:
            continue
        duration = (end - start) / len(characters)
        sentence = " ".join(characters)
        for index, character in enumerate(characters):
            history.append(character)
            events.append(
                {
                    "type": "Word",
                    "text": character,
                    "start": start + index * duration,
                    "duration": max(duration, 1e-4),
                    "timeline": "default",
                    "subject": "default",
                    "language": "chinese",
                    "modality": "heard",
                    "sequence_id": sequence_id,
                    "sentence": sentence,
                    "sentence_char": index * 2,
                    "context": " ".join(history),
                }
            )
    if not events:
        raise RuntimeError(f"No timed text found in {transcript_path}")
    return events


def make_text_audio_events(
    audio_path: Path, transcript_path: Path
) -> pd.DataFrame:
    from neuralset.events.utils import standardize_events

    audio_event = {
        "type": "Audio",
        "filepath": str(audio_path),
        "start": 0.0,
        "duration": wav_duration(audio_path),
        "timeline": "default",
        "subject": "default",
    }
    return standardize_events(
        pd.DataFrame([audio_event, *transcript_word_events(transcript_path)])
    )


def segment_times(segments: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    durations = []
    for index, segment in enumerate(segments):
        starts.append(float(getattr(segment, "start", index)))
        durations.append(float(getattr(segment, "duration", 1.0)))
    return np.asarray(starts, dtype=np.float32), np.asarray(
        durations, dtype=np.float32
    )


def read_ictr(data_dir: Path, ad_id: str) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(data_dir / "ictr" / f"{ad_id}.csv")
    return (
        table["sec"].to_numpy(dtype=np.float32),
        table["ictr"].to_numpy(dtype=np.float32),
    )


def replace_with_retry(
    source: Path,
    destination: Path,
    attempts: int = 20,
    base_delay_seconds: float = 0.25,
) -> None:
    """Replace a file while tolerating short-lived Windows/OneDrive locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            delay = min(base_delay_seconds * (2**attempt), 2.0)
            time.sleep(delay)


def atomic_temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    replace_with_retry(temporary, path)


def validate_raw_output(path: Path) -> tuple[int, int]:
    with np.load(path) as output:
        predictions = output["brain_activity"]
        starts = output["segment_start_seconds"]
        if predictions.ndim != 2 or predictions.shape[1] != EXPECTED_VERTICES:
            raise RuntimeError(f"Unexpected TRIBE shape in {path}: {predictions.shape}")
        if len(starts) != predictions.shape[0] or np.any(np.diff(starts) < 0):
            raise RuntimeError(f"Invalid segment timestamps in {path}")
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"Non-finite TRIBE values in {path}")
        return int(predictions.shape[0]), int(predictions.shape[1])


def _clean_feature_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "unknown"


def parcel_features(predictions: np.ndarray) -> dict[str, float]:
    from nilearn.datasets import fetch_atlas_surf_destrieux

    atlas = fetch_atlas_surf_destrieux(verbose=0)
    labels = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in atlas.labels
    ]
    features: dict[str, float] = {}
    hemisphere_data = (
        ("lh", np.asarray(atlas.map_left), predictions[:, :10_242]),
        ("rh", np.asarray(atlas.map_right), predictions[:, 10_242:]),
    )
    for hemisphere, mapping, values in hemisphere_data:
        for label_index, label in enumerate(labels):
            if label_index == 0 or "unknown" in label.lower():
                continue
            mask = mapping == label_index
            if not mask.any():
                continue
            series = values[:, mask].mean(axis=1)
            time_axis = np.arange(len(series), dtype=np.float32)
            slope = (
                float(np.polyfit(time_axis, series, 1)[0])
                if len(series) >= 2
                else 0.0
            )
            peak_fraction = (
                float(np.argmax(series) / max(len(series) - 1, 1))
                if len(series)
                else 0.0
            )
            prefix = f"{hemisphere}_{_clean_feature_name(label)}"
            features[f"{prefix}__mean"] = float(np.mean(series))
            features[f"{prefix}__std"] = float(np.std(series))
            features[f"{prefix}__max"] = float(np.max(series))
            features[f"{prefix}__p95"] = float(np.percentile(series, 95))
            features[f"{prefix}__slope"] = slope
            features[f"{prefix}__peak_fraction"] = peak_fraction
    return features


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, default=json_value) + "\n"
        )


def json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def process_one_ad(
    model,
    row: pd.Series,
    data_dir: Path,
    output_dir: Path,
    profile: RuntimeProfile,
    modalities: str,
    resume: bool,
) -> tuple[dict[str, object], dict[str, object] | None]:
    ad_id = str(row["ad_id"])
    raw_path = output_dir / "raw" / f"{ad_id}.npz"
    relative_raw_path = str(raw_path.relative_to(output_dir))
    if resume and raw_path.exists():
        timesteps, vertices = validate_raw_output(raw_path)
        with np.load(raw_path) as existing:
            compact = {
                "ad_id": ad_id,
                **parcel_features(existing["brain_activity"]),
            }
        return (
            {
                **row.to_dict(),
                "status": "complete",
                "runtime_seconds": np.nan,
                "tribe_timesteps": timesteps,
                "tribe_vertices": vertices,
                "profile": profile.name,
                "modalities": modalities,
                "raw_path": relative_raw_path,
                "error": "",
            },
            compact,
        )

    import torch

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    reconstruction_seconds = 0.0
    if modalities == "video":
        with tempfile.TemporaryDirectory(prefix=f"tribe_{ad_id}_") as temporary_dir:
            video_path = Path(temporary_dir) / f"{ad_id}.mp4"
            reconstruct_started = time.perf_counter()
            reconstruct_video(data_dir, ad_id, video_path, profile)
            reconstruction_seconds = time.perf_counter() - reconstruct_started
            events = make_video_events(video_path)
            inference_started = time.perf_counter()
            predictions, segments = model.predict(events=events, verbose=False)
            inference_seconds = time.perf_counter() - inference_started
    else:
        audio_path = data_dir / "audios_16k" / f"{ad_id}.wav"
        transcript_path = data_dir / "transcripts" / f"{ad_id}.json"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio for {ad_id}: {audio_path}")
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Missing transcript for {ad_id}: {transcript_path}"
            )
        events = make_text_audio_events(audio_path, transcript_path)
        inference_started = time.perf_counter()
        predictions, segments = model.predict(events=events, verbose=False)
        inference_seconds = time.perf_counter() - inference_started

    predictions = np.asarray(predictions, dtype=np.float32)
    starts, durations = segment_times(segments)
    ictr_seconds, ictr = read_ictr(data_dir, ad_id)
    atomic_save_npz(
        raw_path,
        brain_activity=predictions,
        segment_start_seconds=starts,
        segment_duration_seconds=durations,
        ictr_seconds=ictr_seconds,
        ictr=ictr,
    )
    timesteps, vertices = validate_raw_output(raw_path)
    compact = {"ad_id": ad_id, **parcel_features(predictions)}
    runtime = time.perf_counter() - started
    record = {
        **row.to_dict(),
        "status": "complete",
        "runtime_seconds": runtime,
        "reconstruction_seconds": reconstruction_seconds,
        "inference_seconds": inference_seconds,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "tribe_timesteps": timesteps,
        "tribe_vertices": vertices,
        "profile": profile.name,
        "modalities": modalities,
        "raw_path": relative_raw_path,
        "mean_ictr": float(np.mean(ictr)),
        "max_ictr": float(np.max(ictr)),
        "error": "",
    }
    return record, compact


def write_parquet_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_temporary_path(path)
    table.to_parquet(temporary, index=False)
    replace_with_retry(temporary, path)


def run_pipeline(args: argparse.Namespace, benchmark: bool) -> int:
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")
    if args.max_consecutive_failures < 1:
        raise ValueError("--max-consecutive-failures must be at least 1")
    if not args.commercial_permission_confirmed:
        raise RuntimeError(
            "This run is configured for commercial use. Pass "
            "--commercial-permission-confirmed only after confirming the separate "
            "AdsTrace and TRIBE v2 permissions."
        )
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]

    source = read_source_table(data_dir)
    split_counts = split_counts_for_sample(args.sample_size)
    selection = stratified_sample(source, split_counts, seed=args.seed)
    write_parquet_atomic(selection, output_dir / "selection.parquet")
    work = select_benchmark(selection, args.benchmark_size) if benchmark else selection
    if args.limit:
        work = work.head(args.limit)

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, revisions, model_load_seconds = load_tribe_model(
        cache_dir,
        profile,
        modalities=args.modalities,
        text_device=args.text_device,
        allow_revision_change=args.allow_revision_change,
    )
    provenance = {
        "tribe_code_revision": TRIBE_CODE_REVISION,
        "model_revisions": revisions,
        "profile": asdict(profile),
        "features_to_use": features_for_mode(args.modalities),
        "modalities": args.modalities,
        "text_device": args.text_device,
        "feature_cache_policy": "retain_until_process_exit",
        "sample_size": args.sample_size,
        "split_counts": split_counts,
        "seed": args.seed,
        "commercial_permission_confirmed": True,
        "original_video_available": False,
        "source_frame_rate_hz": 1,
        "model_load_seconds": model_load_seconds,
        "created_at_unix": time.time(),
        "cache_dir": str(cache_dir),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=json_value),
        encoding="utf-8",
    )

    records: list[dict[str, object]] = []
    feature_records: list[dict[str, object]] = []
    failures_path = output_dir / "failures.jsonl"
    pipeline_started = time.perf_counter()
    failure_count = 0
    consecutive_failures = 0
    for position, (_, row) in enumerate(work.iterrows(), start=1):
        should_stop = False
        ad_id = str(row["ad_id"])
        print(
            f"[{position}/{len(work)}] ad={ad_id} type={row['ad_type']} "
            f"duration={row['duration_seconds']}s",
            flush=True,
        )
        try:
            record, compact = process_one_ad(
                model,
                row,
                data_dir,
                output_dir,
                profile,
                modalities=args.modalities,
                resume=args.resume,
            )
            records.append(record)
            consecutive_failures = 0
            if compact is not None:
                feature_records.append(compact)
            print(
                f"  complete in {record.get('runtime_seconds', float('nan')):.1f}s",
                flush=True,
            )
        except Exception as error:
            failure_count += 1
            consecutive_failures += 1
            failure = {
                **row.to_dict(),
                "status": "failed",
                "profile": profile.name,
                "modalities": args.modalities,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at_unix": time.time(),
            }
            records.append(failure)
            append_jsonl(failures_path, failure)
            print(f"  FAILED: {failure['error']}", file=sys.stderr, flush=True)
            should_stop = (
                args.fail_fast
                or consecutive_failures >= args.max_consecutive_failures
            )

        write_parquet_atomic(pd.DataFrame(records), output_dir / "manifest.parquet")
        if feature_records:
            feature_table = pd.DataFrame(feature_records)
            write_parquet_atomic(
                feature_table,
                output_dir / "compact_features.parquet",
            )
            feature_names = [
                column for column in feature_table.columns if column != "ad_id"
            ]
            (output_dir / "feature_schema.json").write_text(
                json.dumps(feature_names, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if position % args.progress_every == 0 or position == len(work):
            elapsed = time.perf_counter() - pipeline_started
            eta = elapsed / position * (len(work) - position)
            percent = position / len(work) * 100
            print(
                f"PROGRESS {percent:5.1f}% ({position}/{len(work)}) | "
                f"elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)} | failures {failure_count}",
                flush=True,
            )
        if should_stop:
            print(
                "STOPPED SAFELY after "
                f"{consecutive_failures} consecutive failure(s). "
                "Completed raw outputs remain resumable.",
                file=sys.stderr,
                flush=True,
            )
            break

    manifest = pd.DataFrame(records)
    successful = manifest[manifest["status"] == "complete"] if not manifest.empty else manifest
    measured = successful[
        successful.get("runtime_seconds", pd.Series(dtype=float)).notna()
    ]
    if benchmark and not measured.empty:
        # TRIBE lazily initializes (and may download) modality encoders inside
        # the first predict() call. On a fresh machine that one-time setup can
        # dwarf actual inference, so exclude the first row when later clean
        # measurements are available.  A resumed warm-up already has NaN
        # runtime and therefore is not present in `measured`.
        warmup_excluded = measured.iloc[0:0]
        measured_for_estimate = measured
        if len(measured) > 1 and measured.index[0] == 0:
            warmup_excluded = measured.iloc[[0]]
            measured_for_estimate = measured.iloc[1:]
        factors = (
            measured_for_estimate["runtime_seconds"].astype(float)
            / measured_for_estimate["duration_seconds"].astype(float)
        )
        selected_seconds = float(selection["duration_seconds"].sum())
        report = {
            "profile": asdict(profile),
            "modalities": args.modalities,
            "model_load_seconds": model_load_seconds,
            "benchmark_requested": len(work),
            "benchmark_completed": len(successful),
            "benchmark_failed": int((manifest["status"] == "failed").sum()),
            "benchmark_timed": len(measured_for_estimate),
            "warmup_excluded": len(warmup_excluded),
            "median_processing_seconds_per_ad_second": float(factors.median()),
            "p90_processing_seconds_per_ad_second": float(
                np.percentile(factors, 90)
            ),
            "selected_ad_seconds": selected_seconds,
            "estimated_full_hours_median": float(
                selected_seconds * factors.median() / 3600
            ),
            "estimated_full_hours_p90": float(
                selected_seconds * np.percentile(factors, 90) / 3600
            ),
            "rows": measured_for_estimate[
                [
                    "ad_id",
                    "ad_type",
                    "duration_seconds",
                    "runtime_seconds",
                    "reconstruction_seconds",
                    "inference_seconds",
                    "peak_cuda_memory_gib",
                ]
            ].to_dict(orient="records"),
        }
        (output_dir / "benchmark.json").write_text(
            json.dumps(
                report, ensure_ascii=False, indent=2, default=json_value
            ),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=json_value))
    return 0 if len(successful) == len(work) else 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a TRIBE v2 to AdsTrace metrics dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("benchmark", "run", "retry-failures"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-dir", type=Path, default=Path("data/AdsTrace"))
        child.add_argument(
            "--output-dir",
            type=Path,
            default=Path("data/AdsTraceTribeV2_video_1500"),
        )
        child.add_argument(
            "--cache-dir",
            type=Path,
            default=Path("C:/td") if os.name == "nt" else Path.home() / ".cache" / "tribe_dopa",
            help="Short cache path; avoids Windows MAX_PATH failures.",
        )
        child.add_argument("--sample-size", type=int, default=1_500)
        child.add_argument("--seed", type=int, default=33)
        child.add_argument("--profile", choices=PROFILES, default="lowmem8")
        child.add_argument(
            "--modalities",
            choices=("video", "text-audio"),
            default="video",
        )
        child.add_argument(
            "--text-device",
            choices=("cpu", "cuda"),
            default="cpu",
            help="CPU is safest for the 3B text model on an 8 GB GPU.",
        )
        child.add_argument("--benchmark-size", type=int, default=8)
        child.add_argument("--limit", type=int)
        child.add_argument(
            "--progress-every",
            type=int,
            default=5,
            help="Print percentage, elapsed time, and ETA every N ads (default: 5).",
        )
        child.add_argument("--resume", action="store_true")
        child.add_argument("--fail-fast", action="store_true")
        child.add_argument(
            "--max-consecutive-failures",
            type=int,
            default=3,
            help="Stop safely after N consecutive failures (default: 3).",
        )
        child.add_argument("--allow-revision-change", action="store_true")
        child.add_argument(
            "--commercial-permission-confirmed", action="store_true"
        )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.command == "retry-failures":
        args.resume = True
        return run_pipeline(args, benchmark=False)
    return run_pipeline(args, benchmark=args.command == "benchmark")


if __name__ == "__main__":
    raise SystemExit(main())
