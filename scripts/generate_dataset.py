from __future__ import annotations

r"""
Portable AdsTrace → TRIBE v2 dataset generator.

Reproduces the exact dataset files in data/:
  text-audio.parquet  – brain features from text + audio inputs
  video.parquet       – brain features from video inputs
  combined.parquet    – 1800 features from both conditions side-by-side

Quick start (Windows, Python 3.12, NVIDIA GPU):

    py -3.12 scripts/generate_dataset.py setup
    .\.tribe_env\Scripts\python.exe scripts/generate_dataset.py run --license-accepted

TRIBE v2 and AdsTrace are non-commercially licensed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import wave
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_REPO = "Xiuze/AdsTrace"
DATASET_REVISION = "d449d193c20fec73ed913697edd13e07de748f9f"
TRIBE_CODE_REVISION = "af58661791a351a448a489042a28f6c37e1c14b7"
TRIBE_MODEL_ID = "facebook/tribev2"
TRIBE_MODEL_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
VJEPA_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
VJEPA_MODEL_REVISION = "875c192b7b704b87d1e1d99345769632dd5f739a"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B"
AUDIO_MODEL_ID = "facebook/w2v-bert-2.0"
EXPECTED_ADS = 2_833
EXPECTED_VERTICES = 20_484
DEFAULT_SAMPLE_SIZE = 1_500
DEFAULT_SPLIT_COUNTS = {"train": 1_200, "val": 150, "test": 150}

MODALITY_CONFIGS = {
    "text-audio": {
        "features_to_use": ["text", "audio"],
        "model_ids": [TRIBE_MODEL_ID, LLAMA_MODEL_ID, AUDIO_MODEL_ID],
    },
    "video": {
        "features_to_use": ["video"],
        "model_ids": [TRIBE_MODEL_ID, VJEPA_MODEL_ID],
    },
}

STRIP_COLS = [
    "status", "runtime_seconds", "reconstruction_seconds", "inference_seconds",
    "peak_cuda_memory_gib", "tribe_timesteps", "tribe_vertices", "profile",
    "modalities", "raw_path", "error", "traceback", "failed_at_unix",
]


@dataclass(frozen=True)
class SourceFile:
    name: str
    size: int
    sha256: str | None = None
    archive: bool = False


SOURCE_FILES = (
    SourceFile("tags_cn.csv", 214_008),
    SourceFile("products_cn.json", 6_120),
    SourceFile("products_en.json", 6_905),
    SourceFile("split.json", 53_896),
    SourceFile("ictr.zip", 1_172_508,
               "9414bcaf4984c6ee88082899a8f73a34b8c370c1bb6b7d7b4a4513e4c8c42a7d", True),
    SourceFile("transcripts.zip", 1_876_336,
               "605ebc46b2e1bd835d3dcc4d430a4cead4517351386179a92d22c951dc67eea4", True),
    SourceFile("audios_16k.zip", 1_851_914_587,
               "a93fcd2bdfe4f1ae62ff965a4bf9b93b66d8c8c4f7bda7c1740ff0dc370389a4", True),
    SourceFile("frames.zip", 8_332_717_273,
               "c37e1c9b442337a9fcbdab77dba02aafe6575b348fad4d277e07da070fcadc6c", True),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                       default=_json_default))


def _json_default(value: object) -> object:
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    log(f"[command] {subprocess.list2cmdline(command)}")
    result = subprocess.run(command, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


def venv_python(environment: Path) -> Path:
    return (environment / "Scripts" / "python.exe" if os.name == "nt"
            else environment / "bin" / "python")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def source_url(name: str) -> str:
    return (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
            f"{DATASET_REVISION}/{name}?download=true")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            d.update(chunk)
    return d.hexdigest()


def download_with_resume(source: SourceFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.size:
        if source.sha256 is None or sha256_file(destination) == source.sha256:
            log(f"[download] verified existing {destination.name}")
            return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    current = partial.stat().st_size if partial.exists() else 0
    if current > source.size:
        partial.unlink()
        current = 0
    req = urllib.request.Request(source_url(source.name))
    if current:
        req.add_header("Range", f"bytes={current}-")
    log(f"[download] {source.name}: {current / 1024**3:.2f} / "
        f"{source.size / 1024**3:.2f} GiB")
    started = time.monotonic()
    last = started
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if current and e.code == 416 and current == source.size:
            resp = None
        else:
            raise
    if resp is not None:
        mode = "ab" if current and getattr(resp, "status", resp.getcode()) == 206 else "wb"
        if mode == "wb":
            current = 0
        with resp, partial.open(mode) as f:
            while chunk := resp.read(8 * 1024 * 1024):
                f.write(chunk)
                current += len(chunk)
                now = time.monotonic()
                if now - last >= 15:
                    elapsed = max(now - started, 0.001)
                    log(f"  {100 * current / source.size:5.1f}%  "
                        f"{current / 1024**3:6.2f} GiB  "
                        f"{(current / 1024**2) / elapsed:6.1f} MiB/s")
                    last = now
    if partial.stat().st_size != source.size:
        raise RuntimeError(f"Size mismatch: {partial.stat().st_size} != {source.size}")
    if source.sha256 and sha256_file(partial) != source.sha256:
        raise RuntimeError(f"SHA-256 mismatch: {source.name}")
    os.replace(partial, destination)
    return destination


def safe_extract_zip(archive: Path, data_dir: Path) -> None:
    root = data_dir.resolve()
    log(f"[extract] {archive.name}")
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            if not (data_dir / m.filename).resolve().is_relative_to(root):
                raise RuntimeError(f"Unsafe path: {m.filename}")
        z.extractall(data_dir)


def archive_is_extracted(source: SourceFile, data_dir: Path) -> bool:
    checks = {
        "frames.zip": lambda: len(list((data_dir / "frames").glob("*/*.jpg"))) >= EXPECTED_ADS,
        "audios_16k.zip": lambda: len(list((data_dir / "audios_16k").glob("*.wav"))) == EXPECTED_ADS,
        "ictr.zip": lambda: len(list((data_dir / "ictr").glob("*.csv"))) == EXPECTED_ADS,
        "transcripts.zip": lambda: len(list((data_dir / "transcripts").glob("*.json"))) == EXPECTED_ADS,
    }
    return checks[source.name]() if source.name in checks else False


def validate_dataset(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "tags_cn.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = {str(r["ID"]) for r in rows}
    if len(rows) != EXPECTED_ADS or len(ids) != EXPECTED_ADS:
        raise RuntimeError(f"Expected {EXPECTED_ADS} unique rows, got {len(rows)} rows / {len(ids)} ids")
    folders = {
        "frames": {p.name for p in (data_dir / "frames").iterdir() if p.is_dir()},
        "audios_16k": {p.stem for p in (data_dir / "audios_16k").glob("*.wav")},
        "transcripts": {p.stem for p in (data_dir / "transcripts").glob("*.json")},
        "ictr": {p.stem for p in (data_dir / "ictr").glob("*.csv")},
    }
    for name, available in folders.items():
        missing = ids - available
        if missing:
            raise RuntimeError(f"{name} missing {len(missing)} ids")
    splits = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    expected = {"train": 2_266, "val": 283, "test": 284}
    counts = {k: len(v) for k, v in splits.items()}
    if counts != expected:
        raise RuntimeError(f"Unexpected splits: {counts}")
    report = dict(dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION,
                  num_ads=len(ids), split_counts=counts,
                  modalities=["text", "audio", "video"], validated_at_unix=time.time())
    write_json(data_dir / "download_validation.json", report)
    return report


def download_dataset(data_dir: Path, keep_archives: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_dir).free / 1024**3
    if free < 24:
        raise RuntimeError(f"Need 24 GiB free, only {free:.1f} GiB")
    log(f"[preflight] data={data_dir.resolve()} free={free:.1f} GiB")
    for src in SOURCE_FILES:
        dst = data_dir / src.name
        if src.archive and archive_is_extracted(src, data_dir):
            log(f"[extract] already complete: {src.name}")
            continue
        download_with_resume(src, dst)
        if src.archive:
            safe_extract_zip(dst, data_dir)
            if not archive_is_extracted(src, data_dir):
                raise RuntimeError(f"Extraction failed: {src.name}")
            if not keep_archives:
                dst.unlink()
    validate_dataset(data_dir)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def duration_from_ictr(path: Path) -> int:
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def read_source_table(data_dir: Path):
    import pandas as pd
    tags = pd.read_csv(data_dir / "tags_cn.csv", encoding="utf-8-sig")
    tags["ID"] = tags["ID"].astype(str)
    tags = tags.rename(columns={
        "ID": "ad_id", "视频特点": "ad_type", "商品名字": "product_name",
        "主播": "presenter", "脚本": "script_tag", "剪辑技术": "editing_technique",
        "促销活动": "promotion", "件数": "item_count", "ROI": "roi", "CVR": "cvr",
    })
    splits = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    lookup = {str(aid): split for split, vals in splits.items() for aid in vals}
    tags["split"] = tags["ad_id"].map(lookup)
    tags["duration_seconds"] = tags["ad_id"].map(
        lambda a: duration_from_ictr(data_dir / "ictr" / f"{a}.csv"))
    tags["duration_bin"] = pd.qcut(tags["duration_seconds"], q=3,
                                   labels=["short", "medium", "long"],
                                   duplicates="drop").astype(str)
    if tags["split"].isna().any():
        raise RuntimeError("Some ids missing from split.json")
    return tags


def allocate_group_counts(sizes, target: int) -> dict[Any, int]:
    import numpy as np
    if target > int(sizes.sum()):
        raise ValueError("Sample target exceeds available rows")
    raw = sizes / sizes.sum() * target
    allocated = np.floor(raw).astype(int)
    ensure = target >= len(sizes)
    if ensure:
        allocated = allocated.clip(lower=1)
    minimum = 1 if ensure else 0
    while int(allocated.sum()) > target:
        candidates = [k for k in allocated.index if allocated[k] > minimum]
        k = min(candidates, key=lambda x: (raw[x] - allocated[x], str(x)))
        allocated[k] -= 1
    while int(allocated.sum()) < target:
        candidates = [k for k in allocated.index if allocated[k] < sizes[k]]
        k = max(candidates, key=lambda x: (raw[x] - allocated[x], str(x)))
        allocated[k] += 1
    return {k: int(v) for k, v in allocated.items()}


def split_counts_for_sample(sample_size: int) -> dict[str, int]:
    if sample_size == DEFAULT_SAMPLE_SIZE:
        return dict(DEFAULT_SPLIT_COUNTS)
    train = round(sample_size * 0.8)
    val = round(sample_size * 0.1)
    return {"train": train, "val": val, "test": sample_size - train - val}


def stratified_sample(table, sample_size: int, seed: int):
    import pandas as pd
    selected = []
    for split, target in split_counts_for_sample(sample_size).items():
        cand = table[table["split"] == split].copy()
        cand["_stratum"] = list(zip(cand["ad_type"].fillna(""), cand["duration_bin"]))
        sizes = cand.groupby("_stratum", sort=True).size()
        alloc = allocate_group_counts(sizes, target)
        parts = []
        for idx, (stratum, count) in enumerate(sorted(alloc.items(), key=str)):
            group = cand[cand["_stratum"] == stratum]
            parts.append(group.sample(n=count, random_state=seed + idx, replace=False))
        chosen = pd.concat(parts, ignore_index=True).drop(columns="_stratum")
        if len(chosen) != target:
            raise RuntimeError(f"Incorrect {split} sample size: {len(chosen)}")
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True).sort_values(
        ["split", "ad_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# TRIBE video reconstruction
# ---------------------------------------------------------------------------

def locate_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("ffmpeg not found; re-run the setup command")


def frame_sort_key(path: Path) -> tuple[int, str]:
    m = re.findall(r"\d+", path.stem)
    return (int(m[-1]) if m else 0, path.name)


def ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\'")


def reconstruct_silent_video(data_dir: Path, ad_id: str, destination: Path,
                             frames_per_second: int) -> None:
    frames = sorted((data_dir / "frames" / ad_id).glob("*.jpg"), key=frame_sort_key)
    if not frames:
        raise FileNotFoundError(f"No frames for ad {ad_id}")
    concat_path = destination.with_suffix(".frames.txt")
    lines = []
    for f in frames:
        lines.append(f"file '{ffmpeg_concat_path(f)}'")
        lines.append("duration 1")
    lines.append(f"file '{ffmpeg_concat_path(frames[-1])}'")
    atomic_write_text(concat_path, "\n".join(lines) + "\n")
    cmd = [locate_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(concat_path),
           "-vf", f"fps={frames_per_second}", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(destination)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ffmpeg failed for {ad_id}: {r.stderr.strip()}")
    finally:
        concat_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TRIBE model & inference
# ---------------------------------------------------------------------------

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def transcript_word_events(transcript_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    history: deque[str] = deque(maxlen=1024)
    for seq_id, seg in enumerate(payload.get("segments", [])):
        start, end = float(seg["start"]), float(seg["end"])
        chars = [c for c in str(seg.get("text", "")) if not c.isspace()]
        if not chars or end <= start:
            continue
        dur = (end - start) / len(chars)
        for idx, char in enumerate(chars):
            history.append(char)
            events.append(dict(type="Word", text=char, start=start + idx * dur,
                               duration=max(dur, 1e-4), timeline="default",
                               subject="default", language="chinese",
                               modality="heard", sequence_id=seq_id,
                               sentence=" ".join(chars), sentence_char=idx * 2,
                               context=" ".join(history)))
    if not events:
        raise RuntimeError(f"No timed text in {transcript_path}")
    return events


def make_multimodal_events(video_path: Path, audio_path: Path,
                           transcript_path: Path):
    import pandas as pd
    from neuralset.events.utils import standardize_events
    base = dict(start=0.0, timeline="default", subject="default")
    events = [
        {**base, "type": "Video", "filepath": str(video_path)},
        {**base, "type": "Audio", "filepath": str(audio_path),
         "duration": wav_duration(audio_path)},
        *transcript_word_events(transcript_path),
    ]
    return standardize_events(pd.DataFrame(events))


def load_tribe_model(cache_dir: Path, features_to_use: list[str],
                     video_frames: int, text_device: str,
                     feature_device: str):
    import torch
    from tribev2.demo_utils import TribeModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config = {
        "data.features_to_use": features_to_use,
        "data.batch_size": 1, "data.num_workers": 0,
        "data.shuffle_train": False, "data.shuffle_val": False,
        "data.text_feature.batch_size": 1,
        "data.text_feature.device": text_device,
        "data.text_feature.infra.max_jobs": 1,
        "data.text_feature.infra.min_samples_per_job": 1,
        "data.audio_feature.device": feature_device,
        "data.audio_feature.infra.max_jobs": 1,
        "data.audio_feature.infra.min_samples_per_job": 1,
        "data.video_feature.use_audio": False,
        "data.video_feature.num_frames": video_frames,
        "data.video_feature.image.batch_size": 1,
        "data.video_feature.image.device": feature_device,
        "data.video_feature.infra.max_jobs": 1,
        "data.video_feature.infra.min_samples_per_job": 1,
    }
    started = time.perf_counter()
    model = TribeModel.from_pretrained(TRIBE_MODEL_ID, cache_folder=cache_dir,
                                       cluster=None, device="cuda",
                                       config_update=config)
    return model, time.perf_counter() - started


def segment_times(segments: Sequence[Any]):
    import numpy as np
    starts = [float(getattr(s, "start", i)) for i, s in enumerate(segments)]
    durations = [float(getattr(s, "duration", 1.0)) for s in segments]
    return np.asarray(starts, dtype=np.float32), np.asarray(durations, dtype=np.float32)


def read_ictr(data_dir: Path, ad_id: str):
    import numpy as np
    import pandas as pd
    t = pd.read_csv(data_dir / "ictr" / f"{ad_id}.csv")
    return t["sec"].to_numpy(dtype=np.float32), t["ictr"].to_numpy(dtype=np.float32)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def validate_raw_output(path: Path) -> tuple[int, int]:
    import numpy as np
    with np.load(path) as d:
        preds, starts = d["brain_activity"], d["segment_start_seconds"]
        if preds.ndim != 2 or preds.shape[1] != EXPECTED_VERTICES:
            raise RuntimeError(f"Unexpected shape: {preds.shape}")
        if len(starts) != preds.shape[0] or np.any(np.diff(starts) < 0):
            raise RuntimeError("Invalid timestamps")
        if not np.isfinite(preds).all():
            raise RuntimeError("Non-finite predictions")
        return int(preds.shape[0]), int(preds.shape[1])


# ---------------------------------------------------------------------------
# Parcellation
# ---------------------------------------------------------------------------

def clean_feature_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "unknown"


def parcel_features(predictions) -> dict[str, float]:
    import numpy as np
    from nilearn.datasets import fetch_atlas_surf_destrieux
    atlas = fetch_atlas_surf_destrieux(verbose=0)
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas.labels]
    result: dict[str, float] = {}
    hemispheres = (
        ("lh", np.asarray(atlas.map_left), predictions[:, :10_242]),
        ("rh", np.asarray(atlas.map_right), predictions[:, 10_242:]),
    )
    for hemi, mapping, values in hemispheres:
        for idx, label in enumerate(labels):
            if idx == 0 or "unknown" in label.lower():
                continue
            mask = mapping == idx
            if not mask.any():
                continue
            series = values[:, mask].mean(axis=1)
            t = np.arange(len(series), dtype=np.float32)
            slope = float(np.polyfit(t, series, 1)[0]) if len(series) >= 2 else 0.0
            peak = float(np.argmax(series) / max(len(series) - 1, 1)) if len(series) else 0.0
            prefix = f"{hemi}_{clean_feature_name(label)}"
            result[f"{prefix}__mean"] = float(np.mean(series))
            result[f"{prefix}__std"] = float(np.std(series))
            result[f"{prefix}__max"] = float(np.max(series))
            result[f"{prefix}__p95"] = float(np.percentile(series, 95))
            result[f"{prefix}__slope"] = slope
            result[f"{prefix}__peak_fraction"] = peak
    return result


# ---------------------------------------------------------------------------
# Per-ad processing
# ---------------------------------------------------------------------------

def process_one_ad(model: Any, row: Any, data_dir: Path, output_dir: Path,
                   video_frames: int, reconstruction_fps: int,
                   modality_name: str):
    import numpy as np
    import torch
    ad_id = str(row["ad_id"])
    raw_path = output_dir / "raw" / f"{ad_id}.npz"
    record_path = output_dir / "records" / f"{ad_id}.json"
    feature_path = output_dir / "features" / f"{ad_id}.json"
    if raw_path.exists() and record_path.exists() and feature_path.exists():
        validate_raw_output(raw_path)
        return (json.loads(record_path.read_text()),
                json.loads(feature_path.read_text()))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix=f"tribe_{ad_id}_") as tmp:
        video_path = Path(tmp) / f"{ad_id}.mp4"
        reconstruct_silent_video(data_dir, ad_id, video_path, reconstruction_fps)
        events = make_multimodal_events(
            video_path, data_dir / "audios_16k" / f"{ad_id}.wav",
            data_dir / "transcripts" / f"{ad_id}.json")
        preds, segments = model.predict(events=events, verbose=False)
    preds = np.asarray(preds, dtype=np.float32)
    starts, durations = segment_times(segments)
    ictr_secs, ictr = read_ictr(data_dir, ad_id)
    atomic_save_npz(raw_path, brain_activity=preds,
                    segment_start_seconds=starts,
                    segment_duration_seconds=durations,
                    ictr_seconds=ictr_secs, ictr=ictr)
    timesteps, vertices = validate_raw_output(raw_path)
    compact = {"ad_id": ad_id, **parcel_features(preds)}
    record = {**row.to_dict(), "status": "complete",
              "modalities": modality_name, "mean_ictr": float(np.mean(ictr)),
              "max_ictr": float(np.max(ictr))}
    write_json(record_path, record)
    write_json(feature_path, compact)
    return record, compact


# ---------------------------------------------------------------------------
# Modality pipeline
# ---------------------------------------------------------------------------

def preflight(cache_dir: Path) -> dict[str, Any]:
    import torch
    from huggingface_hub import model_info, whoami
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    try:
        whoami()
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face login missing. Run 'hf auth login' and accept "
            f"access for {LLAMA_MODEL_ID}") from e
    expected = {TRIBE_MODEL_ID: TRIBE_MODEL_REVISION,
                VJEPA_MODEL_ID: VJEPA_MODEL_REVISION}
    for repo in list(expected) + [LLAMA_MODEL_ID, AUDIO_MODEL_ID]:
        rev = str(model_info(repo).sha)
        if repo in expected and rev != expected[repo]:
            raise RuntimeError(f"Revision changed for {repo}: expected "
                               f"{expected[repo]}, got {rev}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    report = dict(python=sys.version, platform=platform.platform(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=props.name, gpu_memory_gib=props.total_memory / 1024**3)
    log(json.dumps(report, indent=2))
    return report


def run_modality_pipeline(data_dir: Path, output_dir: Path, cache_dir: Path,
                          selection, modality: str, video_frames: int,
                          reconstruction_fps: int, text_device: str,
                          feature_device: str):
    import pandas as pd
    cfg = MODALITY_CONFIGS[modality]
    model, _ = load_tribe_model(cache_dir, cfg["features_to_use"],
                                video_frames, text_device, feature_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for idx, (_, row) in enumerate(selection.iterrows(), start=1):
        ad_id = str(row["ad_id"])
        log(f"[{modality}] [{idx}/{len(selection)}] ad={ad_id}")
        try:
            process_one_ad(model, row, data_dir, output_dir, video_frames,
                           reconstruction_fps, modality)
            completed += 1
        except Exception as e:
            log(f"  FAILED: {type(e).__name__}: {e}")
    return completed


def collect_results(output_dir: Path):
    import pandas as pd
    records = []
    features = []
    for p in sorted((output_dir / "records").glob("*.json")):
        r = json.loads(p.read_text())
        fpath = output_dir / "features" / f"{p.stem}.json"
        f = json.loads(fpath.read_text()) if fpath.exists() else {}
        records.append(r)
        features.append(f)
    manifest = pd.DataFrame(records)
    feature_table = pd.DataFrame(features)
    return manifest, feature_table


# ---------------------------------------------------------------------------
# Combine & cleanup
# ---------------------------------------------------------------------------

def strip_manifest(manifest):
    cols = [c for c in manifest.columns if c not in STRIP_COLS]
    return manifest[cols].copy()


def combine_modalities(ta_manifest, ta_features, vid_manifest, vid_features):
    import pandas as pd
    ta = ta_features.merge(ta_manifest[["ad_id", "roi", "cvr", "mean_ictr",
                                        "max_ictr", "split", "duration_seconds",
                                        "duration_bin", "ad_type", "product_name",
                                        "presenter", "script_tag",
                                        "editing_technique", "promotion",
                                        "item_count"]],
                           on="ad_id", how="inner")
    vid = vid_features[["ad_id"] + [c for c in vid_features.columns if c != "ad_id"]]
    combined = ta.merge(vid, on="ad_id", how="inner", suffixes=("_ta", "_vid"))
    return combined


def finalize_dataset(output_dir: Path, data_dir: Path):
    import pandas as pd
    ta_dir = output_dir / "text-audio"
    vid_dir = output_dir / "video"

    ta_manifest, ta_features = collect_results(ta_dir)
    vid_manifest, vid_features = collect_results(vid_dir)

    ta_manifest = strip_manifest(ta_manifest)
    vid_manifest = strip_manifest(vid_manifest)

    ta_manifest.to_parquet(data_dir / "text-audio.parquet", index=False)
    vid_manifest.to_parquet(data_dir / "video.parquet", index=False)

    combined = combine_modalities(ta_manifest, ta_features,
                                  vid_manifest, vid_features)
    combined.to_parquet(data_dir / "combined.parquet", index=False)

    log(f"\nFinal dataset:")
    log(f"  text-audio: {len(ta_manifest)} ads, {len(ta_manifest.columns)} cols")
    log(f"  video:      {len(vid_manifest)} ads, {len(vid_manifest.columns)} cols")
    ta_feat_cols = [c for c in combined.columns if c.endswith("_ta")]
    vid_feat_cols = [c for c in combined.columns if c.endswith("_vid")]
    log(f"  combined:   {len(combined)} ads, "
        f"{len(ta_feat_cols)} ta + {len(vid_feat_cols)} vid = "
        f"{len(combined.columns) - 1} total cols")

    shutil.rmtree(output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def require_license(args: argparse.Namespace) -> None:
    if not args.license_accepted:
        raise RuntimeError(
            "AdsTrace and TRIBE v2 are non-commercially licensed. "
            "Re-run with --license-accepted.")


def setup_environment(args: argparse.Namespace) -> int:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        raise RuntimeError("Use Python 3.11 or 3.12")
    env = args.work_dir.resolve() / ".tribe_env"
    env.parent.mkdir(parents=True, exist_ok=True)
    if not venv_python(env).exists():
        run_checked([sys.executable, "-m", "venv", str(env)])
    py = str(venv_python(env))
    run_checked([py, "-m", "pip", "install", "--upgrade", "pip"])
    run_checked([py, "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0",
                 "--index-url", args.torch_index_url])
    pkgs = [
        f"https://github.com/facebookresearch/tribev2/archive/{TRIBE_CODE_REVISION}.zip",
        "accelerate>=1.0,<2", "imageio-ffmpeg>=0.6,<0.7",
        "nilearn>=0.12,<0.13", "pandas>=2.2,<3", "pyarrow>=17", "psutil>=6",
    ]
    run_checked([py, "-m", "pip", "install", *pkgs])
    hf = env / "Scripts" / "hf.exe" if os.name == "nt" else env / "bin" / "hf"
    log("\nSetup complete. Next steps:")
    log(f'  "{hf}" auth login')
    log(f'  "{py}" scripts/generate_dataset.py run --license-accepted')
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    require_license(args)
    download_dataset(args.work_dir.resolve() / args.data_dir,
                     keep_archives=args.keep_archives)
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    require_license(args)
    preflight(args.work_dir.resolve() / args.cache_dir)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import pandas as pd
    require_license(args)
    work_dir = args.work_dir.resolve()
    data_dir = work_dir / args.data_dir
    out_dir = work_dir / "_tribe_output"
    cache_dir = work_dir / args.cache_dir

    if args.download:
        download_dataset(data_dir, keep_archives=args.keep_archives)
    validate_dataset(data_dir)
    preflight(cache_dir)
    source = read_source_table(data_dir)
    selection = stratified_sample(source, args.sample_size, args.seed)

    for modality in ("text-audio", "video"):
        mod_dir = out_dir / modality
        if list(mod_dir.glob("records/*.json")):
            log(f"[skip] {modality} already has records, delete {mod_dir} to re-run")
            continue
        log(f"\n{'='*60}\nProcessing modality: {modality}\n{'='*60}")
        run_modality_pipeline(data_dir, mod_dir, cache_dir, selection,
                              modality, args.video_frames,
                              args.reconstruction_fps, args.text_device,
                              args.feature_device)

    finalize_dataset(out_dir, data_dir)
    log("\nDone.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the TRIBE v2 AdsTrace dataset")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="Create isolated environment")
    s.add_argument("--work-dir", type=Path, default=Path("."))
    s.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu124")

    d = sub.add_parser("download", help="Download AdsTrace")
    d.add_argument("--work-dir", type=Path, default=Path("."))
    d.add_argument("--data-dir", type=Path, default=Path("data/AdsTrace"))
    d.add_argument("--keep-archives", action="store_true")
    d.add_argument("--license-accepted", action="store_true")

    pf = sub.add_parser("preflight", help="Check GPU + Hugging Face access")
    pf.add_argument("--work-dir", type=Path, default=Path("."))
    pf.add_argument("--cache-dir", type=Path, default=Path("cache/tribev2"))
    pf.add_argument("--license-accepted", action="store_true")

    r = sub.add_parser("run", help="Run full pipeline (both modalities + combine)")
    r.add_argument("--work-dir", type=Path, default=Path("."))
    r.add_argument("--data-dir", type=Path, default=Path("data/AdsTrace"))
    r.add_argument("--cache-dir", type=Path, default=Path("cache/tribev2"))
    r.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    r.add_argument("--seed", type=int, default=33)
    r.add_argument("--video-frames", type=int, choices=(8, 64), default=8)
    r.add_argument("--reconstruction-fps", type=int, default=16)
    r.add_argument("--text-device", choices=("cpu", "cuda", "accelerate"), default="cpu")
    r.add_argument("--feature-device", choices=("cpu", "cuda"), default="cuda")
    r.add_argument("--download", action="store_true")
    r.add_argument("--keep-archives", action="store_true")
    r.add_argument("--license-accepted", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "setup":
        return setup_environment(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "preflight":
        return cmd_preflight(args)
    if args.command == "run":
        return cmd_run(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; resumable.")
        raise SystemExit(130)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
