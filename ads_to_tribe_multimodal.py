from __future__ import annotations

r"""
Portable AdsTrace -> TRIBE v2 multimodal dataset builder.

This file is designed to be sent to another person and run on their computer.
It downloads AdsTrace, reconstructs the released 1-FPS visual stream, combines
it with the original 16-kHz audio and supplied Chinese transcript timestamps,
runs TRIBE v2, and pairs the resulting brain-response features with ROI, CVR,
and per-second iCTR targets.

Quick start on Windows with Python 3.12 and an NVIDIA GPU:

    py -3.12 ads_to_tribe_multimodal.py setup
    .\.tribe_env\Scripts\hf.exe auth login
    .\.tribe_env\Scripts\python.exe ads_to_tribe_multimodal.py benchmark --download --license-accepted
    .\.tribe_env\Scripts\python.exe ads_to_tribe_multimodal.py run --license-accepted

TRIBE v2 and AdsTrace are released under non-commercial licenses. Use this
program only for non-commercial work or after obtaining the needed permissions.
"""

import argparse
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


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
    SourceFile(
        "ictr.zip",
        1_172_508,
        "9414bcaf4984c6ee88082899a8f73a34b8c370c1bb6b7d7b4a4513e4c8c42a7d",
        True,
    ),
    SourceFile(
        "transcripts.zip",
        1_876_336,
        "605ebc46b2e1bd835d3dcc4d430a4cead4517351386179a92d22c951dc67eea4",
        True,
    ),
    SourceFile(
        "audios_16k.zip",
        1_851_914_587,
        "a93fcd2bdfe4f1ae62ff965a4bf9b93b66d8c8c4f7bda7c1740ff0dc370389a4",
        True,
    ),
    SourceFile(
        "frames.zip",
        8_332_717_273,
        "c37e1c9b442337a9fcbdab77dba02aafe6575b348fad4d277e07da070fcadc6c",
        True,
    ),
)


def log(message: str) -> None:
    print(message, flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=json_value),
    )


def json_value(value: Any) -> Any:
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
    display = subprocess.list2cmdline(command)
    log(f"[command] {display}")
    result = subprocess.run(command, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {display}")
    return result


def venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def setup_environment(args: argparse.Namespace) -> int:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        raise RuntimeError(
            "Run setup with Python 3.11 or 3.12. Example on Windows: "
            "'py -3.12 ads_to_tribe_multimodal.py setup'."
        )
    environment = args.work_dir.resolve() / ".tribe_env"
    environment.parent.mkdir(parents=True, exist_ok=True)
    if not venv_python(environment).exists():
        run_checked([sys.executable, "-m", "venv", str(environment)])
    python = str(venv_python(environment))
    run_checked([python, "-m", "pip", "install", "--upgrade", "pip"])
    run_checked(
        [
            python,
            "-m",
            "pip",
            "install",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "--index-url",
            args.torch_index_url,
        ]
    )
    packages = [
        (
            "https://github.com/facebookresearch/tribev2/archive/"
            f"{TRIBE_CODE_REVISION}.zip"
        ),
        "accelerate>=1.0,<2",
        "imageio-ffmpeg>=0.6,<0.7",
        "nilearn>=0.12,<0.13",
        "pandas>=2.2,<3",
        "pyarrow>=17",
        "psutil>=6",
    ]
    run_checked([python, "-m", "pip", "install", *packages])
    script = Path(__file__).resolve()
    hf_executable = (
        environment / "Scripts" / "hf.exe"
        if os.name == "nt"
        else environment / "bin" / "hf"
    )
    log("\nSetup complete.")
    log("Authenticate once so the gated Llama text model can be downloaded:")
    log(f'  "{hf_executable}" auth login')
    log("Then run the eight-ad benchmark:")
    log(
        f'  "{python}" "{script}" benchmark --download '
        "--license-accepted"
    )
    return 0


def source_url(name: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
        f"{DATASET_REVISION}/{name}?download=true"
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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

    request = urllib.request.Request(source_url(source.name))
    if current:
        request.add_header("Range", f"bytes={current}-")
    log(
        f"[download] {source.name}: {current / 1024**3:.2f} / "
        f"{source.size / 1024**3:.2f} GiB"
    )
    started = time.monotonic()
    last_report = started
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if current and error.code == 416 and current == source.size:
            response = None
        else:
            raise

    if response is not None:
        status = getattr(response, "status", response.getcode())
        mode = "ab" if current and status == 206 else "wb"
        if mode == "wb":
            current = 0
        with response, partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                current += len(chunk)
                now = time.monotonic()
                if now - last_report >= 15:
                    elapsed = max(now - started, 0.001)
                    speed = (current / 1024**2) / elapsed
                    percent = 100 * current / source.size
                    log(
                        f"  {percent:5.1f}%  {current / 1024**3:6.2f} GiB  "
                        f"{speed:6.1f} MiB/s"
                    )
                    last_report = now

    if partial.stat().st_size != source.size:
        raise RuntimeError(
            f"{source.name} size mismatch: expected {source.size}, "
            f"got {partial.stat().st_size}"
        )
    if source.sha256 and sha256_file(partial) != source.sha256:
        raise RuntimeError(f"{source.name} failed its SHA-256 check")
    os.replace(partial, destination)
    log(f"[download] complete and verified: {destination}")
    return destination


def safe_extract_zip(archive: Path, data_dir: Path) -> None:
    root = data_dir.resolve()
    log(f"[extract] {archive.name}")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (data_dir / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(
                    f"Unsafe path in {archive.name}: {member.filename}"
                )
        bundle.extractall(data_dir)


def archive_is_extracted(source: SourceFile, data_dir: Path) -> bool:
    checks = {
        "frames.zip": lambda: len(list((data_dir / "frames").glob("*/*.jpg")))
        >= EXPECTED_ADS,
        "audios_16k.zip": lambda: len(
            list((data_dir / "audios_16k").glob("*.wav"))
        )
        == EXPECTED_ADS,
        "ictr.zip": lambda: len(list((data_dir / "ictr").glob("*.csv")))
        == EXPECTED_ADS,
        "transcripts.zip": lambda: len(
            list((data_dir / "transcripts").glob("*.json"))
        )
        == EXPECTED_ADS,
    }
    return checks[source.name]() if source.name in checks else False


def validate_dataset(data_dir: Path) -> dict[str, Any]:
    import csv

    with (data_dir / "tags_cn.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    ids = {str(row["ID"]) for row in rows}
    if len(rows) != EXPECTED_ADS or len(ids) != EXPECTED_ADS:
        raise RuntimeError(
            f"Expected {EXPECTED_ADS} unique metadata rows; found "
            f"{len(rows)} rows and {len(ids)} IDs"
        )
    folder_ids = {
        "frames": {
            path.name for path in (data_dir / "frames").iterdir() if path.is_dir()
        },
        "audios_16k": {
            path.stem for path in (data_dir / "audios_16k").glob("*.wav")
        },
        "transcripts": {
            path.stem for path in (data_dir / "transcripts").glob("*.json")
        },
        "ictr": {path.stem for path in (data_dir / "ictr").glob("*.csv")},
    }
    for name, available in folder_ids.items():
        missing = ids - available
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} ad IDs")
    splits = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    expected_splits = {"train": 2_266, "val": 283, "test": 284}
    split_counts = {name: len(values) for name, values in splits.items()}
    if split_counts != expected_splits:
        raise RuntimeError(f"Unexpected official splits: {split_counts}")
    split_ids = {str(value) for values in splits.values() for value in values}
    if split_ids != ids:
        raise RuntimeError("split.json IDs do not match tags_cn.csv")
    report = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "num_ads": len(ids),
        "split_counts": split_counts,
        "modalities": ["text", "audio", "video"],
        "validated_at_unix": time.time(),
    }
    write_json(data_dir / "download_validation.json", report)
    log(f"[validate] complete: {len(ids)} ads")
    return report


def download_dataset(data_dir: Path, keep_archives: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(data_dir).free / 1024**3
    if free_gib < 24:
        raise RuntimeError(
            f"At least 24 GiB free is required; only {free_gib:.1f} GiB is free"
        )
    log(f"[preflight] data={data_dir.resolve()} free={free_gib:.1f} GiB")
    for source in SOURCE_FILES:
        destination = data_dir / source.name
        if source.archive and archive_is_extracted(source, data_dir):
            log(f"[extract] already complete: {source.name}")
            continue
        download_with_resume(source, destination)
        if source.archive:
            safe_extract_zip(destination, data_dir)
            if not archive_is_extracted(source, data_dir):
                raise RuntimeError(f"Extraction validation failed: {source.name}")
            if not keep_archives:
                destination.unlink()
                log(f"[cleanup] removed archive {destination.name}")
    validate_dataset(data_dir)


def require_license_acknowledgement(args: argparse.Namespace) -> None:
    if not args.license_accepted:
        raise RuntimeError(
            "AdsTrace and TRIBE v2 are non-commercially licensed. Re-run with "
            "--license-accepted only if this is non-commercial work or you have "
            "obtained the required permissions."
        )


def preflight_models(cache_dir: Path) -> dict[str, Any]:
    import torch
    from huggingface_hub import model_info, whoami

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install the CUDA PyTorch build and use an NVIDIA "
            "GPU. CPU-only processing is not practical for this workload."
        )
    identity: dict[str, Any] | None
    try:
        identity = whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face login is missing. Run 'hf auth login', accept access "
            f"for {LLAMA_MODEL_ID}, and retry."
        ) from error
    repos = [
        TRIBE_MODEL_ID,
        VJEPA_MODEL_ID,
        LLAMA_MODEL_ID,
        AUDIO_MODEL_ID,
    ]
    revisions: dict[str, str] = {}
    for repo in repos:
        try:
            revisions[repo] = str(model_info(repo).sha)
        except Exception as error:
            raise RuntimeError(
                f"Cannot access {repo}. Check the Hugging Face license/access page "
                "and your login."
            ) from error
    expected = {
        TRIBE_MODEL_ID: TRIBE_MODEL_REVISION,
        VJEPA_MODEL_ID: VJEPA_MODEL_REVISION,
    }
    changed = {
        repo: {"expected": expected[repo], "actual": revisions[repo]}
        for repo in expected
        if revisions[repo] != expected[repo]
    }
    if changed:
        raise RuntimeError(
            "Pinned model revisions changed upstream. Review before updating this "
            f"file: {changed}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "gpu_memory_gib": props.total_memory / 1024**3,
        "huggingface_user": identity.get("name", "authenticated"),
        "revisions": revisions,
    }
    log(json.dumps(report, indent=2))
    return report


def duration_from_ictr(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_source_table(data_dir: Path):
    import pandas as pd

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
    splits = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    lookup = {
        str(ad_id): split for split, values in splits.items() for ad_id in values
    }
    tags["split"] = tags["ad_id"].map(lookup)
    tags["duration_seconds"] = tags["ad_id"].map(
        lambda ad_id: duration_from_ictr(data_dir / "ictr" / f"{ad_id}.csv")
    )
    tags["duration_bin"] = pd.qcut(
        tags["duration_seconds"],
        q=3,
        labels=["short", "medium", "long"],
        duplicates="drop",
    ).astype(str)
    if tags["split"].isna().any():
        raise RuntimeError("Some metadata IDs are absent from split.json")
    return tags


def allocate_group_counts(sizes, target: int) -> dict[Any, int]:
    import numpy as np

    if target > int(sizes.sum()):
        raise ValueError("Sample target exceeds available rows")
    raw = sizes / sizes.sum() * target
    allocated = np.floor(raw).astype(int)
    ensure_coverage = target >= len(sizes)
    if ensure_coverage:
        allocated = allocated.clip(lower=1)
    minimum = 1 if ensure_coverage else 0
    while int(allocated.sum()) > target:
        candidates = [key for key in allocated.index if allocated[key] > minimum]
        key = min(candidates, key=lambda item: (raw[item] - allocated[item], str(item)))
        allocated[key] -= 1
    while int(allocated.sum()) < target:
        candidates = [key for key in allocated.index if allocated[key] < sizes[key]]
        key = max(candidates, key=lambda item: (raw[item] - allocated[item], str(item)))
        allocated[key] += 1
    return {key: int(value) for key, value in allocated.items()}


def split_counts_for_sample(sample_size: int) -> dict[str, int]:
    if sample_size == DEFAULT_SAMPLE_SIZE:
        return dict(DEFAULT_SPLIT_COUNTS)
    train = round(sample_size * 0.8)
    validation = round(sample_size * 0.1)
    return {
        "train": train,
        "val": validation,
        "test": sample_size - train - validation,
    }


def stratified_sample(table, sample_size: int, seed: int):
    import pandas as pd

    selected = []
    for split, target in split_counts_for_sample(sample_size).items():
        candidates = table[table["split"] == split].copy()
        candidates["_stratum"] = list(
            zip(candidates["ad_type"].fillna(""), candidates["duration_bin"])
        )
        sizes = candidates.groupby("_stratum", sort=True).size()
        allocations = allocate_group_counts(sizes, target)
        parts = []
        for index, (stratum, count) in enumerate(
            sorted(allocations.items(), key=str)
        ):
            group = candidates[candidates["_stratum"] == stratum]
            parts.append(
                group.sample(n=count, random_state=seed + index, replace=False)
            )
        chosen = pd.concat(parts, ignore_index=True).drop(columns="_stratum")
        if len(chosen) != target:
            raise RuntimeError(f"Incorrect {split} sample size: {len(chosen)}")
        selected.append(chosen)
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["split", "ad_id"])
        .reset_index(drop=True)
    )


def select_benchmark(selection, count: int, seed: int):
    import pandas as pd

    if count < 3:
        raise ValueError("Benchmark count must be at least 3")
    frequencies = (
        selection.groupby("ad_type")
        .size()
        .sort_values(ascending=False, kind="stable")
    )
    chosen = []
    used: set[str] = set()
    for ad_type in list(frequencies.index[: max(1, count - 2)]):
        group = selection[selection["ad_type"] == ad_type]
        median = group["duration_seconds"].median()
        row = group.iloc[(group["duration_seconds"] - median).abs().argsort().iloc[0]]
        chosen.append(row)
        used.add(str(row["ad_id"]))
    remaining = selection[~selection["ad_id"].isin(used)]
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
        used.add(str(row["ad_id"]))
        remaining = remaining[remaining["ad_id"] != row["ad_id"]]
    if len(chosen) < count:
        fill = remaining.sample(n=count - len(chosen), random_state=seed)
        chosen.extend(row for _, row in fill.iterrows())
    return pd.DataFrame(chosen[:count]).reset_index(drop=True)


def locate_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise RuntimeError("ffmpeg was not found; re-run the setup command")


def frame_sort_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else 0, path.name)


def ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def reconstruct_silent_video(
    data_dir: Path,
    ad_id: str,
    destination: Path,
    frames_per_second: int,
) -> None:
    frames = sorted(
        (data_dir / "frames" / ad_id).glob("*.jpg"),
        key=frame_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"No visual frames found for {ad_id}")
    concat_path = destination.with_suffix(".frames.txt")
    lines = []
    for frame in frames:
        lines.append(f"file '{ffmpeg_concat_path(frame)}'")
        lines.append("duration 1")
    lines.append(f"file '{ffmpeg_concat_path(frames[-1])}'")
    atomic_write_text(concat_path, "\n".join(lines) + "\n")
    command = [
        locate_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={frames_per_second}",
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
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                f"ffmpeg failed for {ad_id}: {result.stderr.strip()}"
            )
    finally:
        concat_path.unlink(missing_ok=True)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def transcript_word_events(transcript_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    history: deque[str] = deque(maxlen=1024)
    for sequence_id, segment in enumerate(payload.get("segments", [])):
        start = float(segment["start"])
        end = float(segment["end"])
        characters = [char for char in str(segment.get("text", "")) if not char.isspace()]
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


def make_multimodal_events(
    video_path: Path,
    audio_path: Path,
    transcript_path: Path,
):
    import pandas as pd
    from neuralset.events.utils import standardize_events

    base = {
        "start": 0.0,
        "timeline": "default",
        "subject": "default",
    }
    events = [
        {
            **base,
            "type": "Video",
            "filepath": str(video_path),
        },
        {
            **base,
            "type": "Audio",
            "filepath": str(audio_path),
            "duration": wav_duration(audio_path),
        },
        *transcript_word_events(transcript_path),
    ]
    return standardize_events(pd.DataFrame(events))


def load_tribe_model(
    cache_dir: Path,
    video_frames: int,
    text_device: str,
    feature_device: str,
):
    import torch
    from tribev2.demo_utils import TribeModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner")
    config_update = {
        "data.features_to_use": ["text", "audio", "video"],
        "data.batch_size": 1,
        "data.num_workers": 0,
        "data.shuffle_train": False,
        "data.shuffle_val": False,
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
    model = TribeModel.from_pretrained(
        TRIBE_MODEL_ID,
        cache_folder=cache_dir,
        cluster=None,
        device="cuda",
        config_update=config_update,
    )
    return model, time.perf_counter() - started


def segment_times(segments: Sequence[Any]):
    import numpy as np

    starts = [
        float(getattr(segment, "start", index))
        for index, segment in enumerate(segments)
    ]
    durations = [
        float(getattr(segment, "duration", 1.0)) for segment in segments
    ]
    return (
        np.asarray(starts, dtype=np.float32),
        np.asarray(durations, dtype=np.float32),
    )


def read_ictr(data_dir: Path, ad_id: str):
    import numpy as np
    import pandas as pd

    table = pd.read_csv(data_dir / "ictr" / f"{ad_id}.csv")
    return (
        table["sec"].to_numpy(dtype=np.float32),
        table["ictr"].to_numpy(dtype=np.float32),
    )


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_raw_output(path: Path) -> tuple[int, int]:
    import numpy as np

    with np.load(path) as output:
        predictions = output["brain_activity"]
        starts = output["segment_start_seconds"]
        if predictions.ndim != 2 or predictions.shape[1] != EXPECTED_VERTICES:
            raise RuntimeError(f"Unexpected TRIBE shape: {predictions.shape}")
        if len(starts) != predictions.shape[0] or np.any(np.diff(starts) < 0):
            raise RuntimeError(f"Invalid timestamps in {path}")
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"Non-finite predictions in {path}")
        return int(predictions.shape[0]), int(predictions.shape[1])


def clean_feature_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "unknown"


def parcel_features(predictions) -> dict[str, float]:
    import numpy as np
    from nilearn.datasets import fetch_atlas_surf_destrieux

    atlas = fetch_atlas_surf_destrieux(verbose=0)
    labels = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in atlas.labels
    ]
    features: dict[str, float] = {}
    hemispheres = (
        ("lh", np.asarray(atlas.map_left), predictions[:, :10_242]),
        ("rh", np.asarray(atlas.map_right), predictions[:, 10_242:]),
    )
    for hemisphere, mapping, values in hemispheres:
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
            prefix = f"{hemisphere}_{clean_feature_name(label)}"
            features[f"{prefix}__mean"] = float(np.mean(series))
            features[f"{prefix}__std"] = float(np.std(series))
            features[f"{prefix}__max"] = float(np.max(series))
            features[f"{prefix}__p95"] = float(np.percentile(series, 95))
            features[f"{prefix}__slope"] = slope
            features[f"{prefix}__peak_fraction"] = peak_fraction
    return features


def process_one_ad(
    model: Any,
    row: Any,
    data_dir: Path,
    output_dir: Path,
    video_frames: int,
    reconstruction_fps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    ad_id = str(row["ad_id"])
    raw_path = output_dir / "raw" / f"{ad_id}.npz"
    record_path = output_dir / "records" / f"{ad_id}.json"
    feature_path = output_dir / "feature_records" / f"{ad_id}.json"
    if raw_path.exists() and record_path.exists() and feature_path.exists():
        validate_raw_output(raw_path)
        return (
            json.loads(record_path.read_text(encoding="utf-8")),
            json.loads(feature_path.read_text(encoding="utf-8")),
        )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix=f"tribe_{ad_id}_") as temporary:
        video_path = Path(temporary) / f"{ad_id}.mp4"
        reconstruction_started = time.perf_counter()
        reconstruct_silent_video(
            data_dir,
            ad_id,
            video_path,
            reconstruction_fps,
        )
        reconstruction_seconds = time.perf_counter() - reconstruction_started
        events = make_multimodal_events(
            video_path,
            data_dir / "audios_16k" / f"{ad_id}.wav",
            data_dir / "transcripts" / f"{ad_id}.json",
        )
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
    record = {
        **row.to_dict(),
        "status": "complete",
        "modalities": "text,audio,video",
        "profile": f"multimodal_video{video_frames}",
        "runtime_seconds": time.perf_counter() - started,
        "reconstruction_seconds": reconstruction_seconds,
        "inference_seconds": inference_seconds,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "tribe_timesteps": timesteps,
        "tribe_vertices": vertices,
        "mean_ictr": float(np.mean(ictr)),
        "max_ictr": float(np.max(ictr)),
        "raw_path": str(raw_path.relative_to(output_dir)),
        "error": "",
    }
    write_json(record_path, record)
    write_json(feature_path, compact)
    return record, compact


def write_parquet_atomic(table: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def consolidate_outputs(output_dir: Path, selection: Any) -> tuple[Any, Any]:
    import pandas as pd

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "records").glob("*.json"))
    ]
    features = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "feature_records").glob("*.json"))
    ]
    manifest = pd.DataFrame(records)
    feature_table = pd.DataFrame(features)
    if not manifest.empty:
        write_parquet_atomic(manifest, output_dir / "manifest.parquet")
    if not feature_table.empty:
        write_parquet_atomic(
            feature_table,
            output_dir / "compact_features.parquet",
        )
        write_json(
            output_dir / "feature_schema.json",
            [column for column in feature_table.columns if column != "ad_id"],
        )
    write_parquet_atomic(selection, output_dir / "selection.parquet")
    return manifest, feature_table


def append_failure(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=json_value) + "\n")


def estimate_runtime(successful: Any, selection: Any) -> dict[str, Any]:
    import numpy as np

    rates = (
        successful["runtime_seconds"].astype(float)
        / successful["duration_seconds"].astype(float).clip(lower=1)
    )
    total_content_seconds = float(selection["duration_seconds"].sum())
    return {
        "successful_ads": int(len(successful)),
        "median_seconds_per_content_second": float(np.median(rates)),
        "p90_seconds_per_content_second": float(np.percentile(rates, 90)),
        "estimated_full_hours_median": float(
            np.median(rates) * total_content_seconds / 3600
        ),
        "estimated_full_hours_p90": float(
            np.percentile(rates, 90) * total_content_seconds / 3600
        ),
        "selected_ads": int(len(selection)),
        "selected_content_hours": total_content_seconds / 3600,
    }


def run_pipeline(args: argparse.Namespace, benchmark: bool) -> int:
    import pandas as pd

    require_license_acknowledgement(args)
    work_dir = args.work_dir.resolve()
    data_dir = (work_dir / args.data_dir).resolve()
    output_dir = (work_dir / args.output_dir).resolve()
    cache_dir = (work_dir / args.cache_dir).resolve()
    if args.download:
        download_dataset(data_dir, keep_archives=args.keep_archives)
    validate_dataset(data_dir)
    machine = preflight_models(cache_dir)
    source = read_source_table(data_dir)
    selection = stratified_sample(source, args.sample_size, args.seed)
    work = (
        select_benchmark(selection, args.benchmark_count, args.seed)
        if benchmark
        else selection
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "created_at_unix": time.time(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tribe_code_revision": TRIBE_CODE_REVISION,
        "tribe_model_revision": TRIBE_MODEL_REVISION,
        "vjepa_model_revision": VJEPA_MODEL_REVISION,
        "features_to_use": ["text", "audio", "video"],
        "transcript_alignment": "AdsTrace sentence times interpolated over Chinese characters",
        "video_limitation": "Reconstructed from released 1-FPS frames; not original MP4",
        "license_acknowledged": True,
        "machine": machine,
        "arguments": vars(args),
    }
    write_json(output_dir / "provenance.json", provenance)
    write_parquet_atomic(selection, output_dir / "selection.parquet")

    model, model_load_seconds = load_tribe_model(
        cache_dir,
        args.video_frames,
        args.text_device,
        args.feature_device,
    )
    log(
        f"[model] loaded in {model_load_seconds / 60:.1f} minutes; "
        f"processing {len(work)} ads"
    )
    completed = 0
    failures = 0
    for index, (_, row) in enumerate(work.iterrows(), start=1):
        ad_id = str(row["ad_id"])
        log(f"[{index}/{len(work)}] ad={ad_id} type={row['ad_type']}")
        try:
            record, _ = process_one_ad(
                model,
                row,
                data_dir,
                output_dir,
                args.video_frames,
                args.reconstruction_fps,
            )
            completed += 1
            log(
                f"  complete in {float(record['runtime_seconds']) / 60:.1f} min; "
                f"peak CUDA {float(record['peak_cuda_memory_gib']):.1f} GiB"
            )
        except Exception as error:
            failures += 1
            append_failure(
                output_dir / "failures.jsonl",
                {
                    "ad_id": ad_id,
                    "time_unix": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            log(f"  FAILED: {type(error).__name__}: {error}")
        if index % 5 == 0 or index == len(work):
            consolidate_outputs(output_dir, selection)

    manifest, _ = consolidate_outputs(output_dir, selection)
    report = {
        "mode": "benchmark" if benchmark else "run",
        "attempted_this_invocation": int(len(work)),
        "completed_this_invocation": completed,
        "failures_this_invocation": failures,
        "model_load_seconds": model_load_seconds,
    }
    if not manifest.empty:
        successful_ids = set(work["ad_id"].astype(str)) & set(
            manifest["ad_id"].astype(str)
        )
        successful = manifest[manifest["ad_id"].astype(str).isin(successful_ids)]
        if len(successful) >= 2:
            report["runtime_estimate"] = estimate_runtime(successful, selection)
    report_name = "benchmark.json" if benchmark else "run_report.json"
    write_json(output_dir / report_name, report)
    log(json.dumps(report, indent=2, default=json_value))
    return 0 if failures == 0 else 1


def add_shared_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=Path("data/AdsTrace"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/AdsTraceTribeV2_multimodal_1500"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/tribev2"))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--video-frames", type=int, choices=(8, 64), default=8)
    parser.add_argument("--reconstruction-fps", type=int, default=16)
    parser.add_argument(
        "--text-device",
        choices=("cpu", "cuda", "accelerate"),
        default="cpu",
        help="CPU is safest for the 3B text model on GPUs with 8-12 GiB VRAM.",
    )
    parser.add_argument(
        "--feature-device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device for audio and video feature extraction.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and validate all AdsTrace modalities before processing.",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--license-accepted",
        action="store_true",
        help="Confirm non-commercial use or that required permissions were obtained.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portable AdsTrace -> TRIBE v2 text+audio+video builder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create an isolated environment")
    setup.add_argument("--work-dir", type=Path, default=Path("."))
    setup.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu124",
    )

    download = subparsers.add_parser(
        "download", help="Download and validate full AdsTrace"
    )
    download.add_argument("--work-dir", type=Path, default=Path("."))
    download.add_argument("--data-dir", type=Path, default=Path("data/AdsTrace"))
    download.add_argument("--keep-archives", action="store_true")
    download.add_argument("--license-accepted", action="store_true")

    preflight = subparsers.add_parser(
        "preflight", help="Check CUDA, Hugging Face access, and model revisions"
    )
    preflight.add_argument("--work-dir", type=Path, default=Path("."))
    preflight.add_argument("--cache-dir", type=Path, default=Path("cache/tribev2"))
    preflight.add_argument("--license-accepted", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="Run a representative tiny batch first"
    )
    add_shared_pipeline_arguments(benchmark)
    benchmark.add_argument("--benchmark-count", type=int, default=8)

    run = subparsers.add_parser("run", help="Run/resume the selected full dataset")
    add_shared_pipeline_arguments(run)
    run.set_defaults(benchmark_count=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "setup":
        return setup_environment(args)
    if args.command == "download":
        require_license_acknowledgement(args)
        data_dir = (args.work_dir.resolve() / args.data_dir).resolve()
        download_dataset(data_dir, keep_archives=args.keep_archives)
        return 0
    if args.command == "preflight":
        require_license_acknowledgement(args)
        cache_dir = (args.work_dir.resolve() / args.cache_dir).resolve()
        preflight_models(cache_dir)
        return 0
    if args.command == "benchmark":
        return run_pipeline(args, benchmark=True)
    if args.command == "run":
        return run_pipeline(args, benchmark=False)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely; downloads and ad outputs are resumable.")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
