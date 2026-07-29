#!/usr/bin/env python3
"""Score one ad video with TRIBE v2 and the video-only average-iCTR model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dopa_api.scoring import VideoAdScorer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            REPO_ROOT
            / "benchmark"
            / "saved_models"
            / "mean_ictr_brain_video_ensemble.joblib"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "cache" / "tribev2",
    )
    parser.add_argument("--video-frames", type=int, choices=(8, 64), default=8)
    args = parser.parse_args()

    scorer = VideoAdScorer(
        model_path=args.model,
        cache_dir=args.cache_dir,
        video_frames=args.video_frames,
    )
    result = scorer.score(args.video)
    payload = {
        "percentage": result.percentage,
        "raw_mean_ictr": result.raw_mean_ictr,
        "top_regions": [asdict(region) for region in result.top_regions],
        "brain_timesteps": result.brain_timesteps,
        "compact_features": result.compact_features,
        "model_load_seconds": result.model_load_seconds,
        "inference_seconds": result.inference_seconds,
        "peak_vram_mib": result.peak_vram_mib,
        "process_rss_mib": result.process_rss_mib,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
