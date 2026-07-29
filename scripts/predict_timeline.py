#!/usr/bin/env python3
"""
CLI Script: Predict Second-by-Second Click-Through Rate Timeline for an Ad Video.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.dataset import load_combined_data
from model.timeline import TimelinePredictor


def main():
    parser = argparse.ArgumentParser(description="Predict Second-by-Second CTR Timeline")
    parser.add_argument("--sample-index", type=int, default=5, help="Row index in combined.parquet to test")
    args = parser.parse_args()

    df = load_combined_data()
    sample_row = df.iloc[[args.sample_index]]
    ad_id = sample_row["ad_id"].values[0]
    duration = int(sample_row["duration_seconds"].values[0]) if "duration_seconds" in sample_row.columns else 15

    print("=" * 80)
    print(f"SECOND-BY-SECOND CLICK TIMELINE PREDICTION FOR AD ID: '{ad_id}' (Duration: {duration}s)")
    print("=" * 80)

    brain_cols = [c for c in sample_row.columns if c.endswith("_ta") or c.endswith("_vid")]
    brain_inputs = sample_row[brain_cols]

    timeline_predictor = TimelinePredictor()
    res = timeline_predictor.predict_timeline(brain_inputs, duration_seconds=duration)

    print(f"\n---> Predicted Peak CTR Second: Second {res['peak_second']} ({res['max_ictr']*100:.2f}% Peak CTR)")
    print(f"---> Predicted Mean CTR        : {res['mean_ictr']*100:.2f}%\n")

    print(f"{'Second':<8} | {'Predicted Click Rate (%)':<26} | {'Creative Phase':<25}")
    print("-" * 65)

    df_t = res["timeline_df"]
    for _, row in df_t.iterrows():
        sec = int(row["second"])
        ctr_str = row["predicted_ctr_pct"]
        phase = row["creative_phase"]
        marker = " ◄ PEAK ATTENTION!" if sec == res["peak_second"] else " ◄ ATTENTION DROP!" if sec in res["drop_off_seconds"] else ""
        print(f"{sec:2d}s      | {ctr_str:<26} | {phase:<25}{marker}")

    print("\n--- CREATIVE OPTIMIZATION RECOMMENDATIONS ---")
    for rec in res["recommendations"]:
        print(f"  • {rec}")
    print("=" * 80)


if __name__ == "__main__":
    main()
