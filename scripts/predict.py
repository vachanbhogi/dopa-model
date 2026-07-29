#!/usr/bin/env python3
"""
Production Inference Script: Predict Ad Metrics from TRIBE v2 fMRI Outputs.
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.dataset import load_combined_data
from model.inference import AdMetricPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict Ad Metrics from TRIBE v2 Output")
    parser.add_argument("--ad-id", type=str, default=None, help="Sample ad ID from dataset to test inference")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index in combined.parquet if ad-id not specified")
    args = parser.parse_args()

    df = load_combined_data()
    if args.ad_id is not None:
        sample_row = df[df["ad_id"] == args.ad_id]
        if sample_row.empty:
            print(f"Error: Ad ID '{args.ad_id}' not found in dataset.", file=sys.stderr)
            sys.exit(1)
    else:
        sample_row = df.iloc[[args.sample_index]]

    ad_id = sample_row["ad_id"].values[0]
    print(f"=== Production Ad Metric Inference for Ad ID: '{ad_id}' ===")

    # Initialize Predictor
    predictor = AdMetricPredictor()

    # Extract 1,800 brain features (*_ta and *_vid)
    brain_cols = [c for c in sample_row.columns if c.endswith("_ta") or c.endswith("_vid")]
    brain_inputs = sample_row[brain_cols]

    # Predict
    predicted_metrics = predictor.predict(brain_inputs)

    print("\n--- Model Predictions vs Ground Truth ---")
    print(f"{'Metric':<15} | {'Predicted Value':<18} | {'Ground Truth Value':<18}")
    print("-" * 58)

    for metric in ["roi", "cvr", "mean_ictr", "max_ictr"]:
        pred_val = predicted_metrics[metric]
        gt_val = sample_row[metric].values[0] if metric in sample_row.columns else None
        gt_str = f"{gt_val:.4f}" if (gt_val is not None and not pd.isna(gt_val)) else "N/A"
        print(f"{metric.upper():<15} | {pred_val:18.4f} | {gt_str:<18}")


if __name__ == "__main__":
    main()
