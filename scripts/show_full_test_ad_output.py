#!/usr/bin/env python3
"""
Run Full Production Model Outputs on a Real Ad from the Held-Out TEST Dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.dataset import load_combined_data
from model.inference import AdMetricPredictor
from model.timeline import TimelinePredictor
from model.advanced_strategies import OrdinalTierClassifier
from model.interpret import extract_top_brain_features


def main():
    df = load_combined_data()

    # Filter strictly for held-out test split
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        print("Error: No test split ads found.")
        sys.exit(1)

    # Pick a representative test sample (e.g. 10th test ad)
    sample_row = test_df.iloc[[10]]
    ad_id = str(sample_row["ad_id"].values[0])
    product_name = str(sample_row["product_name"].values[0]) if "product_name" in sample_row.columns else "N/A"
    ad_type = str(sample_row["ad_type"].values[0]) if "ad_type" in sample_row.columns else "N/A"
    duration = int(sample_row["duration_seconds"].values[0]) if "duration_seconds" in sample_row.columns else 20

    print("=" * 95)
    print(f"FULL PRODUCTION MODEL OUTPUT FOR HELD-OUT TEST AD ID: '{ad_id}'")
    print(f"Split: HELD-OUT TEST (Never seen during training) | Product: '{product_name}' | Type: '{ad_type}' | Duration: {duration}s")
    print("=" * 95)

    # 1. Primary Continuous Metrics (ROI, CVR, MEAN_ICTR, MAX_ICTR)
    predictor = AdMetricPredictor()
    brain_cols = [c for c in sample_row.columns if c.endswith("_ta") or c.endswith("_vid")]
    brain_inputs = sample_row[brain_cols]

    predicted_metrics = predictor.predict(brain_inputs)

    print("\n--- 1. PREDICTED AD METRICS VS GROUND TRUTH ---")
    print(f"{'Metric Name':<20} | {'Model Prediction':<20} | {'Ground Truth (Actual)':<22} | {'Prediction Accuracy':<20}")
    print("-" * 90)

    for metric in ["roi", "cvr", "mean_ictr", "max_ictr"]:
        pred_val = predicted_metrics[metric]
        gt_val = sample_row[metric].values[0] if metric in sample_row.columns else None
        
        gt_str = f"{gt_val:.4f}" if (gt_val is not None and not pd.isna(gt_val)) else "N/A"
        
        if gt_val is not None and not pd.isna(gt_val) and gt_val > 0:
            rel_err = abs(pred_val - gt_val) / max(gt_val, 1e-4)
            acc_str = f"{max(0.0, 1.0 - rel_err)*100:.1f}% Relative Match"
        else:
            acc_str = "N/A"

        print(f"{metric.upper():<20} | {pred_val:20.4f} | {gt_str:<22} | {acc_str:<20}")

    # 2. 5-Tier Performance Quintile Rating
    print("\n--- 2. PREDICTED 5-TIER AD PERFORMANCE RATING ---")
    ordinal_model = OrdinalTierClassifier(n_tiers=5, k_features=150, random_state=42)
    
    # Train ordinal classifier on train split
    train_df = df[df["split"] == "train"]
    ordinal_model.fit(train_df[brain_cols], train_df["roi"])
    
    tier_pred = int(ordinal_model.predict(brain_inputs)[0]) + 1
    tier_labels = {1: "Tier 1 (Flop - Bottom 20%)", 2: "Tier 2 (Below Average)", 3: "Tier 3 (Average Performer)", 4: "Tier 4 (Strong Performer)", 5: "Tier 5 (Viral Winner - Top 20%)"}
    print(f"Predicted Performance Rating : {tier_labels.get(tier_pred, f'Tier {tier_pred}')}")

    # 3. Second-by-Second Click Timeline
    print("\n--- 3. SECOND-BY-SECOND PREDICTED CLICK TIMELINE ---")
    timeline_predictor = TimelinePredictor(predictor=predictor)
    timeline_res = timeline_predictor.predict_timeline(brain_inputs, duration_seconds=duration)

    print(f"Peak Attention Second : Second {timeline_res['peak_second']} ({timeline_res['max_ictr']*100:.2f}% Peak CTR)")
    print(f"Average Click Rate    : {timeline_res['mean_ictr']*100:.2f}%\n")

    print(f"{'Second':<8} | {'Predicted Click Rate (%)':<26} | {'Creative Phase':<25}")
    print("-" * 65)

    df_t = timeline_res["timeline_df"]
    for _, row in df_t.iterrows():
        sec = int(row["second"])
        ctr_str = row["predicted_ctr_pct"]
        phase = row["creative_phase"]
        marker = " ◄ PEAK ATTENTION!" if sec == timeline_res["peak_second"] else " ◄ ATTENTION DROP!" if sec in timeline_res["drop_off_seconds"] else ""
        print(f"{sec:2d}s      | {ctr_str:<26} | {phase:<25}{marker}")

    # 4. Top Brain ROI Predictors
    print("\n--- 4. TOP CORTICAL BRAIN SURFACE REGIONS DRIVING PREDICTION ---")
    model_roi = predictor.models["roi"]
    top_brain_df = extract_top_brain_features(model_roi, list(train_df[brain_cols].columns), top_n=5)
    print(top_brain_df[["feature", "importance", "hemisphere", "anatomical_region", "statistic", "modality"]].to_string(index=False))

    # 5. Creative Recommendations
    print("\n--- 5. AUTOMATED CREATIVE RECOMMENDATIONS ---")
    for rec in timeline_res["recommendations"]:
        print(f"  • {rec}")
    print("=" * 95)


if __name__ == "__main__":
    main()
