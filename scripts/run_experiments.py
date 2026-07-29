#!/usr/bin/env python3
"""
Full Benchmark Runner across all Ad Metrics, Brain Modalities, and Model Architectures.
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TARGET_COLUMNS, MODALITIES, BENCHMARK_DIR
from model.dataset import load_combined_data
from model.train import train_single_target_model


def run_full_benchmark():
    df = load_combined_data()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_report = []
    top_features_all = []

    model_architectures = ["ridge", "hgb", "extra_trees", "ensemble"]

    print("=" * 90)
    print("STARTING FULL TRIBE V2 AD METRIC PREDICTION BENCHMARK")
    print("=" * 90)

    for target in TARGET_COLUMNS:
        for modality in MODALITIES:
            for model_type in model_architectures:
                print(f"Running: Target={target:10s} | Modality={modality:18s} | Model={model_type:12s} ...", end=" ", flush=True)

                res = train_single_target_model(
                    df=df,
                    target_name=target,
                    modality=modality,
                    model_type=model_type,
                    k_features=150,
                    alpha=200.0,
                    save_model=(modality == "brain_combined" and model_type == "ensemble")
                )

                rec = {
                    "target": target,
                    "modality": modality,
                    "model_type": model_type,
                    "val_r2": res["val_metrics"]["r2"],
                    "val_pearson_r": res["val_metrics"]["pearson_r"],
                    "val_spearman_r": res["val_metrics"]["spearman_r"],
                    "val_accuracy_2std": res["val_metrics"]["accuracy_within_2std"],
                    "test_r2": res["test_metrics"]["r2"],
                    "test_pearson_r": res["test_metrics"]["pearson_r"],
                    "test_spearman_r": res["test_metrics"]["spearman_r"],
                    "test_accuracy_2std": res["test_metrics"]["accuracy_within_2std"],
                    "test_tier_accuracy": res["test_metrics"]["tier_classification_accuracy"],
                }
                benchmark_report.append(rec)
                print(f"Done! Test Pearson r={rec['test_pearson_r']:.4f}, Test Acc(2std)={rec['test_accuracy_2std']:.2f}%")

                if modality == "brain_combined" and model_type == "ensemble":
                    top_df = res["top_features"].copy()
                    top_df["target"] = target
                    top_features_all.append(top_df)

    # Save results
    report_path = BENCHMARK_DIR / "benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    # Save feature importances
    if top_features_all:
        feat_df = pd.concat(top_features_all, axis=0, ignore_index=True)
        feat_df.to_csv(BENCHMARK_DIR / "feature_importance.csv", index=False)

    print("\n" + "=" * 90)
    print(f"BENCHMARK COMPLETE! Results saved to '{report_path}'")
    print("=" * 90)


if __name__ == "__main__":
    run_full_benchmark()
