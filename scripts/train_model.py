#!/usr/bin/env python3
"""
CLI Script to train and evaluate an ad metric prediction model on TRIBE v2 brain features.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TARGET_COLUMNS, MODALITIES
from model.dataset import load_combined_data
from model.train import train_single_target_model


def main():
    parser = argparse.ArgumentParser(description="Train TRIBE v2 Ad Metric Model")
    parser.add_argument("--target", type=str, default="roi", choices=TARGET_COLUMNS, help="Target ad metric")
    parser.add_argument("--modality", type=str, default="brain_combined", choices=MODALITIES, help="Feature modality configuration")
    parser.add_argument("--model-type", type=str, default="ensemble", choices=["ridge", "elasticnet", "hgb", "extra_trees", "mlp", "ensemble"], help="Model architecture")
    parser.add_argument("--k-features", type=int, default=150, help="Number of top features to select")
    parser.add_argument("--alpha", type=float, default=200.0, help="L2 Regularization parameter alpha")
    args = parser.parse_args()

    print(f"=== Training Model for Target: '{args.target}' ===")
    print(f"Modality: {args.modality} | Architecture: {args.model_type} | Top Features: {args.k_features} | Alpha: {args.alpha}")

    df = load_combined_data()
    res = train_single_target_model(
        df=df,
        target_name=args.target,
        modality=args.modality,
        model_type=args.model_type,
        k_features=args.k_features,
        alpha=args.alpha,
        save_model=True
    )

    print("\n--- Validation Metrics ---")
    for k, v in res["val_metrics"].items():
        print(f"  {k:30s}: {v:.4f}")

    print("\n--- Test Metrics ---")
    for k, v in res["test_metrics"].items():
        print(f"  {k:30s}: {v:.4f}")

    print("\n--- Top 10 Cortical Brain Regions ---")
    print(res["top_features"][["feature", "importance", "hemisphere", "anatomical_region", "statistic", "modality"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
