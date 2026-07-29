#!/usr/bin/env python3
"""
Train and Save Production Models for all 4 Ad Metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TARGET_COLUMNS, MODEL_SAVE_DIR
from model.dataset import load_combined_data
from model.train import train_single_target_model


def train_and_save():
    print("=" * 80)
    print("TRAINING AND SAVING PRODUCTION AD METRIC PREDICTION MODELS")
    print("=" * 80)

    df = load_combined_data()
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for target in TARGET_COLUMNS:
        print(f"\n---> Training model for Target: {target.upper()} <---")
        res = train_single_target_model(
            df=df,
            target_name=target,
            modality="brain_combined",
            model_type="ensemble",
            k_features=150,
            alpha=200.0,
            save_model=True
        )

        val = res["val_metrics"]
        test = res["test_metrics"]

        print(f"Validation | Pearson r={val['pearson_r']:.4f} | R2={val['r2']:.4f} | Acc(2std)={val['accuracy_within_2std']:.2f}% | TierAcc={val['tier_classification_accuracy']:.2f}%")
        print(f"Test Set   | Pearson r={test['pearson_r']:.4f} | R2={test['r2']:.4f} | Acc(2std)={test['accuracy_within_2std']:.2f}% | TierAcc={test['tier_classification_accuracy']:.2f}%")

    print("\n" + "=" * 80)
    print(f"All production models successfully trained and saved to '{MODEL_SAVE_DIR}'")
    print("=" * 80)


if __name__ == "__main__":
    train_and_save()
