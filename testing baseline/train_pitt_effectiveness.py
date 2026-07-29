from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0)


def json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def replace_with_retry(
    source: Path,
    destination: Path,
    attempts: int = 20,
    base_delay_seconds: float = 0.25,
) -> None:
    """Atomically replace a file while tolerating brief Windows file locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(base_delay_seconds * (2**attempt), 2.0))


def write_parquet_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    table.to_parquet(temporary, index=False)
    replace_with_retry(temporary, path)


def pairwise_winner_accuracy(
    table: pd.DataFrame,
    minimum_gap: float = 1.0,
) -> dict[str, Any]:
    required = {"topic_clean", "effectiveness_mean", "prediction"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Pairwise table is missing columns: {sorted(missing)}")
    correct = 0
    evaluated = 0
    ties = 0
    for _, group in table.groupby("topic_clean", dropna=False):
        rows = group.reset_index(drop=True)
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                actual_delta = float(
                    rows.loc[left, "effectiveness_mean"]
                    - rows.loc[right, "effectiveness_mean"]
                )
                if abs(actual_delta) < minimum_gap:
                    continue
                predicted_delta = float(
                    rows.loc[left, "prediction"] - rows.loc[right, "prediction"]
                )
                if predicted_delta == 0:
                    ties += 1
                    continue
                evaluated += 1
                correct += int(np.sign(actual_delta) == np.sign(predicted_delta))
    return {
        "accuracy": float(correct / evaluated) if evaluated else None,
        "evaluated_pairs": evaluated,
        "prediction_ties": ties,
        "minimum_human_rating_gap": minimum_gap,
    }


def regression_metrics(table: pd.DataFrame) -> dict[str, Any]:
    actual = table["effectiveness_mean"].to_numpy(dtype=float)
    predicted = table["prediction"].to_numpy(dtype=float)
    correlation = spearmanr(actual, predicted)
    return {
        "rows": len(table),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "spearman": (
            float(correlation.statistic)
            if correlation.statistic is not None
            and math.isfinite(float(correlation.statistic))
            else None
        ),
    }


def load_training_table(dataset_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_parquet(dataset_dir / "manifest.parquet")
    features = pd.read_parquet(dataset_dir / "compact_features.parquet")
    if manifest["ad_id"].duplicated().any() or features["ad_id"].duplicated().any():
        raise RuntimeError("Pitt manifests and features must contain unique ad IDs")
    manifest = manifest[manifest["status"] == "complete"].copy()
    table = manifest.merge(features, on="ad_id", how="inner", validate="one_to_one")
    feature_columns = [column for column in features.columns if column != "ad_id"]
    if not feature_columns:
        raise RuntimeError("No compact TRIBE features were found")
    if table[feature_columns].isna().any().any():
        raise RuntimeError("Compact TRIBE features contain missing values")
    required_splits = {"train", "val", "test"}
    if set(table["split"]) & required_splits != required_splits:
        raise RuntimeError(
            f"Expected train/val/test rows, found {sorted(set(table['split']))}"
        )
    return table, feature_columns


def train_and_evaluate(
    dataset_dir: Path,
    output_dir: Path,
    minimum_pair_gap: float = 1.0,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    table, feature_columns = load_training_table(dataset_dir)
    train = table[table["split"] == "train"].copy()
    validation = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    if len(train) < 10:
        raise RuntimeError("At least 10 training rows are required for RidgeCV")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ridge",
                RidgeCV(
                    alphas=np.asarray(alphas, dtype=float),
                    cv=min(5, len(train)),
                    scoring="neg_mean_squared_error",
                ),
            ),
        ]
    )
    model.fit(
        train[feature_columns],
        train["effectiveness_mean"].to_numpy(dtype=float),
    )

    predictions: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    for split_name, split_table in (
        ("train", train),
        ("val", validation),
        ("test", test),
    ):
        scored = split_table[
            ["ad_id", "split", "topic_clean", "effectiveness_mean"]
        ].copy()
        scored["prediction"] = model.predict(split_table[feature_columns])
        predictions.append(scored)
        metrics[split_name] = regression_metrics(scored)
        metrics[split_name]["pairwise"] = pairwise_winner_accuracy(
            scored,
            minimum_gap=minimum_pair_gap,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    write_parquet_atomic(prediction_table, output_dir / "predictions.parquet")
    temporary_model = output_dir / f"ridge_model.joblib.{os.getpid()}.tmp"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": "effectiveness_mean",
        },
        temporary_model,
    )
    replace_with_retry(temporary_model, output_dir / "ridge_model.joblib")
    ridge = model.named_steps["ridge"]
    report = {
        "model": "StandardScaler + RidgeCV",
        "target": "effectiveness_mean",
        "feature_count": len(feature_columns),
        "training_rows": len(train),
        "selected_alpha": float(ridge.alpha_),
        "alpha_candidates": list(alphas),
        "alpha_selection": "5-fold cross-validation on training rows only",
        "metrics": metrics,
        "interpretation": (
            "Offline viewer-rated effectiveness prediction; not causal A/B lift"
        ),
    }
    temporary_metrics = output_dir / f"metrics.json.{os.getpid()}.tmp"
    temporary_metrics.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_value),
        encoding="utf-8",
    )
    replace_with_retry(temporary_metrics, output_dir / "metrics.json")
    return report


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Pitt TRIBE effectiveness baseline"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/PittVideoAdsTribeV2_video"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/PittEffectivenessRidge"),
    )
    parser.add_argument("--minimum-pair-gap", type=float, default=1.0)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    report = train_and_evaluate(
        args.dataset_dir.resolve(),
        args.output_dir.resolve(),
        minimum_pair_gap=args.minimum_pair_gap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
