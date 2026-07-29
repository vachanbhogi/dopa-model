from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DOPA_TARGETS = ("roi", "cvr", "mean_ictr", "max_ictr")
PREDICTION_COLUMNS = tuple(f"predicted_{target}" for target in DOPA_TARGETS)
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


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


def expected_brain_columns(metric_model_repo: Path) -> list[str]:
    import pyarrow.parquet as pq

    combined_path = metric_model_repo / "data" / "combined.parquet"
    if not combined_path.is_file():
        raise FileNotFoundError(
            f"Dopa combined feature schema was not found: {combined_path}"
        )
    columns = pq.ParquetFile(combined_path).schema.names
    brain_columns = [
        column
        for column in columns
        if column.endswith("_ta") or column.endswith("_vid")
    ]
    if len(brain_columns) != 1_800:
        raise RuntimeError(
            "Expected 1,800 Dopa brain features, "
            f"found {len(brain_columns)}"
        )
    return brain_columns


def _load_complete_manifest(dataset_dir: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(dataset_dir / "manifest.parquet")
    if "status" not in manifest or "ad_id" not in manifest:
        raise RuntimeError(f"Invalid Pitt manifest in {dataset_dir}")
    manifest = manifest[manifest["status"] == "complete"].copy()
    manifest["ad_id"] = manifest["ad_id"].astype(str)
    if manifest["ad_id"].duplicated().any():
        raise RuntimeError(f"Duplicate Pitt ad IDs in {dataset_dir}")
    return manifest


def _load_compact_features(dataset_dir: Path) -> pd.DataFrame:
    features = pd.read_parquet(dataset_dir / "compact_features.parquet")
    if "ad_id" not in features:
        raise RuntimeError(f"Compact features lack ad_id in {dataset_dir}")
    features["ad_id"] = features["ad_id"].astype(str)
    if features["ad_id"].duplicated().any():
        raise RuntimeError(f"Duplicate feature ad IDs in {dataset_dir}")
    return features


def load_pitt_brain_inputs(
    video_dataset_dir: Path,
    text_audio_dataset_dir: Path,
    brain_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_manifest = _load_complete_manifest(video_dataset_dir)
    text_audio_manifest = _load_complete_manifest(text_audio_dataset_dir)
    video_features = _load_compact_features(video_dataset_dir)
    text_audio_features = _load_compact_features(text_audio_dataset_dir)

    id_sets = [
        set(video_manifest["ad_id"]),
        set(text_audio_manifest["ad_id"]),
        set(video_features["ad_id"]),
        set(text_audio_features["ad_id"]),
    ]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise RuntimeError(
            "Video and text+audio Pitt manifests/features do not contain "
            "the same ad IDs"
        )

    required_labels = {
        "ad_id",
        "split",
        "effectiveness_mean",
        "topic_clean",
    }
    missing = required_labels - set(video_manifest.columns)
    if missing:
        raise RuntimeError(f"Pitt video manifest lacks {sorted(missing)}")

    video_labels = video_manifest.set_index("ad_id").sort_index()
    text_audio_labels = text_audio_manifest.set_index("ad_id").sort_index()
    for column in ("split", "effectiveness_mean"):
        left = video_labels[column]
        right = text_audio_labels[column]
        if column == "effectiveness_mean":
            agrees = np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            )
        else:
            agrees = left.equals(right)
        if not agrees:
            raise RuntimeError(
                f"Pitt video/text+audio manifests disagree on {column}"
            )

    ordered_ids = video_labels.index.tolist()
    video_features = video_features.set_index("ad_id").loc[ordered_ids]
    text_audio_features = text_audio_features.set_index("ad_id").loc[ordered_ids]

    ta_columns = [column for column in brain_columns if column.endswith("_ta")]
    video_columns = [
        column for column in brain_columns if column.endswith("_vid")
    ]
    ta_base = [column[:-3] for column in ta_columns]
    video_base = [column[:-4] for column in video_columns]
    missing_ta = set(ta_base) - set(text_audio_features.columns)
    missing_video = set(video_base) - set(video_features.columns)
    extra_ta = (set(text_audio_features.columns) - {"ad_id"}) - set(ta_base)
    extra_video = (set(video_features.columns) - {"ad_id"}) - set(video_base)
    if missing_ta or missing_video or extra_ta or extra_video:
        raise RuntimeError(
            "Pitt and Dopa feature schemas differ: "
            f"missing_ta={len(missing_ta)}, missing_video={len(missing_video)}, "
            f"extra_ta={len(extra_ta)}, extra_video={len(extra_video)}"
        )

    ta_block = text_audio_features[ta_base].copy()
    ta_block.columns = ta_columns
    video_block = video_features[video_base].copy()
    video_block.columns = video_columns
    brain_inputs = pd.concat([ta_block, video_block], axis=1)[brain_columns]
    if not np.isfinite(brain_inputs.to_numpy(dtype=float)).all():
        raise RuntimeError("Pitt brain inputs contain non-finite values")

    label_columns = [
        column
        for column in (
            "split",
            "effectiveness_mean",
            "effectiveness_std",
            "effectiveness_n",
            "effectiveness_clean",
            "topic_clean",
            "language_clean",
            "duration_seconds",
            "duplicate_group",
        )
        if column in video_labels
    ]
    labels = video_labels[label_columns].reset_index()
    required_splits = {"train", "val", "test"}
    if set(labels["split"]) != required_splits:
        raise RuntimeError(
            f"Expected Pitt train/val/test, found {sorted(set(labels['split']))}"
        )
    return labels, brain_inputs.reset_index(drop=True)


def load_dopa_predictor(metric_model_repo: Path):
    model_dir = metric_model_repo / "benchmark" / "saved_models"
    missing = [
        str(model_dir / f"{target}_brain_combined_ensemble.joblib")
        for target in DOPA_TARGETS
        if not (
            model_dir / f"{target}_brain_combined_ensemble.joblib"
        ).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Dopa metric model artifacts: " + ", ".join(missing)
        )

    repo_text = str(metric_model_repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from model.inference import AdMetricPredictor

    return AdMetricPredictor(model_dir=model_dir)


def predict_dopa_metrics(predictor, brain_inputs: pd.DataFrame) -> pd.DataFrame:
    predictions: dict[str, np.ndarray] = {}
    for target in DOPA_TARGETS:
        values = np.asarray(
            predictor.models[target].predict(brain_inputs),
            dtype=float,
        ).reshape(-1)
        if len(values) != len(brain_inputs):
            raise RuntimeError(
                f"Dopa {target} model returned {len(values)} predictions "
                f"for {len(brain_inputs)} ads"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Dopa {target} predictions are non-finite")
        predictions[f"predicted_{target}"] = values
    return pd.DataFrame(predictions)


def dopa_source_test_report(
    metric_model_repo: Path,
    predictor,
    brain_columns: list[str],
) -> dict[str, Any]:
    source = pd.read_parquet(metric_model_repo / "data" / "combined.parquet")
    source = source[source["split"] == "test"].copy()
    report: dict[str, Any] = {}
    for target in DOPA_TARGETS:
        target_rows = source[source[target].notna()]
        actual = target_rows[target].to_numpy(dtype=float)
        predicted = np.asarray(
            predictor.models[target].predict(target_rows[brain_columns]),
            dtype=float,
        )
        report[target] = regression_metrics(actual, predicted)
    return report


def _safe_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], Any],
) -> float | None:
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return None
    result = function(actual, predicted)
    statistic = float(result.statistic)
    return statistic if math.isfinite(statistic) else None


def correlation_metrics(
    table: pd.DataFrame,
    prediction_column: str,
) -> dict[str, Any]:
    actual = table["effectiveness_mean"].to_numpy(dtype=float)
    predicted = table[prediction_column].to_numpy(dtype=float)
    return {
        "rows": len(table),
        "pearson": _safe_correlation(actual, predicted, pearsonr),
        "spearman": _safe_correlation(actual, predicted, spearmanr),
    }


def pairwise_winner_accuracy(
    table: pd.DataFrame,
    prediction_column: str,
    minimum_gap: float = 1.0,
) -> dict[str, Any]:
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
                    rows.loc[left, prediction_column]
                    - rows.loc[right, prediction_column]
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


def zero_shot_report(
    table: pd.DataFrame,
    minimum_pair_gap: float = 1.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "interpretation": (
            "Association between frozen AdsTrace metric predictions and Pitt "
            "human effectiveness ratings; raw values are not on the same scale"
        ),
        "primary_output": "predicted_mean_ictr",
        "outputs": {},
    }
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    actual_threshold = float(train["effectiveness_mean"].median())

    for prediction_column in PREDICTION_COLUMNS:
        output: dict[str, Any] = {}
        for split_name in ("all", "train", "val", "test"):
            split_table = (
                table if split_name == "all" else table[table["split"] == split_name]
            )
            output[split_name] = correlation_metrics(
                split_table,
                prediction_column,
            )
            output[split_name]["pairwise"] = pairwise_winner_accuracy(
                split_table,
                prediction_column,
                minimum_gap=minimum_pair_gap,
            )

        prediction_threshold = float(train[prediction_column].median())
        actual_high = (
            test["effectiveness_mean"].to_numpy(dtype=float)
            >= actual_threshold
        )
        predicted_high = (
            test[prediction_column].to_numpy(dtype=float)
            >= prediction_threshold
        )
        output["test"]["median_tier_accuracy"] = float(
            np.mean(actual_high == predicted_high)
        )
        output["test"]["thresholds_from_train"] = {
            "effectiveness": actual_threshold,
            prediction_column: prediction_threshold,
        }
        report["outputs"][prediction_column] = output
    return report


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "rows": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "pearson": _safe_correlation(actual, predicted, pearsonr),
        "spearman": _safe_correlation(actual, predicted, spearmanr),
    }


def make_calibrator(
    candidate: str,
    alpha: float | None,
) -> tuple[Any, list[str]]:
    if candidate == "training_mean":
        return DummyRegressor(strategy="mean"), []
    if candidate == "mean_ictr_affine":
        columns = ["predicted_mean_ictr"]
    elif candidate == "four_metric_ridge":
        columns = list(PREDICTION_COLUMNS)
    else:
        raise ValueError(f"Unknown calibration candidate: {candidate}")
    if alpha is None:
        raise ValueError(f"{candidate} requires alpha")
    return (
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        ),
        columns,
    )


def _model_inputs(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if columns:
        return table[columns]
    return pd.DataFrame({"constant": np.ones(len(table))}, index=table.index)


def select_calibrator(
    table: pd.DataFrame,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "val"]
    test = table[table["split"] == "test"]
    specifications: list[tuple[str, float | None]] = [
        ("training_mean", None),
        *[("mean_ictr_affine", alpha) for alpha in alphas],
        *[("four_metric_ridge", alpha) for alpha in alphas],
    ]

    candidates: list[dict[str, Any]] = []
    for candidate, alpha in specifications:
        model, columns = make_calibrator(candidate, alpha)
        model.fit(
            _model_inputs(train, columns),
            train["effectiveness_mean"].to_numpy(dtype=float),
        )
        val_prediction = np.clip(
            model.predict(_model_inputs(validation, columns)),
            1.0,
            5.0,
        )
        test_prediction = np.clip(
            model.predict(_model_inputs(test, columns)),
            1.0,
            5.0,
        )
        candidates.append(
            {
                "candidate": candidate,
                "alpha": alpha,
                "feature_columns": columns,
                "validation": regression_metrics(
                    validation["effectiveness_mean"].to_numpy(dtype=float),
                    val_prediction,
                ),
                "test": regression_metrics(
                    test["effectiveness_mean"].to_numpy(dtype=float),
                    test_prediction,
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda result: (
            result["validation"]["rmse"],
            result["validation"]["mae"],
            result["candidate"],
            result["alpha"] if result["alpha"] is not None else -1,
        ),
    )
    return selected, candidates


def cross_fitted_scored_table(
    table: pd.DataFrame,
    selected: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, Any]:
    scored = table.copy()
    model, columns = make_calibrator(
        selected["candidate"],
        selected["alpha"],
    )
    train_mask = scored["split"] == "train"
    train = scored[train_mask]
    groups = (
        train["duplicate_group"].astype(str)
        if "duplicate_group" in train
        else None
    )
    if groups is not None and groups.nunique() >= 5:
        cross_validator = GroupKFold(n_splits=5)
        train_prediction = cross_val_predict(
            model,
            _model_inputs(train, columns),
            train["effectiveness_mean"].to_numpy(dtype=float),
            cv=cross_validator,
            groups=groups,
            method="predict",
        )
    else:
        cross_validator = KFold(n_splits=5, shuffle=True, random_state=seed)
        train_prediction = cross_val_predict(
            model,
            _model_inputs(train, columns),
            train["effectiveness_mean"].to_numpy(dtype=float),
            cv=cross_validator,
            method="predict",
        )
    scored.loc[train_mask, "calibrated_effectiveness"] = np.clip(
        train_prediction,
        1.0,
        5.0,
    )
    scored.loc[train_mask, "prediction_source"] = "out_of_fold_train"

    evaluation_model, columns = make_calibrator(
        selected["candidate"],
        selected["alpha"],
    )
    evaluation_model.fit(
        _model_inputs(train, columns),
        train["effectiveness_mean"].to_numpy(dtype=float),
    )
    for split_name in ("val", "test"):
        split_mask = scored["split"] == split_name
        split_table = scored[split_mask]
        scored.loc[split_mask, "calibrated_effectiveness"] = np.clip(
            evaluation_model.predict(_model_inputs(split_table, columns)),
            1.0,
            5.0,
        )
        scored.loc[split_mask, "prediction_source"] = "train_fit"

    scored["prediction_error"] = (
        scored["calibrated_effectiveness"] - scored["effectiveness_mean"]
    )

    production_rows = scored[scored["split"].isin(["train", "val"])]
    production_model, columns = make_calibrator(
        selected["candidate"],
        selected["alpha"],
    )
    production_model.fit(
        _model_inputs(production_rows, columns),
        production_rows["effectiveness_mean"].to_numpy(dtype=float),
    )
    return scored, production_model


def bootstrap_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    seed: int,
) -> dict[str, float | int | None]:
    if iterations <= 0 or len(actual) < 2:
        return {"iterations": 0, "low": None, "high": None}
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        indices = generator.integers(0, len(actual), size=len(actual))
        value = float(statistic(actual[indices], predicted[indices]))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {"iterations": iterations, "low": None, "high": None}
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "iterations": iterations,
        "low": float(low),
        "high": float(high),
    }


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_benchmark(
    video_dataset_dir: Path,
    text_audio_dataset_dir: Path,
    metric_model_repo: Path,
    output_dir: Path,
    minimum_pair_gap: float = 1.0,
    bootstrap_iterations: int = 2_000,
    seed: int = 33,
) -> dict[str, Any]:
    brain_columns = expected_brain_columns(metric_model_repo)
    labels, brain_inputs = load_pitt_brain_inputs(
        video_dataset_dir,
        text_audio_dataset_dir,
        brain_columns,
    )
    predictor = load_dopa_predictor(metric_model_repo)
    metric_predictions = predict_dopa_metrics(predictor, brain_inputs)
    source_test = dopa_source_test_report(
        metric_model_repo,
        predictor,
        brain_columns,
    )
    table = pd.concat(
        [labels.reset_index(drop=True), metric_predictions],
        axis=1,
    )

    zero_shot = zero_shot_report(
        table,
        minimum_pair_gap=minimum_pair_gap,
    )
    selected, candidates = select_calibrator(table)
    scored, production_model = cross_fitted_scored_table(
        table,
        selected,
        seed=seed,
    )

    split_metrics: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        split_table = scored[scored["split"] == split_name]
        actual = split_table["effectiveness_mean"].to_numpy(dtype=float)
        predicted = split_table["calibrated_effectiveness"].to_numpy(dtype=float)
        split_metrics[split_name] = regression_metrics(actual, predicted)
        split_metrics[split_name]["pairwise"] = pairwise_winner_accuracy(
            split_table,
            "calibrated_effectiveness",
            minimum_gap=minimum_pair_gap,
        )

    test = scored[scored["split"] == "test"]
    test_actual = test["effectiveness_mean"].to_numpy(dtype=float)
    test_predicted = test["calibrated_effectiveness"].to_numpy(dtype=float)
    split_metrics["test"]["confidence_intervals_95"] = {
        "mae": bootstrap_interval(
            test_actual,
            test_predicted,
            lambda actual, predicted: mean_absolute_error(actual, predicted),
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "spearman": bootstrap_interval(
            test_actual,
            test_predicted,
            lambda actual, predicted: (
                float(spearmanr(actual, predicted).statistic)
                if np.ptp(actual) > 0 and np.ptp(predicted) > 0
                else float("nan")
            ),
            iterations=bootstrap_iterations,
            seed=seed + 1,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(
        table,
        output_dir / "frozen_metric_predictions.parquet",
    )
    write_parquet_atomic(
        scored,
        output_dir / "calibrated_predictions.parquet",
    )

    artifact = {
        "model": production_model,
        "feature_columns": selected["feature_columns"],
        "candidate": selected["candidate"],
        "alpha": selected["alpha"],
        "target": "effectiveness_mean",
        "output_range": [1.0, 5.0],
        "base_outputs": list(PREDICTION_COLUMNS),
        "dopa_model_revision": git_revision(metric_model_repo),
        "fit_rows": int(scored["split"].isin(["train", "val"]).sum()),
    }
    temporary_model = output_dir / f"pitt_calibrator.joblib.{os.getpid()}.tmp"
    joblib.dump(artifact, temporary_model)
    replace_with_retry(temporary_model, output_dir / "pitt_calibrator.joblib")

    report = {
        "experiment": "Frozen Dopa AdsTrace heads evaluated on Pitt",
        "target": "Pitt mean human effectiveness rating (1-5)",
        "rows": len(table),
        "split_counts": table["split"].value_counts().to_dict(),
        "brain_feature_count": len(brain_columns),
        "base_outputs": list(PREDICTION_COLUMNS),
        "dopa_model": {
            "repository": "https://github.com/vachanbhogi/dopa-dataset",
            "revision": git_revision(metric_model_repo),
            "training_domain": "AdsTrace",
            "training_targets": list(DOPA_TARGETS),
            "frozen": True,
            "source_domain_test_metrics": source_test,
        },
        "zero_shot": zero_shot,
        "calibration": {
            "method": (
                "Validation-selected training mean, mean-iCTR affine Ridge, "
                "or four-output Ridge; Dopa models remain frozen"
            ),
            "selected": selected,
            "all_candidates": candidates,
            "metrics": split_metrics,
            "production_refit": "train+validation only",
        },
        "limitations": [
            (
                "Pitt human effectiveness and AdsTrace ROI/CVR/iCTR are "
                "different constructs; zero-shot results are associations, "
                "not raw-unit accuracy."
            ),
            (
                "The 160-ad Pitt run is a pilot with a 20-ad test split; "
                "confidence intervals are correspondingly wide."
            ),
            (
                "The current test labels have been inspected during model "
                "planning, so this report is exploratory rather than a final "
                "untouched benchmark."
            ),
            (
                "A production claim requires a larger Pitt sample and a new "
                "locked holdout."
            ),
        ],
    }
    temporary_report = output_dir / f"metrics.json.{os.getpid()}.tmp"
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_value),
        encoding="utf-8",
    )
    replace_with_retry(temporary_report, output_dir / "metrics.json")
    return report


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Dopa AdsTrace metric heads against Pitt human "
            "effectiveness ratings and fit a tiny calibration layer"
        )
    )
    parser.add_argument(
        "--video-dataset-dir",
        type=Path,
        default=Path("data/PittVideoAdsTribeV2_video"),
    )
    parser.add_argument(
        "--text-audio-dataset-dir",
        type=Path,
        default=Path("data/PittVideoAdsTribeV2_text_audio"),
    )
    parser.add_argument(
        "--metric-model-repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Local checkout of vachanbhogi/dopa-dataset (defaults to repo root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/PittDopaFrozenBenchmark"),
    )
    parser.add_argument("--minimum-pair-gap", type=float, default=1.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=33)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    report = run_benchmark(
        video_dataset_dir=args.video_dataset_dir.resolve(),
        text_audio_dataset_dir=args.text_audio_dataset_dir.resolve(),
        metric_model_repo=args.metric_model_repo.resolve(),
        output_dir=args.output_dir.resolve(),
        minimum_pair_gap=args.minimum_pair_gap,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    summary = {
        "rows": report["rows"],
        "zero_shot_primary": report["zero_shot"]["outputs"][
            "predicted_mean_ictr"
        ],
        "selected_calibrator": report["calibration"]["selected"],
        "calibrated_metrics": report["calibration"]["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
