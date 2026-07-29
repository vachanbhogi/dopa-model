from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_pitt_model import (
    DOPA_TARGETS,
    PREDICTION_COLUMNS,
    cross_fitted_scored_table,
    load_pitt_brain_inputs,
    predict_dopa_metrics,
    select_calibrator,
    zero_shot_report,
)


def _write_pitt_fixture(root: Path, offset: float) -> None:
    ids = [f"ad-{index:02d}" for index in range(15)]
    splits = ["train"] * 10 + ["val"] * 2 + ["test"] * 3
    manifest = pd.DataFrame(
        {
            "ad_id": ids,
            "status": ["complete"] * len(ids),
            "split": splits,
            "effectiveness_mean": np.linspace(1.5, 4.5, len(ids)),
            "topic_clean": [str(index % 3) for index in range(len(ids))],
            "duplicate_group": ids,
        }
    )
    features = pd.DataFrame(
        {
            "ad_id": ids,
            "lh_example__mean": np.arange(len(ids), dtype=float) + offset,
            "rh_example__std": np.arange(len(ids), dtype=float) * 2 + offset,
        }
    )
    root.mkdir(parents=True)
    manifest.to_parquet(root / "manifest.parquet", index=False)
    features.to_parquet(root / "compact_features.parquet", index=False)


def test_load_pitt_brain_inputs_matches_dopa_schema(tmp_path: Path) -> None:
    video = tmp_path / "video"
    text_audio = tmp_path / "text_audio"
    _write_pitt_fixture(video, offset=100)
    _write_pitt_fixture(text_audio, offset=200)
    expected = [
        "rh_example__std_ta",
        "lh_example__mean_ta",
        "lh_example__mean_vid",
        "rh_example__std_vid",
    ]

    labels, inputs = load_pitt_brain_inputs(video, text_audio, expected)

    assert len(labels) == 15
    assert inputs.columns.tolist() == expected
    assert inputs.iloc[0].tolist() == [200.0, 200.0, 100.0, 100.0]


def test_predict_dopa_metrics_batches_every_row() -> None:
    class Model:
        def __init__(self, value: float):
            self.value = value

        def predict(self, inputs: pd.DataFrame) -> np.ndarray:
            return np.full(len(inputs), self.value)

    class Predictor:
        models = {
            target: Model(index + 1.0)
            for index, target in enumerate(DOPA_TARGETS)
        }

    result = predict_dopa_metrics(
        Predictor(),
        pd.DataFrame({"feature": [1.0, 2.0, 3.0]}),
    )
    assert result.shape == (3, 4)
    assert result.columns.tolist() == list(PREDICTION_COLUMNS)
    assert result.iloc[2].tolist() == [1.0, 2.0, 3.0, 4.0]


def _calibration_fixture() -> pd.DataFrame:
    rows = []
    for index in range(60):
        split = "train" if index < 42 else "val" if index < 51 else "test"
        rating = 1.0 + 4.0 * index / 59
        mean_ictr = rating / 100
        rows.append(
            {
                "ad_id": f"ad-{index:02d}",
                "split": split,
                "effectiveness_mean": rating,
                "topic_clean": str(index % 4),
                "duplicate_group": f"group-{index}",
                "predicted_roi": rating * 0.2,
                "predicted_cvr": rating * 0.01,
                "predicted_mean_ictr": mean_ictr,
                "predicted_max_ictr": mean_ictr * 2,
            }
        )
    return pd.DataFrame(rows)


def test_zero_shot_report_and_calibration_find_monotonic_signal() -> None:
    table = _calibration_fixture()
    zero_shot = zero_shot_report(table)
    assert (
        zero_shot["outputs"]["predicted_mean_ictr"]["all"]["spearman"]
        == 1.0
    )

    selected, candidates = select_calibrator(table)
    assert selected["candidate"] != "training_mean"
    assert len(candidates) > 3

    scored, production_model = cross_fitted_scored_table(
        table,
        selected,
        seed=33,
    )
    assert np.isfinite(scored["calibrated_effectiveness"]).all()
    assert np.isfinite(scored["prediction_error"]).all()
    assert set(scored["prediction_source"]) == {
        "out_of_fold_train",
        "train_fit",
    }
    assert hasattr(production_model, "predict")
