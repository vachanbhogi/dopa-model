from __future__ import annotations

from pathlib import Path

import numpy as np

from dopa_api.scoring import ParcelDefinition, VideoAdScorer


def test_parcel_summary_uses_real_prediction_series(tmp_path: Path) -> None:
    scorer = VideoAdScorer(tmp_path / "model.joblib", tmp_path / "cache")
    mask = np.array([True, True])
    scorer.parcels = [
        ParcelDefinition(
            feature_prefix="lh_S_calcarine",
            label="S_calcarine",
            hemisphere="left",
            vertex_slice=slice(0, 2),
            mask=mask,
        ),
        ParcelDefinition(
            feature_prefix="rh_G_front_sup",
            label="G_front_sup",
            hemisphere="right",
            vertex_slice=slice(10_242, 10_244),
            mask=mask,
        ),
    ]
    scorer.expected_columns = [
        f"{prefix}__{stat}_vid"
        for prefix in ("lh_S_calcarine", "rh_G_front_sup")
        for stat in ("mean", "std", "max", "p95", "slope", "peak_fraction")
    ]
    predictions = np.zeros((3, 20_484), dtype=np.float32)
    predictions[:, :2] = np.array([[0, 0], [2, 2], [4, 4]], dtype=np.float32)
    predictions[:, 10_242:10_244] = 1

    compact, regions = scorer._parcel_features(predictions)

    assert list(compact.columns) == scorer.expected_columns
    assert regions[0].region_id == "lh_S_calcarine"
    assert regions[0].relative_response == 100
    assert regions[0].peak_second == 2
    assert "visual-system" in regions[0].description
    assert regions[1].hemisphere == "right"
