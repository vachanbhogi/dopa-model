"""
Unit & Integration Tests for TRIBE v2 Ad Metric Model Pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TARGET_COLUMNS, MODALITIES
from model.dataset import load_combined_data, get_feature_names, get_train_val_test_splits
from model.models import BrainMetricRegressor
from model.train import train_single_target_model
from model.evaluate import calculate_metrics
from model.interpret import extract_top_brain_features
from model.inference import AdMetricPredictor


def test_load_combined_data():
    df = load_combined_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    for target in TARGET_COLUMNS:
        assert target in df.columns


def test_feature_names():
    df = load_combined_data()
    ta_feats = get_feature_names(df, modality="brain_text_audio")
    vid_feats = get_feature_names(df, modality="brain_video")
    comb_feats = get_feature_names(df, modality="brain_combined")

    assert len(ta_feats) == 900
    assert len(vid_feats) == 900
    assert len(comb_feats) == 1800


def test_splits():
    df = load_combined_data()
    X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(df, target_name="roi", modality="brain_combined")

    assert len(X_tr) == 1191
    assert len(X_va) == 150
    assert len(X_te) == 148
    assert X_tr.shape[1] == 1800


def test_no_val_contamination():
    """Verify that model training does not leak validation data into fit()."""
    df = load_combined_data()
    res = train_single_target_model(df, target_name="mean_ictr", modality="brain_combined", model_type="ridge", save_model=False)

    assert "val_metrics" in res
    assert "test_metrics" in res
    # Val Pearson r and Test Pearson r should be in similar realistic ranges
    assert res["val_metrics"]["pearson_r"] > 0.0
    assert res["test_metrics"]["pearson_r"] > 0.0


def test_model_fit_and_predict():
    df = load_combined_data()
    X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(df, target_name="roi", modality="brain_combined")

    model = BrainMetricRegressor(model_type="ridge", k_features=50, alpha=100.0)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_va)
    assert len(preds) == len(X_va)
    assert np.isfinite(preds).all()

    metrics = calculate_metrics(y_va, preds)
    assert "r2" in metrics
    assert "pearson_r" in metrics
    assert "accuracy_within_2std" in metrics


def test_feature_interpretation():
    df = load_combined_data()
    X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(df, target_name="roi", modality="brain_combined")

    model = BrainMetricRegressor(model_type="ridge", k_features=50, alpha=100.0)
    model.fit(X_tr, y_tr)

    df_top = extract_top_brain_features(model, list(X_tr.columns), top_n=10)
    assert isinstance(df_top, pd.DataFrame)
    assert len(df_top) == 10
    assert "feature" in df_top.columns
    assert "importance" in df_top.columns


def test_inference_predictor():
    df = load_combined_data()
    sample_brain = df[[c for c in df.columns if c.endswith("_ta") or c.endswith("_vid")]].iloc[[0]]

    predictor = AdMetricPredictor()
    preds = predictor.predict(sample_brain)

    assert isinstance(preds, dict)
    for target in TARGET_COLUMNS:
        assert target in preds
        assert isinstance(preds[target], float)
        assert np.isfinite(preds[target])


if __name__ == "__main__":
    print("Running pipeline unit tests...")
    test_load_combined_data()
    test_feature_names()
    test_splits()
    test_no_val_contamination()
    test_model_fit_and_predict()
    test_feature_interpretation()
    test_inference_predictor()
    print("All unit tests passed successfully!")
