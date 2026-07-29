"""
Evaluation metrics and statistical validation routines.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray
) -> Dict[str, float]:
    """Calculate regression, correlation, tolerance accuracy, and tier classification metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 1. Standard Regression Metrics
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 2. Correlation Metrics
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    pearson_r = float(pr) if np.isfinite(pr) else 0.0
    spearman_r = float(sr) if np.isfinite(sr) else 0.0

    # 3. Tolerance Accuracy Metrics
    std_true = float(np.std(y_true))
    acc_1std = float(np.mean(np.abs(y_true - y_pred) <= std_true) * 100.0)
    acc_2std = float(np.mean(np.abs(y_true - y_pred) <= (2.0 * std_true)) * 100.0)

    # 4. Tier Classification Accuracy (Top vs Bottom Performance)
    median_val = float(np.median(y_true))
    true_tier = (y_true >= median_val).astype(int)
    pred_tier = (y_pred >= median_val).astype(int)
    tier_acc = float(accuracy_score(true_tier, pred_tier) * 100.0)

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "accuracy_within_1std": acc_1std,
        "accuracy_within_2std": acc_2std,
        "tier_classification_accuracy": tier_acc
    }
