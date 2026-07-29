"""
Training engine and cross-validation runner with strict zero-leakage evaluation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from model.config import TARGET_COLUMNS, LOG_TRANSFORM_TARGETS, MODEL_SAVE_DIR
from model.dataset import load_combined_data, get_train_val_test_splits
from model.models import BrainMetricRegressor
from model.evaluate import calculate_metrics
from model.interpret import extract_top_brain_features


def train_single_target_model(
    df: pd.DataFrame,
    target_name: str,
    modality: str = "brain_combined",
    model_type: str = "ensemble",
    k_features: int = 150,
    alpha: float = 200.0,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Train and evaluate model for a single ad metric target using strict un-contaminated splits.
    Fit is performed ONLY on X_tr (train). Validation is evaluated on X_va (val). Test is evaluated on X_te (test).
    """
    log_trans = LOG_TRANSFORM_TARGETS.get(target_name, False)

    X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(
        df, target_name, modality=modality
    )

    # 1. Instantiate evaluation model
    eval_model = BrainMetricRegressor(
        model_type=model_type,
        k_features=k_features,
        alpha=alpha,
        log_transform=log_trans
    )

    # 2. Fit STRICTLY on X_tr, y_tr (Zero Contamination)
    eval_model.fit(X_tr, y_tr)

    # 3. Predictions on unseen Val and Test partitions
    preds_val = eval_model.predict(X_va)
    preds_test = eval_model.predict(X_te)

    val_metrics = calculate_metrics(y_va, preds_val)
    test_metrics = calculate_metrics(y_te, preds_test)

    # Top brain features extracted from evaluation model
    top_features = extract_top_brain_features(eval_model, list(X_tr.columns), top_n=15)

    # 4. Save production model checkpoint (fitted on train+val for maximum production sample size)
    prod_model = eval_model
    if save_model:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        prod_model = BrainMetricRegressor(
            model_type=model_type,
            k_features=k_features,
            alpha=alpha,
            log_transform=log_trans
        )
        X_train_full = pd.concat([X_tr, X_va], axis=0)
        y_train_full = pd.concat([y_tr, y_va], axis=0)
        prod_model.fit(X_train_full, y_train_full)

        save_path = MODEL_SAVE_DIR / f"{target_name}_{modality}_{model_type}.joblib"
        joblib.dump(prod_model, save_path)

    return {
        "target": target_name,
        "modality": modality,
        "model_type": model_type,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "top_features": top_features,
        "model": prod_model
    }


def train_all_targets(
    data_path: Optional[str] = None,
    modality: str = "brain_combined",
    model_type: str = "ensemble",
    k_features: int = 150,
    alpha: float = 200.0
) -> Dict[str, Any]:
    """Train models across all 4 ad metric targets and return aggregate evaluation report."""
    df = load_combined_data(data_path)
    results = {}

    for target in TARGET_COLUMNS:
        res = train_single_target_model(
            df=df,
            target_name=target,
            modality=modality,
            model_type=model_type,
            k_features=k_features,
            alpha=alpha
        )
        results[target] = res

    return results
