"""
Feature engineering, scaling, selection, and target transformation utilities.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA


class Log1pTargetTransformer:
    """Helper class for log1p transform and inverse transform on continuous targets."""

    def __init__(self, active: bool = True):
        self.active = active

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self.active:
            return np.asarray(y)
        return np.log1p(np.asarray(y))

    def inverse_transform(self, y_pred: np.ndarray) -> np.ndarray:
        if not self.active:
            return np.asarray(y_pred)
        inv = np.expm1(np.asarray(y_pred))
        return np.clip(inv, 0.0, None)


def build_preprocessing_pipeline(
    scale_features: bool = True,
    k_best_features: Optional[int] = 100,
    pca_components: Optional[int] = None,
    random_state: int = 42
):
    """Construct sklearn preprocessing pipeline combining scaling, feature selection, and/or PCA."""
    from sklearn.pipeline import Pipeline

    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))

    if k_best_features is not None and k_best_features > 0:
        steps.append(("select_k_best", SelectKBest(f_regression, k=k_best_features)))

    if pca_components is not None and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))

    return Pipeline(steps) if steps else None
