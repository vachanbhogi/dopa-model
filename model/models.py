"""
Model implementations and ensemble wrappers for ad metric prediction.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor

from model.features import Log1pTargetTransformer


class BrainMetricRegressor(BaseEstimator, RegressorMixin):
    """
    Unified regressor wrapper supporting feature scaling, feature selection,
    model fitting, and target log transformations.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        k_features: int = 100,
        alpha: float = 500.0,
        log_transform: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        self.model_type = model_type
        self.k_features = k_features
        self.alpha = alpha
        self.log_transform = log_transform
        self.random_state = random_state
        self.kwargs = kwargs

        self.target_transformer_ = Log1pTargetTransformer(active=log_transform)
        self.scaler_ = StandardScaler()
        self.selector_ = None
        self.model_ = None

    def _build_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_state, **self.kwargs)
        elif self.model_type == "elasticnet":
            return ElasticNet(alpha=self.alpha, l1_ratio=0.1, max_iter=2000, random_state=self.random_state, **self.kwargs)
        elif self.model_type == "hgb":
            return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, random_state=self.random_state, **self.kwargs)
        elif self.model_type == "extra_trees":
            return ExtraTreesRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=self.random_state, **self.kwargs)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=self.random_state, **self.kwargs)
        elif self.model_type == "mlp":
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=self.random_state, **self.kwargs)
        elif self.model_type == "ensemble":
            r1 = Ridge(alpha=self.alpha, random_state=self.random_state)
            r2 = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.03, random_state=self.random_state)
            r3 = ExtraTreesRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=self.random_state)
            return VotingRegressor(estimators=[('ridge', r1), ('hgb', r2), ('et', r3)])
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        y_trans = self.target_transformer_.transform(y)

        X_scaled = self.scaler_.fit_transform(X)

        num_feats = X.shape[1]
        if self.k_features is not None and 0 < self.k_features < num_feats:
            self.selector_ = SelectKBest(f_regression, k=min(self.k_features, num_feats))
            X_opt = self.selector_.fit_transform(X_scaled, y_trans)
        else:
            self.selector_ = None
            X_opt = X_scaled

        self.model_ = self._build_model()
        self.model_.fit(X_opt, y_trans)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        if self.selector_ is not None:
            X_opt = self.selector_.transform(X_scaled)
        else:
            X_opt = X_scaled

        y_pred_trans = self.model_.predict(X_opt)
        return self.target_transformer_.inverse_transform(y_pred_trans)
