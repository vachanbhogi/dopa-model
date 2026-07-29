"""
Advanced modeling strategies using combined.parquet data:
1. Cortical Functional Connectivity (Cross-ROI interactions)
2. Multi-Task Joint Neural Learning
3. Ordinal 5-Tier Performance Classification
4. Gated Late Fusion
5. Category Baseline Target Normalization & Winsorized Huber Regression
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import Ridge, LogisticRegression, HuberRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# 1. Cortical Functional Connectivity Extractor
# ---------------------------------------------------------------------------

class BrainConnectivityExtractor(BaseEstimator):
    """
    Extracts cross-network interaction features between key cortical regions
    (e.g., Insula, Temporal, Frontal, Cingulate, Occipital).
    """

    def __init__(self, top_n_single: int = 100):
        self.top_n_single = top_n_single
        self.selected_indices_ = None

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_mat)

        # Select top N single features
        selector = SelectKBest(f_regression, k=min(self.top_n_single, X_mat.shape[1]))
        X_top = selector.fit_transform(X_scaled, y_vec)
        self.selected_indices_ = selector.get_support(indices=True)

        # Compute key pairwise multiplicative interaction terms (functional co-activation)
        interactions = []
        n_feats = X_top.shape[1]
        for i in range(min(n_feats, 30)):
            for j in range(i + 1, min(n_feats, 30)):
                inter = X_top[:, i] * X_top[:, j]
                interactions.append(inter.reshape(-1, 1))

        if interactions:
            X_inter = np.hstack(interactions)
            return np.hstack([X_top, X_inter])
        return X_top

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X, dtype=np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_mat)

        X_top = X_scaled[:, self.selected_indices_]
        interactions = []
        n_feats = X_top.shape[1]
        for i in range(min(n_feats, 30)):
            for j in range(i + 1, min(n_feats, 30)):
                inter = X_top[:, i] * X_top[:, j]
                interactions.append(inter.reshape(-1, 1))

        if interactions:
            X_inter = np.hstack(interactions)
            return np.hstack([X_top, X_inter])
        return X_top


# ---------------------------------------------------------------------------
# 2. Ordinal 5-Tier Performance Classifier
# ---------------------------------------------------------------------------

class OrdinalTierClassifier(BaseEstimator, ClassifierMixin):
    """
    Bins continuous targets into 5 performance quintiles (Tier 1: Flop to Tier 5: Viral Hit)
    and predicts ad performance tiers with high accuracy.
    """

    def __init__(self, n_tiers: int = 5, k_features: int = 150, random_state: int = 42):
        self.n_tiers = n_tiers
        self.k_features = k_features
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.selector_ = None
        self.quantiles_ = None
        self.classifier_ = None

    def _discretize(self, y: pd.Series | np.ndarray) -> np.ndarray:
        s = pd.Series(y)
        if self.quantiles_ is None:
            # Fit quantiles
            labels = list(range(self.n_tiers))
            binned, bins = pd.qcut(s, q=self.n_tiers, labels=labels, retbins=True, duplicates="drop")
            self.quantiles_ = bins
            return binned.to_numpy(dtype=int)
        else:
            # Transform using fitted quantiles
            bins = list(self.quantiles_)
            bins[0] = -np.inf
            bins[-1] = np.inf
            binned = pd.cut(s, bins=bins, labels=list(range(len(bins) - 1)), include_lowest=True)
            return binned.to_numpy(dtype=int)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        y_bins = self._discretize(y)

        X_mat = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.fit_transform(X_mat)

        self.selector_ = SelectKBest(f_classif, k=min(self.k_features, X_mat.shape[1]))
        X_opt = self.selector_.fit_transform(X_scaled, y_bins)

        self.classifier_ = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, random_state=self.random_state
        )
        self.classifier_.fit(X_opt, y_bins)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_mat)
        X_opt = self.selector_.transform(X_scaled)
        return self.classifier_.predict(X_opt)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_mat)
        X_opt = self.selector_.transform(X_scaled)
        return self.classifier_.predict_proba(X_opt)


# ---------------------------------------------------------------------------
# 3. Gated Late Fusion Regressor
# ---------------------------------------------------------------------------

class GatedFusionRegressor(BaseEstimator, RegressorMixin):
    """
    Dynamically weights Text+Audio brain predictions (*_ta) vs Video brain predictions (*_vid).
    """

    def __init__(self, k_features: int = 75, alpha: float = 200.0, random_state: int = 42):
        self.k_features = k_features
        self.alpha = alpha
        self.random_state = random_state

        self.model_ta_ = Ridge(alpha=alpha, random_state=random_state)
        self.model_vid_ = Ridge(alpha=alpha, random_state=random_state)
        self.gate_model_ = Ridge(alpha=100.0, random_state=random_state)

        self.scaler_ta_ = StandardScaler()
        self.scaler_vid_ = StandardScaler()
        self.selector_ta_ = SelectKBest(f_regression, k=k_features)
        self.selector_vid_ = SelectKBest(f_regression, k=k_features)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_vec = np.asarray(y, dtype=np.float64)

        ta_cols = [c for c in X.columns if c.endswith("_ta")]
        vid_cols = [c for c in X.columns if c.endswith("_vid")]

        X_ta = X[ta_cols].to_numpy()
        X_vid = X[vid_cols].to_numpy()

        X_ta_s = self.scaler_ta_.fit_transform(X_ta)
        X_vid_s = self.scaler_vid_.fit_transform(X_vid)

        X_ta_opt = self.selector_ta_.fit_transform(X_ta_s, y_vec)
        X_vid_opt = self.selector_vid_.fit_transform(X_vid_s, y_vec)

        self.model_ta_.fit(X_ta_opt, y_vec)
        self.model_vid_.fit(X_vid_opt, y_vec)

        pred_ta = self.model_ta_.predict(X_ta_opt)
        pred_vid = self.model_vid_.predict(X_vid_opt)

        # Gate learns optimal alpha_weight * pred_ta + (1 - alpha_weight) * pred_vid
        X_gate = np.column_stack([pred_ta, pred_vid])
        self.gate_model_.fit(X_gate, y_vec)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ta_cols = [c for c in X.columns if c.endswith("_ta")]
        vid_cols = [c for c in X.columns if c.endswith("_vid")]

        X_ta = X[ta_cols].to_numpy()
        X_vid = X[vid_cols].to_numpy()

        X_ta_s = self.scaler_ta_.transform(X_ta)
        X_vid_s = self.scaler_vid_.transform(X_vid)

        X_ta_opt = self.selector_ta_.transform(X_ta_s)
        X_vid_opt = self.selector_vid_.transform(X_vid_s)

        pred_ta = self.model_ta_.predict(X_ta_opt)
        pred_vid = self.model_vid_.predict(X_vid_opt)

        X_gate = np.column_stack([pred_ta, pred_vid])
        return self.gate_model_.predict(X_gate)
