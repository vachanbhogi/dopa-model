"""
Production inference pipeline for predicting ad metrics from TRIBE v2 fMRI outputs.
Integrates 10-network cognitive ROI scoring and multi-metric predictions.
"""

from typing import Dict, Any, Union
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from model.config import TARGET_COLUMNS, MODEL_SAVE_DIR
from model.dataset import load_combined_data, get_feature_names
from model.roi import extract_cognitive_network_scores
from model.train import train_single_target_model


class AdMetricPredictor:
    """
    Production Predictor class that receives TRIBE v2 text-audio fMRI features (900 *_ta)
    and video fMRI features (900 *_vid) and predicts all ad metrics.
    """

    def __init__(self, model_dir: Union[str, Path] = MODEL_SAVE_DIR):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        self._load_or_train_models()

    def _load_or_train_models(self):
        df = None
        for target in TARGET_COLUMNS:
            model_path = self.model_dir / f"{target}_brain_combined_ensemble.joblib"
            if model_path.exists():
                self.models[target] = joblib.load(model_path)
            else:
                if df is None:
                    df = load_combined_data()
                res = train_single_target_model(
                    df, target_name=target, modality="brain_combined", model_type="ensemble", save_model=True
                )
                self.models[target] = res["model"]

    def _extract_brain_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        brain_cols = [c for c in df.columns if c.endswith("_ta") or c.endswith("_vid")]
        if brain_cols:
            return df[brain_cols].copy()
        return df

    def predict(self, brain_features: Union[pd.DataFrame, Dict[str, float]]) -> Dict[str, float]:
        """
        Predict ad metrics from 1800 TRIBE v2 brain surface features.

        :param brain_features: DataFrame or dict containing the 1,800 *_ta and *_vid features.
        :return: Dictionary containing predicted 'roi', 'cvr', 'mean_ictr', and 'max_ictr'.
        """
        if isinstance(brain_features, dict):
            df_feat = pd.DataFrame([brain_features])
        elif isinstance(brain_features, pd.DataFrame):
            df_feat = brain_features
        else:
            raise TypeError("brain_features must be a dictionary or pandas DataFrame")

        X_brain = self._extract_brain_cols(df_feat)

        predictions = {}
        for target in TARGET_COLUMNS:
            model = self.models[target]
            pred_val = float(model.predict(X_brain)[0])
            predictions[target] = pred_val

        return predictions

    def predict_full_report(self, brain_features: Union[pd.DataFrame, Dict[str, float]]) -> Dict[str, Any]:
        """
        Full prediction report including target predictions, 10 cognitive network scores,
        and executive scorecard metrics.
        """
        preds = self.predict(brain_features)
        cog_res = extract_cognitive_network_scores(brain_features)

        mean_ctr = preds.get("mean_ictr", 0.025)
        ctr_pct = mean_ctr * 100.0
        views_pctile = float(np.clip((mean_ctr / 0.05) * 100.0, 1.0, 99.0))
        att_standout = cog_res["networks"]["visual_cortex"]["score"]
        cog_load = cog_res["networks"]["dlpfc"]["score"]

        scorecard = {
            "predicted_ctr_pct": round(ctr_pct, 2),
            "predicted_views_percentile": round(views_pctile, 1),
            "attention_standout_score": round(att_standout, 1),
            "cognitive_load_index": round(cog_load, 1),
            "performance_verdict": "Strong" if ctr_pct > 3.0 else "Typical" if ctr_pct > 1.2 else "Weak",
        }

        return {
            "metrics": preds,
            "scorecard": scorecard,
            "cognitive_networks": cog_res["networks"],
            "hook_retention_score": cog_res["hook_retention_score"],
            "hook_verdict": cog_res["hook_verdict"],
        }
