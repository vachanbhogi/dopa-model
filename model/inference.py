"""
Production inference pipeline for predicting ad metrics from TRIBE v2 fMRI outputs.
"""

from typing import Dict, Any, Union
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from model.config import TARGET_COLUMNS, MODEL_SAVE_DIR
from model.dataset import load_combined_data
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

        predictions = {}
        for target in TARGET_COLUMNS:
            model = self.models[target]
            pred_val = float(model.predict(df_feat)[0])
            predictions[target] = pred_val

        return predictions
