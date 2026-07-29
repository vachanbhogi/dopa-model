"""
TRIBE v2 Ad Metric Prediction Library.
"""

from model.config import TARGET_COLUMNS, MODALITIES
from model.dataset import load_combined_data, get_train_val_test_splits
from model.models import BrainMetricRegressor
from model.train import train_single_target_model, train_all_targets
from model.evaluate import calculate_metrics
from model.interpret import extract_top_brain_features
from model.inference import AdMetricPredictor
from model.advanced_strategies import BrainConnectivityExtractor, OrdinalTierClassifier, GatedFusionRegressor
from model.timeline import TimelinePredictor

__all__ = [
    "TARGET_COLUMNS",
    "MODALITIES",
    "load_combined_data",
    "get_train_val_test_splits",
    "BrainMetricRegressor",
    "train_single_target_model",
    "train_all_targets",
    "calculate_metrics",
    "extract_top_brain_features",
    "AdMetricPredictor",
    "BrainConnectivityExtractor",
    "OrdinalTierClassifier",
    "GatedFusionRegressor",
    "TimelinePredictor",
]
