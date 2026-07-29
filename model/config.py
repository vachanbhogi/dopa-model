"""
Configuration and constants for TRIBE v2 Ad Metric Prediction.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "combined.parquet"
BENCHMARK_DIR = BASE_DIR / "benchmark"
MODEL_SAVE_DIR = BENCHMARK_DIR / "saved_models"

# Target Metrics
TARGET_COLUMNS = ["roi", "cvr", "mean_ictr", "max_ictr"]

# Target log-transform options (recommended for right-skewed variables)
LOG_TRANSFORM_TARGETS = {
    "roi": False,
    "cvr": True,
    "mean_ictr": True,
    "max_ictr": True,
}

# Modality definitions
MODALITIES = [
    "brain_text_audio",
    "brain_video",
    "brain_combined",
    "multimodal_full",
]

# Metadata columns for optional multimodal fusion
METADATA_COLUMNS = [
    "duration_seconds",
    "ad_type",
    "product_name",
    "presenter",
    "editing_technique",
    "promotion",
    "item_count",
]

# Default training & validation splits
DEFAULT_SPLIT_COL = "split"
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15
RANDOM_SEED = 42
