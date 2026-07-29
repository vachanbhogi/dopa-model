"""
Dataset loading, preprocessing, and feature matrix preparation for TRIBE v2 ad metric models.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from model.config import DATA_PATH, TARGET_COLUMNS, METADATA_COLUMNS, DEFAULT_SPLIT_COL


def load_combined_data(data_path: str | None = None) -> pd.DataFrame:
    """Load combined parquet file containing TRIBE v2 brain features and ad metrics."""
    path = data_path or DATA_PATH
    df = pd.read_parquet(path)
    return df


def get_feature_names(
    df: pd.DataFrame,
    modality: str = "brain_combined"
) -> List[str]:
    """Retrieve list of feature names based on specified modality strategy."""
    ta_cols = [c for c in df.columns if c.endswith("_ta")]
    vid_cols = [c for c in df.columns if c.endswith("_vid")]

    if modality == "brain_text_audio":
        return ta_cols
    elif modality == "brain_video":
        return vid_cols
    elif modality == "brain_combined":
        return ta_cols + vid_cols
    elif modality == "brain_combined_roi":
        roi_cols = [c for c in df.columns if c.startswith("roi_")]
        return ta_cols + vid_cols + roi_cols
    elif modality == "multimodal_full":
        # Brain features + encoded metadata
        brain_cols = ta_cols + vid_cols
        return brain_cols + METADATA_COLUMNS
    else:
        raise ValueError(f"Unknown modality: '{modality}'. Choose from: brain_text_audio, brain_video, brain_combined, brain_combined_roi, multimodal_full")


def prepare_features_and_targets(
    df: pd.DataFrame,
    target_name: str,
    modality: str = "brain_combined"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y for a specific target,
    cleaning null target values and encoding metadata if required.
    """
    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"Target '{target_name}' not in supported targets: {TARGET_COLUMNS}")

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_name]).copy()

    # If ROI features requested and not in clean_df, compute them dynamically
    if modality == "brain_combined_roi" and not any(c.startswith("roi_") for c in clean_df.columns):
        from model.roi import compute_dataset_roi_features
        clean_df = compute_dataset_roi_features(clean_df)

    ta_cols = [c for c in clean_df.columns if c.endswith("_ta")]
    vid_cols = [c for c in clean_df.columns if c.endswith("_vid")]
    brain_cols = ta_cols + vid_cols

    if modality == "brain_text_audio":
        X = clean_df[ta_cols].copy()
    elif modality == "brain_video":
        X = clean_df[vid_cols].copy()
    elif modality == "brain_combined":
        X = clean_df[brain_cols].copy()
    elif modality == "brain_combined_roi":
        roi_cols = [c for c in clean_df.columns if c.startswith("roi_")]
        X = clean_df[brain_cols + roi_cols].copy()
    elif modality == "multimodal_full":
        # One-hot encode categoricals for metadata fusion
        X_brain = clean_df[brain_cols].copy()
        meta_subset = clean_df[['ad_type', 'product_name', 'duration_seconds']].fillna('missing')
        X_meta = pd.get_dummies(meta_subset, drop_first=True)
        X = pd.concat([X_brain, X_meta], axis=1)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    y = clean_df[target_name].copy()
    return X, y, clean_df['split'] if 'split' in clean_df.columns else None


def get_train_val_test_splits(
    df: pd.DataFrame,
    target_name: str,
    modality: str = "brain_combined",
    split_col: str = DEFAULT_SPLIT_COL
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split dataset into train, val, and test partitions according to the AdsTrace split column.
    """
    X, y, splits = prepare_features_and_targets(df, target_name, modality=modality)

    if splits is None or split_col not in df.columns:
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=42)
        return X_train, y_train, X_val, y_val, X_test, y_test

    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask]
    )
