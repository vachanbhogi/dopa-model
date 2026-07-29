"""
Brain feature interpretation and cortical region importance extraction.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def extract_top_brain_features(
    model,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """Extract top cortical features based on model weights or feature selector scores."""
    scores = None

    if hasattr(model, "selector_") and model.selector_ is not None:
        selected_mask = model.selector_.get_support()
        selected_features = [f for f, s in zip(feature_names, selected_mask) if s]

        inner_model = getattr(model, "model_", None)
        if inner_model is not None and hasattr(inner_model, "coef_"):
            coefs = np.abs(inner_model.coef_)
            df_feat = pd.DataFrame({"feature": selected_features, "importance": coefs})
        elif hasattr(model.selector_, "scores_"):
            sel_scores = model.selector_.scores_[selected_mask]
            df_feat = pd.DataFrame({"feature": selected_features, "importance": sel_scores})
        else:
            df_feat = pd.DataFrame({"feature": selected_features, "importance": np.ones(len(selected_features))})
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        df_feat = pd.DataFrame({"feature": feature_names, "importance": coefs})
    else:
        df_feat = pd.DataFrame({"feature": feature_names, "importance": np.ones(len(feature_names))})

    df_feat = df_feat.sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Parse Destrieux anatomical atlas ROI components
    def parse_roi(feat_name: str) -> Dict[str, str]:
        parts = feat_name.split("__")
        stat = parts[1] if len(parts) > 1 else "unknown"
        hemi_roi = parts[0]
        modality = "text-audio" if feat_name.endswith("_ta") else "video" if feat_name.endswith("_vid") else "other"

        hemi = "Left" if hemi_roi.startswith("lh_") else "Right" if hemi_roi.startswith("rh_") else "Unknown"
        region = hemi_roi.replace("lh_", "").replace("rh_", "").replace("_ta", "").replace("_vid", "")
        return {"hemisphere": hemi, "anatomical_region": region, "statistic": stat, "modality": modality}

    parsed = df_feat["feature"].apply(parse_roi).apply(pd.Series)
    df_result = pd.concat([df_feat, parsed], axis=1)
    return df_result.head(top_n)
