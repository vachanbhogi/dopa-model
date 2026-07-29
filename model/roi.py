"""
Neuroanatomical ROI Feature Engineering & Functional Network Extractor.

Maps 20,484 fsaverage5 vertex features / 75 Destrieux anatomical regions into 10 Core Functional Cognitive Networks:
1. Visual Cortex (Attention & Standout)
2. Auditory Cortex (Audio & Speech Engagement)
3. Dorsolateral Prefrontal Cortex / dlPFC (Cognitive Load)
4. Ventromedial Prefrontal Cortex / vMPFC (Reward & Value Offer)
5. Default Mode Network / DMN (Narrative Immersion)
6. Fusiform Face Area / FFA (Spokesperson & Face Prominence)
7. Insula & Visceral Sensory (Visceral Reaction)
8. Motor & Mirror Neurons (Physical Empathy & Relatability)
9. Language Network (Message Clarity)
10. Limbic System (Emotional Stakes)
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

COGNITIVE_NETWORKS = {
    "visual_cortex": {
        "name": "Visual Attention · Standout",
        "description": "Measures visual contrast, motion intensity, and focal point concentration in early visual cortex (V1-V4).",
        "keywords": ["occipital", "cuneus", "calcarine", "Lingual"],
        "ideal_range": (60, 95),
    },
    "auditory_cortex": {
        "name": "Audio Engagement · Soundscape",
        "description": "Reflects neural tracking of music dynamics, voice pitch variation, and sound design clarity.",
        "keywords": ["Tesschel", "temporal_transverse", "temp_sup", "Planum_temporale"],
        "ideal_range": (50, 90),
    },
    "dlpfc": {
        "name": "Cognitive Load · Complexity",
        "description": "Dorsolateral prefrontal cortex load—high values indicate dense text or rapid scene switching.",
        "keywords": ["front_middle", "front_sup"],
        "ideal_range": (20, 60),  # Lower is better for clarity
    },
    "vmpfc": {
        "name": "Reward & Value Offer",
        "description": "Ventromedial prefrontal activation—signals perceived utility, price value, and offer desirability.",
        "keywords": ["frontomargin", "rectus", "orbital"],
        "ideal_range": (65, 95),
    },
    "default_mode_network": {
        "name": "Narrative Immersion · Story",
        "description": "Default Mode Network activation—reflects storyline engagement, character identification, and immersion.",
        "keywords": ["precuneus", "cingul_Post", "cingul_Mid_Post", "Angular"],
        "ideal_range": (55, 90),
    },
    "fusiform_face_area": {
        "name": "Face & Spokesperson Prominence",
        "description": "Ventral stream activation corresponding to presenter presence, human faces, and gaze tracking.",
        "keywords": ["fusifor", "oc_temp_lat"],
        "ideal_range": (40, 85),
    },
    "insula": {
        "name": "Visceral & Sensory Reaction",
        "description": "Insular cortex activation—signals visceral gut reactions, taste/touch cues, and sensory immersion.",
        "keywords": ["insula"],
        "ideal_range": (45, 85),
    },
    "mirror_neuron_system": {
        "name": "Physical Empathy & Relatability",
        "description": "Premotor and parietal co-activation—measures viewer relatability to actions shown on screen.",
        "keywords": ["precentral", "postcentral"],
        "ideal_range": (50, 90),
    },
    "language_network": {
        "name": "Message Clarity · Wernicke/Broca",
        "description": "Broca & Wernicke network tracking speech comprehension, transcript flow, and key claims.",
        "keywords": ["Opercular", "Triangul", "Lat_Fis"],
        "ideal_range": (60, 95),
    },
    "limbic_system": {
        "name": "Emotional Stakes & Arousal",
        "description": "Cingulate and anterior temporal activation—signals emotional tension, urgency, and drive.",
        "keywords": ["cingul_Mid_Ant", "temp_inf"],
        "ideal_range": (50, 90),
    },
}


def _min_max_scale(val: float, v_min: float, v_max: float) -> float:
    if v_max <= v_min:
        return 50.0
    scaled = (val - v_min) / (v_max - v_min) * 100.0
    return float(np.clip(scaled, 0.0, 100.0))


def extract_cognitive_network_scores(brain_features: pd.DataFrame | Dict[str, float]) -> Dict[str, Any]:
    """
    Extract 10 functional network cognitive scores from 1,800 TRIBE v2 brain surface features.
    """
    if isinstance(brain_features, dict):
        df_row = pd.Series(brain_features)
    elif isinstance(brain_features, pd.DataFrame):
        df_row = brain_features.iloc[0]
    else:
        raise TypeError("brain_features must be dict or DataFrame")

    network_results = {}

    for net_key, net_info in COGNITIVE_NETWORKS.items():
        matching_cols = [
            col for col in df_row.index
            if any(kw in col for kw in net_info["keywords"])
        ]

        if not matching_cols:
            score = 50.0
            peak_sec = 2.5
            slope_val = 0.0
        else:
            vals = df_row[matching_cols].astype(float)
            mean_val = float(vals.mean())

            # Reference empirical distribution thresholds from AdsTrace cohort
            if net_key in ["visual_cortex", "auditory_cortex"]:
                score = _min_max_scale(mean_val, -0.05, 0.45)
            elif net_key == "dlpfc":
                score = _min_max_scale(mean_val, -0.02, 0.35)
            elif net_key == "vmpfc":
                score = _min_max_scale(mean_val, -0.04, 0.40)
            elif net_key == "default_mode_network":
                score = _min_max_scale(mean_val, -0.03, 0.38)
            else:
                score = _min_max_scale(mean_val, -0.05, 0.40)

            slope_cols = [c for c in matching_cols if "slope" in c]
            slope_val = float(df_row[slope_cols].mean()) if slope_cols else 0.0

            pf_cols = [c for c in matching_cols if "peak_fraction" in c]
            peak_frac = float(df_row[pf_cols].mean()) if pf_cols else 0.25
            peak_sec = float(np.clip(peak_frac * 15.0, 0.5, 14.5))

        # Categorize verdict
        ideal_low, ideal_high = net_info["ideal_range"]
        if net_key == "dlpfc":
            # For cognitive load, lower/moderate is optimal
            verdict = "Optimal" if score <= ideal_high else "High (Overload)" if score > 75 else "Low"
        else:
            verdict = "Strong" if score >= 75 else "Typical" if score >= 40 else "Weak"

        network_results[net_key] = {
            "key": net_key,
            "name": net_info["name"],
            "description": net_info["description"],
            "score": round(score, 1),
            "verdict": verdict,
            "peak_second": round(peak_sec, 1),
            "slope": round(slope_val, 4),
        }

    # Compute overall Hook Retention Score (0-3s intensity)
    visual_score = network_results["visual_cortex"]["score"]
    auditory_score = network_results["auditory_cortex"]["score"]
    dlpfc_score = network_results["dlpfc"]["score"]

    hook_score = float(np.clip(0.55 * visual_score + 0.35 * auditory_score + 0.10 * (100 - dlpfc_score), 0, 100))

    return {
        "networks": network_results,
        "hook_retention_score": round(hook_score, 1),
        "hook_verdict": "Strong Hook" if hook_score >= 70 else "Moderate Hook" if hook_score >= 45 else "Weak Hook",
    }


def compute_dataset_roi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized computation of 10 summary network features across entire dataset.
    """
    roi_dict = {}

    for net_key, net_info in COGNITIVE_NETWORKS.items():
        matching_cols = [
            col for col in df.columns
            if any(kw in col for kw in net_info["keywords"])
        ]
        if matching_cols:
            mean_vals = df[matching_cols].mean(axis=1).values
            v_min, v_max = (-0.05, 0.45) if net_key in ["visual_cortex", "auditory_cortex"] else (-0.02, 0.35) if net_key == "dlpfc" else (-0.04, 0.40) if net_key == "vmpfc" else (-0.03, 0.38) if net_key == "default_mode_network" else (-0.05, 0.40)
            scores = np.clip((mean_vals - v_min) / (v_max - v_min) * 100.0, 0.0, 100.0)

            slope_cols = [c for c in matching_cols if "slope" in c]
            slopes = df[slope_cols].mean(axis=1).values if slope_cols else np.zeros(len(df))
        else:
            scores = np.full(len(df), 50.0)
            slopes = np.zeros(len(df))

        roi_dict[f"roi_{net_key}_score"] = scores
        roi_dict[f"roi_{net_key}_slope"] = slopes

    vis_scores = roi_dict["roi_visual_cortex_score"]
    aud_scores = roi_dict["roi_auditory_cortex_score"]
    dlp_scores = roi_dict["roi_dlpfc_score"]

    roi_dict["roi_hook_retention_score"] = np.clip(
        0.55 * vis_scores + 0.35 * aud_scores + 0.10 * (100.0 - dlp_scores), 0.0, 100.0
    )

    df_roi = pd.DataFrame(roi_dict, index=df.index)
    return pd.concat([df, df_roi], axis=1)
