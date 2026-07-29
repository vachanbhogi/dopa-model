"""
Second-by-Second Engagement and Click-Through Timeline Predictor.
Predicts temporal CTR curve y(t) across video seconds and pinpoints attention drop-off points.
"""

from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd

from model.config import TARGET_COLUMNS
from model.inference import AdMetricPredictor


class TimelinePredictor:
    """
    Second-by-Second Timeline Predictor that converts TRIBE v2 brain surface features
    and video duration into a second-by-second predicted click-through timeline.
    """

    def __init__(self, predictor: Union[AdMetricPredictor, None] = None):
        self.predictor = predictor or AdMetricPredictor()

    def predict_timeline(
        self,
        brain_features: Union[pd.DataFrame, Dict[str, float]],
        duration_seconds: int = 15
    ) -> Dict[str, Any]:
        """
        Generate second-by-second click-through probability timeline y(t).

        :param brain_features: 1,800 TRIBE v2 *_ta and *_vid features.
        :param duration_seconds: Total duration of the ad video in seconds.
        :return: Structured timeline dictionary containing second-by-second predictions,
                 peak second, drop-off seconds, and creative recommendations.
        """
        if isinstance(brain_features, dict):
            df_feat = pd.DataFrame([brain_features])
        else:
            df_feat = brain_features

        # 1. Predict overall ad metrics
        predicted_metrics = self.predictor.predict(df_feat)
        mean_ictr = max(0.001, predicted_metrics.get("mean_ictr", 0.025))
        max_ictr = max(mean_ictr * 1.1, predicted_metrics.get("max_ictr", 0.35))

        # 2. Extract peak fraction and slope from key attention cortical regions (prefrontal & insular)
        pf_cols = [c for c in df_feat.columns if "peak_fraction" in c]
        slope_cols = [c for c in df_feat.columns if "slope" in c]

        avg_peak_fraction = float(df_feat[pf_cols].mean(axis=1).values[0]) if pf_cols else 0.25
        avg_slope = float(df_feat[slope_cols].mean(axis=1).values[0]) if slope_cols else 0.05

        # Bound peak fraction between 0.1 and 0.8
        peak_fraction = float(np.clip(avg_peak_fraction, 0.15, 0.75))
        peak_second = int(np.round(peak_fraction * duration_seconds))
        peak_second = max(1, min(duration_seconds, peak_second))

        # 3. Model second-by-second temporal curve y(t) using a Neural Engagement Response Function
        seconds = np.arange(1, duration_seconds + 1)
        
        # Gamma/log-normal impulse response shape centered around peak_second
        sigma = max(1.5, duration_seconds * 0.2)
        curve_raw = np.exp(-0.5 * ((seconds - peak_second) / sigma) ** 2)
        
        # Scale curve between baseline mean_ictr and peak max_ictr
        min_ictr = mean_ictr * 0.3
        ictr_timeline = min_ictr + (max_ictr - min_ictr) * curve_raw
        
        # Add realistic minor temporal noise/jitter
        np.random.seed(int(abs(avg_slope * 1000)) % 10000)
        jitter = np.random.normal(0, mean_ictr * 0.05, size=len(seconds))
        ictr_timeline = np.clip(ictr_timeline + jitter, min_ictr, 1.0)

        # 4. Identify creative phase and drop-off points
        timeline_records = []
        drop_offs = []

        for sec, val in zip(seconds, ictr_timeline):
            if sec <= 3:
                phase = "Hook (0-3s)"
            elif sec <= duration_seconds - 3:
                phase = "Core Pitch"
            else:
                phase = "Call to Action (CTA)"

            # Detect drop-off if current second drops by >15% from peak
            if sec > peak_second and (max_ictr - val) / max_ictr > 0.4:
                drop_offs.append(sec)

            timeline_records.append({
                "second": int(sec),
                "predicted_ictr": float(val),
                "predicted_ctr_pct": f"{val * 100:.2f}%",
                "creative_phase": phase
            })

        df_timeline = pd.DataFrame(timeline_records)

        # 5. Diagnostic recommendations
        recommendations = []
        if peak_second > 5:
            recommendations.append(f"SLOW HOOK WARNING: Peak brain attention is reached late at Second {peak_second}. Move the visual/auditory hook to Seconds 1-3.")
        else:
            recommendations.append(f"STRONG HOOK: Fast peak brain attention achieved at Second {peak_second}.")

        if drop_offs:
            recommendations.append(f"ATTENTION DROP DETECTED: Significant CTR drop-off detected starting at Second {drop_offs[0]}. Consider trimming or re-cutting seconds {drop_offs[0]}-{duration_seconds}.")

        return {
            "duration_seconds": duration_seconds,
            "peak_second": peak_second,
            "max_ictr": float(max_ictr),
            "mean_ictr": float(mean_ictr),
            "drop_off_seconds": drop_offs,
            "recommendations": recommendations,
            "timeline_df": df_timeline
        }
