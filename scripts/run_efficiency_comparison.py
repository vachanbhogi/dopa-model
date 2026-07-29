"""
Benchmarking & Efficiency Comparison Runner.

Compares Baseline TRIBE v2 model against Enhanced Neuro-ROI & Multi-Metric pipeline across:
1. Predictive accuracy (R2, Pearson r, Spearman r, MAE, RMSE, Tier Classification Accuracy)
2. Execution efficiency (Feature Extraction ms, Training s, Inference ms, Memory MB)
3. Output Richness (Number of metrics, cognitive scores, action directives, confidence scores)
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TARGET_COLUMNS, BENCHMARK_DIR
from model.dataset import load_combined_data, get_train_val_test_splits
from model.models import BrainMetricRegressor
from model.evaluate import calculate_metrics
from model.roi import extract_cognitive_network_scores, compute_dataset_roi_features
from model.timeline import TimelinePredictor
from model.inference import AdMetricPredictor


def run_efficiency_comparison():
    print("=================================================================")
    print("   TRIBE v2 AD METRIC MODEL: EFFICIENCY & ENHANCEMENT BENCHMARK  ")
    print("=================================================================\n")

    tracemalloc.start()

    # 1. Load data
    t0 = time.perf_counter()
    df = load_combined_data()
    t_load = (time.perf_counter() - t0) * 1000.0

    print(f"Loaded dataset: {len(df)} samples, {df.shape[1]} columns ({t_load:.1f} ms)\n")

    # 2. Benchmark Baseline Pipeline (brain_combined)
    print("--- 1. BENCHMARKING BASELINE MODEL (brain_combined: 1800 vertices) ---")
    baseline_results = {}
    baseline_train_times = []

    for target in TARGET_COLUMNS:
        X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(
            df, target, modality="brain_combined"
        )

        t_start = time.perf_counter()
        model = BrainMetricRegressor(model_type="ridge", k_features=150, alpha=200.0)
        model.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t_start
        baseline_train_times.append(t_fit)

        preds_val = model.predict(X_va)
        preds_test = model.predict(X_te)

        val_m = calculate_metrics(y_va, preds_val)
        test_m = calculate_metrics(y_te, preds_test)

        baseline_results[target] = {
            "fit_time_seconds": round(t_fit, 3),
            "val_metrics": val_m,
            "test_metrics": test_m,
        }
        print(f" Target [{target:10s}] -> Val R2: {val_m['r2']:.4f}, Val Pearson r: {val_m['pearson_r']:.4f}, Test R2: {test_m['r2']:.4f}, Fit: {t_fit:.2f}s")

    # 3. Benchmark Enhanced Pipeline (brain_combined_roi)
    print("\n--- 2. BENCHMARKING ENHANCED NEURO-ROI MODEL (10 Cognitive Networks + Directives) ---")
    t0_roi = time.perf_counter()
    df_roi = compute_dataset_roi_features(df)
    t_roi_extract = (time.perf_counter() - t0_roi) * 1000.0 / len(df)  # per sample ms

    enhanced_results = {}
    enhanced_train_times = []

    for target in TARGET_COLUMNS:
        X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(
            df_roi, target, modality="brain_combined_roi"
        )

        t_start = time.perf_counter()
        model = BrainMetricRegressor(model_type="ridge", k_features=150, alpha=200.0)
        model.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t_start
        enhanced_train_times.append(t_fit)

        preds_val = model.predict(X_va)
        preds_test = model.predict(X_te)

        val_m = calculate_metrics(y_va, preds_val)
        test_m = calculate_metrics(y_te, preds_test)

        enhanced_results[target] = {
            "fit_time_seconds": round(t_fit, 3),
            "val_metrics": val_m,
            "test_metrics": test_m,
        }
        print(f" Target [{target:10s}] -> Val R2: {val_m['r2']:.4f}, Val Pearson r: {val_m['pearson_r']:.4f}, Test R2: {test_m['r2']:.4f}, Fit: {t_fit:.2f}s")

    # 4. Latency & Inference Benchmark
    print("\n--- 3. INFERENCE LATENCY & MEMORY BENCHMARK ---")
    predictor = AdMetricPredictor()
    timeline_predictor = TimelinePredictor(predictor=predictor)
    sample_brain = df.iloc[[0]]

    # Measure single ad inference time for Baseline Predictor
    t_inf_base = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = predictor.predict(sample_brain)
        t_inf_base.append((time.perf_counter() - t0) * 1000.0)

    # Measure single ad inference time for Enhanced TimelinePredictor
    t_inf_enh = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = timeline_predictor.predict_timeline(sample_brain, duration_seconds=15)
        t_inf_enh.append((time.perf_counter() - t0) * 1000.0)

    avg_base_inf_ms = float(np.mean(t_inf_base))
    avg_enh_inf_ms = float(np.mean(t_inf_enh))

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem / (1024.0 * 1024.0)

    print(f" Baseline Single-Ad Inference Latency: {avg_base_inf_ms:.2f} ms")
    print(f" Enhanced Single-Ad Inference Latency:  {avg_enh_inf_ms:.2f} ms (includes 10 ROI scores + Timeline + Directives)")
    print(f" Feature Engineering Extraction Speed:  {t_roi_extract:.3f} ms / sample")
    print(f" Peak Memory Overhead:                  {peak_mem_mb:.2f} MB")

    # 5. Output Richness Comparison
    richness_comparison = {
        "baseline": {
            "features_used": 1800,
            "metrics_predicted": len(TARGET_COLUMNS),
            "cognitive_networks_mapped": 0,
            "action_directives_generated": 0,
            "executive_scorecard_provided": False,
            "confidence_scores_provided": False,
        },
        "enhanced": {
            "features_used": 1821,
            "metrics_predicted": len(TARGET_COLUMNS),
            "cognitive_networks_mapped": 10,
            "action_directives_generated": 3,  # average per ad
            "executive_scorecard_provided": True,
            "confidence_scores_provided": True,
        }
    }

    # 6. Aggregate Comparison Metrics
    base_val_r2 = float(np.mean([baseline_results[t]["val_metrics"]["r2"] for t in TARGET_COLUMNS]))
    enh_val_r2 = float(np.mean([enhanced_results[t]["val_metrics"]["r2"] for t in TARGET_COLUMNS]))

    base_val_pearson = float(np.mean([baseline_results[t]["val_metrics"]["pearson_r"] for t in TARGET_COLUMNS]))
    enh_val_pearson = float(np.mean([enhanced_results[t]["val_metrics"]["pearson_r"] for t in TARGET_COLUMNS]))

    base_val_tier_acc = float(np.mean([baseline_results[t]["val_metrics"]["tier_classification_accuracy"] for t in TARGET_COLUMNS]))
    enh_val_tier_acc = float(np.mean([enhanced_results[t]["val_metrics"]["tier_classification_accuracy"] for t in TARGET_COLUMNS]))

    comparison_report = {
        "summary": {
            "baseline_mean_val_r2": round(base_val_r2, 4),
            "enhanced_mean_val_r2": round(enh_val_r2, 4),
            "r2_improvement_pct": round(((enh_val_r2 - base_val_r2) / max(0.001, abs(base_val_r2))) * 100.0, 2),

            "baseline_mean_val_pearson_r": round(base_val_pearson, 4),
            "enhanced_mean_val_pearson_r": round(enh_val_pearson, 4),
            "pearson_r_improvement_pct": round(((enh_val_pearson - base_val_pearson) / max(0.001, abs(base_val_pearson))) * 100.0, 2),

            "baseline_mean_tier_acc": round(base_val_tier_acc, 2),
            "enhanced_mean_tier_acc": round(enh_val_tier_acc, 2),

            "baseline_avg_train_time_sec": round(float(np.mean(baseline_train_times)), 2),
            "enhanced_avg_train_time_sec": round(float(np.mean(enhanced_train_times)), 2),

            "baseline_inference_ms": round(avg_base_inf_ms, 2),
            "enhanced_inference_ms": round(avg_enh_inf_ms, 2),

            "peak_memory_mb": round(peak_mem_mb, 2),
        },
        "output_richness": richness_comparison,
        "baseline_model_results": baseline_results,
        "enhanced_model_results": enhanced_results,
    }

    # Save benchmark file
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BENCHMARK_DIR / "efficiency_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison_report, f, indent=2)

    print("\n=================================================================")
    print("   SUMMARY COMPARISON RESULTS")
    print("=================================================================")
    print(f" Mean Val Pearson r:     Baseline {base_val_pearson:.4f} -> Enhanced {enh_val_pearson:.4f} (+{comparison_report['summary']['pearson_r_improvement_pct']}%)")
    print(f" Mean Val R2:            Baseline {base_val_r2:.4f} -> Enhanced {enh_val_r2:.4f}")
    print(f" Tier Accuracy:          Baseline {base_val_tier_acc:.2f}% -> Enhanced {enh_val_tier_acc:.2f}%")
    print(f" Single-Ad Inference:    {avg_enh_inf_ms:.2f} ms (Enhanced multi-metric pipeline)")
    print(f" Report saved to:        {out_path}")
    print("=================================================================\n")

    return comparison_report


if __name__ == "__main__":
    run_efficiency_comparison()
