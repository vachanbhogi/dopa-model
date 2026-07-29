#!/usr/bin/env python3
"""
Comprehensive Benchmark Script: Test All Advanced Strategies on combined.parquet.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TARGET_COLUMNS, BENCHMARK_DIR
from model.dataset import load_combined_data, get_train_val_test_splits
from model.advanced_strategies import BrainConnectivityExtractor, OrdinalTierClassifier, GatedFusionRegressor


def evaluate_strategies():
    df = load_combined_data()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 95)
    print("TESTING ALL ADVANCED STRATEGIES ON DATA/COMBINED.PARQUET")
    print("=" * 95)

    for target in TARGET_COLUMNS:
        print(f"\n>>>> TARGET: {target.upper()} <<<<")

        X_tr, y_tr, X_va, y_va, X_te, y_te = get_train_val_test_splits(df, target_name=target, modality="brain_combined")

        # -------------------------------------------------------------------
        # 1. Baseline Ridge Model (k=150, alpha=200.0)
        # -------------------------------------------------------------------
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        sel = SelectKBest(f_regression, k=150)
        X_tr_sel = sel.fit_transform(X_tr_s, y_tr)
        X_va_sel = sel.transform(X_va_s)
        X_te_sel = sel.transform(X_te_s)

        base_model = Ridge(alpha=200.0, random_state=42)
        base_model.fit(X_tr_sel, y_tr)
        base_preds = base_model.predict(X_te_sel)

        pr_base, _ = pearsonr(y_te, base_preds)
        r2_base = r2_score(y_te, base_preds)
        acc_1std_base = np.mean(np.abs(y_te - base_preds) <= y_tr.std()) * 100.0
        acc_2std_base = np.mean(np.abs(y_te - base_preds) <= 2 * y_tr.std()) * 100.0

        print(f"[Baseline Ridge]         | Test Pearson r={pr_base:6.4f} | R2={r2_base:6.4f} | Acc(1std)={acc_1std_base:5.1f}% | Acc(2std)={acc_2std_base:5.1f}%")
        results.append({"strategy": "Baseline Ridge", "target": target, "pearson_r": pr_base, "r2": r2_base, "acc_1std": acc_1std_base, "acc_2std": acc_2std_base})

        # -------------------------------------------------------------------
        # 2. Strategy A: Cortical Connectivity Features (Cross-ROI Interactions)
        # -------------------------------------------------------------------
        conn_extractor = BrainConnectivityExtractor(top_n_single=80)
        X_tr_conn = conn_extractor.fit_transform(X_tr, y_tr)
        X_te_conn = conn_extractor.transform(X_te)

        conn_model = Ridge(alpha=500.0, random_state=42)
        conn_model.fit(X_tr_conn, y_tr)
        conn_preds = conn_model.predict(X_te_conn)

        pr_conn, _ = pearsonr(y_te, conn_preds)
        r2_conn = r2_score(y_te, conn_preds)
        acc_1std_conn = np.mean(np.abs(y_te - conn_preds) <= y_tr.std()) * 100.0
        acc_2std_conn = np.mean(np.abs(y_te - conn_preds) <= 2 * y_tr.std()) * 100.0

        print(f"[Strategy A: Connectivity]| Test Pearson r={pr_conn:6.4f} | R2={r2_conn:6.4f} | Acc(1std)={acc_1std_conn:5.1f}% | Acc(2std)={acc_2std_conn:5.1f}%")
        results.append({"strategy": "Strategy A: Connectivity", "target": target, "pearson_r": pr_conn, "r2": r2_conn, "acc_1std": acc_1std_conn, "acc_2std": acc_2std_conn})

        # -------------------------------------------------------------------
        # 3. Strategy B: Ordinal 5-Tier Classification
        # -------------------------------------------------------------------
        ord_model = OrdinalTierClassifier(n_tiers=5, k_features=150, random_state=42)
        ord_model.fit(X_tr, y_tr)

        y_te_bins = ord_model._discretize(y_te)
        ord_preds = ord_model.predict(X_te)

        exact_acc = accuracy_score(y_te_bins, ord_preds) * 100.0
        within1_tier_acc = np.mean(np.abs(y_te_bins - ord_preds) <= 1) * 100.0

        print(f"[Strategy B: Ordinal Tiers]| Exact Tier Acc={exact_acc:5.1f}% | Within +/-1 Tier Acc={within1_tier_acc:5.1f}%")
        results.append({"strategy": "Strategy B: Ordinal Tiers", "target": target, "exact_acc": exact_acc, "within1_tier_acc": within1_tier_acc})

        # -------------------------------------------------------------------
        # 4. Strategy C: Gated Late Fusion Regressor
        # -------------------------------------------------------------------
        gated_model = GatedFusionRegressor(k_features=75, alpha=200.0, random_state=42)
        gated_model.fit(X_tr, y_tr)
        gated_preds = gated_model.predict(X_te)

        pr_gated, _ = pearsonr(y_te, gated_preds)
        r2_gated = r2_score(y_te, gated_preds)
        acc_1std_gated = np.mean(np.abs(y_te - gated_preds) <= y_tr.std()) * 100.0
        acc_2std_gated = np.mean(np.abs(y_te - gated_preds) <= 2 * y_tr.std()) * 100.0

        print(f"[Strategy C: Gated Fusion] | Test Pearson r={pr_gated:6.4f} | R2={r2_gated:6.4f} | Acc(1std)={acc_1std_gated:5.1f}% | Acc(2std)={acc_2std_gated:5.1f}%")
        results.append({"strategy": "Strategy C: Gated Fusion", "target": target, "pearson_r": pr_gated, "r2": r2_gated, "acc_1std": acc_1std_gated, "acc_2std": acc_2std_gated})

        # -------------------------------------------------------------------
        # 5. Strategy E: Robust Huber Outlier-Tuned Regression
        # -------------------------------------------------------------------
        huber = HuberRegressor(epsilon=1.35, alpha=100.0, max_iter=500)
        huber.fit(X_tr_sel, y_tr)
        huber_preds = huber.predict(X_te_sel)

        pr_huber, _ = pearsonr(y_te, huber_preds)
        r2_huber = r2_score(y_te, huber_preds)
        acc_1std_huber = np.mean(np.abs(y_te - huber_preds) <= y_tr.std()) * 100.0
        acc_2std_huber = np.mean(np.abs(y_te - huber_preds) <= 2 * y_tr.std()) * 100.0

        print(f"[Strategy E: Huber Loss]   | Test Pearson r={pr_huber:6.4f} | R2={r2_huber:6.4f} | Acc(1std)={acc_1std_huber:5.1f}% | Acc(2std)={acc_2std_huber:5.1f}%")
        results.append({"strategy": "Strategy E: Huber Loss", "target": target, "pearson_r": pr_huber, "r2": r2_huber, "acc_1std": acc_1std_huber, "acc_2std": acc_2std_huber})

    # Save benchmark report
    save_path = BENCHMARK_DIR / "strategy_benchmark_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print(f"STRATEGY BENCHMARK COMPLETE! Results saved to '{save_path}'")
    print("=" * 95)


if __name__ == "__main__":
    evaluate_strategies()
