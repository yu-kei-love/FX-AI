# ===========================================
# fx_optuna_v2.py
# Optuna re-optimization with reduced search space
# - Only 4 LightGBM params (n_estimators, learning_rate, max_depth, min_child_samples)
# - Walk-Forward expanding window validation (no random split)
# - Optimizes Profit Factor (not accuracy)
# - 50 trials max
# - Compares best PF vs current default parameters
# ===========================================

import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Common modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, add_regime_features, FEATURE_COLS
from research.common.labels import build_triple_barrier_labels
from research.common.validation import walk_forward_splits, compute_metrics
from research.common.economic_surprise import add_surprise_features, INDICATORS
from research.common.cftc_positions import add_cot_features, COT_FEATURE_COLS

script_dir = Path(__file__).resolve().parent

# ===== Fixed parameters (not optimized) =====
BARRIER_UP = 0.005
BARRIER_DOWN = -0.003
BARRIER_T = 24
VOL_MULT = 2.0
REGIME_CHANGE_THRESH = 2
PRICE_CHANGE_MULT = 3.0
ADOPTION_TARGET = 0.4

# Walk-Forward settings (same as 14_main_system.py)
WF_MIN_TRAIN = 4320   # ~6 months
WF_TEST_SIZE = 720    # ~1 month

# Default parameters (current production values from 14_main_system.py)
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_samples": 20,  # LightGBM default
}

# ===== Data loading (once) =====
print("Loading data...")
df = load_usdjpy_1h()
df = add_technical_features(df)
df = add_regime_features(df)
df = add_rate_features(df)
df = add_daily_trend_features(df)
df = add_surprise_features(df)
df = add_cot_features(df)

df["Close_4h_later"] = df["Close"].shift(-4)
df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)
df["Return_4h"] = (df["Close_4h_later"] - df["Close"]) / df["Close"]
df["Regime_changes_3h"] = (df["Regime"].diff().fillna(0) != 0).astype(int).rolling(3).sum().fillna(0)
df["Abs_ret_1h"] = df["Return_1"].abs()

surprise_cols = [f"surprise_{sid}" for sid in INDICATORS] + ["surprise_composite"]
cot_cols = [c for c in COT_FEATURE_COLS if c in df.columns]
feature_cols = FEATURE_COLS + [c for c in surprise_cols if c in df.columns] + cot_cols
df = df.dropna(subset=feature_cols + ["Label", "Return_4h"])
X = df[feature_cols]
y_direction = df["Label"].values
close_arr = df["Close"].values
ret4_arr = df["Return_4h"].values
regime_all = df["Regime"].values
vol_all = df["Volatility_24"].values
rc_all = df["Regime_changes_3h"].values
abs_ret_all = df["Abs_ret_1h"].values
n_total = len(df)

print(f"Data: {n_total} bars")

# Triple-Barrier labels (fixed parameters)
y_triple = build_triple_barrier_labels(close_arr, BARRIER_UP, BARRIER_DOWN, BARRIER_T)

# Walk-Forward splits (expanding window)
wf_splits = walk_forward_splits(n_total, WF_MIN_TRAIN, WF_TEST_SIZE)
print(f"Walk-Forward windows: {len(wf_splits)}")


def evaluate_wf(params):
    """
    Evaluate parameters using Walk-Forward expanding window.
    Returns mean Profit Factor across all windows.
    Uses Model A (Triple-Barrier) + Model B (Meta-Labeling) same as 11_optuna_optimize.py.
    """
    lgb_params = {
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "min_child_samples": params["min_child_samples"],
        "random_state": 42,
        "verbosity": -1,
    }

    pf_list = []

    for train_idx, test_idx in wf_splits:
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_dir_tr = y_direction[train_idx]
        y_triple_tr = y_triple[train_idx]
        regime_te = regime_all[test_idx]
        vol_te = vol_all[test_idx]
        rc_te = rc_all[test_idx]
        abs_te = abs_ret_all[test_idx]
        ret4_te = ret4_arr[test_idx]
        hist_vol = vol_all[train_idx].mean()
        hist_abs = abs_ret_all[train_idx].mean()
        if hist_abs <= 0:
            hist_abs = 1e-8

        # Model A (Triple-Barrier)
        mask_a = (y_triple_tr == 0) | (y_triple_tr == 1)
        if mask_a.sum() < 10:
            continue
        model_a = lgb.LGBMClassifier(**lgb_params)
        model_a.fit(X_tr[mask_a], y_triple_tr[mask_a].astype(int))

        # Model B (Direction + Meta-Labeling)
        model_b_p = lgb.LGBMClassifier(**lgb_params)
        model_b_p.fit(X_tr, y_dir_tr)
        X_tr_meta = X_tr.copy()
        X_tr_meta["primary_proba"] = model_b_p.predict_proba(X_tr)[:, 1]
        y_meta = (model_b_p.predict(X_tr) == y_dir_tr).astype(int)
        model_b_s = lgb.LGBMClassifier(**lgb_params)
        model_b_s.fit(X_tr_meta, y_meta)

        X_te_meta = X_te.copy()
        X_te_meta["primary_proba"] = model_b_p.predict_proba(X_te)[:, 1]
        proba_adopt = model_b_s.predict_proba(X_te_meta)[:, 1]
        thresh = np.percentile(
            model_b_s.predict_proba(X_tr_meta)[:, 1],
            100 - ADOPTION_TARGET * 100,
        )
        adopt_b = proba_adopt >= thresh

        # Wait conditions
        wait_c = (
            (vol_te > VOL_MULT * hist_vol)
            | (rc_te >= REGIME_CHANGE_THRESH)
            | (abs_te > PRICE_CHANGE_MULT * hist_abs)
        )

        pred_a = model_a.predict(X_te)
        pred_b = model_b_p.predict(X_te)

        # Trade return calculation
        trade_returns = []
        for i in range(len(test_idx)):
            r = regime_te[i]
            if r == 0:
                direction = int(pred_a[i])
            elif r == 1:
                if adopt_b[i]:
                    direction = int(pred_b[i])
                else:
                    continue
            else:
                continue
            if wait_c[i]:
                continue
            direction_mult = 1.0 if direction == 1 else -1.0
            trade_returns.append(ret4_te[i] * direction_mult)

        if len(trade_returns) >= 20:
            m = compute_metrics(np.array(trade_returns))
            if not np.isnan(m["pf"]) and not np.isinf(m["pf"]):
                pf_list.append(m["pf"])

    if len(pf_list) == 0:
        return 0.0, []

    return np.mean(pf_list), pf_list


def objective(trial):
    """Optuna objective: maximize Walk-Forward mean Profit Factor."""
    # Only 4 LightGBM parameters
    n_estimators = trial.suggest_int("n_estimators", 100, 400, step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    min_child_samples = trial.suggest_int("min_child_samples", 10, 100, step=10)

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
    }

    mean_pf, pf_list = evaluate_wf(params)

    # Cap PF at 3.0 to penalize suspiciously high values (overfitting)
    capped_pf = min(mean_pf, 3.0)

    # Log per-window PFs for monitoring
    if pf_list:
        trial.set_user_attr("pf_per_window", [round(p, 4) for p in pf_list])
        trial.set_user_attr("pf_std", round(float(np.std(pf_list)), 4))
        trial.set_user_attr("n_windows_valid", len(pf_list))

    return capped_pf


# ===== Evaluate default parameters first =====
print("\n" + "=" * 60)
print("Evaluating DEFAULT parameters (baseline)...")
print(f"  n_estimators={DEFAULT_PARAMS['n_estimators']}, "
      f"learning_rate={DEFAULT_PARAMS['learning_rate']}, "
      f"max_depth={DEFAULT_PARAMS['max_depth']}, "
      f"min_child_samples={DEFAULT_PARAMS['min_child_samples']}")

t0 = time.time()
default_mean_pf, default_pf_list = evaluate_wf(DEFAULT_PARAMS)
elapsed_default = time.time() - t0

print(f"  Default Walk-Forward mean PF: {default_mean_pf:.4f}")
if default_pf_list:
    print(f"  Per-window PFs: {[round(p, 4) for p in default_pf_list]}")
    print(f"  PF std: {np.std(default_pf_list):.4f}")
    print(f"  Windows with PF >= 1.0: {sum(1 for p in default_pf_list if p >= 1.0)}/{len(default_pf_list)}")
print(f"  Time: {elapsed_default:.1f}s")

# ===== Optuna optimization =====
print("\n" + "=" * 60)
print("Optuna Walk-Forward Profit Factor optimization (50 trials, 4 params)")
print("Search space:")
print("  n_estimators:     [100, 400] step=50")
print("  learning_rate:    [0.01, 0.1] (log)")
print("  max_depth:        [3, 8]")
print("  min_child_samples:[10, 100] step=10")
print("=" * 60)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
t0 = time.time()
study.optimize(objective, n_trials=50, show_progress_bar=True)
elapsed_optuna = time.time() - t0

# ===== Results =====
best = study.best_params
best_value = study.best_value
best_trial = study.best_trial

print(f"\n{'=' * 60}")
print("OPTIMIZATION RESULTS")
print(f"{'=' * 60}")
print(f"\nBest parameters:")
print(f"  n_estimators:      {best['n_estimators']}")
print(f"  learning_rate:     {best['learning_rate']:.6f}")
print(f"  max_depth:         {best['max_depth']}")
print(f"  min_child_samples: {best['min_child_samples']}")
print(f"\nBest Walk-Forward mean PF: {best_value:.4f}")

if "pf_per_window" in best_trial.user_attrs:
    print(f"  Per-window PFs: {best_trial.user_attrs['pf_per_window']}")
    print(f"  PF std: {best_trial.user_attrs['pf_std']}")
    print(f"  Valid windows: {best_trial.user_attrs['n_windows_valid']}")

print(f"\n--- Comparison: Best vs Default ---")
print(f"  Default mean PF: {default_mean_pf:.4f}")
print(f"  Best mean PF:    {best_value:.4f}")
delta = best_value - default_mean_pf
pct = (delta / default_mean_pf * 100) if default_mean_pf > 0 else float("inf")
print(f"  Delta:           {delta:+.4f} ({pct:+.1f}%)")

if default_pf_list:
    default_std = np.std(default_pf_list)
else:
    default_std = float("nan")
best_std = best_trial.user_attrs.get("pf_std", float("nan"))
print(f"  Default PF std:  {default_std:.4f}")
print(f"  Best PF std:     {best_std}")

print(f"\nTotal optimization time: {elapsed_optuna:.1f}s ({elapsed_optuna/50:.1f}s/trial)")

# ===== Top 5 trials =====
print(f"\n--- Top 5 Trials ---")
sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)
for i, t in enumerate(sorted_trials[:5]):
    print(f"  #{i+1} PF={t.value:.4f} | n_est={t.params['n_estimators']}, "
          f"lr={t.params['learning_rate']:.4f}, depth={t.params['max_depth']}, "
          f"min_child={t.params['min_child_samples']}")

# ===== Save results =====
results_path = script_dir / "fx_optuna_v2_results.txt"
with open(results_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("fx_optuna_v2 - Optuna Re-optimization Results\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")

    f.write("SEARCH SPACE (4 parameters only):\n")
    f.write("  n_estimators:      [100, 400] step=50\n")
    f.write("  learning_rate:     [0.01, 0.1] (log scale)\n")
    f.write("  max_depth:         [3, 8]\n")
    f.write("  min_child_samples: [10, 100] step=10\n\n")

    f.write("VALIDATION: Walk-Forward Expanding Window\n")
    f.write(f"  Min train size: {WF_MIN_TRAIN} bars (~6 months)\n")
    f.write(f"  Test size:      {WF_TEST_SIZE} bars (~1 month)\n")
    f.write(f"  Windows:        {len(wf_splits)}\n\n")

    f.write("OBJECTIVE: Profit Factor (capped at 3.0)\n")
    f.write(f"  Trials: 50\n\n")

    f.write("-" * 60 + "\n")
    f.write("DEFAULT PARAMETERS (baseline):\n")
    f.write(f"  n_estimators:      {DEFAULT_PARAMS['n_estimators']}\n")
    f.write(f"  learning_rate:     {DEFAULT_PARAMS['learning_rate']}\n")
    f.write(f"  max_depth:         {DEFAULT_PARAMS['max_depth']}\n")
    f.write(f"  min_child_samples: {DEFAULT_PARAMS['min_child_samples']}\n")
    f.write(f"  Walk-Forward mean PF: {default_mean_pf:.4f}\n")
    if default_pf_list:
        f.write(f"  Per-window PFs:       {[round(p, 4) for p in default_pf_list]}\n")
        f.write(f"  PF std:               {default_std:.4f}\n")
        f.write(f"  Windows PF>=1.0:      {sum(1 for p in default_pf_list if p >= 1.0)}/{len(default_pf_list)}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("BEST PARAMETERS (optimized):\n")
    f.write(f"  n_estimators:      {best['n_estimators']}\n")
    f.write(f"  learning_rate:     {best['learning_rate']:.6f}\n")
    f.write(f"  max_depth:         {best['max_depth']}\n")
    f.write(f"  min_child_samples: {best['min_child_samples']}\n")
    f.write(f"  Walk-Forward mean PF: {best_value:.4f}\n")
    if "pf_per_window" in best_trial.user_attrs:
        f.write(f"  Per-window PFs:       {best_trial.user_attrs['pf_per_window']}\n")
        f.write(f"  PF std:               {best_trial.user_attrs['pf_std']}\n")
        f.write(f"  Valid windows:        {best_trial.user_attrs['n_windows_valid']}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("COMPARISON:\n")
    f.write(f"  Default mean PF: {default_mean_pf:.4f}\n")
    f.write(f"  Best mean PF:    {best_value:.4f}\n")
    f.write(f"  Delta:           {delta:+.4f} ({pct:+.1f}%)\n")
    f.write(f"  Default PF std:  {default_std:.4f}\n")
    f.write(f"  Best PF std:     {best_std}\n\n")

    f.write("-" * 60 + "\n")
    f.write("TOP 5 TRIALS:\n")
    for i, t in enumerate(sorted_trials[:5]):
        f.write(f"  #{i+1} PF={t.value:.4f} | n_est={t.params['n_estimators']}, "
                f"lr={t.params['learning_rate']:.4f}, depth={t.params['max_depth']}, "
                f"min_child={t.params['min_child_samples']}\n")
    f.write("\n")

    f.write("-" * 60 + "\n")
    f.write("OVERFITTING RISK ASSESSMENT:\n")
    f.write(f"  Search params:       4 (reduced from 14)\n")
    f.write(f"  Validation method:   Walk-Forward Expanding Window (no lookahead)\n")
    f.write(f"  PF cap:              3.0 (prevents rewarding extreme values)\n")
    best_vs_default_ratio = best_value / default_mean_pf if default_mean_pf > 0 else float("inf")
    f.write(f"  Best/Default ratio:  {best_vs_default_ratio:.2f}x\n")
    if best_vs_default_ratio > 1.5:
        f.write("  WARNING: >1.5x improvement may indicate remaining overfitting risk\n")
    elif best_vs_default_ratio < 1.1:
        f.write("  NOTE: <1.1x improvement suggests default params are already near-optimal\n")
    else:
        f.write("  OK: Moderate improvement, reasonable optimization gain\n")

print(f"\nResults saved to: {results_path}")
print("Done.")
