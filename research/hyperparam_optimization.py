"""
Hyperparameter optimization for FX ensemble model v3.1
Walk-forward validation across n_estimators x learning_rate grid.
"""

import sys
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, FEATURE_COLS
from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits, compute_metrics

# Import feature engineering from paper_trade
from research.paper_trade import (
    add_multi_timeframe_features,
    add_volatility_regime_features,
    add_calendar_awareness_features,
    CONFIDENCE_THRESHOLD,
    FORECAST_HORIZON,
    MIN_AGREEMENT,
    VOL_MULT,
)


def prepare_full_data():
    """Prepare data with all v3.1 features."""
    print("Loading data...")
    df = load_usdjpy_1h(use_5y=True)
    print(f"  Raw data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    df = add_technical_features(df)
    df = add_rate_features(df)
    df = add_daily_trend_features(df)
    df = add_multi_timeframe_features(df)

    # Interaction features (same as paper_trade.py)
    df["Return"] = df["Close"].pct_change(24)
    df["Volatility"] = df["Return"].rolling(24).std()
    df["RSI_x_Vol"] = df["RSI_14"] * df["Volatility_24"]
    df["MACD_norm"] = df["MACD"] / df["Volatility_24"].replace(0, np.nan)
    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / bb_range
    df["MA_cross"] = (df["MA_5"] - df["MA_75"]) / df["Close"]
    df["Momentum_accel"] = df["Return_1"] - df["Return_1"].shift(1)
    df["Vol_change"] = df["Volatility_24"].pct_change(6)
    df["HL_ratio"] = (df["High"] - df["Low"]) / df["Close"]
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Close_position"] = (df["Close"] - df["Low"]) / hl_range
    df["Return_skew_12"] = df["Return_1"].rolling(12).apply(
        lambda x: (x > 0).sum() / len(x) - 0.5, raw=True
    )

    df = add_volatility_regime_features(df)
    df = add_calendar_awareness_features(df)

    # Feature columns (same as paper_trade.py)
    interaction_cols = [
        "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
        "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
        "Return_skew_12",
    ]
    mtf_cols = [
        "RSI_4h", "MACD_4h", "BB_width_4h",
        "RSI_daily", "MACD_daily", "BB_width_daily",
    ]
    vol_regime_cols = ["Vol_percentile", "Vol_of_vol"]
    calendar_cols = ["Hour_x_DoW", "Session_tokyo", "Session_london", "Session_ny", "Session_overlap"]
    base_feature_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]
    feature_cols = base_feature_cols + interaction_cols + mtf_cols + vol_regime_cols + calendar_cols

    df = df.dropna(subset=feature_cols)

    # Label: 12h direction
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df = df.dropna(subset=["Label", "Return_Nh"])

    print(f"  After feature engineering: {len(df)} rows")
    return df, feature_cols


def run_walk_forward(df, feature_cols, n_estimators, learning_rate):
    """Run walk-forward validation for a single hyperparameter combination."""
    min_train_size = 4320   # 6 months
    test_size = 720         # 1 month
    step_size = 720         # 1 month step

    splits = walk_forward_splits(
        n_total=len(df),
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
    )

    if len(splits) == 0:
        print(f"  WARNING: No walk-forward splits possible")
        return None

    all_returns = []

    for i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df["Label"].values
        X_test = test_df[feature_cols].values
        ret_test = test_df["Return_Nh"].values

        # Train ensemble
        ensemble = EnsembleClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        ensemble.fit(X_train, y_train)

        # Predict
        preds, agreement = ensemble.predict_with_agreement(X_test)
        proba = ensemble.predict_proba(X_test)[:, 1]
        confidence = np.maximum(proba, 1.0 - proba)

        # Volatility filter
        hist_vol = train_df["Volatility_24"].mean()
        vol_test = test_df["Volatility_24"].values
        vol_mask = vol_test <= VOL_MULT * hist_vol

        # Apply filters
        trade_mask = (confidence >= CONFIDENCE_THRESHOLD) & (agreement >= MIN_AGREEMENT) & vol_mask

        if trade_mask.sum() > 0:
            direction = np.where(preds[trade_mask] == 1, 1.0, -1.0)
            returns = ret_test[trade_mask] * direction
            all_returns.extend(returns.tolist())

    if len(all_returns) == 0:
        return None

    metrics = compute_metrics(np.array(all_returns))
    metrics["total_splits"] = len(splits)
    return metrics


def main():
    # Hyperparameter grid
    n_estimators_list = [150, 200, 300, 500]
    learning_rate_list = [0.03, 0.05, 0.08]

    # Prepare data once
    df, feature_cols = prepare_full_data()

    results = []
    total = len(n_estimators_list) * len(learning_rate_list)
    count = 0

    print(f"\n{'='*70}")
    print(f"Walk-Forward Hyperparameter Optimization")
    print(f"Grid: {len(n_estimators_list)} x {len(learning_rate_list)} = {total} combinations")
    print(f"Filters: confidence >= {CONFIDENCE_THRESHOLD}, agreement >= {MIN_AGREEMENT}/5")
    print(f"Forecast horizon: {FORECAST_HORIZON}h")
    print(f"{'='*70}\n")

    for n_est, lr in itertools.product(n_estimators_list, learning_rate_list):
        count += 1
        print(f"[{count}/{total}] n_estimators={n_est}, learning_rate={lr}")

        metrics = run_walk_forward(df, feature_cols, n_est, lr)

        if metrics is None:
            print(f"  -> No trades generated\n")
            continue

        results.append({
            "n_estimators": n_est,
            "learning_rate": lr,
            "PF": metrics["pf"],
            "Sharpe": metrics["sharpe"],
            "Win%": metrics["win_rate"],
            "MDD%": metrics["mdd"],
            "Trades": metrics["n_trades"],
            "ExpVal": metrics["exp_value_net"],
            "Sortino": metrics["sortino"],
            "Payoff": metrics["payoff"],
        })

        print(f"  -> PF={metrics['pf']:.2f}, Sharpe={metrics['sharpe']:.2f}, "
              f"Win={metrics['win_rate']:.1f}%, Trades={metrics['n_trades']}, "
              f"MDD={metrics['mdd']:.2f}%\n")

    # Summary
    if not results:
        print("No valid results!")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("PF", ascending=False)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY (sorted by PF)")
    print(f"{'='*70}")
    print(results_df.to_string(index=False, float_format="%.4f"))

    # Find best by PF
    best_pf = results_df.iloc[0]
    print(f"\n--- Best by PF ---")
    print(f"  n_estimators={int(best_pf['n_estimators'])}, learning_rate={best_pf['learning_rate']}")
    print(f"  PF={best_pf['PF']:.4f}, Sharpe={best_pf['Sharpe']:.4f}")

    # Find best by Sharpe
    best_sharpe = results_df.sort_values("Sharpe", ascending=False).iloc[0]
    print(f"\n--- Best by Sharpe ---")
    print(f"  n_estimators={int(best_sharpe['n_estimators'])}, learning_rate={best_sharpe['learning_rate']}")
    print(f"  PF={best_sharpe['PF']:.4f}, Sharpe={best_sharpe['Sharpe']:.4f}")

    # Combined score: normalize PF and Sharpe, weight equally
    pf_vals = results_df["PF"].replace([np.inf, -np.inf], np.nan)
    sharpe_vals = results_df["Sharpe"].replace([np.inf, -np.inf], np.nan)
    pf_min, pf_max = pf_vals.min(), pf_vals.max()
    sh_min, sh_max = sharpe_vals.min(), sharpe_vals.max()

    if pf_max > pf_min and sh_max > sh_min:
        results_df["PF_norm"] = (pf_vals - pf_min) / (pf_max - pf_min)
        results_df["Sharpe_norm"] = (sharpe_vals - sh_min) / (sh_max - sh_min)
        results_df["Combined"] = 0.5 * results_df["PF_norm"] + 0.5 * results_df["Sharpe_norm"]
        best_combined = results_df.sort_values("Combined", ascending=False).iloc[0]
        print(f"\n--- Best Combined (PF+Sharpe) ---")
        print(f"  n_estimators={int(best_combined['n_estimators'])}, learning_rate={best_combined['learning_rate']}")
        print(f"  PF={best_combined['PF']:.4f}, Sharpe={best_combined['Sharpe']:.4f}, Combined={best_combined['Combined']:.4f}")

    # Baseline comparison
    baseline_row = results_df[
        (results_df["n_estimators"] == 200) & (results_df["learning_rate"] == 0.05)
    ]
    if len(baseline_row) > 0:
        baseline_pf = baseline_row.iloc[0]["PF"]
        baseline_sharpe = baseline_row.iloc[0]["Sharpe"]
        print(f"\n--- Baseline (n_est=200, lr=0.05) ---")
        print(f"  PF={baseline_pf:.4f}, Sharpe={baseline_sharpe:.4f}")

        best = best_combined if "Combined" in results_df.columns else best_pf
        pf_improvement = (best["PF"] - baseline_pf) / baseline_pf * 100
        sharpe_improvement = (best["Sharpe"] - baseline_sharpe) / baseline_sharpe * 100 if baseline_sharpe != 0 else float("nan")
        print(f"\n--- Improvement (best combined vs baseline) ---")
        print(f"  PF improvement: {pf_improvement:+.2f}%")
        print(f"  Sharpe improvement: {sharpe_improvement:+.2f}%")
        print(f"  PF improvement > 5%: {'YES' if pf_improvement > 5 else 'NO'}")

    # Save results
    out_path = Path(__file__).resolve().parent.parent / "data" / "hyperparam_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
