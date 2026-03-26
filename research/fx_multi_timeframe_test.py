"""
fx_multi_timeframe_test.py
Multi-Timeframe Feature Walk-Forward Backtest

Compares two feature sets:
  - Baseline: 1h-only features (FEATURE_COLS from features.py)
  - Enhanced: 1h + 4h + daily timeframe features

Multi-timeframe features are derived by resampling 1h OHLCV data
(no additional data files required).

New features:
  4h:  RSI_4h, MACD_hist_4h, BB_width_4h
  Daily: RSI_daily, MACD_hist_daily, BB_width_daily, Daily_trend_dir
"""

import sys
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Setup paths
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from research.common.data_loader import load_usdjpy_1h
from research.common.features import (
    add_technical_features,
    add_regime_features_wf,
    FEATURE_COLS,
)
from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits, compute_metrics

RESULTS_PATH = script_dir / "fx_multi_timeframe_results.txt"

# Model parameters (same as paper_trade v3.2)
N_ESTIMATORS = 500
LEARNING_RATE = 0.03
FORECAST_HORIZON = 12
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4

# Walk-Forward settings
MIN_TRAIN_HOURS = 4320  # 6 months
TEST_HOURS = 720         # 1 month
STEP_HOURS = 720         # non-overlapping


# ============================================================
# Multi-timeframe feature computation (resampling from 1h)
# ============================================================

def _resample_ohlcv(df, rule):
    """Resample OHLCV to lower frequency."""
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


def _compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line


def _compute_bb_width(close, period=20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std) / sma.replace(0, np.nan)


def add_multi_timeframe_features(df):
    """Add 4h and daily timeframe indicators resampled from 1h data.

    Features added:
      RSI_4h, MACD_hist_4h, BB_width_4h
      RSI_daily, MACD_hist_daily, BB_width_daily, Daily_trend_dir
    """
    for rule, suffix in [("4h", "4h"), ("1D", "daily")]:
        resampled = _resample_ohlcv(
            df[["Open", "High", "Low", "Close", "Volume"]], rule
        )
        close = resampled["Close"]
        feat = pd.DataFrame(index=resampled.index)
        feat[f"RSI_{suffix}"] = _compute_rsi(close, 14)
        feat[f"MACD_hist_{suffix}"] = _compute_macd_hist(close)
        feat[f"BB_width_{suffix}"] = _compute_bb_width(close)

        if suffix == "daily":
            sma20 = close.rolling(20).mean()
            feat["Daily_trend_dir"] = (close > sma20).astype(int)

        feat = feat.reindex(df.index, method="ffill")
        for col in feat.columns:
            df[col] = feat[col]
    return df


# New MTF columns
MTF_COLS = [
    "RSI_4h", "MACD_hist_4h", "BB_width_4h",
    "RSI_daily", "MACD_hist_daily", "BB_width_daily",
    "Daily_trend_dir",
]


# ============================================================
# Walk-Forward backtest engine
# ============================================================

def run_walk_forward(df_raw, feature_cols, label=""):
    """Run Walk-Forward backtest with given feature set.

    Returns dict of aggregated metrics or None.
    """
    print(f"\n{'='*60}")
    print(f"  Walk-Forward: {label}")
    print(f"  Features: {len(feature_cols)} cols")
    print(f"{'='*60}")

    df = df_raw.copy()

    # Technical features
    df = add_technical_features(df)

    # Return/Volatility for regime
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change(24)
    if "Volatility" not in df.columns:
        df["Volatility"] = df["Return"].rolling(24).std()

    # Multi-timeframe features (always compute; baseline just won't use them)
    df = add_multi_timeframe_features(df)

    # Label: 12h forward return direction
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]
    df = df.dropna(subset=["Return_Nh"])

    if len(df) < MIN_TRAIN_HOURS + TEST_HOURS + 200:
        print(f"  SKIP: Insufficient data ({len(df)} bars)")
        return None

    splits = walk_forward_splits(len(df), MIN_TRAIN_HOURS, TEST_HOURS, STEP_HOURS)
    print(f"  Walk-Forward splits: {len(splits)}")

    if len(splits) == 0:
        print(f"  SKIP: No valid WF splits")
        return None

    all_returns = []
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        df_fold = df.copy()
        df_fold = add_regime_features_wf(df_fold, train_end_idx=train_idx[-1] + 1)

        # Check required feature columns exist and drop NaN
        missing = [c for c in feature_cols if c not in df_fold.columns]
        if missing:
            print(f"    Fold {fold_i+1}: SKIP missing cols {missing}")
            continue

        valid_mask = df_fold[feature_cols].notna().all(axis=1)
        df_fold_clean = df_fold[valid_mask]

        train_mask = df_fold_clean.index.isin(df.index[train_idx])
        test_mask = df_fold_clean.index.isin(df.index[test_idx])

        train_df = df_fold_clean[train_mask]
        test_df = df_fold_clean[test_mask]

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        X_train = train_df[feature_cols].values
        y_train = (train_df["Close_Nh_later"] > train_df["Close"]).astype(int).values
        X_test = test_df[feature_cols].values
        y_test_dir = (test_df["Close_Nh_later"] > test_df["Close"]).astype(int).values
        ret_test = test_df["Return_Nh"].values

        # Train ensemble
        ensemble = EnsembleClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE
        )
        ensemble.fit(X_train, y_train)

        # Predict
        preds, agreement = ensemble.predict_with_agreement(X_test)
        proba = ensemble.predict_proba(X_test)[:, 1]
        confidence = np.maximum(proba, 1.0 - proba)

        # Volatility filter
        hist_vol = train_df["Volatility_24"].mean()
        vol_test = test_df["Volatility_24"].values
        vol_mask = vol_test <= 2.0 * hist_vol

        # Apply filters
        trade_mask = (
            (confidence >= CONFIDENCE_THRESHOLD)
            & (agreement >= MIN_AGREEMENT)
            & vol_mask
        )

        if trade_mask.sum() == 0:
            continue

        direction = np.where(preds[trade_mask] == 1, 1.0, -1.0)
        fold_returns = ret_test[trade_mask] * direction

        all_returns.extend(fold_returns.tolist())

        fold_metrics = compute_metrics(fold_returns)
        fold_results.append({
            "fold": fold_i + 1,
            "n_trades": fold_metrics["n_trades"],
            "win_rate": fold_metrics["win_rate"],
            "pf": fold_metrics["pf"],
        })

        pf_str = f"{fold_metrics['pf']:.2f}" if not math.isinf(fold_metrics['pf']) else "inf"
        print(
            f"    Fold {fold_i+1}: trades={fold_metrics['n_trades']}, "
            f"WR={fold_metrics['win_rate']:.1f}%, PF={pf_str}"
        )

    if len(all_returns) == 0:
        print(f"  RESULT: No trades executed")
        return None

    all_returns = np.array(all_returns)
    metrics = compute_metrics(all_returns)

    pf_str = f"{metrics['pf']:.2f}" if not math.isinf(metrics['pf']) else "inf"
    sharpe_str = f"{metrics['sharpe']:.2f}" if not math.isnan(metrics['sharpe']) else "N/A"

    print(f"\n  --- {label} Overall Results ---")
    print(f"  Trades:    {metrics['n_trades']}")
    print(f"  Win Rate:  {metrics['win_rate']:.1f}%")
    print(f"  PF:        {pf_str}")
    print(f"  Sharpe:    {sharpe_str}")
    print(f"  MDD:       {metrics['mdd']:.2f}%")
    print(f"  Exp Value: {metrics['exp_value_net']:+.6f}")
    print(f"  Payoff:    {metrics['payoff']:.2f}")

    return {
        "label": label,
        "n_folds": len(splits),
        "n_trade_folds": len(fold_results),
        **metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  FX Multi-Timeframe Feature Walk-Forward Comparison")
    print(f"  Model: Ensemble (5 models, n_est={N_ESTIMATORS}, lr={LEARNING_RATE})")
    print(f"  Horizon: {FORECAST_HORIZON}h | Conf: {CONFIDENCE_THRESHOLD} | Agreement: {MIN_AGREEMENT}/5")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load 1h data
    print("\nLoading USD/JPY 1h data...")
    df_raw = load_usdjpy_1h(use_5y=True)
    print(f"  Data: {df_raw.index[0]} ~ {df_raw.index[-1]} ({len(df_raw)} bars)")

    # If 5y data not available, fall back
    if len(df_raw) < MIN_TRAIN_HOURS + TEST_HOURS + 500:
        print("  5y data insufficient, trying 2y data...")
        df_raw = load_usdjpy_1h(use_5y=False)
        print(f"  Data: {df_raw.index[0]} ~ {df_raw.index[-1]} ({len(df_raw)} bars)")

    # --- Baseline: 1h features only ---
    baseline_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]
    # Add Regime_duration which is in FEATURE_COLS and computed by add_regime_features_wf
    baseline_cols.append("Regime_duration")

    baseline_result = run_walk_forward(df_raw, baseline_cols, label="Baseline (1h only)")

    # --- Enhanced: 1h + 4h + daily ---
    enhanced_cols = baseline_cols + MTF_COLS

    enhanced_result = run_walk_forward(df_raw, enhanced_cols, label="Enhanced (1h + 4h + daily)")

    # --- Summary ---
    print("\n\n")
    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)

    results_text = []
    results_text.append("=" * 70)
    results_text.append("  FX Multi-Timeframe Feature Walk-Forward Comparison")
    results_text.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    results_text.append(f"  Model: Ensemble 5 models, n_est={N_ESTIMATORS}, lr={LEARNING_RATE}")
    results_text.append(f"  Horizon: {FORECAST_HORIZON}h | Conf: {CONFIDENCE_THRESHOLD} | Agreement: {MIN_AGREEMENT}/5")
    results_text.append(f"  Data: {df_raw.index[0]} ~ {df_raw.index[-1]} ({len(df_raw)} bars)")
    results_text.append("=" * 70)
    results_text.append("")

    header = f"{'Config':<28} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'Sharpe':>8} {'MDD%':>7} {'ExpVal':>10} {'Payoff':>7}"
    print(header)
    results_text.append(header)
    print("-" * 90)
    results_text.append("-" * 90)

    for r in [baseline_result, enhanced_result]:
        if r is None:
            line = f"{'N/A':<28} {'---':>7} {'---':>8} {'---':>7} {'---':>8} {'---':>7} {'---':>10} {'---':>7}"
        else:
            pf_str = f"{r['pf']:.2f}" if not math.isinf(r['pf']) else "inf"
            sharpe_str = f"{r['sharpe']:.2f}" if not math.isnan(r['sharpe']) else "N/A"
            payoff_str = f"{r['payoff']:.2f}" if not math.isnan(r['payoff']) else "N/A"
            line = (
                f"{r['label']:<28} {r['n_trades']:>7} {r['win_rate']:>7.1f}% "
                f"{pf_str:>7} {sharpe_str:>8} {r['mdd']:>6.2f}% "
                f"{r['exp_value_net']:>+10.6f} {payoff_str:>7}"
            )
        print(line)
        results_text.append(line)

    print("-" * 90)
    results_text.append("-" * 90)

    # Delta analysis
    results_text.append("")
    if baseline_result and enhanced_result:
        delta_pf = enhanced_result["pf"] - baseline_result["pf"]
        delta_wr = enhanced_result["win_rate"] - baseline_result["win_rate"]
        delta_sharpe = enhanced_result["sharpe"] - baseline_result["sharpe"]
        delta_ev = enhanced_result["exp_value_net"] - baseline_result["exp_value_net"]

        delta_lines = [
            "  Delta (Enhanced - Baseline):",
            f"    PF:        {delta_pf:+.2f}" if not (math.isinf(delta_pf) or math.isnan(delta_pf)) else f"    PF:        N/A",
            f"    Win Rate:  {delta_wr:+.1f}%",
            f"    Sharpe:    {delta_sharpe:+.2f}" if not math.isnan(delta_sharpe) else f"    Sharpe:    N/A",
            f"    Exp Value: {delta_ev:+.6f}" if not math.isnan(delta_ev) else f"    Exp Value: N/A",
        ]

        # Verdict
        positive = False
        if not (math.isnan(delta_pf) or math.isinf(delta_pf)):
            if delta_pf > 0 and delta_wr > 0:
                verdict = "POSITIVE - Multi-timeframe features improve both PF and Win Rate"
                positive = True
            elif delta_pf > 0:
                verdict = "POSITIVE - Multi-timeframe features improve PF"
                positive = True
            elif delta_sharpe > 0 and delta_ev > 0:
                verdict = "MIXED POSITIVE - Sharpe and ExpVal improved, but PF decreased"
                positive = True
            else:
                verdict = "NEGATIVE - Multi-timeframe features did not improve performance"
        else:
            verdict = "INCONCLUSIVE"

        delta_lines.append("")
        delta_lines.append(f"  VERDICT: {verdict}")
        delta_lines.append("")
        delta_lines.append("  New MTF feature columns:")
        for col in MTF_COLS:
            delta_lines.append(f"    - {col}")

        for line in delta_lines:
            print(line)
            results_text.append(line)
    else:
        msg = "  Could not compare: one or both configurations returned no results."
        print(msg)
        results_text.append(msg)

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
