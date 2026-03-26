# ===========================================
# fx_confidence_optimizer.py
# Confidence threshold optimizer for FX trading signals
#
# Walk-Forward validation with expanding window to find the
# optimal confidence cutoff that maximizes PF while keeping
# sufficient trade count (>100).
# ===========================================

import sys
import warnings
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

# ---- Parameters (v3.2) ----
N_ESTIMATORS = 500
LEARNING_RATE = 0.03
FORECAST_HORIZON = 12
MIN_AGREEMENT = 4
MIN_TRAIN_SIZE = 4320  # 6 months of 1h bars
TEST_SIZE = 720        # 1 month of 1h bars
MAX_FOLDS = 8

THRESHOLDS = np.arange(0.50, 0.82, 0.02)

script_dir = Path(__file__).resolve().parent
RESULTS_PATH = script_dir / "fx_confidence_results.txt"


# ---- Feature engineering (same as paper_trade.py) ----

def _resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


def _compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line


def _compute_bb_width(close, period=20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std) / sma.replace(0, np.nan)


def add_multi_timeframe_features(df):
    for rule, suffix in [("4h", "4h"), ("1D", "daily")]:
        resampled = _resample_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]], rule)
        close = resampled["Close"]
        feat = pd.DataFrame(index=resampled.index)
        feat[f"RSI_{suffix}"] = _compute_rsi(close, 14)
        feat[f"MACD_{suffix}"] = _compute_macd(close)
        feat[f"BB_width_{suffix}"] = _compute_bb_width(close)
        feat = feat.reindex(df.index, method="ffill")
        for col in feat.columns:
            df[col] = feat[col]
    return df


def add_volatility_regime_features(df):
    vol = df["Volatility_24"]
    df["Vol_percentile"] = vol.rolling(720, min_periods=72).apply(
        lambda x: (x[-1] >= x).sum() / len(x), raw=True
    )
    df["Vol_of_vol"] = vol.rolling(120, min_periods=24).std()
    return df


def add_calendar_awareness_features(df):
    h = df.index.hour
    dow = df.index.dayofweek
    df["Hour_x_DoW"] = h * 10 + dow
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    df["Session_overlap"] = ((h >= 13) & (h < 16)).astype(int)
    return df


def prepare_data():
    """Load USDJPY 1h and build v3.1 feature set."""
    print("Loading USDJPY 1h data...")
    df = load_usdjpy_1h()
    df = add_technical_features(df)
    df = add_rate_features(df)
    df = add_daily_trend_features(df)
    df = add_multi_timeframe_features(df)

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
    print(f"Data loaded: {len(df)} rows, {len(feature_cols)} features")
    return df, feature_cols


def compute_model_weights(ensemble, X_val, y_val):
    """Per-model accuracy-cubed weighting."""
    weights = []
    for model in ensemble.models:
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        weights.append(acc)
    weights = np.array(weights)
    weights = weights ** 3
    weights = weights / weights.sum()
    return weights


def weighted_predict(ensemble, X, weights):
    """Weighted ensemble prediction returning preds, agreement, probabilities."""
    probas = np.array([m.predict_proba(X)[:, 1] for m in ensemble.models])
    weighted_proba = (probas * weights[:, None]).sum(axis=0)
    preds = (weighted_proba >= 0.5).astype(int)
    individual_preds = np.array([m.predict(X) for m in ensemble.models])
    vote_sum = individual_preds.sum(axis=0)
    agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
    return preds, agreement, weighted_proba


def run_optimization():
    """Main: Walk-Forward with confidence threshold sweep."""
    df, feature_cols = prepare_data()

    # Create 12h direction label
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]
    df = df.dropna(subset=["Return_Nh"])

    n = len(df)
    print(f"\nTotal usable rows: {n}")

    # Walk-Forward splits (expanding window)
    splits = walk_forward_splits(n, MIN_TRAIN_SIZE, TEST_SIZE)
    if len(splits) > MAX_FOLDS:
        # Take evenly spaced folds for speed
        indices = np.linspace(0, len(splits) - 1, MAX_FOLDS, dtype=int)
        splits = [splits[i] for i in indices]
    print(f"Walk-Forward folds: {len(splits)}")

    # Collect per-trade data across all folds
    all_confidences = []
    all_returns = []
    all_predictions = []
    all_agreements = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_i + 1}/{len(splits)} ---")
        print(f"  Train: {len(train_idx)} rows, Test: {len(test_idx)} rows")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols].values
        y_train = (train_df["Close_Nh_later"] > train_df["Close"]).astype(int).values

        # Split train into train/val for weighting
        val_size = min(500, len(train_df) // 5)
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        # Train 5-model ensemble
        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        ensemble.fit(X_tr, y_tr)

        # Compute performance-based weights
        weights = compute_model_weights(ensemble, X_val, y_val)

        # Predict on test set
        X_test = test_df[feature_cols].values
        preds, agreement, weighted_proba = weighted_predict(ensemble, X_test, weights)
        confidence = np.maximum(weighted_proba, 1.0 - weighted_proba)
        ret_test = test_df["Return_Nh"].values

        all_confidences.append(confidence)
        all_returns.append(ret_test)
        all_predictions.append(preds)
        all_agreements.append(agreement)

        print(f"  Avg confidence: {confidence.mean():.4f}")
        print(f"  Trades at 0.50: {len(confidence)}, at 0.60: {(confidence >= 0.60).sum()}")

    # Concatenate all folds
    all_conf = np.concatenate(all_confidences)
    all_ret = np.concatenate(all_returns)
    all_pred = np.concatenate(all_predictions)
    all_agree = np.concatenate(all_agreements)

    print(f"\n{'=' * 70}")
    print(f"CONFIDENCE THRESHOLD OPTIMIZATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total OOS predictions: {len(all_conf)}")
    print(f"Confidence range: [{all_conf.min():.4f}, {all_conf.max():.4f}]")
    print(f"Mean confidence: {all_conf.mean():.4f}")

    # Sweep thresholds
    results = []
    header = f"{'Threshold':>10} {'N_Trades':>10} {'WinRate':>10} {'PF':>10} {'Sharpe':>10} {'MDD':>10} {'ExpVal':>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for thresh in THRESHOLDS:
        # Filter by confidence >= threshold AND agreement >= MIN_AGREEMENT
        mask = (all_conf >= thresh) & (all_agree >= MIN_AGREEMENT)
        if mask.sum() == 0:
            results.append({
                "threshold": thresh, "n_trades": 0,
                "win_rate": np.nan, "pf": np.nan, "sharpe": np.nan,
                "mdd": np.nan, "exp_value_net": np.nan
            })
            print(f"{thresh:>10.2f} {'0':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12}")
            continue

        direction = np.where(all_pred[mask] == 1, 1.0, -1.0)
        trade_returns = all_ret[mask] * direction

        metrics = compute_metrics(trade_returns)
        results.append({
            "threshold": thresh,
            "n_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "pf": metrics["pf"],
            "sharpe": metrics["sharpe"],
            "mdd": metrics["mdd"],
            "exp_value_net": metrics["exp_value_net"],
        })

        pf_str = f"{metrics['pf']:.2f}" if not np.isnan(metrics['pf']) else "N/A"
        sharpe_str = f"{metrics['sharpe']:.2f}" if not np.isnan(metrics['sharpe']) else "N/A"
        mdd_str = f"{metrics['mdd']:.2f}%" if not np.isnan(metrics['mdd']) else "N/A"
        ev_str = f"{metrics['exp_value_net']:+.6f}" if not np.isnan(metrics['exp_value_net']) else "N/A"
        print(f"{thresh:>10.2f} {metrics['n_trades']:>10} {metrics['win_rate']:>9.1f}% {pf_str:>10} {sharpe_str:>10} {mdd_str:>10} {ev_str:>12}")

    # Find optimal threshold (best PF with >100 trades)
    df_results = pd.DataFrame(results)
    viable = df_results[(df_results["n_trades"] > 100) & (df_results["pf"].notna())]

    print(f"\n{'=' * 70}")
    if len(viable) == 0:
        print("No viable threshold found with >100 trades.")
        optimal = None
    else:
        best_idx = viable["pf"].idxmax()
        optimal = viable.loc[best_idx]
        print(f"OPTIMAL THRESHOLD: {optimal['threshold']:.2f}")
        print(f"  Profit Factor:  {optimal['pf']:.2f}")
        print(f"  Win Rate:       {optimal['win_rate']:.1f}%")
        print(f"  N Trades:       {int(optimal['n_trades'])}")
        print(f"  Sharpe Ratio:   {optimal['sharpe']:.2f}")
        print(f"  MDD:            {optimal['mdd']:.2f}%")
        print(f"  Exp Value (net):{optimal['exp_value_net']:+.6f}")

    # Baseline comparison (threshold=0.50, i.e., all trades)
    baseline = df_results[df_results["threshold"] == 0.50].iloc[0]
    print(f"\nBASELINE (threshold=0.50):")
    print(f"  PF={baseline['pf']:.2f}, WinRate={baseline['win_rate']:.1f}%, N={int(baseline['n_trades'])}, Sharpe={baseline['sharpe']:.2f}")

    if optimal is not None:
        pf_improvement = ((optimal['pf'] / baseline['pf']) - 1) * 100 if baseline['pf'] > 0 else float('nan')
        wr_improvement = optimal['win_rate'] - baseline['win_rate']
        trade_reduction = (1 - optimal['n_trades'] / baseline['n_trades']) * 100
        print(f"\nIMPROVEMENT vs baseline:")
        print(f"  PF:       {pf_improvement:+.1f}%")
        print(f"  WinRate:  {wr_improvement:+.1f}pp")
        print(f"  Trades reduced: {trade_reduction:.1f}%")
    print(f"{'=' * 70}")

    # Save results to file
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("FX Confidence Threshold Optimization Results\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Model: v3.2 5-model ensemble (LGB/XGB/CatBoost/RF/ExtraTrees)\n")
        f.write(f"Params: n_estimators={N_ESTIMATORS}, learning_rate={LEARNING_RATE}\n")
        f.write(f"Data: USDJPY 1h\n")
        f.write(f"Forecast horizon: {FORECAST_HORIZON}h\n")
        f.write(f"Min agreement: {MIN_AGREEMENT}/5\n")
        f.write(f"Validation: Walk-Forward (expanding window, {len(splits)} folds)\n")
        f.write(f"Total OOS predictions: {len(all_conf)}\n\n")

        f.write(f"{'Threshold':>10} {'N_Trades':>10} {'WinRate':>10} {'PF':>10} {'Sharpe':>10} {'MDD':>10} {'ExpVal':>12}\n")
        f.write("-" * 72 + "\n")
        for r in results:
            pf_s = f"{r['pf']:.2f}" if not np.isnan(r.get('pf', np.nan)) else "N/A"
            sh_s = f"{r['sharpe']:.2f}" if not np.isnan(r.get('sharpe', np.nan)) else "N/A"
            mdd_s = f"{r['mdd']:.2f}%" if not np.isnan(r.get('mdd', np.nan)) else "N/A"
            ev_s = f"{r['exp_value_net']:+.6f}" if not np.isnan(r.get('exp_value_net', np.nan)) else "N/A"
            wr_s = f"{r['win_rate']:.1f}%" if not np.isnan(r.get('win_rate', np.nan)) else "N/A"
            f.write(f"{r['threshold']:>10.2f} {r['n_trades']:>10} {wr_s:>10} {pf_s:>10} {sh_s:>10} {mdd_s:>10} {ev_s:>12}\n")

        f.write(f"\n{'=' * 70}\n")
        if optimal is not None:
            f.write(f"OPTIMAL THRESHOLD: {optimal['threshold']:.2f}\n")
            f.write(f"  Profit Factor:  {optimal['pf']:.2f}\n")
            f.write(f"  Win Rate:       {optimal['win_rate']:.1f}%\n")
            f.write(f"  N Trades:       {int(optimal['n_trades'])}\n")
            f.write(f"  Sharpe Ratio:   {optimal['sharpe']:.2f}\n")
            f.write(f"  MDD:            {optimal['mdd']:.2f}%\n")
            f.write(f"  Exp Value (net):{optimal['exp_value_net']:+.6f}\n")
        else:
            f.write("No viable threshold found with >100 trades.\n")

        f.write(f"\nBASELINE (threshold=0.50, all trades with {MIN_AGREEMENT}/5 agreement):\n")
        f.write(f"  PF={baseline['pf']:.2f}, WinRate={baseline['win_rate']:.1f}%, N={int(baseline['n_trades'])}, Sharpe={baseline['sharpe']:.2f}\n")

        if optimal is not None:
            f.write(f"\nIMPROVEMENT vs baseline:\n")
            f.write(f"  PF:       {pf_improvement:+.1f}%\n")
            f.write(f"  WinRate:  {wr_improvement:+.1f}pp\n")
            f.write(f"  Trades reduced: {trade_reduction:.1f}%\n")
        f.write(f"{'=' * 70}\n")

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_optimization()
