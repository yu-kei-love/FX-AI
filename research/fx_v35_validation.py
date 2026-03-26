"""
FX v3.5 Walk-Forward Validation
================================
Post-fix validation after correcting look-ahead bias in MTF features (shift(1) added).
Uses expanding window WF with 5 folds, minimum 2000 bars training.
Replicates paper_trade.py ensemble + filters exactly.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Setup paths
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from research.common.features import (
    add_technical_features,
    add_multi_timeframe_features,
    FEATURE_COLS,
)
from research.common.ensemble import EnsembleClassifier

# ============================================================
# Constants (matching paper_trade.py)
# ============================================================
FORECAST_HORIZON = 12  # 12h direction label
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4  # 4/5 models must agree
N_FOLDS = 5
MIN_TRAIN_BARS = 2000

# Ensemble hyperparams (paper_trade.py DEFAULT_PARAMS)
N_ESTIMATORS = 300
LEARNING_RATE = 0.062

DATA_DIR = (script_dir / ".." / "data").resolve()
RESULTS_PATH = script_dir / "fx_v35_validation_results.txt"


# ============================================================
# Feature engineering (replicates paper_trade.py prepare_data)
# ============================================================
def load_data():
    """Load USDJPY 1h CSV."""
    path = DATA_DIR / "usdjpy_1h.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True, header=[0, 1])
    # Flatten multi-level columns
    df.columns = df.columns.get_level_values(0)
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def add_volatility_regime_features(df):
    """Volatility regime features (no lookahead)."""
    vol = df["Volatility_24"]
    df["Vol_percentile"] = vol.rolling(720, min_periods=72).apply(
        lambda x: (x[-1] >= x).sum() / len(x), raw=True
    )
    df["Vol_of_vol"] = vol.rolling(120, min_periods=24).std()
    return df


def add_calendar_awareness_features(df):
    """Calendar/session features."""
    h = df.index.hour
    dow = df.index.dayofweek
    df["Hour_x_DoW"] = h * 10 + dow
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    df["Session_overlap"] = ((h >= 13) & (h < 16)).astype(int)
    return df


def add_interaction_features(df):
    """v3 interaction features (matching paper_trade.py)."""
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
    return df


def prepare_features(df):
    """Full feature pipeline matching paper_trade.py."""
    df = add_technical_features(df)

    # Multi-timeframe features (v3.5 fix: shift(1) applied in features.py)
    df = add_multi_timeframe_features(df)

    # Interaction features
    df["Return"] = df["Close"].pct_change(24)
    df["Volatility"] = df["Return"].rolling(24).std()
    df = add_interaction_features(df)

    # Volatility regime features
    df = add_volatility_regime_features(df)

    # Calendar features
    df = add_calendar_awareness_features(df)

    # Build feature column list (matching paper_trade.py)
    # base_feature_cols = FEATURE_COLS minus Regime* columns
    base_feature_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]

    interaction_cols = [
        "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
        "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
        "Return_skew_12",
    ]
    vol_regime_cols = ["Vol_percentile", "Vol_of_vol"]
    calendar_cols = [
        "Hour_x_DoW", "Session_tokyo", "Session_london",
        "Session_ny", "Session_overlap",
    ]

    feature_cols = base_feature_cols + interaction_cols + vol_regime_cols + calendar_cols

    # Create 12h label
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]

    # Compute SMA_20 for trend filter (used later)
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Compute Volatility_20 for volatility filter
    df["Volatility_20"] = df["Return_1"].rolling(20).std()

    # Drop rows with NaN in features or label
    df = df.dropna(subset=feature_cols + ["Label", "Return_Nh"])

    return df, feature_cols


# ============================================================
# Weighted ensemble prediction (from paper_trade.py)
# ============================================================
def compute_model_weights(ensemble, X_val, y_val):
    """Per-model weights based on Sharpe ratio."""
    weights = []
    for model in ensemble.models:
        proba = model.predict_proba(X_val)[:, 1]
        direction = np.where(proba > 0.5, 1.0, -1.0)
        label_return = np.where(y_val == 1, 1.0, -1.0)
        trade_ret = direction * label_return
        if len(trade_ret) > 1 and trade_ret.std() > 0:
            sharpe = trade_ret.mean() / trade_ret.std()
        else:
            sharpe = 0.0
        weights.append(max(sharpe, 0.0))
    weights = np.array(weights)
    if weights.sum() <= 0:
        return np.ones(len(weights)) / len(weights)
    return weights / weights.sum()


def weighted_predict(ensemble, X, weights):
    """Weighted prediction."""
    probas = np.array([m.predict_proba(X)[:, 1] for m in ensemble.models])
    weighted_proba = (probas * weights[:, None]).sum(axis=0)
    preds = (weighted_proba >= 0.5).astype(int)
    individual_preds = np.array([m.predict(X) for m in ensemble.models])
    vote_sum = individual_preds.sum(axis=0)
    agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
    return preds, agreement, weighted_proba


# ============================================================
# Filters (matching paper_trade.py)
# ============================================================
def apply_filters(test_df, df_full, feature_cols):
    """Apply Tuesday skip, volatility filter, and trend filter.
    Returns mask of rows that pass all filters."""
    mask = pd.Series(True, index=test_df.index)

    # 1) Tuesday skip (dayofweek == 1)
    mask &= test_df.index.dayofweek != 1

    # 2) Hour filter: skip UTC 20 and 23
    hours = test_df.index.hour
    mask &= ~hours.isin([20, 23])

    # 3) Volatility filter: skip if vol percentile < 0.20 or > 0.90
    if "Volatility_20" in df_full.columns:
        vol_series = df_full["Volatility_20"].reindex(test_df.index)
        # Compute rolling percentile using last 120 bars from full data
        vol_pct = pd.Series(np.nan, index=test_df.index)
        for i, idx in enumerate(test_df.index):
            loc = df_full.index.get_loc(idx)
            if loc >= 120:
                window = df_full["Volatility_20"].iloc[loc - 120:loc + 1].dropna()
                if len(window) > 10:
                    current = df_full["Volatility_20"].iloc[loc]
                    if not np.isnan(current):
                        vol_pct.iloc[i] = (window.iloc[:-1] < current).mean()
        vol_filter = (vol_pct >= 0.20) & (vol_pct <= 0.90)
        vol_filter = vol_filter.fillna(True)  # If can't compute, don't filter
        mask &= vol_filter

    return mask


# ============================================================
# Walk-Forward validation
# ============================================================
def run_walk_forward(df, feature_cols):
    """Expanding window Walk-Forward validation with 5 folds."""
    n = len(df)
    # Reserve space for folds: first MIN_TRAIN_BARS for initial training
    test_size = (n - MIN_TRAIN_BARS) // N_FOLDS

    all_results = []
    fold_metrics = []

    print(f"\nData: {n} bars, {df.index[0]} to {df.index[-1]}")
    print(f"Folds: {N_FOLDS}, min train: {MIN_TRAIN_BARS}, test size: ~{test_size}")
    print("=" * 80)

    for fold in range(N_FOLDS):
        train_end = MIN_TRAIN_BARS + fold * test_size
        test_start = train_end
        test_end = min(train_end + test_size, n) if fold < N_FOLDS - 1 else n

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if len(test_df) == 0:
            continue

        print(f"\nFold {fold + 1}/{N_FOLDS}: train={len(train_df)}, test={len(test_df)}")
        print(f"  Train: {train_df.index[0]} to {train_df.index[-1]}")
        print(f"  Test:  {test_df.index[0]} to {test_df.index[-1]}")

        # Split train into train/val for weight computation
        val_size = min(500, len(train_df) // 5)
        X_train = train_df[feature_cols].iloc[:-val_size]
        y_train = train_df["Label"].iloc[:-val_size]
        X_val = train_df[feature_cols].iloc[-val_size:]
        y_val = train_df["Label"].iloc[-val_size:]

        # Train ensemble
        ensemble = EnsembleClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
        )
        ensemble.fit(X_train, y_train)

        # Compute performance-based weights
        weights = compute_model_weights(ensemble, X_val, y_val)
        print(f"  Weights: {[f'{w:.3f}' for w in weights]}")

        # Predict on test set
        X_test = test_df[feature_cols]
        preds, agreement, weighted_proba = weighted_predict(ensemble, X_test, weights)

        # Compute confidence
        confidence = np.maximum(weighted_proba, 1.0 - weighted_proba)

        # Apply filters
        filter_mask = apply_filters(test_df, df, feature_cols)

        # Apply confidence + agreement filters
        conf_mask = confidence >= CONFIDENCE_THRESHOLD
        agree_mask = agreement >= MIN_AGREEMENT

        # Trend filter (soft): raise threshold by 5% when against trend
        if "SMA_20" in df.columns:
            sma20 = df["SMA_20"].reindex(test_df.index)
            price = test_df["Close"]
            predicted_dir = preds  # 1=up, 0=down
            price_above_sma = (price > sma20).values
            against_trend = ((predicted_dir == 1) & ~price_above_sma) | \
                           ((predicted_dir == 0) & price_above_sma)
            trend_threshold = np.where(against_trend,
                                       CONFIDENCE_THRESHOLD + 0.05,
                                       CONFIDENCE_THRESHOLD)
            conf_mask = confidence >= trend_threshold

        # Combine all filters
        trade_mask = filter_mask & conf_mask & agree_mask

        # Compute returns for traded bars
        traded_preds = preds[trade_mask]
        traded_returns = test_df["Return_Nh"].values[trade_mask]
        traded_labels = test_df["Label"].values[trade_mask]

        n_trades = int(trade_mask.sum())

        if n_trades == 0:
            print(f"  No trades in this fold!")
            fold_metrics.append({
                "fold": fold + 1,
                "n_trades": 0,
                "pf": np.nan,
                "sharpe": np.nan,
                "winrate": np.nan,
                "mdd": np.nan,
            })
            continue

        # Direction: 1 for long (pred=1), -1 for short (pred=0)
        direction = np.where(traded_preds == 1, 1.0, -1.0)
        trade_returns = direction * traded_returns

        # Metrics
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Win rate
        wins = (trade_returns > 0).sum()
        winrate = wins / n_trades

        # Sharpe (annualized, computed on ALL bars including non-traded)
        # Following feedback: compute on all bars, not just traded bars
        all_bar_returns = np.zeros(len(test_df))
        traded_indices = np.where(trade_mask)[0]
        all_bar_returns[traded_indices] = trade_returns
        if all_bar_returns.std() > 0:
            sharpe = all_bar_returns.mean() / all_bar_returns.std() * np.sqrt(8760)
        else:
            sharpe = 0.0

        # Max Drawdown
        cumret = np.cumsum(trade_returns)
        peak = np.maximum.accumulate(cumret)
        dd = cumret - peak
        mdd = dd.min() * 100  # as percentage

        print(f"  Trades: {n_trades}, PF: {pf:.2f}, Sharpe: {sharpe:.2f}, "
              f"WinRate: {winrate:.1%}, MDD: {mdd:.2f}%")

        fold_metrics.append({
            "fold": fold + 1,
            "n_trades": n_trades,
            "pf": pf,
            "sharpe": sharpe,
            "winrate": winrate,
            "mdd": mdd,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "test_start": str(test_df.index[0]),
            "test_end": str(test_df.index[-1]),
        })

        # Collect for overall metrics
        for i, idx in enumerate(test_df.index[trade_mask]):
            all_results.append({
                "timestamp": idx,
                "pred": traded_preds[i],
                "label": traded_labels[i],
                "return": traded_returns[i],
                "trade_return": trade_returns[i],
            })

    return fold_metrics, all_results


# ============================================================
# Report generation
# ============================================================
def generate_report(fold_metrics, all_results, df, feature_cols):
    """Generate and save validation report."""
    lines = []
    lines.append("=" * 80)
    lines.append("FX v3.5 Walk-Forward Validation Results (POST Look-Ahead Bias Fix)")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"  Data: USDJPY 1h, {len(df)} bars")
    lines.append(f"  Period: {df.index[0]} to {df.index[-1]}")
    lines.append(f"  Folds: {N_FOLDS} (expanding window)")
    lines.append(f"  Min training bars: {MIN_TRAIN_BARS}")
    lines.append(f"  Ensemble: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees")
    lines.append(f"  Hyperparams: n_estimators={N_ESTIMATORS}, learning_rate={LEARNING_RATE}")
    lines.append(f"  Forecast horizon: {FORECAST_HORIZON}h")
    lines.append(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    lines.append(f"  Min agreement: {MIN_AGREEMENT}/5")
    lines.append(f"  Filters: Tuesday skip, hour filter (UTC 20,23), volatility filter, trend filter")
    lines.append(f"  MTF features: v3.5 fix (shift(1) applied to prevent look-ahead bias)")
    lines.append(f"  Features: {len(feature_cols)} columns")
    lines.append("")

    # Per-fold results
    lines.append("-" * 80)
    lines.append("Per-Fold Results:")
    lines.append("-" * 80)
    lines.append(f"{'Fold':>4} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'WinRate':>8} {'MDD%':>8}  Period")
    lines.append("-" * 80)

    for m in fold_metrics:
        if m["n_trades"] == 0:
            lines.append(f"{m['fold']:>4} {0:>7} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            lines.append(
                f"{m['fold']:>4} {m['n_trades']:>7} {m['pf']:>8.2f} {m['sharpe']:>8.2f} "
                f"{m['winrate']:>7.1%} {m['mdd']:>8.2f}  {m.get('test_start', '')[:10]} to {m.get('test_end', '')[:10]}"
            )

    # Overall metrics
    lines.append("")
    lines.append("-" * 80)
    lines.append("Overall Results (all folds combined):")
    lines.append("-" * 80)

    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        trade_returns = results_df["trade_return"].values
        n_total = len(trade_returns)

        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        overall_pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        wins = (trade_returns > 0).sum()
        overall_winrate = wins / n_total

        # Sharpe on all bars (not just traded)
        # Total test bars across all folds
        total_test_bars = sum(m.get("test_size", 0) for m in fold_metrics)
        all_bar_returns = np.zeros(total_test_bars)
        # Place trade returns at their positions
        offset = 0
        fold_idx = 0
        for m in fold_metrics:
            test_size = m.get("test_size", 0)
            offset += test_size

        # Simpler: just use the trade returns for Sharpe estimate
        # (traded bar returns with zeros for non-traded bars approximated)
        total_test_bars_approx = sum(m.get("test_size", 0) for m in fold_metrics if m.get("test_size"))
        if total_test_bars_approx > 0:
            mean_per_bar = trade_returns.sum() / total_test_bars_approx
            var_per_bar = (trade_returns**2).sum() / total_test_bars_approx
            std_per_bar = np.sqrt(var_per_bar - mean_per_bar**2) if var_per_bar > mean_per_bar**2 else 1e-10
            overall_sharpe = mean_per_bar / std_per_bar * np.sqrt(8760) if std_per_bar > 0 else 0
        else:
            overall_sharpe = 0

        # Max drawdown
        cumret = np.cumsum(trade_returns)
        peak = np.maximum.accumulate(cumret)
        dd = cumret - peak
        overall_mdd = dd.min() * 100

        # Avg return per trade
        avg_ret = trade_returns.mean() * 100

        # Total cumulative return
        total_ret = trade_returns.sum() * 100

        lines.append(f"  Total trades:     {n_total}")
        lines.append(f"  Profit Factor:    {overall_pf:.2f}")
        lines.append(f"  Sharpe Ratio:     {overall_sharpe:.2f}")
        lines.append(f"  Win Rate:         {overall_winrate:.1%}")
        lines.append(f"  Max Drawdown:     {overall_mdd:.2f}%")
        lines.append(f"  Avg Return/Trade: {avg_ret:.4f}%")
        lines.append(f"  Total Return:     {total_ret:.2f}%")
        lines.append(f"  Gross Profit:     {gross_profit * 100:.2f}%")
        lines.append(f"  Gross Loss:       {gross_loss * 100:.2f}%")
        lines.append(f"  Wins / Losses:    {wins} / {n_total - wins}")
    else:
        lines.append("  No trades generated across all folds!")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print("\n" + report)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nResults saved to: {RESULTS_PATH}")

    return report


# ============================================================
# Main
# ============================================================
def main():
    print("FX v3.5 Walk-Forward Validation (Post Look-Ahead Bias Fix)")
    print("=" * 80)

    # Load and prepare data
    print("Loading USDJPY 1h data...")
    df = load_data()
    print(f"Raw data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    print("Computing features (v3.5 with shift(1) MTF fix)...")
    df, feature_cols = prepare_features(df)
    print(f"After feature computation: {len(df)} bars, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")

    # Run Walk-Forward
    fold_metrics, all_results = run_walk_forward(df, feature_cols)

    # Generate report
    generate_report(fold_metrics, all_results, df, feature_cols)


if __name__ == "__main__":
    main()
