"""
fx_improvement_vol_regime.py
Volatility Regime Filter Analysis for FX Model (USDJPY + AUDJPY)

Tests whether skipping low-volatility periods improves performance.
Uses Vol_percentile (already computed) to define regimes:
  - Low vol: percentile < 20th
  - Normal vol: 20th-80th
  - High vol: > 80th

Walk-Forward validation to avoid overfitting.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, FEATURE_COLS
from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits, compute_metrics

# ============================================================
# Parameters (v3.3)
# ============================================================
N_ESTIMATORS = 500
LEARNING_RATE = 0.03
FORECAST_HORIZON = 12
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4
VOL_MULT = 2.0
MAX_FOLDS = 8
MIN_TRAIN_SIZE = 4320
TEST_SIZE = 720
SKIP_HOURS = {20, 23}

OUTPUT_PATH = Path(__file__).resolve().parent / "fx_improvement_vol_results.txt"


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
    return ema_fast - ema_slow


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


def prepare_full_features(df):
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

    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]

    df = df.dropna(subset=feature_cols + ["Label", "Return_Nh"])
    return df, feature_cols


def run_wf_with_vol_filter(df, feature_cols, min_vol_pct=None, max_vol_pct=None, label=""):
    """Run Walk-Forward with optional volatility percentile filter.

    min_vol_pct: skip if Vol_percentile < this (e.g., 0.20 to skip low vol)
    max_vol_pct: skip if Vol_percentile > this (e.g., 0.80 to skip high vol)
    """
    n = len(df)
    splits = walk_forward_splits(n, MIN_TRAIN_SIZE, TEST_SIZE)
    if len(splits) > MAX_FOLDS:
        splits = splits[-MAX_FOLDS:]

    all_returns = []
    all_skipped = 0

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["Label"]
        hist_vol = train_df["Volatility_24"].mean() if "Volatility_24" in train_df.columns else 0

        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        ensemble.fit(X_train, y_train)

        X_test = test_df[feature_cols]
        preds, agreements = ensemble.predict_with_agreement(X_test)
        probas = ensemble.predict_proba(X_test)

        for j in range(len(test_df)):
            row = test_df.iloc[j]
            ts = test_df.index[j]

            # Existing filters
            if ts.hour in SKIP_HOURS:
                all_skipped += 1
                continue
            vol = row.get("Volatility_24", 0)
            if hist_vol > 0 and vol > VOL_MULT * hist_vol:
                all_skipped += 1
                continue

            # Vol percentile filter (NEW)
            vol_pct = row.get("Vol_percentile", 0.5)
            if min_vol_pct is not None and vol_pct < min_vol_pct:
                all_skipped += 1
                continue
            if max_vol_pct is not None and vol_pct > max_vol_pct:
                all_skipped += 1
                continue

            # Confidence & agreement
            p = probas[j]
            confidence = max(p[1], 1.0 - p[1])
            if confidence < CONFIDENCE_THRESHOLD:
                all_skipped += 1
                continue
            if agreements[j] < MIN_AGREEMENT:
                all_skipped += 1
                continue

            direction = 1 if preds[j] == 1 else -1
            trade_return = direction * row["Return_Nh"]
            all_returns.append(trade_return)

    metrics = compute_metrics(np.array(all_returns))
    return metrics


def analyze_vol_regime_pf(df, feature_cols):
    """Analyze PF by vol percentile bucket."""
    n = len(df)
    splits = walk_forward_splits(n, MIN_TRAIN_SIZE, TEST_SIZE)
    if len(splits) > MAX_FOLDS:
        splits = splits[-MAX_FOLDS:]

    bucket_returns = {b: [] for b in ["<10%", "10-20%", "20-40%", "40-60%", "60-80%", "80-90%", ">90%"]}

    def get_bucket(pct):
        if pct < 0.10: return "<10%"
        if pct < 0.20: return "10-20%"
        if pct < 0.40: return "20-40%"
        if pct < 0.60: return "40-60%"
        if pct < 0.80: return "60-80%"
        if pct < 0.90: return "80-90%"
        return ">90%"

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["Label"]
        hist_vol = train_df["Volatility_24"].mean() if "Volatility_24" in train_df.columns else 0

        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        ensemble.fit(X_train, y_train)

        X_test = test_df[feature_cols]
        preds, agreements = ensemble.predict_with_agreement(X_test)
        probas = ensemble.predict_proba(X_test)

        for j in range(len(test_df)):
            row = test_df.iloc[j]
            ts = test_df.index[j]

            if ts.hour in SKIP_HOURS:
                continue
            vol = row.get("Volatility_24", 0)
            if hist_vol > 0 and vol > VOL_MULT * hist_vol:
                continue

            p = probas[j]
            confidence = max(p[1], 1.0 - p[1])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            if agreements[j] < MIN_AGREEMENT:
                continue

            direction = 1 if preds[j] == 1 else -1
            trade_return = direction * row["Return_Nh"]
            vol_pct = row.get("Vol_percentile", 0.5)
            bucket = get_bucket(vol_pct)
            bucket_returns[bucket].append(trade_return)

    return bucket_returns


def load_audjpy_1h():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    path = data_dir / "audjpy_1h.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def main():
    results = []
    results.append("=" * 70)
    results.append("Volatility Regime Filter Analysis (Walk-Forward)")
    results.append("=" * 70)

    for pair_name, loader in [("USDJPY", load_usdjpy_1h), ("AUDJPY", load_audjpy_1h)]:
        results.append(f"\n{'='*50}")
        results.append(f"Pair: {pair_name}")
        results.append(f"{'='*50}")

        print(f"\n[{pair_name}] Loading data...")
        df_raw = loader()
        df, feature_cols = prepare_full_features(df_raw)
        print(f"[{pair_name}] Data: {len(df)} rows")
        results.append(f"Data: {len(df)} rows")

        # 1) Per-bucket analysis
        print(f"[{pair_name}] Analyzing vol regime buckets...")
        bucket_returns = analyze_vol_regime_pf(df, feature_cols)
        results.append("\n--- Vol Regime Bucket Analysis ---")
        for bucket, rets in bucket_returns.items():
            if len(rets) == 0:
                results.append(f"  {bucket:>8s}: No trades")
                continue
            m = compute_metrics(np.array(rets))
            results.append(
                f"  {bucket:>8s}: PF={m['pf']:.2f}, WinRate={m['win_rate']:.1f}%, "
                f"Trades={m['n_trades']}, ExpVal={m['exp_value_net']:+.6f}"
            )

        # 2) Baseline
        print(f"[{pair_name}] Running baseline...")
        base_m = run_wf_with_vol_filter(df, feature_cols)
        results.append(f"\n--- Baseline ---")
        results.append(
            f"  PF={base_m['pf']:.2f}, Sharpe={base_m['sharpe']:.2f}, "
            f"WinRate={base_m['win_rate']:.1f}%, Trades={base_m['n_trades']}, "
            f"ExpVal={base_m['exp_value_net']:+.6f}"
        )

        # 3) Test vol filters
        vol_configs = [
            ("Skip low vol (<10pct)", 0.10, None),
            ("Skip low vol (<15pct)", 0.15, None),
            ("Skip low vol (<20pct)", 0.20, None),
            ("Skip low vol (<25pct)", 0.25, None),
            ("Skip extreme vol (>90pct)", None, 0.90),
            ("Skip both (<15pct, >90pct)", 0.15, 0.90),
            ("Skip both (<20pct, >90pct)", 0.20, 0.90),
            ("Only moderate vol (20-80pct)", 0.20, 0.80),
        ]

        results.append(f"\n--- Vol Filter Comparisons ---")
        best_config = None
        best_pf = base_m["pf"]

        for config_name, min_v, max_v in vol_configs:
            print(f"[{pair_name}] Testing {config_name}...")
            m = run_wf_with_vol_filter(df, feature_cols, min_vol_pct=min_v, max_vol_pct=max_v)
            results.append(
                f"  {config_name}: PF={m['pf']:.2f}, Sharpe={m['sharpe']:.2f}, "
                f"WinRate={m['win_rate']:.1f}%, Trades={m['n_trades']}, "
                f"ExpVal={m['exp_value_net']:+.6f}"
            )
            # Better PF, better EV, and at least 60% trades retained
            if (m["pf"] > best_pf and
                m["exp_value_net"] > base_m["exp_value_net"] and
                m["n_trades"] >= base_m["n_trades"] * 0.60):
                best_config = (config_name, min_v, max_v)
                best_pf = m["pf"]

        if best_config:
            results.append(f"\n  RECOMMENDATION: Apply '{best_config[0]}'")
            results.append(f"    min_vol_pct={best_config[1]}, max_vol_pct={best_config[2]}")
        else:
            results.append(f"\n  RECOMMENDATION: No vol regime filter beneficial")

    output = "\n".join(results)
    print(output)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
