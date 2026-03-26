"""
fx_time_filter_analysis.py
Time-of-Day Filter Analysis for FX Model (USDJPY)

Walk-Forward validation (expanding window, max 8 folds) to identify
which hours of the day (0-23 UTC) are profitable vs unprofitable.

Uses v3.2 params: n_estimators=500, learning_rate=0.03
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, FEATURE_COLS
from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits, compute_metrics

# ============================================================
# Parameters (v3.2)
# ============================================================
N_ESTIMATORS = 500
LEARNING_RATE = 0.03
FORECAST_HORIZON = 12
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4
VOL_MULT = 2.0
MAX_FOLDS = 8
MIN_TRAIN_SIZE = 4320  # 6 months in hours
TEST_SIZE = 720        # 1 month in hours

OUTPUT_PATH = Path(__file__).resolve().parent / "fx_time_filter_results.txt"


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
    """Load USDJPY 1h and compute all v3.1 features."""
    print("Loading USDJPY 1h data...")
    df = load_usdjpy_1h()
    df = add_technical_features(df)
    df = add_rate_features(df)
    df = add_daily_trend_features(df)
    df = add_multi_timeframe_features(df)

    # Interaction features
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

    # Feature columns (v3.1 - same as paper_trade.py)
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

    # Label: 12h direction
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]

    df = df.dropna(subset=feature_cols + ["Label", "Return_Nh"])
    print(f"Data prepared: {len(df)} rows, {len(feature_cols)} features")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df, feature_cols


def run_walk_forward(df, feature_cols):
    """Run Walk-Forward validation and collect per-trade hour information."""
    n_total = len(df)
    splits = walk_forward_splits(n_total, MIN_TRAIN_SIZE, TEST_SIZE)

    # Limit to MAX_FOLDS (take the last N folds for recency)
    if len(splits) > MAX_FOLDS:
        splits = splits[-MAX_FOLDS:]

    print(f"\nWalk-Forward: {len(splits)} folds (expanding window)")
    print(f"  min_train={MIN_TRAIN_SIZE}h, test={TEST_SIZE}h")
    print(f"  v3.2 params: n_estimators={N_ESTIMATORS}, lr={LEARNING_RATE}")

    # Collect all trades: (hour, direction_return)
    all_trades = []  # list of dict: {hour, return, pred, actual, confidence, agreement}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df["Label"].values
        X_test = test_df[feature_cols].values

        print(f"\n  Fold {fold_i+1}/{len(splits)}: "
              f"train={len(train_idx)} ({df.index[train_idx[0]].strftime('%Y-%m-%d')} ~ "
              f"{df.index[train_idx[-1]].strftime('%Y-%m-%d')}), "
              f"test={len(test_idx)} ({df.index[test_idx[0]].strftime('%Y-%m-%d')} ~ "
              f"{df.index[test_idx[-1]].strftime('%Y-%m-%d')})")

        # Train ensemble
        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        ensemble.fit(X_train, y_train)

        # Historical volatility for vol filter
        hist_vol = train_df["Volatility_24"].mean()

        # Predict on test
        preds, agreement = ensemble.predict_with_agreement(X_test)
        proba = ensemble.predict_proba(X_test)
        confidence = np.maximum(proba[:, 1], 1.0 - proba[:, 1])

        # Volatility filter
        vol_test = test_df["Volatility_24"].values
        vol_ok = vol_test <= VOL_MULT * hist_vol

        # Get actual returns
        ret_test = test_df["Return_Nh"].values
        hours_test = test_df.index.hour

        # Apply filters (confidence + agreement + vol)
        trade_mask = (confidence >= CONFIDENCE_THRESHOLD) & (agreement >= MIN_AGREEMENT) & vol_ok

        n_trades_fold = trade_mask.sum()
        print(f"    Trades: {n_trades_fold} / {len(test_idx)} "
              f"({n_trades_fold/len(test_idx)*100:.1f}%)")

        for i in range(len(test_idx)):
            if not trade_mask[i]:
                continue
            direction = 1.0 if preds[i] == 1 else -1.0
            trade_return = ret_test[i] * direction
            all_trades.append({
                "hour": int(hours_test[i]),
                "return": float(trade_return),
                "pred": int(preds[i]),
                "confidence": float(confidence[i]),
                "agreement": int(agreement[i]),
                "fold": fold_i,
            })

    print(f"\nTotal trades collected: {len(all_trades)}")
    return all_trades


def analyze_by_hour(all_trades):
    """Group trades by hour and compute metrics."""
    if not all_trades:
        print("No trades to analyze!")
        return None, None

    trades_df = pd.DataFrame(all_trades)

    # Overall metrics
    overall_returns = trades_df["return"].values
    overall_metrics = compute_metrics(overall_returns)

    results = []
    for hour in range(24):
        hour_trades = trades_df[trades_df["hour"] == hour]
        n = len(hour_trades)
        if n == 0:
            results.append({
                "hour": hour, "n_trades": 0, "pf": np.nan,
                "win_rate": np.nan, "avg_return": np.nan, "total_return": np.nan,
                "avg_confidence": np.nan, "avg_agreement": np.nan,
            })
            continue

        returns = hour_trades["return"].values
        metrics = compute_metrics(returns)
        results.append({
            "hour": hour,
            "n_trades": n,
            "pf": metrics["pf"],
            "win_rate": metrics["win_rate"],
            "avg_return": float(returns.mean()),
            "total_return": float(returns.sum()),
            "sharpe": metrics["sharpe"],
            "avg_confidence": float(hour_trades["confidence"].mean()),
            "avg_agreement": float(hour_trades["agreement"].mean()),
        })

    hour_df = pd.DataFrame(results)
    return hour_df, overall_metrics


def calculate_improvement(all_trades, bad_hours):
    """Calculate how much improvement if bad hours are excluded."""
    trades_df = pd.DataFrame(all_trades)

    # Before filter
    before_returns = trades_df["return"].values
    before_metrics = compute_metrics(before_returns)

    # After filter
    filtered = trades_df[~trades_df["hour"].isin(bad_hours)]
    after_returns = filtered["return"].values
    after_metrics = compute_metrics(after_returns)

    return before_metrics, after_metrics


def format_results(hour_df, overall_metrics, before_metrics, after_metrics, bad_hours, good_hours):
    """Format all results as a string for output."""
    lines = []
    lines.append("=" * 70)
    lines.append("FX TIME-OF-DAY FILTER ANALYSIS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Model: v3.2 Ensemble (LightGBM+XGBoost+CatBoost+RF+ExtraTrees)")
    lines.append(f"Params: n_estimators={N_ESTIMATORS}, learning_rate={LEARNING_RATE}")
    lines.append(f"Forecast horizon: {FORECAST_HORIZON}h")
    lines.append(f"Filters: confidence>={CONFIDENCE_THRESHOLD}, agreement>={MIN_AGREEMENT}/5, vol<={VOL_MULT}x")
    lines.append(f"Walk-Forward: expanding window, max {MAX_FOLDS} folds")
    lines.append(f"Fee: 0.03% (spread)")
    lines.append("")

    # Overall metrics
    lines.append("-" * 70)
    lines.append("OVERALL METRICS (all hours)")
    lines.append("-" * 70)
    lines.append(f"  Total trades: {overall_metrics['n_trades']}")
    lines.append(f"  Win rate:     {overall_metrics['win_rate']:.1f}%")
    lines.append(f"  PF:           {overall_metrics['pf']:.2f}")
    lines.append(f"  Sharpe:       {overall_metrics['sharpe']:.2f}")
    lines.append(f"  Exp value:    {overall_metrics['exp_value_net']:+.6f}")
    lines.append("")

    # Per-hour table
    lines.append("-" * 70)
    lines.append("PER-HOUR BREAKDOWN (UTC)")
    lines.append("-" * 70)
    lines.append(f"{'Hour':>4} {'JST':>5} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'AvgRet':>10} {'TotalRet':>10} {'Status':>10}")
    lines.append("-" * 70)

    for _, row in hour_df.iterrows():
        h = int(row["hour"])
        jst = (h + 9) % 24
        n = int(row["n_trades"])
        if n == 0:
            lines.append(f"{h:>4} {jst:>4}h {0:>7} {'---':>8} {'---':>7} {'---':>10} {'---':>10} {'NO DATA':>10}")
            continue

        pf = row["pf"]
        wr = row["win_rate"]
        avg_r = row["avg_return"]
        tot_r = row["total_return"]

        if h in bad_hours:
            status = "** BAD **"
        elif h in good_hours:
            status = "GOOD"
        else:
            status = "OK"

        pf_str = f"{pf:.2f}" if not np.isinf(pf) else "inf"
        lines.append(f"{h:>4} {jst:>4}h {n:>7} {wr:>7.1f}% {pf_str:>7} {avg_r:>+10.6f} {tot_r:>+10.4f} {status:>10}")

    lines.append("")

    # Bad hours summary
    lines.append("-" * 70)
    lines.append("BAD HOURS (PF < 1.0 = losing hours)")
    lines.append("-" * 70)
    if bad_hours:
        bad_strs = []
        for h in sorted(bad_hours):
            jst = (h + 9) % 24
            row_data = hour_df[hour_df["hour"] == h].iloc[0]
            pf_val = row_data["pf"]
            pf_str = f"{pf_val:.2f}" if not np.isinf(pf_val) else "inf"
            bad_strs.append(f"  {h:02d}:00 UTC ({jst:02d}:00 JST) - PF={pf_str}, WR={row_data['win_rate']:.1f}%, n={int(row_data['n_trades'])}")
        lines.extend(bad_strs)
    else:
        lines.append("  None - all hours are profitable!")
    lines.append("")

    # Good hours
    lines.append("-" * 70)
    lines.append("BEST HOURS (PF >= 1.3)")
    lines.append("-" * 70)
    if good_hours:
        for h in sorted(good_hours):
            jst = (h + 9) % 24
            row_data = hour_df[hour_df["hour"] == h].iloc[0]
            pf_val = row_data["pf"]
            pf_str = f"{pf_val:.2f}" if not np.isinf(pf_val) else "inf"
            lines.append(f"  {h:02d}:00 UTC ({jst:02d}:00 JST) - PF={pf_str}, WR={row_data['win_rate']:.1f}%, n={int(row_data['n_trades'])}")
    lines.append("")

    # Improvement analysis
    lines.append("-" * 70)
    lines.append("IMPROVEMENT IF BAD HOURS EXCLUDED")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<20} {'Before':>12} {'After':>12} {'Change':>12}")
    lines.append(f"  {'-'*56}")

    def fmt_metric(name, before_val, after_val, fmt=".2f", pct=False):
        suffix = "%" if pct else ""
        bstr = f"{before_val:{fmt}}{suffix}"
        astr = f"{after_val:{fmt}}{suffix}"
        if pct:
            diff = after_val - before_val
            dstr = f"{diff:+{fmt}}pp"
        else:
            diff = after_val - before_val
            dstr = f"{diff:+{fmt}}"
        return f"  {name:<20} {bstr:>12} {astr:>12} {dstr:>12}"

    lines.append(fmt_metric("Trades", before_metrics["n_trades"], after_metrics["n_trades"], fmt=".0f"))
    lines.append(fmt_metric("Win Rate", before_metrics["win_rate"], after_metrics["win_rate"], fmt=".1f", pct=True))
    lines.append(fmt_metric("PF", before_metrics["pf"], after_metrics["pf"]))
    lines.append(fmt_metric("Sharpe", before_metrics["sharpe"], after_metrics["sharpe"]))
    lines.append(fmt_metric("Exp Value (net)", before_metrics["exp_value_net"], after_metrics["exp_value_net"], fmt=".6f"))
    lines.append("")

    # Recommendation
    lines.append("=" * 70)
    lines.append("RECOMMENDATION")
    lines.append("=" * 70)
    if bad_hours:
        bad_utc = ", ".join(f"{h:02d}" for h in sorted(bad_hours))
        bad_jst = ", ".join(f"{(h+9)%24:02d}" for h in sorted(bad_hours))
        lines.append(f"  AVOID trading at hours (UTC): {bad_utc}")
        lines.append(f"  AVOID trading at hours (JST): {bad_jst}")
        lines.append("")
        good_utc = ", ".join(f"{h:02d}" for h in sorted(good_hours))
        good_jst = ", ".join(f"{(h+9)%24:02d}" for h in sorted(good_hours))
        lines.append(f"  BEST hours to trade (UTC): {good_utc}")
        lines.append(f"  BEST hours to trade (JST): {good_jst}")
        lines.append("")
        pf_improvement = after_metrics["pf"] - before_metrics["pf"]
        wr_improvement = after_metrics["win_rate"] - before_metrics["win_rate"]
        lines.append(f"  Expected improvement: PF +{pf_improvement:.2f}, WinRate +{wr_improvement:.1f}pp")
        trade_reduction = (1 - after_metrics["n_trades"] / before_metrics["n_trades"]) * 100
        lines.append(f"  Trade reduction: {trade_reduction:.1f}%")
    else:
        lines.append("  No time filter needed - all hours are profitable.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("FX Time-of-Day Filter Analysis")
    print("=" * 60)

    # Step 1: Prepare data
    df, feature_cols = prepare_data()

    # Step 2: Walk-Forward validation
    all_trades = run_walk_forward(df, feature_cols)

    if not all_trades:
        print("ERROR: No trades generated. Check data/parameters.")
        return

    # Step 3: Analyze by hour
    hour_df, overall_metrics = analyze_by_hour(all_trades)

    # Step 4: Identify bad and good hours
    valid_hours = hour_df[hour_df["n_trades"] >= 5]  # need minimum 5 trades for significance
    bad_hours = set(valid_hours[valid_hours["pf"] < 1.0]["hour"].astype(int).tolist())
    good_hours = set(valid_hours[valid_hours["pf"] >= 1.3]["hour"].astype(int).tolist())

    # Step 5: Calculate improvement
    before_metrics, after_metrics = calculate_improvement(all_trades, bad_hours)

    # Step 6: Format and save results
    report = format_results(hour_df, overall_metrics, before_metrics, after_metrics, bad_hours, good_hours)

    print("\n" + report)

    # Save to file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
