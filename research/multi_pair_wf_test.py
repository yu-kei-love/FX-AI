"""
multi_pair_wf_test.py
Multi-currency pair Walk-Forward backtest using the v3.2 model architecture.

Tests the same LightGBM ensemble model (n_estimators=500, lr=0.03) across
all available currency pairs and compares PF, Sharpe, Win Rate, MDD.

Results are saved to research/fx_multi_pair_results.txt
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

from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits, compute_metrics

DATA_DIR = (script_dir / ".." / "data").resolve()
RESULTS_PATH = script_dir / "fx_multi_pair_results.txt"

# v3.2 model parameters
N_ESTIMATORS = 500
LEARNING_RATE = 0.03
FORECAST_HORIZON = 12  # 12h prediction horizon
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4  # 4/5 models must agree

# Walk-Forward settings
MIN_TRAIN_HOURS = 4320  # 6 months
TEST_HOURS = 720  # 1 month
STEP_HOURS = 720  # step = test size (non-overlapping)


# ============================================================
# Data loading
# ============================================================

def load_pair_1h(pair_name: str) -> pd.DataFrame:
    """Load 1h OHLCV data for a currency pair.
    Handles both formats: clean CSV and USDJPY's 3-header-row format.
    """
    filename = f"{pair_name.lower()}_1h.csv"
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    if pair_name.lower() == "usdjpy":
        # USDJPY has 3 header rows (Price, Ticker, Datetime)
        df = pd.read_csv(
            path, skiprows=3,
            names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
        )
    else:
        df = pd.read_csv(path)

    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Datetime" in df.columns:
        df["Datetime"] = df["Datetime"].astype(str).str.slice(0, 19)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime", "Close"])
        df = df.set_index("Datetime")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.dropna(subset=["Close"])
    df = df.sort_index()
    return df


# ============================================================
# Feature engineering (same as paper_trade.py v3.2)
# ============================================================

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Core technical indicators."""
    close = df["Close"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss_s.rolling(14).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss.replace(0, np.nan))
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, np.nan)

    # Moving Averages
    df["MA_5"] = close.rolling(5).mean()
    df["MA_25"] = close.rolling(25).mean()
    df["MA_75"] = close.rolling(75).mean()

    # Returns & Volatility
    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Return_6"] = close.pct_change(6)
    df["Return_24"] = close.pct_change(24)
    df["Volatility_24"] = df["Return_1"].rolling(24).std()

    # Time features
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek

    return df


def add_regime_features_wf(df: pd.DataFrame, train_end_idx: int) -> pd.DataFrame:
    """Walk-Forward safe HMM regime features (train on train data only)."""
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM
    from numpy.linalg import LinAlgError

    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change(24)
    if "Volatility" not in df.columns:
        df["Volatility"] = df["Return"].rolling(24).std()

    df_train = df.iloc[:train_end_idx].copy()
    df_clean_train = df_train.dropna(subset=["Return", "Volatility"])

    if len(df_clean_train) < 100:
        df["Regime"] = 0
        df["Regime_changed"] = 0
        df["Regime_duration"] = 1
        return df

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_clean_train[["Return", "Volatility"]].values)

    best_model = None
    best_score = -np.inf
    for k in range(5):
        try:
            model = GaussianHMM(n_components=3, covariance_type="diag",
                                n_iter=200, random_state=42 + k)
            model.fit(X_train)
            score = model.score(X_train)
            if score > best_score:
                best_score = score
                best_model = model
        except (LinAlgError, ValueError):
            continue

    if best_model is None:
        df["Regime"] = 0
        df["Regime_changed"] = 0
        df["Regime_duration"] = 1
        return df

    df_clean = df.dropna(subset=["Return", "Volatility"])
    X_all = scaler.transform(df_clean[["Return", "Volatility"]].values)
    states = best_model.predict(X_all)

    df["Regime"] = np.nan
    df.loc[df_clean.index, "Regime"] = states
    df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)
    df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
    regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
    df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1

    return df


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
    """4h and daily timeframe indicators."""
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


def add_interaction_features(df):
    """v3 interaction features."""
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


# Feature column list (same as paper_trade.py v3.2, excluding rate/daily_trend which are USDJPY-specific)
BASE_FEATURE_COLS = [
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "MA_5", "MA_25", "MA_75",
    "Return_1", "Return_3", "Return_6", "Return_24",
    "Volatility_24", "Hour", "DayOfWeek",
    "Regime", "Regime_changed", "Regime_duration",
]

INTERACTION_COLS = [
    "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
    "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
    "Return_skew_12",
]

MTF_COLS = [
    "RSI_4h", "MACD_4h", "BB_width_4h",
    "RSI_daily", "MACD_daily", "BB_width_daily",
]

VOL_REGIME_COLS = ["Vol_percentile", "Vol_of_vol"]

CALENDAR_COLS = [
    "Hour_x_DoW", "Session_tokyo", "Session_london",
    "Session_ny", "Session_overlap",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + INTERACTION_COLS + MTF_COLS + VOL_REGIME_COLS + CALENDAR_COLS


# ============================================================
# Walk-Forward backtest
# ============================================================

def run_walk_forward(pair_name: str) -> dict:
    """Run Walk-Forward backtest for a single currency pair."""
    print(f"\n{'='*60}")
    print(f"  {pair_name} Walk-Forward Backtest")
    print(f"{'='*60}")

    # Load data
    try:
        df = load_pair_1h(pair_name)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    print(f"  Data: {df.index[0]} ~ {df.index[-1]} ({len(df)} bars)")

    # Add technical features (pair-agnostic)
    df = add_technical_features(df)

    # Return/Volatility for regime
    df["Return"] = df["Close"].pct_change(24)
    df["Volatility"] = df["Return"].rolling(24).std()

    # Multi-timeframe features
    df = add_multi_timeframe_features(df)

    # Interaction features
    df = add_interaction_features(df)

    # Volatility regime features
    df = add_volatility_regime_features(df)

    # Calendar features
    df = add_calendar_awareness_features(df)

    # Create label: 12h forward return direction
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]

    # Drop rows without label
    df = df.dropna(subset=["Return_Nh"])

    # Check minimum data
    if len(df) < MIN_TRAIN_HOURS + TEST_HOURS + 200:
        print(f"  SKIP: Insufficient data ({len(df)} bars, need {MIN_TRAIN_HOURS + TEST_HOURS + 200})")
        return None

    # Walk-Forward splits
    splits = walk_forward_splits(len(df), MIN_TRAIN_HOURS, TEST_HOURS, STEP_HOURS)
    print(f"  Walk-Forward splits: {len(splits)}")

    if len(splits) == 0:
        print(f"  SKIP: No valid WF splits")
        return None

    all_returns = []
    all_preds = []
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # Add regime features with WF-safe method (train on train only)
        df_fold = df.copy()
        df_fold = add_regime_features_wf(df_fold, train_end_idx=train_idx[-1] + 1)

        # Drop NaN in feature columns
        valid_mask = df_fold[ALL_FEATURE_COLS].notna().all(axis=1)
        df_fold_clean = df_fold[valid_mask]

        # Re-map indices after cleaning
        train_mask = df_fold_clean.index.isin(df.index[train_idx])
        test_mask = df_fold_clean.index.isin(df.index[test_idx])

        train_df = df_fold_clean[train_mask]
        test_df = df_fold_clean[test_mask]

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        X_train = train_df[ALL_FEATURE_COLS].values
        y_train = (train_df["Close_Nh_later"] > train_df["Close"]).astype(int).values
        X_test = test_df[ALL_FEATURE_COLS].values
        y_test_dir = (test_df["Close_Nh_later"] > test_df["Close"]).astype(int).values
        ret_test = test_df["Return_Nh"].values

        # Train ensemble
        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
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
        trade_mask = (confidence >= CONFIDENCE_THRESHOLD) & (agreement >= MIN_AGREEMENT) & vol_mask

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

        print(f"    Fold {fold_i+1}: trades={fold_metrics['n_trades']}, "
              f"WR={fold_metrics['win_rate']:.1f}%, PF={fold_metrics['pf']:.2f}")

    if len(all_returns) == 0:
        print(f"  RESULT: No trades executed across all folds")
        return None

    # Overall metrics
    all_returns = np.array(all_returns)
    metrics = compute_metrics(all_returns)

    print(f"\n  --- {pair_name} Overall Results ---")
    print(f"  Trades:    {metrics['n_trades']}")
    print(f"  Win Rate:  {metrics['win_rate']:.1f}%")
    print(f"  PF:        {metrics['pf']:.2f}")
    print(f"  Sharpe:    {metrics['sharpe']:.2f}")
    print(f"  MDD:       {metrics['mdd']:.2f}%")
    print(f"  Sortino:   {metrics['sortino']:.2f}")
    print(f"  Exp Value: {metrics['exp_value_net']:+.6f}")
    print(f"  Payoff:    {metrics['payoff']:.2f}")

    return {
        "pair": pair_name,
        "n_bars": len(df),
        "n_folds": len(splits),
        "n_trade_folds": len(fold_results),
        **metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    # Find all available pairs
    pair_files = {
        "USDJPY": "usdjpy_1h.csv",
        "EURUSD": "eurusd_1h.csv",
        "GBPUSD": "gbpusd_1h.csv",
        "EURJPY": "eurjpy_1h.csv",
        "AUDJPY": "audjpy_1h.csv",
    }

    available_pairs = []
    for pair, fname in pair_files.items():
        if (DATA_DIR / fname).exists():
            available_pairs.append(pair)

    print("=" * 60)
    print("  FX Multi-Pair Walk-Forward Backtest")
    print(f"  Model: v3.2 Ensemble (5 models, n_est={N_ESTIMATORS}, lr={LEARNING_RATE})")
    print(f"  Horizon: {FORECAST_HORIZON}h | Confidence: {CONFIDENCE_THRESHOLD} | Agreement: {MIN_AGREEMENT}/5")
    print(f"  Available pairs: {', '.join(available_pairs)}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results = []
    for pair in available_pairs:
        try:
            r = run_walk_forward(pair)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"\n  ERROR on {pair}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n\n")
    print("=" * 80)
    print("  MULTI-PAIR COMPARISON SUMMARY")
    print("=" * 80)

    if not results:
        print("  No results to display.")
        return

    header = f"{'Pair':<10} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'Sharpe':>8} {'MDD%':>7} {'Sortino':>8} {'ExpVal':>10}"
    print(header)
    print("-" * 80)

    for r in results:
        pf_str = f"{r['pf']:.2f}" if not math.isinf(r['pf']) else "inf"
        sharpe_str = f"{r['sharpe']:.2f}" if not math.isnan(r['sharpe']) else "N/A"
        sortino_str = f"{r['sortino']:.2f}" if not (math.isnan(r['sortino']) or math.isinf(r['sortino'])) else "inf"
        line = (f"{r['pair']:<10} {r['n_trades']:>7} {r['win_rate']:>7.1f}% "
                f"{pf_str:>7} {sharpe_str:>8} {r['mdd']:>6.2f}% "
                f"{sortino_str:>8} {r['exp_value_net']:>+10.6f}")
        print(line)

    print("-" * 80)

    # Find best pair by PF (excluding inf)
    valid_pf = [r for r in results if not math.isinf(r['pf']) and not math.isnan(r['pf'])]
    if valid_pf:
        best_pf = max(valid_pf, key=lambda x: x['pf'])
        best_sharpe = max(results, key=lambda x: x['sharpe'] if not math.isnan(x['sharpe']) else -999)
        print(f"\n  Best PF:     {best_pf['pair']} (PF={best_pf['pf']:.2f})")
        if not math.isnan(best_sharpe['sharpe']):
            print(f"  Best Sharpe: {best_sharpe['pair']} (Sharpe={best_sharpe['sharpe']:.2f})")

    # Save results to file
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  FX Multi-Pair Walk-Forward Backtest Results\n")
        f.write(f"  Model: v3.2 Ensemble (5 models, n_est={N_ESTIMATORS}, lr={LEARNING_RATE})\n")
        f.write(f"  Horizon: {FORECAST_HORIZON}h | Confidence >= {CONFIDENCE_THRESHOLD} | Agreement >= {MIN_AGREEMENT}/5\n")
        f.write(f"  WF Settings: min_train={MIN_TRAIN_HOURS}h, test={TEST_HOURS}h, step={STEP_HOURS}h\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        for r in results:
            pf_str = f"{r['pf']:.2f}" if not math.isinf(r['pf']) else "inf"
            sharpe_str = f"{r['sharpe']:.2f}" if not math.isnan(r['sharpe']) else "N/A"
            sortino_str = f"{r['sortino']:.2f}" if not (math.isnan(r['sortino']) or math.isinf(r['sortino'])) else "inf"
            line = (f"{r['pair']:<10} {r['n_trades']:>7} {r['win_rate']:>7.1f}% "
                    f"{pf_str:>7} {sharpe_str:>8} {r['mdd']:>6.2f}% "
                    f"{sortino_str:>8} {r['exp_value_net']:>+10.6f}")
            f.write(line + "\n")
        f.write("-" * 80 + "\n")

        if valid_pf:
            best_pf = max(valid_pf, key=lambda x: x['pf'])
            best_sharpe = max(results, key=lambda x: x['sharpe'] if not math.isnan(x['sharpe']) else -999)
            f.write(f"\nBest PF:     {best_pf['pair']} (PF={best_pf['pf']:.2f})\n")
            if not math.isnan(best_sharpe['sharpe']):
                f.write(f"Best Sharpe: {best_sharpe['pair']} (Sharpe={best_sharpe['sharpe']:.2f})\n")

        f.write("\n\nDetailed per-pair results:\n")
        f.write("=" * 80 + "\n")
        for r in results:
            f.write(f"\n{r['pair']}:\n")
            f.write(f"  Data bars:      {r['n_bars']}\n")
            f.write(f"  WF folds:       {r['n_folds']} (traded in {r['n_trade_folds']})\n")
            f.write(f"  Total trades:   {r['n_trades']}\n")
            f.write(f"  Win rate:       {r['win_rate']:.1f}%\n")
            pf_str = f"{r['pf']:.4f}" if not math.isinf(r['pf']) else "inf"
            f.write(f"  Profit Factor:  {pf_str}\n")
            sharpe_str = f"{r['sharpe']:.4f}" if not math.isnan(r['sharpe']) else "N/A"
            f.write(f"  Sharpe Ratio:   {sharpe_str}\n")
            f.write(f"  Max Drawdown:   {r['mdd']:.2f}%\n")
            sortino_str = f"{r['sortino']:.4f}" if not (math.isnan(r['sortino']) or math.isinf(r['sortino'])) else "inf"
            f.write(f"  Sortino Ratio:  {sortino_str}\n")
            f.write(f"  Exp Value Net:  {r['exp_value_net']:+.6f}\n")
            f.write(f"  Payoff Ratio:   {r['payoff']:.4f}\n")

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
