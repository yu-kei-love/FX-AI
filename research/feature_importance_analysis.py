"""
Feature Importance Analysis Across All Models
==============================================
Standalone script: trains LightGBM on each model's data, extracts gain-based
feature importances, checks correlations, and writes a comprehensive report.

Usage:
    python research/feature_importance_analysis.py
"""

import sys
import warnings
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Output
REPORT_PATH = PROJECT_ROOT / "research" / "feature_importance_report.txt"

# ======================================================================
# Utility
# ======================================================================

def train_lgbm_and_get_importance(X, y, feature_names, model_name, n_estimators=300):
    """Train a LightGBM classifier and return gain-based feature importances."""
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    # Remove rows with NaN
    mask = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr))
    X_arr = X_arr[mask]
    y_arr = y_arr[mask]

    if len(X_arr) < 100:
        print(f"  [SKIP] {model_name}: insufficient data ({len(X_arr)} rows)")
        return None

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_arr, y_arr)

    importances = model.feature_importances_  # gain-based by default
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
    imp_df["cumulative_pct"] = imp_df["importance_pct"].cumsum()

    return imp_df


def find_correlated_pairs(X, feature_names, threshold=0.95):
    """Find highly correlated feature pairs."""
    X_df = pd.DataFrame(X, columns=feature_names) if not isinstance(X, pd.DataFrame) else X.copy()
    X_df = X_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    X_df = X_df.fillna(X_df.median())

    corr = X_df.corr().abs()
    pairs = []
    seen = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            c1, c2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            if val >= threshold and (c1, c2) not in seen:
                pairs.append((c1, c2, val))
                seen.add((c1, c2))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def identify_weak_features(imp_df, bottom_pct=0.10):
    """Identify features in the bottom N% by importance."""
    if imp_df is None or len(imp_df) == 0:
        return []
    n = max(1, int(len(imp_df) * bottom_pct))
    weak = imp_df.tail(n)
    return weak


# ======================================================================
# Model 1: FX (USD/JPY)
# ======================================================================

def analyze_fx_model():
    """Analyze FX model feature importance."""
    print("\n" + "=" * 60)
    print("MODEL 1: FX (USD/JPY) - paper_trade.py")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    from research.common.data_loader import load_usdjpy_1h
    from research.common.features import add_technical_features, add_regime_features, FEATURE_COLS

    # Load data
    try:
        df = load_usdjpy_1h(use_5y=True)
    except Exception:
        df = load_usdjpy_1h(use_5y=False)

    if df is None or len(df) == 0:
        print("  [SKIP] No FX data available")
        return None, None, None

    # Build features
    df = add_technical_features(df)
    try:
        df = add_regime_features(df)
    except Exception:
        df["Regime"] = 0
        df["Regime_changed"] = 0
        df["Regime_duration"] = 1

    # Label: 12h forward return direction
    df["Label"] = (df["Close"].shift(-12) > df["Close"]).astype(int)
    df = df.dropna(subset=FEATURE_COLS + ["Label"])

    feature_names = FEATURE_COLS
    X = df[feature_names].copy()
    y = df["Label"].copy()

    print(f"  Data: {len(X)} rows, {len(feature_names)} features")

    imp_df = train_lgbm_and_get_importance(X, y, feature_names, "FX")
    weak = identify_weak_features(imp_df)
    corr_pairs = find_correlated_pairs(X, feature_names)

    return imp_df, weak, corr_pairs


# ======================================================================
# Model 2: Japan Stock
# ======================================================================

def analyze_stock_model():
    """Analyze Japan Stock model feature importance."""
    print("\n" + "=" * 60)
    print("MODEL 2: Japan Stock - japan_stock_model.py")
    print("=" * 60)

    # Load pre-downloaded data from data/japan_stocks/
    stock_dir = DATA_DIR / "japan_stocks"
    if not stock_dir.exists():
        print("  [SKIP] No stock data directory")
        return None, None, None

    # Build all_data dict similar to download_data()
    all_data = {}
    US_INDICES = {"^GSPC": "SP500", "^IXIC": "NASDAQ", "^DJI": "DOW", "^VIX": "VIX"}
    US_SECTOR_ETFS = {"QQQ": "US_Tech", "XLF": "US_Finance", "XLE": "US_Energy", "XLV": "US_Healthcare"}
    JP_STOCKS = {"^N225": "Nikkei225", "1306.T": "TOPIX_ETF", "7203.T": "Toyota", "9984.T": "SoftBank_Group",
                 "6758.T": "Sony", "8306.T": "MUFG"}

    ticker_name_map = {**US_INDICES, **US_SECTOR_ETFS, **JP_STOCKS}
    name_ticker_map = {v: k for k, v in ticker_name_map.items()}

    for csv_file in stock_dir.glob("*.csv"):
        name = csv_file.stem
        if name in name_ticker_map:
            ticker = name_ticker_map[name]
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
            except Exception:
                pass

    if "^N225" not in all_data:
        print("  [SKIP] No Nikkei225 data")
        return None, None, None

    # Build features using japan_stock_model.make_features logic
    target_ticker = "^N225"
    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)

    features = pd.DataFrame(index=target_df.index)

    # US returns
    for ticker, name in US_INDICES.items():
        if ticker in all_data and ticker != "^VIX":
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_returns = us_df["Close"].pct_change()
            us_returns.name = f"{name}_Return"
            features = features.join(us_returns, how="left")
            us_ret_5d = us_df["Close"].pct_change(5)
            us_ret_5d.name = f"{name}_Return_5d"
            features = features.join(us_ret_5d, how="left")

    # US sector returns
    for ticker, name in US_SECTOR_ETFS.items():
        if ticker in all_data:
            sec_df = all_data[ticker].copy()
            sec_df.index = pd.to_datetime(sec_df.index)
            sec_returns = sec_df["Close"].pct_change()
            sec_returns.name = f"{name}_Return"
            features = features.join(sec_returns, how="left")

    # VIX
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_level = vix_df["Close"].copy()
        vix_level.name = "VIX_Level"
        features = features.join(vix_level, how="left")
        vix_change = vix_df["Close"].pct_change()
        vix_change.name = "VIX_Change"
        features = features.join(vix_change, how="left")
        vix_high = (vix_df["Close"] > 25).astype(int)
        vix_high.name = "VIX_High"
        features = features.join(vix_high, how="left")

    # Target momentum
    for period in [1, 5, 20]:
        ret = target_df["Close"].pct_change(period)
        ret.name = f"Target_Return_{period}d"
        features = features.join(ret, how="left")

    # Volume features
    if "Volume" in target_df.columns:
        vol_change = target_df["Volume"].pct_change()
        vol_change.name = "Volume_Change"
        features = features.join(vol_change, how="left")
        vol_ratio = target_df["Volume"] / target_df["Volume"].rolling(20).mean()
        vol_ratio.name = "Volume_Ratio"
        features = features.join(vol_ratio, how="left")

    # Rolling correlations
    target_returns = target_df["Close"].pct_change()
    for ticker, name in US_INDICES.items():
        if ticker in all_data and ticker != "^VIX":
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_ret = us_df["Close"].pct_change()
            combined = pd.DataFrame({"target": target_returns, "us": us_ret}).dropna()
            if len(combined) > 20:
                rolling_corr = combined["target"].rolling(20).corr(combined["us"])
                rolling_corr.name = f"Corr_{name}_20d"
                features = features.join(rolling_corr, how="left")

    # VIX regime features
    if "^VIX" in all_data:
        vix_close = all_data["^VIX"]["Close"].reindex(features.index, method="ffill")
        features["VIX_Percentile"] = vix_close.rolling(120, min_periods=20).rank(pct=True)
        features["VIX_of_VIX"] = vix_close.rolling(20, min_periods=5).std()
        features["VIX_Return_5d"] = vix_close.pct_change(5)

    # Momentum
    target_close = target_df["Close"]
    daily_up = (target_close.pct_change() > 0).astype(float)
    features["RSI_Proxy_10d"] = daily_up.rolling(10, min_periods=5).mean()
    features["Cumulative_Return_5d"] = target_close.pct_change(5)
    features["Cumulative_Return_10d"] = target_close.pct_change(10)
    features["Target_Volatility_20d"] = target_close.pct_change().rolling(20, min_periods=5).std()

    # Calendar
    features["DayOfWeek"] = features.index.dayofweek
    features["Month"] = features.index.month
    features["IsMonday"] = (features.index.dayofweek == 0).astype(int)
    features["IsFriday"] = (features.index.dayofweek == 4).astype(int)

    # Interaction features
    for ticker, name in US_INDICES.items():
        corr_col = f"Corr_{name}_20d"
        if corr_col in features.columns:
            features[f"CorrMom_{name}_5d"] = features[corr_col].diff(5)

    if "VIX_Level" in features.columns:
        features["VIX_Regime"] = np.where(features["VIX_Level"] < 15, 0, np.where(features["VIX_Level"] < 25, 1, 2))
        if "SP500_Return" in features.columns:
            features["VIX_x_SP500"] = features["VIX_Regime"] * features["SP500_Return"]
        if "NASDAQ_Return" in features.columns:
            features["VIX_x_NASDAQ"] = features["VIX_Regime"] * features["NASDAQ_Return"]

    if "US_Tech_Return" in features.columns and "US_Finance_Return" in features.columns:
        features["TechFinance_Spread"] = features["US_Tech_Return"] - features["US_Finance_Return"]
        features["TechFinance_Spread_5d"] = features["TechFinance_Spread"].rolling(5).sum()

    if "US_Energy_Return" in features.columns and "US_Tech_Return" in features.columns:
        features["EnergyTech_Spread"] = features["US_Energy_Return"] - features["US_Tech_Return"]

    if "VIX_Percentile" in features.columns and "Target_Return_5d" in features.columns:
        features["VIXPct_x_Mom5d"] = features["VIX_Percentile"] * features["Target_Return_5d"]

    # Target
    future_return = target_df["Close"].pct_change().shift(-1)
    y = (future_return > 0).astype(int)
    y.name = "Target"

    features = features.ffill()
    common_idx = features.index.intersection(y.dropna().index)
    features = features.loc[common_idx]
    y = y.loc[common_idx]
    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]
    y = y.loc[valid_mask]

    feature_names = features.columns.tolist()
    print(f"  Data: {len(features)} rows, {len(feature_names)} features")

    imp_df = train_lgbm_and_get_importance(features, y, feature_names, "Stock")
    weak = identify_weak_features(imp_df)
    corr_pairs = find_correlated_pairs(features, feature_names)

    return imp_df, weak, corr_pairs


# ======================================================================
# Model 3: Boat Racing
# ======================================================================

def analyze_boat_model():
    """Analyze Boat Racing model feature importance."""
    print("\n" + "=" * 60)
    print("MODEL 3: Boat Racing - boat_model.py")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))

    # Try to import generate_training_data from boat_model
    try:
        from research.boat.boat_model import generate_training_data, create_features, FEATURE_COLS
        df = generate_training_data(n_races=10000, seed=42)
        df = create_features(df)

        feature_names = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_names].copy()
        y = df["win"].copy()

        # Fill NaN
        X = X.fillna(0)

        print(f"  Data: {len(X)} rows, {len(feature_names)} features")

        imp_df = train_lgbm_and_get_importance(X, y, feature_names, "Boat")
        weak = identify_weak_features(imp_df)
        corr_pairs = find_correlated_pairs(X, feature_names)

        return imp_df, weak, corr_pairs

    except Exception as e:
        print(f"  [ERROR] {e}")
        # Fallback: use saved training data
        boat_data = DATA_DIR / "boat" / "model_training_data.csv"
        if boat_data.exists():
            df = pd.read_csv(boat_data)
            print(f"  Loaded saved training data: {len(df)} rows")
            # Try to identify features
            BASE_FEATURE_COLS = [
                "lane", "racer_class", "racer_win_rate", "racer_place_rate",
                "racer_3place_rate", "racer_local_win_rate", "racer_local_2place_rate",
                "motor_2place_rate", "boat_2place_rate", "avg_start_timing",
                "flying_count", "late_count", "racer_weight",
                "weather_wind_speed", "weather_condition", "wave_height", "course_type",
            ]
            feature_names = [c for c in BASE_FEATURE_COLS if c in df.columns]
            # Also add derived features that exist
            for c in df.columns:
                if c not in feature_names and c not in ["win", "place_top2", "race_id", "race_date"]:
                    if df[c].dtype in [np.float64, np.int64, float, int]:
                        feature_names.append(c)

            if "win" in df.columns:
                X = df[feature_names].fillna(0)
                y = df["win"]
                imp_df = train_lgbm_and_get_importance(X, y, feature_names, "Boat")
                weak = identify_weak_features(imp_df)
                corr_pairs = find_correlated_pairs(X, feature_names)
                return imp_df, weak, corr_pairs

        return None, None, None


# ======================================================================
# Model 4: Crypto (BTC)
# ======================================================================

def analyze_crypto_model():
    """Analyze Crypto model feature importance."""
    print("\n" + "=" * 60)
    print("MODEL 4: Crypto (BTC) - hybrid_model.py")
    print("=" * 60)

    crypto_data = DATA_DIR / "crypto" / "btc_1h.csv"
    if not crypto_data.exists():
        # Try daily
        crypto_data = DATA_DIR / "crypto" / "btc_1d.csv"
    if not crypto_data.exists():
        print("  [SKIP] No crypto data available")
        return None, None, None

    try:
        from research.crypto.hybrid_model import build_features, generate_labels, FORECAST_HORIZON, TRANSACTION_COST
    except ImportError as e:
        print(f"  [ERROR] Cannot import crypto model: {e}")
        return None, None, None

    try:
        df = pd.read_csv(crypto_data, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_index()

        # Merge ETH data if available
        eth_path = DATA_DIR / "crypto" / "eth_1h.csv"
        if eth_path.exists():
            try:
                eth = pd.read_csv(eth_path, index_col=0, parse_dates=True)
                if isinstance(eth.columns, pd.MultiIndex):
                    eth.columns = eth.columns.get_level_values(0)
                df["ETH_Close"] = pd.to_numeric(eth["Close"], errors="coerce")
            except Exception:
                pass

        # Merge auxiliary data
        for aux_name, aux_col in [("fear_greed.csv", "fear_greed_index"), ("funding_rate.csv", "funding_rate")]:
            aux_path = DATA_DIR / "crypto" / aux_name
            if aux_path.exists():
                try:
                    aux = pd.read_csv(aux_path, index_col=0, parse_dates=True)
                    if aux_col in aux.columns:
                        df[aux_col] = aux[aux_col]
                    elif len(aux.columns) > 0:
                        df[aux_col] = aux.iloc[:, 0]
                except Exception:
                    pass

        df, feature_names = build_features(df)
        df = generate_labels(df, FORECAST_HORIZON, TRANSACTION_COST)
        df = df.dropna(subset=["label"])

        X = df[feature_names].fillna(0)
        y = df["label"].astype(int)

        print(f"  Data: {len(X)} rows, {len(feature_names)} features")

        imp_df = train_lgbm_and_get_importance(X, y, feature_names, "Crypto")
        weak = identify_weak_features(imp_df)
        corr_pairs = find_correlated_pairs(X, feature_names)

        return imp_df, weak, corr_pairs

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()
        return None, None, None


# ======================================================================
# Model 5: Keiba (Horse Racing)
# ======================================================================

def analyze_keiba_model():
    """Analyze Keiba model feature importance."""
    print("\n" + "=" * 60)
    print("MODEL 5: Keiba (Horse Racing) - keiba_model.py")
    print("=" * 60)

    try:
        from research.keiba.keiba_model import generate_synthetic_data, FEATURE_COLS as KEIBA_FEATURE_COLS
    except ImportError:
        KEIBA_FEATURE_COLS = None

    # Try real data first
    keiba_csv = DATA_DIR / "keiba" / "race_results.csv"
    keiba_csv_v2 = DATA_DIR / "keiba" / "race_results_v2.csv"

    if keiba_csv_v2.exists():
        keiba_csv = keiba_csv_v2

    try:
        from research.keiba.keiba_model import load_real_data, FEATURE_COLS, REAL_FEATURE_COLS
        df = load_real_data()
        if df is not None and len(df) > 0:
            # Use real feature cols
            feature_names = [c for c in REAL_FEATURE_COLS if c in df.columns]
            if len(feature_names) < 5:
                feature_names = [c for c in FEATURE_COLS if c in df.columns]
            X = df[feature_names].fillna(0)
            y = df["win"] if "win" in df.columns else (df["finish"] == 1).astype(int)
            print(f"  Data (real): {len(X)} rows, {len(feature_names)} features")

            imp_df = train_lgbm_and_get_importance(X, y, feature_names, "Keiba")
            weak = identify_weak_features(imp_df)
            corr_pairs = find_correlated_pairs(X, feature_names)
            return imp_df, weak, corr_pairs
    except Exception as e:
        print(f"  [INFO] Real data failed: {e}")

    # Fallback: synthetic data
    try:
        from research.keiba.keiba_model import generate_synthetic_data, FEATURE_COLS as KEIBA_COLS
        df = generate_synthetic_data(n_races=3000, seed=42)
        feature_names = [c for c in KEIBA_COLS if c in df.columns]
        X = df[feature_names].fillna(0)
        y = df["win"]
        print(f"  Data (synthetic): {len(X)} rows, {len(feature_names)} features")

        imp_df = train_lgbm_and_get_importance(X, y, feature_names, "Keiba")
        weak = identify_weak_features(imp_df)
        corr_pairs = find_correlated_pairs(X, feature_names)
        return imp_df, weak, corr_pairs
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback; traceback.print_exc()
        return None, None, None


# ======================================================================
# Report Generation
# ======================================================================

def generate_report(results):
    """Generate comprehensive text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("FEATURE IMPORTANCE ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    model_names = ["FX (USD/JPY)", "Japan Stock (Nikkei225)", "Boat Racing", "Crypto (BTC)", "Keiba (Horse Racing)"]

    all_weak_features = {}  # model -> list of weak features
    all_corr_pairs = {}     # model -> list of correlated pairs

    for i, (name, (imp_df, weak, corr_pairs)) in enumerate(zip(model_names, results)):
        lines.append("-" * 70)
        lines.append(f"MODEL {i+1}: {name}")
        lines.append("-" * 70)

        if imp_df is None:
            lines.append("  ** SKIPPED (no data or import error) **")
            lines.append("")
            continue

        # Feature importance ranking
        lines.append("")
        lines.append("  FEATURE IMPORTANCE RANKING (gain-based):")
        lines.append(f"  {'Rank':<5} {'Feature':<35} {'Importance':<12} {'Pct(%)':<8} {'Cumul(%)':<8}")
        lines.append("  " + "-" * 68)
        for idx, row in imp_df.iterrows():
            lines.append(f"  {idx+1:<5} {row['feature']:<35} {row['importance']:<12.0f} {row['importance_pct']:<8.1f} {row['cumulative_pct']:<8.1f}")

        # Weak features (bottom 10%)
        lines.append("")
        lines.append("  WEAK FEATURES (bottom 10% by importance):")
        if weak is not None and len(weak) > 0:
            weak_names = []
            for _, row in weak.iterrows():
                weak_names.append(row["feature"])
                lines.append(f"    - {row['feature']:<35} importance={row['importance']:.0f}  ({row['importance_pct']:.2f}%)")
            all_weak_features[name] = weak_names
        else:
            lines.append("    (none identified)")

        # Correlated pairs
        lines.append("")
        lines.append("  HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.95):")
        if corr_pairs and len(corr_pairs) > 0:
            all_corr_pairs[name] = corr_pairs
            for f1, f2, r in corr_pairs:
                lines.append(f"    - {f1} <-> {f2}  (r={r:.4f})")
        else:
            lines.append("    (none found)")

        lines.append("")

    # ======================================================================
    # Summary & Recommendations
    # ======================================================================
    lines.append("")
    lines.append("=" * 70)
    lines.append("SUMMARY & RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")

    for name in model_names:
        lines.append(f"--- {name} ---")
        weak_list = all_weak_features.get(name, [])
        corr_list = all_corr_pairs.get(name, [])

        if not weak_list and not corr_list:
            lines.append("  No issues found (or model was skipped).")
        else:
            if weak_list:
                lines.append(f"  Pruning candidates (near-zero importance): {len(weak_list)} features")
                for f in weak_list:
                    lines.append(f"    PRUNE: {f}")
                lines.append("  -> These features add noise without predictive power. Remove to reduce overfitting.")

            if corr_list:
                lines.append(f"  Redundant pairs (>0.95 correlation): {len(corr_list)} pairs")
                for f1, f2, r in corr_list:
                    lines.append(f"    REDUNDANT: {f1} vs {f2} (r={r:.4f})")
                lines.append("  -> Keep the feature with higher importance from each pair, drop the other.")

        lines.append("")

    # Overall recommendations
    lines.append("=" * 70)
    lines.append("OVERALL RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. IMMEDIATE ACTIONS:")
    lines.append("   - Remove features with near-zero importance (bottom 10%)")
    lines.append("   - For correlated pairs, keep the one with higher importance")
    lines.append("")
    lines.append("2. OVERFITTING RISK ASSESSMENT:")
    lines.append("   - Models with many low-importance features are most at risk")
    lines.append("   - High feature count relative to data size increases overfitting")
    lines.append("")
    lines.append("3. MONITORING:")
    lines.append("   - Re-run this analysis after feature pruning to verify improvement")
    lines.append("   - Compare Walk-Forward PF before/after pruning")
    lines.append("")

    report_text = "\n".join(lines)
    return report_text


# ======================================================================
# Main
# ======================================================================

def main():
    print("Feature Importance Analysis - Starting...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")

    results = []

    # 1. FX Model
    try:
        r = analyze_fx_model()
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] FX: {e}")
        import traceback; traceback.print_exc()
        results.append((None, None, None))

    # 2. Stock Model
    try:
        r = analyze_stock_model()
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Stock: {e}")
        import traceback; traceback.print_exc()
        results.append((None, None, None))

    # 3. Boat Model
    try:
        r = analyze_boat_model()
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Boat: {e}")
        import traceback; traceback.print_exc()
        results.append((None, None, None))

    # 4. Crypto Model
    try:
        r = analyze_crypto_model()
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Crypto: {e}")
        import traceback; traceback.print_exc()
        results.append((None, None, None))

    # 5. Keiba Model
    try:
        r = analyze_keiba_model()
        results.append(r)
    except Exception as e:
        print(f"  [ERROR] Keiba: {e}")
        import traceback; traceback.print_exc()
        results.append((None, None, None))

    # Generate report
    report = generate_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()
