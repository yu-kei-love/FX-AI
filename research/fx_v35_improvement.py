"""
FX v3.5 改善実験スクリプト
==========================
v3.5 (MTF shift(1)修正後) の PF 1.19 を改善するための実験。

実験一覧:
  1. 追加特徴量: ATR, ADX, 長期MA距離, 連続陰陽線
  2. max_depth=4 (6から下げる、過学習抑制)
  3. ドローダウンベース動的閾値
  4. Fold内再学習（3ヶ月ごとにモデル更新）
  5. 全改善統合テスト
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from research.common.features import (
    add_technical_features,
    add_multi_timeframe_features,
    FEATURE_COLS,
)
from research.common.ensemble import EnsembleClassifier

# ============================================================
# Constants
# ============================================================
FORECAST_HORIZON = 12
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4
N_FOLDS = 5
MIN_TRAIN_BARS = 2000
N_ESTIMATORS = 300
LEARNING_RATE = 0.062

DATA_DIR = (script_dir / ".." / "data").resolve()
RESULTS_PATH = script_dir / "fx_v35_improvement_results.txt"

LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


# ============================================================
# Data Loading
# ============================================================
def load_data():
    path = DATA_DIR / "usdjpy_1h.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df


# ============================================================
# Feature Engineering
# ============================================================
def base_features(df):
    """ベースライン特徴量 (v3.5と同一)"""
    df = add_technical_features(df)
    df = add_multi_timeframe_features(df)

    df["Return"] = df["Close"].pct_change(24)
    df["Volatility"] = df["Return"].rolling(24).std()

    # Interaction features
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

    # Vol regime
    vol = df["Volatility_24"]
    df["Vol_percentile"] = vol.rolling(720, min_periods=72).apply(
        lambda x: (x[-1] >= x).sum() / len(x), raw=True
    )
    df["Vol_of_vol"] = vol.rolling(120, min_periods=24).std()

    # Calendar
    h = df.index.hour
    dow = df.index.dayofweek
    df["Hour_x_DoW"] = h * 10 + dow
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    df["Session_overlap"] = ((h >= 13) & (h < 16)).astype(int)

    return df


def get_base_feature_cols():
    """ベースライン特徴量リスト"""
    base = [c for c in FEATURE_COLS if not c.startswith("Regime")]
    interaction = [
        "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
        "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
        "Return_skew_12",
    ]
    vol_regime = ["Vol_percentile", "Vol_of_vol"]
    calendar = [
        "Hour_x_DoW", "Session_tokyo", "Session_london",
        "Session_ny", "Session_overlap",
    ]
    return base + interaction + vol_regime + calendar


def add_extra_features(df):
    """実験1: 追加特徴量"""
    # ATR (Average True Range) - 14期間
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()
    df["ATR_norm"] = df["ATR_14"] / df["Close"]  # 正規化ATR

    # ADX (Average Directional Index) - 14期間
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr14 = df["ATR_14"].replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["ADX_14"] = dx.rolling(14).mean()
    df["DI_diff"] = plus_di - minus_di  # トレンド方向の強さ

    # 長期MA距離 (トレンドの位置)
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["Close"]
    df["Dist_SMA200"] = (df["Close"] - df["SMA_200"]) / df["Close"]
    df["SMA50_above_200"] = (df["SMA_50"] > df["SMA_200"]).astype(int)

    # 連続陰陽線カウント
    up = (df["Close"] > df["Close"].shift(1)).astype(int)
    groups = (up != up.shift(1)).cumsum()
    df["Consec_dir"] = up.groupby(groups).cumcount() + 1
    df["Consec_dir"] = df["Consec_dir"] * np.where(up == 1, 1, -1)

    # ローソク足パターン
    body = (df["Close"] - df["Open"]).abs()
    total_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Body_ratio"] = body / total_range  # 実体/全体 (大きい=方向性あり)
    df["Upper_shadow"] = (df["High"] - df[["Close", "Open"]].max(axis=1)) / total_range
    df["Lower_shadow"] = (df[["Close", "Open"]].min(axis=1) - df["Low"]) / total_range

    return df


EXTRA_FEATURE_COLS = [
    "ATR_norm", "ADX_14", "DI_diff",
    "Dist_SMA50", "Dist_SMA200", "SMA50_above_200",
    "Consec_dir", "Body_ratio", "Upper_shadow", "Lower_shadow",
]


# ============================================================
# Filters
# ============================================================
def apply_filters(test_df, df_full):
    """Apply Tuesday skip, hour filter, vol filter."""
    mask = pd.Series(True, index=test_df.index)
    mask &= test_df.index.dayofweek != 1
    mask &= ~test_df.index.hour.isin([20, 23])

    if "Volatility_20" in df_full.columns:
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
        vol_filter = vol_filter.fillna(True)
        mask &= vol_filter

    return mask


def apply_trend_filter(confidence, preds, test_df, df_full, threshold):
    """Soft trend filter."""
    conf_mask = confidence >= threshold
    if "SMA_20" in df_full.columns:
        sma20 = df_full["SMA_20"].reindex(test_df.index)
        price = test_df["Close"]
        price_above_sma = (price > sma20).values
        against_trend = ((preds == 1) & ~price_above_sma) | ((preds == 0) & price_above_sma)
        adj_threshold = np.where(against_trend, threshold + 0.05, threshold)
        conf_mask = confidence >= adj_threshold
    return conf_mask


# ============================================================
# Weighted ensemble
# ============================================================
def compute_model_weights(ensemble, X_val, y_val):
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
    probas = np.array([m.predict_proba(X)[:, 1] for m in ensemble.models])
    weighted_proba = (probas * weights[:, None]).sum(axis=0)
    preds = (weighted_proba >= 0.5).astype(int)
    individual_preds = np.array([m.predict(X) for m in ensemble.models])
    vote_sum = individual_preds.sum(axis=0)
    agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
    return preds, agreement, weighted_proba


# ============================================================
# Walk-Forward Engine
# ============================================================
def run_wf(df, feature_cols, max_depth=6, use_dd_control=False, retrain_interval=0):
    """
    Walk-Forward validation.
    max_depth: ensemble tree depth
    use_dd_control: ドローダウンベース動的閾値
    retrain_interval: 0=fold毎, >0=X本毎に再学習
    """
    n = len(df)
    test_size = (n - MIN_TRAIN_BARS) // N_FOLDS

    all_trade_returns = []
    fold_metrics = []

    for fold in range(N_FOLDS):
        train_end = MIN_TRAIN_BARS + fold * test_size
        test_start = train_end
        test_end = min(train_end + test_size, n) if fold < N_FOLDS - 1 else n

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        if len(test_df) == 0:
            continue

        val_size = min(500, len(train_df) // 5)
        X_train = train_df[feature_cols].iloc[:-val_size]
        y_train = train_df["Label"].iloc[:-val_size]
        X_val = train_df[feature_cols].iloc[-val_size:]
        y_val = train_df["Label"].iloc[-val_size:]

        # Build ensemble with specified max_depth
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        # Override max_depth
        ensemble.model_lgb.set_params(max_depth=max_depth)
        ensemble.model_xgb.set_params(max_depth=max_depth)
        ensemble.model_cat.set_params(depth=min(max_depth, 8))
        ensemble.model_rf.set_params(max_depth=max_depth + 2)
        ensemble.model_et.set_params(max_depth=max_depth + 2)

        if retrain_interval > 0:
            # Fold内で定期再学習
            fold_returns = _run_fold_with_retrain(
                df, train_end, test_start, test_end, feature_cols,
                ensemble, retrain_interval, use_dd_control
            )
        else:
            ensemble.fit(X_train, y_train)
            weights = compute_model_weights(ensemble, X_val, y_val)
            fold_returns = _evaluate_fold(
                df, test_df, feature_cols, ensemble, weights, use_dd_control
            )

        all_trade_returns.extend(fold_returns)

        # Fold metrics
        tr = np.array(fold_returns) if fold_returns else np.array([])
        n_trades = len(tr)
        if n_trades > 0:
            gp = tr[tr > 0].sum()
            gl = abs(tr[tr < 0].sum())
            pf = gp / gl if gl > 0 else np.inf
            wr = (tr > 0).sum() / n_trades
            cumret = np.cumsum(tr)
            peak = np.maximum.accumulate(cumret)
            mdd = (cumret - peak).min() * 100
        else:
            pf, wr, mdd = np.nan, np.nan, np.nan

        fold_metrics.append({
            "fold": fold + 1, "n_trades": n_trades,
            "pf": pf, "winrate": wr, "mdd": mdd,
            "test_start": str(test_df.index[0])[:10],
            "test_end": str(test_df.index[-1])[:10],
        })

    return fold_metrics, np.array(all_trade_returns)


def _evaluate_fold(df, test_df, feature_cols, ensemble, weights, use_dd_control):
    """Evaluate a single fold."""
    X_test = test_df[feature_cols]
    preds, agreement, weighted_proba = weighted_predict(ensemble, X_test, weights)
    confidence = np.maximum(weighted_proba, 1.0 - weighted_proba)

    filter_mask = apply_filters(test_df, df)
    agree_mask = agreement >= MIN_AGREEMENT

    threshold = CONFIDENCE_THRESHOLD
    if use_dd_control:
        # 動的閾値: 直近のドローダウンに応じて閾値を上げる
        conf_mask = _dynamic_threshold(confidence, preds, test_df, df, agreement)
    else:
        conf_mask = apply_trend_filter(confidence, preds, test_df, df, threshold)

    trade_mask = filter_mask & conf_mask & agree_mask
    traded_preds = preds[trade_mask]
    traded_returns = test_df["Return_Nh"].values[trade_mask]

    direction = np.where(traded_preds == 1, 1.0, -1.0)
    return list(direction * traded_returns)


def _dynamic_threshold(confidence, preds, test_df, df, agreement):
    """ドローダウンベース動的閾値"""
    conf_mask = np.zeros(len(test_df), dtype=bool)
    cum_return = 0.0
    peak_return = 0.0
    current_dd = 0.0

    for i in range(len(test_df)):
        # ドローダウンが深いほど閾値を上げる
        dd_penalty = min(abs(current_dd) * 2.0, 0.15)  # 最大+15%
        threshold = CONFIDENCE_THRESHOLD + dd_penalty

        # トレンドフィルター
        idx = test_df.index[i]
        if "SMA_20" in df.columns:
            sma20_val = df["SMA_20"].get(idx, np.nan)
            price_val = test_df["Close"].iloc[i]
            if not np.isnan(sma20_val):
                against_trend = ((preds[i] == 1) and (price_val <= sma20_val)) or \
                               ((preds[i] == 0) and (price_val > sma20_val))
                if against_trend:
                    threshold += 0.05

        if confidence[i] >= threshold and agreement[i] >= MIN_AGREEMENT:
            conf_mask[i] = True

        # リターン追跡（次のバーで更新）
        if i > 0 and conf_mask[i - 1]:
            ret = test_df["Return_Nh"].iloc[i - 1]
            d = 1.0 if preds[i - 1] == 1 else -1.0
            trade_ret = d * ret
            cum_return += trade_ret
            peak_return = max(peak_return, cum_return)
            current_dd = cum_return - peak_return

    # 基本フィルタも適用
    base_filter = apply_filters(test_df, df)
    return pd.Series(conf_mask, index=test_df.index) & base_filter


def _run_fold_with_retrain(df, train_end, test_start, test_end, feature_cols,
                           ensemble, retrain_interval, use_dd_control):
    """Fold内で定期的に再学習"""
    all_returns = []
    current_train_end = train_end

    for chunk_start in range(test_start, test_end, retrain_interval):
        chunk_end = min(chunk_start + retrain_interval, test_end)
        train_df = df.iloc[:current_train_end].copy()
        test_df = df.iloc[chunk_start:chunk_end].copy()

        if len(test_df) == 0:
            break

        val_size = min(500, len(train_df) // 5)
        X_train = train_df[feature_cols].iloc[:-val_size]
        y_train = train_df["Label"].iloc[:-val_size]
        X_val = train_df[feature_cols].iloc[-val_size:]
        y_val = train_df["Label"].iloc[-val_size:]

        ensemble.fit(X_train, y_train)
        weights = compute_model_weights(ensemble, X_val, y_val)

        chunk_returns = _evaluate_fold(df, test_df, feature_cols, ensemble, weights, use_dd_control)
        all_returns.extend(chunk_returns)

        # 次のチャンクではこのチャンクのデータも学習に含める
        current_train_end = chunk_end

    return all_returns


# ============================================================
# Metrics
# ============================================================
def calc_overall_metrics(trade_returns):
    tr = np.asarray(trade_returns, dtype=float)
    n = len(tr)
    if n == 0:
        return {"n_trades": 0, "pf": np.nan, "wr": np.nan, "mdd": np.nan, "total_ret": np.nan}

    gp = tr[tr > 0].sum()
    gl = abs(tr[tr < 0].sum())
    pf = gp / gl if gl > 0 else np.inf
    wr = (tr > 0).sum() / n
    cumret = np.cumsum(tr)
    peak = np.maximum.accumulate(cumret)
    mdd = (cumret - peak).min() * 100
    total_ret = tr.sum() * 100

    return {"n_trades": n, "pf": round(pf, 4), "wr": round(wr * 100, 2),
            "mdd": round(mdd, 2), "total_ret": round(total_ret, 2)}


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 70)
    log("FX v3.5 改善実験")
    log(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 70)

    # Load data
    log("\n[データ読み込み] USDJPY 1h...")
    df = load_data()
    log(f"  {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Base features
    log("\n[特徴量計算] ベースライン特徴量...")
    df = base_features(df)

    # Extra features
    log("[特徴量計算] 追加特徴量 (ATR, ADX, SMA距離, etc.)...")
    df = add_extra_features(df)

    # Label
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Volatility_20"] = df["Return_1"].rolling(20).std()

    base_cols = get_base_feature_cols()
    all_cols = base_cols + EXTRA_FEATURE_COLS

    # Ensure no NaN in feature columns
    df = df.dropna(subset=all_cols + ["Label", "Return_Nh"])
    log(f"  データ: {len(df)} bars, ベース特徴量: {len(base_cols)}, 追加含む: {len(all_cols)}")

    # ============================================================
    # BASELINE (v3.5 そのまま)
    # ============================================================
    log("\n" + "=" * 70)
    log("BASELINE (v3.5, max_depth=6)")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(df, base_cols, max_depth=6)
    baseline = calc_overall_metrics(trade_returns)
    log(f"  PF={baseline['pf']}, WR={baseline['wr']}%, "
        f"MDD={baseline['mdd']}%, Trades={baseline['n_trades']}, "
        f"TotalRet={baseline['total_ret']}%")
    for fm in fold_metrics:
        log(f"  Fold {fm['fold']}: trades={fm['n_trades']}, PF={fm.get('pf', 'N/A'):.2f}" if fm['n_trades'] > 0
            else f"  Fold {fm['fold']}: trades=0")

    # ============================================================
    # 実験1: 追加特徴量 (ATR, ADX, SMA距離, etc.)
    # ============================================================
    log("\n" + "=" * 70)
    log("TEST 1: 追加特徴量 (ATR, ADX, SMA距離, 連続陰陽線)")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(df, all_cols, max_depth=6)
    exp1 = calc_overall_metrics(trade_returns)
    delta_pf = exp1['pf'] - baseline['pf'] if not np.isnan(exp1['pf']) else 0
    log(f"  PF={exp1['pf']} (delta={delta_pf:+.4f}), WR={exp1['wr']}%, "
        f"MDD={exp1['mdd']}%, Trades={exp1['n_trades']}, "
        f"TotalRet={exp1['total_ret']}%")
    verdict1 = "BENEFICIAL" if delta_pf > 0.05 else "NOT BENEFICIAL"
    log(f"  -> {verdict1}")

    # ============================================================
    # 実験2: max_depth=4 (過学習抑制)
    # ============================================================
    log("\n" + "=" * 70)
    log("TEST 2: max_depth=4 (ベース特徴量)")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(df, base_cols, max_depth=4)
    exp2 = calc_overall_metrics(trade_returns)
    delta_pf = exp2['pf'] - baseline['pf'] if not np.isnan(exp2['pf']) else 0
    log(f"  PF={exp2['pf']} (delta={delta_pf:+.4f}), WR={exp2['wr']}%, "
        f"MDD={exp2['mdd']}%, Trades={exp2['n_trades']}, "
        f"TotalRet={exp2['total_ret']}%")
    verdict2 = "BENEFICIAL" if delta_pf > 0.05 else "NOT BENEFICIAL"
    log(f"  -> {verdict2}")

    # ============================================================
    # 実験3: max_depth=4 + 追加特徴量
    # ============================================================
    log("\n" + "=" * 70)
    log("TEST 3: max_depth=4 + 追加特徴量")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(df, all_cols, max_depth=4)
    exp3 = calc_overall_metrics(trade_returns)
    delta_pf = exp3['pf'] - baseline['pf'] if not np.isnan(exp3['pf']) else 0
    log(f"  PF={exp3['pf']} (delta={delta_pf:+.4f}), WR={exp3['wr']}%, "
        f"MDD={exp3['mdd']}%, Trades={exp3['n_trades']}, "
        f"TotalRet={exp3['total_ret']}%")
    verdict3 = "BENEFICIAL" if delta_pf > 0.05 else "NOT BENEFICIAL"
    log(f"  -> {verdict3}")

    # ============================================================
    # 実験4: ドローダウンベース動的閾値
    # ============================================================
    log("\n" + "=" * 70)
    log("TEST 4: ドローダウンベース動的閾値 (ベース特徴量, depth=6)")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(df, base_cols, max_depth=6, use_dd_control=True)
    exp4 = calc_overall_metrics(trade_returns)
    delta_pf = exp4['pf'] - baseline['pf'] if not np.isnan(exp4['pf']) else 0
    log(f"  PF={exp4['pf']} (delta={delta_pf:+.4f}), WR={exp4['wr']}%, "
        f"MDD={exp4['mdd']}%, Trades={exp4['n_trades']}, "
        f"TotalRet={exp4['total_ret']}%")
    verdict4 = "BENEFICIAL" if delta_pf > 0.05 else "NOT BENEFICIAL"
    log(f"  -> {verdict4}")

    # ============================================================
    # 実験5: Fold内再学習 (720本=約1ヶ月ごと)
    # ============================================================
    log("\n" + "=" * 70)
    log("TEST 5: Fold内再学習 (720本毎, ベース特徴量, depth=6)")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(df, base_cols, max_depth=6,
                                          retrain_interval=720)
    exp5 = calc_overall_metrics(trade_returns)
    delta_pf = exp5['pf'] - baseline['pf'] if not np.isnan(exp5['pf']) else 0
    log(f"  PF={exp5['pf']} (delta={delta_pf:+.4f}), WR={exp5['wr']}%, "
        f"MDD={exp5['mdd']}%, Trades={exp5['n_trades']}, "
        f"TotalRet={exp5['total_ret']}%")
    verdict5 = "BENEFICIAL" if delta_pf > 0.05 else "NOT BENEFICIAL"
    log(f"  -> {verdict5}")

    # ============================================================
    # 実験6: 最良の組み合わせ
    # ============================================================
    # 有効だった改善を全て組み合わせる
    best_depth = 4 if exp2['pf'] > baseline['pf'] else 6
    best_cols = all_cols if exp1['pf'] > baseline['pf'] else base_cols
    best_dd = exp4['pf'] > baseline['pf']
    best_retrain = 720 if exp5['pf'] > baseline['pf'] else 0

    log("\n" + "=" * 70)
    log(f"TEST 6: 最良組み合わせ (depth={best_depth}, "
        f"extra_feat={'Y' if best_cols == all_cols else 'N'}, "
        f"dd_ctrl={'Y' if best_dd else 'N'}, "
        f"retrain={best_retrain})")
    log("=" * 70)
    fold_metrics, trade_returns = run_wf(
        df, best_cols, max_depth=best_depth,
        use_dd_control=best_dd, retrain_interval=best_retrain
    )
    exp6 = calc_overall_metrics(trade_returns)
    delta_pf = exp6['pf'] - baseline['pf'] if not np.isnan(exp6['pf']) else 0
    log(f"  PF={exp6['pf']} (delta={delta_pf:+.4f}), WR={exp6['wr']}%, "
        f"MDD={exp6['mdd']}%, Trades={exp6['n_trades']}, "
        f"TotalRet={exp6['total_ret']}%")
    verdict6 = "BENEFICIAL" if delta_pf > 0.05 else "NOT BENEFICIAL"
    log(f"  -> {verdict6}")

    # ============================================================
    # サマリー
    # ============================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"{'実験':<40} {'PF':>8} {'WR%':>8} {'MDD%':>8} {'Trades':>7} {'判定':<15}")
    log("-" * 90)
    exps = [
        ("BASELINE (v3.5, depth=6)", baseline, "---"),
        ("1. 追加特徴量", exp1, verdict1),
        ("2. max_depth=4", exp2, verdict2),
        ("3. depth=4 + 追加特徴量", exp3, verdict3),
        ("4. DD動的閾値", exp4, verdict4),
        ("5. Fold内再学習(720)", exp5, verdict5),
        ("6. 最良組み合わせ", exp6, verdict6),
    ]
    for name, m, v in exps:
        log(f"{name:<40} {m['pf']:>8} {m['wr']:>8} {m['mdd']:>8} {m['n_trades']:>7} {v:<15}")

    log("\n" + "=" * 70)
    log("実験完了")
    log("=" * 70)

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"\n結果保存: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
