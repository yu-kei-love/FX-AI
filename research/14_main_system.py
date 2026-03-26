# ===========================================
# 14_main_system.py v3.1
# Walk-Forward検証付き統合メインシステム
# v3.0 Changes:
#   - HMM fitted per-window (no lookahead bias)
#   - 5-model ensemble (LGB+XGB+CatBoost+RF+ET)
#   - Confidence threshold optimized per-window
#   - Simplified signal logic (no regime switching)
#   - Embargo increased to 72h
# v3.1 Changes:
#   - Multi-timeframe features (4h, daily RSI/MACD/BB_width)
#   - Volatility regime features (vol percentile, vol-of-vol)
#   - Economic calendar awareness (hour×dow, session indicators)
#   - Performance-based ensemble weighting per window
#   - Finer confidence threshold candidates
# ===========================================

import json
import sys
import math
from datetime import datetime, date
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, add_regime_features_wf, FEATURE_COLS
from research.common.labels import build_triple_barrier_labels, build_volatility_barriers
from research.common.validation import walk_forward_splits, compute_metrics, print_metrics
from research.common.ensemble import EnsembleClassifier

# ----- 定数 -----
BARRIER_UP, BARRIER_DOWN, BARRIER_T = 0.005, -0.003, 24
# 待機条件
VOL_MULT = 2.0
REGIME_CHANGE_THRESH = 2
PRICE_CHANGE_MULT = 3.0
# 自信度フィルター（動的に最適化する）— v3.1: finer granularity
CONFIDENCE_CANDIDATES = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]
# Walk-Forward 設定（1ヶ月 ≈ 720本, 最小学習期間 6ヶ月 ≈ 4320本）
WF_MIN_TRAIN = 4320
WF_TEST_SIZE = 720
# アンサンブル一致条件
MIN_AGREEMENT = 4  # 5人中4人以上一致

script_dir = Path(__file__).resolve().parent


# ----- 経済指標イベント -----
def generate_economic_events(start_dt, end_dt):
    events = []
    y, m = start_dt.year, start_dt.month
    end_y, end_m = end_dt.year, end_dt.month
    while (y, m) <= (end_y, end_m):
        for d in range(1, 8):
            try:
                cand = date(y, m, d)
                if cand.weekday() == 4:
                    events.append((datetime(y, m, d, 14, 0, 0), 3))
                    break
            except ValueError:
                pass
        wed_count = 0
        for d in range(1, monthrange(y, m)[1] + 1):
            try:
                cand = date(y, m, d)
                if cand.weekday() == 2:
                    wed_count += 1
                    if wed_count == 3:
                        events.append((datetime(y, m, d, 14, 0, 0), 3))
                        break
            except ValueError:
                pass
        try:
            events.append((datetime(y, m, 15, 14, 0, 0), 2))
        except ValueError:
            pass
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return events


def build_economic_wait_flag(dt_series, events, hours_before=2, hours_after=1, min_importance=2):
    wait = np.zeros(len(dt_series), dtype=bool)
    dt_arr = pd.to_datetime(dt_series)
    for i, t in enumerate(dt_arr):
        t = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        if t.tzinfo:
            t = t.replace(tzinfo=None)
        for ev_dt, imp in events:
            if imp < min_importance:
                continue
            if ev_dt.tzinfo:
                ev_dt = ev_dt.replace(tzinfo=None)
            delta_h = (ev_dt - t).total_seconds() / 3600
            if 0 <= delta_h <= hours_before or -hours_after <= delta_h < 0:
                wait[i] = True
                break
    return wait


def optimize_confidence_threshold(X_val, y_val, ret_val, ensemble, wait_mask=None):
    """Validation setで最適な信頼度閾値を見つける"""
    preds, agreement = ensemble.predict_with_agreement(X_val)
    proba = ensemble.predict_proba(X_val)[:, 1]

    best_pf = 0.0
    best_thresh = 0.60

    for thresh in CONFIDENCE_CANDIDATES:
        confidence = np.maximum(proba, 1.0 - proba)
        trade_mask = (confidence >= thresh) & (agreement >= MIN_AGREEMENT)
        if wait_mask is not None:
            trade_mask &= ~wait_mask

        if trade_mask.sum() < 20:
            continue

        direction = np.where(preds[trade_mask] == 1, 1.0, -1.0)
        returns = ret_val[trade_mask] * direction
        profit = returns[returns > 0].sum()
        loss = abs(returns[returns < 0].sum())
        pf = profit / loss if loss > 0 else (float('inf') if profit > 0 else 0.0)

        # PFが高すぎるのは過学習の兆候 → capする
        pf = min(pf, 5.0)

        if pf > best_pf:
            best_pf = pf
            best_thresh = thresh

    return best_thresh, best_pf


# ----- Multi-timeframe features (v3.1) -----
def _resample_ohlcv(df, rule):
    """Resample OHLCV to a coarser timeframe."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    resampled = df.resample(rule).agg(agg).dropna()
    return resampled


def _compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss.replace(0, np.nan))
    return pd.Series(100.0 - (100.0 / (1.0 + rs)), index=close.index)


def _compute_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _compute_bb_width(close, period=20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    bb_width = (bb_upper - bb_lower) / sma.replace(0, np.nan)
    return bb_width


def add_multi_timeframe_features(df):
    """Add 4h and daily timeframe indicators to 1h DataFrame."""
    for rule, suffix in [("4h", "4h"), ("1D", "daily")]:
        resampled = _resample_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]], rule)
        close = resampled["Close"]

        feat = pd.DataFrame(index=resampled.index)
        feat[f"RSI_{suffix}"] = _compute_rsi(close, 14)
        macd_l, _, macd_h = _compute_macd(close)
        feat[f"MACD_{suffix}"] = macd_l
        feat[f"BB_width_{suffix}"] = _compute_bb_width(close)

        # Forward-fill onto 1h index
        feat = feat.reindex(df.index, method="ffill")
        for col in feat.columns:
            df[col] = feat[col]
    return df


# ----- Volatility regime features (v3.1) -----
def add_volatility_regime_features(df):
    """Add realized vol percentile and vol-of-vol (no lookahead)."""
    vol = df["Volatility_24"]
    # Realized volatility percentile: current vol vs rolling 30-day (720h) distribution
    df["Vol_percentile"] = vol.rolling(720, min_periods=72).apply(
        lambda x: (x[-1] >= x).sum() / len(x), raw=True
    )
    # Volatility of volatility: rolling std of volatility over 5 days (120h)
    df["Vol_of_vol"] = vol.rolling(120, min_periods=24).std()
    return df


# ----- Economic calendar awareness features (v3.1) -----
def add_calendar_awareness_features(df):
    """Add hour×dow interaction and session indicators as features."""
    h = df.index.hour
    dow = df.index.dayofweek
    # Hour × DayOfWeek interaction (encoded cyclically already exists, add direct interaction)
    df["Hour_x_DoW"] = h * 10 + dow  # simple interaction encoding
    # Tokyo/London/NY session indicators (already in features.py but may not be in FEATURE_COLS)
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    df["Session_overlap"] = ((h >= 13) & (h < 16)).astype(int)
    return df


# ----- Performance-based ensemble weighting (v3.1) -----
def compute_model_weights(ensemble, X_val, y_val):
    """Compute per-model weights based on validation accuracy."""
    weights = []
    for model in ensemble.models:
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        weights.append(acc)
    weights = np.array(weights)
    # Sharpen weights: exponentiate to give more weight to better models
    weights = weights ** 3
    weights = weights / weights.sum()
    return weights


def weighted_predict(ensemble, X, weights):
    """Weighted prediction using per-model weights."""
    probas = np.array([m.predict_proba(X)[:, 1] for m in ensemble.models])
    weighted_proba = (probas * weights[:, None]).sum(axis=0)
    preds = (weighted_proba >= 0.5).astype(int)
    # Agreement: count how many models agree with the weighted prediction
    individual_preds = np.array([m.predict(X) for m in ensemble.models])
    vote_sum = individual_preds.sum(axis=0)
    agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
    return preds, agreement, weighted_proba


# ===== データ準備 =====
print("データ読み込み中...")
df = load_usdjpy_1h()
df = add_technical_features(df)
df = add_rate_features(df)
df = add_daily_trend_features(df)

# v3.1: Multi-timeframe features
df = add_multi_timeframe_features(df)

# Return/Volatility
df["Return"] = df["Close"].pct_change(24)
df["Volatility"] = df["Return"].rolling(24).std()

# === 新特徴量: 交互作用・相対指標 ===
# RSI × ボラティリティ（高ボラ時のRSI反転は強い）
df["RSI_x_Vol"] = df["RSI_14"] * df["Volatility_24"]
# MACD相対強度（MACDをボラティリティで正規化）
df["MACD_norm"] = df["MACD"] / df["Volatility_24"].replace(0, np.nan)
# BB位置（0=lower, 1=upper）
bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
df["BB_position"] = (df["Close"] - df["BB_lower"]) / bb_range
# 短期 vs 長期トレンド乖離
df["MA_cross"] = (df["MA_5"] - df["MA_75"]) / df["Close"]
# モメンタム加速度（2次微分）
df["Momentum_accel"] = df["Return_1"] - df["Return_1"].shift(1)
# ボラティリティ変化率
df["Vol_change"] = df["Volatility_24"].pct_change(6)
# 高値安値レンジ比
df["HL_ratio"] = (df["High"] - df["Low"]) / df["Close"]
# 終値位置（ローソク内での位置 0=安値, 1=高値）
hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
df["Close_position"] = (df["Close"] - df["Low"]) / hl_range
# リターンの非対称性（正のリターンの比率 − 負のリターンの比率、12h window）
df["Return_skew_12"] = df["Return_1"].rolling(12).apply(
    lambda x: (x > 0).sum() / len(x) - 0.5, raw=True
)

# v3.1: Volatility regime features
df = add_volatility_regime_features(df)

# v3.1: Economic calendar awareness features
df = add_calendar_awareness_features(df)

# ラベル（12h先を予測 — 4hでは手数料に対してリターンが小さすぎる）
FORECAST_HORIZON = 12
df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]
df["Abs_ret_1h"] = df["Return_1"].abs()

# 経済指標待機フラグ
events = generate_economic_events(
    df.index.min().to_pydatetime() if hasattr(df.index.min(), "to_pydatetime") else df.index.min(),
    df.index.max().to_pydatetime() if hasattr(df.index.max(), "to_pydatetime") else df.index.max(),
)
df["wait_economic"] = build_economic_wait_flag(df.index, events, 2, 1, 2)

# 特徴量カラム（Regime系なし + 新交互作用特徴量）
interaction_cols = [
    "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
    "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
    "Return_skew_12",
]
# v3.1: Multi-timeframe features
mtf_cols = [
    "RSI_4h", "MACD_4h", "BB_width_4h",
    "RSI_daily", "MACD_daily", "BB_width_daily",
]
# v3.1: Volatility regime features
vol_regime_cols = ["Vol_percentile", "Vol_of_vol"]
# v3.1: Calendar awareness features
calendar_cols = ["Hour_x_DoW", "Session_tokyo", "Session_london", "Session_ny", "Session_overlap"]
base_feature_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]
regime_cols = []  # v3.0: レジームなし（HMM per-windowでも効果薄）

# 欠損値除去
all_feature_cols = base_feature_cols + interaction_cols + mtf_cols + vol_regime_cols + calendar_cols
df = df.dropna(subset=all_feature_cols + ["Label", "Return_Nh"])
n_total = len(df)

print(f"データ: {n_total}本 ({df.index.min()} ~ {df.index.max()})")


# ===== 1ウィンドウ分の学習・予測・評価 =====
def run_window(train_idx, test_idx):
    """Walk-Forward 1ウィンドウ分の処理。"""
    feature_cols = all_feature_cols

    X_train = df[feature_cols].iloc[train_idx].values
    X_test = df[feature_cols].iloc[test_idx].values

    y_train = df["Label"].iloc[train_idx].values
    y_test_arr = df["Label"].iloc[test_idx].values

    # 待機条件用の配列
    vol_test = df["Volatility"].iloc[test_idx].values
    rc_test = np.zeros(len(test_idx))  # レジームなし
    abs_test = df["Abs_ret_1h"].iloc[test_idx].values
    ret4_test = df["Return_Nh"].iloc[test_idx].values
    wait_econ_test = df["wait_economic"].iloc[test_idx].values
    hist_vol = df["Volatility"].iloc[train_idx].mean()
    hist_abs = df["Abs_ret_1h"].iloc[train_idx].mean()
    if hist_abs <= 0:
        hist_abs = 1e-8

    # --- アンサンブルモデル学習 ---
    # Validation split (最後の15%をthreshold最適化に使用)
    val_size = int(len(train_idx) * 0.15)
    train_size = len(train_idx) - val_size

    X_fit = X_train[:train_size]
    y_fit = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    ret_val = df["Return_Nh"].iloc[train_idx[train_size:]].values

    ensemble = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
    ensemble.fit(X_fit, y_fit)

    # v3.1: Compute per-model weights based on validation performance
    model_weights = compute_model_weights(ensemble, X_val, y_val)

    # 信頼度閾値を最適化
    wait_val = df["wait_economic"].iloc[train_idx[train_size:]].values.astype(bool)
    best_thresh, val_pf = optimize_confidence_threshold(
        X_val, y_val, ret_val, ensemble, wait_val
    )

    # 全trainデータで再学習
    ensemble_full = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
    ensemble_full.fit(X_train, y_train)

    # --- 予測 (v3.1: weighted ensemble) ---
    preds, agreement, proba = weighted_predict(ensemble_full, X_test, model_weights)

    # 待機条件
    wait_c = (
        (vol_test > VOL_MULT * hist_vol)
        | (rc_test >= REGIME_CHANGE_THRESH)
        | (abs_test > PRICE_CHANGE_MULT * hist_abs)
    )

    # シグナル生成
    trade_returns = []
    signals = []
    # ドローダウンベース停止: 累積損が10%を超えたら回復まで取引停止
    DD_STOP_THRESH = 0.10
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    dd_stopped = False

    for i in range(len(test_idx)):
        ts = df.index[test_idx[i]]
        ts_str = ts.strftime("%Y-%m-%d %H:%M")

        # ドローダウン停止チェック
        current_dd = peak_pnl - cumulative_pnl
        if dd_stopped:
            if cumulative_pnl >= peak_pnl * 0.5:  # ピークの50%まで回復したら再開
                dd_stopped = False
            else:
                signals.append({"ts": ts_str, "mode": "停止", "reason": "DD超過"})
                continue
        elif current_dd > DD_STOP_THRESH:
            dd_stopped = True
            signals.append({"ts": ts_str, "mode": "停止", "reason": "DD超過"})
            continue

        # 経済指標待機
        if wait_econ_test[i]:
            signals.append({"ts": ts_str, "mode": "待機", "reason": "経済指標"})
            continue

        # 待機条件C
        if wait_c[i]:
            signals.append({"ts": ts_str, "mode": "待機", "reason": "待機条件C"})
            continue

        # アンサンブル一致度フィルター
        if agreement[i] < MIN_AGREEMENT:
            signals.append({"ts": ts_str, "mode": "見送り", "reason": f"一致不足({agreement[i]}/5)"})
            continue

        # 自信度フィルター (per-window最適化済み)
        confidence = max(proba[i], 1.0 - proba[i])
        if confidence < best_thresh:
            signals.append({"ts": ts_str, "mode": "見送り", "reason": "自信不足",
                            "confidence": round(float(confidence), 4)})
            continue

        # トレード実行
        direction = int(preds[i])
        direction_mult = 1.0 if direction == 1 else -1.0
        ret = ret4_test[i] * direction_mult
        trade_returns.append(ret)
        cumulative_pnl += ret
        peak_pnl = max(peak_pnl, cumulative_pnl)
        signals.append({
            "ts": ts_str, "mode": "ENS",
            "direction": "買い" if direction == 1 else "売り",
            "return": float(ret),
            "agreement": int(agreement[i]),
            "confidence": round(float(confidence), 4),
        })

    return trade_returns, signals, best_thresh, val_pf


# ===== Walk-Forward 検証 =====
print(f"\n{'='*60}")
print("Walk-Forward 検証開始 (v3.1: MTF + VolRegime + Calendar + WeightedEnsemble)")
print(f"最小学習期間: {WF_MIN_TRAIN}本 ({WF_MIN_TRAIN/720:.1f}ヶ月)")
print(f"テスト期間: {WF_TEST_SIZE}本 ({WF_TEST_SIZE/720:.1f}ヶ月)")
print(f"アンサンブル: LGB+XGB+CatBoost+RF+ET (5モデル)")
print(f"最小一致: {MIN_AGREEMENT}/5")
print(f"信頼度候補: {CONFIDENCE_CANDIDATES}")
print(f"{'='*60}")

splits = walk_forward_splits(n_total, WF_MIN_TRAIN, WF_TEST_SIZE)
print(f"ウィンドウ数: {len(splits)}")

all_trade_returns = []
all_signals = []
window_metrics = []
window_thresholds = []

for w_idx, (train_idx, test_idx) in enumerate(splits):
    train_start = df.index[train_idx[0]].strftime("%Y-%m-%d")
    train_end = df.index[train_idx[-1]].strftime("%Y-%m-%d")
    test_start = df.index[test_idx[0]].strftime("%Y-%m-%d")
    test_end = df.index[test_idx[-1]].strftime("%Y-%m-%d")

    print(f"\n--- Window {w_idx+1}/{len(splits)} ---")
    print(f"  学習: {train_start} ~ {train_end} ({len(train_idx)}本)")
    print(f"  テスト: {test_start} ~ {test_end} ({len(test_idx)}本)")

    returns, signals, best_thresh, val_pf = run_window(train_idx, test_idx)
    all_trade_returns.extend(returns)
    all_signals.extend(signals)
    window_thresholds.append(best_thresh)

    print(f"  最適信頼度閾値: {best_thresh} (val PF: {val_pf:.2f})")

    if returns:
        m = compute_metrics(np.array(returns))
        window_metrics.append(m)
        print(f"  トレード数: {m['n_trades']}, PF: {m['pf']:.2f}, "
              f"勝率: {m['win_rate']:.1f}%, Sharpe: {m['sharpe']:.2f}")
    else:
        print("  トレードなし")

# ===== 全体の評価 =====
print(f"\n{'='*60}")
print("全体評価（Walk-Forward 全ウィンドウ合算）v3.1")
print(f"{'='*60}")

if all_trade_returns:
    overall = compute_metrics(np.array(all_trade_returns))
    print_metrics(overall, "Walk-Forward 全体")

    # 各ウィンドウの安定性
    print("\n【ウィンドウ別 PF/Sharpe の安定性】")
    pf_list = [m["pf"] for m in window_metrics if not np.isnan(m["pf"])]
    sharpe_list = [m["sharpe"] for m in window_metrics if not np.isnan(m["sharpe"])]
    if pf_list:
        print(f"  PF: min={min(pf_list):.2f}, max={max(pf_list):.2f}, "
              f"mean={np.mean(pf_list):.2f}, std={np.std(pf_list):.2f}")
        pf_above_1 = sum(1 for p in pf_list if p >= 1.0)
        print(f"  PF >= 1.0 のウィンドウ: {pf_above_1}/{len(pf_list)}")
    if sharpe_list:
        print(f"  Sharpe: min={min(sharpe_list):.2f}, max={max(sharpe_list):.2f}, "
              f"mean={np.mean(sharpe_list):.2f}, std={np.std(sharpe_list):.2f}")

    print(f"\n【各ウィンドウの最適信頼度閾値】")
    for i, th in enumerate(window_thresholds):
        print(f"  Window {i+1}: {th}")

    # 本番移行判断
    cond_pf = overall["pf"] >= 1.3
    cond_mdd = overall["mdd"] <= 20.0
    cond_sharpe = overall["sharpe"] >= 1.0
    cond_trades = overall["n_trades"] >= 100
    deploy_ok = cond_pf and cond_mdd and cond_sharpe and cond_trades
    print("\n【本番移行判断】")
    print("  PF >= 1.3: {} ({:.2f})".format("OK" if cond_pf else "NG", overall["pf"]))
    print("  MDD <= 20%: {} ({:.2f}%)".format("OK" if cond_mdd else "NG", overall["mdd"]))
    print("  Sharpe >= 1.0: {} ({:.2f})".format("OK" if cond_sharpe else "NG", overall["sharpe"]))
    print("  Trade >= 100: {} ({})".format("OK" if cond_trades else "NG", overall["n_trades"]))
    print("  -> {}".format("達成" if deploy_ok else "未達"))
else:
    overall = {}
    print("トレードが発生しませんでした。")

# ===== ダッシュボード用状態保存 =====
state_dir = (script_dir.parent / "data").resolve()
state_dir.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "df": df,
    "signals": all_signals,
    "all_trade_returns": all_trade_returns,
    "window_metrics": window_metrics,
    "overall_metrics": overall,
    "n_total": n_total,
    "splits": [(list(t), list(te)) for t, te in splits],
    "version": "3.1",
    "window_thresholds": window_thresholds,
}, state_dir / "dashboard_state.joblib")
print("\nダッシュボード用状態を保存しました。")
