# ===========================================
# paper_trade.py
# ペーパートレード（デモトレード）スクリプト
#
# 2つのモードで動作:
# 1. CSV記録モード（デフォルト）: OANDAなしで動作。予測結果をCSVに記録
# 2. OANDAモード: デモ口座で実際に注文を出す（要APIキー設定）
#
# 使い方:
#   python research/paper_trade.py          # 1回だけ予測して記録
#   python research/paper_trade.py --loop   # 1時間ごとに自動で繰り返し
# ===========================================

import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, add_multi_timeframe_features, FEATURE_COLS
from research.common.ensemble import EnsembleClassifier
from research.common.risk_manager import RiskManager
from research.common.stop_loss import StopLossManager
from research.common.economic_calendar import is_safe_to_trade
from research.telegram_bot import send_signal_sync

script_dir = Path(__file__).resolve().parent
DATA_DIR = (script_dir / ".." / "data").resolve()
LOG_DIR = DATA_DIR / "paper_trade_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# モデルパラメータ（best_params.json から読み込み）
PARAMS_PATH = DATA_DIR / "best_params.json"

# 固定パラメータ（フォールバック）
DEFAULT_PARAMS = {
    "triple_barrier": {"barrier_up": 0.005, "barrier_down": -0.003, "barrier_t": 24},
    "wait_mode": {"vol_mult": 2.0, "regime_change_thresh": 2, "price_change_mult": 3.0},
    "meta_labeling": {"adoption_target": 0.4},
    # Optuna v2最適化 v3.4: PF 1.011→1.187 (+17.4%), max_depth 6→4で汎化性能向上
    "lgbm": {"n_estimators": 300, "learning_rate": 0.062, "max_depth": 4, "min_child_samples": 10},
}

# v3.3 optimized: SL 2.5×ATR / TP 1.5×ATR, 時間帯フィルター(UTC20,23除外) from WF analysis
# v3.3 pruned: Removed Regime/Regime_changed, MA_5/MA_75, BB_upper/BB_lower, MACD_signal
# v3.2 Walk-Forward検証: PF=1.56→1.07(raw)+filter, Sharpe=13.79, Win=55.5%
# Hyperparamグリッドサーチ結果: n_est=500, lr=0.03がベスト (PF+4%)
CONFIDENCE_THRESHOLD = 0.60  # v3: lower threshold OK with ensemble + interaction features
FORECAST_HORIZON = 12  # v3: 12h instead of 4h
MIN_AGREEMENT = 4  # v3: 4/5 instead of 5/5

# ペーパートレード用の仮想口座残高（円）
INITIAL_BALANCE = 1_000_000  # 100万円

# 複数通貨ペア設定
PAIRS = ["USDJPY", "AUDJPY"]

# リスク管理・SL/TP管理の初期化
RISK_STATE_PATH = LOG_DIR / "risk_state.json"


def _init_risk_manager():
    """リスクマネージャーを初期化（状態があれば復元）"""
    if RISK_STATE_PATH.exists():
        try:
            rm = RiskManager.load_state(RISK_STATE_PATH)
            print(f"リスク状態復元: 残高={rm.account_balance:.0f}, 連敗={rm.losing_streak}")
            return rm
        except Exception as e:
            print(f"リスク状態復元失敗（新規作成）: {e}")
    return RiskManager(account_balance=INITIAL_BALANCE)


# グローバルインスタンス
risk_manager = _init_risk_manager()
sl_manager = StopLossManager(
    atr_period=14,
    sl_multiplier=2.5,   # WF分析最適化: 1.5→2.5 ATR (ノイズSL回避)
    tp_multiplier=1.5,    # WF分析最適化: 2.0→1.5 ATR (利確を早める)
    max_hold_hours=24,
)


def load_params():
    """最適化済みパラメータを読み込む"""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"パラメータ読み込み: {PARAMS_PATH}")
        return params
    print("best_params.json が見つかりません。デフォルトパラメータを使用")
    return DEFAULT_PARAMS


def update_data():
    """01_data_fetch.py を実行してデータを最新に更新する"""
    fetch_script = script_dir / "01_data_fetch.py"
    if not fetch_script.exists():
        print("警告: 01_data_fetch.py が見つかりません。既存データを使用します")
        return
    print("データを最新に更新中...")
    try:
        subprocess.run(
            [sys.executable, str(fetch_script)],
            cwd=str(script_dir.parent),
            timeout=120,
            capture_output=True,
        )
        print("データ更新完了")
    except Exception as e:
        print(f"データ更新失敗（既存データを使用）: {e}")


def load_audjpy_1h():
    """AUD/JPY 1時間足データを読み込む（audjpy_1h.csvはヘッダー付きCSV）"""
    path = DATA_DIR / "audjpy_1h.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def _resample_ohlcv(df, rule):
    """Resample OHLCV data to a lower frequency."""
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
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bb_width(close, period=20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std) / sma.replace(0, np.nan)


# add_multi_timeframe_features は research.common.features からインポート (v3.4)


def add_volatility_regime_features(df):
    """Add realized vol percentile and vol-of-vol (no lookahead)."""
    vol = df["Volatility_24"]
    df["Vol_percentile"] = vol.rolling(720, min_periods=72).apply(
        lambda x: (x[-1] >= x).sum() / len(x), raw=True
    )
    df["Vol_of_vol"] = vol.rolling(120, min_periods=24).std()
    return df


def add_calendar_awareness_features(df):
    """Add hour x dow interaction and session indicators."""
    h = df.index.hour
    dow = df.index.dayofweek
    df["Hour_x_DoW"] = h * 10 + dow
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    df["Session_overlap"] = ((h >= 13) & (h < 16)).astype(int)
    return df


def compute_model_weights(ensemble, X_val, y_val):
    """Compute per-model weights based on Sharpe ratio (optimizer result)."""
    import math
    weights = []
    for model in ensemble.models:
        proba = model.predict_proba(X_val)[:, 1]
        direction = np.where(proba > 0.5, 1.0, -1.0)
        # Approximate trade returns using label as proxy
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
    """Weighted prediction using per-model weights."""
    probas = np.array([m.predict_proba(X)[:, 1] for m in ensemble.models])
    weighted_proba = (probas * weights[:, None]).sum(axis=0)
    preds = (weighted_proba >= 0.5).astype(int)
    individual_preds = np.array([m.predict(X) for m in ensemble.models])
    vote_sum = individual_preds.sum(axis=0)
    agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
    return preds, agreement, weighted_proba


def prepare_data(pair="USDJPY"):
    """最新データを読み込んでv3.1特徴量を生成する（ペア指定可）"""
    print(f"[{pair}] データ読み込み中...")
    if pair == "AUDJPY":
        df = load_audjpy_1h()
    else:
        df = load_usdjpy_1h()
    df = add_technical_features(df)
    # 金利・日足トレンド特徴量（USD/JPY固有だがtry/exceptで0フォールバック）
    df = add_rate_features(df)
    df = add_daily_trend_features(df)

    # v3.1: Multi-timeframe features
    df = add_multi_timeframe_features(df)

    # v3: 交互作用特徴量
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

    # v3.1: Volatility regime features
    df = add_volatility_regime_features(df)

    # v3.1: Calendar awareness features
    df = add_calendar_awareness_features(df)

    # v3.1 feature columns
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
    return df, feature_cols


def train_models(df, feature_cols, params, pair="USDJPY"):
    """直近データで5モデルアンサンブルを学習する（v3.1: 性能ベース重み付け）"""
    n_est = params["lgbm"]["n_estimators"]
    lr = params["lgbm"]["learning_rate"]

    # 直近のデータを学習に使う（最新FORECAST_HORIZON本は未確定なので除く）
    train_df = df.iloc[:-FORECAST_HORIZON].copy()

    # v3: 12h方向ラベル
    train_df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON).loc[train_df.index]
    train_df["Label"] = (train_df["Close_Nh_later"] > train_df["Close"]).astype(int)
    train_df = train_df.dropna(subset=["Label"])

    # v3.1: Split into train/val for performance-based weighting
    val_size = min(500, len(train_df) // 5)
    X_train = train_df[feature_cols].iloc[:-val_size]
    y_train = train_df["Label"].iloc[:-val_size]
    X_val = train_df[feature_cols].iloc[-val_size:]
    y_val = train_df["Label"].iloc[-val_size:]

    # 5モデルアンサンブルを学習
    ensemble = EnsembleClassifier(n_estimators=n_est, learning_rate=lr)
    ensemble.fit(X_train, y_train)

    # v3.1: Performance-based ensemble weighting
    weights = compute_model_weights(ensemble, X_val, y_val)

    # 学習期間の統計値（待機条件に使う）
    hist_vol = train_df["Volatility_24"].mean()

    models = {
        "ensemble": ensemble,
        "hist_vol": hist_vol,
        "weights": weights,
    }

    print(f"[{pair}] 学習完了（v3.1アンサンブル, {FORECAST_HORIZON}h horizon）: {len(X_train)}本学習, {val_size}本検証")
    print(f"  モデル重み: {[f'{w:.3f}' for w in weights]}")
    return models


def predict_latest(df, feature_cols, models, params, pair="USDJPY"):
    """最新の1本に対して5モデルアンサンブルで予測を行う"""
    latest = df.iloc[-1]
    X_latest = df[feature_cols].iloc[[-1]]
    ts = df.index[-1]

    result = {
        "pair": pair,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "close": float(latest["Close"]),
    }

    # 時間帯フィルター: JST 05:00 (UTC 20) / JST 08:00 (UTC 23) はPF<1.0のためスキップ
    # WF分析最適化 v3.3
    utc_hour = ts.hour if ts.tzinfo is None else ts.tz_convert("UTC").hour
    if utc_hour in (20, 23):
        result["action"] = "SKIP"
        result["reason"] = f"時間帯フィルター (UTC {utc_hour}:00 = JST {(utc_hour+9)%24}:00, PF<1.0)"
        return result

    # 曜日フィルター: 火曜日はPF<1.0のためスキップ
    # WF分析最適化 v3.4: USDJPY PF 1.55→1.84, AUDJPY PF 1.58→1.68
    if ts.dayofweek == 1:  # 0=月, 1=火, 2=水, 3=木, 4=金
        result["action"] = "SKIP"
        result["reason"] = "曜日フィルター (火曜日, PF<1.0)"
        return result

    # ボラティリティフィルター: 低ボラ(<20pct)と極端高ボラ(>90pct)をスキップ
    # WF分析最適化 v3.4: PF 1.58→2.07 (+31%), Sharpe 15.17→25.16
    if "Volatility_20" in df.columns:
        vol_series = df["Volatility_20"].dropna()
        if len(vol_series) > 120:
            current_vol = vol_series.iloc[-1]
            vol_pct = (vol_series.iloc[-120:] < current_vol).mean()
            if vol_pct < 0.20 or vol_pct > 0.90:
                result["action"] = "SKIP"
                result["reason"] = f"ボラフィルター (vol_pct={vol_pct:.2f}, 範囲外)"
                return result

    # 経済カレンダーチェック（高インパクトイベント前後は見送り）
    safe, event_name = is_safe_to_trade(datetime.now())
    if not safe:
        result["action"] = "SKIP"
        result["reason"] = f"経済イベント回避: {event_name}"
        return result

    # リスク管理チェック
    can, risk_reason = risk_manager.can_trade(pair=pair, direction="long")
    if not can:
        result["action"] = "SKIP"
        result["reason"] = f"リスク管理: {risk_reason}"
        return result

    # 待機条件（ボラティリティ異常）
    vol = latest["Volatility_24"] if "Volatility_24" in latest.index else 0
    vol_mult = params["wait_mode"]["vol_mult"]
    if vol > vol_mult * models["hist_vol"]:
        result["action"] = "SKIP"
        result["reason"] = f"ボラティリティ異常 ({vol:.6f} > {vol_mult * models['hist_vol']:.6f})"
        return result

    # v3.1: 性能ベース重み付きアンサンブル予測
    ensemble = models["ensemble"]
    weights = models.get("weights")
    if weights is not None:
        pred, agreement, weighted_proba = weighted_predict(ensemble, X_latest, weights)
        pred = pred[0]
        agreement = int(agreement[0])
        proba = np.array([1 - weighted_proba[0], weighted_proba[0]])
    else:
        proba = ensemble.predict_proba(X_latest)[0]
        pred, agreement = ensemble.predict_with_agreement(X_latest)
        pred = pred[0]
        agreement = int(agreement[0])

    # トレンドフィルター (soft): 逆トレンド時は信頼度要求を+5%上げる
    # WF分析最適化 v3.4: PF 1.58→1.65, トレード数維持
    trend_penalty = 0.0
    if "SMA_20" in df.columns:
        sma20 = df["SMA_20"].iloc[-1]
        current_price = float(latest["Close"])
        predicted_dir = 1 if proba[1] > 0.5 else 0
        price_above_sma = current_price > sma20
        if (predicted_dir == 1 and not price_above_sma) or (predicted_dir == 0 and price_above_sma):
            trend_penalty = 0.05  # 逆トレンド時は+5%要求

    # v3.5: ドローダウンベース動的閾値
    # 損失が深いほど信頼度閾値を引き上げてトレードを絞る（WF実験: PF 1.19→1.57, MDD -36%→-20%）
    dd_penalty = 0.0
    if risk_manager is not None:
        current_dd = risk_manager._current_drawdown()
        dd_penalty = min(current_dd * 2.0, 0.15)  # DDの2倍を加算、最大+15%

    # 自信度フィルター
    confidence = max(proba[1], 1.0 - proba[1])
    effective_threshold = CONFIDENCE_THRESHOLD + trend_penalty + dd_penalty
    if confidence < effective_threshold:
        result["action"] = "SKIP"
        dd_str = f" +DD{dd_penalty:.0%}" if dd_penalty > 0 else ""
        result["reason"] = f"自信不足 ({confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}{dd_str})"
        result["confidence"] = float(confidence)
        result["agreement"] = agreement
        return result

    # v3: MIN_AGREEMENT人以上一致
    if agreement < MIN_AGREEMENT:
        result["action"] = "SKIP"
        result["reason"] = f"一致不足 ({agreement}/5人、{MIN_AGREEMENT}人以上が必要)"
        result["confidence"] = float(confidence)
        result["agreement"] = agreement
        return result

    direction = "BUY" if pred == 1 else "SELL"

    # ATRベースのSL/TP計算
    atr_series = sl_manager.calculate_atr(df)
    current_atr = float(atr_series.iloc[-1])
    levels = sl_manager.get_levels(direction, float(latest["Close"]), current_atr)

    result["action"] = direction
    result["mode"] = f"アンサンブル({agreement}/5人一致)"
    result["confidence"] = float(confidence)
    result["proba_up"] = float(proba[1])
    result["proba_down"] = float(proba[0])
    result["agreement"] = agreement
    result["stop_loss"] = levels["stop_loss"]
    result["take_profit"] = levels["take_profit"]
    result["atr"] = current_atr
    result["risk_reward"] = levels["risk_reward_ratio"]
    return result


def log_prediction(result):
    """予測結果をCSVに記録する"""
    log_file = LOG_DIR / "predictions.csv"

    # 旧CSVフォーマットの場合は自動移行（check_past_predictionsが先に実行されるが念のため）
    _migrate_csv_if_needed(log_file)

    # 固定カラム順で記録（追記時にカラムがズレないようにする）
    columns = [
        "logged_at", "pair", "timestamp", "close", "action",
        "mode", "reason", "confidence", "agreement", "proba_up", "proba_down",
        "stop_loss", "take_profit", "atr",
        "exit_price", "exit_reason", "actual_return", "net_return", "result",
    ]
    row = {"logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    for col in columns:
        if col == "logged_at":
            continue
        row[col] = result.get(col, "")

    df_row = pd.DataFrame([row], columns=columns)
    if log_file.exists():
        df_row.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_file, index=False)

    print(f"記録完了: {log_file}")


def check_past_predictions(df, pair="USDJPY"):
    """過去の予測結果を実際の値動きと照合する（ペア別）

    ATR-based SL/TP: 保有期間中にSL/TPに到達したか確認し、到達した場合は
    そのバーの終値で決済。到達しなかった場合はFORECAST_HORIZON時間後に決済。
    """
    log_file = LOG_DIR / "predictions.csv"
    if not log_file.exists():
        return

    # 旧CSVフォーマットの場合は自動移行
    _migrate_csv_if_needed(log_file)

    logs = pd.read_csv(log_file)
    # actionがBUYまたはSELLのもので、まだ結果が記録されていないものを探す
    trades = logs[logs["action"].isin(["BUY", "SELL"])].copy()
    # pairカラムがある場合は該当ペアのみフィルター（旧データ互換: pairなし→USDJPY扱い）
    if "pair" in trades.columns:
        pair_mask = trades["pair"].fillna("USDJPY") == pair
        trades = trades[pair_mask]
    elif pair != "USDJPY":
        return  # 旧データにはAUDJPYの記録がない

    if "actual_return" in trades.columns:
        unchecked = trades[trades["actual_return"].isna()]
    else:
        unchecked = trades

    if unchecked.empty:
        return

    updated = False
    for idx, row in unchecked.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        # 保有期間のバーを取得（エントリーの次のバー〜FORECAST_HORIZON時間後まで）
        target_ts = ts + timedelta(hours=FORECAST_HORIZON)

        # エントリーの次のバーから最大保持期間までの範囲を取得
        holding_bars = df.loc[(df.index > ts) & (df.index <= target_ts)]
        if holding_bars.empty:
            continue  # まだデータが足りない

        entry_price = float(row["close"])
        is_buy = row["action"] == "BUY"

        # SL/TP レベルを取得（CSVに記録されている場合はそれを使う、なければフォールバック）
        sl_price = row.get("stop_loss", np.nan)
        tp_price = row.get("take_profit", np.nan)

        # SL/TP が NaN の場合（旧データ）は現在のATRから再計算を試みる
        if pd.isna(sl_price) or pd.isna(tp_price):
            # エントリー時点のATRを計算して推定
            entry_idx = df.index.get_indexer([ts], method="nearest")[0]
            if entry_idx >= 14:
                atr_series = sl_manager.calculate_atr(df.iloc[:entry_idx + 1])
                current_atr = float(atr_series.iloc[-1])
                direction_str = "BUY" if is_buy else "SELL"
                levels = sl_manager.get_levels(direction_str, entry_price, current_atr)
                sl_price = levels["stop_loss"]
                tp_price = levels["take_profit"]
            else:
                sl_price = np.nan
                tp_price = np.nan

        # SL/TPの有無に応じて決済ロジックを分岐
        exit_price = None
        exit_reason = "timeout"  # デフォルト: FORECAST_HORIZON到達

        if not pd.isna(sl_price) and not pd.isna(tp_price):
            # 保有期間中にSL/TPヒットを確認
            for bar_ts, bar in holding_bars.iterrows():
                if is_buy:
                    # BUY: 安値がSL以下でSLヒット、高値がTP以上でTPヒット
                    if bar["Low"] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        break
                    if bar["High"] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        break
                else:
                    # SELL: 高値がSL以上でSLヒット、安値がTP以下でTPヒット
                    if bar["High"] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        break
                    if bar["Low"] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        break

        # SL/TPに到達しなかった場合はFORECAST_HORIZON時間後の終値で決済
        if exit_price is None:
            if target_ts in df.index:
                exit_price = float(df.loc[target_ts, "Close"])
                exit_reason = "timeout"
            elif len(holding_bars) >= FORECAST_HORIZON:
                # 完全一致のtargetバーがなくても保有期間分のバーが揃っていれば最終バーで決済
                exit_price = float(holding_bars.iloc[-1]["Close"])
                exit_reason = "timeout"
            else:
                continue  # データ不足、次回に持ち越し

        raw_return = (exit_price - entry_price) / entry_price
        actual_return = raw_return if is_buy else -raw_return

        # 手数料（スプレッド）を引く
        spread = 0.0005 if pair == "AUDJPY" else 0.0003  # AUDJPYはスプレッド広め
        net_return = actual_return - spread

        logs.loc[idx, "exit_price"] = exit_price
        logs.loc[idx, "exit_reason"] = exit_reason
        logs.loc[idx, "actual_return"] = actual_return
        logs.loc[idx, "net_return"] = net_return
        logs.loc[idx, "result"] = "WIN" if net_return > 0 else "LOSE"
        updated = True

        direction_label = "BUY" if is_buy else "SELL"
        print(f"  [{pair}] {direction_label} @ {entry_price:.3f} → {exit_price:.3f} "
              f"({exit_reason}) = {net_return:+.6f} {'WIN' if net_return > 0 else 'LOSE'}")

        # リスクマネージャーにトレード結果を記録
        pnl_amount = net_return * risk_manager.account_balance
        direction_str = "long" if is_buy else "short"
        risk_manager.record_trade(pair, direction_str, entry_price, exit_price, pnl_amount)

    if updated:
        logs.to_csv(log_file, index=False)
        risk_manager.save_state(RISK_STATE_PATH)
        # 成績サマリーを表示（ペア別）
        completed = logs[logs["result"].isin(["WIN", "LOSE"])]
        if "pair" in completed.columns:
            pair_completed = completed[completed["pair"].fillna("USDJPY") == pair]
        else:
            pair_completed = completed if pair == "USDJPY" else pd.DataFrame()
        if len(pair_completed) > 0:
            wins = (pair_completed["result"] == "WIN").sum()
            total = len(pair_completed)
            net_returns = pair_completed["net_return"].astype(float)
            cum_return = net_returns.sum()
            print(f"\n【ペーパートレード成績 {pair}】")
            print(f"  完了トレード: {total}回")
            print(f"  勝率: {wins}/{total} ({wins/total*100:.1f}%)")
            print(f"  累積リターン: {cum_return:+.6f}")


def _migrate_csv_if_needed(log_file):
    """CSVのカラム形式を最新に移行する（旧14列→新18列、pair列欠損の修正など）"""
    NEW_COLUMNS = [
        "logged_at", "pair", "timestamp", "close", "action",
        "mode", "reason", "confidence", "agreement", "proba_up", "proba_down",
        "stop_loss", "take_profit", "atr",
        "exit_price", "exit_reason", "actual_return", "net_return", "result",
    ]
    if not log_file.exists():
        return

    logs = pd.read_csv(log_file)
    current_cols = list(logs.columns)

    # 旧フォーマット（14列、pair列なし）の検出
    old_columns_no_pair = [
        "logged_at", "timestamp", "close", "action",
        "mode", "reason", "confidence", "agreement", "proba_up", "proba_down",
        "exit_price", "actual_return", "net_return", "result",
    ]

    if current_cols == old_columns_no_pair:
        print("CSV移行: 旧フォーマット(14列)を新フォーマット(19列)に変換中...")

        # 行ごとにフィールド数を確認して修正（pair列が途中から追加された場合の対応）
        # 14列ヘッダーに15列データが混在している可能性がある
        raw_lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        header_fields = len(raw_lines[0].split(","))
        fixed_rows = []
        for i, line in enumerate(raw_lines):
            if i == 0:
                continue  # ヘッダーはスキップ
            fields = line.split(",")
            if len(fields) == header_fields + 1 and fields[1] in PAIRS:
                # 15フィールド行: pair列が挿入されている → そのまま使う
                row_dict = dict(zip(["logged_at", "pair"] + old_columns_no_pair[1:], fields))
                fixed_rows.append(row_dict)
            elif len(fields) == header_fields:
                # 14フィールド行: pair列なし → USDJPY補完
                row_dict = dict(zip(old_columns_no_pair, fields))
                row_dict["pair"] = "USDJPY"
                fixed_rows.append(row_dict)
            else:
                # その他のフィールド数 → best effort
                row_dict = dict(zip(old_columns_no_pair, fields[:header_fields]))
                row_dict["pair"] = "USDJPY"
                fixed_rows.append(row_dict)

        logs = pd.DataFrame(fixed_rows)
        # SL/TP/ATR列を追加
        for col in ["stop_loss", "take_profit", "atr"]:
            if col not in logs.columns:
                logs[col] = np.nan
        # exit_reason列を追加
        if "exit_reason" not in logs.columns:
            # 既に結果があるものはtimeoutと推定
            logs["exit_reason"] = ""
            logs.loc[logs["result"].isin(["WIN", "LOSE"]), "exit_reason"] = "timeout"
        # カラム順を合わせる
        for col in NEW_COLUMNS:
            if col not in logs.columns:
                logs[col] = ""
        logs = logs[NEW_COLUMNS]
        logs.to_csv(log_file, index=False)
        print(f"CSV移行完了: {len(logs)}行を新フォーマットに変換")
    elif set(NEW_COLUMNS) - set(current_cols):
        # 一部カラムが欠けている場合は追加
        missing = set(NEW_COLUMNS) - set(current_cols)
        print(f"CSV移行: 欠損カラム {missing} を追加中...")
        for col in missing:
            logs[col] = "" if col in ("pair", "exit_reason") else np.nan
        if "pair" in missing:
            logs["pair"] = logs["pair"].replace("", "USDJPY")
        logs = logs.reindex(columns=NEW_COLUMNS, fill_value="")
        logs.to_csv(log_file, index=False)
        print(f"CSV移行完了")


def _build_reasons(result):
    """Telegram通知用の理由リストを生成する"""
    return [
        f"{result.get('agreement', '?')}/5モデル一致",
        f"自信度 {result.get('confidence', 0):.1%}",
        f"上昇確率: {result.get('proba_up', 0):.1%}",
        f"予測期間: {FORECAST_HORIZON}h",
    ]


def _pair_display_name(pair):
    """ペアコードを表示用名称に変換する（例: USDJPY → USD/JPY）"""
    return f"{pair[:3]}/{pair[3:]}"


def _run_pair(pair, params, skip_update=False):
    """1つのペアに対して予測を実行する"""
    df, feature_cols = prepare_data(pair=pair)
    models = train_models(df, feature_cols, params, pair=pair)

    # 過去の予測結果を照合
    check_past_predictions(df, pair=pair)

    # 最新の予測
    result = predict_latest(df, feature_cols, models, params, pair=pair)

    # 結果を表示
    display_name = _pair_display_name(pair)
    print(f"\n{'='*50}")
    print(f"【予測結果 {display_name}】{result['timestamp']}")
    print(f"  現在レート: {result['close']:.3f}")
    print(f"  予測期間: {FORECAST_HORIZON}h先")

    if result["action"] == "SKIP":
        print(f"  判断: 見送り")
        print(f"  理由: {result['reason']}")
    else:
        print(f"  判断: {result['action']}（{'買い' if result['action'] == 'BUY' else '売り'}）")
        print(f"  モデル: {result.get('mode', '?')}")
        print(f"  一致人数: {result.get('agreement', '?')}/5人")
        print(f"  自信度: {result['confidence']:.1%}")
        print(f"  上昇確率: {result.get('proba_up', 0):.1%}")
        if "stop_loss" in result:
            print(f"  SL: {result['stop_loss']:.3f} / TP: {result['take_profit']:.3f}")
            print(f"  ATR: {result.get('atr', 0):.5f} / R:R = 1:{result.get('risk_reward', 0):.2f}")
    print(f"{'='*50}")

    # CSV記録
    log_prediction(result)

    # Telegram通知 + リスク管理登録
    if result["action"] in ("BUY", "SELL"):
        # リスクマネージャーにポジション登録
        sl_price = result.get("stop_loss", result["close"])
        lot = risk_manager.calculate_position_size(pair, result["close"], sl_price)
        direction_str = "long" if result["action"] == "BUY" else "short"
        risk_manager.open_position(pair, direction_str, result["close"], sl_price, lot)
        risk_manager.save_state(RISK_STATE_PATH)
        print(f"[{pair}] ポジション登録: {lot:.2f}ロット (SL={sl_price:.3f})")

        try:
            reasons = _build_reasons(result)
            if "stop_loss" in result:
                reasons.append(f"SL: {result['stop_loss']:.3f} / TP: {result['take_profit']:.3f}")
            signal = {
                "pair": display_name,
                "action": result["action"],
                "price": result["close"],
                "confidence": result.get("confidence", 0),
                "agreement": result.get("agreement", 0),
                "reasons": reasons,
                "category": "FX",
            }
            send_signal_sync(signal)
            print(f"[{pair}] Telegram通知送信完了")
        except Exception as e:
            print(f"[{pair}] Telegram通知失敗: {e}")

    return result


def run_once(skip_update=False):
    """1回だけ予測を実行する（全ペア）"""
    if not skip_update:
        update_data()
    params = load_params()

    results = {}
    for pair in PAIRS:
        print(f"\n{'#'*50}")
        print(f"# {_pair_display_name(pair)} の予測")
        print(f"{'#'*50}")
        try:
            results[pair] = _run_pair(pair, params, skip_update=True)
        except Exception as e:
            print(f"[{pair}] エラー: {e}")
            results[pair] = {"pair": pair, "action": "ERROR", "reason": str(e)}

    # リスク管理状態を表示（全ペア共通）
    status = risk_manager.get_status()
    print(f"\n【リスク状態（全体）】残高={status['account_balance']:.0f} "
          f"DD={status['drawdown_pct']:.1%} 連敗={status['losing_streak']} "
          f"ポジション={status['open_positions']}/{status['max_positions']}")

    return results


def run_loop(interval_minutes=60):
    """指定間隔で繰り返し実行する"""
    print(f"ペーパートレード開始（{interval_minutes}分間隔で自動実行）")
    print("Ctrl+C で停止\n")

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 実行中...")
            run_once()
        except Exception as e:
            print(f"エラー: {e}")

        next_run = datetime.now() + timedelta(minutes=interval_minutes)
        print(f"次回実行: {next_run.strftime('%H:%M:%S')}")

        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n停止しました")
            break


def run_backfill(n_days=30):
    """過去n日分のバックフィル（paper trade backtest）- 全ペア"""
    from research.common.validation import compute_metrics as calc_metrics

    params = load_params()
    for pair in PAIRS:
        print(f"\n{'#'*60}")
        print(f"# {_pair_display_name(pair)} バックフィル: 過去{n_days}日")
        print(f"{'#'*60}")
        try:
            _run_backfill_pair(pair, n_days, params, calc_metrics)
        except Exception as e:
            print(f"[{pair}] バックフィルエラー: {e}")


def _run_backfill_pair(pair, n_days, params, calc_metrics):
    """1ペアの過去n日分バックフィル"""
    df, feature_cols = prepare_data(pair=pair)

    # 12h方向ラベル（バックフィル用）
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]
    df = df.dropna(subset=["Return_Nh"])

    # 過去n_days日分のデータを使う
    hours = n_days * 24
    if len(df) < hours + 4320:
        print(f"データ不足: {len(df)}本（{hours + 4320}本必要）")
        return

    test_start = len(df) - hours
    train_df = df.iloc[:test_start]
    test_df = df.iloc[test_start:]

    # 学習
    X_train = train_df[feature_cols]
    y_train = (train_df["Close_Nh_later"] > train_df["Close"]).astype(int)

    ensemble = EnsembleClassifier(
        n_estimators=params["lgbm"]["n_estimators"],
        learning_rate=params["lgbm"]["learning_rate"],
    )
    ensemble.fit(X_train.values, y_train.values)
    hist_vol = train_df["Volatility_24"].mean()

    # テスト期間で予測
    X_test = test_df[feature_cols].values
    preds, agreement = ensemble.predict_with_agreement(X_test)
    proba = ensemble.predict_proba(X_test)[:, 1]
    confidence = np.maximum(proba, 1.0 - proba)
    ret_test = test_df["Return_Nh"].values

    # フィルター
    trade_mask = (confidence >= CONFIDENCE_THRESHOLD) & (agreement >= MIN_AGREEMENT)
    vol_test = test_df["Volatility_24"].values
    vol_mask = vol_test <= VOL_MULT * hist_vol if "VOL_MULT" in dir() else np.ones(len(test_df), dtype=bool)
    trade_mask &= vol_mask

    if trade_mask.sum() == 0:
        print("トレードなし")
        return

    # 結果計算
    direction = np.where(preds[trade_mask] == 1, 1.0, -1.0)
    returns = ret_test[trade_mask] * direction
    metrics = calc_metrics(returns)

    print(f"\n{'='*60}")
    print(f"ペーパートレード バックフィル結果 [{pair}]")
    print(f"{'='*60}")
    print(f"  期間: {test_df.index[0]} ~ {test_df.index[-1]}")
    print(f"  トレード数: {metrics['n_trades']}")
    print(f"  勝率: {metrics['win_rate']:.1f}%")
    print(f"  PF: {metrics['pf']:.2f}")
    print(f"  MDD: {metrics['mdd']:.2f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  手数料込み期待値: {metrics['exp_value_net']:+.6f}")
    print(f"  一致条件: {MIN_AGREEMENT}/5人以上")
    print(f"  信頼度閾値: {CONFIDENCE_THRESHOLD}")
    print(f"  予測期間: {FORECAST_HORIZON}h")
    print(f"{'='*60}")

    # Buy & Hold比較
    bh_ret = (test_df["Close"].iloc[-1] - test_df["Close"].iloc[0]) / test_df["Close"].iloc[0]
    strategy_ret = returns.sum()
    print(f"  戦略リターン: {strategy_ret*100:+.2f}%")
    print(f"  Buy&Hold: {bh_ret*100:+.2f}%")
    print(f"  Alpha: {(strategy_ret - bh_ret)*100:+.2f}%")


# VOL_MULT for backfill
VOL_MULT = 2.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FX AI ペーパートレード v3.3")
    parser.add_argument("--loop", action="store_true", help="自動繰り返しモード")
    parser.add_argument("--interval", type=int, default=60, help="繰り返し間隔（分）")
    parser.add_argument("--backfill", type=int, default=0, help="過去N日分のバックフィル")
    args = parser.parse_args()

    if args.backfill > 0:
        run_backfill(args.backfill)
    elif args.loop:
        run_loop(args.interval)
    else:
        run_once()
