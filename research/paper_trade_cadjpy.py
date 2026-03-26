# ===========================================
# paper_trade_cadjpy.py
# CAD/JPY ペーパートレード（デモトレード）スクリプト
#
# USD/JPY版をCAD/JPYに適応:
#   - yfinanceでCADJPY=Xデータ取得（2年分1時間足）
#   - 原油価格（CL=F）を追加特徴量として使用
#   - 閾値: conf>=0.75, agree>=3（multi_currency_pipeline結果）
#   - スプレッド: 0.0004（CAD/JPYはUSD/JPYより広い）
#
# 使い方:
#   python research/paper_trade_cadjpy.py          # 1回実行
#   python research/paper_trade_cadjpy.py --loop   # 自動ループ
# ===========================================

import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.features import add_technical_features, add_regime_features, FEATURE_COLS
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
    "lgbm": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
}

# multi_currency_pipeline検証で最適化された閾値
# CAD/JPY: conf>=0.75, agree>=3
CONFIDENCE_THRESHOLD = 0.75
AGREEMENT_THRESHOLD = 3

# スプレッドコスト（CAD/JPYはUSD/JPYより広い）
SPREAD_COST = 0.0004

# ペーパートレード用の仮想口座残高（円）
INITIAL_BALANCE = 1_000_000  # 100万円

# リスク管理（USD/JPYと共有のリスク状態ファイル）
RISK_STATE_PATH = LOG_DIR / "risk_state.json"

# 原油価格の追加特徴量カラム
OIL_FEATURE_COLS = [
    "Oil_Return_1", "Oil_Return_6", "Oil_Return_24", "Oil_Volatility",
]


def _init_risk_manager():
    """リスクマネージャーを初期化（USD/JPYと共有状態を復元）"""
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
sl_manager = StopLossManager(atr_period=14, sl_multiplier=1.5, tp_multiplier=2.0, max_hold_hours=24)


def load_params():
    """最適化済みパラメータを読み込む"""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"パラメータ読み込み: {PARAMS_PATH}")
        return params
    print("best_params.json が見つかりません。デフォルトパラメータを使用")
    return DEFAULT_PARAMS


def load_cadjpy_1h() -> pd.DataFrame:
    """yfinanceでCAD/JPY 1時間足データを取得する（直近2年分）"""
    print("CAD/JPYデータをyfinanceから取得中...")
    ticker = yf.Ticker("CADJPY=X")
    # yfinanceの1h足は最大730日まで取得可能
    df = ticker.history(period="2y", interval="1h")

    if df.empty:
        raise RuntimeError("CAD/JPYデータの取得に失敗しました")

    # タイムゾーンをUTCに統一してからnaive化
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # カラム名を統一
    df = df.rename(columns={
        "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"]).sort_index()
    print(f"CAD/JPYデータ取得完了: {len(df)}本 ({df.index.min()} ~ {df.index.max()})")
    return df


def load_oil_1h() -> pd.DataFrame:
    """yfinanceで原油（WTI）1時間足データを取得する"""
    print("原油データ（CL=F）をyfinanceから取得中...")
    ticker = yf.Ticker("CL=F")
    df = ticker.history(period="2y", interval="1h")

    if df.empty:
        print("警告: 原油データの取得に失敗しました。空のDataFrameを返します")
        return pd.DataFrame()

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df = df[["Close"]].rename(columns={"Close": "Oil_Close"}).copy()
    df = df.dropna().sort_index()
    print(f"原油データ取得完了: {len(df)}本")
    return df


def add_oil_features(df: pd.DataFrame, df_oil: pd.DataFrame) -> pd.DataFrame:
    """原油価格の特徴量を追加する"""
    if df_oil.empty:
        # データ取得失敗時はゼロ埋め
        for col in OIL_FEATURE_COLS:
            df[col] = 0.0
        return df

    # CAD/JPYのインデックスに合わせてリインデックス（前方補間）
    oil = df_oil["Oil_Close"].reindex(df.index, method="ffill")

    # 原油リターン（1時間、6時間、24時間）
    df["Oil_Return_1"] = oil.pct_change(1)
    df["Oil_Return_6"] = oil.pct_change(6)
    df["Oil_Return_24"] = oil.pct_change(24)

    # 原油ボラティリティ（24時間）
    df["Oil_Volatility"] = oil.pct_change(1).rolling(24).std()

    return df


def prepare_data():
    """CAD/JPYデータを読み込んで特徴量を生成する"""
    print("データ読み込み中...")

    # CAD/JPYの価格データ
    df = load_cadjpy_1h()

    # テクニカル指標 + レジーム特徴量
    df = add_technical_features(df)
    df = add_regime_features(df)

    # 原油データの取得と特徴量追加
    df_oil = load_oil_1h()
    df = add_oil_features(df, df_oil)

    # 特徴量カラム（基本 + 原油）
    feature_cols = FEATURE_COLS + OIL_FEATURE_COLS

    df = df.dropna(subset=feature_cols)
    return df, feature_cols


def train_models(df, feature_cols, params):
    """直近データで5モデルアンサンブルを学習する"""
    n_est = params["lgbm"]["n_estimators"]
    lr = params["lgbm"]["learning_rate"]

    # 直近のデータを学習に使う（最新4本は未確定なので除く）
    train_df = df.iloc[:-4].copy()

    # 方向ラベル（4時間後に上がったか）
    train_df["Close_4h_later"] = df["Close"].shift(-4).loc[train_df.index]
    train_df["Label"] = (train_df["Close_4h_later"] > train_df["Close"]).astype(int)
    train_df = train_df.dropna(subset=["Label"])
    X_train = train_df[feature_cols]
    y_train = train_df["Label"]

    # 5モデルアンサンブルを学習
    ensemble = EnsembleClassifier(n_estimators=n_est, learning_rate=lr)
    ensemble.fit(X_train, y_train)

    # 学習期間の統計値（待機条件に使う）
    hist_vol = train_df["Volatility_24"].mean()

    models = {
        "ensemble": ensemble,
        "hist_vol": hist_vol,
    }

    print(f"学習完了（5モデルアンサンブル）: {len(X_train)}本使用 ({train_df.index.min()} ~ {train_df.index.max()})")
    return models


def predict_latest(df, feature_cols, models, params):
    """最新の1本に対して5モデルアンサンブルで予測を行う"""
    latest = df.iloc[-1]
    X_latest = df[feature_cols].iloc[[-1]]
    ts = df.index[-1]

    result = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "close": float(latest["Close"]),
        "regime": int(latest["Regime"]),
    }

    # 経済カレンダーチェック（高インパクトイベント前後は見送り）
    safe, event_name = is_safe_to_trade(datetime.now())
    if not safe:
        result["action"] = "SKIP"
        result["reason"] = f"経済イベント回避: {event_name}"
        return result

    # リスク管理チェック
    can, risk_reason = risk_manager.can_trade(pair="CADJPY", direction="long")
    if not can:
        result["action"] = "SKIP"
        result["reason"] = f"リスク管理: {risk_reason}"
        return result

    # レジーム判定
    regime = int(latest["Regime"])
    if regime == 2:
        result["action"] = "SKIP"
        result["reason"] = "高ボラティリティ（相場が荒れている）"
        return result

    # 待機条件（ボラティリティ異常）
    vol = latest["Volatility_24"]
    vol_mult = params["wait_mode"]["vol_mult"]
    if vol > vol_mult * models["hist_vol"]:
        result["action"] = "SKIP"
        result["reason"] = f"ボラティリティ異常 ({vol:.6f} > {vol_mult * models['hist_vol']:.6f})"
        return result

    # 5モデルアンサンブル予測
    ensemble = models["ensemble"]
    proba = ensemble.predict_proba(X_latest)[0]
    pred, agreement = ensemble.predict_with_agreement(X_latest)
    pred = pred[0]
    agreement = int(agreement[0])

    # 自信度フィルター
    confidence = max(proba[1], 1.0 - proba[1])
    if confidence < CONFIDENCE_THRESHOLD:
        result["action"] = "SKIP"
        result["reason"] = f"自信不足 ({confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%})"
        result["confidence"] = float(confidence)
        result["agreement"] = agreement
        return result

    # 一致人数フィルター（CAD/JPY: 3人以上で十分）
    if agreement < AGREEMENT_THRESHOLD:
        result["action"] = "SKIP"
        result["reason"] = f"一致不足 ({agreement}/5人、{AGREEMENT_THRESHOLD}人以上が必要)"
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
    """予測結果をCSVに記録する（CAD/JPY専用ログ）"""
    log_file = LOG_DIR / "predictions_cadjpy.csv"

    # 固定カラム順で記録
    columns = [
        "logged_at", "timestamp", "close", "regime", "action",
        "mode", "reason", "confidence", "agreement", "proba_up", "proba_down",
        "exit_price", "actual_return", "net_return", "result",
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


def check_past_predictions(df):
    """過去の予測結果を実際の値動きと照合する"""
    log_file = LOG_DIR / "predictions_cadjpy.csv"
    if not log_file.exists():
        return

    logs = pd.read_csv(log_file)
    # actionがBUYまたはSELLのもので、まだ結果が記録されていないものを探す
    trades = logs[logs["action"].isin(["BUY", "SELL"])].copy()
    if "actual_return" in trades.columns:
        unchecked = trades[trades["actual_return"].isna()]
    else:
        unchecked = trades

    if unchecked.empty:
        return

    updated = False
    for idx, row in unchecked.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        # 4時間後の終値を確認
        target_ts = ts + timedelta(hours=4)
        if target_ts not in df.index:
            continue

        entry_price = row["close"]
        exit_price = df.loc[target_ts, "Close"]
        raw_return = (exit_price - entry_price) / entry_price

        if row["action"] == "BUY":
            actual_return = raw_return
        else:
            actual_return = -raw_return

        # 手数料（スプレッド）を引く — CAD/JPYは0.0004
        net_return = actual_return - SPREAD_COST

        logs.loc[idx, "exit_price"] = exit_price
        logs.loc[idx, "actual_return"] = actual_return
        logs.loc[idx, "net_return"] = net_return
        logs.loc[idx, "result"] = "WIN" if net_return > 0 else "LOSE"
        updated = True

        # リスクマネージャーにトレード結果を記録
        pnl_amount = net_return * risk_manager.account_balance
        direction_str = "long" if row["action"] == "BUY" else "short"
        risk_manager.record_trade("CADJPY", direction_str, entry_price, exit_price, pnl_amount)

    if updated:
        logs.to_csv(log_file, index=False)
        risk_manager.save_state(RISK_STATE_PATH)
        # 成績サマリーを表示
        completed = logs[logs["result"].isin(["WIN", "LOSE"])]
        if len(completed) > 0:
            wins = (completed["result"] == "WIN").sum()
            total = len(completed)
            net_returns = completed["net_return"].astype(float)
            cum_return = net_returns.sum()
            print(f"\n【CAD/JPY ペーパートレード成績】")
            print(f"  完了トレード: {total}回")
            print(f"  勝率: {wins}/{total} ({wins/total*100:.1f}%)")
            print(f"  累積リターン: {cum_return:+.6f}")


def _build_reasons(result):
    """Telegram通知用の理由リストを生成する"""
    regime_names = {0: "トレンド", 1: "レンジ", 2: "高ボラ"}
    return [
        f"{result.get('agreement', '?')}/5モデル一致",
        f"自信度 {result.get('confidence', 0):.1%}",
        f"レジーム: {regime_names.get(result.get('regime', 0), '不明')}",
        f"上昇確率: {result.get('proba_up', 0):.1%}",
    ]


def run_once():
    """1回だけ予測を実行する"""
    params = load_params()
    df, feature_cols = prepare_data()
    models = train_models(df, feature_cols, params)

    # 過去の予測結果を照合
    check_past_predictions(df)

    # 最新の予測
    result = predict_latest(df, feature_cols, models, params)

    # 結果を表示
    print(f"\n{'='*50}")
    print(f"【CAD/JPY 予測結果】{result['timestamp']}")
    print(f"  現在レート: {result['close']:.3f}")
    regime_names = {0: "トレンド", 1: "レンジ", 2: "高ボラ"}
    print(f"  レジーム: {regime_names.get(result['regime'], '不明')}")

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

    # リスク管理状態を表示
    status = risk_manager.get_status()
    print(f"【リスク状態】残高={status['account_balance']:.0f} "
          f"DD={status['drawdown_pct']:.1%} 連敗={status['losing_streak']} "
          f"ポジション={status['open_positions']}/{status['max_positions']}")

    # CSV記録
    log_prediction(result)

    # Telegram通知 + リスク管理登録
    if result["action"] in ("BUY", "SELL"):
        # リスクマネージャーにポジション登録
        sl_price = result.get("stop_loss", result["close"])
        lot = risk_manager.calculate_position_size("CADJPY", result["close"], sl_price)
        direction_str = "long" if result["action"] == "BUY" else "short"
        risk_manager.open_position("CADJPY", direction_str, result["close"], sl_price, lot)
        risk_manager.save_state(RISK_STATE_PATH)
        print(f"ポジション登録: {lot:.2f}ロット (SL={sl_price:.3f})")

        try:
            reasons = _build_reasons(result)
            if "stop_loss" in result:
                reasons.append(f"SL: {result['stop_loss']:.3f} / TP: {result['take_profit']:.3f}")
            signal = {
                "pair": "CAD/JPY",
                "action": result["action"],
                "price": result["close"],
                "confidence": result.get("confidence", 0),
                "agreement": result.get("agreement", 0),
                "reasons": reasons,
                "category": "FX",
            }
            send_signal_sync(signal)
            print("Telegram通知送信完了")
        except Exception as e:
            print(f"Telegram通知失敗: {e}")

    return result


def run_loop(interval_minutes=60):
    """指定間隔で繰り返し実行する"""
    print(f"CAD/JPY ペーパートレード開始（{interval_minutes}分間隔で自動実行）")
    print("Ctrl+C で停止\n")

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CAD/JPY 実行中...")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAD/JPY ペーパートレード")
    parser.add_argument("--loop", action="store_true", help="自動繰り返しモード")
    parser.add_argument("--interval", type=int, default=60, help="繰り返し間隔（分）")
    args = parser.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        run_once()
