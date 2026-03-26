# ===========================================
# paper_trade_stocks.py
# 日本株ペーパートレード v3（日足ベース）
#
# japan_stock_model.py v3.0 のパイプラインを直接利用:
#   - Expanding Window Walk-Forward
#   - 14銘柄スクリーニング
#   - 性能ベースアンサンブル重み付け
#   - ボラティリティレジーム特徴量
#   - CONFIDENCE_THRESHOLD = 0.60
#   - AGREEMENT_THRESHOLD = 4
#
# 使い方:
#   python research/paper_trade_stocks.py          # 1回だけ予測
#   python research/paper_trade_stocks.py --loop   # 毎日自動で繰り返し
# ===========================================

import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.ensemble import EnsembleClassifier
from research.common.risk_manager import RiskManager
from research.common.stop_loss import StopLossManager
from research.common.economic_calendar import is_safe_to_trade
from research.telegram_bot import send_signal_sync

# japan_stock_model v3 のコアモジュールをインポート
from research.japan_stock_model import (
    download_data as jsm_download_data,
    make_features as jsm_make_features,
    train_and_predict as jsm_train_and_predict,
    JP_STOCKS,
    SCREENING_TARGETS,
    CONFIDENCE_THRESHOLD,
    AGREEMENT_THRESHOLD,
    TRAIN_DAYS,
    TEST_DAYS,
)

script_dir = Path(__file__).resolve().parent
DATA_DIR = (script_dir / ".." / "data").resolve()
LOG_DIR = DATA_DIR / "paper_trade_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# スプレッドコスト（全銘柄共通）
SPREAD_COST = 0.001

# ─── リスク管理 ───
INITIAL_BALANCE = 1_000_000
RISK_STATE_PATH = LOG_DIR / "risk_state_stocks.json"


def _init_risk_manager():
    """リスクマネージャーを初期化（共有状態があれば復元）"""
    if RISK_STATE_PATH.exists():
        try:
            rm = RiskManager.load_state(RISK_STATE_PATH)
            print(f"リスク状態復元: 残高={rm.account_balance:.0f}, 連敗={rm.losing_streak}")
            return rm
        except Exception as e:
            print(f"リスク状態復元失敗（新規作成）: {e}")
    return RiskManager(account_balance=INITIAL_BALANCE)


risk_manager = _init_risk_manager()
sl_manager = StopLossManager(atr_period=14, sl_multiplier=1.5, tp_multiplier=2.0, max_hold_hours=24)


# ===========================================
# 1. ログ記録
# ===========================================
def log_prediction(result, ticker):
    """予測結果をCSVに記録する"""
    log_file = LOG_DIR / "predictions_stocks.csv"

    columns = [
        "logged_at", "ticker", "timestamp", "close", "action",
        "mode", "reason", "confidence", "agreement", "proba_up", "proba_down",
        "exit_price", "actual_return", "net_return", "result",
    ]
    row = {
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
    }
    for col in columns:
        if col in row:
            continue
        row[col] = result.get(col, "")

    df_row = pd.DataFrame([row], columns=columns)
    if log_file.exists():
        df_row.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_file, index=False)
    print(f"記録完了: {log_file}")


# ===========================================
# 2. 過去予測の照合
# ===========================================
def check_past_predictions(ticker, all_data):
    """過去の予測と実際の翌日終値を照合する"""
    log_file = LOG_DIR / "predictions_stocks.csv"
    if not log_file.exists():
        return

    logs = pd.read_csv(log_file)
    # この銘柄のBUY/SELLで未照合のものだけ
    mask = (logs["ticker"] == ticker) & logs["action"].isin(["BUY", "SELL"])
    if "actual_return" in logs.columns:
        mask = mask & logs["actual_return"].isna()
    trades = logs[mask]
    if trades.empty:
        return

    # 対象銘柄の株価データを取得
    if ticker not in all_data:
        return
    df = all_data[ticker].copy()
    df.index = pd.to_datetime(df.index)

    updated = False
    pair_name = ticker.replace(".", "")
    name = JP_STOCKS.get(ticker, ticker)
    for idx, row in trades.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        # 翌営業日の終値を探す
        future = df.loc[df.index > ts]
        if future.empty:
            continue

        entry_price = row["close"]
        exit_price = float(future.iloc[0]["Close"])
        raw_return = (exit_price - entry_price) / entry_price

        if row["action"] == "BUY":
            actual_return = raw_return
        else:
            actual_return = -raw_return

        net_return = actual_return - SPREAD_COST

        logs.loc[idx, "exit_price"] = exit_price
        logs.loc[idx, "actual_return"] = actual_return
        logs.loc[idx, "net_return"] = net_return
        logs.loc[idx, "result"] = "WIN" if net_return > 0 else "LOSE"
        updated = True

        # リスクマネージャーに記録
        pnl_amount = net_return * risk_manager.account_balance
        direction_str = "long" if row["action"] == "BUY" else "short"
        risk_manager.record_trade(pair_name, direction_str, entry_price, exit_price, pnl_amount)

    if updated:
        logs.to_csv(log_file, index=False)
        risk_manager.save_state(RISK_STATE_PATH)
        # 成績サマリー
        completed = logs[(logs["ticker"] == ticker) & logs["result"].isin(["WIN", "LOSE"])]
        if len(completed) > 0:
            wins = (completed["result"] == "WIN").sum()
            total = len(completed)
            cum = completed["net_return"].astype(float).sum()
            print(f"\n【{name} 成績】{wins}/{total}勝 ({wins/total*100:.1f}%) 累積={cum:+.4f}")


# ===========================================
# 3. 1銘柄の予測パイプライン（v2: Expanding WF）
# ===========================================
def predict_stock(ticker, all_data):
    """japan_stock_model v2 のExpanding WFパイプラインで1銘柄を予測する

    Returns:
        result dict or None
    """
    name = JP_STOCKS.get(ticker, ticker)
    pair_name = ticker.replace(".", "")

    print(f"\n{'='*50}")
    print(f"【{name}({ticker})】v2 Expanding WF")
    print(f"{'='*50}")

    # 特徴量生成（v2: インタラクション特徴量込み）
    X, y, feature_names = jsm_make_features(all_data, target_ticker=ticker)
    if X is None or len(X) < TRAIN_DAYS + TEST_DAYS:
        print(f"  データ不足でスキップ")
        return None

    # Expanding WF学習・予測
    results, latest_pred = jsm_train_and_predict(X, y, feature_names)
    if latest_pred is None:
        print(f"  最新予測なし")
        return None

    # 結果を整形
    confidence = latest_pred["confidence"]
    agreement = latest_pred["agreement"]
    direction = latest_pred["direction"]  # "UP" or "DOWN"
    pred_date = latest_pred["date"]

    # 株価取得
    if ticker in all_data:
        close_price = float(all_data[ticker]["Close"].iloc[-1])
    else:
        close_price = 0.0

    result = {
        "timestamp": pred_date.strftime("%Y-%m-%d") if hasattr(pred_date, "strftime") else str(pred_date),
        "close": close_price,
        "confidence": confidence,
        "agreement": agreement,
        "proba_up": confidence if direction == "UP" else 1.0 - confidence,
        "proba_down": 1.0 - confidence if direction == "UP" else confidence,
    }

    # 経済カレンダーチェック
    safe, event_name = is_safe_to_trade(datetime.now())
    if not safe:
        result["action"] = "SKIP"
        result["reason"] = f"経済イベント回避: {event_name}"
        return result

    # リスク管理チェック
    can, risk_reason = risk_manager.can_trade(pair=pair_name, direction="long")
    if not can:
        result["action"] = "SKIP"
        result["reason"] = f"リスク管理: {risk_reason}"
        return result

    # 自信度フィルター（v2: 0.60）
    if confidence < CONFIDENCE_THRESHOLD:
        result["action"] = "SKIP"
        result["reason"] = f"自信不足 ({confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%})"
        return result

    # 一致数フィルター（v2: 4/5）
    if agreement < AGREEMENT_THRESHOLD:
        result["action"] = "SKIP"
        result["reason"] = f"一致不足 ({agreement}/5人、{AGREEMENT_THRESHOLD}人以上必要)"
        return result

    action = "BUY" if direction == "UP" else "SELL"

    # ATRベースSL/TP
    target_df = all_data[ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)
    atr_series = sl_manager.calculate_atr(target_df)
    current_atr = float(atr_series.iloc[-1])
    levels = sl_manager.get_levels(action, close_price, current_atr)

    result["action"] = action
    result["mode"] = f"v2 Expanding WF ({agreement}/5人一致)"
    result["stop_loss"] = levels["stop_loss"]
    result["take_profit"] = levels["take_profit"]
    result["atr"] = current_atr
    result["risk_reward"] = levels["risk_reward_ratio"]

    # 結果表示
    print(f"\n日付: {result['timestamp']} / 終値: {close_price:.1f}")
    print(f"判断: {action}（{'買い' if action == 'BUY' else '売り'}）")
    print(f"モデル: {result['mode']} / 自信度: {confidence:.1%}")
    print(f"SL: {levels['stop_loss']:.1f} / TP: {levels['take_profit']:.1f} / R:R=1:{levels['risk_reward_ratio']:.2f}")

    return result


# ===========================================
# 4. Telegram通知
# ===========================================
def send_notification(result, ticker):
    """シグナルをTelegramに送信する"""
    name = JP_STOCKS.get(ticker, ticker)
    pair_name = ticker.replace(".", "")

    sl_price = result.get("stop_loss", result["close"])
    lot = risk_manager.calculate_position_size(pair_name, result["close"], sl_price)
    direction_str = "long" if result["action"] == "BUY" else "short"
    risk_manager.open_position(pair_name, direction_str, result["close"], sl_price, lot)
    risk_manager.save_state(RISK_STATE_PATH)

    try:
        reasons = [
            f"{result.get('agreement', '?')}/5モデル一致",
            f"自信度 {result.get('confidence', 0):.1%}",
            f"上昇確率: {result.get('proba_up', 0):.1%}",
        ]
        if "stop_loss" in result:
            reasons.append(f"SL: {result['stop_loss']:.1f} / TP: {result['take_profit']:.1f}")

        signal = {
            "pair": f"{name}({ticker})",
            "action": result["action"],
            "price": result["close"],
            "confidence": result.get("confidence", 0),
            "agreement": result.get("agreement", 0),
            "reasons": reasons,
            "category": "株",
        }
        send_signal_sync(signal)
        print("Telegram通知送信完了")
    except Exception as e:
        print(f"Telegram通知失敗: {e}")


# ===========================================
# 5. 全銘柄を実行（v2: 14銘柄スクリーニング）
# ===========================================
def run_once():
    """全14対象銘柄に対してv2パイプラインを実行する"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 日本株ペーパートレード v2 開始")
    print(f"設定: CONFIDENCE >= {CONFIDENCE_THRESHOLD:.0%}, AGREEMENT >= {AGREEMENT_THRESHOLD}/5")
    print(f"対象銘柄: {len(SCREENING_TARGETS)}銘柄")
    print(f"方式: Expanding Window Walk-Forward")

    # データ一括ダウンロード（japan_stock_model.py の関数を使用）
    all_data = jsm_download_data(period="2y")
    if len(all_data) < 3:
        print("[NG] 十分なデータが取得できませんでした")
        return {}

    results = {}
    signals_sent = 0

    for ticker in SCREENING_TARGETS:
        try:
            # 過去予測照合
            check_past_predictions(ticker, all_data)

            # 予測
            result = predict_stock(ticker, all_data)
            if result is None:
                continue

            results[ticker] = result

            # CSV記録
            log_prediction(result, ticker)

            # Telegram通知（BUY/SELLのみ）
            if result["action"] in ("BUY", "SELL"):
                send_notification(result, ticker)
                signals_sent += 1

        except Exception as e:
            print(f"エラー ({ticker}): {e}")
            import traceback
            traceback.print_exc()

    # サマリー
    print(f"\n{'='*50}")
    print(f"【サマリー】")
    total = len(results)
    trades = sum(1 for r in results.values() if r.get("action") in ("BUY", "SELL"))
    skips = total - trades
    print(f"処理: {total}銘柄 / シグナル: {trades}件 / 見送り: {skips}件")

    # リスク管理状態を表示
    status = risk_manager.get_status()
    print(f"【リスク状態】残高={status['account_balance']:.0f} "
          f"DD={status['drawdown_pct']:.1%} 連敗={status['losing_streak']} "
          f"ポジション={status['open_positions']}/{status['max_positions']}")
    return results


# ===========================================
# 6. 日次ループ
# ===========================================
def run_loop(interval_minutes=1440):
    """指定間隔（デフォルト24時間）で繰り返し実行する"""
    print(f"日本株ペーパートレード v2 開始（{interval_minutes}分間隔で自動実行）")
    print("Ctrl+C で停止\n")

    while True:
        try:
            run_once()
        except Exception as e:
            print(f"エラー: {e}")
            import traceback
            traceback.print_exc()

        next_run = datetime.now() + timedelta(minutes=interval_minutes)
        print(f"\n次回実行: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n停止しました")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日本株 AI ペーパートレード v2")
    parser.add_argument("--loop", action="store_true", help="自動繰り返しモード")
    parser.add_argument("--interval", type=int, default=1440, help="繰り返し間隔（分、デフォルト1440=24時間）")
    args = parser.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        run_once()
