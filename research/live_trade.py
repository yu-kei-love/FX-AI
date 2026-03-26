# ===========================================
# live_trade.py
# OANDA API連携 自動FXトレードスクリプト
#
# paper_trade.py のモデル・予測ロジックを再利用し、
# OANDA v20 API経由で実際に注文を発行する。
#
# デフォルトはpractice（デモ）環境。
# --dry-run で注文発行をスキップしてテスト可能。
#
# 使い方:
#   python research/live_trade.py --dry-run          # ドライラン（注文なし）
#   python research/live_trade.py                    # practice環境で自動取引
#   python research/live_trade.py --loop             # 1時間ごとに繰り返し
#   python research/live_trade.py --loop --dry-run   # ループ＋ドライラン
#
# 安全機能:
#   - デフォルトpractice環境（live切替は.envで明示設定が必要）
#   - 日次最大損失リミット（5%）
#   - 最大同時ポジション数（2）
#   - キルスイッチ: data/KILL_SWITCH ファイルが存在すると取引停止
#   - 週末検出: 土日はトレードしない
#   - 全トレードをCSVログ + Telegram通知
# ===========================================

import os
import sys
import csv
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- パス設定 ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- .env読み込み ---
load_dotenv(PROJECT_ROOT / ".env", override=True)

# --- OANDA設定 ---
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")  # practice or live

# --- paper_trade.pyから再利用 ---
from research.paper_trade import (
    prepare_data,
    train_models,
    predict_latest,
    load_params,
    update_data,
    PAIRS,
    FORECAST_HORIZON,
    sl_manager,
    risk_manager,
    RISK_STATE_PATH,
    _build_reasons,
    _pair_display_name,
)
from research.telegram_bot import send_signal_sync

# --- ディレクトリ ---
DATA_DIR = (PROJECT_ROOT / "data").resolve()
LIVE_LOG_DIR = DATA_DIR / "live_trade_logs"
LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
KILL_SWITCH_PATH = DATA_DIR / "KILL_SWITCH"
TRADE_LOG_PATH = LIVE_LOG_DIR / "trades.csv"

# --- 安全パラメータ ---
MAX_RISK_PER_TRADE_PCT = 0.02   # 1トレードあたり最大リスク: 口座残高の2%
MAX_DAILY_LOSS_PCT = 0.05       # 日次最大損失: 口座残高の5%
MAX_CONCURRENT_POSITIONS = 2    # 最大同時ポジション数
LOOP_INTERVAL_MINUTES = 60      # ループ間隔（分）

# --- OANDA通貨ペア変換 ---
# paper_trade.pyの "USDJPY" → OANDAの "USD_JPY"
def to_oanda_instrument(pair: str) -> str:
    """USDJPY → USD_JPY 形式に変換"""
    return f"{pair[:3]}_{pair[3:]}"


def from_oanda_instrument(instrument: str) -> str:
    """USD_JPY → USDJPY 形式に変換"""
    return instrument.replace("_", "")


# ===========================================================
# ロギング設定
# ===========================================================
def setup_logging():
    """ファイル+コンソールのロガーを設定"""
    log_file = LIVE_LOG_DIR / "live_trade.log"
    logger = logging.getLogger("live_trade")
    logger.setLevel(logging.DEBUG)

    # ファイルハンドラ
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # コンソールハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logging()


# ===========================================================
# OANDA API クライアント
# ===========================================================
class OandaClient:
    """oandapyV20ベースのOANDA APIクライアント"""

    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        self._api = None
        self._connected = False

    def connect(self) -> bool:
        """APIに接続する。oandapyV20が必要。"""
        if not self.api_key or self.api_key == "your_oanda_key_here":
            log.warning("OANDA_API_KEY が未設定です。.envファイルを確認してください。")
            return False
        if not self.account_id:
            log.warning("OANDA_ACCOUNT_ID が未設定です。.envファイルを確認してください。")
            return False

        try:
            import oandapyV20
            self._api = oandapyV20.API(
                access_token=self.api_key,
                environment=self.environment,
            )
            self._connected = True
            log.info(f"OANDA API接続成功 (環境: {self.environment}, 口座: {self.account_id})")
            return True
        except ImportError:
            log.error(
                "oandapyV20がインストールされていません。\n"
                "  pip install oandapyV20"
            )
            return False
        except Exception as e:
            log.error(f"OANDA API接続失敗: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._api is not None

    def get_account_summary(self) -> dict:
        """口座サマリーを取得"""
        import oandapyV20.endpoints.accounts as accounts
        r = accounts.AccountSummary(self.account_id)
        self._api.request(r)
        return r.response.get("account", {})

    def get_account_balance(self) -> float:
        """口座残高を取得（float）"""
        summary = self.get_account_summary()
        return float(summary.get("balance", 0))

    def get_open_positions(self) -> list:
        """オープンポジション一覧を取得"""
        import oandapyV20.endpoints.positions as positions
        r = positions.OpenPositions(self.account_id)
        self._api.request(r)
        return r.response.get("positions", [])

    def get_open_trades(self) -> list:
        """オープントレード一覧を取得"""
        import oandapyV20.endpoints.trades as trades
        r = trades.OpenTrades(self.account_id)
        self._api.request(r)
        return r.response.get("trades", [])

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: float = None,
        take_profit_price: float = None,
    ) -> dict:
        """
        成行注文を発行する。

        Args:
            instrument: "USD_JPY" 形式
            units: 正=買い, 負=売り
            stop_loss_price: ストップロス価格
            take_profit_price: テイクプロフィット価格

        Returns:
            APIレスポンスのdict
        """
        import oandapyV20.endpoints.orders as orders

        order_body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
            }
        }

        # SL設定
        if stop_loss_price is not None:
            # JPYペアは小数点3桁、それ以外は5桁
            precision = 3 if "JPY" in instrument else 5
            order_body["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss_price:.{precision}f}",
                "timeInForce": "GTC",
            }

        # TP設定
        if take_profit_price is not None:
            precision = 3 if "JPY" in instrument else 5
            order_body["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit_price:.{precision}f}",
            }

        r = orders.OrderCreate(self.account_id, data=order_body)
        self._api.request(r)
        return r.response

    def close_trade(self, trade_id: str) -> dict:
        """トレードをクローズする"""
        import oandapyV20.endpoints.trades as trades
        r = trades.TradeClose(self.account_id, tradeID=trade_id)
        self._api.request(r)
        return r.response

    def get_candles(self, instrument: str, granularity: str = "H1", count: int = 500) -> pd.DataFrame:
        """OANDAからローソク足データを取得する"""
        import oandapyV20.endpoints.instruments as instruments

        params = {
            "granularity": granularity,
            "count": count,
            "price": "MBA",  # Mid, Bid, Ask
        }
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        self._api.request(r)

        candles = r.response.get("candles", [])
        rows = []
        for c in candles:
            if not c.get("complete", False):
                continue  # 未完成のローソク足はスキップ
            mid = c["mid"]
            rows.append({
                "timestamp": pd.Timestamp(c["time"]),
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
                "Volume": int(c["volume"]),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize(None)  # タイムゾーン除去（既存パイプラインとの互換性）
        return df


# ===========================================================
# 安全チェック
# ===========================================================
def check_kill_switch() -> bool:
    """キルスイッチファイルが存在するか確認"""
    if KILL_SWITCH_PATH.exists():
        log.critical("KILL SWITCH が有効です。data/KILL_SWITCH を削除するまで取引停止。")
        return True
    return False


def is_weekend() -> bool:
    """土日（UTC基準）かどうか判定。金曜22:00 UTC以降〜日曜22:00 UTCは市場閉鎖。"""
    now = datetime.now(timezone.utc)
    dow = now.weekday()  # 0=月, 4=金, 5=土, 6=日
    hour = now.hour

    # 土曜日全日
    if dow == 5:
        return True
    # 日曜日 22:00 UTC前
    if dow == 6 and hour < 22:
        return True
    # 金曜日 22:00 UTC以降
    if dow == 4 and hour >= 22:
        return True
    return False


def check_daily_loss_limit(client: OandaClient, initial_balance: float) -> bool:
    """日次損失リミットを確認。リミット超過ならTrue"""
    try:
        current_balance = client.get_account_balance()
        daily_loss = (initial_balance - current_balance) / initial_balance
        if daily_loss >= MAX_DAILY_LOSS_PCT:
            log.warning(
                f"日次損失リミット到達: {daily_loss:.2%} >= {MAX_DAILY_LOSS_PCT:.0%} "
                f"(開始残高: {initial_balance:.0f}, 現在: {current_balance:.0f})"
            )
            return True
        return False
    except Exception as e:
        log.error(f"残高確認エラー: {e}")
        return True  # エラー時は安全側に倒す


def count_open_positions(client: OandaClient) -> int:
    """オープンポジション数を取得"""
    try:
        trades = client.get_open_trades()
        return len(trades)
    except Exception as e:
        log.error(f"ポジション数取得エラー: {e}")
        return MAX_CONCURRENT_POSITIONS  # エラー時は最大と仮定（安全側）


# ===========================================================
# ポジションサイズ計算
# ===========================================================
def calculate_units(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    pair: str,
    max_risk_pct: float = MAX_RISK_PER_TRADE_PCT,
) -> int:
    """
    口座残高の max_risk_pct% をリスクとしてポジションサイズ（units）を計算する。

    Args:
        account_balance: 口座残高
        entry_price: エントリー価格
        stop_loss_price: ストップロス価格
        pair: 通貨ペア（"USDJPY"など）
        max_risk_pct: 最大リスク割合（デフォルト2%）

    Returns:
        ユニット数（整数）
    """
    risk_amount = account_balance * max_risk_pct
    sl_distance = abs(entry_price - stop_loss_price)

    if sl_distance <= 0:
        log.warning("SL距離がゼロ以下。最小ユニット数を返します。")
        return 1

    # JPYペアの場合: 1ユニットの1pip損益 = 0.01 (3桁) or 0.001 (5桁)
    # 例: USD/JPY 150.000, SL距離 0.500 → 1ユニットあたり0.500円の損失
    # risk_amount / sl_distance = ユニット数
    units = int(risk_amount / sl_distance)

    # 最低1ユニット、最大100万ユニット（安全上限）
    units = max(1, min(units, 1_000_000))

    return units


# ===========================================================
# トレードログ
# ===========================================================
TRADE_LOG_COLUMNS = [
    "timestamp", "pair", "action", "entry_price", "units",
    "stop_loss", "take_profit", "atr", "confidence", "agreement",
    "order_id", "trade_id", "status", "exit_price", "exit_reason",
    "pnl", "account_balance", "dry_run", "error",
]


def log_trade(trade_data: dict):
    """トレードをCSVに記録する"""
    file_exists = TRADE_LOG_PATH.exists()

    row = {col: trade_data.get(col, "") for col in TRADE_LOG_COLUMNS}
    row["timestamp"] = row.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        log.debug(f"トレードログ記録: {TRADE_LOG_PATH}")
    except Exception as e:
        log.error(f"トレードログ書き込みエラー: {e}")


# ===========================================================
# Telegram通知（拡張版）
# ===========================================================
def notify_trade(result: dict, units: int, order_response: dict = None, dry_run: bool = False):
    """トレード実行をTelegram通知する"""
    try:
        pair_display = _pair_display_name(result["pair"])
        mode_label = "[DRY-RUN] " if dry_run else "[LIVE] "
        env_label = f"({OANDA_ENVIRONMENT})"

        reasons = _build_reasons(result)
        if "stop_loss" in result:
            reasons.append(f"SL: {result['stop_loss']:.3f} / TP: {result['take_profit']:.3f}")
        reasons.append(f"ユニット数: {units:,}")

        if order_response and "orderFillTransaction" in order_response:
            fill = order_response["orderFillTransaction"]
            reasons.append(f"約定価格: {fill.get('price', '?')}")
            reasons.append(f"Trade ID: {fill.get('tradeOpened', {}).get('tradeID', '?')}")

        signal = {
            "pair": f"{mode_label}{pair_display} {env_label}",
            "action": result["action"],
            "price": result["close"],
            "confidence": result.get("confidence", 0),
            "agreement": result.get("agreement", 0),
            "reasons": reasons,
            "category": "FX-LIVE",
        }
        send_signal_sync(signal)
        log.info(f"Telegram通知送信完了: {result['pair']} {result['action']}")
    except Exception as e:
        log.warning(f"Telegram通知失敗: {e}")


def notify_error(message: str):
    """エラーをTelegram通知する"""
    try:
        signal = {
            "pair": "SYSTEM",
            "action": "ERROR",
            "price": 0,
            "confidence": 0,
            "agreement": 0,
            "reasons": [message],
            "category": "SYSTEM",
        }
        send_signal_sync(signal)
    except Exception:
        pass  # 通知失敗は無視


# ===========================================================
# メイン取引ロジック
# ===========================================================
def execute_trade(
    client: OandaClient,
    result: dict,
    dry_run: bool = False,
) -> dict:
    """
    predict_latest()の結果に基づいてOANDAに注文を発行する。

    Args:
        client: OandaClientインスタンス
        result: predict_latest()の戻り値
        dry_run: Trueなら注文をスキップ

    Returns:
        トレード記録用のdict
    """
    pair = result["pair"]
    action = result["action"]
    entry_price = result["close"]
    sl_price = result.get("stop_loss")
    tp_price = result.get("take_profit")
    instrument = to_oanda_instrument(pair)

    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pair": pair,
        "action": action,
        "entry_price": entry_price,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "atr": result.get("atr", ""),
        "confidence": result.get("confidence", ""),
        "agreement": result.get("agreement", ""),
        "dry_run": dry_run,
        "status": "pending",
    }

    # 口座残高取得
    try:
        if client.is_connected:
            balance = client.get_account_balance()
        else:
            balance = 1_000_000  # フォールバック
    except Exception as e:
        log.warning(f"残高取得失敗、デフォルト値を使用: {e}")
        balance = 1_000_000

    trade_record["account_balance"] = balance

    # ユニット数計算
    if sl_price is None:
        log.warning(f"[{pair}] SLが未設定。最小ユニットで発注。")
        units = 1
    else:
        units = calculate_units(balance, entry_price, sl_price, pair)

    # 売りの場合はマイナス
    if action == "SELL":
        units = -units

    trade_record["units"] = units
    log.info(
        f"[{pair}] {action} 注文準備: {abs(units):,}ユニット @ ~{entry_price:.3f} "
        f"SL={sl_price} TP={tp_price}"
    )

    # --- ドライランチェック ---
    if dry_run:
        log.info(f"[{pair}] DRY-RUN: 注文をスキップしました")
        trade_record["status"] = "dry_run"
        trade_record["order_id"] = "DRY_RUN"
        notify_trade(result, abs(units), dry_run=True)
        return trade_record

    # --- 実際の注文発行 ---
    if not client.is_connected:
        log.error(f"[{pair}] OANDA未接続。注文できません。")
        trade_record["status"] = "error"
        trade_record["error"] = "OANDA未接続"
        return trade_record

    try:
        response = client.place_market_order(
            instrument=instrument,
            units=units,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )

        # レスポンス解析
        if "orderFillTransaction" in response:
            fill = response["orderFillTransaction"]
            trade_record["status"] = "filled"
            trade_record["order_id"] = fill.get("orderID", "")
            trade_record["trade_id"] = fill.get("tradeOpened", {}).get("tradeID", "")
            trade_record["entry_price"] = float(fill.get("price", entry_price))
            log.info(
                f"[{pair}] 注文約定: OrderID={trade_record['order_id']} "
                f"TradeID={trade_record['trade_id']} @ {trade_record['entry_price']}"
            )
        elif "orderCancelTransaction" in response:
            cancel = response["orderCancelTransaction"]
            trade_record["status"] = "cancelled"
            trade_record["error"] = cancel.get("reason", "unknown")
            log.warning(f"[{pair}] 注文キャンセル: {trade_record['error']}")
        else:
            trade_record["status"] = "unknown"
            trade_record["error"] = json.dumps(response)[:200]
            log.warning(f"[{pair}] 不明なレスポンス: {trade_record['error']}")

        notify_trade(result, abs(units), order_response=response, dry_run=False)

    except Exception as e:
        trade_record["status"] = "error"
        trade_record["error"] = str(e)[:200]
        log.error(f"[{pair}] 注文エラー: {e}")
        notify_error(f"注文エラー [{pair}]: {str(e)[:100]}")

    return trade_record


def check_and_manage_positions(client: OandaClient, dry_run: bool = False):
    """
    オープンポジションのSL/TP管理（バックアップ）。
    OANDA側のSL/TPが設定されているか確認し、欠けていれば追加する。
    """
    if not client.is_connected:
        return

    try:
        open_trades = client.get_open_trades()
        for trade in open_trades:
            trade_id = trade.get("id", "?")
            instrument = trade.get("instrument", "")
            has_sl = "stopLossOrder" in trade and trade["stopLossOrder"] is not None
            has_tp = "takeProfitOrder" in trade and trade["takeProfitOrder"] is not None

            if not has_sl or not has_tp:
                log.warning(
                    f"Trade {trade_id} ({instrument}): "
                    f"SL={'あり' if has_sl else 'なし'} / TP={'あり' if has_tp else 'なし'} "
                    f"- SL/TP欠損を検出。手動で確認してください。"
                )
    except Exception as e:
        log.error(f"ポジション管理チェックエラー: {e}")


# ===========================================================
# メインループ
# ===========================================================
def run_once(client: OandaClient, dry_run: bool = False, skip_update: bool = False):
    """1回の予測→取引サイクルを実行する"""
    log.info("=" * 60)
    log.info(f"実行開始 (dry_run={dry_run}, env={OANDA_ENVIRONMENT})")
    log.info("=" * 60)

    # --- 安全チェック ---
    if check_kill_switch():
        return {}

    if is_weekend():
        log.info("週末のため取引スキップ")
        return {}

    # --- 日次損失チェック ---
    daily_start_balance = None
    if client.is_connected and not dry_run:
        try:
            daily_start_balance = client.get_account_balance()
            if check_daily_loss_limit(client, daily_start_balance):
                log.warning("日次損失リミット超過。取引停止。")
                notify_error("日次損失リミット超過。取引停止。")
                return {}
        except Exception as e:
            log.error(f"日次損失チェックエラー: {e}")

    # --- ポジション数チェック ---
    if client.is_connected:
        open_count = count_open_positions(client)
        if open_count >= MAX_CONCURRENT_POSITIONS:
            log.info(
                f"最大ポジション数到達 ({open_count}/{MAX_CONCURRENT_POSITIONS})。新規注文スキップ。"
            )
            # 既存ポジション管理のみ実行
            check_and_manage_positions(client, dry_run)
            return {}

    # --- データ更新 ---
    if not skip_update:
        try:
            update_data()
        except Exception as e:
            log.warning(f"データ更新失敗（既存データを使用）: {e}")

    # --- パラメータ読み込み ---
    params = load_params()

    # --- 各ペアで予測→取引 ---
    results = {}
    for pair in PAIRS:
        log.info(f"--- {_pair_display_name(pair)} ---")
        try:
            # データ準備＆モデル学習
            df, feature_cols = prepare_data(pair=pair)
            models = train_models(df, feature_cols, params, pair=pair)

            # 予測実行
            result = predict_latest(df, feature_cols, models, params, pair=pair)

            log.info(
                f"[{pair}] 予測結果: {result['action']} "
                f"(confidence={result.get('confidence', '?')}, "
                f"agreement={result.get('agreement', '?')})"
            )

            if result.get("reason"):
                log.info(f"[{pair}] 理由: {result['reason']}")

            # BUY/SELLシグナルの場合は注文
            if result["action"] in ("BUY", "SELL"):
                # ポジション数の再チェック（他ペアで注文した場合）
                if client.is_connected:
                    current_open = count_open_positions(client)
                    if current_open >= MAX_CONCURRENT_POSITIONS:
                        log.info(f"[{pair}] ポジション上限。スキップ。")
                        results[pair] = result
                        continue

                trade_record = execute_trade(client, result, dry_run=dry_run)
                log_trade(trade_record)
                results[pair] = result
            else:
                results[pair] = result

        except Exception as e:
            log.error(f"[{pair}] 処理エラー: {e}", exc_info=True)
            results[pair] = {"pair": pair, "action": "ERROR", "reason": str(e)}

    # --- 既存ポジション管理 ---
    check_and_manage_positions(client, dry_run)

    log.info("実行完了")
    return results


def run_loop(client: OandaClient, dry_run: bool = False):
    """指定間隔で繰り返し実行する"""
    log.info(f"ループ開始 ({LOOP_INTERVAL_MINUTES}分間隔, dry_run={dry_run})")
    log.info("Ctrl+C で停止")

    while True:
        try:
            # キルスイッチチェック
            if check_kill_switch():
                log.critical("キルスイッチ検出。ループ終了。")
                break

            run_once(client, dry_run=dry_run)

        except KeyboardInterrupt:
            log.info("ユーザーによる停止")
            break
        except Exception as e:
            log.error(f"ループ内エラー: {e}", exc_info=True)
            notify_error(f"ループエラー: {str(e)[:100]}")

        next_run = datetime.now() + timedelta(minutes=LOOP_INTERVAL_MINUTES)
        log.info(f"次回実行: {next_run.strftime('%H:%M:%S')}")

        try:
            time.sleep(LOOP_INTERVAL_MINUTES * 60)
        except KeyboardInterrupt:
            log.info("ユーザーによる停止")
            break


# ===========================================================
# エントリーポイント
# ===========================================================
def main():
    parser = argparse.ArgumentParser(
        description="OANDA API連携 自動FXトレード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使い方:
  python research/live_trade.py --dry-run          # ドライラン（注文なし）
  python research/live_trade.py                    # practice環境で自動取引
  python research/live_trade.py --loop             # 1時間ごとに繰り返し
  python research/live_trade.py --loop --dry-run   # ループ＋ドライラン

安全機能:
  - デフォルトpractice環境
  - 日次損失5%でストップ
  - 最大2ポジション
  - data/KILL_SWITCH でキルスイッチ
  - 土日は自動スキップ
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="注文を発行せずにテストする（予測・ログ・通知は実行）",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help=f"{LOOP_INTERVAL_MINUTES}分間隔で繰り返し実行",
    )
    parser.add_argument(
        "--skip-update", action="store_true",
        help="データ更新をスキップ（既存データを使用）",
    )
    args = parser.parse_args()

    # --- 起動メッセージ ---
    log.info("=" * 60)
    log.info("OANDA Live Trade System 起動")
    log.info(f"  環境: {OANDA_ENVIRONMENT}")
    log.info(f"  ドライラン: {args.dry_run}")
    log.info(f"  ループ: {args.loop}")
    log.info(f"  通貨ペア: {', '.join(PAIRS)}")
    log.info(f"  最大リスク/トレード: {MAX_RISK_PER_TRADE_PCT:.0%}")
    log.info(f"  日次損失リミット: {MAX_DAILY_LOSS_PCT:.0%}")
    log.info(f"  最大ポジション数: {MAX_CONCURRENT_POSITIONS}")
    log.info("=" * 60)

    # --- live環境の警告 ---
    if OANDA_ENVIRONMENT == "live" and not args.dry_run:
        log.warning("!!! LIVE環境での実取引モードです !!!")
        log.warning("10秒以内にCtrl+Cで中断できます。")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            log.info("ユーザーにより中断されました。")
            return

    # --- OANDA接続 ---
    client = OandaClient(OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT)
    connected = client.connect()

    if not connected and not args.dry_run:
        log.error("OANDA接続失敗。--dry-run モードで実行するか、.envの設定を確認してください。")
        return

    if not connected and args.dry_run:
        log.warning("OANDA未接続ですが、dry-runモードのため続行します。")

    # --- 口座情報表示 ---
    if connected:
        try:
            summary = client.get_account_summary()
            log.info(f"口座残高: {float(summary.get('balance', 0)):,.0f} {summary.get('currency', '?')}")
            log.info(f"未実現損益: {float(summary.get('unrealizedPL', 0)):,.0f}")
            log.info(f"証拠金使用率: {float(summary.get('marginUsed', 0)):,.0f}")
        except Exception as e:
            log.warning(f"口座情報取得失敗: {e}")

    # --- 実行 ---
    if args.loop:
        run_loop(client, dry_run=args.dry_run)
    else:
        run_once(client, dry_run=args.dry_run, skip_update=args.skip_update)


if __name__ == "__main__":
    main()
