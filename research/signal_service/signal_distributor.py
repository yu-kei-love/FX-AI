# ===========================================
# signal_distributor.py
# 競艇AIシグナル配信サービス
#
# 機能:
#   - 日次レース予測シグナルの生成・配信
#   - オーナーチャンネル（全詳細）と購読者チャンネル（公開版）の二系統配信
#   - 日次結果サマリー配信
#   - 週次パフォーマンスレポート（毎週日曜）
#   - 成績自動トラッキング（勝率・PF・累計損益）
#
# スケジュール:
#   朝  09:00 - 当日予測シグナル配信
#   夜  21:00 - 当日結果サマリー配信
#   日曜 21:30 - 週次レポート配信
# ===========================================

import sys
import os
import json
import csv
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ログ設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "signal_service.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# パス定義
DATA_DIR = PROJECT_ROOT / "data"
SIGNAL_DATA_DIR = DATA_DIR / "signal_service"
BOAT_DATA_DIR = DATA_DIR / "boat"
PERFORMANCE_CSV = SIGNAL_DATA_DIR / "performance.csv"
DAILY_SIGNALS_DIR = SIGNAL_DATA_DIR / "daily_signals"
STATE_FILE = SIGNAL_DATA_DIR / "service_state.json"

# ディレクトリ作成
for d in [SIGNAL_DATA_DIR, DAILY_SIGNALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 会場名マッピング
VENUE_NAMES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

# ベットタイプ日本語
BET_TYPE_JP = {
    "win": "単勝",
    "exacta": "2連単",
    "quinella": "2連複",
}

# 信頼度の星表示
def _confidence_stars(ev: float) -> str:
    """EVに基づいて信頼度を星で表示"""
    if ev >= 3.5:
        return "★★★★★"
    elif ev >= 3.0:
        return "★★★★☆"
    elif ev >= 2.5:
        return "★★★☆☆"
    elif ev >= 2.0:
        return "★★☆☆☆"
    else:
        return "★☆☆☆☆"


# =============================================================
# Telegram送信
# =============================================================

def _get_telegram_config() -> Dict[str, Any]:
    """Telegram設定を読み込む"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    signal_channel_id = os.getenv("SIGNAL_CHANNEL_ID")

    # オーナーのchat_id
    chat_id_file = DATA_DIR / "telegram_chat_id.txt"
    owner_chat_id = None
    if chat_id_file.exists():
        try:
            owner_chat_id = int(chat_id_file.read_text().strip())
        except (ValueError, TypeError):
            pass

    return {
        "bot_token": bot_token,
        "owner_chat_id": owner_chat_id,
        "signal_channel_id": signal_channel_id,
    }


async def _send_telegram_async(
    bot_token: str,
    chat_id: int | str,
    text: str,
    parse_mode: str = None,
) -> bool:
    """Telegramにメッセージを非同期送信する"""
    try:
        import telegram
        bot = telegram.Bot(token=bot_token)
        max_len = 4000
        for i in range(0, len(text), max_len):
            chunk = text[i:i + max_len]
            await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode,
            )
        return True
    except Exception as e:
        logger.error(f"Telegram送信エラー (chat_id={chat_id}): {e}")
        return False


def send_to_owner(text: str, parse_mode: str = None) -> bool:
    """オーナーチャンネルにメッセージ送信（全詳細付き）"""
    config = _get_telegram_config()
    if not config["bot_token"] or not config["owner_chat_id"]:
        logger.warning("オーナーチャンネル設定なし")
        return False

    try:
        asyncio.run(_send_telegram_async(
            config["bot_token"], config["owner_chat_id"], text, parse_mode
        ))
        logger.info("オーナーチャンネル送信完了")
        return True
    except Exception as e:
        logger.error(f"オーナーチャンネル送信エラー: {e}")
        return False


def send_to_subscribers(text: str, parse_mode: str = None) -> bool:
    """購読者チャンネルにメッセージ送信（公開版）"""
    config = _get_telegram_config()
    if not config["bot_token"] or not config["signal_channel_id"]:
        logger.warning("購読者チャンネル設定なし（SIGNAL_CHANNEL_IDを.envに設定してください）")
        return False

    try:
        asyncio.run(_send_telegram_async(
            config["bot_token"], config["signal_channel_id"], text, parse_mode
        ))
        logger.info("購読者チャンネル送信完了")
        return True
    except Exception as e:
        logger.error(f"購読者チャンネル送信エラー: {e}")
        return False


def send_to_both(owner_text: str, subscriber_text: str, parse_mode: str = None) -> Dict[str, bool]:
    """両チャンネルにそれぞれのメッセージを送信"""
    results = {
        "owner": send_to_owner(owner_text, parse_mode),
        "subscribers": send_to_subscribers(subscriber_text, parse_mode),
    }
    return results


# =============================================================
# パフォーマンス管理
# =============================================================

def _init_performance_csv():
    """パフォーマンスCSVを初期化する"""
    if not PERFORMANCE_CSV.exists():
        with open(PERFORMANCE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "date", "race_venue", "race_number", "race_time",
                "bet_type", "lane", "ev", "confidence",
                "odds", "bet_amount", "result", "payout",
                "profit", "cumulative_profit",
            ])


def record_signal(
    signal_date: str,
    venue: str,
    race_number: int,
    race_time: str,
    bet_type: str,
    lane: int,
    ev: float,
    confidence: float,
    odds: float,
    bet_amount: int = 500,
) -> None:
    """シグナルをCSVに記録する（結果は後から更新）"""
    _init_performance_csv()
    with open(PERFORMANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            signal_date, venue, race_number, race_time,
            bet_type, lane, round(ev, 3), round(confidence, 4),
            round(odds, 1), bet_amount, "", 0, 0, 0,
        ])


def update_results(signal_date: str, results: List[Dict[str, Any]]) -> None:
    """
    当日の結果を更新する。
    results: [{"venue": str, "race_number": int, "winner_lane": int, "actual_odds": float}, ...]
    """
    if not PERFORMANCE_CSV.exists():
        logger.warning("パフォーマンスCSVが存在しません")
        return

    # 全行読み込み
    rows = []
    with open(PERFORMANCE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # 結果をマッピング
    result_map = {}
    for r in results:
        key = (r["venue"], int(r["race_number"]))
        result_map[key] = r

    # 累計利益を再計算
    cumulative = 0.0
    if rows:
        # 当日以前の累計を取得
        for row in rows:
            if row["date"] < signal_date and row["profit"]:
                try:
                    cumulative = float(row["cumulative_profit"])
                except (ValueError, TypeError):
                    pass

    # 当日の結果を更新
    for row in rows:
        if row["date"] != signal_date:
            continue

        key = (row["race_venue"], int(row["race_number"]))
        if key not in result_map:
            continue

        r = result_map[key]
        winner_lane = int(r["winner_lane"])
        bet_lane = int(row["lane"])
        bet_amount = int(row["bet_amount"]) if row["bet_amount"] else 500
        actual_odds = float(r.get("actual_odds", row["odds"]))

        if bet_lane == winner_lane:
            payout = int(bet_amount * actual_odds)
            profit = payout - bet_amount
            row["result"] = "WIN"
        else:
            payout = 0
            profit = -bet_amount
            row["result"] = "LOSE"

        row["payout"] = payout
        row["profit"] = profit
        cumulative += profit
        row["cumulative_profit"] = cumulative

    # CSV書き直し
    with open(PERFORMANCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"結果更新完了: {signal_date} ({len(results)}レース)")


def get_performance_summary(
    target_date: Optional[str] = None,
    period: str = "daily",
) -> Dict[str, Any]:
    """
    パフォーマンスサマリーを取得する。

    Args:
        target_date: 対象日 (YYYY-MM-DD)。Noneで本日。
        period: "daily", "weekly", "monthly"

    Returns:
        サマリー辞書
    """
    if target_date is None:
        target_date = date.today().isoformat()

    if not PERFORMANCE_CSV.exists():
        return {"wins": 0, "losses": 0, "total_bets": 0, "profit": 0, "pf": 0.0}

    rows = []
    with open(PERFORMANCE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"wins": 0, "losses": 0, "total_bets": 0, "profit": 0, "pf": 0.0}

    # 期間でフィルタ
    td = date.fromisoformat(target_date)
    if period == "daily":
        filtered = [r for r in rows if r["date"] == target_date]
    elif period == "weekly":
        week_start = td - timedelta(days=td.weekday())
        filtered = [
            r for r in rows
            if r["date"] >= week_start.isoformat() and r["date"] <= target_date
        ]
    elif period == "monthly":
        month_start = td.replace(day=1).isoformat()
        filtered = [r for r in rows if r["date"] >= month_start and r["date"] <= target_date]
    else:
        filtered = rows

    # 結果が入っているもののみ集計
    settled = [r for r in filtered if r.get("result") in ("WIN", "LOSE")]

    wins = sum(1 for r in settled if r["result"] == "WIN")
    losses = sum(1 for r in settled if r["result"] == "LOSE")
    total_bets = wins + losses

    gross_profit = sum(float(r["profit"]) for r in settled if r["result"] == "WIN")
    gross_loss = abs(sum(float(r["profit"]) for r in settled if r["result"] == "LOSE"))
    net_profit = sum(float(r["profit"]) for r in settled)

    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    win_rate = wins / total_bets if total_bets > 0 else 0.0

    # 累計利益
    all_settled = [r for r in rows if r.get("result") in ("WIN", "LOSE")]
    cumulative_profit = sum(float(r["profit"]) for r in all_settled)

    # 未確定シグナル数
    pending = len([r for r in filtered if r.get("result") not in ("WIN", "LOSE")])

    return {
        "period": period,
        "target_date": target_date,
        "wins": wins,
        "losses": losses,
        "total_bets": total_bets,
        "pending": pending,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "pf": pf,
        "cumulative_profit": cumulative_profit,
    }


# =============================================================
# シグナル生成
# =============================================================

def _load_boat_model():
    """競艇モデルをロードする"""
    try:
        from research.boat.boat_model import load_models
        models = load_models()
        if models:
            logger.info("競艇モデルロード完了")
            return models
    except Exception as e:
        logger.error(f"競艇モデルロードエラー: {e}")
    return None


def _load_today_races(target_date: Optional[str] = None) -> Optional[Any]:
    """
    当日のレースデータを読み込む。
    data/boat/daily/ に当日のCSVがある前提。
    なければ data/boat/real_race_data.csv から当日分を抽出。
    """
    import pandas as pd

    if target_date is None:
        target_date = date.today().strftime("%Y%m%d")
    else:
        target_date = target_date.replace("-", "")

    # 1. daily/YYYYMMDD.csv を確認
    daily_path = BOAT_DATA_DIR / "daily" / f"{target_date}.csv"
    if daily_path.exists():
        df = pd.read_csv(daily_path, encoding="utf-8-sig")
        logger.info(f"当日レースデータ読込: {daily_path} ({len(df)}行)")
        return df

    # 2. real_race_data.csv から当日分を抽出
    real_path = BOAT_DATA_DIR / "real_race_data.csv"
    if real_path.exists():
        df = pd.read_csv(real_path, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date_str"] = df["date"].astype(str).str.replace(".0", "", regex=False)
            today_df = df[df["date_str"] == target_date].copy()
            if len(today_df) > 0:
                logger.info(f"リアルデータから当日分抽出: {len(today_df)}レース")
                return today_df

    logger.warning(f"当日のレースデータが見つかりません: {target_date}")
    return None


def generate_signals(target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    当日のレース予測シグナルを生成する。

    Returns:
        シグナルのリスト。各シグナルは:
        {
            "venue": str, "venue_name": str, "race_number": int,
            "race_time": str, "lane": int, "bet_type": str,
            "ev": float, "confidence": float, "odds": float,
            "model_prob": float, "agreement": int,
        }
    """
    import pandas as pd
    from research.boat.boat_model import (
        create_features, predict_with_agreement,
        normalize_race_probs, find_value_bets,
        FEATURE_COLS, APPROX_ODDS, CLASS_NAMES,
    )

    if target_date is None:
        target_date = date.today().isoformat()

    logger.info(f"シグナル生成開始: {target_date}")

    # モデルロード
    models = _load_boat_model()
    if models is None:
        logger.error("モデルが利用できません。先にboat_model.pyでトレーニングしてください。")
        return []

    # レースデータ読込
    race_data = _load_today_races(target_date)
    if race_data is None or len(race_data) == 0:
        logger.warning("レースデータなし")
        return []

    signals = []

    # レースごとに処理
    if "race_id" not in race_data.columns and "venue_code" in race_data.columns:
        # real_race_data形式: 1行=1レース、6艇分のデータが横に展開
        from research.boat.boat_model import load_real_data, CLASS_MAP, WIND_DIR_MAP
        # load_real_dataの形式に変換済みのデータを使用
        processed = _process_raw_race_data(race_data)
        if processed is not None:
            race_data = processed

    # レースIDでグループ化して予測
    if "race_id" in race_data.columns:
        race_data = create_features(race_data)

        # 利用可能な特徴量のみ使用
        available_cols = [c for c in FEATURE_COLS if c in race_data.columns]
        missing_cols = [c for c in FEATURE_COLS if c not in race_data.columns]
        for mc in missing_cols:
            race_data[mc] = 0

        X = race_data[FEATURE_COLS].values
        raw_probs, agreement = predict_with_agreement(models, X)
        race_data["raw_prob"] = raw_probs
        race_data["model_agreement"] = agreement

        # レース内で正規化
        race_data = normalize_race_probs(race_data, prob_col="raw_prob")

        # レースごとにバリューベット検出
        for race_id, group in race_data.groupby("race_id"):
            bets = find_value_bets(group, bet_type="win", min_ev=2.00)

            for bet in bets:
                lane = int(bet["lane"])
                venue_code = str(group.iloc[0].get("venue_code", "00")).zfill(2) if "venue_code" in group.columns else "00"
                venue_name = VENUE_NAMES.get(venue_code, f"会場{venue_code}")
                race_number = int(group.iloc[0].get("race_number", 0)) if "race_number" in group.columns else 0
                race_time = str(group.iloc[0].get("race_time", "")) if "race_time" in group.columns else ""

                signal = {
                    "date": target_date,
                    "venue_code": venue_code,
                    "venue_name": venue_name,
                    "race_number": race_number,
                    "race_time": race_time,
                    "lane": lane,
                    "bet_type": "win",
                    "ev": round(float(bet["ev"]), 2),
                    "confidence": round(float(bet["model_prob"]), 4),
                    "odds": round(float(bet["odds"]), 1),
                    "model_prob": round(float(bet["model_prob"]), 4),
                    "agreement": int(bet.get("model_agreement", 0)) if "model_agreement" in bet else 5,
                    "kelly_fraction": round(float(bet.get("kelly_fraction", 0)), 4),
                }
                signals.append(signal)

    signals.sort(key=lambda x: (x["venue_name"], x["race_number"]))
    logger.info(f"シグナル生成完了: {len(signals)}件")

    # シグナルを保存
    _save_daily_signals(target_date, signals)

    return signals


def _process_raw_race_data(raw_df) -> Optional[Any]:
    """
    raw_race_data形式（1行=1レース）を、モデル入力形式（1行=1艇）に変換する。
    """
    import pandas as pd
    from research.boat.boat_model import CLASS_MAP, WIND_DIR_MAP

    rough_venues = {"02", "06", "10", "17", "21", "24"}
    weather_map = {"晴": 0, "曇り": 1, "曇": 1, "雨": 2, "雪": 2, "霧": 1}

    rows = []
    for idx, race in raw_df.iterrows():
        if pd.isna(race.get("date")):
            continue

        venue_code = str(int(race["venue_code"])).zfill(2) if pd.notna(race.get("venue_code")) else "00"
        weather_condition = weather_map.get(str(race.get("weather", "")).strip(), 0)
        wind_speed = float(race.get("wind_speed", 0)) if pd.notna(race.get("wind_speed")) else 0.0
        wave_height = float(race.get("wave_height", 0)) if pd.notna(race.get("wave_height")) else 0.0
        course_type = 1 if venue_code in rough_venues or wave_height >= 5 else 0
        wind_dir_str = str(race.get("wind_direction", "")).strip()
        wind_direction = WIND_DIR_MAP.get(wind_dir_str, 0)

        race_number = int(race.get("race_number", 0)) if pd.notna(race.get("race_number")) else 0
        race_time = str(race.get("race_time", "")) if pd.notna(race.get("race_time")) else ""

        # 勝者特定
        winner_lane = None
        second_lane = None
        for lane in range(1, 7):
            finish_val = race.get(f"lane{lane}_finish")
            if pd.notna(finish_val):
                try:
                    finish = int(float(finish_val))
                    if finish == 1:
                        winner_lane = lane
                    elif finish == 2:
                        second_lane = lane
                except (ValueError, TypeError):
                    pass

        for lane in range(1, 7):
            prefix = f"lane{lane}_"
            racer_class_str = str(race.get(f"{prefix}class", "B1")).strip()
            racer_class = CLASS_MAP.get(racer_class_str, 2)

            row = {
                "race_id": idx,
                "venue_code": venue_code,
                "race_number": race_number,
                "race_time": race_time,
                "lane": lane,
                "racer_class": racer_class,
                "racer_win_rate": float(race.get(f"{prefix}win_rate", 4.5)) if pd.notna(race.get(f"{prefix}win_rate")) else 4.5,
                "racer_place_rate": float(race.get(f"{prefix}place_rate", 30.0)) if pd.notna(race.get(f"{prefix}place_rate")) else 30.0,
                "racer_local_win_rate": float(race.get(f"{prefix}local_win_rate", 4.0)) if pd.notna(race.get(f"{prefix}local_win_rate")) else 4.0,
                "racer_local_2place_rate": float(race.get(f"{prefix}local_2place_rate", 25.0)) if pd.notna(race.get(f"{prefix}local_2place_rate")) else 25.0,
                "motor_2place_rate": float(race.get(f"{prefix}motor_2place", 40.0)) if pd.notna(race.get(f"{prefix}motor_2place")) else 40.0,
                "boat_2place_rate": float(race.get(f"{prefix}boat_2place", 40.0)) if pd.notna(race.get(f"{prefix}boat_2place")) else 40.0,
                "avg_start_timing": float(race.get(f"{prefix}start_timing", 0.17)) if pd.notna(race.get(f"{prefix}start_timing")) else 0.17,
                "flying_count": int(race.get(f"{prefix}flying", 0)) if pd.notna(race.get(f"{prefix}flying")) else 0,
                "racer_weight": float(race.get(f"{prefix}weight", 52.0)) if pd.notna(race.get(f"{prefix}weight")) else 52.0,
                "weather_wind_speed": wind_speed,
                "weather_condition": weather_condition,
                "wave_height": wave_height,
                "course_type": course_type,
                "wind_direction": wind_direction,
                "exhibition_time_raw": float(race.get(f"{prefix}exhibition_time", 6.75)) if pd.notna(race.get(f"{prefix}exhibition_time")) else 6.75,
                "exhibition_start_raw": float(race.get(f"{prefix}exhibition_start", 0.17)) if pd.notna(race.get(f"{prefix}exhibition_start")) else 0.17,
                "win": 1 if lane == winner_lane else 0,
                "place_top2": 1 if lane in (winner_lane, second_lane) else 0,
            }

            # 実オッズがあれば使用
            odds_col = f"{prefix}odds"
            if odds_col in race.index and pd.notna(race.get(odds_col)):
                row["odds"] = float(race[odds_col])

            rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)


def _save_daily_signals(target_date: str, signals: List[Dict[str, Any]]) -> None:
    """日次シグナルをJSONに保存"""
    date_str = target_date.replace("-", "")
    path = DAILY_SIGNALS_DIR / f"{date_str}_signals.json"
    path.write_text(
        json.dumps(signals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"シグナル保存: {path}")


def _load_daily_signals(target_date: str) -> List[Dict[str, Any]]:
    """日次シグナルをJSONから読み込み"""
    date_str = target_date.replace("-", "")
    path = DAILY_SIGNALS_DIR / f"{date_str}_signals.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


# =============================================================
# メッセージフォーマット
# =============================================================

def format_signal_message_owner(
    signals: List[Dict[str, Any]],
    target_date: Optional[str] = None,
) -> str:
    """
    オーナー向けシグナルメッセージ（全詳細付き）
    """
    if target_date is None:
        target_date = date.today().isoformat()

    display_date = target_date.replace("-", "/")
    yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    yesterday_perf = get_performance_summary(yesterday, period="daily")
    monthly_perf = get_performance_summary(target_date, period="monthly")

    lines = []
    lines.append(f"🏁 競艇AI予測シグナル | {display_date}")
    lines.append(f"【オーナー版・詳細データ付き】")
    lines.append("")

    if not signals:
        lines.append("本日の推奨レースはありません。")
    else:
        for s in signals:
            venue_display = s["venue_name"]
            race_num = s["race_number"]
            race_time = s["race_time"]
            lane = s["lane"]
            bet_type_jp = BET_TYPE_JP.get(s["bet_type"], s["bet_type"])
            ev = s["ev"]
            stars = _confidence_stars(ev)

            lines.append(f"📍 {venue_display} {race_num}R ({race_time})")
            lines.append(f"  推奨: {lane}号艇 {bet_type_jp}")
            lines.append(f"  信頼度: {stars} (EV {ev:.2f})")
            lines.append(f"  モデル確率: {s['model_prob']:.1%} / オッズ: {s['odds']:.1f}")
            lines.append(f"  合意度: {s.get('agreement', '-')}/5モデル")
            lines.append(f"  Kelly: {s.get('kelly_fraction', 0):.2%}")
            lines.append("")

    # サマリー
    lines.append("─" * 30)
    lines.append(f"本日の推奨: {len(signals)}レース")

    if yesterday_perf["total_bets"] > 0:
        y_w = yesterday_perf["wins"]
        y_l = yesterday_perf["losses"]
        y_pnl = yesterday_perf["net_profit"]
        lines.append(f"昨日の成績: {y_w}勝{y_l}敗 / {y_pnl:+,.0f}円")

    if monthly_perf["total_bets"] > 0:
        m_pnl = monthly_perf["net_profit"]
        m_pf = monthly_perf["pf"]
        lines.append(f"今月累計: {m_pnl:+,.0f}円 (PF {m_pf:.2f})")

    return "\n".join(lines)


def format_signal_message_subscriber(
    signals: List[Dict[str, Any]],
    target_date: Optional[str] = None,
) -> str:
    """
    購読者向けシグナルメッセージ（競争優位性を保護、内部詳細なし）
    """
    if target_date is None:
        target_date = date.today().isoformat()

    display_date = target_date.replace("-", "/")
    yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    yesterday_perf = get_performance_summary(yesterday, period="daily")
    monthly_perf = get_performance_summary(target_date, period="monthly")

    lines = []
    lines.append(f"🏁 競艇AI予測シグナル | {display_date}")
    lines.append("")

    if not signals:
        lines.append("本日の推奨レースはありません。")
    else:
        for s in signals:
            venue_display = s["venue_name"]
            race_num = s["race_number"]
            race_time = s["race_time"]
            lane = s["lane"]
            bet_type_jp = BET_TYPE_JP.get(s["bet_type"], s["bet_type"])
            ev = s["ev"]
            stars = _confidence_stars(ev)

            lines.append(f"📍 {venue_display} {race_num}R ({race_time})")
            lines.append(f"  推奨: {lane}号艇 {bet_type_jp}")
            lines.append(f"  信頼度: {stars} (EV {ev:.2f})")
            lines.append("")

    # サマリー (内部モデル情報なし)
    lines.append("─" * 30)
    lines.append(f"本日のAI推奨: {len(signals)}レース")

    if yesterday_perf["total_bets"] > 0:
        y_w = yesterday_perf["wins"]
        y_l = yesterday_perf["losses"]
        y_pnl = yesterday_perf["net_profit"]
        lines.append(f"昨日の成績: {y_w}勝{y_l}敗 / {y_pnl:+,.0f}円")

    if monthly_perf["total_bets"] > 0:
        m_pnl = monthly_perf["net_profit"]
        m_pf = monthly_perf["pf"]
        lines.append(f"今月累計: {m_pnl:+,.0f}円 (PF {m_pf:.2f})")

    return "\n".join(lines)


def format_daily_results(target_date: Optional[str] = None) -> str:
    """日次結果サマリーメッセージを生成"""
    if target_date is None:
        target_date = date.today().isoformat()

    display_date = target_date.replace("-", "/")
    perf = get_performance_summary(target_date, period="daily")
    monthly = get_performance_summary(target_date, period="monthly")

    lines = []
    lines.append(f"📊 本日の結果 | {display_date}")
    lines.append("")

    if perf["total_bets"] == 0:
        lines.append("本日のベットはありませんでした。")
        return "\n".join(lines)

    wins = perf["wins"]
    losses = perf["losses"]
    win_rate = perf["win_rate"]
    net_profit = perf["net_profit"]

    lines.append(f"成績: {wins}勝{losses}敗 (的中率 {win_rate:.1%})")
    lines.append(f"損益: {net_profit:+,.0f}円")
    lines.append("")

    # 個別レース結果
    signals = _load_daily_signals(target_date)
    if PERFORMANCE_CSV.exists():
        with open(PERFORMANCE_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            day_rows = [r for r in reader if r["date"] == target_date]

        for row in day_rows:
            result_mark = "✅" if row["result"] == "WIN" else "❌" if row["result"] == "LOSE" else "⏳"
            venue = row["race_venue"]
            race_num = row["race_number"]
            lane = row["lane"]
            profit = float(row["profit"]) if row["profit"] else 0
            lines.append(f"  {result_mark} {venue} {race_num}R {lane}号艇 → {profit:+,.0f}円")

    lines.append("")
    lines.append("─" * 30)
    lines.append(f"今月累計: {monthly['net_profit']:+,.0f}円 (PF {monthly['pf']:.2f})")
    lines.append(f"今月通算: {monthly['wins']}勝{monthly['losses']}敗 (的中率 {monthly['win_rate']:.1%})")

    return "\n".join(lines)


def format_weekly_report(target_date: Optional[str] = None) -> str:
    """週次パフォーマンスレポートを生成"""
    if target_date is None:
        target_date = date.today().isoformat()

    display_date = target_date.replace("-", "/")
    weekly = get_performance_summary(target_date, period="weekly")
    monthly = get_performance_summary(target_date, period="monthly")

    lines = []
    lines.append(f"📈 週次パフォーマンスレポート")
    lines.append(f"集計日: {display_date}")
    lines.append("")

    lines.append("【今週の成績】")
    if weekly["total_bets"] > 0:
        lines.append(f"  ベット数: {weekly['total_bets']}")
        lines.append(f"  成績: {weekly['wins']}勝{weekly['losses']}敗")
        lines.append(f"  的中率: {weekly['win_rate']:.1%}")
        lines.append(f"  損益: {weekly['net_profit']:+,.0f}円")
        lines.append(f"  PF: {weekly['pf']:.2f}")
    else:
        lines.append("  ベットなし")

    lines.append("")
    lines.append("【今月の成績】")
    if monthly["total_bets"] > 0:
        lines.append(f"  ベット数: {monthly['total_bets']}")
        lines.append(f"  成績: {monthly['wins']}勝{monthly['losses']}敗")
        lines.append(f"  的中率: {monthly['win_rate']:.1%}")
        lines.append(f"  損益: {monthly['net_profit']:+,.0f}円")
        lines.append(f"  PF: {monthly['pf']:.2f}")
    else:
        lines.append("  ベットなし")

    lines.append("")
    lines.append("─" * 30)
    lines.append(f"累計損益: {monthly['cumulative_profit']:+,.0f}円")

    return "\n".join(lines)


# =============================================================
# メインパイプライン
# =============================================================

def run_morning_signals(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    朝のシグナル配信パイプライン。
    1. シグナル生成
    2. パフォーマンスCSVに記録
    3. 両チャンネルに配信
    """
    if target_date is None:
        target_date = date.today().isoformat()

    logger.info("=" * 60)
    logger.info(f"  朝のシグナル配信開始: {target_date}")
    logger.info("=" * 60)

    result = {
        "timestamp": datetime.now().isoformat(),
        "date": target_date,
        "signals": [],
        "errors": [],
    }

    # 1. シグナル生成
    try:
        signals = generate_signals(target_date)
        result["signals"] = signals
        result["signal_count"] = len(signals)
    except Exception as e:
        error_msg = f"シグナル生成エラー: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        return result

    if not signals:
        logger.info("推奨シグナルなし")
        # データなしでも通知
        msg = f"🏁 競艇AI予測シグナル | {target_date.replace('-', '/')}\n\n本日の推奨レースはありません。"
        send_to_both(msg, msg)
        return result

    # 2. パフォーマンスCSVに記録
    _init_performance_csv()
    for s in signals:
        record_signal(
            signal_date=target_date,
            venue=s["venue_name"],
            race_number=s["race_number"],
            race_time=s["race_time"],
            bet_type=s["bet_type"],
            lane=s["lane"],
            ev=s["ev"],
            confidence=s["confidence"],
            odds=s["odds"],
        )

    # 3. 両チャンネルに配信
    owner_msg = format_signal_message_owner(signals, target_date)
    subscriber_msg = format_signal_message_subscriber(signals, target_date)

    send_results = send_to_both(owner_msg, subscriber_msg)
    result["send_results"] = send_results

    logger.info(f"シグナル配信完了: {len(signals)}件")
    return result


def run_evening_results(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    夕方の結果配信パイプライン。
    1. 結果の読み込み・更新
    2. 結果サマリー生成
    3. 両チャンネルに配信
    """
    if target_date is None:
        target_date = date.today().isoformat()

    logger.info("=" * 60)
    logger.info(f"  夕方の結果配信開始: {target_date}")
    logger.info("=" * 60)

    result = {
        "timestamp": datetime.now().isoformat(),
        "date": target_date,
        "errors": [],
    }

    # 結果サマリー生成
    try:
        msg = format_daily_results(target_date)
        send_to_both(msg, msg)
        result["message_sent"] = True
    except Exception as e:
        error_msg = f"結果配信エラー: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)

    return result


def run_weekly_report(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    週次レポート配信パイプライン（毎週日曜）。
    """
    if target_date is None:
        target_date = date.today().isoformat()

    logger.info("=" * 60)
    logger.info(f"  週次レポート配信: {target_date}")
    logger.info("=" * 60)

    result = {
        "timestamp": datetime.now().isoformat(),
        "date": target_date,
        "errors": [],
    }

    try:
        msg = format_weekly_report(target_date)
        send_to_both(msg, msg)
        result["message_sent"] = True
    except Exception as e:
        error_msg = f"週次レポート配信エラー: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)

    return result


# =============================================================
# スケジュール実行
# =============================================================

def run_scheduled(
    morning_time: str = "09:00",
    evening_time: str = "21:00",
    weekly_time: str = "21:30",
) -> None:
    """
    スケジュール実行を開始する。

    Args:
        morning_time: 朝シグナル配信時刻
        evening_time: 夕方結果配信時刻
        weekly_time: 週次レポート配信時刻（日曜日）
    """
    try:
        import schedule
    except ImportError:
        logger.error("scheduleライブラリが必要です: pip install schedule")
        return

    import time

    logger.info("=" * 60)
    logger.info("  競艇AIシグナル配信サービス開始")
    logger.info(f"  朝シグナル: 毎日 {morning_time}")
    logger.info(f"  結果配信: 毎日 {evening_time}")
    logger.info(f"  週次レポート: 毎週日曜 {weekly_time}")
    logger.info("=" * 60)

    # 朝シグナル（毎日）
    schedule.every().day.at(morning_time).do(run_morning_signals)

    # 結果配信（毎日）
    schedule.every().day.at(evening_time).do(run_evening_results)

    # 週次レポート（毎週日曜）
    schedule.every().sunday.at(weekly_time).do(run_weekly_report)

    logger.info("スケジュール登録完了。Ctrl+Cで終了します。")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("シグナル配信サービスを終了しました。")


# =============================================================
# 状態管理
# =============================================================

def _load_state() -> Dict[str, Any]:
    """サービスの状態を読み込む"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {
        "last_morning_run": None,
        "last_evening_run": None,
        "last_weekly_run": None,
        "total_signals_sent": 0,
    }


def _save_state(state: Dict[str, Any]) -> None:
    """サービスの状態を保存する"""
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================
# CLIエントリーポイント
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="競艇AIシグナル配信サービス")
    parser.add_argument(
        "--action",
        choices=["morning", "evening", "weekly", "schedule", "status", "test"],
        default="morning",
        help="実行アクション (default: morning)",
    )
    parser.add_argument("--date", type=str, default=None, help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--morning-time", type=str, default="09:00", help="朝シグナル配信時刻")
    parser.add_argument("--evening-time", type=str, default="21:00", help="結果配信時刻")
    parser.add_argument("--weekly-time", type=str, default="21:30", help="週次レポート時刻")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  競艇AIシグナル配信: action={args.action}")
    logger.info("=" * 60)

    if args.action == "morning":
        result = run_morning_signals(args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "evening":
        result = run_evening_results(args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "weekly":
        result = run_weekly_report(args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "schedule":
        run_scheduled(
            morning_time=args.morning_time,
            evening_time=args.evening_time,
            weekly_time=args.weekly_time,
        )

    elif args.action == "status":
        state = _load_state()
        perf = get_performance_summary(period="monthly")
        print("=== サービス状態 ===")
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print("\n=== 今月のパフォーマンス ===")
        print(json.dumps(perf, ensure_ascii=False, indent=2, default=str))

    elif args.action == "test":
        # テスト: メッセージフォーマットの確認
        test_signals = [
            {
                "date": date.today().isoformat(),
                "venue_code": "24",
                "venue_name": "大村",
                "race_number": 4,
                "race_time": "13:30",
                "lane": 1,
                "bet_type": "win",
                "ev": 2.45,
                "confidence": 0.42,
                "odds": 2.8,
                "model_prob": 0.42,
                "agreement": 5,
                "kelly_fraction": 0.035,
            },
            {
                "date": date.today().isoformat(),
                "venue_code": "12",
                "venue_name": "住之江",
                "race_number": 7,
                "race_time": "15:00",
                "lane": 4,
                "bet_type": "win",
                "ev": 3.12,
                "confidence": 0.28,
                "odds": 11.0,
                "model_prob": 0.28,
                "agreement": 4,
                "kelly_fraction": 0.022,
            },
        ]

        print("=== オーナー向けメッセージ ===")
        print(format_signal_message_owner(test_signals))
        print()
        print("=== 購読者向けメッセージ ===")
        print(format_signal_message_subscriber(test_signals))
        print()
        print("=== 結果サマリー ===")
        print(format_daily_results())
        print()
        print("=== 週次レポート ===")
        print(format_weekly_report())
