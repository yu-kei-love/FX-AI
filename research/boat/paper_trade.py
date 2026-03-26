# ===========================================
# paper_trade.py
# ボートレース ペーパートレード
#
# 使い方:
#   python research/boat/paper_trade.py --today        # 今日のレースを予測
#   python research/boat/paper_trade.py --backfill 30  # 過去30日分をバックテスト
#   python research/boat/paper_trade.py --loop          # 毎時自動予測ループ
# ===========================================

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.boat.boat_model import (
    load_real_data, create_features, predict_proba, predict_place_proba,
    predict_with_agreement, normalize_race_probs, find_value_bets, load_models,
    FEATURE_COLS, CLASS_NAMES, APPROX_ODDS,
)
from research.boat.fetch_real_data import fetch_day_results, VENUES

# ===== 定数 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "boat"
TRADE_LOG = DATA_DIR / "paper_trade_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DATA_DIR / "paper_trade.log"), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_or_create_log():
    """ペーパートレードログを読み込む。なければ新規作成。"""
    if TRADE_LOG.exists():
        return pd.read_csv(TRADE_LOG, encoding="utf-8-sig")
    return pd.DataFrame(columns=[
        "timestamp", "date", "venue", "race_no", "lane", "bet_type",
        "model_prob", "odds", "ev", "bet_amount", "result",
        "payout", "pnl", "cumulative_pnl",
    ])


def save_log(df):
    """ペーパートレードログを保存する。"""
    df.to_csv(TRADE_LOG, index=False, encoding="utf-8-sig")


def predict_and_bet_day(date_str, models, bet_amount=500, strategy="win", dry_run=False):
    """
    指定日のレースを予測し、バリューベットを記録する。

    Args:
        date_str: "20260325" 形式の日付
        models: 訓練済みモデル
        bet_amount: 1ベット金額
        strategy: "win", "exacta", "quinella"
        dry_run: True なら結果を取得せずに予測のみ

    Returns:
        bets: ベットリスト
    """
    logger.info(f"=== {date_str} のレース予測 ===")

    # 全会場のレースデータを取得
    all_bets = []

    for venue_code in VENUES:
        try:
            # レース結果を取得（結果があれば既に終了したレース）
            fetch_day_results(venue_code, date_str, skip_odds=True)
        except Exception:
            pass  # 開催なしの場合

    # real_race_data.csv からこの日のデータを取得
    real_path = DATA_DIR / "real_race_data.csv"
    if not real_path.exists():
        logger.warning("real_race_data.csv が見つかりません")
        return []

    raw = pd.read_csv(real_path, encoding="utf-8-sig")
    day_data = raw[raw["date"] == int(date_str)]

    if len(day_data) == 0:
        logger.info(f"  {date_str} のレースデータなし")
        return []

    logger.info(f"  {len(day_data)}レース検出")

    # load_real_data の処理をこの日のデータだけに適用
    from research.boat.boat_model import CLASS_MAP
    weather_map = {"晴": 0, "曇り": 1, "曇": 1, "雨": 2, "雪": 2, "霧": 1}
    rough_venues = {"02", "06", "10", "17", "21", "24"}

    rows = []
    for idx, race in day_data.iterrows():
        if pd.isna(race.get("date")):
            continue
        race_id = idx
        venue_code = str(int(race["venue_code"])).zfill(2) if pd.notna(race.get("venue_code")) else "00"
        venue_name = race.get("venue_name", VENUES.get(venue_code, "不明"))

        weather_condition = weather_map.get(str(race.get("weather", "")).strip(), 0)
        wind_speed = float(race.get("wind_speed", 0)) if pd.notna(race.get("wind_speed")) else 0.0
        wave_height = float(race.get("wave_height", 0)) if pd.notna(race.get("wave_height")) else 0.0
        course_type = 1 if venue_code in rough_venues or wave_height >= 5 else 0

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
            racer_class = CLASS_MAP.get(str(race.get(f"{prefix}class", "B1")).strip(), 2)
            win_rate = float(race.get(f"{prefix}win_rate", 4.0)) if pd.notna(race.get(f"{prefix}win_rate")) else 4.0
            place_rate = float(race.get(f"{prefix}2place_rate", 20.0)) if pd.notna(race.get(f"{prefix}2place_rate")) else 20.0
            motor_rate = float(race.get(f"{prefix}motor_2rate", 40.0)) if pd.notna(race.get(f"{prefix}motor_2rate")) else 40.0
            boat_rate = float(race.get(f"{prefix}boat_2rate", 40.0)) if pd.notna(race.get(f"{prefix}boat_2rate")) else 40.0
            start_timing = float(race.get(f"{prefix}start_timing", 0.15)) if pd.notna(race.get(f"{prefix}start_timing")) else 0.15
            flying = int(float(race.get(f"{prefix}flying_count", 0))) if pd.notna(race.get(f"{prefix}flying_count")) else 0
            local_wr = float(race.get(f"{prefix}local_win_rate", win_rate)) if pd.notna(race.get(f"{prefix}local_win_rate")) else win_rate

            three_place_rate = float(race.get(f"{prefix}3place_rate", place_rate * 1.3)) if pd.notna(race.get(f"{prefix}3place_rate")) else place_rate * 1.3
            local_2place = float(race.get(f"{prefix}local_2place_rate", local_wr * 5)) if pd.notna(race.get(f"{prefix}local_2place_rate")) else local_wr * 5
            late_cnt = int(float(race.get(f"{prefix}late_count", 0))) if pd.notna(race.get(f"{prefix}late_count")) else 0
            weight = float(race.get(f"{prefix}weight", 52.0)) if pd.notna(race.get(f"{prefix}weight")) else 52.0

            odds_col = f"odds_{lane}"
            odds = float(race.get(odds_col, 5.0)) if pd.notna(race.get(odds_col)) else 5.0

            rows.append({
                "race_id": race_id,
                "race_date": pd.to_datetime(date_str, format="%Y%m%d"),
                "venue_code": venue_code,
                "venue_name": venue_name,
                "race_no": race.get("race_no", 0),
                "lane": lane,
                "racer_class": racer_class,
                "racer_win_rate": win_rate,
                "racer_place_rate": place_rate,
                "racer_3place_rate": three_place_rate,
                "racer_local_win_rate": local_wr,
                "racer_local_2place_rate": local_2place,
                "motor_2place_rate": motor_rate,
                "boat_2place_rate": boat_rate,
                "avg_start_timing": start_timing,
                "flying_count": flying,
                "late_count": late_cnt,
                "racer_weight": weight,
                "weather_wind_speed": wind_speed,
                "weather_condition": weather_condition,
                "wave_height": wave_height,
                "course_type": course_type,
                "odds": odds,
                "win": 1 if lane == winner_lane else 0,
                "place_top2": 1 if lane in (winner_lane, second_lane) else 0,
            })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df = create_features(df)

    # 予測
    import xgboost as xgb
    X = df[FEATURE_COLS].values
    df["raw_prob"] = predict_proba(models, X)
    df = normalize_race_probs(df)

    # v3.4: モデル一致度を計算 (agreement >= 4 のみベット対象)
    _, agreement = predict_with_agreement(models, X)
    df["model_agreement"] = agreement

    # Place prediction (v3.1)
    if "lgb_place" in models:
        df["raw_place_prob"] = predict_place_proba(models, X)
        race_place_sums = df.groupby("race_id")["raw_place_prob"].transform("sum")
        df["pred_place_prob"] = df["raw_place_prob"] / race_place_sums * 2.0

    # バリューベット検出
    bets = []
    for rid in df["race_id"].unique():
        race_data = df[df["race_id"] == rid]
        race_bets = find_value_bets(race_data, bet_type=strategy)

        for bet in race_bets:
            race_row = race_data.iloc[0]
            venue_name = race_row.get("venue_name", "不明")
            race_no = race_row.get("race_no", "?")

            won = bet["win"] == 1
            payout = bet_amount * bet["odds"] if won else 0
            pnl = payout - bet_amount

            bet_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": date_str,
                "venue": venue_name,
                "race_no": int(race_no) if pd.notna(race_no) else 0,
                "lane": bet["lane"],
                "bet_type": strategy,
                "model_prob": round(bet["model_prob"], 4),
                "odds": bet["odds"],
                "ev": round(bet["ev"], 3),
                "bet_amount": bet_amount,
                "result": "WIN" if won else "LOSE",
                "payout": payout,
                "pnl": pnl,
            }
            bets.append(bet_record)

            status = "○" if won else "×"
            logger.info(
                f"  {status} {venue_name} {int(race_no) if pd.notna(race_no) else '?'}R "
                f"{bet['lane']}号艇 odds={bet['odds']:.1f} "
                f"prob={bet['model_prob']:.3f} ev={bet['ev']:.2f} "
                f"PnL={pnl:+.0f}円"
            )

    if bets:
        total_pnl = sum(b["pnl"] for b in bets)
        n_wins = sum(1 for b in bets if b["result"] == "WIN")
        logger.info(f"  日計: {len(bets)}ベット, {n_wins}的中, PnL={total_pnl:+,.0f}円")

    return bets


def run_backfill(days=30, strategy="win", bet_amount=500):
    """
    過去N日分のペーパートレードバックフィルを実行する。
    既存のreal_race_data.csvから直接読み込み（ウェブ取得なし）。
    """
    logger.info(f"=== ペーパートレード バックフィル ({days}日間) ===")

    try:
        models = load_models()
        logger.info("モデルロード成功")
    except Exception as e:
        logger.error(f"モデルロード失敗: {e}")
        logger.info("モデルを再訓練します...")
        from research.boat.boat_model import run_pipeline
        results = run_pipeline()
        models = results["models"]

    # real_race_data.csv全体をload_real_dataで変換
    df_all = load_real_data()
    if df_all is None or len(df_all) == 0:
        logger.error("リアルデータなし")
        return

    df_all = create_features(df_all)
    df_all = df_all.sort_values("race_date").reset_index(drop=True)

    # 日付でフィルタ
    dates = sorted(df_all["race_date"].dt.strftime("%Y%m%d").unique())
    if len(dates) < days:
        target_dates = dates
    else:
        target_dates = dates[-days:]

    logger.info(f"対象期間: {target_dates[0]} ~ {target_dates[-1]} ({len(target_dates)}日)")

    # 予測
    import xgboost as xgb
    X_all = df_all[FEATURE_COLS].values
    df_all["raw_prob"] = predict_proba(models, X_all)
    df_all = normalize_race_probs(df_all)

    # v3.4: モデル一致度を計算
    _, agreement = predict_with_agreement(models, X_all)
    df_all["model_agreement"] = agreement

    # Place prediction (v3.1)
    if "lgb_place" in models:
        df_all["raw_place_prob"] = predict_place_proba(models, X_all)
        race_place_sums = df_all.groupby("race_id")["raw_place_prob"].transform("sum")
        df_all["pred_place_prob"] = df_all["raw_place_prob"] / race_place_sums * 2.0

    # 日付ごとにバリューベット検出
    all_bets = []
    for date_str in target_dates:
        day_mask = df_all["race_date"].dt.strftime("%Y%m%d") == date_str
        day_df = df_all[day_mask]

        day_bets = []
        for rid in day_df["race_id"].unique():
            race_data = day_df[day_df["race_id"] == rid]
            bets = find_value_bets(race_data, bet_type=strategy)

            for bet in bets:
                race_row = race_data.iloc[0]
                venue_name = race_row.get("venue_code", "??")
                race_no = race_row.get("race_no", 0)

                won = bet["win"] == 1
                payout = bet_amount * bet["odds"] if won else 0
                pnl = payout - bet_amount

                day_bets.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "date": date_str,
                    "venue": venue_name,
                    "race_no": int(race_no) if pd.notna(race_no) else 0,
                    "lane": bet["lane"],
                    "bet_type": strategy,
                    "model_prob": round(bet["model_prob"], 4),
                    "odds": bet["odds"],
                    "ev": round(bet["ev"], 3),
                    "bet_amount": bet_amount,
                    "result": "WIN" if won else "LOSE",
                    "payout": payout,
                    "pnl": pnl,
                })

        if day_bets:
            n_wins = sum(1 for b in day_bets if b["result"] == "WIN")
            day_pnl = sum(b["pnl"] for b in day_bets)
            logger.info(f"  {date_str}: {len(day_bets)}ベット, {n_wins}的中, PnL={day_pnl:+,.0f}円")
            all_bets.extend(day_bets)

    if not all_bets:
        logger.info("ベットなし")
        return

    df = pd.DataFrame(all_bets)
    df["cumulative_pnl"] = df["pnl"].cumsum()
    save_log(df)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"バックフィル結果 ({strategy})")
    logger.info(f"{'=' * 50}")

    n_bets = len(df)
    n_wins = (df["result"] == "WIN").sum()
    total_invest = df["bet_amount"].sum()
    total_payout = df["payout"].sum()
    total_pnl = df["pnl"].sum()
    recovery = total_payout / total_invest if total_invest > 0 else 0
    hit_rate = n_wins / n_bets if n_bets > 0 else 0

    cum_pnl = df["cumulative_pnl"]
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    max_dd = drawdown.max()

    logger.info(f"  ベット数: {n_bets}")
    logger.info(f"  的中数: {n_wins}")
    logger.info(f"  的中率: {hit_rate:.1%}")
    logger.info(f"  投資額: {total_invest:,.0f}円")
    logger.info(f"  回収額: {total_payout:,.0f}円")
    logger.info(f"  損益: {total_pnl:+,.0f}円")
    logger.info(f"  回収率: {recovery:.3f} ({recovery * 100:.1f}%)")
    logger.info(f"  最大DD: {max_dd:,.0f}円")
    logger.info(f"  最終累計: {cum_pnl.iloc[-1]:+,.0f}円")
    logger.info(f"\nログ保存: {TRADE_LOG}")


def run_today(strategy="win", bet_amount=500):
    """今日のレースを予測・記録する。"""
    today = datetime.now().strftime("%Y%m%d")
    logger.info(f"=== 今日のレース予測 ({today}) ===")

    try:
        models = load_models()
    except Exception:
        logger.error("モデルロード失敗。先に boat_model.py を実行してください。")
        return

    bets = predict_and_bet_day(today, models, bet_amount, strategy)

    if bets:
        # 既存ログに追記
        existing = load_or_create_log()
        new_df = pd.DataFrame(bets)

        if len(existing) > 0:
            last_cum = existing["cumulative_pnl"].iloc[-1]
        else:
            last_cum = 0

        new_df["cumulative_pnl"] = last_cum + new_df["pnl"].cumsum()
        combined = pd.concat([existing, new_df], ignore_index=True)
        save_log(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ボートレース ペーパートレード")
    parser.add_argument("--today", action="store_true", help="今日のレースを予測")
    parser.add_argument("--backfill", type=int, default=0, help="過去N日分のバックフィル")
    parser.add_argument("--strategy", default="win", choices=["win", "exacta", "quinella"])
    parser.add_argument("--bet", type=int, default=500, help="1ベット金額")
    args = parser.parse_args()

    if args.backfill > 0:
        run_backfill(days=args.backfill, strategy=args.strategy, bet_amount=args.bet)
    elif args.today:
        run_today(strategy=args.strategy, bet_amount=args.bet)
    else:
        parser.print_help()
