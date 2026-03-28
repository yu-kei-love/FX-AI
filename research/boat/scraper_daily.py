# ===========================================
# scraper_daily.py
# ボートレース 毎朝08:00 事前取得バッチ
#
# 取得内容：
#   - 当日の全開催会場・全レースの出走表
#   - 選手情報・モーター・ボート
#   - 天候・環境情報
#   - レースグレード・種別
#
# タスクスケジューラ設定：
#   毎朝 08:00 実行
#   python research/boat/scraper_daily.py
# ===========================================

import re
import sys
import time
import logging
import random
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.boat.db_manager import (
    init_db, insert_race, insert_entries_batch,
    log_fetch_result, VENUE_CODE_TO_ID, get_stats
)
from research.boat.scraper_historical import (
    VENUES, HEADERS, _get_soup, _polite_sleep,
    _safe_float, _safe_int, _fw2int, CLASS_MAP,
    WIND_DIR_SIN_COS, _parse_entry_row, _parse_weather,
    fetch_race_entry_info,
)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race"

LOG_FILE = PROJECT_ROOT / "data" / "boat" / "scraper_daily.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================
# 当日の開催会場を取得
# =============================================================

def fetch_todays_venues(date_str=None):
    """
    当日（または指定日）の開催会場リストを取得する。

    Returns:
        list[str]: 会場コードのリスト ["01", "02", ...]
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        logger.warning("トップページ取得失敗 -> 全会場を試みる")
        return list(VENUES.keys())

    venue_codes = []
    try:
        # 開催会場リンクを探す
        links = soup.find_all("a", href=True)
        for a in links:
            href = a["href"]
            m = re.search(r"jcd=(\d{2})", href)
            if m:
                code = m.group(1)
                if code in VENUES and code not in venue_codes:
                    venue_codes.append(code)
    except Exception as e:
        logger.warning(f"会場リスト取得失敗: {e}")
        return list(VENUES.keys())

    if not venue_codes:
        logger.info("当日開催会場が検出されなかった -> 全会場を試みる")
        return list(VENUES.keys())

    logger.info(f"当日開催会場: {[VENUES.get(c, c) for c in venue_codes]}")
    return venue_codes


def fetch_race_schedule_info(venue_code, date_str):
    """
    出走表トップページからレース一覧・時刻・グレード・種別を取得する。

    Returns:
        list[dict]: [{race_no, race_time, grade, race_type}, ...]
    """
    url = f"{BASE_URL}/raceindex?jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    schedule = []
    try:
        # レース一覧テーブルを探す
        rows = soup.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            if not tds:
                continue
            text = row.get_text(strip=True)
            # "1R", "2R" ... "12R" のパターンを探す
            m = re.search(r"(\d{1,2})R", text)
            if m:
                race_no = int(m.group(1))
                # 時刻パターン "14:30" を探す
                time_m = re.search(r"(\d{2}:\d{2})", text)
                race_time = time_m.group(1) if time_m else None
                # グレード
                grade = None
                for g in ["SG", "G1", "G2", "G3"]:
                    if g in text:
                        grade = g
                        break
                if grade is None:
                    grade = "一般"
                # 種別
                race_type = None
                for t in ["優勝戦", "準優勝戦", "準優", "予選"]:
                    if t in text:
                        race_type = t.replace("準優勝戦", "準優")
                        break

                schedule.append({
                    "race_no": race_no,
                    "race_time": race_time,
                    "grade": grade,
                    "race_type": race_type,
                })
    except Exception as e:
        logger.debug(f"スケジュールパース失敗 {venue_code}: {e}")

    return schedule


# =============================================================
# メイン処理
# =============================================================

def run_daily_scrape(date_str=None, db_path=None):
    """
    当日分の出走表を全会場取得してDBに保存する。

    Parameters:
        date_str (str|None): YYYYMMDD（Noneなら今日）
        db_path  (Path|None): DBパス
    """
    init_db(db_path)

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    logger.info(f"=== 日次バッチ開始 {date_str} ===")

    # 当日の開催会場を取得
    _polite_sleep()
    venues = fetch_todays_venues(date_str)

    total_races = 0
    now_str = datetime.now().isoformat()

    for venue_code in venues:
        venue_id = VENUE_CODE_TO_ID.get(venue_code, 0)
        venue_name = VENUES.get(venue_code, venue_code)

        logger.info(f"  [{venue_name}] 処理中...")

        # レーススケジュール情報（時刻・グレード・種別）
        _polite_sleep()
        schedule = fetch_race_schedule_info(venue_code, date_str)
        schedule_map = {s["race_no"]: s for s in schedule}

        races_in_venue = 0
        for race_no in range(1, 13):
            _polite_sleep()
            entries_raw = fetch_race_entry_info(venue_code, race_no, date_str)

            if entries_raw is None:
                if race_no == 1:
                    logger.info(f"  [{venue_name}] 開催なし")
                    break
                continue

            race_id = f"{date_str}_{venue_id:02d}_{race_no:02d}"
            sched = schedule_map.get(race_no, {})

            race_data = {
                "race_id": race_id,
                "date": date_str,
                "venue_id": venue_id,
                "race_no": race_no,
                "grade": sched.get("grade"),
                "race_type": sched.get("race_type"),
                "race_time": sched.get("race_time"),
                "weather": None,
                "wind_speed": None,
                "wind_direction": None,
                "wind_direction_sin": None,
                "wind_direction_cos": None,
                "wave_height": None,
                "rain_amount": None,
                "air_pressure": None,
                "temperature": None,
                "water_type": None,
                "tide_level": None,
                "created_at": now_str,
            }
            insert_race(race_data, db_path)

            # エントリ保存（まだ結果・course_takenは不明）
            entries_to_insert = []
            for raw in entries_raw:
                entry = {
                    "race_id": race_id,
                    "lane": raw["lane"],
                    "course_taken": None,        # スタート展示後に更新
                    "racer_id": raw.get("racer_id"),
                    "racer_name": raw.get("racer_name"),
                    "racer_class": raw.get("racer_class"),
                    "racer_weight": raw.get("racer_weight"),
                    "racer_age": None,
                    "national_win_rate":    raw.get("national_win_rate"),
                    "national_2place_rate": raw.get("national_2place_rate"),
                    "national_3place_rate": raw.get("national_3place_rate"),
                    "local_win_rate":    raw.get("local_win_rate"),
                    "local_2place_rate": raw.get("local_2place_rate"),
                    "avg_start_timing": raw.get("avg_start_timing"),
                    "flying_count": raw.get("flying_count"),
                    "late_count":   raw.get("late_count"),
                    "motor_no":          raw.get("motor_no"),
                    "motor_2place_rate": raw.get("motor_2place_rate"),
                    "boat_no":          raw.get("boat_no"),
                    "boat_2place_rate": raw.get("boat_2place_rate"),
                    "exhibition_time": None,
                    "exhibition_st": None,
                    "finish": None,             # レース後に更新
                    "created_at": now_str,
                }
                entries_to_insert.append(entry)

            insert_entries_batch(entries_to_insert, db_path)
            races_in_venue += 1
            logger.info(f"    R{race_no:02d}: {len(entries_to_insert)}艇 保存 "
                        f"(時刻={sched.get('race_time', '?')}, "
                        f"グレード={sched.get('grade', '?')})")

        log_fetch_result(date_str, venue_id,
                         "success" if races_in_venue > 0 else "no_race",
                         races_in_venue, db_path=db_path)
        total_races += races_in_venue

    stats = get_stats(db_path)
    logger.info(f"=== 日次バッチ完了: {total_races}レース取得 ===")
    logger.info(f"  DB総レース数: {stats['races']}")
    logger.info(f"  DB総エントリ数: {stats['entries']}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ボートレース日次バッチ")
    parser.add_argument("--date", default=None, help="日付 YYYYMMDD（省略時は今日）")
    args = parser.parse_args()
    run_daily_scrape(args.date)


if __name__ == "__main__":
    main()
