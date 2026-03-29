# ===========================================
# scraper_realtime.py
# ボートレース リアルタイム直前取得スクリプト
#
# 取得タイミング（7回）：
#   120min: 単勝オッズ（ベースライン）
#    60min: 単勝・2連単・2連複オッズ
#    30min: 全券種オッズ
#    15min: 進入コース★・展示タイム・全券種オッズ  ← 最重要
#     5min: 全券種オッズ（最終市場の総意）
#     1min: 全券種オッズ（急変検知）
#    final: 全券種オッズ（確定値）
#
# 使い方:
#   python scraper_realtime.py --venue 02 --race 8 --timing 15min
#   python scraper_realtime.py --venue 02 --race 8 --timing final
# ===========================================

import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.boat.db_manager import (
    init_db, insert_odds_batch, get_conn,
    VENUE_CODE_TO_ID, DB_PATH
)
from research.boat.scraper_historical import (
    VENUES, HEADERS, _get_soup, _polite_sleep,
    _safe_float, _safe_int,
)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race"

LOG_FILE = PROJECT_ROOT / "data" / "boat" / "scraper_realtime.log"
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

VALID_TIMINGS = ["120min", "60min", "30min", "15min", "5min", "1min", "final"]


# =============================================================
# オッズ取得
# =============================================================

def fetch_win_odds_realtime(venue_code, race_no, date_str):
    """単勝オッズを取得する。"""
    # 単勝オッズは oddstf ではなく oddsw ページにある
    url = f"{BASE_URL}/oddsw1?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    results = []
    try:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    lane = _safe_int(tds[0].get_text(strip=True))
                    if lane and 1 <= lane <= 6:
                        odds_val = _safe_float(tds[-1].get_text(strip=True))
                        if odds_val and odds_val > 1.0:
                            results.append({
                                "combination": str(lane),
                                "odds_value": odds_val,
                            })
    except Exception as e:
        logger.debug(f"単勝オッズ取得失敗: {e}")

    return results


def fetch_exacta_odds_realtime(venue_code, race_no, date_str):
    """2連単オッズを取得する。"""
    url = f"{BASE_URL}/odds2tf?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    results = []
    try:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    combo_text = tds[0].get_text(strip=True)
                    m = re.match(r"([1-6])-([1-6])", combo_text)
                    if m:
                        combo = f"{m.group(1)}-{m.group(2)}"
                        odds_val = _safe_float(tds[-1].get_text(strip=True))
                        if odds_val:
                            results.append({
                                "combination": combo,
                                "odds_value": odds_val,
                            })
    except Exception as e:
        logger.debug(f"2連単オッズ取得失敗: {e}")

    return results


def fetch_quinella_odds_realtime(venue_code, race_no, date_str):
    """2連複オッズを取得する。"""
    url = f"{BASE_URL}/odds2f?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    results = []
    try:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    combo_text = tds[0].get_text(strip=True)
                    m = re.match(r"([1-6])-([1-6])", combo_text)
                    if m:
                        combo = f"{m.group(1)}-{m.group(2)}"
                        odds_val = _safe_float(tds[-1].get_text(strip=True))
                        if odds_val:
                            results.append({
                                "combination": combo,
                                "odds_value": odds_val,
                            })
    except Exception as e:
        logger.debug(f"2連複オッズ取得失敗: {e}")

    return results


def fetch_trifecta_odds_realtime(venue_code, race_no, date_str):
    """3連単オッズを取得する（120通り）。"""
    url = f"{BASE_URL}/oddstf?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    results = []
    try:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    combo_text = tds[0].get_text(strip=True)
                    m = re.match(r"([1-6])-([1-6])-([1-6])", combo_text)
                    if m:
                        combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        odds_val = _safe_float(tds[-1].get_text(strip=True))
                        if odds_val:
                            results.append({
                                "combination": combo,
                                "odds_value": odds_val,
                            })
    except Exception as e:
        logger.debug(f"3連単オッズ取得失敗: {e}")

    return results


# =============================================================
# スタート展示取得（15min タイミングで実行）★最重要
# =============================================================

def fetch_start_exhibition(venue_code, race_no, date_str, db_path=None):
    """
    スタート展示ページから実際の進入コースを取得してDBを更新する。
    15minタイミング専用。

    Returns:
        dict: {lane: course_taken}
    """
    url = f"{BASE_URL}/beforeinfo?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return {}

    course_taken = {}
    exhibition_times = {}
    exhibition_sts = {}

    try:
        # スタート展示テーブル: 各行が1コースに対応（1行1セル構造）
        # Row 0: ヘッダ, Row 1: コース1, ..., Row 6: コース6
        # セル内テキスト先頭が艇番、".XX"がST
        st_table = soup.find("table", class_="is-h292__3rdadd")
        if st_table is None:
            st_table = soup.find("table", class_=re.compile("is-h292"))
        tables_to_check = [st_table] if st_table else soup.find_all("table")

        for table in tables_to_check:
            rows = table.find_all("tr")
            for row_idx, row in enumerate(rows):
                if row_idx == 0:
                    continue  # ヘッダ行をスキップ
                course = row_idx  # 行インデックス=コース番号
                tds = row.find_all("td")
                for td in tds:
                    text = td.get_text(strip=True)
                    # "1.16" 形式 → 艇番=1, ST=0.16
                    m = re.match(r"([1-6])\.(\d{2})", text[:4])
                    if m:
                        lane = int(m.group(1))
                        st_val = float(f"0.{m.group(2)}")
                        if 1 <= course <= 6 and 1 <= lane <= 6:
                            course_taken[lane] = course
                            exhibition_sts[lane] = st_val

        # 展示タイム取得
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    lane = _safe_int(tds[0].get_text(strip=True))
                    if lane and 1 <= lane <= 6:
                        time_val = _safe_float(tds[1].get_text(strip=True))
                        if time_val and 6.0 <= time_val <= 8.0:
                            exhibition_times[lane] = time_val

    except Exception as e:
        logger.warning(f"スタート展示パース失敗: {e}")

    # DBのcourse_taken・exhibition情報を更新
    if course_taken and db_path is not None:
        venue_id = VENUE_CODE_TO_ID.get(venue_code, 0)
        race_id = f"{date_str}_{venue_id:02d}_{race_no:02d}"
        try:
            with get_conn(db_path) as conn:
                for lane, course in course_taken.items():
                    et = exhibition_times.get(lane)
                    est = exhibition_sts.get(lane)
                    conn.execute("""
                        UPDATE entries
                        SET course_taken=?, exhibition_time=?, exhibition_st=?
                        WHERE race_id=? AND lane=?
                    """, (course, et, est, race_id, lane))
            logger.info(f"  course_taken更新: {course_taken}")
        except Exception as e:
            logger.error(f"course_taken DB更新失敗: {e}")

    return course_taken


def apply_odds_surge_filter(lane, current_odds, prev_odds):
    """
    オッズ急上昇フィルター。

    Parameters:
        lane         : 艇番
        current_odds : 現在のオッズ
        prev_odds    : 前タイミングのオッズ

    Returns:
        dict: {
            "surge_rate": float,
            "action": "cancel"/"reduce_half"/"reduce_slight"/"ok",
            "confidence_mult": float (1.0=変更なし、0.5=半減)
        }
    """
    if prev_odds is None or prev_odds <= 0:
        return {"surge_rate": 0, "action": "ok", "confidence_mult": 1.0}

    surge_rate = (current_odds - prev_odds) / prev_odds

    if surge_rate >= 0.20:
        return {
            "surge_rate": surge_rate,
            "action": "cancel",
            "confidence_mult": 0.0,
            "reason": f"オッズ{surge_rate:.0%}急上昇：未知の悪材料の可能性",
        }
    elif surge_rate >= 0.10:
        return {
            "surge_rate": surge_rate,
            "action": "reduce_half",
            "confidence_mult": 0.5,
            "reason": f"オッズ{surge_rate:.0%}急上昇：信頼度を半減",
        }
    elif surge_rate >= 0.05:
        return {
            "surge_rate": surge_rate,
            "action": "reduce_slight",
            "confidence_mult": 0.8,
            "reason": f"オッズ{surge_rate:.0%}急上昇：信頼度をやや低下",
        }
    else:
        return {
            "surge_rate": surge_rate,
            "action": "ok",
            "confidence_mult": 1.0,
            "reason": "通常変動範囲",
        }


# =============================================================
# タイミング別の取得処理
# =============================================================

def run_realtime_fetch(venue_code, race_no, timing, date_str=None, db_path=None):
    """
    指定タイミングのオッズ・展示情報を取得してDBに保存する。

    Parameters:
        venue_code (str): 場コード "02"
        race_no    (int): レース番号 1〜12
        timing     (str): "120min"/"60min"/"30min"/"15min"/"5min"/"1min"/"final"
        date_str   (str|None): YYYYMMDD（Noneなら今日）
        db_path    (Path|None): DBパス
    """
    init_db(db_path)

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    if timing not in VALID_TIMINGS:
        logger.error(f"不正なタイミング: {timing}。有効値: {VALID_TIMINGS}")
        return

    venue_id = VENUE_CODE_TO_ID.get(venue_code, 0)
    venue_name = VENUES.get(venue_code, venue_code)
    race_id = f"{date_str}_{venue_id:02d}_{race_no:02d}"
    now_iso = datetime.now().isoformat()

    logger.info(f"=== リアルタイム取得 [{venue_name} R{race_no}] timing={timing} ===")

    odds_to_save = []

    # 120min: 単勝のみ
    if timing == "120min":
        win_odds = fetch_win_odds_realtime(venue_code, race_no, date_str)
        for o in win_odds:
            odds_to_save.append({
                "race_id": race_id, "odds_type": "単勝",
                "combination": o["combination"], "odds_value": o["odds_value"],
                "recorded_at": now_iso, "timing": timing,
            })
        logger.info(f"  単勝オッズ: {len(win_odds)}件")

    # 60min: 単勝 + 2連単 + 2連複
    elif timing == "60min":
        for odds_list, odds_type in [
            (fetch_win_odds_realtime(venue_code, race_no, date_str), "単勝"),
            (fetch_exacta_odds_realtime(venue_code, race_no, date_str), "2連単"),
            (fetch_quinella_odds_realtime(venue_code, race_no, date_str), "2連複"),
        ]:
            for o in odds_list:
                odds_to_save.append({
                    "race_id": race_id, "odds_type": odds_type,
                    "combination": o["combination"], "odds_value": o["odds_value"],
                    "recorded_at": now_iso, "timing": timing,
                })
        logger.info(f"  オッズ取得: {len(odds_to_save)}件")

    # 15min: 進入コース★ + 全券種オッズ
    elif timing == "15min":
        # 最重要: スタート展示から進入コースを取得
        course_taken = fetch_start_exhibition(
            venue_code, race_no, date_str, db_path
        )
        logger.info(f"  進入コース取得: {course_taken}")

        # 全券種オッズ
        _polite_sleep()
        for odds_list, odds_type in [
            (fetch_win_odds_realtime(venue_code, race_no, date_str), "単勝"),
            (fetch_exacta_odds_realtime(venue_code, race_no, date_str), "2連単"),
            (fetch_quinella_odds_realtime(venue_code, race_no, date_str), "2連複"),
            (fetch_trifecta_odds_realtime(venue_code, race_no, date_str), "3連単"),
        ]:
            _polite_sleep()
            for o in odds_list:
                odds_to_save.append({
                    "race_id": race_id, "odds_type": odds_type,
                    "combination": o["combination"], "odds_value": o["odds_value"],
                    "recorded_at": now_iso, "timing": timing,
                })
        logger.info(f"  全券種オッズ: {len(odds_to_save)}件")

    # 30min / 5min / 1min / final: 全券種オッズ
    else:
        for odds_list, odds_type in [
            (fetch_win_odds_realtime(venue_code, race_no, date_str), "単勝"),
            (fetch_exacta_odds_realtime(venue_code, race_no, date_str), "2連単"),
            (fetch_quinella_odds_realtime(venue_code, race_no, date_str), "2連複"),
            (fetch_trifecta_odds_realtime(venue_code, race_no, date_str), "3連単"),
        ]:
            _polite_sleep()
            for o in odds_list:
                odds_to_save.append({
                    "race_id": race_id, "odds_type": odds_type,
                    "combination": o["combination"], "odds_value": o["odds_value"],
                    "recorded_at": now_iso, "timing": timing,
                })
        logger.info(f"  全券種オッズ: {len(odds_to_save)}件")

    # DB保存
    if odds_to_save:
        insert_odds_batch(odds_to_save, db_path)
        logger.info(f"  → DB保存完了: {len(odds_to_save)}件")

    # finalタイミングでレース結果も取得
    if timing == "final":
        _polite_sleep()
        _update_race_result(venue_code, race_no, date_str, race_id, db_path)


def _update_race_result(venue_code, race_no, date_str, race_id, db_path=None):
    """
    レース結果（着順・course_taken）をDBに更新する。
    finalタイミングで呼び出す。
    """
    from research.boat.scraper_historical import fetch_race_result_and_course

    result = fetch_race_result_and_course(venue_code, race_no, date_str)
    if result is None:
        return

    try:
        with get_conn(db_path) as conn:
            for lane, finish in result["finish_order"].items():
                course = result["course_taken"].get(lane)
                st = result["start_timings"].get(lane)
                conn.execute("""
                    UPDATE entries
                    SET finish=?, course_taken=?, exhibition_st=?
                    WHERE race_id=? AND lane=?
                """, (finish, course, st, race_id, lane))

            # 天候情報を races テーブルに更新
            weather = result["weather"]
            conn.execute("""
                UPDATE races
                SET weather=?, wind_speed=?, wind_direction=?,
                    wind_direction_sin=?, wind_direction_cos=?, wave_height=?
                WHERE race_id=?
            """, (
                weather.get("weather"),
                weather.get("wind_speed"),
                weather.get("wind_direction"),
                weather.get("wind_direction_sin"),
                weather.get("wind_direction_cos"),
                weather.get("wave_height"),
                race_id,
            ))

        logger.info(f"  レース結果・course_taken更新完了: {race_id}")

    except Exception as e:
        logger.error(f"レース結果DB更新失敗: {e}")


# =============================================================
# メイン
# =============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ボートレース リアルタイム取得")
    parser.add_argument("--venue",  required=True, help="場コード（例: 02）")
    parser.add_argument("--race",   required=True, type=int, help="レース番号 1〜12")
    parser.add_argument("--timing", required=True,
                        choices=VALID_TIMINGS,
                        help="取得タイミング")
    parser.add_argument("--date",   default=None, help="日付 YYYYMMDD（省略時は今日）")
    args = parser.parse_args()

    run_realtime_fetch(
        venue_code=args.venue,
        race_no=args.race,
        timing=args.timing,
        date_str=args.date,
    )


if __name__ == "__main__":
    main()
