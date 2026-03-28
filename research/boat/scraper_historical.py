# ===========================================
# scraper_historical.py
# ボートレース 過去データ一括取得スクリプト
#
# 目標：2年分の実データ（約200,000レース）
# 取得内容：
#   - 出走表（選手情報・モーター・ボート）
#   - レース結果（着順）
#   - スタート展示（進入コース ★最重要）
#   - 確定オッズ（3連単）
#
# 使い方:
#   python scraper_historical.py --start 20240101 --end 20260327
#   python scraper_historical.py --start 20240101 --end 20260327 --venue 02
#   python scraper_historical.py --resume   # 途中から再開
# ===========================================

import re
import sys
import time
import math
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.boat.db_manager import (
    init_db, is_already_fetched, log_fetch_result,
    insert_race, insert_entries_batch, insert_odds_batch,
    VENUE_CODE_TO_ID, get_stats, DB_PATH
)

# ===== 定数 =====
BASE_URL = "https://www.boatrace.jp/owpc/pc/race"

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
    "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国",
    "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀",
    "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

WIND_DIR_SIN_COS = {
    "北":  (0.0,   -1.0),
    "北東": (0.707, -0.707),
    "東":  (1.0,   0.0),
    "南東": (0.707, 0.707),
    "南":  (0.0,   1.0),
    "南西": (-0.707, 0.707),
    "西":  (-1.0,  0.0),
    "北西": (-0.707, -0.707),
    "無風": (0.0,  0.0),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

# ===== ロガー設定 =====
LOG_FILE = PROJECT_ROOT / "data" / "boat" / "scraper_historical.log"

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
# HTTP ユーティリティ
# =============================================================

def _get_soup(url, retries=3, timeout=20):
    """URLからBeautifulSoupを取得する。失敗時はNone。"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** attempt + random.random()
                logger.debug(f"リトライ {attempt+1}/{retries}: {url} -> {e}")
                time.sleep(wait)
            else:
                logger.warning(f"取得失敗: {url} -> {e}")
                return None


def _polite_sleep():
    """サーバー負荷配慮のスリープ（2〜3秒）。"""
    time.sleep(2.0 + random.random())


def _safe_float(text, default=None):
    if text is None:
        return default
    try:
        cleaned = str(text).strip().replace(",", "")
        if cleaned in ("", "-", "−", "欠場", "失格", "転覆", "落水",
                       "沈没", "不完走", "妨害失格", "エンスト", "F", "L", "K"):
            return default
        return float(cleaned)
    except (ValueError, AttributeError):
        return default


def _safe_int(text, default=None):
    v = _safe_float(text)
    if v is None:
        return default
    return int(v)


_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")

def _fw2int(text, default=None):
    if text is None:
        return default
    try:
        return int(str(text).strip().translate(_FULLWIDTH))
    except (ValueError, AttributeError):
        return default


# =============================================================
# スクレイピング関数
# =============================================================

def fetch_schedule(year_month):
    """
    月間スケジュールページから開催会場を取得する。

    Returns:
        dict: {date_str: [venue_code, ...]}
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/monthlyschedule?ym={year_month}"
    soup = _get_soup(url)
    if soup is None:
        return {}

    schedule = {}
    try:
        # スケジュールテーブルを探す
        table = soup.find("table", class_=re.compile("is-w.*schedule", re.I))
        if table is None:
            table = soup.find("table")
        if table is None:
            return {}

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            for cell in cells:
                links = cell.find_all("a", href=True)
                for a in links:
                    href = a["href"]
                    m = re.search(r"hd=(\d{8}).*jcd=(\d{2})", href)
                    if m:
                        date_str = m.group(1)
                        venue_code = m.group(2)
                        if date_str not in schedule:
                            schedule[date_str] = []
                        if venue_code not in schedule[date_str]:
                            schedule[date_str].append(venue_code)
    except Exception as e:
        logger.warning(f"スケジュールパース失敗 {year_month}: {e}")

    return schedule


def fetch_race_entry_info(venue_code, race_no, date_str):
    """
    出走表ページから選手・モーター・ボート情報を取得する。

    Returns:
        list[dict] | None: 6艇分の情報
    """
    url = f"{BASE_URL}/racelist?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return None

    boats = []
    try:
        # 6 tbodyを持つテーブルを探す
        table = None
        for t in soup.find_all("table"):
            tbodies = t.find_all("tbody")
            if len(tbodies) >= 6:
                table = t
                break

        if table is None:
            return None

        tbodies = table.find_all("tbody")
        for idx, tbody in enumerate(tbodies[:6]):
            info = _parse_entry_row(tbody, idx + 1)
            boats.append(info)

    except Exception as e:
        logger.warning(f"出走表パース失敗 {venue_code} R{race_no} {date_str}: {e}")
        return None

    return boats if len(boats) == 6 else None


def _parse_entry_row(tbody, lane):
    """tbodyから1艇分の情報をパースする。"""
    info = {
        "lane": lane,
        "racer_id": None,
        "racer_name": None,
        "racer_class": None,
        "racer_weight": None,
        "national_win_rate": None,
        "national_2place_rate": None,
        "national_3place_rate": None,
        "local_win_rate": None,
        "local_2place_rate": None,
        "avg_start_timing": None,
        "flying_count": None,
        "late_count": None,
        "motor_no": None,
        "motor_2place_rate": None,
        "boat_no": None,
        "boat_2place_rate": None,
    }
    try:
        tds = tbody.find_all("td")

        # td[2]: 登番/級別/名前/体重
        if len(tds) > 2:
            cell_text = tds[2].get_text(separator=" ", strip=True)
            m = re.search(r"(\d{4})", cell_text)
            if m:
                info["racer_id"] = int(m.group(1))
            for cls in ("A1", "A2", "B1", "B2"):
                if cls in cell_text:
                    info["racer_class"] = CLASS_MAP.get(cls)
                    break
            wm = re.search(r"(\d{2}(?:\.\d)?)\s*kg", cell_text)
            if wm:
                info["racer_weight"] = float(wm.group(1))
            links = tbody.find_all("a")
            for a in links:
                name = a.get_text(strip=True)
                if len(name) >= 2 and not name.isdigit():
                    info["racer_name"] = name
                    break

        # td[3]: F回数/L回数/平均ST "F0L00.18"
        if len(tds) > 3:
            fl_text = tds[3].get_text(strip=True)
            fm = re.search(r"F(\d+)", fl_text)
            lm = re.search(r"L(\d+)", fl_text)
            sm = re.search(r"(\d\.\d{2})", fl_text)
            if fm:
                info["flying_count"] = int(fm.group(1))
            if lm:
                info["late_count"] = int(lm.group(1))
            if sm:
                info["avg_start_timing"] = float(sm.group(1))

        # td[4]: 全国勝率/2連率/3連率 "4.7221.2844.68"
        if len(tds) > 4:
            t4 = tds[4].get_text(strip=True)
            m = re.match(r"(\d\.\d{2})(\d{2}\.\d{2})(\d{2}\.\d{2})", t4)
            if m:
                info["national_win_rate"]    = float(m.group(1))
                info["national_2place_rate"] = float(m.group(2))
                info["national_3place_rate"] = float(m.group(3))

        # td[5]: 当地勝率/2連率/3連率
        if len(tds) > 5:
            t5 = tds[5].get_text(strip=True)
            m = re.match(r"(\d\.\d{2})(\d{2}\.\d{2})(\d{2}\.\d{2})", t5)
            if m:
                info["local_win_rate"]    = float(m.group(1))
                info["local_2place_rate"] = float(m.group(2))

        # td[6]: モーター番号+2連率
        if len(tds) > 6:
            t6 = tds[6].get_text(strip=True)
            m = re.match(r"(\d{2,3})(\d{2}\.\d{2})", t6)
            if m:
                info["motor_no"]         = int(m.group(1))
                info["motor_2place_rate"] = float(m.group(2))

        # td[7]: ボート番号+2連率
        if len(tds) > 7:
            t7 = tds[7].get_text(strip=True)
            m = re.match(r"(\d{2,3})(\d{2}\.\d{2})", t7)
            if m:
                info["boat_no"]         = int(m.group(1))
                info["boat_2place_rate"] = float(m.group(2))

    except Exception as e:
        logger.debug(f"エントリパース失敗 lane={lane}: {e}")

    return info


def fetch_race_result_and_course(venue_code, race_no, date_str):
    """
    レース結果ページから着順・実際の進入コース・天候を取得する。

    スタートタイミング表の形式:
      コース順: 1コース側から「艇番」が並ぶ
      → 表の列インデックスがコース番号に対応

    Returns:
        dict | None: {
            "finish_order": {lane: finish},    # 艇番→着順
            "course_taken": {lane: course},    # 艇番→実際のコース
            "start_timings": {lane: st},       # 艇番→ST(秒)
            "weather": {...}
        }
    """
    url = f"{BASE_URL}/raceresult?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return None

    result = {
        "finish_order": {},
        "course_taken": {},
        "start_timings": {},
        "weather": {},
    }

    try:
        # --- 着順テーブル ---
        result_tables = soup.find_all("table", class_="is-w495")
        if result_tables:
            for row in result_tables[0].find_all("tr"):
                tds = row.find_all("td")
                if len(tds) >= 2:
                    finish_pos = _fw2int(tds[0].get_text(strip=True))
                    lane = _safe_int(tds[1].get_text(strip=True))
                    if finish_pos and lane and 1 <= lane <= 6:
                        result["finish_order"][lane] = finish_pos

        # --- スタートタイミング表 → 進入コース取得 ---
        # HTML構造: class="is-h292__3rdadd" のテーブル
        # ヘッダ行: コース 1 | コース 2 | ... | コース 6
        # データ行: 艇番.タイミング 形式 "1.16", "3.18" など
        # → カラム位置がコース番号、値の整数部が艇番
        st_table = soup.find("table", class_="is-h292__3rdadd")
        if st_table is None:
            # 別のクラスを試す
            st_table = soup.find("table", class_=re.compile("start", re.I))

        if st_table:
            rows = st_table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if not tds:
                    tds = row.find_all("th")
                for col_idx, td in enumerate(tds):
                    text = td.get_text(strip=True)
                    # "1.16" 形式 → 艇番=1, ST=0.16
                    m = re.match(r"([1-6])\.(\d{2})", text[:4])
                    if m:
                        lane = int(m.group(1))
                        st_val = float(f"0.{m.group(2)}")
                        course = col_idx + 1   # 列インデックス=コース番号（1〜6）
                        if 1 <= course <= 6 and 1 <= lane <= 6:
                            result["course_taken"][lane] = course
                            result["start_timings"][lane] = st_val

        # --- 天候情報 ---
        result["weather"] = _parse_weather(soup)

    except Exception as e:
        logger.warning(f"結果パース失敗 {venue_code} R{race_no} {date_str}: {e}")

    return result


def _parse_weather(soup):
    """天候・風速・風向き・波高をパースする。"""
    info = {
        "weather": None,
        "wind_speed": None,
        "wind_direction": None,
        "wind_direction_sin": None,
        "wind_direction_cos": None,
        "wave_height": None,
    }
    try:
        # 天候セクション（複数のクラス名を試す）
        section = (soup.find("div", class_="weather1")
                   or soup.find("div", class_="is-weather")
                   or soup.find("div", class_=re.compile("weather", re.I)))
        if section:
            text = section.get_text()
        else:
            text = soup.get_text()

        for w in ["晴", "曇り", "曇", "雨", "雪", "霧"]:
            if w in text:
                info["weather"] = w
                break

        wm = re.search(r"(\d+)\s*m", text)
        if wm:
            info["wind_speed"] = int(wm.group(1))

        wave_m = re.search(r"(\d+)\s*cm", text)
        if wave_m:
            info["wave_height"] = int(wave_m.group(1))

        for dir_name, (sin_v, cos_v) in WIND_DIR_SIN_COS.items():
            if dir_name in text:
                info["wind_direction"] = dir_name
                info["wind_direction_sin"] = sin_v
                info["wind_direction_cos"] = cos_v
                break

    except Exception as e:
        logger.debug(f"天候パース失敗: {e}")

    return info


def fetch_trifecta_odds(venue_code, race_no, date_str):
    """
    3連単の確定オッズを取得する。

    Returns:
        list[dict]: [{combination, odds_value}, ...]
    """
    url = f"{BASE_URL}/oddstf?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    odds_list = []
    try:
        # 3連単テーブル: 120通り
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    # 組み合わせ列と確率列を探す
                    combo_text = tds[0].get_text(strip=True)
                    m = re.match(r"(\d)-(\d)-(\d)", combo_text)
                    if m:
                        combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        odds_val = _safe_float(tds[-1].get_text(strip=True))
                        if odds_val and odds_val > 0:
                            odds_list.append({
                                "combination": combo,
                                "odds_value": odds_val,
                            })
    except Exception as e:
        logger.debug(f"オッズパース失敗 {venue_code} R{race_no} {date_str}: {e}")

    return odds_list


def fetch_win_odds(venue_code, race_no, date_str):
    """
    単勝の確定オッズを取得する（oddstf ページから）。

    Returns:
        list[dict]: [{lane, odds_value}, ...]
    """
    url = f"{BASE_URL}/oddstf?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return []

    win_odds = []
    try:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    lane_text = tds[0].get_text(strip=True)
                    lane = _safe_int(lane_text)
                    if lane and 1 <= lane <= 6:
                        odds_val = _safe_float(tds[1].get_text(strip=True))
                        if odds_val:
                            win_odds.append({"lane": lane, "odds_value": odds_val})
    except Exception as e:
        logger.debug(f"単勝オッズパース失敗: {e}")

    return win_odds


# =============================================================
# 1日・1会場分の取得処理
# =============================================================

def fetch_venue_day(venue_code, date_str, db_path=None):
    """
    1会場・1日分（最大12レース）のデータを取得してDBに保存する。

    Returns:
        int: 取得成功したレース数（0=開催なし）
    """
    venue_id = VENUE_CODE_TO_ID.get(venue_code)
    if venue_id is None:
        logger.error(f"未知の会場コード: {venue_code}")
        return 0

    venue_name = VENUES.get(venue_code, "不明")
    races_fetched = 0
    now_str = datetime.now().isoformat()

    for race_no in range(1, 13):
        # 出走表取得
        _polite_sleep()
        entries_raw = fetch_race_entry_info(venue_code, race_no, date_str)

        if entries_raw is None:
            # このレースは存在しない（開催なし or ページなし）
            if race_no == 1:
                # 第1Rが取れないならこの会場は開催なし
                return 0
            continue

        # レース結果 + 進入コース取得
        _polite_sleep()
        result_data = fetch_race_result_and_course(venue_code, race_no, date_str)

        # オッズ取得（3連単）
        _polite_sleep()
        odds_raw = fetch_trifecta_odds(venue_code, race_no, date_str)

        # --- レースIDを構築 ---
        race_id = f"{date_str}_{venue_id:02d}_{race_no:02d}"

        # --- races テーブルに保存 ---
        weather = result_data["weather"] if result_data else {}
        race_data = {
            "race_id": race_id,
            "date": date_str,
            "venue_id": venue_id,
            "race_no": race_no,
            "grade": None,
            "race_type": None,
            "race_time": None,
            "weather": weather.get("weather"),
            "wind_speed": weather.get("wind_speed"),
            "wind_direction": weather.get("wind_direction"),
            "wind_direction_sin": weather.get("wind_direction_sin"),
            "wind_direction_cos": weather.get("wind_direction_cos"),
            "wave_height": weather.get("wave_height"),
            "rain_amount": None,
            "air_pressure": None,
            "temperature": None,
            "water_type": None,
            "tide_level": None,
            "created_at": now_str,
        }
        insert_race(race_data, db_path)

        # --- entries テーブルに保存 ---
        entries_to_insert = []
        for raw in entries_raw:
            lane = raw["lane"]
            course_taken = None
            finish = None
            exhibition_st = None

            if result_data:
                course_taken = result_data["course_taken"].get(lane)
                finish       = result_data["finish_order"].get(lane)
                exhibition_st = result_data["start_timings"].get(lane)

            entry = {
                "race_id": race_id,
                "lane": lane,
                "course_taken": course_taken,
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
                "exhibition_st": exhibition_st,
                "finish": finish,
                "created_at": now_str,
            }
            entries_to_insert.append(entry)

        insert_entries_batch(entries_to_insert, db_path)

        # --- odds テーブルに保存 ---
        if odds_raw:
            now_iso = datetime.now().isoformat()
            odds_to_insert = [
                {
                    "race_id": race_id,
                    "odds_type": "3連単",
                    "combination": o["combination"],
                    "odds_value": o["odds_value"],
                    "recorded_at": now_iso,
                    "timing": "final",
                }
                for o in odds_raw
            ]
            insert_odds_batch(odds_to_insert, db_path)

        races_fetched += 1
        logger.info(
            f"  [{venue_name} R{race_no:02d}] "
            f"エントリ={len(entries_to_insert)} "
            f"course_taken={sum(1 for e in entries_to_insert if e['course_taken'])} "
            f"オッズ={len(odds_raw)}"
        )

    return races_fetched


# =============================================================
# メインループ
# =============================================================

def run_historical_scrape(
    start_date_str,
    end_date_str,
    venue_filter=None,
    db_path=None,
    use_schedule=True,
):
    """
    指定期間の全会場・全レースを取得する。

    Parameters:
        start_date_str (str): 開始日 YYYYMMDD
        end_date_str   (str): 終了日 YYYYMMDD
        venue_filter   (str|None): 特定会場コードのみ取得（例: "02"）
        db_path        (Path|None): DBパス（Noneならデフォルト）
        use_schedule   (bool): 月間スケジュールを使って開催会場を絞るか
    """
    # DB初期化
    init_db(db_path)
    logger.info(f"=== 過去データ取得開始 {start_date_str} → {end_date_str} ===")

    start_dt = datetime.strptime(start_date_str, "%Y%m%d")
    end_dt   = datetime.strptime(end_date_str,   "%Y%m%d")
    total_days = (end_dt - start_dt).days + 1

    # 月間スケジュールキャッシュ
    schedule_cache = {}

    current_dt = start_dt
    day_count = 0

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        ym = current_dt.strftime("%Y%m")
        day_count += 1

        logger.info(
            f"[{day_count}/{total_days}] {date_str} 処理中..."
        )

        # 月間スケジュール取得（月が変わるたびに1回だけ）
        if use_schedule and ym not in schedule_cache:
            _polite_sleep()
            logger.info(f"  月間スケジュール取得: {ym}")
            schedule_cache[ym] = fetch_schedule(ym)

        # この日の開催会場を決定
        if use_schedule and date_str in schedule_cache.get(ym, {}):
            day_venues = schedule_cache[ym][date_str]
            logger.info(f"  開催会場: {[VENUES.get(v, v) for v in day_venues]}")
        else:
            # スケジュール取得失敗 or 該当なし → 全会場を試す
            day_venues = list(VENUES.keys())

        # venue_filter があれば絞り込む
        if venue_filter:
            day_venues = [v for v in day_venues if v == venue_filter]

        for venue_code in day_venues:
            venue_id = VENUE_CODE_TO_ID.get(venue_code, 0)

            # 取得済みならスキップ
            if is_already_fetched(date_str, venue_id, db_path):
                logger.info(f"  [{VENUES.get(venue_code)}] スキップ（取得済み）")
                continue

            logger.info(f"  [{VENUES.get(venue_code, venue_code)}] 取得開始...")

            try:
                races_count = fetch_venue_day(venue_code, date_str, db_path)

                if races_count == 0:
                    log_fetch_result(date_str, venue_id, "no_race", 0, db_path=db_path)
                    logger.info(f"  [{VENUES.get(venue_code)}] 開催なし")
                else:
                    log_fetch_result(date_str, venue_id, "success",
                                     races_count, db_path=db_path)
                    logger.info(
                        f"  [{VENUES.get(venue_code)}] 完了 ({races_count}レース)"
                    )

            except Exception as e:
                log_fetch_result(date_str, venue_id, "error", 0,
                                 error_msg=str(e), db_path=db_path)
                logger.error(
                    f"  [{VENUES.get(venue_code)}] エラー: {e}", exc_info=True
                )

        current_dt += timedelta(days=1)

    # 最終統計
    stats = get_stats(db_path)
    logger.info("=== 取得完了 ===")
    logger.info(f"  レース数: {stats['races']}")
    logger.info(f"  エントリ数: {stats['entries']}")
    logger.info(
        f"  course_taken取得済み: {stats['entries_with_course_taken']}"
        f" ({stats['entries_with_course_taken']/max(stats['entries'],1)*100:.1f}%)"
    )
    logger.info(f"  オッズ数: {stats['odds']}")
    logger.info(f"  期間: {stats['date_min']} 〜 {stats['date_max']}")


def main():
    parser = argparse.ArgumentParser(
        description="ボートレース過去データ一括取得"
    )
    parser.add_argument("--start", required=False,
                        help="開始日 YYYYMMDD（例: 20240101）")
    parser.add_argument("--end", required=False,
                        help="終了日 YYYYMMDD（例: 20260327）")
    parser.add_argument("--venue", default=None,
                        help="特定会場コードのみ取得（例: 02）")
    parser.add_argument("--resume", action="store_true",
                        help="2年前から今日まで取得（デフォルト期間）")
    parser.add_argument("--no-schedule", action="store_true",
                        help="月間スケジュール取得をスキップして全会場を試す")
    args = parser.parse_args()

    if args.resume or (not args.start and not args.end):
        # デフォルト: 2年前から昨日まで
        end_dt = datetime.now() - timedelta(days=1)
        start_dt = end_dt - timedelta(days=730)
        start_str = start_dt.strftime("%Y%m%d")
        end_str   = end_dt.strftime("%Y%m%d")
        logger.info(f"--resume モード: {start_str} → {end_str}")
    else:
        start_str = args.start
        end_str   = args.end

    run_historical_scrape(
        start_date_str=start_str,
        end_date_str=end_str,
        venue_filter=args.venue,
        use_schedule=not args.no_schedule,
    )


if __name__ == "__main__":
    main()
