# ===========================================
# fetch_real_data.py
# ボートレース公式サイトからレースデータを取得
#
# データソース: https://www.boatrace.jp/
#   - レース結果: raceresult
#   - 出走表: racelist
#   - オッズ: oddstf
#
# 使い方:
#   python fetch_real_data.py --start 20260101 --end 20260324
#   python fetch_real_data.py --venue 01  # 桐生のみ
# ===========================================

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import threading
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# スレッドセーフなCSV書き込み用ロック
_csv_lock = threading.Lock()

# ===== 定数 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "boat"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race"

# 場コード
VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
    "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国",
    "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀",
    "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

# 選手級別の数値エンコーディング（boat_model.py と統一）
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

# HTTPリクエストヘッダー
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ロガー設定（バッファリングなし + ファイル出力）
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_handler = _FlushHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

# ファイルにも出力（進捗確認用）
_PROGRESS_FILE = DATA_DIR / "fetch_progress.log"
_file_handler = logging.FileHandler(str(_PROGRESS_FILE), mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logging.root.handlers = [_handler, _file_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================
# ユーティリティ
# =============================================================

def _safe_float(text, default=None):
    """テキストを安全にfloatに変換する。"""
    if text is None:
        return default
    try:
        cleaned = text.strip().replace(",", "")
        if cleaned in ("", "-", "−", "欠場", "失格", "転覆", "落水", "沈没", "不完走",
                        "妨害失格", "エンスト", "F", "L", "K"):
            return default
        return float(cleaned)
    except (ValueError, AttributeError):
        return default


def _safe_int(text, default=None):
    """テキストを安全にintに変換する。"""
    val = _safe_float(text, default=None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return int(val)


# 全角→半角数字変換
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

def _fw_to_int(text, default=None):
    """全角数字を含むテキストをintに変換する。"""
    if text is None:
        return default
    try:
        cleaned = text.strip().translate(_FULLWIDTH_DIGITS)
        return int(cleaned)
    except (ValueError, AttributeError):
        return default


def _get_soup(url, timeout=15, retries=1):
    """URLからBeautifulSoupオブジェクトを返す。エラー時はNone。"""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(1)
                continue
            logger.warning(f"リクエスト失敗: {url} -> {e}")
            return None


def _polite_sleep(min_sec=0.3, max_sec=0.8):
    """サーバーに優しいランダムスリープ。"""
    time.sleep(min_sec + (max_sec - min_sec) * np.random.random())


# =============================================================
# Function 1: レース結果を取得
# =============================================================

def fetch_race_result(venue_code, race_no, date_str):
    """
    レース結果ページから着順・選手情報・タイムを取得する。

    Parameters:
        venue_code (str): 場コード（例: "01"）
        race_no (int): レース番号（1-12）
        date_str (str): 日付（YYYYMMDD）

    Returns:
        dict: レース結果データ。取得失敗時はNone。
    """
    url = f"{BASE_URL}/raceresult?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return None

    result = {
        "date": date_str,
        "venue_code": venue_code,
        "venue_name": VENUES.get(venue_code, "不明"),
        "race_no": race_no,
        "finish_order": {},       # {lane: 着順}
        "racers": {},             # {lane: {"name": ..., "reg_no": ...}}
        "race_times": {},         # {lane: タイム文字列}
        "start_timings": {},      # {lane: ST(秒)}
    }

    try:
        # --- 着順テーブル ---
        # class="is-w495" の最初のテーブル: [着, 枠, ボートレーサー, レースタイム]
        # 着は全角数字（１２３...）、枠は半角数字
        result_tables = soup.find_all("table", class_="is-w495")
        result_table = result_tables[0] if result_tables else None

        if result_table:
            rows = result_table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 2:
                    finish_pos = _fw_to_int(tds[0].get_text(strip=True))
                    lane = _safe_int(tds[1].get_text(strip=True))
                    if finish_pos is not None and lane is not None and 1 <= lane <= 6:
                        result["finish_order"][lane] = finish_pos

                        # ボートレーサー列: "4895門間　　雄大" → 登番+名前
                        if len(tds) >= 3:
                            racer_text = tds[2].get_text(strip=True)
                            import re
                            m = re.match(r"(\d{4})(.*)", racer_text)
                            if m:
                                result["racers"][lane] = {
                                    "name": m.group(2).strip(),
                                    "reg_no": int(m.group(1)),
                                }

                        # レースタイム
                        if len(tds) >= 4:
                            race_time = tds[3].get_text(strip=True)
                            if race_time:
                                result["race_times"][lane] = race_time

        # --- スタートタイミング ---
        # class="is-w495 is-h292__3rdadd" テーブル: 各TDが "枠番.タイミング" 形式
        import re
        st_table = soup.find("table", class_="is-h292__3rdadd")
        if st_table:
            tds = st_table.find_all("td")
            for td in tds:
                text = td.get_text(strip=True)[:4]  # "1.16" etc, truncate noise
                m = re.match(r"(\d)\.(\d{2})", text)
                if m:
                    lane = int(m.group(1))
                    timing = float(f"0.{m.group(2)}")
                    if 1 <= lane <= 6:
                        result["start_timings"][lane] = timing

        # --- 天候情報 ---
        result["weather"] = _parse_weather(soup)

    except Exception as e:
        logger.warning(f"結果パース失敗: {venue_code} R{race_no} {date_str} -> {e}")

    return result


def _parse_weather(soup):
    """天候・風・波情報をパースする。"""
    weather_info = {
        "weather": None,
        "wind_direction": None,
        "wind_speed": None,
        "wave_height": None,
    }

    try:
        # 天候セクションを探す
        weather_section = soup.find("div", class_="weather1")
        if weather_section is None:
            # 別のクラス名
            weather_section = soup.find("div", class_="is-weather")
        if weather_section is None:
            # bodyからテキスト検索
            body_text = soup.get_text()
            if "天候" in body_text:
                weather_section = soup

        if weather_section:
            text = weather_section.get_text()

            # 天候
            for w in ["晴", "曇り", "曇", "雨", "雪", "霧"]:
                if w in text:
                    weather_info["weather"] = w
                    break

            # 風速（数値 + "m"）
            import re
            wind_match = re.search(r"(\d+)\s*m", text)
            if wind_match:
                weather_info["wind_speed"] = int(wind_match.group(1))

            # 波高
            wave_match = re.search(r"(\d+)\s*cm", text)
            if wave_match:
                weather_info["wave_height"] = int(wave_match.group(1))

    except Exception as e:
        logger.debug(f"天候パース失敗: {e}")

    return weather_info


# =============================================================
# Function 2: 出走表（レース前情報）を取得
# =============================================================

def fetch_race_before_info(venue_code, race_no, date_str):
    """
    出走表ページから各艇の選手情報を取得する。

    Parameters:
        venue_code (str): 場コード
        race_no (int): レース番号（1-12）
        date_str (str): 日付（YYYYMMDD）

    Returns:
        list[dict]: 6艇分の情報。取得失敗時はNone。
    """
    url = f"{BASE_URL}/racelist?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return None

    boats = []

    try:
        # 出走表テーブル: 6 tbody（各艇1つ）を持つテーブルを探す
        tables = soup.find_all("table")
        table = None
        for t in tables:
            tbodies = t.find_all("tbody")
            if len(tbodies) >= 6:
                table = t
                break

        if table is None:
            logger.warning(f"出走表テーブル未検出: {venue_code} R{race_no} {date_str}")
            return None

        tbodies = table.find_all("tbody")
        for idx, tbody in enumerate(tbodies[:6]):
            lane = idx + 1
            boat_info = _parse_boat_row(tbody, lane)
            boats.append(boat_info)

    except Exception as e:
        logger.warning(f"出走表パース失敗: {venue_code} R{race_no} {date_str} -> {e}")
        return None

    return boats if len(boats) == 6 else None


def _parse_boat_row(tbody, lane):
    """
    tbody要素から1艇分の情報をパースする。

    実際のHTML構造（2025年時点）:
      td[0]: 枠番（全角: １２３...）
      td[2]: "3463\\n /B1乙津　　康志" — 登番/級別/名前
      td[3]: "F0L00.18" — フライング回数/出遅れ/平均ST
      td[4]: "4.7221.2844.68" — 全国勝率/全国2連率/全国3連率
      td[5]: "5.0830.5650.00" — 当地勝率/当地2連率/当地3連率
      td[6]: "3540.5664.34" — モーター番号+2連率+3連率
      td[7]: "2245.5159.62" — ボート番号+2連率+3連率
    """
    import re

    info = {
        "lane": lane,
        "racer_name": None,
        "racer_reg_no": None,
        "racer_class": None,
        "racer_win_rate": None,
        "racer_2place_rate": None,
        "racer_3place_rate": None,
        "racer_local_win_rate": None,
        "racer_local_2place_rate": None,
        "flying_count": None,
        "late_count": None,
        "motor_no": None,
        "motor_2place_rate": None,
        "boat_no": None,
        "boat_2place_rate": None,
        "weight": None,
    }

    try:
        tds = tbody.find_all("td")

        # --- td[2]: 登番/級別/名前/支部/年齢 ---
        if len(tds) > 2:
            cell_text = tds[2].get_text(separator=" ", strip=True)
            m = re.search(r"(\d{4})", cell_text)
            if m:
                info["racer_reg_no"] = int(m.group(1))
            for cls in ("A1", "A2", "B1", "B2"):
                if cls in cell_text:
                    info["racer_class"] = cls
                    break
            # 体重: "52kg" or "52.0"
            wm = re.search(r"(\d{2}(?:\.\d)?)\s*kg", cell_text)
            if wm:
                info["weight"] = float(wm.group(1))
            # 選手名: リンクタグのうち名前らしいもの
            links = tbody.find_all("a")
            for a in links:
                name = a.get_text(strip=True)
                if len(name) >= 2 and not name.isdigit() and not name.startswith("http"):
                    info["racer_name"] = name
                    break

        # --- td[3]: F回数/L回数/平均ST ---
        # "F0L00.18" → F=0, L=0, avg_st=0.18
        if len(tds) > 3:
            fl_text = tds[3].get_text(strip=True)
            f_match = re.search(r"F(\d+)", fl_text)
            l_match = re.search(r"L(\d+)", fl_text)
            st_match = re.search(r"(\d\.\d{2})", fl_text)
            if f_match:
                info["flying_count"] = int(f_match.group(1))
            if l_match:
                info["late_count"] = int(l_match.group(1))
            if st_match:
                info["avg_start_timing"] = float(st_match.group(1))

        # --- td[4]: 全国勝率/2連率/3連率 ---
        # "4.7221.2844.68" → 勝率=4.72, 2連率=21.28, 3連率=44.68
        if len(tds) > 4:
            rates_text = tds[4].get_text(strip=True)
            m = re.match(r"(\d\.\d{2})(\d{2}\.\d{2})(\d{2}\.\d{2})", rates_text)
            if m:
                info["racer_win_rate"] = float(m.group(1))
                info["racer_2place_rate"] = float(m.group(2))
                info["racer_3place_rate"] = float(m.group(3))

        # --- td[5]: 当地勝率/2連率/3連率 ---
        # "5.0830.5650.00"
        if len(tds) > 5:
            local_text = tds[5].get_text(strip=True)
            m = re.match(r"(\d\.\d{2})(\d{2}\.\d{2})(\d{2}\.\d{2})", local_text)
            if m:
                info["racer_local_win_rate"] = float(m.group(1))
                info["racer_local_2place_rate"] = float(m.group(2))

        # --- td[6]: モーター番号+2連率+3連率 ---
        if len(tds) > 6:
            motor_text = tds[6].get_text(strip=True)
            m = re.match(r"(\d{2,3})(\d{2}\.\d{2})(\d{2}\.\d{2})", motor_text)
            if m:
                info["motor_no"] = int(m.group(1))
                info["motor_2place_rate"] = float(m.group(2))

        # --- td[7]: ボート番号+2連率+3連率 ---
        if len(tds) > 7:
            boat_text = tds[7].get_text(strip=True)
            m = re.match(r"(\d{2,3})(\d{2}\.\d{2})(\d{2}\.\d{2})", boat_text)
            if m:
                info["boat_no"] = int(m.group(1))
                info["boat_2place_rate"] = float(m.group(2))

    except Exception as e:
        logger.debug(f"ボート行パース失敗 (lane={lane}): {e}")

    return info


# =============================================================
# Function 3: オッズを取得
# =============================================================

def fetch_odds(venue_code, race_no, date_str):
    """
    単勝オッズを取得する。

    Parameters:
        venue_code (str): 場コード
        race_no (int): レース番号（1-12）
        date_str (str): 日付（YYYYMMDD）

    Returns:
        dict: {1: odds1, 2: odds2, ..., 6: odds6}。取得失敗時はNone。
    """
    url = f"{BASE_URL}/oddstf?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return None

    odds = {}

    try:
        # 単勝オッズテーブル: class="is-w495" で "単勝オッズ" ヘッダーを持つもの
        # 構造: [枠, ボートレーサー, 単勝オッズ]
        odds_tables = soup.find_all("table", class_="is-w495")
        odds_table = None
        for t in odds_tables:
            ths = t.find_all("th")
            for th in ths:
                if "単勝" in th.get_text():
                    odds_table = t
                    break
            if odds_table:
                break

        # フォールバック: 最初のis-w495テーブル
        if odds_table is None and odds_tables:
            odds_table = odds_tables[0]

        if odds_table:
            rows = odds_table.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 3:
                    # [枠, ボートレーサー, オッズ]
                    lane = _safe_int(tds[0].get_text(strip=True))
                    odds_val = _safe_float(tds[2].get_text(strip=True))
                    if lane and 1 <= lane <= 6 and odds_val is not None:
                        odds[lane] = odds_val

    except Exception as e:
        logger.warning(f"オッズパース失敗: {venue_code} R{race_no} {date_str} -> {e}")

    return odds if odds else None


# =============================================================
# Function 3b: 展示タイム・チルトを取得
# =============================================================

def fetch_exhibition(venue_code, race_no, date_str):
    """
    直前情報ページから展示タイム・チルト角を取得する。

    Returns:
        dict: {lane: {"exhibition_time": float, "tilt": float}}。取得失敗時はNone。
    """
    url = f"{BASE_URL}/beforeinfo?rno={race_no}&jcd={venue_code}&hd={date_str}"
    soup = _get_soup(url)
    if soup is None:
        return None

    exhibition = {}
    import re

    try:
        # beforeinfoページのテーブルから展示タイムを探す
        tables = soup.find_all("table")
        for t in tables:
            rows = t.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) >= 4:
                    # 枠番を探す
                    lane_text = tds[0].get_text(strip=True)
                    lane = _fw_to_int(lane_text)
                    if lane and 1 <= lane <= 6:
                        ex_data = {}
                        # 残りのtdから展示タイム（6.XX形式）とチルト（-0.5等）を探す
                        for td in tds[1:]:
                            text = td.get_text(strip=True)
                            # 展示タイム: 6.XX形式（6秒台）
                            m_ex = re.match(r"^(\d\.\d{2})$", text)
                            if m_ex:
                                val = float(m_ex.group(1))
                                if 6.0 <= val <= 7.5:
                                    ex_data["exhibition_time"] = val
                            # チルト: -0.5, 0.0, 0.5 等
                            m_tilt = re.match(r"^(-?\d+\.?\d*)$", text)
                            if m_tilt:
                                val = float(m_tilt.group(1))
                                if -3.0 <= val <= 3.0 and "exhibition_time" not in ex_data:
                                    ex_data["tilt"] = val

                        if ex_data:
                            exhibition[lane] = ex_data

    except Exception as e:
        logger.debug(f"展示情報パース失敗: {venue_code} R{race_no} {date_str} -> {e}")

    return exhibition if exhibition else None


# =============================================================
# Function 4: 1日分の全レースを取得
# =============================================================

def fetch_day_results(venue_code, date_str, skip_odds=False):
    """
    指定会場・日付の全12レース分のデータを取得する。

    Parameters:
        venue_code (str): 場コード
        date_str (str): 日付（YYYYMMDD）
        skip_odds (bool): オッズ取得をスキップ（高速化）

    Returns:
        list[dict]: 各レースのデータ。開催なしの場合は空リスト。
    """
    races = []

    # まずR1の結果ページだけで開催有無を判定（高速チェック）
    test_result = fetch_race_result(venue_code, 1, date_str)
    if test_result is None or not test_result.get("finish_order"):
        logger.debug(f"{VENUES.get(venue_code, venue_code)} {date_str} 開催なし")
        return []

    for race_no in range(1, 13):
        # レース結果（R1は再利用）
        if race_no == 1:
            result = test_result
        else:
            result = fetch_race_result(venue_code, race_no, date_str)
            _polite_sleep()

        # 出走表（レース前情報）
        before_info = fetch_race_before_info(venue_code, race_no, date_str)
        _polite_sleep()

        # オッズ（オプション）
        odds = None
        if not skip_odds:
            odds = fetch_odds(venue_code, race_no, date_str)
            _polite_sleep()

        # データが何もなければスキップ
        if before_info is None and result is None:
            continue

        race_data = {
            "date": date_str,
            "venue_code": venue_code,
            "venue_name": VENUES.get(venue_code, "不明"),
            "race_no": race_no,
            "before_info": before_info,
            "result": result,
            "odds": odds,
        }
        races.append(race_data)

    return races


# =============================================================
# Function 5: 期間指定でデータ取得 + CSV保存
# =============================================================

def _race_to_csv_row(race_data):
    """1レース分のデータをCSV行（flat dict）に変換する。"""
    row = {
        "date": race_data["date"],
        "venue_code": race_data["venue_code"],
        "venue_name": race_data["venue_name"],
        "race_no": race_data["race_no"],
    }

    before_info = race_data.get("before_info") or [None] * 6
    result = race_data.get("result") or {}
    odds = race_data.get("odds") or {}

    finish_order = result.get("finish_order", {}) if isinstance(result, dict) else {}
    start_timings = result.get("start_timings", {}) if isinstance(result, dict) else {}
    weather = result.get("weather", {}) if isinstance(result, dict) else {}

    for lane in range(1, 7):
        prefix = f"lane{lane}_"
        info = before_info[lane - 1] if before_info and lane - 1 < len(before_info) else None

        if info and isinstance(info, dict):
            row[prefix + "racer"] = info.get("racer_name")
            row[prefix + "class"] = info.get("racer_class")
            row[prefix + "win_rate"] = info.get("racer_win_rate")
            row[prefix + "2place_rate"] = info.get("racer_2place_rate")
            row[prefix + "3place_rate"] = info.get("racer_3place_rate")
            row[prefix + "local_win_rate"] = info.get("racer_local_win_rate")
            row[prefix + "local_2place_rate"] = info.get("racer_local_2place_rate")
            row[prefix + "flying_count"] = info.get("flying_count")
            row[prefix + "late_count"] = info.get("late_count")
            row[prefix + "motor_no"] = info.get("motor_no")
            row[prefix + "motor_2rate"] = info.get("motor_2place_rate")
            row[prefix + "boat_2rate"] = info.get("boat_2place_rate")
            row[prefix + "weight"] = info.get("weight")
        else:
            row[prefix + "racer"] = None
            row[prefix + "class"] = None
            row[prefix + "win_rate"] = None
            row[prefix + "2place_rate"] = None
            row[prefix + "3place_rate"] = None
            row[prefix + "local_win_rate"] = None
            row[prefix + "local_2place_rate"] = None
            row[prefix + "flying_count"] = None
            row[prefix + "late_count"] = None
            row[prefix + "motor_no"] = None
            row[prefix + "motor_2rate"] = None
            row[prefix + "boat_2rate"] = None
            row[prefix + "weight"] = None

        row[prefix + "start_timing"] = start_timings.get(lane)
        row[prefix + "finish"] = finish_order.get(lane)

    # オッズ
    for lane in range(1, 7):
        row[f"odds_{lane}"] = odds.get(lane) if odds else None

    # 天候
    if isinstance(weather, dict):
        row["weather"] = weather.get("weather")
        row["wind_direction"] = weather.get("wind_direction")
        row["wind_speed"] = weather.get("wind_speed")
        row["wave_height"] = weather.get("wave_height")
    else:
        row["weather"] = None
        row["wind_direction"] = None
        row["wind_speed"] = None
        row["wave_height"] = None

    return row


def fetch_historical_data(start_date, end_date, venues=None, skip_odds=False):
    """
    期間・会場を指定してデータを取得し、CSVに保存する。

    Parameters:
        start_date (str): 開始日（YYYYMMDD）
        end_date (str): 終了日（YYYYMMDD）
        venues (list[str], optional): 場コードのリスト。Noneなら全24場。
        skip_odds (bool): オッズ取得をスキップして高速化。

    Returns:
        pd.DataFrame: 取得したデータ。
    """
    if venues is None:
        venues = list(VENUES.keys())
    elif isinstance(venues, str):
        venues = [venues]

    csv_path = DATA_DIR / "real_race_data.csv"

    # 既存データの読み込み（中断再開用）
    existing_keys = set()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
            for _, r in existing_df.iterrows():
                key = f"{r['date']}_{r['venue_code']}_{r['race_no']}"
                existing_keys.add(key)
            logger.info(f"既存データ: {len(existing_df)}レース読み込み済み")
        except Exception as e:
            logger.warning(f"既存CSV読み込み失敗: {e}")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    # 日付リストを生成
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    total_races = 0
    new_rows = []

    for date_str in dates:
        for venue_code in venues:
            # 既に全レース取得済みならスキップ
            first_key = f"{date_str}_{venue_code}_1"
            if first_key in existing_keys:
                logger.debug(f"スキップ（取得済み）: {VENUES[venue_code]} {date_str}")
                continue

            races = fetch_day_results(venue_code, date_str, skip_odds=skip_odds)

            if not races:
                continue

            for race in races:
                key = f"{race['date']}_{race['venue_code']}_{race['race_no']}"
                if key not in existing_keys:
                    csv_row = _race_to_csv_row(race)
                    new_rows.append(csv_row)
                    existing_keys.add(key)

            total_new = len(new_rows)
            total_all = len(existing_keys)
            venue_name = VENUES.get(venue_code, venue_code)
            logger.info(
                f"{date_str} {venue_name} {len(races)}R完了 "
                f"(新規: {total_new}, 累計: {total_all}レース)"
            )

            # 定期的にCSVに保存（12レースごと = 1日分）
            if len(new_rows) >= 12:
                _append_to_csv(csv_path, new_rows, existing_df)
                existing_df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
                new_rows = []

    # 残りを保存
    if new_rows:
        _append_to_csv(csv_path, new_rows, existing_df)

    # 最終データを読み込んで返す
    if csv_path.exists():
        final_df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
        logger.info(f"完了: 合計 {len(final_df)} レース -> {csv_path}")
        return final_df
    else:
        logger.info("取得データなし")
        return pd.DataFrame()


def _append_to_csv(csv_path, new_rows, existing_df):
    """新しい行をCSVに追記する（スレッドセーフ）。"""
    with _csv_lock:
        # ロック中に最新のCSVを再読込（他スレッドの書き込みを反映）
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
            except Exception:
                pass
        new_df = pd.DataFrame(new_rows)
        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        # 重複排除（同じレースが複数回保存されるのを防ぐ）
        if "date" in combined.columns and "venue_code" in combined.columns and "race_no" in combined.columns:
            combined = combined.drop_duplicates(subset=["date", "venue_code", "race_no"], keep="last")
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV保存: {len(combined)}レース -> {csv_path}")


# =============================================================
# モデル用フォーマット変換
# =============================================================

def convert_to_model_format(csv_path=None):
    """
    生データCSVをboat_model.pyが期待するDataFrame形式に変換する。

    boat_model.pyの期待カラム:
        race_id, race_date, lane, racer_class, racer_win_rate, racer_place_rate,
        racer_local_win_rate, motor_2place_rate, boat_2place_rate, avg_start_timing,
        flying_count, weather_wind_speed, weather_condition, course_type,
        win, place_top2

    Parameters:
        csv_path (str or Path, optional): 生データCSVパス。デフォルトはdata/boat/real_race_data.csv。

    Returns:
        pd.DataFrame: モデル学習用データフレーム。
    """
    if csv_path is None:
        csv_path = DATA_DIR / "real_race_data.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.error(f"データファイルが見つかりません: {csv_path}")
        return None

    raw = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
    logger.info(f"生データ読み込み: {len(raw)}レース")

    rows = []
    for idx, race in raw.iterrows():
        # レースIDを生成（日付_会場_レース番号）
        race_id = idx

        # 日付をdatetime化
        try:
            race_date = datetime.strptime(str(race["date"]), "%Y%m%d")
        except (ValueError, TypeError):
            race_date = None

        # 天候条件の数値化
        weather_map = {"晴": 0, "曇り": 1, "曇": 1, "雨": 2, "雪": 2, "霧": 1}
        weather_condition = weather_map.get(race.get("weather"), 0)
        weather_wind_speed = _safe_float(str(race.get("wind_speed", 0)), default=0.0)

        # 難水面の判定（江戸川・びわこ・福岡は難水面とする）
        hard_courses = {"03", "11", "22"}
        course_type = 1 if str(race.get("venue_code", "")) in hard_courses else 0

        # 着順情報を取得
        finish_cols = {lane: race.get(f"lane{lane}_finish") for lane in range(1, 7)}

        # 勝者・2着を特定
        winner_lane = None
        second_lane = None
        for lane, finish in finish_cols.items():
            f = _safe_int(str(finish)) if finish is not None else None
            if f == 1:
                winner_lane = lane
            elif f == 2:
                second_lane = lane

        # 着順データがなければスキップ
        if winner_lane is None:
            continue

        for lane in range(1, 7):
            prefix = f"lane{lane}_"

            # 級別を数値化
            racer_class_str = race.get(prefix + "class")
            racer_class = CLASS_MAP.get(str(racer_class_str), 2) if racer_class_str else 2

            racer_win_rate = _safe_float(str(race.get(prefix + "win_rate", "")), default=5.0)
            racer_place_rate = _safe_float(str(race.get(prefix + "2place_rate", "")), default=30.0)
            motor_2place_rate = _safe_float(str(race.get(prefix + "motor_2rate", "")), default=40.0)
            boat_2place_rate = _safe_float(str(race.get(prefix + "boat_2rate", "")), default=40.0)
            avg_start_timing = _safe_float(str(race.get(prefix + "start_timing", "")), default=0.17)

            # 新特徴量
            racer_3place_rate = _safe_float(str(race.get(prefix + "3place_rate", "")), default=50.0)
            local_win_rate = _safe_float(str(race.get(prefix + "local_win_rate", "")), default=racer_win_rate)
            local_2place_rate = _safe_float(str(race.get(prefix + "local_2place_rate", "")), default=racer_place_rate)
            flying_count = _safe_int(str(race.get(prefix + "flying_count", "")), default=0)
            late_count = _safe_int(str(race.get(prefix + "late_count", "")), default=0)
            weight = _safe_float(str(race.get(prefix + "weight", "")), default=52.0)
            wave_height = _safe_float(str(race.get("wave_height", 0)), default=0.0)

            # オッズ
            odds_val = _safe_float(str(race.get(f"odds_{lane}", "")), default=None)

            rows.append({
                "race_id": race_id,
                "race_date": race_date,
                "venue_code": str(race.get("venue_code", "")),
                "lane": lane,
                "racer_class": racer_class,
                "racer_win_rate": racer_win_rate,
                "racer_place_rate": racer_place_rate,
                "racer_3place_rate": racer_3place_rate,
                "racer_local_win_rate": local_win_rate,
                "racer_local_2place_rate": local_2place_rate,
                "motor_2place_rate": motor_2place_rate,
                "boat_2place_rate": boat_2place_rate,
                "avg_start_timing": avg_start_timing,
                "flying_count": flying_count,
                "late_count": late_count,
                "weight": weight,
                "weather_wind_speed": weather_wind_speed,
                "weather_condition": weather_condition,
                "wave_height": wave_height,
                "course_type": course_type,
                "odds": odds_val,
                "win": 1 if lane == winner_lane else 0,
                "place_top2": 1 if lane in (winner_lane, second_lane) else 0,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("変換結果が空です")
        return df

    # 統計情報の表示
    n_races = df["race_id"].nunique()
    logger.info(f"変換完了: {n_races}レース × 6艇 = {len(df)}行")
    logger.info("[枠番別勝率]")
    for lane in range(1, 7):
        wr = df[df["lane"] == lane]["win"].mean()
        logger.info(f"  {lane}号艇: {wr:.1%}")

    # モデル用CSVに保存
    model_csv_path = DATA_DIR / "model_training_data.csv"
    df.to_csv(model_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"モデル用データ保存: {model_csv_path}")

    return df


# =============================================================
# メイン
# =============================================================

if __name__ == "__main__":
    today = datetime.now()
    default_start = (today - timedelta(days=90)).strftime("%Y%m%d")
    default_end = today.strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="ボートレース公式サイトからレースデータを取得する"
    )
    parser.add_argument(
        "--start", default=default_start,
        help=f"開始日 YYYYMMDD (デフォルト: {default_start})"
    )
    parser.add_argument(
        "--end", default=default_end,
        help=f"終了日 YYYYMMDD (デフォルト: {default_end})"
    )
    parser.add_argument(
        "--venue", default=None,
        help="特定の場コード (例: 01=桐生, 24=大村)。省略時は全24場"
    )
    parser.add_argument(
        "--skip-odds", action="store_true",
        help="オッズ取得をスキップして高速化"
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="データ取得せず、既存CSVをモデル用フォーマットに変換のみ"
    )

    args = parser.parse_args()

    if args.convert_only:
        logger.info("=== モデル用フォーマット変換のみ ===")
        df = convert_to_model_format()
        if df is not None and not df.empty:
            logger.info(f"変換完了: {len(df)}行")
    else:
        logger.info("=== ボートレースデータ取得開始 ===")
        logger.info(f"期間: {args.start} ~ {args.end}")
        venues = [args.venue] if args.venue else None
        if venues:
            logger.info(f"対象会場: {VENUES.get(args.venue, args.venue)}")
        else:
            logger.info("対象会場: 全24場")

        df = fetch_historical_data(args.start, args.end, venues=venues, skip_odds=args.skip_odds)

        if df is not None and not df.empty:
            logger.info("=== モデル用フォーマット変換 ===")
            model_df = convert_to_model_format()
            if model_df is not None:
                logger.info(f"パイプライン完了: {len(model_df)}行のモデル用データ作成")
