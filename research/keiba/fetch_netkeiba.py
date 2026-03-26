# ===========================================
# fetch_netkeiba.py
# netkeiba.comからJRA中央競馬の過去レース結果をスクレイピング
#
# 使い方:
#   python fetch_netkeiba.py                    # 直近数日分のテストスクレイプ
#   python fetch_netkeiba.py --year 2024        # 2024年の全レース
#   python fetch_netkeiba.py --from 2020 --to 2025  # 2020-2025年
#   python fetch_netkeiba.py --date 20240101    # 特定日のレース
#
# 出力: data/keiba/real_race_results.csv (インクリメンタル保存)
# ===========================================

import re
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ===== 設定 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "keiba"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "real_race_results.csv"
HORSE_CACHE_CSV = DATA_DIR / "horse_details_cache.csv"
PROGRESS_FILE = DATA_DIR / "scrape_progress.json"

# netkeiba URL patterns
BASE_URL = "https://db.netkeiba.com"
RACE_LIST_URL = "https://db.netkeiba.com/race/list/{date}/"
RACE_DETAIL_URL = "https://db.netkeiba.com/race/{race_id}/"
HORSE_URL = "https://db.netkeiba.com/horse/{horse_id}/"

# JRA venue codes
VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

# HTTP settings
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
}
REQUEST_DELAY = 1.5  # seconds between requests (polite scraping)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================
# HTTP helper
# ===========================================

class NetkeibaSession:
    """Polite HTTP session with retry and delay."""

    def __init__(self, delay: float = REQUEST_DELAY):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self._last_request_time = 0.0

    def get(self, url: str, encoding: str = None) -> Optional[BeautifulSoup]:
        """Fetch a page with polite delay and return parsed HTML."""
        # Respect delay between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            resp = self.session.get(url, timeout=30)
            self._last_request_time = time.time()

            if resp.status_code == 404:
                log.debug(f"404: {url}")
                return None
            resp.raise_for_status()

            # netkeiba uses EUC-JP for older pages, UTF-8 for newer
            if encoding:
                resp.encoding = encoding
            elif "euc-jp" in resp.text[:500].lower() or "euc_jp" in resp.text[:500].lower():
                resp.encoding = "euc-jp"
            elif resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding

            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Request failed: {url} -> {e}")
            return None


# ===========================================
# Race list scraper (find race IDs for a given date)
# ===========================================

def get_race_ids_for_date(session: NetkeibaSession, date_str: str) -> list:
    """
    Get all race IDs for a given date (format: YYYYMMDD).
    Scrapes the race list page and extracts race IDs from links.
    """
    url = RACE_LIST_URL.format(date=date_str)
    soup = session.get(url)
    if soup is None:
        return []

    race_ids = []
    # Look for links to individual race pages
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Pattern: /race/YYYYNNDDRRXX/ or /race/XXXXXXXXXXXX/
        match = re.search(r"/race/(\d{12})/", href)
        if match:
            rid = match.group(1)
            if rid not in race_ids:
                race_ids.append(rid)

    log.info(f"Date {date_str}: found {len(race_ids)} races")
    return race_ids


# ===========================================
# Individual race result scraper
# ===========================================

def parse_race_result(session: NetkeibaSession, race_id: str) -> Optional[pd.DataFrame]:
    """
    Parse a single race result page from netkeiba.
    Returns a DataFrame with one row per horse.
    """
    url = RACE_DETAIL_URL.format(race_id=race_id)
    soup = session.get(url)
    if soup is None:
        return None

    rows = []

    # --- Race info (header) ---
    race_info = _parse_race_info(soup, race_id)
    if race_info is None:
        log.warning(f"Could not parse race info for {race_id}")
        return None

    # --- Result table ---
    # The main result table has class "race_table_01" on netkeiba
    result_table = soup.find("table", class_="race_table_01")
    if result_table is None:
        # Try alternative table selectors
        result_table = soup.find("table", {"summary": re.compile("レース結果", re.I)})
    if result_table is None:
        log.warning(f"No result table found for race {race_id}")
        return None

    tbody = result_table.find("tbody")
    if tbody is None:
        tbody = result_table

    tr_list = tbody.find_all("tr")
    for tr in tr_list:
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue

        try:
            horse_row = _parse_result_row(tds, race_id, race_info)
            if horse_row is not None:
                rows.append(horse_row)
        except Exception as e:
            log.debug(f"Error parsing row in race {race_id}: {e}")
            continue

    if not rows:
        log.warning(f"No horse rows parsed for race {race_id}")
        return None

    df = pd.DataFrame(rows)
    df["field_size"] = len(df)
    return df


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> Optional[dict]:
    """Extract race metadata from the page header."""
    info = {"race_id": race_id}

    # Race name
    race_name_tag = soup.find("h1", class_="racedata_title") or soup.find("dl", class_="racedata")
    if race_name_tag is None:
        # Alternative: look for the race data section
        data_intro = soup.find("div", class_="data_intro")
        if data_intro:
            h1 = data_intro.find("h1")
            if h1:
                info["race_name"] = h1.get_text(strip=True)
        else:
            info["race_name"] = ""
    else:
        info["race_name"] = race_name_tag.get_text(strip=True)

    # Race conditions (distance, track type, weather, track condition)
    # Usually in a <span> or <diary_snap_cut> or <p> with race details
    race_data_span = None
    data_intro = soup.find("div", class_="data_intro")
    if data_intro:
        spans = data_intro.find_all("span")
        for s in spans:
            text = s.get_text(strip=True)
            if "m" in text or "芝" in text or "ダ" in text:
                race_data_span = text
                break
        if race_data_span is None:
            # Try the diary_snap_cut or p tags
            p_tags = data_intro.find_all("p")
            for p in p_tags:
                text = p.get_text(strip=True)
                if "m" in text or "芝" in text or "ダ" in text:
                    race_data_span = text
                    break

    if race_data_span is None:
        # Fallback: try to find in any element containing distance info
        for tag in soup.find_all(["span", "p", "div"]):
            text = tag.get_text(strip=True)
            if re.search(r"(芝|ダ)[左右内外]*\d{3,4}m", text):
                race_data_span = text
                break

    if race_data_span:
        # Parse track type
        if "芝" in race_data_span:
            info["track_type"] = 0  # 芝
        elif "ダ" in race_data_span:
            info["track_type"] = 1  # ダート
        else:
            info["track_type"] = 0

        # Parse distance
        dist_match = re.search(r"(\d{3,4})m", race_data_span)
        if dist_match:
            info["distance"] = int(dist_match.group(1))
        else:
            info["distance"] = 0

        # Parse track condition
        cond_map = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不良": 3}
        info["track_condition"] = 0
        for cond_text, cond_val in cond_map.items():
            if cond_text in race_data_span:
                info["track_condition"] = cond_val
                break
    else:
        info["track_type"] = 0
        info["distance"] = 0
        info["track_condition"] = 0

    # Race date from race_id (first 4 digits = year, but actual date needs checking)
    # race_id format: YYYYVVDDRRXX where YYYY=year, VV=venue, DD=kai, RR=day, XX=race_num
    # Actually on netkeiba the format can vary. Let's extract from page.
    date_tag = None
    for tag in soup.find_all(["p", "div", "span"]):
        text = tag.get_text(strip=True)
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if date_match:
            y, m, d = date_match.groups()
            info["race_date"] = f"{y}-{int(m):02d}-{int(d):02d}"
            date_tag = True
            break
    if not date_tag:
        # Fallback: extract from race_id (first 4 chars = year)
        year = race_id[:4]
        info["race_date"] = f"{year}-01-01"

    # Race class (G1, G2, G3, Listed, Open, etc.)
    race_name = info.get("race_name", "")
    info["race_class"] = _classify_race(race_name, soup)

    # Venue
    venue_code = race_id[4:6]
    info["venue"] = VENUE_CODES.get(venue_code, "不明")

    return info


def _classify_race(race_name: str, soup: BeautifulSoup) -> int:
    """Classify race level: G1=5, G2=4, G3=3, Listed=2, Open=1, 条件=0"""
    text = race_name
    # Also check for grade icons/images
    for img in soup.find_all("img"):
        alt = img.get("alt", "")
        text += " " + alt

    if re.search(r"G\s*[IⅠ]\b|GI\b|G1\b", text, re.I):
        # Make sure it's not G2 or G3
        if not re.search(r"G\s*[IⅠ]{2,}|GII|G2|GIII|G3", text, re.I):
            return 5
    if re.search(r"G\s*[IⅠ]{2}|GII\b|G2\b", text, re.I):
        return 4
    if re.search(r"G\s*[IⅠ]{3}|GIII\b|G3\b", text, re.I):
        return 3
    if "リステッド" in text or "Listed" in text or "(L)" in text:
        return 2
    if "オープン" in text or "OP" in text or "Open" in text:
        return 1
    return 0


def _parse_result_row(tds: list, race_id: str, race_info: dict) -> Optional[dict]:
    """Parse a single row from the result table."""
    row = {}

    # Column layout for netkeiba race result table:
    # 0: 着順, 1: 枠番, 2: 馬番, 3: 馬名, 4: 性齢, 5: 斤量,
    # 6: 騎手, 7: タイム, 8: 着差, 9: (単勝)人気, 10: 単勝オッズ,
    # 11: 後3F(上がり3F), 12: コーナー通過順, 13: 厩舎(調教師), 14: 馬体重
    # Note: column order may vary slightly

    # 着順 (finish position)
    finish_text = tds[0].get_text(strip=True)
    if not finish_text.isdigit():
        # 中止, 除外, 取消 etc.
        return None
    row["finish"] = int(finish_text)

    # 枠番 (gate number)
    waku_text = tds[1].get_text(strip=True)
    row["post_position"] = int(waku_text) if waku_text.isdigit() else 0

    # 馬番 (horse number)
    umaban_text = tds[2].get_text(strip=True)
    row["horse_number"] = int(umaban_text) if umaban_text.isdigit() else 0

    # 馬名 (horse name) + horse_id
    horse_tag = tds[3].find("a")
    if horse_tag:
        row["horse_name"] = horse_tag.get_text(strip=True)
        href = horse_tag.get("href", "")
        horse_id_match = re.search(r"/horse/(\w+)/", href)
        row["horse_id"] = horse_id_match.group(1) if horse_id_match else ""
    else:
        row["horse_name"] = tds[3].get_text(strip=True)
        row["horse_id"] = ""

    # 性齢 (sex + age)
    sex_age_text = tds[4].get_text(strip=True)
    row["sex"], row["horse_age"] = _parse_sex_age(sex_age_text)

    # 斤量 (weight carried)
    kinryo_text = tds[5].get_text(strip=True)
    try:
        row["weight_carried"] = float(kinryo_text)
    except (ValueError, TypeError):
        row["weight_carried"] = 55.0

    # 騎手 (jockey) + jockey_id
    jockey_tag = tds[6].find("a")
    if jockey_tag:
        row["jockey_name"] = jockey_tag.get_text(strip=True)
        href = jockey_tag.get("href", "")
        jockey_id_match = re.search(r"/jockey/(\w+)/", href)
        row["jockey_id"] = jockey_id_match.group(1) if jockey_id_match else ""
    else:
        row["jockey_name"] = tds[6].get_text(strip=True)
        row["jockey_id"] = ""

    # タイム (race time)
    time_text = tds[7].get_text(strip=True)
    row["race_time"] = _parse_time(time_text)

    # 着差 (margin)
    margin_text = tds[8].get_text(strip=True)
    row["margin"] = margin_text

    # Handle varying column layouts - netkeiba sometimes has different column counts
    n_tds = len(tds)

    # Try to find 人気 (popularity), オッズ (odds), 上がり3F, 通過順, 調教師, 馬体重
    # These may be at different indices depending on the page version
    row["popularity"] = 0
    row["odds"] = 0.0
    row["last_3f"] = 0.0
    row["corner_positions"] = ""
    row["trainer_name"] = ""
    row["trainer_id"] = ""
    row["horse_weight"] = 0
    row["weight_change"] = 0

    # netkeiba.com result table has ~25 columns:
    # [0]=着順, [1]=枠番, [2]=馬番, [3]=馬名, [4]=性齢, [5]=斤量, [6]=騎手, [7]=タイム,
    # [8]=着差, [9-13]=various, [14]=通過順, [15]=上がり3F, [16]=単勝オッズ, [17]=人気,
    # [18]=馬体重, [22]=調教師

    if n_tds >= 15:
        # コーナー通過順 (column 14)
        corner_text = tds[14].get_text(strip=True) if n_tds > 14 else ""
        row["corner_positions"] = corner_text

    if n_tds >= 16:
        # 上がり3F (last 3 furlongs time, column 15)
        f3_text = tds[15].get_text(strip=True) if n_tds > 15 else ""
        try:
            row["last_3f"] = float(f3_text)
        except (ValueError, TypeError):
            row["last_3f"] = 0.0

    if n_tds >= 17:
        # 単勝オッズ (win odds, column 16)
        odds_text = tds[16].get_text(strip=True) if n_tds > 16 else ""
        try:
            row["odds"] = float(odds_text.replace(",", ""))
        except (ValueError, TypeError):
            row["odds"] = 0.0

    if n_tds >= 18:
        # 人気 (popularity, column 17)
        pop_text = tds[17].get_text(strip=True) if n_tds > 17 else ""
        row["popularity"] = int(pop_text) if pop_text.isdigit() else 0

    if n_tds >= 19:
        # 馬体重 (horse weight, column 18) - format: "470(+4)" or "470(-2)"
        weight_text = tds[18].get_text(strip=True) if n_tds > 18 else ""
        row["horse_weight"], row["weight_change"] = _parse_horse_weight(weight_text)

    if n_tds >= 23:
        # 調教師 (trainer, column 22)
        trainer_tag = tds[22].find("a") if n_tds > 22 else None
        if trainer_tag:
            row["trainer_name"] = trainer_tag.get_text(strip=True)
            href = trainer_tag.get("href", "")
            trainer_id_match = re.search(r"/trainer/(\w+)/", href)
            row["trainer_id"] = trainer_id_match.group(1) if trainer_id_match else ""

    # Add race info
    row["race_id"] = race_id
    row["race_date"] = race_info.get("race_date", "")
    row["race_name"] = race_info.get("race_name", "")
    row["distance"] = race_info.get("distance", 0)
    row["track_type"] = race_info.get("track_type", 0)
    row["track_condition"] = race_info.get("track_condition", 0)
    row["race_class"] = race_info.get("race_class", 0)
    row["venue"] = race_info.get("venue", "")

    return row


def _parse_sex_age(text: str) -> tuple:
    """Parse '牡3' -> (1, 3), '牝4' -> (0, 4), 'セ5' -> (2, 5)"""
    sex_map = {"牝": 0, "牡": 1, "セ": 2}
    sex = 1  # default 牡
    age = 3  # default
    if text:
        for k, v in sex_map.items():
            if k in text:
                sex = v
                age_part = text.replace(k, "").strip()
                if age_part.isdigit():
                    age = int(age_part)
                break
    return sex, age


def _parse_time(text: str) -> float:
    """Parse race time: '1:34.5' -> 94.5 seconds"""
    if not text or text == "--":
        return 0.0
    try:
        if ":" in text:
            parts = text.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(text)
    except (ValueError, TypeError):
        return 0.0


def _parse_horse_weight(text: str) -> tuple:
    """Parse '470(+4)' -> (470, 4), '470(-2)' -> (470, -2)"""
    if not text:
        return 0, 0
    # Remove spaces
    text = text.strip()
    match = re.match(r"(\d+)\s*\(([+-]?\d+)\)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Just weight, no change
    weight_match = re.match(r"(\d+)", text)
    if weight_match:
        return int(weight_match.group(1)), 0
    return 0, 0


# ===========================================
# Horse detail scraper (bloodline, trainer, past results)
# ===========================================

def fetch_horse_details(session: NetkeibaSession, horse_id: str) -> Optional[dict]:
    """
    Fetch horse detail page for bloodline (sire, broodmare sire) info.
    Returns dict with sire, broodmare_sire, trainer.
    """
    if not horse_id:
        return None

    url = HORSE_URL.format(horse_id=horse_id)
    soup = session.get(url)
    if soup is None:
        return None

    details = {"horse_id": horse_id}

    # Bloodline table (血統テーブル)
    blood_table = soup.find("table", class_="blood_table")
    if blood_table is None:
        blood_table = soup.find("table", {"summary": re.compile("血統", re.I)})

    if blood_table:
        tds = blood_table.find_all("td")
        # Typically: Father at index 0 or 1, Mother's Father deeper in the table
        links = blood_table.find_all("a")
        if len(links) >= 1:
            details["sire"] = links[0].get_text(strip=True)
        if len(links) >= 3:
            details["broodmare_sire"] = links[2].get_text(strip=True)

    # Profile section for trainer
    profile = soup.find("div", class_="db_prof_area_02")
    if profile is None:
        profile = soup.find("table", class_="db_prof_table")
    if profile:
        for a_tag in profile.find_all("a", href=True):
            if "/trainer/" in a_tag["href"]:
                details["trainer"] = a_tag.get_text(strip=True)
                break

    return details


# ===========================================
# Date generation helpers
# ===========================================

def generate_jra_dates(year: int) -> list:
    """
    Generate all Saturdays and Sundays in a given year (JRA race days).
    JRA races are held on weekends (Sat/Sun), with some exceptions.
    """
    dates = []
    current = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    while current <= end:
        if current.weekday() in (5, 6):  # Saturday=5, Sunday=6
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def generate_date_range(start_year: int, end_year: int) -> list:
    """Generate all weekend dates across multiple years."""
    all_dates = []
    for year in range(start_year, end_year + 1):
        all_dates.extend(generate_jra_dates(year))
    return all_dates


# ===========================================
# Incremental save / progress tracking
# ===========================================

def load_existing_data() -> pd.DataFrame:
    """Load existing scraped data if available."""
    if OUTPUT_CSV.exists():
        try:
            df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
            log.info(f"Loaded existing data: {len(df)} rows, "
                     f"{df['race_id'].nunique()} races")
            return df
        except Exception as e:
            log.warning(f"Could not load existing data: {e}")
    return pd.DataFrame()


def save_data(df: pd.DataFrame):
    """Save data to CSV (incremental)."""
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info(f"Saved {len(df)} rows ({df['race_id'].nunique()} races) -> {OUTPUT_CSV}")


def load_progress() -> set:
    """Load set of already-scraped dates."""
    import json
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
            return set(data.get("scraped_dates", []))
        except Exception:
            pass
    return set()


def save_progress(scraped_dates: set):
    """Save progress (scraped dates)."""
    import json
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"scraped_dates": sorted(scraped_dates)}, f, indent=2)


# ===========================================
# Main scraper orchestration
# ===========================================

def scrape_races(dates: list, fetch_horse_info: bool = False,
                 max_races: int = None) -> pd.DataFrame:
    """
    Main scraping function. Scrapes race results for given dates.

    Args:
        dates: List of date strings in YYYYMMDD format
        fetch_horse_info: Whether to also fetch horse bloodline info
        max_races: Maximum number of races to scrape (for testing)

    Returns:
        DataFrame with all race results
    """
    session = NetkeibaSession()
    existing_df = load_existing_data()
    scraped_dates = load_progress()

    all_rows = []
    if not existing_df.empty:
        all_rows.append(existing_df)

    total_new_races = 0
    horse_details_cache = {}

    # Load horse details cache
    if HORSE_CACHE_CSV.exists() and fetch_horse_info:
        try:
            hdf = pd.read_csv(HORSE_CACHE_CSV, encoding="utf-8-sig")
            for _, row in hdf.iterrows():
                horse_details_cache[row["horse_id"]] = row.to_dict()
        except Exception:
            pass

    for date_str in dates:
        if date_str in scraped_dates:
            log.debug(f"Skipping already-scraped date: {date_str}")
            continue

        if max_races and total_new_races >= max_races:
            log.info(f"Reached max_races limit ({max_races})")
            break

        log.info(f"Scraping date: {date_str}")
        race_ids = get_race_ids_for_date(session, date_str)

        if not race_ids:
            log.info(f"  No races found for {date_str}")
            scraped_dates.add(date_str)
            save_progress(scraped_dates)
            continue

        date_frames = []
        for rid in race_ids:
            if max_races and total_new_races >= max_races:
                break

            # Check if we already have this race
            if not existing_df.empty and rid in existing_df["race_id"].values:
                log.debug(f"  Race {rid} already in data, skipping")
                continue

            race_df = parse_race_result(session, rid)
            if race_df is not None and not race_df.empty:
                # Optionally fetch horse bloodline info
                if fetch_horse_info:
                    for idx, row in race_df.iterrows():
                        hid = row.get("horse_id", "")
                        if hid and hid not in horse_details_cache:
                            details = fetch_horse_details(session, hid)
                            if details:
                                horse_details_cache[hid] = details
                        if hid in horse_details_cache:
                            d = horse_details_cache[hid]
                            race_df.loc[idx, "sire"] = d.get("sire", "")
                            race_df.loc[idx, "broodmare_sire"] = d.get("broodmare_sire", "")

                date_frames.append(race_df)
                total_new_races += 1
                log.info(f"  Race {rid}: {len(race_df)} horses parsed")

        if date_frames:
            date_df = pd.concat(date_frames, ignore_index=True)
            all_rows.append(date_df)

            # Incremental save every date
            combined = pd.concat(all_rows, ignore_index=True)
            combined = combined.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
            save_data(combined)

        scraped_dates.add(date_str)
        save_progress(scraped_dates)

    # Final save
    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
        result = result.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
        save_data(result)

        # Save horse details cache
        if horse_details_cache and fetch_horse_info:
            hdf = pd.DataFrame(list(horse_details_cache.values()))
            hdf.to_csv(HORSE_CACHE_CSV, index=False, encoding="utf-8-sig")

        return result
    else:
        return existing_df if not existing_df.empty else pd.DataFrame()


# ===========================================
# CLI
# ===========================================

def main():
    parser = argparse.ArgumentParser(description="Scrape race results from netkeiba.com")
    parser.add_argument("--year", type=int, help="Scrape a specific year")
    parser.add_argument("--from-year", type=int, dest="from_year",
                        help="Start year for range scraping")
    parser.add_argument("--to-year", type=int, dest="to_year",
                        help="End year for range scraping")
    parser.add_argument("--date", type=str, help="Scrape a specific date (YYYYMMDD)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: scrape just 2-3 recent dates")
    parser.add_argument("--horse-info", action="store_true", dest="horse_info",
                        help="Also fetch horse bloodline info (slower)")
    parser.add_argument("--max-races", type=int, dest="max_races",
                        help="Max number of races to scrape")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("netkeiba.com Race Results Scraper")
    log.info("=" * 50)

    if args.date:
        dates = [args.date]
    elif args.year:
        dates = generate_jra_dates(args.year)
        log.info(f"Year {args.year}: {len(dates)} weekend dates")
    elif args.from_year and args.to_year:
        dates = generate_date_range(args.from_year, args.to_year)
        log.info(f"Range {args.from_year}-{args.to_year}: {len(dates)} weekend dates")
    elif args.test:
        # Test mode: use a few known recent dates
        # Use dates from late 2024 (known to have races)
        dates = ["20241221", "20241222", "20241228"]
        log.info(f"Test mode: {len(dates)} dates")
    else:
        # Default: scrape 2024 data
        dates = generate_jra_dates(2024)
        log.info(f"Default: year 2024, {len(dates)} weekend dates")

    result = scrape_races(
        dates,
        fetch_horse_info=args.horse_info,
        max_races=args.max_races,
    )

    if not result.empty:
        log.info("=" * 50)
        log.info(f"Total races: {result['race_id'].nunique()}")
        log.info(f"Total rows: {len(result)}")
        log.info(f"Date range: {result['race_date'].min()} ~ {result['race_date'].max()}")
        log.info(f"Output: {OUTPUT_CSV}")
    else:
        log.warning("No data was scraped.")

    log.info("Done.")


if __name__ == "__main__":
    main()
