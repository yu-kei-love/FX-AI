# ===========================================
# fetch_keirin_data.py
# 競輪レース結果データの取得
#
# keirin-station.comから公開レース結果をスクレイピング
# 2秒のディレイを入れてポライトスクレイピング
# ===========================================

import sys
import time
import csv
import re
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "keirin"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "real_race_results.csv"
PROGRESS_FILE = DATA_DIR / "scrape_progress.json"

# ===== 設定 =====
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en;q=0.9",
}

# ポライトスクレイピング: リクエスト間隔(秒)
REQUEST_DELAY = 2.0

BASE_URL = "https://keirin-station.com"

# 競輪場コード (主要場)
STADIUM_CODES = {
    "11": "函館", "12": "青森", "13": "いわき平",
    "21": "弥彦", "22": "前橋", "23": "取手",
    "24": "宇都宮", "25": "大宮", "26": "西武園",
    "27": "京王閣", "28": "立川", "31": "松戸",
    "32": "千葉", "33": "川崎", "34": "平塚",
    "35": "小田原", "36": "伊東温泉", "38": "静岡",
    "42": "名古屋", "43": "岐阜", "44": "大垣",
    "45": "豊橋", "46": "富山", "47": "松阪",
    "48": "四日市", "51": "福井", "52": "奈良",
    "53": "向日町", "54": "和歌山", "55": "岸和田",
    "56": "岸和田", "61": "玉野", "62": "広島",
    "63": "防府", "71": "高松", "72": "小松島",
    "73": "高知", "74": "松山", "81": "小倉",
    "82": "久留米", "83": "武雄", "84": "佐世保",
    "85": "別府", "86": "熊本",
}

# バンクサイズ (m)
BANK_SIZES = {
    "前橋": 335, "大宮": 500, "立川": 400, "京王閣": 400,
    "松戸": 333, "川崎": 400, "平塚": 400, "名古屋": 400,
    "岸和田": 400, "小倉": 400, "久留米": 400, "函館": 400,
}


class KeirinScraper:
    """keirin-station.comからレースデータを取得"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, url, retries=2):
        """GETリクエスト (ディレイ付き、リトライあり)"""
        time.sleep(REQUEST_DELAY)
        for attempt in range(retries + 1):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 404:
                    return None  # Race doesn't exist
                if resp.status_code == 500 and attempt < retries:
                    time.sleep(3)
                    continue
                resp.encoding = "utf-8"
                return BeautifulSoup(resp.text, "html.parser")
            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(3)
                    continue
                return None
        return None

    def get_active_stadiums(self, date_str: str) -> list:
        """Get list of stadium codes that had races on this date."""
        url = f"{BASE_URL}/keirindb/race/resultlist/"
        soup = self._get(url)
        if soup is None:
            return []

        codes = set()
        # Look for result links matching this date
        pattern = re.compile(rf"/keirindb/race/result/(\d+)/{date_str}/")
        for a_tag in soup.find_all("a", href=True):
            match = pattern.search(a_tag["href"])
            if match:
                codes.add(match.group(1))

        # Also try date-specific result list
        if not codes:
            for code in list(STADIUM_CODES.keys())[:5]:  # Try first 5
                list_url = f"{BASE_URL}/keirindb/race/resultlist/{code}/{date_str}/"
                soup2 = self._get(list_url)
                if soup2 and soup2.find("table"):
                    codes.add(code)

        return list(codes)

    def get_race_list_for_date(self, date_str: str) -> list:
        """
        指定日のレース一覧を取得
        date_str: "YYYYMMDD"
        Returns: list of (stadium_code, race_no) tuples
        """
        # keirin-station.comの結果一覧ページ
        url = f"{BASE_URL}/keirindb/race/resultlist/"
        soup = self._get(url)
        if soup is None:
            return []

        races = []
        # 結果リンクを探す: /keirindb/race/result/{stadium_id}/{date}/{race_no}/
        pattern = re.compile(rf"/keirindb/race/result/(\d+)/{date_str}/(\d+)/")

        for a_tag in soup.find_all("a", href=True):
            match = pattern.search(a_tag["href"])
            if match:
                stadium_code = match.group(1)
                race_no = int(match.group(2))
                races.append((stadium_code, race_no))

        # Also try the date-specific result list page for each known stadium
        if not races:
            for code in STADIUM_CODES:
                list_url = f"{BASE_URL}/keirindb/race/resultlist/{code}/{date_str}/"
                soup = self._get(list_url)
                if soup is None:
                    continue
                for a_tag in soup.find_all("a", href=True):
                    match = pattern.search(a_tag["href"])
                    if match:
                        stadium_code = match.group(1)
                        race_no = int(match.group(2))
                        races.append((stadium_code, race_no))
                if races:
                    break  # Found races for this date

        return list(set(races))  # Remove duplicates

    def parse_race_result(self, stadium_code: str, date_str: str, race_no: int) -> list:
        """
        個別レース結果をパース
        Returns: list of dicts (one per racer)
        """
        url = f"{BASE_URL}/keirindb/race/result/{stadium_code}/{date_str}/{race_no}/"
        soup = self._get(url)
        if soup is None:
            return []

        rows = []
        venue = STADIUM_CODES.get(stadium_code, f"Code{stadium_code}")

        # レースメタデータ
        race_info = {
            "race_date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            "venue": venue,
            "stadium_code": stadium_code,
            "race_no": race_no,
            "race_id": f"{date_str}_{stadium_code}_{race_no:02d}",
        }

        # グレード情報を探す
        grade_tag = soup.find(string=re.compile(r"[FGAS][12I]|GP"))
        race_info["grade"] = grade_tag.strip() if grade_tag else ""

        # 結果テーブルを探す
        result_table = None
        tables = soup.find_all("table")
        for table in tables:
            # 着順テーブルを特定
            ths = table.find_all("th")
            header_text = " ".join(th.get_text(strip=True) for th in ths)
            if "着" in header_text and ("車番" in header_text or "選手" in header_text):
                result_table = table
                break

        if result_table is None:
            # Try finding the table with 8 rows (header + 7 racers) and 8+ columns
            for table in tables:
                trs = table.find_all("tr")
                if len(trs) >= 8:
                    tds_first = trs[1].find_all("td") if len(trs) > 1 else []
                    if len(tds_first) >= 6:
                        result_table = table
                        break

        if result_table is None:
            return []

        # テーブル行をパース
        trs = result_table.find_all("tr")
        for tr in trs[1:]:  # Skip header
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            row = dict(race_info)

            try:
                # Fixed column layout (11 tds):
                # [0]=着(1着), [1]=車番(img alt), [2]=選手名(a), [3]=府県,
                # [4]=期別, [5]=級班, [6]=着差/ライン, [7]=上り, [8]=決まり手,
                # [9]=H/B, [10]=備考

                # 着順
                finish_text = tds[0].get_text(strip=True)
                finish_match = re.search(r"(\d+)", finish_text)
                if not finish_match:
                    continue
                row["finish"] = int(finish_match.group(1))

                # 車番 (post position) - in img alt attribute
                img = tds[1].find("img") if len(tds) > 1 else None
                if img and img.get("alt", "").isdigit():
                    row["post_position"] = int(img["alt"])
                else:
                    row["post_position"] = 0

                # 選手名 (racer name)
                if len(tds) > 2:
                    name_tag = tds[2].find("a")
                    if name_tag:
                        row["racer_name"] = name_tag.get_text(strip=True)
                        href = name_tag.get("href", "")
                        racer_id_match = re.search(r"/player/detail/(\d+)/", href)
                        row["racer_id"] = racer_id_match.group(1) if racer_id_match else ""
                    else:
                        row["racer_name"] = tds[2].get_text(strip=True)
                        row["racer_id"] = ""
                else:
                    row["racer_name"] = ""
                    row["racer_id"] = ""

                # 府県 (prefecture)
                row["prefecture"] = tds[3].get_text(strip=True) if len(tds) > 3 else ""

                # 期別 (term)
                row["term"] = tds[4].get_text(strip=True) if len(tds) > 4 else ""

                # 級班 (class: S1, S2, A1, A2, A3, L1)
                row["racer_class"] = tds[5].get_text(strip=True) if len(tds) > 5 else ""

                # 上り (last lap time)
                agari_text = tds[7].get_text(strip=True) if len(tds) > 7 else ""
                try:
                    row["last_lap"] = float(agari_text)
                except (ValueError, TypeError):
                    row["last_lap"] = 0.0

                # 決まり手 (winning technique)
                row["winning_move"] = tds[8].get_text(strip=True) if len(tds) > 8 else ""

                rows.append(row)
            except Exception as e:
                continue

        # 払戻金（オッズ）を探す
        payout_info = self._parse_payouts(soup)
        for row in rows:
            row.update(payout_info)

        return rows

    def _parse_payouts(self, soup) -> dict:
        """払戻金（配当金）をパース"""
        payouts = {
            "win_payout": 0,
            "exacta_payout": 0,
            "quinella_payout": 0,
            "trifecta_payout": 0,
            "trio_payout": 0,
        }

        def _extract_payout(text):
            """Extract payout from text like '2=4170円(1)' -> 170"""
            # Pattern: combo(single digits with = or -) followed by amount followed by 円
            # e.g., "2=4170円" -> combo "2=4", payout 170
            m = re.search(r"\d[=-]\d(?:[=-]\d)?(\d[\d,]*)円", text)
            if m:
                return int(m.group(1).replace(",", ""))
            # Fallback: just extract digits before 円
            m = re.search(r"(\d[\d,]*)円", text)
            if m:
                return int(m.group(1).replace(",", ""))
            return 0

        for table in soup.find_all("table"):
            table_text = table.get_text()

            if "車番連" in table_text:
                trs = table.find_all("tr")
                for tr in trs:
                    cells = tr.find_all(["td", "th"])
                    row_text = " ".join(c.get_text(strip=True) for c in cells)

                    if "複" in row_text and "円" in row_text:
                        payouts["quinella_payout"] = _extract_payout(row_text)
                    elif "単" in row_text and "円" in row_text:
                        payouts["exacta_payout"] = _extract_payout(row_text)

            if "3連勝" in table_text:
                trs = table.find_all("tr")
                for tr in trs:
                    cells = tr.find_all(["td", "th"])
                    row_text = " ".join(c.get_text(strip=True) for c in cells)

                    if "複" in row_text and "円" in row_text:
                        payouts["trio_payout"] = _extract_payout(row_text)
                    elif "単" in row_text and "円" in row_text:
                        payouts["trifecta_payout"] = _extract_payout(row_text)

        return payouts


def load_progress() -> set:
    """Load set of already-scraped dates."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
            return set(data.get("scraped_dates", []))
        except Exception:
            pass
    return set()


def save_progress(scraped_dates: set):
    """Save progress."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"scraped_dates": sorted(scraped_dates)}, f, indent=2)


def load_existing_data() -> pd.DataFrame:
    """Load existing scraped data."""
    if OUTPUT_CSV.exists():
        try:
            return pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        except Exception:
            pass
    return pd.DataFrame()


def save_data(df: pd.DataFrame):
    """Save data to CSV."""
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def generate_dates(from_date: str, to_date: str) -> list:
    """Generate date strings between from_date and to_date (YYYYMMDD)."""
    start = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(to_date, "%Y%m%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def scrape_keirin(from_date: str, to_date: str, max_races: int = None):
    """
    Main scraping function.
    Scrapes race results from keirin-station.com.
    """
    scraper = KeirinScraper()
    existing_df = load_existing_data()
    scraped_dates = load_progress()

    all_rows = []
    if not existing_df.empty:
        all_rows.extend(existing_df.to_dict("records"))

    dates = generate_dates(from_date, to_date)
    total_new = 0

    print(f"Scraping keirin results: {from_date} to {to_date} ({len(dates)} dates)")

    for date_str in dates:
        if date_str in scraped_dates:
            continue

        if max_races and total_new >= max_races:
            break

        print(f"  Date: {date_str}")

        # First, try to get the result list page to find active stadiums
        active_stadiums = scraper.get_active_stadiums(date_str)
        if not active_stadiums:
            # Fallback: try all stadiums
            active_stadiums = list(STADIUM_CODES.keys())

        found_any = False
        for code in active_stadiums:
            if max_races and total_new >= max_races:
                break

            stadium_found = False
            # Try races 1-12 for this stadium
            for race_no in range(1, 13):
                rows = scraper.parse_race_result(code, date_str, race_no)
                if rows:
                    all_rows.extend(rows)
                    total_new += 1
                    found_any = True
                    stadium_found = True
                    print(f"    {STADIUM_CODES.get(code, code)} R{race_no}: {len(rows)} racers")
                elif stadium_found:
                    # Had races but this one doesn't exist, no more races at this stadium
                    break

            if found_any:
                # Save incrementally
                df = pd.DataFrame(all_rows)
                df = df.drop_duplicates(subset=["race_id", "post_position"], keep="last")
                save_data(df)

        scraped_dates.add(date_str)
        save_progress(scraped_dates)

        if found_any:
            print(f"    Total: {len(all_rows)} rows so far")

    # Final save
    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset=["race_id", "post_position"], keep="last")
        save_data(df)
        print(f"\nDone: {len(df)} rows, {df['race_id'].nunique()} races -> {OUTPUT_CSV}")
        return df
    return pd.DataFrame()


# ===========================================
# CLI
# ===========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape keirin race results")
    parser.add_argument("--from-date", default="20250101",
                        help="Start date YYYYMMDD (default: 20250101)")
    parser.add_argument("--to-date", default="20260325",
                        help="End date YYYYMMDD (default: 20260325)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: scrape 1 date only")
    parser.add_argument("--max-races", type=int, default=None,
                        help="Maximum races to scrape")
    args = parser.parse_args()

    if args.test:
        # Test with a known date
        scrape_keirin("20260325", "20260325", max_races=5)
    else:
        scrape_keirin(args.from_date, args.to_date, args.max_races)
