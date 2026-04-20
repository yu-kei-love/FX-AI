# ===========================================
# scraper/scraper_netkeirin_api.py
# netkeirin の AplRaceOdds API を直叩きする軽量スクレイパー
# (Playwright 不要、requests のみ)
#
# 発見:
#   POST https://keirin.netkeiba.com/api/race/
#   data: input=UTF-8&output=json&class=AplRaceOdds&method=get&compress=1&race_id={12桁}
#   → {"status":"OK","data":{"nkrace_odds::{race_id}": base64+zlib圧縮}}
#
# 圧縮解除後の JSON:
#   {
#     "official_dt": 日時,
#     "list_5": 2車複 (上位),
#     "list_6": 2車単 (上位),
#     "list_7": ワイド (上位),
#     "list_8": 3連複 (上位),
#     "list_9": 3連単 (上位 60combos),
#   }
#   各要素: [combo_key, odds, ?, popularity_rank]
#
# race_id 形式:
#   12桁 YYYYMMDDJJRR (JJ = JKA code)
#
# DB 上の race_id は {venue_id:02d}_{YYYYMMDD}_{RR:02d} なので変換が必要
#
# 倫理: delay 3.0s 厳守、商用不可
# ===========================================

import argparse
import base64
import json
import logging
import sqlite3
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCRAPER_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRAPER_DIR.parent
_DATA_DIR = _PROJECT_DIR / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

try:
    from bank_master import BANK_MASTER
    VENUE_ID_TO_JKA = {
        int(info["venue_id"]): info["jka_code"]
        for info in BANK_MASTER.values()
        if info.get("venue_id") and info.get("jka_code")
    }
except ImportError:
    BANK_MASTER = {}
    VENUE_ID_TO_JKA = {}

PROJECT_ROOT = _PROJECT_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"
FAILED_LOG = _SCRAPER_DIR / "failed_netkeirin.log"

API_URL = "https://keirin.netkeiba.com/api/race/"


class NetkeirinApiScraper:
    """netkeirin AplRaceOdds API 直叩きスクレイパー"""

    def __init__(self, db_path=None, delay=3.0, jyo_cds=None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = max(delay, 3.0)
        self.jyo_cds = [int(c) for c in jyo_cds] if jyo_cds else None
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
            "Accept-Language": "ja,en;q=0.9",
        })
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS odds_netkeirin (
                race_id    TEXT NOT NULL,
                odds_type  TEXT NOT NULL,    -- '2t','2f','wide','trio','3t'
                combo_key  TEXT NOT NULL,     -- '010203' 等
                odds       REAL,
                popularity INTEGER,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (race_id, odds_type, combo_key)
            );
            CREATE INDEX IF NOT EXISTS idx_ok_race
                ON odds_netkeirin(race_id);
        """)
        conn.commit()
        conn.close()

    def _polite_sleep(self):
        time.sleep(self.delay)

    def _fetch_odds_json(self, race_id_nk, retry=3):
        """AplRaceOdds API を叩いて decompress 済み JSON を返す"""
        for attempt in range(retry):
            try:
                referer = f"https://keirin.netkeiba.com/race/odds/?race_id={race_id_nk}"
                self.session.headers["Referer"] = referer
                r = self.session.post(API_URL, data={
                    "input": "UTF-8", "output": "json",
                    "class": "AplRaceOdds", "method": "get",
                    "compress": "1", "race_id": race_id_nk,
                }, timeout=20)
                if r.status_code != 200:
                    logger.debug("status %d race=%s", r.status_code, race_id_nk)
                    time.sleep(2 ** attempt)
                    continue
                j = json.loads(r.text)
                if j.get("status") != "OK":
                    return None  # データなし (NG)
                data = j.get("data") or {}
                key = f"nkrace_odds::{race_id_nk}"
                b64 = data.get(key)
                if not b64:
                    return None
                raw = base64.b64decode(b64)
                dec = zlib.decompress(raw)
                return json.loads(dec)
            except requests.RequestException as e:
                logger.warning("req err %s: %s", race_id_nk, e)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.debug("parse err %s: %s", race_id_nk, e)
                return None
        return None

    def _parse_list(self, raw_list, expected_combo_chars):
        """
        list_X の要素 = [combo_key, odds, ?, popularity]
        combo_key は 2桁 or 4桁 or 6桁 (例: '0102', '010203')
        """
        out = []
        for entry in raw_list:
            try:
                ck = str(entry[0])
                odds = float(entry[1])
                pop = int(entry[3]) if len(entry) > 3 else 0
                if len(ck) != expected_combo_chars:
                    continue
                out.append((ck, odds, pop))
            except (ValueError, TypeError, IndexError):
                continue
        return out

    def race_id_db_to_nk(self, race_id_db):
        """DB形式(venue_id_YYYYMMDD_RR) → netkeirin形式(YYYYMMDDJJRR)"""
        try:
            parts = race_id_db.split("_")
            venue_id = int(parts[0])
            date_str = parts[1]
            race_no = int(parts[2])
            jka = VENUE_ID_TO_JKA.get(venue_id)
            if jka is None:
                return None
            return f"{date_str}{jka}{race_no:02d}"
        except Exception:
            return None

    def scrape_one_race(self, race_id_db):
        nk_id = self.race_id_db_to_nk(race_id_db)
        if nk_id is None:
            return None

        data = self._fetch_odds_json(nk_id)
        if data is None:
            return None

        # 各券種をパース
        result = {
            "2f":   self._parse_list(data.get("list_5", []), 4),   # 2車複
            "2t":   self._parse_list(data.get("list_6", []), 4),   # 2車単
            "wide": self._parse_list(data.get("list_7", []), 4),   # ワイド
            "trio": self._parse_list(data.get("list_8", []), 6),   # 3連複
            "3t":   self._parse_list(data.get("list_9", []), 6),   # 3連単
        }
        return result

    def save_odds(self, race_id_db, odds_dict):
        if not odds_dict:
            return 0
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n = 0
        for odds_type, lst in odds_dict.items():
            for ck, o, pop in lst:
                try:
                    cur.execute("""
                        INSERT INTO odds_netkeirin
                            (race_id, odds_type, combo_key, odds,
                             popularity, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(race_id, odds_type, combo_key)
                        DO NOTHING
                    """, (race_id_db, odds_type, ck, o, pop, now))
                    n += cur.rowcount
                except sqlite3.Error:
                    pass
        conn.commit()
        conn.close()
        return n

    def _log_failed(self, race_id, reason):
        try:
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\t{race_id}\t{reason}\n")
        except Exception:
            pass

    def _get_target_races(self):
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_id FROM odds_netkeirin")
        done = {row[0] for row in cur.fetchall()}
        sql = """
            SELECT race_id, jyo_cd, race_date, race_no
            FROM races
            WHERE race_date BETWEEN '20220101' AND '20241231'
        """
        params = []
        if self.jyo_cds:
            ph = ",".join(["?"] * len(self.jyo_cds))
            sql += f" AND jyo_cd IN ({ph})"
            params.extend(self.jyo_cds)
        sql += " ORDER BY race_date, jyo_cd, race_no"
        cur.execute(sql, params)
        all_races = cur.fetchall()
        conn.close()
        targets = [r for r in all_races if r[0] not in done]
        logger.info("対象: %d (既存: %d / 全体: %d)",
                    len(targets), len(done), len(all_races))
        return targets

    def backfill(self, limit=None):
        targets = self._get_target_races()
        if limit:
            targets = targets[:limit]
        total = len(targets)
        if total == 0:
            logger.info("処理対象なし")
            return
        success = failed = 0
        for idx, (race_id, jyo_cd, race_date, race_no) in enumerate(targets, 1):
            try:
                odds = self.scrape_one_race(race_id)
                if not odds or all(len(v) == 0 for v in odds.values()):
                    failed += 1
                    self._log_failed(race_id, "no_odds")
                else:
                    saved = self.save_odds(race_id, odds)
                    success += 1
                    if idx % 50 == 0 or idx < 5:
                        n_3t = len(odds.get("3t", []))
                        logger.info("[%d/%d] %s: 3t=%d 保存=%d (s=%d f=%d)",
                                    idx, total, race_id, n_3t, saved,
                                    success, failed)
                if failed >= 10 and failed == idx:
                    logger.error("10件連続失敗。停止")
                    break
            except KeyboardInterrupt:
                logger.info("中断")
                break
            except Exception as e:
                logger.error("err %s: %s", race_id, e)
                failed += 1
                self._log_failed(race_id, str(e))
            self._polite_sleep()
        logger.info("完了: s=%d f=%d total=%d", success, failed, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str,
                        help="DB race_id (venue_id_YYYYMMDD_RR) でテスト")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--jyo_cds", type=str, default=None,
                        help="カンマ区切り venue_id")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    jyo_cds = None
    if args.jyo_cds:
        jyo_cds = [int(x.strip()) for x in args.jyo_cds.split(",") if x.strip()]

    sc = NetkeirinApiScraper(db_path=args.db, delay=args.delay,
                             jyo_cds=jyo_cds)

    if args.test:
        logger.info("test: %s → nk=%s", args.test,
                    sc.race_id_db_to_nk(args.test))
        t0 = time.time()
        odds = sc.scrape_one_race(args.test)
        elapsed = time.time() - t0
        if odds:
            print(f"所要: {elapsed:.1f}秒")
            for t, lst in odds.items():
                print(f"  {t}: {len(lst)} items")
                for c, o, p in lst[:3]:
                    print(f"    {c}: odds={o}, pop={p}")
            saved = sc.save_odds(args.test, odds)
            print(f"DB保存: {saved}")
        else:
            print("取得失敗")
        return

    if args.backfill:
        sc.backfill(limit=args.limit)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
