# ===========================================
# scraper/scraper_oddspark.py
# 競輪 - オッズパーク過去3連単確定オッズスクレイパー
#
# 対象: oddspark.com の Odds.do ページ
# 取得: 3連単確定オッズ（レース後の確定値）
#
# URL:
#   https://www.oddspark.com/keirin/Odds.do
#   ?joCode={JKA}&kaisaiBi={YYYYMMDD}&raceNo={N}
#   &betType=9&shaban={K}&jikuCode=1
#
# 重要:
#   - 1レース = 7 shaban ページ (1着=1..7) イテレート
#   - デフォルトは shaban=1 のみ取得モード（partial）
#   - --full で全 shaban 取得モード（約10日）
#   - delay 3.0 秒厳守、2並列まで
#
# 倫理:
#   - robots.txt: Disallow なし（Noindex は SEO 向け）
#   - 商用利用しない・データ再配布しない
#
# 注意:
#   - joCode と JKA の対応は多くが一致（確認済み範囲）
#   - 取得失敗レースは failed ログに記録
# ===========================================

import argparse
import io
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
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
    # venue_id (int) -> JKA code (string "NN")
    VENUE_ID_TO_JKA = {
        int(info["venue_id"]): info["jka_code"]
        for info in BANK_MASTER.values()
        if info.get("venue_id") and info.get("jka_code")
    }
except ImportError:
    BANK_MASTER = {}
    VENUE_ID_TO_JKA = {}

BASE_URL = "https://www.oddspark.com"
PROJECT_ROOT = _PROJECT_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"
FAILED_LOG = _SCRAPER_DIR / "failed_oddspark.log"


class OddsparkScraper:
    """オッズパークから過去3連単確定オッズを取得する"""

    def __init__(self, db_path=None, delay=3.0, jyo_cds=None, full=False):
        """
        Parameters:
            db_path: SQLite DBパス
            delay: リクエスト間ディレイ（秒、最低 3.0 推奨）
            jyo_cds: 対象 venue_id リスト（None で全会場）
            full: True なら全 shaban (1..N) 取得、False なら shaban=1 のみ
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = max(delay, 3.0)
        self.jyo_cds = [int(c) for c in jyo_cds] if jyo_cds else None
        self.full = full
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KeirinResearchBot/1.0 (personal non-commercial research)",
            "Accept-Language": "ja,en;q=0.9",
        })
        self._init_db()

    def _init_db(self):
        """odds_trifecta_final テーブルを作成する"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS odds_trifecta_final (
                race_id    TEXT NOT NULL,
                sha_ban_1  INTEGER NOT NULL,
                sha_ban_2  INTEGER NOT NULL,
                sha_ban_3  INTEGER NOT NULL,
                odds       REAL,
                source     TEXT DEFAULT 'oddspark',
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (race_id, sha_ban_1, sha_ban_2, sha_ban_3)
            );
            CREATE INDEX IF NOT EXISTS idx_otf_race
                ON odds_trifecta_final(race_id);
        """)
        conn.commit()
        conn.close()
        logger.info("DB 初期化完了: %s", self.db_path)

    def _polite_sleep(self):
        time.sleep(self.delay)

    def _log_failed(self, race_id, reason):
        try:
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\t{race_id}\t{reason}\n")
        except Exception:
            pass

    def _fetch(self, url, retry=3):
        """HTTP GET。3回リトライ。"""
        for i in range(retry):
            try:
                r = self.session.get(url, timeout=20)
                if r.status_code == 200:
                    return r.text
                if r.status_code == 404:
                    return None
                if r.status_code in (403, 429):
                    logger.warning("rate limit/ban: %d %s", r.status_code, url)
                    time.sleep(30)
                    continue
                time.sleep(2 ** i)
            except requests.RequestException as e:
                logger.warning("req error %s: %s", url, e)
                time.sleep(2 ** i)
        return None

    def get_url(self, jka, kaisai_bi, race_no, shaban=1):
        return (
            f"{BASE_URL}/keirin/Odds.do"
            f"?joCode={jka}&kaisaiBi={kaisai_bi}"
            f"&raceNo={race_no}&betType=9"
            f"&shaban={shaban}&jikuCode=1"
        )

    def _parse_trifecta_table(self, html, first_sha):
        """
        3連単オッズテーブルをパース。

        Table 構造 (pivot):
          cell[r][c] = 3連単 (first=first_sha, 2着=c, 3着=r) の odds

        Returns:
            list of (sha1, sha2, sha3, odds) tuples
        """
        try:
            dfs = pd.read_html(io.StringIO(html))
        except Exception:
            return []
        if len(dfs) < 2:
            return []
        df = dfs[1]  # Table[1] が 3連単オッズ本体
        if df.shape[0] < 3 or df.shape[1] < 3:
            return []

        results = []
        # 2着列ラベル (row 1)
        try:
            second_cars = {}
            for c in range(2, df.shape[1]):
                val = df.iloc[1, c]
                if pd.isna(val):
                    continue
                try:
                    second_cars[c] = int(float(val))
                except (ValueError, TypeError):
                    continue

            # 3着行ラベル (col 1) + odds values (col 2..)
            for r in range(2, df.shape[0]):
                label = df.iloc[r, 1]
                if pd.isna(label):
                    continue
                try:
                    third_car = int(float(label))
                except (ValueError, TypeError):
                    continue
                for c, second_car in second_cars.items():
                    val = df.iloc[r, c]
                    if pd.isna(val):
                        continue
                    try:
                        odds = float(val)
                    except (ValueError, TypeError):
                        continue
                    if second_car == first_sha or third_car == first_sha:
                        continue
                    if second_car == third_car:
                        continue
                    results.append((first_sha, second_car, third_car, odds))
        except Exception as e:
            logger.debug("parse error: %s", e)
        return results

    def scrape_one_race(self, race_id):
        """
        1レース分の 3連単オッズ取得。

        shaban=1 のみ (self.full=False) or 全 shaban (self.full=True)

        Returns:
            list of (sha1, sha2, sha3, odds)
        """
        # race_id = "venue_id_YYYYMMDD_RR"
        try:
            parts = race_id.split("_")
            venue_id = int(parts[0])
            date_str = parts[1]
            race_no = int(parts[2])
        except Exception as e:
            logger.warning("race_id parse err %s: %s", race_id, e)
            return []

        jka = VENUE_ID_TO_JKA.get(venue_id)
        if jka is None:
            return []

        all_odds = []
        shabans = list(range(1, 10)) if self.full else [1]
        for sb in shabans:
            url = self.get_url(jka, date_str, race_no, shaban=sb)
            html = self._fetch(url)
            if html is None:
                self._polite_sleep()
                continue
            parsed = self._parse_trifecta_table(html, first_sha=sb)
            all_odds.extend(parsed)
            self._polite_sleep()
            # shaban=1 で空データなら早期終了（そもそも開催なし or 取得不能）
            if sb == 1 and not parsed:
                break

        return all_odds

    def save_odds(self, race_id, odds_list):
        """DBに保存。ON CONFLICT DO NOTHING"""
        if not odds_list:
            return 0
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n = 0
        for sb1, sb2, sb3, odds in odds_list:
            try:
                cur.execute("""
                    INSERT INTO odds_trifecta_final
                        (race_id, sha_ban_1, sha_ban_2, sha_ban_3,
                         odds, source, fetched_at)
                    VALUES (?, ?, ?, ?, ?, 'oddspark', ?)
                    ON CONFLICT(race_id, sha_ban_1, sha_ban_2, sha_ban_3)
                    DO NOTHING
                """, (race_id, sb1, sb2, sb3, odds, now))
                n += cur.rowcount
            except sqlite3.Error as e:
                logger.debug("insert err %s: %s", race_id, e)
        conn.commit()
        conn.close()
        return n

    def _get_target_races(self):
        """
        対象レース一覧。
        既に odds_trifecta_final に記録済みのレースはスキップ。
        jyo_cds 指定時はフィルタ。
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        # 既に登録済みレース
        cur.execute("SELECT DISTINCT race_id FROM odds_trifecta_final")
        done = {row[0] for row in cur.fetchall()}
        # 全 races (2022〜2024のみ対象: 直近データ)
        params = []
        sql = """
            SELECT race_id, jyo_cd, race_date, race_no
            FROM races
            WHERE race_date BETWEEN '20220101' AND '20241231'
        """
        if self.jyo_cds:
            placeholders = ",".join(["?"] * len(self.jyo_cds))
            sql += f" AND jyo_cd IN ({placeholders})"
            params.extend(self.jyo_cds)
        sql += " ORDER BY race_date, jyo_cd, race_no"
        cur.execute(sql, params)
        all_races = cur.fetchall()
        conn.close()

        targets = [r for r in all_races if r[0] not in done]
        logger.info("対象レース: %d (既存: %d, 全体: %d)",
                    len(targets), len(done), len(all_races))
        return targets

    def backfill(self, limit=None):
        """既存 races のうち未取得レースを順次処理"""
        targets = self._get_target_races()
        if limit:
            targets = targets[:limit]
        total = len(targets)
        if total == 0:
            logger.info("処理対象なし")
            return

        success = 0
        failed = 0
        for idx, (race_id, jyo_cd, race_date, race_no) in enumerate(targets, 1):
            try:
                odds = self.scrape_one_race(race_id)
                if not odds:
                    failed += 1
                    self._log_failed(race_id, "no_odds")
                else:
                    saved = self.save_odds(race_id, odds)
                    success += 1
                    if idx % 10 == 0 or idx < 5:
                        logger.info("[%d/%d] %s: %d 保存 (cum s=%d f=%d)",
                                    idx, total, race_id, saved, success, failed)
                # 連続失敗検出
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

        logger.info("完了: success=%d failed=%d total=%d",
                    success, failed, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true",
                        help="全 races 対象でバックフィル")
    parser.add_argument("--jyo_cds", type=str, default=None,
                        help="カンマ区切り venue_id リスト")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--full", action="store_true",
                        help="全 shaban (全1着) 取得。時間10倍")
    parser.add_argument("--limit", type=int, default=None,
                        help="最初のN件のみ処理（テスト用）")
    parser.add_argument("--test", type=str, default=None,
                        help="race_id 指定でテスト実行")
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    jyo_cds = None
    if args.jyo_cds:
        jyo_cds = [int(x.strip()) for x in args.jyo_cds.split(",")
                   if x.strip()]

    scraper = OddsparkScraper(
        db_path=args.db, delay=args.delay,
        jyo_cds=jyo_cds, full=args.full,
    )

    if args.test:
        odds = scraper.scrape_one_race(args.test)
        print(f"取得オッズ数: {len(odds)}")
        for o in odds[:5]:
            print(f"  {o}")
        if odds:
            n = scraper.save_odds(args.test, odds)
            print(f"DB保存: {n}")
        return

    if args.backfill:
        scraper.backfill(limit=args.limit)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
