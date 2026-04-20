# ===========================================
# scraper/scraper_netkeirin_playwright.py
# netkeirin (keirin.netkeiba.com) から過去 3連単オッズを
# Playwright で取得する。
#
# netkeirin は SPA で JS レンダリング必須のため、
# 実ブラウザで描画して HTML から抽出する。
#
# 戦略:
#   1. Network イベントで /api/race/?...class=... リクエストを捕捉
#   2. そのレスポンス JSON を保存（将来の直接スクレイプに活用）
#   3. 同時に DOM レンダリング後のテーブルからオッズ抽出
#
# 倫理:
#   - delay 5.0 秒以上 (重いブラウザ自動化)
#   - 1並列まで
#   - 非商用・データ再配布なし
# ===========================================

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCRAPER_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRAPER_DIR.parent
PROJECT_ROOT = _PROJECT_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

BASE_URL = "https://keirin.netkeiba.com"


class NetkeirinPlaywrightScraper:
    """netkeirin 3連単オッズを Playwright で取得"""

    def __init__(self, db_path=None, delay=5.0, headless=True):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = max(delay, 5.0)
        self.headless = headless
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS odds_trifecta_nk (
                race_id    TEXT NOT NULL,
                sha_ban_1  INTEGER NOT NULL,
                sha_ban_2  INTEGER NOT NULL,
                sha_ban_3  INTEGER NOT NULL,
                odds       REAL,
                source     TEXT DEFAULT 'netkeirin',
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (race_id, sha_ban_1, sha_ban_2, sha_ban_3)
            );
            CREATE INDEX IF NOT EXISTS idx_otnk_race
                ON odds_trifecta_nk(race_id);
            CREATE TABLE IF NOT EXISTS netkeirin_api_log (
                race_id    TEXT NOT NULL,
                api_url    TEXT NOT NULL,
                status     INTEGER,
                resp_size  INTEGER,
                fetched_at TEXT,
                PRIMARY KEY (race_id, api_url)
            );
        """)
        conn.commit()
        conn.close()

    def _polite_sleep(self):
        time.sleep(self.delay)

    def scrape_one_race(self, race_id):
        """
        1 レース分の 3連単オッズを取得。
        Returns:
            {
              "api_calls": [{"url": str, "status": int, "body_sample": str}...],
              "odds": list[(sha1, sha2, sha3, odds)],
              "html_size": int,
            }
        """
        url = f"{BASE_URL}/race/odds/?race_id={race_id}"
        api_calls = []
        json_bodies = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"),
            )
            page = context.new_page()

            # リクエスト＆レスポンス捕捉 (URL+POST body+response body)
            request_store = {}

            def on_request(request):
                try:
                    if "/api/race/" in request.url:
                        request_store[request.url] = {
                            "method": request.method,
                            "post_data": request.post_data,
                            "headers": dict(request.headers),
                        }
                except Exception:
                    pass

            def on_response(response):
                try:
                    u = response.url
                    if "/api/race/" in u or "odds" in u.lower():
                        try:
                            body = response.text()
                        except Exception:
                            body = ""
                        req = request_store.get(u, {})
                        api_calls.append({
                            "url": u,
                            "method": req.get("method", "GET"),
                            "post_data": req.get("post_data"),
                            "status": response.status,
                            "body_size": len(body),
                            "body_sample": body[:2000] if body else "",
                        })
                        if body.strip().startswith("{") or body.strip().startswith("["):
                            try:
                                json_bodies.append({
                                    "url": u,
                                    "method": req.get("method", "GET"),
                                    "post_data": req.get("post_data"),
                                    "json": json.loads(body),
                                })
                            except Exception:
                                pass
                except Exception:
                    pass

            page.on("request", on_request)

            page.on("response", on_response)
            try:
                page.goto(url, timeout=30000, wait_until="networkidle")
            except PwTimeout:
                logger.warning("timeout: %s", race_id)

            # 待機（JS 読み込み完了待ち）
            try:
                # 3連単オッズリンクがあれば移動（もしくは直接 odds_sanrentan へ）
                page.goto(f"{BASE_URL}/race/odds_sanrentan/?race_id={race_id}",
                          timeout=30000, wait_until="networkidle")
            except PwTimeout:
                pass

            try:
                page.wait_for_timeout(2000)  # JS完了待ち
            except Exception:
                pass

            html = page.content()
            browser.close()

        # odds 抽出
        odds = self._parse_trifecta_from_html(html)

        return {
            "race_id": race_id,
            "api_calls": api_calls,
            "json_bodies": json_bodies,
            "odds": odds,
            "html_size": len(html),
        }

    def _parse_trifecta_from_html(self, html):
        """
        レンダリング後の HTML から 3連単オッズを抽出。
        netkeirin DOM 構造に依存するため、複数のパターンを試行。
        """
        odds_list = []
        # パターン1: td 内に "1-2-3" 形式のキーとオッズが並ぶ
        #           正則: 数字3つ '-' 区切り + 数値
        for m in re.finditer(
            r"(\d+)[\-\u2013](\d+)[\-\u2013](\d+)\s*</td>\s*<td[^>]*>\s*([\d\.]+)",
            html,
        ):
            try:
                sb1, sb2, sb3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                o = float(m.group(4))
                odds_list.append((sb1, sb2, sb3, o))
            except (ValueError, TypeError):
                continue

        # パターン2: JSON 形式で埋め込まれている可能性
        # oddsData = [...] のような
        for m in re.finditer(r"(\d+)-(\d+)-(\d+)&quot;:&quot;([\d\.]+)", html):
            try:
                sb1, sb2, sb3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                o = float(m.group(4))
                odds_list.append((sb1, sb2, sb3, o))
            except (ValueError, TypeError):
                continue

        # パターン3: data-* 属性内
        for m in re.finditer(r'data-combo="(\d+)-(\d+)-(\d+)"[^>]*data-odds="([\d\.]+)"',
                             html):
            try:
                sb1, sb2, sb3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                o = float(m.group(4))
                odds_list.append((sb1, sb2, sb3, o))
            except (ValueError, TypeError):
                continue

        return odds_list

    def save_odds(self, race_id, odds_list):
        if not odds_list:
            return 0
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n = 0
        for sb1, sb2, sb3, o in odds_list:
            try:
                cur.execute("""
                    INSERT INTO odds_trifecta_nk
                        (race_id, sha_ban_1, sha_ban_2, sha_ban_3,
                         odds, source, fetched_at)
                    VALUES (?, ?, ?, ?, ?, 'netkeirin', ?)
                    ON CONFLICT(race_id, sha_ban_1, sha_ban_2, sha_ban_3)
                    DO NOTHING
                """, (race_id, sb1, sb2, sb3, o, now))
                n += cur.rowcount
            except sqlite3.Error:
                pass
        conn.commit()
        conn.close()
        return n

    def save_api_log(self, race_id, api_calls):
        if not api_calls:
            return
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for a in api_calls:
            try:
                cur.execute("""
                    INSERT INTO netkeirin_api_log
                        (race_id, api_url, status, resp_size, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(race_id, api_url) DO NOTHING
                """, (race_id, a["url"], a["status"], a["body_size"], now))
            except sqlite3.Error:
                pass
        conn.commit()
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="race_id を指定してテスト実行")
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--headed", action="store_true",
                        help="GUI ブラウザで実行 (debug)")
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    sc = NetkeirinPlaywrightScraper(
        db_path=args.db, delay=args.delay, headless=not args.headed,
    )

    if args.test:
        logger.info("テスト取得: %s", args.test)
        t0 = time.time()
        result = sc.scrape_one_race(args.test)
        elapsed = time.time() - t0
        logger.info("所要: %.1f秒, html=%d, api_calls=%d, odds=%d",
                    elapsed, result["html_size"],
                    len(result["api_calls"]), len(result["odds"]))
        print("\n=== API calls (詳細) ===")
        for a in result["api_calls"][:20]:
            print(f"  [{a.get('method','?')}] {a['status']} ({a['body_size']}b)")
            print(f"    url: {a['url'][:200]}")
            if a.get('post_data'):
                print(f"    POST: {a['post_data'][:300]}")
            if a.get('body_sample'):
                print(f"    body: {a['body_sample'][:300]}")
            print()

        print(f"\n=== Odds extracted: {len(result['odds'])} ===")
        for o in result["odds"][:5]:
            print(f"  {o}")

        print(f"\n=== JSON response 詳細 ({len(result['json_bodies'])}) ===")
        for j in result["json_bodies"][:10]:
            keys = list(j["json"].keys()) if isinstance(j["json"], dict) else "list"
            print(f"  [{j.get('method','?')}] {j['url']}")
            if j.get('post_data'):
                print(f"    POST: {j['post_data']}")
            print(f"    keys: {keys}")
            d = j["json"].get("data") if isinstance(j["json"], dict) else None
            if d is not None:
                if isinstance(d, dict):
                    print(f"    data keys: {list(d.keys())[:20]}")
                elif isinstance(d, list):
                    print(f"    data list len: {len(d)}")
            print()
        # save
        sc.save_api_log(args.test, result["api_calls"])
        saved = sc.save_odds(args.test, result["odds"])
        print(f"\n保存: odds={saved}, api_calls logged")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
