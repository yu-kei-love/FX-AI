# ===========================================
# scraper/scraper_comment.py
# 競輪 - Gambooスクレイパー（コメント・並び予想）
#
# 対象：gamboo.jp（公開ページ・ログイン不要）
# 取得項目：
#   ①選手コメント（marutoku.aspx の「選手コメント」列）
#   ②並び予想（popup/osusume の「☆並び」）
#
# URL構造（v0.25以降）:
#   コメント: /keirin/yoso/marutoku.aspx?rdt={YYYY-MM-DD}&pid={jka}&rno={r}
#     → 1レース1リクエスト（選手別コメントテキスト）
#   並び予想: /keirin/yoso/popup/osusume?rdt={YYYY-MM-DD}&pid={jka}&rno={r}
#     → 実は会場単位で1ページに全レースの並びが集約されている
#       任意の rno で同じページが返るため、1会場1リクエストで済む
#
# 設計メモ:
#   - aokei.aspx（アオケイ予想新聞）はログイン必須のため対象外
#   - JKAコードはゼロパディングなし（bank_master.jka_code そのまま）
#   - chariloto DBのracesテーブルを参照し、既存レースのみ補完
#
# robots.txt（2026-04-07）:
#   Disallow: /keirin/yoso/analyzer/
#   → /keirin/yoso/marutoku.aspx と /keirin/yoso/popup/osusume はアクセス可
#
# 注意：このファイルはスクレイピングコードのため
#       note販売パッケージには含めない
# ===========================================

import argparse
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# bank_master.py を参照（JKAコード取得）
_BANK_PATH = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(_BANK_PATH))
from bank_master import BANK_MASTER  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"
FAILED_LOG = Path(__file__).resolve().parent / "failed_gamboo.log"

BASE_URL = "https://gamboo.jp"

# robots.txt Disallow パス
DISALLOWED_PATHS = [
    "/keirin/yoso/analyzer/",
    "/autorace/analyzer/marklist/",
]


# =========================================================
# コメント解析ユーティリティ（line_predictor から参照される）
# =========================================================

def parse_comment_for_lines(comment_text):
    """
    コメントテキストからライン意図を解析する。

    戻り値: {"role": "lead"|"follow"|"single"|"unknown",
            "target": str|None, "confidence": float}
    """
    patterns = {
        "lead": [
            r"(先行|自力|前から|逃げ|前で)",
            r"(先手で|先頭で)",
        ],
        "follow": [
            r"(\S+)選手(を|の後ろ|の番手)(を?追|につ)",
            r"(\S+)番(の後ろ|番手|を追)",
            r"(番手で|マークで|後ろにつ)",
        ],
        "single": [
            r"(単騎|一人で|単独で)",
        ],
    }

    for role, role_patterns in patterns.items():
        for pattern in role_patterns:
            m = re.search(pattern, comment_text or "")
            if m:
                target = None
                if role == "follow" and m.lastindex and m.lastindex >= 1:
                    raw = m.group(1)
                    target = raw if raw.strip() else None
                return {
                    "role": role,
                    "target": target,
                    "confidence": 0.80 if role in ("lead", "single") else 0.70,
                }

    return {"role": "unknown", "target": None, "confidence": 0.30}


# =========================================================
# GambooScraper クラス
# =========================================================

class GambooScraper:
    """gamboo.jp から選手コメント・並び予想を取得する"""

    def __init__(self, db_path=None, delay=3.0):
        """
        Parameters:
            db_path: SQLiteファイルのパス
            delay: リクエスト間の待機秒数（Gambooは長めに設定）
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (KeirinDataCollector/1.0; "
                          "personal research use)",
            "Accept-Language": "ja,en;q=0.9",
        })

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DB が見つかりません: {self.db_path}\n"
                "先に chariloto スクレイパーで races テーブルを作成してください"
            )

        # jyo_cd (bank_master venue_id int) → jka_code マップ
        self.venue_id_to_jka = {}
        for info in BANK_MASTER.values():
            vid = info.get("venue_id")
            jka = info.get("jka_code")
            if vid and jka:
                self.venue_id_to_jka[int(vid)] = jka

        self._init_db()

    # =========================================================
    # DB
    # =========================================================

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self):
        """comments / reporter_predictions テーブルを作成する"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS comments (
                race_id       TEXT NOT NULL,
                sha_ban       INTEGER NOT NULL,
                senshu_name   TEXT,
                comment_text  TEXT,
                comment_date  TEXT,
                created_at    TEXT NOT NULL,
                PRIMARY KEY (race_id, sha_ban),
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );

            CREATE TABLE IF NOT EXISTS reporter_predictions (
                race_id        TEXT NOT NULL,
                reporter_name  TEXT,
                predicted_line TEXT,
                confidence     TEXT,
                created_at     TEXT NOT NULL,
                PRIMARY KEY (race_id, reporter_name),
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );

            CREATE INDEX IF NOT EXISTS idx_comments_race
                ON comments(race_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_race
                ON reporter_predictions(race_id);
        """)
        conn.commit()
        conn.close()

    # =========================================================
    # HTTP ヘルパ
    # =========================================================

    def _polite_sleep(self):
        time.sleep(self.delay)

    def _is_disallowed(self, path):
        for d in DISALLOWED_PATHS:
            if path.startswith(d):
                return True
        return False

    def _fetch_html(self, url, max_retries=3):
        """URL取得。リトライ付き。"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if self._is_disallowed(parsed.path):
            logger.warning("robots.txt 違反パス: %s", parsed.path)
            return None

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 404:
                    logger.debug("404: %s", url)
                    return None
                if resp.status_code == 503:
                    logger.warning("503 %s (%d/%d)",
                                   url, attempt, max_retries)
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                resp.encoding = "utf-8"
                return resp.text
            except requests.RequestException as e:
                logger.warning("req error %s (%d/%d): %s",
                               url, attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return None

    def _log_failed(self, identifier, reason):
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}\t{identifier}\t{reason}\n")

    # =========================================================
    # DBから対象レース取得
    # =========================================================

    def _get_races_for_date(self, date_compact):
        """
        指定日の(race_id, jyo_cd, race_no) 一覧を取得する。

        Parameters:
            date_compact: "YYYYMMDD" 形式
        """
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT race_id, jyo_cd, race_no FROM races "
            "WHERE race_date = ? ORDER BY jyo_cd, race_no",
            (date_compact,)
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    def _get_existing_comment_race_ids(self):
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_id FROM comments")
        ids = {r[0] for r in cur.fetchall()}
        conn.close()
        return ids

    def _get_existing_prediction_race_ids(self):
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_id FROM reporter_predictions")
        ids = {r[0] for r in cur.fetchall()}
        conn.close()
        return ids

    # =========================================================
    # 並び予想（osusume: 1会場1ページで全レース分取得）
    # =========================================================

    def _fetch_narabi_for_venue(self, date_str, jka_code, sample_rno=1):
        """
        osusume ページから1会場の全レース並びを取得する。

        URL: /keirin/yoso/popup/osusume?rdt=YYYY-MM-DD&pid={jka}&rno={r}
        サンプル rno で1会場の全レース分の並びが返ってくる。

        Returns:
            dict: {race_no(int): predicted_line(str)}
            例: {1: "1-4-2-6,3,5", 2: "1-2-5,3-4-6", ...}
        """
        url = (f"{BASE_URL}/keirin/yoso/popup/osusume"
               f"?rdt={date_str}&pid={jka_code}&rno={sample_rno}")
        html = self._fetch_html(url)
        if html is None:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        result = {}

        # テーブル内の行を走査して「1R」「2R」パターンと並び文字列を抽出
        for tr in soup.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            race_label = cells[0].get_text(strip=True)
            narabi_raw = cells[1].get_text(strip=True)

            m = re.match(r"(\d+)R", race_label)
            if not m:
                continue
            race_no = int(m.group(1))

            # 「☆1426・3・5」形式 → "1-4-2-6,3,5"
            narabi = self._parse_narabi_text(narabi_raw)
            if narabi:
                result[race_no] = narabi

        return result

    def _parse_narabi_text(self, text):
        """
        Gamboo 並び表記を正規化する。

        入力例:
          "☆1426・3・5"    → "1-4-2-6,3,5"
          "☆125・346"      → "1-2-5,3-4-6"
          "☆34・51・627"   → "3-4,5-1,6-2-7"

        形式ルール:
          - 先頭の「☆」「★」等の記号は除去
          - 「・」がライン区切り
          - 区切られた各グループは数字の連続で、各桁が車番
          - 同グループ内の車番をハイフン区切り、グループ間をカンマ区切り
        """
        if not text:
            return None

        # 先頭の記号を除去
        t = text.strip()
        t = re.sub(r"^[☆★◎○△▲◇◆※\s]+", "", t)

        if not t:
            return None

        # 全角「・」と半角「.」「,」を統一
        t = t.replace("・", "|").replace("、", "|")

        # 数字以外と | のみ残す（空白等除去）
        t = re.sub(r"[^\d|]", "", t)
        if not t:
            return None

        groups = [g for g in t.split("|") if g]
        line_strs = []
        for g in groups:
            # 各桁が車番（1〜9）
            cars = [c for c in g if c in "123456789"]
            if not cars:
                continue
            line_strs.append("-".join(cars))

        if not line_strs:
            return None

        return ",".join(line_strs)

    # =========================================================
    # コメント（marutoku: 1レース1ページ）
    # =========================================================

    def _fetch_comments_for_race(self, date_str, jka_code, race_no):
        """
        marutoku.aspx から1レースの選手コメントを取得する。

        Returns:
            list[dict]: [{"sha_ban", "senshu_name", "comment_text"}, ...]
        """
        url = (f"{BASE_URL}/keirin/yoso/marutoku.aspx"
               f"?rdt={date_str}&pid={jka_code}&rno={race_no}")
        html = self._fetch_html(url)
        if html is None:
            return []

        return self._parse_marutoku(html)

    def _parse_marutoku(self, html):
        """
        marutoku.aspx の HTML から選手コメントを抽出する。

        「選手コメント」を th に持つ table を走査し、
        各行の最後のセルがコメントテキストに相当する。
        """
        soup = BeautifulSoup(html, "html.parser")

        # 「選手コメント」th を含む table を探す
        target_table = None
        for th in soup.find_all("th"):
            if "選手コメント" in th.get_text(strip=True):
                target_table = th.find_parent("table")
                break

        if target_table is None:
            return []

        results = []
        # 各 tr を走査
        # ヘッダ行はセル数が少ない（5個など）ので、
        # 9セル行（実データ）だけを対象にする
        for tr in target_table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 9:
                continue

            # 選手名セル（最初）例: "山元大夢|石川(-)/123/24"
            name_cell = cells[0].get_text(strip=True, separator="|")
            parts = name_cell.split("|")
            senshu_name = parts[0].strip() if parts else ""

            # 車番: 選手名セルの前に別列として存在する場合もあるが
            # 今回の構造では table[0] にある枠番/車番列から取る必要あり。
            # しかし実データでは、td の最初の要素が選手名で、
            # 行番号順に車番=1,2,3... となる仕様のため、
            # 行のインデックスで代用する。
            # ただし確実性のため、選手名横の「枠番/車番」テーブルから
            # 別途取得する。→ ここでは順番で代用。
            # （見出し2行をスキップした後の順番）
            # より正確に取るには別テーブルをパースする必要あり

            # コメントセル（最後）
            comment_text = cells[-1].get_text(strip=True)

            results.append({
                "senshu_name": senshu_name,
                "comment_text": comment_text if comment_text else None,
            })

        # sha_ban は行順に1から振る
        for i, r in enumerate(results, 1):
            r["sha_ban"] = i

        return results

    # =========================================================
    # DB保存
    # =========================================================

    def _save_comments(self, comments):
        """
        {race_id, sha_ban, senshu_name, comment_text, comment_date} のリスト。
        comment_text が None/空 は保存しない。
        """
        if not comments:
            return 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect_db()
        cur = conn.cursor()
        saved = 0
        for c in comments:
            if not c.get("comment_text"):
                continue
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO comments
                    (race_id, sha_ban, senshu_name, comment_text,
                     comment_date, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    c["race_id"],
                    c["sha_ban"],
                    c.get("senshu_name"),
                    c["comment_text"],
                    c.get("comment_date"),
                    now,
                ))
                saved += cur.rowcount
            except sqlite3.Error as e:
                logger.warning("comments INSERT %s: %s",
                               c.get("race_id"), e)
        conn.commit()
        conn.close()
        return saved

    def _save_predictions(self, predictions):
        """
        {race_id, reporter_name, predicted_line} のリスト。
        """
        if not predictions:
            return 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect_db()
        cur = conn.cursor()
        saved = 0
        for p in predictions:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO reporter_predictions
                    (race_id, reporter_name, predicted_line,
                     confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    p["race_id"],
                    p.get("reporter_name", "gamboo"),
                    p.get("predicted_line"),
                    p.get("confidence"),
                    now,
                ))
                saved += cur.rowcount
            except sqlite3.Error as e:
                logger.warning("predictions INSERT %s: %s",
                               p.get("race_id"), e)
        conn.commit()
        conn.close()
        return saved

    # =========================================================
    # 1日分の処理
    # =========================================================

    def scrape_comments_for_date(self, date_str):
        """
        指定日の全会場のコメント＋並びを取得して保存する。

        Parameters:
            date_str: "YYYY-MM-DD" 形式
        """
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("日付パース失敗: %s", date_str)
            return
        date_compact = dt.strftime("%Y%m%d")

        races = self._get_races_for_date(date_compact)
        if not races:
            logger.info("[%s] DBにレースなし", date_str)
            return

        # 会場別にグルーピング
        by_venue = {}
        for race_id, jyo_cd, race_no in races:
            jka = self.venue_id_to_jka.get(int(jyo_cd))
            if jka is None:
                continue
            by_venue.setdefault(jka, []).append((race_id, race_no))

        existing_c = self._get_existing_comment_race_ids()
        existing_p = self._get_existing_prediction_race_ids()

        total_c = 0
        total_p = 0

        for jka, race_list in by_venue.items():
            # 1. 並び予想を1リクエストで取得
            narabi_map = self._fetch_narabi_for_venue(date_str, jka)
            self._polite_sleep()

            preds = []
            for race_id, race_no in race_list:
                if race_id in existing_p:
                    continue
                narabi = narabi_map.get(int(race_no))
                if narabi:
                    preds.append({
                        "race_id": race_id,
                        "reporter_name": "gamboo",
                        "predicted_line": narabi,
                        "confidence": None,
                    })
            total_p += self._save_predictions(preds)

            # 2. 各レースのコメントを取得
            for race_id, race_no in race_list:
                if race_id in existing_c:
                    continue
                comments_raw = self._fetch_comments_for_race(
                    date_str, jka, race_no
                )
                self._polite_sleep()
                if not comments_raw:
                    self._log_failed(
                        f"comments_{race_id}",
                        "marutoku取得失敗"
                    )
                    continue

                comments = []
                for c in comments_raw:
                    if c.get("comment_text"):
                        comments.append({
                            "race_id": race_id,
                            "sha_ban": c["sha_ban"],
                            "senshu_name": c.get("senshu_name"),
                            "comment_text": c["comment_text"],
                            "comment_date": date_compact,
                        })
                total_c += self._save_comments(comments)

        logger.info(
            "[%s] 完了: コメント %d件, 並び予想 %d件",
            date_str, total_c, total_p,
        )

    def scrape_predictions_for_date(self, date_str):
        """
        並び予想のみを取得する（コメントなし）。
        """
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return
        date_compact = dt.strftime("%Y%m%d")

        races = self._get_races_for_date(date_compact)
        if not races:
            logger.info("[%s] DBにレースなし", date_str)
            return

        by_venue = {}
        for race_id, jyo_cd, race_no in races:
            jka = self.venue_id_to_jka.get(int(jyo_cd))
            if jka is None:
                continue
            by_venue.setdefault(jka, []).append((race_id, race_no))

        existing = self._get_existing_prediction_race_ids()
        total = 0

        for jka, race_list in by_venue.items():
            narabi_map = self._fetch_narabi_for_venue(date_str, jka)
            self._polite_sleep()

            preds = []
            for race_id, race_no in race_list:
                if race_id in existing:
                    continue
                narabi = narabi_map.get(int(race_no))
                if narabi:
                    preds.append({
                        "race_id": race_id,
                        "reporter_name": "gamboo",
                        "predicted_line": narabi,
                        "confidence": None,
                    })
            total += self._save_predictions(preds)

        logger.info("[%s] 並び予想: %d件保存", date_str, total)

    def scrape_backfill_range(self, start_date, end_date):
        """
        指定期間のコメント・並び予想をバックフィルする（並列実行用）。

        既に comments/reporter_predictions に入っている race_id は
        scrape_comments_for_date 内部でスキップされる。

        Parameters:
            start_date: "YYYY-MM-DD" 形式
            end_date:   "YYYY-MM-DD" 形式
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error("日付パース失敗: %s", e)
            return

        start_c = start_dt.strftime("%Y%m%d")
        end_c = end_dt.strftime("%Y%m%d")

        # 期間内で races に存在する日付を取得
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT race_date FROM races
            WHERE race_date >= ? AND race_date <= ?
            ORDER BY race_date
        """, (start_c, end_c))
        dates = [row[0] for row in cur.fetchall()]
        conn.close()

        if not dates:
            logger.info("[%s 〜 %s] DBにレースなし", start_date, end_date)
            return

        logger.info("=== バックフィル開始 ===")
        logger.info("期間: %s 〜 %s（%d 日）",
                    start_date, end_date, len(dates))

        for i, dc in enumerate(dates, 1):
            try:
                dt = datetime.strptime(dc, "%Y%m%d")
            except ValueError:
                continue
            ds = dt.strftime("%Y-%m-%d")
            logger.info("[%d/%d] %s", i, len(dates), ds)
            try:
                self.scrape_comments_for_date(ds)
            except Exception as e:
                logger.error("[%s] エラー: %s", ds, e)
                self._log_failed(f"backfill_{ds}", str(e))

        logger.info("=== バックフィル完了 ===")

    def scrape_missing_comments(self):
        """DB内でコメント未取得のrace_dateを自動検出して順次補完する"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r.race_date
            FROM races r
            LEFT JOIN comments c ON r.race_id = c.race_id
            WHERE c.race_id IS NULL
            ORDER BY r.race_date
        """)
        dates = [row[0] for row in cur.fetchall()]
        conn.close()

        if not dates:
            logger.info("コメント未取得なし")
            return

        logger.info("コメント未取得: %d 日", len(dates))
        for i, dc in enumerate(dates, 1):
            try:
                dt = datetime.strptime(dc, "%Y%m%d")
            except ValueError:
                continue
            ds = dt.strftime("%Y-%m-%d")
            logger.info("[%d/%d] %s", i, len(dates), ds)
            try:
                self.scrape_comments_for_date(ds)
            except Exception as e:
                logger.error("[%s] %s", ds, e)
                self._log_failed(f"missing_{ds}", str(e))

    def scrape_missing_predictions(self):
        """並び予想未取得のrace_dateを自動検出して順次補完する"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r.race_date
            FROM races r
            LEFT JOIN reporter_predictions rp ON r.race_id = rp.race_id
            WHERE rp.race_id IS NULL
            ORDER BY r.race_date
        """)
        dates = [row[0] for row in cur.fetchall()]
        conn.close()

        if not dates:
            logger.info("並び予想未取得なし")
            return

        logger.info("並び予想未取得: %d 日", len(dates))
        for i, dc in enumerate(dates, 1):
            try:
                dt = datetime.strptime(dc, "%Y%m%d")
            except ValueError:
                continue
            ds = dt.strftime("%Y-%m-%d")
            logger.info("[%d/%d] %s", i, len(dates), ds)
            try:
                self.scrape_predictions_for_date(ds)
            except Exception as e:
                logger.error("[%s] %s", ds, e)
                self._log_failed(f"missing_pred_{ds}", str(e))

    # =========================================================
    # 進捗表示
    # =========================================================

    def get_status(self):
        conn = self._connect_db()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM races")
        race_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM comments")
        c_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT race_id) FROM comments")
        c_race = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM reporter_predictions")
        p_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT race_id) FROM reporter_predictions")
        p_race = cur.fetchone()[0]

        conn.close()

        print("\n========== Gamboo 補完進捗 ==========")
        print(f"  レース総数:   {race_count:,}")
        print(f"  コメント:     {c_count:,} ({c_race:,} レース)")
        print(f"  並び予想:     {p_count:,} ({p_race:,} レース)")
        print("=====================================\n")

        return {
            "race_count": race_count,
            "comment_count": c_count,
            "prediction_count": p_count,
        }


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gambooスクレイパー（コメント・並び予想）v0.25"
    )
    parser.add_argument("--date", type=str, help="対象日 YYYY-MM-DD")
    parser.add_argument("--comments", action="store_true",
                        help="コメント取得")
    parser.add_argument("--predictions", action="store_true",
                        help="並び予想取得")
    parser.add_argument("--missing", action="store_true",
                        help="未取得分を自動検出して取得（両方）")
    parser.add_argument("--backfill_range", type=str, nargs=2,
                        metavar=("START", "END"),
                        help="期間バックフィル（例: 2022-01-01 2022-12-31）")
    parser.add_argument("--status", action="store_true",
                        help="取得済み件数を表示")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="リクエスト間隔（秒）")
    parser.add_argument("--db", type=str, default=None, help="DBパス")

    args = parser.parse_args()

    try:
        scraper = GambooScraper(db_path=args.db, delay=args.delay)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if args.status:
        scraper.get_status()
        return

    if args.missing:
        scraper.scrape_missing_comments()
        scraper.get_status()
        return

    if args.backfill_range:
        start, end = args.backfill_range
        scraper.scrape_backfill_range(start, end)
        scraper.get_status()
        return

    if args.date:
        if args.comments or (not args.comments and not args.predictions):
            scraper.scrape_comments_for_date(args.date)
        elif args.predictions:
            scraper.scrape_predictions_for_date(args.date)
        scraper.get_status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
