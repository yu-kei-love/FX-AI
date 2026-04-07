# ===========================================
# scraper/scraper_comment.py
# 競輪 - Gambooスクレイパー（コメント・記者予想）
#
# 対象：gamboo.jp（公開ページ・ログイン不要）
# 取得項目：
#   ①選手コメント（前日コメント）
#   ②記者予想ライン（並び予想）
#
# chariloto DBのracesテーブルを参照し、
# 取得済みレースのコメント・予想のみ取得する
# （Gamboo単独クロールはしない）
#
# robots.txt確認（2026-04-07）:
#   Disallow: /keirin/yoso/analyzer/
#   Disallow: /keirin/yoso/analyzer/*/ranking
#   → /keirin/comment/ と /keirin/yoso/{jyo_cd}/{date}/ はアクセス可
#
# 注意：このファイルはスクレイピングコードのため
#       note販売パッケージには含めない
# ===========================================

import argparse
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ログ設定
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

# robots.txt で Disallow されているパス（アクセス禁止）
DISALLOWED_PATHS = [
    "/keirin/yoso/analyzer/",
    "/autorace/analyzer/marklist/",
]


# =========================================================
# コメント解析ユーティリティ（スタブ版から引き継ぎ）
# =========================================================

def parse_comment_for_lines(comment_text):
    """
    コメントテキストからライン情報を抽出する。

    解析ルール：
    「○○選手を追います」→ {"role": "follow", "target": "○○"}
    「先行します」「自力で行きます」→ {"role": "lead", "target": None}
    「番手で勝負します」→ {"role": "second", "target": None}
    「単騎で行きます」→ {"role": "single", "target": None}

    Parameters:
        comment_text: 選手コメントの文字列

    Returns:
        {"role": str, "target": str|None, "confidence": float}
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
            m = re.search(pattern, comment_text)
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
    """gamboo.jp から競輪の選手コメント・記者予想を取得する"""

    def __init__(self, db_path=None, delay=3.0):
        """
        Parameters:
            db_path: SQLiteファイルのパス（デフォルト: data/keirin/keirin.db）
            delay: リクエスト間の待機秒数（Gambooは長めに設定）
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KeirinDataCollector/1.0 (personal research use)",
            "Accept-Language": "ja,en;q=0.9",
        })
        self._init_db()

    def _init_db(self):
        """comments・reporter_predictions テーブルを初期化する"""
        if not self.db_path.exists():
            logger.error("DB が見つかりません: %s", self.db_path)
            logger.error("先に chariloto スクレイパーで races テーブルを作成してください")
            raise FileNotFoundError(
                f"DB が見つかりません: {self.db_path}\n"
                "先に chariloto スクレイパーを実行してください"
            )

        conn = sqlite3.connect(str(self.db_path))
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
        logger.info("Gamboo テーブル初期化完了: %s", self.db_path)

    # =========================================================
    # HTTP / ユーティリティ
    # =========================================================

    def _polite_sleep(self):
        """サーバー負荷を考慮した待機"""
        time.sleep(self.delay)

    def _is_disallowed(self, path):
        """robots.txt の Disallow に該当するか判定する"""
        for d in DISALLOWED_PATHS:
            if path.startswith(d):
                return True
        return False

    def _fetch_html(self, url, max_retries=3):
        """
        URLからHTMLを取得する。リトライ付き。

        Returns:
            str: HTMLテキスト。取得失敗時は None。
        """
        # robots.txt チェック
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if self._is_disallowed(parsed.path):
            logger.warning("robots.txt で Disallow されているパス: %s", parsed.path)
            return None

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 404:
                    logger.debug("404 Not Found: %s", url)
                    return None
                if resp.status_code == 503:
                    logger.warning("503 Service Unavailable: %s (attempt %d/%d)",
                                   url, attempt, max_retries)
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return resp.text
            except requests.RequestException as e:
                logger.warning("リクエストエラー: %s (attempt %d/%d): %s",
                               url, attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return None

    def _log_failed(self, identifier, reason):
        """失敗した取得をログに記録する"""
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}\t{identifier}\t{reason}\n")

    # =========================================================
    # chariloto DB からの対象レース取得
    # =========================================================

    def _get_race_ids_for_date(self, date_str):
        """
        指定日のrace_id一覧をchariloto DBから取得する。

        Parameters:
            date_str: "YYYYMMDD" 形式

        Returns:
            list[dict]: [{"race_id": str, "jyo_cd": int, "race_no": int}]
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT race_id, jyo_cd, race_no FROM races WHERE race_date = ?",
            (date_str,)
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {"race_id": r[0], "jyo_cd": r[1], "race_no": r[2]}
            for r in rows
        ]

    def _get_jyo_codes_for_date(self, date_str):
        """
        指定日に開催がある会場コードの一覧を取得する。

        Returns:
            list[int]: 会場コードのリスト（重複なし）
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT jyo_cd FROM races WHERE race_date = ?",
            (date_str,)
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _get_existing_comment_race_ids(self):
        """既にコメント取得済みのrace_idセットを返す"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_id FROM comments")
        ids = {r[0] for r in cur.fetchall()}
        conn.close()
        return ids

    def _get_existing_prediction_race_ids(self):
        """既に記者予想取得済みのrace_idセットを返す"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_id FROM reporter_predictions")
        ids = {r[0] for r in cur.fetchall()}
        conn.close()
        return ids

    def _get_all_race_dates(self):
        """DB内の全レース日付を取得する（古い順）"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_date FROM races ORDER BY race_date")
        dates = [r[0] for r in cur.fetchall()]
        conn.close()
        return dates

    # =========================================================
    # 選手コメントのスクレイピング
    # =========================================================

    def scrape_comments_for_date(self, date_str):
        """
        指定日の全会場のコメントを取得・保存する。

        Parameters:
            date_str: "YYYYMMDD" 形式
        """
        jyo_codes = self._get_jyo_codes_for_date(date_str)
        if not jyo_codes:
            logger.info("[%s] DB にレースなし（コメント取得スキップ）", date_str)
            return

        race_list = self._get_race_ids_for_date(date_str)
        existing = self._get_existing_comment_race_ids()

        # race_id → race_no のマッピング
        race_map = {}
        for r in race_list:
            race_map[r["race_id"]] = r

        total_saved = 0

        for jyo_cd in jyo_codes:
            jyo_cd_str = f"{jyo_cd:02d}"
            url = f"{BASE_URL}/keirin/comment/{jyo_cd_str}/{date_str}/"
            html = self._fetch_html(url)
            self._polite_sleep()

            if html is None:
                logger.debug("[%s] 会場 %s: コメントページなし", date_str, jyo_cd_str)
                continue

            parsed = self._parse_comment_page(html, jyo_cd_str, date_str)

            if not parsed:
                logger.debug("[%s] 会場 %s: コメント抽出0件", date_str, jyo_cd_str)
                continue

            # 取得済みスキップ & コメントなしスキップ
            comments_to_save = []
            for c in parsed:
                if c["race_id"] in existing:
                    continue
                if not c.get("comment_text"):
                    continue
                comments_to_save.append(c)

            if comments_to_save:
                self._save_comments(comments_to_save)
                total_saved += len(comments_to_save)
                logger.info("[%s] 会場 %s: %d 件コメント保存",
                            date_str, jyo_cd_str, len(comments_to_save))

        logger.info("[%s] コメント合計: %d 件保存", date_str, total_saved)

    def _parse_comment_page(self, html, jyo_cd_str, date_str):
        """
        コメントページのHTMLをパースする。

        Gambooのコメントページはレース別にセクション分割されている想定。
        テーブル or リスト形式でコメントが記載されている。

        Parameters:
            html: HTMLテキスト
            jyo_cd_str: 会場コード文字列（"01"〜"43"）
            date_str: "YYYYMMDD" 形式

        Returns:
            list[dict]: コメントデータのリスト
        """
        soup = BeautifulSoup(html, "html.parser")
        comments = []

        # 方法1: pd.read_html でテーブルから取得を試行
        try:
            dfs = pd.read_html(html)
            for i, df in enumerate(dfs):
                race_no = i + 1
                race_id = f"{jyo_cd_str}_{date_str}_{race_no:02d}"
                extracted = self._extract_comments_from_df(df, race_id, date_str)
                comments.extend(extracted)

            if comments:
                return comments
        except ValueError:
            pass

        # 方法2: BeautifulSoup でセクション/divから取得を試行
        # レース番号ごとのセクションを探す
        race_sections = soup.find_all(
            ["div", "section", "table"],
            class_=re.compile(r"race|comment|result", re.IGNORECASE)
        )

        current_race_no = 0
        for section in race_sections:
            # レース番号を検出
            race_no_match = re.search(r"(\d+)\s*[Rレース]", section.get_text())
            if race_no_match:
                current_race_no = int(race_no_match.group(1))

            if current_race_no < 1:
                continue

            race_id = f"{jyo_cd_str}_{date_str}_{current_race_no:02d}"

            # コメント行を探す
            rows = section.find_all(["tr", "li", "div"],
                                    class_=re.compile(r"player|racer|entry",
                                                      re.IGNORECASE))
            for row in rows:
                text = row.get_text(separator=" ", strip=True)
                parsed = self._extract_comment_from_text(text, race_id, date_str)
                if parsed:
                    comments.append(parsed)

        # 方法3: ページ全体のテキストからパターンマッチ
        if not comments:
            full_text = soup.get_text(separator="\n", strip=True)
            comments = self._extract_comments_from_fulltext(
                full_text, jyo_cd_str, date_str
            )

        return comments

    def _extract_comments_from_df(self, df, race_id, date_str):
        """DataFrameからコメントを抽出する"""
        if df.empty:
            return []

        df.columns = [str(c).strip().replace("\n", "").replace(" ", "")
                      for c in df.columns]

        # 車番カラムの候補
        sha_ban_col = self._find_column(df, ["車番", "車", "枠"])
        name_col = self._find_column(df, ["選手名", "選手", "氏名"])
        comment_col = self._find_column(df, ["コメント", "前日コメント", "選手コメント",
                                              "comment"])

        if comment_col is None:
            return []

        results = []
        for _, row in df.iterrows():
            sha_ban = self._to_int(row.get(sha_ban_col)) if sha_ban_col else None
            if sha_ban is None or sha_ban < 1 or sha_ban > 9:
                continue

            comment_text = str(row.get(comment_col, "")).strip()
            if not comment_text or comment_text == "nan":
                continue

            senshu_name = str(row.get(name_col, "")).strip() if name_col else ""

            results.append({
                "race_id": race_id,
                "sha_ban": sha_ban,
                "senshu_name": senshu_name,
                "comment_text": comment_text,
                "comment_date": date_str,
            })

        return results

    def _extract_comment_from_text(self, text, race_id, date_str):
        """テキスト行からコメントを抽出する"""
        # パターン: "1番 選手名 コメントテキスト"
        m = re.match(
            r"(\d)\s*番?\s+(\S+)\s+(.+)",
            text.strip()
        )
        if m:
            sha_ban = int(m.group(1))
            senshu_name = m.group(2).strip()
            comment_text = m.group(3).strip()
            if comment_text and len(comment_text) >= 5:
                return {
                    "race_id": race_id,
                    "sha_ban": sha_ban,
                    "senshu_name": senshu_name,
                    "comment_text": comment_text,
                    "comment_date": date_str,
                }
        return None

    def _extract_comments_from_fulltext(self, full_text, jyo_cd_str, date_str):
        """ページ全体のテキストからコメントをパターンマッチで抽出する"""
        comments = []
        current_race_no = 0

        for line in full_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # レース番号の検出
            race_match = re.match(r"(\d{1,2})\s*[Rレース]", line)
            if race_match:
                current_race_no = int(race_match.group(1))
                continue

            if current_race_no < 1:
                continue

            race_id = f"{jyo_cd_str}_{date_str}_{current_race_no:02d}"

            # 車番＋コメントの検出
            m = re.match(r"(\d)\s*番?\s+(\S+)\s+(.{5,})", line)
            if m:
                sha_ban = int(m.group(1))
                senshu_name = m.group(2).strip()
                comment_text = m.group(3).strip()
                comments.append({
                    "race_id": race_id,
                    "sha_ban": sha_ban,
                    "senshu_name": senshu_name,
                    "comment_text": comment_text,
                    "comment_date": date_str,
                })

        return comments

    # =========================================================
    # 記者予想のスクレイピング
    # =========================================================

    def scrape_predictions_for_date(self, date_str):
        """
        指定日の全会場の記者予想を取得・保存する。

        Parameters:
            date_str: "YYYYMMDD" 形式
        """
        jyo_codes = self._get_jyo_codes_for_date(date_str)
        if not jyo_codes:
            logger.info("[%s] DB にレースなし（記者予想取得スキップ）", date_str)
            return

        existing = self._get_existing_prediction_race_ids()
        total_saved = 0

        for jyo_cd in jyo_codes:
            jyo_cd_str = f"{jyo_cd:02d}"
            url = f"{BASE_URL}/keirin/yoso/{jyo_cd_str}/{date_str}/"

            html = self._fetch_html(url)
            self._polite_sleep()

            if html is None:
                logger.debug("[%s] 会場 %s: 記者予想ページなし", date_str, jyo_cd_str)
                self._log_failed(
                    f"pred_{jyo_cd_str}_{date_str}",
                    "記者予想ページ取得失敗"
                )
                continue

            parsed = self._parse_prediction_page(html, jyo_cd_str, date_str)

            if not parsed:
                logger.debug("[%s] 会場 %s: 記者予想抽出0件", date_str, jyo_cd_str)
                self._log_failed(
                    f"pred_{jyo_cd_str}_{date_str}",
                    "記者予想パース失敗（0件）"
                )
                continue

            # 取得済みスキップ
            predictions_to_save = [
                p for p in parsed if p["race_id"] not in existing
            ]

            if predictions_to_save:
                self._save_predictions(predictions_to_save)
                total_saved += len(predictions_to_save)
                logger.info("[%s] 会場 %s: %d 件記者予想保存",
                            date_str, jyo_cd_str, len(predictions_to_save))

        logger.info("[%s] 記者予想合計: %d 件保存", date_str, total_saved)

    def _parse_prediction_page(self, html, jyo_cd_str, date_str):
        """
        記者予想ページのHTMLをパースする。

        Parameters:
            html: HTMLテキスト
            jyo_cd_str: 会場コード文字列
            date_str: "YYYYMMDD" 形式

        Returns:
            list[dict]: 記者予想データのリスト
        """
        soup = BeautifulSoup(html, "html.parser")
        predictions = []

        # 方法1: pd.read_html でテーブルから取得を試行
        try:
            dfs = pd.read_html(html)
            for i, df in enumerate(dfs):
                race_no = i + 1
                race_id = f"{jyo_cd_str}_{date_str}_{race_no:02d}"
                extracted = self._extract_predictions_from_df(df, race_id)
                predictions.extend(extracted)

            if predictions:
                return predictions
        except ValueError:
            pass

        # 方法2: BeautifulSoup でセクションから取得
        race_sections = soup.find_all(
            ["div", "section", "table"],
            class_=re.compile(r"race|yoso|prediction|forecast", re.IGNORECASE)
        )

        current_race_no = 0
        for section in race_sections:
            # レース番号を検出
            race_no_match = re.search(r"(\d+)\s*[Rレース]", section.get_text())
            if race_no_match:
                current_race_no = int(race_no_match.group(1))

            if current_race_no < 1:
                continue

            race_id = f"{jyo_cd_str}_{date_str}_{current_race_no:02d}"
            text = section.get_text(separator="\n", strip=True)
            extracted = self._extract_predictions_from_text(text, race_id)
            predictions.extend(extracted)

        # 方法3: ページ全体テキストからパターンマッチ
        if not predictions:
            full_text = soup.get_text(separator="\n", strip=True)
            predictions = self._extract_predictions_from_fulltext(
                full_text, jyo_cd_str, date_str
            )

        return predictions

    def _extract_predictions_from_df(self, df, race_id):
        """DataFrameから記者予想を抽出する"""
        if df.empty:
            return []

        df.columns = [str(c).strip().replace("\n", "").replace(" ", "")
                      for c in df.columns]

        # ライン予想カラムの候補
        line_col = self._find_column(df, ["並び", "ライン", "予想ライン",
                                           "展開予想", "line"])
        reporter_col = self._find_column(df, ["記者", "予想者", "担当"])
        conf_col = self._find_column(df, ["自信度", "信頼度", "確度", "◎○△"])

        if line_col is None:
            # ライン専用カラムがない場合、テキスト全体から検索
            return []

        results = []
        for _, row in df.iterrows():
            line_text = str(row.get(line_col, "")).strip()
            if not line_text or line_text == "nan":
                continue

            # ライン形式（"1-2-3　4-5　6-7"）かチェック
            if not re.search(r"\d[-\-ー]\d", line_text):
                continue

            reporter = str(row.get(reporter_col, "")).strip() if reporter_col else "unknown"
            if reporter == "nan":
                reporter = "unknown"

            confidence = str(row.get(conf_col, "")).strip() if conf_col else None
            if confidence == "nan":
                confidence = None

            results.append({
                "race_id": race_id,
                "reporter_name": reporter,
                "predicted_line": line_text,
                "confidence": confidence,
            })

        return results

    def _extract_predictions_from_text(self, text, race_id):
        """テキストブロックから記者予想を抽出する"""
        predictions = []

        # ライン形式: "1-2-3　4-5　6-7" を検出
        line_pattern = re.compile(
            r"(\d[-\-ー]\d(?:[-\-ー]\d)*(?:[\s　]+\d[-\-ー]\d(?:[-\-ー]\d)*)*)"
        )

        for line in text.split("\n"):
            line = line.strip()
            m = line_pattern.search(line)
            if m:
                predicted_line = m.group(1).strip()
                # ハイフンを統一
                predicted_line = re.sub(r"[\-ー]", "-", predicted_line)

                # 記者名を検出（ライン行の前に名前がある場合）
                reporter = "unknown"
                name_match = re.match(r"^(\S{2,6})\s*[:：]?\s*" + re.escape(predicted_line), line)
                if name_match:
                    reporter = name_match.group(1)

                predictions.append({
                    "race_id": race_id,
                    "reporter_name": reporter,
                    "predicted_line": predicted_line,
                    "confidence": None,
                })

        return predictions

    def _extract_predictions_from_fulltext(self, full_text, jyo_cd_str, date_str):
        """ページ全体テキストから記者予想を抽出する"""
        predictions = []
        current_race_no = 0

        line_pattern = re.compile(
            r"(\d[-\-ー]\d(?:[-\-ー]\d)*(?:[\s　]+\d[-\-ー]\d(?:[-\-ー]\d)*)*)"
        )

        for line in full_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # レース番号の検出
            race_match = re.match(r"(\d{1,2})\s*[Rレース]", line)
            if race_match:
                current_race_no = int(race_match.group(1))
                continue

            if current_race_no < 1:
                continue

            race_id = f"{jyo_cd_str}_{date_str}_{current_race_no:02d}"

            m = line_pattern.search(line)
            if m:
                predicted_line = m.group(1).strip()
                predicted_line = re.sub(r"[\-ー]", "-", predicted_line)

                predictions.append({
                    "race_id": race_id,
                    "reporter_name": "unknown",
                    "predicted_line": predicted_line,
                    "confidence": None,
                })

        return predictions

    # =========================================================
    # ユーティリティ
    # =========================================================

    def _find_column(self, df, candidates):
        """候補のカラム名からDataFrame内に存在するものを返す"""
        for c in candidates:
            for col in df.columns:
                if c in str(col):
                    return col
        return None

    def _to_int(self, val):
        """値をintに変換。失敗時はNone。"""
        if pd.isna(val):
            return None
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return None

    # =========================================================
    # DB保存
    # =========================================================

    def _save_comments(self, comments):
        """
        コメントをDBに保存する。
        comment_text が None/空の場合は保存しない。

        Parameters:
            comments: list[dict]
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

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
            except sqlite3.Error as e:
                logger.warning("comments INSERT エラー (race_id=%s, sha_ban=%s): %s",
                               c["race_id"], c.get("sha_ban"), e)

        conn.commit()
        conn.close()

    def _save_predictions(self, predictions):
        """
        記者予想をDBに保存する。

        Parameters:
            predictions: list[dict]
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        for p in predictions:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO reporter_predictions
                    (race_id, reporter_name, predicted_line,
                     confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    p["race_id"],
                    p.get("reporter_name", "unknown"),
                    p.get("predicted_line"),
                    p.get("confidence"),
                    now,
                ))
            except sqlite3.Error as e:
                logger.warning("predictions INSERT エラー (race_id=%s): %s",
                               p["race_id"], e)

        conn.commit()
        conn.close()

    # =========================================================
    # コメント未取得レースの自動検出・取得
    # =========================================================

    def scrape_missing_comments(self):
        """
        DBにレースが存在するがコメント未取得のrace_idを自動検出し、
        日付単位でまとめてコメントを取得する。
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # races にあるが comments にないレースの日付一覧
        cur.execute("""
            SELECT DISTINCT r.race_date
            FROM races r
            LEFT JOIN comments c ON r.race_id = c.race_id
            WHERE c.race_id IS NULL
            ORDER BY r.race_date
        """)
        missing_dates = [row[0] for row in cur.fetchall()]
        conn.close()

        if not missing_dates:
            logger.info("コメント未取得のレースはありません")
            return

        logger.info("コメント未取得: %d 日分", len(missing_dates))

        for i, date_str in enumerate(missing_dates):
            logger.info("[%d/%d] %s のコメント取得中...",
                        i + 1, len(missing_dates), date_str)
            try:
                self.scrape_comments_for_date(date_str)
            except Exception as e:
                logger.error("[%s] コメント取得エラー: %s", date_str, e)
                self._log_failed(f"comments_{date_str}", str(e))

    # =========================================================
    # 記者予想未取得レースの自動検出・取得
    # =========================================================

    def scrape_missing_predictions(self):
        """
        DBにレースが存在するが記者予想未取得のrace_idを自動検出し、
        日付単位でまとめて記者予想を取得する。
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        cur.execute("""
            SELECT DISTINCT r.race_date
            FROM races r
            LEFT JOIN reporter_predictions rp ON r.race_id = rp.race_id
            WHERE rp.race_id IS NULL
            ORDER BY r.race_date
        """)
        missing_dates = [row[0] for row in cur.fetchall()]
        conn.close()

        if not missing_dates:
            logger.info("記者予想未取得のレースはありません")
            return

        logger.info("記者予想未取得: %d 日分", len(missing_dates))

        for i, date_str in enumerate(missing_dates):
            logger.info("[%d/%d] %s の記者予想取得中...",
                        i + 1, len(missing_dates), date_str)
            try:
                self.scrape_predictions_for_date(date_str)
            except Exception as e:
                logger.error("[%s] 記者予想取得エラー: %s", date_str, e)
                self._log_failed(f"pred_{date_str}", str(e))

    # =========================================================
    # 進捗表示
    # =========================================================

    def get_status(self):
        """取得済み件数・進捗を表示する"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # racesテーブルの件数
        cur.execute("SELECT COUNT(*) FROM races")
        race_count = cur.fetchone()[0]

        # コメント件数
        cur.execute("SELECT COUNT(*) FROM comments")
        comment_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT race_id) FROM comments")
        comment_race_count = cur.fetchone()[0]

        # 記者予想件数
        cur.execute("SELECT COUNT(*) FROM reporter_predictions")
        pred_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT race_id) FROM reporter_predictions")
        pred_race_count = cur.fetchone()[0]

        # コメント未取得レース数
        cur.execute("""
            SELECT COUNT(DISTINCT r.race_id)
            FROM races r
            LEFT JOIN comments c ON r.race_id = c.race_id
            WHERE c.race_id IS NULL
        """)
        missing_comment = cur.fetchone()[0]

        # 記者予想未取得レース数
        cur.execute("""
            SELECT COUNT(DISTINCT r.race_id)
            FROM races r
            LEFT JOIN reporter_predictions rp ON r.race_id = rp.race_id
            WHERE rp.race_id IS NULL
        """)
        missing_pred = cur.fetchone()[0]

        conn.close()

        print("\n========== Gamboo スクレイピング進捗 ==========")
        print(f"  DB: {self.db_path}")
        print(f"  レース総数（chariloto）:  {race_count:,}")
        print(f"  ---")
        print(f"  コメント数:              {comment_count:,}")
        print(f"  コメント取得済みレース:  {comment_race_count:,}")
        print(f"  コメント未取得レース:    {missing_comment:,}")
        print(f"  ---")
        print(f"  記者予想数:              {pred_count:,}")
        print(f"  記者予想取得済みレース:  {pred_race_count:,}")
        print(f"  記者予想未取得レース:    {missing_pred:,}")
        print("================================================\n")

        return {
            "race_count": race_count,
            "comment_count": comment_count,
            "comment_race_count": comment_race_count,
            "missing_comment": missing_comment,
            "pred_count": pred_count,
            "pred_race_count": pred_race_count,
            "missing_pred": missing_pred,
        }


# =========================================================
# コマンドラインインターフェース
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gamboo スクレイパー（選手コメント・記者予想）"
    )
    parser.add_argument("--date", type=str,
                        help="取得対象日（YYYYMMDD）")
    parser.add_argument("--comments", action="store_true",
                        help="選手コメントを取得")
    parser.add_argument("--predictions", action="store_true",
                        help="記者予想を取得")
    parser.add_argument("--missing", action="store_true",
                        help="未取得分を自動検出して取得")
    parser.add_argument("--status", action="store_true",
                        help="取得済み件数・進捗を表示")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="リクエスト間隔（秒、デフォルト: 3.0）")
    parser.add_argument("--db", type=str, default=None,
                        help="DBファイルパス")

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
        logger.info("=== 未取得コメント・記者予想の自動取得 ===")
        scraper.scrape_missing_comments()
        scraper.scrape_missing_predictions()
        scraper.get_status()
        return

    if args.date:
        if args.comments or (not args.comments and not args.predictions):
            logger.info("=== コメント取得: %s ===", args.date)
            scraper.scrape_comments_for_date(args.date)

        if args.predictions or (not args.comments and not args.predictions):
            logger.info("=== 記者予想取得: %s ===", args.date)
            scraper.scrape_predictions_for_date(args.date)

        scraper.get_status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
