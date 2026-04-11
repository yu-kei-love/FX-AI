# ===========================================
# scraper/scraper_historical.py
# 競輪 - 過去データスクレイパー（chariloto.com）
#
# 対象：chariloto.com（公開ページ・ログイン不要）
# 保存先：SQLite（data/keirin/keirin.db）
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

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"
FAILED_LOG = Path(__file__).resolve().parent / "failed_races.log"
PROGRESS_DIR = Path(__file__).resolve().parent

BASE_URL = "https://www.chariloto.com"

# chariloto.com で使われる会場コード（01〜43）
# bank_master.py の venue_id に対応
JYO_CODES = [f"{i:02d}" for i in range(1, 44)]


class ChariLotoScraper:
    """chariloto.com から競輪の過去レース結果・出走表を取得する"""

    def __init__(self, db_path=None, delay=2.0, jyo_cds=None):
        """
        Parameters:
            db_path: SQLiteファイルのパス（デフォルト: data/keirin/keirin.db）
            delay: リクエスト間の待機秒数（サーバー負荷配慮）
            jyo_cds: 対象会場コードのリスト（int または "01"〜"43" 形式）
                     None の場合は全43会場を対象
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = delay
        # 会場コードを "01"〜"43" 形式に正規化
        if jyo_cds is None:
            self.jyo_cds = list(JYO_CODES)
        else:
            self.jyo_cds = [f"{int(c):02d}" for c in jyo_cds]
        # 並列実行時の進捗ファイル（対象会場グループ別）
        self.progress_file = self._make_progress_file()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KeirinDataCollector/1.0 (personal research use)",
            "Accept-Language": "ja,en;q=0.9",
        })
        self._init_db()

    def _make_progress_file(self) -> Path:
        """
        対象会場グループごとに進捗ファイルを分離する。
        並列実行時に互いの進捗を上書きしないため。
        """
        if len(self.jyo_cds) == len(JYO_CODES):
            return PROGRESS_DIR / ".scrape_progress"
        # 先頭と末尾のコードでサフィックスを作る
        suffix = f"{self.jyo_cds[0]}_{self.jyo_cds[-1]}"
        return PROGRESS_DIR / f".scrape_progress_{suffix}"

    def _connect_db(self) -> sqlite3.Connection:
        """
        DBに接続し WAL モードを有効化する。

        WAL モード:
          複数プロセスから同時書き込みしても競合しない。
          synchronous=NORMAL でディスクI/Oを軽減。
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self):
        """DBスキーマを初期化する（WALモード有効化も実施）"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect_db()
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS races (
                race_id     TEXT PRIMARY KEY,
                jyo_cd      INTEGER NOT NULL,
                race_date   TEXT NOT NULL,
                race_no     INTEGER NOT NULL,
                grade       TEXT,
                stage       TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results (
                race_id     TEXT NOT NULL,
                rank        INTEGER NOT NULL,
                sha_ban     INTEGER NOT NULL,
                senshu_name TEXT,
                kimari_te   TEXT,
                line_id     INTEGER,
                line_pos    INTEGER,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (race_id, rank),
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );

            CREATE TABLE IF NOT EXISTS entries (
                race_id       TEXT NOT NULL,
                sha_ban       INTEGER NOT NULL,
                senshu_name   TEXT,
                age           INTEGER,
                ki_betsu      INTEGER,
                todofuken     TEXT,
                kyoso_tokuten REAL,
                gear_ratio    REAL,
                back_count    INTEGER,
                home_count    INTEGER,
                kyakushitsu   TEXT,
                created_at    TEXT NOT NULL,
                PRIMARY KEY (race_id, sha_ban),
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );

            CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
            CREATE INDEX IF NOT EXISTS idx_results_race ON results(race_id);
            CREATE INDEX IF NOT EXISTS idx_entries_race ON entries(race_id);
        """)
        conn.commit()
        conn.close()
        logger.info("DB初期化完了: %s", self.db_path)

    def _polite_sleep(self):
        """サーバー負荷を考慮した待機"""
        time.sleep(self.delay)

    def _fetch_html(self, url, max_retries=3):
        """
        URLからHTMLを取得する。リトライ付き。

        Returns:
            str: HTMLテキスト。取得失敗時はNone。
        """
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

    def _log_failed(self, race_id, reason):
        """失敗したrace_idをログに記録する"""
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}\t{race_id}\t{reason}\n")

    def _get_scraped_race_ids(self):
        """取得済みのrace_idセットを返す"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT race_id FROM races")
        ids = {row[0] for row in cur.fetchall()}
        conn.close()
        return ids

    def _get_scraped_dates(self):
        """取得済みの日付セットを返す"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT race_date FROM races")
        dates = {row[0] for row in cur.fetchall()}
        conn.close()
        return dates

    # =========================================================
    # 結果ページのスクレイピング
    # =========================================================

    def scrape_race_result(self, jyo_cd, date_str):
        """
        1会場・1日分のレース結果を取得する。

        URL: https://www.chariloto.com/keirin/results/{jyo_cd}/{date}

        Parameters:
            jyo_cd: 会場コード（"01"〜"43"）
            date_str: "YYYYMMDD" 形式

        Returns:
            list[dict]: レース結果のリスト。取得失敗時は空リスト。
        """
        # URLの日付フォーマット（YYYY-MM-DD or YYYYMMDD はサイトに合わせる）
        url = f"{BASE_URL}/keirin/results/{jyo_cd}/{date_str}"
        html = self._fetch_html(url)
        if html is None:
            return []

        try:
            dfs = pd.read_html(html)
        except ValueError:
            # テーブルが見つからない
            logger.debug("テーブルなし: %s", url)
            return []

        if not dfs:
            return []

        results = []
        for i, df in enumerate(dfs):
            race_no = i + 1
            race_id = f"{jyo_cd}_{date_str}_{race_no:02d}"

            # 結果テーブルから着順・車番・選手名・決まり手を抽出
            parsed = self._parse_result_table(df, race_id)
            if parsed:
                results.append({
                    "race_id": race_id,
                    "jyo_cd": int(jyo_cd),
                    "race_date": date_str,
                    "race_no": race_no,
                    "results": parsed["results"],
                    "line_info": parsed.get("line_info", []),
                })

        return results

    def _parse_result_table(self, df, race_id):
        """
        結果テーブルのDataFrameをパースする。

        カラム名はサイトのHTML構造に依存するため、
        位置ベース・名前ベースの両方で対応する。

        Returns:
            dict: {"results": [...], "line_info": [...]}
            パースできない場合はNone。
        """
        if df.empty:
            return None

        # カラム名を正規化（全角スペース・改行除去）
        df.columns = [str(c).strip().replace("\n", "").replace(" ", "")
                      for c in df.columns]

        results = []
        line_raw = None

        # 着順カラムの候補
        rank_col = self._find_column(df, ["着順", "着", "順位"])
        sha_ban_col = self._find_column(df, ["車番", "車", "枠番"])
        name_col = self._find_column(df, ["選手名", "選手", "氏名"])
        kimari_col = self._find_column(df, ["決まり手", "決り手", "決手"])

        if rank_col is None or sha_ban_col is None:
            # カラムが見つからない場合、位置ベースで試行
            if len(df.columns) >= 3:
                rank_col = df.columns[0]
                sha_ban_col = df.columns[1]
                name_col = df.columns[2] if len(df.columns) > 2 else None
                kimari_col = None
            else:
                return None

        for _, row in df.iterrows():
            try:
                rank_val = row[rank_col]
                # 着順が数値でない行（ヘッダー重複等）をスキップ
                rank = self._to_int(rank_val)
                if rank is None or rank < 1 or rank > 9:
                    continue

                sha_ban = self._to_int(row[sha_ban_col])
                if sha_ban is None:
                    continue

                senshu_name = str(row[name_col]).strip() if name_col else ""
                kimari_te = str(row[kimari_col]).strip() if kimari_col and pd.notna(row.get(kimari_col)) else ""

                # 1〜3着のみ保存
                if rank <= 3:
                    results.append({
                        "race_id": race_id,
                        "rank": rank,
                        "sha_ban": sha_ban,
                        "senshu_name": senshu_name,
                        "kimari_te": kimari_te,
                    })
            except Exception as e:
                logger.debug("行パースエラー (race_id=%s): %s", race_id, e)
                continue

        if not results:
            return None

        return {"results": results, "line_info": []}

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

    def _to_float(self, val):
        """値をfloatに変換。失敗時はNone。"""
        if pd.isna(val):
            return None
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return None

    # =========================================================
    # 並び（ライン）のパース
    # =========================================================

    def _parse_line_from_result(self, line_text):
        """
        並びテキストをライン番号に変換する。

        入力例: "1-2-3　4-5　6-7"
        出力: {1: (1,1), 2: (1,2), 3: (1,3), 4: (2,1), 5: (2,2), 6: (3,1), 7: (3,2)}
              → {車番: (line_id, line_pos)}

        単騎（ハイフンなしの単独番号）はライン番号=自動採番、ポジション=1

        Parameters:
            line_text: "1-2-3　4-5　6-7" 形式の文字列

        Returns:
            dict: {sha_ban: (line_id, line_pos)}
        """
        if not line_text or not isinstance(line_text, str):
            return {}

        # 全角スペース・半角スペース・タブで分割
        groups = re.split(r'[\s　]+', line_text.strip())
        result = {}
        line_id = 1

        for group in groups:
            if not group:
                continue
            # ハイフン区切りで車番を分解（全角・半角対応）
            members = re.split(r'[-\-ー]', group)
            pos = 1
            for m in members:
                m = m.strip()
                if not m:
                    continue
                try:
                    sha_ban = int(m)
                    result[sha_ban] = (line_id, pos)
                    pos += 1
                except ValueError:
                    continue
            if pos > 1:  # 少なくとも1人はパースできた
                line_id += 1

        return result

    # =========================================================
    # 出走表のスクレイピング
    # =========================================================

    def scrape_race_entry(self, jyo_cd, date_str, race_no):
        """
        1レース分の出走表を取得する。

        URL: https://www.chariloto.com/keirin/race/{jyo_cd}/{date}/{race_no}

        Parameters:
            jyo_cd: 会場コード
            date_str: "YYYYMMDD" 形式
            race_no: レース番号

        Returns:
            list[dict]: 出走選手リスト。取得失敗時は空リスト。
        """
        url = f"{BASE_URL}/keirin/race/{jyo_cd}/{date_str}/{race_no}"
        html = self._fetch_html(url)
        if html is None:
            return []

        try:
            dfs = pd.read_html(html)
        except ValueError:
            logger.debug("テーブルなし: %s", url)
            return []

        if not dfs:
            return []

        race_id = f"{jyo_cd}_{date_str}_{race_no:02d}"
        entries = self._parse_entry_table(dfs[0], race_id)
        return entries

    def _parse_entry_table(self, df, race_id):
        """
        出走表のDataFrameをパースする。

        Returns:
            list[dict]: 出走選手データのリスト
        """
        if df.empty:
            return []

        # カラム名正規化
        df.columns = [str(c).strip().replace("\n", "").replace(" ", "")
                      for c in df.columns]

        sha_ban_col = self._find_column(df, ["車番", "車", "枠"])
        name_col = self._find_column(df, ["選手名", "選手", "氏名"])
        age_col = self._find_column(df, ["年齢", "齢"])
        ki_col = self._find_column(df, ["期別", "期"])
        pref_col = self._find_column(df, ["府県", "都道府県", "県", "地区"])
        tokuten_col = self._find_column(df, ["競走得点", "得点", "得"])
        gear_col = self._find_column(df, ["ギア", "ギヤ", "gear", "G倍"])
        back_col = self._find_column(df, ["バック", "B回", "B数"])
        home_col = self._find_column(df, ["ホーム", "H回", "H数"])
        style_col = self._find_column(df, ["脚質", "脚"])

        entries = []
        for _, row in df.iterrows():
            sha_ban = self._to_int(row.get(sha_ban_col)) if sha_ban_col else None
            if sha_ban is None or sha_ban < 1 or sha_ban > 9:
                continue

            entry = {
                "race_id": race_id,
                "sha_ban": sha_ban,
                "senshu_name": str(row.get(name_col, "")).strip() if name_col else "",
                "age": self._to_int(row.get(age_col)) if age_col else None,
                "ki_betsu": self._to_int(row.get(ki_col)) if ki_col else None,
                "todofuken": str(row.get(pref_col, "")).strip() if pref_col else None,
                "kyoso_tokuten": self._to_float(row.get(tokuten_col)) if tokuten_col else None,
                "gear_ratio": self._to_float(row.get(gear_col)) if gear_col else None,
                "back_count": self._to_int(row.get(back_col)) if back_col else None,
                "home_count": self._to_int(row.get(home_col)) if home_col else None,
                "kyakushitsu": str(row.get(style_col, "")).strip() if style_col else None,
            }
            entries.append(entry)

        return entries

    # =========================================================
    # DB保存
    # =========================================================

    def _save_to_db(self, races, results, entries):
        """
        レース・結果・出走表をDBに保存する。
        取得済みのrace_idはスキップ（重複防止）。

        Parameters:
            races: list[dict]   - racesテーブル用データ
            results: list[dict] - resultsテーブル用データ
            entries: list[dict] - entriesテーブル用データ
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect_db()
        cur = conn.cursor()

        for race in races:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO races
                    (race_id, jyo_cd, race_date, race_no, grade, stage, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    race["race_id"],
                    race["jyo_cd"],
                    race["race_date"],
                    race["race_no"],
                    race.get("grade"),
                    race.get("stage"),
                    now,
                ))
            except sqlite3.Error as e:
                logger.warning("races INSERT エラー (race_id=%s): %s",
                               race["race_id"], e)

        for r in results:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO results
                    (race_id, rank, sha_ban, senshu_name, kimari_te,
                     line_id, line_pos, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r["race_id"],
                    r["rank"],
                    r["sha_ban"],
                    r.get("senshu_name"),
                    r.get("kimari_te"),
                    r.get("line_id"),
                    r.get("line_pos"),
                    now,
                ))
            except sqlite3.Error as e:
                logger.warning("results INSERT エラー (race_id=%s, rank=%d): %s",
                               r["race_id"], r["rank"], e)

        for e in entries:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO entries
                    (race_id, sha_ban, senshu_name, age, ki_betsu,
                     todofuken, kyoso_tokuten, gear_ratio,
                     back_count, home_count, kyakushitsu, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    e["race_id"],
                    e["sha_ban"],
                    e.get("senshu_name"),
                    e.get("age"),
                    e.get("ki_betsu"),
                    e.get("todofuken"),
                    e.get("kyoso_tokuten"),
                    e.get("gear_ratio"),
                    e.get("back_count"),
                    e.get("home_count"),
                    e.get("kyakushitsu"),
                    now,
                ))
            except sqlite3.Error as e_err:
                logger.warning("entries INSERT エラー (race_id=%s, sha_ban=%s): %s",
                               e["race_id"], e.get("sha_ban"), e_err)

        conn.commit()
        conn.close()

    # =========================================================
    # 1日分のスクレイピング
    # =========================================================

    def scrape_day(self, date_str):
        """
        指定日の全会場のレース結果・出走表を取得する。

        Parameters:
            date_str: "YYYYMMDD" 形式
        """
        scraped_ids = self._get_scraped_race_ids()
        day_races = []
        day_results = []
        day_entries = []

        for jyo_cd in self.jyo_cds:
            # 結果ページを取得
            race_data_list = self.scrape_race_result(jyo_cd, date_str)
            if not race_data_list:
                continue

            logger.info("[%s] 会場 %s: %d レース検出",
                        date_str, jyo_cd, len(race_data_list))

            for race_data in race_data_list:
                race_id = race_data["race_id"]

                # 取得済みスキップ
                if race_id in scraped_ids:
                    logger.debug("[SKIP] %s は取得済み", race_id)
                    continue

                # races テーブル用
                day_races.append({
                    "race_id": race_id,
                    "jyo_cd": race_data["jyo_cd"],
                    "race_date": date_str,
                    "race_no": race_data["race_no"],
                })

                # results テーブル用
                day_results.extend(race_data["results"])

                # 出走表を取得
                self._polite_sleep()
                entry_list = self.scrape_race_entry(
                    jyo_cd, date_str, race_data["race_no"]
                )
                day_entries.extend(entry_list)

            self._polite_sleep()

        # DB保存
        if day_races:
            self._save_to_db(day_races, day_results, day_entries)
            logger.info("[%s] 保存完了: %d レース, %d 結果, %d 出走",
                        date_str, len(day_races), len(day_results),
                        len(day_entries))
        else:
            logger.info("[%s] 取得対象なし（開催なし or 取得済み）", date_str)

    # =========================================================
    # 期間指定でまとめて取得
    # =========================================================

    def scrape_range(self, start_date, end_date):
        """
        指定期間のレースデータをまとめて取得する。

        Parameters:
            start_date: "YYYY-MM-DD" or "YYYYMMDD" 形式
            end_date:   "YYYY-MM-DD" or "YYYYMMDD" 形式
        """
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)

        if start_dt is None or end_dt is None:
            logger.error("日付のパースに失敗しました: start=%s, end=%s",
                         start_date, end_date)
            return

        total_days = (end_dt - start_dt).days + 1
        current = start_dt
        day_count = 0

        logger.info("=== スクレイピング開始 ===")
        logger.info("期間: %s 〜 %s（%d日間）",
                     start_dt.strftime("%Y-%m-%d"),
                     end_dt.strftime("%Y-%m-%d"),
                     total_days)
        logger.info("対象会場: %d場 (%s)",
                    len(self.jyo_cds),
                    ",".join(self.jyo_cds))
        logger.info("進捗ファイル: %s", self.progress_file.name)

        while current <= end_dt:
            date_str = current.strftime("%Y%m%d")
            day_count += 1
            logger.info("[%d/%d] %s", day_count, total_days,
                        current.strftime("%Y-%m-%d"))

            try:
                self.scrape_day(date_str)
            except Exception as e:
                logger.error("[%s] 予期せぬエラー: %s", date_str, e)
                self._log_failed(f"day_{date_str}", str(e))

            # 進捗を保存（--resume用）
            self._save_progress(current.strftime("%Y-%m-%d"),
                                end_dt.strftime("%Y-%m-%d"))

            current += timedelta(days=1)

        logger.info("=== スクレイピング完了 ===")
        self.get_status()

    def _parse_date(self, date_str):
        """日付文字列をdatetimeに変換する"""
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _save_progress(self, last_date, end_date):
        """進捗をファイルに保存する（--resume用・グループ別）"""
        with open(self.progress_file, "w", encoding="utf-8") as f:
            f.write(f"{last_date}\n{end_date}\n")

    def _load_progress(self):
        """保存された進捗を読み込む（グループ別）"""
        if not self.progress_file.exists():
            return None, None
        try:
            lines = self.progress_file.read_text(encoding="utf-8").strip().split("\n")
            if len(lines) >= 2:
                return lines[0], lines[1]
        except Exception:
            pass
        return None, None

    # =========================================================
    # 進捗表示
    # =========================================================

    def get_status(self):
        """取得済み件数・進捗を表示する"""
        conn = self._connect_db()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM races")
        race_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM results")
        result_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM entries")
        entry_count = cur.fetchone()[0]

        cur.execute("SELECT MIN(race_date), MAX(race_date) FROM races")
        date_range = cur.fetchone()

        cur.execute("SELECT COUNT(DISTINCT race_date) FROM races")
        day_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT jyo_cd) FROM races")
        venue_count = cur.fetchone()[0]

        conn.close()

        print("\n========== スクレイピング進捗 ==========")
        print(f"  DB: {self.db_path}")
        print(f"  レース数:   {race_count:,}")
        print(f"  結果数:     {result_count:,}")
        print(f"  出走表数:   {entry_count:,}")
        print(f"  日付範囲:   {date_range[0]} 〜 {date_range[1]}")
        print(f"  取得日数:   {day_count:,}")
        print(f"  会場数:     {venue_count}")
        print("========================================\n")

        # 保存された進捗
        last_date, end_date = self._load_progress()
        if last_date and end_date:
            print(f"  前回の進捗: {last_date} まで取得済み（目標: {end_date}）")

        return {
            "race_count": race_count,
            "result_count": result_count,
            "entry_count": entry_count,
            "date_min": date_range[0],
            "date_max": date_range[1],
            "day_count": day_count,
            "venue_count": venue_count,
        }


# =========================================================
# コマンドラインインターフェース
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="競輪過去データスクレイパー（chariloto.com）"
    )
    parser.add_argument("--start", type=str,
                        help="開始日（YYYY-MM-DD）")
    parser.add_argument("--end", type=str,
                        help="終了日（YYYY-MM-DD）")
    parser.add_argument("--resume", action="store_true",
                        help="前回の続きから再開")
    parser.add_argument("--status", action="store_true",
                        help="取得済み件数・進捗を表示")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="リクエスト間隔（秒、デフォルト: 2.0）")
    parser.add_argument("--db", type=str, default=None,
                        help="DBファイルパス（デフォルト: data/keirin/keirin.db）")
    parser.add_argument("--jyo_cds", type=str, default=None,
                        help="対象会場コード（カンマ区切り、例: 1,2,3,4）。"
                             "省略時は全43会場を対象")

    args = parser.parse_args()

    # --jyo_cds のパース
    jyo_cds = None
    if args.jyo_cds:
        try:
            jyo_cds = [int(c.strip()) for c in args.jyo_cds.split(",")
                       if c.strip()]
            if not jyo_cds:
                raise ValueError("会場コードが空")
            for c in jyo_cds:
                if c < 1 or c > 43:
                    raise ValueError(f"会場コードは1〜43: {c}")
        except ValueError as e:
            logger.error("--jyo_cds の形式が不正です: %s", e)
            return

    scraper = ChariLotoScraper(
        db_path=args.db,
        delay=args.delay,
        jyo_cds=jyo_cds,
    )

    if args.status:
        scraper.get_status()
        return

    if args.resume:
        last_date, end_date = scraper._load_progress()
        if last_date is None or end_date is None:
            # --start/--end が指定されていればそちらを優先して続行
            if args.start and args.end:
                logger.info("進捗ファイルなし。--start/--end で新規実行します")
                scraper.scrape_range(args.start, args.end)
                return
            logger.error("再開用の進捗ファイルが見つかりません: %s",
                         scraper.progress_file)
            return
        # 前回の翌日から再開
        resume_dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        resume_date = resume_dt.strftime("%Y-%m-%d")
        logger.info("前回の続きから再開: %s 〜 %s", resume_date, end_date)
        scraper.scrape_range(resume_date, end_date)
        return

    if args.start and args.end:
        scraper.scrape_range(args.start, args.end)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
