# ===========================================
# scraper/scraper_historical.py
# 競輪 - 過去データスクレイパー（chariloto.com）
#
# 対象：chariloto.com（公開ページ・ログイン不要）
# 保存先：SQLite（data/keirin/keirin.db）
#
# v0.13:
#   - JKA公式会場コード（11〜87）を使うように変更
#   - 日付フォーマットを YYYY-MM-DD に変更
#   - pd.read_html() を io.StringIO() でラップ
#   - 年別インデックス（?year=YYYY）から実在日付のみ取得
#   - 結果ページ1枚に全レース（1R〜12R）が含まれることを前提に
#     出走表の別ページ取得を廃止
#   - 周回予想（ライン）を <span class="p10"> 区切りでパース
#
# 注意：このファイルはスクレイピングコードのため
#       note販売パッケージには含めない
# ===========================================

import argparse
import io
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# bank_master.py を参照（JKA会場コード取得）
_BANK_PATH = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(_BANK_PATH))
from bank_master import BANK_MASTER  # noqa: E402

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
FAILED_KDREAMS_LOG = Path(__file__).resolve().parent / "failed_kdreams.log"
PROGRESS_DIR = Path(__file__).resolve().parent

BASE_URL = "https://www.chariloto.com"
KDREAMS_BASE_URL = "https://keirin.kdreams.jp"

# bank_master から JKA会場コード一覧を生成
# 例: ["11", "12", "13", "21", "22", ...]
JKA_CODES = sorted({info["jka_code"] for info in BANK_MASTER.values()
                    if info.get("jka_code")})

# jka_code → venue_id（bank_master内部ID "01"〜"43"）の逆引き
JKA_TO_VENUE_ID = {
    info["jka_code"]: info["venue_id"]
    for info in BANK_MASTER.values()
    if info.get("jka_code")
}


class ChariLotoScraper:
    """chariloto.com から競輪の過去レース結果を取得する"""

    def __init__(self, db_path=None, delay=2.0, jyo_cds=None):
        """
        Parameters:
            db_path: SQLiteファイルのパス（デフォルト: data/keirin/keirin.db）
            delay: リクエスト間の待機秒数（サーバー負荷配慮）
            jyo_cds: 対象会場コードのリスト（JKAコード: 11〜87）
                     int または str を受け付け、内部で "11"〜"87" 形式に正規化
                     None の場合は全43会場を対象
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = delay
        # 会場コードを "11"〜"87" 形式（JKAコード）に正規化
        if jyo_cds is None:
            self.jyo_cds = list(JKA_CODES)
        else:
            self.jyo_cds = [f"{int(c):02d}" for c in jyo_cds]
            # 未定義の JKA コードをチェック
            invalid = [c for c in self.jyo_cds if c not in JKA_CODES]
            if invalid:
                logger.warning("未定義のJKAコードを除外: %s", invalid)
                self.jyo_cds = [c for c in self.jyo_cds if c in JKA_CODES]
        self.progress_file = self._make_progress_file()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (KeirinDataCollector/1.0; "
                          "personal research use)",
            "Accept-Language": "ja,en;q=0.9",
        })
        self._init_db()

    def _make_progress_file(self) -> Path:
        """
        対象会場グループごとに進捗ファイルを分離する。
        並列実行時に互いの進捗を上書きしないため。
        """
        if len(self.jyo_cds) == len(JKA_CODES):
            return PROGRESS_DIR / ".scrape_progress"
        suffix = f"{self.jyo_cds[0]}_{self.jyo_cds[-1]}"
        return PROGRESS_DIR / f".scrape_progress_{suffix}"

    def _connect_db(self) -> sqlite3.Connection:
        """
        DBに接続し WAL モードを有効化する。
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self):
        """DBスキーマを初期化する"""
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
                agari_time  REAL,
                chakusa     TEXT,
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
                start_count   INTEGER,
                kyakushitsu   TEXT,
                created_at    TEXT NOT NULL,
                PRIMARY KEY (race_id, sha_ban),
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );

            CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
            CREATE INDEX IF NOT EXISTS idx_results_race ON results(race_id);
            CREATE INDEX IF NOT EXISTS idx_entries_race ON entries(race_id);
        """)

        # 既存DBにカラムがなければ追加（マイグレーション）
        cur.execute("PRAGMA table_info(entries)")
        entry_cols = {row[1] for row in cur.fetchall()}
        if "start_count" not in entry_cols:
            try:
                cur.execute("ALTER TABLE entries ADD COLUMN start_count INTEGER")
                logger.info("entries テーブルに start_count カラムを追加")
            except sqlite3.OperationalError as e:
                logger.warning("start_count 追加失敗: %s", e)

        cur.execute("PRAGMA table_info(results)")
        result_cols = {row[1] for row in cur.fetchall()}
        for col, typ in [("agari_time", "REAL"), ("chakusa", "TEXT")]:
            if col not in result_cols:
                try:
                    cur.execute(f"ALTER TABLE results ADD COLUMN {col} {typ}")
                    logger.info("results テーブルに %s カラムを追加", col)
                except sqlite3.OperationalError as e:
                    logger.warning("%s 追加失敗: %s", col, e)

        conn.commit()
        conn.close()
        logger.info("DB初期化完了: %s", self.db_path)

    def _polite_sleep(self):
        """サーバー負荷を考慮した待機"""
        time.sleep(self.delay)

    def _fetch_html(self, url, max_retries=3):
        """URLからHTMLを取得する。リトライ付き。"""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 404:
                    logger.debug("404 Not Found: %s", url)
                    return None
                if resp.status_code == 503:
                    logger.warning("503 Service Unavailable: %s (%d/%d)",
                                   url, attempt, max_retries)
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                resp.encoding = "utf-8"
                return resp.text
            except requests.RequestException as e:
                logger.warning("リクエストエラー: %s (%d/%d): %s",
                               url, attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return None

    def _log_failed(self, identifier, reason):
        """失敗情報をログファイルに記録する"""
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}\t{identifier}\t{reason}\n")

    def _get_scraped_race_ids(self):
        """取得済みのrace_idセットを返す"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT race_id FROM races")
        ids = {row[0] for row in cur.fetchall()}
        conn.close()
        return ids

    # =========================================================
    # 年別インデックスから実在日付リストを取得
    # =========================================================

    def get_race_dates_for_venue_year(self, jka_code, year):
        """
        指定会場・年の開催日リストを取得する。

        URL: /keirin/results/{jka_code}?year={year}

        Parameters:
            jka_code: JKA会場コード（"11"〜"87"）
            year    : 西暦（int）

        Returns:
            list[str]: "YYYY-MM-DD" 形式の開催日リスト（昇順）
        """
        url = f"{BASE_URL}/keirin/results/{jka_code}?year={year}"
        html = self._fetch_html(url)
        if html is None:
            return []

        # /keirin/results/{jka_code}/{YYYY-MM-DD} のリンクを全抽出
        pattern = re.compile(
            rf"/keirin/results/{jka_code}/(\d{{4}}-\d{{2}}-\d{{2}})"
        )
        dates = sorted(set(pattern.findall(html)))
        return dates

    # =========================================================
    # 1日分の結果ページを取得・パース
    # =========================================================

    def scrape_race_result(self, jka_code, date_str):
        """
        1会場・1日分のレース結果ページを取得し、全レースをパースする。

        URL: https://www.chariloto.com/keirin/results/{jka_code}/{YYYY-MM-DD}

        Parameters:
            jka_code : JKA会場コード（"11"〜"87"）
            date_str : "YYYY-MM-DD" 形式

        Returns:
            list[dict]: 各レースのデータ。取得失敗時は空リスト。
              [
                {
                  "race_id", "jyo_cd", "race_date", "race_no",
                  "results":  [着順情報のリスト],
                  "entries":  [出走情報のリスト],
                  "line_info": {sha_ban: (line_id, line_pos)},
                },
                ...
              ]
        """
        url = f"{BASE_URL}/keirin/results/{jka_code}/{date_str}"
        html = self._fetch_html(url)
        if html is None:
            return []

        # venue_id（bank_master内部ID "01"〜"43"）を取得
        venue_id = JKA_TO_VENUE_ID.get(jka_code)
        if venue_id is None:
            logger.warning("venue_id が見つかりません: jka_code=%s", jka_code)
            return []

        # YYYYMMDD 形式の日付（race_id生成用）
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_compact = dt.strftime("%Y%m%d")
        except ValueError:
            logger.error("日付パース失敗: %s", date_str)
            return []

        # BeautifulSoup と pandas 両方でパース
        try:
            soup = BeautifulSoup(html, "html.parser")
            dfs = pd.read_html(io.StringIO(html))
        except Exception as e:
            logger.debug("HTMLパース失敗 %s: %s", url, e)
            return []

        if not dfs:
            return []

        # レース数を table[0]（開催レース一覧）から取得
        # 例: ['開催レース', '1R', '2R', '3R', ..., '12R']
        n_races = 0
        try:
            first_row = list(dfs[0].iloc[0].values)
            race_cells = [str(v) for v in first_row[1:]
                          if re.match(r"^\d+R$", str(v))]
            n_races = len(race_cells)
        except Exception:
            pass

        if n_races == 0:
            logger.debug("レース数不明: %s", url)
            return []

        # 結果テーブル(着順)は table[0] の次から順番に並ぶ
        # パターン: [result_table, shukai_prediction, narabi, haito, haito, ...]
        # 結果テーブルは columns に "着", "車番", "選手名" を含むことで判別

        race_results = self._extract_race_tables(dfs)
        # 周回予想（ライン情報）を BS4 から取得
        line_info_list = self._extract_line_info(soup)

        # レース別に組み立て
        parsed_races = []
        for i in range(n_races):
            if i >= len(race_results):
                break
            result_rows = race_results[i]
            if not result_rows:
                continue

            race_no = i + 1
            race_id = f"{venue_id}_{date_compact}_{race_no:02d}"

            # ライン情報（i番目が取れなければ空）
            line_info = line_info_list[i] if i < len(line_info_list) else {}

            # 1〜3着の結果
            results = []
            entries = []
            for row in result_rows:
                rank = row.get("rank")
                sha_ban = row.get("sha_ban")
                if sha_ban is None:
                    continue

                # 1〜3着のみ results テーブルへ
                if rank is not None and 1 <= rank <= 3:
                    line_id, line_pos = line_info.get(sha_ban, (None, None))
                    results.append({
                        "race_id": race_id,
                        "rank": rank,
                        "sha_ban": sha_ban,
                        "senshu_name": row.get("senshu_name"),
                        "kimari_te": row.get("kimari_te"),
                        "line_id": line_id,
                        "line_pos": line_pos,
                        "agari_time": row.get("agari_time"),
                        "chakusa": row.get("chakusa"),
                    })

                # 全選手 entries テーブルへ
                entries.append({
                    "race_id": race_id,
                    "sha_ban": sha_ban,
                    "senshu_name": row.get("senshu_name"),
                    "age": row.get("age"),
                    "ki_betsu": row.get("ki_betsu"),
                    "todofuken": row.get("todofuken"),
                    # 以下は結果ページには無い（別ページ or 現状取得不可）
                    "kyoso_tokuten": None,
                    "gear_ratio": None,
                    "back_count": None,
                    "home_count": None,
                    "kyakushitsu": None,
                })

            if results:
                parsed_races.append({
                    "race_id": race_id,
                    "jyo_cd": int(venue_id),
                    "race_date": date_compact,
                    "race_no": race_no,
                    "results": results,
                    "entries": entries,
                    "line_info": line_info,
                })

        return parsed_races

    def _extract_race_tables(self, dfs):
        """
        pd.read_html() の結果リストから、各レースの着順テーブルを抽出する。

        着順テーブルは列に "着", "車番", "選手名" を持つ。

        Returns:
            list[list[dict]]: レース順に並んだ着順行リスト
        """
        race_list = []
        for df in dfs:
            try:
                cols = [str(c).strip() for c in df.columns]
            except Exception:
                continue

            has_rank = any("着" == c or c == "順位" for c in cols)
            has_sha_ban = any("車番" in c for c in cols)
            has_name = any("選手名" in c for c in cols)
            if not (has_rank and has_sha_ban and has_name):
                continue

            rank_col = self._find_col(cols, ["着", "順位"])
            sha_ban_col = self._find_col(cols, ["車番"])
            name_col = self._find_col(cols, ["選手名"])
            age_col = self._find_col(cols, ["年齢"])
            pref_col = self._find_col(cols, ["府県", "都道府県"])
            ki_col = self._find_col(cols, ["期別"])
            kimari_col = self._find_col(cols, ["決まり手", "決り手"])
            agari_col = self._find_col(cols, ["上り", "上がり"])
            chakusa_col = self._find_col(cols, ["着差"])

            rows = []
            for _, row in df.iterrows():
                rank = self._to_int(row.get(rank_col))
                sha_ban = self._to_int(row.get(sha_ban_col))
                if sha_ban is None:
                    continue
                rows.append({
                    "rank": rank,
                    "sha_ban": sha_ban,
                    "senshu_name": self._clean_str(row.get(name_col)),
                    "age": self._to_int(row.get(age_col)) if age_col else None,
                    "todofuken": self._clean_str(row.get(pref_col))
                                 if pref_col else None,
                    "ki_betsu": self._to_int(row.get(ki_col))
                                if ki_col else None,
                    "kimari_te": self._clean_str(row.get(kimari_col))
                                 if kimari_col else None,
                    "agari_time": self._to_float(row.get(agari_col))
                                  if agari_col else None,
                    "chakusa": self._clean_str(row.get(chakusa_col))
                               if chakusa_col else None,
                })

            if rows:
                race_list.append(rows)

        return race_list

    def _extract_line_info(self, soup):
        """
        周回予想（ライン情報）を抽出する。

        HTML構造:
          <tr>
            <th>周回予想</th>
            <td>
              <table>
                <tr>
                  <td><span class="square ... bg-1">1</span></td>
                  <td><span class="square ... bg-4">4</span></td>
                  ...
                  <td><span class="p10"></span></td>  ← ライン区切り（空セル）
                  <td><span class="square ... bg-3">3</span></td>
                </tr>
              </table>
            </td>
          </tr>

        Returns:
            list[dict]: レース順に並んだ {sha_ban: (line_id, line_pos)} 辞書
        """
        result = []

        # "周回予想" を含む tr を全て取得
        for th in soup.find_all("th"):
            if th.get_text(strip=True) != "周回予想":
                continue

            parent_tr = th.find_parent("tr")
            if parent_tr is None:
                continue

            # 隣の td 内の inner table の各 td を順に走査
            td = th.find_next_sibling("td")
            if td is None:
                continue

            cells = td.find_all("td")
            line_map = {}  # sha_ban → (line_id, line_pos)
            line_id = 1
            line_pos = 1
            last_was_empty = False

            for cell in cells:
                span = cell.find("span")
                if span is None:
                    continue
                classes = span.get("class", [])
                # 空セル = ライン区切り
                is_empty = "p10" in classes
                if is_empty:
                    # 連続する空セルは1回のライン区切り扱い
                    if not last_was_empty and line_pos > 1:
                        line_id += 1
                        line_pos = 1
                    last_was_empty = True
                    continue

                # 車番セル
                text = span.get_text(strip=True)
                try:
                    sha_ban = int(text)
                except ValueError:
                    continue

                line_map[sha_ban] = (line_id, line_pos)
                line_pos += 1
                last_was_empty = False

            result.append(line_map)

        return result

    def _find_col(self, cols, candidates):
        """カラム候補から一致するものを返す"""
        for cand in candidates:
            for c in cols:
                if cand in c:
                    return c
        return None

    def _to_int(self, val):
        """値をintに変換。失敗時はNone。"""
        if val is None or pd.isna(val):
            return None
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return None

    def _clean_str(self, val):
        """文字列を正規化。nanや空はNone。"""
        if val is None or pd.isna(val):
            return None
        s = str(val).strip().replace("\u3000", " ")
        if s in ("", "nan", "NaN", "None"):
            return None
        return s

    # =========================================================
    # DB保存
    # =========================================================

    def _save_to_db(self, races, results, entries):
        """レース・結果・出走表をDBに保存する"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect_db()
        cur = conn.cursor()

        for race in races:
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO races
                    (race_id, jyo_cd, race_date, race_no,
                     grade, stage, created_at)
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
                     line_id, line_pos, agari_time, chakusa, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r["race_id"],
                    r["rank"],
                    r["sha_ban"],
                    r.get("senshu_name"),
                    r.get("kimari_te"),
                    r.get("line_id"),
                    r.get("line_pos"),
                    r.get("agari_time"),
                    r.get("chakusa"),
                    now,
                ))
            except sqlite3.Error as e:
                logger.warning("results INSERT エラー (race_id=%s rank=%d): %s",
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
                logger.warning("entries INSERT エラー (race_id=%s sha_ban=%s): %s",
                               e["race_id"], e.get("sha_ban"), e_err)

        conn.commit()
        conn.close()

    # =========================================================
    # 期間指定でまとめて取得
    # =========================================================

    def scrape_range(self, start_date, end_date):
        """
        指定期間のレースデータをまとめて取得する。

        年別インデックス（?year=YYYY）から実在開催日のみを抽出してアクセス。
        """
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)

        if start_dt is None or end_dt is None:
            logger.error("日付のパース失敗: start=%s end=%s",
                         start_date, end_date)
            return

        logger.info("=== スクレイピング開始 ===")
        logger.info("期間: %s 〜 %s",
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"))
        logger.info("対象会場: %d場 (%s)",
                    len(self.jyo_cds),
                    ",".join(self.jyo_cds))
        logger.info("進捗ファイル: %s", self.progress_file.name)

        scraped_ids = self._get_scraped_race_ids()

        # 期間内の年を列挙
        years = set()
        current = start_dt
        while current <= end_dt:
            years.add(current.year)
            current += timedelta(days=32)
            current = current.replace(day=1)
        years = sorted(years)

        total_saved = 0

        # 会場 × 年 × 実在日付 のループ
        for jka_code in self.jyo_cds:
            venue_name = self._get_venue_name(jka_code)
            for year in years:
                try:
                    dates = self.get_race_dates_for_venue_year(jka_code, year)
                except Exception as e:
                    logger.error("年別インデックス取得失敗 %s/%d: %s",
                                 jka_code, year, e)
                    self._log_failed(f"index_{jka_code}_{year}", str(e))
                    continue

                # 期間内の日付だけ残す
                target_dates = [
                    d for d in dates
                    if start_dt <= datetime.strptime(d, "%Y-%m-%d") <= end_dt
                ]
                if not target_dates:
                    self._polite_sleep()
                    continue

                logger.info("[%s/%s] %d年: %d日分",
                            jka_code, venue_name, year, len(target_dates))

                for date_str in target_dates:
                    try:
                        parsed_races = self.scrape_race_result(
                            jka_code, date_str
                        )
                    except Exception as e:
                        logger.error("[%s %s] 取得失敗: %s",
                                     jka_code, date_str, e)
                        self._log_failed(f"{jka_code}_{date_str}", str(e))
                        self._polite_sleep()
                        continue

                    if not parsed_races:
                        self._polite_sleep()
                        continue

                    # 取得済みスキップ
                    day_races = []
                    day_results = []
                    day_entries = []
                    for pr in parsed_races:
                        if pr["race_id"] in scraped_ids:
                            continue
                        day_races.append({
                            "race_id": pr["race_id"],
                            "jyo_cd": pr["jyo_cd"],
                            "race_date": pr["race_date"],
                            "race_no": pr["race_no"],
                        })
                        day_results.extend(pr["results"])
                        day_entries.extend(pr["entries"])
                        scraped_ids.add(pr["race_id"])

                    if day_races:
                        self._save_to_db(day_races, day_results, day_entries)
                        total_saved += len(day_races)
                        logger.info("  [%s] 保存: %d レース, %d 結果, %d 出走",
                                    date_str, len(day_races),
                                    len(day_results), len(day_entries))

                    self._polite_sleep()

                # 年ごとの進捗保存
                self._save_progress(
                    f"{jka_code}_{year}",
                    end_dt.strftime("%Y-%m-%d"),
                )

        logger.info("=== スクレイピング完了: %d レース保存 ===", total_saved)
        self.get_status()

    def _get_venue_name(self, jka_code):
        """jka_code から会場名を取得"""
        for name, info in BANK_MASTER.items():
            if info.get("jka_code") == jka_code:
                return name
        return "unknown"

    def _parse_date(self, date_str):
        """日付文字列をdatetimeに変換"""
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _save_progress(self, last_marker, end_date):
        """進捗をファイルに保存"""
        with open(self.progress_file, "w", encoding="utf-8") as f:
            f.write(f"{last_marker}\n{end_date}\n")

    def _load_progress(self):
        """保存された進捗を読み込む"""
        if not self.progress_file.exists():
            return None, None
        try:
            lines = self.progress_file.read_text(
                encoding="utf-8"
            ).strip().split("\n")
            if len(lines) >= 2:
                return lines[0], lines[1]
        except Exception:
            pass
        return None, None

    # =========================================================
    # 進捗表示
    # =========================================================

    def get_status(self):
        """取得済み件数を表示"""
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
# Kドリームズ補完スクレイパー
# =========================================================

class KdreamsSupplementScraper:
    """
    keirin.kdreams.jp から選手統計を取得して
    chariloto の entries テーブルの None 項目を補完する。

    取得項目:
      - kyoso_tokuten（競走得点）
      - gear_ratio（ギア倍数）
      - back_count（B: バック回数）
      - start_count（S: スタート回数）
      - kyakushitsu（脚質: 逃/捲/差/追/自）

    注意:
      - home_count（H: ホーム回数）は kdreams 側のページに存在しないため
        補完対象外（None のまま）
      - robots.txt は 200 だが text/html を返す（= robots.txt 非設置）
        → Disallow なしと解釈してスクレイピング可
    """

    def __init__(self, db_path=None, delay=2.0):
        """
        Parameters:
            db_path: SQLiteファイルのパス
            delay: リクエスト間の待機秒数
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
                "先に ChariLotoScraper でデータを取得してください"
            )

        # jyo_cd(bank_master venue_id int) → jka_code マップ
        self.venue_id_to_jka = {}
        for info in BANK_MASTER.values():
            vid = info.get("venue_id")
            jka = info.get("jka_code")
            if vid and jka:
                self.venue_id_to_jka[int(vid)] = jka

        # start_count カラムが存在しなければ追加（マイグレーション）
        self._ensure_start_count_column()

    def _ensure_start_count_column(self):
        """entriesテーブルに start_count カラムがなければ追加する"""
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(entries)")
        cols = {row[1] for row in cur.fetchall()}
        if "start_count" not in cols:
            try:
                cur.execute(
                    "ALTER TABLE entries ADD COLUMN start_count INTEGER"
                )
                conn.commit()
                logger.info("entries テーブルに start_count カラムを追加")
            except sqlite3.OperationalError as e:
                logger.warning("start_count 追加失敗: %s", e)
        conn.close()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _polite_sleep(self):
        time.sleep(self.delay)

    def _fetch_html(self, url, max_retries=3):
        """URL からHTMLを取得。リトライ付き。"""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 404:
                    logger.debug("404 Not Found: %s", url)
                    return None
                if resp.status_code == 503:
                    logger.warning("503 SRV Unavailable: %s (%d/%d)",
                                   url, attempt, max_retries)
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                resp.encoding = "utf-8"
                return resp.text
            except requests.RequestException as e:
                logger.warning("リクエストエラー: %s (%d/%d): %s",
                               url, attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        return None

    def _log_failed(self, identifier, reason):
        """失敗情報を failed_kdreams.log に記録"""
        with open(FAILED_KDREAMS_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}\t{identifier}\t{reason}\n")

    # =========================================================
    # URL発見
    # =========================================================

    def _fetch_racecard_urls(self, date_str):
        """
        /racecard/{YYYY}/{MM}/{DD}/ から
        その日の全レース詳細 URL を取得する。

        Parameters:
            date_str: "YYYY-MM-DD" 形式

        Returns:
            list[dict]: [
                {
                  "venue_romaji": str,   # "matsusaka"
                  "jka_code":     str,   # "47"
                  "race_no":      int,
                  "url":          str,
                  "race_id_kd":   str,   # kdreams内部の16桁ID
                },
                ...
            ]
        """
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("日付パース失敗: %s", date_str)
            return []

        url = f"{KDREAMS_BASE_URL}/racecard/{dt:%Y}/{dt:%m}/{dt:%d}/"
        html = self._fetch_html(url)
        if html is None:
            return []

        # /{venue_romaji}/racedetail/{16桁race_id} を抽出
        pattern = re.compile(r"/([a-z_]+)/racedetail/(\d{16})")
        matches = sorted(set(pattern.findall(html)))

        results = []
        for venue_romaji, race_id_kd in matches:
            if venue_romaji in ("racecard", "sp", "pc", "css", "js",
                                "images", "img"):
                continue
            # race_id_kd = JKA(2) + 初日YYYYMMDD(8) + 日数(2) + race_no(4)
            jka = race_id_kd[:2]
            try:
                race_no = int(race_id_kd[-4:])
            except ValueError:
                continue
            results.append({
                "venue_romaji": venue_romaji,
                "jka_code": jka,
                "race_no": race_no,
                "url": f"{KDREAMS_BASE_URL}/{venue_romaji}"
                       f"/racedetail/{race_id_kd}",
                "race_id_kd": race_id_kd,
            })

        return results

    # =========================================================
    # レース詳細ページのパース
    # =========================================================

    def _parse_racedetail(self, url):
        """
        レース詳細ページから選手統計を取得する。

        Parameters:
            url: kdreams の racedetail URL

        Returns:
            dict: {sha_ban: {
                "senshu_name": str,
                "kyoso_tokuten": float or None,
                "gear_ratio": float or None,
                "back_count": int or None,
                "start_count": int or None,
                "kyakushitsu": str or None,
            }}
            取得失敗時は空dict。
        """
        html = self._fetch_html(url)
        if html is None:
            return {}

        try:
            dfs = pd.read_html(io.StringIO(html))
        except Exception as e:
            logger.debug("read_html 失敗 %s: %s", url, e)
            return {}

        if not dfs:
            return {}

        # table[0] を使う（最も情報量が多い）
        # 必要なら他テーブルもfallbackとして走査する
        for df in dfs[:3]:
            parsed = self._extract_player_stats(df)
            if parsed:
                return parsed

        return {}

    def _extract_player_stats(self, df):
        """
        DataFrame（MultiIndex列）から選手統計を抽出する。

        Returns:
            dict: {sha_ban: {...}}
        """
        if df.empty:
            return {}

        # MultiIndex 列を flatten（最下層のラベルを使う）
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = []
            for c in df.columns:
                last = c[-1]
                flat_cols.append(self._normalize_col_name(str(last)))
            df = df.copy()
            df.columns = flat_cols
        else:
            df = df.copy()
            df.columns = [self._normalize_col_name(str(c))
                          for c in df.columns]

        cols = list(df.columns)

        sha_ban_col = self._find_col(cols, ["車番"])
        name_col = self._find_col(cols, ["選手名"])
        kyakushitsu_col = self._find_col(cols, ["脚質"])
        gear_col = self._find_col(cols, ["ギヤ倍数", "ギア倍数", "ギヤ", "ギア"])
        tokuten_col = self._find_col(cols, ["競走得点"])
        s_col = self._find_col_exact(cols, ["S"])
        b_col = self._find_col_exact(cols, ["B"])

        if sha_ban_col is None or tokuten_col is None:
            return {}

        result = {}
        for _, row in df.iterrows():
            sha_ban = self._to_int(row.get(sha_ban_col))
            if sha_ban is None or sha_ban < 1 or sha_ban > 9:
                continue

            # 選手名のパース: "山元 大夢  石　川/24/123"
            senshu_name = None
            if name_col:
                raw = str(row.get(name_col, "")).strip()
                if raw and raw != "nan":
                    # "  " (2スペース) または "\u3000" で分割
                    parts = re.split(r"\s{2,}|\u3000{2,}|  ", raw, maxsplit=1)
                    senshu_name = parts[0].strip() if parts else None

            result[sha_ban] = {
                "senshu_name": senshu_name,
                "kyoso_tokuten": self._to_float(row.get(tokuten_col)),
                "gear_ratio": self._to_float(row.get(gear_col))
                              if gear_col else None,
                "back_count": self._to_int(row.get(b_col)) if b_col else None,
                "start_count": self._to_int(row.get(s_col)) if s_col else None,
                "kyakushitsu": self._clean_str(row.get(kyakushitsu_col))
                               if kyakushitsu_col else None,
            }

        return result

    def _normalize_col_name(self, name):
        """列名を正規化（全空白除去）"""
        return re.sub(r"\s+", "", name)

    def _find_col(self, cols, candidates):
        """候補の部分一致で列名を検索"""
        for cand in candidates:
            for c in cols:
                if cand in c:
                    return c
        return None

    def _find_col_exact(self, cols, candidates):
        """候補の完全一致で列名を検索（S,B の誤マッチ防止）"""
        for cand in candidates:
            for c in cols:
                if c == cand:
                    return c
        return None

    def _to_int(self, val):
        if val is None or pd.isna(val):
            return None
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return None

    def _to_float(self, val):
        if val is None or pd.isna(val):
            return None
        try:
            s = str(val).strip().replace(",", "")
            if s in ("", "nan", "-", "---"):
                return None
            return float(s)
        except (ValueError, TypeError):
            return None

    def _clean_str(self, val):
        if val is None or pd.isna(val):
            return None
        s = str(val).strip().replace("\u3000", "")
        s = re.sub(r"\s+", "", s)
        if s in ("", "nan", "None"):
            return None
        return s

    # =========================================================
    # マッチング用ヘルパ
    # =========================================================

    def _names_match(self, a, b):
        """
        選手名の簡易比較。
        全角半角スペース・空白類を全て除去して比較する。
        """
        if not a or not b:
            return False
        na = re.sub(r"[\s\u3000]+", "", str(a))
        nb = re.sub(r"[\s\u3000]+", "", str(b))
        return na == nb

    # =========================================================
    # 日付指定で補完
    # =========================================================

    def supplement_entries_for_date(self, date_str):
        """
        指定日のentriesテーブルで kyoso_tokuten が None の選手を
        Kドリームズから取得して補完する。

        Parameters:
            date_str: "YYYY-MM-DD" 形式
        """
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("日付パース失敗: %s", date_str)
            return
        date_compact = dt.strftime("%Y%m%d")

        # DB から該当日の補完対象を取得
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT e.race_id, e.sha_ban, e.senshu_name, r.jyo_cd
            FROM entries e
            JOIN races r ON e.race_id = r.race_id
            WHERE r.race_date = ? AND e.kyoso_tokuten IS NULL
            ORDER BY e.race_id, e.sha_ban
        """, (date_compact,))
        missing = cur.fetchall()
        conn.close()

        if not missing:
            logger.info("[%s] 補完対象なし", date_str)
            return

        # race_id 別にグルーピング
        missing_by_race = {}
        target_jkas = set()
        for race_id, sha_ban, senshu_name, jyo_cd in missing:
            jka = self.venue_id_to_jka.get(int(jyo_cd))
            if jka is None:
                logger.debug("jyo_cd=%s に対応する jka_code なし", jyo_cd)
                continue
            target_jkas.add(jka)
            missing_by_race.setdefault(race_id, []).append({
                "sha_ban": sha_ban,
                "senshu_name": senshu_name,
                "jka_code": jka,
            })

        logger.info("[%s] 補完対象: %d entries / %d races / %d venues",
                    date_str, len(missing), len(missing_by_race),
                    len(target_jkas))

        # kdreams から該当日のレース詳細URLを取得
        racecard_entries = self._fetch_racecard_urls(date_str)
        self._polite_sleep()

        if not racecard_entries:
            logger.warning("[%s] racecard 取得失敗", date_str)
            self._log_failed(f"racecard_{date_str}", "empty")
            return

        # (jka_code, race_no) → url マップ
        url_map = {}
        for e in racecard_entries:
            if e["jka_code"] not in target_jkas:
                continue
            key = (e["jka_code"], e["race_no"])
            url_map[key] = e["url"]

        logger.info("[%s] マッチする kdreams URL: %d",
                    date_str, len(url_map))

        # 同じ URL を複数選手で使い回すためキャッシュ
        url_cache = {}
        updates = []
        matched_races = 0
        unmatched_races = 0

        for race_id, misses in missing_by_race.items():
            # chariloto race_id: "{venue_id}_{YYYYMMDD}_{RR}"
            parts = race_id.split("_")
            if len(parts) != 3:
                continue
            try:
                race_no = int(parts[2])
            except ValueError:
                continue
            jka = misses[0]["jka_code"]

            url = url_map.get((jka, race_no))
            if url is None:
                unmatched_races += 1
                continue

            if url not in url_cache:
                url_cache[url] = self._parse_racedetail(url)
                self._polite_sleep()

            player_stats = url_cache[url]
            if not player_stats:
                self._log_failed(f"parse_{race_id}", "parse empty")
                continue

            matched_races += 1
            for miss in misses:
                sha_ban = miss["sha_ban"]
                target_name = miss["senshu_name"]

                stat = player_stats.get(sha_ban)
                if stat is None:
                    continue

                # 選手名で念のため確認（違っていてもログのみ、更新は実施）
                if target_name and stat.get("senshu_name"):
                    if not self._names_match(target_name, stat["senshu_name"]):
                        logger.debug(
                            "name mismatch %s: chariloto=%r kdreams=%r",
                            race_id, target_name, stat["senshu_name"],
                        )

                updates.append({
                    "race_id": race_id,
                    "sha_ban": sha_ban,
                    "kyoso_tokuten": stat.get("kyoso_tokuten"),
                    "gear_ratio": stat.get("gear_ratio"),
                    "back_count": stat.get("back_count"),
                    "start_count": stat.get("start_count"),
                    "kyakushitsu": stat.get("kyakushitsu"),
                })

        # 一括 UPDATE（kyoso_tokuten が NULL のもののみ更新）
        if updates:
            self._apply_updates(updates)

        logger.info(
            "[%s] 完了: %d updates / matched_races=%d unmatched_races=%d",
            date_str, len(updates), matched_races, unmatched_races,
        )

    def _apply_updates(self, updates):
        """
        entries テーブルに一括更新をかける。
        kyoso_tokuten が NULL のレコードのみ更新（上書き防止）。
        """
        conn = self._connect_db()
        cur = conn.cursor()

        for u in updates:
            try:
                cur.execute("""
                    UPDATE entries
                    SET kyoso_tokuten = COALESCE(?, kyoso_tokuten),
                        gear_ratio    = COALESCE(?, gear_ratio),
                        back_count    = COALESCE(?, back_count),
                        start_count   = COALESCE(?, start_count),
                        kyakushitsu   = COALESCE(?, kyakushitsu)
                    WHERE race_id = ?
                      AND sha_ban = ?
                      AND kyoso_tokuten IS NULL
                """, (
                    u["kyoso_tokuten"],
                    u["gear_ratio"],
                    u["back_count"],
                    u["start_count"],
                    u["kyakushitsu"],
                    u["race_id"],
                    u["sha_ban"],
                ))
            except sqlite3.Error as e:
                logger.warning(
                    "UPDATE エラー race_id=%s sha_ban=%s: %s",
                    u["race_id"], u["sha_ban"], e,
                )

        conn.commit()
        conn.close()

    # =========================================================
    # DB全体の未補完を一括処理
    # =========================================================

    def supplement_missing_all(self):
        """
        DB の entries テーブル全体を走査して
        kyoso_tokuten が None のレコードを日付別に補完する。
        """
        self._supplement_missing_dates_between(None, None)

    def supplement_missing_range(self, start_date, end_date):
        """
        指定期間の未補完日付だけを処理する（並列実行用）。

        Parameters:
            start_date: "YYYY-MM-DD" 形式
            end_date:   "YYYY-MM-DD" 形式
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        self._supplement_missing_dates_between(
            start_dt.strftime("%Y%m%d"),
            end_dt.strftime("%Y%m%d"),
        )

    def _supplement_missing_dates_between(self, start_compact, end_compact,
                                          max_retries=3, retry_delay=120):
        """
        未補完日付リストを取得して順次補完する内部実装。
        1巡目の完了後、まだ未補完が残っていれば自動リトライする。

        Parameters:
            start_compact: "YYYYMMDD" 形式 or None
            end_compact:   "YYYYMMDD" 形式 or None
            （None の場合は全期間）
            max_retries: 自動リトライ回数（デフォルト3回）
            retry_delay: リトライ間の待機秒数（デフォルト120秒、毎回2倍に増加）
        """
        for attempt in range(1 + max_retries):
            dates = self._get_null_dates(start_compact, end_compact)

            if not dates:
                if attempt == 0:
                    logger.info("補完対象の日付なし")
                else:
                    logger.info("リトライ %d: 未補完なし → 完了", attempt)
                return

            # リトライ回の場合はログと待機
            if attempt > 0:
                wait = retry_delay * (2 ** (attempt - 1))
                logger.info(
                    "=== リトライ %d/%d: %d日分が未補完 → %d秒待機後に再試行 ===",
                    attempt, max_retries, len(dates), wait,
                )
                time.sleep(wait)

            range_label = ""
            if start_compact or end_compact:
                range_label = (
                    f" (range {start_compact or '-'}〜{end_compact or '-'})"
                )
            label = f"[pass {attempt + 1}] " if attempt > 0 else ""
            logger.info("%s補完対象日数: %d%s", label, len(dates), range_label)

            for i, date_compact in enumerate(dates, 1):
                try:
                    dt = datetime.strptime(date_compact, "%Y%m%d")
                except ValueError:
                    continue
                date_str = dt.strftime("%Y-%m-%d")
                logger.info("[%d/%d] %s の補完", i, len(dates), date_str)
                try:
                    self.supplement_entries_for_date(date_str)
                except Exception as e:
                    logger.error("[%s] 補完エラー: %s", date_str, e)
                    self._log_failed(f"supplement_{date_str}", str(e))

        # 最終チェック
        remaining = self._get_null_dates(start_compact, end_compact)
        if remaining:
            logger.warning(
                "リトライ %d回後も %d日分が未補完のまま残っています",
                max_retries, len(remaining),
            )

    def _get_null_dates(self, start_compact, end_compact):
        """指定範囲内の kyoso_tokuten IS NULL の日付一覧を返す"""
        conn = self._connect_db()
        cur = conn.cursor()
        sql = """
            SELECT DISTINCT r.race_date
            FROM entries e
            JOIN races r ON e.race_id = r.race_id
            WHERE e.kyoso_tokuten IS NULL
        """
        params = []
        if start_compact is not None:
            sql += " AND r.race_date >= ?"
            params.append(start_compact)
        if end_compact is not None:
            sql += " AND r.race_date <= ?"
            params.append(end_compact)
        sql += " ORDER BY r.race_date"
        cur.execute(sql, params)
        dates = [row[0] for row in cur.fetchall()]
        conn.close()
        return dates


    # =========================================================
    # 上がりタイム・着差のバックフィル
    # =========================================================

    def backfill_agari(self, start_date, end_date):
        """
        agari_time が NULL の race_id を再取得して
        agari_time と chakusa を埋める。

        既存の race_id を chariloto から再取得し、
        results テーブルを UPDATE する。
        """
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)
        if start_dt is None or end_dt is None:
            logger.error("日付パース失敗: %s, %s", start_date, end_date)
            return

        conn = self._connect_db()
        cur = conn.cursor()

        # agari_time が NULL のレース日付+会場を取得
        cur.execute("""
            SELECT DISTINCT rc.race_date, rc.jyo_cd
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.agari_time IS NULL
              AND rc.race_date >= ?
              AND rc.race_date <= ?
            ORDER BY rc.race_date
        """, (start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")))
        targets = cur.fetchall()
        conn.close()

        if not targets:
            logger.info("backfill_agari: 対象なし")
            return

        logger.info("backfill_agari: %d 会場日を処理", len(targets))

        # venue_id → jka_code マップ
        vid_to_jka = {}
        for info in BANK_MASTER.values():
            vid = info.get("venue_id")
            jka = info.get("jka_code")
            if vid and jka:
                vid_to_jka[int(vid)] = jka

        updated_total = 0
        for i, (race_date, jyo_cd) in enumerate(targets, 1):
            jka = vid_to_jka.get(int(jyo_cd))
            if jka is None:
                continue

            try:
                dt = datetime.strptime(race_date, "%Y%m%d")
            except ValueError:
                continue
            date_str = dt.strftime("%Y-%m-%d")

            logger.info("[%d/%d] %s jyo=%s → jka=%s",
                        i, len(targets), date_str, jyo_cd, jka)

            try:
                parsed_races = self.scrape_race_result(jka, date_str)
            except Exception as e:
                logger.error("取得失敗 %s/%s: %s", jka, date_str, e)
                self._polite_sleep()
                continue

            if not parsed_races:
                self._polite_sleep()
                continue

            # UPDATE（agari_time / chakusa のみ）
            conn = self._connect_db()
            cur = conn.cursor()
            n_updated = 0
            for pr in parsed_races:
                for r in pr["results"]:
                    if r.get("agari_time") is not None:
                        cur.execute("""
                            UPDATE results
                            SET agari_time = ?, chakusa = ?
                            WHERE race_id = ? AND rank = ?
                              AND agari_time IS NULL
                        """, (
                            r["agari_time"],
                            r.get("chakusa"),
                            r["race_id"],
                            r["rank"],
                        ))
                        n_updated += cur.rowcount
            conn.commit()
            conn.close()

            if n_updated > 0:
                updated_total += n_updated
                logger.info("  → %d 件更新", n_updated)

            self._polite_sleep()

        logger.info("backfill_agari 完了: %d 件更新", updated_total)


# =========================================================
# コマンドラインインターフェース
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="競輪過去データスクレイパー（chariloto.com）"
    )
    parser.add_argument("--start", type=str, help="開始日（YYYY-MM-DD）")
    parser.add_argument("--end", type=str, help="終了日（YYYY-MM-DD）")
    parser.add_argument("--resume", action="store_true",
                        help="前回の続きから再開")
    parser.add_argument("--status", action="store_true",
                        help="取得済み件数を表示")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="リクエスト間隔（秒、デフォルト: 2.0）")
    parser.add_argument("--db", type=str, default=None,
                        help="DBファイルパス")
    parser.add_argument("--jyo_cds", type=str, default=None,
                        help="対象JKAコード（カンマ区切り、例: 47,81）"
                             "省略時は全43会場")
    parser.add_argument("--supplement", type=str, default=None,
                        help="Kドリームズ補完を指定日(YYYY-MM-DD)で実行")
    parser.add_argument("--supplement_all", action="store_true",
                        help="Kドリームズ補完をDB全体の未補完分に実行")
    parser.add_argument("--supplement_range", type=str, default=None,
                        help="Kドリームズ補完を指定期間の未補完分に実行"
                             "（例: 2022-04-20,2023-08-26）")
    parser.add_argument("--backfill_agari", action="store_true",
                        help="agari_timeがNULLのレースの上がりタイムを再取得")

    args = parser.parse_args()

    jyo_cds = None
    if args.jyo_cds:
        try:
            jyo_cds = [int(c.strip()) for c in args.jyo_cds.split(",")
                       if c.strip()]
            if not jyo_cds:
                raise ValueError("会場コードが空")
        except ValueError as e:
            logger.error("--jyo_cds の形式が不正: %s", e)
            return

    # 上がりタイムバックフィル
    if args.backfill_agari:
        if not args.start or not args.end:
            logger.error("--backfill_agari には --start と --end が必要")
            return
        scraper = ChariLotoScraper(db_path=args.db, delay=args.delay)
        scraper.backfill_agari(args.start, args.end)
        return

    # Kドリームズ補完モード
    if args.supplement or args.supplement_all or args.supplement_range:
        try:
            supp = KdreamsSupplementScraper(
                db_path=args.db,
                delay=args.delay,
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            return
        if args.supplement_all:
            supp.supplement_missing_all()
        elif args.supplement_range:
            parts = args.supplement_range.split(",")
            if len(parts) != 2:
                logger.error("--supplement_range は 'START,END' 形式で指定")
                return
            try:
                supp.supplement_missing_range(parts[0].strip(), parts[1].strip())
            except ValueError as e:
                logger.error("--supplement_range 日付パース失敗: %s", e)
                return
        else:
            supp.supplement_entries_for_date(args.supplement)
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
        last_marker, end_date = scraper._load_progress()
        if last_marker is None or end_date is None:
            if args.start and args.end:
                logger.info("進捗なし。--start/--end で新規実行")
                scraper.scrape_range(args.start, args.end)
                return
            logger.error("再開用の進捗ファイルなし: %s", scraper.progress_file)
            return
        logger.info("前回の続きから再開: marker=%s end=%s",
                    last_marker, end_date)
        # 新規実装: 進捗は「年別インデックスで取得済みの会場×年」の最終位置
        # を保持するが、シンプルに最初から再実行して取得済みスキップに任せる
        if args.start:
            scraper.scrape_range(args.start, end_date)
        else:
            logger.error("--start が必要（年の開始を指定してください）")
        return

    if args.start and args.end:
        scraper.scrape_range(args.start, args.end)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
