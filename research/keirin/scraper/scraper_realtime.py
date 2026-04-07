# ===========================================
# scraper/scraper_realtime.py
# 競輪 - リアルタイムオッズスクレイパー（Kドリームズ）
#
# 対象：keirin.kdreams.jp（公開ページ・ログイン不要）
# 取得項目：オッズのみ（3連単・2車単）
#
# レース結果・選手情報はcharilotoで取得済みのため不要。
# chariloto DBのracesテーブルから当日レースを特定して
# オッズスナップショットを取得する。
#
# robots.txt確認（2026-04-08）:
#   404（ファイルなし）→ Disallow なし
#
# 取得タイミング設計：
#   DEFAULT_SNAPSHOT_MINUTES は仮の値。
#   データが溜まった後に統計的に最適値を検証して変更する。
#   変更時は DEFAULT_SNAPSHOT_MINUTES の1行を直すだけ。
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
FAILED_LOG = Path(__file__).resolve().parent / "failed_realtime.log"

BASE_URL = "https://keirin.kdreams.jp"

# 締切何分前にオッズを取得するか（仮の値・後で統計的に最適化）
# 変更時はここだけ直す。例：[45, 20, 5, 0]
DEFAULT_SNAPSHOT_MINUTES = [60, 30, 10, 0]


class RealtimeScraper:
    """keirin.kdreams.jp から競輪オッズを取得する（リアルタイム専用）"""

    def __init__(self, db_path=None, delay=2.0, snapshot_minutes=None):
        """
        Parameters:
            db_path: SQLiteファイルのパス（デフォルト: data/keirin/keirin.db）
            delay: リクエスト間の待機秒数
            snapshot_minutes: 締切何分前に取得するかのリスト
                              Noneの場合はDEFAULT_SNAPSHOT_MINUTESを使う
                              例：[45, 20, 5, 0] に変更するだけで動く設計
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.delay = delay
        self.snapshot_minutes = snapshot_minutes or DEFAULT_SNAPSHOT_MINUTES
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KeirinDataCollector/1.0 (personal research use)",
            "Accept-Language": "ja,en;q=0.9",
        })
        self._init_db()

    def _init_db(self):
        """odds_history テーブルを初期化する"""
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
            CREATE TABLE IF NOT EXISTS odds_history (
                race_id        TEXT NOT NULL,
                odds_type      TEXT NOT NULL,
                snapshot_time  TEXT NOT NULL,
                minutes_before INTEGER NOT NULL,
                sha_ban_1      INTEGER NOT NULL,
                sha_ban_2      INTEGER NOT NULL,
                sha_ban_3      INTEGER,
                odds           REAL,
                created_at     TEXT NOT NULL,
                PRIMARY KEY (race_id, odds_type, minutes_before,
                             sha_ban_1, sha_ban_2, sha_ban_3),
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            );

            CREATE INDEX IF NOT EXISTS idx_odds_history_race
                ON odds_history(race_id);
            CREATE INDEX IF NOT EXISTS idx_odds_history_snapshot
                ON odds_history(race_id, odds_type, minutes_before);
        """)
        conn.commit()
        conn.close()
        logger.info("odds_history テーブル初期化完了: %s", self.db_path)

    # =========================================================
    # HTTP / ユーティリティ
    # =========================================================

    def _polite_sleep(self):
        """サーバー負荷を考慮した待機"""
        time.sleep(self.delay)

    def _fetch_html(self, url, max_retries=3):
        """
        URLからHTMLを取得する。リトライ付き。

        Returns:
            str: HTMLテキスト。取得失敗時は None。
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

    def _log_failed(self, identifier, reason):
        """失敗した取得をログに記録する"""
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}\t{identifier}\t{reason}\n")

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
        if val is None:
            return None
        try:
            s = str(val).strip().replace(",", "")
            if s in ("", "nan", "-", "---", "取消"):
                return None
            return float(s)
        except (ValueError, TypeError):
            return None

    # =========================================================
    # chariloto DB からの当日レース取得
    # =========================================================

    def _get_today_races(self):
        """
        当日のレース一覧をchariloto DBから取得する。

        Returns:
            list[dict]: [{"race_id": str, "jyo_cd": int, "race_no": int}]
        """
        today = datetime.now().strftime("%Y%m%d")
        return self._get_races_for_date(today)

    def _get_races_for_date(self, date_str):
        """
        指定日のレース一覧を取得する。

        Parameters:
            date_str: "YYYYMMDD" 形式

        Returns:
            list[dict]
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

    def _get_existing_snapshots(self, race_id, odds_type):
        """取得済みの minutes_before セットを返す"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT minutes_before FROM odds_history "
            "WHERE race_id = ? AND odds_type = ?",
            (race_id, odds_type)
        )
        mins = {r[0] for r in cur.fetchall()}
        conn.close()
        return mins

    # =========================================================
    # オッズスナップショットの取得
    # =========================================================

    def scrape_odds_snapshot(self, jyo_cd, date_str, race_no, minutes_before):
        """
        1レース分のオッズスナップショットを取得する。
        3連単と2車単の両方を取得。

        Parameters:
            jyo_cd: 会場コード（int）
            date_str: "YYYYMMDD" 形式
            race_no: レース番号（int）
            minutes_before: 締切何分前か

        Returns:
            dict: {"3t": list[dict], "2t": list[dict]}
        """
        jyo_cd_str = f"{jyo_cd:02d}" if isinstance(jyo_cd, int) else str(jyo_cd)
        race_id = f"{jyo_cd_str}_{date_str}_{race_no:02d}"
        snapshot_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        result = {"3t": [], "2t": []}

        # 3連単オッズ
        url_3t = (f"{BASE_URL}/race/odds/3t/"
                  f"{jyo_cd_str}/{date_str}/{race_no}/")
        odds_3t = self._fetch_and_parse_odds(
            url_3t, race_id, "3t", snapshot_time, minutes_before
        )
        result["3t"] = odds_3t
        self._polite_sleep()

        # 2車単オッズ
        url_2t = (f"{BASE_URL}/race/odds/2t/"
                  f"{jyo_cd_str}/{date_str}/{race_no}/")
        odds_2t = self._fetch_and_parse_odds(
            url_2t, race_id, "2t", snapshot_time, minutes_before
        )
        result["2t"] = odds_2t

        # DB保存
        all_odds = odds_3t + odds_2t
        if all_odds:
            self._save_odds(all_odds)
            logger.info("[%s] %d分前: 3連単 %d件, 2車単 %d件",
                        race_id, minutes_before,
                        len(odds_3t), len(odds_2t))
        else:
            logger.debug("[%s] %d分前: オッズ取得0件", race_id, minutes_before)
            self._log_failed(
                f"{race_id}_m{minutes_before}",
                "オッズ取得0件"
            )

        return result

    def _fetch_and_parse_odds(self, url, race_id, odds_type,
                              snapshot_time, minutes_before):
        """
        オッズページを取得・パースする。

        Returns:
            list[dict]: オッズデータのリスト
        """
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

        odds_list = []
        for df in dfs:
            parsed = self._parse_odds_table(
                df, race_id, odds_type, snapshot_time, minutes_before
            )
            odds_list.extend(parsed)

        return odds_list

    def _parse_odds_table(self, df, race_id, odds_type,
                          snapshot_time, minutes_before):
        """
        オッズテーブルのDataFrameをパースする。

        3連単: 1着-2着-3着 + オッズ
        2車単: 1着-2着 + オッズ

        Returns:
            list[dict]
        """
        if df.empty:
            return []

        # カラム名正規化
        df.columns = [str(c).strip().replace("\n", "").replace(" ", "")
                      for c in df.columns]

        results = []

        # パターン1: 「組合せ」+「オッズ」カラム
        combo_col = self._find_column(df, ["組合せ", "組み合わせ", "買目",
                                            "組合", "車番"])
        odds_col = self._find_column(df, ["オッズ", "払戻", "倍率"])

        if combo_col and odds_col:
            for _, row in df.iterrows():
                combo_str = str(row[combo_col]).strip()
                odds_val = self._to_float(row[odds_col])
                if odds_val is None:
                    continue

                nums = self._parse_combo(combo_str)
                if nums is None:
                    continue

                if odds_type == "3t" and len(nums) == 3:
                    results.append({
                        "race_id": race_id,
                        "odds_type": odds_type,
                        "snapshot_time": snapshot_time,
                        "minutes_before": minutes_before,
                        "sha_ban_1": nums[0],
                        "sha_ban_2": nums[1],
                        "sha_ban_3": nums[2],
                        "odds": odds_val,
                    })
                elif odds_type == "2t" and len(nums) >= 2:
                    results.append({
                        "race_id": race_id,
                        "odds_type": odds_type,
                        "snapshot_time": snapshot_time,
                        "minutes_before": minutes_before,
                        "sha_ban_1": nums[0],
                        "sha_ban_2": nums[1],
                        "sha_ban_3": None,
                        "odds": odds_val,
                    })
            return results

        # パターン2: 1着・2着・3着が別カラム
        col_1 = self._find_column(df, ["1着", "一着", "1st"])
        col_2 = self._find_column(df, ["2着", "二着", "2nd"])
        col_3 = self._find_column(df, ["3着", "三着", "3rd"])

        if col_1 and col_2 and odds_col:
            for _, row in df.iterrows():
                s1 = self._to_int(row.get(col_1))
                s2 = self._to_int(row.get(col_2))
                odds_val = self._to_float(row.get(odds_col))
                if s1 is None or s2 is None or odds_val is None:
                    continue

                s3 = self._to_int(row.get(col_3)) if col_3 else None

                if odds_type == "3t" and s3 is not None:
                    results.append({
                        "race_id": race_id,
                        "odds_type": odds_type,
                        "snapshot_time": snapshot_time,
                        "minutes_before": minutes_before,
                        "sha_ban_1": s1,
                        "sha_ban_2": s2,
                        "sha_ban_3": s3,
                        "odds": odds_val,
                    })
                elif odds_type == "2t":
                    results.append({
                        "race_id": race_id,
                        "odds_type": odds_type,
                        "snapshot_time": snapshot_time,
                        "minutes_before": minutes_before,
                        "sha_ban_1": s1,
                        "sha_ban_2": s2,
                        "sha_ban_3": None,
                        "odds": odds_val,
                    })
            return results

        # パターン3: 全カラムを走査して数値+オッズのパターンを探す
        for _, row in df.iterrows():
            row_text = " ".join(str(v) for v in row.values)
            nums_in_row = re.findall(r"\d+", row_text)
            floats_in_row = re.findall(r"\d+\.\d+", row_text)

            if not floats_in_row:
                continue

            odds_val = self._to_float(floats_in_row[-1])
            if odds_val is None or odds_val <= 0:
                continue

            if odds_type == "3t" and len(nums_in_row) >= 4:
                s1 = self._to_int(nums_in_row[0])
                s2 = self._to_int(nums_in_row[1])
                s3 = self._to_int(nums_in_row[2])
                if all(1 <= x <= 9 for x in [s1, s2, s3] if x):
                    results.append({
                        "race_id": race_id,
                        "odds_type": odds_type,
                        "snapshot_time": snapshot_time,
                        "minutes_before": minutes_before,
                        "sha_ban_1": s1,
                        "sha_ban_2": s2,
                        "sha_ban_3": s3,
                        "odds": odds_val,
                    })
            elif odds_type == "2t" and len(nums_in_row) >= 3:
                s1 = self._to_int(nums_in_row[0])
                s2 = self._to_int(nums_in_row[1])
                if all(1 <= x <= 9 for x in [s1, s2] if x):
                    results.append({
                        "race_id": race_id,
                        "odds_type": odds_type,
                        "snapshot_time": snapshot_time,
                        "minutes_before": minutes_before,
                        "sha_ban_1": s1,
                        "sha_ban_2": s2,
                        "sha_ban_3": None,
                        "odds": odds_val,
                    })

        return results

    def _parse_combo(self, combo_str):
        """
        組合せ文字列を車番リストに変換する。

        "1-2-3" → [1, 2, 3]
        "1=2=3" → [1, 2, 3]
        "1→2→3" → [1, 2, 3]

        Returns:
            list[int] or None
        """
        # 区切り文字を統一して分割
        nums = re.split(r"[-=→\-ー＝]", combo_str)
        result = []
        for n in nums:
            n = n.strip()
            if not n:
                continue
            try:
                val = int(n)
                if 1 <= val <= 9:
                    result.append(val)
            except ValueError:
                continue
        return result if len(result) >= 2 else None

    def _find_column(self, df, candidates):
        """候補のカラム名からDataFrame内に存在するものを返す"""
        for c in candidates:
            for col in df.columns:
                if c in str(col):
                    return col
        return None

    # =========================================================
    # DB保存
    # =========================================================

    def _save_odds(self, odds_list):
        """
        オッズデータをDBに保存する。

        Parameters:
            odds_list: list[dict]
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        for o in odds_list:
            try:
                cur.execute("""
                    INSERT OR REPLACE INTO odds_history
                    (race_id, odds_type, snapshot_time, minutes_before,
                     sha_ban_1, sha_ban_2, sha_ban_3, odds, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    o["race_id"],
                    o["odds_type"],
                    o["snapshot_time"],
                    o["minutes_before"],
                    o["sha_ban_1"],
                    o["sha_ban_2"],
                    o.get("sha_ban_3"),
                    o["odds"],
                    now,
                ))
            except sqlite3.Error as e:
                logger.warning("odds INSERT エラー (race_id=%s): %s",
                               o["race_id"], e)

        conn.commit()
        conn.close()

    # =========================================================
    # 当日全レースのオッズ取得
    # =========================================================

    def scrape_all_active_races(self):
        """
        当日開催中の全レースを対象にオッズスナップショットを取得する。
        chariloto DBのracesテーブルから当日レースを特定。
        """
        races = self._get_today_races()
        if not races:
            logger.info("当日のレースが DB に見つかりません")
            return

        logger.info("=== 当日オッズ取得開始: %d レース ===", len(races))

        for minutes_before in self.snapshot_minutes:
            logger.info("--- %d分前スナップショット ---", minutes_before)

            for race in races:
                # 取得済みスキップ
                existing_3t = self._get_existing_snapshots(
                    race["race_id"], "3t"
                )
                existing_2t = self._get_existing_snapshots(
                    race["race_id"], "2t"
                )
                if (minutes_before in existing_3t
                        and minutes_before in existing_2t):
                    logger.debug("[SKIP] %s %d分前は取得済み",
                                 race["race_id"], minutes_before)
                    continue

                try:
                    self.scrape_odds_snapshot(
                        race["jyo_cd"],
                        race["race_id"].split("_")[1],  # date_str
                        race["race_no"],
                        minutes_before,
                    )
                except Exception as e:
                    logger.error("[%s] オッズ取得エラー: %s", race["race_id"], e)
                    self._log_failed(
                        f"{race['race_id']}_m{minutes_before}",
                        str(e)
                    )

                self._polite_sleep()

        logger.info("=== 当日オッズ取得完了 ===")
        self.get_status()

    # =========================================================
    # オッズ急変検出
    # =========================================================

    def detect_odds_surge(self, race_id, threshold=0.3):
        """
        直前2スナップショット間のオッズ変動を検出する。

        snapshot_minutes[-2]（最後から2番目）と
        snapshot_minutes[-1]（最終）を比較。
        上位人気（3連単10倍未満）が threshold 以上変動したら True。

        将来的に threshold も最適化対象にする。

        Parameters:
            race_id: レースID
            threshold: 変動率の閾値（デフォルト: 0.3 = 30%）

        Returns:
            True  → 急変あり（予測可能性スコアを40%カット推奨）
            False → 正常範囲
        """
        if len(self.snapshot_minutes) < 2:
            return False

        prev_min = self.snapshot_minutes[-2]
        curr_min = self.snapshot_minutes[-1]

        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # 前回スナップショットの上位人気（3連単10倍未満）
        cur.execute("""
            SELECT sha_ban_1, sha_ban_2, sha_ban_3, odds
            FROM odds_history
            WHERE race_id = ? AND odds_type = '3t'
              AND minutes_before = ? AND odds < 10.0
        """, (race_id, prev_min))
        prev_odds = {
            (r[0], r[1], r[2]): r[3] for r in cur.fetchall()
        }

        if not prev_odds:
            conn.close()
            return False

        # 最新スナップショット
        cur.execute("""
            SELECT sha_ban_1, sha_ban_2, sha_ban_3, odds
            FROM odds_history
            WHERE race_id = ? AND odds_type = '3t'
              AND minutes_before = ?
        """, (race_id, curr_min))
        curr_odds = {
            (r[0], r[1], r[2]): r[3] for r in cur.fetchall()
        }
        conn.close()

        # 変動率チェック
        for combo, prev_val in prev_odds.items():
            curr_val = curr_odds.get(combo)
            if curr_val is None or prev_val <= 0:
                continue

            change_rate = abs(curr_val - prev_val) / prev_val
            if change_rate >= threshold:
                logger.warning(
                    "[SURGE] %s: %s %.1f→%.1f (%.0f%%変動)",
                    race_id,
                    "-".join(str(x) for x in combo if x),
                    prev_val, curr_val,
                    change_rate * 100
                )
                return True

        return False

    # =========================================================
    # 進捗表示
    # =========================================================

    def get_status(self):
        """取得済み件数・直近のオッズ急変レース一覧を表示する"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # オッズ総件数
        cur.execute("SELECT COUNT(*) FROM odds_history")
        total_odds = cur.fetchone()[0]

        # タイプ別件数
        cur.execute("""
            SELECT odds_type, COUNT(*) FROM odds_history
            GROUP BY odds_type
        """)
        type_counts = {r[0]: r[1] for r in cur.fetchall()}

        # スナップショット別件数
        cur.execute("""
            SELECT minutes_before, COUNT(DISTINCT race_id)
            FROM odds_history
            GROUP BY minutes_before
            ORDER BY minutes_before DESC
        """)
        snapshot_counts = cur.fetchall()

        # レース数
        cur.execute("SELECT COUNT(DISTINCT race_id) FROM odds_history")
        race_count = cur.fetchone()[0]

        # 日付範囲
        cur.execute("""
            SELECT MIN(snapshot_time), MAX(snapshot_time)
            FROM odds_history
        """)
        time_range = cur.fetchone()

        conn.close()

        print("\n========== オッズスクレイピング進捗 ==========")
        print(f"  DB: {self.db_path}")
        print(f"  オッズ総件数:     {total_odds:,}")
        print(f"  3連単:            {type_counts.get('3t', 0):,}")
        print(f"  2車単:            {type_counts.get('2t', 0):,}")
        print(f"  対象レース数:     {race_count:,}")
        if time_range[0]:
            print(f"  取得期間:         {time_range[0]} 〜 {time_range[1]}")
        print(f"  ---")
        print(f"  スナップショット設定: {self.snapshot_minutes}")
        for mins, cnt in snapshot_counts:
            print(f"    {mins}分前: {cnt} レース")
        print("================================================")

        # オッズ急変レースの検出
        self._show_surge_races()

        print()
        return {
            "total_odds": total_odds,
            "type_counts": type_counts,
            "race_count": race_count,
        }

    def _show_surge_races(self):
        """直近のオッズ急変レースを表示する"""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # 当日のレースIDを取得
        today = datetime.now().strftime("%Y%m%d")
        cur.execute(
            "SELECT DISTINCT race_id FROM odds_history "
            "WHERE race_id LIKE ?",
            (f"%_{today}_%",)
        )
        today_race_ids = [r[0] for r in cur.fetchall()]
        conn.close()

        if not today_race_ids:
            return

        surge_races = []
        for race_id in today_race_ids:
            if self.detect_odds_surge(race_id):
                surge_races.append(race_id)

        if surge_races:
            print(f"  ---")
            print(f"  [警告] オッズ急変レース: {len(surge_races)} 件")
            for rid in surge_races:
                print(f"    - {rid}")


# =========================================================
# コマンドラインインターフェース
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="競輪リアルタイムオッズスクレイパー（Kドリームズ）"
    )
    parser.add_argument("--today", action="store_true",
                        help="当日の全レースのオッズを自動取得")
    parser.add_argument("--date", type=str,
                        help="指定日のレースのオッズを取得（YYYYMMDD）")
    parser.add_argument("--status", action="store_true",
                        help="取得済み件数・直近のオッズ急変レース一覧")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="リクエスト間隔（秒、デフォルト: 2.0）")
    parser.add_argument("--db", type=str, default=None,
                        help="DBファイルパス")
    parser.add_argument("--snapshots", type=str, default=None,
                        help="スナップショット分数（カンマ区切り、例: 45,20,5,0）")

    args = parser.parse_args()

    # スナップショット分数のパース
    snapshot_minutes = None
    if args.snapshots:
        try:
            snapshot_minutes = [int(x.strip()) for x in args.snapshots.split(",")]
        except ValueError:
            logger.error("--snapshots の形式が不正です（例: 60,30,10,0）")
            return

    try:
        scraper = RealtimeScraper(
            db_path=args.db,
            delay=args.delay,
            snapshot_minutes=snapshot_minutes,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if args.status:
        scraper.get_status()
        return

    if args.today:
        scraper.scrape_all_active_races()
        return

    if args.date:
        races = scraper._get_races_for_date(args.date)
        if not races:
            logger.info("指定日のレースが DB に見つかりません: %s", args.date)
            return

        logger.info("=== %s のオッズ取得: %d レース ===", args.date, len(races))
        for minutes_before in scraper.snapshot_minutes:
            logger.info("--- %d分前スナップショット ---", minutes_before)
            for race in races:
                try:
                    scraper.scrape_odds_snapshot(
                        race["jyo_cd"], args.date,
                        race["race_no"], minutes_before,
                    )
                except Exception as e:
                    logger.error("[%s] エラー: %s", race["race_id"], e)
                scraper._polite_sleep()

        scraper.get_status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
