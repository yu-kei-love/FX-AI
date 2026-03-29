# ===========================================
# scraper/scraper_historical.py
# 競輪 - 過去データスクレイパー
#
# 対象：keirin-station.com / チャリロト / 公式サイト
# 保存先：SQLite（data/keirin/keirin.db）
#
# 注意：このファイルはスクレイピングコードのため
#       note販売パッケージには含めない
# ===========================================

import time
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

# スクレイピング間隔（サーバー負荷配慮）
SLEEP_MIN = 1.0
SLEEP_MAX = 2.0


def _polite_sleep():
    """サーバー負荷を考慮した待機"""
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def init_db():
    """
    SQLiteのスキーマを初期化する。

    テーブル：
    - races    : レース情報
    - entries  : 出走情報（car_no, racer_id, style 等）
    - results  : レース結果
    - odds     : オッズ時系列
    - lines    : ライン情報
    - comments : 選手コメント
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS races (
            race_id         TEXT PRIMARY KEY,
            date            TEXT NOT NULL,
            venue_name      TEXT NOT NULL,
            venue_id        TEXT,
            race_no         INTEGER,
            grade           TEXT,
            race_type       TEXT,
            wind_speed      REAL,
            wind_direction  REAL,
            is_rain         INTEGER,
            weather         TEXT,
            is_dome         INTEGER,
            scraped_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS entries (
            entry_id        TEXT PRIMARY KEY,
            race_id         TEXT NOT NULL,
            car_no          INTEGER,
            racer_id        TEXT,
            prefecture      TEXT,
            district        TEXT,
            racer_class     TEXT,
            grade_score     REAL,
            style           TEXT,
            gear_ratio      REAL,
            back_count      INTEGER,
            home_count      INTEGER,
            start_count     INTEGER,
            win_rate        REAL,
            second_rate     REAL,
            third_rate      REAL,
            recent_5_results TEXT,
            age             INTEGER,
            term            INTEGER,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );

        CREATE TABLE IF NOT EXISTS results (
            result_id       TEXT PRIMARY KEY,
            race_id         TEXT NOT NULL,
            rank            INTEGER,
            car_no          INTEGER,
            finish_time     REAL,
            deciding_factor TEXT,
            actual_line     TEXT,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );

        CREATE TABLE IF NOT EXISTS odds (
            odds_id         TEXT PRIMARY KEY,
            race_id         TEXT NOT NULL,
            car_no          INTEGER,
            win_odds        REAL,
            timing          TEXT,
            trifecta_odds   REAL,
            scraped_at      TEXT,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );

        CREATE TABLE IF NOT EXISTS lines (
            line_id         TEXT PRIMARY KEY,
            race_id         TEXT NOT NULL,
            car_nos         TEXT,
            car_no          INTEGER,
            line_position   TEXT,
            source          TEXT,
            confidence      REAL,
            scraped_at      TEXT,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );

        CREATE TABLE IF NOT EXISTS comments (
            comment_id      TEXT PRIMARY KEY,
            race_id         TEXT NOT NULL,
            car_no          INTEGER,
            comment_text    TEXT,
            comment_date    TEXT,
            source          TEXT,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );
    """)

    conn.commit()
    conn.close()
    print(f"DB初期化完了: {DB_PATH}")


def get_scraped_dates() -> set:
    """取得済みの日付セットを返す（重複スキップ用）"""
    if not DB_PATH.exists():
        return set()
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute("SELECT DISTINCT date FROM races")
    rows = cur.fetchall()
    conn.close()
    return {row[0] for row in rows}


def scrape_race_list(date_str: str) -> list:
    """
    指定日のレース一覧を取得する。

    Parameters:
        date_str: "YYYYMMDD" 形式

    Returns:
        [{"race_id": str, "venue_name": str, "race_no": int, ...}]
    """
    raise NotImplementedError(
        "scrape_race_list: 実装待ち（データソース選定後に実装）\n"
        "候補：keirin-station.com / チャリロト公式"
    )


def scrape_race_detail(race_id: str) -> dict:
    """
    レース詳細（出走情報・オッズ・ライン）を取得する。

    Returns:
        {"entries": DataFrame, "odds": DataFrame, "lines": DataFrame}
    """
    raise NotImplementedError(
        "scrape_race_detail: 実装待ち（データソース選定後に実装）"
    )


def scrape_race_result(race_id: str) -> dict:
    """
    レース結果（着順・決まり手・実際のライン）を取得する。

    Returns:
        {"results": DataFrame}
    """
    raise NotImplementedError(
        "scrape_race_result: 実装待ち（データソース選定後に実装）"
    )


def run_historical_scraping(
    start_date: str,
    end_date: str,
):
    """
    指定期間の過去データを取得する。

    取得済みの日付はスキップする（重複防止）。
    """
    init_db()
    scraped = get_scraped_dates()

    current = datetime.strptime(start_date, "%Y%m%d")
    end_dt  = datetime.strptime(end_date, "%Y%m%d")

    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        if date_str in scraped:
            print(f"[SKIP] {date_str} は取得済み")
            current += timedelta(days=1)
            continue

        print(f"[FETCH] {date_str}")
        try:
            races = scrape_race_list(date_str)
            for race in races:
                detail = scrape_race_detail(race["race_id"])
                result = scrape_race_result(race["race_id"])
                # DBに保存（実装後に追記）
            _polite_sleep()
        except NotImplementedError as e:
            print(f"  未実装: {e}")
            break
        except Exception as e:
            print(f"  エラー: {e}")

        current += timedelta(days=1)


if __name__ == "__main__":
    # 動作確認用
    init_db()
    print("DBスキーマ初期化完了")
