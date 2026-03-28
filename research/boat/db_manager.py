# ===========================================
# db_manager.py
# ボートレースAI - SQLiteデータベース管理
#
# 設計方針：
#   - 全特徴量を保存し、後で取捨選択できる構造
#   - course_taken（実際の進入コース）を最重要カラムとして管理
#   - オッズは7タイミング全て保存（精度最大化設計）
# ===========================================

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "boat" / "boatrace.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


SCHEMA_SQL = """
-- 会場マスタ
CREATE TABLE IF NOT EXISTS venues (
    venue_id   INTEGER PRIMARY KEY,
    venue_name TEXT NOT NULL,
    water_type INTEGER DEFAULT 0,          -- 海水=1 / 淡水=0
    is_river   INTEGER DEFAULT 0,          -- 川=1
    historical_course1_win_rate REAL,      -- 1コース歴史的勝率
    historical_upset_rate       REAL,      -- 荒れやすさ指数
    historical_avg_trifecta_odds REAL      -- 3連単平均配当
);

-- 選手マスタ
CREATE TABLE IF NOT EXISTS racers (
    racer_id   INTEGER PRIMARY KEY,        -- 登録番号
    racer_name TEXT,
    home_venue_id INTEGER,
    birth_date TEXT,
    debut_year INTEGER
);

-- レース情報
CREATE TABLE IF NOT EXISTS races (
    race_id   TEXT PRIMARY KEY,            -- "{date}_{venue_id}_{race_no}" e.g. "20260101_02_08"
    date      TEXT NOT NULL,               -- YYYYMMDD
    venue_id  INTEGER NOT NULL,
    race_no   INTEGER NOT NULL,            -- 1〜12
    grade     TEXT,                        -- SG/G1/G2/G3/一般
    race_type TEXT,                        -- 予選/準優/優勝戦
    race_time TEXT,                        -- 締め切り時刻 "14:30"
    weather   TEXT,
    wind_speed REAL,
    wind_direction TEXT,
    wind_direction_sin REAL,
    wind_direction_cos REAL,
    wave_height REAL,
    rain_amount REAL,
    air_pressure REAL,
    temperature REAL,
    water_type INTEGER,
    tide_level REAL,
    created_at TEXT,
    FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
);

-- 出走情報（最重要テーブル）
CREATE TABLE IF NOT EXISTS entries (
    entry_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id    TEXT NOT NULL,
    lane       INTEGER NOT NULL,           -- 艇番 1〜6
    course_taken INTEGER,                  -- 実際の進入コース 1〜6 ★最重要
    racer_id   INTEGER,
    racer_name TEXT,
    racer_class INTEGER,                   -- A1=4/A2=3/B1=2/B2=1
    racer_weight REAL,
    racer_age  INTEGER,
    home_venue_id INTEGER,
    is_home    INTEGER DEFAULT 0,          -- 地元フラグ

    -- 全国成績
    national_win_rate    REAL,
    national_2place_rate REAL,
    national_3place_rate REAL,

    -- 当地成績
    local_win_rate    REAL,
    local_2place_rate REAL,

    -- コース別勝率（1〜6コース）
    course1_win_rate REAL, course2_win_rate REAL,
    course3_win_rate REAL, course4_win_rate REAL,
    course5_win_rate REAL, course6_win_rate REAL,

    -- コース別2着率（1〜6コース）
    course1_2place_rate REAL, course2_2place_rate REAL,
    course3_2place_rate REAL, course4_2place_rate REAL,
    course5_2place_rate REAL, course6_2place_rate REAL,

    -- コース別平均ST（1〜6コース）
    course1_avg_st REAL, course2_avg_st REAL,
    course3_avg_st REAL, course4_avg_st REAL,
    course5_avg_st REAL, course6_avg_st REAL,

    -- スタート関連
    avg_start_timing REAL,
    flying_count  INTEGER,
    late_count    INTEGER,
    days_since_last_flying INTEGER,
    is_flying_return INTEGER DEFAULT 0,

    -- コンディション
    consecutive_race_days INTEGER,
    days_since_last_race  INTEGER,
    recent_5_avg_finish   REAL,
    recent_5_trend        INTEGER,

    -- モーター情報
    motor_no INTEGER,
    motor_win_rate    REAL,
    motor_2place_rate REAL,
    motor_maintenance_count INTEGER,

    -- ボート情報
    boat_no INTEGER,
    boat_2place_rate REAL,

    -- 展示情報（直前取得）
    exhibition_time    REAL,
    exhibition_st      REAL,
    exhibition_dashi   REAL,       -- 出足
    exhibition_yukiashi REAL,      -- 行き足
    exhibition_nobiashi REAL,      -- 伸び足
    exhibition_mawariashi REAL,    -- まわり足

    -- 結果
    finish INTEGER,                -- 着順 1〜6

    created_at TEXT,
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (racer_id) REFERENCES racers(racer_id)
);

-- インデックス（クエリ高速化）
CREATE INDEX IF NOT EXISTS idx_entries_race_id ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_entries_racer_id ON entries(racer_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_venue ON races(venue_id);

-- オッズ（7タイミング保存）
CREATE TABLE IF NOT EXISTS odds (
    odds_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id    TEXT NOT NULL,
    odds_type  TEXT NOT NULL,      -- 単勝/2連単/2連複/3連単/3連複
    combination TEXT NOT NULL,     -- "1-2-3" 形式
    odds_value REAL,
    recorded_at TEXT,
    timing     TEXT,               -- "120min"/"60min"/"30min"/"15min"/"5min"/"1min"/"final"
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_race_id ON odds(race_id);
CREATE INDEX IF NOT EXISTS idx_odds_timing ON odds(timing);

-- 選手対戦履歴
CREATE TABLE IF NOT EXISTS head_to_head (
    racer_a_id INTEGER,
    racer_b_id INTEGER,
    racer_a_win_count INTEGER DEFAULT 0,
    total_races INTEGER DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (racer_a_id, racer_b_id)
);

-- 取得済み日付の記録（重複防止）
CREATE TABLE IF NOT EXISTS fetch_log (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL,
    venue_id   INTEGER NOT NULL,
    status     TEXT NOT NULL,      -- "success"/"error"/"no_race"
    races_fetched INTEGER DEFAULT 0,
    fetched_at TEXT NOT NULL,
    error_msg  TEXT,
    UNIQUE(date, venue_id)
);
"""

# 会場マスタ初期データ
VENUE_MASTER = [
    (1,  "桐生",   0, 0, 0.556, 0.45, 5800),
    (2,  "戸田",   0, 1, 0.520, 0.52, 6200),
    (3,  "江戸川", 0, 1, 0.488, 0.58, 7100),
    (4,  "平和島", 1, 0, 0.510, 0.53, 6500),
    (5,  "多摩川", 0, 1, 0.528, 0.48, 5900),
    (6,  "浜名湖", 1, 0, 0.535, 0.47, 5700),
    (7,  "蒲郡",   0, 0, 0.548, 0.44, 5600),
    (8,  "常滑",   1, 0, 0.557, 0.43, 5500),
    (9,  "津",     1, 0, 0.542, 0.45, 5700),
    (10, "三国",   1, 0, 0.538, 0.46, 5800),
    (11, "びわこ", 0, 0, 0.518, 0.50, 6100),
    (12, "住之江", 0, 0, 0.562, 0.42, 5400),
    (13, "尼崎",   1, 0, 0.545, 0.45, 5600),
    (14, "鳴門",   1, 0, 0.533, 0.47, 5800),
    (15, "丸亀",   1, 0, 0.540, 0.46, 5700),
    (16, "児島",   1, 0, 0.536, 0.47, 5800),
    (17, "宮島",   1, 0, 0.530, 0.49, 5900),
    (18, "徳山",   1, 0, 0.541, 0.46, 5700),
    (19, "下関",   1, 0, 0.537, 0.47, 5800),
    (20, "若松",   1, 0, 0.543, 0.45, 5600),
    (21, "芦屋",   1, 0, 0.547, 0.44, 5600),
    (22, "福岡",   1, 0, 0.549, 0.44, 5500),
    (23, "唐津",   1, 0, 0.544, 0.45, 5600),
    (24, "大村",   1, 0, 0.555, 0.43, 5500),
]

# 会場コード→venue_id マッピング
VENUE_CODE_TO_ID = {str(i).zfill(2): i for i in range(1, 25)}


@contextmanager
def get_conn(db_path=None):
    """SQLite接続のコンテキストマネージャ。"""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=None):
    """データベースを初期化してスキーマとマスタデータを投入する。"""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA_SQL)

        # 会場マスタ投入（存在しない場合のみ）
        for row in VENUE_MASTER:
            conn.execute("""
                INSERT OR IGNORE INTO venues
                (venue_id, venue_name, water_type, is_river,
                 historical_course1_win_rate, historical_upset_rate,
                 historical_avg_trifecta_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, row)

    print(f"DB初期化完了: {db_path or DB_PATH}")


def is_already_fetched(date_str, venue_id, db_path=None):
    """指定日付・会場が取得済みかどうか確認する。"""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM fetch_log WHERE date=? AND venue_id=?",
            (date_str, venue_id)
        ).fetchone()
    return row is not None and row["status"] == "success"


def log_fetch_result(date_str, venue_id, status, races_fetched=0,
                     error_msg=None, db_path=None):
    """取得結果をfetch_logに記録する。"""
    now = datetime.now().isoformat()
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO fetch_log
            (date, venue_id, status, races_fetched, fetched_at, error_msg)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date_str, venue_id, status, races_fetched, now, error_msg))


def insert_race(race_data, db_path=None):
    """
    レース情報をDBに挿入する。

    Parameters:
        race_data (dict): races テーブルに対応するデータ
    """
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO races
            (race_id, date, venue_id, race_no, grade, race_type, race_time,
             weather, wind_speed, wind_direction, wind_direction_sin, wind_direction_cos,
             wave_height, rain_amount, air_pressure, temperature,
             water_type, tide_level, created_at)
            VALUES
            (:race_id, :date, :venue_id, :race_no, :grade, :race_type, :race_time,
             :weather, :wind_speed, :wind_direction, :wind_direction_sin, :wind_direction_cos,
             :wave_height, :rain_amount, :air_pressure, :temperature,
             :water_type, :tide_level, :created_at)
        """, race_data)


def insert_entry(entry_data, db_path=None):
    """
    出走情報をDBに挿入する。

    Parameters:
        entry_data (dict): entries テーブルに対応するデータ
    """
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO entries
            (race_id, lane, course_taken, racer_id, racer_name,
             racer_class, racer_weight, racer_age,
             national_win_rate, national_2place_rate, national_3place_rate,
             local_win_rate, local_2place_rate,
             avg_start_timing, flying_count, late_count,
             motor_no, motor_2place_rate, boat_no, boat_2place_rate,
             exhibition_time, exhibition_st,
             finish, created_at)
            VALUES
            (:race_id, :lane, :course_taken, :racer_id, :racer_name,
             :racer_class, :racer_weight, :racer_age,
             :national_win_rate, :national_2place_rate, :national_3place_rate,
             :local_win_rate, :local_2place_rate,
             :avg_start_timing, :flying_count, :late_count,
             :motor_no, :motor_2place_rate, :boat_no, :boat_2place_rate,
             :exhibition_time, :exhibition_st,
             :finish, :created_at)
        """, entry_data)


def insert_entries_batch(entries, db_path=None):
    """複数エントリを一括挿入する。"""
    if not entries:
        return
    with get_conn(db_path) as conn:
        for entry_data in entries:
            conn.execute("""
                INSERT OR REPLACE INTO entries
                (race_id, lane, course_taken, racer_id, racer_name,
                 racer_class, racer_weight, racer_age,
                 national_win_rate, national_2place_rate, national_3place_rate,
                 local_win_rate, local_2place_rate,
                 avg_start_timing, flying_count, late_count,
                 motor_no, motor_2place_rate, boat_no, boat_2place_rate,
                 exhibition_time, exhibition_st,
                 finish, created_at)
                VALUES
                (:race_id, :lane, :course_taken, :racer_id, :racer_name,
                 :racer_class, :racer_weight, :racer_age,
                 :national_win_rate, :national_2place_rate, :national_3place_rate,
                 :local_win_rate, :local_2place_rate,
                 :avg_start_timing, :flying_count, :late_count,
                 :motor_no, :motor_2place_rate, :boat_no, :boat_2place_rate,
                 :exhibition_time, :exhibition_st,
                 :finish, :created_at)
            """, entry_data)


def insert_odds(odds_data, db_path=None):
    """オッズデータをDBに挿入する。"""
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO odds
            (race_id, odds_type, combination, odds_value, recorded_at, timing)
            VALUES (:race_id, :odds_type, :combination, :odds_value, :recorded_at, :timing)
        """, odds_data)


def insert_odds_batch(odds_list, db_path=None):
    """複数オッズを一括挿入する。"""
    if not odds_list:
        return
    with get_conn(db_path) as conn:
        for o in odds_list:
            conn.execute("""
                INSERT OR REPLACE INTO odds
                (race_id, odds_type, combination, odds_value, recorded_at, timing)
                VALUES (:race_id, :odds_type, :combination, :odds_value, :recorded_at, :timing)
            """, o)


def get_stats(db_path=None):
    """DB内のデータ件数サマリーを返す。"""
    with get_conn(db_path) as conn:
        races_count = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        entries_count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        course_taken_count = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE course_taken IS NOT NULL"
        ).fetchone()[0]
        odds_count = conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM races"
        ).fetchone()
    return {
        "races": races_count,
        "entries": entries_count,
        "entries_with_course_taken": course_taken_count,
        "odds": odds_count,
        "date_min": date_range[0],
        "date_max": date_range[1],
    }


if __name__ == "__main__":
    init_db()
    stats = get_stats()
    print("DB統計:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
