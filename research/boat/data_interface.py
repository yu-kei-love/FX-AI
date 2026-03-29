# ===========================================
# data_interface.py
# ボートレースAI - データ取得の抽象化層
#
# 設計方針：
#   スクレイピングコードなしでも動作確認できる。
#   note販売時に購入者がCSVやmockで試せる。
#
#   モード：
#   "sqlite" : SQLiteDBから読む（デフォルト）
#   "csv"    : CSVファイルから読む
#   "mock"   : 動作確認用サンプルデータ（学習・評価禁止）
#
# 重要：
#   mockモードは動作確認・デバッグ専用。
#   合成データで学習・評価することを禁止する。
# ===========================================

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB   = PROJECT_ROOT / "data" / "boat" / "boatrace.db"

# 全24会場の固定データ（実際の値）
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
    (12, "住之江", 1, 0, 0.522, 0.48, 6000),
    (13, "尼崎",   1, 0, 0.543, 0.44, 5600),
    (14, "鳴門",   1, 0, 0.523, 0.51, 6100),
    (15, "丸亀",   1, 0, 0.538, 0.46, 5800),
    (16, "児島",   1, 0, 0.530, 0.49, 5900),
    (17, "宮島",   1, 0, 0.548, 0.44, 5600),
    (18, "徳山",   1, 0, 0.553, 0.43, 5500),
    (19, "下関",   1, 0, 0.545, 0.45, 5700),
    (20, "若松",   1, 0, 0.545, 0.45, 5700),
    (21, "芦屋",   1, 0, 0.525, 0.50, 6000),
    (22, "福岡",   1, 0, 0.548, 0.44, 5600),
    (23, "唐津",   1, 0, 0.518, 0.51, 6100),
    (24, "大村",   1, 0, 0.532, 0.48, 5900),
]


class DataInterface:
    """
    データ取得の抽象化層。

    note販売時：
    スクレイピングコードなしでも
    csvモードかmockモードで動作確認できる。

    モード：
    "sqlite" : SQLiteDBから読む（デフォルト）
    "csv"    : CSVファイルから読む
    "mock"   : 動作確認・デバッグ専用（学習・評価禁止）
    """

    def __init__(
        self,
        mode:     str  = "sqlite",
        db_path:  str  = None,
        csv_dir:  str  = None,
    ):
        if mode not in ("sqlite", "csv", "mock"):
            raise ValueError(f"modeは 'sqlite' / 'csv' / 'mock' のいずれかです: {mode}")

        self.mode    = mode
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.csv_dir = Path(csv_dir) if csv_dir else None

        if mode == "mock":
            print("=" * 50)
            print("警告：モックデータを使用しています")
            print("  動作確認・デバッグ専用です")
            print("  学習・評価・実運用には使わないこと")
            print("=" * 50)

    # ==========================================================
    # 公開API
    # ==========================================================

    def get_races(
        self,
        start_date: str = None,
        end_date:   str = None,
    ) -> pd.DataFrame:
        """
        レース情報を取得する。

        Returns:
            DataFrame（columns: race_id, date, venue_id, race_no,
                       grade, race_type, weather,
                       wind_speed, wind_direction_sin,
                       wind_direction_cos, wave_height）
        """
        if self.mode == "sqlite":
            return self._from_sqlite_races(start_date, end_date)
        elif self.mode == "csv":
            return self._from_csv("races.csv")
        else:
            return self._mock_races()

    def get_entries(
        self,
        race_id:    str = None,
        start_date: str = None,
        end_date:   str = None,
    ) -> pd.DataFrame:
        """
        出走情報を取得する。

        Returns:
            DataFrame（columns: entry_id, race_id, lane, course_taken,
                       racer_class, national_win_rate, local_win_rate,
                       avg_start_timing, motor_win_rate, exhibition_time 等）
        """
        if self.mode == "sqlite":
            return self._from_sqlite_entries(race_id, start_date, end_date)
        elif self.mode == "csv":
            return self._from_csv("entries.csv")
        else:
            return self._mock_entries()

    def get_odds(
        self,
        race_id: str = None,
        timing:  str = None,
    ) -> pd.DataFrame:
        """
        オッズ情報を取得する。

        Args:
            timing: None=全タイミング / "final"=確定値のみ

        Returns:
            DataFrame（columns: race_id, combination, timing,
                       odds_type, odds_value）
        """
        if self.mode == "sqlite":
            return self._from_sqlite_odds(race_id, timing)
        elif self.mode == "csv":
            return self._from_csv("odds.csv")
        else:
            return self._mock_odds()

    def get_venues(self) -> pd.DataFrame:
        """
        会場マスタを取得する。

        Returns:
            DataFrame（columns: venue_id, venue_name, water_type,
                       historical_course1_win_rate,
                       historical_upset_rate,
                       historical_avg_trifecta_odds）
        """
        if self.mode == "sqlite":
            return self._from_sqlite_venues()
        else:
            return self._mock_venues()

    # ==========================================================
    # SQLite 内部メソッド
    # ==========================================================

    def _from_sqlite_races(self, start_date, end_date) -> pd.DataFrame:
        """SQLiteからレース情報を取得する。"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DBファイルが見つかりません: {self.db_path}\n"
                "mode='mock' で動作確認できます。"
            )

        where, params = [], []
        if start_date:
            where.append("date >= ?")
            params.append(start_date)
        if end_date:
            where.append("date <= ?")
            params.append(end_date)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(
            f"SELECT * FROM races {where_sql} ORDER BY date, venue_id, race_no",
            conn, params=params,
        )
        conn.close()
        return df

    def _from_sqlite_entries(self, race_id, start_date, end_date) -> pd.DataFrame:
        """SQLiteから出走情報を取得する。"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DBファイルが見つかりません: {self.db_path}\n"
                "mode='mock' で動作確認できます。"
            )

        where, params = [], []
        if race_id:
            where.append("e.race_id = ?")
            params.append(race_id)
        if start_date:
            where.append("r.date >= ?")
            params.append(start_date)
        if end_date:
            where.append("r.date <= ?")
            params.append(end_date)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(
            f"""
            SELECT e.*
            FROM entries e
            JOIN races r ON e.race_id = r.race_id
            {where_sql}
            ORDER BY r.date, r.venue_id, r.race_no, e.lane
            """,
            conn, params=params,
        )
        conn.close()
        return df

    def _from_sqlite_odds(self, race_id, timing) -> pd.DataFrame:
        """SQLiteからオッズ情報を取得する。"""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DBファイルが見つかりません: {self.db_path}\n"
                "mode='mock' で動作確認できます。"
            )

        where, params = [], []
        if race_id:
            where.append("o.race_id = ?")
            params.append(race_id)
        if timing:
            where.append("o.timing = ?")
            params.append(timing)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(
            f"SELECT * FROM odds o {where_sql}",
            conn, params=params,
        )
        conn.close()
        return df

    def _from_sqlite_venues(self) -> pd.DataFrame:
        """SQLiteから会場マスタを取得する。"""
        if not self.db_path.exists():
            return self._mock_venues()

        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query("SELECT * FROM venues ORDER BY venue_id", conn)
        conn.close()

        # データがない場合はマスタデータで補完
        if df.empty:
            return self._mock_venues()
        return df

    # ==========================================================
    # CSV 内部メソッド
    # ==========================================================

    def _from_csv(self, filename: str) -> pd.DataFrame:
        """CSVファイルから読み込む。"""
        if self.csv_dir is None:
            raise ValueError("csv_dir が未設定です。DataInterface(mode='csv', csv_dir='path/to/dir') で指定してください。")

        fpath = self.csv_dir / filename
        if not fpath.exists():
            raise FileNotFoundError(
                f"CSVファイルが見つかりません: {fpath}\n"
                f"以下の列を持つCSVを用意してください: {filename}"
            )

        return pd.read_csv(str(fpath))

    # ==========================================================
    # モック生成（動作確認・デバッグ専用）
    # ==========================================================

    def _mock_races(self, n: int = 100) -> pd.DataFrame:
        """
        モックのレースデータを生成する（動作確認専用）。
        n レース分の合成データを生成する。
        """
        rng = np.random.default_rng(42)

        race_ids = [f"20240101_{v:02d}_{r:02d}" for v in range(1, 9) for r in range(1, 13)]
        race_ids = race_ids[:n]
        n = len(race_ids)  # 実際の件数に合わせる

        venues = [int(rid.split("_")[1]) for rid in race_ids]
        races  = [int(rid.split("_")[2]) for rid in race_ids]

        return pd.DataFrame({
            "race_id":             race_ids,
            "date":                ["20240101"] * n,
            "venue_id":            venues,
            "race_no":             races,
            "grade":               rng.choice(["一般", "G3", "G2", "G1", "SG"], n, p=[0.6, 0.2, 0.1, 0.07, 0.03]),
            "race_type":           rng.choice(["予選", "準優", "優勝戦"], n, p=[0.7, 0.2, 0.1]),
            "race_time":           ["14:30"] * n,
            "weather":             rng.choice(["晴", "曇", "雨"], n, p=[0.6, 0.3, 0.1]),
            "wind_speed":          rng.uniform(0, 8, n).round(1),
            "wind_direction_sin":  rng.uniform(-1, 1, n).round(4),
            "wind_direction_cos":  rng.uniform(-1, 1, n).round(4),
            "wave_height":         rng.uniform(0, 20, n).round(1),
            "rain_amount":         rng.uniform(0, 5, n).round(1),
            "air_pressure":        rng.uniform(1000, 1020, n).round(1),
            "temperature":         rng.uniform(5, 35, n).round(1),
            "water_type":          rng.choice([0, 1], n),
            "tide_level":          rng.uniform(0, 200, n).round(1),
        })

    def _mock_entries(self, n_races: int = 100) -> pd.DataFrame:
        """
        モックの出走データを生成する（動作確認専用）。
        n_races × 6艇 = n_races*6 件のデータ。
        """
        rng = np.random.default_rng(42)

        race_ids  = [f"20240101_{v:02d}_{r:02d}" for v in range(1, 9) for r in range(1, 13)]
        race_ids  = race_ids[:n_races]

        rows = []
        for race_id in race_ids:
            # course_takenは前付けも考慮（ランダムシャッフル）
            courses = rng.permutation(6) + 1
            for lane in range(1, 7):
                course_taken = int(courses[lane - 1])
                rows.append({
                    "race_id":             race_id,
                    "lane":                lane,
                    "course_taken":        course_taken,  # 最重要・艇番と混同しない
                    "racer_id":            rng.integers(1000, 9999),
                    "racer_class":         rng.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.4, 0.2]),
                    "racer_weight":        rng.uniform(45, 65),
                    "racer_age":           rng.integers(20, 55),
                    "is_home":             rng.choice([0, 1], p=[0.8, 0.2]),
                    "national_win_rate":   rng.uniform(0.3, 0.8),
                    "national_2place_rate": rng.uniform(0.4, 0.85),
                    "national_3place_rate": rng.uniform(0.5, 0.90),
                    "local_win_rate":      rng.uniform(0.2, 0.80),
                    "local_2place_rate":   rng.uniform(0.3, 0.85),
                    "avg_start_timing":    rng.uniform(0.10, 0.25),
                    "flying_count":        rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03]),
                    "late_count":          rng.choice([0, 1], p=[0.95, 0.05]),
                    "days_since_last_flying": rng.integers(0, 365),
                    "is_flying_return":    rng.choice([0, 1], p=[0.95, 0.05]),
                    "consecutive_race_days": rng.integers(1, 6),
                    "days_since_last_race": rng.integers(0, 90),
                    "recent_5_avg_finish": rng.uniform(1, 6),
                    "recent_5_trend":      rng.choice([-1, 0, 1]),
                    "motor_win_rate":      rng.uniform(0.3, 0.7),
                    "motor_2place_rate":   rng.uniform(0.4, 0.75),
                    "motor_maintenance_count": rng.integers(0, 5),
                    "boat_2place_rate":    rng.uniform(0.3, 0.7),
                    "exhibition_time":     rng.uniform(6.5, 7.5),
                    "exhibition_st":       rng.uniform(0.10, 0.25),
                    "exhibition_dashi":    rng.uniform(1, 5),
                    "exhibition_yukiashi": rng.uniform(1, 5),
                    "exhibition_nobiashi": rng.uniform(1, 5),
                    "exhibition_mawariashi": rng.uniform(1, 5),
                    # コース別勝率（1〜6コース）
                    **{f"course{c}_win_rate": rng.uniform(0.1, 0.8) for c in range(1, 7)},
                    **{f"course{c}_2place_rate": rng.uniform(0.2, 0.85) for c in range(1, 7)},
                    **{f"course{c}_avg_st": rng.uniform(0.10, 0.25) for c in range(1, 7)},
                    "finish":              rng.integers(1, 7),
                })

        return pd.DataFrame(rows)

    def _mock_odds(self, n_races: int = 100) -> pd.DataFrame:
        """
        モックのオッズデータを生成する（動作確認専用）。
        単勝オッズのみ生成（確定値・final タイミング）。
        """
        rng = np.random.default_rng(42)

        race_ids = [f"20240101_{v:02d}_{r:02d}" for v in range(1, 9) for r in range(1, 13)]
        race_ids = race_ids[:n_races]

        timings = ["120min", "60min", "30min", "15min", "5min", "1min", "final"]
        rows = []

        for race_id in race_ids:
            base_odds = rng.uniform(1.5, 20.0, 6)  # 6艇の基準オッズ
            for t_idx, timing in enumerate(timings):
                for lane in range(1, 7):
                    # タイミングが後になるほどオッズが変動
                    noise = rng.uniform(0.9, 1.1)
                    odds = float(base_odds[lane - 1] * noise)
                    rows.append({
                        "race_id":    race_id,
                        "odds_type":  "単勝",
                        "combination": str(lane),
                        "odds_value": round(odds, 1),
                        "timing":     timing,
                    })

        return pd.DataFrame(rows)

    def _mock_venues(self) -> pd.DataFrame:
        """全24会場の固定データを返す（実際の値を使用）。"""
        return pd.DataFrame(
            VENUE_MASTER,
            columns=[
                "venue_id", "venue_name", "water_type", "is_river",
                "historical_course1_win_rate",
                "historical_upset_rate",
                "historical_avg_trifecta_odds",
            ]
        )
