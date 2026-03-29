# ===========================================
# model/data_interface.py
# 競輪AI - データ抽象化レイヤー
#
# 3モードに対応：
#   sqlite : SQLiteDBから読む（本番・研究用）
#   csv    : CSVファイルから読む（データ共有用）
#   mock   : 動作確認用の合成データ（評価に使用禁止）
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

import sys
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

# 全43会場名（bank_master.pyと同じ順序）
VENUE_NAMES = [
    "函館", "青森", "いわき平", "郡山",
    "前橋", "取手", "宇都宮", "大宮", "西武園", "京王閣", "立川",
    "松戸", "川崎", "平塚", "小田原", "伊東",
    "静岡", "浜松", "豊橋", "名古屋", "岐阜", "大垣", "松阪",
    "富山", "福井",
    "奈良", "和歌山", "岸和田", "向日町",
    "玉野", "広島", "防府",
    "高松", "小松島", "高知", "松山",
    "福岡", "小倉", "久留米", "武雄", "佐世保", "別府", "熊本",
]

DISTRICTS = ["北日本", "関東", "南関東", "中部", "北信越", "近畿", "中国", "四国", "九州"]

RACER_CLASSES = ["SS", "S1", "S2", "A1", "A2", "A3"]

GRADES = ["GP", "G1", "G2", "G3", "F1", "F2"]

STYLES = ["逃げ", "追込", "両"]

LINE_POSITIONS = ["先頭", "番手", "3番手", "単騎"]

DECIDING_FACTORS = ["逃げ", "捲り", "差し", "番手捲り", "マーク", "不明"]

_MOCK_WARNING = """
==================================================
競輪：モックデータを使用しています
  動作確認・デバッグ用です
  学習・評価・note掲載には使わないこと
==================================================
"""


class DataInterface:
    """
    競輪データの抽象化レイヤー。

    Parameters:
        mode    : "sqlite" / "csv" / "mock"
        db_path : sqliteモード時のDBパス（省略でデフォルト使用）
        csv_dir : csvモード時のディレクトリパス
    """

    def __init__(self, mode: str = "sqlite",
                 db_path: str = None,
                 csv_dir: str = None):
        if mode not in ("sqlite", "csv", "mock"):
            raise ValueError(f"modeは 'sqlite'/'csv'/'mock' のいずれかです: {mode}")
        self.mode = mode
        self.db_path = db_path or str(DB_PATH)
        self.csv_dir = csv_dir

        if mode == "mock":
            print(_MOCK_WARNING, file=sys.stderr)

    # =============================================================
    # 公開メソッド
    # =============================================================

    def get_races(self, start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
        """
        レース情報を取得する。

        Returns:
            DataFrame columns:
            race_id, date, venue_name, venue_id, race_no,
            grade, race_type,
            wind_speed, wind_direction_sin, wind_direction_cos,
            is_rain, weather,
            is_dome
        """
        if self.mode == "sqlite":
            return self._sqlite_races(start_date, end_date)
        elif self.mode == "csv":
            return self._csv_races(start_date, end_date)
        else:
            return self._mock_races()

    def get_entries(self, race_id: str = None,
                    start_date: str = None,
                    end_date: str = None) -> pd.DataFrame:
        """
        出走情報を取得する。

        Returns:
            DataFrame columns:
            entry_id, race_id, car_no, racer_id,
            prefecture, district,
            racer_class, grade_score,
            style, gear_ratio,
            back_count, home_count, start_count,
            win_rate, second_rate, third_rate,
            recent_5_results,
            age, term
        """
        if self.mode == "sqlite":
            return self._sqlite_entries(race_id, start_date, end_date)
        elif self.mode == "csv":
            return self._csv_entries(race_id, start_date, end_date)
        else:
            return self._mock_entries()

    def get_lines(self, race_id: str = None,
                  start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
        """
        ライン情報を取得する（競輪固有）。

        Returns:
            DataFrame columns:
            race_id, line_id, car_nos（例:"3-7-4"）,
            line_position（先頭/番手/3番手/単騎）,
            source（"gamboo"/"chariloto"/"comment"/"mock"）,
            confidence（信頼度 0〜1）
        """
        if self.mode == "sqlite":
            return self._sqlite_lines(race_id, start_date, end_date)
        elif self.mode == "csv":
            return self._csv_lines(race_id, start_date, end_date)
        else:
            return self._mock_lines()

    def get_comments(self, race_id: str = None,
                     start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
        """
        選手コメントを取得する。

        Returns:
            DataFrame columns:
            race_id, car_no, comment_text,
            comment_date, source
        """
        if self.mode == "sqlite":
            return self._sqlite_comments(race_id, start_date, end_date)
        elif self.mode == "csv":
            return self._csv_comments(race_id, start_date, end_date)
        else:
            return self._mock_comments()

    def get_odds(self, race_id: str = None,
                 timing: str = None) -> pd.DataFrame:
        """
        オッズ情報を取得する。

        Parameters:
            timing : None=全タイミング / "5min"/"1min"/"final" 等

        Returns:
            DataFrame columns:
            race_id, car_no, win_odds,
            timing（"120min"/"60min"/"30min"/"15min"/"5min"/"1min"/"final"）,
            trifecta_odds（3連単代表値）
        """
        if self.mode == "sqlite":
            return self._sqlite_odds(race_id, timing)
        elif self.mode == "csv":
            return self._csv_odds(race_id, timing)
        else:
            return self._mock_odds()

    def get_results(self, race_id: str = None,
                    start_date: str = None,
                    end_date: str = None) -> pd.DataFrame:
        """
        レース結果を取得する。

        Returns:
            DataFrame columns:
            race_id, rank, car_no,
            finish_time, deciding_factor（決まり手）,
            actual_line（実際のライン：過去データから）
        """
        if self.mode == "sqlite":
            return self._sqlite_results(race_id, start_date, end_date)
        elif self.mode == "csv":
            return self._csv_results(race_id, start_date, end_date)
        else:
            return self._mock_results()

    # =============================================================
    # SQLiteモード
    # =============================================================

    def _sqlite_where(self, start_date, end_date,
                      table_alias="r"):
        clauses, params = [], []
        if start_date:
            clauses.append(f"{table_alias}.date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append(f"{table_alias}.date <= ?")
            params.append(end_date)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where_sql, params

    def _sqlite_races(self, start_date, end_date):
        where_sql, params = self._sqlite_where(start_date, end_date)
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM races r {where_sql} ORDER BY r.date, r.venue_id, r.race_no",
            conn, params=params,
        )
        conn.close()
        return df

    def _sqlite_entries(self, race_id, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        if race_id:
            df = pd.read_sql_query(
                "SELECT * FROM entries WHERE race_id = ? ORDER BY car_no",
                conn, params=[race_id],
            )
        else:
            where_sql, params = self._sqlite_where(start_date, end_date)
            df = pd.read_sql_query(
                f"""SELECT e.* FROM entries e
                    JOIN races r ON e.race_id = r.race_id
                    {where_sql} ORDER BY r.date, r.venue_id, r.race_no, e.car_no""",
                conn, params=params,
            )
        conn.close()
        return df

    def _sqlite_lines(self, race_id, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        if race_id:
            df = pd.read_sql_query(
                "SELECT * FROM lines WHERE race_id = ?",
                conn, params=[race_id],
            )
        else:
            where_sql, params = self._sqlite_where(start_date, end_date)
            df = pd.read_sql_query(
                f"""SELECT l.* FROM lines l
                    JOIN races r ON l.race_id = r.race_id
                    {where_sql} ORDER BY r.date, r.venue_id, r.race_no""",
                conn, params=params,
            )
        conn.close()
        return df

    def _sqlite_comments(self, race_id, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        if race_id:
            df = pd.read_sql_query(
                "SELECT * FROM comments WHERE race_id = ?",
                conn, params=[race_id],
            )
        else:
            where_sql, params = self._sqlite_where(start_date, end_date)
            df = pd.read_sql_query(
                f"""SELECT c.* FROM comments c
                    JOIN races r ON c.race_id = r.race_id
                    {where_sql}""",
                conn, params=params,
            )
        conn.close()
        return df

    def _sqlite_odds(self, race_id, timing):
        conn = sqlite3.connect(self.db_path)
        if race_id and timing:
            df = pd.read_sql_query(
                "SELECT * FROM odds WHERE race_id = ? AND timing = ?",
                conn, params=[race_id, timing],
            )
        elif race_id:
            df = pd.read_sql_query(
                "SELECT * FROM odds WHERE race_id = ?",
                conn, params=[race_id],
            )
        else:
            df = pd.read_sql_query("SELECT * FROM odds", conn)
        conn.close()
        return df

    def _sqlite_results(self, race_id, start_date, end_date):
        conn = sqlite3.connect(self.db_path)
        if race_id:
            df = pd.read_sql_query(
                "SELECT * FROM results WHERE race_id = ? ORDER BY rank",
                conn, params=[race_id],
            )
        else:
            where_sql, params = self._sqlite_where(start_date, end_date)
            df = pd.read_sql_query(
                f"""SELECT res.* FROM results res
                    JOIN races r ON res.race_id = r.race_id
                    {where_sql} ORDER BY r.date, res.rank""",
                conn, params=params,
            )
        conn.close()
        return df

    # =============================================================
    # CSVモード
    # =============================================================

    def _csv_load(self, filename):
        path = Path(self.csv_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {path}")
        return pd.read_csv(str(path))

    def _csv_filter_dates(self, df, start_date, end_date):
        if start_date and "date" in df.columns:
            df = df[df["date"] >= start_date]
        if end_date and "date" in df.columns:
            df = df[df["date"] <= end_date]
        return df

    def _csv_races(self, start_date, end_date):
        df = self._csv_load("races.csv")
        return self._csv_filter_dates(df, start_date, end_date)

    def _csv_entries(self, race_id, start_date, end_date):
        df = self._csv_load("entries.csv")
        if race_id:
            df = df[df["race_id"] == race_id]
        return df

    def _csv_lines(self, race_id, start_date, end_date):
        df = self._csv_load("lines.csv")
        if race_id:
            df = df[df["race_id"] == race_id]
        return df

    def _csv_comments(self, race_id, start_date, end_date):
        df = self._csv_load("comments.csv")
        if race_id:
            df = df[df["race_id"] == race_id]
        return df

    def _csv_odds(self, race_id, timing):
        df = self._csv_load("odds.csv")
        if race_id:
            df = df[df["race_id"] == race_id]
        if timing:
            df = df[df["timing"] == timing]
        return df

    def _csv_results(self, race_id, start_date, end_date):
        df = self._csv_load("results.csv")
        if race_id:
            df = df[df["race_id"] == race_id]
        return df

    # =============================================================
    # モックモード（動作確認専用・評価に使用禁止）
    # =============================================================

    def _mock_races(self, n: int = 200) -> pd.DataFrame:
        """モックレースデータ（動作確認専用）"""
        rng = np.random.default_rng(42)

        # 1会場12レース × n_venues
        n_venues = max(1, n // 12)
        venues = VENUE_NAMES[:n_venues]
        rows = []
        for venue in venues:
            for race_no in range(1, 13):
                race_id = f"20240101_{venue}_{race_no:02d}"
                wind_deg = rng.uniform(0, 360)
                wind_rad = np.radians(wind_deg)
                rows.append({
                    "race_id":              race_id,
                    "date":                 "20240101",
                    "venue_name":           venue,
                    "venue_id":             f"{VENUE_NAMES.index(venue)+1:02d}",
                    "race_no":              race_no,
                    "grade":                rng.choice(GRADES, p=[0.02, 0.08, 0.10, 0.20, 0.40, 0.20]),
                    "race_type":            rng.choice(["予選", "準決", "決勝"], p=[0.70, 0.20, 0.10]),
                    "wind_speed":           round(float(rng.uniform(0, 8)), 1),
                    "wind_direction_sin":   round(float(np.sin(wind_rad)), 4),
                    "wind_direction_cos":   round(float(np.cos(wind_rad)), 4),
                    "is_rain":              int(rng.choice([0, 1], p=[0.85, 0.15])),
                    "weather":              rng.choice(["晴", "曇", "雨"], p=[0.60, 0.25, 0.15]),
                    "is_dome":              int(venue in ("前橋", "小倉")),
                })
        df = pd.DataFrame(rows)
        return df.head(n)

    def _mock_entries(self, n_races: int = None) -> pd.DataFrame:
        """モック出走データ（動作確認専用）"""
        rng = np.random.default_rng(42)
        races = self._mock_races(n=n_races or 200)
        rows = []
        for _, race in races.iterrows():
            n_entries = int(rng.choice([7, 8, 9], p=[0.30, 0.45, 0.25]))
            district_pool = rng.choice(DISTRICTS, size=n_entries, replace=True,
                                       p=[0.10, 0.18, 0.12, 0.15, 0.05, 0.12, 0.08, 0.08, 0.12])
            for car_no in range(1, n_entries + 1):
                rows.append({
                    "entry_id":       f"{race['race_id']}_c{car_no:02d}",
                    "race_id":        race["race_id"],
                    "car_no":         car_no,
                    "racer_id":       int(rng.integers(10000, 99999)),
                    "prefecture":     rng.choice(["東京", "大阪", "神奈川", "愛知", "福岡"]),
                    "district":       district_pool[car_no - 1],
                    "racer_class":    rng.choice(RACER_CLASSES, p=[0.05, 0.20, 0.25, 0.25, 0.15, 0.10]),
                    "grade_score":    round(float(rng.uniform(50, 120)), 2),
                    "style":          rng.choice(STYLES, p=[0.45, 0.40, 0.15]),
                    "gear_ratio":     round(float(rng.uniform(3.3, 3.9)), 2),
                    "back_count":     int(rng.integers(0, 80)),   # 最重要特徴量
                    "home_count":     int(rng.integers(0, 60)),
                    "start_count":    int(rng.integers(0, 100)),
                    "win_rate":       round(float(rng.uniform(0.05, 0.35)), 3),
                    "second_rate":    round(float(rng.uniform(0.10, 0.45)), 3),
                    "third_rate":     round(float(rng.uniform(0.15, 0.55)), 3),
                    "recent_5_results": ",".join(
                        [str(int(rng.integers(1, 10))) for _ in range(5)]
                    ),
                    "age":            int(rng.integers(20, 45)),
                    "term":           int(rng.integers(50, 120)),
                })
        return pd.DataFrame(rows)

    def _mock_lines(self) -> pd.DataFrame:
        """
        モックラインデータ（動作確認専用）。
        地区別に自動でライン構成を生成する。
        """
        rng = np.random.default_rng(42)
        entries = self._mock_entries()
        rows = []

        for race_id, group in entries.groupby("race_id"):
            # 地区ごとに選手をグループ化してライン構成を作る
            district_groups: dict = {}
            for _, row in group.iterrows():
                d = row["district"]
                if d not in district_groups:
                    district_groups[d] = []
                district_groups[d].append(row["car_no"])

            line_id = 0
            for district, car_nos in district_groups.items():
                if len(car_nos) >= 2:
                    # 地区内でライン構成（2〜3車）
                    rng.shuffle(np.array(car_nos, dtype=int))  # シャッフル
                    for i in range(0, len(car_nos), 3):
                        chunk = car_nos[i:i + 3]
                        line_str = "-".join(str(c) for c in chunk)
                        positions = ["先頭", "番手", "3番手"][:len(chunk)]
                        for pos_idx, car_no in enumerate(chunk):
                            rows.append({
                                "race_id":       race_id,
                                "line_id":       f"{race_id}_L{line_id:02d}",
                                "car_nos":       line_str,
                                "car_no":        car_no,
                                "line_position": positions[pos_idx],
                                "source":        "mock",
                                "confidence":    0.50,  # モックなので低め
                            })
                        line_id += 1
                else:
                    # 1人 → 単騎
                    rows.append({
                        "race_id":       race_id,
                        "line_id":       f"{race_id}_L{line_id:02d}",
                        "car_nos":       str(car_nos[0]),
                        "car_no":        car_nos[0],
                        "line_position": "単騎",
                        "source":        "mock",
                        "confidence":    0.50,
                    })
                    line_id += 1

        return pd.DataFrame(rows)

    def _mock_comments(self) -> pd.DataFrame:
        """モックコメントデータ（動作確認専用）"""
        rng = np.random.default_rng(42)
        entries = self._mock_entries()
        COMMENT_TEMPLATES = [
            "先行で勝負します",
            "自力で行きます",
            "番手で勝負します",
            "{target}選手を追います",
            "後ろにつきます",
            "単騎で行きます",
        ]
        rows = []
        for _, row in entries.iterrows():
            # 全選手の約70%がコメントあり
            if rng.random() > 0.70:
                continue
            tmpl = rng.choice(COMMENT_TEMPLATES)
            comment = tmpl.replace("{target}", str(rng.integers(1, 10)))
            rows.append({
                "race_id":      row["race_id"],
                "car_no":       row["car_no"],
                "comment_text": comment,
                "comment_date": "2024-01-01",
                "source":       "mock",
            })
        return pd.DataFrame(rows)

    def _mock_odds(self) -> pd.DataFrame:
        """モックオッズデータ（動作確認専用）"""
        rng = np.random.default_rng(42)
        entries = self._mock_entries()
        TIMINGS = ["120min", "60min", "30min", "15min", "5min", "1min", "final"]
        rows = []
        for race_id, group in entries.groupby("race_id"):
            base_odds = rng.uniform(1.5, 20.0, len(group))
            for timing in TIMINGS:
                noise = rng.uniform(0.95, 1.05, len(group))
                for i, (_, entry) in enumerate(group.iterrows()):
                    rows.append({
                        "race_id":        race_id,
                        "car_no":         entry["car_no"],
                        "win_odds":       round(float(base_odds[i] * noise[i]), 1),
                        "timing":         timing,
                        "trifecta_odds":  round(float(base_odds[i] * 10 * noise[i]), 1),
                    })
        return pd.DataFrame(rows)

    def _mock_results(self) -> pd.DataFrame:
        """モックレース結果データ（動作確認専用）"""
        rng = np.random.default_rng(42)
        entries = self._mock_entries()
        rows = []
        for race_id, group in entries.groupby("race_id"):
            car_nos = list(group["car_no"])
            rng.shuffle(np.array(car_nos))
            for rank, car_no in enumerate(car_nos, 1):
                rows.append({
                    "race_id":        race_id,
                    "rank":           rank,
                    "car_no":         car_no,
                    "finish_time":    round(float(rng.uniform(70, 80)), 2),
                    "deciding_factor": rng.choice(DECIDING_FACTORS),
                    "actual_line":    None,  # 実データなので None
                })
        return pd.DataFrame(rows)
