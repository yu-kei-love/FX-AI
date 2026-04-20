# ===========================================
# model/meet_rhythm.py
# v0.40: 節リズム 4 特徴量
#
# K-01 days_since_last_race  前回出走からの経過日数
# K-02 meet_day_number       当節の何日目か (1, 2, 3, 4)
# K-03 intra_meet_rank_avg   当節内の平均着順 (当該日より前)
# K-04 rest_quality_score    休養の質 (大きいほど好調復帰想定)
#                             = 1 / max(1, days_since_last_race) × 100
#                             ×  long-term win_rate 補正
#
# data-leak 防止:
#   全て race_date < current_race_date のデータのみ使用
#
# 使い方:
#   mr = MeetRhythmFeatures(db_path)
#   mr.preload(race_dates)  # バッチ事前ロード
#   values = mr.get_for(senshu_name, race_date)
#   → dict {K01..K04}
# ===========================================

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd


MEET_RHYTHM_FEATURE_NAMES = [
    "K01_days_since_last_race",
    "K02_meet_day_number",
    "K03_intra_meet_rank_avg",
    "K04_rest_quality_score",
]


class MeetRhythmFeatures:
    """節リズム特徴量の batch 計算"""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        # { senshu_name: [(race_date_compact, race_id, rank, win_rate_at_time), ...] }
        # race_date_compact = 'YYYYMMDD'
        self._history = None

    def preload(self):
        """全 results / entries を事前ロード"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT r.race_id, r.rank, r.sha_ban,
                   e.senshu_name, e.win_rate,
                   rc.race_date
            FROM results r
            JOIN entries e ON r.race_id = e.race_id AND r.sha_ban = e.sha_ban
            JOIN races rc ON r.race_id = rc.race_id
            WHERE e.senshu_name IS NOT NULL
              AND r.rank IS NOT NULL
            ORDER BY e.senshu_name, rc.race_date
        """, conn)
        conn.close()

        hist = defaultdict(list)
        for _, row in df.iterrows():
            if pd.isna(row["senshu_name"]):
                continue
            try:
                date_str = str(row["race_date"])
                rank = int(row["rank"])
                wr = float(row["win_rate"] or 0.0) if not pd.isna(row["win_rate"]) else 0.0
            except (ValueError, TypeError):
                continue
            hist[row["senshu_name"]].append((date_str, rank, wr))
        # sort by date
        for name in hist:
            hist[name].sort(key=lambda x: x[0])
        self._history = hist
        return len(hist)

    def _parse_date(self, date_compact):
        try:
            return datetime.strptime(date_compact, "%Y%m%d")
        except (ValueError, TypeError):
            return None

    def get_for(self, senshu_name, race_date):
        """
        Parameters:
            senshu_name: str
            race_date:   str "YYYYMMDD"
        Returns:
            dict (4 values, default 0)
        """
        default = {k: 0.0 for k in MEET_RHYTHM_FEATURE_NAMES}
        default["K02_meet_day_number"] = 1.0   # 節初日が default
        default["K01_days_since_last_race"] = 30.0  # 初出走扱い (default)
        default["K04_rest_quality_score"] = 0.0

        if self._history is None:
            return default
        history = self._history.get(senshu_name)
        if not history:
            return default

        cur_dt = self._parse_date(race_date)
        if cur_dt is None:
            return default

        # race_date より前のレースのみ
        prev_races = [(d, r, wr) for d, r, wr in history if d < race_date]
        if not prev_races:
            return default

        # K-01: 直近レースからの日数
        last_date_str, last_rank, last_wr = prev_races[-1]
        last_dt = self._parse_date(last_date_str)
        if last_dt is None:
            return default
        days_since = (cur_dt - last_dt).days
        if days_since < 0:
            days_since = 0

        # K-02: 節の何日目か
        # 連続出走判定: 直近の出走日を遡って、3日以内連続なら同一節
        meet_day = 1
        expected_prev = cur_dt
        for i in range(len(prev_races) - 1, -1, -1):
            d_str, _, _ = prev_races[i]
            d_dt = self._parse_date(d_str)
            if d_dt is None:
                break
            gap = (expected_prev - d_dt).days
            if gap == 0:
                continue
            if 1 <= gap <= 1:  # 当日 or 翌日
                meet_day += 1
                expected_prev = d_dt
            else:
                break
        # 最大 meet_day = 1 (当日スタート) + 連続前日数
        # 多くの節は 4日制 / 3日制

        # K-03: 当節内の平均着順 (meet_day > 1 のときのみ)
        intra_ranks = []
        if meet_day > 1:
            cutoff_dt = cur_dt - timedelta(days=meet_day)
            for d_str, rank, _ in prev_races[-meet_day + 1 - 1:]:
                d_dt = self._parse_date(d_str)
                if d_dt is None:
                    continue
                if cutoff_dt < d_dt < cur_dt:
                    intra_ranks.append(rank)
        intra_avg = float(sum(intra_ranks) / len(intra_ranks)) if intra_ranks else 0.0

        # K-04: 休養の質
        # days_since と最新の win_rate で補正
        # idea: 休養が長すぎても短すぎても score 下がる
        # optimal rest ~ 7-14 日仮定
        # rest_quality = exp(-((days_since - 10) / 10)^2) × win_rate_scale
        import math
        optimal = 10.0
        scale = 10.0
        diff = days_since - optimal
        rest_bell = math.exp(-(diff * diff) / (scale * scale))
        # win_rate は 0〜1 の範囲を想定。0 の場合 bell のみ。
        wr_mult = 1.0 + min(max(last_wr, 0.0), 1.0)
        rest_quality = rest_bell * wr_mult

        return {
            "K01_days_since_last_race": float(days_since),
            "K02_meet_day_number":      float(meet_day),
            "K03_intra_meet_rank_avg":  float(intra_avg),
            "K04_rest_quality_score":   float(rest_quality),
        }

    def compute_for_df(self, entries_df, races_df):
        """
        entries_df (senshu_name 列) + races_df (race_id, race_date)
        から各行に 4 特徴量を付加する。
        """
        if self._history is None:
            self.preload()
        race_date_map = dict(zip(races_df["race_id"], races_df["race_date"]))
        rows = []
        for _, row in entries_df.iterrows():
            name = row.get("senshu_name")
            rid = row.get("race_id")
            rd = race_date_map.get(rid, "")
            if not name or not rd:
                rows.append({k: 0.0 for k in MEET_RHYTHM_FEATURE_NAMES})
                continue
            rows.append(self.get_for(name, str(rd)))
        return pd.DataFrame(rows)
