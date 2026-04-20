# ===========================================
# model/niche_features.py
# v0.41: ニッチ特徴量 3 個
#
# L-01 same_line_cooperation_count
#       過去 60 日で同じラインメンバーと走った回数
#       (同じレースに同じ都道府県/地区選手がいた回数で近似)
# L-02 recent_kimarite_pattern_score
#       直近 10 レースの決まり手パターンスコア
#       (nige=3, makuri=2, sashi=1 の加重平均)
# L-03 prev_meet_top_finish
#       前節の最高着順 (節 = 同一会場の連続開催)
#
# data-leak 防止: race_date < current_race_date のデータのみ使用
# ===========================================

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd


NICHE_FEATURE_NAMES = [
    "L01_same_line_coop_count",
    "L02_recent_kimarite_pattern_score",
    "L03_prev_meet_top_finish",
]


class NicheFeatures:
    """v0.41 ニッチ特徴量 batch 計算"""

    KIMARITE_WEIGHTS = {
        "逃": 3.0, "逃げ": 3.0,
        "捲": 2.0, "捲り": 2.0,
        "差": 1.0, "差し": 1.0,
        "マ": 0.5, "マーク": 0.5,
    }

    def __init__(self, db_path):
        self.db_path = str(db_path)
        # { senshu_name: [(date, race_id, rank, kimari_te, prefecture, jyo_cd), ...] }
        self._history = None
        # { race_id: [(senshu_name, prefecture), ...] } (他選手の情報)
        self._race_entries = None

    def preload(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT r.race_id, r.rank, r.kimari_te,
                   e.senshu_name, e.todofuken AS prefecture,
                   rc.race_date, rc.jyo_cd
            FROM results r
            JOIN entries e ON r.race_id = e.race_id AND r.sha_ban = e.sha_ban
            JOIN races rc ON r.race_id = rc.race_id
            WHERE e.senshu_name IS NOT NULL
              AND r.rank IS NOT NULL
            ORDER BY e.senshu_name, rc.race_date
        """, conn)
        entries_all = pd.read_sql_query("""
            SELECT e.race_id, e.senshu_name, e.todofuken AS prefecture
            FROM entries e
            WHERE e.senshu_name IS NOT NULL
        """, conn)
        conn.close()

        hist = defaultdict(list)
        for _, row in df.iterrows():
            try:
                hist[row["senshu_name"]].append({
                    "date": str(row["race_date"]),
                    "race_id": row["race_id"],
                    "rank": int(row["rank"]),
                    "kimari_te": row.get("kimari_te") or "",
                    "prefecture": row.get("prefecture") or "",
                    "jyo_cd": int(row.get("jyo_cd") or 0),
                })
            except (ValueError, TypeError):
                continue
        for name in hist:
            hist[name].sort(key=lambda x: x["date"])

        race_ent = defaultdict(list)
        for _, row in entries_all.iterrows():
            race_ent[row["race_id"]].append(
                (row["senshu_name"], row.get("prefecture") or "")
            )

        self._history = hist
        self._race_entries = race_ent
        return len(hist)

    def _parse_date(self, date_str):
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except (ValueError, TypeError):
            return None

    def get_for(self, senshu_name, race_date, current_race_id=None):
        default = {k: 0.0 for k in NICHE_FEATURE_NAMES}
        default["L03_prev_meet_top_finish"] = 9.0

        if self._history is None:
            return default
        hist = self._history.get(senshu_name)
        if not hist:
            return default

        cur_dt = self._parse_date(race_date)
        if cur_dt is None:
            return default

        prev_races = [h for h in hist if h["date"] < race_date]
        if not prev_races:
            return default

        # L-01: same line cooperation
        # 過去 60 日以内のレースで、同一都道府県のメンバーと走った回数
        cutoff_dt = cur_dt - timedelta(days=60)
        cutoff_str = cutoff_dt.strftime("%Y%m%d")
        my_pref = None
        # 自分の都道府県を最新から
        if prev_races:
            my_pref = prev_races[-1]["prefecture"]
        coop = 0
        if my_pref and self._race_entries is not None:
            for h in prev_races:
                if h["date"] < cutoff_str:
                    continue
                rid = h["race_id"]
                entries = self._race_entries.get(rid, [])
                for other_name, other_pref in entries:
                    if other_name != senshu_name and other_pref == my_pref:
                        coop += 1

        # L-02: recent kimarite pattern score
        recent10 = prev_races[-10:]
        scores = []
        for h in recent10:
            kt = h.get("kimari_te") or ""
            # 部分一致
            w = 0.0
            for key, val in self.KIMARITE_WEIGHTS.items():
                if key in kt:
                    w = max(w, val)
                    break
            scores.append(w)
        kim_score = sum(scores) / len(scores) if scores else 0.0

        # L-03: prev meet top finish
        # 「前節」= 直近連続の同会場レース（4日以内連続を同節扱い）
        prev_meet = []
        if prev_races:
            last = prev_races[-1]
            prev_meet.append(last)
            last_jyo = last["jyo_cd"]
            last_dt = self._parse_date(last["date"])
            for i in range(len(prev_races) - 2, -1, -1):
                h = prev_races[i]
                if h["jyo_cd"] != last_jyo:
                    break
                d = self._parse_date(h["date"])
                if d is None:
                    break
                if last_dt and (last_dt - d).days <= 4:
                    prev_meet.append(h)
                    last_dt = d
                else:
                    break
        prev_top = min((h["rank"] for h in prev_meet), default=9)

        return {
            "L01_same_line_coop_count":         float(coop),
            "L02_recent_kimarite_pattern_score": float(kim_score),
            "L03_prev_meet_top_finish":         float(prev_top),
        }
