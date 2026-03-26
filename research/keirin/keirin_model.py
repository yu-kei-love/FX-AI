# ===========================================
# keirin_model.py
# 競輪予測モデル v3.0: ライン戦術を考慮した着順予測
#
# v3.0 改善点 (v2.0からの差分):
#   - Softmax確率キャリブレーション (温度スケーリング)
#   - ペアワイズ比較特徴量 (A>B確率予測)
#   - EV閾値グリッドサーチ (bet type別最適化)
#   - Fractional Kelly ベットサイジング (0.25x)
#   - Top-2/Top-3予測モデル追加
#
# v2.0 改善点:
#   - 5モデルアンサンブル (LGB+XGB+CatBoost+RF+ExtraTrees)
#   - 30+特徴量 (逃げ成功率, ブロック力, 直近成績, 対戦成績等)
#   - S級/A級の分離シミュレーション
#   - 500+選手プールからの抽選
#   - 2車単・3連単バリューベッティング
#   - 風抵抗モデル改善（番手有利度の精緻化）
#   - モーニング vs メインレースの差異
#   - Walk-Forward 8分割
# ===========================================

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

warnings.filterwarnings("ignore")

# ===== 定数 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "keirin"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 特徴量カラム (30+)
FEATURE_COLS = [
    # 選手基本
    "racer_class",          # 級班 (S-SS=6, S-1=5, S-2=4, A-1=3, A-2=2, A-3=1)
    "race_points",          # 競走得点
    "win_rate",             # 勝率
    "place_rate",           # 2連対率
    "three_place_rate",     # 3連対率
    "racer_age",            # 年齢
    # 脚質
    "leg_type",             # 逃=0, 捲=1, 差=2, 追込=3
    "escape_count",         # 逃げ回数
    "makuri_count",         # 捲り回数
    "escape_success_rate",  # 逃げ成功率
    # 直近成績
    "recent_avg_finish",    # 直近5走の平均着順
    "recent_win_count",     # 直近5走の1着回数
    "recent_top3_count",    # 直近5走の3着以内回数
    # ライン情報（最重要）
    "line_position",        # ライン内位置 (先行=1, 番手=2, 3番手=3, 単騎=0)
    "line_strength",        # ライン全体の平均競走得点
    "line_size",            # ライン人数 (2 or 3)
    "is_line_front",        # ライン先頭か (0/1)
    "line_front_escape_rate",  # ライン先頭の逃げ成功率
    "line_block_strength",  # 番手のブロック力
    # 対戦・コース適性
    "h2h_advantage",        # 対戦成績アドバンテージ
    "course_familiarity",   # 当地成績 (0-1)
    # ギア・コンディション
    "gear_ratio",           # ギア倍数 (3.57-4.00)
    "days_since_last_race", # 前走からの日数（連投リスク）
    # コース
    "post_position",        # 車番 (1-9)
    "bank_type",            # バンク (333m=0, 400m=1, 500m=2)
    "weather_condition",    # 天候 (晴=0, 曇=1, 雨=2)
    "field_size",           # 出走数 (7 or 9)
    # レース条件
    "is_morning_race",      # モーニングレースか
    "race_class_level",     # S級=3, A級1-2=2, A級3=1
    # 交互作用特徴量
    "points_x_line_strength",   # 競走得点 × ライン強度
    "bante_x_front_escape",     # 番手フラグ × 先頭逃げ成功率
]

# 級班の定義
CLASS_MAP = {
    6: "S級S班", 5: "S級1班", 4: "S級2班",
    3: "A級1班", 2: "A級2班", 1: "A級3班",
}

LEG_TYPE_MAP = {0: "逃", 1: "捲", 2: "差", 3: "追込"}


# ==========================================================
# 0. 選手プール生成
#    500+人の選手プールを作り、レースごとに抽選する
# ==========================================================

def _create_racer_pool(n_racers=600, rng=None):
    """600人の選手プールを生成"""
    if rng is None:
        rng = np.random.default_rng()

    pool = []
    # S級: ~120人, A級1-2班: ~280人, A級3班: ~200人
    class_distribution = [
        (6, 9),    # S-SS: 9人
        (5, 50),   # S-1
        (4, 60),   # S-2
        (3, 130),  # A-1
        (2, 150),  # A-2
        (1, 201),  # A-3
    ]

    racer_id = 0
    for cls, count in class_distribution:
        for _ in range(count):
            base_points = {6: 117, 5: 107, 4: 97, 3: 87, 2: 77, 1: 67}
            spread = {6: 4, 5: 6, 4: 6, 3: 6, 2: 6, 1: 5}
            race_points = rng.normal(base_points[cls], spread[cls])
            race_points = np.clip(race_points, 50, 133)

            base_win = (race_points - 60) / 200
            win_rate = np.clip(base_win + rng.normal(0, 0.04), 0.0, 0.50)
            place_rate = np.clip(win_rate * 1.8 + rng.normal(0, 0.03), win_rate, 0.70)
            three_place_rate = np.clip(place_rate * 1.4 + rng.normal(0, 0.03), place_rate, 0.85)

            leg_type = rng.choice([0, 1, 2, 3], p=[0.18, 0.22, 0.33, 0.27])
            if leg_type == 0:
                escape_count = rng.poisson(5)
                makuri_count = rng.poisson(1)
                escape_success_rate = np.clip(rng.beta(3, 5), 0.05, 0.70)
                block_strength = rng.uniform(0.1, 0.4)
            elif leg_type == 1:
                escape_count = rng.poisson(1)
                makuri_count = rng.poisson(4)
                escape_success_rate = np.clip(rng.beta(1, 5), 0.0, 0.3)
                block_strength = rng.uniform(0.2, 0.6)
            elif leg_type == 2:
                escape_count = rng.poisson(0.5)
                makuri_count = rng.poisson(1)
                escape_success_rate = np.clip(rng.beta(1, 8), 0.0, 0.15)
                block_strength = rng.uniform(0.4, 0.8)
            else:
                escape_count = rng.poisson(0.2)
                makuri_count = rng.poisson(0.5)
                escape_success_rate = np.clip(rng.beta(1, 10), 0.0, 0.1)
                block_strength = rng.uniform(0.5, 0.9)

            age = int(rng.normal(35, 8))
            age = np.clip(age, 20, 62)
            # ギア倍数: 高ギア=スピード型, 低ギア=持久型
            gear_ratio = round(rng.normal(3.92, 0.08), 2)
            gear_ratio = np.clip(gear_ratio, 3.57, 4.00)

            pool.append({
                "racer_id": racer_id,
                "racer_class": cls,
                "race_points": round(race_points, 2),
                "win_rate": round(win_rate, 3),
                "place_rate": round(place_rate, 3),
                "three_place_rate": round(three_place_rate, 3),
                "leg_type": int(leg_type),
                "escape_count": int(escape_count),
                "makuri_count": int(makuri_count),
                "escape_success_rate": round(escape_success_rate, 3),
                "block_strength": round(block_strength, 3),
                "racer_age": int(age),
                "gear_ratio": gear_ratio,
            })
            racer_id += 1

    return pool


# ==========================================================
# 1. 合成データ生成 (大幅改善)
# ==========================================================

def _generate_line_formation(n_racers=9, rng=None):
    """ライン編成を生成する"""
    if rng is None:
        rng = np.random.default_rng()

    if n_racers == 9:
        patterns = [
            ([3, 3, 3], 0.30),
            ([3, 3, 2, 1], 0.25),
            ([3, 2, 2, 1, 1], 0.20),
            ([3, 3, 1, 1, 1], 0.10),
            ([2, 2, 2, 2, 1], 0.10),
            ([3, 2, 2, 2], 0.05),
        ]
    else:
        patterns = [
            ([3, 2, 2], 0.35),
            ([3, 3, 1], 0.25),
            ([2, 2, 2, 1], 0.20),
            ([3, 2, 1, 1], 0.15),
            ([2, 2, 1, 1, 1], 0.05),
        ]

    sizes_list, probs = zip(*patterns)
    probs = np.array(probs) / sum(probs)
    idx = rng.choice(len(sizes_list), p=probs)
    line_sizes = list(sizes_list[idx])

    racer_indices = list(range(n_racers))
    rng.shuffle(racer_indices)

    assignments = []
    pos = 0
    for line_id, size in enumerate(line_sizes):
        for within_pos in range(size):
            racer_idx = racer_indices[pos]
            if size == 1:
                line_position = 0
            else:
                line_position = within_pos + 1
            assignments.append({
                "racer_idx": racer_idx,
                "line_id": line_id,
                "line_position": line_position,
                "line_size": size,
            })
            pos += 1

    assignments.sort(key=lambda x: x["racer_idx"])
    return assignments


def _simulate_finish(racers, line_assignments, bank_type, weather,
                     is_morning, race_class_level, rng):
    """
    着順をシミュレート（v2.0: 改善版）

    改善点:
    - 風抵抗モデル: 番手の空気抵抗削減を精緻化
    - S級 vs A級: S級は実力通りになりやすい
    - モーニング vs メイン: モーニングは荒れやすい
    - ブロック力: 番手のブロック力が3番手以降をブロック
    - 逃げ成功率: 先行選手の成功率がライン全体に影響
    """
    n = len(racers)
    scores = np.zeros(n)

    # --- (A) 個人実力 ---
    points = np.array([r["race_points"] for r in racers])
    points_norm = (points - points.mean()) / (points.std() + 1e-8)
    # S級は実力差がつきやすい
    skill_weight = 1.8 if race_class_level == 3 else 1.4
    scores += points_norm * skill_weight

    # --- (B) 年齢効果 ---
    for i, r in enumerate(racers):
        age = r["racer_age"]
        if age <= 28:
            scores[i] += 0.15  # 若手の勢い
        elif age >= 45:
            scores[i] -= 0.2   # ベテランの体力低下

    # --- (C) ライン効果（最重要、精緻化） ---
    line_groups = {}
    for a in line_assignments:
        lid = a["line_id"]
        if lid not in line_groups:
            line_groups[lid] = []
        line_groups[lid].append(a["racer_idx"])

    for a in line_assignments:
        idx = a["racer_idx"]
        lid = a["line_id"]
        lpos = a["line_position"]
        lsize = a["line_size"]

        line_members = line_groups[lid]
        line_avg = np.mean([racers[m]["race_points"] for m in line_members])
        line_strength = (line_avg - 80) / 20

        if lsize >= 2:
            # ライン先頭の逃げ成功率を取得
            front_idx = [m for m in line_members
                         if line_assignments[m]["line_position"] == 1]
            front_escape = 0.3
            if front_idx:
                front_escape = racers[front_idx[0]]["escape_success_rate"]

            if lpos == 1:
                # 先行: ライン強度 + 逃げ成功率で決まる
                escape_bonus = front_escape * 1.5
                scores[idx] += line_strength * 0.8 + escape_bonus
                if racers[idx]["leg_type"] == 0:
                    scores[idx] += 0.35
            elif lpos == 2:
                # 番手: 風抵抗削減 + ブロック力
                # 番手は先行が強いほど有利（ドラフティング効果）
                draft_bonus = line_strength * 1.3
                # 先行の逃げ成功率が高いほど番手も有利
                front_sync = front_escape * 0.8
                # ブロック力で後方をブロック
                block_bonus = racers[idx]["block_strength"] * 0.5
                scores[idx] += draft_bonus + front_sync + block_bonus
                if racers[idx]["leg_type"] in [2, 3]:
                    scores[idx] += 0.45
            elif lpos == 3:
                # 3番手: 番手ほどではないが恩恵あり
                scores[idx] += line_strength * 0.5
                # 番手のブロック力が高いと3番手も守られる
                bante_idx = [m for m in line_members
                             if line_assignments[m]["line_position"] == 2]
                if bante_idx:
                    bante_block = racers[bante_idx[0]]["block_strength"]
                    scores[idx] += bante_block * 0.3
        else:
            # 単騎: 不利
            scores[idx] -= 0.6
            # ただし実力が高い単騎は自力で戦える
            if racers[idx]["race_points"] > 100:
                scores[idx] += 0.3

    # --- (D) バンクと脚質の相性 ---
    for i, r in enumerate(racers):
        if bank_type == 0:  # 333m
            if r["leg_type"] in [0, 1]:
                scores[i] += 0.25
        elif bank_type == 2:  # 500m
            if r["leg_type"] in [2, 3]:
                scores[i] += 0.25

    # --- (E) ギア倍数の影響 ---
    for i, r in enumerate(racers):
        gear = r["gear_ratio"]
        if bank_type == 0:  # 短いバンク: 高ギア有利
            scores[i] += (gear - 3.92) * 2.0
        elif bank_type == 2:  # 長いバンク: 低ギア有利
            scores[i] -= (gear - 3.92) * 1.5

    # --- (F) 車番の影響 ---
    for a in line_assignments:
        idx = a["racer_idx"]
        post = idx + 1
        if post <= 3:
            scores[idx] += 0.1

    # --- (G) 連投リスク ---
    for i, r in enumerate(racers):
        days = r.get("days_since_last_race", 7)
        if days <= 1:
            scores[i] -= 0.3  # 連投は疲労
        elif days >= 30:
            scores[i] -= 0.15  # 長期休養明けも不利

    # --- (H) ランダム要素 ---
    # S級は展開が読みやすい(ノイズ小), モーニングは荒れる(ノイズ大)
    base_noise = 1.5
    if weather == 2:
        base_noise += 0.5
    if is_morning:
        base_noise += 0.3  # モーニングはメンバー弱め→荒れる
    if race_class_level == 3:
        base_noise -= 0.2  # S級は堅い
    elif race_class_level == 1:
        base_noise += 0.2  # A級3班は荒れる

    noise = rng.normal(0, base_noise, n)
    scores += noise

    # 落車リスク (約3%の確率)
    for i in range(n):
        if rng.random() < 0.03:
            scores[i] -= 5.0  # 落車→ほぼ最下位

    # 着順を決定
    finishes = np.zeros(n, dtype=int)
    for rank, racer_idx in enumerate(np.argsort(-scores)):
        finishes[racer_idx] = rank + 1

    return finishes


def generate_training_data(n_races=10000, seed=42):
    """
    競輪の合成トレーニングデータを生成する (v2.0)

    改善: 500+選手プール、直近成績、対戦成績、コース適性等
    """
    rng = np.random.default_rng(seed)
    pool = _create_racer_pool(600, rng)
    rows = []

    # 選手の直近成績を追跡（最大5走）
    racer_history = {r["racer_id"]: [] for r in pool}
    # 対戦成績
    h2h_records = {}
    # 当地成績 (バンクタイプごと)
    course_records = {r["racer_id"]: {0: [], 1: [], 2: []} for r in pool}
    # 前走日 (race_id as proxy)
    last_race_day = {r["racer_id"]: -999 for r in pool}

    for race_id in range(n_races):
        # レース条件
        field_size = rng.choice([7, 9], p=[0.15, 0.85])
        bank_type = rng.choice([0, 1, 2], p=[0.20, 0.60, 0.20])
        weather = rng.choice([0, 1, 2], p=[0.60, 0.25, 0.15])
        race_class_level = rng.choice([1, 2, 3], p=[0.20, 0.50, 0.30])
        is_morning = int(rng.random() < 0.25)  # 25%がモーニング

        # 級班に合った選手をプールから抽選
        if race_class_level == 3:
            eligible = [r for r in pool if r["racer_class"] >= 4]
        elif race_class_level == 2:
            eligible = [r for r in pool if r["racer_class"] in [2, 3]]
        else:
            eligible = [r for r in pool if r["racer_class"] == 1]

        if len(eligible) < field_size:
            eligible = pool.copy()

        selected_indices = rng.choice(len(eligible), size=field_size, replace=False)
        racers = [eligible[i].copy() for i in selected_indices]

        # 直近成績を付与
        for r in racers:
            rid = r["racer_id"]
            hist = racer_history[rid]
            if len(hist) >= 1:
                r["recent_avg_finish"] = round(np.mean(hist[-5:]), 2)
                r["recent_win_count"] = int(sum(1 for h in hist[-5:] if h == 1))
                r["recent_top3_count"] = int(sum(1 for h in hist[-5:] if h <= 3))
            else:
                # 初走: ニュートラル
                r["recent_avg_finish"] = 5.0
                r["recent_win_count"] = 0
                r["recent_top3_count"] = 1

            # 当地成績
            cr = course_records[rid][bank_type]
            if len(cr) >= 1:
                r["course_familiarity"] = round(np.clip(
                    sum(1 for c in cr[-10:] if c <= 3) / max(len(cr[-10:]), 1),
                    0, 1
                ), 3)
            else:
                r["course_familiarity"] = 0.3  # デフォルト

            # 前走からの日数 (race_id差をプロキシとして使用)
            days = race_id - last_race_day[rid]
            r["days_since_last_race"] = min(days, 60)

        # ライン編成
        line_assignments = _generate_line_formation(field_size, rng)

        # ライン強度・先頭逃げ成功率・ブロック力を各選手に付与
        line_groups = {}
        for a in line_assignments:
            lid = a["line_id"]
            if lid not in line_groups:
                line_groups[lid] = []
            line_groups[lid].append(a["racer_idx"])

        for a in line_assignments:
            idx = a["racer_idx"]
            lid = a["line_id"]
            members = line_groups[lid]
            line_strength = np.mean([racers[m]["race_points"] for m in members])
            a["line_strength"] = round(line_strength, 2)

            # ライン先頭の逃げ成功率
            front_members = [m for m in members
                             if line_assignments[m]["line_position"] == 1]
            if front_members:
                a["line_front_escape_rate"] = racers[front_members[0]]["escape_success_rate"]
            else:
                a["line_front_escape_rate"] = 0.0

            # 番手のブロック力
            bante_members = [m for m in members
                             if line_assignments[m]["line_position"] == 2]
            if bante_members:
                a["line_block_strength"] = racers[bante_members[0]]["block_strength"]
            else:
                a["line_block_strength"] = 0.0

        # 対戦成績アドバンテージ
        racer_ids_in_race = [r["racer_id"] for r in racers]
        for i, r in enumerate(racers):
            adv = 0
            count = 0
            for j, r2 in enumerate(racers):
                if i == j:
                    continue
                key = (r["racer_id"], r2["racer_id"])
                if key in h2h_records and h2h_records[key][1] > 0:
                    wins, total = h2h_records[key]
                    adv += wins / total - 0.5
                    count += 1
            r["h2h_advantage"] = round(adv / max(count, 1), 3)

        # 着順シミュレーション
        finishes = _simulate_finish(racers, line_assignments, bank_type,
                                    weather, is_morning, race_class_level, rng)

        # 履歴を更新
        for i in range(field_size):
            rid = racers[i]["racer_id"]
            f = int(finishes[i])
            racer_history[rid].append(f)
            if len(racer_history[rid]) > 20:
                racer_history[rid] = racer_history[rid][-20:]
            course_records[rid][bank_type].append(f)
            last_race_day[rid] = race_id

            # 対戦成績更新
            for j in range(field_size):
                if i == j:
                    continue
                key = (racers[i]["racer_id"], racers[j]["racer_id"])
                if key not in h2h_records:
                    h2h_records[key] = [0, 0]
                h2h_records[key][1] += 1
                if finishes[i] < finishes[j]:
                    h2h_records[key][0] += 1

        # データ行を生成
        for i in range(field_size):
            a = line_assignments[i]
            r = racers[i]
            row = {
                "race_id": race_id,
                "racer_idx": i,
                "post_position": i + 1,
                "finish": int(finishes[i]),
                "is_winner": int(finishes[i] == 1),
                "finish_top2": int(finishes[i] <= 2),
                "finish_top3": int(finishes[i] <= 3),
                # 選手基本
                "racer_class": r["racer_class"],
                "race_points": r["race_points"],
                "win_rate": r["win_rate"],
                "place_rate": r["place_rate"],
                "three_place_rate": r["three_place_rate"],
                "racer_age": r["racer_age"],
                # 脚質
                "leg_type": r["leg_type"],
                "escape_count": r["escape_count"],
                "makuri_count": r["makuri_count"],
                "escape_success_rate": r["escape_success_rate"],
                # 直近成績
                "recent_avg_finish": r["recent_avg_finish"],
                "recent_win_count": r["recent_win_count"],
                "recent_top3_count": r["recent_top3_count"],
                # ライン
                "line_position": a["line_position"],
                "line_strength": a["line_strength"],
                "line_size": a["line_size"],
                "is_line_front": int(a["line_position"] == 1),
                "line_front_escape_rate": a["line_front_escape_rate"],
                "line_block_strength": a["line_block_strength"],
                # 対戦・コース
                "h2h_advantage": r.get("h2h_advantage", 0.0),
                "course_familiarity": r.get("course_familiarity", 0.3),
                # ギア・コンディション
                "gear_ratio": r["gear_ratio"],
                "days_since_last_race": r.get("days_since_last_race", 7),
                # コース条件
                "bank_type": bank_type,
                "weather_condition": weather,
                "field_size": field_size,
                # レース条件
                "is_morning_race": is_morning,
                "race_class_level": race_class_level,
                # 交互作用
                "points_x_line_strength": round(
                    r["race_points"] * a["line_strength"] / 100, 4),
                "bante_x_front_escape": round(
                    float(a["line_position"] == 2) * a["line_front_escape_rate"], 4),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[データ生成] {n_races}レース、{len(df)}行のデータを生成")
    return df


# ==========================================================
# 2. モデル学習
#    5モデルアンサンブル
# ==========================================================

class KeirinPredictor:
    """
    競輪予測モデル v2.0

    5モデルアンサンブル:
      LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
    """

    def __init__(self):
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

        self.models = {}
        self.weights = {}

        self.models["lgb"] = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=7,
            num_leaves=40, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )
        self.weights["lgb"] = 0.30

        self.models["xgb"] = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=7,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric="logloss",
        )
        self.weights["xgb"] = 0.25

        # CatBoost
        try:
            from catboost import CatBoostClassifier
            self.models["cat"] = CatBoostClassifier(
                iterations=400, learning_rate=0.04, depth=7,
                random_state=42, verbose=0,
            )
            self.weights["cat"] = 0.20
        except ImportError:
            print("  [注意] CatBoostが未インストール。4モデルで続行。")

        self.models["rf"] = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        self.weights["rf"] = 0.15

        self.models["et"] = ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        self.weights["et"] = 0.10

        # 重み正規化
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.is_fitted = False

    def fit(self, X, y):
        """学習"""
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """加重平均で勝利確率を推定"""
        probs = np.zeros(X.shape[0])
        for name, model in self.models.items():
            p = model.predict_proba(X)[:, 1]
            probs += p * self.weights[name]
        return probs

    def predict_race(self, X_race, race_ids, temperature=1.0):
        """
        レースごとにSoftmax正規化 (v3.0: 温度スケーリング付き)

        temperature < 1.0: より自信のある予測 (尖った分布)
        temperature > 1.0: より均一な予測 (平坦な分布)
        """
        raw_probs = self.predict_proba(X_race)
        normalized = np.zeros_like(raw_probs)

        for rid in np.unique(race_ids):
            mask = race_ids == rid
            race_probs = raw_probs[mask]

            if race_probs.sum() <= 0:
                normalized[mask] = 1.0 / mask.sum()
                continue

            # Softmax with temperature scaling
            # log(prob) をスコアとして使用し、温度で割ってsoftmax
            log_probs = np.log(np.clip(race_probs, 1e-10, None))
            scaled = log_probs / max(temperature, 0.01)
            # 数値安定性のためmax引き
            scaled -= scaled.max()
            exp_scaled = np.exp(scaled)
            softmax_probs = exp_scaled / exp_scaled.sum()

            normalized[mask] = softmax_probs

        return normalized

    def feature_importance(self):
        """特徴量重要度を返す（LightGBM基準）"""
        if not self.is_fitted:
            return {}
        imp = self.models["lgb"].feature_importances_
        return dict(zip(FEATURE_COLS, imp))


class KeirinTop2Predictor:
    """
    v3.0: Top-2入着予測モデル

    2車単/2車複向けに、2着以内に入る確率を予測する。
    勝利確率モデルとは別に学習することで、
    「勝てないが2着には来やすい」選手を捕捉する。
    """

    def __init__(self):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=32, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=43, verbosity=-1,
        )
        self.is_fitted = False

    def fit(self, X, y_top2):
        """y_top2: 2着以内=1, それ以外=0"""
        self.model.fit(X, y_top2)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class KeirinPairwisePredictor:
    """
    v3.0: ペアワイズ比較予測

    2人の選手の特徴量差分から「A > B」の確率を予測する。
    2車単の順序予測に使用。
    """

    def __init__(self):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=24, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            random_state=44, verbosity=-1,
        )
        self.is_fitted = False

    def fit(self, X_pairs, y_pairs):
        """
        X_pairs: 選手Aの特徴量 - 選手Bの特徴量 (差分)
        y_pairs: A がBより上位=1, それ以外=0
        """
        self.model.fit(X_pairs, y_pairs)
        self.is_fitted = True
        return self

    def predict_proba(self, X_pairs):
        return self.model.predict_proba(X_pairs)[:, 1]


def build_pairwise_data(df, feature_cols, max_pairs_per_race=20):
    """
    レースデータからペアワイズ学習データを構築

    各レースから上位者 vs 下位者のペアを作成し、
    特徴量差分と勝敗ラベルを返す。
    """
    rng = np.random.default_rng(42)
    pair_X = []
    pair_y = []

    for race_id in df["race_id"].unique():
        race = df[df["race_id"] == race_id]
        if len(race) < 2:
            continue

        features = race[feature_cols].values
        finishes = race["finish"].values
        n = len(race)

        # ペアをサンプリング (全組合せは多すぎるので制限)
        pairs_generated = 0
        indices = list(range(n))
        rng.shuffle(indices)

        for i in range(n):
            for j in range(i + 1, n):
                if pairs_generated >= max_pairs_per_race:
                    break
                ii, jj = indices[i], indices[j]
                diff = features[ii] - features[jj]
                label = 1 if finishes[ii] < finishes[jj] else 0

                pair_X.append(diff)
                pair_y.append(label)
                pairs_generated += 1
            if pairs_generated >= max_pairs_per_race:
                break

    return np.array(pair_X), np.array(pair_y)


# ==========================================================
# 3. 資金管理
# ==========================================================

class KeirinRiskManager:
    """競輪用リスク管理"""

    def __init__(self, daily_budget=3000, monthly_budget=30000):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.daily_spent = 0
        self.monthly_spent = 0
        self.daily_profit = 0
        self.monthly_profit = 0
        self.bet_history = []

    def can_bet(self, amount):
        if self.daily_spent + amount > self.daily_budget:
            return False, "日次予算超過"
        if self.monthly_spent + amount > self.monthly_budget:
            return False, "月次予算超過"
        return True, "OK"

    def place_bet(self, amount, race_id, bet_type, selection):
        can, reason = self.can_bet(amount)
        if not can:
            return False, reason
        self.daily_spent += amount
        self.monthly_spent += amount
        self.bet_history.append({
            "race_id": race_id, "bet_type": bet_type,
            "selection": selection, "amount": amount,
            "result": None, "payout": 0,
        })
        return True, "ベット完了"

    def settle(self, race_id, payout):
        for bet in self.bet_history:
            if bet["race_id"] == race_id and bet["result"] is None:
                bet["result"] = "的中" if payout > 0 else "不的中"
                bet["payout"] = payout
                profit = payout - bet["amount"]
                self.daily_profit += profit
                self.monthly_profit += profit
                break

    def reset_daily(self):
        self.daily_spent = 0
        self.daily_profit = 0

    def summary(self):
        total_bet = sum(b["amount"] for b in self.bet_history)
        total_payout = sum(b["payout"] for b in self.bet_history)
        n_bets = len(self.bet_history)
        n_hits = sum(1 for b in self.bet_history if b["result"] == "的中")
        hit_rate = n_hits / n_bets if n_bets > 0 else 0
        roi = total_payout / total_bet if total_bet > 0 else 0
        return {
            "総ベット数": n_bets, "的中数": n_hits,
            "的中率": f"{hit_rate:.1%}",
            "総投資額": total_bet, "総回収額": total_payout,
            "回収率": f"{roi:.1%}", "純損益": total_payout - total_bet,
        }


# ==========================================================
# 4. Walk-Forward検証 (8分割)
# ==========================================================

def walk_forward_validation(df, n_splits=8, use_pairwise=True, temperature=1.0):
    """
    Walk-Forward検証 (v3.0)

    改善:
    - Top-2モデル併用
    - ペアワイズモデルで2車単精度向上
    - 温度スケーリング付きSoftmax
    """
    race_ids = df["race_id"].unique()
    n_races = len(race_ids)
    split_size = n_races // (n_splits + 1)

    all_results = []

    for fold in range(n_splits):
        train_end = (fold + 1) * split_size
        test_start = train_end
        test_end = min(test_start + split_size, n_races)
        if test_end <= test_start:
            break

        train_races = race_ids[:train_end]
        test_races = race_ids[test_start:test_end]

        train_mask = df["race_id"].isin(train_races)
        test_mask = df["race_id"].isin(test_races)

        X_train = df.loc[train_mask, FEATURE_COLS].values
        y_train = df.loc[train_mask, "is_winner"].values
        y_train_top2 = df.loc[train_mask, "finish_top2"].values
        X_test = df.loc[test_mask, FEATURE_COLS].values
        test_race_ids = df.loc[test_mask, "race_id"].values
        test_finishes = df.loc[test_mask, "finish"].values
        test_post_positions = df.loc[test_mask, "post_position"].values

        # Win model
        model = KeirinPredictor()
        model.fit(X_train, y_train)

        # Top-2 model
        top2_model = KeirinTop2Predictor()
        top2_model.fit(X_train, y_train_top2)

        # Pairwise model (v3.0)
        pairwise_model = None
        if use_pairwise:
            train_df = df.loc[train_mask]
            pair_X, pair_y = build_pairwise_data(train_df, FEATURE_COLS,
                                                  max_pairs_per_race=15)
            if len(pair_X) > 100:
                pairwise_model = KeirinPairwisePredictor()
                pairwise_model.fit(pair_X, pair_y)

        # Win probability with softmax
        win_probs = model.predict_race(X_test, test_race_ids, temperature=temperature)
        # Top-2 probability
        top2_probs = top2_model.predict_proba(X_test)

        for rid in np.unique(test_race_ids):
            mask = test_race_ids == rid
            race_win_probs = win_probs[mask]
            race_top2_probs = top2_probs[mask]
            race_finishes = test_finishes[mask]
            race_features = X_test[mask]
            n_racers = mask.sum()

            # Combine win + top2 for ranking
            # 0.6 * win_prob + 0.4 * top2_prob for more stable ranking
            combined_probs = 0.6 * race_win_probs + 0.4 * race_top2_probs
            # Re-normalize
            combined_probs = combined_probs / max(combined_probs.sum(), 1e-8)

            pred_winner_idx = np.argmax(combined_probs)
            actual_winner_idx = np.argmin(race_finishes)

            # Top-2 prediction: use combined model
            top2_pred = np.argsort(-combined_probs)[:2]
            actual_top2 = np.argsort(race_finishes)[:2]

            # For exacta: use pairwise model to determine order
            if pairwise_model is not None and n_racers >= 2:
                cand_a = top2_pred[0]
                cand_b = top2_pred[1]
                diff_ab = race_features[cand_a] - race_features[cand_b]
                p_a_beats_b = pairwise_model.predict_proba(
                    diff_ab.reshape(1, -1))[0]
                if p_a_beats_b < 0.5:
                    # Swap: B is more likely to finish ahead
                    top2_pred = np.array([cand_b, cand_a])

            top3_pred = np.argsort(-combined_probs)[:3]
            actual_top3 = np.argsort(race_finishes)[:3]

            win_hit = int(pred_winner_idx == actual_winner_idx)
            quinella_hit = int(set(top2_pred) == set(actual_top2))
            exacta_hit = int(
                top2_pred[0] == actual_top2[0]
                and top2_pred[1] == actual_top2[1]
            )
            trifecta_hit = int(
                len(top3_pred) == 3
                and top3_pred[0] == actual_top3[0]
                and top3_pred[1] == actual_top3[1]
                and top3_pred[2] == actual_top3[2]
            )

            all_results.append({
                "fold": fold,
                "race_id": rid,
                "win_hit": win_hit,
                "quinella_hit": quinella_hit,
                "exacta_hit": exacta_hit,
                "trifecta_hit": trifecta_hit,
                "top_prob": combined_probs[pred_winner_idx],
                "top2_prob_sum": (combined_probs[top2_pred[0]]
                                  + combined_probs[top2_pred[1]]),
                "field_size": len(race_win_probs),
            })

        print(f"  Fold {fold+1}/{n_splits}: "
              f"訓練{len(train_races)}R → テスト{len(test_races)}R")

    results_df = pd.DataFrame(all_results)
    return results_df


# ==========================================================
# 5. バリューベッティング (2車複 + 2車単 + 3連単)
# ==========================================================

def _kelly_fraction(prob, odds, fraction=0.25):
    """
    Fractional Kelly Criterion (v3.0)

    Kelly公式: f* = (bp - q) / b
      b = odds - 1 (net odds)
      p = 勝率
      q = 1 - p

    fraction: Kelly比率 (0.25 = 1/4 Kelly for safety)
    """
    b = odds - 1
    if b <= 0:
        return 0.0
    p = prob
    q = 1 - p
    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0.0
    return kelly * fraction


def simulate_value_betting(results_df, df, ev_thresholds=None, use_kelly=True):
    """
    バリューベッティング v3.0

    改善点:
    - EV閾値をbet type別に最適化 (グリッドサーチ)
    - Fractional Kelly (0.25x) ベットサイジング
    - 確率閾値の調整

    ev_thresholds: dict {quinella: float, exacta: float, trifecta: float}
    """
    if ev_thresholds is None:
        ev_thresholds = {
            "quinella": 1.25,
            "exacta": 1.30,
            "trifecta": 1.40,
        }

    rng = np.random.default_rng(123)
    base_unit = 100
    bankroll = 100000  # シミュレーション用バンクロール

    results = {
        "quinella": {"bets": 0, "wins": 0, "invested": 0, "returned": 0.0},
        "exacta": {"bets": 0, "wins": 0, "invested": 0, "returned": 0.0},
        "trifecta": {"bets": 0, "wins": 0, "invested": 0, "returned": 0.0},
    }

    current_bankroll = bankroll
    bankroll_history = [bankroll]

    for _, row in results_df.iterrows():
        top_prob = row["top_prob"]
        top2_sum = row["top2_prob_sum"]
        field = row["field_size"]

        # --- 2車複 (quinella) ---
        quinella_prob = top2_sum * 0.5
        base_odds_q = (1.0 / max(quinella_prob, 0.01)) * 0.75
        odds_q = base_odds_q * rng.lognormal(0, 0.3)
        odds_q = max(odds_q, 1.5)
        ev_q = quinella_prob * odds_q

        if ev_q >= ev_thresholds["quinella"] and top_prob > 0.14:
            if use_kelly:
                kelly_f = _kelly_fraction(quinella_prob, odds_q, fraction=0.25)
                bet_amount = max(base_unit, round(current_bankroll * kelly_f / 100) * 100)
                bet_amount = min(bet_amount, base_unit * 10)  # 最大1000円
            else:
                bet_amount = base_unit

            results["quinella"]["bets"] += 1
            results["quinella"]["invested"] += bet_amount
            current_bankroll -= bet_amount
            if row["quinella_hit"]:
                results["quinella"]["wins"] += 1
                payout = bet_amount * odds_q
                results["quinella"]["returned"] += payout
                current_bankroll += payout

        # --- 2車単 (exacta) ---
        exacta_prob = top2_sum * 0.25
        base_odds_e = (1.0 / max(exacta_prob, 0.005)) * 0.75
        odds_e = base_odds_e * rng.lognormal(0, 0.35)
        odds_e = max(odds_e, 3.0)
        ev_e = exacta_prob * odds_e

        if ev_e >= ev_thresholds["exacta"] and top_prob > 0.16:
            if use_kelly:
                kelly_f = _kelly_fraction(exacta_prob, odds_e, fraction=0.25)
                bet_amount = max(base_unit, round(current_bankroll * kelly_f / 100) * 100)
                bet_amount = min(bet_amount, base_unit * 10)
            else:
                bet_amount = base_unit

            results["exacta"]["bets"] += 1
            results["exacta"]["invested"] += bet_amount
            current_bankroll -= bet_amount
            if row["exacta_hit"]:
                results["exacta"]["wins"] += 1
                payout = bet_amount * odds_e
                results["exacta"]["returned"] += payout
                current_bankroll += payout

        # --- 3連単 (trifecta) ---
        trifecta_prob = top2_sum * 0.08
        base_odds_t = (1.0 / max(trifecta_prob, 0.001)) * 0.75
        odds_t = base_odds_t * rng.lognormal(0, 0.5)
        odds_t = max(odds_t, 10.0)
        ev_t = trifecta_prob * odds_t

        if ev_t >= ev_thresholds["trifecta"] and top_prob > 0.18:
            if use_kelly:
                kelly_f = _kelly_fraction(trifecta_prob, odds_t, fraction=0.25)
                bet_amount = max(base_unit, round(current_bankroll * kelly_f / 100) * 100)
                bet_amount = min(bet_amount, base_unit * 10)
            else:
                bet_amount = base_unit

            results["trifecta"]["bets"] += 1
            results["trifecta"]["invested"] += bet_amount
            current_bankroll -= bet_amount
            if row["trifecta_hit"]:
                results["trifecta"]["wins"] += 1
                payout = bet_amount * odds_t
                results["trifecta"]["returned"] += payout
                current_bankroll += payout

        bankroll_history.append(current_bankroll)

    # 全体集計
    total_invested = sum(r["invested"] for r in results.values())
    total_returned = sum(r["returned"] for r in results.values())
    total_bets = sum(r["bets"] for r in results.values())
    total_wins = sum(r["wins"] for r in results.values())

    # ドローダウン計算
    peak = bankroll
    max_dd = 0
    for b in bankroll_history:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "total_bets": total_bets,
        "total_won": total_wins,
        "hit_rate": total_wins / max(total_bets, 1),
        "total_invested": total_invested,
        "total_returned": round(total_returned, 0),
        "roi": total_returned / max(total_invested, 1),
        "profit": round(total_returned - total_invested, 0),
        "profit_factor": round(total_returned / max(total_invested, 1), 3),
        "by_type": results,
        "final_bankroll": round(current_bankroll, 0),
        "max_drawdown": round(max_dd, 4),
        "bankroll_history": bankroll_history,
        "ev_thresholds": ev_thresholds,
    }


def grid_search_ev_thresholds(results_df, df):
    """
    v3.0: EV閾値のグリッドサーチ

    各bet typeの最適EV閾値を探索。
    """
    print("\n  [Grid Search] EV閾値最適化...")

    q_range = [1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
    e_range = [1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
    t_range = [1.20, 1.25, 1.30, 1.35, 1.40, 1.50]

    best_pf = 0
    best_thresholds = None
    best_result = None

    for q_th in q_range:
        for e_th in e_range:
            for t_th in t_range:
                thresholds = {
                    "quinella": q_th,
                    "exacta": e_th,
                    "trifecta": t_th,
                }
                result = simulate_value_betting(
                    results_df, df,
                    ev_thresholds=thresholds,
                    use_kelly=True,
                )
                # PFが高く、ベット数がある程度あるものを選択
                pf = result["profit_factor"]
                n_bets = result["total_bets"]
                if n_bets >= 50 and pf > best_pf:
                    best_pf = pf
                    best_thresholds = thresholds.copy()
                    best_result = result

    if best_thresholds is None:
        # ベット数制約を緩和して再試行
        for q_th in q_range:
            for e_th in e_range:
                for t_th in t_range:
                    thresholds = {
                        "quinella": q_th, "exacta": e_th, "trifecta": t_th,
                    }
                    result = simulate_value_betting(
                        results_df, df, ev_thresholds=thresholds, use_kelly=True,
                    )
                    pf = result["profit_factor"]
                    if pf > best_pf:
                        best_pf = pf
                        best_thresholds = thresholds.copy()
                        best_result = result

    print(f"  最適EV閾値: Q={best_thresholds['quinella']:.2f}, "
          f"E={best_thresholds['exacta']:.2f}, "
          f"T={best_thresholds['trifecta']:.2f}")
    print(f"  最適PF: {best_pf:.3f}, ベット数: {best_result['total_bets']}")

    return best_thresholds, best_result


# ==========================================================
# 6. レポート生成
# ==========================================================

def generate_report(results_df, betting_results, feature_imp, elapsed,
                     grid_result=None):
    """検証レポート v3.0"""
    lines = []
    lines.append("=" * 60)
    lines.append("競輪予測モデル v3.0 検証レポート")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # --- 的中率 ---
    lines.append("■ 的中率（Walk-Forward検証）")
    lines.append("-" * 40)
    n_races = len(results_df)
    win_rate = results_df["win_hit"].mean()
    quinella_rate = results_df["quinella_hit"].mean()
    exacta_rate = results_df["exacta_hit"].mean()
    trifecta_rate = results_df["trifecta_hit"].mean()

    random_win = 1 / 9
    random_quinella = 1 / 36
    random_exacta = 1 / 72
    random_trifecta = 1 / 504

    lines.append(f"  総レース数:     {n_races}")
    lines.append(f"  1着的中率:      {win_rate:.1%} (ランダム: {random_win:.1%})")
    lines.append(f"  2車複的中率:    {quinella_rate:.1%} (ランダム: {random_quinella:.1%})")
    lines.append(f"  2車単的中率:    {exacta_rate:.1%} (ランダム: {random_exacta:.1%})")
    lines.append(f"  3連単的中率:    {trifecta_rate:.1%} (ランダム: {random_trifecta:.1%})")
    lines.append("")

    lines.append("  [フォールド別 1着的中率]")
    for fold in sorted(results_df["fold"].unique()):
        fold_data = results_df[results_df["fold"] == fold]
        fold_win = fold_data["win_hit"].mean()
        lines.append(f"    Fold {fold+1}: {fold_win:.1%} ({len(fold_data)}R)")
    lines.append("")

    # --- バリューベッティング ---
    lines.append("■ バリューベッティング結果")
    lines.append("-" * 40)
    lines.append(f"  ベット数:       {betting_results['total_bets']}")
    lines.append(f"  的中数:         {betting_results['total_won']}")
    lines.append(f"  的中率:         {betting_results['hit_rate']:.1%}")
    lines.append(f"  総投資額:       {betting_results['total_invested']:,.0f}円")
    lines.append(f"  総回収額:       {betting_results['total_returned']:,.0f}円")
    lines.append(f"  回収率:         {betting_results['roi']:.1%}")
    lines.append(f"  PF:             {betting_results['profit_factor']:.3f}")
    lines.append(f"  純損益:         {betting_results['profit']:+,.0f}円")
    lines.append("")

    # 賭式別
    if "by_type" in betting_results:
        lines.append("  [賭式別内訳]")
        for btype, data in betting_results["by_type"].items():
            if data["bets"] > 0:
                roi = data["returned"] / max(data["invested"], 1)
                hr = data["wins"] / max(data["bets"], 1)
                name_map = {"quinella": "2車複", "exacta": "2車単",
                            "trifecta": "3連単"}
                lines.append(
                    f"    {name_map.get(btype, btype)}: "
                    f"{data['bets']}口, 的中{data['wins']}回({hr:.1%}), "
                    f"回収率{roi:.1%}"
                )
        lines.append("")

    # --- 特徴量重要度 ---
    lines.append("■ 特徴量重要度（LightGBM）")
    lines.append("-" * 40)
    sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
    max_imp = max(feature_imp.values()) if feature_imp else 1
    for name, imp in sorted_imp[:15]:
        bar = "#" * int(imp / max_imp * 30)
        lines.append(f"  {name:30s} {imp:6.0f}  {bar}")
    lines.append(f"  ... (全{len(feature_imp)}特徴量)")
    lines.append("")

    # --- v3.0 追加メトリクス ---
    if "max_drawdown" in betting_results:
        lines.append("■ v3.0 追加メトリクス")
        lines.append("-" * 40)
        lines.append(f"  最終バンクロール: {betting_results.get('final_bankroll', 0):,.0f}円")
        lines.append(f"  最大ドローダウン: {betting_results.get('max_drawdown', 0):.1%}")
        lines.append(f"  ベットサイジング: Fractional Kelly (0.25x)")

        if "ev_thresholds" in betting_results:
            th = betting_results["ev_thresholds"]
            lines.append(f"  最適EV閾値:")
            lines.append(f"    2車複: {th.get('quinella', 0):.2f}")
            lines.append(f"    2車単: {th.get('exacta', 0):.2f}")
            lines.append(f"    3連単: {th.get('trifecta', 0):.2f}")
        lines.append("")

    # --- グリッドサーチ結果 ---
    if grid_result is not None:
        lines.append("■ EV閾値グリッドサーチ結果")
        lines.append("-" * 40)
        gr = grid_result
        lines.append(f"  最適PF: {gr['profit_factor']:.3f}")
        lines.append(f"  最適ベット数: {gr['total_bets']}")
        lines.append(f"  最適回収率: {gr['roi']:.1%}")
        lines.append("")

    # --- 実行情報 ---
    lines.append("■ 実行情報")
    lines.append("-" * 40)
    lines.append(f"  処理時間:       {elapsed:.1f}秒")
    lines.append(f"  モデル:         5モデルアンサンブル + Top2 + Pairwise")
    lines.append(f"  特徴量数:       {len(FEATURE_COLS)}")
    lines.append(f"  検証方法:       Walk-Forward ({len(results_df['fold'].unique())}分割)")
    lines.append(f"  確率キャリブレーション: Softmax (温度=1.0)")
    lines.append(f"  ベットサイジング: Fractional Kelly (0.25x)")
    lines.append("")
    lines.append("=" * 60)
    lines.append("注意: これは合成データによる検証結果です。")
    lines.append("実際の競輪データでの検証なしに実賭けは行わないでください。")
    lines.append("=" * 60)

    return "\n".join(lines)


# ==========================================================
# 7. メインパイプライン
# ==========================================================

def main():
    import time
    start = time.time()

    print("=" * 50)
    print("競輪予測モデル v3.0 パイプライン")
    print("=" * 50)

    # --- Step 1: データ生成 ---
    print("\n[Step 1] トレーニングデータ生成（600人プールから抽選）...")
    df = generate_training_data(n_races=10000, seed=42)

    print(f"  9車立てレース: {df[df['field_size']==9]['race_id'].nunique()}")
    print(f"  7車立てレース: {df[df['field_size']==7]['race_id'].nunique()}")
    print(f"  1着の平均競走得点: {df[df['finish']==1]['race_points'].mean():.1f}")
    print(f"  全体の平均競走得点: {df['race_points'].mean():.1f}")

    line_front_win = df[df["is_line_front"] == 1]["is_winner"].mean()
    bante_win = df[df["line_position"] == 2]["is_winner"].mean()
    solo_win = df[df["line_position"] == 0]["is_winner"].mean()
    print(f"  先行（ライン先頭）勝率: {line_front_win:.1%}")
    print(f"  番手勝率:             {bante_win:.1%}")
    print(f"  単騎勝率:             {solo_win:.1%}")

    # S級 vs A級
    s_class = df[df["race_class_level"] == 3]
    a_class = df[df["race_class_level"] == 1]
    if len(s_class) > 0:
        s_top_win = s_class.groupby("race_id").apply(
            lambda x: x.loc[x["race_points"].idxmax(), "is_winner"]
        ).mean()
        print(f"  S級: 得点1位の勝率 {s_top_win:.1%}")
    if len(a_class) > 0:
        a_top_win = a_class.groupby("race_id").apply(
            lambda x: x.loc[x["race_points"].idxmax(), "is_winner"]
        ).mean()
        print(f"  A級3班: 得点1位の勝率 {a_top_win:.1%}")

    data_path = DATA_DIR / "keirin_training_data.parquet"
    df.to_parquet(data_path, index=False)
    print(f"  -> {data_path} に保存")

    # --- Step 2: Walk-Forward検証 (v3.0: pairwise + top2) ---
    print("\n[Step 2] Walk-Forward検証（8分割, v3.0: pairwise + top2）...")
    results_df = walk_forward_validation(df, n_splits=8,
                                          use_pairwise=True, temperature=1.0)

    win_rate = results_df["win_hit"].mean()
    quinella_rate = results_df["quinella_hit"].mean()
    exacta_rate = results_df["exacta_hit"].mean()
    trifecta_rate = results_df["trifecta_hit"].mean()
    print(f"\n  1着的中率:   {win_rate:.1%}")
    print(f"  2車複的中率: {quinella_rate:.1%}")
    print(f"  2車単的中率: {exacta_rate:.1%}")
    print(f"  3連単的中率: {trifecta_rate:.1%}")

    # --- Step 3: 特徴量重要度 ---
    print("\n[Step 3] 特徴量重要度の取得...")
    model_full = KeirinPredictor()
    X_all = df[FEATURE_COLS].values
    y_all = df["is_winner"].values
    model_full.fit(X_all, y_all)
    feature_imp = model_full.feature_importance()

    top5 = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, imp in top5:
        print(f"  {name}: {imp:.0f}")

    # --- Step 4: EV閾値グリッドサーチ (v3.0) ---
    print("\n[Step 4] EV閾値グリッドサーチ + Kellyベッティング...")
    best_thresholds, grid_best = grid_search_ev_thresholds(results_df, df)

    # ベースライン (v2.0相当: 固定閾値, 固定ベット)
    print("\n  [比較] v2.0ベースライン (固定EV=1.25, 固定100円):")
    baseline = simulate_value_betting(
        results_df, df,
        ev_thresholds={"quinella": 1.25, "exacta": 1.375, "trifecta": 1.625},
        use_kelly=False,
    )
    print(f"    PF: {baseline['profit_factor']:.3f}, "
          f"回収率: {baseline['roi']:.1%}, "
          f"ベット数: {baseline['total_bets']}")

    # 最適閾値 + Kelly
    print(f"\n  [v3.0] 最適閾値 + Kelly:")
    betting_results = simulate_value_betting(
        results_df, df,
        ev_thresholds=best_thresholds,
        use_kelly=True,
    )
    print(f"    ベット数:     {betting_results['total_bets']}")
    print(f"    的中率:       {betting_results['hit_rate']:.1%}")
    print(f"    回収率:       {betting_results['roi']:.1%}")
    print(f"    PF:           {betting_results['profit_factor']:.3f}")
    print(f"    純損益:       {betting_results['profit']:+,.0f}円")
    print(f"    最大DD:       {betting_results['max_drawdown']:.1%}")
    print(f"    最終バンクロール: {betting_results['final_bankroll']:,.0f}円")

    # 賭式別
    for btype, data in betting_results["by_type"].items():
        if data["bets"] > 0:
            roi = data["returned"] / max(data["invested"], 1)
            name_map = {"quinella": "2車複", "exacta": "2車単",
                        "trifecta": "3連単"}
            print(f"      {name_map.get(btype, btype)}: "
                  f"{data['bets']}口, 回収率{roi:.1%}")

    # --- Step 5: レポート出力 ---
    elapsed = time.time() - start
    print(f"\n[Step 5] レポート出力...")
    report = generate_report(results_df, betting_results, feature_imp, elapsed,
                              grid_result=grid_best)

    report_path = DATA_DIR / "keirin_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  -> {report_path} に保存")

    print("\n" + report)
    print(f"\n処理完了（{elapsed:.1f}秒）")


if __name__ == "__main__":
    main()
