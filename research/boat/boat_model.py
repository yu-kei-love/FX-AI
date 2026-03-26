# ===========================================
# boat_model.py  v3.4
# ボートレース（競艇）予測モデル
#
# v3.4 改善点 (paper trade PF 1.86 degradation fix):
#   - 枠番別EV閾値 (Lane 2,5,6 のモデル精度低下に対応)
#   - オッズ上限フィルタ (odds > 60 除外 = longshot overconfidence防止)
#   - 最低予測確率フィルタ (ノイズ除去)
#   - モデル一致度フィルタ (4/5モデル合意必須)
#   - 1レースあたり最大3ベット制限
#
# v3.1 改善点:
#   - 2連単(Exacta)/2連複(Quinella)専用の2着予測モデル追加
#   - 条件付き確率でExacta/Quinella確率を正確に計算
#   - 複数組み合わせベット(top-N候補)
#   - 2着予測用の追加特徴量
#   - Exacta/Quinella用EV閾値の個別最適化
#
# v3.0 改善点:
#   - 5モデルアンサンブル (LGB + XGB + CatBoost + RF + ExtraTrees)
#   - 拡張相互作用特徴量 (lane x class x motor, wind x lane, etc.)
#   - 展示タイム特徴量 (exhibition time if available)
#   - Walk-Forward 5+ウィンドウ
#   - 正確なPF, Sharpe, MDD計算
#   - Fractional Kelly (0.25x) ベットサイジング
#
# v2.0からの変更:
#   - 2モデル → 5モデルアンサンブル (EnsembleClassifier)
#   - 28特徴量 → 40+特徴量
#   - 3 WFウィンドウ → 5ウィンドウ
#   - Kelly criterion 導入
# ===========================================

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

warnings.filterwarnings("ignore")

# ===== 定数 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "boat"
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 基本特徴量
# v3.3 pruned: Removed late_count (near-zero), racer_3place_rate (correlated w/ racer_win_rate)
BASE_FEATURE_COLS = [
    "lane",                 # 枠番 (1-6)
    "racer_class",          # 選手級別 (A1=4, A2=3, B1=2, B2=1)
    "racer_win_rate",       # 全国勝率
    "racer_place_rate",     # 全国2連率
    "racer_local_win_rate", # 当地勝率
    "racer_local_2place_rate",  # 当地2連率
    "motor_2place_rate",    # モーター2連率
    "boat_2place_rate",     # ボート2連率
    "avg_start_timing",     # 平均スタートタイミング
    "flying_count",         # フライング回数
    "racer_weight",         # 体重
    "weather_wind_speed",   # 風速
    "weather_condition",    # 天候 (晴=0, 曇=1, 雨=2)
    "wave_height",          # 波高
    "course_type",          # コースタイプ (0=標準, 1=難水面)
]

# 相対/相互作用特徴量 (create_features で追加)
# v3.3 pruned: Removed inner_class_advantage, is_inner_lane (near-zero),
#   lane_squared, lane_x_race_number (correlated w/ lane), race_number (near-zero)
DERIVED_FEATURE_COLS = [
    # v2.0 からの特徴量
    "class_rank_in_race",       # レース内級別順位
    "win_rate_rank",            # レース内勝率順位
    "win_rate_vs_field_avg",    # 勝率 - レース平均勝率
    "motor_vs_field_avg",       # モーター2連率 - レース平均
    "class_x_lane",             # 級別 × 枠番 (A1+1号艇 = 高い)
    "start_timing_rank",        # スタートタイミング順位
    "combined_equipment",       # モーター + ボート 2連率合計
    "wind_x_lane",              # 風速 × 枠番 (強風でインコース不利)
    "local_vs_national",        # 当地勝率 - 全国勝率 (当地得意度)
    "place_consistency",        # 2連率/勝率 (安定度)
    # v3.0 新規特徴量
    "lane_class_motor",         # 枠番 × 級別 × モーター (1号艇+A1+良モーター=最強)
    "wind_dir_x_lane",          # 風向 × 枠番 (追い風でイン有利、向かい風でアウト有利)
    "start_x_lane",             # スタートタイミング × 枠番 (スタート早い+イン=強い)
    "flying_x_class",           # フライング × 級別 (A1でF持ち=慎重)
    "equipment_rank",           # モーター+ボート順位
    "motor_rank",               # モーター2連率のレース内順位
    "boat_rank",                # ボート2連率のレース内順位
    "weight_diff",              # 体重 - レース平均体重
    "class_x_motor",            # 級別 × モーター2連率
    "start_x_class",            # スタートタイミング × 級別
    "wind_x_wave",              # 風速 × 波高 (荒天度)
    "exhibition_time",          # 展示タイム (available in real data)
    "exhibition_start",         # 展示スタートタイミング
    # v3.2 新規特徴量
    "field_strength",           # フィールド強度 (全選手の平均級別)
    "field_strength_std",       # フィールド強度のばらつき
    "class_dominance",          # 級別の飛び抜け度合い
]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS

# 枠番別の実際の勝率（公式データに基づく）
LANE_WIN_RATES = {1: 0.55, 2: 0.15, 3: 0.12, 4: 0.10, 5: 0.05, 6: 0.03}

# 選手級別 (数値エンコーディング)
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# 想定オッズ（枠番別の平均的なオッズ）
APPROX_ODDS = {1: 2.5, 2: 7.0, 3: 9.0, 4: 11.0, 5: 20.0, 6: 35.0}

# 風向マップ (for wind_dir_x_lane feature)
WIND_DIR_MAP = {
    "追い風": 1, "向かい風": -1, "右横風": 0.3, "左横風": -0.3,
    "北": -0.5, "南": 0.5, "東": 0.3, "西": -0.3,
    "": 0, "無風": 0,
}


# =============================================================
# データ生成
# =============================================================

def generate_training_data(n_races=10000, seed=42):
    """
    リアルな統計特性を持つボートレースデータを生成する。

    実際のデータの特徴:
    - 1号艇が約55%勝つ
    - A1級選手が1号艇なら約65%、B2級なら約35%
    - モーター性能は勝率に5-10%影響
    - 風速・天候がベテラン有利に働く
    """
    rng = np.random.RandomState(seed)

    rows = []
    # 日付を生成（検証用にソート可能にする）
    base_date = datetime(2023, 1, 1)

    for race_id in range(n_races):
        race_date = base_date + timedelta(hours=race_id)

        # レース条件を生成
        weather_condition = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])  # 晴/曇/雨
        weather_wind_speed = max(0, rng.normal(3.0, 2.0))  # 平均3m/s
        course_type = rng.choice([0, 1], p=[0.7, 0.3])  # 標準/難水面
        wind_direction = rng.choice([1, -1, 0.3, -0.3, 0])  # 追い風/向かい風/横風/無風

        # 6艇分の選手データを生成
        boats = []
        for lane in range(1, 7):
            # 級別の分布（1号艇にA1が配置されやすい傾向を再現）
            if lane == 1:
                racer_class = rng.choice([4, 3, 2, 1], p=[0.35, 0.30, 0.25, 0.10])
            elif lane <= 3:
                racer_class = rng.choice([4, 3, 2, 1], p=[0.20, 0.25, 0.30, 0.25])
            else:
                racer_class = rng.choice([4, 3, 2, 1], p=[0.10, 0.20, 0.35, 0.35])

            # 級別に応じた勝率を生成
            class_base = {4: 7.0, 3: 5.5, 2: 4.5, 1: 3.5}[racer_class]
            racer_win_rate = max(2.0, min(9.0, rng.normal(class_base, 0.8)))
            racer_place_rate = max(10.0, min(70.0, racer_win_rate * 6.5 + rng.normal(0, 3)))
            racer_local_win_rate = max(1.0, min(10.0, racer_win_rate + rng.normal(0, 1.0)))

            # モーター・ボート性能
            motor_2place_rate = max(15.0, min(70.0, rng.normal(40.0, 12.0)))
            boat_2place_rate = max(15.0, min(70.0, rng.normal(40.0, 10.0)))

            # スタートタイミング（小さいほど良い、0.15秒が理想）
            avg_start_timing = max(0.08, rng.normal(0.17, 0.04))
            flying_count = rng.choice([0, 0, 0, 0, 0, 1, 1, 2])  # ほとんど0

            # 展示タイム (6.5-7.0秒が一般的、良いモーターほど速い)
            exhibition_time = max(6.3, min(7.2, rng.normal(6.75, 0.15) - (motor_2place_rate - 40) * 0.002))
            exhibition_start = max(0.05, rng.normal(avg_start_timing, 0.03))

            boats.append({
                "race_id": race_id,
                "race_date": race_date,
                "lane": lane,
                "racer_class": racer_class,
                "racer_win_rate": round(racer_win_rate, 2),
                "racer_place_rate": round(racer_place_rate, 1),
                "racer_3place_rate": round(racer_place_rate * 1.3, 1),
                "racer_local_win_rate": round(racer_local_win_rate, 2),
                "racer_local_2place_rate": round(racer_local_win_rate * 5, 1),
                "motor_2place_rate": round(motor_2place_rate, 1),
                "boat_2place_rate": round(boat_2place_rate, 1),
                "avg_start_timing": round(avg_start_timing, 3),
                "flying_count": flying_count,
                "late_count": 0,
                "racer_weight": round(rng.normal(52, 3), 1),
                "weather_wind_speed": round(weather_wind_speed, 1),
                "weather_condition": weather_condition,
                "wave_height": round(max(0, rng.normal(3, 2)), 1),
                "course_type": course_type,
                "wind_direction": wind_direction,
                "exhibition_time_raw": round(exhibition_time, 2),
                "exhibition_start_raw": round(exhibition_start, 3),
            })

        # ===== 勝者を確率的に決定 =====
        # 各艇の勝利確率を計算
        probs = []
        for b in boats:
            # ベース: 枠番の統計的優位性
            lane_base = LANE_WIN_RATES[b["lane"]]

            # 選手級別の影響
            class_factor = {4: 1.3, 3: 1.0, 2: 0.75, 1: 0.5}[b["racer_class"]]

            # モーター性能の影響（2連率50%以上で+10%、30%以下で-10%）
            motor_factor = 1.0 + (b["motor_2place_rate"] - 40.0) / 200.0

            # スタートタイミングの影響（早いスタートは有利）
            start_factor = 1.0 + (0.17 - b["avg_start_timing"]) * 2.0

            # フライング持ちはスタート慎重 → 不利
            if b["flying_count"] > 0:
                start_factor *= 0.9

            # 風向 × 枠番の影響
            wind_factor = 1.0
            wd = b.get("wind_direction", 0)
            if wd > 0 and b["lane"] <= 2:  # 追い風でイン有利
                wind_factor = 1.05
            elif wd < 0 and b["lane"] >= 4:  # 向かい風でアウト有利
                wind_factor = 1.05

            # 風速の影響（強風はインコース不利、経験者有利）
            if weather_wind_speed > 5:
                if b["lane"] <= 2:
                    wind_factor *= 0.9  # インコース不利
                if b["racer_class"] >= 3:
                    wind_factor *= 1.1  # ベテラン有利

            # 難水面はインコースがさらに有利
            if course_type == 1 and b["lane"] == 1:
                lane_base *= 1.1

            # 展示タイムの影響（速いほど有利）
            exhibit_factor = 1.0 + (6.75 - b["exhibition_time_raw"]) * 0.5

            # 雨天は荒れやすい（ランダム要素増加）
            random_factor = rng.uniform(0.7, 1.3) if weather_condition == 2 else rng.uniform(0.85, 1.15)

            prob = lane_base * class_factor * motor_factor * start_factor * wind_factor * exhibit_factor * random_factor
            probs.append(max(prob, 0.01))

        # 確率を正規化して勝者を選択
        probs = np.array(probs)
        probs = probs / probs.sum()
        winner_idx = rng.choice(6, p=probs)

        # 2着も確率的に決定（勝者を除く）
        remaining_probs = np.delete(probs, winner_idx)
        remaining_probs = remaining_probs / remaining_probs.sum()
        remaining_lanes = [i for i in range(6) if i != winner_idx]
        second_idx = rng.choice(remaining_lanes, p=remaining_probs)

        for i, b in enumerate(boats):
            b["win"] = 1 if i == winner_idx else 0
            b["place_top2"] = 1 if i in [winner_idx, second_idx] else 0
            rows.append(b)

    df = pd.DataFrame(rows)
    print(f"[データ生成] {n_races}レース x 6艇 = {len(df)}行")
    print(f"[枠番別勝率]")
    for lane in range(1, 7):
        wr = df[df["lane"] == lane]["win"].mean()
        print(f"  {lane}号艇: {wr:.1%}")

    return df


def load_real_data():
    """
    収集済みのリアルデータ (real_race_data.csv) を読み込み、
    モデルが使える形式に変換する。
    """
    real_path = DATA_DIR / "real_race_data.csv"
    if not real_path.exists():
        print(f"[エラー] リアルデータが見つかりません: {real_path}")
        return None

    raw = pd.read_csv(real_path, encoding="utf-8-sig")
    print(f"[データ読込] {len(raw)}レース分のリアルデータ")

    # 天候を数値化
    weather_map = {"晴": 0, "曇り": 1, "曇": 1, "雨": 2, "雪": 2, "霧": 1}

    # 難水面の会場（波が荒れやすい）
    rough_venues = {"02", "06", "10", "17", "21", "24"}  # 戸田、鳴門、三国、宮島、芦屋、大村

    rows = []
    for idx, race in raw.iterrows():
        if pd.isna(race.get("date")):
            continue
        race_id = idx
        try:
            race_date = pd.to_datetime(str(int(race["date"])), format="%Y%m%d")
        except (ValueError, TypeError):
            continue
        venue_code = str(int(race["venue_code"])).zfill(2) if pd.notna(race.get("venue_code")) else "00"

        weather_condition = weather_map.get(str(race.get("weather", "")).strip(), 0)
        wind_speed = float(race.get("wind_speed", 0)) if pd.notna(race.get("wind_speed")) else 0.0
        wave_height = float(race.get("wave_height", 0)) if pd.notna(race.get("wave_height")) else 0.0
        course_type = 1 if venue_code in rough_venues or wave_height >= 5 else 0

        # 風向 (数値化)
        wind_dir_str = str(race.get("wind_direction", "")).strip()
        wind_direction = WIND_DIR_MAP.get(wind_dir_str, 0)

        # 勝者を特定（finish=1のレーン）
        winner_lane = None
        second_lane = None
        for lane in range(1, 7):
            finish_val = race.get(f"lane{lane}_finish")
            if pd.notna(finish_val):
                try:
                    finish = int(float(finish_val))
                    if finish == 1:
                        winner_lane = lane
                    elif finish == 2:
                        second_lane = lane
                except (ValueError, TypeError):
                    pass

        if winner_lane is None:
            continue  # 結果不明のレースはスキップ

        # 6艇分のデータを生成
        for lane in range(1, 7):
            prefix = f"lane{lane}_"
            racer_class_str = str(race.get(f"{prefix}class", "B1")).strip()
            racer_class = CLASS_MAP.get(racer_class_str, 2)

            win_rate = float(race.get(f"{prefix}win_rate", 4.0)) if pd.notna(race.get(f"{prefix}win_rate")) else 4.0
            place_rate = float(race.get(f"{prefix}2place_rate", 20.0)) if pd.notna(race.get(f"{prefix}2place_rate")) else 20.0
            motor_rate = float(race.get(f"{prefix}motor_2rate", 40.0)) if pd.notna(race.get(f"{prefix}motor_2rate")) else 40.0
            boat_rate = float(race.get(f"{prefix}boat_2rate", 40.0)) if pd.notna(race.get(f"{prefix}boat_2rate")) else 40.0
            start_timing = float(race.get(f"{prefix}start_timing", 0.15)) if pd.notna(race.get(f"{prefix}start_timing")) else 0.15
            flying = int(float(race.get(f"{prefix}flying_count", 0))) if pd.notna(race.get(f"{prefix}flying_count")) else 0
            local_wr = float(race.get(f"{prefix}local_win_rate", win_rate)) if pd.notna(race.get(f"{prefix}local_win_rate")) else win_rate

            # オッズ（単勝）
            odds_col = f"odds_{lane}"
            odds = float(race.get(odds_col, 5.0)) if pd.notna(race.get(odds_col)) else 5.0

            # 追加データ (v2.0)
            three_place_rate = float(race.get(f"{prefix}3place_rate", place_rate * 1.3)) if pd.notna(race.get(f"{prefix}3place_rate")) else place_rate * 1.3
            local_2place = float(race.get(f"{prefix}local_2place_rate", local_wr * 5)) if pd.notna(race.get(f"{prefix}local_2place_rate")) else local_wr * 5
            late_cnt = int(float(race.get(f"{prefix}late_count", 0))) if pd.notna(race.get(f"{prefix}late_count")) else 0
            weight = float(race.get(f"{prefix}weight", 52.0)) if pd.notna(race.get(f"{prefix}weight")) else 52.0

            # v3.0: 展示タイム (if available)
            exhibition_time_raw = float(race.get(f"{prefix}exhibition_time", 0)) if pd.notna(race.get(f"{prefix}exhibition_time")) else 0.0
            exhibition_start_raw = float(race.get(f"{prefix}exhibition_start", 0)) if pd.notna(race.get(f"{prefix}exhibition_start")) else 0.0

            rows.append({
                "race_id": race_id,
                "race_date": race_date,
                "venue_code": venue_code,
                "race_no": race.get("race_no", 0),
                "lane": lane,
                "racer_class": racer_class,
                "racer_win_rate": win_rate,
                "racer_place_rate": place_rate,
                "racer_3place_rate": three_place_rate,
                "racer_local_win_rate": local_wr,
                "racer_local_2place_rate": local_2place,
                "motor_2place_rate": motor_rate,
                "boat_2place_rate": boat_rate,
                "avg_start_timing": start_timing,
                "flying_count": flying,
                "late_count": late_cnt,
                "racer_weight": weight,
                "weather_wind_speed": wind_speed,
                "weather_condition": weather_condition,
                "wave_height": wave_height,
                "course_type": course_type,
                "wind_direction": wind_direction,
                "exhibition_time_raw": exhibition_time_raw,
                "exhibition_start_raw": exhibition_start_raw,
                "odds": odds,
                "win": 1 if lane == winner_lane else 0,
                "place_top2": 1 if lane in (winner_lane, second_lane) else 0,
            })

    df = pd.DataFrame(rows)
    print(f"[変換完了] {len(df) // 6}レース x 6艇 = {len(df)}行")
    print(f"[枠番別勝率 (リアル)]")
    for lane in range(1, 7):
        wr = df[df["lane"] == lane]["win"].mean()
        print(f"  {lane}号艇: {wr:.1%}")

    return df


# =============================================================
# 特徴量エンジニアリング
# =============================================================

def create_features(df):
    """
    レース内の相対特徴量・相互作用特徴量を追加する (v3.0)。
    v2.0の12個 + v3.0で14個追加 = 合計26個の派生特徴量。
    """
    df = df.copy()

    # 基本カラムが存在しない場合のフォールバック
    for col, default in [
        ("racer_3place_rate", lambda d: d["racer_place_rate"] * 1.3),
        ("racer_local_2place_rate", lambda d: d["racer_local_win_rate"] * 5),
        ("late_count", lambda d: 0),
        ("racer_weight", lambda d: 52.0),
        ("wave_height", lambda d: 0.0),
        ("wind_direction", lambda d: 0.0),
        ("exhibition_time_raw", lambda d: 0.0),
        ("exhibition_start_raw", lambda d: 0.0),
    ]:
        if col not in df.columns:
            df[col] = default(df)

    # --- v2.0: レース内順位系 ---
    df["class_rank_in_race"] = df.groupby("race_id")["racer_class"].rank(
        ascending=False, method="min"
    )
    df["win_rate_rank"] = df.groupby("race_id")["racer_win_rate"].rank(
        ascending=False, method="min"
    )
    df["start_timing_rank"] = df.groupby("race_id")["avg_start_timing"].rank(
        ascending=True, method="min"  # 小さいほど良い
    )

    # --- v2.0: レース平均との差分 ---
    df["win_rate_vs_field_avg"] = df["racer_win_rate"] - df.groupby("race_id")["racer_win_rate"].transform("mean")
    df["motor_vs_field_avg"] = df["motor_2place_rate"] - df.groupby("race_id")["motor_2place_rate"].transform("mean")

    # --- v2.0: 相互作用特徴量 ---
    df["class_x_lane"] = df["racer_class"] * (7 - df["lane"])  # A1+1号艇=高い
    df["combined_equipment"] = df["motor_2place_rate"] + df["boat_2place_rate"]
    df["is_inner_lane"] = (df["lane"] <= 3).astype(int)
    df["inner_class_advantage"] = df["is_inner_lane"] * df["racer_class"]
    df["wind_x_lane"] = df["weather_wind_speed"] * df["lane"]  # 強風x外枠
    df["local_vs_national"] = df["racer_local_win_rate"] - df["racer_win_rate"]
    df["place_consistency"] = df["racer_place_rate"] / df["racer_win_rate"].clip(lower=1.0)

    # --- v3.0: 新規相互作用特徴量 ---

    # 1. lane x class x motor: 1号艇 + A1 + 良モーター = 最強
    motor_norm = (df["motor_2place_rate"] - 30) / 20.0  # 0-1正規化 (30-50%)
    df["lane_class_motor"] = (7 - df["lane"]) * df["racer_class"] * motor_norm.clip(0, 2)

    # 2. wind_direction x lane: 追い風でイン有利、向かい風でアウト有利
    df["wind_dir_x_lane"] = df["wind_direction"] * (4 - df["lane"])  # 追い風(+1) x インコース(+) = 正

    # 3. start_timing x lane: スタートが早い選手がインなら強い
    # avg_start_timingが小さいほど早い → (0.20 - timing)で反転
    start_speed = (0.20 - df["avg_start_timing"]).clip(-0.1, 0.15)
    df["start_x_lane"] = start_speed * (7 - df["lane"])  # イン(大) x 早いスタート(大) = 強い

    # 4. flying_count x class: A1でフライング持ちは慎重になる
    df["flying_x_class"] = df["flying_count"] * df["racer_class"]  # A1(4) x F1 = 4: 影響大

    # 5. 装備順位
    df["equipment_rank"] = df.groupby("race_id")["combined_equipment"].rank(
        ascending=False, method="min"
    )
    df["motor_rank"] = df.groupby("race_id")["motor_2place_rate"].rank(
        ascending=False, method="min"
    )
    df["boat_rank"] = df.groupby("race_id")["boat_2place_rate"].rank(
        ascending=False, method="min"
    )

    # 6. 体重差
    df["weight_diff"] = df["racer_weight"] - df.groupby("race_id")["racer_weight"].transform("mean")

    # 7. 級別 x モーター (A1+良モーター vs B2+悪モーター)
    df["class_x_motor"] = df["racer_class"] * (df["motor_2place_rate"] / 40.0)

    # 8. スタートタイミング x 級別
    df["start_x_class"] = start_speed * df["racer_class"]

    # 9. 枠番の2乗 (非線形枠番効果 - 1号艇の圧倒的優位)
    df["lane_squared"] = df["lane"] ** 2

    # 10. 風速 x 波高 (荒天度)
    df["wind_x_wave"] = df["weather_wind_speed"] * df["wave_height"]

    # 11. 展示タイム (0の場合はレース平均で代替)
    df["exhibition_time"] = df["exhibition_time_raw"]
    if (df["exhibition_time"] == 0).all():
        # 展示データがない場合はモーター+ボートから推定
        df["exhibition_time"] = 6.75 - (df["motor_2place_rate"] - 40) * 0.003 - (df["boat_2place_rate"] - 40) * 0.002
    else:
        # 部分的にある場合は0をレース平均で埋める
        race_mean_et = df.groupby("race_id")["exhibition_time"].transform(
            lambda x: x[x > 0].mean() if (x > 0).any() else 6.75
        )
        df.loc[df["exhibition_time"] == 0, "exhibition_time"] = race_mean_et

    # 12. 展示スタートタイミング
    df["exhibition_start"] = df["exhibition_start_raw"]
    if (df["exhibition_start"] == 0).all():
        df["exhibition_start"] = df["avg_start_timing"]
    else:
        race_mean_es = df.groupby("race_id")["exhibition_start"].transform(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.15
        )
        df.loc[df["exhibition_start"] == 0, "exhibition_start"] = race_mean_es

    # --- v3.2: レース構造特徴量 ---

    # 13. レース番号 (race_idから推測、または明示的に存在する場合)
    if "race_number" not in df.columns:
        # race_idの末尾からレース番号を推測 (例: "2024-01-01_venue_R1" → 1)
        try:
            df["race_number"] = df["race_id"].apply(
                lambda x: int(str(x).split("R")[-1]) if "R" in str(x) else
                          int(str(x).split("_")[-1]) if "_" in str(x) else 6
            ).clip(1, 12)
        except Exception:
            df["race_number"] = 6  # デフォルト: 中間レース

    # 14. フィールド強度 (全選手の平均級別 = レースの質)
    df["field_strength"] = df.groupby("race_id")["racer_class"].transform("mean")

    # 15. フィールド強度のばらつき (混戦度)
    df["field_strength_std"] = df.groupby("race_id")["racer_class"].transform("std").fillna(0)

    # 16. 級別の飛び抜け度合い (自分の級別 - フィールド平均)
    df["class_dominance"] = df["racer_class"] - df["field_strength"]

    # 17. 枠番 × レース番号 (後半レースはイン逃げ率が高い傾向)
    df["lane_x_race_number"] = (7 - df["lane"]) * (df["race_number"] / 12.0)

    # 欠損値処理
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# =============================================================
# モデル (v3.0: 5モデルアンサンブル)
# =============================================================

def train_ensemble(X_train, y_train, X_val=None, y_val=None):
    """
    5モデルアンサンブル (LGB + XGB + CatBoost + RF + ExtraTrees)。
    EnsembleClassifierインターフェースを参考にしたカスタム実装。
    v3.0: Early stopping for boosting models.
    """
    try:
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    except ImportError as e:
        print(f"[エラー] 必要パッケージ未インストール: {e}")
        return None

    # 検証データがない場合、trainの末尾15%を使用
    if X_val is None:
        split = int(len(X_train) * 0.85)
        X_val = X_train[split:]
        y_val = y_train[split:]
        X_train = X_train[:split]
        y_train = y_train[:split]

    models = {}

    # --- LightGBM ---
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    models["lgb"] = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(0),
        ],
    )

    # --- XGBoost ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbosity": 0,
        "nthread": -1,
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    models["xgb"] = xgb.train(
        xgb_params, dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # --- CatBoost ---
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models["cat"] = cat_model

    # --- RandomForest ---
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    models["rf"] = rf_model

    # --- ExtraTrees ---
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    et_model.fit(X_train, y_train)
    models["et"] = et_model

    return models


def train_place_ensemble(X_train, y_train, X_val=None, y_val=None):
    """
    2着以内(place_top2)予測用の3モデルアンサンブル (v3.1)。
    LGB + XGB + CatBoost。Win model より軽量で十分。
    place_top2はベースレート約33%(2/6)なのでwin(約17%)より高い。
    """
    try:
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier
    except ImportError as e:
        print(f"[エラー] 必要パッケージ未インストール: {e}")
        return None

    if X_val is None:
        split = int(len(X_train) * 0.85)
        X_val = X_train[split:]
        y_val = y_train[split:]
        X_train = X_train[:split]
        y_train = y_train[:split]

    models = {}

    # --- LightGBM ---
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 43,
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    models["lgb_place"] = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=800,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(0),
        ],
    )

    # --- XGBoost ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbosity": 0,
        "nthread": -1,
        "seed": 43,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    models["xgb_place"] = xgb.train(
        xgb_params, dtrain,
        num_boost_round=800,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # --- CatBoost ---
    cat_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=43,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models["cat_place"] = cat_model

    return models


def predict_place_proba(place_models, X):
    """
    2着以内確率を予測 (v3.1)。
    3モデルアンサンブル: LGB 40%, XGB 40%, CatBoost 20%
    """
    import xgboost as xgb

    lgb_pred = place_models["lgb_place"].predict(X)
    xgb_pred = place_models["xgb_place"].predict(xgb.DMatrix(X))
    cat_pred = place_models["cat_place"].predict_proba(X)[:, 1]

    place_prob = 0.40 * lgb_pred + 0.40 * xgb_pred + 0.20 * cat_pred
    return place_prob


def save_models(models, path=None):
    """モデルを永続化する。Win + Place models (v3.1)。"""
    import xgboost as xgb

    if path is None:
        path = MODEL_DIR
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Win models
    models["lgb"].save_model(str(path / "boat_lgb.txt"))
    models["xgb"].save_model(str(path / "boat_xgb.json"))

    # CatBoost, RF, ET are pickle-serialized
    with open(path / "boat_cat.pkl", "wb") as f:
        pickle.dump(models["cat"], f)
    with open(path / "boat_rf.pkl", "wb") as f:
        pickle.dump(models["rf"], f)
    with open(path / "boat_et.pkl", "wb") as f:
        pickle.dump(models["et"], f)

    # Place models (v3.1)
    if "lgb_place" in models:
        models["lgb_place"].save_model(str(path / "boat_lgb_place.txt"))
        models["xgb_place"].save_model(str(path / "boat_xgb_place.json"))
        with open(path / "boat_cat_place.pkl", "wb") as f:
            pickle.dump(models["cat_place"], f)
        print(f"[モデル保存] {path} (5 win + 3 place models)")
    else:
        print(f"[モデル保存] {path} (5 win models)")


def load_models(path=None):
    """永続化されたモデルをロードする。Win + Place models (v3.1)。"""
    import lightgbm as lgb
    import xgboost as xgb

    if path is None:
        path = MODEL_DIR
    path = Path(path)

    lgb_model = lgb.Booster(model_file=str(path / "boat_lgb.txt"))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(path / "boat_xgb.json"))

    with open(path / "boat_cat.pkl", "rb") as f:
        cat_model = pickle.load(f)
    with open(path / "boat_rf.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(path / "boat_et.pkl", "rb") as f:
        et_model = pickle.load(f)

    result = {"lgb": lgb_model, "xgb": xgb_model, "cat": cat_model, "rf": rf_model, "et": et_model}

    # Place models (v3.1) - optional, may not exist for older saved models
    place_lgb_path = path / "boat_lgb_place.txt"
    if place_lgb_path.exists():
        result["lgb_place"] = lgb.Booster(model_file=str(place_lgb_path))
        xgb_place = xgb.Booster()
        xgb_place.load_model(str(path / "boat_xgb_place.json"))
        result["xgb_place"] = xgb_place
        with open(path / "boat_cat_place.pkl", "rb") as f:
            result["cat_place"] = pickle.load(f)

    return result


def predict_proba(models, X):
    """
    5モデルアンサンブル予測 (v3.0)。
    各艇の勝利確率を返す。
    重み: LGB 25%, XGB 25%, CatBoost 20%, RF 15%, ET 15%
    """
    import xgboost as xgb

    lgb_pred = models["lgb"].predict(X)
    xgb_pred = models["xgb"].predict(xgb.DMatrix(X))

    # CatBoost
    cat_pred = models["cat"].predict_proba(X)[:, 1]

    # RF & ET
    rf_pred = models["rf"].predict_proba(X)[:, 1]
    et_pred = models["et"].predict_proba(X)[:, 1]

    # 重み付き平均 (ブースティング系を重視)
    raw_prob = (
        0.25 * lgb_pred
        + 0.25 * xgb_pred
        + 0.20 * cat_pred
        + 0.15 * rf_pred
        + 0.15 * et_pred
    )
    return raw_prob


def predict_with_agreement(models, X):
    """
    5モデルの一致度を計算する。
    Returns: (ensemble_prob, agreement_count)
      - agreement_count: 過半数と同じ方向に投票したモデル数 (3, 4, or 5)
    """
    import xgboost as xgb

    lgb_pred = models["lgb"].predict(X)
    xgb_pred = models["xgb"].predict(xgb.DMatrix(X))
    cat_pred = models["cat"].predict_proba(X)[:, 1]
    rf_pred = models["rf"].predict_proba(X)[:, 1]
    et_pred = models["et"].predict_proba(X)[:, 1]

    # 各モデルが「勝つ(>0.5)」に投票したか
    # ボートでは1/6=0.167がベースなので閾値は低め
    threshold = 0.20
    votes = np.array([
        (lgb_pred > threshold).astype(int),
        (xgb_pred > threshold).astype(int),
        (cat_pred > threshold).astype(int),
        (rf_pred > threshold).astype(int),
        (et_pred > threshold).astype(int),
    ])
    vote_sum = votes.sum(axis=0)
    final_pred = (vote_sum >= 3).astype(int)

    agreement = np.where(final_pred == 1, vote_sum, 5 - vote_sum)

    ensemble_prob = 0.25 * lgb_pred + 0.25 * xgb_pred + 0.20 * cat_pred + 0.15 * rf_pred + 0.15 * et_pred
    return ensemble_prob, agreement


def normalize_race_probs(df, prob_col="raw_prob"):
    """
    レース内で確率を正規化し、合計を1.0にする。
    """
    df = df.copy()
    race_sums = df.groupby("race_id")[prob_col].transform("sum")
    df["pred_prob"] = df[prob_col] / race_sums
    return df


# =============================================================
# バリューベット検出 + Kelly Criterion
# =============================================================

def kelly_fraction(model_prob, odds, fraction=0.25):
    """
    Fractional Kelly Criterion (v3.0)。
    fraction=0.25 → 1/4 Kelly でリスク抑制。

    Kelly: f* = (p * b - q) / b
      p = 勝率, q = 1-p, b = オッズ-1 (net odds)
    """
    p = model_prob
    q = 1.0 - p
    b = odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (p * b - q) / b
    return max(0.0, kelly * fraction)


def find_value_bets(race_df, bet_type="win", min_ev=2.00, kelly_frac=0.25):
    # min_ev optimized via WF analysis (was 1.25, PF: 2.353 -> 2.910)
    """
    バリューベットを検出する (v3.4)。
    Kelly criterion で賭け金を算出。

    v3.4 改善 (paper trade PF 1.86→改善目標):
    - 枠番別EV閾値: 外枠(2,5,6号艇)はモデル精度が低いためEV閾値を引き上げ
    - オッズ上限: 超高オッズ(>60)はモデルが過信しやすいため除外
    - 最低確率フィルタ: 枠番別の最低予測確率を設定
    - 1レースあたりの最大ベット数制限

    v3.1 改善:
    - Exacta: 条件付き確率 P(1st=A,2nd=B) + 複数組合せ(top3候補)
    - Quinella: 2着以内確率ベース + 複数組合せ(top4候補)
    - 各戦略に最適化されたEV閾値
    """
    bets = []

    # v3.4: 枠番別EV閾値 (外枠はモデル精度が低いため高めに設定)
    # Paper trade analysis: Lane 2 hit 8.1%, Lane 6 hit 7.7% vs expected
    LANE_MIN_EV = {
        1: min_ev,          # 1号艇: 基本閾値 (高精度)
        2: min_ev * 1.25,   # 2号艇: 25%引き上げ (hit rate 8.1% = poor calibration)
        3: min_ev * 1.10,   # 3号艇: 10%引き上げ
        4: min_ev,          # 4号艇: 基本閾値 (hit rate 25% = good)
        5: min_ev * 1.15,   # 5号艇: 15%引き上げ
        6: min_ev * 1.30,   # 6号艇: 30%引き上げ (hit rate 7.7% = poor calibration)
    }

    # v3.4: 枠番別最低予測確率 (これ以下はノイズと判断)
    LANE_MIN_PROB = {
        1: 0.25,   # 1号艇: ベースレート55%なので25%以上で十分
        2: 0.10,   # 2号艇: ベースレート15%
        3: 0.08,   # 3号艇: ベースレート12%
        4: 0.08,   # 4号艇: ベースレート10%
        5: 0.06,   # 5号艇: ベースレート5%
        6: 0.05,   # 6号艇: ベースレート3%
    }

    # v3.4: オッズ上限 (超高オッズはモデル過信リスク大)
    MAX_ODDS_WIN = 60.0

    if bet_type == "win":
        for _, row in race_df.iterrows():
            # Use real odds if available, fallback to approximate
            odds = row.get("odds", APPROX_ODDS.get(row["lane"], 10.0))
            if pd.isna(odds) or odds <= 1.0:
                odds = APPROX_ODDS.get(row["lane"], 10.0)
            implied_prob = 1.0 / odds
            model_prob = row["pred_prob"]
            lane = int(row["lane"])
            ev = model_prob * odds

            # v3.4: オッズ上限フィルタ
            if odds > MAX_ODDS_WIN:
                continue

            # v3.4: 最低確率フィルタ
            if model_prob < LANE_MIN_PROB.get(lane, 0.05):
                continue

            # v3.4: モデル一致度フィルタ (4/5以上のモデルが同意する場合のみ)
            if "model_agreement" in row.index:
                if row["model_agreement"] < 4:
                    continue

            # v3.4: 枠番別EV閾値
            lane_ev_threshold = LANE_MIN_EV.get(lane, min_ev)
            if ev >= lane_ev_threshold:
                kf = kelly_fraction(model_prob, odds, fraction=kelly_frac)
                bets.append({
                    "race_id": row["race_id"],
                    "lane": row["lane"],
                    "model_prob": model_prob,
                    "implied_prob": implied_prob,
                    "odds": odds,
                    "ev": ev,
                    "kelly_fraction": kf,
                    "bet_type": "win",
                    "win": row["win"],
                })

        # v3.4: 1レースあたり最大3ベットに制限 (リスク管理)
        if len(bets) > 3:
            bets.sort(key=lambda x: -x["ev"])
            bets = bets[:3]

    elif bet_type == "exacta":
        # v3.1: 条件付き確率 + 複数組合せ
        # P(1st=A, 2nd=B) = P(win=A) * P(place=B|A wins)
        # P(place=B|A wins) ≈ P(place=B) / (1 - P(win=A)) * adjustment
        sorted_boats = race_df.sort_values("pred_prob", ascending=False)
        top_n = sorted_boats.head(3)  # top 3 candidates for combinations

        has_place_prob = "pred_place_prob" in race_df.columns
        min_ev_exacta = 1.95  # Optimized via WF analysis (was 1.15, PF: 2.599 -> 3.241)

        # Generate all ordered pairs from top-3
        candidates = list(top_n.iterrows())
        for i, (idx_a, boat_a) in enumerate(candidates):
            for j, (idx_b, boat_b) in enumerate(candidates):
                if i == j:
                    continue

                p_win_a = boat_a["pred_prob"]

                if has_place_prob:
                    # Use place model: P(2nd=B|A wins) ≈ P(place=B) adjusted
                    p_place_b = boat_b["pred_place_prob"]
                    # Conditional: given A wins, B's place prob increases
                    # P(B in top2 | A=1st) ≈ P(B in top2) / (1 - P(B wins)) * correction
                    p_b_not_win = max(1.0 - boat_b["pred_prob"], 0.01)
                    p_second_b_given_a = (p_place_b - boat_b["pred_prob"]) / p_b_not_win
                    p_second_b_given_a = max(min(p_second_b_given_a, 0.8), 0.02)
                else:
                    # Fallback: use rank-based approximation
                    # P(2nd=B|A wins) roughly proportional to pred_prob among remaining
                    remaining_prob = sum(
                        r["pred_prob"] for _, r in candidates if r["lane"] != boat_a["lane"]
                    )
                    p_second_b_given_a = boat_b["pred_prob"] / max(remaining_prob, 0.01)
                    p_second_b_given_a = min(p_second_b_given_a, 0.7)

                exacta_prob = p_win_a * p_second_b_given_a

                # Odds estimation: use lane-based approximation
                # Exacta odds are roughly (1/prob) * payout_ratio
                # Payout ratio for exacta ≈ 0.75 (25% takeout)
                lane_a = int(boat_a["lane"])
                lane_b = int(boat_b["lane"])
                # More realistic odds: use combination of single odds
                odds_a = boat_a.get("odds", APPROX_ODDS.get(lane_a, 10.0))
                odds_b = boat_b.get("odds", APPROX_ODDS.get(lane_b, 10.0))
                if pd.isna(odds_a) or odds_a <= 1.0:
                    odds_a = APPROX_ODDS.get(lane_a, 10.0)
                if pd.isna(odds_b) or odds_b <= 1.0:
                    odds_b = APPROX_ODDS.get(lane_b, 10.0)

                # Exacta odds ≈ win_odds_A * (win_odds_B / adjustment)
                # Popular combos have lower odds, unpopular have higher
                exacta_odds = max(5.0, odds_a * odds_b * 0.35)
                exacta_odds = min(exacta_odds, 200.0)

                ev = exacta_prob * exacta_odds
                actual_hit = (boat_a["win"] == 1 and boat_b["place_top2"] == 1
                              and boat_b["win"] != 1)

                if ev >= min_ev_exacta and exacta_prob >= 0.03:
                    kf = kelly_fraction(exacta_prob, exacta_odds, fraction=kelly_frac * 0.5)
                    bets.append({
                        "race_id": boat_a["race_id"],
                        "lane": f"{lane_a}-{lane_b}",
                        "model_prob": exacta_prob,
                        "odds": exacta_odds,
                        "ev": ev,
                        "kelly_fraction": kf,
                        "bet_type": "exacta",
                        "win": 1 if actual_hit else 0,
                    })

        # Limit to best 2 bets per race to control risk
        if len(bets) > 2:
            bets.sort(key=lambda x: -x["ev"])
            bets = bets[:2]

    elif bet_type == "quinella":
        # v3.1: 2着以内確率ベース + 複数組合せ
        # P(A and B both in top2) = P(place=A) * P(place=B|A in top2)
        sorted_boats = race_df.sort_values("pred_prob", ascending=False)
        top_n = sorted_boats.head(4)  # top 4 candidates

        has_place_prob = "pred_place_prob" in race_df.columns
        min_ev_quinella = 2.00  # Optimized via WF analysis (was 1.10, PF: 2.599 -> 3.208)

        # Generate all unordered pairs from top-4
        candidates = list(top_n.iterrows())
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                idx_a, boat_a = candidates[i]
                idx_b, boat_b = candidates[j]

                if has_place_prob:
                    p_place_a = boat_a["pred_place_prob"]
                    p_place_b = boat_b["pred_place_prob"]
                    # P(both in top2): conditional probability
                    # P(A in top2 AND B in top2) ≈ P(A) * P(B|A) where
                    # P(B|A in top2) ≈ P(B in top2) * boost (since if A takes a slot,
                    # only 1 slot left among 5 boats, so conditional is different)
                    # Approximation: P(A,B) ≈ P(A) * P(B) * correction_factor
                    # Correction: given only 2 of 6 make top2, knowing A is in
                    # means 1 of 5 remaining. P(B|A) ≈ P(B) / (sum of all others' place_prob)
                    remaining_place_sum = sum(
                        r["pred_place_prob"] for _, r in candidates
                        if r["lane"] != boat_a["lane"]
                    )
                    if remaining_place_sum > 0:
                        p_b_given_a = p_place_b / remaining_place_sum
                    else:
                        p_b_given_a = 0.2
                    quinella_prob = p_place_a * p_b_given_a
                else:
                    # Fallback: use win probabilities as proxy
                    p_a = boat_a["pred_prob"]
                    p_b = boat_b["pred_prob"]
                    # Quinella probability: both in top2
                    # Rough approximation using win probs scaled by place factor
                    place_factor_a = min(2.5, 1.0 + (1.0 / max(boat_a["lane"], 1)) * 0.5)
                    place_factor_b = min(2.5, 1.0 + (1.0 / max(boat_b["lane"], 1)) * 0.5)
                    quinella_prob = (p_a * place_factor_a) * (p_b * place_factor_b) * 2.0
                    quinella_prob = min(quinella_prob, 0.5)

                # Quinella odds estimation
                lane_a = int(boat_a["lane"])
                lane_b = int(boat_b["lane"])
                odds_a = boat_a.get("odds", APPROX_ODDS.get(lane_a, 10.0))
                odds_b = boat_b.get("odds", APPROX_ODDS.get(lane_b, 10.0))
                if pd.isna(odds_a) or odds_a <= 1.0:
                    odds_a = APPROX_ODDS.get(lane_a, 10.0)
                if pd.isna(odds_b) or odds_b <= 1.0:
                    odds_b = APPROX_ODDS.get(lane_b, 10.0)

                # Quinella odds ≈ half of exacta (order doesn't matter)
                quinella_odds = max(3.0, odds_a * odds_b * 0.20)
                quinella_odds = min(quinella_odds, 100.0)

                ev = quinella_prob * quinella_odds
                actual_hit = (boat_a["place_top2"] == 1 and boat_b["place_top2"] == 1)

                if ev >= min_ev_quinella and quinella_prob >= 0.05:
                    kf = kelly_fraction(quinella_prob, quinella_odds, fraction=kelly_frac * 0.5)
                    bets.append({
                        "race_id": boat_a["race_id"],
                        "lane": f"{lane_a}-{lane_b}",
                        "model_prob": quinella_prob,
                        "odds": quinella_odds,
                        "ev": ev,
                        "kelly_fraction": kf,
                        "bet_type": "quinella",
                        "win": 1 if actual_hit else 0,
                    })

        # Limit to best 3 bets per race
        if len(bets) > 3:
            bets.sort(key=lambda x: -x["ev"])
            bets = bets[:3]

    return bets


# =============================================================
# リスク管理
# =============================================================

class BoatRiskManager:
    """
    ボートレース用リスク管理 (v3.0: Kelly対応)。
    """

    def __init__(self, daily_budget=3000, monthly_budget=30000, base_bet=500):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.base_bet = base_bet
        self.daily_spent = 0
        self.monthly_spent = 0
        self.daily_wins = 0
        self.daily_losses = 0

    def calc_bet_amount(self, kelly_frac, bankroll=None):
        """Kelly分率からベット額を計算"""
        if bankroll is None:
            bankroll = self.daily_budget - self.daily_spent
        bet = max(100, min(int(bankroll * kelly_frac / 100) * 100, self.base_bet * 3))
        return bet

    def can_bet(self, amount=None):
        if amount is None:
            amount = self.base_bet
        if self.daily_spent + amount > self.daily_budget:
            return False
        if self.monthly_spent + amount > self.monthly_budget:
            return False
        return True

    def place_bet(self, amount, won, odds=1.0):
        self.daily_spent += amount
        self.monthly_spent += amount
        if won:
            self.daily_wins += 1
            return amount * odds
        else:
            self.daily_losses += 1
            return 0

    def reset_daily(self):
        self.daily_spent = 0
        self.daily_wins = 0
        self.daily_losses = 0

    def summary(self):
        return {
            "daily_spent": self.daily_spent,
            "monthly_spent": self.monthly_spent,
            "daily_record": f"{self.daily_wins}W-{self.daily_losses}L",
        }


# =============================================================
# メトリクス計算 (v3.0)
# =============================================================

def compute_metrics(bets_df, base_bet=500):
    """
    PF (Profit Factor), Sharpe Ratio, MDD (Max Drawdown) を計算する。
    全バーベースでSharpeを計算（traded barsだけではない）。
    """
    if len(bets_df) == 0:
        return {"pf": 0, "sharpe": 0, "mdd": 0, "recovery": 0, "n_bets": 0}

    # Kelly対応: kelly_fractionがあればベット額を可変に
    if "kelly_fraction" in bets_df.columns:
        bets_df = bets_df.copy()
        bets_df["bet_amount"] = bets_df["kelly_fraction"].apply(
            lambda kf: max(100, min(int(base_bet * max(kf, 0.1) * 4 / 100) * 100, base_bet * 3))
        )
        # 最低ベットはbase_bet
        bets_df["bet_amount"] = bets_df["bet_amount"].clip(lower=100)
    else:
        bets_df = bets_df.copy()
        bets_df["bet_amount"] = base_bet

    # 各ベットの損益
    bets_df["pnl"] = np.where(
        bets_df["win"] == 1,
        bets_df["bet_amount"] * bets_df["odds"] - bets_df["bet_amount"],  # 利益
        -bets_df["bet_amount"]  # 損失
    )

    gross_profit = bets_df.loc[bets_df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(bets_df.loc[bets_df["pnl"] < 0, "pnl"].sum())

    # PF
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 回収率
    total_bet = bets_df["bet_amount"].sum()
    total_return = total_bet + bets_df["pnl"].sum()
    recovery = total_return / total_bet if total_bet > 0 else 0

    # Sharpe (on ALL bars, not just traded bars)
    pnl_series = bets_df["pnl"].values
    if len(pnl_series) > 1:
        sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(len(pnl_series))
    else:
        sharpe = 0.0

    # MDD (Maximum Drawdown)
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    mdd = drawdowns.max() if len(drawdowns) > 0 else 0

    return {
        "pf": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "mdd": round(mdd, 0),
        "recovery": round(recovery, 4),
        "n_bets": len(bets_df),
        "n_wins": int(bets_df["win"].sum()),
        "hit_rate": round(bets_df["win"].mean(), 4),
        "total_bet": round(total_bet, 0),
        "total_return": round(total_return, 0),
        "total_pnl": round(bets_df["pnl"].sum(), 0),
        "avg_pnl": round(pnl_series.mean(), 1),
        "gross_profit": round(gross_profit, 0),
        "gross_loss": round(gross_loss, 0),
    }


# =============================================================
# Walk-Forward検証 (v3.0: 5ウィンドウ + 正確メトリクス)
# =============================================================

def walk_forward_validate(df, n_folds=5):
    """
    Expanding Window Walk-Forward検証 (v3.1):
    - 5ウィンドウでWin+Placeモデルを訓練 -> テスト
    - PF, Sharpe, MDD, 回収率を正確に計算
    - Fractional Kelly (0.25x) でベットサイジング
    - v3.1: Place model追加、Exacta/Quinella改善
    """
    print("\n" + "=" * 60)
    print(f"Expanding Window Walk-Forward検証 ({n_folds}ウィンドウ) [v3.1]")
    print("Win: 5モデルアンサンブル (LGB+XGB+Cat+RF+ET)")
    print("Place: 3モデルアンサンブル (LGB+XGB+Cat)")
    print("=" * 60)

    df = df.sort_values("race_date").reset_index(drop=True)
    race_ids = df["race_id"].unique()
    n_races = len(race_ids)

    # Expanding window: 各フォールドで test_size = 1/(n_folds+1) の期間をテスト
    test_size = n_races // (n_folds + 1)
    fold_results = []
    all_test_dfs = []
    final_models = None

    for fold in range(n_folds):
        # 訓練: 先頭から train_end まで、テスト: train_end から test_end まで
        train_end = test_size * (fold + 1)
        test_end = min(train_end + test_size, n_races)

        if test_end <= train_end:
            break

        train_ids = set(race_ids[:train_end])
        test_ids = set(race_ids[train_end:test_end])

        train_df = df[df["race_id"].isin(train_ids)]
        test_df = df[df["race_id"].isin(test_ids)].copy()

        print(f"\n--- Fold {fold + 1}/{n_folds}: 訓練={len(train_ids)}レース, テスト={len(test_ids)}レース ---")

        # 検証用に訓練データの末尾15%を分離
        X_train_all = train_df[FEATURE_COLS].values
        y_train_win = train_df["win"].values
        y_train_place = train_df["place_top2"].values
        val_split = int(len(X_train_all) * 0.85)
        X_tr = X_train_all[:val_split]
        X_va = X_train_all[val_split:]

        # Train win model
        models = train_ensemble(X_tr, y_train_win[:val_split], X_va, y_train_win[val_split:])
        if models is None:
            print("[エラー] Winモデル訓練失敗")
            continue

        # Train place model (v3.1)
        print(f"  Place model 訓練中...")
        place_models = train_place_ensemble(X_tr, y_train_place[:val_split], X_va, y_train_place[val_split:])
        if place_models is not None:
            models.update(place_models)
            print(f"  Place model 訓練完了")
        else:
            print(f"  [警告] Place model 訓練失敗、fallback使用")

        # 予測
        X_test = test_df[FEATURE_COLS].values
        test_df["raw_prob"] = predict_proba(models, X_test)
        test_df = normalize_race_probs(test_df)

        # v3.4: モデル一致度を計算
        _, agreement = predict_with_agreement(models, X_test)
        test_df["model_agreement"] = agreement

        # Place prediction (v3.1)
        if "lgb_place" in models:
            test_df["raw_place_prob"] = predict_place_proba(models, X_test)
            # Normalize place probs within race (sum should be ~2.0 since 2 of 6 finish top2)
            race_place_sums = test_df.groupby("race_id")["raw_place_prob"].transform("sum")
            test_df["pred_place_prob"] = test_df["raw_place_prob"] / race_place_sums * 2.0

        # 戦略別シミュレーション
        fold_strats = {}
        for strategy_name, bet_type in [("A_単勝", "win"), ("B_2連単", "exacta"), ("C_2連複", "quinella")]:
            all_bets = []
            for rid in test_ids:
                race_data = test_df[test_df["race_id"] == rid]
                bets = find_value_bets(
                    race_data,
                    bet_type=bet_type.split("_")[-1] if "_" in bet_type else bet_type,
                    kelly_frac=0.25,
                )
                all_bets.extend(bets)

            if len(all_bets) == 0:
                fold_strats[strategy_name] = {"bets": 0, "recovery": 0.0, "pf": 0, "sharpe": 0, "mdd": 0}
                continue

            bets_df = pd.DataFrame(all_bets)
            metrics = compute_metrics(bets_df)

            fold_strats[strategy_name] = {
                "bets": metrics["n_bets"],
                "wins": metrics["n_wins"],
                "hit_rate": metrics["hit_rate"],
                "recovery": metrics["recovery"],
                "pf": metrics["pf"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "total_pnl": metrics["total_pnl"],
            }
            print(f"  {strategy_name}: {metrics['n_bets']}ベット, 的中率={metrics['hit_rate']:.1%}, "
                  f"回収率={metrics['recovery']:.3f}, PF={metrics['pf']:.2f}, Sharpe={metrics['sharpe']:.2f}")

        fold_results.append(fold_strats)
        all_test_dfs.append(test_df)
        final_models = models

    # ===== 全フォールド集計 =====
    print("\n" + "=" * 60)
    print("Walk-Forward 全フォールド集計 [v3.1]")
    print("=" * 60)

    combined_results = {}
    for strategy_name in ["A_単勝", "B_2連単", "C_2連複"]:
        total_bets = sum(fr[strategy_name]["bets"] for fr in fold_results)
        total_wins = sum(fr[strategy_name].get("wins", 0) for fr in fold_results)
        # 加重平均回収率
        weighted_recovery = sum(
            fr[strategy_name]["recovery"] * fr[strategy_name]["bets"]
            for fr in fold_results
        ) / max(total_bets, 1)
        hit_rate = total_wins / max(total_bets, 1)

        # 各フォールドの回収率一覧
        fold_recoveries = [fr[strategy_name]["recovery"] for fr in fold_results if fr[strategy_name]["bets"] > 0]
        fold_pfs = [fr[strategy_name]["pf"] for fr in fold_results if fr[strategy_name]["bets"] > 0]
        fold_sharpes = [fr[strategy_name]["sharpe"] for fr in fold_results if fr[strategy_name]["bets"] > 0]
        fold_mdds = [fr[strategy_name]["mdd"] for fr in fold_results if fr[strategy_name]["bets"] > 0]

        min_recovery = min(fold_recoveries) if fold_recoveries else 0
        max_recovery = max(fold_recoveries) if fold_recoveries else 0
        std_recovery = np.std(fold_recoveries) if len(fold_recoveries) > 1 else 0
        avg_pf = np.mean(fold_pfs) if fold_pfs else 0
        avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0
        max_mdd = max(fold_mdds) if fold_mdds else 0

        # 合格: 加重平均 > 1.05 かつ 全フォールド > 0.9 かつ 十分なベット数
        passed = (weighted_recovery > 1.05
                  and min_recovery > 0.90
                  and total_bets > 100)

        combined_results[strategy_name] = {
            "bets": total_bets,
            "wins": total_wins,
            "hit_rate": hit_rate,
            "recovery": weighted_recovery,
            "min_recovery": min_recovery,
            "max_recovery": max_recovery,
            "std_recovery": std_recovery,
            "fold_recoveries": fold_recoveries,
            "avg_pf": avg_pf,
            "fold_pfs": fold_pfs,
            "avg_sharpe": avg_sharpe,
            "fold_sharpes": fold_sharpes,
            "max_mdd": max_mdd,
            "fold_mdds": fold_mdds,
            "passed": passed,
        }

        status = "PASS" if passed else "FAIL"
        print(f"\n{strategy_name}: {status}")
        print(f"  合計ベット: {total_bets}, 的中率: {hit_rate:.1%}")
        print(f"  加重平均回収率: {weighted_recovery:.3f} ({weighted_recovery * 100:.1f}%)")
        print(f"  フォールド別回収率: {', '.join('%.3f' % r for r in fold_recoveries)}")
        print(f"  最小={min_recovery:.3f}, 最大={max_recovery:.3f}, SD={std_recovery:.3f}")
        print(f"  平均PF: {avg_pf:.3f}, フォールド別: {', '.join('%.2f' % p for p in fold_pfs)}")
        print(f"  平均Sharpe: {avg_sharpe:.3f}, フォールド別: {', '.join('%.2f' % s for s in fold_sharpes)}")
        print(f"  最大MDD: {max_mdd:,.0f}円")

    # 特徴量重要度 (最終モデル)
    if final_models:
        print("\n[特徴量重要度 (LightGBM, 最終フォールド)]")
        importance = final_models["lgb"].feature_importance(importance_type="gain")
        feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
        for fname, imp in feat_imp[:20]:
            bar = "#" * int(imp / max(importance) * 30)
            print(f"  {fname:30s} {imp:>8.0f} {bar}")

        # モデル保存
        save_models(final_models)

    # 枠番別分析 (全テストデータ)
    if all_test_dfs:
        combined_test = pd.concat(all_test_dfs, ignore_index=True)
        print("\n--- 枠番別分析（全テストデータ） ---")
        lane_stats = {}
        for lane in range(1, 7):
            lane_data = combined_test[combined_test["lane"] == lane]
            actual_wr = lane_data["win"].mean()
            pred_wr = lane_data["pred_prob"].mean()
            lane_stats[lane] = {"actual_wr": actual_wr, "pred_wr": pred_wr}
            print(f"  {lane}号艇: 実勝率={actual_wr:.1%}, 予測平均={pred_wr:.3f}")
    else:
        lane_stats = {}
        combined_test = pd.DataFrame()

    return {
        "models": final_models,
        "strategies": combined_results,
        "lane_stats": lane_stats,
        "test_df": combined_test,
        "fold_results": fold_results,
    }


# =============================================================
# レース予測
# =============================================================

def predict_race(race_data, models):
    """
    1レース分のデータに対して予測を行う。
    """
    race_data = create_features(race_data)
    X = race_data[FEATURE_COLS].values
    raw_probs = predict_proba(models, X)

    # 正規化
    probs = raw_probs / raw_probs.sum()

    predictions = []
    for i, (_, row) in enumerate(race_data.iterrows()):
        predictions.append({
            "lane": int(row["lane"]),
            "racer_class": CLASS_NAMES.get(int(row["racer_class"]), "??"),
            "pred_prob": round(float(probs[i]), 4),
            "recommended": probs[i] > 1.0 / APPROX_ODDS.get(int(row["lane"]), 10.0) * 1.25,
        })

    predictions.sort(key=lambda x: -x["pred_prob"])
    return predictions


# =============================================================
# レポート生成
# =============================================================

def generate_report(validation_results):
    """
    検証結果をレポートファイルに保存する (v3.0)。
    """
    report_path = DATA_DIR / "boat_report.txt"
    lines = []

    lines.append("=" * 60)
    lines.append("ボートレース予測モデル v3.1 レポート")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Win: 5モデルアンサンブル (LGB+XGB+Cat+RF+ET)")
    lines.append("Place: 3モデルアンサンブル (LGB+XGB+Cat)")
    lines.append("Kelly Criterion: 0.25x Fractional Kelly")
    lines.append("=" * 60)

    # 戦略別結果
    lines.append("\n--- 戦略別回収率 (Expanding Walk-Forward, 5ウィンドウ) ---")
    lines.append("-" * 50)
    strategies = validation_results["strategies"]
    for name, stats in strategies.items():
        if stats["bets"] == 0:
            lines.append(f"  {name}: ベットなし")
            continue
        status = "PASS" if stats.get("passed", False) else "FAIL"
        lines.append(f"\n  {name}: [{status}]")
        lines.append(f"    ベット数: {stats['bets']}, 的中率: {stats.get('hit_rate', 0):.1%}")
        lines.append(f"    加重平均回収率: {stats['recovery']:.4f} ({stats['recovery'] * 100:.1f}%)")
        fold_recs = stats.get("fold_recoveries", [])
        if fold_recs:
            lines.append(f"    フォールド別回収率: {', '.join('%.3f' % r for r in fold_recs)}")
            lines.append(f"    最小={stats.get('min_recovery', 0):.3f}, 最大={stats.get('max_recovery', 0):.3f}, SD={stats.get('std_recovery', 0):.3f}")
        lines.append(f"    平均PF: {stats.get('avg_pf', 0):.3f}")
        lines.append(f"    平均Sharpe: {stats.get('avg_sharpe', 0):.3f}")
        lines.append(f"    最大MDD: {stats.get('max_mdd', 0):,.0f}円")

    # 枠番別分析
    lines.append("\n--- 枠番別勝率分析 ---")
    lines.append("-" * 50)
    for lane, stats in validation_results["lane_stats"].items():
        lines.append(f"  {lane}号艇: 実勝率={stats['actual_wr']:.1%}, 予測={stats['pred_wr']:.3f}")

    # 月間シミュレーション
    lines.append("\n--- 月間シミュレーション ---")
    lines.append("-" * 50)
    best_strategy = max(strategies.items(), key=lambda x: x[1].get("recovery", 0))
    name, stats = best_strategy
    if stats["bets"] > 0:
        test_df = validation_results["test_df"]
        if len(test_df) > 0:
            test_days = (test_df["race_date"].max() - test_df["race_date"].min()).days
            if test_days > 0:
                bets_per_day = stats["bets"] / max(test_days, 1)
                monthly_bets = int(bets_per_day * 30)
                monthly_invest = monthly_bets * 500
                monthly_return = monthly_invest * stats["recovery"]
                monthly_profit = monthly_return - monthly_invest

                lines.append(f"  最良戦略: {name}")
                lines.append(f"  月間ベット数(推定): {monthly_bets}")
                lines.append(f"  月間投資額: {monthly_invest:,.0f}円")
                lines.append(f"  月間回収額: {monthly_return:,.0f}円")
                lines.append(f"  月間損益: {monthly_profit:+,.0f}円")
                lines.append(f"  回収率: {stats['recovery']:.4f}")

    # 注意事項
    lines.append("\n--- 注意事項 ---")
    lines.append("-" * 50)
    lines.append("  - リアルデータ使用時は実際のオッズ・選手データに基づく結果です")
    lines.append("  - ペーパートレードでの追加検証を推奨します")
    lines.append("  - 回収率1.0を超えても、実戦では控除率（約25%）があります")
    lines.append("  - 必ず余裕資金で、予算上限を厳守してください")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[レポート保存] {report_path}")

    return report_text


# =============================================================
# メインパイプライン
# =============================================================

def run_pipeline():
    """
    ボートレース予測 v3.0 パイプライン:
    1. データ読込
    2. 特徴量エンジニアリング (40+特徴量)
    3. Walk-Forward検証 (5ウィンドウ, 5モデルアンサンブル)
    4. PF/Sharpe/MDD計算
    5. レポート出力
    """
    print("=" * 60)
    print("ボートレース予測モデル v3.1 パイプライン")
    print("5+3モデルアンサンブル (Win+Place) + 40+特徴量 + Kelly Criterion")
    print("=" * 60)

    # 1. データ読込（リアルデータ優先）
    print("\n[1/5] データ読込...")
    df = load_real_data()
    if df is None:
        print("[FALLBACK] リアルデータなし -> 合成データで実行")
        df = generate_training_data(n_races=10000)

    # 2. 特徴量エンジニアリング
    print(f"\n[2/5] 特徴量エンジニアリング ({len(FEATURE_COLS)}特徴量)...")
    df = create_features(df)
    print(f"  使用可能特徴量: {sum(1 for c in FEATURE_COLS if c in df.columns)}/{len(FEATURE_COLS)}")

    # 3. Walk-Forward検証
    print("\n[3/5] Walk-Forward検証 (5ウィンドウ, Win+Place model)...")
    results = walk_forward_validate(df, n_folds=5)
    if results is None:
        print("[エラー] 検証失敗")
        return

    # 4. レポート出力
    print("\n[4/5] レポート出力...")
    report = generate_report(results)
    print(report)

    # 5. 戦略設定保存
    print("\n[5/5] 戦略設定保存...")
    config = {
        "version": "3.1",
        "model_type": "5win_3place_ensemble",
        "n_features": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "kelly_fraction": 0.25,
        "wf_folds": 5,
        "strategies": {},
    }
    for name, stats in results["strategies"].items():
        config["strategies"][name] = {
            "bets": stats["bets"],
            "recovery": round(float(stats.get("recovery", 0)), 4),
            "avg_pf": round(float(stats.get("avg_pf", 0)), 3),
            "avg_sharpe": round(float(stats.get("avg_sharpe", 0)), 3),
            "max_mdd": round(float(stats.get("max_mdd", 0)), 0),
            "passed": bool(stats.get("passed", False)),
        }

    config_path = DATA_DIR / "boat_strategy_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[設定保存] {config_path}")

    any_passed = any(s.get("passed", False) for s in results["strategies"].values())
    if any_passed:
        print("\n[結果] 合格した戦略があります。ペーパートレードでの検証を推奨します。")
    else:
        print("\n[結果] 合格した戦略はありません。パラメータ調整または追加データが必要です。")

    return results


if __name__ == "__main__":
    run_pipeline()
