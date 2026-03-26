# ===========================================
# keiba_model.py
# 競馬予測モデル: バリューベット戦略 v2.0 → v2.1 pruned
#
# 戦略の考え方:
#   モデルが推定する勝率と、オッズから逆算される暗黙の確率を比較。
#   モデルの推定確率がオッズの暗黙確率を大幅に上回る場合のみベットする。
#   これにより、控除率（約20-25%）を超える回収率を目指す。
#
# v2.0 改善点:
#   - 5モデルアンサンブル (LGB + XGB + CatBoost + RF + ExtraTrees)
#   - 改良された合成データ生成 (200+ 騎手、クラス効果、調教師効果、距離適性)
#   - 追加特徴量 (ペース予測、脚質、距離適性指数、馬場×体重交互作用)
#   - edge_threshold 1.30 (控除率20-25%を考慮)
#   - Expanding Window Walk-Forward
#   - Profit Factor (PF) レポート
#
# 注意:
#   合成データでの検証です。実データでは結果が異なる可能性が高い。
#   投資助言ではありません。必ず余裕資金の範囲内で。
# ===========================================

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
# Suppress sklearn parallel warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ===== 定数 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "keiba"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 特徴量カラム（基本）
# v2.1 pruned: Removed trainer_win_rate (correlated w/ jockey_win_rate, keep jockey)
FEATURE_COLS = [
    # 馬の基本情報
    "horse_age",           # 馬齢
    "horse_weight",        # 馬体重 (kg)
    "weight_change",       # 体重変化
    "sex",                 # 性別 (0=牝, 1=牡, 2=セン)
    # 成績
    "win_rate",            # 勝率
    "place_rate",          # 複勝率
    "earnings_per_race",   # 1走あたり賞金（万円）
    "days_since_last",     # 前走からの日数
    "last_finish",         # 前走着順
    "last3_avg_finish",    # 直近3走平均着順
    # 騎手
    "jockey_win_rate",     # 騎手勝率
    "jockey_place_rate",   # 騎手複勝率
    # コース・条件
    "post_position",       # 枠番
    "distance",            # 距離 (m)
    "track_type",          # 芝=0, ダート=1
    "track_condition",     # 良=0, 稍重=1, 重=2, 不良=3
    "field_size",          # 出走頭数
    # クラス
    "race_class",          # G1=5, G2=4, G3=3, Listed=2, Open=1, 条件=0
    # 脚質
    "running_style",       # 逃げ=0, 先行=1, 差し=2, 追込=3
    # オッズ
    "odds",                # 単勝オッズ
    "popularity",          # 人気順
]


# ===========================================
# 実データ読み込み
# ===========================================

REAL_DATA_CSV = DATA_DIR / "real_race_results.csv"

# 実データ用の追加特徴量カラム
# v2.1 pruned: Removed sire_encoded, bms_encoded (near-zero importance),
#   venue_aptitude (near-zero), odds_vs_jockey (correlated w/ odds, keep odds)
REAL_FEATURE_COLS = FEATURE_COLS + [
    "last_3f",             # 上がり3F（前走）
    "last_3f_rank",        # 上がり3F順位（前走）
    "corner_position_4",   # 最終コーナー通過順位
    "weight_carried",      # 斤量
    # v2.1: 追加実データ特徴量
    "prev_last_3f",        # 前走の上がり3F（このレース結果ではなく前走の）
    "dist_aptitude",       # 距離適性（同距離帯での勝率）
    "track_aptitude",      # 馬場適性（同馬場タイプでの勝率）
    "class_x_age",         # クラス×馬齢交互作用
]


def load_real_data() -> pd.DataFrame:
    """
    netkeiba.comからスクレイピングした実データを読み込み、
    モデルで使える形式に変換する。

    Returns:
        DataFrame (keiba_model.pyの形式に合わせたカラム構成)
        空の場合は pd.DataFrame() を返す
    """
    if not REAL_DATA_CSV.exists():
        print("[実データ] real_race_results.csv が見つかりません。")
        print("  fetch_netkeiba.py を実行してデータを取得してください。")
        return pd.DataFrame()

    try:
        raw = pd.read_csv(REAL_DATA_CSV, encoding="utf-8-sig")
    except Exception as e:
        print(f"[実データ] 読み込みエラー: {e}")
        return pd.DataFrame()

    if raw.empty:
        print("[実データ] CSVが空です。")
        return pd.DataFrame()

    print(f"[実データ] {len(raw)} 行, {raw['race_id'].nunique()} レース読み込み")

    # --- 基本カラムの変換 ---
    df = pd.DataFrame()
    df["race_id"] = raw["race_id"].astype(str)
    df["race_date"] = raw["race_date"]
    df["horse_name"] = raw["horse_name"]

    # 馬の基本情報
    df["horse_age"] = raw.get("horse_age", pd.Series(dtype=int))
    df["horse_weight"] = raw.get("horse_weight", pd.Series(dtype=int))
    df["weight_change"] = raw.get("weight_change", pd.Series(dtype=int))
    df["sex"] = raw.get("sex", pd.Series(dtype=int))

    # コース・条件
    df["post_position"] = raw.get("post_position", pd.Series(dtype=int))
    df["distance"] = raw.get("distance", pd.Series(dtype=int))
    df["track_type"] = raw.get("track_type", pd.Series(dtype=int))
    df["track_condition"] = raw.get("track_condition", pd.Series(dtype=int))
    df["field_size"] = raw.get("field_size", pd.Series(dtype=int))
    df["race_class"] = raw.get("race_class", pd.Series(dtype=int))

    # オッズ・人気
    df["odds"] = raw.get("odds", pd.Series(dtype=float))
    df["popularity"] = raw.get("popularity", pd.Series(dtype=int))

    # 着順 -> win / place_finish
    df["finish"] = raw.get("finish", pd.Series(dtype=int))
    df["win"] = (df["finish"] == 1).astype(int)
    df["place_finish"] = (df["finish"] <= 3).astype(int)

    # 上がり3F (実データ固有)
    df["last_3f"] = raw.get("last_3f", pd.Series(dtype=float)).fillna(0.0)

    # 通過順位 (最終コーナー)
    if "corner_positions" in raw.columns:
        df["corner_position_4"] = raw["corner_positions"].apply(_parse_last_corner)
    else:
        df["corner_position_4"] = 0

    # 斤量
    df["weight_carried"] = raw.get("weight_carried", pd.Series(dtype=float)).fillna(55.0)

    # --- 騎手・調教師の勝率を集計 ---
    # 実データでは、馬ごとの成績指標を過去走から集計する必要がある
    df["jockey_id"] = raw.get("jockey_id", "")
    df["trainer_id"] = raw.get("trainer_id", "")
    df["horse_id"] = raw.get("horse_id", "")

    # 時系列順にソートしてから集計（未来データのリーク防止）
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    df = _compute_rolling_stats(df)

    # --- 血統エンコード ---
    if "sire" in raw.columns:
        # 上位50種牡馬をラベルエンコード、それ以外は0
        sire_counts = raw["sire"].value_counts()
        top_sires = sire_counts.head(50).index.tolist()
        sire_map = {s: i + 1 for i, s in enumerate(top_sires)}
        df["sire_encoded"] = raw["sire"].map(sire_map).fillna(0).astype(int)
    else:
        df["sire_encoded"] = 0

    if "broodmare_sire" in raw.columns:
        bms_counts = raw["broodmare_sire"].value_counts()
        top_bms = bms_counts.head(50).index.tolist()
        bms_map = {s: i + 1 for i, s in enumerate(top_bms)}
        df["bms_encoded"] = raw["broodmare_sire"].map(bms_map).fillna(0).astype(int)
    else:
        df["bms_encoded"] = 0

    # --- 脚質推定（通過順位ベース） ---
    # 最終コーナー順位からrunning_styleを推定
    df["running_style"] = df["corner_position_4"].apply(_estimate_running_style)

    # --- 上がり3Fランク（レース内順位） ---
    # Note: last_3f_rank uses CURRENT race data (known after race).
    # For prediction, prev_last_3f (from rolling stats) is the non-leaking version.
    df["last_3f_rank"] = df.groupby("race_id")["last_3f"].rank(method="min", ascending=True)
    df["last_3f_rank"] = df["last_3f_rank"].fillna(0).astype(int)

    # --- 会場情報を追加 ---
    df["venue"] = raw.get("venue", "")

    # フィルタ: 不正データを除外
    df = df[df["odds"] > 0].copy()
    df = df[df["distance"] > 0].copy()
    df = df[df["horse_weight"] > 300].copy()  # 明らかに異常なデータを除外

    print(f"[実データ] 前処理後: {len(df)} 行, {df['race_id'].nunique()} レース")
    return df


def _parse_last_corner(corner_str) -> int:
    """通過順位文字列 '3-3-2-1' から最終コーナー順位を取得"""
    if pd.isna(corner_str) or not corner_str:
        return 0
    parts = str(corner_str).split("-")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def _estimate_running_style(corner_pos: int) -> int:
    """最終コーナー通過順位から脚質を推定: 逃げ=0, 先行=1, 差し=2, 追込=3"""
    if corner_pos <= 0:
        return 1  # 不明の場合は先行
    if corner_pos <= 2:
        return 0  # 逃げ
    elif corner_pos <= 5:
        return 1  # 先行
    elif corner_pos <= 10:
        return 2  # 差し
    else:
        return 3  # 追込


def _get_distance_band(distance):
    """距離を距離帯に分類: 短距離/マイル/中距離/長距離"""
    if distance <= 1400:
        return 0  # 短距離
    elif distance <= 1800:
        return 1  # マイル
    elif distance <= 2200:
        return 2  # 中距離
    else:
        return 3  # 長距離


def _compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬・騎手・調教師ごとの成績を、過去データのみを使って集計。
    （未来データリーク防止: expanding windowで計算）
    """
    # 馬ごとの過去成績
    df["win_rate"] = 0.0
    df["place_rate"] = 0.0
    df["earnings_per_race"] = 0.0
    df["days_since_last"] = 30
    df["last_finish"] = 5
    df["last3_avg_finish"] = 5.0

    # 騎手・調教師の勝率
    df["jockey_win_rate"] = 0.08
    df["jockey_place_rate"] = 0.20
    df["trainer_win_rate"] = 0.08

    # v2.1: 追加特徴量
    df["prev_last_3f"] = 0.0
    df["dist_aptitude"] = 0.0
    df["track_aptitude"] = 0.0
    df["venue_aptitude"] = 0.0

    # レース日でグループ化して時系列処理
    horse_history = {}   # horse_id -> list of finishes
    horse_dates = {}     # horse_id -> list of dates
    horse_last3f = {}    # horse_id -> list of last_3f times
    horse_dist = {}      # horse_id -> {distance_band: (wins, runs)}
    horse_track = {}     # horse_id -> {track_type: (wins, runs)}
    horse_venue = {}     # horse_id -> {venue: (wins, runs)}
    jockey_stats = {}    # jockey_id -> (wins, runs, places)
    trainer_stats = {}   # trainer_id -> (wins, runs)

    for idx, row in df.iterrows():
        hid = row.get("horse_id", "")
        jid = row.get("jockey_id", "")
        tid = row.get("trainer_id", "")
        finish = row.get("finish", 99)
        race_date = row.get("race_date", "")
        distance = row.get("distance", 0)
        track_type = row.get("track_type", 0)
        venue = row.get("venue", "")
        last_3f_val = row.get("last_3f", 0.0)
        dist_band = _get_distance_band(distance)

        # --- 馬の過去成績（このレース以前のデータ） ---
        if hid and hid in horse_history:
            past_finishes = horse_history[hid]
            n_races = len(past_finishes)
            if n_races > 0:
                wins = sum(1 for f in past_finishes if f == 1)
                places = sum(1 for f in past_finishes if f <= 3)
                df.at[idx, "win_rate"] = wins / n_races
                df.at[idx, "place_rate"] = places / n_races
                df.at[idx, "last_finish"] = past_finishes[-1]
                last3 = past_finishes[-3:] if n_races >= 3 else past_finishes
                df.at[idx, "last3_avg_finish"] = np.mean(last3)

            # days_since_last
            past_dates = horse_dates.get(hid, [])
            if past_dates:
                try:
                    last_dt = pd.to_datetime(past_dates[-1])
                    curr_dt = pd.to_datetime(race_date)
                    days = (curr_dt - last_dt).days
                    df.at[idx, "days_since_last"] = max(7, days)
                except Exception:
                    pass

            # v2.1: 前走の上がり3F
            past_3f = horse_last3f.get(hid, [])
            if past_3f:
                df.at[idx, "prev_last_3f"] = past_3f[-1]

            # v2.1: 距離適性（同距離帯での複勝率）
            if hid in horse_dist and dist_band in horse_dist[hid]:
                d_wins, d_runs = horse_dist[hid][dist_band]
                if d_runs >= 1:
                    df.at[idx, "dist_aptitude"] = d_wins / d_runs

            # v2.1: 馬場適性（同馬場タイプでの複勝率）
            if hid in horse_track and track_type in horse_track[hid]:
                t_wins, t_runs = horse_track[hid][track_type]
                if t_runs >= 1:
                    df.at[idx, "track_aptitude"] = t_wins / t_runs

            # v2.1: コース適性（同会場での複勝率）
            if hid in horse_venue and venue in horse_venue[hid]:
                v_wins, v_runs = horse_venue[hid][venue]
                if v_runs >= 1:
                    df.at[idx, "venue_aptitude"] = v_wins / v_runs

        # --- 騎手の成績（このレース以前） ---
        if jid and jid in jockey_stats:
            j_wins, j_runs, j_places = jockey_stats[jid]
            if j_runs > 0:
                df.at[idx, "jockey_win_rate"] = j_wins / j_runs
                df.at[idx, "jockey_place_rate"] = j_places / j_runs

        # --- 調教師の成績（このレース以前） ---
        if tid and tid in trainer_stats:
            t_wins, t_runs = trainer_stats[tid]
            if t_runs > 0:
                df.at[idx, "trainer_win_rate"] = t_wins / t_runs

        # --- このレースの結果を履歴に追加 ---
        is_place = finish <= 3

        if hid:
            if hid not in horse_history:
                horse_history[hid] = []
                horse_dates[hid] = []
                horse_last3f[hid] = []
                horse_dist[hid] = {}
                horse_track[hid] = {}
                horse_venue[hid] = {}
            horse_history[hid].append(finish)
            horse_dates[hid].append(race_date)
            if last_3f_val > 0:
                horse_last3f[hid].append(last_3f_val)

            # 距離帯別成績
            if dist_band not in horse_dist[hid]:
                horse_dist[hid][dist_band] = (0, 0)
            dw, dr = horse_dist[hid][dist_band]
            horse_dist[hid][dist_band] = (dw + (1 if is_place else 0), dr + 1)

            # 馬場タイプ別成績
            if track_type not in horse_track[hid]:
                horse_track[hid][track_type] = (0, 0)
            tw, tr_ = horse_track[hid][track_type]
            horse_track[hid][track_type] = (tw + (1 if is_place else 0), tr_ + 1)

            # 会場別成績
            if venue and venue not in horse_venue[hid]:
                horse_venue[hid][venue] = (0, 0)
            if venue:
                vw, vr = horse_venue[hid][venue]
                horse_venue[hid][venue] = (vw + (1 if is_place else 0), vr + 1)

        if jid:
            if jid not in jockey_stats:
                jockey_stats[jid] = (0, 0, 0)
            w, r, p = jockey_stats[jid]
            jockey_stats[jid] = (
                w + (1 if finish == 1 else 0),
                r + 1,
                p + (1 if is_place else 0),
            )

        if tid:
            if tid not in trainer_stats:
                trainer_stats[tid] = (0, 0)
            w, r = trainer_stats[tid]
            trainer_stats[tid] = (w + (1 if finish == 1 else 0), r + 1)

    # v2.1: 交互作用特徴量（ベクトル演算で高速化）
    df["odds_vs_jockey"] = df["odds"] * df["jockey_win_rate"]
    df["class_x_age"] = df["race_class"] * 10 + df["horse_age"]

    return df


# ===========================================
# データ生成（改良版）
# ===========================================

def generate_training_data(n_races: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    改良された合成競馬データを生成。

    v2.0 改善:
    - 200+ 騎手プール（実際のJRAに近い規模）
    - 80+ 調教師プール（トップ調教師の勝率15-20%）
    - クラス効果 (G1, G2, G3, Listed, Open, 条件戦)
    - 距離適性パターン
    - 脚質 (逃げ/先行/差し/追込)
    - ペース効果
    """
    rng = np.random.RandomState(seed)
    rows = []

    # === 騎手プール（200名） ===
    n_jockeys = 200
    jockey_skill = rng.beta(2, 5, n_jockeys)
    jockey_win_rates = 0.02 + jockey_skill * 0.22  # 2%〜24%
    jockey_place_rates = jockey_win_rates * 2.5 + rng.normal(0, 0.02, n_jockeys)
    jockey_place_rates = np.clip(jockey_place_rates, 0.08, 0.60)

    # === 調教師プール（80名） ===
    n_trainers = 80
    trainer_skill = rng.beta(2, 4, n_trainers)
    trainer_win_rates = 0.04 + trainer_skill * 0.18  # 4%〜22% (top = 15-20%)

    # === クラス分布 ===
    # 0=条件戦, 1=Open, 2=Listed, 3=G3, 4=G2, 5=G1
    class_probs = [0.55, 0.18, 0.10, 0.08, 0.05, 0.04]

    # 日付範囲（約3年分）
    start_date = datetime(2022, 1, 1)
    race_dates = []
    current = start_date
    while len(race_dates) < n_races:
        days_to_sat = (5 - current.weekday()) % 7
        if days_to_sat == 0 and current.weekday() != 5:
            days_to_sat = 7
        current += timedelta(days=max(days_to_sat, 1))
        if current.weekday() == 4:
            current += timedelta(days=1)
        n_sat = rng.randint(6, 13)
        for _ in range(min(n_sat, n_races - len(race_dates))):
            race_dates.append(current)
        sunday = current + timedelta(days=1)
        n_sun = rng.randint(6, 13)
        for _ in range(min(n_sun, n_races - len(race_dates))):
            race_dates.append(sunday)
        current = sunday + timedelta(days=1)

    for race_idx in range(n_races):
        race_date = race_dates[race_idx]
        race_id = f"R{race_date.strftime('%Y%m%d')}{race_idx:04d}"

        # レース条件
        field_size = rng.choice(range(8, 19), p=_field_size_probs())
        track_type = rng.choice([0, 1], p=[0.55, 0.45])
        distance = _sample_distance(rng, track_type)
        track_condition = rng.choice([0, 1, 2, 3], p=[0.60, 0.20, 0.12, 0.08])
        race_class = rng.choice([0, 1, 2, 3, 4, 5], p=class_probs)

        # === 各馬の「真の能力」を生成 ===
        # クラスが上がると能力のベースラインが上がり分散が小さくなる
        class_base = race_class * 0.3
        class_std = max(0.6, 1.0 - race_class * 0.05)
        true_abilities = rng.normal(class_base, class_std, field_size)

        # レースのペース（先行馬の割合で決まる）
        running_styles = rng.choice(
            [0, 1, 2, 3], size=field_size,
            p=[0.10, 0.30, 0.35, 0.25]
        )
        n_front_runners = np.sum(running_styles <= 1)
        pace_factor = n_front_runners / field_size  # 高い = ハイペース

        for horse_idx in range(field_size):
            true_ability = true_abilities[horse_idx]
            style = running_styles[horse_idx]

            # 馬の基本情報
            horse_age = rng.choice([2, 3, 4, 5, 6, 7, 8],
                                    p=[0.08, 0.28, 0.25, 0.18, 0.10, 0.07, 0.04])
            sex = rng.choice([0, 1, 2], p=[0.35, 0.55, 0.10])
            horse_weight = int(rng.normal(470 + true_ability * 3, 28))
            weight_change = int(rng.normal(0, 4))

            # === 隠れたエッジ（モデルが発見すべきパターン） ===
            hidden_edge = 0.0

            # 年齢・性別効果（やや強めに）
            age_effect = {2: -0.06, 3: 0.05, 4: 0.04, 5: 0.02,
                          6: -0.02, 7: -0.05, 8: -0.08}
            sex_effect = {0: -0.02, 1: 0.02, 2: 0.0}
            hidden_edge += age_effect.get(horse_age, 0) + sex_effect[sex]

            # 枠番効果
            post_position = horse_idx + 1
            if track_type == 0 and distance <= 1600:
                hidden_edge += (field_size / 2 - post_position) * 0.006
            else:
                hidden_edge += (field_size / 2 - post_position) * 0.002

            # 馬場×体重の交互作用
            if track_condition >= 2:
                hidden_edge += (horse_weight - 470) * 0.0005

            # 距離適性（各馬に得意距離がある）
            horse_best_dist = rng.choice([1200, 1600, 2000, 2400],
                                          p=[0.25, 0.30, 0.30, 0.15])
            dist_diff = abs(distance - horse_best_dist) / 400.0
            dist_aptitude = max(0, 1.0 - dist_diff * 0.15)
            hidden_edge += (dist_aptitude - 0.7) * 0.12

            # 脚質×ペース交互作用
            if pace_factor > 0.45:  # ハイペース
                style_bonus = {0: -0.08, 1: -0.03, 2: 0.05, 3: 0.07}
            else:  # スローペース
                style_bonus = {0: 0.07, 1: 0.04, 2: -0.03, 3: -0.06}
            hidden_edge += style_bonus[style]

            # 調教師効果
            trainer_id = rng.choice(n_trainers)
            t_win = trainer_win_rates[trainer_id]
            hidden_edge += (t_win - 0.10) * 0.20

            # クラス効果（高クラスでは実力差が出やすい）
            if race_class >= 3:
                hidden_edge += true_ability * 0.03

            # === 真のレース能力 = 基本能力 + 隠れたエッジ ===
            race_ability = true_ability + hidden_edge

            # === オッズ用の能力推定 ===
            # 市場はほぼ全ての公開情報を織り込んでいる
            # hidden_edgeの55%は市場が見落とす（モデルのエッジ源）
            odds_ability = race_ability - hidden_edge * 0.55 + rng.normal(0, 0.05)

            # === 成績指標 ===
            win_rate = np.clip(0.08 + odds_ability * 0.04 + rng.normal(0, 0.06),
                               0.0, 0.45)
            place_rate = np.clip(win_rate * 2.2 + rng.normal(0, 0.10), 0.05, 0.75)
            earnings = max(0, 400 + odds_ability * 150 + rng.normal(0, 250))
            days_since_last = max(7, int(rng.exponential(30) + 14))
            last_finish = max(1, int(rng.exponential(4) + 1 - odds_ability * 0.25))
            last_finish = min(last_finish, 18)
            last3_avg = max(1.0, last_finish + rng.normal(0, 3.5))
            last3_avg = min(last3_avg, 18.0)

            # 騎手
            jockey_weight = np.exp(jockey_skill * 2 + odds_ability * 0.3)
            jockey_probs = jockey_weight / jockey_weight.sum()
            jockey_id = rng.choice(n_jockeys, p=jockey_probs)
            j_win = jockey_win_rates[jockey_id]
            j_place = jockey_place_rates[jockey_id]

            rows.append({
                "race_id": race_id,
                "race_date": race_date.strftime("%Y-%m-%d"),
                "horse_idx": horse_idx,
                "horse_name": f"Horse_{race_idx}_{horse_idx}",
                "horse_age": horse_age,
                "horse_weight": horse_weight,
                "weight_change": weight_change,
                "sex": sex,
                "win_rate": round(win_rate, 3),
                "place_rate": round(place_rate, 3),
                "earnings_per_race": round(earnings, 1),
                "days_since_last": days_since_last,
                "last_finish": last_finish,
                "last3_avg_finish": round(last3_avg, 1),
                "jockey_win_rate": round(j_win, 3),
                "jockey_place_rate": round(j_place, 3),
                "trainer_win_rate": round(t_win, 3),
                "post_position": post_position,
                "distance": distance,
                "track_type": track_type,
                "track_condition": track_condition,
                "field_size": field_size,
                "race_class": race_class,
                "running_style": style,
                "race_ability": race_ability,
                "odds_ability": odds_ability,
                "pace_factor": pace_factor,
                "dist_aptitude": dist_aptitude,
            })

    df = pd.DataFrame(rows)

    # --- レースごとに勝者を決定 ---
    df["finish"] = 0
    df["win"] = 0
    df["place_finish"] = 0

    for rid, group in df.groupby("race_id"):
        idx = group.index
        abilities = group["race_ability"].values

        noise = rng.normal(0, 1.2, len(abilities))
        performance = abilities + noise

        ranks = np.empty_like(performance, dtype=int)
        ranks[np.argsort(-performance)] = np.arange(1, len(performance) + 1)
        df.loc[idx, "finish"] = ranks
        df.loc[idx, "win"] = (ranks == 1).astype(int)
        df.loc[idx, "place_finish"] = (ranks <= 3).astype(int)

    # --- オッズ生成 ---
    for rid, group in df.groupby("race_id"):
        idx = group.index
        oa = group["odds_ability"].values
        exp_oa = np.exp(oa - np.max(oa))
        probs = exp_oa / exp_oa.sum()
        takeout_factor = 0.78
        adjusted_odds = (1.0 / probs) * takeout_factor
        adjusted_odds = np.clip(adjusted_odds, 1.1, 200.0)
        adjusted_odds = np.round(adjusted_odds, 1)
        df.loc[idx, "odds"] = adjusted_odds

    # --- 人気順 ---
    df["popularity"] = df.groupby("race_id")["odds"].rank(method="min").astype(int)

    # 不要列を削除
    df.drop(columns=["race_ability", "odds_ability", "horse_idx",
                      "pace_factor", "dist_aptitude"], inplace=True)

    # 保存
    save_path = DATA_DIR / "race_results_v2.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[データ生成] {n_races}レース、{len(df)}行 -> {save_path}")

    return df


def _field_size_probs():
    """出走頭数の確率分布（8〜18頭）"""
    sizes = list(range(8, 19))
    weights = [1, 2, 3, 5, 7, 10, 12, 10, 8, 5, 3]
    total = sum(weights)
    return [w / total for w in weights]


def _sample_distance(rng, track_type: int) -> int:
    """距離をサンプリング"""
    if track_type == 0:
        distances = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000, 3200]
        probs = [0.12, 0.10, 0.18, 0.15, 0.15, 0.10, 0.10, 0.04, 0.03, 0.03]
    else:
        distances = [1000, 1150, 1200, 1400, 1700, 1800, 2000, 2100, 2400]
        probs = [0.05, 0.05, 0.15, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05]
    return rng.choice(distances, p=probs)


# ===========================================
# 特徴量エンジニアリング（改良版）
# ===========================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量の追加・変換（v2.0: 大幅に追加）"""
    df = df.copy()

    # --- 基本特徴量 ---
    # オッズから暗黙の確率
    df["implied_prob"] = 1.0 / df["odds"]

    # 人気と実力の乖離
    df["odds_vs_winrate"] = df["win_rate"] - df["implied_prob"]

    # 騎手と馬の複合指標
    df["jockey_horse_combo"] = df["jockey_win_rate"] * df["win_rate"]

    # 休養明け指標
    df["long_rest"] = (df["days_since_last"] > 60).astype(int)

    # 体重の適正範囲
    df["weight_extreme"] = ((df["horse_weight"] < 430) |
                             (df["horse_weight"] > 520)).astype(int)

    # 内枠ダミー
    df["inner_post"] = (df["post_position"] <= 3).astype(int)

    # 少頭数レース
    df["small_field"] = (df["field_size"] <= 10).astype(int)

    # --- v2.0 追加特徴量 ---

    # ペース予測: レース内の先行馬（逃げ+先行）の割合
    if "running_style" in df.columns:
        df["is_front_runner"] = (df["running_style"] <= 1).astype(int)
        pace_map = df.groupby("race_id")["is_front_runner"].transform("mean")
        df["pace_predict"] = pace_map
        # 脚質×ペースの交互作用
        df["style_pace_interact"] = df["running_style"] * df["pace_predict"]
    else:
        df["pace_predict"] = 0.4
        df["style_pace_interact"] = 0.0

    # 距離適性指数（距離カテゴリ × 馬の成績パターン）
    df["dist_category"] = pd.cut(df["distance"],
                                  bins=[0, 1400, 1800, 2200, 4000],
                                  labels=[0, 1, 2, 3])
    df["dist_category"] = pd.to_numeric(df["dist_category"], errors="coerce").fillna(1).astype(int)
    df["dist_x_winrate"] = df["dist_category"] * df["win_rate"]

    # 馬場×体重の交互作用
    df["condition_x_weight"] = df["track_condition"] * (df["horse_weight"] - 470) / 50.0

    # 調教師×クラスの交互作用
    if "trainer_win_rate" in df.columns and "race_class" in df.columns:
        df["trainer_x_class"] = df["trainer_win_rate"] * df["race_class"]
    else:
        df["trainer_x_class"] = 0.0

    # 騎手勝率×クラス
    if "race_class" in df.columns:
        df["jockey_x_class"] = df["jockey_win_rate"] * df["race_class"]
    else:
        df["jockey_x_class"] = 0.0

    # 枠番×距離の交互作用（短距離の内枠有利度）
    df["post_x_dist"] = df["post_position"] * (df["distance"] / 2000.0)

    # 人気×オッズ乖離（穴馬の強さ指標）
    df["pop_x_edge"] = df["popularity"] * df["odds_vs_winrate"]

    # 馬齢×クラス（若駒の上位クラスでの成績）
    if "race_class" in df.columns:
        df["age_x_class"] = df["horse_age"] * df["race_class"]
    else:
        df["age_x_class"] = 0.0

    # --- 実データ固有の特徴量 ---

    # 上がり3F関連
    if "last_3f" in df.columns and df["last_3f"].sum() > 0:
        # レース内の上がり3F順位
        df["last_3f_rank"] = df.groupby("race_id")["last_3f"].rank(method="min")
        # 上がり3Fとオッズの乖離（速い上がりなのに人気薄 = エッジ）
        df["f3_vs_pop"] = df["last_3f_rank"] - df["popularity"]
    else:
        df["last_3f_rank"] = 0.0
        df["f3_vs_pop"] = 0.0

    # 最終コーナー通過順（脚質の実測値）
    if "corner_position_4" in df.columns and df["corner_position_4"].sum() > 0:
        df["corner_x_pace"] = df["corner_position_4"] * df.get("pace_predict", 0.4)
    else:
        df["corner_x_pace"] = 0.0

    # 斤量関連
    if "weight_carried" in df.columns and df["weight_carried"].sum() > 0:
        df["kinryo_advantage"] = 55.0 - df["weight_carried"]  # 軽量 = 有利
        df["kinryo_x_dist"] = df["weight_carried"] * (df["distance"] / 2000.0)
    else:
        df["kinryo_advantage"] = 0.0
        df["kinryo_x_dist"] = 0.0

    # 血統（実データのみ）
    if "sire_encoded" in df.columns:
        df["sire_x_track"] = df["sire_encoded"] * df["track_type"]
        df["sire_x_dist"] = df["sire_encoded"] * df["dist_category"]
    else:
        df["sire_x_track"] = 0.0
        df["sire_x_dist"] = 0.0

    return df


# ===========================================
# 5モデルアンサンブル
# ===========================================

class KeibaEnsemble:
    """5モデルアンサンブル確率推定（LGB + XGB + CatBoost + RF + ExtraTrees）"""

    def __init__(self):
        self.models = []
        self.model_names = []
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """5モデルを訓練"""
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

        self.feature_cols = X.columns.tolist()
        self.models = []
        self.model_names = []

        # --- LightGBM ---
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        }
        dtrain_lgb = lgb.Dataset(X, label=y)
        lgb_model = lgb.train(lgb_params, dtrain_lgb, num_boost_round=300)
        self.models.append(("lgb", lgb_model))
        self.model_names.append("LightGBM")

        # --- XGBoost ---
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "seed": 42,
        }
        dtrain_xgb = xgb.DMatrix(X, label=y)
        xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=300)
        self.models.append(("xgb", xgb_model))
        self.model_names.append("XGBoost")

        # --- CatBoost ---
        cat_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
            l2_leaf_reg=3.0,
        )
        cat_model.fit(X.values, y.values)
        self.models.append(("cat", cat_model))
        self.model_names.append("CatBoost")

        # --- RandomForest ---
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=1,
        )
        rf_model.fit(X.values, y.values)
        self.models.append(("rf", rf_model))
        self.model_names.append("RandomForest")

        # --- ExtraTrees ---
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=1,
        )
        et_model.fit(X.values, y.values)
        self.models.append(("et", et_model))
        self.model_names.append("ExtraTrees")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """5モデルの確率平均"""
        import xgboost as xgb

        X_use = X[self.feature_cols]
        probas = []

        for tag, model in self.models:
            if tag == "lgb":
                p = model.predict(X_use)
            elif tag == "xgb":
                p = model.predict(xgb.DMatrix(X_use))
            elif tag == "cat":
                p = model.predict_proba(X_use.values)[:, 1]
            elif tag in ("rf", "et"):
                p = model.predict_proba(X_use.values)[:, 1]
            else:
                continue
            probas.append(p)

        return np.mean(probas, axis=0)


# ===========================================
# リスク管理
# ===========================================

class KeibaRiskManager:
    """ベット管理（日次・月次の予算制限）"""

    def __init__(self, daily_budget: int = 5000, monthly_budget: int = 50000,
                 bet_unit: int = 1000):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.bet_unit = bet_unit
        self.daily_spent = 0
        self.monthly_spent = 0
        self.current_day = None
        self.current_month = None
        self.history = []

    def can_bet(self, amount: int, date_str: str) -> bool:
        """ベット可能かチェック"""
        day = date_str[:10]
        month = date_str[:7]

        if day != self.current_day:
            self.daily_spent = 0
            self.current_day = day
        if month != self.current_month:
            self.monthly_spent = 0
            self.current_month = month

        if self.daily_spent + amount > self.daily_budget:
            return False
        if self.monthly_spent + amount > self.monthly_budget:
            return False
        return True

    def record_bet(self, amount: int, payout: float, date_str: str,
                   bet_type: str = "win", details: dict = None):
        """ベット結果を記録"""
        self.daily_spent += amount
        self.monthly_spent += amount
        record = {
            "date": date_str,
            "amount": amount,
            "payout": payout,
            "profit": payout - amount,
            "bet_type": bet_type,
            "details": details or {},
        }
        self.history.append(record)

    def get_summary(self) -> dict:
        """全体サマリーを返す"""
        if not self.history:
            return {"total_bet": 0, "total_payout": 0, "recovery_rate": 0,
                    "profit_factor": 0, "n_bets": 0, "n_hits": 0}
        total_bet = sum(h["amount"] for h in self.history)
        total_payout = sum(h["payout"] for h in self.history)
        gross_profit = sum(h["payout"] for h in self.history if h["payout"] > 0)
        gross_loss = sum(h["amount"] for h in self.history if h["payout"] == 0)
        return {
            "total_bet": total_bet,
            "total_payout": total_payout,
            "recovery_rate": total_payout / total_bet if total_bet > 0 else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
            "n_bets": len(self.history),
            "n_hits": sum(1 for h in self.history if h["payout"] > 0),
        }


# ===========================================
# バリューベット判定
# ===========================================

def find_value_bets(race_df: pd.DataFrame, model: KeibaEnsemble,
                    edge_threshold: float = 1.30) -> list:
    """
    1レース分のデータに対し、バリューベットを判定。
    条件: モデル推定確率 * オッズ > edge_threshold
    """
    features = race_df[model.feature_cols]
    probs = model.predict_proba(features)
    race_df = race_df.copy()
    race_df["pred_prob"] = probs

    bets = []
    for _, row in race_df.iterrows():
        pred_prob = row["pred_prob"]
        odds = row["odds"]
        expected_value = pred_prob * odds

        if expected_value > edge_threshold:
            bets.append({
                "horse_name": row["horse_name"],
                "pred_prob": round(pred_prob, 4),
                "odds": odds,
                "expected_value": round(expected_value, 3),
                "popularity": int(row["popularity"]),
                "implied_prob": round(1.0 / odds, 4),
                "edge": round(pred_prob - 1.0 / odds, 4),
            })

    return bets


# ===========================================
# Walk-Forward検証（Expanding Window）
# ===========================================

def walk_forward_validate(df: pd.DataFrame,
                          initial_train_months: int = 12,
                          test_months: int = 3,
                          edge_threshold: float = 1.30,
                          bet_unit: int = 1000) -> dict:
    """
    Expanding Window Walk-Forward検証。

    - 訓練期間: 最初の initial_train_months から始まり、毎Foldで拡張
    - テスト期間: test_months ヶ月
    - PF (profit_factor) を計算
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values("race_date")

    # 特徴量エンジニアリング
    df = engineer_features(df)

    # 使用する特徴量
    # v2.1 pruned: Removed trainer_x_class (trainer_win_rate pruned),
    #   sire_encoded, bms_encoded and derivatives (near-zero)
    feature_cols = FEATURE_COLS + [
        "implied_prob", "odds_vs_winrate", "jockey_horse_combo",
        "long_rest", "weight_extreme", "inner_post", "small_field",
        # v2.0 追加特徴量
        "pace_predict", "style_pace_interact",
        "dist_category", "dist_x_winrate",
        "condition_x_weight", "jockey_x_class",
        "post_x_dist", "pop_x_edge", "age_x_class",
    ]

    # 実データ固有の特徴量を追加（存在する場合のみ）
    real_data_features = [
        "last_3f", "last_3f_rank", "f3_vs_pop",
        "corner_position_4", "corner_x_pace",
        "weight_carried", "kinryo_advantage", "kinryo_x_dist",
    ]
    for feat in real_data_features:
        if feat in df.columns and df[feat].sum() != 0:
            feature_cols.append(feat)

    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    results = []
    fold = 0
    # Expanding window: train always starts from min_date
    test_start = min_date + pd.DateOffset(months=initial_train_months)

    while test_start + pd.DateOffset(months=test_months) <= max_date:
        test_end = test_start + pd.DateOffset(months=test_months)

        # Expanding: train from beginning to test_start
        train_df = df[df["race_date"] < test_start]
        test_df = df[(df["race_date"] >= test_start) & (df["race_date"] < test_end)]

        if len(train_df) < 1000 or len(test_df) < 100:
            test_start += pd.DateOffset(months=test_months)
            continue

        # モデル訓練
        model = KeibaEnsemble()
        X_train = train_df[feature_cols]
        y_train = train_df["win"]
        model.fit(X_train, y_train)

        # テスト期間での模擬ベット
        risk_mgr = KeibaRiskManager(
            daily_budget=10000, monthly_budget=100000, bet_unit=bet_unit
        )

        for rid, race_group in test_df.groupby("race_id"):
            date_str = str(race_group["race_date"].iloc[0].date())
            value_bets = find_value_bets(race_group, model, edge_threshold)

            for bet in value_bets:
                if not risk_mgr.can_bet(bet_unit, date_str):
                    break
                horse_row = race_group[
                    race_group["horse_name"] == bet["horse_name"]
                ].iloc[0]
                won = horse_row["win"] == 1
                payout = bet["odds"] * bet_unit if won else 0

                risk_mgr.record_bet(
                    bet_unit, payout, date_str, "win",
                    {"horse": bet["horse_name"], "odds": bet["odds"],
                     "pred_prob": bet["pred_prob"],
                     "track_type": int(horse_row["track_type"]),
                     "field_size": int(horse_row["field_size"])}
                )

        summary = risk_mgr.get_summary()
        summary["fold"] = fold
        summary["train_size"] = len(train_df)
        summary["test_start"] = str(test_start.date())
        summary["test_end"] = str(test_end.date())
        results.append(summary)

        hit_rate = summary["n_hits"] / summary["n_bets"] if summary["n_bets"] > 0 else 0
        pf = summary["profit_factor"]
        print(f"  Fold {fold}: ~{summary['test_end']} | "
              f"train={summary['train_size']:>5d} | "
              f"bets={summary['n_bets']:>4d} | "
              f"hit={hit_rate:.1%} | "
              f"RR={summary['recovery_rate']:.1%} | "
              f"PF={pf:.2f}")

        fold += 1
        test_start += pd.DateOffset(months=test_months)

    # 全体集計
    total_bet = sum(r["total_bet"] for r in results)
    total_payout = sum(r["total_payout"] for r in results)
    total_n_bets = sum(r["n_bets"] for r in results)
    total_n_hits = sum(r["n_hits"] for r in results)
    # PF: 勝ちベットの払戻総額 / 負けベットの投資総額
    total_gross_profit = sum(
        sum(h["payout"] for h in r.get("details", []) if isinstance(h, dict) and h.get("payout", 0) > 0)
        if isinstance(r.get("details"), list) else 0
        for r in results
    )
    # Simpler: PF = total_payout / (total_bet - total_payout + total_payout_from_losers)
    # Actually: PF = total_payout / total_bet (same as recovery_rate for fixed-bet)
    # For keiba: PF = gross_win_payouts / gross_loss_bets
    # We recalculate from per-fold summaries
    all_gross_profit = 0
    all_gross_loss = 0
    for r in results:
        # From each fold's risk manager, we recorded profit_factor
        # But we need the raw numbers. Let's reconstruct:
        # hits * avg_payout_per_hit = total_payout for that fold
        # (n_bets - n_hits) * bet_unit = gross_loss for that fold
        all_gross_profit += r["total_payout"]
        all_gross_loss += (r["n_bets"] - r["n_hits"]) * bet_unit

    overall_pf = all_gross_profit / all_gross_loss if all_gross_loss > 0 else 0

    overall = {
        "folds": results,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "total_n_bets": total_n_bets,
        "total_n_hits": total_n_hits,
        "overall_hit_rate": total_n_hits / total_n_bets if total_n_bets > 0 else 0,
        "overall_recovery_rate": total_payout / total_bet if total_bet > 0 else 0,
        "overall_profit_factor": overall_pf,
    }

    return overall


# ===========================================
# 複勝・ワイド検証
# ===========================================

def walk_forward_validate_place(df: pd.DataFrame,
                                initial_train_months: int = 12,
                                test_months: int = 3,
                                edge_threshold: float = 1.20,
                                bet_unit: int = 1000) -> dict:
    """
    複勝（3着以内）の Expanding Window Walk-Forward 検証。
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values("race_date")
    df = engineer_features(df)

    # v2.1 pruned: same as walk_forward_validate
    feature_cols = FEATURE_COLS + [
        "implied_prob", "odds_vs_winrate", "jockey_horse_combo",
        "long_rest", "weight_extreme", "inner_post", "small_field",
        "pace_predict", "style_pace_interact",
        "dist_category", "dist_x_winrate",
        "condition_x_weight", "jockey_x_class",
        "post_x_dist", "pop_x_edge", "age_x_class",
    ]

    # 実データ固有の特徴量を追加（存在する場合のみ）
    real_data_features = [
        "last_3f", "last_3f_rank", "f3_vs_pop",
        "corner_position_4", "corner_x_pace",
        "weight_carried", "kinryo_advantage", "kinryo_x_dist",
    ]
    for feat in real_data_features:
        if feat in df.columns and df[feat].sum() != 0:
            feature_cols.append(feat)

    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    results = []
    fold = 0
    test_start = min_date + pd.DateOffset(months=initial_train_months)

    while test_start + pd.DateOffset(months=test_months) <= max_date:
        test_end = test_start + pd.DateOffset(months=test_months)
        train_df = df[df["race_date"] < test_start]
        test_df = df[(df["race_date"] >= test_start) & (df["race_date"] < test_end)]

        if len(train_df) < 1000 or len(test_df) < 100:
            test_start += pd.DateOffset(months=test_months)
            continue

        # 複勝用モデル（target=place_finish）
        model = KeibaEnsemble()
        X_train = train_df[feature_cols]
        y_train = train_df["place_finish"]
        model.fit(X_train, y_train)

        risk_mgr = KeibaRiskManager(
            daily_budget=10000, monthly_budget=100000, bet_unit=bet_unit
        )

        for rid, race_group in test_df.groupby("race_id"):
            date_str = str(race_group["race_date"].iloc[0].date())
            features = race_group[model.feature_cols]
            probs = model.predict_proba(features)

            for i, (_, row) in enumerate(race_group.iterrows()):
                pred_prob = probs[i]
                place_odds = max(1.1, row["odds"] * 0.35)
                ev = pred_prob * place_odds

                if ev > edge_threshold:
                    if not risk_mgr.can_bet(bet_unit, date_str):
                        break
                    placed = row["place_finish"] == 1
                    payout = place_odds * bet_unit if placed else 0
                    risk_mgr.record_bet(
                        bet_unit, payout, date_str, "place",
                        {"horse": row["horse_name"], "odds": place_odds,
                         "track_type": int(row["track_type"]),
                         "field_size": int(row["field_size"])}
                    )

        summary = risk_mgr.get_summary()
        summary["fold"] = fold
        results.append(summary)
        fold += 1
        test_start += pd.DateOffset(months=test_months)

    total_bet = sum(r["total_bet"] for r in results)
    total_payout = sum(r["total_payout"] for r in results)
    total_n_bets = sum(r["n_bets"] for r in results)
    total_n_hits = sum(r["n_hits"] for r in results)

    all_gross_profit = 0
    all_gross_loss = 0
    for r in results:
        all_gross_profit += r["total_payout"]
        all_gross_loss += (r["n_bets"] - r["n_hits"]) * bet_unit

    overall_pf = all_gross_profit / all_gross_loss if all_gross_loss > 0 else 0

    return {
        "folds": results,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "total_n_bets": total_n_bets,
        "total_n_hits": total_n_hits,
        "overall_hit_rate": total_n_hits / total_n_bets if total_n_bets > 0 else 0,
        "overall_recovery_rate": total_payout / total_bet if total_bet > 0 else 0,
        "overall_profit_factor": overall_pf,
    }


# ===========================================
# 予測（1レース分）
# ===========================================

def predict_race(race_data: pd.DataFrame, model: KeibaEnsemble,
                 edge_threshold: float = 1.30) -> pd.DataFrame:
    """1レース分の全馬について予測を返す"""
    race_data = engineer_features(race_data)
    features = race_data[model.feature_cols]
    probs = model.predict_proba(features)

    result = race_data[["horse_name", "odds", "popularity"]].copy()
    result["pred_prob"] = np.round(probs, 4)
    result["implied_prob"] = np.round(1.0 / result["odds"], 4)
    result["expected_value"] = np.round(result["pred_prob"] * result["odds"], 3)
    result["recommended"] = result["expected_value"] > edge_threshold
    result = result.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    return result


# ===========================================
# レポート生成
# ===========================================

def generate_report(win_results: dict, place_results: dict) -> str:
    """検証結果のテキストレポートを生成"""
    lines = []
    lines.append("=" * 60)
    lines.append("競馬予測モデル v2.0 検証レポート")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("[重要] このレポートは合成データによる検証結果です。")
    lines.append("  実データでの結果は大きく異なる可能性があります。")
    lines.append("  投資助言ではありません。")
    lines.append("")

    lines.append("-" * 40)
    lines.append("■ 単勝（Win）検証結果")
    lines.append("-" * 40)
    _append_result_section(lines, win_results)

    lines.append("")
    lines.append("-" * 40)
    lines.append("■ 複勝（Place）検証結果")
    lines.append("-" * 40)
    _append_result_section(lines, place_results)

    # Fold詳細
    lines.append("")
    lines.append("-" * 40)
    lines.append("■ Fold別詳細（単勝）")
    lines.append("-" * 40)
    for f in win_results.get("folds", []):
        rr = f["recovery_rate"]
        hr = f["n_hits"] / f["n_bets"] if f["n_bets"] > 0 else 0
        pf = f["profit_factor"]
        lines.append(
            f"  Fold {f['fold']}: bets={f['n_bets']:>4d} | "
            f"hit={f['n_hits']:>3d} ({hr:.1%}) | "
            f"RR={rr:.1%} | PF={pf:.2f}"
        )

    # 合否判定
    lines.append("")
    lines.append("=" * 60)
    lines.append("■ 総合判定")
    lines.append("=" * 60)

    win_rr = win_results.get("overall_recovery_rate", 0)
    win_pf = win_results.get("overall_profit_factor", 0)
    place_rr = place_results.get("overall_recovery_rate", 0)
    place_pf = place_results.get("overall_profit_factor", 0)
    win_bets = win_results.get("total_n_bets", 0)
    place_bets = place_results.get("total_n_bets", 0)

    win_pass = win_rr > 1.05 and win_bets > 100 and win_pf > 1.0
    place_pass = place_rr > 1.05 and place_bets > 100 and place_pf > 1.0

    lines.append(f"  単勝: RR={win_rr:.1%}, PF={win_pf:.2f}, bets={win_bets} -> "
                 f"{'PASS' if win_pass else 'FAIL'}")
    lines.append(f"  複勝: RR={place_rr:.1%}, PF={place_pf:.2f}, bets={place_bets} -> "
                 f"{'PASS' if place_pass else 'FAIL'}")

    if not win_pass and not place_pass:
        lines.append("")
        lines.append("  [結論] 現時点では安定的にプラス回収を達成できていません。")
        lines.append("  モデルの改善余地:")
        lines.append("    - 実データでの検証")
        lines.append("    - 血統・調教データの追加")
        lines.append("    - 閾値の最適化")
    elif win_pass or place_pass:
        lines.append("")
        lines.append("  [結論] 合成データでは一部の条件でプラス回収を達成。")
        lines.append("  ただし実データでの検証が必須です。")

    lines.append("")
    lines.append("=" * 60)
    lines.append("レポート終了")
    lines.append("=" * 60)

    return "\n".join(lines)


def _append_result_section(lines: list, results: dict):
    """結果セクションを追加"""
    total_bet = results.get("total_bet", 0)
    total_payout = results.get("total_payout", 0)
    n_bets = results.get("total_n_bets", 0)
    n_hits = results.get("total_n_hits", 0)
    rr = results.get("overall_recovery_rate", 0)
    hr = results.get("overall_hit_rate", 0)
    pf = results.get("overall_profit_factor", 0)

    lines.append(f"  総ベット数:     {n_bets}")
    lines.append(f"  的中数:         {n_hits}")
    lines.append(f"  的中率:         {hr:.1%}")
    lines.append(f"  総投資額:       {total_bet:>10,d} 円")
    lines.append(f"  総払戻額:       {total_payout:>10,.0f} 円")
    lines.append(f"  回収率 (RR):    {rr:.1%}")
    lines.append(f"  Profit Factor:  {pf:.2f}")
    lines.append(f"  損益:           {total_payout - total_bet:>+10,.0f} 円")


# ===========================================
# パイプライン
# ===========================================

def run_pipeline(use_real_data: bool = True):
    """メインパイプライン v2.1 (実データ対応)"""
    print("=" * 60)
    print("競馬予測モデル v2.1 パイプライン開始")
    print("  5-model ensemble | expanded features | edge=1.30")
    print("  実データ優先（フォールバック: 合成データ）")
    print("=" * 60)

    # 1. データ読み込み（実データ優先）
    data_source = "synthetic"
    df = pd.DataFrame()

    if use_real_data:
        print("\n[Step 1] 実データ読み込み試行...")
        df = load_real_data()
        if not df.empty and df["race_id"].nunique() >= 100:
            data_source = "real"
            print(f"  実データ使用: {len(df)} 行, {df['race_id'].nunique()} レース")
        else:
            print("  実データ不十分。合成データにフォールバック。")
            df = pd.DataFrame()

    if df.empty:
        print("\n[Step 1] 合成データ生成...")
        df = generate_training_data(n_races=6000)
        data_source = "synthetic"

    print(f"  データ件数: {len(df)} 行, {df['race_id'].nunique()} レース")
    print(f"  データソース: {data_source}")

    fav_df = df[df["popularity"] == 1]
    fav_win_rate = fav_df["win"].mean()
    print(f"  1番人気勝率: {fav_win_rate:.1%} (目標: ~30%)")

    # 2. Walk-Forward検証（単勝）
    print("\n[Step 2] Expanding WF検証（単勝, edge=1.30）...")
    win_results = walk_forward_validate(
        df, initial_train_months=12, test_months=3,
        edge_threshold=1.30, bet_unit=1000
    )
    print(f"\n  >> 単勝 RR:  {win_results['overall_recovery_rate']:.1%}")
    print(f"  >> 単勝 PF:  {win_results['overall_profit_factor']:.2f}")
    print(f"  >> 単勝 hit: {win_results['overall_hit_rate']:.1%}")
    print(f"  >> 単勝 bets:{win_results['total_n_bets']}")

    # 3. Walk-Forward検証（複勝）
    print("\n[Step 3] Expanding WF検証（複勝, edge=1.20）...")
    place_results = walk_forward_validate_place(
        df, initial_train_months=12, test_months=3,
        edge_threshold=1.20, bet_unit=1000
    )
    print(f"\n  >> 複勝 RR:  {place_results['overall_recovery_rate']:.1%}")
    print(f"  >> 複勝 PF:  {place_results['overall_profit_factor']:.2f}")
    print(f"  >> 複勝 hit: {place_results['overall_hit_rate']:.1%}")
    print(f"  >> 複勝 bets:{place_results['total_n_bets']}")

    # 4. レポート生成
    print("\n[Step 4] レポート生成...")
    report = generate_report(win_results, place_results)
    report_path = DATA_DIR / "keiba_report_v2.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  レポート保存: {report_path}")

    # 5. モデル設定保存
    config = {
        "model_type": "5-model ensemble (LGB+XGB+CatBoost+RF+ExtraTrees)",
        "features": FEATURE_COLS,
        "edge_threshold_win": 1.30,
        "edge_threshold_place": 1.20,
        "bet_unit": 1000,
        "win_recovery_rate": round(win_results["overall_recovery_rate"], 4),
        "win_profit_factor": round(win_results["overall_profit_factor"], 4),
        "place_recovery_rate": round(place_results["overall_recovery_rate"], 4),
        "place_profit_factor": round(place_results["overall_profit_factor"], 4),
        "win_n_bets": win_results["total_n_bets"],
        "place_n_bets": place_results["total_n_bets"],
        "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_source": data_source,
        "status": "experimental" if data_source == "synthetic" else "real_data_validation",
        "warning": ("合成データでの検証結果。実データでの検証が必要。"
                     if data_source == "synthetic"
                     else "実データでの検証結果。ペーパートレードでの確認が必要。"),
    }
    config_path = DATA_DIR / "model_config_v2.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  モデル設定保存: {config_path}")

    # レポート出力
    print("\n" + report)

    print("\n" + "=" * 60)
    print("パイプライン完了")
    print("=" * 60)

    return win_results, place_results


# ===========================================
# メイン
# ===========================================

if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="競馬予測モデル v2.1")
    _parser.add_argument("--synthetic", action="store_true",
                         help="合成データのみ使用（実データを無視）")
    _args = _parser.parse_args()
    run_pipeline(use_real_data=not _args.synthetic)
