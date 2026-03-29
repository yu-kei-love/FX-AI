# ===========================================
# feature_engine.py
# ボートレースAI - 74特徴量エンジン
#
# 設計方針：
#   - DBからデータを読み込んで74特徴量を計算する
#   - course_takenを艇番と混同しない（最重要）
#   - wind_directionはsin/cos変換
#   - 欠損値はレース内平均で補完
#   - 正規化不要（ツリーモデルのため）
#   - 全特徴量を保存・削除しない
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

from data_interface import DataInterface

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "boat" / "boatrace.db"

# 会場別の1コース勝率（マスタデータ）
VENUE_COURSE1_WIN_RATE = {
    1: 0.556, 2: 0.520, 3: 0.488, 4: 0.510, 5: 0.528,
    6: 0.535, 7: 0.548, 8: 0.557, 9: 0.542, 10: 0.538,
    11: 0.518, 12: 0.522, 13: 0.543, 14: 0.523, 15: 0.538,
    16: 0.530, 17: 0.548, 18: 0.553, 19: 0.545, 20: 0.545,
    21: 0.525, 22: 0.548, 23: 0.518, 24: 0.532,
}

# グレード・レースタイプの数値化
GRADE_MAP   = {"SG": 4, "G1": 3, "G2": 2, "G3": 1, "一般": 0, None: 0}
RACETYPE_MAP = {"優勝戦": 3, "準優": 2, "予選": 1, None: 0}


# =============================================================
# データ読み込み
# =============================================================

def load_race_data(start_date=None, end_date=None, data_interface=None, db_path=None):
    """
    DBからレースデータを読み込む。
    DataInterfaceを経由することでsqlite/csv/mockモードを切り替え可能。

    Parameters:
        start_date      : 開始日 "YYYYMMDD"（Noneの場合は全期間）
        end_date        : 終了日 "YYYYMMDD"（Noneの場合は全期間）
        data_interface  : DataInterfaceインスタンス（指定なければ自動生成）
        db_path         : 後方互換性のためdb_pathも受け付ける

    Returns:
        races_df   : レース情報 DataFrame
        entries_df : 出走情報 DataFrame（course_taken含む）
        odds_df    : オッズ時系列 DataFrame
    """
    # DataInterfaceが未指定の場合は自動生成
    if data_interface is None:
        if db_path:
            data_interface = DataInterface(mode="sqlite", db_path=str(db_path))
        else:
            data_interface = DataInterface(mode="sqlite", db_path=str(DB_PATH))

    races_df = data_interface.get_races(start_date=start_date, end_date=end_date)
    entries_df = data_interface.get_entries(start_date=start_date, end_date=end_date)
    odds_df = data_interface.get_odds()

    return races_df, entries_df, odds_df


# =============================================================
# メイン特徴量生成
# =============================================================

def create_features(entries_df, races_df, odds_df):
    """
    74特徴量を計算して返す。

    Parameters:
        entries_df : 出走情報 DataFrame
        races_df   : レース情報 DataFrame
        odds_df    : オッズ時系列 DataFrame

    Returns:
        features_df : 特徴量 DataFrame（race_id・lane・course_taken付き）
    """
    # レース情報を出走情報にマージ
    df = entries_df.merge(races_df, on="race_id", how="left", suffixes=("", "_race"))

    # 各カテゴリの特徴量を計算
    df = calc_racer_features(df)
    df = calc_course_features(df)
    df = calc_condition_features(df)
    df = calc_equipment_features(df)
    df = calc_environment_features(df)
    df = calc_venue_features(df)
    df = calc_race_features(df)
    df = calc_odds_features(df, odds_df)
    df = calc_h2h_features(df)
    df = calc_relative_features(df)

    # 欠損値を補完（レース内平均）
    df = _fill_na_inrace(df)

    return df


def _fill_na_inrace(df):
    """欠損値をレース内平均で補完する。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # race_id・lane・finish・course_takenは補完しない
    skip_cols = {"entry_id", "lane", "finish", "course_taken", "racer_id",
                 "home_venue_id", "venue_id", "race_no", "motor_no", "boat_no"}
    target_cols = [c for c in numeric_cols if c not in skip_cols]

    for col in target_cols:
        df[col] = df.groupby("race_id")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    return df


# =============================================================
# カテゴリA：選手基本（10特徴量）
# =============================================================

def calc_racer_features(df):
    """
    A-01 racer_class（級別 A1=4/A2=3/B1=2/B2=1）
    A-02 national_win_rate（全国勝率）
    A-03 national_2place_rate（全国2着率）
    A-04 national_3place_rate（全国3着率）
    A-05 local_win_rate（当地勝率）
    A-06 local_2place_rate（当地2着率）
    A-07 is_home（地元フラグ）
    A-08 local_vs_national（当地 - 全国勝率）
    A-09 racer_weight（体重）
    A-10 racer_age（年齢）
    """
    # DBの列名をそのまま使う（既存列名と一致）
    # A-08のみ計算が必要
    df["local_vs_national"] = (
        df["local_win_rate"].fillna(0) - df["national_win_rate"].fillna(0)
    )
    return df


# =============================================================
# カテゴリB：コース・スタート（12特徴量）
# =============================================================

def calc_course_features(df):
    """
    B-01 course_taken（実際の進入コース）← 最重要
    B-02 lane（艇番）
    B-03 course_vs_lane（進入コース - 艇番）
    B-04 is_course_changed（枠番と違うフラグ）
    B-05 course_win_rate（実際のコースでの選手勝率）
    B-06 course_2place_rate（同2着率）
    B-07 course_avg_st（実際のコースでの平均ST）
    B-08 avg_start_timing（全体平均ST）
    B-09 flying_count（フライング回数）
    B-10 late_count（出遅れ回数）
    B-11 days_since_last_flying（最後のFから経過日数）
    B-12 is_flying_return（F休み明けフラグ）
    """
    # B-03: コースと艇番の差（前付けの度合い）
    df["course_vs_lane"] = df["course_taken"].fillna(df["lane"]) - df["lane"]

    # B-04: コースが変わったかフラグ
    df["is_course_changed"] = (df["course_taken"] != df["lane"]).astype(int)

    # B-05〜B-07: 実際の進入コースに対応するコース別成績を取得
    # DBにはcourse1_win_rate〜course6_win_rateが保存されている
    def get_course_win_rate(row):
        c = int(row["course_taken"]) if pd.notna(row["course_taken"]) else int(row["lane"])
        col = f"course{c}_win_rate"
        return row.get(col, np.nan)

    def get_course_2place_rate(row):
        c = int(row["course_taken"]) if pd.notna(row["course_taken"]) else int(row["lane"])
        col = f"course{c}_2place_rate"
        return row.get(col, np.nan)

    def get_course_avg_st(row):
        c = int(row["course_taken"]) if pd.notna(row["course_taken"]) else int(row["lane"])
        col = f"course{c}_avg_st"
        return row.get(col, np.nan)

    df["course_win_rate"]   = df.apply(get_course_win_rate, axis=1)
    df["course_2place_rate"] = df.apply(get_course_2place_rate, axis=1)
    df["course_avg_st"]     = df.apply(get_course_avg_st, axis=1)

    return df


# =============================================================
# カテゴリC：コンディション（4特徴量）
# =============================================================

def calc_condition_features(df):
    """
    C-01 consecutive_race_days（当節連続出走日数）
    C-02 days_since_last_race（前節からの休養日数）
    C-03 recent_5_avg_finish（直近5走の平均着順）
    C-04 recent_5_trend（直近フォームのトレンド）

    DBの既存カラムをそのまま使う。
    """
    return df


# =============================================================
# カテゴリD：モーター・ボート・展示（12特徴量）
# =============================================================

def calc_equipment_features(df):
    """
    D-01 motor_win_rate（モーター勝率）
    D-02 motor_2place_rate（モーター2着率）
    D-03 motor_maintenance_count（今節整備回数）
    D-04 boat_2place_rate（ボート2着率）
    D-05 exhibition_time（展示タイム）
    D-06 exhibition_time_rank（展示タイム順位 6艇中）
    D-07 exhibition_time_vs_field（展示タイム - フィールド平均）
    D-08 exhibition_st（展示スタートタイミング）
    D-09 exhibition_dashi（出足）
    D-10 exhibition_yukiashi（行き足）
    D-11 exhibition_nobiashi（伸び足）
    D-12 exhibition_mawariashi（まわり足）
    """
    # D-06: 展示タイム順位（速い方が小さい値 → 昇順でランク）
    df["exhibition_time_rank"] = (
        df.groupby("race_id")["exhibition_time"]
        .rank(method="min", ascending=True)  # タイムは小さいほど速い
    )

    # D-07: 展示タイム - フィールド平均
    df["race_avg_exhibition"] = df.groupby("race_id")["exhibition_time"].transform("mean")
    df["exhibition_time_vs_field"] = df["exhibition_time"] - df["race_avg_exhibition"]
    df.drop(columns=["race_avg_exhibition"], inplace=True)

    return df


# =============================================================
# カテゴリE：環境（12特徴量）
# =============================================================

def calc_environment_features(df):
    """
    E-01 wind_speed（風速）
    E-02 wind_direction_sin（風向きsin変換）
    E-03 wind_direction_cos（風向きcos変換）
    E-04 wave_height（波高）
    E-05 rain_amount（雨量）
    E-06 air_pressure（気圧）
    E-07 temperature（気温）
    E-08 water_type（海水=1/淡水=0）
    E-09 tide_level（潮位）
    E-10 month_sin（月のsin変換）
    E-11 month_cos（月のcos変換）
    E-12 hour_of_race（レース時刻帯）

    注意：wind_directionは必ずsin/cos変換する
    （数値のままでは方向の循環性を表現できない）
    """
    # E-02/E-03: DBに既にsin/cosが保存されているが、未保存の場合に備えて計算
    if "wind_direction_sin" not in df.columns or df["wind_direction_sin"].isna().all():
        if "wind_direction" in df.columns:
            # 風向きを度数（0-360）として扱いsin/cosに変換
            wind_deg = pd.to_numeric(df["wind_direction"], errors="coerce")
            df["wind_direction_sin"] = np.sin(np.radians(wind_deg))
            df["wind_direction_cos"] = np.cos(np.radians(wind_deg))

    # E-10/E-11: 月のsin/cos変換（季節性の循環性）
    if "date" in df.columns:
        month = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # E-12: レース時刻帯（時間部分を数値化）
    if "race_time" in df.columns:
        df["hour_of_race"] = pd.to_datetime(
            df["race_time"], format="%H:%M", errors="coerce"
        ).dt.hour.fillna(-1).astype(int)

    return df


# =============================================================
# カテゴリF：会場（4特徴量）
# =============================================================

def calc_venue_features(df):
    """
    F-01 venue_id（会場ID）
    F-02 venue_course1_win_rate（会場の1コース歴史的勝率）
    F-03 venue_upset_rate（会場の荒れやすさ指数）
    F-04 venue_avg_trifecta_odds（会場の3連単平均配当）
    """
    # venue_idはracesテーブルから来ている
    # F-02〜F-04: DBのvenuesテーブルから取得済みの場合はそのまま使う
    # マスタデータでフォールバック
    if "venue_course1_win_rate" not in df.columns:
        df["venue_course1_win_rate"] = df["venue_id"].map(VENUE_COURSE1_WIN_RATE)

    return df


# =============================================================
# カテゴリG：レース構成（6特徴量）
# =============================================================

def calc_race_features(df):
    """
    G-01 race_no（レース番号 1〜12）
    G-02 grade_num（グレード SG=4/G1=3/G2=2/G3=1/一般=0）
    G-03 race_type_num（優勝戦=3/準優=2/予選=1）
    G-04 field_strength（フィールド強度 全選手の平均ランク）
    G-05 field_strength_std（フィールド強度のばらつき）
    G-06 upset_potential（荒れやすさ指数）
    """
    # G-02: グレード数値化
    df["grade_num"] = df["grade"].map(GRADE_MAP).fillna(0).astype(int)

    # G-03: レースタイプ数値化
    df["race_type_num"] = df["race_type"].map(RACETYPE_MAP).fillna(0).astype(int)

    # G-04: フィールド強度（racer_classの平均）
    df["field_strength"] = df.groupby("race_id")["racer_class"].transform("mean")

    # G-05: フィールド強度のばらつき
    df["field_strength_std"] = df.groupby("race_id")["racer_class"].transform("std")

    # G-06: 荒れやすさ指数（実力差が小さいほど荒れやすい）
    # std が小さい → 実力が拮抗 → 荒れやすい → 指数を高くする
    df["upset_potential"] = 1.0 / (df["field_strength_std"].fillna(1.0) + 0.1)

    return df


# =============================================================
# カテゴリH：オッズ時系列（16特徴量）
# =============================================================

ODDS_TIMINGS = ["120min", "60min", "30min", "15min", "5min", "1min", "final"]

def calc_odds_features(entries_df, odds_df):
    """
    H-01 win_odds_final（単勝オッズ確定値）
    H-02 win_odds_rank（単勝オッズ順位）
    H-03 implied_prob_final（市場確率 1/オッズ）
    H-04 odds_change_120to60（2時間前→1時間前の変化率）
    H-05 odds_change_60to30（1時間前→30分前の変化率）
    H-06 odds_change_30to15（30分前→15分前の変化率）
    H-07 odds_change_15to5（15分前→5分前の変化率）
    H-08 odds_change_5to1（5分前→1分前の変化率）
    H-09 odds_change_total（2時間前→確定の総変化率）
    H-10 odds_acceleration（変化率の変化速度）
    H-11 is_sharp_drop（5分間で10%以上下落フラグ）
    H-12 is_suspicious_move（他の艇と逆方向に動いているフラグ）
    H-13 is_late_surge（締め切り1分前からの急変フラグ）
    H-14 odds_rank_change（オッズ人気順位の変化）
    H-15 odds_stability（全7タイミングの変動係数）
    H-16 odds_120min_baseline（2時間前時点のベースライン値）
    """
    if odds_df is None or odds_df.empty:
        # オッズデータなし → NaN列を追加して返す
        odds_cols = [
            "win_odds_final", "win_odds_rank", "implied_prob_final",
            "odds_change_120to60", "odds_change_60to30", "odds_change_30to15",
            "odds_change_15to5", "odds_change_5to1", "odds_change_total",
            "odds_acceleration", "is_sharp_drop", "is_suspicious_move",
            "is_late_surge", "odds_rank_change", "odds_stability",
            "odds_120min_baseline",
        ]
        for col in odds_cols:
            entries_df[col] = np.nan
        return entries_df

    # 単勝オッズのみ抽出してピボット
    win_odds = odds_df[odds_df["odds_type"] == "単勝"].copy()
    # combinationが艇番（"1"〜"6"）
    win_odds["lane"] = pd.to_numeric(win_odds["combination"], errors="coerce").astype("Int64")

    # タイミング別にピボット
    pivot = win_odds.pivot_table(
        index=["race_id", "lane"],
        columns="timing",
        values="odds_value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # 存在しないタイミングはNaN
    for t in ODDS_TIMINGS:
        if t not in pivot.columns:
            pivot[t] = np.nan

    # H-01: 確定単勝オッズ
    pivot["win_odds_final"] = pivot["final"]

    # H-02: 単勝オッズ順位（安い方が1位）
    pivot["win_odds_rank"] = pivot.groupby("race_id")["final"].rank(
        method="min", ascending=True
    )

    # H-03: 市場確率
    pivot["implied_prob_final"] = 1.0 / pivot["final"].replace(0, np.nan)

    # H-04〜H-08: タイミング間の変化率
    pairs = [
        ("odds_change_120to60", "120min", "60min"),
        ("odds_change_60to30",  "60min",  "30min"),
        ("odds_change_30to15",  "30min",  "15min"),
        ("odds_change_15to5",   "15min",  "5min"),
        ("odds_change_5to1",    "5min",   "1min"),
    ]
    for col, t1, t2 in pairs:
        pivot[col] = (pivot[t2] - pivot[t1]) / pivot[t1].replace(0, np.nan)

    # H-09: 総変化率（2時間前→確定）
    pivot["odds_change_total"] = (
        pivot["final"] - pivot["120min"]
    ) / pivot["120min"].replace(0, np.nan)

    # H-10: 変化率の加速度（直近変化率 - 前変化率）
    pivot["odds_acceleration"] = pivot["odds_change_5to1"] - pivot["odds_change_15to5"]

    # H-11: 急落フラグ（5分間で10%以上下落 → オッズ低下＝人気上昇）
    pivot["is_sharp_drop"] = (pivot["odds_change_5to1"] <= -0.10).astype(int)

    # H-12: 他の艇と逆方向に動いているフラグ
    #   レース内で平均と逆符号かつ絶対値が大きい
    pivot["race_avg_change"] = pivot.groupby("race_id")["odds_change_5to1"].transform("mean")
    pivot["is_suspicious_move"] = (
        (pivot["odds_change_5to1"] * pivot["race_avg_change"] < 0)
        & (pivot["odds_change_5to1"].abs() > 0.10)
    ).astype(int)
    pivot.drop(columns=["race_avg_change"], inplace=True)

    # H-13: 直前急変フラグ（1分前→確定で5%以上変化）
    pivot["is_late_surge"] = (
        ((pivot["final"] - pivot["1min"]) / pivot["1min"].replace(0, np.nan)).abs() >= 0.05
    ).astype(int)

    # H-14: オッズ人気順位の変化（120min→final）
    pivot["rank_120min"] = pivot.groupby("race_id")["120min"].rank(method="min")
    pivot["rank_final"]  = pivot["win_odds_rank"]
    pivot["odds_rank_change"] = pivot["rank_final"] - pivot["rank_120min"]
    pivot.drop(columns=["rank_120min", "rank_final"], inplace=True)

    # H-15: 変動係数（全7タイミングの標準偏差/平均）
    timing_cols_exist = [t for t in ODDS_TIMINGS if t in pivot.columns]
    pivot["odds_stability"] = (
        pivot[timing_cols_exist].std(axis=1) / pivot[timing_cols_exist].mean(axis=1)
    )

    # H-16: 2時間前ベースライン
    pivot["odds_120min_baseline"] = pivot["120min"]

    # 不要な元タイミング列を削除（race_id・laneは残す）
    drop_cols = [t for t in ODDS_TIMINGS if t in pivot.columns]
    pivot.drop(columns=drop_cols, inplace=True)

    # entries_dfにマージ
    entries_df = entries_df.merge(pivot, on=["race_id", "lane"], how="left")
    return entries_df


# =============================================================
# カテゴリI：対戦履歴（1特徴量）
# =============================================================

def calc_h2h_features(df):
    """
    I-01 h2h_win_rate_vs_field（同レース選手への過去勝率平均）
    """
    # head_to_headテーブルのデータが必要だが、
    # 現状DBに十分なデータが蓄積されていない段階では
    # NaNのまま（欠損値補完でレース内平均になる）
    if "h2h_win_rate_vs_field" not in df.columns:
        df["h2h_win_rate_vs_field"] = np.nan
    return df


# =============================================================
# カテゴリJ：レース内相対指標（7特徴量）
# =============================================================

def calc_relative_features(df):
    """
    J-01 class_rank_in_race（レース内ランク順位）
    J-02 win_rate_rank（レース内勝率順位）
    J-03 course_win_rate_rank（レース内コース別勝率順位）
    J-04 motor_rank（レース内モーターランク）
    J-05 exhibition_rank（レース内展示タイム順位）
    J-06 weight_diff（体重 - フィールド平均）
    J-07 motor_vs_field（モーター率 - フィールド平均）
    """
    # J-01: 級別のレース内ランク（高いほど上位）
    df["class_rank_in_race"] = df.groupby("race_id")["racer_class"].rank(
        method="min", ascending=False
    )

    # J-02: 全国勝率のレース内順位（高いほど上位）
    df["win_rate_rank"] = df.groupby("race_id")["national_win_rate"].rank(
        method="min", ascending=False
    )

    # J-03: コース別勝率のレース内順位
    df["course_win_rate_rank"] = df.groupby("race_id")["course_win_rate"].rank(
        method="min", ascending=False
    )

    # J-04: モーター勝率のレース内順位
    df["motor_rank"] = df.groupby("race_id")["motor_win_rate"].rank(
        method="min", ascending=False
    )

    # J-05: 展示タイム順位（早い方が上位 → 降順で小さい方が1位）
    df["exhibition_rank"] = df.groupby("race_id")["exhibition_time"].rank(
        method="min", ascending=True
    )

    # J-06: 体重偏差
    df["race_avg_weight"] = df.groupby("race_id")["racer_weight"].transform("mean")
    df["weight_diff"] = df["racer_weight"] - df["race_avg_weight"]
    df.drop(columns=["race_avg_weight"], inplace=True)

    # J-07: モーター勝率偏差
    df["race_avg_motor"] = df.groupby("race_id")["motor_win_rate"].transform("mean")
    df["motor_vs_field"] = df["motor_win_rate"] - df["race_avg_motor"]
    df.drop(columns=["race_avg_motor"], inplace=True)

    return df


# =============================================================
# 特徴量重要度の保存
# =============================================================

def save_feature_importance(model, feature_names, save_path):
    """
    学習後に特徴量重要度を自動出力する。
    削除はしない・記録として残す。

    Parameters:
        model        : 学習済みLightGBMモデル
        feature_names: 特徴量名のリスト
        save_path    : 保存先パス（CSV）

    出力形式：
        順位, 特徴量名, 重要度(gain), 重要度(split)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gain   = model.feature_importance(importance_type="gain")
    split  = model.feature_importance(importance_type="split")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain":  gain,
        "importance_split": split,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    importance_df.insert(0, "rank", range(1, len(importance_df) + 1))

    # 重要度下位10%に警告（削除はしない）
    threshold = len(importance_df) * 0.9
    low_importance = importance_df[importance_df["rank"] > threshold]
    if not low_importance.empty:
        print(f"[警告] 重要度下位10%の特徴量（削除しない・記録のみ）:")
        for _, row in low_importance.iterrows():
            print(f"  {row['rank']:3d}. {row['feature']:<40} gain={row['importance_gain']:.4f}")

    importance_df.to_csv(save_path, index=False)
    print(f"特徴量重要度を保存: {save_path}")
    return importance_df


# =============================================================
# 特徴量名一覧（74特徴量）
# =============================================================

FEATURE_NAMES_74 = [
    # A: 選手基本
    "racer_class", "national_win_rate", "national_2place_rate",
    "national_3place_rate", "local_win_rate", "local_2place_rate",
    "is_home", "local_vs_national", "racer_weight", "racer_age",
    # B: コース・スタート
    "course_taken", "lane", "course_vs_lane", "is_course_changed",
    "course_win_rate", "course_2place_rate", "course_avg_st",
    "avg_start_timing", "flying_count", "late_count",
    "days_since_last_flying", "is_flying_return",
    # C: コンディション
    "consecutive_race_days", "days_since_last_race",
    "recent_5_avg_finish", "recent_5_trend",
    # D: モーター・展示
    "motor_win_rate", "motor_2place_rate", "motor_maintenance_count",
    "boat_2place_rate", "exhibition_time", "exhibition_time_rank",
    "exhibition_time_vs_field", "exhibition_st",
    "exhibition_dashi", "exhibition_yukiashi",
    "exhibition_nobiashi", "exhibition_mawariashi",
    # E: 環境
    "wind_speed", "wind_direction_sin", "wind_direction_cos",
    "wave_height", "rain_amount", "air_pressure", "temperature",
    "water_type", "tide_level", "month_sin", "month_cos", "hour_of_race",
    # F: 会場
    "venue_id", "venue_course1_win_rate",
    "venue_upset_rate", "venue_avg_trifecta_odds",
    # G: レース構成
    "race_no", "grade_num", "race_type_num",
    "field_strength", "field_strength_std", "upset_potential",
    # H: オッズ時系列
    "win_odds_final", "win_odds_rank", "implied_prob_final",
    "odds_change_120to60", "odds_change_60to30", "odds_change_30to15",
    "odds_change_15to5", "odds_change_5to1", "odds_change_total",
    "odds_acceleration", "is_sharp_drop", "is_suspicious_move",
    "is_late_surge", "odds_rank_change", "odds_stability",
    "odds_120min_baseline",
    # I: 対戦履歴
    "h2h_win_rate_vs_field",
    # J: レース内相対指標
    "class_rank_in_race", "win_rate_rank", "course_win_rate_rank",
    "motor_rank", "exhibition_rank", "weight_diff", "motor_vs_field",
]

assert len(FEATURE_NAMES_74) == 84, f"特徴量数が{len(FEATURE_NAMES_74)}です（84であるべき）"
