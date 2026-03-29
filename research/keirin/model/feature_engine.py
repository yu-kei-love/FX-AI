# ===========================================
# model/feature_engine.py
# 競輪AI - 特徴量エンジン（50特徴量）
#
# カテゴリ構成：
#   A: 選手個人（12）
#   B: ライン（8）  ← 最重要
#   C: バンク×脚質（6）
#   D: 展開予測（4）← 既存AIにない特徴
#   E: 風・天候（5）
#   F: オッズ（6）
#   G: レース構成（5）
#   H: レース内相対（4）
#   合計：50特徴量
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# bank_master.py を参照
_BANK_PATH = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(_BANK_PATH))
from bank_master import get_bank_info, get_style_advantage

from data_interface import DataInterface

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

# グレード数値化マップ
GRADE_MAP = {"GP": 6, "G1": 5, "G2": 4, "G3": 3, "F1": 2, "F2": 1, None: 0}

# レースタイプ数値化マップ
RACE_TYPE_MAP = {"決勝": 3, "準決": 2, "予選": 1, None: 0}

# 脚質数値化マップ
STYLE_MAP = {"逃げ": 2, "追込": 1, "両": 0, None: 0}


# =============================================================
# データ読み込み
# =============================================================

def load_race_data(start_date=None, end_date=None,
                   data_interface=None, db_path=None):
    """
    データを読み込む（DataInterface経由）。

    Parameters:
        start_date     : 開始日 "YYYYMMDD"
        end_date       : 終了日 "YYYYMMDD"
        data_interface : DataInterfaceインスタンス（省略でsqlite）
        db_path        : 後方互換性のため

    Returns:
        races_df, entries_df, odds_df, lines_df
    """
    if data_interface is None:
        if db_path:
            data_interface = DataInterface(mode="sqlite", db_path=str(db_path))
        else:
            data_interface = DataInterface(mode="sqlite", db_path=str(DB_PATH))

    races_df   = data_interface.get_races(start_date=start_date, end_date=end_date)
    entries_df = data_interface.get_entries(start_date=start_date, end_date=end_date)
    odds_df    = data_interface.get_odds()
    lines_df   = data_interface.get_lines(start_date=start_date, end_date=end_date)

    return races_df, entries_df, odds_df, lines_df


# =============================================================
# メイン特徴量生成
# =============================================================

def create_features(entries_df, races_df, odds_df,
                    line_probs=None, bank_info=None):
    """
    全50特徴量を計算して返す。

    Parameters:
        entries_df : 出走情報DataFrame
        races_df   : レース情報DataFrame
        odds_df    : オッズDataFrame
        line_probs : ライン予測確率 {"3-7-4": 0.85, ...}（省略可）
        bank_info  : バンク情報dict（省略時はvenue_nameから自動取得）

    Returns:
        features_df: 1行=1選手の特徴量DataFrame
    """
    if entries_df is None or len(entries_df) == 0:
        return pd.DataFrame()

    # レース情報をエントリーにマージ
    if races_df is not None and len(races_df) > 0:
        df = entries_df.merge(races_df, on="race_id", how="left")
    else:
        df = entries_df.copy()

    # バンク情報を取得（会場名から自動）
    if bank_info is None and "venue_name" in df.columns:
        venue_name = df["venue_name"].iloc[0] if len(df) > 0 else None
        bank_info = get_bank_info(venue_name) if venue_name else {}
    bank_info = bank_info or {}

    # line_probsが未指定の場合は空dict
    line_probs = line_probs or {}

    feature_frames = []

    # カテゴリ別に計算
    feat_a = calc_racer_features(df)
    feat_b = calc_line_features(df, line_probs)
    feat_c = calc_bank_features(df, bank_info)
    feat_d = calc_deployment_features(df, line_probs, bank_info)
    feat_e = calc_weather_features(df, bank_info)
    feat_f = calc_odds_features(odds_df, df)
    feat_g = calc_race_features(df)
    feat_h = calc_relative_features(df)

    feature_frames = [feat_a, feat_b, feat_c, feat_d,
                      feat_e, feat_f, feat_g, feat_h]

    # 全てのカテゴリを横結合
    result = df[["race_id", "car_no"]].copy()
    for feat in feature_frames:
        if feat is not None and len(feat) > 0:
            result = result.merge(feat, on=["race_id", "car_no"], how="left")

    return result


# =============================================================
# カテゴリA：選手個人（12特徴量）
# =============================================================

def calc_racer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A-01 racer_class（SS=5/S1=4/S2=3/A1=2/A2=1/A3=0）
    A-02 grade_score（競走得点）
    A-03 win_rate（勝率）
    A-04 second_rate（2着率）
    A-05 third_rate（3着率）
    A-06 style_num（脚質 逃げ=2/追込=1/両=0）
    A-07 gear_ratio（ギア倍数）← 競輪固有・重要
    A-08 back_count（B：バック回数）← 最重要特徴量
    A-09 home_count（H：ホーム回数）
    A-10 start_count（S：スタート回数）
    A-11 age（年齢）
    A-12 term（期別）
    """
    CLASS_MAP = {"SS": 5, "S1": 4, "S2": 3, "A1": 2, "A2": 1, "A3": 0}

    out = df[["race_id", "car_no"]].copy()
    out["A01_racer_class"]  = df["racer_class"].map(CLASS_MAP).fillna(0).astype(int)
    out["A02_grade_score"]  = df["grade_score"].fillna(df["grade_score"].mean())
    out["A03_win_rate"]     = df["win_rate"].fillna(0.0)
    out["A04_second_rate"]  = df["second_rate"].fillna(0.0)
    out["A05_third_rate"]   = df["third_rate"].fillna(0.0)
    out["A06_style_num"]    = df["style"].map(STYLE_MAP).fillna(0).astype(int)
    out["A07_gear_ratio"]   = df["gear_ratio"].fillna(3.6)   # 未検証：平均値3.6を仮置き
    out["A08_back_count"]   = df["back_count"].fillna(0).astype(int)   # 最重要
    out["A09_home_count"]   = df["home_count"].fillna(0).astype(int)
    out["A10_start_count"]  = df["start_count"].fillna(0).astype(int)
    out["A11_age"]          = df["age"].fillna(30).astype(int)
    out["A12_term"]         = df["term"].fillna(80).astype(int)

    return out


# =============================================================
# カテゴリB：ライン（8特徴量）← 最重要
# =============================================================

def calc_line_features(df: pd.DataFrame,
                        line_probs: dict) -> pd.DataFrame:
    """
    B-01 line_position_num（先頭=3/番手=2/3番手=1/単騎=0）← 最重要
    B-02 line_size（ライン人数 3/2/1）
    B-03 line_confidence（ラインの信頼度 0〜1）
    B-04 line_grade_score（ライン全員の競走得点平均）
    B-05 line_back_sum（ライン内のバック回数合計）← 核心
    B-06 betrayal_risk（裏切りリスク 0〜1：実データで計算）
    B-07 district_num（地区コード）
    B-08 is_home_district（地元地区フラグ 1/0）
    """
    DISTRICT_MAP = {
        "北日本": 1, "関東": 2, "南関東": 3,
        "中部": 4, "北信越": 5, "近畿": 6,
        "中国": 7, "四国": 8, "九州": 9,
    }
    POS_MAP = {"先頭": 3, "番手": 2, "3番手": 1, "単騎": 0}

    out = df[["race_id", "car_no"]].copy()

    # line_probsからcar_noごとのライン情報を逆引き
    car_line_info = _build_car_line_info(df, line_probs)

    out["B01_line_position_num"] = df["car_no"].map(
        lambda c: POS_MAP.get(car_line_info.get(c, {}).get("position", "単騎"), 0)
    )
    out["B02_line_size"]         = df["car_no"].map(
        lambda c: car_line_info.get(c, {}).get("line_size", 1)
    )
    out["B03_line_confidence"]   = df["car_no"].map(
        lambda c: car_line_info.get(c, {}).get("confidence", 0.30)
    )

    # grade_score と back_count はエントリーデータから
    grade_score = dict(zip(df["car_no"].astype(int), df["grade_score"].fillna(70)))
    back_count  = dict(zip(df["car_no"].astype(int), df["back_count"].fillna(0)))

    def _line_grade_avg(car_no):
        line_cars = car_line_info.get(car_no, {}).get("line_cars", [car_no])
        scores = [grade_score.get(c, 70) for c in line_cars]
        return float(np.mean(scores)) if scores else 70.0

    def _line_back_sum(car_no):
        line_cars = car_line_info.get(car_no, {}).get("line_cars", [car_no])
        return sum(back_count.get(c, 0) for c in line_cars)

    out["B04_line_grade_score"] = df["car_no"].astype(int).map(_line_grade_avg)
    out["B05_line_back_sum"]    = df["car_no"].astype(int).map(_line_back_sum)

    # 裏切りリスクは実データから計算。現時点では0固定（未検証）
    out["B06_betrayal_risk"]    = 0.0  # 未検証：実データ取得後に計算

    out["B07_district_num"]     = df["district"].map(DISTRICT_MAP).fillna(0).astype(int)

    # 地元フラグ（会場の都道府県と選手の都道府県が一致）
    # 簡略化：地元フラグはデータ取得後に実装
    out["B08_is_home_district"] = 0  # 未検証：スクレイピングデータ取得後に実装

    return out


def _build_car_line_info(df: pd.DataFrame,
                          line_probs: dict) -> dict:
    """
    line_probsからcar_noごとのライン情報を構築する。

    Returns:
        {car_no: {"position": str, "line_size": int,
                  "confidence": float, "line_cars": list}}
    """
    car_info = {}
    for line_str, conf in line_probs.items():
        cars = [int(c) for c in line_str.split("-") if c.isdigit()]
        positions = ["先頭", "番手", "3番手"]
        for i, car_no in enumerate(cars):
            pos = positions[i] if i < len(positions) else "3番手"
            if len(cars) == 1:
                pos = "単騎"
            car_info[car_no] = {
                "position":   pos,
                "line_size":  len(cars),
                "confidence": conf,
                "line_cars":  cars,
            }
    return car_info


# =============================================================
# カテゴリC：バンク×脚質（6特徴量）
# =============================================================

def calc_bank_features(df: pd.DataFrame,
                        bank_info: dict) -> pd.DataFrame:
    """
    C-01 bank_length（バンク周長 333/400/500）
    C-02 bank_straight（みなし直線距離m）
    C-03 bank_cant（カント角度）
    C-04 style_advantage_escape（逃げ有利度）
    C-05 style_advantage_makuri（捲り有利度）
    C-06 style_advantage_sashi（差し有利度）

    style_advantage_*はbank_master.pyのget_style_advantage()を使う。
    """
    out = df[["race_id", "car_no"]].copy()

    if bank_info:
        out["C01_bank_length"]   = bank_info.get("length", 400)
        out["C02_bank_straight"] = bank_info.get("straight", 54.0)
        out["C03_bank_cant"]     = bank_info.get("cant", 31.0)

        # スタイル有利度はレース内で全選手共通
        # 風情報があれば使う（なければ無風として計算）
        wind_speed = 0.0
        wind_direction = 0.0
        if "wind_speed" in df.columns and len(df) > 0:
            wind_speed = float(df["wind_speed"].iloc[0])
        venue_name = None
        if "venue_name" in df.columns and len(df) > 0:
            venue_name = df["venue_name"].iloc[0]

        if venue_name:
            advantage = get_style_advantage(venue_name, wind_speed, wind_direction)
        else:
            advantage = {"escape": 0.50, "makuri": 0.50, "sashi": 0.50}

        out["C04_style_advantage_escape"] = advantage["escape"]
        out["C05_style_advantage_makuri"] = advantage["makuri"]
        out["C06_style_advantage_sashi"]  = advantage["sashi"]
    else:
        out["C01_bank_length"]            = 400
        out["C02_bank_straight"]          = 54.0
        out["C03_bank_cant"]              = 31.0
        out["C04_style_advantage_escape"] = 0.50
        out["C05_style_advantage_makuri"] = 0.50
        out["C06_style_advantage_sashi"]  = 0.50

    return out


# =============================================================
# カテゴリD：展開予測（4特徴量）← 既存AIにない特徴
# =============================================================

def calc_deployment_features(df: pd.DataFrame,
                               line_probs: dict,
                               bank_info: dict) -> pd.DataFrame:
    """
    D-01 escape_deployment_score（逃げ展開スコア）
    D-02 makuri_deployment_score（捲り展開スコア）
    D-03 sashi_deployment_score（差し展開スコア）
    D-04 chaos_risk（荒れリスク）

    計算ロジック：
    ライン先頭の脚質 × バンク特性 × 風で計算する。
    例：逃げ型先頭×333mバンク×バック追い風 → escape_deployment_scoreが高い
    """
    out = df[["race_id", "car_no"]].copy()

    bank_length   = bank_info.get("length", 400) if bank_info else 400
    is_back_wind  = False
    wind_speed    = 0.0
    if "wind_speed" in df.columns and len(df) > 0:
        wind_speed = float(df["wind_speed"].iloc[0])
    is_rain = False
    if "is_rain" in df.columns and len(df) > 0:
        is_rain = bool(df["is_rain"].iloc[0])

    # バンク特性ベーススコア
    if bank_length <= 335:
        bank_escape_factor = 1.2
        bank_makuri_factor = 0.8
    elif bank_length == 400:
        bank_escape_factor = 1.0
        bank_makuri_factor = 1.0
    else:  # 500m
        bank_escape_factor = 0.7
        bank_makuri_factor = 1.3

    # 先頭選手の脚質を特定
    car_line_info = _build_car_line_info(df, line_probs)
    style_map = dict(zip(df["car_no"].astype(int), df["style"].fillna("追込")))

    def _escape_score(car_no):
        """逃げ展開スコア：ライン先頭が逃げ型の場合に高くなる"""
        info = car_line_info.get(car_no, {})
        if info.get("position") == "先頭":
            style = style_map.get(car_no, "追込")
            base = 0.8 if style == "逃げ" else 0.4
            return min(1.0, base * bank_escape_factor)
        return 0.3

    def _makuri_score(car_no):
        """捲り展開スコア：バンク特性と風で変わる"""
        base = 0.5
        if wind_speed >= 4.0:
            base += 0.15
        return min(1.0, base * bank_makuri_factor)

    def _sashi_score(car_no):
        """差し展開スコア：番手選手に有利"""
        info = car_line_info.get(car_no, {})
        if info.get("position") in ("番手", "3番手"):
            return 0.6
        return 0.3

    out["D01_escape_deployment_score"] = df["car_no"].astype(int).map(_escape_score)
    out["D02_makuri_deployment_score"] = df["car_no"].astype(int).map(_makuri_score)
    out["D03_sashi_deployment_score"]  = df["car_no"].astype(int).map(_sashi_score)

    # 荒れリスク：単騎が多いほど上昇
    n_singles = sum(1 for k in line_probs if "-" not in k)
    chaos_base = min(1.0, n_singles * 0.15)
    if is_rain:
        chaos_base = min(1.0, chaos_base + 0.10)
    if wind_speed >= 4.0:
        chaos_base = min(1.0, chaos_base + 0.10)
    out["D04_chaos_risk"] = round(chaos_base, 3)

    return out


# =============================================================
# カテゴリE：風・天候（5特徴量）
# =============================================================

def calc_weather_features(df: pd.DataFrame,
                            bank_info: dict) -> pd.DataFrame:
    """
    E-01 wind_speed（屋内は0固定）
    E-02 wind_direction_sin（屋内は0固定）
    E-03 wind_direction_cos（屋内は1固定）
    E-04 is_back_headwind（バック向かい風フラグ）
         → 風速3m/s以上かつバック向かい風のとき True
         → 屋内会場は常に False
    E-05 is_rain（雨フラグ）
    """
    is_dome = bank_info.get("is_dome", False) if bank_info else False

    out = df[["race_id", "car_no"]].copy()

    if is_dome:
        # 屋内会場：風を0固定
        out["E01_wind_speed"]          = 0.0
        out["E02_wind_direction_sin"]  = 0.0
        out["E03_wind_direction_cos"]  = 1.0
        out["E04_is_back_headwind"]    = 0
    else:
        out["E01_wind_speed"] = (
            df["wind_speed"].fillna(0.0) if "wind_speed" in df.columns else 0.0
        )
        out["E02_wind_direction_sin"] = (
            df["wind_direction_sin"].fillna(0.0)
            if "wind_direction_sin" in df.columns else 0.0
        )
        out["E03_wind_direction_cos"] = (
            df["wind_direction_cos"].fillna(1.0)
            if "wind_direction_cos" in df.columns else 1.0
        )

        # バック向かい風判定（sin < -0.5 かつ wind_speed >= 3m/s）
        # sin(-90度)=-1 は南から北方向の風（バック向かい風を簡略化）
        wind_speed = (df["wind_speed"].fillna(0.0) if "wind_speed" in df.columns
                      else pd.Series(0.0, index=df.index))
        wind_sin   = (df["wind_direction_sin"].fillna(0.0)
                      if "wind_direction_sin" in df.columns
                      else pd.Series(0.0, index=df.index))
        out["E04_is_back_headwind"] = (
            ((wind_speed >= 3.0) & (wind_sin < -0.5)).astype(int)
        )

    out["E05_is_rain"] = (
        df["is_rain"].fillna(0).astype(int) if "is_rain" in df.columns else 0
    )

    return out


# =============================================================
# カテゴリF：オッズ（6特徴量）
# =============================================================

def calc_odds_features(odds_df: pd.DataFrame,
                        entries_df: pd.DataFrame) -> pd.DataFrame:
    """
    F-01 win_odds_final（単勝オッズ確定値）
    F-02 win_odds_rank（単勝オッズ順位）
    F-03 implied_prob_final（市場確率 1/オッズ）
    F-04 odds_change_total（直前からの変化率：15min→final）
    F-05 is_sharp_move（急変フラグ 10%以上変化したらTrue）
    F-06 odds_stability（全タイミングの変動係数）
    """
    out = entries_df[["race_id", "car_no"]].copy()

    if odds_df is None or len(odds_df) == 0:
        out["F01_win_odds_final"]    = np.nan
        out["F02_win_odds_rank"]     = np.nan
        out["F03_implied_prob"]      = np.nan
        out["F04_odds_change_total"] = 0.0
        out["F05_is_sharp_move"]     = 0
        out["F06_odds_stability"]    = np.nan
        return out

    # finalオッズ
    final_odds = odds_df[odds_df["timing"] == "final"][["race_id", "car_no", "win_odds"]]
    final_odds = final_odds.rename(columns={"win_odds": "F01_win_odds_final"})
    out = out.merge(final_odds, on=["race_id", "car_no"], how="left")
    out["F03_implied_prob"] = (1.0 / out["F01_win_odds_final"]).replace([np.inf], np.nan)

    # レース内オッズ順位（小さいほど人気）
    out["F02_win_odds_rank"] = (
        out.groupby("race_id")["F01_win_odds_final"].rank(ascending=True)
    )

    # オッズ変化率（15min→final）
    odds_15min = odds_df[odds_df["timing"] == "15min"][["race_id", "car_no", "win_odds"]]
    odds_15min = odds_15min.rename(columns={"win_odds": "odds_15min"})
    out = out.merge(odds_15min, on=["race_id", "car_no"], how="left")
    out["F04_odds_change_total"] = (
        (out["F01_win_odds_final"] - out["odds_15min"]) / out["odds_15min"].replace(0, np.nan)
    ).fillna(0.0)
    out["F05_is_sharp_move"] = (out["F04_odds_change_total"].abs() >= 0.10).astype(int)
    out = out.drop(columns=["odds_15min"])

    # オッズ変動係数（全タイミング）
    TIMINGS = ["120min", "60min", "30min", "15min", "5min", "1min", "final"]
    all_timing_odds = odds_df[odds_df["timing"].isin(TIMINGS)].copy()
    stability = (
        all_timing_odds.groupby(["race_id", "car_no"])["win_odds"]
        .agg(lambda x: float(x.std() / x.mean()) if float(x.mean()) > 0 else 0.0)
        .reset_index()
        .rename(columns={"win_odds": "F06_odds_stability"})
    )
    out = out.merge(stability, on=["race_id", "car_no"], how="left")

    return out


# =============================================================
# カテゴリG：レース構成（5特徴量）
# =============================================================

def calc_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G-01 race_no（レース番号）
    G-02 grade_num（G1/G2/G3/F1/F2 を数値化）
    G-03 race_type_num（予選/準決/決勝 を数値化）
    G-04 field_strength（競走得点の平均）
    G-05 n_single（単騎選手の数）
    """
    out = df[["race_id", "car_no"]].copy()

    out["G01_race_no"]      = df["race_no"].fillna(1).astype(int) if "race_no" in df.columns else 1
    out["G02_grade_num"]    = df["grade"].map(GRADE_MAP).fillna(0).astype(int) if "grade" in df.columns else 0
    out["G03_race_type_num"]= df["race_type"].map(RACE_TYPE_MAP).fillna(0).astype(int) if "race_type" in df.columns else 0

    # 競走得点平均（レース内）
    if "grade_score" in df.columns:
        field_avg = df.groupby("race_id")["grade_score"].transform("mean")
        out["G04_field_strength"] = field_avg.fillna(70.0)
    else:
        out["G04_field_strength"] = 70.0

    # 単騎人数（ライン情報があれば計算。ここでは0固定で後で補完）
    out["G05_n_single"] = 0  # 未検証：create_features側からline_probsを受け取って計算

    return out


# =============================================================
# カテゴリH：レース内相対（4特徴量）
# =============================================================

def calc_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    H-01 grade_score_rank（レース内得点順位）
    H-02 back_count_rank（レース内バック回数順位）
    H-03 grade_score_vs_field（得点 - フィールド平均）
    H-04 is_home（地元フラグ）
    """
    out = df[["race_id", "car_no"]].copy()

    if "grade_score" in df.columns:
        out["H01_grade_score_rank"] = (
            df.groupby("race_id")["grade_score"].rank(ascending=False)
        )
        field_avg = df.groupby("race_id")["grade_score"].transform("mean")
        out["H03_grade_score_vs_field"] = df["grade_score"] - field_avg
    else:
        out["H01_grade_score_rank"]     = np.nan
        out["H03_grade_score_vs_field"] = 0.0

    if "back_count" in df.columns:
        out["H02_back_count_rank"] = (
            df.groupby("race_id")["back_count"].rank(ascending=False)
        )
    else:
        out["H02_back_count_rank"] = np.nan

    # 地元フラグ（簡略化：実データ取得後に会場都道府県と選手都道府県を照合）
    out["H04_is_home"] = 0  # 未検証：スクレイピングデータ取得後に実装

    return out


# =============================================================
# 特徴量名一覧（50特徴量）
# =============================================================

FEATURE_NAMES = [
    # A: 選手個人（12）
    "A01_racer_class", "A02_grade_score", "A03_win_rate",
    "A04_second_rate", "A05_third_rate", "A06_style_num",
    "A07_gear_ratio", "A08_back_count", "A09_home_count",
    "A10_start_count", "A11_age", "A12_term",
    # B: ライン（8）← 最重要
    "B01_line_position_num", "B02_line_size", "B03_line_confidence",
    "B04_line_grade_score", "B05_line_back_sum", "B06_betrayal_risk",
    "B07_district_num", "B08_is_home_district",
    # C: バンク×脚質（6）
    "C01_bank_length", "C02_bank_straight", "C03_bank_cant",
    "C04_style_advantage_escape", "C05_style_advantage_makuri",
    "C06_style_advantage_sashi",
    # D: 展開予測（4）
    "D01_escape_deployment_score", "D02_makuri_deployment_score",
    "D03_sashi_deployment_score", "D04_chaos_risk",
    # E: 風・天候（5）
    "E01_wind_speed", "E02_wind_direction_sin", "E03_wind_direction_cos",
    "E04_is_back_headwind", "E05_is_rain",
    # F: オッズ（6）
    "F01_win_odds_final", "F02_win_odds_rank", "F03_implied_prob",
    "F04_odds_change_total", "F05_is_sharp_move", "F06_odds_stability",
    # G: レース構成（5）
    "G01_race_no", "G02_grade_num", "G03_race_type_num",
    "G04_field_strength", "G05_n_single",
    # H: レース内相対（4）
    "H01_grade_score_rank", "H02_back_count_rank",
    "H03_grade_score_vs_field", "H04_is_home",
]

assert len(FEATURE_NAMES) == 50, f"特徴量数が{len(FEATURE_NAMES)}です（50であるべき）"
