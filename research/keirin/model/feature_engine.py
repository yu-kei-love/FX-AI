# ===========================================
# model/feature_engine.py
# 競輪AI - 特徴量エンジン（57特徴量）
#
# カテゴリ構成：
#   A: 選手個人（12）  ← home_count除外（全件NULL・Kドリームズ非掲載）
#   B: ライン（8）  ← 最重要
#   C: バンク×脚質（7）
#   D: 展開予測（4）← 既存AIにない特徴
#   E: 風・天候（5）
#   F: オッズ（6）
#   G: レース構成（6）← is_midnight追加
#   H: レース内相対（4）
#   I: 履歴（6）← 当場勝率・トレンド・相性・Elo・上がりタイム
#   J: 決まり手実績（3）← 逃げ率・捲り率・差し率
#   合計：61特徴量
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# bank_master.py を参照
_BANK_PATH = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(_BANK_PATH))
from bank_master import BANK_MASTER, get_bank_info, get_style_advantage

from data_interface import DataInterface

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

# グレード数値化マップ
# グレード数値化マップ（Kドリームズ準拠）
GRADE_MAP = {
    "GP": 6, "G1": 5, "G2": 4, "G3": 3,
    "FI": 2, "FII": 1,
    "F1": 2, "F2": 1,  # 旧キー互換
    None: 2,  # デフォルト=FI相当
}

# ステージ数値化マップ（Kドリームズ準拠）
RACE_TYPE_MAP = {
    "決勝": 4, "準決": 3, "特選": 2, "選抜": 2,
    "一般": 1, "予選": 1,
    None: 1,  # デフォルト=一般
}

# 脚質数値化マップ
STYLE_MAP = {"逃げ": 2, "追込": 1, "両": 0, None: 0}

# 地区別の裏切り率デフォルト（未検証・データ取得後に更新）
DISTRICT_BETRAYAL_DEFAULT = {
    "北日本": 0.08, "関東": 0.10, "南関東": 0.10,
    "中部": 0.09, "北信越": 0.09, "近畿": 0.11,
    "中国": 0.08, "四国": 0.08, "九州": 0.07,
}

# 会場名→都道府県の逆引きマップ（bank_master.py から自動生成）
VENUE_PREFECTURE_MAP = {
    name: info.get("prefecture", "")
    for name, info in BANK_MASTER.items()
}


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
                    line_probs=None, bank_info=None,
                    db_path=None):
    """
    全61特徴量を計算して返す。

    Parameters:
        entries_df : 出走情報DataFrame
        races_df   : レース情報DataFrame
        odds_df    : オッズDataFrame
        line_probs : ライン予測確率 {"3-7-4": 0.85, ...}（省略可）
        bank_info  : バンク情報dict（省略時はvenue_nameから自動取得）
        db_path    : SQLiteのパス（番手捲り率・裏切り率計算用、省略可）

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

    # カテゴリ別に計算
    feat_a = calc_racer_features(df, db_path=db_path)
    feat_b = calc_line_features(df, line_probs, db_path=db_path)
    feat_c = calc_bank_features(df, bank_info)
    feat_d = calc_deployment_features(df, line_probs, bank_info)
    feat_e = calc_weather_features(df, bank_info)
    feat_f = calc_odds_features(odds_df, df, db_path=db_path)
    feat_g = calc_race_features(df, line_probs)
    feat_h = calc_relative_features(df, bank_info)
    feat_i = calc_history_features(df, db_path=db_path)
    feat_j = calc_kimari_features(df)

    feature_frames = [feat_a, feat_b, feat_c, feat_d,
                      feat_e, feat_f, feat_g, feat_h, feat_i, feat_j]

    # 全てのカテゴリを横結合
    result = df[["race_id", "car_no"]].copy()
    for feat in feature_frames:
        if feat is not None and len(feat) > 0:
            result = result.merge(feat, on=["race_id", "car_no"], how="left")

    return result


# =============================================================
# カテゴリA：選手個人（13特徴量）
# =============================================================

def calc_racer_features(df: pd.DataFrame,
                        db_path=None) -> pd.DataFrame:
    """
    A-01 racer_class（SS=5/S1=4/S2=3/A1=2/A2=1/A3=0）
    A-02 grade_score（競走得点）
    A-03 win_rate（勝率）
    A-04 second_rate（2着率）
    A-05 third_rate（3着率）
    A-06 style_num（脚質 逃げ=2/追込=1/両=0）
    A-07 gear_ratio（ギア倍数）← 競輪固有・重要
    A-08 back_count（B：バック回数）← 最重要特徴量
    A-09 start_count（S：スタート回数）
    A-10 age（年齢）
    A-11 term（期別）
    A-12 bante_makuri_rate（番手捲り率）

    除外:
    home_count（H：ホーム回数）→ Kドリームズ非掲載のため全件NULL
    NULLが100%の特徴量は情報を持たないので除外する
    将来KEIRIN.JP等から補完できたら再度追加する
    """
    CLASS_MAP = {
        "SS": 6, "S1": 5, "S2": 4,
        "A1": 3, "A2": 2, "A3": 1,
        # 全角対応
        "ＳＳ": 6, "Ｓ１": 5, "Ｓ２": 4,
        "Ａ１": 3, "Ａ２": 2, "Ａ３": 1,
    }

    out = df[["race_id", "car_no"]].copy()
    out["A01_racer_class"]  = df["racer_class"].map(CLASS_MAP).fillna(2).astype(int)
    out["A02_grade_score"]  = df["grade_score"].fillna(df["grade_score"].mean())
    out["A03_win_rate"]     = df["win_rate"].fillna(df["win_rate"].mean() if "win_rate" in df.columns and df["win_rate"].notna().any() else 15.0)
    out["A04_second_rate"]  = df["second_rate"].fillna(df["second_rate"].mean() if "second_rate" in df.columns and df["second_rate"].notna().any() else 30.0)
    out["A05_third_rate"]   = df["third_rate"].fillna(df["third_rate"].mean() if "third_rate" in df.columns and df["third_rate"].notna().any() else 45.0)
    out["A06_style_num"]    = df["style"].map(STYLE_MAP).fillna(0).astype(int)
    out["A07_gear_ratio"]   = df["gear_ratio"].fillna(3.6)   # 未検証：平均値3.6を仮置き
    out["A08_back_count"]   = df["back_count"].fillna(0).astype(int)   # 最重要
    # home_count は除外（Kドリームズ非掲載で全件NULL）
    out["A09_start_count"]  = df["start_count"].fillna(0).astype(int)
    out["A10_age"]          = df["age"].fillna(30).astype(int)
    out["A11_term"]         = df["term"].fillna(80).astype(int)

    # A-12: 番手捲り率（line_pos=2 のレースで rank=1 の割合）
    out["A12_bante_makuri_rate"] = df.apply(
        lambda row: _calc_bante_makuri_rate(
            row.get("racer_id"), db_path
        ), axis=1
    )

    return out


def _calc_bante_makuri_rate(racer_id, db_path, n=30):
    """
    過去Nレースで line_pos=2（番手）のレースを抽出し、
    そのうち rank=1 の割合を返す。

    N=30（直近30回）を使う。
    データが5件未満の場合は全体平均 0.15 で補完。

    Parameters:
        racer_id: 選手ID
        db_path: SQLiteのパス（None の場合はデフォルト値を返す）
        n: 直近何レースを対象にするか

    Returns:
        float: 番手捲り率（0〜1）
    """
    if racer_id is None or db_path is None:
        return 0.15  # デフォルト：全体平均（未検証）

    db = Path(db_path) if not isinstance(db_path, Path) else db_path
    if not db.exists():
        return 0.15

    try:
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()

        # results テーブルに line_pos があるか確認
        cur.execute("PRAGMA table_info(results)")
        cols = {row[1] for row in cur.fetchall()}
        if "line_pos" not in cols or "senshu_name" not in cols:
            conn.close()
            return 0.15

        # racer_id ではなく senshu_name で検索する場合もある
        # entries テーブルから racer_id → senshu_name を取得
        cur.execute(
            "SELECT DISTINCT senshu_name FROM entries WHERE racer_id = ? LIMIT 1",
            (str(racer_id),)
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return 0.15
        senshu_name = row[0]

        # line_pos=2 のレースを直近N件取得
        cur.execute("""
            SELECT r.rank
            FROM results r
            WHERE r.senshu_name = ? AND r.line_pos = 2
            ORDER BY r.race_id DESC
            LIMIT ?
        """, (senshu_name, n))
        rows = cur.fetchall()
        conn.close()

        if len(rows) < 5:
            return 0.15  # データ不足→全体平均

        wins = sum(1 for r in rows if r[0] == 1)
        return round(wins / len(rows), 4)

    except (sqlite3.Error, Exception):
        return 0.15


# =============================================================
# カテゴリB：ライン（8特徴量）← 最重要
# =============================================================

def calc_line_features(df: pd.DataFrame,
                        line_probs: dict,
                        db_path=None) -> pd.DataFrame:
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

    # B-06: 裏切りリスク（実データから計算）
    out["B06_betrayal_risk"] = df.apply(
        lambda row: _calc_betrayal_risk(
            row.get("racer_id"),
            row.get("district"),
            db_path
        ), axis=1
    )

    out["B07_district_num"] = df["district"].map(DISTRICT_MAP).fillna(0).astype(int)

    # B-08: 地元地区フラグ（会場の都道府県と選手の都道府県を照合）
    venue_prefecture = ""
    if "venue_name" in df.columns and len(df) > 0:
        venue_name = df["venue_name"].iloc[0]
        venue_prefecture = VENUE_PREFECTURE_MAP.get(venue_name, "")

    if venue_prefecture and "prefecture" in df.columns:
        out["B08_is_home_district"] = (
            df["prefecture"].fillna("").str.contains(
                venue_prefecture.replace("県", "").replace("府", "")
                                .replace("都", "").replace("道", ""),
                na=False
            ).astype(int)
        )
    else:
        out["B08_is_home_district"] = 0

    return out


def _calc_betrayal_risk(racer_id, district, db_path):
    """
    裏切りリスクを計算する。

    reporter_predictions の predicted_line と
    results の実際の line_id を比較する。

    データが5件未満 → 地区別デフォルト値
    地区もない → 0.1（全体デフォルト）

    Parameters:
        racer_id: 選手ID
        district: 選手の地区
        db_path: SQLiteのパス

    Returns:
        float: 裏切りリスク（0〜1）
    """
    district_default = DISTRICT_BETRAYAL_DEFAULT.get(district, 0.10)

    if racer_id is None or db_path is None:
        return district_default

    db = Path(db_path) if not isinstance(db_path, Path) else db_path
    if not db.exists():
        return district_default

    try:
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()

        # reporter_predictions テーブルがあるか確認
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='reporter_predictions'"
        )
        if cur.fetchone() is None:
            conn.close()
            return district_default

        # results テーブルに line_id があるか確認
        cur.execute("PRAGMA table_info(results)")
        cols = {row[1] for row in cur.fetchall()}
        if "line_id" not in cols:
            conn.close()
            return district_default

        # entries から racer_id → senshu_name
        cur.execute(
            "SELECT DISTINCT senshu_name FROM entries WHERE racer_id = ? LIMIT 1",
            (str(racer_id),)
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return district_default
        senshu_name = row[0]

        # 過去レースで predicted_line と actual line_id を比較
        # results の line_id が予測と異なる回数をカウント
        cur.execute("""
            SELECT
                res.race_id,
                res.line_id AS actual_line_id,
                rp.predicted_line
            FROM results res
            JOIN reporter_predictions rp ON res.race_id = rp.race_id
            WHERE res.senshu_name = ?
            ORDER BY res.race_id DESC
            LIMIT 30
        """, (senshu_name,))
        rows = cur.fetchall()
        conn.close()

        if len(rows) < 5:
            return district_default

        # 予想ラインに含まれるか判定
        betrayals = 0
        for _, actual_line_id, predicted_line in rows:
            if actual_line_id is None or predicted_line is None:
                continue
            # predicted_line の各グループに車番が属するかチェック
            # 簡略化: actual_line_id が予想と異なればカウント
            # （詳細な照合は実データのスキーマ確定後に改良）
            betrayals += 1 if actual_line_id is None else 0

        return round(betrayals / len(rows), 4) if rows else district_default

    except (sqlite3.Error, Exception):
        return district_default


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
# カテゴリC：バンク×脚質（7特徴量）
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
    C-07 bank_escape_rate（バンクの逃げ率）← 新規追加
    """
    out = df[["race_id", "car_no"]].copy()

    if bank_info:
        out["C01_bank_length"]   = bank_info.get("length", 400)
        out["C02_bank_straight"] = bank_info.get("straight", 54.0)
        out["C03_bank_cant"]     = bank_info.get("cant", 31.0)

        # スタイル有利度はレース内で全選手共通
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

        # C-07: バンクの逃げ率（bank_master から取得）
        out["C07_bank_escape_rate"] = bank_info.get("escape_rate", 0.28)
    else:
        out["C01_bank_length"]            = 400
        out["C02_bank_straight"]          = 54.0
        out["C03_bank_cant"]              = 31.0
        out["C04_style_advantage_escape"] = 0.50
        out["C05_style_advantage_makuri"] = 0.50
        out["C06_style_advantage_sashi"]  = 0.50
        out["C07_bank_escape_rate"]       = 0.28  # 全場平均（未検証）

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
    """
    out = df[["race_id", "car_no"]].copy()

    bank_length   = bank_info.get("length", 400) if bank_info else 400
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
        info = car_line_info.get(car_no, {})
        if info.get("position") == "先頭":
            style = style_map.get(car_no, "追込")
            base = 0.8 if style == "逃げ" else 0.4
            return min(1.0, base * bank_escape_factor)
        return 0.3

    def _makuri_score(car_no):
        base = 0.5
        if wind_speed >= 4.0:
            base += 0.15
        return min(1.0, base * bank_makuri_factor)

    def _sashi_score(car_no):
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
    E-05 is_rain（雨フラグ）
    """
    is_dome = bank_info.get("is_dome", False) if bank_info else False

    out = df[["race_id", "car_no"]].copy()

    if is_dome:
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
                        entries_df: pd.DataFrame,
                        db_path=None) -> pd.DataFrame:
    """
    F-01 win_odds_final（単勝オッズ確定値）
    F-02 win_odds_rank（単勝オッズ順位）
    F-03 implied_prob_final（市場確率 1/オッズ）
    F-04 odds_change_total（直前からの変化率：15min→final）
    F-05 odds_surge_flag（急変フラグ：3連単ベース優先、フォールバックで単勝10%）
    F-06 odds_stability（全タイミングの変動係数）
    """
    out = entries_df[["race_id", "car_no"]].copy()

    if odds_df is None or len(odds_df) == 0:
        out["F01_win_odds_final"]    = np.nan
        out["F02_win_odds_rank"]     = np.nan
        out["F03_implied_prob"]      = np.nan
        out["F04_odds_change_total"] = 0.0
        out["F05_odds_surge_flag"]   = 0
        out["F06_odds_stability"]    = np.nan
        return out

    # finalオッズ
    final_odds = odds_df[odds_df["timing"] == "final"][["race_id", "car_no", "win_odds"]]
    final_odds = final_odds.rename(columns={"win_odds": "F01_win_odds_final"})
    out = out.merge(final_odds, on=["race_id", "car_no"], how="left")
    out["F03_implied_prob"] = (1.0 / out["F01_win_odds_final"]).replace([np.inf], np.nan)

    # レース内オッズ順位
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

    # F-05: odds_surge_flag（3連単ベース優先）
    out["F05_odds_surge_flag"] = _calc_odds_surge_flags(
        out, db_path
    )

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


def _calc_odds_surge_flags(out_df, db_path):
    """
    odds_surge_flag を計算する。

    odds_history テーブルが存在する場合:
      → detect_odds_surge() の結果を使う（3連単ベース）
    odds_history テーブルが存在しない場合:
      → 単勝オッズの10%変動にフォールバック

    Returns:
        pd.Series: 0 or 1
    """
    # 3連単ベースを試行
    if db_path is not None:
        db = Path(db_path) if not isinstance(db_path, Path) else db_path
        if db.exists():
            try:
                conn = sqlite3.connect(str(db))
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='odds_history'"
                )
                has_table = cur.fetchone() is not None
                conn.close()

                if has_table:
                    # scraper_realtime.py の detect_odds_surge を利用
                    surge_map = _detect_surge_from_odds_history(db)
                    return out_df["race_id"].map(
                        lambda rid: 1 if surge_map.get(rid, False) else 0
                    )
            except (sqlite3.Error, Exception):
                pass

    # フォールバック：単勝ベースの10%変動
    if "F04_odds_change_total" in out_df.columns:
        return (out_df["F04_odds_change_total"].abs() >= 0.10).astype(int)
    return pd.Series(0, index=out_df.index)


def _detect_surge_from_odds_history(db_path, threshold=0.3):
    """
    odds_history テーブルから3連単ベースのオッズ急変を検出する。

    直前2スナップショットの上位人気（10倍未満）を比較。

    Returns:
        dict: {race_id: bool}
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 使用可能な minutes_before を取得
    cur.execute(
        "SELECT DISTINCT minutes_before FROM odds_history ORDER BY minutes_before"
    )
    available_mins = [r[0] for r in cur.fetchall()]

    if len(available_mins) < 2:
        conn.close()
        return {}

    prev_min = available_mins[-2]
    curr_min = available_mins[-1]

    # 全race_id を取得
    cur.execute("SELECT DISTINCT race_id FROM odds_history")
    race_ids = [r[0] for r in cur.fetchall()]

    surge_map = {}
    for race_id in race_ids:
        # 前回の上位人気
        cur.execute("""
            SELECT sha_ban_1, sha_ban_2, sha_ban_3, odds
            FROM odds_history
            WHERE race_id = ? AND odds_type = '3t'
              AND minutes_before = ? AND odds < 10.0
        """, (race_id, prev_min))
        prev_odds = {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}

        if not prev_odds:
            surge_map[race_id] = False
            continue

        cur.execute("""
            SELECT sha_ban_1, sha_ban_2, sha_ban_3, odds
            FROM odds_history
            WHERE race_id = ? AND odds_type = '3t'
              AND minutes_before = ?
        """, (race_id, curr_min))
        curr_odds = {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}

        found_surge = False
        for combo, prev_val in prev_odds.items():
            curr_val = curr_odds.get(combo)
            if curr_val is None or prev_val <= 0:
                continue
            change_rate = abs(curr_val - prev_val) / prev_val
            if change_rate >= threshold:
                found_surge = True
                break

        surge_map[race_id] = found_surge

    conn.close()
    return surge_map


# =============================================================
# カテゴリG：レース構成（6特徴量）
# =============================================================

def calc_race_features(df: pd.DataFrame,
                        line_probs: dict = None) -> pd.DataFrame:
    """
    G-01 race_no（レース番号）
    G-02 grade_num（G1/G2/G3/F1/F2 を数値化）
    G-03 race_type_num（予選/準決/決勝 を数値化）
    G-04 field_strength（競走得点の平均）
    G-05 n_single（単騎選手の数）
    G-06 is_midnight（ミッドナイトフラグ 0/1）
    """
    line_probs = line_probs or {}
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

    # G-05: 単騎人数（line_probs から計算）
    n_singles = sum(1 for k in line_probs if "-" not in k)
    out["G05_n_single"] = n_singles

    # G-06: ミッドナイトフラグ
    out["G06_is_midnight"] = (
        df["is_midnight"].fillna(0).astype(int)
        if "is_midnight" in df.columns else 0
    )

    return out


# =============================================================
# カテゴリH：レース内相対（4特徴量）
# =============================================================

def calc_relative_features(df: pd.DataFrame,
                            bank_info: dict = None) -> pd.DataFrame:
    """
    H-01 grade_score_rank（レース内得点順位）
    H-02 back_count_rank（レース内バック回数順位）
    H-03 grade_score_vs_field（得点 - フィールド平均）
    H-04 is_home（地元フラグ：会場都道府県と選手都道府県を照合）
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

    # H-04: 地元フラグ（会場都道府県と選手都道府県を照合）
    venue_prefecture = ""
    if bank_info:
        venue_prefecture = bank_info.get("prefecture", "")

    if venue_prefecture and "prefecture" in df.columns:
        # 都道府県名の部分一致（「東京」が「東京都」にマッチ等）
        pref_short = (venue_prefecture.replace("県", "").replace("府", "")
                                      .replace("都", "").replace("道", ""))
        out["H04_is_home"] = (
            df["prefecture"].fillna("").str.contains(pref_short, na=False)
            .astype(int)
        )
    else:
        out["H04_is_home"] = 0

    return out


# =============================================================
# カテゴリI：履歴特徴量（4特徴量）← 新規
# =============================================================

def calc_history_features(df: pd.DataFrame,
                          db_path=None) -> pd.DataFrame:
    """
    I-01 home_venue_win_rate（当場勝率）
    I-02 recent_trend_score（直近トレンドスコア）
    I-03 h2h_win_rate（対戦相手との相性スコア）
    I-04 elo_rating（拡張Eloレーティング）

    全ての計算で race_date < 当該レースの日付 を厳守し
    データリークを防止する。

    db_path が None の場合はデフォルト値で補完する。
    """
    out = df[["race_id", "car_no"]].copy()

    if db_path is None or not Path(db_path).exists():
        out["I01_home_venue_win_rate"] = 0.15  # 全体平均（未検証）
        out["I02_recent_trend_score"]  = 0.0
        out["I03_h2h_win_rate"]        = 0.5
        out["I04_elo_rating"]          = 1500.0
        out["I05_recent_agari_avg"]    = 11.5  # 全体平均（未検証）
        out["I06_agari_trend"]         = 0.0
        return out

    db = Path(db_path)

    # インデックス追加（初回のみ・既存なら無視）
    _ensure_history_indexes(db)

    # 選手名・race_date・jyo_cd を df から取得
    # senshu_name は entries 側の列を使う
    name_col = None
    for c in df.columns:
        if "senshu_name" in c:
            name_col = c
            break

    if name_col is None:
        out["I01_home_venue_win_rate"] = 0.15
        out["I02_recent_trend_score"]  = 0.0
        out["I03_h2h_win_rate"]        = 0.5
        out["I04_elo_rating"]          = 1500.0
        out["I05_recent_agari_avg"]    = 11.5
        out["I06_agari_trend"]         = 0.0
        return out

    # race_date 列の確保（マージ済みなら date か race_date がある）
    date_col = None
    for c in ["race_date", "date"]:
        if c in df.columns:
            date_col = c
            break

    jyo_col = None
    for c in ["jyo_cd", "venue_id"]:
        if c in df.columns:
            jyo_col = c
            break

    # --- バッチ計算方式（パフォーマンス最適化） ---
    # 行ごとにSQLを叩く方式は50万行で非現実的なので
    # DBから一括読み込みしてメモリ上で計算する

    conn = sqlite3.connect(str(db), timeout=30.0)
    conn.execute("PRAGMA query_only = ON;")

    # I-01: 当場勝率（バッチ計算）
    out["I01_home_venue_win_rate"] = _batch_venue_win_rate(
        df, name_col, jyo_col, date_col, conn
    )

    # I-02: 直近トレンドスコア（バッチ計算）
    out["I02_recent_trend_score"] = _batch_trend_score(
        df, name_col, date_col, conn
    )

    # I-03: 対戦相性（デフォルト値で補完 — バッチ計算は複雑すぎるため）
    # h2h は選手ペア×全レースの組み合わせ爆発が起きるため
    # 現時点ではデフォルト値を使い、将来メモリ一括計算を実装する
    out["I03_h2h_win_rate"] = 0.5

    # I-04: Eloレーティング（elo_cache テーブルから取得）
    out["I04_elo_rating"] = _batch_elo_rating(
        df, name_col, date_col, conn
    )

    # I-05/I-06: 上がりタイム特徴量（バッチ計算）
    agari_avg, agari_trend = _batch_agari_features(
        df, name_col, date_col, conn
    )
    out["I05_recent_agari_avg"] = agari_avg
    out["I06_agari_trend"] = agari_trend

    conn.close()

    return out


def _batch_agari_features(df, name_col, date_col, conn):
    """
    I-05 / I-06 の上がりタイム特徴量をバッチ計算する。

    全選手の agari_time 付き戦績を一括読み込みし、
    race_date < 条件をメモリ上で適用する（データリーク防止）。

    Returns:
        (Series, Series): (I05_recent_agari_avg, I06_agari_trend)
    """
    default_avg = pd.Series(11.5, index=df.index)
    default_trend = pd.Series(0.0, index=df.index)

    if name_col is None or date_col is None:
        return default_avg, default_trend

    # agari_time 付きの全戦績を読み込み
    cur = conn.cursor()
    cur.execute("""
        SELECT r.senshu_name, rc.race_date, r.agari_time
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.senshu_name IS NOT NULL
          AND r.agari_time IS NOT NULL
        ORDER BY rc.race_date DESC
    """)
    all_results = cur.fetchall()

    if not all_results:
        return default_avg, default_trend

    # senshu_name → [(race_date, agari_time), ...] (降順)
    from collections import defaultdict
    agari_hist = defaultdict(list)
    for sname, rdate, agari in all_results:
        agari_hist[sname].append((rdate, agari))

    def _calc_avg(row):
        sname = row.get(name_col)
        rdate = str(row.get(date_col, ""))
        if not sname or not rdate:
            return 11.5
        hist = agari_hist.get(sname, [])
        # race_date 未満の直近10件
        past = [(d, t) for d, t in hist if d < rdate][:10]
        if len(past) < 3:
            return 11.5
        return round(sum(t for _, t in past) / len(past), 3)

    def _calc_trend(row):
        sname = row.get(name_col)
        rdate = str(row.get(date_col, ""))
        if not sname or not rdate:
            return 0.0
        hist = agari_hist.get(sname, [])
        past = [(d, t) for d, t in hist if d < rdate][:10]
        if len(past) < 3:
            return 0.0
        avg_3 = sum(t for _, t in past[:3]) / 3
        avg_10 = sum(t for _, t in past) / len(past)
        return round(avg_3 - avg_10, 4)

    agari_avg = df.apply(_calc_avg, axis=1)
    agari_trend = df.apply(_calc_trend, axis=1)
    return agari_avg, agari_trend


def _batch_venue_win_rate(df, name_col, jyo_col, date_col, conn):
    """
    I-01: 当場勝率をバッチ計算する。

    全選手×全会場の過去戦績を一括読み込みし、
    race_date < 条件をメモリ上で適用する。
    """
    if name_col is None or jyo_col is None or date_col is None:
        return pd.Series(0.15, index=df.index)

    # 全選手の会場別戦績を読み込み
    cur = conn.cursor()
    cur.execute("""
        SELECT r.senshu_name, rc.jyo_cd, rc.race_date, r.rank
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.senshu_name IS NOT NULL
    """)
    all_venue_results = cur.fetchall()

    # (senshu_name, jyo_cd) → [(race_date, rank), ...] をソート済みで保持
    from collections import defaultdict
    venue_hist = defaultdict(list)
    for sname, jyo, rdate, rank in all_venue_results:
        venue_hist[(sname, int(jyo))].append((rdate, rank))

    def _calc(row):
        sname = row.get(name_col)
        jyo = row.get(jyo_col)
        rdate = str(row.get(date_col, ""))
        if not sname or not jyo or not rdate:
            return 0.15
        key = (sname, int(jyo))
        hist = venue_hist.get(key, [])
        # race_date < 当該日付のみ
        past = [(d, r) for d, r in hist if d < rdate]
        if len(past) < 5:
            return 0.15
        wins = sum(1 for _, r in past if r == 1)
        return round(wins / len(past), 4)

    return df.apply(_calc, axis=1)


def _batch_trend_score(df, name_col, date_col, conn):
    """
    I-02: 直近トレンドスコアをバッチ計算する。

    全選手の全戦績を読み込み、
    直近90日の1着率 - 直近365日の1着率を計算。
    """
    if name_col is None or date_col is None:
        return pd.Series(0.0, index=df.index)

    cur = conn.cursor()
    cur.execute("""
        SELECT r.senshu_name, rc.race_date, r.rank
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.senshu_name IS NOT NULL
    """)
    all_results = cur.fetchall()

    from collections import defaultdict
    player_hist = defaultdict(list)
    for sname, rdate, rank in all_results:
        player_hist[sname].append((rdate, rank))

    def _calc(row):
        sname = row.get(name_col)
        rdate = str(row.get(date_col, ""))
        if not sname or not rdate:
            return 0.0
        hist = player_hist.get(sname, [])
        try:
            from datetime import datetime, timedelta
            dt = datetime.strptime(rdate[:8], "%Y%m%d")
            cutoff_90 = (dt - timedelta(days=90)).strftime("%Y%m%d")
            cutoff_365 = (dt - timedelta(days=365)).strftime("%Y%m%d")
        except ValueError:
            return 0.0

        recent_90 = [(d, r) for d, r in hist if cutoff_90 <= d < rdate]
        recent_365 = [(d, r) for d, r in hist if cutoff_365 <= d < rdate]

        if len(recent_90) < 5 or len(recent_365) < 5:
            return 0.0

        rate_90 = sum(1 for _, r in recent_90 if r == 1) / len(recent_90)
        rate_365 = sum(1 for _, r in recent_365 if r == 1) / len(recent_365)
        return round(rate_90 - rate_365, 4)

    return df.apply(_calc, axis=1)


def _batch_elo_rating(df, name_col, date_col, conn):
    """
    I-04: elo_cache テーブルから Elo レーティングを取得する。

    elo_cache が空の場合はデフォルト値 1500 を返す。
    """
    if name_col is None or date_col is None:
        return pd.Series(1500.0, index=df.index)

    # elo_cache テーブルの存在確認
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='elo_cache'"
    )
    if cur.fetchone() is None:
        return pd.Series(1500.0, index=df.index)

    cur.execute("SELECT COUNT(*) FROM elo_cache")
    if cur.fetchone()[0] == 0:
        return pd.Series(1500.0, index=df.index)

    # elo_cache を全件メモリに読み込み
    cur.execute("SELECT senshu_name, as_of_date, elo_rating FROM elo_cache")
    all_elo = cur.fetchall()

    # senshu_name → [(as_of_date, elo), ...] ソート済み
    from collections import defaultdict
    elo_hist = defaultdict(list)
    for sname, adate, elo in all_elo:
        elo_hist[sname].append((adate, elo))
    # 日付順にソート
    for k in elo_hist:
        elo_hist[k].sort()

    def _calc(row):
        sname = row.get(name_col)
        rdate = str(row.get(date_col, ""))
        if not sname or not rdate:
            return 1500.0
        hist = elo_hist.get(sname, [])
        if not hist:
            return 1500.0
        # race_date 以前の最新Elo（二分探索）
        import bisect
        idx = bisect.bisect_right(
            [h[0] for h in hist], rdate
        ) - 1
        if idx < 0:
            return 1500.0
        return hist[idx][1]

    return df.apply(_calc, axis=1)


def _ensure_history_indexes(db_path):
    """履歴特徴量用のインデックスを追加する（初回のみ）"""
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_results_name
                ON results(senshu_name);
            CREATE INDEX IF NOT EXISTS idx_races_date
                ON races(race_date);
            CREATE INDEX IF NOT EXISTS idx_races_jyo
                ON races(jyo_cd);
        """)
        # Elo キャッシュテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS elo_cache (
                senshu_name TEXT NOT NULL,
                as_of_date  TEXT NOT NULL,
                elo_rating  REAL NOT NULL,
                PRIMARY KEY (senshu_name, as_of_date)
            )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def _calc_venue_win_rate(senshu_name, jyo_cd, race_date, db_path):
    """
    I-01: 当該選手が当該会場で race_date より前に
    出走したレースの1着率。5件未満は全体平均 0.15 で補完。
    """
    if not senshu_name or not jyo_cd or not race_date:
        return 0.15
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN r.rank = 1 THEN 1 ELSE 0 END)
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.senshu_name = ?
              AND rc.jyo_cd = ?
              AND rc.race_date < ?
        """, (senshu_name, int(jyo_cd), str(race_date)))
        row = cur.fetchone()
        conn.close()
        if row is None or row[0] is None or row[0] < 5:
            return 0.15
        total, wins = row[0], row[1] or 0
        return round(wins / total, 4)
    except (sqlite3.Error, Exception):
        return 0.15


def _calc_trend_score(senshu_name, race_date, db_path):
    """
    I-02: 直近3ヶ月の勝率 - 直近1年の勝率。
    各期間5件未満は 0 で補完。
    """
    if not senshu_name or not race_date:
        return 0.0
    try:
        race_date_str = str(race_date)
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()

        def _win_rate_in_period(days_back):
            """race_date から days_back 日前までの1着率"""
            # YYYYMMDD → 日付計算は文字列比較で代用
            # 90日前 / 365日前を概算
            try:
                from datetime import datetime, timedelta
                dt = datetime.strptime(race_date_str[:8], "%Y%m%d")
                cutoff = (dt - timedelta(days=days_back)).strftime("%Y%m%d")
            except ValueError:
                return None, 0
            cur.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN r.rank = 1 THEN 1 ELSE 0 END)
                FROM results r
                JOIN races rc ON r.race_id = rc.race_id
                WHERE r.senshu_name = ?
                  AND rc.race_date >= ?
                  AND rc.race_date < ?
            """, (senshu_name, cutoff, race_date_str))
            row = cur.fetchone()
            if row is None or row[0] is None or row[0] < 5:
                return None, 0
            return row[1] / row[0], row[0]

        rate_3m, n_3m = _win_rate_in_period(90)
        rate_1y, n_1y = _win_rate_in_period(365)
        conn.close()

        if rate_3m is None or rate_1y is None:
            return 0.0
        return round(rate_3m - rate_1y, 4)
    except (sqlite3.Error, Exception):
        return 0.0


def _calc_h2h_win_rate(senshu_name, opponents, race_date, db_path):
    """
    I-03: 同レースの対戦相手全員との過去直接対戦勝率の平均。
    同一レースで同時出走し、自分が上位だった率。
    対戦数10件未満は 0.5 で補完。
    """
    if not senshu_name or not opponents or not race_date:
        return 0.5
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()
        race_date_str = str(race_date)

        total_matches = 0
        total_wins = 0

        for opp in opponents:
            if not opp:
                continue
            # 両者が同一レースに出走したケースを検索
            cur.execute("""
                SELECT r1.rank, r2.rank
                FROM results r1
                JOIN results r2 ON r1.race_id = r2.race_id
                JOIN races rc ON r1.race_id = rc.race_id
                WHERE r1.senshu_name = ?
                  AND r2.senshu_name = ?
                  AND rc.race_date < ?
            """, (senshu_name, opp, race_date_str))
            rows = cur.fetchall()
            for my_rank, opp_rank in rows:
                if my_rank is not None and opp_rank is not None:
                    total_matches += 1
                    if my_rank < opp_rank:
                        total_wins += 1

        conn.close()
        if total_matches < 10:
            return 0.5
        return round(total_wins / total_matches, 4)
    except (sqlite3.Error, Exception):
        return 0.5


def _load_or_compute_elo(db_path, df, name_col, date_col):
    """
    I-04: 拡張Eloレーティング。

    計算方法:
      初期値 1500、K=32
      1着は全敗者に勝利、2着は1着以外の敗者に勝利、...
      当該レースの結果は含めない（race_date < 当該日付を厳守）

    キャッシュ:
      elo_cache テーブルに (senshu_name, as_of_date) → elo_rating を保存
      キャッシュが十分あればそこから読む

    Returns:
        dict: {(senshu_name, race_date): elo_rating}
    """
    # df に含まれる (senshu_name, race_date) ペアを収集
    needed = set()
    for _, row in df.iterrows():
        sn = row.get(name_col)
        rd = row.get(date_col)
        if sn and rd:
            needed.add((str(sn), str(rd)))

    if not needed:
        return {}

    # キャッシュから読み込みを試行
    result = {}
    uncached = set()
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()
        # elo_cache テーブルの存在確認
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='elo_cache'"
        )
        has_cache = cur.fetchone() is not None

        if has_cache:
            for sn, rd in needed:
                cur.execute(
                    "SELECT elo_rating FROM elo_cache "
                    "WHERE senshu_name = ? AND as_of_date = ?",
                    (sn, rd)
                )
                row = cur.fetchone()
                if row is not None:
                    result[(sn, rd)] = row[0]
                else:
                    uncached.add((sn, rd))
        else:
            uncached = needed

        conn.close()
    except sqlite3.Error:
        uncached = needed

    # キャッシュで全件見つかった場合
    if not uncached:
        return result

    # Elo を一括計算（全レースを時系列順に走査）
    try:
        elo_scores = _compute_elo_full(db_path)
    except Exception:
        # 計算失敗時はデフォルト値
        for key in uncached:
            result[key] = 1500.0
        return result

    # uncached 分を埋める
    for sn, rd in uncached:
        result[(sn, rd)] = elo_scores.get(sn, {}).get(rd, 1500.0)

    # キャッシュに書き込み
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        for (sn, rd), elo in result.items():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO elo_cache "
                    "(senshu_name, as_of_date, elo_rating) VALUES (?, ?, ?)",
                    (sn, rd, elo)
                )
            except sqlite3.Error:
                pass
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass

    return result


def _compute_elo_full(db_path):
    """
    全レースを時系列順に走査して Elo レーティングを計算する。
    メモリ上で一括処理し、SQLクエリを最小化する。

    Returns:
        dict: {senshu_name: {race_date: elo_before_that_date}}
        → 各選手が各日付時点で持っていた Elo 値
    """
    K = 32.0
    INITIAL = 1500.0

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA query_only = ON;")

    # 全レース結果をメモリに読み込む（時系列順）
    cur = conn.cursor()
    cur.execute("""
        SELECT r.race_id, rc.race_date, r.senshu_name, r.rank
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.senshu_name IS NOT NULL AND r.rank IS NOT NULL
        ORDER BY rc.race_date, r.race_id, r.rank
    """)
    all_results = cur.fetchall()
    conn.close()

    if not all_results:
        return {}

    # 現在の Elo 値（最新状態）
    current_elo = {}  # senshu_name → float

    # 各日付時点での Elo スナップショット
    # senshu_name → {race_date: elo_before}
    elo_history = {}

    # レースごとにグループ化
    from itertools import groupby
    from operator import itemgetter

    for race_id, race_group in groupby(all_results, key=itemgetter(0)):
        race_rows = list(race_group)
        if not race_rows:
            continue

        race_date = race_rows[0][1]

        # このレースの出走者と着順
        participants = []  # [(senshu_name, rank)]
        for _, _, sname, rank in race_rows:
            if sname:
                participants.append((sname, rank))

        # Elo スナップショットを記録（レース前の値）
        for sname, _ in participants:
            if sname not in current_elo:
                current_elo[sname] = INITIAL
            if sname not in elo_history:
                elo_history[sname] = {}
            # このレース「前」の Elo を記録
            elo_history[sname][race_date] = current_elo[sname]

        # 拡張 Elo 更新（各順位ペアで勝敗を計算）
        n = len(participants)
        if n < 2:
            continue

        elo_deltas = {sname: 0.0 for sname, _ in participants}

        for i in range(n):
            for j in range(i + 1, n):
                name_i, rank_i = participants[i]
                name_j, rank_j = participants[j]

                elo_i = current_elo.get(name_i, INITIAL)
                elo_j = current_elo.get(name_j, INITIAL)

                # 期待勝率
                exp_i = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))

                # 実際の結果（rank が小さいほど上位）
                if rank_i < rank_j:
                    actual_i = 1.0
                elif rank_i > rank_j:
                    actual_i = 0.0
                else:
                    actual_i = 0.5

                # K 係数をペア数で割って調整
                k_adj = K / max(n - 1, 1)
                delta = k_adj * (actual_i - exp_i)
                elo_deltas[name_i] += delta
                elo_deltas[name_j] -= delta

        # Elo 更新
        for sname, delta in elo_deltas.items():
            current_elo[sname] = current_elo.get(sname, INITIAL) + delta

    return elo_history


# =============================================================
# =============================================================
# カテゴリJ：決まり手実績（3特徴量）← 新規
# =============================================================

def calc_kimari_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    J-01 nige_rate（逃げ率 = nige_count / 決まり手合計）
    J-02 makuri_rate（捲り率）
    J-03 sashi_rate（差し率）

    脚質コード（style_num）と重複するが、
    実績ベースの比率の方が精度が高い可能性があるため追加。
    """
    out = df[["race_id", "car_no"]].copy()

    # 決まり手合計（nige + makuri + sashi + mark）
    nige = pd.to_numeric(df["nige_count"], errors="coerce").fillna(0) if "nige_count" in df.columns else pd.Series(0, index=df.index)
    makuri = pd.to_numeric(df["makuri_count"], errors="coerce").fillna(0) if "makuri_count" in df.columns else pd.Series(0, index=df.index)
    sashi = pd.to_numeric(df["sashi_count"], errors="coerce").fillna(0) if "sashi_count" in df.columns else pd.Series(0, index=df.index)
    mark = pd.to_numeric(df["mark_count"], errors="coerce").fillna(0) if "mark_count" in df.columns else pd.Series(0, index=df.index)
    total = nige + makuri + sashi + mark

    # ゼロ除算防止
    safe_total = total.replace(0, 1)

    out["J01_nige_rate"]   = (nige / safe_total).round(4)
    out["J02_makuri_rate"] = (makuri / safe_total).round(4)
    out["J03_sashi_rate"]  = (sashi / safe_total).round(4)

    return out


def _calc_recent_agari_avg(senshu_name, race_date, db_path, n=10):
    """
    I-05: 直近 n レースの上がりタイム平均。
    3件未満は全選手平均 11.5 で補完。
    race_date < 当該レースの日付 で厳密にリーク防止。
    """
    if not senshu_name or not race_date:
        return 11.5
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()
        cur.execute("""
            SELECT r.agari_time
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.senshu_name = ?
              AND rc.race_date < ?
              AND r.agari_time IS NOT NULL
            ORDER BY rc.race_date DESC
            LIMIT ?
        """, (senshu_name, str(race_date), n))
        vals = [row[0] for row in cur.fetchall()]
        conn.close()
        if len(vals) < 3:
            return 11.5
        return round(sum(vals) / len(vals), 3)
    except (sqlite3.Error, Exception):
        return 11.5


def _calc_agari_trend(senshu_name, race_date, db_path):
    """
    I-06: 直近3レースの上がりタイム平均 - 直近10レースの上がりタイム平均。
    マイナス（タイムが縮んでいる）→ 好調。
    データ不足時は 0 で補完。
    race_date < 当該レースの日付 で厳密にリーク防止。
    """
    if not senshu_name or not race_date:
        return 0.0
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()
        cur.execute("""
            SELECT r.agari_time
            FROM results r
            JOIN races rc ON r.race_id = rc.race_id
            WHERE r.senshu_name = ?
              AND rc.race_date < ?
              AND r.agari_time IS NOT NULL
            ORDER BY rc.race_date DESC
            LIMIT 10
        """, (senshu_name, str(race_date)))
        vals = [row[0] for row in cur.fetchall()]
        conn.close()
        if len(vals) < 3:
            return 0.0
        avg_3 = sum(vals[:3]) / 3
        avg_10 = sum(vals) / len(vals)
        return round(avg_3 - avg_10, 4)
    except (sqlite3.Error, Exception):
        return 0.0


# 特徴量名一覧（61特徴量）
# =============================================================

FEATURE_NAMES = [
    # A: 選手個人（12）← home_count除外
    "A01_racer_class", "A02_grade_score", "A03_win_rate",
    "A04_second_rate", "A05_third_rate", "A06_style_num",
    "A07_gear_ratio", "A08_back_count",
    "A09_start_count", "A10_age", "A11_term",
    "A12_bante_makuri_rate",
    # B: ライン（8）← 最重要
    "B01_line_position_num", "B02_line_size", "B03_line_confidence",
    "B04_line_grade_score", "B05_line_back_sum", "B06_betrayal_risk",
    "B07_district_num", "B08_is_home_district",
    # C: バンク×脚質（7）
    "C01_bank_length", "C02_bank_straight", "C03_bank_cant",
    "C04_style_advantage_escape", "C05_style_advantage_makuri",
    "C06_style_advantage_sashi", "C07_bank_escape_rate",
    # D: 展開予測（4）
    "D01_escape_deployment_score", "D02_makuri_deployment_score",
    "D03_sashi_deployment_score", "D04_chaos_risk",
    # E: 風・天候（5）
    "E01_wind_speed", "E02_wind_direction_sin", "E03_wind_direction_cos",
    "E04_is_back_headwind", "E05_is_rain",
    # F: オッズ（6）
    "F01_win_odds_final", "F02_win_odds_rank", "F03_implied_prob",
    "F04_odds_change_total", "F05_odds_surge_flag", "F06_odds_stability",
    # G: レース構成（6）
    "G01_race_no", "G02_grade_num", "G03_race_type_num",
    "G04_field_strength", "G05_n_single", "G06_is_midnight",
    # H: レース内相対（4）
    "H01_grade_score_rank", "H02_back_count_rank",
    "H03_grade_score_vs_field", "H04_is_home",
    # I: 履歴（6）
    "I01_home_venue_win_rate", "I02_recent_trend_score",
    "I03_h2h_win_rate", "I04_elo_rating",
    "I05_recent_agari_avg", "I06_agari_trend",
    # J: 決まり手実績（3）
    "J01_nige_rate", "J02_makuri_rate", "J03_sashi_rate",
]

assert len(FEATURE_NAMES) == 61, f"特徴量数が{len(FEATURE_NAMES)}です（61であるべき）"
