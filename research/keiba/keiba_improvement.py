# ===========================================
# keiba_improvement.py
# 競馬予測モデル改善テスト
#
# 7つの新特徴量を個別にテストし、
# Walk-Forward backtestでPF改善を確認する。
# 改善が認められたもののみ採用する。
#
# テスト項目:
#   1. 馬場状態 x 距離 x 脚質 交互作用
#   2. ペース予測特徴量
#   3. クラス昇降級特徴量
#   4. 調教師-騎手コンビ勝率
#   5. 距離適性スコアリング (精密版)
#   6. 斤量変化の影響
#   7. 休養明け効果 (放牧明け)
# ===========================================

import sys
import warnings
import time
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Import from the existing model
from research.keiba.keiba_model import (
    generate_training_data,
    engineer_features,
    KeibaEnsemble,
    KeibaRiskManager,
    find_value_bets,
    FEATURE_COLS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "keiba"
RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================
# Baseline feature set (from keiba_model.py)
# ===========================================

BASELINE_ENGINEERED_FEATURES = [
    "implied_prob", "odds_vs_winrate", "jockey_horse_combo",
    "long_rest", "weight_extreme", "inner_post", "small_field",
    "pace_predict", "style_pace_interact",
    "dist_category", "dist_x_winrate",
    "condition_x_weight", "jockey_x_class",
    "post_x_dist", "pop_x_edge", "age_x_class",
]


# ===========================================
# 1. Track Condition Interaction Features
#    馬場状態 x 距離 x 脚質
# ===========================================

def add_track_condition_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    馬場状態と距離・脚質の交互作用特徴量を追加。
    重馬場では:
      - パワー型(先行・逃げ)が有利
      - 長距離ほどスタミナ消耗で差がつく
      - 追込は脚が使えず不利
    """
    df = df.copy()
    new_cols = []

    # 馬場状態 x 脚質
    df["cond_x_style"] = df["track_condition"] * df["running_style"]
    new_cols.append("cond_x_style")

    # 馬場状態 x 距離 (重馬場の長距離はスタミナ勝負)
    df["cond_x_dist"] = df["track_condition"] * (df["distance"] / 2000.0)
    new_cols.append("cond_x_dist")

    # 馬場状態 x 距離 x 脚質 (3-way interaction)
    df["cond_dist_style"] = (
        df["track_condition"] * (df["distance"] / 2000.0) * df["running_style"]
    )
    new_cols.append("cond_dist_style")

    # 良馬場の差し追込有利フラグ (良馬場で末脚が生きる)
    df["good_track_closer"] = (
        (df["track_condition"] == 0) & (df["running_style"] >= 2)
    ).astype(int)
    new_cols.append("good_track_closer")

    # 重馬場の先行有利フラグ
    df["heavy_track_front"] = (
        (df["track_condition"] >= 2) & (df["running_style"] <= 1)
    ).astype(int)
    new_cols.append("heavy_track_front")

    return df, new_cols


# ===========================================
# 2. Pace Prediction Features
#    ペース予測 - レースが速くなるか遅くなるか
# ===========================================

def add_pace_prediction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    より精密なペース予測特徴量。
    - 逃げ馬の数
    - 先行馬の割合
    - ハイペース/スローペース指標
    - 脚質ごとのペース有利度
    """
    df = df.copy()
    new_cols = []

    # 逃げ馬の数 (per race)
    df["n_front"] = df.groupby("race_id")["running_style"].transform(
        lambda x: (x <= 0).sum()
    )
    new_cols.append("n_front")

    # 先行馬(逃げ+先行)の割合 (per race)
    df["front_ratio"] = df.groupby("race_id")["running_style"].transform(
        lambda x: (x <= 1).mean()
    )
    new_cols.append("front_ratio")

    # ハイペース指標 (逃げ馬多い or 先行馬の割合高い)
    df["hi_pace_flag"] = (
        (df["n_front"] >= 2) | (df["front_ratio"] > 0.45)
    ).astype(int)
    new_cols.append("hi_pace_flag")

    # スローペース指標
    df["slow_pace_flag"] = (
        (df["n_front"] <= 1) & (df["front_ratio"] < 0.30)
    ).astype(int)
    new_cols.append("slow_pace_flag")

    # 脚質ごとのペース適性スコア
    # ハイペース => 差し追込有利, スローペース => 逃げ先行有利
    pace_style_map = {
        # (hi_pace, style) -> bonus
    }
    def pace_style_score(row):
        fr = row["front_ratio"]
        style = row["running_style"]
        if fr > 0.45:  # ハイペース
            return {0: -0.3, 1: -0.1, 2: 0.2, 3: 0.3}.get(style, 0)
        elif fr < 0.30:  # スローペース
            return {0: 0.3, 1: 0.15, 2: -0.1, 3: -0.25}.get(style, 0)
        else:
            return 0.0

    df["pace_style_score"] = df.apply(pace_style_score, axis=1)
    new_cols.append("pace_style_score")

    # 距離 x ペース (長距離でハイペースだと消耗大)
    df["dist_x_pace"] = (df["distance"] / 2000.0) * df["front_ratio"]
    new_cols.append("dist_x_pace")

    return df, new_cols


# ===========================================
# 3. Class Transition Features
#    昇級/降級の影響
# ===========================================

def add_class_transition_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    クラスの昇降級による影響を特徴量化。
    合成データではhorse_idがないため、間接的に推定:
      - race_class と win_rate / last_finish の乖離
      - 高クラスで成績低迷 = 昇級苦戦
      - 低クラスで成績好調 = 降級で有利
    """
    df = df.copy()
    new_cols = []

    # クラスと成績の乖離 (高クラスなのに勝率低い = 昇級苦戦)
    # race_class: 0-5, win_rate: 0-0.45
    df["class_vs_perf"] = df["race_class"] * 0.2 - df["win_rate"]
    new_cols.append("class_vs_perf")

    # 高クラス初挑戦的指標 (高クラスで勝率0)
    df["class_struggle"] = (
        (df["race_class"] >= 3) & (df["win_rate"] < 0.05)
    ).astype(int)
    new_cols.append("class_struggle")

    # 降級馬的指標 (低クラスで勝率高い)
    df["class_dropper"] = (
        (df["race_class"] <= 1) & (df["win_rate"] > 0.15)
    ).astype(int)
    new_cols.append("class_dropper")

    # クラスとlast_finishの乖離
    df["class_vs_lastfinish"] = df["race_class"] - (df["last_finish"] / 3.0)
    new_cols.append("class_vs_lastfinish")

    # 前走好走 x 高クラス (勢いで昇級)
    df["momentum_upgrade"] = (
        (df["last_finish"] <= 3) & (df["race_class"] >= 3)
    ).astype(int) * df["win_rate"]
    new_cols.append("momentum_upgrade")

    return df, new_cols


# ===========================================
# 4. Trainer-Jockey Combination Win Rates
#    調教師-騎手コンビ勝率
# ===========================================

def add_trainer_jockey_combo_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    調教師と騎手の組み合わせ効果。
    合成データではtrainer_win_rateがあるが、
    コンビとしての相乗効果をモデル化。
    """
    df = df.copy()
    new_cols = []

    # trainer_win_rateが存在するか確認
    if "trainer_win_rate" not in df.columns:
        df["trainer_win_rate"] = 0.08

    # 調教師 x 騎手 コンビ指標 (両方の勝率が高いと相乗効果)
    df["trainer_jockey_combo"] = df["trainer_win_rate"] * df["jockey_win_rate"]
    new_cols.append("trainer_jockey_combo")

    # コンビの総合力 (加法的な効果)
    df["combo_sum"] = df["trainer_win_rate"] + df["jockey_win_rate"]
    new_cols.append("combo_sum")

    # 調教師と騎手の力差 (バランスの良さ)
    df["combo_balance"] = abs(df["trainer_win_rate"] - df["jockey_win_rate"])
    new_cols.append("combo_balance")

    # 一流コンビフラグ (上位調教師 x 上位騎手)
    df["elite_combo"] = (
        (df["trainer_win_rate"] > 0.12) & (df["jockey_win_rate"] > 0.12)
    ).astype(int)
    new_cols.append("elite_combo")

    # 調教師勝率 x クラス (上位クラスでの調教師の強さ)
    df["trainer_x_class_v2"] = df["trainer_win_rate"] * (df["race_class"] + 1)
    new_cols.append("trainer_x_class_v2")

    return df, new_cols


# ===========================================
# 5. Distance Aptitude Scoring (Refined)
#    距離適性スコアリング (精密版)
# ===========================================

def add_distance_aptitude_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    馬の距離適性をより精密にスコアリング。
    - 距離帯別の成績パターン
    - 距離延長/短縮の効果
    - 馬体重と距離の関係
    """
    df = df.copy()
    new_cols = []

    # 距離カテゴリ (0=sprint, 1=mile, 2=mid, 3=long)
    df["_dist_cat"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 4000],
        labels=[0, 1, 2, 3]
    )
    df["_dist_cat"] = pd.to_numeric(df["_dist_cat"], errors="coerce").fillna(1).astype(int)

    # 馬体重と距離の適性 (重い馬は短距離向き、軽い馬は長距離向き)
    df["weight_dist_apt"] = (df["horse_weight"] - 470) / 50.0 * (2.0 - df["_dist_cat"])
    new_cols.append("weight_dist_apt")

    # 距離 x 勝率の非線形交互作用
    df["dist_winrate_sq"] = (df["distance"] / 2000.0) ** 2 * df["win_rate"]
    new_cols.append("dist_winrate_sq")

    # 年齢と距離の適性 (若馬はマイル~中距離、老馬は短距離)
    df["age_dist_apt"] = np.where(
        df["horse_age"] <= 3,
        (2.0 - abs(df["_dist_cat"] - 1.5)) * 0.1,  # 若馬: マイル~中距離
        (2.0 - abs(df["_dist_cat"] - 0.5)) * 0.05   # 老馬: 短距離~マイル
    )
    new_cols.append("age_dist_apt")

    # 性別と距離 (牝馬はマイル以下で強い傾向)
    df["sex_dist"] = np.where(
        (df["sex"] == 0) & (df["_dist_cat"] <= 1), 1, 0
    )
    new_cols.append("sex_dist")

    # 過去成績と距離カテゴリのクロス (dist_aptitudeがあれば使う)
    if "dist_aptitude" in df.columns:
        df["dist_apt_confidence"] = df["dist_aptitude"] * df["place_rate"]
        new_cols.append("dist_apt_confidence")

    df.drop(columns=["_dist_cat"], inplace=True, errors="ignore")
    return df, new_cols


# ===========================================
# 6. Weight Change Impact
#    斤量変化の影響
# ===========================================

def add_weight_change_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    斤量(weight_carried)と体重変化(weight_change)の影響を特徴量化。
    - 軽量馬の有利度
    - 体重増減の非線形効果
    - 距離と斤量の関係
    """
    df = df.copy()
    new_cols = []

    # 体重変化の非線形効果 (大きな増減はマイナス)
    df["weight_change_sq"] = df["weight_change"] ** 2
    new_cols.append("weight_change_sq")

    # 体重減少フラグ (適度な減少は調子が良い可能性)
    df["slight_weight_loss"] = (
        (df["weight_change"] >= -4) & (df["weight_change"] < 0)
    ).astype(int)
    new_cols.append("slight_weight_loss")

    # 体重大幅増加フラグ (太め = 休み明け or 仕上がり不十分)
    df["big_weight_gain"] = (df["weight_change"] > 8).astype(int)
    new_cols.append("big_weight_gain")

    # 体重大幅減少フラグ (絞りすぎ or 体調不良)
    df["big_weight_loss"] = (df["weight_change"] < -8).astype(int)
    new_cols.append("big_weight_loss")

    # 斤量関連 (weight_carriedがあれば)
    if "weight_carried" in df.columns and df["weight_carried"].sum() > 0:
        # 斤量とクラスの関係
        df["kinryo_x_class"] = df["weight_carried"] * df["race_class"]
        new_cols.append("kinryo_x_class")

        # 斤量の相対的な軽さ (レース内)
        df["kinryo_rank"] = df.groupby("race_id")["weight_carried"].rank(
            method="min", ascending=True
        )
        new_cols.append("kinryo_rank")
    else:
        # weight_carriedがない場合は馬体重ベースで代替
        df["weight_x_class"] = df["horse_weight"] * df["race_class"] / 1000.0
        new_cols.append("weight_x_class")

    return df, new_cols


# ===========================================
# 7. Rest Period Effect
#    放牧明けの影響
# ===========================================

def add_rest_period_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    休養期間(days_since_last)の詳細な影響を特徴量化。
    - 短期放牧 (2-4週): ほぼ影響なし
    - 中期放牧 (1-3ヶ月): 調子次第
    - 長期放牧 (3ヶ月+): 仕上がりに不安
    - 超長期放牧 (6ヶ月+): 大幅マイナス
    """
    df = df.copy()
    new_cols = []

    days = df["days_since_last"]

    # 休養カテゴリ
    df["rest_category"] = pd.cut(
        days,
        bins=[0, 21, 45, 90, 180, 9999],
        labels=[0, 1, 2, 3, 4]
    )
    df["rest_category"] = pd.to_numeric(df["rest_category"], errors="coerce").fillna(1).astype(int)
    new_cols.append("rest_category")

    # 連闘フラグ (2週以内)
    df["quick_return"] = (days <= 14).astype(int)
    new_cols.append("quick_return")

    # 中間放牧フラグ (1-3ヶ月)
    df["medium_rest"] = ((days > 45) & (days <= 90)).astype(int)
    new_cols.append("medium_rest")

    # 長期放牧フラグ (3ヶ月超)
    df["long_rest_v2"] = (days > 90).astype(int)
    new_cols.append("long_rest_v2")

    # 超長期放牧フラグ (6ヶ月超)
    df["very_long_rest"] = (days > 180).astype(int)
    new_cols.append("very_long_rest")

    # 休養日数の対数変換 (非線形効果をキャプチャ)
    df["log_days_rest"] = np.log1p(days)
    new_cols.append("log_days_rest")

    # 休養明け x クラス (高クラスで休み明けは不利)
    df["rest_x_class"] = df["rest_category"] * df["race_class"]
    new_cols.append("rest_x_class")

    # 休養明け x 年齢 (若駒は休み明けでも走る)
    df["rest_x_age"] = df["rest_category"] * (df["horse_age"] - 3)
    new_cols.append("rest_x_age")

    return df, new_cols


# ===========================================
# Walk-Forward Backtest (per improvement)
# ===========================================

def run_wf_backtest(df: pd.DataFrame,
                    extra_features: list = None,
                    feature_adder=None,
                    initial_train_months: int = 12,
                    test_months: int = 3,
                    edge_threshold: float = 1.30,
                    bet_unit: int = 1000) -> dict:
    """
    特定の追加特徴量でWalk-Forward backtestを実行。

    Args:
        df: 生データ
        extra_features: 追加する特徴量カラム名リスト
        feature_adder: df -> (df, new_cols) の関数
        initial_train_months: 最初の訓練期間
        test_months: テスト期間
        edge_threshold: エッジ閾値
        bet_unit: ベット単位

    Returns:
        dict with overall results
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values("race_date")

    # Baseline feature engineering
    df = engineer_features(df)

    # Apply additional feature adder if provided
    added_cols = []
    if feature_adder is not None:
        df, added_cols = feature_adder(df)

    # Feature columns
    feature_cols = FEATURE_COLS + BASELINE_ENGINEERED_FEATURES.copy()

    if extra_features:
        feature_cols += extra_features
    if added_cols:
        feature_cols += added_cols

    # Deduplicate
    feature_cols = list(dict.fromkeys(feature_cols))

    # Remove any features not in df
    feature_cols = [c for c in feature_cols if c in df.columns]

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

        model = KeibaEnsemble()
        X_train = train_df[feature_cols]
        y_train = train_df["win"]
        model.fit(X_train, y_train)

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
                risk_mgr.record_bet(bet_unit, payout, date_str, "win")

        summary = risk_mgr.get_summary()
        summary["fold"] = fold
        summary["train_size"] = len(train_df)
        results.append(summary)

        fold += 1
        test_start += pd.DateOffset(months=test_months)

    # Aggregate
    total_bet = sum(r["total_bet"] for r in results)
    total_payout = sum(r["total_payout"] for r in results)
    total_n_bets = sum(r["n_bets"] for r in results)
    total_n_hits = sum(r["n_hits"] for r in results)

    all_gross_profit = sum(r["total_payout"] for r in results)
    all_gross_loss = sum((r["n_bets"] - r["n_hits"]) * bet_unit for r in results)

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
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }


# ===========================================
# Main: Test all 7 improvements
# ===========================================

def main():
    print("=" * 70)
    print("競馬予測モデル改善テスト")
    print(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Generate data
    print("\n[Data] 合成データ生成 (6000レース)...")
    df = generate_training_data(n_races=6000, seed=42)
    print(f"  {len(df)} 行, {df['race_id'].nunique()} レース")

    # --- Baseline ---
    print("\n" + "=" * 70)
    print("BASELINE (現行モデル v2.1)")
    print("=" * 70)
    t0 = time.time()
    baseline = run_wf_backtest(df)
    t_base = time.time() - t0
    print(f"  PF={baseline['overall_profit_factor']:.4f}, "
          f"RR={baseline['overall_recovery_rate']:.1%}, "
          f"bets={baseline['total_n_bets']}, "
          f"hit={baseline['overall_hit_rate']:.1%}, "
          f"features={baseline['n_features']}, "
          f"time={t_base:.1f}s")

    # --- Test each improvement ---
    improvements = [
        {
            "name": "1. Track Condition Interactions (馬場状態 x 距離 x 脚質)",
            "adder": add_track_condition_interactions,
        },
        {
            "name": "2. Pace Prediction Features (ペース予測)",
            "adder": add_pace_prediction_features,
        },
        {
            "name": "3. Class Transition Features (昇級/降級)",
            "adder": add_class_transition_features,
        },
        {
            "name": "4. Trainer-Jockey Combo (調教師-騎手コンビ)",
            "adder": add_trainer_jockey_combo_features,
        },
        {
            "name": "5. Distance Aptitude Scoring (距離適性精密版)",
            "adder": add_distance_aptitude_features,
        },
        {
            "name": "6. Weight Change Impact (斤量変化)",
            "adder": add_weight_change_features,
        },
        {
            "name": "7. Rest Period Effect (放牧明け)",
            "adder": add_rest_period_features,
        },
    ]

    results_all = {}
    beneficial = []

    for imp in improvements:
        print(f"\n{'=' * 70}")
        print(f"TEST: {imp['name']}")
        print(f"{'=' * 70}")
        t0 = time.time()
        result = run_wf_backtest(df, feature_adder=imp["adder"])
        elapsed = time.time() - t0

        pf_delta = result["overall_profit_factor"] - baseline["overall_profit_factor"]
        rr_delta = result["overall_recovery_rate"] - baseline["overall_recovery_rate"]

        is_beneficial = (
            pf_delta > 0.01 and  # PF must improve by at least 0.01
            result["overall_profit_factor"] > baseline["overall_profit_factor"] and
            result["total_n_bets"] >= baseline["total_n_bets"] * 0.8  # Don't lose too many bets
        )

        print(f"  PF={result['overall_profit_factor']:.4f} (delta={pf_delta:+.4f}), "
              f"RR={result['overall_recovery_rate']:.1%} (delta={rr_delta:+.1%}), "
              f"bets={result['total_n_bets']}, "
              f"hit={result['overall_hit_rate']:.1%}, "
              f"features={result['n_features']}, "
              f"time={elapsed:.1f}s")
        print(f"  -> {'BENEFICIAL' if is_beneficial else 'NOT BENEFICIAL'}")

        results_all[imp["name"]] = {
            "result": result,
            "pf_delta": pf_delta,
            "rr_delta": rr_delta,
            "is_beneficial": is_beneficial,
            "adder": imp["adder"],
            "time": elapsed,
        }
        if is_beneficial:
            beneficial.append(imp)

    # --- Test combined beneficial improvements ---
    combined_result = None
    if len(beneficial) > 0:
        print(f"\n{'=' * 70}")
        print(f"COMBINED TEST: {len(beneficial)} beneficial improvements together")
        print(f"{'=' * 70}")

        def combined_adder(df):
            all_cols = []
            for imp in beneficial:
                df, cols = imp["adder"](df)
                all_cols.extend(cols)
            return df, all_cols

        t0 = time.time()
        combined_result = run_wf_backtest(df, feature_adder=combined_adder)
        elapsed = time.time() - t0

        pf_delta = combined_result["overall_profit_factor"] - baseline["overall_profit_factor"]
        rr_delta = combined_result["overall_recovery_rate"] - baseline["overall_recovery_rate"]

        print(f"  PF={combined_result['overall_profit_factor']:.4f} (delta={pf_delta:+.4f}), "
              f"RR={combined_result['overall_recovery_rate']:.1%} (delta={rr_delta:+.1%}), "
              f"bets={combined_result['total_n_bets']}, "
              f"hit={combined_result['overall_hit_rate']:.1%}, "
              f"features={combined_result['n_features']}, "
              f"time={elapsed:.1f}s")

    # --- Generate results report ---
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("競馬予測モデル改善テスト結果")
    report_lines.append(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("[注意] 合成データでの検証結果です。実データでは結果が異なります。")
    report_lines.append("")

    report_lines.append("-" * 50)
    report_lines.append("BASELINE (現行モデル v2.1)")
    report_lines.append("-" * 50)
    report_lines.append(f"  Profit Factor:  {baseline['overall_profit_factor']:.4f}")
    report_lines.append(f"  Recovery Rate:  {baseline['overall_recovery_rate']:.1%}")
    report_lines.append(f"  Total Bets:     {baseline['total_n_bets']}")
    report_lines.append(f"  Hit Rate:       {baseline['overall_hit_rate']:.1%}")
    report_lines.append(f"  Features:       {baseline['n_features']}")
    report_lines.append("")

    for name, data in results_all.items():
        r = data["result"]
        report_lines.append("-" * 50)
        report_lines.append(f"TEST: {name}")
        report_lines.append("-" * 50)
        report_lines.append(f"  Profit Factor:  {r['overall_profit_factor']:.4f} (delta={data['pf_delta']:+.4f})")
        report_lines.append(f"  Recovery Rate:  {r['overall_recovery_rate']:.1%} (delta={data['rr_delta']:+.1%})")
        report_lines.append(f"  Total Bets:     {r['total_n_bets']}")
        report_lines.append(f"  Hit Rate:       {r['overall_hit_rate']:.1%}")
        report_lines.append(f"  Features:       {r['n_features']}")
        report_lines.append(f"  Time:           {data['time']:.1f}s")
        report_lines.append(f"  Verdict:        {'BENEFICIAL - ADOPT' if data['is_beneficial'] else 'NOT BENEFICIAL - SKIP'}")
        report_lines.append("")

    if combined_result:
        pf_delta = combined_result["overall_profit_factor"] - baseline["overall_profit_factor"]
        rr_delta = combined_result["overall_recovery_rate"] - baseline["overall_recovery_rate"]
        report_lines.append("=" * 50)
        report_lines.append(f"COMBINED ({len(beneficial)} beneficial improvements)")
        report_lines.append("=" * 50)
        adopted_names = [imp["name"] for imp in beneficial]
        for n in adopted_names:
            report_lines.append(f"  + {n}")
        report_lines.append(f"  Profit Factor:  {combined_result['overall_profit_factor']:.4f} (delta={pf_delta:+.4f})")
        report_lines.append(f"  Recovery Rate:  {combined_result['overall_recovery_rate']:.1%} (delta={rr_delta:+.1%})")
        report_lines.append(f"  Total Bets:     {combined_result['total_n_bets']}")
        report_lines.append(f"  Hit Rate:       {combined_result['overall_hit_rate']:.1%}")
        report_lines.append(f"  Features:       {combined_result['n_features']}")
        report_lines.append("")

    report_lines.append("=" * 50)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 50)
    n_beneficial = sum(1 for d in results_all.values() if d["is_beneficial"])
    report_lines.append(f"  Tested:     7 improvements")
    report_lines.append(f"  Beneficial: {n_beneficial}")
    report_lines.append(f"  Skipped:    {7 - n_beneficial}")
    report_lines.append("")

    if n_beneficial > 0:
        report_lines.append("  Adopted improvements:")
        for name, data in results_all.items():
            if data["is_beneficial"]:
                report_lines.append(f"    + {name} (PF delta={data['pf_delta']:+.4f})")
    else:
        report_lines.append("  No improvements showed clear benefit.")
        report_lines.append("  The baseline model is retained as-is.")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("テスト完了")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Save results
    results_path = RESULTS_DIR / "keiba_improvement_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Results] Saved to {results_path}")
    print(report_text)

    return baseline, results_all, beneficial, combined_result


if __name__ == "__main__":
    baseline, results_all, beneficial, combined_result = main()
