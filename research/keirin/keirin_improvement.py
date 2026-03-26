# ===========================================
# keirin_improvement.py
# 競輪予測モデル v3.0 改善実験
#
# 6つの改善案をWalk-Forward backtestで検証し、
# ベースラインと比較して効果のあるものだけを採用する。
#
# 改善候補:
#   1. ライン強度スコアリング (合算ライン力)
#   2. バンク形状特徴量 (333m/400m/500m)
#   3. 風向き × ライン戦術
#   4. 連戦疲労効果
#   5. 季節/天候 × ライン戦術
#   6. ライン内位置 × 脚質 交互作用
# ===========================================

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

# ベースラインモデルからインポート
from research.keirin.keirin_model import (
    FEATURE_COLS,
    generate_training_data,
    walk_forward_validation,
    simulate_value_betting,
    KeirinPredictor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "research" / "keirin"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ユーティリティ: Walk-Forward + 評価メトリクス
# ==========================================================

def evaluate_model(df, feature_cols, label="baseline", n_splits=8,
                   temperature=1.0):
    """
    Walk-Forwardで評価し、主要メトリクスを返す。
    feature_colsを動的に切り替え可能。
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

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "is_winner"].values
        y_train_top2 = df.loc[train_mask, "finish_top2"].values
        X_test = df.loc[test_mask, feature_cols].values
        test_race_ids = df.loc[test_mask, "race_id"].values
        test_finishes = df.loc[test_mask, "finish"].values

        # Win model - lightweight for speed: LGB only
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7,
            num_leaves=40, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )
        model.fit(X_train, y_train)

        # Top-2 model
        top2_model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=32, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=43, verbosity=-1,
        )
        top2_model.fit(X_train, y_train_top2)

        # Predict
        raw_win = model.predict_proba(X_test)[:, 1]
        raw_top2 = top2_model.predict_proba(X_test)[:, 1]

        # Softmax normalize per race
        win_probs = np.zeros_like(raw_win)
        for rid in np.unique(test_race_ids):
            mask = test_race_ids == rid
            rp = raw_win[mask]
            log_p = np.log(np.clip(rp, 1e-10, None)) / max(temperature, 0.01)
            log_p -= log_p.max()
            exp_p = np.exp(log_p)
            win_probs[mask] = exp_p / exp_p.sum()

        for rid in np.unique(test_race_ids):
            mask = test_race_ids == rid
            race_win = win_probs[mask]
            race_top2 = raw_top2[mask]
            race_fin = test_finishes[mask]

            combined = 0.6 * race_win + 0.4 * race_top2
            combined = combined / max(combined.sum(), 1e-8)

            pred_winner = np.argmax(combined)
            actual_winner = np.argmin(race_fin)

            top2_pred = np.argsort(-combined)[:2]
            actual_top2 = np.argsort(race_fin)[:2]

            top3_pred = np.argsort(-combined)[:3]
            actual_top3 = np.argsort(race_fin)[:3]

            all_results.append({
                "fold": fold,
                "race_id": rid,
                "win_hit": int(pred_winner == actual_winner),
                "quinella_hit": int(set(top2_pred) == set(actual_top2)),
                "exacta_hit": int(
                    top2_pred[0] == actual_top2[0]
                    and top2_pred[1] == actual_top2[1]),
                "trifecta_hit": int(
                    top3_pred[0] == actual_top3[0]
                    and top3_pred[1] == actual_top3[1]
                    and top3_pred[2] == actual_top3[2]),
                "top_prob": combined[pred_winner],
                "top2_prob_sum": combined[top2_pred[0]] + combined[top2_pred[1]],
                "field_size": mask.sum(),
            })

    results_df = pd.DataFrame(all_results)

    # Betting simulation
    bet_result = simulate_value_betting(results_df, df, use_kelly=True)

    metrics = {
        "label": label,
        "win_rate": results_df["win_hit"].mean(),
        "quinella_rate": results_df["quinella_hit"].mean(),
        "exacta_rate": results_df["exacta_hit"].mean(),
        "trifecta_rate": results_df["trifecta_hit"].mean(),
        "n_races": len(results_df),
        "total_bets": bet_result["total_bets"],
        "roi": bet_result["roi"],
        "profit_factor": bet_result["profit_factor"],
        "profit": bet_result["profit"],
        "max_drawdown": bet_result["max_drawdown"],
    }

    # Feature importance
    full_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=7,
        num_leaves=40, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )
    full_model.fit(
        df[feature_cols].values,
        df["is_winner"].values,
    )
    imp = dict(zip(feature_cols, full_model.feature_importances_))
    metrics["feature_importance"] = imp

    return metrics


def format_metrics(m):
    """メトリクスを文字列に整形"""
    lines = [
        f"  1着的中率:  {m['win_rate']:.2%}",
        f"  2車複的中率: {m['quinella_rate']:.2%}",
        f"  2車単的中率: {m['exacta_rate']:.2%}",
        f"  3連単的中率: {m['trifecta_rate']:.2%}",
        f"  ベット数:   {m['total_bets']}",
        f"  回収率:     {m['roi']:.2%}",
        f"  PF:         {m['profit_factor']:.3f}",
        f"  純損益:     {m['profit']:+,.0f}円",
        f"  最大DD:     {m['max_drawdown']:.2%}",
    ]
    return "\n".join(lines)


def compare_metrics(baseline, improved, metric_keys=None):
    """ベースラインと改善版の差分を返す"""
    if metric_keys is None:
        metric_keys = [
            "win_rate", "quinella_rate", "exacta_rate", "trifecta_rate",
            "roi", "profit_factor", "profit",
        ]
    diffs = {}
    for k in metric_keys:
        diffs[k] = improved[k] - baseline[k]
    return diffs


def is_improvement(diffs, min_win_delta=0.002, min_pf_delta=0.0):
    """
    改善判定:
    - 1着的中率が下がらない (>= -0.002)
    - PFまたは回収率が向上
    - 2車複的中率が下がらない
    """
    win_ok = diffs["win_rate"] >= -min_win_delta
    pf_ok = diffs["profit_factor"] >= min_pf_delta
    roi_ok = diffs["roi"] >= -0.005
    quinella_ok = diffs["quinella_rate"] >= -0.003
    # 少なくとも1つの主要指標が改善
    any_improvement = (
        diffs["win_rate"] > 0.001
        or diffs["quinella_rate"] > 0.001
        or diffs["profit_factor"] > 0.005
        or diffs["roi"] > 0.005
    )
    return win_ok and quinella_ok and roi_ok and any_improvement


# ==========================================================
# 改善1: ライン強度スコアリング (合算ライン力)
# ==========================================================

def add_line_strength_scoring(df):
    """
    ライン強度スコアリング:
    - line_total_points: ライン全員の競走得点合計
    - line_max_points: ライン内最高得点
    - line_min_points: ライン内最低得点
    - line_points_std: ライン内得点の標準偏差（均一性）
    - line_win_rate_avg: ライン全員の平均勝率
    - line_strength_rank: そのレース内でのライン強度順位
    """
    df = df.copy()

    # line_total_points = line_strength * line_size (近似)
    df["line_total_points"] = df["line_strength"] * df["line_size"]

    # ライン内最大/最小は直接計算できないので、race内集計で近似
    # line_strength は平均なので、個人得点とline_strengthの差で近似
    df["line_points_deviation"] = df["race_points"] - df["line_strength"]

    # ライン勝率合算
    df["line_win_rate_avg"] = df.groupby(
        ["race_id"]
    ).apply(
        lambda x: x.groupby("line_strength")["win_rate"].transform("mean")
    ).reset_index(drop=True)

    # race内でのライン強度ランク (高い方が1位)
    df["line_strength_rank"] = df.groupby("race_id")["line_strength"].rank(
        ascending=False, method="dense"
    )

    # ライン合計得点 × 自身の得点（強いラインの強い選手は二重に有利）
    df["line_total_x_personal"] = df["line_total_points"] * df["race_points"] / 10000

    return df


# ==========================================================
# 改善2: バンク形状特徴量
# ==========================================================

def add_bank_type_features(df):
    """
    バンク形状:
    - bank_333: 333mバンクフラグ (短距離→先行有利)
    - bank_500: 500mバンクフラグ (長距離→追込有利)
    - bank_x_leg_type: バンク × 脚質の詳細交互作用
    - bank_x_line_size: バンク × ライン人数
    - bank_x_escape_rate: バンク × 逃げ成功率
    """
    df = df.copy()

    df["bank_333"] = (df["bank_type"] == 0).astype(int)
    df["bank_500"] = (df["bank_type"] == 2).astype(int)

    # 333mバンクでは逃げ(0)・捲り(1)が有利、500mでは差し(2)・追込(3)が有利
    df["bank_leg_advantage"] = 0.0
    mask_333 = df["bank_type"] == 0
    mask_500 = df["bank_type"] == 2
    df.loc[mask_333 & (df["leg_type"].isin([0, 1])), "bank_leg_advantage"] = 1.0
    df.loc[mask_500 & (df["leg_type"].isin([2, 3])), "bank_leg_advantage"] = 1.0
    df.loc[mask_333 & (df["leg_type"].isin([2, 3])), "bank_leg_advantage"] = -0.5
    df.loc[mask_500 & (df["leg_type"].isin([0, 1])), "bank_leg_advantage"] = -0.5

    # バンク × ライン人数 (333mでは大ラインが有利)
    df["bank_x_line_size"] = df["bank_type"] * df["line_size"]

    # バンク × 逃げ成功率 (333mでは逃げの価値が上がる)
    df["bank_x_escape_rate"] = 0.0
    df.loc[mask_333, "bank_x_escape_rate"] = df.loc[mask_333, "escape_success_rate"] * 1.5
    df.loc[mask_500, "bank_x_escape_rate"] = df.loc[mask_500, "escape_success_rate"] * 0.5

    return df


# ==========================================================
# 改善3: 風向き × ライン戦術
# ==========================================================

def add_wind_effect_features(df):
    """
    風向きがラインに与える影響:
    - wind_headwind_penalty: 向かい風ペナルティ（先行に不利）
    - weather_x_line_position: 天候 × ライン内位置
    - rain_draft_bonus: 雨天時のドラフティング効果増大
    - weather_x_escape: 天候 × 逃げ成功率
    - wind_line_front_penalty: 悪天候時の先行不利度
    """
    df = df.copy()

    # 悪天候（雨=2）時は先行が不利（風抵抗増大、路面悪化）
    df["wind_front_penalty"] = 0.0
    rain_mask = df["weather_condition"] == 2
    df.loc[rain_mask & (df["line_position"] == 1), "wind_front_penalty"] = -1.0
    df.loc[rain_mask & (df["line_position"] == 0), "wind_front_penalty"] = -0.5

    # 雨天時は番手のドラフティング効果が増大
    df["rain_draft_bonus"] = 0.0
    df.loc[rain_mask & (df["line_position"] == 2), "rain_draft_bonus"] = 1.0
    df.loc[rain_mask & (df["line_position"] == 3), "rain_draft_bonus"] = 0.5

    # 天候 × 逃げ成功率（雨だと逃げ切りにくい）
    df["weather_x_escape"] = df["weather_condition"] * df["escape_success_rate"]

    # 天候 × ライン強度（悪天候ではライン連携の価値が上がる）
    df["weather_x_line_strength"] = df["weather_condition"] * df["line_strength"] / 100

    # 曇り以上で大ラインが有利に
    cloudy_mask = df["weather_condition"] >= 1
    df["bad_weather_line_bonus"] = 0.0
    df.loc[cloudy_mask & (df["line_size"] >= 3), "bad_weather_line_bonus"] = 1.0

    return df


# ==========================================================
# 改善4: 連戦疲労効果
# ==========================================================

def add_fatigue_features(df):
    """
    連戦の疲労効果:
    - fatigue_score: 疲労度スコア (低days=高疲労)
    - fatigue_x_age: 疲労 × 年齢 (年配者は疲労の影響大)
    - optimal_rest: 適切な休養かどうか (3-14日が最適)
    - rest_x_class: 休養 × 級班 (S級は短い休みでも影響少)
    - consecutive_race_risk: 連投リスク (1-2日以内)
    """
    df = df.copy()

    days = df["days_since_last_race"]

    # 疲労スコア (0=疲労なし, 高い=疲労大)
    df["fatigue_score"] = np.where(
        days <= 2, 3.0 - days,       # 0-2日: 高疲労
        np.where(
            days <= 7, 0.0,           # 3-7日: 最適
            np.where(
                days <= 14, 0.0,      # 8-14日: まだOK
                (days - 14) / 30      # 15日+: 休み明けで鈍い
            )
        )
    )

    # 疲労 × 年齢（35歳以上は疲労の影響大）
    df["fatigue_x_age"] = df["fatigue_score"] * np.maximum(df["racer_age"] - 30, 0) / 20

    # 適切な休養 (3-14日が最適)
    df["optimal_rest"] = ((days >= 3) & (days <= 14)).astype(float)

    # 休養 × 級班 (S級は管理が良いので休養の影響小)
    df["rest_x_class"] = df["optimal_rest"] * df["racer_class"] / 6

    # 連投リスク (days <= 2 で脚質が逃げなら特に不利)
    df["consecutive_race_risk"] = 0.0
    consecutive_mask = days <= 2
    df.loc[consecutive_mask, "consecutive_race_risk"] = 1.0
    df.loc[consecutive_mask & (df["leg_type"] == 0), "consecutive_race_risk"] = 2.0

    return df


# ==========================================================
# 改善5: 季節/天候 × ライン戦術
# ==========================================================

def add_season_weather_features(df):
    """
    季節/天候 × ライン戦術:
    - season: 季節 (simulated from race_id)
    - season_x_line_position: 季節 × ライン内位置
    - weather_volatility: 天候による荒れ度
    - morning_weather_interaction: モーニング × 天候
    - class_weather_interaction: 級班 × 天候
    """
    df = df.copy()

    # 季節をrace_idから疑似生成 (4季節, 0-3)
    df["season"] = (df["race_id"] % 365 // 91).astype(int)  # 0=春, 1=夏, 2=秋, 3=冬

    # 冬は先行不利（寒さで脚が動きにくい）、夏は追込有利（持久力勝負）
    df["season_x_line_front"] = 0.0
    winter_mask = df["season"] == 3
    summer_mask = df["season"] == 1
    df.loc[winter_mask & (df["is_line_front"] == 1), "season_x_line_front"] = -0.5
    df.loc[summer_mask & (df["line_position"].isin([2, 3])), "season_x_line_front"] = 0.5

    # 天候による荒れ度（レース予測が難しくなる度合い）
    df["weather_volatility"] = df["weather_condition"] * 0.5 + df["is_morning_race"] * 0.3

    # モーニング × 天候 (モーニング+雨は大荒れ)
    df["morning_weather"] = df["is_morning_race"] * df["weather_condition"]

    # 級班 × 天候 (S級は悪天候でも実力通り)
    df["class_weather"] = df["race_class_level"] * (2 - df["weather_condition"]) / 6

    # 季節 × バンク (冬の500mは特に追込有利)
    df["season_x_bank"] = 0.0
    df.loc[winter_mask & (df["bank_type"] == 2), "season_x_bank"] = 1.0
    df.loc[summer_mask & (df["bank_type"] == 0), "season_x_bank"] = 1.0

    return df


# ==========================================================
# 改善6: ライン内位置 × 脚質 交互作用
# ==========================================================

def add_position_legtype_features(df):
    """
    ライン内位置 × 脚質の交互作用:
    - 先行×逃げ: 理想的な配置
    - 番手×差し/追込: 理想的な番手
    - 3番手×追込: 3番手追込は有利
    - ミスマッチ: 先行なのに追込タイプなど
    - position_legtype_score: 配置適正スコア
    """
    df = df.copy()

    # 配置適正スコア
    # 先行(1) × 逃(0): 最適 = +2
    # 先行(1) × 捲(1): 適正 = +1
    # 番手(2) × 差(2): 最適 = +2
    # 番手(2) × 追込(3): 適正 = +1.5
    # 3番手(3) × 追込(3): 適正 = +1.5
    # 単騎(0) × 捲(1): 自力で行ける = +1
    # ミスマッチ: -1

    conditions = [
        # 先行 × 逃
        (df["line_position"] == 1) & (df["leg_type"] == 0),
        # 先行 × 捲
        (df["line_position"] == 1) & (df["leg_type"] == 1),
        # 番手 × 差
        (df["line_position"] == 2) & (df["leg_type"] == 2),
        # 番手 × 追込
        (df["line_position"] == 2) & (df["leg_type"] == 3),
        # 3番手 × 追込
        (df["line_position"] == 3) & (df["leg_type"] == 3),
        # 3番手 × 差
        (df["line_position"] == 3) & (df["leg_type"] == 2),
        # 単騎 × 捲
        (df["line_position"] == 0) & (df["leg_type"] == 1),
        # 単騎 × 逃
        (df["line_position"] == 0) & (df["leg_type"] == 0),
        # 先行 × 追込 (ミスマッチ)
        (df["line_position"] == 1) & (df["leg_type"] == 3),
        # 番手 × 逃 (ミスマッチ)
        (df["line_position"] == 2) & (df["leg_type"] == 0),
    ]
    values = [2.0, 1.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, -1.0, -0.5]

    df["position_legtype_score"] = 0.0
    for cond, val in zip(conditions, values):
        df.loc[cond, "position_legtype_score"] = val

    # 配置適正 × 個人実力
    df["position_fit_x_points"] = df["position_legtype_score"] * df["race_points"] / 100

    # 先行逃げの組合せ強度
    df["front_escape_combo"] = (
        (df["line_position"] == 1).astype(float)
        * (df["leg_type"] == 0).astype(float)
        * df["escape_success_rate"]
    )

    # 番手追込の組合せ強度
    df["bante_oikomi_combo"] = (
        (df["line_position"] == 2).astype(float)
        * (df["leg_type"].isin([2, 3])).astype(float)
        * df["line_block_strength"]
    )

    # ミスマッチ度
    df["position_mismatch"] = (df["position_legtype_score"] < 0).astype(float)

    return df


# ==========================================================
# メイン: 全改善を個別テスト + 最終統合テスト
# ==========================================================

def main():
    start_time = time.time()
    output_lines = []

    def log(msg):
        print(msg)
        output_lines.append(msg)

    log("=" * 70)
    log("競輪予測モデル 改善実験")
    log(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # --- データ生成 ---
    log("\n[データ生成] 10000レース (seed=42)...")
    df = generate_training_data(n_races=10000, seed=42)
    log(f"  {len(df)}行, {df['race_id'].nunique()}レース")

    # --- ベースライン ---
    log("\n" + "=" * 70)
    log("[BASELINE] 既存特徴量 ({} features)".format(len(FEATURE_COLS)))
    log("=" * 70)
    t0 = time.time()
    baseline = evaluate_model(df, FEATURE_COLS, label="baseline")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(baseline))

    improvements_applied = []
    all_experiments = []

    # ==========================================================
    # 改善1: ライン強度スコアリング
    # ==========================================================
    log("\n" + "-" * 70)
    log("[実験1] ライン強度スコアリング (ライン強度合算特徴量)")
    log("-" * 70)

    df1 = add_line_strength_scoring(df)
    new_cols_1 = [
        "line_total_points", "line_points_deviation",
        "line_strength_rank", "line_total_x_personal",
    ]
    # line_win_rate_avg can have NaN issues, check
    if "line_win_rate_avg" in df1.columns and df1["line_win_rate_avg"].notna().all():
        new_cols_1.append("line_win_rate_avg")
    feat_1 = FEATURE_COLS + [c for c in new_cols_1 if c in df1.columns]

    t0 = time.time()
    result_1 = evaluate_model(df1, feat_1, label="exp1_line_strength")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(result_1))

    diffs_1 = compare_metrics(baseline, result_1)
    log(f"  [差分] 1着: {diffs_1['win_rate']:+.3%}, 2車複: {diffs_1['quinella_rate']:+.3%}, "
        f"PF: {diffs_1['profit_factor']:+.3f}, ROI: {diffs_1['roi']:+.3%}")

    exp1_pass = is_improvement(diffs_1)
    log(f"  -> 採用判定: {'PASS' if exp1_pass else 'FAIL'}")
    all_experiments.append(("exp1_line_strength", result_1, diffs_1, exp1_pass, new_cols_1))
    if exp1_pass:
        improvements_applied.append(("exp1", new_cols_1, add_line_strength_scoring))

    # ==========================================================
    # 改善2: バンク形状特徴量
    # ==========================================================
    log("\n" + "-" * 70)
    log("[実験2] バンク形状特徴量 (333m/400m/500m)")
    log("-" * 70)

    df2 = add_bank_type_features(df)
    new_cols_2 = [
        "bank_333", "bank_500", "bank_leg_advantage",
        "bank_x_line_size", "bank_x_escape_rate",
    ]
    feat_2 = FEATURE_COLS + new_cols_2

    t0 = time.time()
    result_2 = evaluate_model(df2, feat_2, label="exp2_bank_features")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(result_2))

    diffs_2 = compare_metrics(baseline, result_2)
    log(f"  [差分] 1着: {diffs_2['win_rate']:+.3%}, 2車複: {diffs_2['quinella_rate']:+.3%}, "
        f"PF: {diffs_2['profit_factor']:+.3f}, ROI: {diffs_2['roi']:+.3%}")

    exp2_pass = is_improvement(diffs_2)
    log(f"  -> 採用判定: {'PASS' if exp2_pass else 'FAIL'}")
    all_experiments.append(("exp2_bank_features", result_2, diffs_2, exp2_pass, new_cols_2))
    if exp2_pass:
        improvements_applied.append(("exp2", new_cols_2, add_bank_type_features))

    # ==========================================================
    # 改善3: 風向き × ライン戦術
    # ==========================================================
    log("\n" + "-" * 70)
    log("[実験3] 風向き × ライン戦術")
    log("-" * 70)

    df3 = add_wind_effect_features(df)
    new_cols_3 = [
        "wind_front_penalty", "rain_draft_bonus",
        "weather_x_escape", "weather_x_line_strength",
        "bad_weather_line_bonus",
    ]
    feat_3 = FEATURE_COLS + new_cols_3

    t0 = time.time()
    result_3 = evaluate_model(df3, feat_3, label="exp3_wind_effect")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(result_3))

    diffs_3 = compare_metrics(baseline, result_3)
    log(f"  [差分] 1着: {diffs_3['win_rate']:+.3%}, 2車複: {diffs_3['quinella_rate']:+.3%}, "
        f"PF: {diffs_3['profit_factor']:+.3f}, ROI: {diffs_3['roi']:+.3%}")

    exp3_pass = is_improvement(diffs_3)
    log(f"  -> 採用判定: {'PASS' if exp3_pass else 'FAIL'}")
    all_experiments.append(("exp3_wind_effect", result_3, diffs_3, exp3_pass, new_cols_3))
    if exp3_pass:
        improvements_applied.append(("exp3", new_cols_3, add_wind_effect_features))

    # ==========================================================
    # 改善4: 連戦疲労効果
    # ==========================================================
    log("\n" + "-" * 70)
    log("[実験4] 連戦疲労効果")
    log("-" * 70)

    df4 = add_fatigue_features(df)
    new_cols_4 = [
        "fatigue_score", "fatigue_x_age", "optimal_rest",
        "rest_x_class", "consecutive_race_risk",
    ]
    feat_4 = FEATURE_COLS + new_cols_4

    t0 = time.time()
    result_4 = evaluate_model(df4, feat_4, label="exp4_fatigue")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(result_4))

    diffs_4 = compare_metrics(baseline, result_4)
    log(f"  [差分] 1着: {diffs_4['win_rate']:+.3%}, 2車複: {diffs_4['quinella_rate']:+.3%}, "
        f"PF: {diffs_4['profit_factor']:+.3f}, ROI: {diffs_4['roi']:+.3%}")

    exp4_pass = is_improvement(diffs_4)
    log(f"  -> 採用判定: {'PASS' if exp4_pass else 'FAIL'}")
    all_experiments.append(("exp4_fatigue", result_4, diffs_4, exp4_pass, new_cols_4))
    if exp4_pass:
        improvements_applied.append(("exp4", new_cols_4, add_fatigue_features))

    # ==========================================================
    # 改善5: 季節/天候 × ライン戦術
    # ==========================================================
    log("\n" + "-" * 70)
    log("[実験5] 季節/天候 × ライン戦術")
    log("-" * 70)

    df5 = add_season_weather_features(df)
    new_cols_5 = [
        "season", "season_x_line_front", "weather_volatility",
        "morning_weather", "class_weather", "season_x_bank",
    ]
    feat_5 = FEATURE_COLS + new_cols_5

    t0 = time.time()
    result_5 = evaluate_model(df5, feat_5, label="exp5_season_weather")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(result_5))

    diffs_5 = compare_metrics(baseline, result_5)
    log(f"  [差分] 1着: {diffs_5['win_rate']:+.3%}, 2車複: {diffs_5['quinella_rate']:+.3%}, "
        f"PF: {diffs_5['profit_factor']:+.3f}, ROI: {diffs_5['roi']:+.3%}")

    exp5_pass = is_improvement(diffs_5)
    log(f"  -> 採用判定: {'PASS' if exp5_pass else 'FAIL'}")
    all_experiments.append(("exp5_season_weather", result_5, diffs_5, exp5_pass, new_cols_5))
    if exp5_pass:
        improvements_applied.append(("exp5", new_cols_5, add_season_weather_features))

    # ==========================================================
    # 改善6: ライン内位置 × 脚質 交互作用
    # ==========================================================
    log("\n" + "-" * 70)
    log("[実験6] ライン内位置 × 脚質 交互作用")
    log("-" * 70)

    df6 = add_position_legtype_features(df)
    new_cols_6 = [
        "position_legtype_score", "position_fit_x_points",
        "front_escape_combo", "bante_oikomi_combo",
        "position_mismatch",
    ]
    feat_6 = FEATURE_COLS + new_cols_6

    t0 = time.time()
    result_6 = evaluate_model(df6, feat_6, label="exp6_position_legtype")
    log(f"  実行時間: {time.time()-t0:.1f}秒")
    log(format_metrics(result_6))

    diffs_6 = compare_metrics(baseline, result_6)
    log(f"  [差分] 1着: {diffs_6['win_rate']:+.3%}, 2車複: {diffs_6['quinella_rate']:+.3%}, "
        f"PF: {diffs_6['profit_factor']:+.3f}, ROI: {diffs_6['roi']:+.3%}")

    exp6_pass = is_improvement(diffs_6)
    log(f"  -> 採用判定: {'PASS' if exp6_pass else 'FAIL'}")
    all_experiments.append(("exp6_position_legtype", result_6, diffs_6, exp6_pass, new_cols_6))
    if exp6_pass:
        improvements_applied.append(("exp6", new_cols_6, add_position_legtype_features))

    # ==========================================================
    # 統合テスト: PASSした改善を全て適用
    # ==========================================================
    log("\n" + "=" * 70)
    log("[統合テスト] PASSした改善を全て適用")
    log("=" * 70)

    if len(improvements_applied) > 0:
        log(f"  適用する改善: {[imp[0] for imp in improvements_applied]}")

        df_combined = df.copy()
        combined_new_cols = []
        for name, cols, func in improvements_applied:
            df_combined = func(df_combined)
            combined_new_cols.extend(cols)

        # 重複除去
        combined_new_cols = list(dict.fromkeys(combined_new_cols))
        # 存在するカラムのみ
        combined_new_cols = [c for c in combined_new_cols
                             if c in df_combined.columns
                             and df_combined[c].notna().all()]
        feat_combined = FEATURE_COLS + combined_new_cols

        log(f"  統合特徴量数: {len(feat_combined)} (ベース {len(FEATURE_COLS)} + 新規 {len(combined_new_cols)})")

        t0 = time.time()
        result_combined = evaluate_model(df_combined, feat_combined,
                                          label="combined")
        log(f"  実行時間: {time.time()-t0:.1f}秒")
        log(format_metrics(result_combined))

        diffs_combined = compare_metrics(baseline, result_combined)
        log(f"\n  [統合差分 vs ベースライン]")
        log(f"    1着的中率:  {diffs_combined['win_rate']:+.3%}")
        log(f"    2車複的中率: {diffs_combined['quinella_rate']:+.3%}")
        log(f"    2車単的中率: {diffs_combined['exacta_rate']:+.3%}")
        log(f"    3連単的中率: {diffs_combined['trifecta_rate']:+.3%}")
        log(f"    PF:         {diffs_combined['profit_factor']:+.3f}")
        log(f"    回収率:     {diffs_combined['roi']:+.3%}")
        log(f"    純損益:     {diffs_combined['profit']:+,.0f}円")

        combined_pass = is_improvement(diffs_combined)
        log(f"\n  -> 統合採用判定: {'PASS' if combined_pass else 'FAIL'}")

        # 新規特徴量の重要度
        if "feature_importance" in result_combined:
            fi = result_combined["feature_importance"]
            log("\n  [新規特徴量の重要度]")
            new_fi = {k: fi.get(k, 0) for k in combined_new_cols if k in fi}
            sorted_fi = sorted(new_fi.items(), key=lambda x: x[1], reverse=True)
            max_fi = max(fi.values()) if fi else 1
            for name, imp in sorted_fi[:10]:
                bar = "#" * int(imp / max_fi * 25)
                log(f"    {name:30s} {imp:6.0f}  {bar}")
    else:
        log("  改善なし: 全実験がFAIL")
        combined_pass = False
        combined_new_cols = []
        feat_combined = FEATURE_COLS

    # ==========================================================
    # サマリー
    # ==========================================================
    elapsed = time.time() - start_time

    log("\n" + "=" * 70)
    log("実験サマリー")
    log("=" * 70)

    log(f"\n{'実験名':<30s} {'1着':>8s} {'2車複':>8s} {'PF':>8s} {'ROI':>8s} {'判定':>6s}")
    log("-" * 70)
    log(f"{'BASELINE':<30s} {baseline['win_rate']:>8.2%} {baseline['quinella_rate']:>8.2%} "
        f"{baseline['profit_factor']:>8.3f} {baseline['roi']:>8.2%} {'---':>6s}")

    for name, result, diffs, passed, cols in all_experiments:
        status = "PASS" if passed else "FAIL"
        log(f"{name:<30s} {result['win_rate']:>8.2%} {result['quinella_rate']:>8.2%} "
            f"{result['profit_factor']:>8.3f} {result['roi']:>8.2%} {status:>6s}")

    log(f"\n  採用された改善: {len(improvements_applied)}件")
    for name, cols, _ in improvements_applied:
        log(f"    - {name}: {cols}")

    if combined_pass:
        log(f"\n  統合テスト: PASS")
        log(f"  新規特徴量: {combined_new_cols}")
        log(f"  最終特徴量数: {len(feat_combined)}")
    else:
        log(f"\n  統合テスト: {'PASS' if len(improvements_applied) > 0 and combined_pass else 'FAIL / 対象なし'}")

    log(f"\n  総実行時間: {elapsed:.1f}秒")

    # --- 結果ファイル保存 ---
    results_path = RESULTS_DIR / "keirin_improvement_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log(f"\n  -> 結果を {results_path} に保存")

    # --- 適用すべき変更のまとめを返す ---
    return {
        "improvements_applied": improvements_applied,
        "combined_pass": combined_pass,
        "combined_new_cols": combined_new_cols,
        "feat_combined": feat_combined,
        "baseline": baseline,
        "all_experiments": all_experiments,
    }


if __name__ == "__main__":
    result = main()
