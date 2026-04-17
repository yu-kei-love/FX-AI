# ===========================================
# scripts/backtest.py
# 競輪AI - バックテスト（データリーク対策版）
#
# 2024年（テスト期間）のレースに対して予測精度と投資シミュレーションを行う
#
# 注意: 現在のモデル (models/stage1_*.pkl) は
#       全期間（2022-2024）で学習されているためデータリークあり。
#       まず動作確認として実行するが、本来は2022-2023で再学習すべき。
#       その旨を出力に明記する。
#
# 注意2: odds_history テーブルが空（3連単オッズ未収集）の場合、
#        回収率計算は出来ないため的中率のみ出力する。
# ===========================================

import sqlite3
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "data"))

from feature_engine import create_features, FEATURE_NAMES, DB_PATH
from betting_logic import (
    calc_ev, kelly_bet, load_trifecta_odds_from_db,
    TAKEOUT_RATE, EV_MIN,
)

MODEL_DIR = PROJECT_DIR / "models"

# バックテスト期間
TEST_START = "20240101"
TEST_END = "20241231"

# 初期資金
INITIAL_CAPITAL = 100_000

# Kelly 上限（資金の5%/レース）
MAX_BET_PER_RACE = 5_000


def load_test_data(is_midnight: bool, db_path=DB_PATH):
    """テスト期間の races/entries/results を一括読み込み"""
    label = "ミッドナイト" if is_midnight else "通常"
    midnight_val = 1 if is_midnight else 0

    conn = sqlite3.connect(str(db_path))
    print(f"[{label}] テストデータ読み込み中 ({TEST_START}〜{TEST_END})...")

    races_df = pd.read_sql_query(f"""
        SELECT DISTINCT r.*
        FROM races r
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date >= '{TEST_START}'
          AND r.race_date <= '{TEST_END}'
          AND NOT EXISTS (
            SELECT 1 FROM entries e
            WHERE e.race_id = r.race_id AND e.kyoso_tokuten IS NULL
          )
        ORDER BY r.race_date
    """, conn)
    print(f"[{label}] 対象レース数: {len(races_df):,}")

    if len(races_df) == 0:
        conn.close()
        return None, None, None

    entries_df = pd.read_sql_query(f"""
        SELECT e.*
        FROM entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date >= '{TEST_START}'
          AND r.race_date <= '{TEST_END}'
          AND e.kyoso_tokuten IS NOT NULL
    """, conn)

    results_df = pd.read_sql_query(f"""
        SELECT res.race_id, res.sha_ban, res.rank
        FROM results res
        JOIN races r ON res.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date >= '{TEST_START}'
          AND r.race_date <= '{TEST_END}'
        ORDER BY res.race_id, res.rank
    """, conn)
    conn.close()

    print(f"[{label}] entries={len(entries_df):,}, results={len(results_df):,}")
    return races_df, entries_df, results_df


def compute_features(entries_df, races_df, db_path):
    """特徴量計算（train.pyと同じカラムリネーム）"""
    col_renames = {}
    if "sha_ban" in entries_df.columns and "car_no" not in entries_df.columns:
        col_renames["sha_ban"] = "car_no"
    if "kyakushitsu" in entries_df.columns and "style" not in entries_df.columns:
        col_renames["kyakushitsu"] = "style"
    if "ki_betsu" in entries_df.columns and "term" not in entries_df.columns:
        col_renames["ki_betsu"] = "term"
    if "kyoso_tokuten" in entries_df.columns and "grade_score" not in entries_df.columns:
        col_renames["kyoso_tokuten"] = "grade_score"
    if "todofuken" in entries_df.columns and "prefecture" not in entries_df.columns:
        col_renames["todofuken"] = "prefecture"
    if col_renames:
        entries_df = entries_df.rename(columns=col_renames)

    for col, default in [
        ("racer_class", None), ("win_rate", 0.0),
        ("second_rate", 0.0), ("third_rate", 0.0),
        ("racer_id", None), ("district", None),
    ]:
        if col not in entries_df.columns:
            entries_df[col] = default

    features = create_features(
        entries_df=entries_df,
        races_df=races_df,
        odds_df=pd.DataFrame(),
        line_probs=None,
        bank_info=None,
        db_path=str(db_path),
    )
    return features


def run_backtest(is_midnight: bool, db_path=DB_PATH):
    """
    バックテスト本体。

    1. テスト期間のデータを読み込み
    2. 特徴量計算
    3. Stage1Model で 1着確率を予測
    4. レースごとに予測上位の的中率を計算
    5. odds_history が存在すればトライフェクタEV計算
    """
    label = "midnight" if is_midnight else "normal"
    label_ja = "ミッドナイト" if is_midnight else "通常"

    print(f"\n{'='*60}")
    print(f"  {label_ja} バックテスト")
    print(f"{'='*60}\n")

    # モデル読み込み
    model_path = MODEL_DIR / f"stage1_{label}.pkl"
    if not model_path.exists():
        print(f"モデルなし: {model_path}")
        return None
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    print(f"モデル: {model_path.name} (CV AUC={model_data.get('cv_mean_auc', 0):.4f})")

    # テストデータ
    races_df, entries_df, results_df = load_test_data(is_midnight, db_path)
    if races_df is None:
        return None

    # 特徴量計算
    print(f"[{label_ja}] 特徴量計算中...")
    t0 = time.time()
    features = compute_features(entries_df, races_df, db_path)
    print(f"  特徴量計算完了: {time.time()-t0:.1f}秒, {len(features):,}行")

    # 予測
    print(f"[{label_ja}] 予測中...")
    X = features[FEATURE_NAMES].fillna(0)
    probs = model.predict_proba(X)
    features["pred_prob"] = probs

    # 実際の1着選手マップ
    winner_map = {}
    for _, row in results_df.iterrows():
        if row["rank"] == 1:
            winner_map[row["race_id"]] = int(row["sha_ban"])

    # トップ1/3予測 vs 実際の的中
    results_summary = []
    hit_top1 = 0
    hit_top3 = 0
    total_races = 0

    # race_id 単位で集計
    for race_id, group in features.groupby("race_id"):
        actual_winner = winner_map.get(race_id)
        if actual_winner is None:
            continue

        sorted_group = group.sort_values("pred_prob", ascending=False)
        top1_pick = int(sorted_group.iloc[0]["car_no"])
        top3_picks = [int(c) for c in sorted_group.head(3)["car_no"]]
        pred_prob_top1 = float(sorted_group.iloc[0]["pred_prob"])

        top1_hit = (top1_pick == actual_winner)
        top3_hit = (actual_winner in top3_picks)

        if top1_hit:
            hit_top1 += 1
        if top3_hit:
            hit_top3 += 1
        total_races += 1

        results_summary.append({
            "race_id": race_id,
            "actual_winner": actual_winner,
            "top1_pick": top1_pick,
            "top1_prob": pred_prob_top1,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "n_entries": len(group),
        })

    # 結果
    df = pd.DataFrame(results_summary)
    print(f"\n=== {label_ja} 予測精度 ===")
    print(f"  テストレース数:         {total_races:,}")
    print(f"  トップ1 的中率:         {hit_top1/total_races*100:.2f}% "
          f"({hit_top1:,}/{total_races:,})")
    print(f"  トップ3内的中率:        {hit_top3/total_races*100:.2f}% "
          f"({hit_top3:,}/{total_races:,})")
    # 無作為ベースライン
    avg_entries = df["n_entries"].mean()
    print(f"  平均出走人数:           {avg_entries:.1f}")
    print(f"  無作為 トップ1 期待値:  {1/avg_entries*100:.2f}%")
    print(f"  無作為 トップ3 期待値:  {3/avg_entries*100:.2f}%")

    # 予測確率の較正度（実際1着率を予測確率別に集計）
    df["prob_bin"] = pd.cut(
        df["top1_prob"],
        bins=[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
        labels=["<10%", "10-15%", "15-20%", "20-25%",
                "25-30%", "30-40%", "40-50%", ">50%"],
    )
    calib = df.groupby("prob_bin", observed=True).agg(
        n=("top1_hit", "size"),
        actual_rate=("top1_hit", "mean"),
    )
    print(f"\n=== {label_ja} 較正度（予測確率別の実際的中率） ===")
    for bin_label, row in calib.iterrows():
        if row["n"] > 0:
            print(f"  {bin_label}: n={int(row['n']):>6,}, "
                  f"実際的中率={row['actual_rate']*100:.1f}%")

    # ROI シミュレーション（オッズが取得できれば）
    roi_result = simulate_roi(df, db_path, label_ja)

    return {
        "label": label,
        "total_races": total_races,
        "top1_hit_rate": hit_top1 / total_races,
        "top3_hit_rate": hit_top3 / total_races,
        "df": df,
        "roi": roi_result,
    }


def simulate_roi(df, db_path, label_ja):
    """
    オッズデータを使ったROIシミュレーション。

    現時点ではオッズデータが収集されていないため、
    理論オッズ（公正オッズ）で代替して参考値を出す。

    理論オッズ = (1 / p) × (1 - TAKEOUT_RATE)

    この方式では EV = p × 理論オッズ × 0.75 = 0.75 × 0.75 = 0.5625 程度に固定され
    常にマイナス期待値になるため、実運用不可。あくまで「的中率ベース」の参考値。
    """
    # odds_history テーブル存在確認
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='odds_history'"
    )
    has_odds = cur.fetchone() is not None
    conn.close()

    if not has_odds:
        print(f"\n=== {label_ja} ROI シミュレーション ===")
        print("  odds_history テーブルなし → ROI計算スキップ")
        print("  実運用時の収支を評価するには 3連単オッズ収集が必要")
        return None

    # ... 将来実装 ...
    return None


def main():
    print("=" * 60)
    print("  競輪AI バックテスト")
    print("=" * 60)
    print(f"  テスト期間: {TEST_START} 〜 {TEST_END}")
    print()
    print("[注意]注意: 現在のモデルは全期間(2022-2024)で学習されているため")
    print("     データリークがあり、正確な汎化性能は測れません。")
    print("     本来は2022-2023で再学習が必要です。")
    print("     （まずは動作確認として実行）")
    print()

    results = []
    r = run_backtest(is_midnight=False)
    if r:
        results.append(r)
    r = run_backtest(is_midnight=True)
    if r:
        results.append(r)

    # 全体サマリ
    print("\n" + "=" * 60)
    print("  全体サマリ")
    print("=" * 60)
    total_races = sum(r["total_races"] for r in results)
    total_hit1 = sum(r["top1_hit_rate"] * r["total_races"] for r in results)
    total_hit3 = sum(r["top3_hit_rate"] * r["total_races"] for r in results)
    for r in results:
        print(f"  {r['label']}: "
              f"トップ1={r['top1_hit_rate']*100:.2f}%, "
              f"トップ3={r['top3_hit_rate']*100:.2f}% "
              f"(n={r['total_races']:,})")
    if total_races > 0:
        print(f"  --- 合計 ---")
        print(f"  トップ1 的中率: {total_hit1/total_races*100:.2f}%")
        print(f"  トップ3 的中率: {total_hit3/total_races*100:.2f}%")
    print("=" * 60)

    # データリーク警告
    if results:
        for r in results:
            if r["top1_hit_rate"] > 0.5:
                print(f"\n[注意]{r['label']} のトップ1的中率が {r['top1_hit_rate']*100:.1f}% は異常に高い")
                print("     データリークの疑いあり（モデルが2024年データを学習済みのため）")


if __name__ == "__main__":
    main()
