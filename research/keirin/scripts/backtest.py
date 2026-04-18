# ===========================================
# scripts/backtest.py
# 競輪AI - バックテスト（データリーク対策版）
#
# 2024年（テスト期間）のレースに対して予測精度と投資シミュレーションを行う
#
# モード:
#   --real なし: 理論オッズでROIシミュレーション (calc_theoretical_roi)
#   --real あり: results.trifecta_payout を使った実払戻ROI (run_backtest_real)
#
# 注意: 現在のモデル (models/stage1_*.pkl) は
#       全期間（2022-2024）で学習されているためデータリークあり。
#       --model_suffix 2023 で 2023年以前学習モデルを指定すること。
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


def load_payout_data(is_midnight: bool, db_path=DB_PATH):
    """
    テスト期間の rank=1 の results から trifecta_combo / trifecta_payout を読み込む。
    払戻がバックフィル済みのレースのみ対象。
    """
    midnight_val = 1 if is_midnight else 0
    conn = sqlite3.connect(str(db_path))
    payout_df = pd.read_sql_query(f"""
        SELECT res.race_id, res.sha_ban AS actual_winner,
               res.trifecta_combo, res.trifecta_payout
        FROM results res
        JOIN races r ON res.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date >= '{TEST_START}'
          AND r.race_date <= '{TEST_END}'
          AND res.rank = 1
          AND res.trifecta_payout IS NOT NULL
          AND res.trifecta_combo IS NOT NULL
    """, conn)
    conn.close()
    return payout_df


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


def run_backtest(is_midnight: bool, db_path=DB_PATH,
                 model_suffix: str = None):
    """
    バックテスト本体（理論ROI版）。

    1. テスト期間のデータを読み込み
    2. 特徴量計算
    3. Stage1Model で 1着確率を予測
    4. レースごとに予測上位の的中率を計算
    5. odds_history が存在すればトライフェクタEV計算

    Parameters:
        model_suffix: モデルファイル名サフィックス
                      例: "2023" → stage1_{label}_2023.pkl
    """
    label = "midnight" if is_midnight else "normal"
    label_ja = "ミッドナイト" if is_midnight else "通常"

    print(f"\n{'='*60}")
    print(f"  {label_ja} バックテスト")
    print(f"{'='*60}\n")

    # モデル読み込み
    if model_suffix:
        model_path = MODEL_DIR / f"stage1_{label}_{model_suffix}.pkl"
    else:
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


def calc_theoretical_roi(df, ev_threshold=1.1):
    """
    理論オッズを使ったROI計算。

    仮定:
    - 市場オッズ = 0.75 / 市場確率
    - 市場確率 = 均等（全組み合わせが同確率、1/N）
    - EV = 予測確率 × N （均等市場比）
    - 的中時の払戻 = 0.75 / 予測確率 （モデル予測＝市場予測の仮定）

    ROI > 75% はモデルが市場予測（ランダム予測）より優れている証拠。

    Parameters:
        df: run_backtest の返り値の DataFrame
            （top1_prob, top1_hit, n_entries を含む）
        ev_threshold: EV 閾値

    Returns:
        (roi_percent, n_bets, n_hits)
    """
    df = df.copy()
    # EV = 予測確率 × 出走人数 (均等市場比)
    df["ev"] = df["top1_prob"] * df["n_entries"]

    filtered = df[df["ev"] > ev_threshold]
    if len(filtered) == 0:
        return 0.0, 0, 0

    # ゼロ除算回避
    valid = filtered[filtered["top1_prob"] > 0.001]
    n_bets = len(valid)
    if n_bets == 0:
        return 0.0, 0, 0

    hits = valid[valid["top1_hit"] == True]
    if len(hits) > 0:
        # 的中レースの理論払戻合計
        theoretical_return = (0.75 / hits["top1_prob"]).sum()
    else:
        theoretical_return = 0.0

    # 1点固定賭け (bet=1) として
    roi = theoretical_return / n_bets * 100
    return roi, n_bets, len(hits)


def simulate_roi(df, db_path, label_ja):
    """
    オッズデータを使ったROIシミュレーション。

    過去3連単オッズは未収集のため、理論ROIで代替。
    EV閾値別の感度分析を出力する。
    """
    print(f"\n=== {label_ja} 理論ROI シミュレーション ===")
    print("  注意: 過去3連単オッズは未収集のため、理論値で代替")
    print("  理論オッズ = 0.75 / 予測確率")
    print("  EV = 予測確率 × N (均等市場仮定)")
    print("  ROI > 75% = モデルがランダム市場より優れている")

    # EV閾値別の感度分析
    print("\n  EV閾値別ROI:")
    print(f"  {'EV閾値':>8}  {'購入':>8}  {'的中':>7}  {'的中率':>7}  {'理論ROI':>9}")
    results = []
    for th in [1.05, 1.1, 1.2, 1.3, 1.5, 2.0]:
        roi, n_bets, n_hits = calc_theoretical_roi(df, ev_threshold=th)
        hit_rate = (n_hits / n_bets * 100) if n_bets > 0 else 0
        print(f"  EV>{th:>4.2f}  {n_bets:>8,}  {n_hits:>7,}  "
              f"{hit_rate:>6.2f}%  {roi:>8.2f}%")
        results.append({
            "threshold": th,
            "n_bets": n_bets,
            "n_hits": n_hits,
            "hit_rate": hit_rate,
            "roi": roi,
        })

    # 100%超えの有無
    over_100 = [r for r in results if r["roi"] > 100]
    if over_100:
        print(f"\n  [発見] 理論ROI > 100% の閾値: "
              f"{[r['threshold'] for r in over_100]}")
        print(f"       = モデルが市場より優れている可能性（理論上）")
    else:
        best = max(results, key=lambda r: r["roi"])
        print(f"\n  最良: EV>{best['threshold']} で ROI={best['roi']:.2f}% "
              f"(75%基準)")

    return {"ev_sensitivity": results}


# =========================================================================
#  実払戻金バックテスト（v0.32）
# =========================================================================

def run_backtest_real(is_midnight: bool, db_path=DB_PATH,
                      model_suffix: str = "2023",
                      ev_threshold: float = 1.1,
                      bet_amount: int = 100):
    """
    理論ROIではなく実際の払戻金でROIを計算する。

    買い目選定:
      1. モデルの Top1 予測を1着固定
      2. Top2・Top3 を2着・3着に配置 → 2通りの3連単:
           (Top1-Top2-Top3) と (Top1-Top3-Top2)
      3. EV > ev_threshold の組み合わせのみ購入

    EV計算（市場オッズなしのため理論値で代替）:
      combo_prob = p1 × p2/(1-p1) × p3/(1-p1-p2)
      EV = combo_prob × 120 × 0.75
         （3連単120通り均等市場・控除率25%の仮定）

    的中判定:
      購入combo と results.trifecta_combo が完全一致
      的中時払戻 = results.trifecta_payout

    出力:
      - EV閾値別の回収率
      - 月別の回収率
      - 総投資額・総払戻額・収支・ROI

    Parameters:
        is_midnight:    モデル種別
        model_suffix:   モデルサフィックス（デフォルト "2023"）
        ev_threshold:   月別サマリに使うEV閾値
        bet_amount:     1点あたりの賭金（デフォルト100円）
    """
    label = "midnight" if is_midnight else "normal"
    label_ja = "ミッドナイト" if is_midnight else "通常"

    print(f"\n{'='*60}")
    print(f"  {label_ja} 実払戻バックテスト")
    print(f"{'='*60}\n")

    # モデル読み込み
    if model_suffix:
        model_path = MODEL_DIR / f"stage1_{label}_{model_suffix}.pkl"
    else:
        model_path = MODEL_DIR / f"stage1_{label}.pkl"
    if not model_path.exists():
        print(f"モデルなし: {model_path}")
        return None
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    print(f"モデル: {model_path.name} (CV AUC={model_data.get('cv_mean_auc', 0):.4f})")

    # 払戻データ（バックフィル済み）
    payout_df = load_payout_data(is_midnight, db_path)
    print(f"払戻バックフィル済み: {len(payout_df):,}レース")
    if len(payout_df) == 0:
        print("[警告] 払戻データがありません。")
        print("        scraper_historical.py --backfill_payout を先に実行してください。")
        return None

    # テスト期間の race/entries/results
    races_df, entries_df, results_df = load_test_data(is_midnight, db_path)
    if races_df is None:
        return None

    # 払戻済みレースだけに絞る
    payout_race_ids = set(payout_df["race_id"])
    races_df = races_df[races_df["race_id"].isin(payout_race_ids)].reset_index(drop=True)
    entries_df = entries_df[entries_df["race_id"].isin(payout_race_ids)].reset_index(drop=True)
    print(f"[{label_ja}] 払戻あり & 特徴量計算対象: {len(races_df):,}レース")
    if len(races_df) == 0:
        print("[警告] 払戻あり & 特徴量条件を満たすレースがありません")
        return None

    # 特徴量計算
    print(f"[{label_ja}] 特徴量計算中...")
    t0 = time.time()
    features = compute_features(entries_df, races_df, db_path)
    print(f"  特徴量計算完了: {time.time()-t0:.1f}秒, {len(features):,}行")

    # 予測
    X = features[FEATURE_NAMES].fillna(0)
    features["pred_prob"] = model.predict_proba(X)

    # race_date をマージ（月別集計用）
    if "race_date" not in features.columns:
        features = features.merge(
            races_df[["race_id", "race_date"]], on="race_id", how="left"
        )

    # 払戻情報をレース単位マップに
    payout_map = {}
    for _, row in payout_df.iterrows():
        combo_str = row["trifecta_combo"]
        payout = row["trifecta_payout"]
        if pd.isna(combo_str) or pd.isna(payout):
            continue
        try:
            combo_tuple = tuple(int(x) for x in str(combo_str).split("-"))
        except ValueError:
            continue
        if len(combo_tuple) != 3:
            continue
        payout_map[row["race_id"]] = (combo_tuple, int(payout))

    # レースごとに購入・的中判定
    bet_records = []
    skipped = 0
    for race_id, group in features.groupby("race_id"):
        if race_id not in payout_map:
            skipped += 1
            continue
        actual_combo, actual_payout = payout_map[race_id]

        sorted_group = group.sort_values("pred_prob", ascending=False)
        if len(sorted_group) < 3:
            skipped += 1
            continue

        top3 = sorted_group.head(3)
        car_nos = [int(c) for c in top3["car_no"]]
        probs = [float(p) for p in top3["pred_prob"]]
        top1_car, top2_car, top3_car = car_nos
        p1, p2, p3 = probs

        denom2 = 1 - p1
        if denom2 <= 0:
            skipped += 1
            continue

        race_date = str(group.iloc[0].get("race_date", ""))
        month = race_date[:6] if len(race_date) >= 6 else ""

        # Combo A: Top1-Top2-Top3
        denom3_a = 1 - p1 - p2
        if denom3_a > 0:
            combo_a = (top1_car, top2_car, top3_car)
            prob_a = p1 * (p2 / denom2) * (p3 / denom3_a)
            ev_a = prob_a * 120 * 0.75
            hit_a = (combo_a == actual_combo)
            bet_records.append({
                "race_id": race_id, "race_date": race_date, "month": month,
                "combo": "-".join(map(str, combo_a)),
                "prob": prob_a, "ev": ev_a,
                "hit": hit_a,
                "payout": actual_payout if hit_a else 0,
                "bet": bet_amount,
            })

        # Combo B: Top1-Top3-Top2
        denom3_b = 1 - p1 - p3
        if denom3_b > 0:
            combo_b = (top1_car, top3_car, top2_car)
            prob_b = p1 * (p3 / denom2) * (p2 / denom3_b)
            ev_b = prob_b * 120 * 0.75
            hit_b = (combo_b == actual_combo)
            bet_records.append({
                "race_id": race_id, "race_date": race_date, "month": month,
                "combo": "-".join(map(str, combo_b)),
                "prob": prob_b, "ev": ev_b,
                "hit": hit_b,
                "payout": actual_payout if hit_b else 0,
                "bet": bet_amount,
            })

    bets_df = pd.DataFrame(bet_records)
    if len(bets_df) == 0:
        print("[警告] 購入候補ゼロ")
        return None

    print(f"\n[{label_ja}] 対象レース: {bets_df['race_id'].nunique():,}")
    print(f"[{label_ja}] 総購入候補: {len(bets_df):,}通り (スキップ {skipped:,})")

    # === EV閾値別 ROI ===
    print(f"\n=== {label_ja} EV閾値別 実ROI ===")
    print(f"  {'EV閾値':>8}  {'購入':>8}  {'的中':>6}  {'的中率':>7}  "
          f"{'投資':>12}  {'払戻':>12}  {'ROI':>9}")
    ev_results = []
    for th in [1.05, 1.10, 1.20, 1.50, 2.00]:
        filtered = bets_df[bets_df["ev"] > th]
        n_bets = len(filtered)
        if n_bets == 0:
            print(f"  EV>{th:>4.2f}  {'-':>8}  {'-':>6}  {'-':>7}  "
                  f"{'-':>12}  {'-':>12}  {'-':>9}")
            ev_results.append({"threshold": th, "n_bets": 0})
            continue
        n_hits = int(filtered["hit"].sum())
        total_bet = int(filtered["bet"].sum())
        total_return = int(filtered["payout"].sum())
        roi = (total_return / total_bet - 1) * 100 if total_bet > 0 else 0
        hit_rate = n_hits / n_bets * 100
        print(f"  EV>{th:>4.2f}  {n_bets:>8,}  {n_hits:>6,}  "
              f"{hit_rate:>6.2f}%  {total_bet:>11,}円  "
              f"{total_return:>11,}円  {roi:>+7.2f}%")
        ev_results.append({
            "threshold": th, "n_bets": n_bets, "n_hits": n_hits,
            "hit_rate": hit_rate, "total_bet": total_bet,
            "total_return": total_return, "roi": roi,
        })

    # === 月別 ROI （EV > ev_threshold で固定） ===
    print(f"\n=== {label_ja} 月別 実ROI (EV > {ev_threshold}) ===")
    print(f"  {'月':>7}  {'購入':>7}  {'的中':>6}  {'的中率':>7}  "
          f"{'投資':>11}  {'払戻':>11}  {'ROI':>9}")
    default_filtered = bets_df[bets_df["ev"] > ev_threshold]
    monthly_results = []
    if len(default_filtered) == 0:
        print(f"  (EV > {ev_threshold} を満たす購入候補なし)")
    else:
        for month, mgroup in default_filtered.groupby("month"):
            n_bets = len(mgroup)
            n_hits = int(mgroup["hit"].sum())
            total_bet = int(mgroup["bet"].sum())
            total_return = int(mgroup["payout"].sum())
            roi = (total_return / total_bet - 1) * 100 if total_bet > 0 else 0
            hit_rate = n_hits / n_bets * 100 if n_bets > 0 else 0
            print(f"  {month:>7}  {n_bets:>7,}  {n_hits:>6,}  "
                  f"{hit_rate:>6.2f}%  {total_bet:>10,}円  "
                  f"{total_return:>10,}円  {roi:>+7.2f}%")
            monthly_results.append({
                "month": month, "n_bets": n_bets, "n_hits": n_hits,
                "hit_rate": hit_rate, "total_bet": total_bet,
                "total_return": total_return, "roi": roi,
            })

    # 全体サマリ（EV > ev_threshold）
    total_bet = int(default_filtered["bet"].sum()) if len(default_filtered) else 0
    total_return = int(default_filtered["payout"].sum()) if len(default_filtered) else 0
    profit = total_return - total_bet
    total_roi = (total_return / total_bet - 1) * 100 if total_bet > 0 else 0
    print(f"\n=== {label_ja} 総合 (EV > {ev_threshold}) ===")
    print(f"  総投資額: {total_bet:,}円")
    print(f"  総払戻額: {total_return:,}円")
    print(f"  収支:     {profit:+,}円")
    print(f"  ROI:      {total_roi:+.2f}%")

    return {
        "label": label,
        "n_races_covered": int(bets_df["race_id"].nunique()),
        "ev_sensitivity": ev_results,
        "monthly": monthly_results,
        "total_bet": total_bet,
        "total_return": total_return,
        "profit": profit,
        "roi": total_roi,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage1 バックテスト")
    parser.add_argument("--model_suffix", type=str, default=None,
                        help="モデルファイル名サフィックス "
                             "（例: 2023 → stage1_*_2023.pkl を使用）")
    parser.add_argument("--real", action="store_true",
                        help="実払戻金でROIを計算 (results.trifecta_payout 使用)")
    parser.add_argument("--ev_threshold", type=float, default=1.1,
                        help="月別ROI計算に使うEV閾値 (デフォルト: 1.1)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "normal", "midnight"],
                        help="対象モデル (both/normal/midnight)")
    args = parser.parse_args()

    print("=" * 60)
    print("  競輪AI バックテスト" + ("（実払戻版）" if args.real else "（理論ROI版）"))
    print("=" * 60)
    print(f"  テスト期間: {TEST_START} 〜 {TEST_END}")
    if args.model_suffix:
        print(f"  モデル: stage1_*_{args.model_suffix}.pkl")
    else:
        print(f"  モデル: stage1_*.pkl (全期間学習・データリーク注意)")
    if args.real:
        print(f"  EV閾値(月別): {args.ev_threshold}")
    print()

    results = []
    targets = []
    if args.mode in ("both", "normal"):
        targets.append(False)
    if args.mode in ("both", "midnight"):
        targets.append(True)

    if args.real:
        # 実払戻バックテスト
        for is_mid in targets:
            r = run_backtest_real(
                is_midnight=is_mid,
                model_suffix=args.model_suffix,
                ev_threshold=args.ev_threshold,
            )
            if r:
                results.append(r)

        # 全体サマリ
        if results:
            print("\n" + "=" * 60)
            print("  実払戻バックテスト 全体サマリ")
            print("=" * 60)
            total_bet = sum(r["total_bet"] for r in results)
            total_return = sum(r["total_return"] for r in results)
            profit = total_return - total_bet
            roi = (total_return / total_bet - 1) * 100 if total_bet > 0 else 0
            for r in results:
                print(f"  {r['label']}: "
                      f"投資 {r['total_bet']:>10,}円, "
                      f"払戻 {r['total_return']:>10,}円, "
                      f"ROI {r['roi']:+7.2f}% (n_races={r['n_races_covered']:,})")
            print(f"  --- 合計 ---")
            print(f"  総投資: {total_bet:,}円  総払戻: {total_return:,}円")
            print(f"  収支:   {profit:+,}円  ROI: {roi:+.2f}%")
            print("=" * 60)
        return

    # ----- 理論ROIバックテスト（従来） -----
    for is_mid in targets:
        r = run_backtest(is_midnight=is_mid, model_suffix=args.model_suffix)
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
