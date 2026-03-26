"""
validate_real_data.py
ボートレースモデルのリアルデータ検証スクリプト

目的: real_race_data.csv のみを使い、walk-forward で
      モデルの真の性能を検証する。
      合成データは一切使わない。

検証方法:
  1. real_race_data.csv を load_real_data() で読み込み
  2. 時系列順にソート
  3. 70% train / 30% holdout のシンプル分割
  4. Walk-forward (5 fold) でも検証
  5. オッズがあるレースと無いレースを分けて分析
  6. PF, ROI, win rate, bet count を報告

重要な注意点:
  - オッズが欠損しているレースでは APPROX_ODDS を使う → 信頼性が低い
  - 真の性能はオッズが利用可能なレースのみで評価すべき
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.boat.boat_model import (
    load_real_data,
    create_features,
    train_ensemble,
    train_place_ensemble,
    predict_proba,
    predict_with_agreement,
    normalize_race_probs,
    find_value_bets,
    compute_metrics,
    walk_forward_validate,
    FEATURE_COLS,
    APPROX_ODDS,
    DATA_DIR,
)

import warnings
warnings.filterwarnings("ignore")


def check_data_quality(raw_path):
    """データの品質を確認する"""
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    print(f"Total races in CSV: {len(raw)}")

    # オッズの欠損確認
    has_odds = raw["odds_1"].notna() & (raw["odds_1"] != "")
    print(f"Races with odds_1: {has_odds.sum()} / {len(raw)} ({has_odds.mean():.1%})")

    # 追加フィールド確認
    has_local = raw["lane1_local_win_rate"].notna()
    print(f"Races with local_win_rate: {has_local.sum()} / {len(raw)} ({has_local.mean():.1%})")

    has_weight = raw["lane1_weight"].notna()
    print(f"Races with weight: {has_weight.sum()} / {len(raw)} ({has_weight.mean():.1%})")

    # 日付レンジ
    dates = pd.to_numeric(raw["date"], errors="coerce").dropna().astype(int)
    print(f"Date range: {dates.min()} to {dates.max()}")

    return raw


def simple_holdout_validation(df, train_ratio=0.70):
    """
    シンプルな時系列ホールドアウト検証。
    前半 train_ratio% で訓練、後半で検証。
    """
    print("\n" + "=" * 60)
    print(f"Simple Holdout Validation (train={train_ratio:.0%}, test={1-train_ratio:.0%})")
    print("=" * 60)

    df = df.sort_values("race_date").reset_index(drop=True)
    race_ids = df["race_id"].unique()
    n_races = len(race_ids)
    split_idx = int(n_races * train_ratio)

    train_ids = set(race_ids[:split_idx])
    test_ids = set(race_ids[split_idx:])

    train_df = df[df["race_id"].isin(train_ids)]
    test_df = df[df["race_id"].isin(test_ids)].copy()

    print(f"Train: {len(train_ids)} races ({len(train_df)} rows)")
    print(f"Test:  {len(test_ids)} races ({len(test_df)} rows)")
    print(f"Train date range: {train_df['race_date'].min()} to {train_df['race_date'].max()}")
    print(f"Test  date range: {test_df['race_date'].min()} to {test_df['race_date'].max()}")

    # 訓練
    X_train = train_df[FEATURE_COLS].values
    y_train_win = train_df["win"].values

    val_split = int(len(X_train) * 0.85)
    X_tr = X_train[:val_split]
    X_va = X_train[val_split:]

    print("\nTraining Win model (5 ensemble)...")
    models = train_ensemble(X_tr, y_train_win[:val_split], X_va, y_train_win[val_split:])
    if models is None:
        print("[ERROR] Training failed")
        return None

    # 予測
    X_test = test_df[FEATURE_COLS].values
    test_df["raw_prob"] = predict_proba(models, X_test)
    test_df = normalize_race_probs(test_df)

    # モデル一致度
    _, agreement = predict_with_agreement(models, X_test)
    test_df["model_agreement"] = agreement

    # 単勝のバリューベット検出
    all_bets = []
    for rid in test_ids:
        race_data = test_df[test_df["race_id"] == rid]
        bets = find_value_bets(race_data, bet_type="win", kelly_frac=0.25)
        all_bets.extend(bets)

    if len(all_bets) == 0:
        print("[WARNING] No bets found!")
        return {"n_bets": 0}

    bets_df = pd.DataFrame(all_bets)
    metrics = compute_metrics(bets_df)

    print(f"\n--- Holdout Results (Win/単勝) ---")
    print(f"  Bets:      {metrics['n_bets']}")
    print(f"  Wins:      {metrics['n_wins']}")
    print(f"  Hit rate:  {metrics['hit_rate']:.1%}")
    print(f"  PF:        {metrics['pf']:.3f}")
    print(f"  ROI:       {metrics['recovery']:.4f} ({metrics['recovery']*100:.1f}%)")
    print(f"  Sharpe:    {metrics['sharpe']:.3f}")
    print(f"  MDD:       {metrics['mdd']:,.0f} yen")
    print(f"  Total bet: {metrics['total_bet']:,.0f} yen")
    print(f"  Total ret: {metrics['total_return']:,.0f} yen")
    print(f"  Total PnL: {metrics['total_pnl']:+,.0f} yen")

    # オッズ別分析: 実オッズ有 vs APPROX_ODDS使用
    has_real_odds = bets_df["odds"].apply(
        lambda o: o not in APPROX_ODDS.values()
    )
    print(f"\n--- Odds Analysis ---")
    print(f"  Bets with likely real odds: {has_real_odds.sum()}")
    print(f"  Bets with approx odds:     {(~has_real_odds).sum()}")

    if has_real_odds.sum() > 10:
        real_odds_metrics = compute_metrics(bets_df[has_real_odds])
        print(f"\n  [Real odds only] PF={real_odds_metrics['pf']:.3f}, "
              f"ROI={real_odds_metrics['recovery']:.4f}, "
              f"Hit={real_odds_metrics['hit_rate']:.1%}, "
              f"Bets={real_odds_metrics['n_bets']}")

    # 枠番別分析
    print(f"\n--- Lane Analysis (Test data) ---")
    lane_results = {}
    for lane in range(1, 7):
        lane_bets = bets_df[bets_df["lane"] == lane]
        if len(lane_bets) > 0:
            lm = compute_metrics(lane_bets)
            lane_results[lane] = lm
            print(f"  Lane {lane}: {lm['n_bets']} bets, hit={lm['hit_rate']:.1%}, "
                  f"PF={lm['pf']:.3f}, ROI={lm['recovery']:.4f}")
        else:
            print(f"  Lane {lane}: no bets")

    return {
        "metrics": metrics,
        "bets_df": bets_df,
        "test_df": test_df,
        "lane_results": lane_results,
    }


def save_results(holdout_result, wf_result, output_path):
    """結果をテキストファイルに保存"""
    lines = []
    lines.append("=" * 70)
    lines.append("  BOAT RACING MODEL - REAL DATA VALIDATION RESULTS")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Data source: real_race_data.csv ONLY (no synthetic data)")
    lines.append("=" * 70)

    # Holdout results
    if holdout_result and holdout_result.get("n_bets", 0) > 0:
        m = holdout_result["metrics"]
        lines.append("")
        lines.append("--- SIMPLE HOLDOUT (70% train / 30% test) ---")
        lines.append(f"  Strategy:  Win (Tansho / 単勝)")
        lines.append(f"  Bets:      {m['n_bets']}")
        lines.append(f"  Wins:      {m['n_wins']}")
        lines.append(f"  Hit rate:  {m['hit_rate']:.1%}")
        lines.append(f"  PF:        {m['pf']:.3f}")
        lines.append(f"  ROI:       {m['recovery']:.4f} ({m['recovery']*100:.1f}%)")
        lines.append(f"  Sharpe:    {m['sharpe']:.3f}")
        lines.append(f"  MDD:       {m['mdd']:,.0f} yen")
        lines.append(f"  Total bet: {m['total_bet']:,.0f} yen")
        lines.append(f"  Total ret: {m['total_return']:,.0f} yen")
        lines.append(f"  Total PnL: {m['total_pnl']:+,.0f} yen")

        # Lane analysis
        if holdout_result.get("lane_results"):
            lines.append("")
            lines.append("  Lane-by-lane breakdown:")
            for lane, lm in sorted(holdout_result["lane_results"].items()):
                lines.append(f"    Lane {lane}: {lm['n_bets']} bets, hit={lm['hit_rate']:.1%}, "
                             f"PF={lm['pf']:.3f}, ROI={lm['recovery']:.4f}")
    else:
        lines.append("")
        lines.append("--- SIMPLE HOLDOUT: No bets generated ---")

    # Walk-forward results
    if wf_result and wf_result.get("strategies"):
        lines.append("")
        lines.append("--- WALK-FORWARD (5 folds, expanding window) ---")
        for name, stats in wf_result["strategies"].items():
            status = "PASS" if stats.get("passed") else "FAIL"
            lines.append(f"")
            lines.append(f"  {name}: [{status}]")
            lines.append(f"    Bets: {stats['bets']}")
            lines.append(f"    Hit rate: {stats.get('hit_rate', 0):.1%}")
            lines.append(f"    Weighted ROI: {stats.get('recovery', 0):.4f} ({stats.get('recovery', 0)*100:.1f}%)")
            lines.append(f"    Avg PF: {stats.get('avg_pf', 0):.3f}")
            lines.append(f"    Avg Sharpe: {stats.get('avg_sharpe', 0):.3f}")
            lines.append(f"    Max MDD: {stats.get('max_mdd', 0):,.0f} yen")
            if stats.get("fold_recoveries"):
                lines.append(f"    Fold ROIs: {', '.join(f'{r:.3f}' for r in stats['fold_recoveries'])}")
            if stats.get("fold_pfs"):
                lines.append(f"    Fold PFs: {', '.join(f'{p:.2f}' for p in stats['fold_pfs'])}")
            lines.append(f"    Min ROI: {stats.get('min_recovery', 0):.3f}, "
                         f"Max ROI: {stats.get('max_recovery', 0):.3f}, "
                         f"SD: {stats.get('std_recovery', 0):.3f}")

    # Verdict
    lines.append("")
    lines.append("=" * 70)
    lines.append("  VERDICT")
    lines.append("=" * 70)

    if holdout_result and holdout_result.get("n_bets", 0) > 0:
        m = holdout_result["metrics"]
        if m["pf"] > 1.0 and m["recovery"] > 1.0:
            lines.append(f"  Holdout: PROFITABLE (PF={m['pf']:.3f}, ROI={m['recovery']*100:.1f}%)")
            if m["pf"] > 1.3:
                lines.append(f"  Assessment: Model shows promise on real data.")
                lines.append(f"  Recommendation: Proceed with paper trading (small stakes).")
            else:
                lines.append(f"  Assessment: Marginal profitability. Needs more data/tuning.")
                lines.append(f"  Recommendation: Continue paper trading before live betting.")
        else:
            lines.append(f"  Holdout: UNPROFITABLE (PF={m['pf']:.3f}, ROI={m['recovery']*100:.1f}%)")
            lines.append(f"  Assessment: Model does NOT work on real data.")
            lines.append(f"  Recommendation: DO NOT bet real money. Needs fundamental redesign.")
    else:
        lines.append(f"  Holdout: No bets generated (filters too strict or data issue)")

    lines.append("")
    lines.append("  IMPORTANT CAVEATS:")
    lines.append("  - Many early races lack real odds data (uses approximate odds)")
    lines.append("  - Walk-forward on 10k races is reasonably robust")
    lines.append("  - Real-world slippage, late scratches, etc. are NOT modeled")
    lines.append("  - The 25% takeout rate is already factored into odds")
    lines.append("  - Past performance does not guarantee future results")
    lines.append("=" * 70)

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"\n[SAVED] {output_path}")
    return text


def main():
    print("=" * 60)
    print("BOAT RACING MODEL - REAL DATA VALIDATION")
    print("Using ONLY real_race_data.csv (no synthetic data)")
    print("=" * 60)

    # 1. データ品質チェック
    raw_path = DATA_DIR / "real_race_data.csv"
    print("\n[1/4] Data quality check...")
    check_data_quality(raw_path)

    # 2. データ読込・特徴量作成
    print("\n[2/4] Loading and processing real data...")
    df = load_real_data()
    if df is None or len(df) == 0:
        print("[ERROR] No real data loaded")
        return
    df = create_features(df)
    n_races = len(df) // 6
    print(f"Processed: {n_races} races, {len(df)} rows, {len(FEATURE_COLS)} features")

    # 3. Simple holdout validation
    print("\n[3/4] Running simple holdout validation...")
    holdout_result = simple_holdout_validation(df, train_ratio=0.70)

    # 4. Walk-forward validation
    print("\n[4/4] Running walk-forward validation (5 folds)...")
    wf_result = walk_forward_validate(df, n_folds=5)

    # 5. 結果保存
    output_path = PROJECT_ROOT / "research" / "boat" / "real_data_validation_results.txt"
    report = save_results(holdout_result, wf_result, output_path)
    print("\n" + report)


if __name__ == "__main__":
    main()
