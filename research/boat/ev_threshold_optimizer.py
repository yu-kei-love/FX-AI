"""
EV Threshold Optimizer for Boat Racing Model
=============================================
Tests EV thresholds from 1.0 to 2.0 (step 0.05) for each bet type
(win, exacta, quinella) using Walk-Forward validation.

Reports PF, hit rate, number of bets, and ROI at each threshold.
Finds optimal threshold maximizing PF with sufficient bet count (>50).

Does NOT modify boat_model.py.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from research.boat.boat_model import (
    FEATURE_COLS,
    APPROX_ODDS,
    create_features,
    generate_training_data,
    train_ensemble,
    train_place_ensemble,
    predict_proba,
    predict_place_proba,
    normalize_race_probs,
    kelly_fraction,
    compute_metrics,
)

DATA_DIR = PROJECT_ROOT / "data" / "boat"
RESULTS_PATH = PROJECT_ROOT / "research" / "boat_ev_optimization_results.txt"

# EV thresholds to test
EV_THRESHOLDS = [round(1.0 + i * 0.05, 2) for i in range(21)]  # 1.00 to 2.00


def find_value_bets_with_threshold(race_df, bet_type, ev_threshold, kelly_frac=0.25):
    """
    Same logic as boat_model.find_value_bets but with configurable EV threshold
    for ALL bet types (the original hardcodes exacta=1.15, quinella=1.10).
    """
    bets = []

    if bet_type == "win":
        for _, row in race_df.iterrows():
            odds = row.get("odds", APPROX_ODDS.get(row["lane"], 10.0))
            if pd.isna(odds) or odds <= 1.0:
                odds = APPROX_ODDS.get(row["lane"], 10.0)
            model_prob = row["pred_prob"]
            ev = model_prob * odds

            if ev >= ev_threshold:
                kf = kelly_fraction(model_prob, odds, fraction=kelly_frac)
                bets.append({
                    "race_id": row["race_id"],
                    "lane": row["lane"],
                    "model_prob": model_prob,
                    "odds": odds,
                    "ev": ev,
                    "kelly_fraction": kf,
                    "bet_type": "win",
                    "win": row["win"],
                })

    elif bet_type == "exacta":
        sorted_boats = race_df.sort_values("pred_prob", ascending=False)
        top_n = sorted_boats.head(3)
        has_place_prob = "pred_place_prob" in race_df.columns

        candidates = list(top_n.iterrows())
        for i, (idx_a, boat_a) in enumerate(candidates):
            for j, (idx_b, boat_b) in enumerate(candidates):
                if i == j:
                    continue

                p_win_a = boat_a["pred_prob"]

                if has_place_prob:
                    p_place_b = boat_b["pred_place_prob"]
                    p_b_not_win = max(1.0 - boat_b["pred_prob"], 0.01)
                    p_second_b_given_a = (p_place_b - boat_b["pred_prob"]) / p_b_not_win
                    p_second_b_given_a = max(min(p_second_b_given_a, 0.8), 0.02)
                else:
                    remaining_prob = sum(
                        r["pred_prob"] for _, r in candidates if r["lane"] != boat_a["lane"]
                    )
                    p_second_b_given_a = boat_b["pred_prob"] / max(remaining_prob, 0.01)
                    p_second_b_given_a = min(p_second_b_given_a, 0.7)

                exacta_prob = p_win_a * p_second_b_given_a

                lane_a = int(boat_a["lane"])
                lane_b = int(boat_b["lane"])
                odds_a = boat_a.get("odds", APPROX_ODDS.get(lane_a, 10.0))
                odds_b = boat_b.get("odds", APPROX_ODDS.get(lane_b, 10.0))
                if pd.isna(odds_a) or odds_a <= 1.0:
                    odds_a = APPROX_ODDS.get(lane_a, 10.0)
                if pd.isna(odds_b) or odds_b <= 1.0:
                    odds_b = APPROX_ODDS.get(lane_b, 10.0)

                exacta_odds = max(5.0, odds_a * odds_b * 0.35)
                exacta_odds = min(exacta_odds, 200.0)

                ev = exacta_prob * exacta_odds
                actual_hit = (boat_a["win"] == 1 and boat_b["place_top2"] == 1
                              and boat_b["win"] != 1)

                if ev >= ev_threshold and exacta_prob >= 0.03:
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

        if len(bets) > 2:
            bets.sort(key=lambda x: -x["ev"])
            bets = bets[:2]

    elif bet_type == "quinella":
        sorted_boats = race_df.sort_values("pred_prob", ascending=False)
        top_n = sorted_boats.head(4)
        has_place_prob = "pred_place_prob" in race_df.columns

        candidates = list(top_n.iterrows())
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                idx_a, boat_a = candidates[i]
                idx_b, boat_b = candidates[j]

                if has_place_prob:
                    p_place_a = boat_a["pred_place_prob"]
                    p_place_b = boat_b["pred_place_prob"]
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
                    p_a = boat_a["pred_prob"]
                    p_b = boat_b["pred_prob"]
                    place_factor_a = min(2.5, 1.0 + (1.0 / max(boat_a["lane"], 1)) * 0.5)
                    place_factor_b = min(2.5, 1.0 + (1.0 / max(boat_b["lane"], 1)) * 0.5)
                    quinella_prob = (p_a * place_factor_a) * (p_b * place_factor_b) * 2.0
                    quinella_prob = min(quinella_prob, 0.5)

                lane_a = int(boat_a["lane"])
                lane_b = int(boat_b["lane"])
                odds_a = boat_a.get("odds", APPROX_ODDS.get(lane_a, 10.0))
                odds_b = boat_b.get("odds", APPROX_ODDS.get(lane_b, 10.0))
                if pd.isna(odds_a) or odds_a <= 1.0:
                    odds_a = APPROX_ODDS.get(lane_a, 10.0)
                if pd.isna(odds_b) or odds_b <= 1.0:
                    odds_b = APPROX_ODDS.get(lane_b, 10.0)

                quinella_odds = max(3.0, odds_a * odds_b * 0.20)
                quinella_odds = min(quinella_odds, 100.0)

                ev = quinella_prob * quinella_odds
                actual_hit = (boat_a["place_top2"] == 1 and boat_b["place_top2"] == 1)

                if ev >= ev_threshold and quinella_prob >= 0.05:
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

        if len(bets) > 3:
            bets.sort(key=lambda x: -x["ev"])
            bets = bets[:3]

    return bets


def load_data():
    """Load training data: prefer model_training_data.csv, fallback to generate."""
    csv_path = DATA_DIR / "model_training_data.csv"
    if csv_path.exists():
        print(f"[Data] Loading {csv_path}")
        df = pd.read_csv(csv_path)
        if "race_date" not in df.columns:
            # Generate date from race_id
            base = pd.Timestamp("2025-01-01")
            df["race_date"] = df["race_id"].apply(lambda rid: base + pd.Timedelta(hours=rid))
        else:
            df["race_date"] = pd.to_datetime(df["race_date"])
        print(f"  {len(df)} rows, {df['race_id'].nunique()} races")
        return df
    else:
        print("[Data] No CSV found, generating synthetic data (10000 races)")
        df = generate_training_data(n_races=10000, seed=42)
        return df


def walk_forward_ev_optimization(df, n_folds=5):
    """
    Walk-Forward validation testing all EV thresholds for each bet type.
    Returns dict: {bet_type: {threshold: aggregated_metrics}}
    """
    df = create_features(df)
    df = df.sort_values("race_date").reset_index(drop=True)
    race_ids = df["race_id"].unique()
    n_races = len(race_ids)
    test_size = n_races // (n_folds + 1)

    print(f"\n{'='*70}")
    print(f"Walk-Forward EV Threshold Optimization ({n_folds} folds)")
    print(f"  Total races: {n_races}, Test size per fold: {test_size}")
    print(f"  EV thresholds: {EV_THRESHOLDS[0]} to {EV_THRESHOLDS[-1]} (step 0.05)")
    print(f"{'='*70}")

    # Store per-fold bets for each threshold x bet_type
    # Structure: {bet_type: {threshold: [list of all bets across folds]}}
    all_bets = {}
    for bt in ["win", "exacta", "quinella"]:
        all_bets[bt] = {t: [] for t in EV_THRESHOLDS}

    for fold in range(n_folds):
        train_end = test_size * (fold + 1)
        test_end = min(train_end + test_size, n_races)
        if test_end <= train_end:
            break

        train_ids = set(race_ids[:train_end])
        test_ids = set(race_ids[train_end:test_end])

        train_df = df[df["race_id"].isin(train_ids)]
        test_df = df[df["race_id"].isin(test_ids)].copy()

        print(f"\n--- Fold {fold+1}/{n_folds}: train={len(train_ids)} races, test={len(test_ids)} races ---")

        # Train/val split
        X_train_all = train_df[FEATURE_COLS].values
        y_train_win = train_df["win"].values
        y_train_place = train_df["place_top2"].values
        val_split = int(len(X_train_all) * 0.85)
        X_tr = X_train_all[:val_split]
        X_va = X_train_all[val_split:]

        # Train win model
        print("  Training win ensemble...")
        models = train_ensemble(X_tr, y_train_win[:val_split], X_va, y_train_win[val_split:])
        if models is None:
            print("  [ERROR] Win model training failed")
            continue

        # Train place model
        print("  Training place ensemble...")
        place_models = train_place_ensemble(X_tr, y_train_place[:val_split], X_va, y_train_place[val_split:])
        if place_models is not None:
            models.update(place_models)
            print("  Place model OK")
        else:
            print("  [WARN] Place model failed, using fallback")

        # Predict on test set
        X_test = test_df[FEATURE_COLS].values
        test_df["raw_prob"] = predict_proba(models, X_test)
        test_df = normalize_race_probs(test_df)

        if "lgb_place" in models:
            test_df["raw_place_prob"] = predict_place_proba(models, X_test)
            race_place_sums = test_df.groupby("race_id")["raw_place_prob"].transform("sum")
            test_df["pred_place_prob"] = test_df["raw_place_prob"] / race_place_sums * 2.0

        # For each threshold x bet_type, collect bets
        for bt in ["win", "exacta", "quinella"]:
            for ev_t in EV_THRESHOLDS:
                fold_bets = []
                for rid in test_ids:
                    race_data = test_df[test_df["race_id"] == rid]
                    bets = find_value_bets_with_threshold(race_data, bt, ev_t, kelly_frac=0.25)
                    fold_bets.extend(bets)
                all_bets[bt][ev_t].extend(fold_bets)

        print(f"  Fold {fold+1} done. Bets collected for all thresholds.")

    return all_bets


def analyze_results(all_bets):
    """Compute metrics for each bet_type x threshold and find optimal."""
    results = {}

    for bt in ["win", "exacta", "quinella"]:
        results[bt] = {}
        for ev_t in EV_THRESHOLDS:
            bets_list = all_bets[bt][ev_t]
            if len(bets_list) == 0:
                results[bt][ev_t] = {
                    "n_bets": 0, "hit_rate": 0, "pf": 0,
                    "recovery": 0, "sharpe": 0, "mdd": 0,
                    "avg_ev": 0, "total_pnl": 0,
                }
                continue

            bets_df = pd.DataFrame(bets_list)
            metrics = compute_metrics(bets_df)
            avg_ev = bets_df["ev"].mean() if len(bets_df) > 0 else 0

            results[bt][ev_t] = {
                "n_bets": metrics["n_bets"],
                "hit_rate": metrics["hit_rate"],
                "pf": metrics["pf"],
                "recovery": metrics["recovery"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "avg_ev": round(avg_ev, 3),
                "total_pnl": metrics["total_pnl"],
            }

    return results


def find_optimal(results, min_bets=50):
    """Find optimal threshold for each bet type: maximize PF with n_bets > min_bets."""
    optimal = {}
    for bt in ["win", "exacta", "quinella"]:
        best_t = None
        best_pf = 0
        best_metrics = None

        for ev_t in EV_THRESHOLDS:
            m = results[bt][ev_t]
            if m["n_bets"] >= min_bets and m["pf"] > best_pf:
                best_pf = m["pf"]
                best_t = ev_t
                best_metrics = m

        optimal[bt] = {"threshold": best_t, "metrics": best_metrics}
    return optimal


def format_report(results, optimal):
    """Format a text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("EV Threshold Optimization Results - Boat Racing Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method: Walk-Forward validation (5 expanding windows)")
    lines.append("EV range tested: 1.00 to 2.00 (step 0.05)")
    lines.append("Optimization target: Maximize PF (Profit Factor) with >= 50 bets")
    lines.append("")

    # Current defaults
    lines.append("-" * 80)
    lines.append("CURRENT MODEL DEFAULTS:")
    lines.append(f"  Win:      min_ev = 1.25")
    lines.append(f"  Exacta:   min_ev = 1.15")
    lines.append(f"  Quinella: min_ev = 1.10")
    lines.append("-" * 80)
    lines.append("")

    # Optimal results
    lines.append("=" * 80)
    lines.append("OPTIMAL THRESHOLDS (maximize PF with >= 50 bets)")
    lines.append("=" * 80)
    for bt in ["win", "exacta", "quinella"]:
        opt = optimal[bt]
        bt_label = {"win": "Win (Tansho)", "exacta": "Exacta (Nirentan)", "quinella": "Quinella (Nirenfuku)"}[bt]
        if opt["threshold"] is not None:
            m = opt["metrics"]
            lines.append(f"\n  {bt_label}:")
            lines.append(f"    Optimal EV threshold: {opt['threshold']:.2f}")
            lines.append(f"    PF:        {m['pf']:.3f}")
            lines.append(f"    Recovery:  {m['recovery']:.4f} ({m['recovery']*100:.1f}%)")
            lines.append(f"    Hit rate:  {m['hit_rate']:.4f} ({m['hit_rate']*100:.1f}%)")
            lines.append(f"    N bets:    {m['n_bets']}")
            lines.append(f"    Sharpe:    {m['sharpe']:.3f}")
            lines.append(f"    MDD:       {m['mdd']:,.0f} yen")
            lines.append(f"    Avg EV:    {m['avg_ev']:.3f}")
            lines.append(f"    Total PnL: {m['total_pnl']:,.0f} yen")
        else:
            lines.append(f"\n  {bt_label}: No threshold found with >= 50 bets")

    # Detailed tables
    for bt in ["win", "exacta", "quinella"]:
        bt_label = {"win": "Win (Tansho)", "exacta": "Exacta (Nirentan)", "quinella": "Quinella (Nirenfuku)"}[bt]
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"DETAILED RESULTS: {bt_label}")
        lines.append("=" * 80)
        lines.append(f"{'EV Thresh':>10} {'N Bets':>8} {'Hit Rate':>10} {'PF':>8} {'Recovery':>10} {'Sharpe':>8} {'MDD':>10} {'Avg EV':>8} {'Total PnL':>12}")
        lines.append("-" * 96)

        for ev_t in EV_THRESHOLDS:
            m = results[bt][ev_t]
            marker = ""
            if optimal[bt]["threshold"] == ev_t:
                marker = " <-- OPTIMAL"
            # Also mark current default
            current_defaults = {"win": 1.25, "exacta": 1.15, "quinella": 1.10}
            if abs(ev_t - current_defaults[bt]) < 0.001:
                marker += " [CURRENT]"

            lines.append(
                f"{ev_t:>10.2f} {m['n_bets']:>8} {m['hit_rate']:>10.4f} {m['pf']:>8.3f} "
                f"{m['recovery']:>10.4f} {m['sharpe']:>8.3f} {m['mdd']:>10,.0f} "
                f"{m['avg_ev']:>8.3f} {m['total_pnl']:>12,.0f}{marker}"
            )

    # Comparison summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("COMPARISON: CURRENT vs OPTIMAL")
    lines.append("=" * 80)
    current_defaults = {"win": 1.25, "exacta": 1.15, "quinella": 1.10}
    for bt in ["win", "exacta", "quinella"]:
        bt_label = {"win": "Win", "exacta": "Exacta", "quinella": "Quinella"}[bt]
        cur_t = current_defaults[bt]
        opt_t = optimal[bt]["threshold"]

        cur_m = results[bt].get(cur_t, results[bt][EV_THRESHOLDS[0]])
        opt_m = optimal[bt]["metrics"] if optimal[bt]["metrics"] else cur_m

        lines.append(f"\n  {bt_label}:")
        lines.append(f"    {'':20s} {'Current':>12s} {'Optimal':>12s} {'Change':>12s}")
        lines.append(f"    {'Threshold':20s} {cur_t:>12.2f} {(opt_t if opt_t else 0):>12.2f}")
        lines.append(f"    {'PF':20s} {cur_m['pf']:>12.3f} {opt_m['pf']:>12.3f} {opt_m['pf']-cur_m['pf']:>+12.3f}")
        lines.append(f"    {'Recovery':20s} {cur_m['recovery']:>12.4f} {opt_m['recovery']:>12.4f} {opt_m['recovery']-cur_m['recovery']:>+12.4f}")
        lines.append(f"    {'Hit Rate':20s} {cur_m['hit_rate']:>12.4f} {opt_m['hit_rate']:>12.4f} {opt_m['hit_rate']-cur_m['hit_rate']:>+12.4f}")
        lines.append(f"    {'N Bets':20s} {cur_m['n_bets']:>12d} {opt_m['n_bets']:>12d} {opt_m['n_bets']-cur_m['n_bets']:>+12d}")
        lines.append(f"    {'Total PnL':20s} {cur_m['total_pnl']:>12,.0f} {opt_m['total_pnl']:>12,.0f} {opt_m['total_pnl']-cur_m['total_pnl']:>+12,.0f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("NOTES:")
    lines.append("  - PF (Profit Factor) = Gross Profit / Gross Loss (>1.0 = profitable)")
    lines.append("  - Recovery = Total Return / Total Bet (>1.0 = net positive)")
    lines.append("  - Higher EV threshold = fewer bets but potentially higher quality")
    lines.append("  - Lower EV threshold = more bets but potentially more noise")
    lines.append("  - Optimal = highest PF with at least 50 bets across all folds")
    lines.append("  - Walk-Forward prevents lookahead bias (expanding window)")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("EV Threshold Optimizer - Boat Racing Model")
    print("=" * 70)

    # Load data
    df = load_data()

    # Run Walk-Forward with all thresholds
    all_bets = walk_forward_ev_optimization(df, n_folds=5)

    # Analyze
    print("\n\nAnalyzing results...")
    results = analyze_results(all_bets)
    optimal = find_optimal(results, min_bets=50)

    # Format report
    report = format_report(results, optimal)

    # Print summary
    print("\n" + report)

    # Save to file
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[SAVED] Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
