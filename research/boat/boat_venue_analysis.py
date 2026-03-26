# ===========================================
# boat_venue_analysis.py
# Paper Trade結果の会場別・レース別・ベット別パターン分析
#
# 分析項目:
#   1. 会場(venue)別の勝率・PF
#   2. レース番号(1R-12R)別の勝率・PF
#   3. bet_type別の勝率・PF
#   4. EV範囲別の勝率・PF
#   5. オッズ範囲別の勝率・PF
#   6. 不採算セグメントの特定
#   7. 枠番(lane)別の勝率・PF
# ===========================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "boat"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "venue_analysis_results.txt"


def load_paper_trade_log():
    """paper_trade_log.csv を読み込む"""
    csv_path = DATA_DIR / "paper_trade_log.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["is_win"] = (df["result"] == "WIN").astype(int)
    return df


def calc_metrics(df, group_col, label=""):
    """グループ別に勝率・PF・ROI等を計算"""
    results = []
    for name, grp in df.groupby(group_col):
        n = len(grp)
        wins = grp["is_win"].sum()
        win_rate = wins / n if n > 0 else 0
        total_bet = grp["bet_amount"].sum()
        total_payout = grp["payout"].sum()
        pf = total_payout / total_bet if total_bet > 0 else 0
        roi = (total_payout - total_bet) / total_bet * 100 if total_bet > 0 else 0
        avg_odds = grp["odds"].mean()
        avg_ev = grp["ev"].mean()
        total_pnl = grp["pnl"].sum()
        results.append({
            "group": name,
            "n_bets": n,
            "wins": wins,
            "win_rate": win_rate,
            "pf": pf,
            "roi": roi,
            "avg_odds": avg_odds,
            "avg_ev": avg_ev,
            "total_pnl": total_pnl,
        })
    return pd.DataFrame(results)


def analyze_by_ev_range(df):
    """EV範囲別の分析"""
    bins = [0, 1.3, 1.5, 2.0, 2.5, 3.0, 5.0, float("inf")]
    labels = ["<1.3", "1.3-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-5.0", "5.0+"]
    df = df.copy()
    df["ev_range"] = pd.cut(df["ev"], bins=bins, labels=labels, right=False)
    return calc_metrics(df, "ev_range", "EV Range")


def analyze_by_odds_range(df):
    """オッズ範囲別の分析"""
    bins = [0, 3, 5, 10, 20, 40, 60, 100, float("inf")]
    labels = ["<3", "3-5", "5-10", "10-20", "20-40", "40-60", "60-100", "100+"]
    df = df.copy()
    df["odds_range"] = pd.cut(df["odds"], bins=bins, labels=labels, right=False)
    return calc_metrics(df, "odds_range", "Odds Range")


def analyze_by_lane(df):
    """枠番別の分析"""
    return calc_metrics(df, "lane", "Lane")


def format_table(df, title):
    """テーブルを整形して文字列にする"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {title}")
    lines.append(f"{'='*70}")
    lines.append(f"{'Group':>12} {'N':>6} {'Wins':>5} {'WinRate':>8} {'PF':>7} {'ROI%':>8} {'AvgOdds':>8} {'AvgEV':>7} {'TotalPnL':>10}")
    lines.append("-" * 85)
    for _, row in df.iterrows():
        lines.append(
            f"{str(row['group']):>12} {row['n_bets']:>6} {row['wins']:>5} "
            f"{row['win_rate']:>7.1%} {row['pf']:>7.3f} {row['roi']:>7.1f}% "
            f"{row['avg_odds']:>8.1f} {row['avg_ev']:>7.2f} {row['total_pnl']:>10,.0f}"
        )
    return "\n".join(lines)


def identify_unprofitable_segments(df):
    """不採算セグメントを特定"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("  UNPROFITABLE SEGMENTS (PF < 1.0, n >= 10)")
    lines.append(f"{'='*70}")

    segments_found = []

    # Race number
    for race_no, grp in df.groupby("race_no"):
        n = len(grp)
        if n < 10:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        if pf < 1.0:
            segments_found.append(("Race", race_no, n, pf, grp["pnl"].sum()))

    # Lane
    for lane, grp in df.groupby("lane"):
        n = len(grp)
        if n < 10:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        if pf < 1.0:
            segments_found.append(("Lane", lane, n, pf, grp["pnl"].sum()))

    # EV ranges
    bins_ev = [0, 1.3, 1.5, 2.0, 2.5, 3.0, 5.0, float("inf")]
    labels_ev = ["<1.3", "1.3-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-5.0", "5.0+"]
    df_temp = df.copy()
    df_temp["ev_range"] = pd.cut(df_temp["ev"], bins=bins_ev, labels=labels_ev, right=False)
    for ev_range, grp in df_temp.groupby("ev_range", observed=True):
        n = len(grp)
        if n < 10:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        if pf < 1.0:
            segments_found.append(("EV_Range", ev_range, n, pf, grp["pnl"].sum()))

    # Odds ranges
    bins_odds = [0, 3, 5, 10, 20, 40, 60, 100, float("inf")]
    labels_odds = ["<3", "3-5", "5-10", "10-20", "20-40", "40-60", "60-100", "100+"]
    df_temp["odds_range"] = pd.cut(df_temp["odds"], bins=bins_odds, labels=labels_odds, right=False)
    for odds_range, grp in df_temp.groupby("odds_range", observed=True):
        n = len(grp)
        if n < 10:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        if pf < 1.0:
            segments_found.append(("Odds_Range", odds_range, n, pf, grp["pnl"].sum()))

    if segments_found:
        lines.append(f"{'Category':>12} {'Value':>12} {'N':>6} {'PF':>7} {'TotalPnL':>10}")
        lines.append("-" * 55)
        for cat, val, n, pf, pnl in sorted(segments_found, key=lambda x: x[3]):
            lines.append(f"{cat:>12} {str(val):>12} {n:>6} {pf:>7.3f} {pnl:>10,.0f}")
    else:
        lines.append("  No unprofitable segments found with n >= 10")

    return "\n".join(lines)


def generate_recommendations(df):
    """改善提案を生成"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("  RECOMMENDATIONS FOR boat_model.py")
    lines.append(f"{'='*70}")

    # 1. Check race numbers
    lines.append("\n--- Race Number Analysis ---")
    for race_no, grp in df.groupby("race_no"):
        n = len(grp)
        if n < 5:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        wr = grp["is_win"].mean()
        if pf < 0.8 and n >= 10:
            lines.append(f"  [FILTER] Race {race_no}: PF={pf:.3f}, WR={wr:.1%}, N={n} -> Consider blacklisting")
        elif pf >= 1.5:
            lines.append(f"  [GOOD]   Race {race_no}: PF={pf:.3f}, WR={wr:.1%}, N={n}")

    # 2. Check lanes
    lines.append("\n--- Lane Analysis ---")
    for lane, grp in df.groupby("lane"):
        n = len(grp)
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        wr = grp["is_win"].mean()
        if pf < 0.8:
            lines.append(f"  [FILTER] Lane {lane}: PF={pf:.3f}, WR={wr:.1%}, N={n} -> Increase EV threshold")
        elif pf >= 1.5:
            lines.append(f"  [GOOD]   Lane {lane}: PF={pf:.3f}, WR={wr:.1%}, N={n}")

    # 3. Check EV ranges
    lines.append("\n--- EV Range Analysis ---")
    bins_ev = [0, 1.3, 1.5, 2.0, 2.5, 3.0, 5.0, float("inf")]
    labels_ev = ["<1.3", "1.3-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-5.0", "5.0+"]
    df_temp = df.copy()
    df_temp["ev_range"] = pd.cut(df_temp["ev"], bins=bins_ev, labels=labels_ev, right=False)
    for ev_range, grp in df_temp.groupby("ev_range", observed=True):
        n = len(grp)
        if n < 5:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        wr = grp["is_win"].mean()
        if pf < 0.8:
            lines.append(f"  [FILTER] EV {ev_range}: PF={pf:.3f}, WR={wr:.1%}, N={n} -> Raise min EV threshold")
        elif pf >= 1.5:
            lines.append(f"  [GOOD]   EV {ev_range}: PF={pf:.3f}, WR={wr:.1%}, N={n}")

    # 4. Check odds ranges
    lines.append("\n--- Odds Range Analysis ---")
    bins_odds = [0, 3, 5, 10, 20, 40, 60, 100, float("inf")]
    labels_odds = ["<3", "3-5", "5-10", "10-20", "20-40", "40-60", "60-100", "100+"]
    df_temp["odds_range"] = pd.cut(df_temp["odds"], bins=bins_odds, labels=labels_odds, right=False)
    for odds_range, grp in df_temp.groupby("odds_range", observed=True):
        n = len(grp)
        if n < 5:
            continue
        pf = grp["payout"].sum() / grp["bet_amount"].sum()
        wr = grp["is_win"].mean()
        if pf < 0.8:
            lines.append(f"  [FILTER] Odds {odds_range}: PF={pf:.3f}, WR={wr:.1%}, N={n} -> Cap max odds")
        elif pf >= 1.5:
            lines.append(f"  [GOOD]   Odds {odds_range}: PF={pf:.3f}, WR={wr:.1%}, N={n}")

    # 5. Cross-analysis: low EV + high odds
    lines.append("\n--- Cross Analysis: Low EV + High Odds ---")
    mask_low_ev_high_odds = (df["ev"] < 1.5) & (df["odds"] > 20)
    subset = df[mask_low_ev_high_odds]
    if len(subset) >= 5:
        pf = subset["payout"].sum() / subset["bet_amount"].sum() if subset["bet_amount"].sum() > 0 else 0
        wr = subset["is_win"].mean()
        lines.append(f"  EV<1.5 & Odds>20: PF={pf:.3f}, WR={wr:.1%}, N={len(subset)}")
        if pf < 0.8:
            lines.append("  -> STRONG FILTER: Remove low-EV high-odds bets")

    # 6. Cross-analysis: EV < 1.3
    mask_very_low_ev = df["ev"] < 1.3
    subset = df[mask_very_low_ev]
    if len(subset) >= 5:
        pf = subset["payout"].sum() / subset["bet_amount"].sum() if subset["bet_amount"].sum() > 0 else 0
        wr = subset["is_win"].mean()
        lines.append(f"\n  EV < 1.3 overall: PF={pf:.3f}, WR={wr:.1%}, N={len(subset)}")
        if pf < 1.0:
            lines.append("  -> STRONG FILTER: Raise global min EV from current level")

    return "\n".join(lines)


def main():
    df = load_paper_trade_log()

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("  BOAT RACE PAPER TRADE - VENUE & PATTERN ANALYSIS")
    output_lines.append(f"  Generated: {pd.Timestamp.now()}")
    output_lines.append("=" * 70)

    # Overall stats
    total_bets = len(df)
    total_wins = df["is_win"].sum()
    total_bet_amount = df["bet_amount"].sum()
    total_payout = df["payout"].sum()
    overall_pf = total_payout / total_bet_amount
    overall_roi = (total_payout - total_bet_amount) / total_bet_amount * 100
    overall_wr = total_wins / total_bets

    output_lines.append(f"\n--- Overall Summary ---")
    output_lines.append(f"  Total Bets:    {total_bets}")
    output_lines.append(f"  Total Wins:    {total_wins}")
    output_lines.append(f"  Win Rate:      {overall_wr:.1%}")
    output_lines.append(f"  Total Wagered: {total_bet_amount:,.0f}")
    output_lines.append(f"  Total Payout:  {total_payout:,.0f}")
    output_lines.append(f"  Profit Factor: {overall_pf:.3f}")
    output_lines.append(f"  ROI:           {overall_roi:+.1f}%")
    output_lines.append(f"  Total P&L:     {df['pnl'].sum():+,.0f}")
    output_lines.append(f"  Venues:        {sorted(df['venue'].unique())}")
    output_lines.append(f"  Bet Types:     {sorted(df['bet_type'].unique())}")
    output_lines.append(f"  Date Range:    {df['date'].min()} - {df['date'].max()}")

    # 1. Venue analysis
    venue_stats = calc_metrics(df, "venue")
    output_lines.append(format_table(venue_stats, "1. VENUE ANALYSIS"))

    # 2. Race number analysis
    race_stats = calc_metrics(df, "race_no")
    output_lines.append(format_table(race_stats, "2. RACE NUMBER ANALYSIS (1R-12R)"))

    # 3. Bet type analysis
    bet_stats = calc_metrics(df, "bet_type")
    output_lines.append(format_table(bet_stats, "3. BET TYPE ANALYSIS"))

    # 4. EV range analysis
    ev_stats = analyze_by_ev_range(df)
    output_lines.append(format_table(ev_stats, "4. EV RANGE ANALYSIS"))

    # 5. Odds range analysis
    odds_stats = analyze_by_odds_range(df)
    output_lines.append(format_table(odds_stats, "5. ODDS RANGE ANALYSIS"))

    # 6. Lane analysis
    lane_stats = analyze_by_lane(df)
    output_lines.append(format_table(lane_stats, "6. LANE ANALYSIS"))

    # 7. Unprofitable segments
    output_lines.append(identify_unprofitable_segments(df))

    # 8. Recommendations
    output_lines.append(generate_recommendations(df))

    # Write results
    result_text = "\n".join(output_lines)
    print(result_text)
    OUTPUT_FILE.write_text(result_text, encoding="utf-8")
    print(f"\n\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
