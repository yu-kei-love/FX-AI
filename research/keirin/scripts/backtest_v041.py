# ===========================================
# scripts/backtest_v041.py
# v0.41: 節リズム + ニッチ特徴量追加版 Stage1 (LGB単体) バックテスト
# FEATURE_NAMES_V041 = FEATURE_NAMES (61) + K01-K04 (4) + L01-L03 (3) = 68
# ===========================================

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "data"))
sys.path.insert(0, str(SCRIPT_DIR))

from feature_engine import FEATURE_NAMES, DB_PATH
from meet_rhythm import MeetRhythmFeatures, MEET_RHYTHM_FEATURE_NAMES
from niche_features import NicheFeatures, NICHE_FEATURE_NAMES
from backtest import (
    TEST_START, TEST_END, MODEL_DIR,
    load_test_data, compute_features,
    _parse_combo_str, TICKET_TYPES, PATTERNS,
    _enumerate_combos, _combo_prob_for_set,
)

PROGRESS_LOG = PROJECT_DIR.parent.parent / "data" / "keirin" / "v039_progress.log"
REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"

FEATURE_NAMES_V041 = FEATURE_NAMES + MEET_RHYTHM_FEATURE_NAMES + NICHE_FEATURE_NAMES


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run(is_midnight=False, suffix="v0.41"):
    label = "midnight" if is_midnight else "normal"
    label_ja = "ミッドナイト" if is_midnight else "通常"
    log(f"\n=== {suffix} {label_ja} backtest ===")

    model_path = MODEL_DIR / f"stage1_{label}_{suffix}.pkl"
    if not model_path.exists():
        log(f"モデルなし: {model_path}")
        return None
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]

    midnight_val = 1 if is_midnight else 0
    conn = sqlite3.connect(str(DB_PATH))
    payout_df = pd.read_sql_query(f"""
        SELECT res.race_id,
               res.exacta_combo, res.exacta_payout,
               res.quinella_combo, res.quinella_payout,
               res.trio_combo, res.trio_payout,
               res.trifecta_combo, res.trifecta_payout,
               res.wide_payouts
        FROM results res
        JOIN races r ON res.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date >= '{TEST_START}'
          AND r.race_date <= '{TEST_END}'
          AND res.rank = 1 AND res.exacta_payout IS NOT NULL
    """, conn)
    conn.close()
    log(f"payout: {len(payout_df):,}")

    races_df, entries_df, _ = load_test_data(is_midnight, DB_PATH)
    if races_df is None:
        return None
    pids = set(payout_df["race_id"])
    races_df = races_df[races_df["race_id"].isin(pids)].reset_index(drop=True)
    entries_df = entries_df[entries_df["race_id"].isin(pids)].reset_index(drop=True)
    if len(races_df) == 0:
        return None

    features = compute_features(entries_df, races_df, DB_PATH)

    if "senshu_name" not in features.columns:
        name_map = {}
        sha_col = "car_no" if "car_no" in entries_df.columns else "sha_ban"
        for _, row in entries_df.iterrows():
            try:
                name_map[(row["race_id"], int(row[sha_col]))] = row.get("senshu_name")
            except (ValueError, TypeError):
                continue
        features["senshu_name"] = features.apply(
            lambda r: name_map.get((r["race_id"], int(r["car_no"]))), axis=1
        )
    if "race_date" not in features.columns:
        date_map = dict(zip(races_df["race_id"], races_df["race_date"]))
        features["race_date"] = features["race_id"].map(date_map)

    log(f"  MeetRhythm preload...")
    mr = MeetRhythmFeatures(DB_PATH)
    mr.preload()
    rows = [mr.get_for(r.get("senshu_name"), str(r.get("race_date", "")))
            for _, r in features.iterrows()]
    mr_df = pd.DataFrame(rows)
    for c in MEET_RHYTHM_FEATURE_NAMES:
        features[c] = mr_df[c].values

    log(f"  Niche preload...")
    nc = NicheFeatures(DB_PATH)
    nc.preload()
    rows = [nc.get_for(r.get("senshu_name"), str(r.get("race_date", "")))
            for _, r in features.iterrows()]
    nc_df = pd.DataFrame(rows)
    for c in NICHE_FEATURE_NAMES:
        features[c] = nc_df[c].values

    X = features[FEATURE_NAMES_V041].fillna(0).values
    features["pred_prob"] = model.predict_proba(X)[:, 1]

    payout_map = {}
    for _, row in payout_df.iterrows():
        wide_list = []
        try:
            if row["wide_payouts"]:
                wide_list = json.loads(row["wide_payouts"])
        except Exception:
            pass
        payout_map[row["race_id"]] = {
            "exacta":   (row["exacta_combo"],   row["exacta_payout"]),
            "quinella": (row["quinella_combo"], row["quinella_payout"]),
            "trio":     (row["trio_combo"],     row["trio_payout"]),
            "trifecta": (row["trifecta_combo"], row["trifecta_payout"]),
            "wide":     wide_list,
        }

    stats = {}
    for tkey, _, _, _ in TICKET_TYPES:
        for pname, (_, _, pth_list) in PATTERNS.items():
            for pth in pth_list:
                stats[(tkey, pname, pth)] = {
                    "n_bets": 0, "n_hits": 0,
                    "total_bet": 0, "total_return": 0,
                }

    n_races = 0
    t0 = time.time()
    for race_id, group in features.groupby("race_id"):
        if race_id not in payout_map:
            continue
        pdata = payout_map[race_id]
        car_list = [(int(r["car_no"]), float(r["pred_prob"]))
                    for _, r in group.iterrows()]
        car_nos = [c for c, _ in car_list]
        car_probs = dict(car_list)
        if len(car_list) < 3:
            continue
        n_races += 1

        for tkey, _, n_cars, ordered in TICKET_TYPES:
            if tkey == "wide":
                wide_hits = pdata["wide"]
                hit_combos = {}
                for w in wide_hits:
                    c = _parse_combo_str(w.get("combo"), ordered=False)
                    if c is not None:
                        hit_combos[tuple(sorted(c))] = int(w.get("payout") or 0)
            else:
                combo_str, payout = pdata[tkey]
                actual_tuple = _parse_combo_str(combo_str, ordered)
                if actual_tuple is None or payout is None or pd.isna(payout):
                    continue
                if ordered:
                    hit_combos = {actual_tuple: int(payout)}
                else:
                    hit_combos = {tuple(sorted(actual_tuple)): int(payout)}
            if not hit_combos:
                continue

            for combo in _enumerate_combos(car_nos, n_cars, ordered):
                cprob = _combo_prob_for_set(car_probs, combo, ordered)
                if cprob is None or cprob <= 0:
                    continue
                match_key = combo if ordered else tuple(sorted(combo))
                is_hit = match_key in hit_combos
                payout_val = hit_combos[match_key] if is_hit else 0

                for pname, (min_odds, max_odds, pth_list) in PATTERNS.items():
                    for pth in pth_list:
                        if cprob <= pth:
                            continue
                        odds_est = 0.75 / cprob if cprob > 0 else 9999
                        if odds_est < min_odds or odds_est >= max_odds:
                            continue
                        s = stats[(tkey, pname, pth)]
                        s["n_bets"] += 1
                        s["total_bet"] += 100
                        if is_hit:
                            s["n_hits"] += 1
                            s["total_return"] += payout_val

    log(f"評価完了: {(time.time()-t0)/60:.1f}分, races={n_races:,}")
    return {"label": label, "n_races": n_races, "stats": stats}


def summarize(results):
    log("\n=== v0.41 結果サマリ ===")
    highlights = [
        ("trifecta", "A_本命", 0.20),
        ("trifecta", "B_中穴", 0.02),
        ("trifecta", "C_穴",   0.005),
        ("exacta",   "A_本命", 0.20),
        ("quinella", "B_中穴", 0.05),
        ("trio",     "C_穴",   0.005),
        ("wide",     "A_本命", 0.20),
    ]
    out = {}
    for r in results:
        lbl = r["label"]
        out[lbl] = {}
        log(f"\n-- {lbl} (n={r['n_races']:,}) --")
        log(f"  {'ticket':>8} {'pattern':<10} {'pth':>7} "
            f"{'n':>8} {'hit%':>6} {'ROI':>8}")
        for tk, pn, pt in highlights:
            s = r["stats"].get((tk, pn, pt))
            if not s or s["n_bets"] == 0:
                continue
            roi = (s["total_return"] / s["total_bet"] - 1) * 100
            hr = s["n_hits"] / s["n_bets"] * 100
            log(f"  {tk:>8} {pn:<10} {pt:>7.3f} {s['n_bets']:>8,} "
                f"{hr:>5.2f}% {roi:>+7.2f}%")
            key = f"{tk}__{pn}__{pt}"
            out[lbl][key] = {
                "n_bets": s["n_bets"], "n_hits": s["n_hits"],
                "hit_rate": round(hr, 2), "roi": round(roi, 2),
                "total_bet": s["total_bet"],
                "total_return": s["total_return"],
            }
        out[lbl]["n_races"] = r["n_races"]
    out_path = REPORT_DIR / "v041_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"保存: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_year", type=str, default=None,
                        help="YYYY を指定するとその年をテスト")
    parser.add_argument("--model_suffix", type=str, default="v0.41")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "normal", "midnight"])
    parser.add_argument("--out_suffix", type=str, default=None,
                        help="出力 JSON のサフィックス")
    args = parser.parse_args()

    # test year override
    if args.test_year:
        import backtest as bt
        bt.TEST_START = f"{args.test_year}0101"
        bt.TEST_END = f"{args.test_year}1231"
        # モジュール変数更新後、このモジュールの import も更新
        global TEST_START, TEST_END
        TEST_START = bt.TEST_START
        TEST_END = bt.TEST_END

    log("=" * 60)
    log(f"backtest: suffix={args.model_suffix} "
        f"test={args.test_year or '2024'}")
    log("=" * 60)
    results = []
    targets = []
    if args.mode in ("both", "normal"):
        targets.append(False)
    if args.mode in ("both", "midnight"):
        targets.append(True)
    for is_mid in targets:
        r = run(is_midnight=is_mid, suffix=args.model_suffix)
        if r:
            results.append(r)
    if results:
        # summarize で出力 JSON path を切り替え
        out_tag = args.out_suffix or (
            f"{args.model_suffix}_on_{args.test_year or '2024'}"
        )
        _save_with_tag(results, out_tag)
    log(f"backtest 完了: {args.model_suffix} on {args.test_year or '2024'}")


def _save_with_tag(results, tag):
    log(f"\n=== 結果サマリ ({tag}) ===")
    highlights = [
        ("trifecta", "A_本命", 0.20),
        ("trifecta", "B_中穴", 0.02),
        ("trifecta", "C_穴",   0.005),
        ("exacta",   "A_本命", 0.20),
        ("quinella", "B_中穴", 0.05),
        ("trio",     "C_穴",   0.005),
        ("wide",     "A_本命", 0.20),
    ]
    out = {}
    for r in results:
        lbl = r["label"]
        out[lbl] = {}
        log(f"\n-- {lbl} (n={r['n_races']:,}) --")
        log(f"  {'ticket':>8} {'pattern':<10} {'pth':>7} "
            f"{'n':>8} {'hit%':>6} {'ROI':>8}")
        for tk, pn, pt in highlights:
            s = r["stats"].get((tk, pn, pt))
            if not s or s["n_bets"] == 0:
                continue
            roi = (s["total_return"] / s["total_bet"] - 1) * 100
            hr = s["n_hits"] / s["n_bets"] * 100
            log(f"  {tk:>8} {pn:<10} {pt:>7.3f} {s['n_bets']:>8,} "
                f"{hr:>5.2f}% {roi:>+7.2f}%")
            key = f"{tk}__{pn}__{pt}"
            out[lbl][key] = {
                "n_bets": s["n_bets"], "n_hits": s["n_hits"],
                "hit_rate": round(hr, 2), "roi": round(roi, 2),
                "total_bet": s["total_bet"],
                "total_return": s["total_return"],
            }
        out[lbl]["n_races"] = r["n_races"]
    out_path = REPORT_DIR / f"backtest_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"保存: {out_path}")


if __name__ == "__main__":
    main()
