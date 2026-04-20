# ===========================================
# scripts/backtest_v039.py
# v0.39: Stage1 + Stage2 併用 バックテスト
#
# 2024年テスト、通常レース対象（midnight も後段で追加可）。
# 3連単 combo 確率:
#   P(X,Y,Z) = p1[X] × p2[Y|X] × p3_approx[Z|X,Y]
# ここで p2 は Stage2 予測、p3 は残余均等近似
#
# data-leak なし:
#   - Stage1/Stage2 は 2022-2023 学習
#   - テストは 2024
#   - オッズ帯は odds_est = 0.75 / combo_prob で事前推定
# ===========================================

import json
import pickle
import sys
import time
from itertools import permutations, combinations
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
from stage2_model import Stage2Model, STAGE2_EXTRA_FEATURES
from backtest import (
    TEST_START, TEST_END, MODEL_DIR,
    load_test_data, compute_features,
    _parse_combo_str, TICKET_TYPES, PATTERNS,
)

PROGRESS_LOG = PROJECT_DIR.parent.parent / "data" / "keirin" / "v039_progress.log"
REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_stage1_model(label, suffix="v1.0"):
    p = MODEL_DIR / f"stage1_{label}_{suffix}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)["model"]


def load_stage2_model(label, suffix="v0.39"):
    p = MODEL_DIR / f"stage2_{label}_{suffix}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)["model"]


def build_stage2_inputs(group_df, stage1_probs, fixed_car):
    """
    1 レースで「X = fixed_car を 1着仮定」した場合の
    各 Y 候補に対する Stage2 入力行を構築。

    group_df: このレースの全エントリ DataFrame (FEATURE_NAMES + car_no + race_id 列)
    stage1_probs: {car_no: stage1_prob}
    fixed_car: 1着候補の車番

    Returns:
        df_s2: DataFrame (candidates only, excluding fixed_car)
        car_nos: list[int] (行順の候補 car_no)
    """
    if fixed_car not in stage1_probs:
        return None, None
    x_stage1 = float(stage1_probs[fixed_car])
    x_row = group_df[group_df["car_no"] == fixed_car]
    if len(x_row) == 0:
        return None, None
    x_row = x_row.iloc[0]
    x_grade = float(x_row.get("A02_grade_score", 0.0) or 0.0)
    x_elo = float(x_row.get("I04_elo_rating", 0.0) or 0.0)
    x_trend = float(x_row.get("I02_recent_trend_score", 0.0) or 0.0)

    candidates = group_df[group_df["car_no"] != fixed_car]
    rows = []
    car_nos = []
    for _, y in candidates.iterrows():
        y_car = int(y["car_no"])
        y_stage1 = float(stage1_probs.get(y_car, 0.0))
        feat = {f: float(y.get(f, 0.0) or 0.0) for f in FEATURE_NAMES}
        feat["stage1_prob_self"] = y_stage1
        feat["stage1_prob_fixed"] = x_stage1
        feat["delta_stage1_prob"] = y_stage1 - x_stage1
        feat["delta_grade_score"] = (
            float(y.get("A02_grade_score", 0.0) or 0.0) - x_grade
        )
        feat["delta_elo_rating"] = (
            float(y.get("I04_elo_rating", 0.0) or 0.0) - x_elo
        )
        feat["delta_recent_trend"] = (
            float(y.get("I02_recent_trend_score", 0.0) or 0.0) - x_trend
        )
        rows.append(feat)
        car_nos.append(y_car)
    if not rows:
        return None, None
    cols = FEATURE_NAMES + STAGE2_EXTRA_FEATURES
    return pd.DataFrame(rows, columns=cols), car_nos


def run(is_midnight=False, stage1_suffix="v1.0", stage2_suffix="v0.39"):
    label = "midnight" if is_midnight else "normal"
    label_ja = "ミッドナイト" if is_midnight else "通常"
    log(f"\n=== {label_ja} v0.39 backtest ===")
    s1 = load_stage1_model(label, stage1_suffix)
    s2 = load_stage2_model(label, stage2_suffix)
    if s1 is None or s2 is None:
        log(f"モデルなし: stage1={s1 is not None} stage2={s2 is not None}")
        return None

    # 払戻データ
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
    log(f"payout 済み: {len(payout_df):,} レース")

    races_df, entries_df, _ = load_test_data(is_midnight, DB_PATH)
    if races_df is None:
        return None
    pids = set(payout_df["race_id"])
    races_df = races_df[races_df["race_id"].isin(pids)].reset_index(drop=True)
    entries_df = entries_df[entries_df["race_id"].isin(pids)].reset_index(drop=True)
    if len(races_df) == 0:
        return None

    features = compute_features(entries_df, races_df, DB_PATH)
    if "race_date" not in features.columns:
        date_map = dict(zip(races_df["race_id"], races_df["race_date"]))
        features["race_date"] = features["race_id"].map(date_map)

    X_all = features[FEATURE_NAMES].fillna(0)
    features["__s1"] = s1.predict_proba(X_all)

    # payout map
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

    # 集計 accumulator: (ticket, pattern, prob_th)
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
        if len(group) < 3:
            continue
        pdata = payout_map[race_id]

        car_nos = [int(c) for c in group["car_no"]]
        stage1_dict = dict(zip(car_nos, group["__s1"].astype(float)))
        group_feat = group

        # Stage2 予測キャッシュ: {fixed_car: {candidate_car: p2}}
        s2_cache = {}
        for fixed in car_nos:
            df_s2, cands = build_stage2_inputs(group_feat, stage1_dict, fixed)
            if df_s2 is None or len(df_s2) == 0:
                s2_cache[fixed] = {}
                continue
            # normalize: 各候補の Stage2 prob を合計 1 になるよう正規化
            raw = s2.predict_proba(df_s2)
            total = raw.sum()
            if total > 0:
                norm = raw / total
            else:
                norm = raw
            s2_cache[fixed] = dict(zip(cands, norm))

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

            # combo 列挙
            enumerator = (permutations if ordered else combinations)
            for combo in enumerator(car_nos, n_cars):
                # combo_prob 計算 (Stage1 + Stage2 併用)
                if n_cars == 2:
                    c1, c2 = combo if ordered else combo
                    if ordered:
                        p1 = stage1_dict[c1]
                        p2 = s2_cache.get(c1, {}).get(c2, 0.0)
                        cprob = p1 * p2
                    else:
                        # unordered: 両方向を合算
                        a, b = combo
                        p_ab = stage1_dict[a] * s2_cache.get(a, {}).get(b, 0.0)
                        p_ba = stage1_dict[b] * s2_cache.get(b, {}).get(a, 0.0)
                        cprob = p_ab + p_ba
                elif n_cars == 3:
                    if ordered:
                        c1, c2, c3 = combo
                        p1 = stage1_dict[c1]
                        p2 = s2_cache.get(c1, {}).get(c2, 0.0)
                        # 3着 = 残余均等近似 (Stage1 prob re-normalized after X,Y)
                        remaining_s1_sum = sum(stage1_dict[c] for c in car_nos
                                               if c != c1 and c != c2)
                        if remaining_s1_sum > 0:
                            p3 = stage1_dict.get(c3, 0.0) / remaining_s1_sum
                        else:
                            p3 = 1.0 / max(1, len(car_nos) - 2)
                        cprob = p1 * p2 * p3
                    else:
                        # trio: 3! permutations 合算
                        total = 0.0
                        for perm in permutations(combo):
                            c1, c2, c3 = perm
                            p1 = stage1_dict[c1]
                            p2 = s2_cache.get(c1, {}).get(c2, 0.0)
                            remaining_s1_sum = sum(stage1_dict[c] for c in car_nos
                                                   if c != c1 and c != c2)
                            if remaining_s1_sum > 0:
                                p3 = stage1_dict.get(c3, 0.0) / remaining_s1_sum
                            else:
                                p3 = 0.0
                            total += p1 * p2 * p3
                        cprob = total
                else:
                    cprob = 0.0

                if cprob <= 0:
                    continue

                if ordered:
                    match_key = combo
                else:
                    match_key = tuple(sorted(combo))
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


def summarize_and_save(results):
    log("\n=== v0.39 結果サマリ (代表パターン) ===")
    highlights = [
        ("trifecta", "A_本命", 0.20),
        ("trifecta", "B_中穴", 0.02),
        ("trifecta", "C_穴",   0.005),
        ("exacta",   "A_本命", 0.20),
        ("exacta",   "B_中穴", 0.05),
        ("quinella", "B_中穴", 0.05),
        ("trio",     "C_穴",   0.005),
        ("wide",     "A_本命", 0.20),
    ]
    out_json = {}
    for r in results:
        lbl = r["label"]
        out_json[lbl] = {}
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
            out_json[lbl][key] = {
                "n_bets": s["n_bets"], "n_hits": s["n_hits"],
                "hit_rate": round(hr, 2), "roi": round(roi, 2),
                "total_bet": s["total_bet"],
                "total_return": s["total_return"],
            }
        out_json[lbl]["n_races"] = r["n_races"]

    out_path = REPORT_DIR / "v039_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    log(f"\n保存: {out_path}")


def main():
    log("=" * 60)
    log("v0.39 Stage1+Stage2 バックテスト (2024年)")
    log("=" * 60)
    results = []
    for is_mid in [False, True]:
        r = run(is_midnight=is_mid)
        if r:
            results.append(r)
    if results:
        summarize_and_save(results)
    log("v0.39 backtest 完了")


if __name__ == "__main__":
    main()
