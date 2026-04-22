# ===========================================
# scripts/backtest_v044_combined.py
# v0.44: モデル × 記者 × 市場 の組み合わせ条件 value betting
#
# 購入条件:
#   A (model):    combo_prob × real_odds > ev_threshold
#   B (reporter): combo の全車が gamboo 推奨ラインに含まれる
#   C (market):   real_odds の人気順位 >= 5 (市場人気下位)
#
# 4 パターン:
#   P1: A のみ (v0.43 相当)
#   P2: A + B
#   P3: A + C
#   P4: A + B + C
#
# データ-leak なし:
#   記者予想・市場オッズはレース前の公開情報
#   購入決定に使っても leak ではない
#
# 券種: trifecta (3連単) のみ、最も有望な 10-100 倍帯に絞る
# ===========================================

import argparse
import json
import pickle
import re
import sys
import time
from itertools import permutations
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
from backtest import load_test_data, compute_features
import backtest as bt

REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"
MODEL_DIR = PROJECT_DIR / "models"
PROGRESS_LOG = REPORT_DIR / "v044_progress.log"

# グリッド
EV_THRESHOLDS = [1.0, 1.05, 1.10, 1.20, 1.30, 1.50]
ODDS_BANDS = [
    (1, 100),   (5, 100),   (10, 100),
    (10, 500),  (10, 1000),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def parse_reporter_line(s):
    if not isinstance(s, str) or not s:
        return None
    parts = re.split(r"[-=]", s)
    try:
        nums_first = [int(x) for x in re.findall(r"\d+", parts[0])]
        if not nums_first:
            return None
        honmei = nums_first[0]
        ni = [int(x) for x in re.findall(r"\d+", parts[1])] if len(parts) > 1 else []
        san = [int(x) for x in re.findall(r"\d+", parts[2])] if len(parts) > 2 else []
        all_set = set([honmei] + ni + san)
        return {"honmei": honmei, "set": all_set}
    except (ValueError, IndexError):
        return None


def load_reporter_map():
    """race_id -> {'honmei': int, 'set': set}"""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("""
        SELECT race_id, predicted_line FROM reporter_predictions
        WHERE reporter_name = 'gamboo' AND predicted_line IS NOT NULL
    """, conn)
    conn.close()
    out = {}
    for _, row in df.iterrows():
        p = parse_reporter_line(row["predicted_line"])
        if p is not None:
            out[row["race_id"]] = p
    return out


def load_trifecta_odds(race_ids):
    """race_id -> {combo_tuple: (odds, popularity)}"""
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join(["?"] * len(race_ids))
    df = pd.read_sql_query(
        f"""SELECT race_id, combo_key, odds, popularity
            FROM odds_netkeirin
            WHERE odds_type='3t' AND race_id IN ({placeholders})""",
        conn, params=list(race_ids),
    )
    conn.close()
    out = {}
    for _, row in df.iterrows():
        ck = str(row["combo_key"])
        if len(ck) != 6:
            continue
        try:
            combo = (int(ck[0:2]), int(ck[2:4]), int(ck[4:6]))
            o = float(row["odds"])
            pop = int(row["popularity"] or 9999)
        except (ValueError, TypeError):
            continue
        out.setdefault(row["race_id"], {})[combo] = (o, pop)
    return out


def load_actual_trifecta(race_ids):
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join(["?"] * len(race_ids))
    df = pd.read_sql_query(
        f"""SELECT race_id, trifecta_combo, trifecta_payout
            FROM results
            WHERE rank=1 AND trifecta_combo IS NOT NULL
              AND race_id IN ({placeholders})""",
        conn, params=list(race_ids),
    )
    conn.close()
    out = {}
    for _, row in df.iterrows():
        try:
            c = tuple(int(x) for x in str(row["trifecta_combo"]).split("-"))
            if len(c) == 3:
                out[row["race_id"]] = (c, int(row["trifecta_payout"]))
        except Exception:
            continue
    return out


def combo_prob_ordered(car_probs, combo):
    ps = [car_probs.get(c) for c in combo]
    if any(p is None for p in ps):
        return None
    p1, p2, p3 = ps
    d2 = 1 - p1
    d3 = 1 - p1 - p2
    if d2 <= 0 or d3 <= 0:
        return None
    return p1 * (p2 / d2) * (p3 / d3)


def run(model_suffix, test_year, bet_amount=100):
    log(f"\n{'='*60}")
    log(f"v0.44 model={model_suffix} test={test_year}")
    log(f"{'='*60}")

    bt.TEST_START = f"{test_year}0101"
    bt.TEST_END = f"{test_year}1231"

    # モデル
    model_path = MODEL_DIR / f"stage1_normal_{model_suffix}.pkl"
    if not model_path.exists():
        log(f"モデルなし: {model_path}")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]

    # 対象レース
    conn = sqlite3.connect(str(DB_PATH))
    valid_ids = pd.read_sql_query(f"""
        SELECT DISTINCT res.race_id FROM results res
        JOIN races r ON res.race_id=r.race_id
        JOIN (SELECT DISTINCT race_id FROM odds_netkeirin WHERE odds_type='3t') nk
             ON res.race_id=nk.race_id
        JOIN reporter_predictions rp
             ON res.race_id=rp.race_id AND rp.reporter_name='gamboo'
        WHERE res.rank=1 AND res.trifecta_payout IS NOT NULL
          AND r.is_midnight=0
          AND r.race_date BETWEEN '{bt.TEST_START}' AND '{bt.TEST_END}'
    """, conn)["race_id"].tolist()
    conn.close()
    race_set = set(valid_ids)
    log(f"  対象 (モデル+オッズ+記者+payout 揃い): {len(race_set):,}")
    if len(race_set) == 0:
        return None

    races_df, entries_df, _ = load_test_data(False, DB_PATH)
    races_df = races_df[races_df["race_id"].isin(race_set)].reset_index(drop=True)
    entries_df = entries_df[entries_df["race_id"].isin(race_set)].reset_index(drop=True)

    log(f"  特徴量計算中...")
    t0 = time.time()
    features = compute_features(entries_df, races_df, DB_PATH)
    X = features[FEATURE_NAMES].fillna(0)
    features["pred_prob"] = model.predict_proba(X)
    log(f"  予測完了 {time.time()-t0:.1f}秒")

    log(f"  ロード中: odds/reporter/actual...")
    t0 = time.time()
    odds_map = load_trifecta_odds(list(race_set))
    reporter_map = load_reporter_map()
    actual_map = load_actual_trifecta(list(race_set))
    log(f"  ロード完了 {time.time()-t0:.1f}秒")

    # stats: {(pattern, ev_th, min_odds, max_odds): {n_bets, n_hits, total_bet, total_return}}
    stats = {}
    patterns = ["A", "AB", "AC", "ABC"]
    for p in patterns:
        for ev_th in EV_THRESHOLDS:
            for mn, mx in ODDS_BANDS:
                stats[(p, ev_th, mn, mx)] = {
                    "n_bets": 0, "n_hits": 0,
                    "total_bet": 0, "total_return": 0,
                }

    log(f"  評価中...")
    t0 = time.time()
    n_races_eval = 0
    for race_id, group in features.groupby("race_id"):
        if race_id not in race_set:
            continue
        car_list = [(int(r["car_no"]), float(r["pred_prob"]))
                    for _, r in group.iterrows()]
        car_nos = [c for c, _ in car_list]
        car_probs = dict(car_list)
        if len(car_list) < 3:
            continue

        r_odds = odds_map.get(race_id, {})
        r_rep = reporter_map.get(race_id)
        r_act = actual_map.get(race_id)
        if not r_odds or r_rep is None or r_act is None:
            continue
        actual_combo, actual_payout = r_act
        reporter_set = r_rep["set"]

        n_races_eval += 1

        for combo in permutations(car_nos, 3):
            if combo not in r_odds:
                continue
            odds_val, pop = r_odds[combo]
            cprob = combo_prob_ordered(car_probs, combo)
            if cprob is None or cprob <= 0:
                continue

            real_ev = cprob * odds_val
            is_hit = (combo == actual_combo)
            payout_val = actual_payout if is_hit else 0

            # B 条件: 全 car が reporter_set 内
            cond_B = all(c in reporter_set for c in combo)
            # C 条件: 市場人気 5 位以下 (オッズ popularity >= 5)
            cond_C = (pop >= 5)

            for p in patterns:
                # B/C 条件
                if "B" in p and not cond_B:
                    continue
                if "C" in p and not cond_C:
                    continue
                for ev_th in EV_THRESHOLDS:
                    if real_ev <= ev_th:
                        continue
                    for mn, mx in ODDS_BANDS:
                        if odds_val < mn or odds_val >= mx:
                            continue
                        s = stats[(p, ev_th, mn, mx)]
                        s["n_bets"] += 1
                        s["total_bet"] += bet_amount
                        if is_hit:
                            s["n_hits"] += 1
                            s["total_return"] += payout_val

    log(f"  評価完了 {(time.time()-t0)/60:.1f}分, races={n_races_eval:,}")

    # 結果整形
    results_by_pattern = {}
    for (p, ev_th, mn, mx), s in stats.items():
        if s["n_bets"] == 0:
            continue
        roi = (s["total_return"] / s["total_bet"] - 1) * 100
        hr = s["n_hits"] / s["n_bets"] * 100
        hit_avg = (s["total_return"] / s["n_hits"] / bet_amount) if s["n_hits"] > 0 else 0
        results_by_pattern.setdefault(p, []).append({
            "ev_th": ev_th, "min_odds": mn, "max_odds": mx,
            "n_bets": s["n_bets"], "n_hits": s["n_hits"],
            "hit_rate": round(hr, 3),
            "hit_avg_odds": round(hit_avg, 2),
            "total_bet": s["total_bet"],
            "total_return": s["total_return"],
            "roi": round(roi, 3),
        })
    for p in results_by_pattern:
        results_by_pattern[p].sort(key=lambda x: x["roi"], reverse=True)

    return {
        "model_suffix": model_suffix,
        "test_year": test_year,
        "n_races": n_races_eval,
        "patterns": results_by_pattern,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_suffix", type=str, default="v1.0")
    parser.add_argument("--test_year", type=str, default="2024")
    parser.add_argument("--out_suffix", type=str, default=None)
    args = parser.parse_args()

    r = run(args.model_suffix, args.test_year)
    if r is None:
        log("失敗")
        return

    tag = args.out_suffix or f"{args.model_suffix}_on_{args.test_year}"
    out_path = REPORT_DIR / f"v044_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    log(f"保存: {out_path}")

    # TOP 3 per pattern
    log(f"\n=== {tag} TOP 3 ROI (各パターン) ===")
    for p in ["A", "AB", "AC", "ABC"]:
        rs = r["patterns"].get(p, [])
        if not rs:
            log(f"  {p}: データなし")
            continue
        log(f"\n  [パターン {p}]")
        log(f"    {'ev':>5} {'odds':>14} {'n_bets':>8} {'hit%':>6} {'hit_avg':>8} {'ROI':>8}")
        for r_ in rs[:3]:
            band = f"[{r_['min_odds']},{r_['max_odds']})"
            log(f"    {r_['ev_th']:>5.2f} {band:>14} "
                f"{r_['n_bets']:>8,} {r_['hit_rate']:>5.2f}% "
                f"{r_['hit_avg_odds']:>6.1f}倍 {r_['roi']:>+7.2f}%")


if __name__ == "__main__":
    main()
