# ===========================================
# scripts/backtest_v045_quinella.py
# v0.45: ai_keirin 手法 — 2車複特化 value betting
#
# ai_keirin:
#   - 券種: 2車複
#   - EV閾値: 150%〜200% (1.5〜2.0)
#   - レース絞り込み: 1着予測値 × 2車複的中予測値
#   - 2023年 1年で回収率 117〜250% 達成
#
# 購入条件 (組合せ):
#   A  (model):    quinella_prob × real_odds > ev_threshold
#   B* (combo):    combo_score > min_combo_score
#                   combo_score = max(p1,p2) × quinella_prob
#   C  (reporter): combo の両車が gamboo 推奨ラインに含まれる
#   D  (market):   real_odds の popularity >= 5
#
# 4 軸グリッド:
#   ev_threshold:     [1.2, 1.5, 1.7, 2.0, 2.3, 2.5, 2.8, 3.0]  (8)
#   min_combo_score:  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]        (7)
#   use_reporter:     [True, False]                                 (2)
#   use_market:       [True, False]                                 (2)
#   → 8 × 7 × 2 × 2 = 224 パターン
#
# 各パターンを 4 OOS で検証 → 896 実験
#
# data-leak なし: Stage1 モデルは OOS 学習期間のみ、実オッズは購入前の公開情報
# ===========================================

import argparse
import json
import pickle
import re
import sys
import time
from itertools import combinations
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
PROGRESS_LOG = REPORT_DIR / "v045_progress.log"

EV_THRESHOLDS     = [1.2, 1.5, 1.7, 2.0, 2.3, 2.5, 2.8, 3.0]
MIN_COMBO_SCORES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
USE_REPORTER      = [False, True]
USE_MARKET        = [False, True]


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
        return {"set": set([honmei] + ni + san)}
    except (ValueError, IndexError):
        return None


def load_reporter_map():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("""
        SELECT race_id, predicted_line FROM reporter_predictions
        WHERE reporter_name='gamboo' AND predicted_line IS NOT NULL
    """, conn)
    conn.close()
    out = {}
    for _, row in df.iterrows():
        p = parse_reporter_line(row["predicted_line"])
        if p is not None:
            out[row["race_id"]] = p["set"]
    return out


def load_quinella_odds(race_ids):
    """race_id -> {pair: (odds, popularity)}  (2f = 2車複)"""
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join(["?"] * len(race_ids))
    df = pd.read_sql_query(
        f"""SELECT race_id, combo_key, odds, popularity
            FROM odds_netkeirin
            WHERE odds_type='2f' AND race_id IN ({placeholders})""",
        conn, params=list(race_ids),
    )
    conn.close()
    out = {}
    for _, row in df.iterrows():
        ck = str(row["combo_key"])
        if len(ck) != 4:
            continue
        try:
            a, b = int(ck[0:2]), int(ck[2:4])
            pair = tuple(sorted([a, b]))
            odds = float(row["odds"])
            pop  = int(row["popularity"] or 9999)
        except (ValueError, TypeError):
            continue
        out.setdefault(row["race_id"], {})[pair] = (odds, pop)
    return out


def load_actual_quinella(race_ids):
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join(["?"] * len(race_ids))
    df = pd.read_sql_query(
        f"""SELECT race_id, quinella_combo, quinella_payout
            FROM results
            WHERE rank=1 AND quinella_combo IS NOT NULL
              AND race_id IN ({placeholders})""",
        conn, params=list(race_ids),
    )
    conn.close()
    out = {}
    for _, row in df.iterrows():
        try:
            c = tuple(sorted(int(x) for x in str(row["quinella_combo"]).split("=")))
            if len(c) == 2:
                out[row["race_id"]] = (c, int(row["quinella_payout"]))
        except Exception:
            continue
    return out


def quinella_prob(p_a, p_b):
    """Stage1 のみで 2車複 joint prob を条件付き近似で計算"""
    d_a = 1.0 - p_a
    d_b = 1.0 - p_b
    if d_a <= 0 or d_b <= 0:
        return None
    return p_a * (p_b / d_a) + p_b * (p_a / d_b)


def run(model_suffix, test_year, bet_amount=100):
    log(f"\n{'='*60}")
    log(f"v0.45 quinella: model={model_suffix} test={test_year}")
    log(f"{'='*60}")

    bt.TEST_START = f"{test_year}0101"
    bt.TEST_END   = f"{test_year}1231"

    model_path = MODEL_DIR / f"stage1_normal_{model_suffix}.pkl"
    if not model_path.exists():
        log(f"モデルなし: {model_path}")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]

    # 対象レース (モデル+quinella オッズ+記者+払戻すべて揃い)
    conn = sqlite3.connect(str(DB_PATH))
    valid = pd.read_sql_query(f"""
        SELECT DISTINCT res.race_id FROM results res
        JOIN races r ON res.race_id=r.race_id
        JOIN (SELECT DISTINCT race_id FROM odds_netkeirin WHERE odds_type='2f') nk
             ON res.race_id=nk.race_id
        JOIN reporter_predictions rp
             ON res.race_id=rp.race_id AND rp.reporter_name='gamboo'
        WHERE res.rank=1 AND res.quinella_payout IS NOT NULL
          AND r.is_midnight=0
          AND r.race_date BETWEEN '{bt.TEST_START}' AND '{bt.TEST_END}'
    """, conn)["race_id"].tolist()
    conn.close()
    race_set = set(valid)
    log(f"  対象: {len(race_set):,}")
    if not race_set:
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

    log(f"  ロード中...")
    t0 = time.time()
    odds_map   = load_quinella_odds(list(race_set))
    rep_map    = load_reporter_map()
    actual_map = load_actual_quinella(list(race_set))
    log(f"  ロード完了 {time.time()-t0:.1f}秒")

    # stats accumulator: (ev, score, use_rep, use_mkt) -> {n_bets,n_hits,total_bet,total_return}
    stats = {}
    for ev in EV_THRESHOLDS:
        for sc in MIN_COMBO_SCORES:
            for ur in USE_REPORTER:
                for um in USE_MARKET:
                    stats[(ev, sc, ur, um)] = {
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
        if len(car_list) < 2:
            continue
        car_probs = dict(car_list)
        car_nos = [c for c, _ in car_list]

        r_odds = odds_map.get(race_id, {})
        r_act  = actual_map.get(race_id)
        r_rep  = rep_map.get(race_id)
        if not r_odds or r_act is None or r_rep is None:
            continue
        actual_pair, actual_payout = r_act
        n_races_eval += 1

        for pair in combinations(car_nos, 2):
            key = tuple(sorted(pair))
            if key not in r_odds:
                continue
            odds_val, pop = r_odds[key]
            a, b = key
            p_a = car_probs.get(a)
            p_b = car_probs.get(b)
            if p_a is None or p_b is None:
                continue
            qp = quinella_prob(p_a, p_b)
            if qp is None or qp <= 0:
                continue
            real_ev = qp * odds_val
            combo_score = max(p_a, p_b) * qp

            is_hit = (key == actual_pair)
            payout = actual_payout if is_hit else 0

            # reporter & market フラグ
            cond_reporter = all(c in r_rep for c in key)
            cond_market   = (pop >= 5)

            for ev in EV_THRESHOLDS:
                if real_ev <= ev:
                    continue
                for sc in MIN_COMBO_SCORES:
                    if combo_score < sc:
                        continue
                    for ur in USE_REPORTER:
                        if ur and not cond_reporter:
                            continue
                        for um in USE_MARKET:
                            if um and not cond_market:
                                continue
                            s = stats[(ev, sc, ur, um)]
                            s["n_bets"] += 1
                            s["total_bet"] += bet_amount
                            if is_hit:
                                s["n_hits"] += 1
                                s["total_return"] += payout

    log(f"  評価完了 {(time.time()-t0)/60:.1f}分, races={n_races_eval:,}")

    # 結果整形
    results = []
    for (ev, sc, ur, um), s in stats.items():
        if s["n_bets"] == 0:
            continue
        roi = (s["total_return"] / s["total_bet"] - 1) * 100
        hr  = s["n_hits"] / s["n_bets"] * 100
        hit_avg = (s["total_return"] / s["n_hits"] / bet_amount) if s["n_hits"] > 0 else 0
        results.append({
            "ev_th": ev, "min_score": sc,
            "use_reporter": ur, "use_market": um,
            "n_bets": s["n_bets"], "n_hits": s["n_hits"],
            "hit_rate": round(hr, 3),
            "hit_avg_odds": round(hit_avg, 2),
            "total_bet": s["total_bet"],
            "total_return": s["total_return"],
            "roi": round(roi, 3),
        })
    results.sort(key=lambda x: x["roi"], reverse=True)

    return {
        "model_suffix": model_suffix,
        "test_year": test_year,
        "n_races": n_races_eval,
        "results": results,
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
    out_path = REPORT_DIR / f"v045_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    log(f"\n保存: {out_path}")

    # TOP 10 ROI (n_bets >= 100 の条件付き)
    filtered = [r_ for r_ in r["results"] if r_["n_bets"] >= 100]
    log(f"\n=== {tag} TOP 10 ROI (n_bets>=100) ===")
    log(f"  {'ev':>4} {'score':>5} {'rep':>4} {'mkt':>4} "
        f"{'n_bets':>8} {'hit%':>6} {'avg':>8} {'ROI':>8}")
    for r_ in filtered[:10]:
        log(f"  {r_['ev_th']:>4.2f} {r_['min_score']:>5.2f} "
            f"{'Y' if r_['use_reporter'] else 'N':>4} "
            f"{'Y' if r_['use_market'] else 'N':>4} "
            f"{r_['n_bets']:>8,} {r_['hit_rate']:>5.2f}% "
            f"{r_['hit_avg_odds']:>6.1f}倍 {r_['roi']:>+7.2f}%")


if __name__ == "__main__":
    main()
