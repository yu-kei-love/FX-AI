# ===========================================
# scripts/backtest_value_real.py
# v0.43: 真の value betting バックテスト
#
# netkeirin 実オッズ (odds_netkeirin) を使用した value betting:
#   real_EV = combo_prob × real_odds
#   購入条件: real_EV > ev_threshold
#             min_odds <= real_odds < max_odds
#
# data-leak なし:
#   - モデルは学習期間 (train_start〜train_end) のみで学習
#   - テスト期間の予測は model.predict_proba(X) のみ
#   - real_odds は買い目選定に使用、actual_payout は return のみ
#
# グリッドスキャン:
#   EV閾値: [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
#   min_odds: [1, 5, 10, 20, 50]
#   max_odds: [100, 500, 1000, 99999]
#   券種: trifecta / trio / exacta / quinella / wide
# ===========================================

import argparse
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
from backtest import (
    load_test_data, compute_features,
)
import backtest as bt

REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"
MODEL_DIR = PROJECT_DIR / "models"
PROGRESS_LOG = REPORT_DIR / "v043_progress.log"

# 券種定義 (odds_type, label_ja, n_cars, ordered)
# combo_key フォーマット: 2車 '0102', 3車 '010203' (2 桁ずつ)
TICKET_SPECS = [
    ("3t",   "trifecta", 3, True),
    ("trio", "trio",     3, False),
    ("2t",   "exacta",   2, True),
    ("2f",   "quinella", 2, False),
    ("wide", "wide",     2, False),
]

# グリッド
EV_THRESHOLDS = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
ODDS_BANDS = [
    (1, 100),    (1, 500),    (1, 1000),   (1, 99999),
    (5, 100),    (5, 500),    (5, 1000),
    (10, 100),   (10, 500),   (10, 1000),  (10, 99999),
    (20, 500),   (20, 1000),
    (50, 1000),  (50, 99999),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def parse_combo_key(ck, n_cars):
    """'0102' / '010203' → (1,2) / (1,2,3) tuple"""
    if not isinstance(ck, str):
        return None
    try:
        return tuple(int(ck[i:i+2]) for i in range(0, n_cars*2, 2))
    except (ValueError, IndexError):
        return None


def combo_prob_ordered(car_probs, combo):
    """条件付き近似: p(A=1, B=2, C=3) ≒ pA × pB/(1-pA) × pC/(1-pA-pB)"""
    ps = [car_probs.get(c) for c in combo]
    if any(p is None for p in ps):
        return None
    if len(ps) == 2:
        p1, p2 = ps
        d = 1 - p1
        if d <= 0:
            return None
        return p1 * (p2 / d)
    elif len(ps) == 3:
        p1, p2, p3 = ps
        d2 = 1 - p1
        d3 = 1 - p1 - p2
        if d2 <= 0 or d3 <= 0:
            return None
        return p1 * (p2 / d2) * (p3 / d3)
    return None


def combo_prob_unordered(car_probs, combo):
    """順不同: 全順列の確率和"""
    total = 0.0
    ok = False
    for perm in permutations(combo):
        p = combo_prob_ordered(car_probs, perm)
        if p is not None:
            total += p
            ok = True
    return total if ok else None


def load_model(label, suffix):
    path = MODEL_DIR / f"stage1_{label}_{suffix}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["model"]


def load_real_odds(race_ids, odds_type):
    """odds_netkeirin から対象レース・指定券種の odds を取得"""
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join(["?"] * len(race_ids))
    df = pd.read_sql_query(
        f"""SELECT race_id, combo_key, odds
            FROM odds_netkeirin
            WHERE odds_type = ?
              AND race_id IN ({placeholders})""",
        conn, params=[odds_type] + list(race_ids),
    )
    conn.close()
    # {race_id: {combo_tuple: odds}}
    out = {}
    # n_cars 推定: 最初の行で decide
    for _, row in df.iterrows():
        ck = row["combo_key"]
        # 3t/trio は 6桁、2t/2f/wide は 4桁
        n_cars = 3 if odds_type in ("3t", "trio") else 2
        combo = parse_combo_key(ck, n_cars)
        if combo is None:
            continue
        try:
            o = float(row["odds"])
        except (ValueError, TypeError):
            continue
        ordered = odds_type in ("3t", "2t")
        key = combo if ordered else tuple(sorted(combo))
        out.setdefault(row["race_id"], {})[key] = o
    return out


def load_actual_payouts(race_ids):
    """results から actual combo/payout 取得 (全券種)"""
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join(["?"] * len(race_ids))
    df = pd.read_sql_query(
        f"""SELECT race_id,
                   trifecta_combo, trifecta_payout,
                   trio_combo, trio_payout,
                   exacta_combo, exacta_payout,
                   quinella_combo, quinella_payout,
                   wide_payouts
            FROM results
            WHERE rank = 1 AND race_id IN ({placeholders})""",
        conn, params=list(race_ids),
    )
    conn.close()
    out = {}
    for _, row in df.iterrows():
        rid = row["race_id"]
        entry = {}
        # trifecta
        if pd.notna(row["trifecta_combo"]) and pd.notna(row["trifecta_payout"]):
            try:
                c = tuple(int(x) for x in str(row["trifecta_combo"]).split("-"))
                if len(c) == 3:
                    entry["trifecta"] = {c: int(row["trifecta_payout"])}
            except Exception:
                pass
        # trio
        if pd.notna(row["trio_combo"]) and pd.notna(row["trio_payout"]):
            try:
                c = tuple(sorted(int(x) for x in str(row["trio_combo"]).split("=")))
                if len(c) == 3:
                    entry["trio"] = {c: int(row["trio_payout"])}
            except Exception:
                pass
        # exacta
        if pd.notna(row["exacta_combo"]) and pd.notna(row["exacta_payout"]):
            try:
                c = tuple(int(x) for x in str(row["exacta_combo"]).split("-"))
                if len(c) == 2:
                    entry["exacta"] = {c: int(row["exacta_payout"])}
            except Exception:
                pass
        # quinella
        if pd.notna(row["quinella_combo"]) and pd.notna(row["quinella_payout"]):
            try:
                c = tuple(sorted(int(x) for x in str(row["quinella_combo"]).split("=")))
                if len(c) == 2:
                    entry["quinella"] = {c: int(row["quinella_payout"])}
            except Exception:
                pass
        # wide (3ペア)
        if pd.notna(row["wide_payouts"]) and row["wide_payouts"]:
            try:
                wl = json.loads(row["wide_payouts"])
                wide_hits = {}
                for w in wl:
                    c_str = w.get("combo")
                    p_val = w.get("payout")
                    if c_str and p_val is not None:
                        c = tuple(sorted(int(x) for x in str(c_str).split("=")))
                        if len(c) == 2:
                            wide_hits[c] = int(p_val)
                entry["wide"] = wide_hits
            except Exception:
                pass
        out[rid] = entry
    return out


def run_value_bet(model_suffix, test_year, bet_amount=100):
    """指定モデル・テスト年で value betting grid scan"""
    log(f"\n{'='*60}")
    log(f"value betting: model={model_suffix}, test={test_year}")
    log(f"{'='*60}")

    # test期間設定
    bt.TEST_START = f"{test_year}0101"
    bt.TEST_END = f"{test_year}1231"

    # 通常レースのみ (midnight は数が少ない)
    model = load_model("normal", model_suffix)
    if model is None:
        log(f"  モデルなし: stage1_normal_{model_suffix}.pkl")
        return None

    races_df, entries_df, _ = load_test_data(False, DB_PATH)
    if races_df is None:
        return None
    # payout あり & netkeirin あり の交集合
    conn = sqlite3.connect(str(DB_PATH))
    valid_ids = pd.read_sql_query(f"""
        SELECT DISTINCT res.race_id FROM results res
        JOIN races r ON res.race_id=r.race_id
        JOIN (SELECT DISTINCT race_id FROM odds_netkeirin WHERE odds_type='3t') nk
             ON res.race_id=nk.race_id
        WHERE res.rank=1 AND res.exacta_payout IS NOT NULL
          AND r.is_midnight=0
          AND r.race_date BETWEEN '{bt.TEST_START}' AND '{bt.TEST_END}'
    """, conn)["race_id"].tolist()
    conn.close()
    race_set = set(valid_ids)
    races_df = races_df[races_df["race_id"].isin(race_set)].reset_index(drop=True)
    entries_df = entries_df[entries_df["race_id"].isin(race_set)].reset_index(drop=True)
    log(f"  対象レース: {len(races_df):,}")
    if len(races_df) == 0:
        return None

    # 特徴量計算・予測
    log(f"  特徴量計算中...")
    t0 = time.time()
    features = compute_features(entries_df, races_df, DB_PATH)
    X = features[FEATURE_NAMES].fillna(0)
    features["pred_prob"] = model.predict_proba(X)
    log(f"  予測完了 {time.time()-t0:.1f}秒")

    # 実オッズ & actual payout ロード
    log(f"  実オッズロード中...")
    t0 = time.time()
    odds_by_type = {}
    for ot, tkey, n_cars, ordered in TICKET_SPECS:
        odds_by_type[tkey] = load_real_odds(list(race_set), ot)
        log(f"    {tkey}: {len(odds_by_type[tkey]):,} races")
    actual_payouts = load_actual_payouts(list(race_set))
    log(f"  ロード完了 {time.time()-t0:.1f}秒")

    # accumulator: (ticket, ev_th, min_odds, max_odds) -> stats
    stats = {}
    for _, tkey, _, _ in TICKET_SPECS:
        for ev_th in EV_THRESHOLDS:
            for mn, mx in ODDS_BANDS:
                stats[(tkey, ev_th, mn, mx)] = {
                    "n_bets": 0, "n_hits": 0,
                    "total_bet": 0, "total_return": 0,
                }

    # レースごとに評価
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
        n_races_eval += 1
        ap = actual_payouts.get(race_id, {})

        for ot, tkey, n_cars, ordered in TICKET_SPECS:
            odds_map = odds_by_type.get(tkey, {}).get(race_id, {})
            if not odds_map:
                continue
            actual_hits = ap.get(tkey, {})  # {combo: payout} (wide は 3 pairs)

            # combo 列挙
            enum = permutations(car_nos, n_cars) if ordered else \
                   combinations(car_nos, n_cars)
            for combo in enum:
                key = combo if ordered else tuple(sorted(combo))
                real_odds = odds_map.get(key)
                if real_odds is None:
                    continue
                if ordered:
                    cprob = combo_prob_ordered(car_probs, combo)
                else:
                    cprob = combo_prob_unordered(car_probs, combo)
                if cprob is None or cprob <= 0:
                    continue
                real_ev = cprob * real_odds

                is_hit = key in actual_hits
                payout_val = actual_hits.get(key, 0)

                # 全グリッドで評価
                for ev_th in EV_THRESHOLDS:
                    if real_ev <= ev_th:
                        continue
                    for mn, mx in ODDS_BANDS:
                        if real_odds < mn or real_odds >= mx:
                            continue
                        s = stats[(tkey, ev_th, mn, mx)]
                        s["n_bets"] += 1
                        s["total_bet"] += bet_amount
                        if is_hit:
                            s["n_hits"] += 1
                            s["total_return"] += payout_val

    log(f"  評価完了 {(time.time()-t0)/60:.1f}分, races={n_races_eval:,}")

    # 結果集計
    results_list = []
    for (tkey, ev_th, mn, mx), s in stats.items():
        if s["n_bets"] == 0:
            continue
        roi = (s["total_return"] / s["total_bet"] - 1) * 100
        hr = s["n_hits"] / s["n_bets"] * 100
        hit_avg_odds = (s["total_return"] / s["n_hits"] / bet_amount) if s["n_hits"] > 0 else 0
        results_list.append({
            "ticket":   tkey,
            "ev_th":    ev_th,
            "min_odds": mn,
            "max_odds": mx,
            "n_bets":   s["n_bets"],
            "n_hits":   s["n_hits"],
            "hit_rate": round(hr, 3),
            "hit_avg_odds": round(hit_avg_odds, 2),
            "total_bet":    s["total_bet"],
            "total_return": s["total_return"],
            "roi":          round(roi, 3),
        })

    # ROI 降順
    results_list.sort(key=lambda x: x["roi"], reverse=True)

    return {
        "model_suffix": model_suffix,
        "test_year":    test_year,
        "n_races":      n_races_eval,
        "results":      results_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_year", type=str, default="2024")
    parser.add_argument("--model_suffix", type=str, default="v1.0")
    parser.add_argument("--out_suffix", type=str, default=None)
    args = parser.parse_args()

    r = run_value_bet(args.model_suffix, args.test_year)
    if r is None:
        log("実行失敗")
        return

    # 保存
    tag = args.out_suffix or f"{args.model_suffix}_on_{args.test_year}"
    out_path = REPORT_DIR / f"value_real_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    log(f"\n保存: {out_path}")

    # TOP10 ROI 表示
    log(f"\n=== TOP 10 (ROI 降順, {args.model_suffix} on {args.test_year}) ===")
    log(f"  {'ticket':>8} {'ev':>5} {'odds':>12} {'n_bets':>8} {'hit%':>6} {'hit_avg':>8} {'ROI':>8}")
    for r in r["results"][:10]:
        band = f"[{r['min_odds']},{r['max_odds']})"
        log(f"  {r['ticket']:>8} {r['ev_th']:>5.2f} {band:>12} "
            f"{r['n_bets']:>8,} {r['hit_rate']:>5.2f}% "
            f"{r['hit_avg_odds']:>6.1f}倍 {r['roi']:>+7.2f}%")


if __name__ == "__main__":
    main()
