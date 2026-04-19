# ===========================================
# scripts/backtest_v038.py
# v0.38 マルチモデル バックテスト
#
# favorite / underdog モデルで 2024 通常レースを評価
# 既存 backtest.py の run_backtest_all_tickets と同じ枠組み
# data-leak なし (purchase = combo_prob のみ、return = actual_payout)
# ===========================================

import json
import pickle
import sys
import time
from itertools import permutations, combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "data"))
sys.path.insert(0, str(SCRIPT_DIR))

from feature_engine import FEATURE_NAMES, DB_PATH
from backtest import (
    TEST_START, TEST_END, MODEL_DIR,
    load_test_data, load_payout_data, compute_features,
    TICKET_TYPES, PATTERNS,
    _combo_prob_for_set, _enumerate_combos, _parse_combo_str,
)
import sqlite3

REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"


def _load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"]


def eval_model(model, label, is_midnight=False, test_year="2024"):
    """単一モデルで 2024 通常を全券種×全パターン評価"""
    # --- test_year 上書き ---
    import backtest as bt
    bt.TEST_START = f"{test_year}0101"
    bt.TEST_END = f"{test_year}1231"

    # 自前で全6券種込みの payout_df 取得
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
          AND r.race_date >= '{bt.TEST_START}'
          AND r.race_date <= '{bt.TEST_END}'
          AND res.rank = 1
          AND res.exacta_payout IS NOT NULL
    """, conn)
    conn.close()
    if len(payout_df) == 0:
        return None
    races_df, entries_df, _ = load_test_data(is_midnight, DB_PATH)
    if races_df is None:
        return None
    payout_ids = set(payout_df["race_id"])
    races_df = races_df[races_df["race_id"].isin(payout_ids)].reset_index(drop=True)
    entries_df = entries_df[entries_df["race_id"].isin(payout_ids)].reset_index(drop=True)

    features = compute_features(entries_df, races_df, DB_PATH)
    X = features[FEATURE_NAMES].fillna(0)
    # favorite/underdog モデルは predict_proba がシンプル (LGB単体)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # underdog/favorite は X を受け取る設計に注意
    features["pred_prob"] = probs

    # 払戻マップ
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = row["race_id"]
        wide_list = []
        try:
            if row["wide_payouts"]:
                wide_list = json.loads(row["wide_payouts"])
        except Exception:
            pass
        payout_map[rid] = {
            "exacta":   (row["exacta_combo"],   row["exacta_payout"]),
            "quinella": (row["quinella_combo"], row["quinella_payout"]),
            "trio":     (row["trio_combo"],     row["trio_payout"]),
            "trifecta": (row["trifecta_combo"], row["trifecta_payout"]),
            "wide":     wide_list,
        }

    # 集計 accumulator
    stats = {}
    for tkey, _, _, _ in TICKET_TYPES:
        for pname, (_, _, pth_list) in PATTERNS.items():
            for pth in pth_list:
                stats[(tkey, pname, pth)] = {
                    "n_bets": 0, "n_hits": 0,
                    "total_bet": 0, "total_return": 0,
                }

    n_races_processed = 0
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
        n_races_processed += 1

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

    return {"label": label, "n_races": n_races_processed, "stats": stats}


class LGBWrapper:
    """LGBMClassifier を受け取って、features DataFrame で predict_proba を返す薄ラッパー"""
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model
    def predict_proba(self, X):
        # lgb_model が sklearn-like なら predict_proba(values)[:,1]
        return self.lgb_model.predict_proba(X.values)[:, 1]


def wrap(model):
    """favorite/underdog モデルはそのまま .predict_proba(X) 動作"""
    return model


def summarize(result, label):
    print(f"\n--- {label} ---")
    stats = result["stats"]
    # 代表パターンだけ表示
    highlights = [
        ("trifecta", "A_本命", 0.20),
        ("trifecta", "B_中穴", 0.02),
        ("trifecta", "C_穴",   0.005),
        ("exacta",   "A_本命", 0.20),
        ("quinella", "B_中穴", 0.05),
        ("trio",     "C_穴",   0.005),
        ("wide",     "A_本命", 0.20),
    ]
    print(f"  {'ticket':>8} {'pattern':<10} {'prob_th':>8} "
          f"{'n_bets':>8} {'hit率':>7} {'ROI':>9}")
    for tkey, pname, pth in highlights:
        s = stats.get((tkey, pname, pth))
        if not s or s["n_bets"] == 0:
            continue
        roi = (s["total_return"] / s["total_bet"] - 1) * 100
        hr = s["n_hits"] / s["n_bets"] * 100
        print(f"  {tkey:>8} {pname:<10} {pth:>8.3f} "
              f"{s['n_bets']:>8,} {hr:>6.2f}% {roi:>+7.2f}%")


def main():
    t0 = time.time()
    fav_path = MODEL_DIR / "stage1_normal_favorite_v0.38.pkl"
    ud_path  = MODEL_DIR / "stage1_normal_underdog_v0.38.pkl"

    print("=" * 60)
    print("  v0.38 バックテスト (2024年, 通常レース)")
    print("=" * 60)

    fav = _load_model(fav_path)
    ud  = _load_model(ud_path)

    print("\n[FAVORITE モデル] 評価中...")
    r_fav = eval_model(wrap(fav), "favorite", is_midnight=False, test_year="2024")
    print(f"  対象レース: {r_fav['n_races']:,}")
    summarize(r_fav, "FAVORITE")

    print("\n[UNDERDOG モデル] 評価中...")
    r_ud = eval_model(wrap(ud), "underdog", is_midnight=False, test_year="2024")
    print(f"  対象レース: {r_ud['n_races']:,}")
    summarize(r_ud, "UNDERDOG")

    # JSON 比較
    def to_summary(r):
        out = {}
        for key, s in r["stats"].items():
            if s["n_bets"] == 0:
                continue
            tkey, pname, pth = key
            kstr = f"{tkey}__{pname}__{pth}"
            roi = (s["total_return"] / s["total_bet"] - 1) * 100
            hr = s["n_hits"] / s["n_bets"] * 100
            out[kstr] = {
                "n_bets": s["n_bets"], "n_hits": s["n_hits"],
                "hit_rate": round(hr, 2),
                "total_bet": s["total_bet"],
                "total_return": s["total_return"],
                "roi": round(roi, 2),
            }
        return out

    comparison = {
        "favorite": to_summary(r_fav),
        "underdog": to_summary(r_ud),
        "test_year": "2024",
        "n_races_favorite": r_fav["n_races"],
        "n_races_underdog": r_ud["n_races"],
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_path = REPORT_DIR / "v038_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")

    # 成功判定 (v1.0 と比較)
    # v1.0 参考値 (backtest_v0.34_final.log より通常):
    #   trifecta A_本命 prob=0.20: ROI -18.08%
    #   trifecta C_穴  prob=0.005: ROI -42.07%
    V10_BASE = {
        "trifecta__A_本命__0.2":   -18.08,
        "trifecta__C_穴__0.005":   -42.07,
    }
    print("\n=== v1.0 比較 (通常モデル, 2024) ===")
    for base_key, v10_roi in V10_BASE.items():
        fav_roi = comparison["favorite"].get(base_key, {}).get("roi")
        ud_roi  = comparison["underdog"].get(base_key, {}).get("roi")
        print(f"  {base_key}")
        print(f"    v1.0: {v10_roi:+7.2f}%")
        if fav_roi is not None:
            print(f"    fav : {fav_roi:+7.2f}% (差 {fav_roi-v10_roi:+.2f}pt)")
        if ud_roi is not None:
            print(f"    ud  : {ud_roi:+7.2f}% (差 {ud_roi-v10_roi:+.2f}pt)")

    elapsed = time.time() - t0
    print(f"\n所要時間: {elapsed/60:.1f}分")


if __name__ == "__main__":
    main()
