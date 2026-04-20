# ===========================================
# scripts/train_v041.py
# v0.41: 節リズム + ニッチ特徴量追加版 Stage1 訓練
#
# FEATURE_NAMES_V041 = FEATURE_NAMES (61) + K01..K04 (4) + L01..L03 (3) = 68
# ===========================================

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

from feature_engine import FEATURE_NAMES, DB_PATH, create_features
from prediction_model import purged_kfold_cv
from meet_rhythm import MeetRhythmFeatures, MEET_RHYTHM_FEATURE_NAMES
from niche_features import NicheFeatures, NICHE_FEATURE_NAMES
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

MODEL_DIR = PROJECT_DIR / "models"
PROGRESS_LOG = PROJECT_DIR.parent.parent / "data" / "keirin" / "v039_progress.log"

FEATURE_NAMES_V041 = FEATURE_NAMES + MEET_RHYTHM_FEATURE_NAMES + NICHE_FEATURE_NAMES


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def build_dataset(is_midnight, start, end):
    midnight_val = 1 if is_midnight else 0
    start_c = start.replace("-", "")
    end_c = end.replace("-", "")
    conn = sqlite3.connect(str(DB_PATH))
    races_df = pd.read_sql_query(f"""
        SELECT DISTINCT r.*
        FROM races r
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date BETWEEN '{start_c}' AND '{end_c}'
          AND NOT EXISTS (
            SELECT 1 FROM entries e
            WHERE e.race_id = r.race_id AND e.kyoso_tokuten IS NULL
          )
        ORDER BY r.race_date
    """, conn)
    entries_df = pd.read_sql_query(f"""
        SELECT e.* FROM entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date BETWEEN '{start_c}' AND '{end_c}'
          AND e.kyoso_tokuten IS NOT NULL
    """, conn)
    results_1st = pd.read_sql_query(f"""
        SELECT res.race_id, res.sha_ban FROM results res
        JOIN races r ON res.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          AND r.race_date BETWEEN '{start_c}' AND '{end_c}'
          AND res.rank = 1
    """, conn)
    conn.close()

    col_renames = {}
    if "sha_ban" in entries_df.columns:
        col_renames["sha_ban"] = "car_no"
    if "kyakushitsu" in entries_df.columns:
        col_renames["kyakushitsu"] = "style"
    if "ki_betsu" in entries_df.columns:
        col_renames["ki_betsu"] = "term"
    if "kyoso_tokuten" in entries_df.columns:
        col_renames["kyoso_tokuten"] = "grade_score"
    if "todofuken" in entries_df.columns:
        col_renames["todofuken"] = "prefecture"
    entries_df = entries_df.rename(columns=col_renames)
    for col, default in [("racer_class", None), ("win_rate", 0.0),
                         ("second_rate", 0.0), ("third_rate", 0.0),
                         ("racer_id", None), ("district", None)]:
        if col not in entries_df.columns:
            entries_df[col] = default

    t0 = time.time()
    features = create_features(
        entries_df=entries_df, races_df=races_df,
        odds_df=pd.DataFrame(), line_probs=None, bank_info=None,
        db_path=str(DB_PATH),
    )
    log(f"  基本特徴量: {time.time()-t0:.1f}秒, {len(features):,}行")

    # senshu_name を features に attach
    if "senshu_name" not in features.columns:
        name_map = {}
        for _, row in entries_df.iterrows():
            name_map[(row["race_id"], int(row["car_no"]))] = row.get("senshu_name")
        features["senshu_name"] = features.apply(
            lambda r: name_map.get((r["race_id"], int(r["car_no"]))), axis=1
        )
    date_map = dict(zip(races_df["race_id"], races_df["race_date"]))
    features["race_date"] = features["race_id"].map(date_map)

    # MeetRhythm
    t0 = time.time()
    mr = MeetRhythmFeatures(DB_PATH)
    n = mr.preload()
    log(f"  MeetRhythm preload: {n:,}, {time.time()-t0:.1f}秒")
    t0 = time.time()
    rows = [mr.get_for(r.get("senshu_name"), str(r.get("race_date", "")))
            for _, r in features.iterrows()]
    mr_df = pd.DataFrame(rows)
    for c in MEET_RHYTHM_FEATURE_NAMES:
        features[c] = mr_df[c].values
    log(f"  MeetRhythm 計算: {time.time()-t0:.1f}秒")

    # Niche
    t0 = time.time()
    nc = NicheFeatures(DB_PATH)
    n2 = nc.preload()
    log(f"  Niche preload: {n2:,}, {time.time()-t0:.1f}秒")
    t0 = time.time()
    rows = [nc.get_for(r.get("senshu_name"), str(r.get("race_date", "")))
            for _, r in features.iterrows()]
    nc_df = pd.DataFrame(rows)
    for c in NICHE_FEATURE_NAMES:
        features[c] = nc_df[c].values
    log(f"  Niche 計算: {time.time()-t0:.1f}秒")

    winners = set((r["race_id"], int(r["sha_ban"])) for _, r in results_1st.iterrows())
    features["__y1"] = features.apply(
        lambda r: 1 if (r["race_id"], int(r["car_no"])) in winners else 0, axis=1
    )
    X = features[FEATURE_NAMES_V041].fillna(0)
    y = features["__y1"]
    dates = features["race_date"]
    return X, y, dates


def train_one(is_midnight, start="2022-01-01", end="2023-12-31", suffix="v0.41"):
    label = "midnight" if is_midnight else "normal"
    log(f"\n=== v0.41 Stage1 {label} ({suffix}) 開始 ===")
    t0 = time.time()
    X, y, dates = build_dataset(is_midnight, start, end)
    log(f"  X shape: {X.shape}")

    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=63, max_depth=6,
        min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        n_estimators=500, verbose=-1, random_state=42, n_jobs=-1,
    )
    log("  Purged KFold CV (5分割)...")
    cv_aucs = []
    for fold, (tr, te) in enumerate(
        purged_kfold_cv(X, y, dates, n_splits=5, gap_days=7)
    ):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr].values, y.iloc[tr].values)
        pred = m.predict_proba(X.iloc[te].values)[:, 1]
        auc = roc_auc_score(y.iloc[te].values, pred)
        cv_aucs.append(auc)
        log(f"    Fold {fold+1}: AUC={auc:.4f}")
    cv_mean = float(np.mean(cv_aucs))
    log(f"  CV 平均: {cv_mean:.4f}")

    log("  最終学習...")
    final = lgb.LGBMClassifier(**params)
    final.fit(X.values, y.values)

    imp = sorted(zip(FEATURE_NAMES_V041, final.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    log(f"  TOP20 重要度:")
    for n, s in imp[:20]:
        mark = ""
        if n in MEET_RHYTHM_FEATURE_NAMES:
            mark = " [MR]"
        elif n in NICHE_FEATURE_NAMES:
            mark = " [NC]"
        log(f"    {n}: {s:.0f}{mark}")

    out = MODEL_DIR / f"stage1_{label}_{suffix}.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "model": final, "cv_mean_auc": cv_mean,
            "feature_names": FEATURE_NAMES_V041, "label": label,
            "train_start": start, "train_end": end,
        }, f)
    log(f"  保存: {out}")
    log(f"  所要時間: {(time.time()-t0)/60:.1f}分")
    return {"label": label, "cv_mean_auc": cv_mean, "importance": imp[:25]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_start", type=str, default="2022-01-01")
    parser.add_argument("--train_end", type=str, default="2023-12-31")
    parser.add_argument("--suffix", type=str, default="v0.41",
                        help="モデルファイル名サフィックス")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "normal", "midnight"])
    args = parser.parse_args()

    log("=" * 60)
    log(f"v0.41 Stage1 訓練: {args.train_start} 〜 {args.train_end} "
        f"suffix={args.suffix}")
    log("=" * 60)
    results = []
    targets = []
    if args.mode in ("both", "normal"):
        targets.append(False)
    if args.mode in ("both", "midnight"):
        targets.append(True)
    for is_mid in targets:
        r = train_one(is_mid, args.train_start, args.train_end,
                      suffix=args.suffix)
        if r:
            results.append(r)
    log(f"\n=== {args.suffix} 結果 ===")
    for r in results:
        log(f"  {r['label']}: CV AUC={r['cv_mean_auc']:.4f}")
    log(f"{args.suffix} 訓練完了")


if __name__ == "__main__":
    main()
