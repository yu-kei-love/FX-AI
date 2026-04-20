# ===========================================
# scripts/train_v040.py
# v0.40: 節リズム 4 特徴量追加版 Stage1 訓練
#
# FEATURE_NAMES_V040 = FEATURE_NAMES (61) + K01..K04 (4) = 65 feat
# 既存 models/stage1_*_v1.0 を壊さず、stage1_*_v0.40.pkl を生成
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
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

MODEL_DIR = PROJECT_DIR / "models"
PROGRESS_LOG = PROJECT_DIR.parent.parent / "data" / "keirin" / "v039_progress.log"

FEATURE_NAMES_V040 = FEATURE_NAMES + MEET_RHYTHM_FEATURE_NAMES


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

    # rename
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
    # still keep senshu_name for meet_rhythm
    # note: col "senshu_name" in entries_df should already exist

    for col, default in [("racer_class", None), ("win_rate", 0.0),
                         ("second_rate", 0.0), ("third_rate", 0.0),
                         ("racer_id", None), ("district", None)]:
        if col not in entries_df.columns:
            entries_df[col] = default

    # 既存特徴量
    t0 = time.time()
    features = create_features(
        entries_df=entries_df, races_df=races_df,
        odds_df=pd.DataFrame(), line_probs=None, bank_info=None,
        db_path=str(DB_PATH),
    )
    log(f"  基本特徴量: {time.time()-t0:.1f}秒, {len(features):,}行")

    # 節リズム
    t0 = time.time()
    mr = MeetRhythmFeatures(DB_PATH)
    n = mr.preload()
    log(f"  MeetRhythm preload: {n:,} 選手, {time.time()-t0:.1f}秒")
    t0 = time.time()
    # entries_df の行順と features の行順は一致する前提（create_features は
    # entries_df をそのまま使用、race_id/car_no 保持）
    # senshu_name を features に attach
    if "senshu_name" not in features.columns:
        name_map = {}
        for _, row in entries_df.iterrows():
            name_map[(row["race_id"], int(row["car_no"]))] = row.get("senshu_name")
        features["senshu_name"] = features.apply(
            lambda r: name_map.get((r["race_id"], int(r["car_no"]))), axis=1
        )
    # K01..K04 計算
    date_map = dict(zip(races_df["race_id"], races_df["race_date"]))
    features["race_date"] = features["race_id"].map(date_map)
    rhythm_rows = []
    for _, row in features.iterrows():
        v = mr.get_for(row.get("senshu_name"), str(row.get("race_date", "")))
        rhythm_rows.append(v)
    rhythm_df = pd.DataFrame(rhythm_rows)
    for col in MEET_RHYTHM_FEATURE_NAMES:
        features[col] = rhythm_df[col].values
    log(f"  MeetRhythm 計算: {time.time()-t0:.1f}秒")

    # target
    winners = set((r["race_id"], int(r["sha_ban"])) for _, r in results_1st.iterrows())
    features["__y1"] = features.apply(
        lambda r: 1 if (r["race_id"], int(r["car_no"])) in winners else 0, axis=1
    )
    X = features[FEATURE_NAMES_V040].fillna(0)
    y = features["__y1"]
    dates = features["race_date"]
    return X, y, dates, features


def train_one(is_midnight, start="2022-01-01", end="2023-12-31"):
    label = "midnight" if is_midnight else "normal"
    log(f"\n=== v0.40 Stage1 {label} 開始 ===")
    t0 = time.time()
    X, y, dates, features = build_dataset(is_midnight, start, end)
    log(f"  X shape: {X.shape}, 1率={y.mean():.4f}")

    # LGB 単体学習 (v1.0 と同じデフォルト相当)
    # 注: Stage1Model (ensemble) は FEATURE_NAMES 固定なので、
    # v0.40 ではまず LGB 単体で検証。効果あれば stacking 実装を別段階で。
    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=63, max_depth=6,
        min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        n_estimators=500, verbose=-1, random_state=42, n_jobs=-1,
    )

    log(f"  Purged KFold CV (5分割)...")
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

    log(f"  最終学習 (全データ)...")
    final = lgb.LGBMClassifier(**params)
    final.fit(X.values, y.values)

    # 重要度
    imp = sorted(zip(FEATURE_NAMES_V040, final.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    log(f"  TOP15 重要度:")
    for n, s in imp[:15]:
        mark = " *" if n in MEET_RHYTHM_FEATURE_NAMES else ""
        log(f"    {n}: {s:.0f}{mark}")

    out = MODEL_DIR / f"stage1_{label}_v0.40.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "model": final,
            "cv_mean_auc": cv_mean,
            "feature_names": FEATURE_NAMES_V040,
            "label": label,
            "train_start": start, "train_end": end,
        }, f)
    log(f"  保存: {out}")
    log(f"  所要時間: {(time.time()-t0)/60:.1f}分")
    return {"label": label, "cv_mean_auc": cv_mean,
            "importance": imp[:20]}


def main():
    log("=" * 60)
    log("v0.40 節リズム4特徴量 Stage1 訓練")
    log("=" * 60)
    log(f"FEATURE_NAMES_V040: {len(FEATURE_NAMES_V040)} features")
    results = []
    for is_mid in [False, True]:
        r = train_one(is_mid)
        if r:
            results.append(r)
    log("\n=== v0.40 結果 ===")
    for r in results:
        log(f"  {r['label']}: CV AUC={r['cv_mean_auc']:.4f}")
    # 節リズム特徴量の重要度
    log("\n=== 節リズム重要度 (normal) ===")
    for r in results:
        if r["label"] != "normal":
            continue
        for n, s in r["importance"]:
            if n in MEET_RHYTHM_FEATURE_NAMES:
                log(f"  {n}: {s}")
    log("v0.40 訓練完了")


if __name__ == "__main__":
    main()
