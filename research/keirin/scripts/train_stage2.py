# ===========================================
# scripts/train_stage2.py
# v0.39: Stage2 モデル訓練
#
# 手順:
#   1. load_training_data で X,y,dates,race_ids 取得
#   2. Stage1 モデルを全データで学習し in-sample prob を取得
#   3. 各レースの actual 1着(X) と 2着(Y=actual 2着, N=not 2着) を集計
#   4. (X, Y or N) ペアでデータ拡張 → Stage2 学習データ
#   5. Purged K-Fold CV で AUC 評価
#   6. 全データで最終学習 → pkl 保存
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

from feature_engine import FEATURE_NAMES, DB_PATH
from prediction_model import Stage1Model, purged_kfold_cv
from stage2_model import Stage2Model, STAGE2_EXTRA_FEATURES
from train import load_training_data
from sklearn.metrics import roc_auc_score

MODEL_DIR = PROJECT_DIR / "models"
PROGRESS_LOG = PROJECT_DIR.parent.parent / "data" / "keirin" / "v039_progress.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_2nd_place_map(is_midnight, db_path, race_ids):
    """race_id → (1着 car_no, 2着 car_no) を取得"""
    conn = sqlite3.connect(str(db_path))
    midnight_val = 1 if is_midnight else 0
    # rank=1 と rank=2 を結合
    q_ids = ",".join(f"'{r}'" for r in race_ids)
    df = pd.read_sql_query(f"""
        SELECT r1.race_id,
               r1.sha_ban AS sha_1,
               r2.sha_ban AS sha_2
        FROM results r1
        JOIN results r2 ON r1.race_id = r2.race_id
        JOIN races rr ON r1.race_id = rr.race_id
        WHERE r1.rank = 1 AND r2.rank = 2
          AND rr.is_midnight = {midnight_val}
          AND r1.race_id IN ({q_ids})
    """, conn)
    conn.close()
    mp = {}
    for _, row in df.iterrows():
        try:
            mp[row["race_id"]] = (int(row["sha_1"]), int(row["sha_2"]))
        except (ValueError, TypeError):
            continue
    return mp


def build_stage2_dataset(X_df, y_1st, race_ids, dates, stage1_probs, top2_map):
    """
    Stage2 学習データセットを構築。

    各レースで (actual 1着 X, 候補 Y) ペアを作成:
      X_feat = Y の 61 feature + stage2 extra features
      y_feat = 1 if Y == actual 2着 else 0

    Returns:
        df_s2: DataFrame (Stage2 入力)
        y_s2:  Series (target)
        race_ids_s2: Series
        dates_s2:    Series
    """
    # df 化
    df = X_df.copy()
    df["__race_id"] = race_ids.values
    df["__date"] = dates.values
    df["__car_no"] = range(len(df))  # placeholder, overwrite below
    df["__stage1"] = stage1_probs

    # race_id でグループ化し、各レースの car_no を付与
    # load_training_data の features には car_no が含まれているので取得
    # load_training_data は features["car_no"] を返すが FEATURE_NAMES には入ってない
    # ここでは race_ids から順序で car_no 取得できない。別途 DB から取得必要。
    # 代わりに top2_map から sha_1, sha_2 を引き、「各レースに何番の車が含まれるか」を DB から
    # 取得する手もあるが、面倒なので feature_engine の結果にある car_no を残す形に変更

    return df  # placeholder; actual construction below


def make_s2_rows_from_features(features_df, stage1_probs, top2_map):
    """
    features_df (train の create_features 結果 = car_no 列含む + FEATURE_NAMES)
    + Stage1 probs (同じ行順)
    + top2_map: race_id → (sha_1, sha_2)
    から、Stage2 学習行を組み立てる。

    各レース: actual 1着 X と、残り N-1 の候補 Y
      features = Y の feature
      extra:
        stage1_prob_self  = stage1[Y]
        stage1_prob_fixed = stage1[X]
        delta_stage1_prob = stage1[Y] - stage1[X]
        delta_grade_score = Y.A02_grade_score - X.A02_grade_score
        delta_elo_rating  = Y.I04_elo_rating - X.I04_elo_rating
        delta_recent_trend = Y.I02_recent_trend_score - X.I02_recent_trend_score
      target = 1 if Y == sha_2 else 0

    Returns:
        pd.DataFrame (Stage2 入力), target Series, race_id Series, date Series
    """
    df = features_df.copy()
    df["__stage1"] = stage1_probs

    rows = []
    targets = []
    rids = []
    dates_out = []

    for race_id, g in df.groupby("race_id", sort=False):
        if race_id not in top2_map:
            continue
        sha1, sha2 = top2_map[race_id]

        # X = actual 1着 の特徴
        X_rows = g[g["car_no"] == sha1]
        if len(X_rows) == 0:
            continue
        x_row = X_rows.iloc[0]
        x_stage1 = float(x_row["__stage1"])
        x_grade = float(x_row.get("A02_grade_score", 0.0) or 0.0)
        x_elo = float(x_row.get("I04_elo_rating", 0.0) or 0.0)
        x_trend = float(x_row.get("I02_recent_trend_score", 0.0) or 0.0)

        race_date = str(x_row.get("race_date", ""))

        for _, y_row in g.iterrows():
            y_car = int(y_row["car_no"])
            if y_car == sha1:
                continue  # 自分 = 1着 → スキップ
            feat_vec = {f: float(y_row.get(f, 0.0) or 0.0) for f in FEATURE_NAMES}
            y_stage1 = float(y_row["__stage1"])
            feat_vec["stage1_prob_self"] = y_stage1
            feat_vec["stage1_prob_fixed"] = x_stage1
            feat_vec["delta_stage1_prob"] = y_stage1 - x_stage1
            feat_vec["delta_grade_score"] = (
                float(y_row.get("A02_grade_score", 0.0) or 0.0) - x_grade
            )
            feat_vec["delta_elo_rating"] = (
                float(y_row.get("I04_elo_rating", 0.0) or 0.0) - x_elo
            )
            feat_vec["delta_recent_trend"] = (
                float(y_row.get("I02_recent_trend_score", 0.0) or 0.0) - x_trend
            )
            rows.append(feat_vec)
            targets.append(1 if y_car == sha2 else 0)
            rids.append(race_id)
            dates_out.append(race_date)

    if not rows:
        return None, None, None, None
    cols = FEATURE_NAMES + STAGE2_EXTRA_FEATURES
    s2_df = pd.DataFrame(rows, columns=cols)
    return s2_df, pd.Series(targets), pd.Series(rids), pd.Series(dates_out)


def train_one(is_midnight, start, end):
    label = "midnight" if is_midnight else "normal"
    log(f"=== Stage2 {label} 開始 ===")
    t0 = time.time()

    # 1. 基本データ（features は全列、race_id/car_no/race_date 必須）
    from train import load_training_data as _lt
    from backtest import compute_features, load_test_data
    # load_training_data はやや違う形 (X が FEATURE_NAMES のみ)
    # ここでは feature 全列を取るため直接実装
    from feature_engine import create_features as _create_features
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
    conn.close()
    log(f"[{label}] races={len(races_df):,} entries={len(entries_df):,}")

    # rename columns
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

    t0f = time.time()
    features = _create_features(
        entries_df=entries_df, races_df=races_df,
        odds_df=pd.DataFrame(), line_probs=None, bank_info=None,
        db_path=str(DB_PATH),
    )
    log(f"[{label}] 特徴量計算: {time.time()-t0f:.1f}秒, {len(features):,}行")

    # race_date を features に attach
    if "race_date" not in features.columns:
        date_map = dict(zip(races_df["race_id"], races_df["race_date"]))
        features["race_date"] = features["race_id"].map(date_map)

    # 2. Stage1 学習 + in-sample prediction
    # Stage1Model は stacking なので tuned_label 使う
    X_s1 = features[FEATURE_NAMES].fillna(0)
    # target 1着
    conn = sqlite3.connect(str(DB_PATH))
    win_df = pd.read_sql_query(f"""
        SELECT r.race_id, r.sha_ban FROM results r
        JOIN races rr ON r.race_id = rr.race_id
        WHERE r.rank = 1 AND rr.is_midnight = {midnight_val}
          AND rr.race_date BETWEEN '{start_c}' AND '{end_c}'
    """, conn)
    conn.close()
    winners = set((row["race_id"], int(row["sha_ban"])) for _, row in win_df.iterrows())
    features["__y1"] = features.apply(
        lambda r: 1 if (r["race_id"], int(r["car_no"])) in winners else 0, axis=1
    )
    y_s1 = features["__y1"]

    log(f"[{label}] Stage1 in-sample 学習...")
    tuned = label
    s1 = Stage1Model(tuned_label=tuned)
    s1.fit(X_s1, y_s1)
    stage1_probs = s1.predict_proba(X_s1)
    features["__stage1"] = stage1_probs
    log(f"[{label}] Stage1 学習完了")

    # 3. top2_map 取得
    conn = sqlite3.connect(str(DB_PATH))
    top2_df = pd.read_sql_query(f"""
        SELECT r1.race_id, r1.sha_ban AS sha_1, r2.sha_ban AS sha_2
        FROM results r1
        JOIN results r2 ON r1.race_id = r2.race_id
        JOIN races rr ON r1.race_id = rr.race_id
        WHERE r1.rank = 1 AND r2.rank = 2
          AND rr.is_midnight = {midnight_val}
          AND rr.race_date BETWEEN '{start_c}' AND '{end_c}'
    """, conn)
    conn.close()
    top2_map = {row["race_id"]: (int(row["sha_1"]), int(row["sha_2"]))
                for _, row in top2_df.iterrows()}
    log(f"[{label}] top2 pairs: {len(top2_map):,}")

    # 4. Stage2 入力データ構築
    t0b = time.time()
    X_s2, y_s2, rid_s2, date_s2 = make_s2_rows_from_features(
        features, stage1_probs, top2_map
    )
    if X_s2 is None:
        log(f"[{label}] Stage2 行 構築失敗（データなし）")
        return None
    log(f"[{label}] Stage2 データ: {len(X_s2):,} 行, 1率={y_s2.mean():.4f}, "
        f"構築 {time.time()-t0b:.1f}秒")

    # 5. Purged K-Fold CV
    log(f"[{label}] Stage2 Purged K-Fold CV (3分割)...")
    # date_s2 は str YYYYMMDD なので dates Series として渡す
    # purged_kfold_cv は dates: pd.Series (日付str or datetime) を期待
    cv_aucs = []
    try:
        for fold, (tr_idx, te_idx) in enumerate(
            purged_kfold_cv(X_s2, y_s2, date_s2, n_splits=3, gap_days=7)
        ):
            m = Stage2Model()
            m.fit(X_s2.iloc[tr_idx], y_s2.iloc[tr_idx])
            pred = m.predict_proba(X_s2.iloc[te_idx])
            auc = roc_auc_score(y_s2.iloc[te_idx], pred)
            cv_aucs.append(auc)
            log(f"  Fold {fold+1}: AUC={auc:.4f}")
    except Exception as e:
        log(f"  CV エラー: {e}")
    cv_mean = float(np.mean(cv_aucs)) if cv_aucs else 0.0

    # 6. 全データで最終学習
    log(f"[{label}] Stage2 最終学習...")
    final = Stage2Model()
    final.fit(X_s2, y_s2)

    imp = final.get_feature_importance()
    log(f"  TOP10 重要度:")
    for fn, sc in imp[:10]:
        log(f"    {fn}: {sc:.0f}")

    out = MODEL_DIR / f"stage2_{label}_v0.39.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "model": final,
            "cv_mean_auc": cv_mean,
            "feature_names": X_s2.columns.tolist(),
            "train_start": start, "train_end": end,
            "label": label,
        }, f)
    log(f"保存: {out}")
    log(f"[{label}] 所要時間: {(time.time()-t0)/60:.1f}分")
    return {"label": label, "cv_mean_auc": cv_mean,
            "n_train": len(X_s2), "importance": imp[:20]}


def main():
    log("=" * 60)
    log("v0.39 Stage2 モデル訓練")
    log("=" * 60)
    results = []
    for is_mid in [False, True]:
        r = train_one(is_mid, "2022-01-01", "2023-12-31")
        if r:
            results.append(r)
    log("\n=== v0.39 Stage2 訓練サマリ ===")
    for r in results:
        log(f"  {r['label']}: CV AUC={r['cv_mean_auc']:.4f} (n={r['n_train']:,})")
    log("v0.39 Stage2 訓練完了")


if __name__ == "__main__":
    main()
