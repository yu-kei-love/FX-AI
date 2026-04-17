# ===========================================
# scripts/train.py
# 競輪AI - Stage1 学習スクリプト
#
# 通常レースとミッドナイトレースで2モデルを学習する
# Purged K-Fold CV で時系列安全な評価を行う
#
# 使い方:
#   cd research/keirin
#   python scripts/train.py
#
# 注意：合成データで評価に使用しないこと（CLAUDE.md準拠）
# ===========================================

import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# パス設定
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "data"))

from feature_engine import create_features, FEATURE_NAMES, DB_PATH
from prediction_model import Stage1Model, purged_kfold_cv

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_training_data(is_midnight: bool, db_path: Path = DB_PATH,
                       train_start: str = None, train_end: str = None):
    """
    学習用データを読み込み、特徴量とターゲットを生成する。

    Parameters:
        is_midnight: True→ミッドナイト、False→通常レース
        db_path: SQLiteのパス
        train_start: 学習データ開始日 "YYYY-MM-DD"（省略時は全期間）
        train_end:   学習データ終了日 "YYYY-MM-DD"（省略時は全期間）

    Returns:
        X: 特徴量DataFrame (1行=1選手)
        y: ターゲットSeries (1着=1, その他=0)
        dates: 日付Series (Purged K-Fold用)
        race_ids: race_id Series
    """
    label = "ミッドナイト" if is_midnight else "通常"
    midnight_val = 1 if is_midnight else 0

    # 日付範囲を YYYYMMDD 形式に（races.race_date の格納形式）
    start_compact = train_start.replace("-", "") if train_start else None
    end_compact = train_end.replace("-", "") if train_end else None

    date_clauses = []
    if start_compact:
        date_clauses.append(f"r.race_date >= '{start_compact}'")
    if end_compact:
        date_clauses.append(f"r.race_date <= '{end_compact}'")
    date_filter = (" AND " + " AND ".join(date_clauses)) if date_clauses else ""

    conn = sqlite3.connect(str(db_path))

    # kyoso_tokuten が補完済みのレースのみ対象
    # （補完されていない選手がいるレースは除外）
    print(f"[{label}] データ読み込み中 (期間: {train_start or '-'} 〜 {train_end or '-'})...")
    races_df = pd.read_sql_query(f"""
        SELECT DISTINCT r.*
        FROM races r
        WHERE r.is_midnight = {midnight_val}
          {date_filter}
          AND NOT EXISTS (
            SELECT 1 FROM entries e
            WHERE e.race_id = r.race_id
              AND e.kyoso_tokuten IS NULL
          )
        ORDER BY r.race_date
    """, conn)
    print(f"[{label}] 対象レース数: {len(races_df):,}")

    if len(races_df) == 0:
        conn.close()
        return None, None, None, None

    # entries を JOIN で一括読み込み（IN句の変数上限を回避）
    entries_df = pd.read_sql_query(f"""
        SELECT e.* FROM entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          {date_filter}
          AND e.kyoso_tokuten IS NOT NULL
    """, conn)

    # results を JOIN で一括読み込み（1着のみ = ターゲット生成用）
    results_df = pd.read_sql_query(f"""
        SELECT res.race_id, res.sha_ban, res.rank
        FROM results res
        JOIN races r ON res.race_id = r.race_id
        WHERE r.is_midnight = {midnight_val}
          {date_filter}
          AND res.rank = 1
    """, conn)
    conn.close()

    print(f"[{label}] entries: {len(entries_df):,}, results(1着): {len(results_df):,}")

    # chariloto DB のカラム名を feature_engine が期待する名前にリネーム
    col_renames = {}
    if "sha_ban" in entries_df.columns and "car_no" not in entries_df.columns:
        col_renames["sha_ban"] = "car_no"
    if "kyakushitsu" in entries_df.columns and "style" not in entries_df.columns:
        col_renames["kyakushitsu"] = "style"
    if "ki_betsu" in entries_df.columns and "term" not in entries_df.columns:
        col_renames["ki_betsu"] = "term"
    if "kyoso_tokuten" in entries_df.columns and "grade_score" not in entries_df.columns:
        col_renames["kyoso_tokuten"] = "grade_score"
    if "todofuken" in entries_df.columns and "prefecture" not in entries_df.columns:
        col_renames["todofuken"] = "prefecture"
    if col_renames:
        entries_df = entries_df.rename(columns=col_renames)
        print(f"[{label}] カラム名リネーム: {col_renames}")

    # feature_engine が期待する追加カラムのフォールバック
    for col, default in [
        ("racer_class", None), ("win_rate", 0.0),
        ("second_rate", 0.0), ("third_rate", 0.0),
        ("racer_id", None), ("district", None),
    ]:
        if col not in entries_df.columns:
            entries_df[col] = default

    # レース情報をエントリーにマージ
    df = entries_df.merge(races_df, on="race_id", how="left")

    # 特徴量を計算（db_path を渡して履歴特徴量も計算）
    print(f"[{label}] 特徴量計算中（{len(FEATURE_NAMES)}特徴量）...")
    t0 = time.time()

    # create_features を一括で渡す
    # v0.24 で履歴特徴量(I-01〜I-06)はバッチプリロード方式に改修済み
    # DBパスを渡すことで I-01 当場勝率、I-02 トレンド、I-04 Elo が
    # 実データで計算される（I-03 h2h と I-05/I-06 agari はデフォルト値のまま）
    features = create_features(
        entries_df=entries_df,
        races_df=races_df,
        odds_df=pd.DataFrame(),  # オッズは未取得のため空
        line_probs=None,
        bank_info=None,
        db_path=str(db_path),
    )
    elapsed = time.time() - t0
    print(f"[{label}] 特徴量計算完了: {elapsed:.1f}秒, {len(features):,}行")

    # ターゲット: 1着=1, その他=0
    # results_df の sha_ban は car_no に対応
    winner_map = set(
        (row["race_id"], int(row["sha_ban"]))
        for _, row in results_df.iterrows()
    )
    features["target"] = features.apply(
        lambda row: 1 if (row["race_id"], int(row["car_no"])) in winner_map else 0,
        axis=1,
    )

    # race_date を dates として抽出
    if "race_date" in features.columns:
        dates = features["race_date"]
    elif "date" in features.columns:
        dates = features["date"]
    else:
        # races_df からマージ
        dates = features["race_id"].map(
            dict(zip(races_df["race_id"], races_df["race_date"]))
        )

    X = features[FEATURE_NAMES].fillna(0)
    y = features["target"]
    race_ids = features["race_id"]

    print(f"[{label}] X shape: {X.shape}")
    print(f"[{label}] 1着率: {y.mean():.4f}")

    return X, y, dates, race_ids


def train_and_evaluate(is_midnight: bool, db_path: Path = DB_PATH,
                       train_start: str = None, train_end: str = None,
                       model_suffix: str = None):
    """
    モデルを学習し、Purged K-Fold CV で評価する。

    Parameters:
        is_midnight: モデル種別
        train_start/train_end: 学習期間（省略時は全期間）
        model_suffix: モデル保存ファイル名のサフィックス
                       例: "2023" → stage1_normal_2023.pkl
    """
    label = "midnight" if is_midnight else "normal"
    label_ja = "ミッドナイト" if is_midnight else "通常"

    print(f"\n{'='*60}")
    print(f"  {label_ja} モデル学習")
    print(f"{'='*60}\n")

    t_start = time.time()

    X, y, dates, race_ids = load_training_data(
        is_midnight, db_path,
        train_start=train_start, train_end=train_end,
    )
    if X is None:
        print(f"[{label_ja}] データなし。スキップ。")
        return

    # Purged K-Fold CV
    print(f"\n[{label_ja}] Purged K-Fold CV (5分割, gap=7日)...")
    try:
        cv_results = []
        from sklearn.metrics import roc_auc_score
        for fold_idx, (train_idx, test_idx) in enumerate(
            purged_kfold_cv(X, y, dates, n_splits=5, gap_days=7)
        ):
            model = Stage1Model()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict_proba(X.iloc[test_idx])
            auc = roc_auc_score(y.iloc[test_idx], y_pred)
            cv_results.append(auc)
            print(f"  Fold {fold_idx+1}: AUC={auc:.4f} "
                  f"(train={len(train_idx):,}, test={len(test_idx):,})")

        cv_mean = np.mean(cv_results)
        cv_std = np.std(cv_results)
        print(f"\n  CV平均: AUC={cv_mean:.4f} ± {cv_std:.4f}")
    except Exception as e:
        print(f"  CV エラー: {e}")
        cv_mean = 0.0

    # 全データで最終学習
    print(f"\n[{label_ja}] 全データで最終学習...")
    final_model = Stage1Model()
    final_model.fit(X, y)

    # 特徴量重要度
    importance = final_model.get_feature_importance()
    print(f"\n=== 特徴量重要度 TOP20 ({label_ja}) ===")
    for feat, score in importance[:20]:
        print(f"  {feat}: {score:.1f}")

    # モデル保存（サフィックスがあれば stage1_{label}_{suffix}.pkl）
    if model_suffix:
        model_path = MODEL_DIR / f"stage1_{label}_{model_suffix}.pkl"
    else:
        model_path = MODEL_DIR / f"stage1_{label}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "cv_mean_auc": cv_mean,
            "cv_results": cv_results if 'cv_results' in dir() else [],
            "feature_names": FEATURE_NAMES,
            "n_train": len(X),
            "label": label,
            "train_start": train_start,
            "train_end": train_end,
        }, f)
    print(f"\nモデル保存: {model_path}")

    elapsed = time.time() - t_start
    print(f"所要時間: {elapsed/60:.1f}分")

    return {
        "label": label,
        "cv_mean": cv_mean,
        "n_train": len(X),
        "importance": importance[:20],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage1 学習")
    parser.add_argument("--train_start", type=str, default=None,
                        help="学習開始日 YYYY-MM-DD（省略で全期間）")
    parser.add_argument("--train_end", type=str, default=None,
                        help="学習終了日 YYYY-MM-DD（省略で全期間）")
    parser.add_argument("--model_suffix", type=str, default=None,
                        help="モデルファイル名サフィックス "
                             "（例: 2023 → stage1_normal_2023.pkl）")
    args = parser.parse_args()

    print("=" * 60)
    print("  競輪AI Stage1 学習")
    print(f"  期間: {args.train_start or '-'} 〜 {args.train_end or '-'}")
    if args.model_suffix:
        print(f"  サフィックス: _{args.model_suffix}")
    print("=" * 60)

    results = []

    # 通常レースモデル
    r = train_and_evaluate(
        is_midnight=False,
        train_start=args.train_start, train_end=args.train_end,
        model_suffix=args.model_suffix,
    )
    if r:
        results.append(r)

    # ミッドナイトモデル
    r = train_and_evaluate(
        is_midnight=True,
        train_start=args.train_start, train_end=args.train_end,
        model_suffix=args.model_suffix,
    )
    if r:
        results.append(r)

    print("\n" + "=" * 60)
    print("  学習結果サマリー")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']}: AUC={r['cv_mean']:.4f} "
              f"(n={r['n_train']:,})")
    print("=" * 60)
