# ===========================================
# scripts/train_v038.py
# v0.38 本命/穴モデル分離 — 訓練・重要度分析
#
# 通常レースを対象（ミッドナイトは小サンプル・後回し）
# 2022-2023 学習、2024 テスト
# ===========================================

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "data"))
sys.path.insert(0, str(SCRIPT_DIR))

from feature_engine import FEATURE_NAMES, DB_PATH
from favorite_model import FavoriteStage1Model
from underdog_model import UnderdogStage1Model
from prediction_model import purged_kfold_cv
from train import load_training_data

import sqlite3

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_with_payouts(is_midnight, start, end):
    """load_training_data + 各エントリに当レース trifecta_payout を broadcast"""
    X, y, dates, race_ids = load_training_data(
        is_midnight, train_start=start, train_end=end,
    )
    if X is None:
        return None, None, None, None, None

    # race_id -> trifecta_payout 取得
    conn = sqlite3.connect(str(DB_PATH))
    payouts_df = pd.read_sql_query("""
        SELECT race_id, trifecta_payout
        FROM results WHERE rank=1 AND trifecta_payout IS NOT NULL
    """, conn)
    conn.close()
    payout_map = dict(zip(payouts_df["race_id"], payouts_df["trifecta_payout"]))

    payouts = race_ids.map(payout_map).fillna(3000.0)  # デフォルト平均値
    return X, y, dates, race_ids, payouts


def train_and_eval(model_cls, X, y, dates, payouts, label):
    """Purged K-Fold CV で AUC 評価 + 最終学習"""
    print(f"\n=== {label} モデル訓練 ===")
    cv_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(
        purged_kfold_cv(X, y, dates, n_splits=5, gap_days=7)
    ):
        m = model_cls()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], payouts.iloc[tr_idx])
        pred = m.predict_proba(X.iloc[te_idx])
        auc = roc_auc_score(y.iloc[te_idx], pred)
        cv_aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.4f}")
    cv_mean = float(np.mean(cv_aucs))
    print(f"  CV平均: {cv_mean:.4f}")

    print(f"  最終学習 (全データ)...")
    final = model_cls()
    final.fit(X, y, payouts)
    imp = final.get_feature_importance(FEATURE_NAMES)
    print(f"  TOP10 重要度:")
    for f, v in imp[:10]:
        print(f"    {f}: {v:.0f}")
    return final, cv_mean, imp


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_start", type=str, default="2022-01-01")
    parser.add_argument("--train_end", type=str, default="2023-12-31")
    parser.add_argument("--suffix", type=str, default="v0.38")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"  v0.38 本命/穴モデル分離訓練 {args.train_start} 〜 {args.train_end}")
    print(f"  suffix: {args.suffix}")
    print("=" * 60)

    # 通常レース 学習
    X, y, dates, race_ids, payouts = load_with_payouts(
        is_midnight=False, start=args.train_start, end=args.train_end,
    )
    if X is None:
        print("データなし")
        return
    print(f"\nX shape: {X.shape}, 1率: {y.mean():.4f}")
    print(f"payouts 分布: 本命<1k:{(payouts<1000).sum():,}, "
          f"中穴:{((payouts>=1000)&(payouts<10000)).sum():,}, "
          f"穴>=10k:{(payouts>=10000).sum():,}")

    # favorite
    fav, fav_cv, fav_imp = train_and_eval(
        FavoriteStage1Model, X, y, dates, payouts, "FAVORITE"
    )
    fav_path = MODEL_DIR / f"stage1_normal_favorite_{args.suffix}.pkl"
    with open(fav_path, "wb") as f:
        pickle.dump({
            "model": fav, "cv_mean_auc": fav_cv,
            "feature_names": FEATURE_NAMES,
            "train_start": args.train_start, "train_end": args.train_end,
        }, f)
    print(f"保存: {fav_path.name}")

    # underdog
    ud, ud_cv, ud_imp = train_and_eval(
        UnderdogStage1Model, X, y, dates, payouts, "UNDERDOG"
    )
    ud_path = MODEL_DIR / f"stage1_normal_underdog_{args.suffix}.pkl"
    with open(ud_path, "wb") as f:
        pickle.dump({
            "model": ud, "cv_mean_auc": ud_cv,
            "feature_names": FEATURE_NAMES,
            "train_start": args.train_start, "train_end": args.train_end,
        }, f)
    print(f"保存: {ud_path.name}")

    # 特徴量重要度比較 JSON 出力
    analysis = {
        "favorite_cv_auc": fav_cv,
        "underdog_cv_auc": ud_cv,
        "favorite_top20": [{"feat": f, "score": float(s)} for f, s in fav_imp[:20]],
        "underdog_top20": [{"feat": f, "score": float(s)} for f, s in ud_imp[:20]],
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    out = REPORT_DIR / f"v038_feature_analysis_{args.suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n重要度分析: {out}")

    # 差異発見
    fav_top_set = set(f for f, _ in fav_imp[:10])
    ud_top_set = set(f for f, _ in ud_imp[:10])
    only_fav = fav_top_set - ud_top_set
    only_ud = ud_top_set - fav_top_set
    print(f"\n本命のみ TOP10入り: {only_fav}")
    print(f"穴のみ   TOP10入り: {only_ud}")

    elapsed = time.time() - t0
    print(f"\n所要時間: {elapsed/60:.1f}分")


if __name__ == "__main__":
    main()
