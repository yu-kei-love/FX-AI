# ===========================================
# scripts/tune_hyperparams.py
# 競輪AI - LightGBM ハイパーパラメータ最適化（Optuna）
#
# 現行の Stage1Model.LGB_PARAMS はデフォルト近似値のため、
# Purged K-Fold CV の平均 AUC を最大化するパラメータを探索する。
#
# 対象: LightGBM 単体（MLP・LR は含まない = 時間対効果優先）
# CV: Purged K-Fold (5splits, gap=7日)
# 学習期間: 2022-01-01 〜 2023-12-31 (デフォルト)
#
# 使い方:
#   python scripts/tune_hyperparams.py --trials 10   # 軽量テスト
#   python scripts/tune_hyperparams.py --trials 100  # 本番
#   python scripts/tune_hyperparams.py --trials 100 --normal_only
#   python scripts/tune_hyperparams.py --trials 100 --midnight_only
#
# 結果: models/best_params_{normal|midnight}.json に保存
# ===========================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "data"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from feature_engine import DB_PATH, FEATURE_NAMES  # noqa: E402
from prediction_model import purged_kfold_cv  # noqa: E402

# train.py の load_training_data を再利用
from train import load_training_data  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def objective(trial, X, y, dates, n_splits=5, gap_days=7):
    """
    Optuna 目的関数。Purged K-Fold CV の平均 AUC を返す。
    """
    params = {
        "objective":          "binary",
        "metric":             "auc",
        "verbose":            -1,
        "random_state":       42,
        "n_jobs":             -1,
        "learning_rate":      trial.suggest_float(
                                  "learning_rate", 0.01, 0.3, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
        "max_depth":          trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":  trial.suggest_int(
                                  "min_child_samples", 5, 100),
        "feature_fraction":   trial.suggest_float(
                                  "feature_fraction", 0.5, 1.0),
        "bagging_fraction":   trial.suggest_float(
                                  "bagging_fraction", 0.5, 1.0),
        "bagging_freq":       trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l1":          trial.suggest_float(
                                  "lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2":          trial.suggest_float(
                                  "lambda_l2", 1e-8, 10.0, log=True),
        "n_estimators":       trial.suggest_int("n_estimators", 100, 1000),
    }

    scores = []
    X_np = X.values  # speed
    y_np = y.values
    for fold_idx, (train_idx, test_idx) in enumerate(
        purged_kfold_cv(X, y, dates, n_splits=n_splits, gap_days=gap_days)
    ):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_np[train_idx], y_np[train_idx])
        y_pred = model.predict_proba(X_np[test_idx])[:, 1]
        auc = roc_auc_score(y_np[test_idx], y_pred)
        scores.append(auc)
        # Optuna Pruner 連携（途中経過の報告）
        trial.report(auc, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune(is_midnight: bool, n_trials: int = 100,
         train_start: str = "2022-01-01",
         train_end: str = "2023-12-31"):
    label = "midnight" if is_midnight else "normal"
    print(f"\n{'='*60}")
    print(f"  {label} ハイパラ最適化 (n_trials={n_trials})")
    print(f"{'='*60}\n")

    t0 = time.time()
    X, y, dates, _race_ids = load_training_data(
        is_midnight=is_midnight,
        train_start=train_start, train_end=train_end,
    )
    if X is None:
        print(f"[{label}] データなし")
        return None
    print(f"[{label}] X: {X.shape}, y 1率: {y.mean():.4f}, "
          f"準備 {time.time()-t0:.1f}秒")

    # Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, X, y, dates),
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )
    elapsed_min = (time.time() - t0) / 60

    print(f"\n=== 最適化完了 ({label}) ===")
    print(f"所要時間: {elapsed_min:.1f} 分")
    print(f"Best CV AUC: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = MODEL_DIR / f"best_params_{label}.json"
    payload = {
        "best_value":  study.best_value,
        "best_params": study.best_params,
        "n_trials":    n_trials,
        "train_start": train_start,
        "train_end":   train_end,
        "tuned_at":    pd.Timestamp.now().isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_path}")

    return study.best_params


def main():
    parser = argparse.ArgumentParser(
        description="LightGBM ハイパラ最適化 (Optuna)"
    )
    parser.add_argument("--trials", type=int, default=100,
                        help="試行回数 (デフォルト: 100)")
    parser.add_argument("--midnight_only", action="store_true")
    parser.add_argument("--normal_only", action="store_true")
    parser.add_argument("--train_start", type=str, default="2022-01-01")
    parser.add_argument("--train_end", type=str, default="2023-12-31")
    args = parser.parse_args()

    # optuna のログレベルを抑制
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not args.midnight_only:
        tune(is_midnight=False, n_trials=args.trials,
             train_start=args.train_start, train_end=args.train_end)
    if not args.normal_only:
        tune(is_midnight=True, n_trials=args.trials,
             train_start=args.train_start, train_end=args.train_end)


if __name__ == "__main__":
    main()
