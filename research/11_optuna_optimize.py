# ===========================================
# 11_optuna_optimize.py
# Optuna最適化（探索空間削減版）
# - 14パラメータ → 4パラメータに削減
# - 正解率 → Profit Factor を最適化
# - Purged CV + Walk-Forward で評価
# ===========================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, add_regime_features, FEATURE_COLS
from research.common.labels import build_triple_barrier_labels
from research.common.validation import PurgedKFold, compute_metrics
from research.common.economic_surprise import add_surprise_features, INDICATORS
from research.common.cftc_positions import add_cot_features, COT_FEATURE_COLS

script_dir = Path(__file__).resolve().parent

# ===== 固定パラメータ（探索しない） =====
BARRIER_UP = 0.005
BARRIER_DOWN = -0.003
BARRIER_T = 24
VOL_MULT = 2.0
REGIME_CHANGE_THRESH = 2
PRICE_CHANGE_MULT = 3.0
ADOPTION_TARGET = 0.4

# ===== データ読み込み（1回だけ） =====
print("データ読み込み中...")
df = load_usdjpy_1h()
df = add_technical_features(df)
df = add_regime_features(df)
df = add_rate_features(df)
df = add_daily_trend_features(df)
df = add_surprise_features(df)
df = add_cot_features(df)

df["Close_4h_later"] = df["Close"].shift(-4)
df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)
df["Return_4h"] = (df["Close_4h_later"] - df["Close"]) / df["Close"]
df["Regime_changes_3h"] = (df["Regime"].diff().fillna(0) != 0).astype(int).rolling(3).sum().fillna(0)
df["Abs_ret_1h"] = df["Return_1"].abs()

surprise_cols = [f"surprise_{sid}" for sid in INDICATORS] + ["surprise_composite"]
cot_cols = [c for c in COT_FEATURE_COLS if c in df.columns]
feature_cols = FEATURE_COLS + [c for c in surprise_cols if c in df.columns] + cot_cols
df = df.dropna(subset=feature_cols + ["Label", "Return_4h"])
X = df[feature_cols]
y_direction = df["Label"].values
close_arr = df["Close"].values
ret4_arr = df["Return_4h"].values
regime_all = df["Regime"].values
vol_all = df["Volatility_24"].values
rc_all = df["Regime_changes_3h"].values
abs_ret_all = df["Abs_ret_1h"].values
n_total = len(df)

print(f"データ: {n_total}本")

# Triple-Barrier ラベルを事前計算（固定パラメータ）
y_triple = build_triple_barrier_labels(close_arr, BARRIER_UP, BARRIER_DOWN, BARRIER_T)


def evaluate_fold(train_idx, test_idx, params):
    """1フォールドでモデルA/Bを学習・予測し、Profit Factorを返す"""
    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_dir_tr = y_direction[train_idx]
    y_triple_tr = y_triple[train_idx]
    regime_te = regime_all[test_idx]
    vol_te = vol_all[test_idx]
    rc_te = rc_all[test_idx]
    abs_te = abs_ret_all[test_idx]
    ret4_te = ret4_arr[test_idx]
    hist_vol = vol_all[train_idx].mean()
    hist_abs = abs_ret_all[train_idx].mean()
    if hist_abs <= 0:
        hist_abs = 1e-8

    lgb_params = {
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": 6,  # 固定
        "random_state": 42,
        "verbosity": -1,
    }

    # モデルA（Triple-Barrier）
    mask_a = (y_triple_tr == 0) | (y_triple_tr == 1)
    if mask_a.sum() < 10:
        return float("nan")
    model_a = lgb.LGBMClassifier(**lgb_params)
    model_a.fit(X_tr[mask_a], y_triple_tr[mask_a].astype(int))

    # モデルB（方向 + Meta-Labeling）
    model_b_p = lgb.LGBMClassifier(**lgb_params)
    model_b_p.fit(X_tr, y_dir_tr)
    X_tr_meta = X_tr.copy()
    X_tr_meta["primary_proba"] = model_b_p.predict_proba(X_tr)[:, 1]
    y_meta = (model_b_p.predict(X_tr) == y_dir_tr).astype(int)
    model_b_s = lgb.LGBMClassifier(**lgb_params)
    model_b_s.fit(X_tr_meta, y_meta)

    X_te_meta = X_te.copy()
    X_te_meta["primary_proba"] = model_b_p.predict_proba(X_te)[:, 1]
    proba_adopt = model_b_s.predict_proba(X_te_meta)[:, 1]
    thresh = np.percentile(
        model_b_s.predict_proba(X_tr_meta)[:, 1],
        100 - params["adoption_target"] * 100,
    )
    adopt_b = proba_adopt >= thresh

    # 待機条件
    wait_c = (
        (vol_te > params["vol_mult"] * hist_vol)
        | (rc_te >= REGIME_CHANGE_THRESH)
        | (abs_te > PRICE_CHANGE_MULT * hist_abs)
    )

    pred_a = model_a.predict(X_te)
    pred_b = model_b_p.predict(X_te)

    # トレードリターン計算
    trade_returns = []
    for i in range(len(test_idx)):
        r = regime_te[i]
        if r == 0:
            direction = int(pred_a[i])
        elif r == 1:
            if adopt_b[i]:
                direction = int(pred_b[i])
            else:
                continue
        else:
            continue
        if wait_c[i]:
            continue
        direction_mult = 1.0 if direction == 1 else -1.0
        trade_returns.append(ret4_te[i] * direction_mult)

    if len(trade_returns) < 20:
        return float("nan")

    m = compute_metrics(np.array(trade_returns))
    return m["pf"]


def objective(trial):
    """Optuna目的関数: Purged CVでのProfit Factorを最大化"""
    # 探索するのは4パラメータだけ
    n_estimators = trial.suggest_int("n_estimators", 100, 400, step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    adoption_target = trial.suggest_float("adoption_target", 0.2, 0.6)
    vol_mult = trial.suggest_float("vol_mult", 1.5, 3.0)

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "adoption_target": adoption_target,
        "vol_mult": vol_mult,
    }

    cv = PurgedKFold(n_splits=5, embargo_size=24)
    pf_list = []
    for train_idx, test_idx in cv.split(X):
        pf = evaluate_fold(train_idx, test_idx, params)
        if not np.isnan(pf):
            pf_list.append(pf)

    if len(pf_list) == 0:
        return 0.0

    # 平均PFを返す（ただしPFが極端に高い場合はペナルティ）
    mean_pf = np.mean(pf_list)
    # PFが3を超える場合は過学習の疑い → クリップ
    return min(mean_pf, 3.0)


# ===== Optuna 最適化 =====
print("\nOptuna で Purged CV Profit Factor を最大化（50試行・4パラメータ）...")
print("探索パラメータ: n_estimators, learning_rate, adoption_target, vol_mult")
print("固定パラメータ: barrier(0.005/-0.003/24h), max_depth=6, weights=1.0\n")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ===== 結果表示 =====
best = study.best_params
print(f"\n{'='*60}")
print("【最適パラメータ】")
print(f"  LightGBM: n_estimators={best['n_estimators']}, lr={best['learning_rate']:.4f}")
print(f"  Meta-Labeling: 採用率目標={best['adoption_target']:.0%}")
print(f"  待機: ボラ閾値={best['vol_mult']:.2f}倍")
print(f"  Purged CV 平均PF: {study.best_value:.4f}")

# ===== best_params.json に保存 =====
best_params_to_save = {
    "triple_barrier": {
        "barrier_up": BARRIER_UP,
        "barrier_down": BARRIER_DOWN,
        "barrier_t": BARRIER_T,
    },
    "wait_mode": {
        "vol_mult": best["vol_mult"],
        "regime_change_thresh": REGIME_CHANGE_THRESH,
        "price_change_mult": PRICE_CHANGE_MULT,
    },
    "meta_labeling": {
        "adoption_target": best["adoption_target"],
    },
    "lgbm": {
        "n_estimators": best["n_estimators"],
        "learning_rate": best["learning_rate"],
        "max_depth": 6,
    },
}

best_params_path = (script_dir / ".." / "data" / "best_params.json").resolve()
best_params_path.parent.mkdir(parents=True, exist_ok=True)
with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(best_params_to_save, f, ensure_ascii=False, indent=2)
print(f"\n最適パラメータを保存: {best_params_path}")

# ===== MLflow 記録 =====
mlflow.set_experiment("fx_ai_optuna_v2")
with mlflow.start_run():
    for k, v in best.items():
        mlflow.log_param(k, v)
    mlflow.log_param("n_search_params", 4)
    mlflow.log_param("fixed_barrier", f"{BARRIER_UP}/{BARRIER_DOWN}/{BARRIER_T}")
    mlflow.log_metric("purged_cv_pf", study.best_value)
    mlflow.log_metric("n_trials", 50)
    print("MLflow に記録しました（実験名: fx_ai_optuna_v2）")
