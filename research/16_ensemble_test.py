# ===========================================
# 16_ensemble_test.py
# アンサンブル（多数決）+ 5年データの効果検証
#
# 比較パターン:
#   A: LightGBM単体 + 2年データ（現行モデル）
#   B: LightGBM単体 + 5年データ
#   C: アンサンブル(5モデル) + 2年データ（3人以上一致）
#   D: アンサンブル(5モデル) + 5年データ（3人以上一致）
#   E: アンサンブル(5モデル) + 5年データ（4人以上一致）
#   F: アンサンブル(5モデル) + 5年データ（5人一致）
# ===========================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, add_regime_features, FEATURE_COLS
from research.common.economic_surprise import add_surprise_features, INDICATORS
from research.common.cftc_positions import add_cot_features, COT_FEATURE_COLS
from research.common.validation import walk_forward_splits, compute_metrics
from research.common.ensemble import EnsembleClassifier

CONFIDENCE_THRESHOLD = 0.60
WF_MIN_TRAIN = 4320
WF_TEST_SIZE = 720
LGB_N_EST = 200
LGB_LR = 0.05


def prepare(use_5y=False):
    """データ準備"""
    df = load_usdjpy_1h(use_5y=use_5y)
    df = add_technical_features(df)
    df = add_regime_features(df)
    df = add_rate_features(df)
    df = add_daily_trend_features(df)
    df = add_surprise_features(df)
    df = add_cot_features(df)

    df["Close_4h_later"] = df["Close"].shift(-4)
    df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)
    df["Return_4h"] = (df["Close_4h_later"] - df["Close"]) / df["Close"]

    surprise_cols = [f"surprise_{sid}" for sid in INDICATORS] + ["surprise_composite"]
    cot_cols = [c for c in COT_FEATURE_COLS if c in df.columns]
    feature_cols = FEATURE_COLS + [c for c in surprise_cols if c in df.columns] + cot_cols
    df = df.dropna(subset=feature_cols + ["Label", "Return_4h"])
    return df, feature_cols


def run_walkforward(df, feature_cols, use_ensemble=False, min_agreement=3, label=""):
    """Walk-Forward検証を実行

    min_agreement: アンサンブル時、最低何人の一致が必要か (3, 4, or 5)
    """
    X = df[feature_cols]
    y = df["Label"]
    ret4 = df["Return_4h"].values
    n = len(df)

    splits = walk_forward_splits(n, WF_MIN_TRAIN, WF_TEST_SIZE)
    all_returns = []
    window_pfs = []

    for w_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        ret_test = ret4[test_idx]

        # モデル学習
        if use_ensemble:
            model = EnsembleClassifier(n_estimators=LGB_N_EST, learning_rate=LGB_LR)
        else:
            model = lgb.LGBMClassifier(
                n_estimators=LGB_N_EST, learning_rate=LGB_LR,
                max_depth=6, random_state=42, verbosity=-1,
            )
        model.fit(X_train, y_train)

        # 予測
        proba = model.predict_proba(X_test)[:, 1]

        if use_ensemble:
            _, agreement = model.predict_with_agreement(X_test)

        # トレードリターン計算（自信度フィルター付き）
        window_returns = []
        for i in range(len(test_idx)):
            confidence = max(proba[i], 1.0 - proba[i])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            # アンサンブルの場合: 一致人数フィルター
            if use_ensemble and agreement[i] < min_agreement:
                continue
            direction = 1.0 if proba[i] > 0.5 else -1.0
            window_returns.append(ret_test[i] * direction)

        all_returns.extend(window_returns)
        if window_returns:
            m = compute_metrics(np.array(window_returns))
            window_pfs.append(m["pf"])

    if not all_returns:
        print(f"  {label}: トレードなし")
        return None

    overall = compute_metrics(np.array(all_returns))
    pf_above_1 = sum(1 for p in window_pfs if p >= 1.0)
    print(f"  {label}:")
    print(f"    PF={overall['pf']:.2f}, Sharpe={overall['sharpe']:.2f}, "
          f"勝率={overall['win_rate']:.1f}%, トレード={overall['n_trades']}, "
          f"期待値={overall['exp_value_net']:+.6f}, "
          f"PF>=1.0: {pf_above_1}/{len(window_pfs)}")
    return overall


# ===== 6パターン比較 =====
print("=" * 60)
print("5モデルアンサンブル + 5年データ 効果比較")
print("=" * 60)

# A: 現行（LightGBM単体 + 2年）
print("\n[A] LightGBM単体 + 2年データ（現行モデル）")
df_2y, fcols_2y = prepare(use_5y=False)
print(f"  データ: {len(df_2y)}本")
result_a = run_walkforward(df_2y, fcols_2y, use_ensemble=False, label="LGB+2y")

# B: LightGBM単体 + 5年
print("\n[B] LightGBM単体 + 5年データ")
df_5y, fcols_5y = prepare(use_5y=True)
print(f"  データ: {len(df_5y)}本")
result_b = run_walkforward(df_5y, fcols_5y, use_ensemble=False, label="LGB+5y")

# C: 5モデルアンサンブル + 2年（3人以上一致）
print("\n[C] 5モデルアンサンブル + 2年データ（3人以上一致）")
result_c = run_walkforward(df_2y, fcols_2y, use_ensemble=True, min_agreement=3, label="5Ensemble+2y(3+)")

# D: 5モデルアンサンブル + 5年（3人以上一致）
print("\n[D] 5モデルアンサンブル + 5年データ（3人以上一致）")
result_d = run_walkforward(df_5y, fcols_5y, use_ensemble=True, min_agreement=3, label="5Ensemble+5y(3+)")

# E: 5モデルアンサンブル + 5年（4人以上一致）
print("\n[E] 5モデルアンサンブル + 5年データ（4人以上一致）")
result_e = run_walkforward(df_5y, fcols_5y, use_ensemble=True, min_agreement=4, label="5Ensemble+5y(4+)")

# F: 5モデルアンサンブル + 5年（5人全員一致）
print("\n[F] 5モデルアンサンブル + 5年データ（5人全員一致）")
result_f = run_walkforward(df_5y, fcols_5y, use_ensemble=True, min_agreement=5, label="5Ensemble+5y(5)")

# ===== 比較まとめ =====
print(f"\n{'='*60}")
print("【比較まとめ】")
print(f"{'パターン':25s} {'PF':>6s} {'Sharpe':>8s} {'勝率':>7s} {'期待値':>12s} {'トレード':>8s}")
print("-" * 75)
for name, r in [("A: LGB+2y(現行)", result_a), ("B: LGB+5y", result_b),
                ("C: 5Ens+2y(3+)", result_c), ("D: 5Ens+5y(3+)", result_d),
                ("E: 5Ens+5y(4+)", result_e), ("F: 5Ens+5y(5人)", result_f)]:
    if r:
        print(f"  {name:23s} {r['pf']:6.2f} {r['sharpe']:8.2f} {r['win_rate']:6.1f}% {r['exp_value_net']:+12.6f} {r['n_trades']:8d}")
    else:
        print(f"  {name:23s}  --- トレードなし ---")
