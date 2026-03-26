# ===========================================
# 17_ensemble_detailed.py
# 詳細分析:
#   1. 時間減衰ウェイト（古いデータの重みを下げる）
#   2. 一致度別の正解率（2人一致 vs 3人一致）
# ===========================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, add_regime_features, FEATURE_COLS
from research.common.economic_surprise import add_surprise_features, INDICATORS
from research.common.cftc_positions import add_cot_features, COT_FEATURE_COLS
from research.common.validation import walk_forward_splits, compute_metrics

CONFIDENCE_THRESHOLD = 0.60
WF_MIN_TRAIN = 4320
WF_TEST_SIZE = 720


def prepare(use_5y=False):
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


def make_time_weights(index, half_life_days=365):
    """時間減衰ウェイトを作る

    古いデータほど重みが小さくなる。
    half_life_days: この日数で重みが半分になる
    例: half_life_days=365 → 1年前のデータの重みは0.5、2年前は0.25
    """
    latest = index.max()
    days_ago = (latest - index).total_seconds() / 86400
    weights = np.power(0.5, days_ago / half_life_days)
    return weights.values if hasattr(weights, 'values') else np.array(weights)


def run_analysis(df, feature_cols, use_5y=False):
    """Walk-Forwardで一致度別の詳細分析を実行"""
    X = df[feature_cols]
    y = df["Label"].values
    ret4 = df["Return_4h"].values
    n = len(df)

    splits = walk_forward_splits(n, WF_MIN_TRAIN, WF_TEST_SIZE)

    # 結果を貯める
    results_2agree = []  # 2人一致のトレード
    results_3agree = []  # 3人一致のトレード
    results_weighted = []  # 時間減衰ウェイト付き学習のトレード
    results_weighted_3agree = []  # 時間減衰 + 3人一致

    for w_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        ret_test = ret4[test_idx]
        y_test = y[test_idx]

        # === パターン1: 通常学習（重み均等）===
        m_lgb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)
        m_xgb = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0, eval_metric="logloss")
        m_rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

        m_lgb.fit(X_train, y_train)
        m_xgb.fit(X_train, y_train)
        m_rf.fit(X_train, y_train)

        pred_lgb = m_lgb.predict(X_test)
        pred_xgb = m_xgb.predict(X_test)
        pred_rf = m_rf.predict(X_test)

        proba_avg = (m_lgb.predict_proba(X_test)[:, 1] +
                     m_xgb.predict_proba(X_test)[:, 1] +
                     m_rf.predict_proba(X_test)[:, 1]) / 3.0

        # 一致度を計算
        votes = np.array([pred_lgb, pred_xgb, pred_rf])
        vote_sum = votes.sum(axis=0)  # 0=全員売り, 1=2人売り, 2=2人買い, 3=全員買い

        for i in range(len(test_idx)):
            confidence = max(proba_avg[i], 1.0 - proba_avg[i])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            direction = 1.0 if proba_avg[i] > 0.5 else -1.0
            ret = ret_test[i] * direction
            actual_correct = (1 if direction > 0 else 0) == y_test[i]

            # 3人一致（vote_sum == 0 or 3）
            if vote_sum[i] == 0 or vote_sum[i] == 3:
                results_3agree.append({"ret": ret, "correct": actual_correct})
            # 2人一致（vote_sum == 1 or 2）
            else:
                results_2agree.append({"ret": ret, "correct": actual_correct})

        # === パターン2: 時間減衰ウェイト付き学習 ===
        weights = make_time_weights(df.index[train_idx])

        m_lgb_w = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)
        m_xgb_w = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0, eval_metric="logloss")
        # RandomForestはsample_weightに対応
        m_rf_w = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

        m_lgb_w.fit(X_train, y_train, sample_weight=weights)
        m_xgb_w.fit(X_train, y_train, sample_weight=weights)
        m_rf_w.fit(X_train, y_train, sample_weight=weights)

        pred_lgb_w = m_lgb_w.predict(X_test)
        pred_xgb_w = m_xgb_w.predict(X_test)
        pred_rf_w = m_rf_w.predict(X_test)

        proba_avg_w = (m_lgb_w.predict_proba(X_test)[:, 1] +
                       m_xgb_w.predict_proba(X_test)[:, 1] +
                       m_rf_w.predict_proba(X_test)[:, 1]) / 3.0

        votes_w = np.array([pred_lgb_w, pred_xgb_w, pred_rf_w])
        vote_sum_w = votes_w.sum(axis=0)

        for i in range(len(test_idx)):
            confidence = max(proba_avg_w[i], 1.0 - proba_avg_w[i])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            direction = 1.0 if proba_avg_w[i] > 0.5 else -1.0
            ret = ret_test[i] * direction

            results_weighted.append(ret)
            if vote_sum_w[i] == 0 or vote_sum_w[i] == 3:
                results_weighted_3agree.append(ret)

        print(f"  Window {w_idx+1}/{len(splits)} 完了")

    # ===== 結果表示 =====
    print(f"\n{'='*60}")
    print("【分析1: 2人一致 vs 3人一致の正解率】")
    print(f"{'='*60}")

    if results_2agree:
        correct_2 = sum(1 for r in results_2agree if r["correct"])
        total_2 = len(results_2agree)
        rets_2 = np.array([r["ret"] for r in results_2agree])
        m2 = compute_metrics(rets_2)
        print(f"  2人一致（意見が割れた）:")
        print(f"    トレード数: {total_2}")
        print(f"    正解率: {correct_2}/{total_2} ({correct_2/total_2*100:.1f}%)")
        print(f"    PF: {m2['pf']:.2f}, 期待値: {m2['exp_value_net']:+.6f}")

    if results_3agree:
        correct_3 = sum(1 for r in results_3agree if r["correct"])
        total_3 = len(results_3agree)
        rets_3 = np.array([r["ret"] for r in results_3agree])
        m3 = compute_metrics(rets_3)
        print(f"\n  3人一致（全員同意見）:")
        print(f"    トレード数: {total_3}")
        print(f"    正解率: {correct_3}/{total_3} ({correct_3/total_3*100:.1f}%)")
        print(f"    PF: {m3['pf']:.2f}, 期待値: {m3['exp_value_net']:+.6f}")

    if results_2agree and results_3agree:
        diff = correct_3/total_3*100 - correct_2/total_2*100
        print(f"\n  → 3人一致は2人一致より正解率が {diff:+.1f}% 高い")

    print(f"\n{'='*60}")
    print("【分析2: 時間減衰ウェイトの効果（半減期1年）】")
    print(f"{'='*60}")
    print("  古いデータの重み: 1年前→50%, 2年前→25%, 3年前→12.5%")

    if results_weighted:
        mw = compute_metrics(np.array(results_weighted))
        print(f"\n  時間減衰あり（全トレード）:")
        print(f"    トレード: {mw['n_trades']}, PF: {mw['pf']:.2f}, "
              f"勝率: {mw['win_rate']:.1f}%, 期待値: {mw['exp_value_net']:+.6f}")

    if results_weighted_3agree:
        mw3 = compute_metrics(np.array(results_weighted_3agree))
        print(f"\n  時間減衰あり + 3人一致のみ（最も厳選）:")
        print(f"    トレード: {mw3['n_trades']}, PF: {mw3['pf']:.2f}, "
              f"勝率: {mw3['win_rate']:.1f}%, 期待値: {mw3['exp_value_net']:+.6f}")


# ===== 実行 =====
print("5年データで詳細分析を実行中...\n")
df, fcols = prepare(use_5y=True)
print(f"データ: {len(df)}本 ({df.index.min()} ~ {df.index.max()})\n")
run_analysis(df, fcols, use_5y=True)
