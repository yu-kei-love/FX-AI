# ===========================================
# 08_pbo_dsr.py v3.0
# PBO（過学習確率）とDSR（Deflated Sharpe Ratio）
# v3.0: アンサンブル + 12h horizon + 交互作用特徴量
# ===========================================

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, FEATURE_COLS
from research.common.validation import compute_metrics
from research.common.ensemble import EnsembleClassifier

script_dir = Path(__file__).resolve().parent

# ===== データ読み込み（v3と同じ前処理） =====
print("データ読み込み中...")
df = load_usdjpy_1h()
df = add_technical_features(df)
df = add_rate_features(df)
df = add_daily_trend_features(df)

df["Return"] = df["Close"].pct_change(24)
df["Volatility"] = df["Return"].rolling(24).std()

# 交互作用特徴量（v3と同じ）
df["RSI_x_Vol"] = df["RSI_14"] * df["Volatility_24"]
df["MACD_norm"] = df["MACD"] / df["Volatility_24"].replace(0, np.nan)
bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
df["BB_position"] = (df["Close"] - df["BB_lower"]) / bb_range
df["MA_cross"] = (df["MA_5"] - df["MA_75"]) / df["Close"]
df["Momentum_accel"] = df["Return_1"] - df["Return_1"].shift(1)
df["Vol_change"] = df["Volatility_24"].pct_change(6)
df["HL_ratio"] = (df["High"] - df["Low"]) / df["Close"]
hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
df["Close_position"] = (df["Close"] - df["Low"]) / hl_range
df["Return_skew_12"] = df["Return_1"].rolling(12).apply(
    lambda x: (x > 0).sum() / len(x) - 0.5, raw=True
)

# 12h horizon
FORECAST_HORIZON = 12
df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]

interaction_cols = [
    "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
    "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
    "Return_skew_12",
]
base_feature_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]
feature_cols = base_feature_cols + interaction_cols

df = df.dropna(subset=feature_cols + ["Label", "Return_Nh"])
X = df[feature_cols]
y = df["Label"]
return_Nh = df["Return_Nh"]

# ===== IS / OOS 分割（80/20） =====
n = len(X)
split_idx = int(n * 0.8)
X_is, X_oos = X.iloc[:split_idx], X.iloc[split_idx:]
y_is, y_oos = y.iloc[:split_idx], y.iloc[split_idx:]
ret_is = return_Nh.iloc[:split_idx]
ret_oos = return_Nh.iloc[split_idx:]

ANNUALIZATION = math.sqrt(24 * 365)


def strategy_returns(pred, actual_return):
    return (2 * pred - 1) * actual_return


def sharpe_ratio(returns):
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return float(np.mean(r) / r.std() * ANNUALIZATION)


def profit_factor(returns):
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    profit = r[r > 0].sum()
    loss = r[r < 0].sum()
    if loss == 0:
        return float("inf") if profit > 0 else float("nan")
    return profit / -loss


# ===== パラメータセット（アンサンブルの設定を変える） =====
param_sets = [
    {"n_estimators": 100, "learning_rate": 0.01},
    {"n_estimators": 100, "learning_rate": 0.05},
    {"n_estimators": 100, "learning_rate": 0.1},
    {"n_estimators": 200, "learning_rate": 0.01},
    {"n_estimators": 200, "learning_rate": 0.05},
    {"n_estimators": 200, "learning_rate": 0.1},
    {"n_estimators": 300, "learning_rate": 0.01},
    {"n_estimators": 300, "learning_rate": 0.05},
    {"n_estimators": 300, "learning_rate": 0.1},
    {"n_estimators": 500, "learning_rate": 0.05},
]

# ===== 各モデルのIS/OOS指標を計算 =====
sharpe_is_list = []
sharpe_oos_list = []
pf_is_list = []
pf_oos_list = []

print(f"\nデータ: IS={split_idx}本, OOS={n-split_idx}本")
print(f"特徴量: {len(feature_cols)}個（交互作用含む）")
print(f"予測ホライゾン: {FORECAST_HORIZON}h")
print("\n【各アンサンブルモデルの評価】")

for i, params in enumerate(param_sets):
    ens = EnsembleClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
    )
    ens.fit(X_is.values, y_is.values)

    pred_is = ens.predict(X_is.values)
    pred_oos = ens.predict(X_oos.values)
    ret_s_is = strategy_returns(pred_is, ret_is.values)
    ret_s_oos = strategy_returns(pred_oos, ret_oos.values)

    s_is = sharpe_ratio(ret_s_is)
    s_oos = sharpe_ratio(ret_s_oos)
    p_is = profit_factor(ret_s_is)
    p_oos = profit_factor(ret_s_oos)

    sharpe_is_list.append(s_is)
    sharpe_oos_list.append(s_oos)
    pf_is_list.append(p_is)
    pf_oos_list.append(p_oos)

    print(f"  モデル{i+1:2}: n_est={params['n_estimators']:3}, lr={params['learning_rate']}"
          f"  IS: Sharpe={s_is:7.3f}, PF={p_is:5.2f}"
          f"  OOS: Sharpe={s_oos:7.3f}, PF={p_oos:5.2f}")

sharpe_is_arr = np.array(sharpe_is_list)
sharpe_oos_arr = np.array(sharpe_oos_list)
pf_is_arr = np.array(pf_is_list)
pf_oos_arr = np.array(pf_oos_list)

# ===== PBO（過学習確率）=====
N_STRATEGIES = len(param_sets)
B = 1000
rng = np.random.RandomState(42)

# Sharpeベース PBO
oos_rank_sharpe = np.argsort(np.argsort(-sharpe_oos_arr)) + 1
pbo_count_sharpe = 0
for _ in range(B):
    idx = rng.randint(0, N_STRATEGIES, size=N_STRATEGIES)
    best_is_idx = idx[np.argmax(sharpe_is_arr[idx])]
    if oos_rank_sharpe[best_is_idx] > N_STRATEGIES // 2:
        pbo_count_sharpe += 1
PBO_sharpe = pbo_count_sharpe / B

# PFベース PBO
pf_oos_clean = np.where(np.isinf(pf_oos_arr), 10.0, pf_oos_arr)
pf_is_clean = np.where(np.isinf(pf_is_arr), 10.0, pf_is_arr)
oos_rank_pf = np.argsort(np.argsort(-pf_oos_clean)) + 1
pbo_count_pf = 0
rng2 = np.random.RandomState(42)
for _ in range(B):
    idx = rng2.randint(0, N_STRATEGIES, size=N_STRATEGIES)
    best_is_idx = idx[np.argmax(pf_is_clean[idx])]
    if oos_rank_pf[best_is_idx] > N_STRATEGIES // 2:
        pbo_count_pf += 1
PBO_pf = pbo_count_pf / B

print(f"\n{'='*60}")
print(f"【PBO（過学習確率）】")
print(f"  Sharpeベース: {PBO_sharpe:.4f} ({PBO_sharpe*100:.1f}%)")
print(f"  PFベース:     {PBO_pf:.4f} ({PBO_pf*100:.1f}%)")

# ===== DSR =====
best_is_idx = int(np.argmax(sharpe_is_arr))
sharpe_oos_selected = sharpe_oos_arr[best_is_idx]
DSR = sharpe_oos_selected * math.sqrt(max(0, 1 - PBO_sharpe))
print(f"\n【DSR（Deflated Sharpe Ratio）】 {DSR:.4f}")
print(f"  IS最良モデル: n_est={param_sets[best_is_idx]['n_estimators']}, "
      f"lr={param_sets[best_is_idx]['learning_rate']}")
print(f"  OOS Sharpe: {sharpe_oos_selected:.4f}")

# ===== OOS最良モデルの詳細評価 =====
best_oos_idx = int(np.argmax(sharpe_oos_arr))
best_ens = EnsembleClassifier(
    n_estimators=param_sets[best_oos_idx]["n_estimators"],
    learning_rate=param_sets[best_oos_idx]["learning_rate"],
)
best_ens.fit(X_is.values, y_is.values)
pred_oos_best = best_ens.predict(X_oos.values)
ret_oos_best = strategy_returns(pred_oos_best, ret_oos.values)
metrics_oos = compute_metrics(ret_oos_best)

print(f"\n【OOS最良モデルの詳細評価】")
print(f"  モデル: n_est={param_sets[best_oos_idx]['n_estimators']}, "
      f"lr={param_sets[best_oos_idx]['learning_rate']}")
print(f"  トレード数: {metrics_oos['n_trades']}")
print(f"  勝率: {metrics_oos['win_rate']:.2f}%")
print(f"  PF: {metrics_oos['pf']:.2f}")
print(f"  MDD: {metrics_oos['mdd']:.2f}%")
print(f"  Sharpe: {metrics_oos['sharpe']:.2f}")
print(f"  手数料込み期待値: {metrics_oos['exp_value_net']:+.6f}")

# ===== 結果の解釈 =====
print(f"\n{'='*60}")
print("【結果の解釈】")
if PBO_sharpe > 0.5:
    print("  PBO(Sharpe) > 50%: バックテストの好結果は過学習の可能性が高い")
elif PBO_sharpe > 0.3:
    print("  PBO(Sharpe) 30-50%: 中程度の過学習リスク")
else:
    print("  PBO(Sharpe) <= 30%: 過学習リスクは低い")

if DSR < 0:
    print("  DSR < 0: 統計的に有意な優位性なし")
elif DSR < 1:
    print("  0 < DSR < 1: わずかな優位性の可能性あり（慎重に）")
else:
    print("  DSR >= 1: 補正後も一定の優位性あり")

# Walk-Forward結果と比較
print(f"\n【v3 Walk-Forward結果との比較】")
print(f"  PBO IS/OOS: Sharpe(IS最良のOOS)={sharpe_oos_selected:.2f}")
print(f"  v3 WF: PF=1.30, Sharpe=8.56, PF>=1.0 in 6/10 windows")
print(f"  → WFの6/10 windows PF>=1.0 は、PBO 40%相当")
