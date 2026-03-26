# ===========================================
# 10_regime_mode_switcher.py
# HMMレジームに応じてモードを自動切替する統合システム
# モデルA（トレンド）・B（レンジ）・C（高ボラ待機）を同時学習
# ===========================================

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from hmmlearn.hmm import GaussianHMM

# ----- パス設定 -----
script_dir = Path(__file__).resolve().parent
data_path = (script_dir / ".." / "data" / "usdjpy_1h.csv").resolve()

# ----- データ読み込み（05/09と同じ） -----
df = pd.read_csv(
    data_path,
    skiprows=3,
    names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Datetime"] = df["Datetime"].astype(str).str.slice(0, 19)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime", "Close"])
df = df.set_index("Datetime")
df = df.sort_index()

# ----- HMMレジーム（05と同じ） -----
df["Return"] = df["Close"].pct_change(24)
df["Volatility"] = df["Return"].rolling(24).std()
df_clean = df.dropna(subset=["Return", "Volatility"])
X_hmm = df_clean[["Return", "Volatility"]].values
model_hmm = GaussianHMM(
    n_components=3, covariance_type="full", n_iter=100, random_state=42,
)
model_hmm.fit(X_hmm)
states = model_hmm.predict(X_hmm)
df["Regime"] = np.nan
df.loc[df_clean.index, "Regime"] = states
df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)
df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1

# ----- 特徴量（05と同じ） -----
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss.replace(0, np.nan))
df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
df["BB_upper"] = sma20 + 2 * std20
df["BB_lower"] = sma20 - 2 * std20
df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, np.nan)
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_25"] = df["Close"].rolling(25).mean()
df["MA_75"] = df["Close"].rolling(75).mean()
df["Return_1"] = df["Close"].pct_change(1)
df["Return_3"] = df["Close"].pct_change(3)
df["Return_6"] = df["Close"].pct_change(6)
df["Return_24"] = df["Close"].pct_change(24)
ret_1h = df["Close"].pct_change(1)
df["Volatility_24"] = ret_1h.rolling(24).std()
df["Hour"] = df.index.hour
df["DayOfWeek"] = df.index.dayofweek
df["Close_4h_later"] = df["Close"].shift(-4)
df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)

feature_cols = [
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "MA_5", "MA_25", "MA_75",
    "Return_1", "Return_3", "Return_6", "Return_24",
    "Volatility_24", "Hour", "DayOfWeek",
    "Regime", "Regime_changed", "Regime_duration",
]
df = df.dropna(subset=feature_cols + ["Label"])
X = df[feature_cols]
y_direction = df["Label"]
close_arr = df["Close"].values
n_total = len(df)

# ----- Triple-Barrier ラベル（モデルA用） -----
# 上限+0.5%（利確）、下限-0.3%（損切り）、時間24時間。利確=1、損切り=0、時間切れ=2（学習から除外）
BARRIER_UP = 0.005
BARRIER_DOWN = -0.003
BARRIER_T = 24
y_triple = np.full(n_total, np.nan)
for i in range(n_total - BARRIER_T):
    c0 = close_arr[i]
    label = 2
    for t in range(1, BARRIER_T + 1):
        ret = (close_arr[i + t] - c0) / c0
        if ret >= BARRIER_UP:
            label = 1
            break
        if ret <= BARRIER_DOWN:
            label = 0
            break
    y_triple[i] = label
df["Label_triple"] = y_triple

# ----- 学習・テスト分割（80% / 20%） -----
split_idx = int(n_total * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train_dir = y_direction.iloc[:split_idx]
y_test_dir = y_direction.iloc[split_idx:]
regime_test = df["Regime"].iloc[split_idx:].values
close_test = df["Close"].iloc[split_idx:].values

# ----- モデルA：Triple-Barrier（時間切れは除外して学習） -----
mask_a = (df["Label_triple"] == 0) | (df["Label_triple"] == 1)
mask_a_train = mask_a.iloc[:split_idx]
X_a = X_train[mask_a_train]
y_a = (df.loc[X_a.index, "Label_triple"]).astype(int)
model_a = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_a.fit(X_a, y_a)

# ----- モデルB：通常ラベル + Meta-Labeling（採用率40%以下に絞る） -----
model_b_primary = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_b_primary.fit(X_train, y_train_dir)
proba_b_train = model_b_primary.predict_proba(X_train)[:, 1]
X_train_meta = X_train.copy()
X_train_meta["primary_proba"] = proba_b_train
y_meta = (model_b_primary.predict(X_train) == y_train_dir.values).astype(int)
model_b_secondary = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_b_secondary.fit(X_train_meta, y_meta)
X_test_meta = X_test.copy()
X_test_meta["primary_proba"] = model_b_primary.predict_proba(X_test)[:, 1]
proba_adopt_test = model_b_secondary.predict_proba(X_test_meta)[:, 1]
# 採用率40%：二次モデル確率の上位40%のみ採用
thresh_40 = np.percentile(proba_adopt_test, 100 - 40)
adopt_b = (proba_adopt_test >= thresh_40)  # 1次元 boolean 配列

# ----- モデルC用：過去平均（学習期間） -----
hist_avg_vol = df["Volatility_24"].iloc[:split_idx].mean()
hist_avg_abs_ret = df["Return_1"].abs().iloc[:split_idx].mean()
if hist_avg_abs_ret == 0:
    hist_avg_abs_ret = 1e-8
# 直近3バーでのレジーム変化回数（モデルC 条件2用）
regime_changes_rolling = (df["Regime"].diff().fillna(0) != 0).astype(int).rolling(3).sum()
df["Regime_changes_3h"] = regime_changes_rolling.fillna(0)
abs_ret_1h = df["Return_1"].abs()

# ----- テスト期間でレジームに応じて A/B/C を切替えて評価 -----
# レジーム0=トレンド→A、1=レンジ→B（採用時のみ）、2=高ボラ→C（待機判定、待機でなければA）
pred_a_test = model_a.predict(X_test)
pred_b_test = model_b_primary.predict(X_test)
y_test_values = y_test_dir.values
n_test = len(y_test_values)
vol_test = df["Volatility_24"].iloc[split_idx:].values
regime_changes_3h_test = df["Regime_changes_3h"].iloc[split_idx:].values
abs_ret_test = abs_ret_1h.iloc[split_idx:].values

# モデルC：待機判定（3条件のいずれかで待機）
cond1 = vol_test > (2 * hist_avg_vol)
cond2 = regime_changes_3h_test >= 2
cond3 = abs_ret_test > (3 * hist_avg_abs_ret)
wait_c = cond1 | cond2 | cond3

# 各サンプルでどのモードで予測するか・予測値
pred_final = np.full(n_test, np.nan)
mode_used = np.full(n_test, -1)  # 0=A, 1=B, 2=wait(C), 3=B不採用

for i in range(n_test):
    r = regime_test[i]
    if r == 0:
        pred_final[i] = pred_a_test[i]
        mode_used[i] = 0
    elif r == 1:
        if adopt_b[i]:
            pred_final[i] = pred_b_test[i]
            mode_used[i] = 1
        else:
            mode_used[i] = 3
    else:
        if wait_c[i]:
            mode_used[i] = 2
        else:
            pred_final[i] = pred_a_test[i]
            mode_used[i] = 0

# 予測を出したサンプルのみで正解率を集計
mask_pred = ~np.isnan(pred_final)
n_pred = mask_pred.sum()
correct_overall = (pred_final[mask_pred] == y_test_values[mask_pred]).sum()

# 各モードの発動回数と正解率
print("【各モードの発動回数と正解率】")
for mode_id, name in [(0, "A(トレンド)"), (1, "B(レンジ)")]:
    m = (mode_used == mode_id)
    cnt = m.sum()
    if cnt > 0:
        acc = (pred_final[m] == y_test_values[m]).mean()
        print("  {}: 発動 {} 回, 正解率 {:.4f} ({:.2f}%)".format(name, cnt, acc, acc * 100))
    else:
        print("  {}: 発動 0 回".format(name))

# 待機モード（C）
wait_count = (mode_used == 2).sum()
wait_rate = wait_count / n_test
print("  待機モード(C): 発動 {} 回, 発動率 {:.4f} ({:.2f}%)".format(wait_count, wait_rate, wait_rate * 100))
print("  B不採用（レンジで採用されず）: {} 回".format((mode_used == 3).sum()))

# 全体の正解率と09との比較
acc_overall = correct_overall / n_pred if n_pred > 0 else 0.0
acc_09 = 0.5236
diff = acc_overall - acc_09
print("\n【統合評価】")
print("  予測を行ったサンプル数: {} / {}".format(int(n_pred), n_test))
print("  全体の正解率: {:.4f} ({:.2f}%)".format(acc_overall, acc_overall * 100))
print("  09の結果（52.36%）との差: {:.4f} ({:+.2f}%)".format(diff, diff * 100))

# ----- MLflow 記録 -----
mlflow.set_experiment("fx_ai_phase1")
with mlflow.start_run():
    mlflow.log_param("script", "10_regime_mode_switcher")
    mlflow.log_param("barrier_up", BARRIER_UP)
    mlflow.log_param("barrier_down", BARRIER_DOWN)
    mlflow.log_param("barrier_t", BARRIER_T)
    mlflow.log_param("adoption_rate_target", 0.4)
    mlflow.log_metric("test_accuracy_overall", acc_overall)
    mlflow.log_metric("accuracy_diff_vs_09", diff)
    mlflow.log_metric("n_mode_a", int((mode_used == 0).sum()))
    mlflow.log_metric("n_mode_b", int((mode_used == 1).sum()))
    mlflow.log_metric("n_wait", int(wait_count))
    mlflow.log_metric("wait_rate", wait_rate)
    mlflow.log_metric("n_predictions", int(n_pred))
    if (mode_used == 0).sum() > 0:
        mlflow.log_metric("accuracy_mode_a", (pred_final[mode_used == 0] == y_test_values[mode_used == 0]).mean())
    if (mode_used == 1).sum() > 0:
        mlflow.log_metric("accuracy_mode_b", (pred_final[mode_used == 1] == y_test_values[mode_used == 1]).mean())
    mlflow.sklearn.log_model(model_a, "model_a")
    mlflow.sklearn.log_model(model_b_primary, "model_b_primary")
    mlflow.sklearn.log_model(model_b_secondary, "model_b_secondary")
    print("\nMLflow に記録しました（実験名: fx_ai_phase1）")
