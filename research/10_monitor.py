# ===========================================
# 10_monitor.py
# AIシステムの監視・異常停止（精度低下・連続損失・ボラ異常）
# データ・特徴量・モデルは09と同じ
# ===========================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from hmmlearn.hmm import GaussianHMM

# ----- パス設定 -----
script_dir = Path(__file__).resolve().parent
data_path = (script_dir / ".." / "data" / "usdjpy_1h.csv").resolve()

# ----- データ読み込み（09と同じ） -----
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

# ----- HMMレジーム（09と同じ） -----
df["Return"] = df["Close"].pct_change(24)
df["Volatility"] = df["Return"].rolling(24).std()
df_clean = df.dropna(subset=["Return", "Volatility"])
X_hmm = df_clean[["Return", "Volatility"]].values
model_hmm = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=100,
    random_state=42,
)
model_hmm.fit(X_hmm)
states = model_hmm.predict(X_hmm)
df["Regime"] = np.nan
df.loc[df_clean.index, "Regime"] = states
df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)
df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1

# ----- 特徴量（09と同じ） -----
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
y = df["Label"]

# ----- 学習80%・テスト20%（09と同じ） -----
n = len(X)
split_idx = int(n * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ----- 一次・二次モデル（09と同じ） -----
model_primary = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_primary.fit(X_train, y_train)
proba_primary_train = model_primary.predict_proba(X_train)[:, 1]
proba_primary_test = model_primary.predict_proba(X_test)[:, 1]
X_train_meta = X_train.copy()
X_train_meta["primary_proba"] = proba_primary_train
X_test_meta = X_test.copy()
X_test_meta["primary_proba"] = proba_primary_test
y_meta_train = (model_primary.predict(X_train) == y_train.values).astype(int)
model_secondary = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_secondary.fit(X_train_meta, y_meta_train)

pred_primary_test = model_primary.predict(X_test)
y_test_values = y_test.values
test_len = len(y_test_values)

# テスト期間の価格・ボラティリティ（df と X は同じ行対応）
df_test = df.iloc[split_idx:]
close_test = df_test["Close"].values
vol_test = df_test["Volatility_24"].values
times_test = df_test.index
hist_avg_vol = df.iloc[:split_idx]["Volatility_24"].mean()

# ----- ローリング監視：直近100件の予測精度 -----
# 精度が48%未満で警告、45%未満で停止フラグ（100件溜まるまで精度は算出しない）
correct = (pred_primary_test == y_test_values).astype(float)
rolling_acc = pd.Series(correct).rolling(100, min_periods=100).mean().values
warn_rolling = (~np.isnan(rolling_acc)) & (rolling_acc < 0.48)
stop_rolling = (~np.isnan(rolling_acc)) & (rolling_acc < 0.45)

# ----- 連続損失監視：3回連続外れで警告、5回連続で停止フラグ -----
wrong = pred_primary_test != y_test_values
consec_wrong = np.zeros(test_len)
for i in range(test_len):
    if wrong[i]:
        consec_wrong[i] = consec_wrong[i - 1] + 1 if i > 0 else 1
    else:
        consec_wrong[i] = 0
warn_consec = consec_wrong >= 3
stop_consec = consec_wrong >= 5

# ----- ボラティリティ異常：直近24hが過去平均の3倍超で停止フラグ -----
stop_vol = vol_test > (3 * hist_avg_vol)
# 過去平均が0のときは除く
if hist_avg_vol <= 0:
    stop_vol = np.zeros_like(stop_vol, dtype=bool)

# ----- 停止フラグ（いずれかで立つ） -----
stop_flag = stop_rolling | stop_consec | stop_vol

# ----- 停止フラグが立った回数と時期を表示 -----
stop_times = times_test[stop_flag]
stop_reasons = []
for i in np.where(stop_flag)[0]:
    reasons = []
    if stop_rolling[i]:
        reasons.append("ローリング精度<45%")
    if stop_consec[i]:
        reasons.append("5回連続損失")
    if stop_vol[i]:
        reasons.append("ボラ異常(3倍超)")
    stop_reasons.append(" / ".join(reasons))

print("【停止フラグ】")
print("  停止フラグが立った回数: {} 回".format(stop_flag.sum()))
if stop_flag.any():
    print("  時期と理由:")
    for t, r in zip(stop_times[:20], stop_reasons[:20]):  # 最大20件表示
        print("    {}  {}".format(t, r))
    if stop_flag.sum() > 20:
        print("    ... 他 {} 件".format(stop_flag.sum() - 20))
else:
    print("  時期: なし")

# ----- 監視結果をグラフで表示 -----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, height_ratios=[1.2, 1, 0.8])

# 上段：価格チャート
ax1.plot(times_test, close_test, linewidth=0.7, color="steelblue")
ax1.set_ylabel("価格（円）")
ax1.set_title("ドル円 1時間足（テスト期間）")
ax1.grid(True, alpha=0.3)

# 中段：ローリング精度（警告・停止ラインを表示）
ax2.plot(times_test, rolling_acc, linewidth=0.8, color="green", label="ローリング精度(100件)")
ax2.axhline(y=0.48, color="orange", linestyle="--", linewidth=1, label="警告ライン(48%)")
ax2.axhline(y=0.45, color="red", linestyle="--", linewidth=1, label="停止ライン(45%)")
ax2.set_ylabel("精度")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)

# 下段：ボラティリティ（3倍ライン表示）
ax3.plot(times_test, vol_test, linewidth=0.7, color="purple", label="直近24hボラティリティ")
ax3.axhline(y=3 * hist_avg_vol, color="red", linestyle="--", linewidth=1, label="停止ライン(過去平均の3倍)")
ax3.set_ylabel("ボラティリティ")
ax3.set_xlabel("日付")
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.3)

plt.suptitle("AIシステム 監視ダッシュボード（ローリング精度・連続損失・ボラ異常）", fontsize=12)
plt.tight_layout()
plt.show()

# ----- MLflow 記録 -----
mlflow.set_experiment("fx_ai_phase1")
with mlflow.start_run():
    mlflow.log_param("script", "10_monitor")
    mlflow.log_param("rolling_window", 100)
    mlflow.log_param("warn_accuracy_threshold", 0.48)
    mlflow.log_param("stop_accuracy_threshold", 0.45)
    mlflow.log_param("consec_warn", 3)
    mlflow.log_param("consec_stop", 5)
    mlflow.log_param("vol_multiple", 3)
    mlflow.log_metric("n_stop_flags", int(stop_flag.sum()))
    mlflow.log_metric("hist_avg_volatility", hist_avg_vol)
    mlflow.log_metric("test_mean_rolling_accuracy", float(np.nanmean(rolling_acc)))
    if stop_flag.any():
        mlflow.log_metric("first_stop_at", stop_times[0].timestamp())
    print("\nMLflow に記録しました（実験名: fx_ai_phase1）")
