# ===========================================
# 09_meta_labeling.py
# Meta-Labeling：一次モデル（方向）＋二次モデル（採用可否）の2段構成
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

# ----- データ読み込み（05と同じ） -----
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

# ----- ラベル（4時間後が上がるか下がるか） -----
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

# ----- 時系列で学習80%・テスト20%に分割 -----
n = len(X)
split_idx = int(n * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ----- 一次モデル（方向予測：4時間後が上がるか下がるか） -----
model_primary = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)
model_primary.fit(X_train, y_train)

# 一次モデルの予測と「クラス1の確率」（二次モデル用特徴量）
pred_primary_train = model_primary.predict(X_train)
pred_primary_test = model_primary.predict(X_test)
proba_primary_train = model_primary.predict_proba(X_train)[:, 1]
proba_primary_test = model_primary.predict_proba(X_test)[:, 1]

# ----- 二次モデル用データ -----
# 特徴量 = 元の特徴量 + 一次モデルの予測確率（採用可否の手がかり）
# ラベル = 一次モデルが正解したか（1=正解, 0=不正解）
X_train_meta = X_train.copy()
X_train_meta["primary_proba"] = proba_primary_train
X_test_meta = X_test.copy()
X_test_meta["primary_proba"] = proba_primary_test
y_meta_train = (pred_primary_train == y_train.values).astype(int)

model_secondary = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)
model_secondary.fit(X_train_meta, y_meta_train)

# 二次モデル：1=シグナル採用、0=不採用
pred_secondary_test = model_secondary.predict(X_test_meta)

# ----- テストデータで評価 -----
acc_primary_only = (pred_primary_test == y_test.values).mean()
adopt_mask = pred_secondary_test == 1
n_adopted = adopt_mask.sum()
if n_adopted > 0:
    acc_adopted = (pred_primary_test[adopt_mask] == y_test.values[adopt_mask]).mean()
else:
    acc_adopted = np.nan
adoption_rate = pred_secondary_test.mean()

print("【テストデータでの評価】")
print("  一次モデルのみの正解率:     {:.4f} ({:.2f}%)".format(acc_primary_only, acc_primary_only * 100))
print("  二次で採用したシグナルの正解率: {:.4f} ({:.2f}%)".format(acc_adopted, acc_adopted * 100) if n_adopted > 0 else "  二次で採用したシグナルの正解率: （採用0件のため算出不可）")
print("  採用率（全シグナル中採用した割合）: {:.4f} ({:.2f}%)".format(adoption_rate, adoption_rate * 100))
print("  採用シグナル数: {} / {}".format(int(n_adopted), len(y_test)))

# ----- MLflow 記録 -----
mlflow.set_experiment("fx_ai_phase1")
with mlflow.start_run():
    mlflow.log_param("model_type", "meta_labeling")
    mlflow.log_param("split_ratio", 0.8)
    mlflow.log_metric("test_accuracy_primary_only", acc_primary_only)
    mlflow.log_metric("test_accuracy_adopted", acc_adopted if n_adopted > 0 else 0.0)
    mlflow.log_metric("adoption_rate", adoption_rate)
    mlflow.log_metric("n_adopted", int(n_adopted))
    mlflow.log_metric("n_test", len(y_test))
    mlflow.sklearn.log_model(model_primary, "model_primary")
    mlflow.sklearn.log_model(model_secondary, "model_secondary")
    print("\nMLflow に記録しました（実験名: fx_ai_phase1）")
