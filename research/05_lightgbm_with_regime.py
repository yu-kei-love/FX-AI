# ===========================================
# 05_lightgbm_with_regime.py
# HMMレジームを特徴量に追加したLightGBM（04の拡張）
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

# ----- データ読み込み -----
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

# ----- HMMレジーム用の中間特徴量（03と同じ） -----
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

# 追加特徴量：レジームが変化したフラグ（0か1）
df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)

# 追加特徴量：同じレジームが続いている時間数
regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1

# ----- 04と同じ特徴量の作成 -----

# 1. RSI（14期間）
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss.replace(0, np.nan))
df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

# 2. MACD（短期12、長期26、シグナル9）
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

# 3. ボリンジャーバンド（20期間）
sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
df["BB_upper"] = sma20 + 2 * std20
df["BB_lower"] = sma20 - 2 * std20
df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, np.nan)

# 4. 移動平均（5・25・75期間）
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_25"] = df["Close"].rolling(25).mean()
df["MA_75"] = df["Close"].rolling(75).mean()

# 5. 前の足からのリターン（1・3・6・24期間）
df["Return_1"] = df["Close"].pct_change(1)
df["Return_3"] = df["Close"].pct_change(3)
df["Return_6"] = df["Close"].pct_change(6)
df["Return_24"] = df["Close"].pct_change(24)

# 6. ボラティリティ（過去24時間のリターン標準偏差）
ret_1h = df["Close"].pct_change(1)
df["Volatility_24"] = ret_1h.rolling(24).std()

# 7. 時間帯（0〜23）
df["Hour"] = df.index.hour
# 8. 曜日（0〜6、月=0）
df["DayOfWeek"] = df.index.dayofweek

# ----- ラベルの作成（04と同じ：4時間後が上がるか下がるか） -----
df["Close_4h_later"] = df["Close"].shift(-4)
df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)

# 特徴量リスト（04の特徴量 + Regime, Regime_changed, Regime_duration）
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

# ----- 時系列でデータ分割（最初80%＝学習、残り20%＝テスト） -----
n = len(X)
split_idx = int(n * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"学習件数: {len(X_train)}, テスト件数: {len(X_test)}")

# ----- LightGBM で学習 -----
model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)
model.fit(X_train, y_train)

# テストデータでの正解率
acc = (model.predict(X_test) == y_test).mean()
ACC_04 = 0.5080  # 04_lightgbm_model.py の結果
diff = acc - ACC_04
print(f"\n【テスト正解率】 {acc:.4f} ({acc*100:.2f}%)")
print(f"04の結果（{ACC_04*100:.2f}%）との差: {diff:+.4f} ({diff*100:+.2f}%)")

# 特徴量の重要度トップ10
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n【特徴量重要度 トップ10】")
for i, (name, imp) in enumerate(importance.head(10).items(), 1):
    print(f"  {i:2}. {name}: {imp}")

# ----- MLflow で結果を記録 -----
mlflow.set_experiment("fx_ai_phase1")
with mlflow.start_run():
    mlflow.log_param("model_type", "lightgbm_with_regime")
    mlflow.log_param("split_ratio", 0.8)
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("accuracy_diff_vs_04", diff)
    for name, imp in importance.items():
        mlflow.log_metric(f"importance_{name}", imp)
    mlflow.sklearn.log_model(model, "model")
    print("\nMLflow に記録しました（実験名: fx_ai_phase1）")
