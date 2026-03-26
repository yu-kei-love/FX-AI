# ===========================================
# 07_walk_forward.py
# Walk-Forward最適化（過去で学習・未来でテストを繰り返す）
# 05 と同じデータ・特徴量を使用
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

# ----- ラベル（05と同じ） -----
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

# ----- Walk-Forward の設定 -----
# 学習期間 2000 時間（約3ヶ月）、テスト期間 500 時間（約3週間）、500 時間ずつ前進
TRAIN_SIZE = 2000
TEST_SIZE = 500
STEP_SIZE = 500

n = len(X)
# テスト終端が n を超えない範囲でウィンドウを生成
n_windows = max(0, (n - TRAIN_SIZE - TEST_SIZE) // STEP_SIZE + 1)
results = []  # (test_start_date, test_end_date, accuracy)

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)

print("【Walk-Forward 各ウィンドウ】")
print("  学習期間: {} 時間, テスト期間: {} 時間, ステップ: {} 時間\n".format(TRAIN_SIZE, TEST_SIZE, STEP_SIZE))

for k in range(n_windows):
    train_start = k * STEP_SIZE
    train_end = train_start + TRAIN_SIZE
    test_start = train_end
    test_end = min(test_start + TEST_SIZE, n)
    if test_end - test_start < TEST_SIZE:
        break
    X_train_w = X.iloc[train_start:train_end]
    y_train_w = y.iloc[train_start:train_end]
    X_test_w = X.iloc[test_start:test_end]
    y_test_w = y.iloc[test_start:test_end]
    model.fit(X_train_w, y_train_w)
    acc = (model.predict(X_test_w) == y_test_w).mean()
    start_date = X.index[test_start]
    end_date = X.index[test_end - 1]
    results.append((start_date, end_date, acc))
    print("  ウィンドウ {:3}: テスト {} ～ {}  正解率 = {:.4f} ({:.2f}%)".format(
        k + 1, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), acc, acc * 100
    ))

if not results:
    print("  ※ データ不足のためウィンドウがありません（学習{}h + テスト{}h が必要）".format(TRAIN_SIZE, TEST_SIZE))
else:
    accuracies = np.array([r[2] for r in results])
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    print("\n【全ウィンドウ 平均正解率】 {:.4f} ± {:.4f} ({:.2f}% ± {:.2f}%)".format(
        mean_acc, std_acc, mean_acc * 100, std_acc * 100
    ))

    # ----- 正解率の推移をグラフで表示 -----
    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = [r[1].strftime("%Y-%m-%d") for r in results]
    ax.plot(range(len(results)), accuracies, marker="o", linestyle="-", color="steelblue", markersize=6)
    ax.axhline(y=mean_acc, color="gray", linestyle="--", alpha=0.8, label="平均")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("正解率", fontsize=12)
    ax.set_xlabel("テスト期間（終了日）", fontsize=12)
    ax.set_title("Walk-Forward 正解率の推移（モデル安定性の確認）", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # ----- MLflow 記録 -----
    mlflow.set_experiment("fx_ai_phase1")
    with mlflow.start_run():
        mlflow.log_param("eval_method", "walk_forward")
        mlflow.log_param("train_size_hours", TRAIN_SIZE)
        mlflow.log_param("test_size_hours", TEST_SIZE)
        mlflow.log_param("step_size_hours", STEP_SIZE)
        mlflow.log_param("n_windows", len(results))
        for i, (_, _, a) in enumerate(results):
            mlflow.log_metric("window_{}_accuracy".format(i + 1), float(a))
        mlflow.log_metric("mean_accuracy", mean_acc)
        mlflow.log_metric("std_accuracy", std_acc)
        print("\nMLflow に記録しました（実験名: fx_ai_phase1）")
