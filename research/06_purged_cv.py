# ===========================================
# 06_purged_cv.py
# Purged K-Fold CV で時系列リークを防いだ評価
# 05 と同じデータ・特徴量を使用
# ===========================================

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from hmmlearn.hmm import GaussianHMM


# ----- Purged K-Fold クラス -----
# 学習データとテストデータの間に禁止期間（Embargo）を設け、時系列リークを防ぐ
class PurgedKFold:
    """
    時系列向け Purged K-Fold。
    テスト区間の直前に embargo_size サンプル、直後に embargo_size サンプルを
    学習から除外する。
    """

    def __init__(self, n_splits=5, embargo_size=24):
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def split(self, X, y=None):
        n = len(X)
        # 時系列を K 個の連続ブロックに分割
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(np.concatenate([[0], fold_sizes]))
        for k in range(self.n_splits):
            test_start = int(starts[k])
            test_end = int(starts[k + 1])
            test_idx = np.arange(test_start, test_end)
            # 学習側：テスト区間の直前・直後を embargo で除外
            train_before_end = max(0, test_start - self.embargo_size)
            train_after_start = min(n, test_end + self.embargo_size)
            train_idx = np.concatenate([
                np.arange(0, train_before_end),
                np.arange(train_after_start, n),
            ])
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx


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

# ----- Purged K-Fold CV -----
# n_splits=5, embargo_size=24（24時間分をバッファとして除外）
purged_cv = PurgedKFold(n_splits=5, embargo_size=24)
fold_accuracies = []
model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)

for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
    X_train_f, X_test_f = X.iloc[train_idx], X.iloc[test_idx]
    y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train_f, y_train_f)
    acc_f = (model.predict(X_test_f) == y_test_f).mean()
    fold_accuracies.append(acc_f)
    print(f"  Fold {fold + 1}: 正解率 = {acc_f:.4f} ({acc_f*100:.2f}%)")

fold_accuracies = np.array(fold_accuracies)
mean_acc = float(np.mean(fold_accuracies))
std_acc = float(np.std(fold_accuracies))
print(f"\n【Purged CV 平均正解率】 {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")

# ----- 05の結果（52.36%）と比較 -----
ACC_05 = 0.5236
diff = mean_acc - ACC_05
print(f"05の結果（{ACC_05*100:.2f}%）との差: {diff:+.4f} ({diff*100:+.2f}%)")

# ----- MLflow 記録 -----
mlflow.set_experiment("fx_ai_phase1")
with mlflow.start_run():
    mlflow.log_param("eval_method", "purged_kfold")
    mlflow.log_param("n_splits", 5)
    mlflow.log_param("embargo_size", 24)
    for i, a in enumerate(fold_accuracies):
        mlflow.log_metric(f"fold_{i+1}_accuracy", float(a))
    mlflow.log_metric("mean_accuracy", mean_acc)
    mlflow.log_metric("std_accuracy", std_acc)
    mlflow.log_metric("accuracy_diff_vs_05", diff)
    print("\nMLflow に記録しました（実験名: fx_ai_phase1）")
