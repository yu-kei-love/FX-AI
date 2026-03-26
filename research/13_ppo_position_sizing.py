# ===========================================
# 13_ppo_position_sizing.py
# PPOでポジションサイズを最適化（方向はLightGBM、サイズはPPO）
# ===========================================

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from hmmlearn.hmm import GaussianHMM
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt


# ----- FXポジションサイジング用 Gymnasium 環境 -----
class FXPositionSizingEnv(gym.Env):
    """
    状態: LightGBM予測確率, レジーム(0-2), 直近ボラ, 直近24hリターン, 現在ポジションサイズ
    行動: ポジションサイズ 0.0〜1.0（連続）
    報酬: 利益時 +リターン×サイズ、損失時 -リターン×サイズ×1.5、高ボラ×大ポジで追加ペナルティ
    """

    metadata = {"render_modes": []}

    def __init__(self, lgb_proba, regime, volatility, return_24h, return_4h, vol_75_train):
        super().__init__()
        self.lgb_proba = np.asarray(lgb_proba, dtype=np.float32)
        self.regime = np.asarray(regime, dtype=np.float32)
        self.volatility = np.asarray(volatility, dtype=np.float32)
        self.return_24h = np.asarray(return_24h, dtype=np.float32)
        self.return_4h = np.asarray(return_4h, dtype=np.float32)
        self.vol_75 = float(vol_75_train)
        self.n = len(self.lgb_proba)
        # return_4h が有効な範囲（t の次に t+4 が存在するため t <= n-5）
        self.max_t = self.n - 5
        assert self.max_t > 0, "データが短すぎます（return_4h のため5件以上必要）"
        # 状態: [予測確率, レジーム/2, ボラ正規化, 24hリターンクリップ, 現在ポジション]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._t = 0
        self._position = 0.0
        self._vol_scale = np.max(self.volatility) + 1e-8

    def _get_obs(self):
        vol_norm = self.volatility[self._t] / self._vol_scale
        ret24 = np.clip(self.return_24h[self._t], -0.05, 0.05) / 0.05 * 0.5 + 0.5
        return np.array([
            self.lgb_proba[self._t],
            self.regime[self._t] / 2.0,
            min(vol_norm, 1.0),
            ret24,
            self._position,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._position = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = float(np.clip(action[0], 0.0, 1.0))
        # 方向: 予測確率0.5以上ならロング(1)、未満ならショート(-1)
        direction = 1.0 if self.lgb_proba[self._t] >= 0.5 else -1.0
        ret4 = self.return_4h[self._t]
        # PnL = 4hリターン × 方向 × ポジションサイズ
        pnl = ret4 * direction * action
        if pnl >= 0:
            reward = pnl
        else:
            reward = pnl * 1.5
        # ボラが高くてポジションが大きいとき追加ペナルティ
        if self.volatility[self._t] >= self.vol_75 and action > 0.5:
            reward -= 0.01 * action
        self._position = action
        self._t += 1
        terminated = self._t >= self.max_t
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}


# ----- データ読み込み（11と同じベース・特徴量） -----
script_dir = Path(__file__).resolve().parent
data_path = (script_dir / ".." / "data" / "usdjpy_1h.csv").resolve()
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
df["Return_4h"] = (df["Close_4h_later"] - df["Close"]) / df["Close"]

feature_cols = [
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "MA_5", "MA_25", "MA_75",
    "Return_1", "Return_3", "Return_6", "Return_24",
    "Volatility_24", "Hour", "DayOfWeek",
    "Regime", "Regime_changed", "Regime_duration",
]
df = df.dropna(subset=feature_cols + ["Label", "Return_4h"])
X = df[feature_cols]
y_dir = df["Label"]
n_total = len(df)
split_idx = int(n_total * 0.8)

# ----- 11の最適パラメータでLightGBMを学習（方向予測＋確率） -----
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y_dir.iloc[:split_idx]
model_lgb = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_lgb.fit(X_train, y_train)
proba_train = model_lgb.predict_proba(X_train)[:, 1]
proba_test = model_lgb.predict_proba(X_test)[:, 1]
proba_all = np.concatenate([proba_train, proba_test])

# 環境用の配列（return_4h が有効な範囲まで）
regime_arr = df["Regime"].values
vol_arr = df["Volatility_24"].values
ret24_arr = df["Return_24"].values
ret4_arr = df["Return_4h"].values
vol_75_train = np.percentile(vol_arr[:split_idx], 75)

# ----- 学習用環境（80%データ） -----
def make_train_env():
    return FXPositionSizingEnv(
        proba_all[:split_idx],
        regime_arr[:split_idx],
        vol_arr[:split_idx],
        ret24_arr[:split_idx],
        ret4_arr[:split_idx],
        vol_75_train,
    )

# ----- PPO 学習（50000ステップ） -----
print("PPO でポジションサイズを学習中（50,000ステップ）...")
train_env = DummyVecEnv([make_train_env])
model_ppo = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=0,
    seed=42,
)
model_ppo.learn(total_timesteps=50_000)
train_env.close()

# ----- テストデータで評価 -----
def evaluate_cumulative_return(proba, regime, vol, ret24, ret4, vol_75, policy_fn):
    """policy_fn(obs) -> action でステップし、累積リターンを返す"""
    n = len(proba)
    max_t = n - 5
    rewards = []
    position = 0.0
    vol_scale = np.max(vol) + 1e-8
    for t in range(max_t):
        vol_norm = min(vol[t] / vol_scale, 1.0)
        ret24_norm = np.clip(ret24[t], -0.05, 0.05) / 0.05 * 0.5 + 0.5
        obs = np.array([proba[t], regime[t] / 2.0, vol_norm, ret24_norm, position], dtype=np.float32)
        action = policy_fn(obs)
        action = np.clip(float(action[0]), 0.0, 1.0)
        direction = 1.0 if proba[t] >= 0.5 else -1.0
        pnl = ret4[t] * direction * action
        reward = pnl if pnl >= 0 else pnl * 1.5
        if vol[t] >= vol_75 and action > 0.5:
            reward -= 0.01 * action
        rewards.append(reward)
        position = action
    return np.array(rewards)

# PPO ポリシー
def ppo_policy(obs):
    return model_ppo.predict(obs, deterministic=True)[0]

# LightGBMのみ（常にポジション1.0）
def lgb_only_policy(obs):
    return np.array([1.0], dtype=np.float32)

proba_test_arr = proba_test
regime_test_arr = regime_arr[split_idx:]
vol_test_arr = vol_arr[split_idx:]
ret24_test_arr = ret24_arr[split_idx:]
ret4_test_arr = ret4_arr[split_idx:]
vol_75_test = np.percentile(vol_test_arr, 75)

rewards_ppo = evaluate_cumulative_return(
    proba_test_arr, regime_test_arr, vol_test_arr,
    ret24_test_arr, ret4_test_arr, vol_75_test, ppo_policy,
)
rewards_lgb = evaluate_cumulative_return(
    proba_test_arr, regime_test_arr, vol_test_arr,
    ret24_test_arr, ret4_test_arr, vol_75_test, lgb_only_policy,
)

cum_ppo = np.cumsum(rewards_ppo)
cum_lgb = np.cumsum(rewards_lgb)
print("\n【テスト期間 累積リターン】")
print("  PPO（ポジションサイズ最適化）: {:.4f}".format(cum_ppo[-1]))
print("  LightGBMのみ（常時サイズ1.0）: {:.4f}".format(cum_lgb[-1]))

# ----- 累積リターンをグラフで表示 -----
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(cum_ppo))
ax.plot(x, cum_ppo, label="PPO（ポジションサイズ最適化）", color="green")
ax.plot(x, cum_lgb, label="LightGBMのみ（サイズ1.0）", color="blue", alpha=0.8)
ax.set_xlabel("ステップ（時間）")
ax.set_ylabel("累積リターン")
ax.set_title("テスト期間 累積リターン比較")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----- MLflow（実験名: fx_ai_phase5） -----
mlflow.set_experiment("fx_ai_phase5")
with mlflow.start_run():
    mlflow.log_param("script", "13_ppo_position_sizing")
    mlflow.log_param("total_timesteps", 50_000)
    mlflow.log_metric("test_cumulative_return_ppo", float(cum_ppo[-1]))
    mlflow.log_metric("test_cumulative_return_lgb_only", float(cum_lgb[-1]))
    mlflow.log_metric("test_mean_reward_ppo", float(np.mean(rewards_ppo)))
    mlflow.log_metric("test_mean_reward_lgb_only", float(np.mean(rewards_lgb)))
    mlflow.sklearn.log_model(model_lgb, "model_lgb")
    # PPO は sb3 の save/load 形式のため、ここではメトリクスのみ記録
    print("\nMLflow に記録しました（実験名: fx_ai_phase5）")
