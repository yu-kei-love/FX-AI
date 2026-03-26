# ===========================================
# 03_hmm_regime.py
# HMMでドル円のレジーム（5種類）を推定
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

# ----- パス設定 -----
script_dir = Path(__file__).resolve().parent
data_path = (script_dir / ".." / "data" / "usdjpy_1h.csv").resolve()

# ----- CSV の読み込み -----
df = pd.read_csv(
    data_path,
    skiprows=3,
    names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
)

# Close を数値に変換
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Datetime は先頭19文字（YYYY-MM-DD HH:MM:SS）で切り取り、日時に変換してインデックスに
df["Datetime"] = df["Datetime"].astype(str).str.slice(0, 19)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime", "Close"])
df = df.set_index("Datetime")

# ----- 特徴量の計算 -----
# 1. リターン：Close の前日比変化率（1時間足なので24時間前との変化率）
df["Return"] = df["Close"].pct_change(24)

# 2. ボラティリティ：リターンの過去24時間の標準偏差
df["Volatility"] = df["Return"].rolling(24).std()

# 特徴量が揃うまで欠損を落とす
df_clean = df.dropna(subset=["Return", "Volatility"])

# ----- GaussianHMM で5状態に分類 -----
# n_components=5（強トレンド/弱トレンド/低ボラレンジ/高ボラレンジ/危機）、full 分散、反復100回
X = df_clean[["Return", "Volatility"]].values
model = GaussianHMM(
    n_components=5,
    covariance_type="full",
    n_iter=100,
    random_state=42,
)
model.fit(X)
states = model.predict(X)

# 分類結果を df に「Regime」列として追加（欠損部分は後で前穴埋め）
df["Regime"] = np.nan
df.loc[df_clean.index, "Regime"] = states
df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)

# ----- 各レジームの出現回数と割合を表示 -----
regime_counts = df["Regime"].value_counts().sort_index()
n_total = len(df)
regime_labels = {
    0: "強トレンド（大きく一方向に動く）",
    1: "弱トレンド（緩やかに一方向に動く）",
    2: "低ボラレンジ（狭いレンジで推移）",
    3: "高ボラレンジ（広いレンジで推移）",
    4: "危機・急変動（異常なボラティリティ）",
}
print("【レジーム別 出現回数と割合】")
for r in range(5):
    cnt = regime_counts.get(r, 0)
    pct = 100.0 * cnt / n_total
    name = regime_labels.get(r, f"レジーム {r}")
    print(f"  {r}: {cnt:>6} 件  ({pct:>5.2f}%)  - {name}")
print(f"  合計:     {n_total:>6} 件")

# ----- グラフ表示 -----
# 上段：ドル円終値の折れ線 / 下段：レジームを色分け（0=青、1=オレンジ、2=赤）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, height_ratios=[1.5, 0.8])

# 上段：終値
ax1.plot(df.index, df["Close"], linewidth=0.6, color="steelblue")
ax1.set_ylabel("価格（円）", fontsize=12)
ax1.set_title("ドル円 1時間足 終値", fontsize=12)
ax1.grid(True, alpha=0.3)

# 下段：レジーム（0〜4を5色で表示）
colors = ["blue", "orange", "green", "red", "purple"]
for r in range(5):
    mask = df["Regime"] == r
    ax2.fill_between(df.index, 0, 1, where=mask, color=colors[r], alpha=0.6, label=f"Regime {r}")
ax2.set_ylim(-0.05, 1.05)
ax2.set_ylabel("レジーム", fontsize=12)
ax2.set_xlabel("日付", fontsize=12)
ax2.legend(loc="upper right", ncol=3)
ax2.grid(True, alpha=0.3)

plt.suptitle("ドル円 1時間足 と HMM レジーム分類", fontsize=14)
plt.tight_layout()
plt.show()
