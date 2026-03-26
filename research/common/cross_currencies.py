"""
他通貨ペア特徴量モジュール

USD/JPYの予測に有効な他の通貨ペアの情報を特徴量として追加する。
通貨は単独で動くのではなく、通貨間の力関係で動く。

例: EUR/USD・GBP/USDが同時にドル高方向 → USD/JPYも上がりやすい

使う通貨ペア:
  - EUR/USD: 世界最大の取引量。ドルの強弱の最良指標
  - EUR/JPY: 円単体の強弱（ドルを除外）
  - GBP/USD: ドルの強弱の補助指標
  - AUD/JPY: リスクオン・オフの代理変数
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "data").resolve()

# 取得する通貨ペア
CROSS_PAIRS = {
    "EURUSD=X": "eurusd",
    "EURJPY=X": "eurjpy",
    "GBPUSD=X": "gbpusd",
    "AUDJPY=X": "audjpy",
}

# 追加される特徴量カラム
CROSS_FEATURE_COLS = []
for prefix in CROSS_PAIRS.values():
    CROSS_FEATURE_COLS.extend([
        f"{prefix}_ret1",     # 1時間リターン
        f"{prefix}_ret6",     # 6時間リターン
        f"{prefix}_ret24",    # 24時間リターン
        f"{prefix}_rsi14",    # RSI(14)
        f"{prefix}_vol24",    # 24時間ボラティリティ
    ])
# ドル強弱の総合指標
CROSS_FEATURE_COLS.append("usd_strength")


def fetch_cross_data():
    """他通貨ペアの1時間足データをyfinanceから取得して保存する"""
    import yfinance as yf

    for ticker, prefix in CROSS_PAIRS.items():
        print(f"  {ticker} 取得中...")
        df = yf.download(ticker, period="2y", interval="1h")
        if hasattr(df.columns, 'droplevel'):
            df.columns = df.columns.droplevel(1)
        save_path = DATA_DIR / f"{prefix}_1h.csv"
        df.to_csv(save_path)
        print(f"    {len(df)} rows -> {save_path}")


def load_cross_pair(prefix):
    """保存済みの通貨ペアデータを読み込む"""
    path = DATA_DIR / f"{prefix}_1h.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def compute_cross_features(df_cross, prefix):
    """1つの通貨ペアから特徴量を計算する"""
    close = df_cross["Close"]

    features = pd.DataFrame(index=df_cross.index)
    features[f"{prefix}_ret1"] = close.pct_change(1)
    features[f"{prefix}_ret6"] = close.pct_change(6)
    features[f"{prefix}_ret24"] = close.pct_change(24)

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features[f"{prefix}_rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # ボラティリティ
    features[f"{prefix}_vol24"] = close.pct_change(1).rolling(24).std()

    return features


def add_cross_features(df):
    """1時間足DataFrameに他通貨ペアの特徴量を追加する

    データがない場合は0で埋める（モデル学習に影響しにくい）
    """
    all_features = []

    for ticker, prefix in CROSS_PAIRS.items():
        df_cross = load_cross_pair(prefix)
        if df_cross is not None and len(df_cross) > 0:
            features = compute_cross_features(df_cross, prefix)
            # USD/JPYのインデックスに合わせてリインデックス
            features = features.reindex(df.index, method="ffill")
            all_features.append(features)
        else:
            # データがない場合はゼロ埋め
            for col in [f"{prefix}_ret1", f"{prefix}_ret6", f"{prefix}_ret24",
                        f"{prefix}_rsi14", f"{prefix}_vol24"]:
                df[col] = 0.0

    # 通貨ペアの特徴量をマージ
    for features in all_features:
        for col in features.columns:
            df[col] = features[col]

    # ドル強弱指標: EUR/USDとGBP/USDのリターンの平均（マイナス=ドル高）
    eurusd_ret = df.get("eurusd_ret1", pd.Series(0, index=df.index))
    gbpusd_ret = df.get("gbpusd_ret1", pd.Series(0, index=df.index))
    df["usd_strength"] = -(eurusd_ret + gbpusd_ret) / 2

    # NaNを0で埋める
    for col in CROSS_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df
