"""
データ読み込みモジュール
USD/JPY 1時間足・日足・金利データの読み込みと前処理
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "data").resolve()


def load_usdjpy_1h(use_5y: bool = False) -> pd.DataFrame:
    """USD/JPY 1時間足データを読み込み、インデックスをDatetimeに設定して返す

    use_5y=True: 5年分データ（2021～、古い部分は日足からの補間）を使う
    use_5y=False: 2年分の本物の1時間足データを使う（デフォルト）
    """
    if use_5y:
        # 本物の1時間足データを優先（Dukascopyから取得したもの）
        path_5y_real = DATA_DIR / "usdjpy_1h_5y_real.csv"
        path_5y = DATA_DIR / "usdjpy_1h_5y.csv"
        chosen = path_5y_real if path_5y_real.exists() else path_5y
        if chosen.exists():
            df = pd.read_csv(chosen, index_col=0, parse_dates=True)
            for c in ["Close", "High", "Low", "Open", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Close"]).sort_index()
            return df

    path = DATA_DIR / "usdjpy_1h.csv"
    df = pd.read_csv(
        path,
        skiprows=3,
        names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
    )
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Datetime"] = df["Datetime"].astype(str).str.slice(0, 19)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", "Close"])
    df = df.set_index("Datetime")
    df = df.sort_index()
    return df


def load_usdjpy_1d() -> pd.DataFrame:
    """USD/JPY 日足データを読み込む"""
    path = DATA_DIR / "usdjpy_1d.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df


def load_rates() -> pd.DataFrame:
    """米国10年債利回りと日米金利差を読み込む"""
    us_path = DATA_DIR / "us_10y_rate.csv"
    diff_path = DATA_DIR / "rate_diff.csv"
    df_us = pd.read_csv(us_path, parse_dates=["Date"]).set_index("Date").sort_index()
    df_diff = pd.read_csv(diff_path, parse_dates=["Date"]).set_index("Date").sort_index()
    df_rates = df_us.join(df_diff[["rate_diff"]], how="outer").sort_index()
    return df_rates


def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """1時間足DataFrameに金利関連特徴量を追加する"""
    try:
        df_rates = load_rates()
        df_rates["ma20"] = df_rates["rate_us_10y"].rolling(20).mean()
        df_rates["rate_trend"] = (df_rates["rate_us_10y"] > df_rates["ma20"]).astype(int)
        df_rates = df_rates.reindex(df.index, method="ffill")
        df["rate_us_10y"] = df_rates["rate_us_10y"]
        df["rate_diff"] = df_rates["rate_diff"]
        df["rate_trend"] = df_rates["rate_trend"]
    except Exception:
        df["rate_us_10y"] = 0.0
        df["rate_diff"] = 0.0
        df["rate_trend"] = 0
    return df


def add_daily_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """1時間足DataFrameに日足トレンド特徴量を追加する"""
    try:
        df_d = load_usdjpy_1d()
        close_d = df_d["Close"]
        ma200 = close_d.rolling(200).mean()
        df["trend_200d"] = (close_d > ma200).astype(int).reindex(df.index, method="ffill").fillna(0).astype(int)
        df["trend_strength"] = ((close_d - ma200) / ma200.replace(0, pd.NA)).reindex(df.index, method="ffill").fillna(0.0)
    except Exception:
        df["trend_200d"] = 0
        df["trend_strength"] = 0.0
    return df
