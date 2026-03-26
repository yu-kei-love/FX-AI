"""
経済指標サプライズ特徴量モジュール

「サプライズ」= 実際の値が「直近の傾向」からどれだけズレたか
（本来は「市場予想との差」を使うべきだが、予想値は無料で入手困難。
 代わりに「直近の移動平均からのズレ」を使う。
 これでも「予想外に良かった/悪かった」をある程度捉えられる。）

取得する指標（全てFREDから無料で取得可能）:
  - UNRATE:   アメリカの失業率（月次）→ 雇用の健康度
  - CPIAUCSL: アメリカの消費者物価指数（月次）→ インフレの度合い
  - FEDFUNDS: アメリカの政策金利（月次）→ 金利の方向
  - PAYEMS:   アメリカの非農業部門雇用者数（月次）→ 雇用統計の本体
  - INDPRO:   アメリカの鉱工業生産指数（月次）→ 景気の強さ
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# .envからAPIキーを読み込む
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "data").resolve()

# 取得する経済指標の一覧
INDICATORS = {
    "UNRATE":   {"name": "失業率",         "direction": "lower_is_good"},
    "CPIAUCSL": {"name": "消費者物価指数", "direction": "neutral"},
    "FEDFUNDS": {"name": "政策金利",       "direction": "neutral"},
    "PAYEMS":   {"name": "非農業雇用者数", "direction": "higher_is_good"},
    "INDPRO":   {"name": "鉱工業生産",     "direction": "higher_is_good"},
}


def fetch_fred_series(series_id: str, start_date: str = "2020-01-01") -> pd.Series:
    """FREDから1つの経済指標の時系列データを取得する"""
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY が .env に設定されていません")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    observations = r.json().get("observations", [])

    rows = []
    for obs in observations:
        val = obs.get("value", ".")
        if val == ".":
            continue
        rows.append((obs["date"], float(val)))

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Value"].sort_index()


def compute_surprise(series: pd.Series, window: int = 6) -> pd.Series:
    """
    サプライズスコアを計算する

    サプライズ = (実際の値 - 直近N回の平均) / 直近N回の標準偏差

    つまり「普段と比べてどれだけ予想外か」を数値化する。
    +2.0 = 普段より大幅に高い
    -2.0 = 普段より大幅に低い
     0.0 = 普段通り
    """
    rolling_mean = series.rolling(window, min_periods=3).mean()
    rolling_std = series.rolling(window, min_periods=3).std()
    # 標準偏差が0（全く同じ値が続いた場合）はサプライズ0とする
    rolling_std = rolling_std.replace(0, np.nan)
    surprise = (series - rolling_mean) / rolling_std
    return surprise.fillna(0.0)


def fetch_all_indicators() -> pd.DataFrame:
    """全ての経済指標を取得し、サプライズスコアを計算する"""
    all_data = {}

    for series_id, info in INDICATORS.items():
        try:
            series = fetch_fred_series(series_id)
            surprise = compute_surprise(series)

            # 「低い方が良い」指標（失業率など）はサプライズの符号を反転
            # → プラス = 経済に良い方のサプライズ、に統一
            if info["direction"] == "lower_is_good":
                surprise = -surprise

            all_data[f"surprise_{series_id}"] = surprise
            all_data[f"value_{series_id}"] = series
            print(f"  {info['name']}({series_id}): {len(series)}件取得")
        except Exception as e:
            print(f"  {info['name']}({series_id}): 取得失敗 ({e})")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def save_indicators():
    """経済指標データを取得してCSVに保存する"""
    print("経済指標データを取得中...")
    df = fetch_all_indicators()
    if df.empty:
        print("データが取得できませんでした")
        return df

    path = DATA_DIR / "economic_indicators.csv"
    df.to_csv(path)
    print(f"保存完了: {path} ({len(df)}行)")
    return df


def load_indicators() -> pd.DataFrame:
    """保存済みの経済指標データを読み込む"""
    path = DATA_DIR / "economic_indicators.csv"
    if not path.exists():
        # ファイルがなければ取得して保存
        return save_indicators()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def add_surprise_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    1時間足のDataFrameに経済指標サプライズ特徴量を追加する

    月次データを1時間足に展開する方法:
    - 経済指標は月1回発表される
    - 発表後、次の発表まで同じ値を使う（前方補完）
    - つまり「直近の経済指標サプライズがどうだったか」を常に参照できる
    """
    try:
        df_econ = load_indicators()
    except Exception as e:
        print(f"経済指標データの読み込み失敗: {e}")
        # 失敗しても動作するようにゼロ埋めで返す
        for series_id in INDICATORS:
            df_hourly[f"surprise_{series_id}"] = 0.0
        df_hourly["surprise_composite"] = 0.0
        return df_hourly

    # サプライズカラムだけ取り出す
    surprise_cols = [c for c in df_econ.columns if c.startswith("surprise_")]

    # 月次データを1時間足のインデックスに展開（前方補完）
    for col in surprise_cols:
        series = df_econ[col].dropna()
        if series.empty:
            df_hourly[col] = 0.0
            continue
        # 月次の日付を1時間足のインデックスに合わせる
        df_hourly[col] = series.reindex(df_hourly.index, method="ffill").fillna(0.0)

    # 総合サプライズ（全指標の平均）
    if surprise_cols:
        existing = [c for c in surprise_cols if c in df_hourly.columns]
        df_hourly["surprise_composite"] = df_hourly[existing].mean(axis=1)
    else:
        df_hourly["surprise_composite"] = 0.0

    return df_hourly


# 直接実行時はデータ取得＆確認
if __name__ == "__main__":
    df = save_indicators()
    if not df.empty:
        print("\n【最新のサプライズスコア】")
        surprise_cols = [c for c in df.columns if c.startswith("surprise_")]
        latest = df[surprise_cols].iloc[-1]
        for col, val in latest.items():
            series_id = col.replace("surprise_", "")
            name = INDICATORS.get(series_id, {}).get("name", series_id)
            print(f"  {name}: {val:+.2f}")
