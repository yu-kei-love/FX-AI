# ===========================================
# fetch_dukascopy.py
# Dukascopyから本物の1分足データを取得して1時間足に変換
# 無料・登録不要・合法
# ===========================================

import sys
import lzma
import struct
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()

# Dukascopyの1分足キャンドルURL
# monthは0始まり (0=1月, 11=12月)
BASE_URL = "https://datafeed.dukascopy.com/datafeed/USDJPY"

# USD/JPYの価格はpip単位で格納（1/1000）
# 通貨ペアによって異なる: JPYペアは1/1000、その他は1/100000
PRICE_DIVISOR = 1000  # JPYペアの場合


def fetch_day(year, month, day):
    """1日分の1分足データを取得する"""
    # Dukascopyはmonthが0始まり
    m = month - 1
    url = f"{BASE_URL}/{year}/{m:02d}/{day:02d}/BID_candles_min_1.bi5"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) == 0:
            return None

        data = lzma.decompress(r.content)
        record_size = 24
        n_records = len(data) // record_size

        if n_records == 0:
            return None

        base_dt = datetime(year, month, day)
        rows = []
        for i in range(n_records):
            offset = i * record_size
            time_sec, o, h, l, c = struct.unpack(">5i", data[offset:offset+20])
            vol = struct.unpack(">f", data[offset+20:offset+24])[0]

            dt = base_dt + timedelta(seconds=time_sec)
            rows.append({
                "Datetime": dt,
                "Open": o / PRICE_DIVISOR,
                "High": h / PRICE_DIVISOR,
                "Low": l / PRICE_DIVISOR,
                "Close": c / PRICE_DIVISOR,
                "Volume": vol,
            })

        return pd.DataFrame(rows)
    except (lzma.LZMAError, struct.error):
        return None
    except requests.RequestException:
        return None


def fetch_range(start_date, end_date):
    """指定範囲の1分足データを取得する"""
    all_data = []
    current = start_date
    total_days = (end_date - start_date).days
    fetched = 0

    while current <= end_date:
        # 土日はスキップ（FX市場は休み）
        if current.weekday() < 5:
            df_day = fetch_day(current.year, current.month, current.day)
            if df_day is not None and len(df_day) > 0:
                all_data.append(df_day)
                fetched += len(df_day)

            # 進捗表示（10日ごと）
            days_done = (current - start_date).days
            if days_done % 10 == 0:
                print(f"  {current.strftime('%Y-%m-%d')} ... {fetched:,}本取得済み")

            # サーバーに負荷をかけないよう少し待つ
            time.sleep(0.1)

        current += timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df.set_index("Datetime").sort_index()
    return df


def resample_to_1h(df_1min):
    """1分足を1時間足に変換する"""
    df_1h = df_1min.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])
    return df_1h


if __name__ == "__main__":
    # 5年分: 2021年1月 ～ 2024年3月（既存の2年分の手前まで）
    # 既存データ: 2024-03-25 ～ 2026-03-23
    start = datetime(2021, 1, 4)  # 2021年最初の営業日
    end = datetime(2024, 3, 24)   # 既存データの直前

    print(f"Dukascopy USD/JPY 1分足データ取得")
    print(f"期間: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
    print(f"（約{(end-start).days}日分 → 1時間足に変換）")
    print()

    # 1分足を取得
    df_1min = fetch_range(start, end)
    if df_1min.empty:
        print("データが取得できませんでした")
        sys.exit(1)

    print(f"\n1分足: {len(df_1min):,}本取得完了")
    print(f"  期間: {df_1min.index.min()} ~ {df_1min.index.max()}")

    # 1時間足に変換
    df_1h = resample_to_1h(df_1min)
    print(f"\n1時間足に変換: {len(df_1h):,}本")

    # 既存の2年分データと結合
    existing_path = DATA_DIR / "usdjpy_1h.csv"
    df_existing = pd.read_csv(
        existing_path, skiprows=3,
        names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
    )
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df_existing[c] = pd.to_numeric(df_existing[c], errors="coerce")
    df_existing["Datetime"] = pd.to_datetime(df_existing["Datetime"].astype(str).str.slice(0, 19), errors="coerce")
    df_existing = df_existing.dropna(subset=["Datetime", "Close"]).set_index("Datetime").sort_index()
    print(f"既存データ: {len(df_existing):,}本 ({df_existing.index.min()} ~ {df_existing.index.max()})")

    # 重複を除いて結合（本物データを優先）
    df_combined = pd.concat([df_1h, df_existing]).sort_index()
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]

    print(f"\n結合結果: {len(df_combined):,}本")
    print(f"  期間: {df_combined.index.min()} ~ {df_combined.index.max()}")

    # 保存
    save_path = DATA_DIR / "usdjpy_1h_5y_real.csv"
    df_combined.to_csv(save_path)
    print(f"\n保存完了: {save_path}")
