# ドル円と金利関連データを取得するスクリプト
# 1時間足・日足・米国10年債利回り・日米金利差を取得して保存する

import os
import yfinance as yf
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

# ── パス設定 ──────────────────────────────
script_dir = Path(__file__).parent
data_dir = (script_dir / ".." / "data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)
path_1h = (data_dir / "usdjpy_1h.csv").resolve()
path_1d = (data_dir / "usdjpy_1d.csv").resolve()
path_us10 = (data_dir / "us_10y_rate.csv").resolve()
path_diff = (data_dir / "rate_diff.csv").resolve()

# .env から FRED_API_KEY を読み込む
load_dotenv(script_dir.parent / ".env")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── ドル円 1時間足データ ──────────────────
print("ドル円 1時間足データを取得中...")
df_1h = yf.download(
    tickers="USDJPY=X",
    period="2y",
    interval="1h",
    auto_adjust=True,
)
print(f"\n1時間足 取得件数：{len(df_1h)}件")
df_1h.to_csv(path_1h)
print(f"1時間足 保存完了：{path_1h}")

# ── ドル円 日足データ ─────────────────────
print("\nドル円 日足データを取得中...")
df_1d = yf.download(
    tickers="USDJPY=X",
    period="5y",
    interval="1d",
    auto_adjust=True,
)
print(f"\n日足 取得件数：{len(df_1d)}件")
df_1d.to_csv(path_1d)
print(f"日足 保存完了：{path_1d}")

# ── FRED 金利データ（米国10年債 & 日米金利差） ─────────
if not FRED_API_KEY:
    print("\n警告: FRED_API_KEY が設定されていないため、金利データ取得をスキップします。")
else:
    print("\nFRED から金利データを取得中...")
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    # 直近5年分を目安に取得
    start_str = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

    # 米国10年債利回り（DGS10） 日次
    params_us = {
        "series_id": "DGS10",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_str,
    }
    r_us = requests.get(base_url, params=params_us, timeout=30)
    r_us.raise_for_status()
    obs_us = r_us.json().get("observations", [])
    rows_us = []
    for o in obs_us:
        v = o.get("value", ".")
        if v == ".":
            continue
        rows_us.append((o.get("date"), float(v)))
    df_us = pd.DataFrame(rows_us, columns=["Date", "rate_us_10y"])
    df_us["Date"] = pd.to_datetime(df_us["Date"])
    df_us = df_us.set_index("Date").sort_index()
    df_us.to_csv(path_us10)
    print(f"米国10年債利回り 保存完了：{path_us10}")

    # 日本10年債利回り（月次, IRLTLT01JPM156N）
    params_jp = {
        "series_id": "IRLTLT01JPM156N",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_str,
    }
    r_jp = requests.get(base_url, params=params_jp, timeout=30)
    r_jp.raise_for_status()
    obs_jp = r_jp.json().get("observations", [])
    rows_jp = []
    for o in obs_jp:
        v = o.get("value", ".")
        if v == ".":
            continue
        rows_jp.append((o.get("date"), float(v)))
    df_jp = pd.DataFrame(rows_jp, columns=["Date", "rate_jp_10y"])
    df_jp["Date"] = pd.to_datetime(df_jp["Date"])
    df_jp = df_jp.set_index("Date").sort_index()

    # 日米金利差を計算（日本は月次なので前方埋めで日次に合わせる）
    df_rate = df_us.join(df_jp, how="left")
    df_rate["rate_jp_10y"] = df_rate["rate_jp_10y"].ffill()
    df_rate["rate_diff"] = df_rate["rate_us_10y"] - df_rate["rate_jp_10y"]
    df_rate.to_csv(path_diff)
    print(f"日米金利差 保存完了：{path_diff}")