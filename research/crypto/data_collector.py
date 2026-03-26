"""
Cryptocurrency Data Collection Pipeline
========================================
Fetches BTC/USDT, ETH/USDT OHLCV data from Binance public API,
plus Fear & Greed Index, funding rates, and open interest.
All endpoints are public (no API key required).

Usage:
    python data_collector.py
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # FX-AI/
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "crypto"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup — console + file
# ---------------------------------------------------------------------------
logger = logging.getLogger("crypto_collector")
logger.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = logging.FileHandler(LOG_DIR / "crypto_collector.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BINANCE_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FEAR_GREED_BASE = "https://api.alternative.me/fng"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "FX-AI-CryptoCollector/1.0"})

# Binance rate limit: 1200 request weight / minute.  Each klines call = 1-5.
# We stay conservative: sleep 0.3s between paginated requests.
RATE_LIMIT_SLEEP = 0.3


# ===================================================================
# Helper utilities
# ===================================================================

def _ms(dt: datetime) -> int:
    """Datetime → Unix milliseconds."""
    return int(dt.timestamp() * 1000)


def _dt(ms: int) -> datetime:
    """Unix milliseconds → UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _safe_get(url: str, params: dict | None = None,
              retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """GET with retry + exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = backoff ** attempt
                logger.warning("Rate limited (429). Sleeping %.1fs ...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = backoff ** attempt
            logger.warning("Request failed (%s). Retry %d/%d in %.1fs",
                           exc, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError("Should not reach here")


# ===================================================================
# 1. Binance OHLCV (public klines)
# ===================================================================

def fetch_binance_klines(symbol: str, interval: str,
                         start: datetime, end: datetime) -> pd.DataFrame:
    """
    Paginated fetch of Binance klines.
    Max 1000 candles per request; we loop with startTime/endTime.

    Parameters
    ----------
    symbol : e.g. "BTCUSDT", "ETHUSDT"
    interval : "1h" or "1d"
    start, end : UTC datetimes defining the range
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_rows: list[list] = []
    current_start_ms = _ms(start)
    end_ms = _ms(end)

    logger.info("Fetching Binance %s %s  %s → %s",
                symbol, interval,
                start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    while current_start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = _safe_get(url, params=params)
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        # Advance past the last candle's open time
        last_open_ms = data[-1][0]
        current_start_ms = last_open_ms + 1
        logger.debug("  fetched %d candles (total %d), last: %s",
                      len(data), len(all_rows), _dt(last_open_ms))

        if len(data) < 1000:
            break  # no more data

        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        logger.warning("No data returned for %s %s", symbol, interval)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    # Keep only what we need
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.rename(columns={"open_time": "timestamp"}, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("  → %d candles for %s %s", len(df), symbol, interval)
    return df


def fetch_binance_ohlcv(symbol: str = "BTCUSDT",
                         interval: str = "1h",
                         years_back: int = 2) -> pd.DataFrame:
    """Convenience wrapper: fetch N years of Binance data up to now."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years_back * 365)
    return fetch_binance_klines(symbol, interval, start, end)


# ===================================================================
# 2. CoinGecko fallback (BTC OHLC, limited granularity)
# ===================================================================

def fetch_coingecko_ohlc(coin_id: str = "bitcoin",
                         vs_currency: str = "usd",
                         days: int = 365) -> pd.DataFrame:
    """
    CoinGecko /coins/{id}/ohlc — free tier.
    Max 'days' depends on plan; granularity auto-selected by API:
      1-2 days → 30min, 3-30 days → 4h, 31+ days → 4 days.
    Useful only as a fallback / sanity check.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}

    logger.info("Fetching CoinGecko OHLC for %s (%d days)", coin_id, days)
    try:
        resp = _safe_get(url, params=params)
        data = resp.json()
    except Exception as exc:
        logger.error("CoinGecko OHLC failed: %s", exc)
        return pd.DataFrame()

    if not data or isinstance(data, dict):
        logger.warning("CoinGecko returned empty / error: %s",
                        str(data)[:200])
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["volume"] = np.nan  # CoinGecko OHLC doesn't include volume
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("  → %d candles from CoinGecko", len(df))
    return df


# ===================================================================
# 3. Fear & Greed Index (alternative.me)
# ===================================================================

def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
    """
    Fetch Crypto Fear & Greed Index history.
    limit=0 → all available history (typically ~2000+ days).
    """
    url = FEAR_GREED_BASE + "/"
    params = {"limit": str(limit), "format": "json"}

    logger.info("Fetching Fear & Greed Index (limit=%s)", limit)
    try:
        resp = _safe_get(url, params=params)
        payload = resp.json()
    except Exception as exc:
        logger.error("Fear & Greed fetch failed: %s", exc)
        return pd.DataFrame()

    data = payload.get("data", [])
    if not data:
        logger.warning("No Fear & Greed data returned")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Fields: value, value_classification, timestamp, time_until_update
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="s", utc=True
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["timestamp", "value", "value_classification"]].copy()
    df.rename(columns={"value": "fear_greed_value",
                        "value_classification": "fear_greed_class"},
              inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("  → %d days of Fear & Greed data", len(df))
    return df


# ===================================================================
# 4. Binance Futures — Funding Rate (public)
# ===================================================================

def fetch_funding_rate(symbol: str = "BTCUSDT",
                       days_back: int = 365 * 2) -> pd.DataFrame:
    """
    Fetch historical funding rate from Binance Futures.
    Endpoint: GET /fapi/v1/fundingRate  (public, max 1000 per request).
    Funding is settled every 8 hours → ~3 entries/day.
    """
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
    end_ms = _ms(datetime.now(timezone.utc))
    start_ms = _ms(datetime.now(timezone.utc) - timedelta(days=days_back))
    all_rows: list[dict] = []

    logger.info("Fetching funding rate for %s (%d days back)", symbol, days_back)

    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = _safe_get(url, params=params)
            data = resp.json()
        except Exception as exc:
            logger.error("Funding rate request failed: %s", exc)
            break

        if not data:
            break

        all_rows.extend(data)
        last_ts = data[-1]["fundingTime"]
        current_start = last_ts + 1

        if len(data) < 1000:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        logger.warning("No funding rate data for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["timestamp", "funding_rate"]].copy()
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("  → %d funding rate records", len(df))
    return df


# ===================================================================
# 5. Binance Futures — Open Interest (public)
# ===================================================================

def fetch_open_interest_hist(symbol: str = "BTCUSDT",
                              period: str = "1h",
                              days_back: int = 30) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance Futures.
    Endpoint: GET /futures/data/openInterestHist (public).
    Max 500 per request. Available periods: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d.
    Note: Binance only keeps ~30 days of granular OI history publicly.
    """
    url = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
    end_ms = _ms(datetime.now(timezone.utc))
    start_ms = _ms(datetime.now(timezone.utc) - timedelta(days=days_back))
    all_rows: list[dict] = []

    logger.info("Fetching open interest history for %s (%s, %d days)",
                symbol, period, days_back)

    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 500,
        }
        try:
            resp = _safe_get(url, params=params)
            data = resp.json()
        except Exception as exc:
            logger.error("Open interest request failed: %s", exc)
            break

        if not data or isinstance(data, dict):
            # API may return error dict
            logger.warning("OI response: %s", str(data)[:200])
            break

        all_rows.extend(data)
        last_ts = data[-1]["timestamp"]
        current_start = last_ts + 1

        if len(data) < 500:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        logger.warning("No open interest data for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["sumOpenInterest", "sumOpenInterestValue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["timestamp"] + [c for c in ["sumOpenInterest", "sumOpenInterestValue"]
                            if c in df.columns]
    df = df[keep].copy()
    df.rename(columns={
        "sumOpenInterest": "open_interest",
        "sumOpenInterestValue": "open_interest_value",
    }, inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("  → %d open interest records", len(df))
    return df


# ===================================================================
# 6. USDT → JPY conversion
# ===================================================================

def load_usdjpy_hourly() -> pd.Series:
    """
    Load USDJPY hourly close prices from the existing FX data.
    Returns a Series indexed by UTC timestamp.
    """
    path = PROJECT_ROOT / "data" / "usdjpy_1h.csv"
    if not path.exists():
        logger.warning("USDJPY file not found at %s", path)
        return pd.Series(dtype=float)

    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    # The CSV has multi-level header: (Price, Ticker) then (Close, USDJPY=X)
    # Flatten to get Close column
    close_col = None
    for col in df.columns:
        if "close" in str(col).lower():
            close_col = col
            break
    if close_col is None:
        # Try first numeric column
        close_col = df.columns[0]

    series = df[close_col].dropna()
    series.index = pd.to_datetime(series.index, utc=True)
    series.name = "usdjpy"
    logger.info("Loaded %d USDJPY hourly records", len(series))
    return series


def convert_usdt_to_jpy(df: pd.DataFrame,
                         usdjpy: pd.Series) -> pd.DataFrame:
    """
    Convert USDT-denominated OHLCV to JPY using USDJPY rates.
    Uses nearest-available USDJPY rate (forward-fill within 24h tolerance).

    Parameters
    ----------
    df : DataFrame with 'timestamp' column and OHLCV in USDT
    usdjpy : Series with DatetimeIndex of USDJPY close prices

    Returns
    -------
    DataFrame with _jpy columns added
    """
    if usdjpy.empty:
        logger.warning("No USDJPY data available — skipping JPY conversion")
        return df

    df = df.copy()
    df = df.set_index("timestamp")

    # Reindex USDJPY to match crypto timestamps (forward-fill, 24h limit)
    usdjpy_aligned = usdjpy.reindex(df.index, method="ffill",
                                      tolerance=pd.Timedelta("24h"))

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[f"{col}_jpy"] = df[col] * usdjpy_aligned

    df = df.reset_index()
    n_converted = df["close_jpy"].notna().sum() if "close_jpy" in df.columns else 0
    logger.info("  JPY conversion: %d / %d rows have USDJPY rate",
                n_converted, len(df))
    return df


# ===================================================================
# 7. Save utilities
# ===================================================================

def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to CSV in the crypto data directory."""
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved %s  (%d rows, %.1f KB)",
                filename, len(df), path.stat().st_size / 1024)
    return path


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    logger.info("=" * 60)
    logger.info("Crypto Data Collection Pipeline — START")
    logger.info("=" * 60)

    now = datetime.now(timezone.utc)

    # Load USDJPY for later conversion
    usdjpy = load_usdjpy_hourly()

    # ------------------------------------------------------------------
    # BTC/USDT 1h — past 2 years
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    btc_1h = fetch_binance_ohlcv("BTCUSDT", "1h", years_back=2)
    if btc_1h.empty:
        logger.warning("Binance BTC 1h failed, trying CoinGecko fallback...")
        btc_1h = fetch_coingecko_ohlc("bitcoin", "usd", days=730)

    if not btc_1h.empty:
        btc_1h = convert_usdt_to_jpy(btc_1h, usdjpy)
        save_csv(btc_1h, "btc_1h.csv")

    # ------------------------------------------------------------------
    # BTC/USDT 1d — past 5 years
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    btc_1d = fetch_binance_ohlcv("BTCUSDT", "1d", years_back=5)
    if btc_1d.empty:
        logger.warning("Binance BTC 1d failed, trying CoinGecko fallback...")
        btc_1d = fetch_coingecko_ohlc("bitcoin", "usd", days=365 * 5)

    if not btc_1d.empty:
        btc_1d = convert_usdt_to_jpy(btc_1d, usdjpy)
        save_csv(btc_1d, "btc_1d.csv")

    # ------------------------------------------------------------------
    # ETH/USDT 1h — past 2 years
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    eth_1h = fetch_binance_ohlcv("ETHUSDT", "1h", years_back=2)
    if not eth_1h.empty:
        eth_1h = convert_usdt_to_jpy(eth_1h, usdjpy)
        save_csv(eth_1h, "eth_1h.csv")

    # ------------------------------------------------------------------
    # Fear & Greed Index
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    fg = fetch_fear_greed(limit=0)
    if not fg.empty:
        save_csv(fg, "fear_greed.csv")

    # ------------------------------------------------------------------
    # Funding Rate (BTC, 2 years)
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    fr = fetch_funding_rate("BTCUSDT", days_back=365 * 2)
    if not fr.empty:
        save_csv(fr, "funding_rate.csv")

    # ------------------------------------------------------------------
    # Open Interest (BTC, ~30 days — API limitation)
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    oi = fetch_open_interest_hist("BTCUSDT", period="1h", days_back=30)
    if not oi.empty:
        save_csv(oi, "open_interest.csv")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Collection complete. Files in %s:", DATA_DIR)
    for f in sorted(DATA_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        logger.info("  %s  (%.1f KB)", f.name, size_kb)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
