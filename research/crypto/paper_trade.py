"""
Crypto Paper Trading System
============================
Runs the trained hybrid model on live BTC/USDT data from Binance (public API).
Logs all signals and simulated trades to CSV for out-of-sample validation.

Modes:
  --once       : Run one prediction cycle and exit
  --loop       : Run continuously every FORECAST_HORIZON hours
  --backfill N : Simulate the last N hours from saved data (for testing)

Usage:
    python paper_trade.py --once
    python paper_trade.py --loop
    python paper_trade.py --backfill 720
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "crypto"
MODEL_DIR = Path(__file__).resolve().parent / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG = DATA_DIR / "paper_trades.csv"
PORTFOLIO_LOG = DATA_DIR / "paper_portfolio.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "paper_trade.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("paper_trade")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INITIAL_CAPITAL = 10000.0  # USD
FORECAST_HORIZON_HOURS = 12  # matched to hybrid_model.py FORECAST_HORIZON
LOOKBACK_HOURS = 1000  # need ~610 for feature warmup + 168 for LSTM sequence


def fetch_klines(symbol: str = SYMBOL, interval: str = "1h", limit: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Fetch recent klines from Binance public API."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.set_index("timestamp").sort_index()
    return df


def fetch_eth_klines(interval: str = "1h", limit: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Fetch ETH/USDT klines for cross-asset features."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": "ETHUSDT", "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["eth_close"] = df["Close"].astype(float)
    df = df[["timestamp", "eth_close"]].set_index("timestamp").sort_index()
    return df


def fetch_funding_rate(symbol: str = SYMBOL, limit: int = 100) -> pd.DataFrame:
    """Fetch recent funding rates from Binance Futures."""
    url = f"https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].set_index("timestamp").sort_index()
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch funding rate: {e}")
        return pd.DataFrame(columns=["funding_rate"])


def fetch_fear_greed() -> pd.DataFrame:
    """Fetch Fear & Greed index."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=30", timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]

        records = []
        for item in data:
            records.append({
                "timestamp": pd.to_datetime(int(item["timestamp"]), unit="s", utc=True),
                "fear_greed": int(item["value"]),
            })
        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch Fear&Greed: {e}")
        return pd.DataFrame(columns=["fear_greed"])


class PaperTrader:
    """Simulated trader that logs signals and tracks portfolio."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None  # {"direction": "BUY"/"SELL", "entry_price": float, "size_usd": float, "entry_time": str}
        self.trade_history = []
        self.portfolio_history = []

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Resume from existing trade log if present."""
        if TRADE_LOG.exists():
            try:
                df = pd.read_csv(TRADE_LOG)
                if len(df) > 0:
                    self.trade_history = df.to_dict("records")
                    # Reconstruct capital from completed trades
                    completed = df[df["status"] == "CLOSED"]
                    if len(completed) > 0:
                        self.capital = self.initial_capital + completed["pnl"].sum()
                    # Check for open position
                    open_trades = df[df["status"] == "OPEN"]
                    if len(open_trades) > 0:
                        last_open = open_trades.iloc[-1]
                        self.position = {
                            "direction": last_open["direction"],
                            "entry_price": last_open["entry_price"],
                            "size_usd": last_open["size_usd"],
                            "entry_time": last_open["entry_time"],
                        }
                    logger.info(f"Resumed: capital=${self.capital:.2f}, trades={len(completed)}, open={self.position is not None}")
            except Exception as e:
                logger.warning(f"Could not load trade history: {e}")

        if PORTFOLIO_LOG.exists():
            try:
                df = pd.read_csv(PORTFOLIO_LOG)
                self.portfolio_history = df.to_dict("records")
            except Exception:
                pass

    def _save_trades(self):
        """Persist trade log to CSV."""
        if self.trade_history:
            pd.DataFrame(self.trade_history).to_csv(TRADE_LOG, index=False)

    def _save_portfolio(self):
        """Persist portfolio snapshot."""
        if self.portfolio_history:
            pd.DataFrame(self.portfolio_history).to_csv(PORTFOLIO_LOG, index=False)

    def close_position(self, current_price: float, timestamp: str, reason: str = "signal"):
        """Close current open position."""
        if self.position is None:
            return

        entry = self.position["entry_price"]
        size_usd = self.position["size_usd"]
        direction = self.position["direction"]

        if direction == "BUY":
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry

        pnl_usd = size_usd * pnl_pct - size_usd * 0.002  # transaction cost

        self.capital += pnl_usd

        # Update the open trade record to CLOSED
        for trade in reversed(self.trade_history):
            if trade.get("status") == "OPEN":
                trade["exit_price"] = current_price
                trade["exit_time"] = timestamp
                trade["pnl"] = round(pnl_usd, 2)
                trade["pnl_pct"] = round(pnl_pct * 100, 4)
                trade["status"] = "CLOSED"
                trade["close_reason"] = reason
                break

        logger.info(
            f"CLOSED {direction}: entry={entry:.2f} exit={current_price:.2f} "
            f"pnl=${pnl_usd:.2f} ({pnl_pct*100:+.2f}%) capital=${self.capital:.2f}"
        )
        self.position = None
        self._save_trades()

    def open_position(self, direction: str, confidence: float, position_size: float,
                      current_price: float, timestamp: str):
        """Open a new position (v3.1: vol_scalar applied in hybrid_model.predict)."""
        size_usd = min(position_size, self.capital * 0.1)  # cap at 10% of capital
        if size_usd < 10:  # minimum $10
            logger.info(f"Position size too small (${size_usd:.2f}), skipping")
            return

        self.position = {
            "direction": direction,
            "entry_price": current_price,
            "size_usd": size_usd,
            "entry_time": timestamp,
        }

        trade_record = {
            "trade_id": len(self.trade_history) + 1,
            "entry_time": timestamp,
            "exit_time": None,
            "direction": direction,
            "confidence": round(confidence, 4),
            "entry_price": current_price,
            "exit_price": None,
            "size_usd": round(size_usd, 2),
            "pnl": None,
            "pnl_pct": None,
            "status": "OPEN",
            "close_reason": None,
        }
        self.trade_history.append(trade_record)

        logger.info(
            f"OPENED {direction}: price={current_price:.2f} size=${size_usd:.2f} "
            f"confidence={confidence:.4f} capital=${self.capital:.2f}"
        )
        self._save_trades()

    def record_portfolio_snapshot(self, timestamp: str, btc_price: float,
                                  signal: str, confidence: float):
        """Record portfolio state for performance tracking."""
        unrealized = 0.0
        if self.position is not None:
            entry = self.position["entry_price"]
            size = self.position["size_usd"]
            if self.position["direction"] == "BUY":
                unrealized = size * (btc_price - entry) / entry
            else:
                unrealized = size * (entry - btc_price) / entry

        equity = self.capital + unrealized
        snapshot = {
            "timestamp": timestamp,
            "capital": round(self.capital, 2),
            "unrealized_pnl": round(unrealized, 2),
            "equity": round(equity, 2),
            "btc_price": round(btc_price, 2),
            "signal": signal,
            "confidence": round(confidence, 4),
            "position": self.position["direction"] if self.position else "FLAT",
            "total_trades": sum(1 for t in self.trade_history if t.get("status") == "CLOSED"),
        }
        self.portfolio_history.append(snapshot)
        self._save_portfolio()
        return equity

    def get_summary(self) -> dict:
        """Compute portfolio performance summary."""
        closed = [t for t in self.trade_history if t.get("status") == "CLOSED" and t.get("pnl") is not None]
        if not closed:
            return {"total_trades": 0, "capital": self.capital}

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        # Max drawdown from portfolio history
        if self.portfolio_history:
            equities = [s["equity"] for s in self.portfolio_history]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        return {
            "total_trades": len(closed),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 4),
            "total_return": round(total_return * 100, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "capital": round(self.capital, 2),
        }


def update_live_data():
    """Fetch latest data from Binance and append to saved CSV files."""
    btc_path = DATA_DIR / "btc_1h.csv"

    # Fetch latest BTC klines
    btc_df = fetch_klines(limit=LOOKBACK_HOURS)
    current_price = float(btc_df["Close"].iloc[-1])
    logger.info(f"BTC price: ${current_price:.2f}, fetched {len(btc_df)} rows")

    # Merge with existing saved data for full history
    if btc_path.exists():
        existing = pd.read_csv(btc_path, parse_dates=["timestamp"])
        existing = existing.set_index("timestamp").sort_index()
        # Rename to match fetched data
        col_map = {c: c.capitalize() for c in existing.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        existing = existing.rename(columns=col_map)
        existing = existing[["Open", "High", "Low", "Close", "Volume"]]

        # Combine: existing data + new data (new overwrites overlapping timestamps)
        combined = pd.concat([existing, btc_df[["Open", "High", "Low", "Close", "Volume"]]])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = btc_df[["Open", "High", "Low", "Close", "Volume"]]

    # Save updated data
    combined.index.name = "timestamp"
    combined.to_csv(btc_path)
    logger.info(f"Updated {btc_path}: {len(combined)} total rows")

    # Also update ETH data
    try:
        eth_df = fetch_eth_klines(limit=LOOKBACK_HOURS)
        eth_path = DATA_DIR / "eth_1h.csv"
        if eth_path.exists():
            existing_eth = pd.read_csv(eth_path, parse_dates=["timestamp"])
            existing_eth = existing_eth.set_index("timestamp").sort_index()
            eth_save = eth_df[["eth_close"]].copy()
            eth_save.columns = ["close"]
            combined_eth = pd.concat([existing_eth[["close"]] if "close" in existing_eth.columns else existing_eth, eth_save])
            combined_eth = combined_eth[~combined_eth.index.duplicated(keep="last")]
            combined_eth = combined_eth.sort_index()
        else:
            combined_eth = eth_df[["eth_close"]].copy()
            combined_eth.columns = ["close"]
        combined_eth.index.name = "timestamp"
        combined_eth.to_csv(eth_path)
    except Exception as e:
        logger.warning(f"ETH update failed: {e}")

    # Update funding rate
    try:
        fr_df = fetch_funding_rate()
        if len(fr_df) > 0:
            fr_path = DATA_DIR / "funding_rate.csv"
            if fr_path.exists():
                existing_fr = pd.read_csv(fr_path, parse_dates=["timestamp"])
                existing_fr = existing_fr.set_index("timestamp").sort_index()
                combined_fr = pd.concat([existing_fr, fr_df])
                combined_fr = combined_fr[~combined_fr.index.duplicated(keep="last")]
                combined_fr = combined_fr.sort_index()
            else:
                combined_fr = fr_df
            combined_fr.index.name = "timestamp"
            combined_fr.to_csv(fr_path)
    except Exception as e:
        logger.warning(f"Funding rate update failed: {e}")

    return current_price


def run_prediction_cycle(model, trader: PaperTrader):
    """Execute one prediction cycle: fetch data, predict, trade."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Update saved CSV files with latest data from Binance
    logger.info("Fetching and updating live data...")
    try:
        current_price = update_live_data()
    except Exception as e:
        logger.error(f"Failed to fetch live data: {e}")
        return

    # Generate prediction using saved CSV (same pipeline as training!)
    btc_path = str(DATA_DIR / "btc_1h.csv")
    try:
        signal, confidence, position_size = model.predict(btc_path, trader.capital)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    logger.info(f"Signal: {signal} (confidence={confidence:.4f}, position=${position_size:.2f})")

    # Trade logic
    if trader.position is not None:
        # Check if we should close
        if signal != trader.position["direction"] and signal != "HOLD":
            # Reverse signal → close and open new
            trader.close_position(current_price, timestamp, reason="signal_reversal")
            trader.open_position(signal, confidence, position_size, current_price, timestamp)
        elif signal == "HOLD":
            # HOLD with open position → close
            trader.close_position(current_price, timestamp, reason="hold_signal")
        # else: same direction → hold position
    else:
        # No position → open if signal is directional
        if signal in ("BUY", "SELL"):
            trader.open_position(signal, confidence, position_size, current_price, timestamp)

    # Record portfolio state
    equity = trader.record_portfolio_snapshot(timestamp, current_price, signal, confidence)

    # Print summary
    summary = trader.get_summary()
    logger.info(
        f"Portfolio: equity=${equity:.2f} | trades={summary['total_trades']} | "
        f"win_rate={summary.get('win_rate', 0)*100:.1f}% | "
        f"PF={summary.get('profit_factor', 0):.2f} | "
        f"return={summary.get('total_return', 0):+.2f}%"
    )


def run_backfill(model, hours: int = 720):
    """
    Simulate paper trading on the last N hours of saved data.
    Uses the SAME pipeline as training by writing temp CSV files.
    """
    logger.info(f"Backfill mode: simulating last {hours} hours")

    import tempfile

    # Load saved data
    btc_path = DATA_DIR / "btc_1h.csv"
    if not btc_path.exists():
        logger.error("No saved BTC data found. Run data_collector.py first.")
        return

    df = pd.read_csv(btc_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    if len(df) < hours:
        logger.warning(f"Not enough data: {len(df)} rows, reducing to {len(df) - 100}")
        hours = len(df) - 100

    trader = PaperTrader(initial_capital=INITIAL_CAPITAL)
    # Clear existing logs for clean backfill
    trader.trade_history = []
    trader.portfolio_history = []
    trader.capital = INITIAL_CAPITAL
    trader.position = None

    start_idx = len(df) - hours
    step = FORECAST_HORIZON_HOURS

    logger.info(f"Simulating from index {start_idx} to {len(df)}, step={step}h")

    for i in range(start_idx, len(df), step):
        # Write all data up to current point as a temp CSV
        # This ensures _prepare_data() processes the full history identically to training
        window = df.iloc[:i + 1].copy()

        close_col = "Close" if "Close" in window.columns else "close"
        current_price = float(window.iloc[-1][close_col])
        timestamp = str(window.index[-1])

        # Save to temp file with correct column names
        window_save = window.copy()
        col_map = {c: c.capitalize() for c in window_save.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        window_save = window_save.rename(columns=col_map)
        window_save = window_save[["Open", "High", "Low", "Close", "Volume"]]
        window_save.index.name = "Timestamp"

        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", dir=str(DATA_DIR)) as f:
                tmp_path = f.name
                window_save.to_csv(f)

            signal, confidence, position_size = model.predict(tmp_path, trader.capital)
        except Exception as e:
            logger.debug(f"Prediction at {timestamp} failed: {e}")
            continue
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Trade logic
        if trader.position is not None:
            if signal != trader.position["direction"] and signal != "HOLD":
                trader.close_position(current_price, timestamp, reason="signal_reversal")
                trader.open_position(signal, confidence, position_size, current_price, timestamp)
            elif signal == "HOLD":
                trader.close_position(current_price, timestamp, reason="hold_signal")
        else:
            if signal in ("BUY", "SELL"):
                trader.open_position(signal, confidence, position_size, current_price, timestamp)

        trader.record_portfolio_snapshot(timestamp, current_price, signal, confidence)

        # Progress
        done = (i - start_idx) // step + 1
        total = (len(df) - start_idx) // step
        if done % 10 == 0:
            logger.info(f"Progress: {done}/{total} steps, trades={len([t for t in trader.trade_history if t.get('status')=='CLOSED'])}")

    # Close any remaining position
    if trader.position is not None:
        close_col = "Close" if "Close" in df.columns else "close"
        final_price = float(df.iloc[-1][close_col])
        trader.close_position(final_price, str(df.index[-1]), reason="backfill_end")
        trader.record_portfolio_snapshot(str(df.index[-1]), final_price, "CLOSE", 0.0)

    # Print final summary
    summary = trader.get_summary()
    print("\n" + "=" * 60)
    print("PAPER TRADE BACKFILL RESULTS")
    print("=" * 60)
    close_col = "Close" if "Close" in df.columns else "close"
    print(f"  Period: {df.index[start_idx]} -> {df.index[-1]}")
    print(f"  Hours simulated: {hours}")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Win rate: {summary.get('win_rate', 0) * 100:.1f}%")
    print(f"  Avg win: ${summary.get('avg_win', 0):.2f}")
    print(f"  Avg loss: ${summary.get('avg_loss', 0):.2f}")
    print(f"  Profit factor: {summary.get('profit_factor', 0):.2f}")
    print(f"  Total return: {summary.get('total_return', 0):+.2f}%")
    print(f"  Max drawdown: {summary.get('max_drawdown', 0):.2f}%")
    print(f"  Final capital: ${summary['capital']:.2f}")
    print("=" * 60)

    # Buy & Hold comparison
    bh_start = float(df.iloc[start_idx][close_col])
    bh_end = float(df.iloc[-1][close_col])
    bh_return = (bh_end - bh_start) / bh_start * 100
    print(f"  Buy&Hold return: {bh_return:+.2f}% (${bh_start:.0f} -> ${bh_end:.0f})")
    print(f"  Alpha: {summary.get('total_return', 0) - bh_return:+.2f}%")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Crypto Paper Trading System")
    parser.add_argument("--once", action="store_true", help="Run one prediction cycle")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--backfill", type=int, default=0, help="Backfill N hours from saved data")
    args = parser.parse_args()

    # Load model
    from research.crypto.hybrid_model import load_model
    logger.info("Loading trained model...")
    model = load_model(str(MODEL_DIR))
    logger.info("Model loaded successfully")

    if args.backfill > 0:
        run_backfill(model, args.backfill)
        return

    trader = PaperTrader(initial_capital=INITIAL_CAPITAL)

    if args.once:
        run_prediction_cycle(model, trader)
    elif args.loop:
        logger.info(f"Starting paper trade loop (every {FORECAST_HORIZON_HOURS}h)")
        while True:
            try:
                run_prediction_cycle(model, trader)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            # Sleep until next cycle
            sleep_seconds = FORECAST_HORIZON_HOURS * 3600
            logger.info(f"Sleeping {FORECAST_HORIZON_HOURS}h until next cycle...")
            try:
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
    else:
        # Default: run once
        run_prediction_cycle(model, trader)


if __name__ == "__main__":
    main()
