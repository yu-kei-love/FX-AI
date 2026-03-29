# ===========================================
# regime_detector.py
# FXモデル - 相場環境自動判定
#
# 背景：
#   OOS期間（2025年8月以降）でモデルが機能しない。
#   原因：低ボラ環境（ATR低下）でスプレッドコストの
#         相対的な影響が増大した。
#
# 解決策：
#   ATR・ADX・ボラティリティから相場環境を3分類し、
#   環境ごとにパラメータを切り替える。
#
# 注意：パラメータを再最適化しないこと
#       （またOOSで崩壊するリスクがある）
# ===========================================

from enum import Enum

import numpy as np
import pandas as pd


# =============================================================
# 相場環境の分類
# =============================================================

class MarketRegime(Enum):
    """相場環境の3分類。"""
    HIGH_VOL_TREND = "HIGH_VOL_TREND"   # 高ボラ・トレンド（モデルが機能する）
    LOW_VOL_TREND  = "LOW_VOL_TREND"    # 低ボラ・トレンド（スプレッドコストに注意）
    RANGE          = "RANGE"            # レンジ相場（見送り推奨）


# =============================================================
# 判定パラメータ（未検証 → グリッドサーチ必要）
# =============================================================

ATR_HIGH_THRESHOLD = 0.25   # ATR(14) > 0.25 → 高ボラ判定（未検証）
ADX_TREND_THRESHOLD = 25.0  # ADX > 25 → トレンド判定（未検証）


# =============================================================
# インジケーター計算
# =============================================================

def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR) を計算する。

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)

    Parameters:
        df    : OHLC DataFrame（列: high, low, close）
        window: 期間（デフォルト14）

    Returns:
        atr: ATRの Series
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(span=window, adjust=False).mean()


def calc_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) を計算する。
    トレンドの強さを0〜100で示す（25以上でトレンド相場）。

    Parameters:
        df    : OHLC DataFrame
        window: 期間

    Returns:
        adx: ADXの Series
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    dm_plus  = np.where((high - prev_high) > (prev_low - low),
                        np.maximum(high - prev_high, 0), 0)
    dm_minus = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0), 0)

    dm_plus_s  = pd.Series(dm_plus,  index=df.index).ewm(span=window, adjust=False).mean()
    dm_minus_s = pd.Series(dm_minus, index=df.index).ewm(span=window, adjust=False).mean()
    tr_s       = tr.ewm(span=window, adjust=False).mean()

    di_plus  = dm_plus_s  / tr_s.replace(0, np.nan) * 100
    di_minus = dm_minus_s / tr_s.replace(0, np.nan) * 100

    dx  = (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100
    adx = dx.ewm(span=window, adjust=False).mean()

    return adx


# =============================================================
# レジーム判定
# =============================================================

def detect_market_regime(
    df:     pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    ATR・ADXからレース環境を3分類する。

    判定基準（未検証 → OOS検証で調整が必要）：
    ATR(14) > 0.25 かつ ADX > 25 → HIGH_VOL_TREND
    ATR(14) ≤ 0.25 かつ ADX > 25 → LOW_VOL_TREND
    ADX ≤ 25                      → RANGE

    Parameters:
        df    : OHLC DataFrame（列: open, high, low, close）
        window: ADX・ATRの計算ウィンドウ

    Returns:
        regime: MarketRegimeのSeries
    """
    atr = calc_atr(df, window=14)
    adx = calc_adx(df, window=window)

    conditions = [
        (atr > ATR_HIGH_THRESHOLD) & (adx > ADX_TREND_THRESHOLD),
        (atr <= ATR_HIGH_THRESHOLD) & (adx > ADX_TREND_THRESHOLD),
    ]
    choices = [MarketRegime.HIGH_VOL_TREND, MarketRegime.LOW_VOL_TREND]

    regime_arr = np.select(conditions, choices, default=MarketRegime.RANGE)
    return pd.Series(regime_arr, index=df.index, name="regime")


def get_current_regime(df: pd.DataFrame, lookback: int = 20) -> MarketRegime:
    """
    直近の相場環境を返す。

    Parameters:
        df      : OHLC DataFrame
        lookback: 判定に使う最近のバー数

    Returns:
        regime: MarketRegime
    """
    regime_series = detect_market_regime(df)
    if regime_series.empty:
        return MarketRegime.RANGE
    return regime_series.iloc[-1]


def get_regime_stats(df: pd.DataFrame) -> dict:
    """
    期間内のレジーム分布を集計する。

    Returns:
        stats: {"HIGH_VOL_TREND": 0.35, "LOW_VOL_TREND": 0.30, "RANGE": 0.35}
    """
    regime_series = detect_market_regime(df)
    counts = regime_series.value_counts()
    total  = len(regime_series)

    return {
        regime.value: round(counts.get(regime, 0) / total, 4)
        for regime in MarketRegime
    }
