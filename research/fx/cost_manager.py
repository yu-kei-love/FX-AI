# ===========================================
# cost_manager.py
# FXモデル - スプレッドコストの動的管理
#
# 背景：
#   低ボラ環境ではATRが小さいため、
#   スプレッドコストの相対的な負担が増大する。
#   相対コストに基づいてエントリー基準を動的に調整する。
# ===========================================

import numpy as np
import pandas as pd
from typing import Optional


# =============================================================
# 定数（未検証 → 実データで検証が必要）
# =============================================================

BASE_SPREAD_PIPS  = 0.3    # USD/JPYの平常スプレッド（0.3pips）未検証
SPREAD_WARN_PIPS  = 1.5    # スプレッド警告閾値（1.5pips）未検証
SPREAD_STOP_PIPS  = 3.0    # スプレッド見送り閾値（3.0pips）未検証


# =============================================================
# スプレッドコストの動的計算
# =============================================================

def calc_effective_spread(
    atr:          float,
    spread_pips:  float = BASE_SPREAD_PIPS,
) -> dict:
    """
    ATRに対するスプレッドの相対コストを計算する。

    相対コスト = スプレッド / ATR
    相対コストが高い → エントリー基準を上げる
    相対コストが低い → 通常通り

    Parameters:
        atr         : ATR値（直近の平均真の値幅）
        spread_pips : スプレッド（pips）

    Returns:
        result: {
            "spread_pips":        float,
            "atr":                float,
            "relative_cost":      float,  # スプレッド/ATR
            "cost_level":         str,    # "low"/"medium"/"high"/"very_high"
            "confidence_penalty": float,  # 確信度への乗数（0〜1）
        }
    """
    if atr <= 0:
        return {
            "spread_pips":        spread_pips,
            "atr":                atr,
            "relative_cost":      float("inf"),
            "cost_level":         "very_high",
            "confidence_penalty": 0.0,
        }

    relative_cost = spread_pips / atr

    # コストレベルの判定（未検証 → データで検証が必要）
    if relative_cost >= 0.50:      # スプレッドがATRの50%以上 → 非常に高い
        cost_level         = "very_high"
        confidence_penalty = 0.0   # 見送り
    elif relative_cost >= 0.30:    # 30%以上 → 高い
        cost_level         = "high"
        confidence_penalty = 0.5
    elif relative_cost >= 0.15:    # 15%以上 → 中程度
        cost_level         = "medium"
        confidence_penalty = 0.8
    else:
        cost_level         = "low"
        confidence_penalty = 1.0   # 通常通り

    return {
        "spread_pips":        spread_pips,
        "atr":                atr,
        "relative_cost":      round(relative_cost, 4),
        "cost_level":         cost_level,
        "confidence_penalty": confidence_penalty,
    }


# =============================================================
# スプレッドの時系列モニタリング
# =============================================================

def calc_dynamic_spread(
    bid_prices: pd.Series,
    ask_prices: pd.Series,
) -> pd.Series:
    """
    Bid/Askから動的スプレッドを計算する（pips換算）。

    Parameters:
        bid_prices : Bid価格のSeries
        ask_prices : Ask価格のSeries

    Returns:
        spread_pips: スプレッド（pips）のSeries
    """
    spread_raw = ask_prices - bid_prices
    # USD/JPYは0.01=1pip
    spread_pips = spread_raw * 100
    return spread_pips


def get_spread_baseline(
    spread_history: pd.Series,
    window:         int = 100,
    percentile:     int = 50,
) -> float:
    """
    スプレッドのベースライン（通常時の水準）を計算する。

    Parameters:
        spread_history : スプレッド履歴
        window         : 計算ウィンドウ
        percentile     : パーセンタイル（50=中央値）

    Returns:
        baseline: スプレッドのベースライン値
    """
    recent = spread_history.iloc[-window:] if len(spread_history) >= window else spread_history
    return float(np.percentile(recent.dropna(), percentile))


# =============================================================
# スプレッドフィルター（エントリー判定）
# =============================================================

def check_spread_for_entry(
    current_spread: float,
    atr:            float,
    baseline_spread: Optional[float] = None,
) -> dict:
    """
    エントリー時にスプレッドが許容範囲かチェックする。

    判定基準（CLAUDE.mdより）：
    ・スプレッドが閾値を超える場合は見送り
    ・スプレッド急拡大フィルターを適用

    Parameters:
        current_spread  : 現在のスプレッド（pips）
        atr             : ATR値
        baseline_spread : 平常時ベースライン（Noneの場合は絶対値のみで判定）

    Returns:
        result: {"allow_entry": bool, "reason": str, "confidence_mult": float}
    """
    # 絶対閾値チェック（スプレッドが一定以上なら見送り）
    if current_spread >= SPREAD_STOP_PIPS:
        return {
            "allow_entry":    False,
            "reason":         f"スプレッド{current_spread:.2f}pips（絶対上限{SPREAD_STOP_PIPS}超）",
            "confidence_mult": 0.0,
        }

    # 相対コストチェック
    cost_info = calc_effective_spread(atr, current_spread)
    if cost_info["cost_level"] == "very_high":
        return {
            "allow_entry":    False,
            "reason":         f"相対コスト{cost_info['relative_cost']:.2%}（ATR比50%超）",
            "confidence_mult": 0.0,
        }

    # スプレッド急拡大チェック（ベースラインがある場合）
    confidence_mult = cost_info["confidence_penalty"]
    reason_parts = [f"相対コスト{cost_info['relative_cost']:.2%}"]

    if baseline_spread and baseline_spread > 0:
        from research.common.market_filter import MarketAnomalyFilter
        filt = MarketAnomalyFilter()
        result = filt.check_fx_spread_widening(
            current_spread, baseline_spread, max_spread_pips=SPREAD_STOP_PIPS
        )
        confidence_mult = min(confidence_mult, result.confidence_mult)
        reason_parts.append(result.reason)

    if current_spread >= SPREAD_WARN_PIPS:
        reason_parts.append(f"警告：スプレッド{current_spread:.2f}pips")

    return {
        "allow_entry":    confidence_mult > 0,
        "reason":         " / ".join(reason_parts),
        "confidence_mult": confidence_mult,
    }


# =============================================================
# コスト調整後の期待値計算
# =============================================================

def calc_net_expected_value(
    raw_ev:        float,
    spread_pips:   float,
    position_pips: float,
) -> float:
    """
    スプレッドコストを考慮した純期待値を計算する。

    net_EV = raw_EV - (spread_cost / position_size)

    Parameters:
        raw_ev        : モデルが計算したEV（コスト考慮前）
        spread_pips   : スプレッド（pips）
        position_pips : ポジションサイズ（pips単位のTP/SL幅）

    Returns:
        net_ev: スプレッドコスト差し引き後のEV
    """
    if position_pips <= 0:
        return raw_ev

    spread_cost_ratio = spread_pips / position_pips
    return raw_ev - spread_cost_ratio
