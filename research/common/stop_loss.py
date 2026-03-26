"""
ATRベースのストップロス・テイクプロフィット管理モジュール。

リスク管理を最優先とし、ATRに基づいた動的なSL/TP設定、
トレーリングストップ、時間ベースのエグジットを提供する。
"""

import numpy as np
import pandas as pd


class StopLossManager:
    """ATRベースのストップロス・テイクプロフィット計算クラス。"""

    def __init__(
        self,
        atr_period: int = 14,
        sl_multiplier: float = 1.5,
        tp_multiplier: float = 2.0,
        max_hold_hours: int = 24,
    ):
        """
        パラメータ初期化。

        Args:
            atr_period: ATR計算期間（デフォルト14）
            sl_multiplier: ストップロスのATR倍率（デフォルト1.5）
            tp_multiplier: テイクプロフィットのATR倍率（デフォルト2.0）
            max_hold_hours: 最大保有時間（デフォルト24時間）
        """
        self.atr_period = atr_period
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.max_hold_hours = max_hold_hours

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        ATR（Average True Range）を計算する。

        Args:
            df: High, Low, Close カラムを持つOHLCデータフレーム

        Returns:
            ATR値のSeries
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Rangeの3要素を計算
        tr1 = high - low  # 当日の高値-安値
        tr2 = (high - close.shift(1)).abs()  # 当日高値-前日終値
        tr3 = (low - close.shift(1)).abs()  # 当日安値-前日終値

        # True Range = 3要素の最大値
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = True Rangeの指数移動平均
        atr = true_range.ewm(span=self.atr_period, min_periods=self.atr_period).mean()

        return atr

    def get_levels(
        self, direction: str, entry_price: float, atr_value: float
    ) -> dict:
        """
        ストップロスとテイクプロフィットのレベルを計算する。

        Args:
            direction: "BUY" または "SELL"
            entry_price: エントリー価格
            atr_value: 現在のATR値

        Returns:
            stop_loss, take_profit, risk_pips, reward_pips, risk_reward_ratio を含む辞書
        """
        risk_pips = atr_value * self.sl_multiplier
        reward_pips = atr_value * self.tp_multiplier

        if direction == "BUY":
            stop_loss = entry_price - risk_pips
            take_profit = entry_price + reward_pips
        elif direction == "SELL":
            stop_loss = entry_price + risk_pips
            take_profit = entry_price - reward_pips
        else:
            raise ValueError(f"directionは 'BUY' または 'SELL' を指定: {direction}")

        # リスクリワード比の計算（ゼロ除算防止）
        risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0.0

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_pips": risk_pips,
            "reward_pips": reward_pips,
            "risk_reward_ratio": round(risk_reward_ratio, 4),
        }

    def update_trailing_stop(
        self,
        direction: str,
        current_price: float,
        current_stop: float,
        atr_value: float,
    ) -> float:
        """
        トレーリングストップを更新する。
        BUYの場合は上方向のみ、SELLの場合は下方向のみ移動する（ラチェット方式）。

        Args:
            direction: "BUY" または "SELL"
            current_price: 現在価格
            current_stop: 現在のストップロス価格
            atr_value: 現在のATR値

        Returns:
            更新後のストップロス価格
        """
        trail_distance = atr_value * self.sl_multiplier

        if direction == "BUY":
            # BUY: 価格上昇に合わせてストップを引き上げる（下げない）
            new_stop = current_price - trail_distance
            return max(new_stop, current_stop)
        elif direction == "SELL":
            # SELL: 価格下落に合わせてストップを引き下げる（上げない）
            new_stop = current_price + trail_distance
            return min(new_stop, current_stop)
        else:
            raise ValueError(f"directionは 'BUY' または 'SELL' を指定: {direction}")

    def should_exit(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        entry_time: pd.Timestamp,
        current_time: pd.Timestamp,
        stop_loss: float,
        take_profit: float,
    ) -> tuple:
        """
        エグジット条件を判定する。

        優先順位: ストップロス > テイクプロフィット > 時間ベースエグジット

        Args:
            direction: "BUY" または "SELL"
            entry_price: エントリー価格
            current_price: 現在価格
            entry_time: エントリー時刻
            current_time: 現在時刻
            stop_loss: ストップロス価格
            take_profit: テイクプロフィット価格

        Returns:
            (エグジットすべきか, 理由) のタプル
            理由: "stop_loss", "take_profit", "time_exit", None
        """
        # ストップロス判定（最優先 — 損失拡大を防ぐ）
        if direction == "BUY" and current_price <= stop_loss:
            return (True, "stop_loss")
        if direction == "SELL" and current_price >= stop_loss:
            return (True, "stop_loss")

        # テイクプロフィット判定
        if direction == "BUY" and current_price >= take_profit:
            return (True, "take_profit")
        if direction == "SELL" and current_price <= take_profit:
            return (True, "take_profit")

        # 時間ベースエグジット判定
        hold_duration = current_time - entry_time
        max_hold = pd.Timedelta(hours=self.max_hold_hours)
        if hold_duration >= max_hold:
            return (True, "time_exit")

        # エグジット条件なし
        return (False, None)
