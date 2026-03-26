"""
ラベル生成モジュール
Triple-Barrier ラベリングと方向ラベル
"""

import numpy as np


def build_triple_barrier_labels(
    close_arr: np.ndarray,
    barrier_up: float = 0.005,
    barrier_down: float = -0.003,
    barrier_t: int = 24,
) -> np.ndarray:
    """
    Triple-Barrier ラベルを生成
    - 1: 利確（上限バリアに到達）
    - 0: 損切り（下限バリアに到達）
    - 2: 時間切れ（どちらにも到達せず）
    - NaN: 計算不能（末尾のデータ）
    """
    n = len(close_arr)
    barrier_t = int(min(barrier_t, n - 1))
    if barrier_t < 1:
        return np.full(n, np.nan)

    y = np.full(n, np.nan)
    for i in range(n - barrier_t):
        c0 = close_arr[i]
        label = 2
        for t in range(1, barrier_t + 1):
            ret = (close_arr[i + t] - c0) / c0
            if ret >= barrier_up:
                label = 1
                break
            if ret <= barrier_down:
                label = 0
                break
        y[i] = label
    return y


def build_volatility_barriers(
    close_arr: np.ndarray,
    volatility_arr: np.ndarray,
    up_mult: float = 2.0,
    down_mult: float = 1.5,
    barrier_t: int = 24,
) -> np.ndarray:
    """
    ボラティリティベースの動的Triple-Barrierラベル
    バリア幅をボラティリティに応じて自動調整する
    """
    n = len(close_arr)
    barrier_t = int(min(barrier_t, n - 1))
    if barrier_t < 1:
        return np.full(n, np.nan)

    y = np.full(n, np.nan)
    for i in range(n - barrier_t):
        c0 = close_arr[i]
        vol = volatility_arr[i] if not np.isnan(volatility_arr[i]) else 0.005
        b_up = vol * up_mult
        b_down = -vol * down_mult
        label = 2
        for t in range(1, barrier_t + 1):
            ret = (close_arr[i + t] - c0) / c0
            if ret >= b_up:
                label = 1
                break
            if ret <= b_down:
                label = 0
                break
        y[i] = label
    return y
