"""
検証モジュール
Purged K-Fold CV, Walk-Forward 検証, 評価指標計算
"""

import math
import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation
    時系列データのリークを防ぐため、テストフォールドの前後にエンバーゴ期間を設ける
    """

    def __init__(self, n_splits: int = 5, embargo_size: int = 24):
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def split(self, X, y=None):
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(np.concatenate([[0], fold_sizes]))
        for k in range(self.n_splits):
            test_start = int(starts[k])
            test_end = int(starts[k + 1])
            test_idx = np.arange(test_start, test_end)
            train_before_end = max(0, test_start - self.embargo_size)
            train_after_start = min(n, test_end + self.embargo_size)
            train_idx = np.concatenate([
                np.arange(0, train_before_end),
                np.arange(train_after_start, n),
            ])
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx


def walk_forward_splits(
    n_total: int,
    min_train_size: int,
    test_size: int,
    step_size: int = None,
) -> list:
    """
    Expanding Window Walk-Forward の分割インデックスを生成

    Parameters:
        n_total: 全データ数
        min_train_size: 最小学習サイズ（例: 6ヶ月分 = 6*30*24 = 4320）
        test_size: テストサイズ（例: 1ヶ月分 = 30*24 = 720）
        step_size: ステップサイズ（デフォルト = test_size）

    Returns:
        list of (train_indices, test_indices) tuples
    """
    if step_size is None:
        step_size = test_size

    splits = []
    test_start = min_train_size
    while test_start + test_size <= n_total:
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        test_start += step_size

    return splits


def compute_metrics(trade_returns: np.ndarray, fee_rate: float = 0.0003) -> dict:
    """
    トレードリターンの配列から評価指標を計算する

    Parameters:
        trade_returns: 各トレードのリターン（方向 × 実際のリターン）
        fee_rate: 1トレードあたりの手数料率（デフォルト: 0.03% = スプレッド相当）

    Returns:
        dict with keys: pf, mdd, sharpe, sortino, exp_value_net, win_rate, payoff, n_trades
    """
    tr = np.asarray(trade_returns, dtype=float)
    tr = tr[~np.isnan(tr)]
    n_trades = len(tr)

    if n_trades == 0:
        return {
            "pf": float("nan"), "mdd": float("nan"), "sharpe": float("nan"),
            "sortino": float("nan"), "exp_value_net": float("nan"),
            "win_rate": float("nan"), "payoff": float("nan"), "n_trades": 0,
        }

    # Profit Factor
    profit = tr[tr > 0].sum()
    loss_sum = tr[tr < 0].sum()
    pf = (profit / -loss_sum) if loss_sum < 0 else (float("inf") if profit > 0 else float("nan"))

    # Maximum Drawdown (%)
    equity = 1.0 + np.cumsum(tr)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100.0
    mdd = float(drawdown.max())

    # Sharpe Ratio (annualized, 1h bars → 24*365)
    mean_r = float(tr.mean())
    std_r = float(tr.std(ddof=1)) if n_trades > 1 else float("nan")
    annualize = math.sqrt(24 * 365)
    sharpe = (mean_r / std_r * annualize) if std_r and std_r > 0 else float("nan")

    # Sortino Ratio
    neg_r = tr[tr < 0]
    if len(neg_r) > 0:
        std_neg = float(neg_r.std(ddof=1))
        sortino = (mean_r / std_neg * annualize) if std_neg > 0 else float("nan")
    else:
        sortino = float("inf")

    # Expected value net of fees
    exp_value_net = mean_r - fee_rate

    # Win rate & Payoff ratio
    wins = tr[tr > 0]
    losses = tr[tr < 0]
    win_rate = float(len(wins) / n_trades * 100.0)
    avg_profit = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    payoff = (avg_profit / -avg_loss) if avg_loss < 0 else float("nan")

    return {
        "pf": pf, "mdd": mdd, "sharpe": sharpe, "sortino": sortino,
        "exp_value_net": exp_value_net, "win_rate": win_rate, "payoff": payoff,
        "n_trades": n_trades,
    }


def print_metrics(metrics: dict, label: str = ""):
    """評価指標を見やすく表示する"""
    prefix = f"【{label}】" if label else "【評価指標】"
    print(f"\n{prefix}")
    print(f"  トレード数: {metrics['n_trades']}")
    print(f"  勝率: {metrics['win_rate']:.2f}%")
    print(f"  プロフィットファクター: {metrics['pf']:.2f}（目標: 1.3以上）")
    print(f"  最大ドローダウン: {metrics['mdd']:.2f}%（目標: 20%以下）")
    print(f"  シャープレシオ: {metrics['sharpe']:.2f}（目標: 1.0以上）")
    print(f"  ソルティノレシオ: {metrics['sortino']:.2f}")
    print(f"  手数料込み期待値: {metrics['exp_value_net']:+.6f}")
    print(f"  損益比: {metrics['payoff']:.2f}")
