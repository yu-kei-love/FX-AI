# ===========================================
# evaluation.py
# 全市場共通 - 評価フレームワーク
#
# 設計方針：
#   - ボートレース・FX・株・暗号通貨で同じ評価指標を使う
#   - Sharpe比のann_factorは実際のトレード頻度から計算する
#   - ROI・MDD・Sharpe・EV実現率・カリブレーションを全て計算する
# ===========================================

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================
# 全市場で使えるユニバーサル評価指標
# =============================================================

def calc_universal_metrics(
    returns:           list,
    capital_history:   list,
    trades_per_period: float,
    period:            str = "day",
    investment_list:   list = None,
    payout_list:       list = None,
) -> dict:
    """
    全市場で同じ評価指標を計算する。

    Parameters:
        returns          : 1取引あたりのリターン率リスト（例: [0.25, -1.0, ...]）
        capital_history  : 資金推移リスト
        trades_per_period: 1期間あたりのトレード数（期間単位はperiodで指定）
        period           : 期間の単位 "day" / "hour" / "minute"
        investment_list  : 投資額リスト（ROI計算用）
        payout_list      : 払戻額リスト（ROI計算用）

    Returns:
        metrics: 全指標のdict
    """
    arr = np.array(returns, dtype=np.float64) if returns else np.array([])
    cap = np.array(capital_history, dtype=np.float64) if capital_history else np.array([])

    # ROI
    roi = 0.0
    if investment_list and payout_list:
        total_inv = sum(investment_list)
        total_pay = sum(payout_list)
        roi = (total_pay - total_inv) / total_inv * 100.0 if total_inv > 0 else 0.0
    elif len(arr) > 0:
        roi = float(arr.mean() * 100.0)

    # 最大ドローダウン
    max_dd = 0.0
    if len(cap) > 1:
        peak = np.maximum.accumulate(cap)
        dds  = np.where(peak > 0, (peak - cap) / peak, 0.0)
        max_dd = float(dds.max() * 100.0)

    # Sharpe比（実際のトレード頻度ベース・固定値を使わない）
    sharpe = 0.0
    if len(arr) >= 2:
        mean_r = arr.mean()
        std_r  = arr.std(ddof=1)
        if std_r > 0 and trades_per_period > 0:
            # period → 年間取引数を計算
            period_multipliers = {"day": 252.0, "hour": 252.0 * 24, "minute": 252.0 * 24 * 60}
            annual_mult = period_multipliers.get(period, 252.0)
            ann_factor = np.sqrt(annual_mult * trades_per_period)
            sharpe = float(mean_r / std_r * ann_factor)

    # Sortino比（下方リスクのみ）
    sortino = 0.0
    if len(arr) >= 2:
        mean_r   = arr.mean()
        neg_arr  = arr[arr < 0]
        if len(neg_arr) > 1:
            downside_std = neg_arr.std(ddof=1)
            if downside_std > 0 and trades_per_period > 0:
                period_multipliers = {"day": 252.0, "hour": 252.0 * 24, "minute": 252.0 * 24 * 60}
                annual_mult = period_multipliers.get(period, 252.0)
                ann_factor = np.sqrt(annual_mult * trades_per_period)
                sortino = float(mean_r / downside_std * ann_factor)

    # Profit Factor
    pf = 0.0
    if len(arr) > 0:
        profits = arr[arr > 0].sum()
        losses  = abs(arr[arr < 0].sum())
        pf = float(profits / losses) if losses > 0 else float("inf")

    # 勝率
    win_rate = float((arr > 0).mean()) if len(arr) > 0 else 0.0

    return {
        "roi":            round(roi, 4),
        "max_drawdown":   round(max_dd, 4),
        "sharpe":         round(sharpe, 4),
        "sortino":        round(sortino, 4),
        "profit_factor":  round(pf, 4),
        "win_rate":       round(win_rate, 4),
        "n_trades":       len(arr),
        "mean_return":    round(float(arr.mean()), 6) if len(arr) > 0 else 0.0,
        "std_return":     round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else 0.0,
    }


# =============================================================
# EV実現率
# =============================================================

def calc_ev_realization_rate(
    predicted_evs: list,
    actual_returns: list,
) -> float:
    """
    EV実現率 = 実際の平均リターン / 予測平均EV

    1.0に近いほどモデルの確率推定が正確。
    0.5以下ならモデルの過信を示す。

    Parameters:
        predicted_evs  : 予測EV（各取引）
        actual_returns : 実際のリターン（各取引）

    Returns:
        realization_rate
    """
    if not predicted_evs or not actual_returns:
        return 0.0

    pred_mean   = np.mean(predicted_evs)
    actual_mean = np.mean(actual_returns)

    if pred_mean <= 0:
        return 0.0

    return float(actual_mean / pred_mean)


# =============================================================
# カリブレーション
# =============================================================

def calc_calibration(
    predicted_probs: list,
    actual_results:  list,
    n_bins:          int = 10,
) -> dict:
    """
    モデルのカリブレーション（確率の精度）を評価する。

    「モデルが30%と言ったら実際に30%当たるか」

    Parameters:
        predicted_probs : 予測確率リスト（0〜1）
        actual_results  : 実際の結果（1=的中/0=外れ）リスト
        n_bins          : ビン数（デフォルト10→10%刻み）

    Returns:
        result: {"bins": [...], "max_error": float, "is_well_calibrated": bool}
    """
    preds  = np.array(predicted_probs, dtype=np.float64)
    actual = np.array(actual_results,  dtype=np.float64)

    bins_data = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask  = (preds >= low) & (preds < high)
        count = int(mask.sum())

        if count == 0:
            bins_data.append({
                "range": (round(low, 2), round(high, 2)),
                "count": 0, "predicted": None, "actual": None, "error": None,
            })
            continue

        pred_mean   = float(preds[mask].mean())
        actual_rate = float(actual[mask].mean())
        bins_data.append({
            "range":     (round(low, 2), round(high, 2)),
            "count":     count,
            "predicted": round(pred_mean, 4),
            "actual":    round(actual_rate, 4),
            "error":     round(abs(pred_mean - actual_rate), 4),
        })

    errors    = [b["error"] for b in bins_data if b["error"] is not None]
    max_error = float(max(errors)) if errors else 0.0

    return {
        "bins":               bins_data,
        "max_error":          round(max_error, 4),
        "is_well_calibrated": max_error < 0.10,
    }


# =============================================================
# 市場別ペーパートレード移行判定
# =============================================================

PAPER_TRADE_CRITERIA = {
    "boat": {
        "roi_min":          0.0,
        "max_drawdown_max": 30.0,
        "calibration_max":  0.10,
        "n_months_min":     3.0,
        "description":      "ボートレース移行条件",
    },
    "fx": {
        "profit_factor_min": 1.3,
        "max_drawdown_max":  20.0,
        "sharpe_min":        1.0,
        "n_trades_min":      200,
        "n_months_min":      3.0,
        "description":       "FX移行条件",
    },
    "stock": {
        "roi_min":          0.0,
        "max_drawdown_max": 20.0,
        "n_months_min":     3.0,
        "description":      "株移行条件",
    },
    "crypto": {
        "roi_min":          0.0,
        "max_drawdown_max": 30.0,
        "n_months_min":     3.0,
        "description":      "暗号通貨移行条件",
    },
}


def check_paper_trade_ready(market: str, metrics: dict) -> dict:
    """
    ペーパートレード移行条件を確認する。

    Parameters:
        market  : "boat" / "fx" / "stock" / "crypto"
        metrics : calc_universal_metrics等で計算した評価指標

    Returns:
        result: {"ready": bool, "passed": {条件: bool}, "failed": [条件名]}
    """
    criteria = PAPER_TRADE_CRITERIA.get(market, {})
    passed = {}
    failed = []

    check_map = {
        "roi_min":           ("roi",           lambda v, t: v >= t,  f"ROI ≥ {criteria.get('roi_min',0):.1f}%"),
        "profit_factor_min": ("profit_factor", lambda v, t: v >= t,  f"PF ≥ {criteria.get('profit_factor_min',1.0):.1f}"),
        "max_drawdown_max":  ("max_drawdown",  lambda v, t: v <= t,  f"MDD ≤ {criteria.get('max_drawdown_max',30):.0f}%"),
        "sharpe_min":        ("sharpe",        lambda v, t: v >= t,  f"Sharpe ≥ {criteria.get('sharpe_min',1.0):.1f}"),
        "n_trades_min":      ("n_trades",      lambda v, t: v >= t,  f"取引数 ≥ {criteria.get('n_trades_min',0)}"),
        "n_months_min":      ("n_months",      lambda v, t: v >= t,  f"期間 ≥ {criteria.get('n_months_min',3):.0f}ヶ月"),
        "calibration_max":   ("calibration_max_error", lambda v, t: v <= t,
                              f"カリブレーション ≤ {criteria.get('calibration_max',0.1):.0%}"),
    }

    print(f"\n=== {criteria.get('description', market)} ===")
    for key, (metric_key, cond_fn, label) in check_map.items():
        if key not in criteria:
            continue
        threshold = criteria[key]
        value = metrics.get(metric_key)
        if value is None:
            ok = False
        else:
            ok = cond_fn(value, threshold)
        passed[label] = ok
        status = "✓" if ok else "✗"
        print(f"  {status} {label}: {value}")
        if not ok:
            failed.append(label)

    ready = len(failed) == 0
    if ready:
        print("\n→ 全条件クリア！ペーパートレードへ移行可能")
    else:
        print(f"\n→ 未達条件: {', '.join(failed)}")

    return {"ready": ready, "passed": passed, "failed": failed}
