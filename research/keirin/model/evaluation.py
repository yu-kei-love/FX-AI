# ===========================================
# model/evaluation.py
# 競輪AI - 評価フレームワーク
#
# ボートレースの evaluation.py と同じ設計に
# 以下を追加：
#   - calc_line_prediction_accuracy（ライン予測正答率）
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from betting_logic import BettingSignal


# =============================================================
# ペーパートレード移行条件
# =============================================================

PAPER_TRADE_CRITERIA = {
    "roi_min":               0.0,    # ROI > 0%
    "max_drawdown_max":      0.30,   # 最大MDD < 30%
    "min_months":            3,      # 検証期間 ≥ 3ヶ月
    "calibration_error_max": 0.10,   # キャリブレーション誤差 < 10%
}


# =============================================================
# 基本指標の計算
# =============================================================

def calc_roi(signals: list, results_df: pd.DataFrame) -> float:
    """
    ROI（投資収益率）を計算する。

    ROI = (回収額 - 投資額) / 投資額

    Parameters:
        signals    : BettingSignalのリスト
        results_df : レース結果DataFrame（race_id, 1st, 2nd, 3rd, trifecta_payout）

    Returns:
        float: ROI（0.05 = 5%）
    """
    total_bet    = 0.0
    total_return = 0.0

    result_lookup = {}
    if results_df is not None and len(results_df) > 0:
        for _, row in results_df.iterrows():
            result_lookup[row["race_id"]] = row

    for signal in signals:
        if not signal.filters_passed:
            continue
        total_bet += signal.kelly_bet
        result = result_lookup.get(signal.race_id)
        if result is not None:
            actual_combo = (
                int(result.get("rank1_car", 0)),
                int(result.get("rank2_car", 0)),
                int(result.get("rank3_car", 0)),
            )
            if actual_combo == signal.combo:
                payout = float(result.get("trifecta_payout", 0))
                total_return += signal.kelly_bet * payout / 100  # 100円あたりの払い戻し

    if total_bet <= 0:
        return 0.0
    return round((total_return - total_bet) / total_bet, 6)


def calc_max_drawdown(capital_series: pd.Series) -> float:
    """
    最大ドローダウンを計算する。

    Parameters:
        capital_series: 資金推移のSeries

    Returns:
        float: 最大MDD（0.15 = 15%）
    """
    if len(capital_series) == 0:
        return 0.0

    peak   = capital_series.expanding().max()
    dd     = (capital_series - peak) / peak
    return float(abs(dd.min()))


def calc_sharpe(returns_series: pd.Series) -> float:
    """
    Sharpe比を計算する（年率換算）。

    Parameters:
        returns_series: 1レースごとのリターンSeries

    Returns:
        float: Sharpe比
    """
    if len(returns_series) < 2:
        return 0.0

    mean_r   = returns_series.mean()
    std_r    = returns_series.std()
    if std_r == 0:
        return 0.0

    # 年間レース数から年率換算
    n_per_year = len(returns_series)  # 実際の年間レース数で換算
    ann_factor = np.sqrt(n_per_year)
    return float(round(mean_r / std_r * ann_factor, 4))


def calc_profit_factor(signals: list, results_df: pd.DataFrame) -> float:
    """
    プロフィットファクター（PF）を計算する。

    PF = 総利益 / 総損失

    Returns:
        float: PF
    """
    wins   = 0.0
    losses = 0.0

    result_lookup = {}
    if results_df is not None:
        for _, row in results_df.iterrows():
            result_lookup[row["race_id"]] = row

    for signal in signals:
        if not signal.filters_passed:
            continue
        result = result_lookup.get(signal.race_id)
        if result is not None:
            actual_combo = (
                int(result.get("rank1_car", 0)),
                int(result.get("rank2_car", 0)),
                int(result.get("rank3_car", 0)),
            )
            if actual_combo == signal.combo:
                payout = float(result.get("trifecta_payout", 0))
                net    = signal.kelly_bet * payout / 100 - signal.kelly_bet
                wins  += max(net, 0)
            else:
                losses += signal.kelly_bet

    if losses == 0:
        return float("inf")
    return round(wins / losses, 4)


# =============================================================
# キャリブレーション確認
# =============================================================

def calc_calibration(signals: list, results_df: pd.DataFrame) -> dict:
    """
    キャリブレーション（確率精度）を計算する。

    10%刻みのビンに分けて、予測確率と実際の当選率を比較する。
    max_error < 10% = well-calibrated

    Returns:
        {
            "bins": [(predicted_prob_center, actual_rate, count)],
            "max_error": float,
            "is_calibrated": bool,
        }
    """
    result_lookup = {}
    if results_df is not None:
        for _, row in results_df.iterrows():
            result_lookup[row["race_id"]] = row

    bins = {i: {"pred_sum": 0.0, "hit": 0, "count": 0}
            for i in range(0, 100, 10)}

    for signal in signals:
        if not signal.filters_passed:
            continue
        bin_key = min(int(signal.predicted_prob * 100 // 10) * 10, 90)
        bins[bin_key]["pred_sum"] += signal.predicted_prob
        bins[bin_key]["count"]    += 1

        result = result_lookup.get(signal.race_id)
        if result is not None:
            actual_combo = (
                int(result.get("rank1_car", 0)),
                int(result.get("rank2_car", 0)),
                int(result.get("rank3_car", 0)),
            )
            if actual_combo == signal.combo:
                bins[bin_key]["hit"] += 1

    calibration_data = []
    errors = []
    for center in range(5, 100, 10):
        bin_key = center - 5
        data    = bins[bin_key]
        if data["count"] == 0:
            continue
        pred_avg   = data["pred_sum"] / data["count"]
        actual_rate = data["hit"] / data["count"]
        error      = abs(pred_avg - actual_rate)
        errors.append(error)
        calibration_data.append((round(pred_avg, 4), round(actual_rate, 4), data["count"]))

    max_error = max(errors) if errors else 0.0
    return {
        "bins":          calibration_data,
        "max_error":     round(max_error, 4),
        "is_calibrated": max_error < PAPER_TRADE_CRITERIA["calibration_error_max"],
    }


# =============================================================
# ライン予測正答率（競輪固有）
# =============================================================

def calc_line_prediction_accuracy(
    predicted_lines: pd.DataFrame,
    actual_lines: pd.DataFrame,
) -> dict:
    """
    ライン予測の正答率を計算する（競輪固有）。

    グレード別・会場別・信頼度別に分解する：
    - high confidence（>0.7）の正答率
    - low confidence（<0.5）の正答率

    → 信頼度が正答率と相関しているかを確認する

    Parameters:
        predicted_lines: DataFrame（race_id, car_no, predicted_line, confidence）
        actual_lines   : DataFrame（race_id, car_no, actual_line）

    Returns:
        {
            "overall": float,
            "by_confidence": {
                "high": float,   # confidence > 0.7
                "mid":  float,   # 0.5 <= confidence <= 0.7
                "low":  float,   # confidence < 0.5
            },
            "by_grade": {"G1": float, "F1": float, ...},
            "by_venue": {"前橋": float, ...},
            "n_samples": int,
        }
    """
    if (predicted_lines is None or len(predicted_lines) == 0
            or actual_lines is None or len(actual_lines) == 0):
        return {
            "overall":        0.0,
            "by_confidence":  {"high": 0.0, "mid": 0.0, "low": 0.0},
            "by_grade":       {},
            "by_venue":       {},
            "n_samples":      0,
        }

    merged = predicted_lines.merge(
        actual_lines[["race_id", "car_no", "actual_line"]],
        on=["race_id", "car_no"],
        how="inner",
    )
    if len(merged) == 0:
        return {
            "overall":        0.0,
            "by_confidence":  {"high": 0.0, "mid": 0.0, "low": 0.0},
            "by_grade":       {},
            "by_venue":       {},
            "n_samples":      0,
        }

    merged["correct"] = (merged["predicted_line"] == merged["actual_line"])

    overall = float(merged["correct"].mean())

    # 信頼度別
    high_mask = merged["confidence"] > 0.7
    mid_mask  = (merged["confidence"] >= 0.5) & ~high_mask
    low_mask  = merged["confidence"] < 0.5

    by_confidence = {
        "high": float(merged.loc[high_mask, "correct"].mean()) if high_mask.sum() > 0 else 0.0,
        "mid":  float(merged.loc[mid_mask,  "correct"].mean()) if mid_mask.sum()  > 0 else 0.0,
        "low":  float(merged.loc[low_mask,  "correct"].mean()) if low_mask.sum()  > 0 else 0.0,
    }

    # グレード別
    by_grade = {}
    if "grade" in merged.columns:
        by_grade = merged.groupby("grade")["correct"].mean().round(4).to_dict()

    # 会場別
    by_venue = {}
    if "venue_name" in merged.columns:
        by_venue = merged.groupby("venue_name")["correct"].mean().round(4).to_dict()

    return {
        "overall":       round(overall, 4),
        "by_confidence": {k: round(v, 4) for k, v in by_confidence.items()},
        "by_grade":      by_grade,
        "by_venue":      by_venue,
        "n_samples":     len(merged),
    }


# =============================================================
# 指標サマリー
# =============================================================

def calc_all_metrics(
    signals: list,
    results_df: pd.DataFrame,
    capital_series: pd.Series,
) -> dict:
    """
    全指標を一括計算してサマリーを返す。

    Returns:
        {
            "roi":          float,
            "max_drawdown": float,
            "sharpe":       float,
            "profit_factor": float,
            "n_bets":       int,
            "calibration":  dict,
        }
    """
    valid_signals = [s for s in signals if s.filters_passed]

    returns_list = []
    result_lookup = {}
    if results_df is not None:
        for _, row in results_df.iterrows():
            result_lookup[row["race_id"]] = row

    for signal in valid_signals:
        result = result_lookup.get(signal.race_id)
        if result is not None:
            actual_combo = (
                int(result.get("rank1_car", 0)),
                int(result.get("rank2_car", 0)),
                int(result.get("rank3_car", 0)),
            )
            if actual_combo == signal.combo:
                payout = float(result.get("trifecta_payout", 0))
                ret    = payout / 100 - 1.0
            else:
                ret = -1.0
            returns_list.append(ret)

    returns_series = pd.Series(returns_list) if returns_list else pd.Series(dtype=float)

    roi          = calc_roi(signals, results_df)
    mdd          = calc_max_drawdown(capital_series)
    sharpe       = calc_sharpe(returns_series)
    pf           = calc_profit_factor(signals, results_df)
    calibration  = calc_calibration(signals, results_df)

    return {
        "roi":           roi,
        "max_drawdown":  mdd,
        "sharpe":        sharpe,
        "profit_factor": pf,
        "n_bets":        len(valid_signals),
        "calibration":   calibration,
    }


def check_paper_trade_ready(metrics: dict) -> tuple:
    """
    ペーパートレード移行条件を確認する。

    Returns:
        (is_ready, reasons)
        is_ready: bool
        reasons : 不合格理由のリスト
    """
    reasons = []

    if metrics.get("roi", -999) <= PAPER_TRADE_CRITERIA["roi_min"]:
        reasons.append(f"ROI不足（{metrics.get('roi', 0)*100:.1f}% ≤ {PAPER_TRADE_CRITERIA['roi_min']*100}%）")

    if metrics.get("max_drawdown", 999) >= PAPER_TRADE_CRITERIA["max_drawdown_max"]:
        reasons.append(
            f"MDD過大（{metrics.get('max_drawdown', 0)*100:.1f}% ≥ {PAPER_TRADE_CRITERIA['max_drawdown_max']*100}%）"
        )

    calib = metrics.get("calibration", {})
    if calib.get("max_error", 999) >= PAPER_TRADE_CRITERIA["calibration_error_max"]:
        reasons.append(
            f"キャリブレーション誤差大（{calib.get('max_error', 0)*100:.1f}% ≥ {PAPER_TRADE_CRITERIA['calibration_error_max']*100}%）"
        )

    is_ready = len(reasons) == 0
    return is_ready, reasons


def print_metrics_report(metrics: dict):
    """指標レポートをコンソールに出力する"""
    print("=" * 50)
    print("競輪AI 評価レポート")
    print("=" * 50)
    print(f"ROI          : {metrics.get('roi', 0)*100:.2f}%")
    print(f"最大MDD      : {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe比     : {metrics.get('sharpe', 0):.4f}")
    print(f"PF           : {metrics.get('profit_factor', 0):.4f}")
    print(f"総ベット数   : {metrics.get('n_bets', 0)}")

    calib = metrics.get("calibration", {})
    print(f"キャリブレーション最大誤差: {calib.get('max_error', 0)*100:.2f}%")
    print(f"  → well-calibrated: {calib.get('is_calibrated', False)}")

    is_ready, reasons = check_paper_trade_ready(metrics)
    print("-" * 50)
    if is_ready:
        print("✓ ペーパートレード移行条件：全て満たしています")
    else:
        print("× ペーパートレード移行条件：未達")
        for r in reasons:
            print(f"  - {r}")
    print("=" * 50)
