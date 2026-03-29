# ===========================================
# model/betting_logic.py
# 競輪AI - EV計算・Kelly基準・買いロジック
#
# ボートレースの betting_logic.py をベースに
# 競輪固有の以下を追加：
#   - ライン信頼度のペナルティ
#   - 裏切りリスクフィルター
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# 控除率 25%（ボートレースと同じ）
TAKEOUT_RATE = 0.25

# Kelly 分数（保守的設定：フルKelly×0.25）
KELLY_FRACTION = 0.25

# 賭け金上限（資金の5%まで）
MAX_BET_RATIO = 0.05

# ドローダウン制御
DD_WARNING_THRESHOLD  = 0.20   # 20% → 賭け金半減
DD_STOP_THRESHOLD     = 0.40   # 40% → 強制停止

# 予測可能性スコア閾値
PREDICTABILITY_MIN = 60  # これ以下は見送り推奨

# EV閾値
EV_MIN = 1.10  # EV ≥ 1.1 のみ購入候補


@dataclass
class BettingSignal:
    """1レースの買いシグナル"""
    race_id:             str
    combo:               tuple          # (1st, 2nd, 3rd) の車番
    bet_type:            str            # "trifecta"（3連単）
    predicted_prob:      float          # 予測確率
    odds:                float          # オッズ
    ev:                  float          # 期待値
    kelly_bet:           float          # Kelly基準の賭け金額
    confidence:          float          # 信頼度（ライン信頼度を考慮）
    predictability_score: int           # 予測可能性スコア
    filters_passed:      bool           # 全フィルターを通過したか
    filter_reason:       str            # フィルター不通過の理由


def calc_ev(prob: float, odds: float) -> float:
    """
    期待値を計算する（控除率25%を考慮）。

    EV = 予測確率 × オッズ × (1 - TAKEOUT_RATE)

    Parameters:
        prob : 予測確率 0〜1
        odds : 3連単オッズ

    Returns:
        EV（1.0より大きければプラス期待値）
    """
    if odds <= 0 or prob <= 0:
        return 0.0
    return prob * odds * (1.0 - TAKEOUT_RATE)


def calc_predictability_score(
    race_data: dict,
    line_probs: dict,
    wind_speed: float,
    is_rain: bool,
) -> int:
    """
    予測可能性スコアを計算する（0〜100）。

    ボートレースとの違い：
    ライン信頼度が低い場合のペナルティを追加（ボートより大きい）

    スコアを下げる要素：
    - ライン信頼度の平均 < 0.5 → -30点
    - 単騎が3人以上             → -20点
    - 風速4m/s以上              → -15点
    - 雨                        → -10点
    - グレードが一般戦（F1/F2） → -5点

    Returns:
        int: 0〜100
    """
    score = 100

    # ライン信頼度チェック
    if line_probs:
        avg_conf = float(np.mean(list(line_probs.values())))
        if avg_conf < 0.50:
            score -= 30

    # 単騎人数チェック
    n_singles = sum(1 for k in line_probs if "-" not in k)
    if n_singles >= 3:
        score -= 20

    # 風速チェック
    if wind_speed >= 4.0:
        score -= 15

    # 雨チェック
    if is_rain:
        score -= 10

    # グレードチェック
    grade = race_data.get("grade", "F1")
    if grade in ("F1", "F2"):
        score -= 5

    return max(0, min(100, score))


def apply_odds_surge_filter(
    signal: BettingSignal,
    odds_history: dict,
) -> BettingSignal:
    """
    オッズ急変フィルター（ボートレースと同じ設計）。

    15分前オッズ → 1分前オッズの変化を監視：
    - ≥20%上昇 → 買いシグナル完全取り消し（filters_passed=False）
    - 10〜20%上昇 → 信頼度を50%カット
    - 5〜10%上昇  → 信頼度を20%カット

    Parameters:
        signal      : BettingSignal
        odds_history: {"15min": float, "1min": float} の dict

    Returns:
        修正済みの BettingSignal
    """
    odds_15min = odds_history.get("15min")
    odds_1min  = odds_history.get("1min")

    if odds_15min is None or odds_1min is None or odds_15min <= 0:
        return signal  # データなしはスルー

    change_rate = (odds_1min - odds_15min) / odds_15min

    if change_rate >= 0.20:
        signal.filters_passed = False
        signal.filter_reason  = f"オッズ急上昇（+{change_rate*100:.1f}%）→ 買い取り消し"
    elif change_rate >= 0.10:
        signal.confidence *= 0.50
        signal.kelly_bet  *= 0.50
        signal.filter_reason = f"オッズ上昇（+{change_rate*100:.1f}%）→ 信頼度50%カット"
    elif change_rate >= 0.05:
        signal.confidence *= 0.80
        signal.kelly_bet  *= 0.80
        signal.filter_reason = f"オッズ上昇（+{change_rate*100:.1f}%）→ 信頼度20%カット"

    return signal


def apply_line_betrayal_filter(
    signal: BettingSignal,
    betrayal_risk: float,
) -> BettingSignal:
    """
    競輪固有：裏切りリスクフィルター。

    裏切りリスクが高い選手が推奨ラインに含まれる場合：
    - 10〜20% → 信頼度を40%カット
    - ≥20%   → 買いシグナルを取り消し

    Parameters:
        signal        : BettingSignal
        betrayal_risk : 裏切り率 0〜1

    Returns:
        修正済みの BettingSignal
    """
    if betrayal_risk >= 0.20:
        signal.filters_passed = False
        signal.filter_reason  = f"裏切りリスク高（{betrayal_risk*100:.1f}%）→ 買い取り消し"
    elif betrayal_risk >= 0.10:
        signal.confidence *= 0.60
        signal.kelly_bet  *= 0.60
        signal.filter_reason = f"裏切りリスク（{betrayal_risk*100:.1f}%）→ 信頼度40%カット"

    return signal


def kelly_bet(
    prob: float,
    odds: float,
    capital: float,
    drawdown_ratio: float = 0.0,
) -> float:
    """
    Kelly基準で賭け金を計算する（ボートレースと同じ設計）。

    - フルKelly × 0.25（保守的設定）
    - 最大5%/レース上限
    - ドローダウン制御

    Parameters:
        prob            : 予測確率
        odds            : オッズ
        capital         : 現在の資金
        drawdown_ratio  : 現在のドローダウン率（0〜1）

    Returns:
        賭け金額（円）
    """
    if odds <= 1 or prob <= 0 or capital <= 0:
        return 0.0

    # Kelly分数 f* = (p*(b+1) - 1) / b
    b = odds - 1
    kelly_f = (prob * (b + 1) - 1.0) / b

    if kelly_f <= 0:
        return 0.0

    # フルKelly × 0.25（保守的）
    fraction = kelly_f * KELLY_FRACTION

    # ドローダウン制御
    if drawdown_ratio >= DD_STOP_THRESHOLD:
        raise RuntimeError(
            f"最大ドローダウン{drawdown_ratio*100:.1f}%に到達。強制停止します。"
        )
    if drawdown_ratio >= DD_WARNING_THRESHOLD:
        fraction *= 0.50  # ドローダウン20%以上で半減

    # 資金の5%上限
    bet_size = min(fraction * capital, capital * MAX_BET_RATIO)

    return round(bet_size, 0)


def check_drawdown(
    peak_capital: float,
    current_capital: float,
) -> tuple:
    """
    現在のドローダウンを確認する。

    Returns:
        (drawdown_ratio, warning_level)
        warning_level: "ok" / "warning" / "stop"
    """
    if peak_capital <= 0:
        return 0.0, "ok"

    dd_ratio = (peak_capital - current_capital) / peak_capital

    if dd_ratio >= DD_STOP_THRESHOLD:
        return dd_ratio, "stop"
    elif dd_ratio >= DD_WARNING_THRESHOLD:
        return dd_ratio, "warning"
    else:
        return dd_ratio, "ok"


def generate_betting_signals(
    trifecta_probs: dict,
    odds_dict: dict,
    race_data: dict,
    line_probs: dict,
    capital: float,
    drawdown_ratio: float = 0.0,
    betrayal_risks: dict = None,
    odds_history: dict = None,
) -> list:
    """
    3連単全通りから買いシグナルを生成する。

    Parameters:
        trifecta_probs  : {(1st,2nd,3rd): 確率}
        odds_dict       : {(1st,2nd,3rd): オッズ}
        race_data       : レース情報（grade等）
        line_probs      : ライン予測確率
        capital         : 現在の資金
        drawdown_ratio  : ドローダウン率
        betrayal_risks  : {car_no: 裏切り率} （省略可）
        odds_history    : {car_no: {"15min": float, "1min": float}} （省略可）

    Returns:
        BettingSignalのリスト（EV≥1.1のもの）
    """
    wind_speed = float(race_data.get("wind_speed", 0.0))
    is_rain    = bool(race_data.get("is_rain", False))

    predictability = calc_predictability_score(
        race_data, line_probs, wind_speed, is_rain
    )

    signals = []

    for combo, prob in trifecta_probs.items():
        if prob <= 0:
            continue

        odds = odds_dict.get(combo, 0.0)
        if odds <= 0:
            continue

        ev = calc_ev(prob, odds)
        if ev < EV_MIN:
            continue

        # ライン内の平均信頼度を計算
        line_confidence = _calc_line_confidence_for_combo(combo, line_probs)

        bet = kelly_bet(prob, odds, capital, drawdown_ratio)

        signal = BettingSignal(
            race_id=race_data.get("race_id", ""),
            combo=combo,
            bet_type="trifecta",
            predicted_prob=round(prob, 6),
            odds=odds,
            ev=round(ev, 4),
            kelly_bet=bet,
            confidence=round(line_confidence, 4),
            predictability_score=predictability,
            filters_passed=True,
            filter_reason="",
        )

        # 予測可能性スコアが低い場合は見送り
        if predictability < PREDICTABILITY_MIN:
            signal.filters_passed = False
            signal.filter_reason  = f"予測可能性スコア低（{predictability}点）"

        # オッズ急変フィルター
        if odds_history and signal.filters_passed:
            car_no_1st = combo[0]
            h = odds_history.get(car_no_1st, {})
            signal = apply_odds_surge_filter(signal, h)

        # 裏切りリスクフィルター
        if betrayal_risks and signal.filters_passed:
            max_risk = max(
                betrayal_risks.get(c, 0.0) for c in combo
            )
            signal = apply_line_betrayal_filter(signal, max_risk)

        signals.append(signal)

    # EVが高い順に並べる
    signals.sort(key=lambda s: s.ev, reverse=True)
    return signals


def _calc_line_confidence_for_combo(
    combo: tuple,
    line_probs: dict,
) -> float:
    """
    3連単コンボに関連するラインの信頼度を取得する。

    先頭車番が含まれるラインの信頼度を返す。
    """
    first_car = combo[0]
    for line_str, conf in line_probs.items():
        cars = [int(c) for c in line_str.split("-") if c.isdigit()]
        if first_car in cars:
            return conf
    return 0.30  # ライン不明の場合は低め


def format_betting_signal(signal: BettingSignal) -> str:
    """買いシグナルを整形して文字列として返す"""
    status = "✓ 買い" if signal.filters_passed else "× スキップ"
    combo_str = "-".join(str(c) for c in signal.combo)
    return (
        f"[{status}] {combo_str} | "
        f"確率:{signal.predicted_prob*100:.2f}% | "
        f"オッズ:{signal.odds:.1f} | "
        f"EV:{signal.ev:.3f} | "
        f"賭け金:{signal.kelly_bet:.0f}円 | "
        f"信頼度:{signal.confidence:.2f} | "
        f"予測スコア:{signal.predictability_score}"
        + (f" | 理由:{signal.filter_reason}" if signal.filter_reason else "")
    )
