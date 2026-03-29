# ===========================================
# kelly.py
# 全市場共通 - Kelly基準による資金管理
#
# 設計方針：
#   - 全市場（ボートレース・FX・株・暗号通貨）で使える
#   - 保守的に1/4 Kelly（過最適化リスクを避ける）
#   - ドローダウンに応じて自動縮小
# ===========================================

import numpy as np


# =============================================================
# 全市場共通のKelly基準
# =============================================================

def kelly_universal(
    prob:          float,
    odds_or_ratio: float,
    capital:       float,
    fraction:      float = 0.25,
    max_ratio:     float = 0.05,
    min_bet:       float = 0.0,
    unit:          float = 1.0,
) -> float:
    """
    全市場で使えるKelly基準による賭け金計算。

    Kelly公式:
        f* = (p × (b + 1) - 1) / b
        f*: 資金に対する最適な賭け比率
        p : 勝率（予測確率）
        b : 純リターン比率（ボートレースはオッズ-1、FXはR/R比率等）

    Parameters:
        prob          : 勝率（0〜1）
        odds_or_ratio : ボートレース→オッズ / FX・株・暗号通貨→利益/損失の比率
        capital       : 現在の資金
        fraction      : Kellyの何分の1を使うか（デフォルト1/4）
        max_ratio     : 1取引の最大賭け比率（デフォルト5%）
        min_bet       : 最小賭け金（0の場合は無視）
        unit          : 丸めの単位（ボートレース→100円、FX→ロット等）

    Returns:
        bet: 賭け金額（単位はcapitalと同じ）
    """
    if prob <= 0 or prob >= 1 or odds_or_ratio <= 1.0 or capital <= 0:
        return 0.0

    b = odds_or_ratio - 1.0
    kelly_full = (prob * (b + 1.0) - 1.0) / b
    kelly_full = max(0.0, kelly_full)

    # 保守的に fraction 倍にする
    kelly_conservative = kelly_full * fraction

    # 賭け金計算
    bet = kelly_conservative * capital

    # 上限: 資金の max_ratio
    bet = min(bet, capital * max_ratio)

    # 下限
    if min_bet > 0:
        bet = max(min_bet, bet) if bet > 0 else 0.0

    # 単位に丸める
    if unit > 0:
        bet = round(bet / unit) * unit

    return float(bet)


# =============================================================
# ドローダウンに応じた賭け金の自動調整
# =============================================================

def kelly_with_drawdown_adjustment(
    prob:          float,
    odds_or_ratio: float,
    capital:       float,
    peak_capital:  float,
    fraction:      float = 0.25,
    max_ratio:     float = 0.05,
    unit:          float = 1.0,
) -> float:
    """
    ドローダウンに応じてKelly賭け金を自動調整する。

    ドローダウン < 10%  : 通常通り
    ドローダウン 10〜20% : 賭け金を75%に縮小
    ドローダウン 20〜30% : 賭け金を50%に縮小（警告）
    ドローダウン 30〜40% : 賭け金を25%に縮小
    ドローダウン ≥ 40%  : 停止（0を返す）

    Parameters:
        capital      : 現在の資金
        peak_capital : 資金のピーク値

    Returns:
        bet: 調整後の賭け金額
    """
    base_bet = kelly_universal(prob, odds_or_ratio, capital, fraction, max_ratio, unit=unit)

    if peak_capital <= 0 or capital >= peak_capital:
        return base_bet

    drawdown = (peak_capital - capital) / peak_capital

    if drawdown >= 0.40:
        return 0.0  # 自動停止
    elif drawdown >= 0.30:
        mult = 0.25
    elif drawdown >= 0.20:
        mult = 0.50
    elif drawdown >= 0.10:
        mult = 0.75
    else:
        mult = 1.0

    adjusted = base_bet * mult
    if unit > 0:
        adjusted = round(adjusted / unit) * unit

    return float(adjusted)


# =============================================================
# ポートフォリオKelly（複数ポジション同時保有時）
# =============================================================

def kelly_portfolio(
    positions: list,
    capital:   float,
    max_total_ratio: float = 0.20,
) -> list:
    """
    複数の取引候補に対してKelly配分を計算する。

    Parameters:
        positions: [
            {"prob": float, "odds": float, "name": str},
            ...
        ]
        capital        : 現在の資金
        max_total_ratio: 全ポジション合計の最大比率（デフォルト20%）

    Returns:
        list of {"name": str, "bet": float, "kelly_f": float}
    """
    results = []
    total_bet = 0.0

    for pos in positions:
        bet = kelly_universal(
            pos["prob"], pos["odds"], capital,
            fraction=0.25, max_ratio=0.05,
        )
        kelly_f = bet / capital if capital > 0 else 0.0
        results.append({"name": pos.get("name", ""), "bet": bet, "kelly_f": kelly_f})
        total_bet += bet

    # 全体が max_total_ratio を超える場合は按分縮小
    if total_bet > capital * max_total_ratio:
        scale = (capital * max_total_ratio) / total_bet
        for r in results:
            r["bet"] *= scale

    return results


# =============================================================
# ユーティリティ
# =============================================================

def expected_value(prob: float, odds: float, takeout: float = 0.0) -> float:
    """
    期待値（EV）を計算する。

    Parameters:
        prob    : 勝率
        odds    : オッズ
        takeout : 控除率（ボートレース=0.25）

    Returns:
        ev: 期待値（1.0=ブレイクイーブン）
    """
    return prob * odds * (1.0 - takeout)


def calc_optimal_kelly_ratio(
    prob: float, odds: float, n_simulations: int = 10000, seed: int = 42
) -> dict:
    """
    シミュレーションでKelly比率の最適値と実際のパフォーマンスを検証する。

    Parameters:
        prob          : 勝率
        odds          : オッズ
        n_simulations : 試行回数

    Returns:
        result: {"full_kelly": float, "quarter_kelly": float,
                 "growth_rate_full": float, "growth_rate_quarter": float}
    """
    rng = np.random.default_rng(seed)
    b = odds - 1.0

    if b <= 0 or prob <= 0:
        return {}

    full_kelly = max(0.0, (prob * (b + 1.0) - 1.0) / b)
    quarter_kelly = full_kelly * 0.25

    def simulate(f):
        capital = 1.0
        for _ in range(n_simulations):
            if rng.random() < prob:
                capital *= (1.0 + f * b)
            else:
                capital *= (1.0 - f)
            if capital <= 0:
                return 0.0
        return capital

    growth_full    = simulate(full_kelly) ** (1.0 / n_simulations)
    growth_quarter = simulate(quarter_kelly) ** (1.0 / n_simulations)

    return {
        "full_kelly":          round(full_kelly, 4),
        "quarter_kelly":       round(quarter_kelly, 4),
        "growth_rate_full":    round(growth_full, 6),
        "growth_rate_quarter": round(growth_quarter, 6),
        "n_simulations":       n_simulations,
    }
