# ===========================================
# betting_logic.py
# ボートレースAI - EV計算・Kelly基準・買い方ロジック
#
# 設計方針：
#   - EV = 予測確率 × オッズ × 0.75（控除率25%を必ず考慮）
#   - オッズ急上昇フィルター必須（市場に謙虚になる設計）
#   - Kelly基準の1/4（保守的運用）
#   - データ不要・即実装完了
# ===========================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# =============================================================
# 定数
# =============================================================

TAKEOUT_RATE    = 0.25   # 控除率25%（ボートレース）
EV_THRESHOLD    = 1.1    # 購入最低EV
MIN_PROB        = 0.01   # 最低予測確率（1%未満は統計的に信頼できない）
MIN_ODDS        = 5.0    # 最低オッズ
MAX_ODDS        = 500.0  # 最大オッズ（異常値除外）
KELLY_FRACTION  = 0.25   # Kelly基準の1/4（保守的）
MAX_BET_RATIO   = 0.05   # 1レース最大投資額（総資金の5%）

# 予測可能性スコアの閾値
PREDICTABILITY_THRESHOLD = 60  # 60未満のレースはスキップ

# オッズ急上昇フィルターの閾値
SURGE_CANCEL_THRESHOLD  = 0.20  # 20%以上上昇 → 買いキャンセル
SURGE_HALF_THRESHOLD    = 0.10  # 10%以上上昇 → 信頼度50%カット
SURGE_REDUCE_THRESHOLD  = 0.05  # 5%以上上昇  → 信頼度20%カット


# =============================================================
# データクラス
# =============================================================

@dataclass
class BettingSignal:
    """買いシグナルの単位。"""
    combination: str          # 例: "2-1-3"
    predicted_prob: float     # モデルの予測確率
    odds: float               # 購入時点のオッズ
    ev: float                 # EV値
    confidence: float = 1.0   # 信頼度（0〜1）
    bet_amount: float = 0.0   # 賭け金（円）
    cancelled: bool = False   # 買いキャンセルフラグ
    cancel_reason: str = ""   # キャンセル理由


@dataclass
class RaceSignal:
    """1レース分の買いシグナル集約。"""
    race_id: str
    venue_name: str
    race_no: int
    deadline_time: str
    course_display: str        # "3-1-2-4-5-6" 形式
    predictability_score: int
    candidates: List[BettingSignal] = field(default_factory=list)
    total_bet: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


# =============================================================
# EV計算（最重要）
# =============================================================

def calc_ev(predicted_prob: float, odds: float) -> float:
    """
    EV = 予測確率 × オッズ × (1 - 控除率)

    注意：控除率を必ず考慮する。
    間違い: EV = prob × odds
    正解:   EV = prob × odds × 0.75

    Parameters:
        predicted_prob : モデルの予測確率（0〜1）
        odds           : オッズ（倍）

    Returns:
        ev: EV値（1.0以上が期待値プラス）
    """
    return predicted_prob * odds * (1.0 - TAKEOUT_RATE)


# =============================================================
# レース予測可能性スコア
# =============================================================

def calc_predictability_score(race_data: dict) -> int:
    """
    予測しやすいレースか判定する。
    スコアが PREDICTABILITY_THRESHOLD（60）未満のレースはスキップする。

    Parameters:
        race_data: {
            "wind_speed": float,
            "wave_height": float,
            "course_taken": dict {lane: course},  # 進入コース情報
            "racer_classes": list[int],            # 各艇の級別
            "flying_racers": list[int],            # F持ち選手の艇番
        }

    Returns:
        score: 0〜100のスコア（高いほど予測しやすい）
    """
    score = 100

    # 強風（7m/s以上） → -25点
    wind_speed = race_data.get("wind_speed", 0) or 0
    if wind_speed >= 7.0:
        score -= 25

    # 高波（15cm以上） → -20点
    wave_height = race_data.get("wave_height", 0) or 0
    if wave_height >= 15.0:
        score -= 20

    # 進入が3艇以上乱れている → -30点
    course_taken = race_data.get("course_taken", {})
    n_changed = sum(1 for lane, course in course_taken.items() if lane != course)
    if n_changed >= 3:
        score -= 30

    # 実力が拮抗している（std < 0.5） → -15点
    racer_classes = race_data.get("racer_classes", [])
    if racer_classes:
        import numpy as np
        std = float(np.std(racer_classes))
        if std < 0.5:
            score -= 15

    # F持ち選手が2人以上 → -20点
    flying_racers = race_data.get("flying_racers", [])
    if len(flying_racers) >= 2:
        score -= 20

    return max(0, score)


# =============================================================
# オッズ急上昇フィルター（最重要）
# =============================================================

def apply_odds_surge_filter(signal: BettingSignal, odds_history: dict) -> BettingSignal:
    """
    モデルが買いと判断した艇のオッズが直前に急上昇している場合は
    信頼度を下げる、または買いをキャンセルする。

    設計思想：「市場が知っていて自分が知らない情報」に謙虚になる。
    自分のモデルが推奨した艇にのみ適用する。

    15分前→1分前の変化率で判定：
        5%未満   → 変更なし
        5〜10%  → 信頼度を20%カット
        10〜20% → 信頼度を50%カット
        20%以上  → 買いシグナルを完全取り消し

    Parameters:
        signal      : 買いシグナル
        odds_history: {"15min": float, "1min": float, ...}

    Returns:
        signal（信頼度・金額が更新される場合あり）
    """
    odds_15min = odds_history.get("15min")
    odds_1min  = odds_history.get("1min")

    if odds_15min is None or odds_1min is None or odds_15min <= 0:
        return signal

    surge_rate = (odds_1min - odds_15min) / odds_15min

    if surge_rate >= SURGE_CANCEL_THRESHOLD:
        signal.cancelled    = True
        signal.confidence   = 0.0
        signal.bet_amount   = 0.0
        signal.cancel_reason = f"オッズ急上昇（{surge_rate:.1%}上昇）→ 買いキャンセル"

    elif surge_rate >= SURGE_HALF_THRESHOLD:
        signal.confidence  *= 0.5
        signal.bet_amount  *= 0.5
        signal.cancel_reason = f"オッズ上昇（{surge_rate:.1%}）→ 信頼度50%カット"

    elif surge_rate >= SURGE_REDUCE_THRESHOLD:
        signal.confidence  *= 0.8
        signal.bet_amount  *= 0.8
        signal.cancel_reason = f"オッズ上昇（{surge_rate:.1%}）→ 信頼度20%カット"

    return signal


# =============================================================
# 購入フィルター
# =============================================================

def filter_bets(
    trifecta_probs: dict,
    odds_dict: dict,
    capital: float,
) -> List[BettingSignal]:
    """
    購入候補を絞り込む。

    全条件を満たすものだけ購入リストに入れる：
    ① EV >= EV_THRESHOLD（1.1）
    ② 予測確率 >= MIN_PROB（1%）
    ③ オッズ >= MIN_ODDS（5倍）
    ④ オッズ <= MAX_ODDS（500倍）

    Parameters:
        trifecta_probs : {"1-2-3": 0.063, ...} モデルの3連単確率
        odds_dict      : {"1-2-3": 25.0, ...}  実際のオッズ
        capital        : 現在の資金（円）

    Returns:
        candidates: フィルタを通過したBettingSignalのリスト（EV降順）
    """
    candidates = []

    for combo, prob in trifecta_probs.items():
        odds = odds_dict.get(combo)
        if odds is None:
            continue

        # フィルター適用
        if prob < MIN_PROB:
            continue
        if odds < MIN_ODDS or odds > MAX_ODDS:
            continue

        ev = calc_ev(prob, odds)
        if ev < EV_THRESHOLD:
            continue

        bet_amt = kelly_bet(prob, odds, capital)

        candidates.append(BettingSignal(
            combination    = combo,
            predicted_prob = prob,
            odds           = odds,
            ev             = ev,
            confidence     = 1.0,
            bet_amount     = bet_amt,
        ))

    # EV降順でソート
    candidates.sort(key=lambda s: s.ev, reverse=True)
    return candidates


# =============================================================
# Kelly基準による賭け金計算
# =============================================================

def kelly_bet(prob: float, odds: float, capital: float) -> float:
    """
    Kelly基準で賭け金を計算する。

    f = (p × (b+1) - 1) / b
    f: 賭け金の割合
    p: 予測的中確率
    b: オッズ - 1

    保守的に1/4にする：
    actual_bet = f × KELLY_FRACTION × capital

    上限：capital × MAX_BET_RATIO（5%）

    Parameters:
        prob    : 予測的中確率
        odds    : オッズ
        capital : 現在の資金（円）

    Returns:
        賭け金額（円）
    """
    if odds <= 1.0 or prob <= 0 or capital <= 0:
        return 0.0

    b = odds - 1.0
    kelly = (prob * (b + 1.0) - 1.0) / b
    kelly = max(0.0, kelly)

    conservative = kelly * KELLY_FRACTION
    bet = conservative * capital

    # 上限: 資金の5%
    max_bet = capital * MAX_BET_RATIO

    # 100円単位に丸める
    bet = min(bet, max_bet)
    bet = max(100.0, round(bet / 100) * 100)

    return bet


# =============================================================
# ドローダウン管理
# =============================================================

def check_drawdown(current_capital: float, peak_capital: float) -> Tuple[str, float]:
    """
    ドローダウンに応じて自動調整する。

    20%減少 → 賭け金を半額に（警告）
    40%減少 → 自動停止（例外を投げる）

    Returns:
        (status, bet_multiplier): ("normal"/"warning"/"stop", 倍率)
    """
    if peak_capital <= 0:
        return ("normal", 1.0)

    drawdown = (peak_capital - current_capital) / peak_capital

    if drawdown >= 0.40:
        raise RuntimeError(
            f"ドローダウン{drawdown:.1%}が40%を超えました。自動停止します。"
            f"（現在資金: {current_capital:,.0f}円 / ピーク: {peak_capital:,.0f}円）"
        )
    elif drawdown >= 0.20:
        return ("warning", 0.5)  # 賭け金を半額に
    else:
        return ("normal", 1.0)


# =============================================================
# 出力フォーマット
# =============================================================

def format_betting_signal(race_signal: RaceSignal) -> str:
    """
    レース直前の推奨購入リストを文字列で返す。

    出力例：
    =====================================
    【戸田 第8レース】14:30締め切り
    進入コース：3-1-2-4-5-6
    予測可能性スコア：78/100
    --- 3連単 推奨購入リスト ---
    1位: 2→1→3  確率6.3%  オッズ25倍  EV=1.18 → ★購入
    2位: 1→2→3  確率8.1%  オッズ12倍  EV=0.73 → 見送り
    推奨ベット：
    2→1→3: 500円 (EV=1.18, Kelly=0.8%)
    =====================================
    """
    lines = [
        "=" * 45,
        f"【{race_signal.venue_name} 第{race_signal.race_no}レース】"
        f"{race_signal.deadline_time}締め切り",
        "",
        f"進入コース：{race_signal.course_display}",
        f"予測可能性スコア：{race_signal.predictability_score}/100",
    ]

    if race_signal.skipped:
        lines.append(f"\n【スキップ】{race_signal.skip_reason}")
        lines.append("=" * 45)
        return "\n".join(lines)

    lines.append("\n--- 3連単 推奨購入リスト ---")

    buy_candidates = [s for s in race_signal.candidates if not s.cancelled]
    skip_candidates = [s for s in race_signal.candidates if s.cancelled]

    for i, sig in enumerate(buy_candidates[:5], 1):
        combo_disp = sig.combination.replace("-", "→")
        lines.append(
            f"{i}位: {combo_disp}  確率{sig.predicted_prob:.1%}  "
            f"オッズ{sig.odds:.0f}倍  EV={sig.ev:.2f} → ★購入"
        )

    for sig in skip_candidates[:3]:
        combo_disp = sig.combination.replace("-", "→")
        reason = sig.cancel_reason or "EV不足"
        lines.append(
            f"   {combo_disp}  確率{sig.predicted_prob:.1%}  "
            f"オッズ{sig.odds:.0f}倍  EV={sig.ev:.2f} → 見送り（{reason}）"
        )

    if buy_candidates:
        lines.append("\n推奨ベット：")
        for sig in buy_candidates[:5]:
            combo_disp = sig.combination.replace("-", "→")
            lines.append(
                f"  {combo_disp}: {sig.bet_amount:,.0f}円 "
                f"(EV={sig.ev:.2f}, Kelly={sig.predicted_prob:.1%})"
            )
        lines.append(f"\n推奨合計: {race_signal.total_bet:,.0f}円")
    else:
        lines.append("\n推奨購入なし（条件を満たす組み合わせなし）")

    lines.append("=" * 45)
    return "\n".join(lines)


# =============================================================
# レース全体の処理フロー
# =============================================================

def process_race(
    race_info: dict,
    trifecta_probs: dict,
    odds_dict: dict,
    odds_history_by_lane: dict,
    capital: float,
    peak_capital: float,
) -> RaceSignal:
    """
    1レース分の買いシグナル処理フロー。

    Parameters:
        race_info            : レース情報（venue_name/race_no/deadline_time等）
        trifecta_probs       : モデルの3連単確率
        odds_dict            : 現在のオッズ
        odds_history_by_lane : {lane: {"15min": float, "1min": float, ...}}
        capital              : 現在の資金
        peak_capital         : 資金のピーク値

    Returns:
        RaceSignal
    """
    # 進入コース表示
    course_taken = race_info.get("course_taken", {})
    course_display = _format_course_display(course_taken)

    # 予測可能性スコア
    pred_score = calc_predictability_score(race_info)

    signal = RaceSignal(
        race_id            = race_info.get("race_id", ""),
        venue_name         = race_info.get("venue_name", ""),
        race_no            = race_info.get("race_no", 0),
        deadline_time      = race_info.get("deadline_time", ""),
        course_display     = course_display,
        predictability_score = pred_score,
    )

    # 予測可能性スコアが低い → スキップ
    if pred_score < PREDICTABILITY_THRESHOLD:
        signal.skipped     = True
        signal.skip_reason = f"予測可能性スコア{pred_score}/100 < 閾値{PREDICTABILITY_THRESHOLD}"
        return signal

    # ドローダウンチェック
    try:
        dd_status, bet_multiplier = check_drawdown(capital, peak_capital)
    except RuntimeError as e:
        signal.skipped     = True
        signal.skip_reason = str(e)
        return signal

    # 購入候補のフィルタリング
    candidates = filter_bets(trifecta_probs, odds_dict, capital)

    # 各候補にオッズ急上昇フィルターを適用
    for cand in candidates:
        # 組み合わせ "X-Y-Z" の1着艇にフィルターを適用
        first_lane = int(cand.combination.split("-")[0])
        odds_hist = odds_history_by_lane.get(first_lane, {})
        cand = apply_odds_surge_filter(cand, odds_hist)
        cand.bet_amount *= bet_multiplier  # DDによる縮小

    signal.candidates = candidates
    signal.total_bet  = sum(
        s.bet_amount for s in candidates if not s.cancelled
    )

    return signal


def _format_course_display(course_taken: dict) -> str:
    """進入コース表示文字列を生成する。例: "3-1-2-4-5-6"""
    if not course_taken:
        return "1-2-3-4-5-6"
    # コース→艇番のマッピングに変換して表示
    course_to_lane = {v: k for k, v in course_taken.items()}
    result = []
    for c in range(1, 7):
        lane = course_to_lane.get(c, "?")
        result.append(str(lane))
    return "-".join(result)
