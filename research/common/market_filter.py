# ===========================================
# market_filter.py
# 全市場共通 - 市場異常検知フィルター
#
# 設計思想：
#   「市場が知っていて自分が知らない情報」
#   に対して謙虚になる。
#   モデルのシグナルより市場の異常な動きを優先する。
#
#   ボートレース → オッズ急上昇フィルター
#   株          → 出来高急増フィルター
#   FX          → スプレッド急拡大フィルター
#   暗号通貨     → Funding Rate急変フィルター
# ===========================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterResult:
    """フィルター判定結果。"""
    passed: bool          # True: シグナル維持 / False: 弱体化または取り消し
    confidence_mult: float  # 信頼度の乗数（1.0=変化なし、0.0=完全キャンセル）
    reason: str           # 判定理由


class MarketAnomalyFilter:
    """
    市場の異常を検知して信頼度を調整する。
    全市場共通インターフェース。
    """

    # =====================
    # ボートレース
    # =====================

    def check_boat_odds_surge(
        self,
        odds_15min: Optional[float],
        odds_1min:  Optional[float],
    ) -> FilterResult:
        """
        ボートレース：15分前→1分前のオッズ急上昇を検出する。

        急上昇はインサイダー情報（欠場・体調不良・八百長等）の
        可能性を示すシグナルとして扱う。

        Parameters:
            odds_15min : 15分前のオッズ
            odds_1min  : 1分前のオッズ

        Returns:
            FilterResult
        """
        if odds_15min is None or odds_1min is None or odds_15min <= 0:
            return FilterResult(passed=True, confidence_mult=1.0, reason="オッズ履歴なし")

        surge = (odds_1min - odds_15min) / odds_15min

        if surge >= 0.20:
            return FilterResult(
                passed=False, confidence_mult=0.0,
                reason=f"オッズ{surge:.1%}急上昇（20%超）→ 完全キャンセル",
            )
        elif surge >= 0.10:
            return FilterResult(
                passed=True, confidence_mult=0.5,
                reason=f"オッズ{surge:.1%}上昇（10〜20%）→ 信頼度50%",
            )
        elif surge >= 0.05:
            return FilterResult(
                passed=True, confidence_mult=0.8,
                reason=f"オッズ{surge:.1%}上昇（5〜10%）→ 信頼度80%",
            )
        else:
            return FilterResult(passed=True, confidence_mult=1.0, reason="正常範囲内")

    # =====================
    # 株
    # =====================

    def check_stock_volume_surge(
        self,
        recent_volume:   float,
        baseline_volume: float,
        threshold:       float = 3.0,
    ) -> FilterResult:
        """
        株：出来高急増を検出する。

        直前の出来高が通常の threshold 倍以上の場合は
        「市場が知っている情報」がある可能性として信頼度を下げる。

        Parameters:
            recent_volume   : 直近の出来高
            baseline_volume : 平常時の出来高ベースライン
            threshold       : 急増判定の倍率（デフォルト3倍）

        Returns:
            FilterResult
        """
        if baseline_volume <= 0:
            return FilterResult(passed=True, confidence_mult=1.0, reason="ベースラインなし")

        ratio = recent_volume / baseline_volume

        if ratio >= threshold * 2:
            return FilterResult(
                passed=False, confidence_mult=0.0,
                reason=f"出来高{ratio:.1f}倍（{threshold*2}倍超）→ 完全キャンセル",
            )
        elif ratio >= threshold:
            return FilterResult(
                passed=True, confidence_mult=0.5,
                reason=f"出来高{ratio:.1f}倍（{threshold}倍超）→ 信頼度50%",
            )
        elif ratio >= threshold * 0.7:
            return FilterResult(
                passed=True, confidence_mult=0.8,
                reason=f"出来高{ratio:.1f}倍（軽微な増加）→ 信頼度80%",
            )
        else:
            return FilterResult(passed=True, confidence_mult=1.0, reason="正常範囲内")

    # =====================
    # FX
    # =====================

    def check_fx_spread_widening(
        self,
        current_spread: float,
        baseline_spread: float,
        max_spread_pips: float = 3.0,
    ) -> FilterResult:
        """
        FX：スプレッド急拡大を検出する。

        スプレッドが閾値を超える場合は
        取引コストが増大するためシグナルを弱体化する。

        Parameters:
            current_spread  : 現在のスプレッド（pips）
            baseline_spread : 平常時のスプレッドベースライン（pips）
            max_spread_pips : 取引見送り閾値（pips）

        Returns:
            FilterResult
        """
        if current_spread >= max_spread_pips:
            return FilterResult(
                passed=False, confidence_mult=0.0,
                reason=f"スプレッド{current_spread:.1f}pips（上限{max_spread_pips}超）→ 見送り",
            )

        if baseline_spread <= 0:
            return FilterResult(passed=True, confidence_mult=1.0, reason="ベースラインなし")

        ratio = current_spread / baseline_spread

        if ratio >= 3.0:
            return FilterResult(
                passed=True, confidence_mult=0.5,
                reason=f"スプレッド{ratio:.1f}倍拡大 → 信頼度50%",
            )
        elif ratio >= 2.0:
            return FilterResult(
                passed=True, confidence_mult=0.8,
                reason=f"スプレッド{ratio:.1f}倍拡大 → 信頼度80%",
            )
        else:
            return FilterResult(passed=True, confidence_mult=1.0, reason="正常範囲内")

    # =====================
    # 暗号通貨
    # =====================

    def check_crypto_funding_rate(
        self,
        funding_rate: float,
        threshold_high:  float =  0.01,   # +1%以上 → 買われすぎ
        threshold_low:   float = -0.005,  # -0.5%以下 → 売られすぎ
    ) -> FilterResult:
        """
        暗号通貨：Funding Rate急変を検出する。

        Funding Rate（資金調達率）は先物市場の需給を示す。
        極端な値はポジションの偏りを示し、
        急反転（スクイーズ）のリスクが高まる。

        Parameters:
            funding_rate   : 現在のFunding Rate（小数表記: 0.01 = 1%）
            threshold_high : 買われすぎ閾値
            threshold_low  : 売られすぎ閾値

        Returns:
            FilterResult
        """
        abs_rate = abs(funding_rate)

        if abs_rate >= abs(threshold_high) * 2:
            return FilterResult(
                passed=False, confidence_mult=0.0,
                reason=f"Funding Rate={funding_rate:.3%}（極端な偏り）→ 完全キャンセル",
            )
        elif funding_rate >= threshold_high:
            return FilterResult(
                passed=True, confidence_mult=0.5,
                reason=f"Funding Rate={funding_rate:.3%}（高い・ロング偏り）→ 信頼度50%",
            )
        elif funding_rate <= threshold_low:
            return FilterResult(
                passed=True, confidence_mult=0.5,
                reason=f"Funding Rate={funding_rate:.3%}（低い・ショート偏り）→ 信頼度50%",
            )
        else:
            return FilterResult(passed=True, confidence_mult=1.0, reason="正常範囲内")

    # =====================
    # 統合API
    # =====================

    def apply(self, market: str, **kwargs) -> FilterResult:
        """
        市場名を指定して対応するフィルターを呼び出す統合API。

        Parameters:
            market : "boat" / "stock" / "fx" / "crypto"
            **kwargs: 各フィルターのパラメータ

        Returns:
            FilterResult
        """
        if market == "boat":
            return self.check_boat_odds_surge(
                kwargs.get("odds_15min"), kwargs.get("odds_1min")
            )
        elif market == "stock":
            return self.check_stock_volume_surge(
                kwargs.get("recent_volume", 0),
                kwargs.get("baseline_volume", 1),
                kwargs.get("threshold", 3.0),
            )
        elif market == "fx":
            return self.check_fx_spread_widening(
                kwargs.get("current_spread", 0),
                kwargs.get("baseline_spread", 0),
                kwargs.get("max_spread_pips", 3.0),
            )
        elif market == "crypto":
            return self.check_crypto_funding_rate(
                kwargs.get("funding_rate", 0),
            )
        else:
            raise ValueError(f"未対応の市場: {market}")
