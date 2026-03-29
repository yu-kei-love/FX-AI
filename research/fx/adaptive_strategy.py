# ===========================================
# adaptive_strategy.py
# FXモデル - 相場環境別パラメータ切り替え
#
# 背景：
#   低ボラ環境でスプレッドコストの相対影響が増大する。
#   環境に応じてパラメータを切り替えることで対応する。
#
# 重要：
#   パラメータを再最適化しない。
#   「見送り判断」をモデルに組み込む。
#   OOS期間（2025-08-25〜2026-03-26）で検証必須。
# ===========================================

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regime_detector import MarketRegime
except ImportError:
    from enum import Enum
    class MarketRegime(Enum):
        HIGH_VOL_TREND = "HIGH_VOL_TREND"
        LOW_VOL_TREND  = "LOW_VOL_TREND"
        RANGE          = "RANGE"


# =============================================================
# 環境別パラメータ
# =============================================================

@dataclass
class StrategyParams:
    """1つの相場環境に対応するパラメータセット。"""
    dd_multiplier:          float         # DDフィルターの乗数
    max_penalty:            float         # 最大ペナルティ
    confidence_threshold:   float         # エントリー確信度の閾値
    take_profit_multiplier: float = 1.0   # 利確の調整倍率（1.0=変更なし）
    stop_loss_multiplier:   float = 1.0   # 損切りの調整倍率
    skip:                   bool  = False # 完全見送り


# 相場環境別パラメータ定義
# HIGH_VOL_TRENDは現在のTEST6パラメータを継承（未検証ラベルなし → 既検証）
REGIME_PARAMS: Dict[MarketRegime, Optional[StrategyParams]] = {
    MarketRegime.HIGH_VOL_TREND: StrategyParams(
        dd_multiplier        = 3.0,
        max_penalty          = 0.20,
        confidence_threshold = 0.60,
        take_profit_multiplier = 1.0,
        stop_loss_multiplier   = 1.0,
    ),
    MarketRegime.LOW_VOL_TREND: StrategyParams(
        dd_multiplier        = 3.0,
        max_penalty          = 0.20,
        confidence_threshold = 0.75,  # 未検証 → 高く設定してエントリーを絞る
        take_profit_multiplier = 0.7, # 未検証 → 利確を早める（スプレッドコスト対策）
        stop_loss_multiplier   = 1.0,
    ),
    MarketRegime.RANGE: StrategyParams(
        dd_multiplier        = 0.0,
        max_penalty          = 0.0,
        confidence_threshold = 1.0,   # 実質的に全てのシグナルを弾く
        skip                 = True,  # 完全見送り
    ),
}


# =============================================================
# アダプティブ戦略クラス
# =============================================================

class AdaptiveStrategy:
    """
    相場環境に応じてパラメータを切り替える戦略クラス。
    """

    def __init__(self):
        self._current_regime = MarketRegime.RANGE
        self._params         = REGIME_PARAMS[MarketRegime.RANGE]

    def update_regime(self, regime: MarketRegime) -> None:
        """
        現在の相場環境を更新してパラメータを切り替える。

        Parameters:
            regime: detect_market_regime() で判定した環境
        """
        self._current_regime = regime
        self._params = REGIME_PARAMS.get(regime)

        if regime == MarketRegime.RANGE:
            print(f"[AdaptiveStrategy] レジーム: RANGE → 全シグナル見送り")
        elif regime == MarketRegime.LOW_VOL_TREND:
            print(f"[AdaptiveStrategy] レジーム: LOW_VOL_TREND → 確信度閾値を厳しくする")
        else:
            print(f"[AdaptiveStrategy] レジーム: HIGH_VOL_TREND → 通常モード")

    def get_params(self, regime: MarketRegime = None) -> Optional[StrategyParams]:
        """
        指定した（または現在の）相場環境のパラメータを返す。

        Returns:
            StrategyParams or None（RANGEの場合）
        """
        if regime is not None:
            return REGIME_PARAMS.get(regime)
        return self._params

    def should_enter(
        self,
        regime:     MarketRegime,
        confidence: float,
    ) -> bool:
        """
        エントリーすべきか判定する。

        Parameters:
            regime    : 現在の相場環境
            confidence: モデルの確信度（0〜1）

        Returns:
            True: エントリーする
            False: スキップ
        """
        params = REGIME_PARAMS.get(regime)

        if params is None or params.skip:
            return False

        return confidence >= params.confidence_threshold

    def adjust_take_profit(
        self,
        base_tp:  float,
        regime:   MarketRegime,
    ) -> float:
        """
        相場環境に応じて利確水準を調整する。

        Parameters:
            base_tp: 基準の利確幅（pips等）
            regime : 現在の相場環境

        Returns:
            adjusted_tp: 調整後の利確幅
        """
        params = REGIME_PARAMS.get(regime)
        if params is None:
            return base_tp
        return base_tp * params.take_profit_multiplier

    def adjust_stop_loss(
        self,
        base_sl: float,
        regime:  MarketRegime,
    ) -> float:
        """
        相場環境に応じて損切り水準を調整する。

        Parameters:
            base_sl: 基準の損切り幅（pips等）
            regime : 現在の相場環境

        Returns:
            adjusted_sl: 調整後の損切り幅
        """
        params = REGIME_PARAMS.get(regime)
        if params is None:
            return base_sl
        return base_sl * params.stop_loss_multiplier


# =============================================================
# OOS再検証の方針（実装後に実行するメモ）
# =============================================================
#
# 上記3つの変更を実装後：
# 1. OOS期間（2025-08-25〜2026-03-26）で再検証
# 2. HIGH_VOL_TREND期間のみのPF・Sharpeを確認
# 3. LOW_VOL_TREND期間のPFが改善したか確認
#
# 判定基準：
# HIGH_VOL_TREND期間：PF ≥ 1.3、Sharpe ≥ 1.0
# LOW_VOL_TREND期間：PF ≥ 1.0（損失を出さなければOK）
#
# 注意：パラメータを再最適化しないこと
# =============================================================
