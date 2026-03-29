# ===========================================
# stat_arb_model.py
# 暗号通貨プロジェクト - 統計的アービトラージ
#
# 設計方針：
#   - 対象: 中小型アルトコインペア（時価総額100〜1000億円）
#   - BTC・ETHの短期は機関投資家の独壇場 → 対象外
#   - Zスコア ≥ +2.0 / ≤ -2.0 でシグナル発生
#   - Zスコア ≥ +4.0 で強制損切り
#   - 損切りを必ず実装（「相関は必ず戻る」は思い込み）
#
# 注意：コードのみ完成。
#       運用はボートレース・FXが安定後。
# ===========================================

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================
# 定数・シグナル定義
# =============================================================

class SignalType(Enum):
    """トレードシグナルの種類。"""
    LONG_A_SHORT_B = "long_A_short_B"   # Aを買い・Bを空売り
    SHORT_A_LONG_B = "short_A_long_B"   # Aを売り・Bを買い
    EXIT           = "exit"             # 決済
    FORCE_EXIT     = "force_exit"       # 強制決済（Zスコア超過）
    NO_SIGNAL      = "no_signal"


@dataclass
class ArbSignal:
    """統計的アービトラージのトレードシグナル。"""
    pair:      Tuple[str, str]    # (coin_A, coin_B)
    signal:    SignalType
    zscore:    float
    timestamp: str
    reason:    str


# =============================================================
# STEP1：ペア選定
# =============================================================

class PairSelector:
    """相関が高いペアを自動で見つける。"""

    def calc_correlation(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        window:  int = 90,
    ) -> float:
        """
        90日間のローリング相関係数を計算する。

        Parameters:
            price_a : コインAの価格Series
            price_b : コインBの価格Series
            window  : ウィンドウ期間（日数）

        Returns:
            corr: 相関係数（-1〜+1）
        """
        log_a = np.log(price_a.replace(0, np.nan))
        log_b = np.log(price_b.replace(0, np.nan))

        corr = log_a.rolling(window).corr(log_b)
        return float(corr.iloc[-1]) if not corr.empty else 0.0

    def find_cointegrated_pairs(
        self,
        prices_dict: Dict[str, pd.Series],
        p_value_threshold: float = 0.05,
    ) -> List[Tuple[str, str, float]]:
        """
        共和分検定で長期的に相関が安定しているペアを選ぶ。

        対象：中小型アルトコイン
        （BTC・ETHは機関投資家の独壇場なので除外）

        Parameters:
            prices_dict       : {"BTC": Series, "ETH": Series, ...}
            p_value_threshold : 共和分検定のp値閾値

        Returns:
            pairs: [(coin_A, coin_B, p_value), ...]（相関が強い順）
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            raise ImportError("statsmodelsが必要です: pip install statsmodels")

        tickers = list(prices_dict.keys())
        cointegrated = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                coin_a = tickers[i]
                coin_b = tickers[j]

                # BTC・ETHは除外
                if coin_a in ("BTC", "ETH", "BTCUSDT", "ETHUSDT"):
                    continue
                if coin_b in ("BTC", "ETH", "BTCUSDT", "ETHUSDT"):
                    continue

                try:
                    series_a = prices_dict[coin_a].dropna()
                    series_b = prices_dict[coin_b].dropna()

                    # 同じ期間のみ使う
                    common_idx = series_a.index.intersection(series_b.index)
                    if len(common_idx) < 60:
                        continue

                    _, p_value, _ = coint(
                        series_a.loc[common_idx],
                        series_b.loc[common_idx],
                    )

                    if p_value < p_value_threshold:
                        cointegrated.append((coin_a, coin_b, round(p_value, 4)))

                except Exception:
                    continue

        # p値が小さい順（相関が強い順）でソート
        cointegrated.sort(key=lambda x: x[2])
        return cointegrated


# =============================================================
# STEP2：スプレッド計算とZスコア
# =============================================================

class SpreadCalculator:
    """2つのコインの価格比率からZスコアを計算する。"""

    def calc_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> pd.Series:
        """
        対数スプレッドを計算する。

        比率 = price_a / price_b
        スプレッド = log(比率) = log(A) - log(B)

        Parameters:
            price_a : コインAの価格Series
            price_b : コインBの価格Series

        Returns:
            spread: 対数スプレッドのSeries
        """
        return np.log(price_a.replace(0, np.nan)) - np.log(price_b.replace(0, np.nan))

    def calc_zscore(
        self,
        spread: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """
        ローリングZスコアを計算する。

        Zスコア = (現在値 - 移動平均) / 移動標準偏差

        読み方：
        Zスコア ≥ +2.0 → Aが割高・Bが割安 → Aを売り・Bを買い
        Zスコア ≤ -2.0 → Aが割安・Bが割高 → Aを買い・Bを売り
        Zスコア ≥ +4.0 → 強制決済（相関が戻らないリスク）

        Parameters:
            spread : 対数スプレッドのSeries
            window : ローリングウィンドウ（デフォルト60）

        Returns:
            zscore: ZスコアのSeries
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std  = spread.rolling(window=window).std(ddof=1)

        zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore


# =============================================================
# STEP3：シグナル生成
# =============================================================

class StatArbSignal:
    """Zスコアからトレードシグナルを生成する。"""

    ENTRY_THRESHOLD = 2.0   # エントリーZスコア
    EXIT_THRESHOLD  = 0.5   # 決済Zスコア（収束判定）
    STOP_THRESHOLD  = 4.0   # 強制損切りZスコア

    def generate_signal(
        self,
        zscore:           float,
        current_position: Optional[SignalType] = None,
        timestamp:        str = "",
        pair:             Tuple[str, str] = ("A", "B"),
    ) -> ArbSignal:
        """
        Zスコアからシグナルを生成する。

        Zスコア ≥ +2.0:
            Aを売り（空売り）・Bを買い
        Zスコア ≤ -2.0:
            Aを買い・Bを売り（空売り）
        |Zスコア| ≤ 0.5 かつポジションあり:
            決済（収束）
        |Zスコア| ≥ 4.0:
            強制損切り

        Parameters:
            zscore           : 現在のZスコア
            current_position : 現在保有しているポジション
            timestamp        : タイムスタンプ
            pair             : (コインA名, コインB名)

        Returns:
            ArbSignal
        """
        abs_z = abs(zscore)

        # 強制損切り
        if abs_z >= self.STOP_THRESHOLD:
            return ArbSignal(
                pair=pair, signal=SignalType.FORCE_EXIT, zscore=zscore,
                timestamp=timestamp,
                reason=f"Zスコア{zscore:.2f}（{self.STOP_THRESHOLD}超）→ 強制損切り",
            )

        # 決済（ポジションがある場合）
        if current_position in (SignalType.LONG_A_SHORT_B, SignalType.SHORT_A_LONG_B):
            if abs_z <= self.EXIT_THRESHOLD:
                return ArbSignal(
                    pair=pair, signal=SignalType.EXIT, zscore=zscore,
                    timestamp=timestamp,
                    reason=f"Zスコア{zscore:.2f}（収束） → 決済",
                )

        # 新規エントリー（ポジションなしの場合）
        if current_position is None or current_position == SignalType.NO_SIGNAL:
            if zscore >= self.ENTRY_THRESHOLD:
                return ArbSignal(
                    pair=pair, signal=SignalType.SHORT_A_LONG_B, zscore=zscore,
                    timestamp=timestamp,
                    reason=f"Zスコア{zscore:.2f}（+{self.ENTRY_THRESHOLD}超）→ A売り・B買い",
                )
            elif zscore <= -self.ENTRY_THRESHOLD:
                return ArbSignal(
                    pair=pair, signal=SignalType.LONG_A_SHORT_B, zscore=zscore,
                    timestamp=timestamp,
                    reason=f"Zスコア{zscore:.2f}（-{self.ENTRY_THRESHOLD}以下）→ A買い・B売り",
                )

        return ArbSignal(
            pair=pair, signal=SignalType.NO_SIGNAL, zscore=zscore,
            timestamp=timestamp, reason="シグナルなし",
        )


# =============================================================
# STEP4：LightGBMで精度向上
# =============================================================

class StatArbPredictor:
    """Zスコアだけでなく追加特徴量も使って予測精度を向上させる。"""

    def __init__(self):
        self.model = None

    def create_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        追加特徴量を作成する。

        基本Zスコア以外に：
        ・ボラティリティ（荒れているとき戻りにくい）
        ・出来高の比率
        ・市場全体のレジーム（強気/弱気）
        ・Fear & Greed Index
        ・Funding Rate（先物市場の需給）
        ・ニュース感情スコア（株モデルと共通化）

        Parameters:
            price_data: OHLCV DataFrame

        Returns:
            features_df
        """
        df = price_data.copy()

        # 基本テクニカル特徴量
        returns = df["close"].pct_change()
        df["volatility_5d"]  = returns.rolling(5).std()
        df["volatility_20d"] = returns.rolling(20).std()
        df["volume_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()

        # 価格モメンタム
        df["return_1d"]  = returns
        df["return_5d"]  = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)

        # RSI（相対力指数）
        df["rsi"] = self._calc_rsi(df["close"], window=14)

        # Fear & Greed Index / Funding Rate は外部データが必要
        # 学習時に追加する（現時点ではNaN）
        df["fear_greed_index"] = np.nan  # 未実装: 外部APIから取得
        df["funding_rate"]     = np.nan  # 未実装: 取引所APIから取得

        return df

    def _calc_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSIを計算する。"""
        delta  = prices.diff()
        gain   = delta.where(delta > 0, 0)
        loss   = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi


# =============================================================
# STEP5：リスク管理
# =============================================================

class StatArbRiskManager:
    """統計的アービトラージのリスク管理。"""

    def calc_position_size(
        self,
        zscore:     float,
        capital:    float,
        volatility: float,
        max_ratio:  float = 0.10,
    ) -> float:
        """
        Kelly基準でポジションサイズを決定する。
        ボートレース・FXと同じ設計。

        Zスコアが大きいほど（割高割安が明確なほど）
        ポジションを大きくする。

        Parameters:
            zscore     : 現在のZスコア
            capital    : 現在の資金
            volatility : スプレッドのボラティリティ
            max_ratio  : 最大ポジション比率

        Returns:
            position_size: ポジションサイズ（資金に対する割合）
        """
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
            from kelly import kelly_universal
        except ImportError:
            def kelly_universal(prob, odds, capital, fraction=0.25, max_ratio=0.05, unit=1.0):
                return capital * fraction * max_ratio

        # Zスコアから勝率・オッズを推定（未検証 → データで調整が必要）
        abs_z      = min(abs(zscore), 4.0)
        est_prob   = 0.5 + (abs_z - 2.0) * 0.05  # Zスコア2→50%, 4→60% (未検証)
        est_odds   = 1.0 / max(volatility, 0.01)  # 暫定（未検証）

        est_prob = max(0.3, min(0.8, est_prob))
        est_odds = max(1.5, min(10.0, est_odds))

        size = kelly_universal(
            prob=est_prob,
            odds_or_ratio=est_odds,
            capital=capital,
            fraction=0.25,
            max_ratio=max_ratio,
        )

        return float(size)

    def check_correlation_stability(
        self,
        recent_corr:     float,
        historical_corr: float,
        threshold:       float = 0.3,
    ) -> dict:
        """
        ペアの相関が崩れていないか確認する。
        崩れていたら強制決済。

        Parameters:
            recent_corr     : 直近の相関係数
            historical_corr : 過去の平均相関係数
            threshold       : 乖離の許容閾値

        Returns:
            {"stable": bool, "reason": str}
        """
        diff = abs(recent_corr - historical_corr)

        if diff >= threshold:
            return {
                "stable": False,
                "reason": (
                    f"相関が崩れています（直近={recent_corr:.3f} vs "
                    f"過去={historical_corr:.3f}、乖離={diff:.3f}）→ 強制決済推奨"
                ),
            }

        return {
            "stable": True,
            "reason": f"相関は安定（乖離={diff:.3f} < {threshold}）",
        }


# =============================================================
# 統合パイプライン
# =============================================================

class StatArbPipeline:
    """統計的アービトラージの全処理を統合するパイプライン。"""

    def __init__(self):
        self.pair_selector   = PairSelector()
        self.spread_calc     = SpreadCalculator()
        self.signal_gen      = StatArbSignal()
        self.predictor       = StatArbPredictor()
        self.risk_manager    = StatArbRiskManager()
        self.active_pairs    = {}   # {(A, B): current_position}

    def run(
        self,
        prices: Dict[str, pd.Series],
        capital: float,
        funding_rates: Dict[str, float] = None,
        timestamp: str = "",
    ) -> List[ArbSignal]:
        """
        全ペアに対してシグナルを計算して返す。

        Parameters:
            prices        : {銘柄名: 価格Series}
            capital       : 現在の資金
            funding_rates : {銘柄名: Funding Rate}
            timestamp     : 現在時刻

        Returns:
            signals: シグナルのリスト
        """
        signals = []

        for (coin_a, coin_b), position in list(self.active_pairs.items()):
            if coin_a not in prices or coin_b not in prices:
                continue

            spread = self.spread_calc.calc_spread(prices[coin_a], prices[coin_b])
            zscore_series = self.spread_calc.calc_zscore(spread)
            if zscore_series.empty or zscore_series.isna().all():
                continue

            current_z = float(zscore_series.iloc[-1])

            signal = self.signal_gen.generate_signal(
                zscore=current_z,
                current_position=position,
                timestamp=timestamp,
                pair=(coin_a, coin_b),
            )

            if signal.signal != SignalType.NO_SIGNAL:
                signals.append(signal)
                # ポジション状態を更新
                if signal.signal in (SignalType.EXIT, SignalType.FORCE_EXIT):
                    self.active_pairs[(coin_a, coin_b)] = None
                else:
                    self.active_pairs[(coin_a, coin_b)] = signal.signal

        return signals
