"""
リスク管理エンジン

FX/株式トレードシステムのリスク管理を一元的に行うモジュール。
「くれぐれも大損のないようにリスク管理を最も大切にする」方針。

主なルール:
  - 2%ルール: 1トレードあたりの最大リスクは口座残高の2%
  - 連敗保護: 連敗数に応じてポジションサイズを自動縮小
  - 日次/週次損失リミット: 日次5%・週次10%超過で取引停止
  - 相関ガード: 高相関ペアの同時ポジションをブロック
  - 最大同時ポジション数: デフォルト3
  - ドローダウンサーキットブレーカー: 15%超で全取引停止
  - 取引ジャーナル: 全リスク判断をCSVに記録
"""

import csv
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── プロジェクトパス ───
DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "data").resolve()
JOURNAL_DIR = DATA_DIR / "paper_trade_logs"
JOURNAL_PATH = JOURNAL_DIR / "risk_journal.csv"

# ─── 通貨ペア相関マップ（絶対値） ───
# ペア名は "AAA/BBB" 形式で統一。キーはタプル (pair_a, pair_b) をソートして格納。
_RAW_CORRELATIONS = {
    ("EURJPY", "GBPJPY"): 0.90,
    ("EURUSD", "GBPUSD"): 0.85,
    ("USDJPY", "EURJPY"): 0.85,
    ("USDJPY", "GBPJPY"): 0.80,
    ("EURUSD", "EURJPY"): 0.60,
    ("GBPUSD", "GBPJPY"): 0.65,
    ("AUDJPY", "EURJPY"): 0.75,
    ("AUDJPY", "GBPJPY"): 0.70,
    ("AUDJPY", "USDJPY"): 0.70,
    ("EURUSD", "USDJPY"): -0.40,  # 逆相関
    ("GBPUSD", "USDJPY"): -0.35,
    ("AUDUSD", "EURUSD"): 0.60,
    ("AUDUSD", "GBPUSD"): 0.55,
    ("NZDUSD", "AUDUSD"): 0.90,
    ("NZDJPY", "AUDJPY"): 0.88,
}

# 正規化: 両方向からルックアップできるよう辞書を構築
CORRELATION_MAP: Dict[Tuple[str, str], float] = {}
for (a, b), corr in _RAW_CORRELATIONS.items():
    CORRELATION_MAP[(a, b)] = corr
    CORRELATION_MAP[(b, a)] = corr

# 相関ブロック閾値（絶対値がこれ以上なら同時ポジション不可）
CORRELATION_THRESHOLD = 0.75

# ─── 連敗時のポジションサイズ縮小ルール ───
# (連敗数, サイズ倍率)  ※リスト末尾の条件が優先
LOSING_STREAK_RULES = [
    (3, 0.50),   # 3連敗: サイズ50%に縮小
    (5, 0.25),   # 5連敗: サイズ25%に縮小
    (7, 0.00),   # 7連敗: 取引停止
]

# ─── ジャーナルCSVヘッダー ───
JOURNAL_HEADERS = [
    "timestamp", "event", "pair", "direction", "entry_price",
    "exit_price", "pnl", "position_size", "account_balance",
    "losing_streak", "daily_loss_pct", "weekly_loss_pct",
    "drawdown_pct", "open_positions", "reason",
]


def _normalize_pair(pair: str) -> str:
    """通貨ペア名を正規化（スラッシュ・スペース除去、大文字化）"""
    return pair.upper().replace("/", "").replace(" ", "").replace("=X", "")


class RiskManager:
    """
    リスク管理エンジン

    全てのトレード判断の前に can_trade() を呼び、
    約定後に record_trade() で結果を記録する。
    """

    def __init__(
        self,
        account_balance: float,
        risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
        max_weekly_loss: float = 0.10,
        max_drawdown: float = 0.15,
        max_positions: int = 3,
    ):
        # ─── パラメータバリデーション ───
        if account_balance <= 0:
            raise ValueError(f"口座残高は正の値が必要です: {account_balance}")
        if not (0 < risk_per_trade <= 0.05):
            raise ValueError(f"risk_per_trade は 0〜5% の範囲で指定: {risk_per_trade}")
        if not (0 < max_daily_loss <= 0.20):
            raise ValueError(f"max_daily_loss は 0〜20% の範囲で指定: {max_daily_loss}")
        if not (0 < max_weekly_loss <= 0.30):
            raise ValueError(f"max_weekly_loss は 0〜30% の範囲で指定: {max_weekly_loss}")
        if not (0 < max_drawdown <= 0.50):
            raise ValueError(f"max_drawdown は 0〜50% の範囲で指定: {max_drawdown}")
        if max_positions < 1:
            raise ValueError(f"max_positions は1以上が必要です: {max_positions}")

        # ─── 設定値 ───
        self.initial_balance: float = account_balance
        self.account_balance: float = account_balance
        self.risk_per_trade: float = risk_per_trade
        self.max_daily_loss: float = max_daily_loss
        self.max_weekly_loss: float = max_weekly_loss
        self.max_drawdown: float = max_drawdown
        self.max_positions: int = max_positions

        # ─── 状態管理 ───
        self.peak_balance: float = account_balance  # 最高残高（ドローダウン計算用）
        self.daily_pnl: float = 0.0                 # 当日の累積損益
        self.weekly_pnl: float = 0.0                # 今週の累積損益
        self.losing_streak: int = 0                  # 現在の連敗数
        self.open_positions: Dict[str, dict] = {}    # {pair: {direction, entry, size, ...}}
        self.trade_history: List[dict] = []          # 取引履歴
        self.is_halted: bool = False                 # 緊急停止フラグ

        # ─── 日付管理 ───
        self.current_date: date = date.today()
        self.week_start_date: date = date.today() - timedelta(
            days=date.today().weekday()
        )  # 今週の月曜日

        # ─── ジャーナル初期化 ───
        self._init_journal()

        logger.info(
            "RiskManager初期化: 残高=%.2f, リスク/トレード=%.1f%%, "
            "日次上限=%.1f%%, 週次上限=%.1f%%, DD上限=%.1f%%, 最大ポジション=%d",
            account_balance, risk_per_trade * 100, max_daily_loss * 100,
            max_weekly_loss * 100, max_drawdown * 100, max_positions,
        )

    # ═══════════════════════════════════════════════════════════════
    # ポジションサイズ計算
    # ═══════════════════════════════════════════════════════════════

    def calculate_position_size(
        self,
        pair: str,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """
        ポジションサイズ（ロット数）を計算する。

        Args:
            pair: 通貨ペア名（例: "USDJPY", "EUR/USD"）
            entry_price: エントリー価格
            stop_loss_price: ストップロス価格

        Returns:
            lot_size: ロット数（0の場合は取引不可）

        ロット計算ロジック:
            risk_amount = account_balance * risk_per_trade
            pip_risk = |entry_price - stop_loss_price|
            lot_size = risk_amount / pip_risk
            ※連敗による縮小を適用
        """
        pair = _normalize_pair(pair)

        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning("無効な価格: entry=%.5f, sl=%.5f", entry_price, stop_loss_price)
            return 0.0

        pip_risk = abs(entry_price - stop_loss_price)
        if pip_risk == 0:
            logger.warning("エントリー価格とストップロスが同じです: %.5f", entry_price)
            return 0.0

        # 基本リスク金額
        risk_amount = self.account_balance * self.risk_per_trade

        # 基本ロットサイズ
        lot_size = risk_amount / pip_risk

        # 連敗による縮小
        streak_multiplier = self._get_streak_multiplier()
        lot_size *= streak_multiplier

        # 最小ロット（0.01）未満は切り捨て
        lot_size = max(0.0, round(lot_size, 2))

        logger.info(
            "ポジションサイズ計算: %s リスク=%.2f, pip_risk=%.5f, "
            "ロット=%.2f (連敗倍率=%.2f)",
            pair, risk_amount, pip_risk, lot_size, streak_multiplier,
        )
        return lot_size

    # ═══════════════════════════════════════════════════════════════
    # 取引可否判定
    # ═══════════════════════════════════════════════════════════════

    def can_trade(
        self,
        pair: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        現在の状態で新規トレードが可能かを判定する。

        Args:
            pair: 取引予定の通貨ペア（相関チェック用、省略可）
            direction: "long" or "short"（相関チェック用、省略可）

        Returns:
            (can_trade, reason): 取引可否とその理由
        """
        # 日付が変わっていれば自動リセット
        self._check_date_rollover()

        # 1. 緊急停止チェック
        if self.is_halted:
            reason = "緊急停止中: 手動で解除されるまで取引不可"
            self._log_event("BLOCKED", reason=reason)
            return False, reason

        # 2. ドローダウン・サーキットブレーカー
        dd = self._current_drawdown()
        if dd >= self.max_drawdown:
            reason = (
                f"ドローダウン上限超過: {dd:.1%} >= {self.max_drawdown:.1%} "
                f"(最高残高: {self.peak_balance:.2f}, 現残高: {self.account_balance:.2f})"
            )
            self._log_event("BLOCKED", reason=reason)
            return False, reason

        # 3. 日次損失リミット
        daily_loss_pct = abs(min(0, self.daily_pnl)) / self.initial_balance
        if daily_loss_pct >= self.max_daily_loss:
            reason = (
                f"日次損失上限超過: {daily_loss_pct:.1%} >= {self.max_daily_loss:.1%} "
                f"(日次損益: {self.daily_pnl:.2f})"
            )
            self._log_event("BLOCKED", reason=reason)
            return False, reason

        # 4. 週次損失リミット
        weekly_loss_pct = abs(min(0, self.weekly_pnl)) / self.initial_balance
        if weekly_loss_pct >= self.max_weekly_loss:
            reason = (
                f"週次損失上限超過: {weekly_loss_pct:.1%} >= {self.max_weekly_loss:.1%} "
                f"(週次損益: {self.weekly_pnl:.2f})"
            )
            self._log_event("BLOCKED", reason=reason)
            return False, reason

        # 5. 連敗停止チェック
        if self._get_streak_multiplier() == 0.0:
            reason = f"連敗停止: {self.losing_streak}連敗により取引停止中"
            self._log_event("BLOCKED", reason=reason)
            return False, reason

        # 6. 最大同時ポジション数
        if len(self.open_positions) >= self.max_positions:
            reason = (
                f"最大ポジション数到達: {len(self.open_positions)}/{self.max_positions} "
                f"(保有中: {list(self.open_positions.keys())})"
            )
            self._log_event("BLOCKED", reason=reason)
            return False, reason

        # 7. 相関ガード（ペアと方向が指定された場合のみ）
        if pair and direction:
            pair = _normalize_pair(pair)
            corr_block = self._check_correlation(pair, direction)
            if corr_block:
                self._log_event("BLOCKED", pair=pair, reason=corr_block)
                return False, corr_block

        reason = "取引可能: 全リスクチェック通過"
        return True, reason

    # ═══════════════════════════════════════════════════════════════
    # 取引記録
    # ═══════════════════════════════════════════════════════════════

    def record_trade(
        self,
        pair: str,
        direction: str,
        entry: float,
        exit_price: float,
        pnl: float,
        position_size: float = 0.0,
    ) -> None:
        """
        完了した取引を記録し、内部状態を更新する。

        Args:
            pair: 通貨ペア
            direction: "long" or "short"
            entry: エントリー価格
            exit_price: 決済価格
            pnl: 損益（金額）
            position_size: ポジションサイズ
        """
        pair = _normalize_pair(pair)

        # 残高更新
        self.account_balance += pnl
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        # 最高残高更新
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

        # 連敗カウント更新
        if pnl < 0:
            self.losing_streak += 1
            logger.warning(
                "損失トレード: %s %s PnL=%.2f (連敗: %d)",
                pair, direction, pnl, self.losing_streak,
            )
        elif pnl > 0:
            self.losing_streak = 0
            logger.info(
                "利益トレード: %s %s PnL=%.2f (連敗リセット)",
                pair, direction, pnl,
            )
        # pnl == 0 の場合は連敗カウントを変更しない

        # ポジション管理から除去
        if pair in self.open_positions:
            del self.open_positions[pair]

        # 履歴に追加
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "pnl": pnl,
            "position_size": position_size,
            "account_balance": self.account_balance,
            "losing_streak": self.losing_streak,
        }
        self.trade_history.append(trade_record)

        # ジャーナルに書き込み
        self._log_event(
            "TRADE_CLOSED",
            pair=pair,
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            pnl=pnl,
            position_size=position_size,
            reason=f"PnL={pnl:+.2f}, 連敗={self.losing_streak}",
        )

    def open_position(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        position_size: float,
    ) -> bool:
        """
        新規ポジションをオープンとして登録する。

        Returns:
            True: 登録成功, False: 取引不可（can_tradeを内部で再チェック）
        """
        pair = _normalize_pair(pair)
        direction = direction.lower()

        can, reason = self.can_trade(pair, direction)
        if not can:
            logger.warning("ポジション登録拒否: %s — %s", pair, reason)
            return False

        self.open_positions[pair] = {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "position_size": position_size,
            "opened_at": datetime.now().isoformat(),
        }

        self._log_event(
            "POSITION_OPENED",
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            reason=f"SL={stop_loss_price:.5f}",
        )
        logger.info(
            "ポジション登録: %s %s @ %.5f (SL=%.5f, size=%.2f)",
            pair, direction, entry_price, stop_loss_price, position_size,
        )
        return True

    # ═══════════════════════════════════════════════════════════════
    # 状態照会
    # ═══════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """現在のリスク状態サマリーを返す。"""
        self._check_date_rollover()
        dd = self._current_drawdown()
        daily_loss_pct = abs(min(0, self.daily_pnl)) / self.initial_balance
        weekly_loss_pct = abs(min(0, self.weekly_pnl)) / self.initial_balance
        streak_mult = self._get_streak_multiplier()

        return {
            "account_balance": round(self.account_balance, 2),
            "initial_balance": round(self.initial_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown_pct": round(dd, 4),
            "drawdown_limit": self.max_drawdown,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_loss_pct": round(daily_loss_pct, 4),
            "daily_loss_limit": self.max_daily_loss,
            "weekly_pnl": round(self.weekly_pnl, 2),
            "weekly_loss_pct": round(weekly_loss_pct, 4),
            "weekly_loss_limit": self.max_weekly_loss,
            "losing_streak": self.losing_streak,
            "streak_multiplier": streak_mult,
            "open_positions": len(self.open_positions),
            "max_positions": self.max_positions,
            "open_pairs": list(self.open_positions.keys()),
            "is_halted": self.is_halted,
            "total_trades": len(self.trade_history),
            "can_trade": self.can_trade()[0],
        }

    def get_losing_streak(self) -> int:
        """現在の連敗数を返す。"""
        return self.losing_streak

    # ═══════════════════════════════════════════════════════════════
    # 日次/週次リセット
    # ═══════════════════════════════════════════════════════════════

    def reset_daily(self) -> None:
        """日次損益をリセットする（スケジューラから呼び出し）。"""
        logger.info("日次リセット: 日次損益 %.2f → 0", self.daily_pnl)
        self._log_event("DAILY_RESET", reason=f"日次損益={self.daily_pnl:+.2f}")
        self.daily_pnl = 0.0
        self.current_date = date.today()

    def reset_weekly(self) -> None:
        """週次損益をリセットする（スケジューラから呼び出し）。"""
        logger.info("週次リセット: 週次損益 %.2f → 0", self.weekly_pnl)
        self._log_event("WEEKLY_RESET", reason=f"週次損益={self.weekly_pnl:+.2f}")
        self.weekly_pnl = 0.0
        self.week_start_date = date.today()

    # ═══════════════════════════════════════════════════════════════
    # 緊急停止 / 解除
    # ═══════════════════════════════════════════════════════════════

    def halt(self, reason: str = "手動停止") -> None:
        """全取引を緊急停止する。"""
        self.is_halted = True
        logger.critical("取引停止: %s", reason)
        self._log_event("HALT", reason=reason)

    def resume(self, reason: str = "手動再開") -> None:
        """緊急停止を解除する。"""
        self.is_halted = False
        logger.info("取引再開: %s", reason)
        self._log_event("RESUME", reason=reason)

    # ═══════════════════════════════════════════════════════════════
    # 残高の外部更新
    # ═══════════════════════════════════════════════════════════════

    def update_balance(self, new_balance: float) -> None:
        """口座残高を外部から更新する（入出金対応）。"""
        if new_balance <= 0:
            raise ValueError(f"口座残高は正の値が必要です: {new_balance}")
        old = self.account_balance
        self.account_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        logger.info("残高更新: %.2f → %.2f", old, new_balance)
        self._log_event("BALANCE_UPDATE", reason=f"{old:.2f} → {new_balance:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 内部メソッド
    # ═══════════════════════════════════════════════════════════════

    def _current_drawdown(self) -> float:
        """現在のドローダウン率を計算する。"""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.account_balance) / self.peak_balance

    def _get_streak_multiplier(self) -> float:
        """連敗数に応じたポジションサイズ倍率を返す。"""
        multiplier = 1.0
        for threshold, mult in LOSING_STREAK_RULES:
            if self.losing_streak >= threshold:
                multiplier = mult
        return multiplier

    def _check_correlation(self, new_pair: str, new_direction: str) -> Optional[str]:
        """
        既存ポジションとの相関をチェックする。

        高相関ペアが同方向で保有されている場合、ブロック理由文字列を返す。
        問題なければ None を返す。
        """
        new_direction = new_direction.lower()

        for held_pair, held_info in self.open_positions.items():
            key = (new_pair, held_pair)
            corr = CORRELATION_MAP.get(key)

            if corr is None:
                continue

            abs_corr = abs(corr)
            if abs_corr < CORRELATION_THRESHOLD:
                continue

            held_direction = held_info["direction"].lower()

            # 正の相関 + 同方向 → リスク集中
            # 正の相関 + 逆方向 → ヘッジ（許可）
            # 負の相関 + 同方向 → ヘッジ（許可）
            # 負の相関 + 逆方向 → リスク集中
            if corr > 0:
                is_risk_concentrated = (new_direction == held_direction)
            else:
                is_risk_concentrated = (new_direction != held_direction)

            if is_risk_concentrated:
                return (
                    f"相関ガード: {new_pair}({new_direction}) と "
                    f"{held_pair}({held_direction}) は相関 {corr:.2f} で"
                    f"リスク集中のため同時保有不可 (閾値: {CORRELATION_THRESHOLD})"
                )

        return None

    def _check_date_rollover(self) -> None:
        """日付が変わっていたら自動的にリセットする。"""
        today = date.today()

        if today != self.current_date:
            self.reset_daily()

        # 週が変わったかチェック（月曜始まり）
        this_monday = today - timedelta(days=today.weekday())
        if this_monday != self.week_start_date:
            self.reset_weekly()

    # ─── ジャーナル（CSVログ） ───

    def _init_journal(self) -> None:
        """ジャーナルCSVファイルを初期化する（ファイルが無ければ作成）。"""
        try:
            JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
            if not JOURNAL_PATH.exists():
                with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(JOURNAL_HEADERS)
                logger.info("ジャーナル作成: %s", JOURNAL_PATH)
        except OSError as e:
            logger.error("ジャーナル初期化失敗: %s", e)

    def _log_event(
        self,
        event: str,
        pair: str = "",
        direction: str = "",
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        pnl: float = 0.0,
        position_size: float = 0.0,
        reason: str = "",
    ) -> None:
        """リスク判断をジャーナルCSVに記録する。"""
        dd = self._current_drawdown()
        daily_loss_pct = abs(min(0, self.daily_pnl)) / max(self.initial_balance, 1)
        weekly_loss_pct = abs(min(0, self.weekly_pnl)) / max(self.initial_balance, 1)

        row = [
            datetime.now().isoformat(),
            event,
            pair,
            direction,
            f"{entry_price:.5f}" if entry_price else "",
            f"{exit_price:.5f}" if exit_price else "",
            f"{pnl:.2f}" if pnl else "",
            f"{position_size:.2f}" if position_size else "",
            f"{self.account_balance:.2f}",
            self.losing_streak,
            f"{daily_loss_pct:.4f}",
            f"{weekly_loss_pct:.4f}",
            f"{dd:.4f}",
            len(self.open_positions),
            reason,
        ]

        try:
            with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except OSError as e:
            logger.error("ジャーナル書き込み失敗: %s", e)

    # ═══════════════════════════════════════════════════════════════
    # シリアライズ / 永続化
    # ═══════════════════════════════════════════════════════════════

    def to_dict(self) -> dict:
        """状態を辞書に変換する（JSON保存用）。"""
        return {
            "initial_balance": self.initial_balance,
            "account_balance": self.account_balance,
            "peak_balance": self.peak_balance,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "losing_streak": self.losing_streak,
            "open_positions": self.open_positions,
            "is_halted": self.is_halted,
            "current_date": self.current_date.isoformat(),
            "week_start_date": self.week_start_date.isoformat(),
            "risk_per_trade": self.risk_per_trade,
            "max_daily_loss": self.max_daily_loss,
            "max_weekly_loss": self.max_weekly_loss,
            "max_drawdown": self.max_drawdown,
            "max_positions": self.max_positions,
        }

    def save_state(self, path: Optional[Path] = None) -> None:
        """状態をJSONファイルに保存する。"""
        path = path or (JOURNAL_DIR / "risk_state.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("リスク状態保存: %s", path)
        except OSError as e:
            logger.error("リスク状態保存失敗: %s", e)

    @classmethod
    def load_state(cls, path: Optional[Path] = None) -> "RiskManager":
        """JSONファイルから状態を復元する。"""
        path = path or (JOURNAL_DIR / "risk_state.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rm = cls(
            account_balance=data["account_balance"],
            risk_per_trade=data.get("risk_per_trade", 0.02),
            max_daily_loss=data.get("max_daily_loss", 0.05),
            max_weekly_loss=data.get("max_weekly_loss", 0.10),
            max_drawdown=data.get("max_drawdown", 0.15),
            max_positions=data.get("max_positions", 3),
        )
        rm.initial_balance = data["initial_balance"]
        rm.peak_balance = data["peak_balance"]
        rm.daily_pnl = data.get("daily_pnl", 0.0)
        rm.weekly_pnl = data.get("weekly_pnl", 0.0)
        rm.losing_streak = data.get("losing_streak", 0)
        rm.open_positions = data.get("open_positions", {})
        rm.is_halted = data.get("is_halted", False)

        if "current_date" in data:
            rm.current_date = date.fromisoformat(data["current_date"])
        if "week_start_date" in data:
            rm.week_start_date = date.fromisoformat(data["week_start_date"])

        logger.info("リスク状態復元: %s", path)
        return rm

    def __repr__(self) -> str:
        dd = self._current_drawdown()
        return (
            f"RiskManager(balance={self.account_balance:.2f}, "
            f"dd={dd:.1%}, streak={self.losing_streak}, "
            f"positions={len(self.open_positions)}/{self.max_positions})"
        )
