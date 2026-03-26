# ===========================================
# performance_report.py
# ペーパートレードのパフォーマンスレポート生成
#
# 日次・週次・月次・全期間のレポートを生成し、
# Telegramで送信する機能を提供する。
# ===========================================

import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# デフォルトのログディレクトリ
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "paper_trade_logs"

# Telegramの設定ファイル
CHAT_ID_FILE = PROJECT_ROOT / "data" / "telegram_chat_id.txt"


class PerformanceReporter:
    """ペーパートレードのパフォーマンスレポートを生成するクラス"""

    def __init__(self, log_dir: Optional[Path] = None):
        """
        初期化

        Args:
            log_dir: predictions.csv があるディレクトリのパス
        """
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.csv_path = self.log_dir / "predictions.csv"

    def _load_data(self) -> pd.DataFrame:
        """CSVからトレードログを読み込む"""
        if not self.csv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(self.csv_path)

        # logged_at を datetime に変換
        if "logged_at" in df.columns:
            df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")

        # timestamp を datetime に変換
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # 数値列の型変換
        for col in ["close", "confidence", "proba_up", "proba_down",
                     "exit_price", "actual_return", "net_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _filter_period(
        self, df: pd.DataFrame, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """指定期間のデータを抽出する"""
        if df.empty or "logged_at" not in df.columns:
            return df
        mask = (df["logged_at"] >= start) & (df["logged_at"] < end)
        return df[mask].copy()

    def _calc_metrics(self, df: pd.DataFrame) -> dict:
        """
        トレードデータから各種指標を計算する

        Returns:
            dict: 各種メトリクス
        """
        metrics = {}

        # 全レコード数
        metrics["total_records"] = len(df)

        # トレード（BUY/SELL）とスキップを分離
        trades = df[df["action"].isin(["BUY", "SELL"])].copy()
        skips = df[df["action"] == "SKIP"].copy()

        metrics["num_trades"] = len(trades)
        metrics["num_buys"] = len(trades[trades["action"] == "BUY"])
        metrics["num_sells"] = len(trades[trades["action"] == "SELL"])
        metrics["num_skips"] = len(skips)

        # スキップ理由の内訳
        if not skips.empty and "reason" in skips.columns:
            reason_counts = (
                skips["reason"]
                .fillna("不明")
                .value_counts()
                .to_dict()
            )
            metrics["skip_reasons"] = reason_counts
        else:
            metrics["skip_reasons"] = {}

        # 結果が記録されているトレードのみで勝率等を計算
        settled = trades.dropna(subset=["net_return"])

        if settled.empty:
            # 決済済みトレードなし
            metrics["win_rate"] = None
            metrics["profit_factor"] = None
            metrics["net_cumulative"] = 0.0
            metrics["max_drawdown"] = 0.0
            metrics["sharpe_ratio"] = None
            metrics["best_trade"] = None
            metrics["worst_trade"] = None
            metrics["current_streak"] = (0, "なし")
            return metrics

        # 勝敗
        wins = settled[settled["net_return"] > 0]
        losses = settled[settled["net_return"] < 0]

        metrics["win_rate"] = len(wins) / len(settled) if len(settled) > 0 else 0.0

        # プロフィットファクター（総利益 / 総損失の絶対値）
        gross_wins = wins["net_return"].sum() if not wins.empty else 0.0
        gross_losses = abs(losses["net_return"].sum()) if not losses.empty else 0.0
        if gross_losses > 0:
            metrics["profit_factor"] = gross_wins / gross_losses
        else:
            metrics["profit_factor"] = float("inf") if gross_wins > 0 else 0.0

        # 累積リターン
        metrics["net_cumulative"] = settled["net_return"].sum()

        # 最大ドローダウン
        cumsum = settled["net_return"].cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        metrics["max_drawdown"] = drawdown.min() if len(drawdown) > 0 else 0.0

        # シャープレシオ（年率換算：1時間足想定 × 24h × 252日）
        returns = settled["net_return"]
        if len(returns) >= 2 and returns.std() > 0:
            periods_per_year = 24 * 252  # 1時間足ベース
            metrics["sharpe_ratio"] = (
                returns.mean() / returns.std() * np.sqrt(periods_per_year)
            )
        else:
            metrics["sharpe_ratio"] = None

        # ベスト/ワーストトレード
        best_idx = settled["net_return"].idxmax()
        worst_idx = settled["net_return"].idxmin()
        best_row = settled.loc[best_idx]
        worst_row = settled.loc[worst_idx]

        metrics["best_trade"] = {
            "return": best_row["net_return"],
            "action": best_row["action"],
            "timestamp": best_row.get("timestamp", ""),
        }
        metrics["worst_trade"] = {
            "return": worst_row["net_return"],
            "action": worst_row["action"],
            "timestamp": worst_row.get("timestamp", ""),
        }

        # 現在の連勝/連敗ストリーク
        results_list = (settled["net_return"] > 0).tolist()
        if results_list:
            current = results_list[-1]
            streak = 0
            for r in reversed(results_list):
                if r == current:
                    streak += 1
                else:
                    break
            streak_type = "連勝" if current else "連敗"
            metrics["current_streak"] = (streak, streak_type)
        else:
            metrics["current_streak"] = (0, "なし")

        return metrics

    def _format_report(self, title: str, period_str: str, metrics: dict) -> str:
        """
        メトリクスを見やすいテキストレポートにフォーマットする

        Args:
            title: レポートタイトル
            period_str: 期間表示文字列
            metrics: _calc_metrics() の戻り値

        Returns:
            str: フォーマット済みレポート
        """
        lines = []
        lines.append(f"{'=' * 30}")
        lines.append(f"  {title}")
        lines.append(f"  {period_str}")
        lines.append(f"{'=' * 30}")

        # トレード件数
        lines.append(f"\n--- 取引概要 ---")
        lines.append(f"  総レコード数: {metrics['total_records']}")
        lines.append(
            f"  取引数: {metrics['num_trades']} "
            f"(買: {metrics['num_buys']} / 売: {metrics['num_sells']})"
        )
        lines.append(f"  スキップ: {metrics['num_skips']}")

        # スキップ理由内訳
        if metrics["skip_reasons"]:
            lines.append(f"  スキップ理由:")
            for reason, count in metrics["skip_reasons"].items():
                lines.append(f"    - {reason}: {count}件")

        # 成績指標
        lines.append(f"\n--- 成績指標 ---")

        if metrics["win_rate"] is not None:
            wr = metrics["win_rate"]
            wr_emoji = "+" if wr >= 0.5 else "-"
            lines.append(f"  [{wr_emoji}] 勝率: {wr:.1%}")
        else:
            lines.append(f"  [?] 勝率: データ不足")

        if metrics["profit_factor"] is not None:
            pf = metrics["profit_factor"]
            if pf == float("inf"):
                lines.append(f"  [+] PF: --- (損失なし)")
            else:
                pf_emoji = "+" if pf >= 1.0 else "-"
                lines.append(f"  [{pf_emoji}] PF: {pf:.2f}")
        else:
            lines.append(f"  [?] PF: データ不足")

        # 累積リターン
        nc = metrics["net_cumulative"]
        nc_emoji = "+" if nc >= 0 else "-"
        lines.append(f"  [{nc_emoji}] 累積リターン: {nc:+.4f}")

        # 最大ドローダウン
        mdd = metrics["max_drawdown"]
        lines.append(f"  [-] 最大DD: {mdd:.4f}")

        # シャープレシオ
        if metrics["sharpe_ratio"] is not None:
            sr = metrics["sharpe_ratio"]
            sr_emoji = "+" if sr >= 0 else "-"
            lines.append(f"  [{sr_emoji}] Sharpe: {sr:.2f}")
        else:
            lines.append(f"  [?] Sharpe: データ不足")

        # ベスト/ワーストトレード
        lines.append(f"\n--- 注目トレード ---")
        if metrics["best_trade"]:
            bt = metrics["best_trade"]
            lines.append(
                f"  [+] ベスト: {bt['action']} "
                f"{bt['return']:+.4f} ({bt['timestamp']})"
            )
        else:
            lines.append(f"  [?] ベスト: なし")

        if metrics["worst_trade"]:
            wt = metrics["worst_trade"]
            lines.append(
                f"  [-] ワースト: {wt['action']} "
                f"{wt['return']:+.4f} ({wt['timestamp']})"
            )
        else:
            lines.append(f"  [?] ワースト: なし")

        # 連勝/連敗ストリーク
        streak_count, streak_type = metrics["current_streak"]
        if streak_count > 0:
            s_emoji = "+" if streak_type == "連勝" else "-"
            lines.append(f"  [{s_emoji}] 現在: {streak_count}{streak_type}中")

        lines.append(f"\n{'=' * 30}")
        return "\n".join(lines)

    def daily_report(self, date: Optional[datetime] = None) -> str:
        """
        日次レポートを生成する

        Args:
            date: 対象日（Noneの場合は今日）

        Returns:
            str: レポートテキスト
        """
        df = self._load_data()
        if df.empty:
            return "データがありません。"

        if date is None:
            date = datetime.now()

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        filtered = self._filter_period(df, start, end)
        if filtered.empty:
            return f"{start.strftime('%Y-%m-%d')} のトレードデータはありません。"

        metrics = self._calc_metrics(filtered)
        period_str = f"{start.strftime('%Y-%m-%d')}"
        return self._format_report("日次レポート", period_str, metrics)

    def weekly_report(self, date: Optional[datetime] = None) -> str:
        """
        週次レポートを生成する（月曜始まり）

        Args:
            date: 対象週に含まれる任意の日（Noneの場合は今週）

        Returns:
            str: レポートテキスト
        """
        df = self._load_data()
        if df.empty:
            return "データがありません。"

        if date is None:
            date = datetime.now()

        # 月曜始まりの週初を計算
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start - timedelta(days=start.weekday())
        end = start + timedelta(days=7)

        filtered = self._filter_period(df, start, end)
        if filtered.empty:
            return (
                f"{start.strftime('%Y-%m-%d')} ~ "
                f"{(end - timedelta(days=1)).strftime('%Y-%m-%d')} "
                f"のトレードデータはありません。"
            )

        metrics = self._calc_metrics(filtered)
        period_str = (
            f"{start.strftime('%Y-%m-%d')} ~ "
            f"{(end - timedelta(days=1)).strftime('%Y-%m-%d')}"
        )
        return self._format_report("週次レポート", period_str, metrics)

    def monthly_report(self, date: Optional[datetime] = None) -> str:
        """
        月次レポートを生成する

        Args:
            date: 対象月に含まれる任意の日（Noneの場合は今月）

        Returns:
            str: レポートテキスト
        """
        df = self._load_data()
        if df.empty:
            return "データがありません。"

        if date is None:
            date = datetime.now()

        # 月初と翌月初を計算
        start = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)

        filtered = self._filter_period(df, start, end)
        if filtered.empty:
            return f"{start.strftime('%Y年%m月')} のトレードデータはありません。"

        metrics = self._calc_metrics(filtered)
        period_str = f"{start.strftime('%Y年%m月')}"
        return self._format_report("月次レポート", period_str, metrics)

    def overall_report(self) -> str:
        """
        全期間のレポートを生成する

        Returns:
            str: レポートテキスト
        """
        df = self._load_data()
        if df.empty:
            return "データがありません。"

        metrics = self._calc_metrics(df)

        # 期間の算出
        if "logged_at" in df.columns and not df["logged_at"].isna().all():
            first = df["logged_at"].min().strftime("%Y-%m-%d")
            last = df["logged_at"].max().strftime("%Y-%m-%d")
            period_str = f"{first} ~ {last}"
        else:
            period_str = "全期間"

        return self._format_report("全期間レポート", period_str, metrics)


# ===========================================
# Telegram送信
# ===========================================

def send_daily_report_telegram(date: Optional[datetime] = None):
    """
    日次レポートを生成してTelegramに送信する

    Args:
        date: 対象日（Noneの場合は今日）
    """
    # レポート生成
    reporter = PerformanceReporter()
    report_text = reporter.daily_report(date)

    # Telegram設定の読み込み
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN が .env に設定されていません")

    if not CHAT_ID_FILE.exists():
        raise FileNotFoundError(
            f"チャットIDファイルが見つかりません: {CHAT_ID_FILE}\n"
            "Telegram Botに /start を送信してください。"
        )

    chat_id = int(CHAT_ID_FILE.read_text().strip())

    # 非同期でTelegram送信
    async def _send():
        from telegram import Bot

        bot = Bot(token=bot_token)
        await bot.send_message(
            chat_id=chat_id,
            text=report_text,
            parse_mode=None,  # プレーンテキストで送信
        )

    asyncio.run(_send())


# ===========================================
# メイン（直接実行用）
# ===========================================

if __name__ == "__main__":
    # テスト用: 全レポートを表示
    reporter = PerformanceReporter()

    print(reporter.daily_report())
    print()
    print(reporter.weekly_report())
    print()
    print(reporter.monthly_report())
    print()
    print(reporter.overall_report())
