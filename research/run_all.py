# ===========================================
# run_all.py
# 統合ランチャー: FX予測 + 日本株予測 + Telegram Bot
#
# 使い方:
#   python research/run_all.py              # 全システム1回実行
#   python research/run_all.py --loop       # 自動繰り返し
#   python research/run_all.py --fx-only    # FXのみ
#   python research/run_all.py --stock-only # 株のみ
#   python research/run_all.py --bot        # Telegram Bot起動
# ===========================================

import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_fx():
    """FXペーパートレードを1回実行（USD/JPY + CAD/JPY）"""
    # USD/JPY
    print(f"\n{'='*60}")
    print(f"[FX USD/JPY] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    try:
        from research.paper_trade import run_once
        run_once()
    except Exception as e:
        print(f"[FX USD/JPY] エラー: {e}")

    # CAD/JPY
    print(f"\n{'='*60}")
    print(f"[FX CAD/JPY] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    try:
        from research.paper_trade_cadjpy import run_once as run_once_cad
        run_once_cad()
    except Exception as e:
        print(f"[FX CAD/JPY] エラー: {e}")


def run_stocks():
    """日本株ペーパートレードを1回実行（ルネサス + ソフトバンクG）"""
    print(f"\n{'='*60}")
    print(f"[株] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    try:
        from research.paper_trade_stocks import run_once as run_once_stocks
        run_once_stocks()
    except Exception as e:
        print(f"[株] エラー: {e}")
        import traceback
        traceback.print_exc()


def run_daily_report():
    """日次レポートをTelegramに送信"""
    try:
        from research.common.performance_report import send_daily_report_telegram
        send_daily_report_telegram()
        print("[レポート] 日次レポート送信完了")
    except Exception as e:
        print(f"[レポート] 送信失敗: {e}")


def run_bot():
    """Telegram Botを起動（常駐プロセス）"""
    from research.telegram_bot import run_bot as start_bot
    start_bot()


def run_loop(interval_minutes=60, fx=True, stocks=True):
    """定期実行ループ"""
    print(f"統合システム開始（{interval_minutes}分間隔）")
    print(f"  FX: {'ON' if fx else 'OFF'}")
    print(f"  株: {'ON' if stocks else 'OFF'}")
    print("Ctrl+C で停止\n")

    last_stock_run = None  # 株は1日1回のみ
    last_report = None

    while True:
        now = datetime.now()

        try:
            # FXは毎回実行
            if fx:
                run_fx()

            # 株は平日の朝8時台に1回だけ実行（東京市場オープン前）
            if stocks:
                is_weekday = now.weekday() < 5
                is_morning = 7 <= now.hour <= 9
                already_ran = last_stock_run and last_stock_run.date() == now.date()
                if is_weekday and is_morning and not already_ran:
                    run_stocks()
                    last_stock_run = now

            # 自動研究は深夜3時台に実行（1日1回）
            if 3 <= now.hour <= 4:
                already_researched = getattr(run_loop, '_last_research', None)
                if not already_researched or already_researched.date() != now.date():
                    try:
                        from research.auto_research import run_daily_research
                        print(f"\n[研究] 自動研究パイプライン開始...")
                        run_daily_research()
                        run_loop._last_research = now
                    except Exception as e:
                        print(f"[研究] エラー: {e}")

            # 日次レポートは18時台に送信
            if 17 <= now.hour <= 18:
                already_reported = last_report and last_report.date() == now.date()
                if not already_reported:
                    run_daily_report()
                    last_report = now

        except Exception as e:
            print(f"エラー: {e}")

        next_run = now + timedelta(minutes=interval_minutes)
        print(f"\n次回実行: {next_run.strftime('%H:%M:%S')}")

        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n停止しました")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FX AI 統合システム")
    parser.add_argument("--loop", action="store_true", help="自動繰り返しモード")
    parser.add_argument("--interval", type=int, default=60, help="繰り返し間隔（分）")
    parser.add_argument("--fx-only", action="store_true", help="FXのみ実行")
    parser.add_argument("--stock-only", action="store_true", help="株のみ実行")
    parser.add_argument("--bot", action="store_true", help="Telegram Bot起動")
    parser.add_argument("--report", action="store_true", help="日次レポート送信")
    args = parser.parse_args()

    if args.bot:
        run_bot()
    elif args.report:
        run_daily_report()
    elif args.loop:
        fx = not args.stock_only
        stocks = not args.fx_only
        run_loop(args.interval, fx=fx, stocks=stocks)
    elif args.fx_only:
        run_fx()
    elif args.stock_only:
        run_stocks()
    else:
        # デフォルト: 両方1回実行
        run_fx()
        run_stocks()
