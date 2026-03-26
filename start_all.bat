@echo off
echo ==========================================
echo  全プロセス一括起動
echo ==========================================
cd /d %~dp0

echo [1/6] FX paper trade (v3.1)...
start /b python research/paper_trade.py --loop

echo [2/6] Crypto paper trade (v3.0)...
start /b python research/crypto/paper_trade.py --loop

echo [3/6] Telegram bot...
start /b python research/telegram_bot.py

echo [4/6] Boat auto collect...
start /b python -u -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8'); from research.boat.auto_collect_and_analyze import main; main()"

echo [5/6] Keiba scraper (netkeiba.com)...
start /b python research/keiba/fetch_netkeiba.py --from-year 2023 --to-year 2025

echo [6/6] Keirin scraper (keirin-station.com)...
start /b python research/keirin/fetch_keirin_data.py --from-date 20250101 --to-date 20260325

echo.
echo 全6プロセス起動完了！
echo タスクマネージャーでpython.exeを確認してください
pause
