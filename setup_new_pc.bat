@echo off
echo ==========================================
echo  FX-AI セットアップスクリプト
echo ==========================================
echo.

REM Python確認
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python がインストールされていません
    echo https://www.python.org/downloads/ からPython 3.11をインストールしてください
    pause
    exit /b 1
)

echo [1/3] 依存ライブラリのインストール...
pip install -r requirements.txt
echo.

echo [2/3] .env ファイルの確認...
if not exist .env (
    echo [WARN] .env ファイルがありません！
    echo 元のPCから .env をコピーしてください
    echo 必要なキー: OANDA_API_KEY, TELEGRAM_BOT_TOKEN, etc.
) else (
    echo [OK] .env ファイルあり
)
echo.

echo [3/3] プロセス起動...
echo 以下のコマンドで各プロセスを起動できます:
echo.
echo   FX paper trade:    start /b python research/paper_trade.py --loop
echo   Crypto paper trade: start /b python research/crypto/paper_trade.py --loop
echo   Telegram bot:       start /b python research/telegram_bot.py
echo   Boat auto collect:  start /b python -u -c "from research.boat.auto_collect_and_analyze import main; main()"
echo   Keiba scraper:      start /b python research/keiba/fetch_netkeiba.py --from-year 2023 --to-year 2025
echo   Keirin scraper:     start /b python research/keirin/fetch_keirin_data.py --from-date 20250101 --to-date 20260325
echo.
echo ==========================================
echo  セットアップ完了
echo ==========================================
pause
