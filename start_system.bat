@echo off
echo ============================================
echo  FX AI 統合トレードシステム
echo ============================================
echo.
echo 1. 全システム1回実行
echo 2. FX自動ループ（1時間ごと）
echo 3. 株のみ実行
echo 4. Telegram Bot起動
echo 5. 日次レポート送信
echo 6. 全自動ループ（FX+株+レポート）
echo.
set /p choice="番号を入力: "

if "%choice%"=="1" python research/run_all.py
if "%choice%"=="2" python research/run_all.py --loop --fx-only
if "%choice%"=="3" python research/run_all.py --stock-only
if "%choice%"=="4" python research/run_all.py --bot
if "%choice%"=="5" python research/run_all.py --report
if "%choice%"=="6" python research/run_all.py --loop

pause
