@echo off
chcp 65001 >nul 2>nul
title FX AI Paper Trade
echo ============================================
echo   FX AI Paper Trade - Auto Mode
echo   Close this window to stop
echo ============================================
echo.
cd /d "%~dp0"
python -X utf8 research/paper_trade.py --loop
pause
