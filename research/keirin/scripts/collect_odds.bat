@echo off
REM ===========================================
REM keirin: 事前オッズ自動収集バッチ
REM Windowsタスクスケジューラで定時実行する
REM 推奨実行時刻: 10:00 / 14:00 / 18:00 / 21:00
REM
REM 実行内容:
REM   当日のレースに対して締切N分前スナップショットを取得
REM   (DEFAULT_SNAPSHOT_MINUTES = [60, 30, 10, 0])
REM
REM 依存: Python 3.11 (C:\Users\yuuga\python311\python.exe)
REM ===========================================

cd /d C:\Users\yuuga\FX-AI\research\keirin

set PY=C:\Users\yuuga\python311\python.exe
set LOGDIR=data\keirin
set LOG=%LOGDIR%\odds_log.txt
set ERRLOG=%LOGDIR%\odds_errors.log

REM ログディレクトリ確保
if not exist %LOGDIR% mkdir %LOGDIR%

echo. >> %LOG%
echo ========== %date% %time% START ========== >> %LOG%

%PY% scraper\scraper_realtime.py --today >> %LOG% 2>> %ERRLOG%
set RC=%errorlevel%

if %RC% EQU 0 (
    echo %date% %time% SUCCESS (rc=%RC%) >> %LOG%
) else (
    echo %date% %time% FAILED (rc=%RC%) >> %LOG%
    echo %date% %time% FAILED (rc=%RC%) >> %ERRLOG%
)

echo ========== %date% %time% END ========== >> %LOG%
