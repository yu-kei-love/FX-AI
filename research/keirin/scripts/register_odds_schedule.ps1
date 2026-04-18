# ===========================================
# keirin: Windowsタスクスケジューラ登録 (PowerShell)
#
# 使い方: 管理者 PowerShell で実行
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\scripts\register_odds_schedule.ps1
#
# 登録内容:
#   タスク名: KeirinOddsCollect_{HHMM}
#   実行時刻: 10:00 / 14:00 / 18:00 / 21:00 (毎日)
#   アクション: collect_odds.bat 実行
# ===========================================

$BatchPath = "C:\Users\yuuga\FX-AI\research\keirin\scripts\collect_odds.bat"
$TaskTimes = @("10:00", "14:00", "18:00", "21:00")

foreach ($t in $TaskTimes) {
    $safeName = $t -replace ":", ""
    $TaskName = "KeirinOddsCollect_$safeName"

    Write-Host "登録中: $TaskName @ $t ..."

    $Action = New-ScheduledTaskAction -Execute $BatchPath
    $Trigger = New-ScheduledTaskTrigger -Daily -At $t
    $Settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 90)
    $Principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive

    # 既存があれば上書き登録
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "競輪オッズ自動収集 ($t)" `
        -Force
}

Write-Host ""
Write-Host "=== 登録完了 ==="
Get-ScheduledTask -TaskName "KeirinOddsCollect_*" |
    Select-Object TaskName, State, @{n="NextRun";e={(Get-ScheduledTaskInfo $_).NextRunTime}}
