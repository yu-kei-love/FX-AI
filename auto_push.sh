#!/bin/bash
# 計算完了後に自動でGitHubにpushするスクリプト
cd "C:/Users/kawanoyuu/Desktop/FX-AI"
git add -A
git commit -m "Auto-save: v3.5 computation results

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push origin master
echo "Push complete: $(date)"
