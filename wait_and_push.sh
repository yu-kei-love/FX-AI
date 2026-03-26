#!/bin/bash
echo "Waiting for background tasks to finish..."
# 5分ごとにチェック、最大6時間待つ
for i in $(seq 1 72); do
    # pythonプロセスがなくなったらpush
    if ! tasklist 2>/dev/null | grep -qi "python"; then
        echo "Python processes finished. Pushing..."
        git add -A
        git commit -m "Auto-save: v3.5 computation results

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
        git push origin master
        echo "Done! $(date)"
        exit 0
    fi
    echo "Still running... check $i/72 ($(date))"
    sleep 300
done
echo "Timeout. Pushing whatever we have..."
git add -A
git commit -m "Auto-save: partial results

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push origin master
