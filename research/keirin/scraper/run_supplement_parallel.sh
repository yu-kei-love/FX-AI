#!/bin/bash
# ===========================================
# run_supplement_parallel.sh
# Kドリームズ補完の2並列実行
#
# 使い方:
#   bash run_supplement_parallel.sh
#
# 設計:
#   - DB内の未補完日付を SQL で取得
#   - 日数を前後半で分割
#   - 2プロセスを --supplement_range で同時起動
#   - delay=2.0s は各プロセスで維持（サーバー負荷を上げない）
#   - DBは WAL モードで書き込み競合なし
#   - COALESCE UPDATE で重複処理されても安全
# ===========================================

set -u

# スクリプト自身のディレクトリを基点にパスを解決
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRAPER="$SCRIPT_DIR/scraper_historical.py"
LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/logs"

if [ ! -f "$SCRAPER" ]; then
    echo "エラー: scraper_historical.py が見つかりません"
    exit 1
fi

mkdir -p "$LOG_DIR"

# Python実行体の決定
if [ -n "${KEIRIN_PYTHON:-}" ] && [ -x "$KEIRIN_PYTHON" ]; then
    PYTHON="$KEIRIN_PYTHON"
elif python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
    PYTHON="python"
elif python3 -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
    PYTHON="python3"
elif [ -x "/c/Users/yuuga/python311/python.exe" ]; then
    PYTHON="/c/Users/yuuga/python311/python.exe"
else
    echo "エラー: Python が見つかりません"
    exit 1
fi
echo "Python: $PYTHON"

# DB パス（scraper 側と同じ相対パス計算）
DB_PATH="$(cd "$SCRIPT_DIR/../../.." && pwd)/data/keirin/keirin.db"
if [ ! -f "$DB_PATH" ]; then
    echo "エラー: DB が見つかりません: $DB_PATH"
    exit 1
fi

# 未補完日付リストを SQL で取得し、中央値で前後半に分ける
SPLIT_RESULT=$("$PYTHON" - <<PYEOF
import sqlite3
from datetime import datetime

conn = sqlite3.connect(r"$DB_PATH")
cur = conn.cursor()
cur.execute("""
    SELECT DISTINCT r.race_date
    FROM entries e
    JOIN races r ON e.race_id = r.race_id
    WHERE e.kyoso_tokuten IS NULL
    ORDER BY r.race_date
""")
dates = [row[0] for row in cur.fetchall()]
conn.close()

if not dates:
    print("NO_DATES")
else:
    # YYYYMMDD → YYYY-MM-DD
    def fmt(d):
        dt = datetime.strptime(d, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")

    n = len(dates)
    mid = n // 2
    first_start = fmt(dates[0])
    first_end = fmt(dates[mid - 1]) if mid >= 1 else fmt(dates[0])
    second_start = fmt(dates[mid])
    second_end = fmt(dates[-1])
    print(f"{n}|{first_start}|{first_end}|{second_start}|{second_end}")
PYEOF
)

if [ "$SPLIT_RESULT" = "NO_DATES" ]; then
    echo "未補完の日付なし。処理不要です。"
    exit 0
fi

IFS='|' read -r TOTAL FIRST_START FIRST_END SECOND_START SECOND_END <<< "$SPLIT_RESULT"

echo "=== Kドリームズ補完 2並列起動 ==="
echo "未補完日数: $TOTAL"
echo "  前半: $FIRST_START 〜 $FIRST_END"
echo "  後半: $SECOND_START 〜 $SECOND_END"
echo ""

# 前半
echo "[Half 1] 前半プロセス起動"
"$PYTHON" "$SCRAPER" \
    --supplement_range "$FIRST_START,$FIRST_END" \
    --delay 2.0 &
PID1=$!

# 後半
echo "[Half 2] 後半プロセス起動"
"$PYTHON" "$SCRAPER" \
    --supplement_range "$SECOND_START,$SECOND_END" \
    --delay 2.0 &
PID2=$!

echo ""
echo "全プロセス起動完了"
echo "  Half 1 PID: $PID1"
echo "  Half 2 PID: $PID2"
echo ""
echo "完了待機中..."

wait $PID1 $PID2

echo ""
echo "=== 全プロセス完了 ==="
"$PYTHON" "$SCRAPER" --status
