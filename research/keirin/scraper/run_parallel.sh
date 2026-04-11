#!/bin/bash
# ===========================================
# run_parallel.sh
# 競輪データの2フェーズ収集パイプライン
#
# 使い方:
#   bash run_parallel.sh 2022-01-01 2024-12-31
#
# 設計:
#   PHASE 1: chariloto.com から4並列でレース結果・出走表を収集
#     - 会場を4グループに分割して並列実行
#     - 各プロセスは独自の進捗ファイル（.scrape_progress_XX_XX）を持つ
#     - SQLite WAL モードで書き込み競合を回避
#     - --resume で途中再開可能
#   PHASE 2: keirin.kdreams.jp から選手統計を補完
#     - DB内の kyoso_tokuten IS NULL のレコードを一括補完
#     - 単一プロセスで順次処理（並列不要）
#   最後に --status で取得件数を表示
# ===========================================

set -u

START=${1:-}
END=${2:-}

if [ -z "$START" ] || [ -z "$END" ]; then
    echo "使い方: bash run_parallel.sh <開始日> <終了日>"
    echo "例: bash run_parallel.sh 2022-01-01 2024-12-31"
    exit 1
fi

# スクリプト自身のディレクトリを基点にパスを解決
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRAPER="$SCRIPT_DIR/scraper_historical.py"

if [ ! -f "$SCRAPER" ]; then
    echo "エラー: scraper_historical.py が見つかりません: $SCRAPER"
    exit 1
fi

# Python実行体の決定
# 優先順位: $KEIRIN_PYTHON → python → python3 → 代表的なWindowsインストール先
if [ -n "${KEIRIN_PYTHON:-}" ] && [ -x "$KEIRIN_PYTHON" ]; then
    PYTHON="$KEIRIN_PYTHON"
elif python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
    PYTHON="python"
elif python3 -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
    PYTHON="python3"
elif [ -x "/c/Users/yuuga/python311/python.exe" ]; then
    PYTHON="/c/Users/yuuga/python311/python.exe"
else
    echo "エラー: Python が見つかりません。環境変数 KEIRIN_PYTHON にパスを設定してください。"
    echo "例: export KEIRIN_PYTHON=/c/Users/yuuga/python311/python.exe"
    exit 1
fi
echo "Python: $PYTHON"

echo "=== PHASE1: chariloto 4並列収集開始 ==="
echo "期間: $START 〜 $END"
echo ""

# グループ1: 北日本（函館01・青森02・いわき平03・弥彦04）
echo "[Group 1] 北日本 (1-4) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 1,2,3,4 --resume &
PID1=$!

# グループ2: 関東（前橋05・取手06・宇都宮07・大宮08・西武園09・京王閣10・立川11）
echo "[Group 2] 関東 (5-11) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 5,6,7,8,9,10,11 --resume &
PID2=$!

# グループ3: 南関東〜中部
# （松戸12・千葉13・川崎14・平塚15・小田原16・伊東17・静岡18・
#   豊橋19・名古屋20・岐阜21・大垣22・松阪23）
echo "[Group 3] 南関東〜中部 (12-23) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 12,13,14,15,16,17,18,19,20,21,22,23 --resume &
PID3=$!

# グループ4: 近畿以西（24〜43 残り全会場）
# 四日市24・富山25・福井26・奈良27・向日町28・和歌山29・岸和田30・
# 玉野31・広島32・防府33・高松34・小松島35・高知36・松山37・
# 小倉38・久留米39・武雄40・佐世保41・別府42・熊本43
echo "[Group 4] 近畿以西 (24-43) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 --resume &
PID4=$!

echo ""
echo "全プロセス起動完了"
echo "  Group 1 PID: $PID1"
echo "  Group 2 PID: $PID2"
echo "  Group 3 PID: $PID3"
echo "  Group 4 PID: $PID4"
echo ""
echo "完了待機中..."

wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=== PHASE1完了 ==="
echo ""
echo "=== PHASE2: Kドリームズ補完開始 ==="
"$PYTHON" "$SCRAPER" --supplement_all
PHASE2_STATUS=$?
if [ $PHASE2_STATUS -ne 0 ]; then
    echo "[警告] PHASE2 が非0で終了: exit=$PHASE2_STATUS"
fi
echo "=== PHASE2完了 ==="
echo ""
echo "=== 全処理完了 ==="
"$PYTHON" "$SCRAPER" --status
