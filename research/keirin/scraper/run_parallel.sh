#!/bin/bash
# ===========================================
# run_parallel.sh
# 競輪データの4並列収集
#
# 使い方:
#   bash run_parallel.sh 2022-01-01 2024-12-31
#
# 設計:
#   - 会場を4グループに分割して並列実行
#   - 各プロセスは独自の進捗ファイル（.scrape_progress_XX_XX）を持つ
#   - SQLite WAL モードで書き込み競合を回避
#   - --resume で途中再開可能
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

echo "=== 4並列スクレイピング開始 ==="
echo "期間: $START 〜 $END"
echo ""

# グループ1: 北日本（函館01・青森02・いわき平03・弥彦04）
echo "[Group 1] 北日本 (1-4) 起動"
python "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 1,2,3,4 --resume &
PID1=$!

# グループ2: 関東（前橋05・取手06・宇都宮07・大宮08・西武園09・京王閣10・立川11）
echo "[Group 2] 関東 (5-11) 起動"
python "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 5,6,7,8,9,10,11 --resume &
PID2=$!

# グループ3: 南関東〜中部
# （松戸12・千葉13・川崎14・平塚15・小田原16・伊東17・静岡18・
#   豊橋19・名古屋20・岐阜21・大垣22・松阪23）
echo "[Group 3] 南関東〜中部 (12-23) 起動"
python "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 12,13,14,15,16,17,18,19,20,21,22,23 --resume &
PID3=$!

# グループ4: 近畿以西（24〜43 残り全会場）
# 四日市24・富山25・福井26・奈良27・向日町28・和歌山29・岸和田30・
# 玉野31・広島32・防府33・高松34・小松島35・高知36・松山37・
# 小倉38・久留米39・武雄40・佐世保41・別府42・熊本43
echo "[Group 4] 近畿以西 (24-43) 起動"
python "$SCRAPER" \
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
echo "=== 全プロセス完了 ==="
