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

# 注意: --jyo_cds に渡すのは JKAコード（11〜87）
# bank_master.py の venue_id (01〜43) ではない
# 全43会場の JKAコード:
#   11,12,13,21,22,23,24,25,26,27,28,31,32,34,35,36,37,38,
#   42,43,44,45,46,47,48,51,53,54,55,56,61,62,63,
#   71,73,74,75,81,83,84,85,86,87

# グループ1: 北日本・関東A (11会場)
# 函館11,青森12,いわき平13,弥彦21,前橋22,取手23,
# 宇都宮24,大宮25,西武園26,京王閣27,立川28
echo "[Group 1] 北日本・関東A (11会場) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 11,12,13,21,22,23,24,25,26,27,28 --resume &
PID1=$!

# グループ2: 南関東・中部A (10会場)
# 松戸31,千葉32,川崎34,平塚35,小田原36,伊東37,静岡38,
# 名古屋42,岐阜43,大垣44
echo "[Group 2] 南関東・中部A (10会場) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 31,32,34,35,36,37,38,42,43,44 --resume &
PID2=$!

# グループ3: 中部B・北信越・近畿・中国 (12会場)
# 豊橋45,富山46,松阪47,四日市48,福井51,奈良53,向日町54,
# 和歌山55,岸和田56,玉野61,広島62,防府63
echo "[Group 3] 中部B・北信越・近畿・中国 (12会場) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 45,46,47,48,51,53,54,55,56,61,62,63 --resume &
PID3=$!

# グループ4: 四国・九州 (10会場)
# 高松71,小松島73,高知74,松山75,小倉81,久留米83,武雄84,
# 佐世保85,別府86,熊本87
echo "[Group 4] 四国・九州 (10会場) 起動"
"$PYTHON" "$SCRAPER" \
    --start "$START" --end "$END" \
    --jyo_cds 71,73,74,75,81,83,84,85,86,87 --resume &
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
