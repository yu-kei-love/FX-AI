# ===========================================
# sales_tracker.py
# note.com 売上分析トラッカー
#
# 機能:
#   - 売上データの手動/自動記録
#   - 時間帯別売上分析
#   - タイトルパターン別売上分析
#   - 価格帯別コンバージョン分析
#   - 週次売上レポート生成
#   - matplotlibによるグラフ生成
#
# データ保存先: data/note_sales/sales_log.csv
# ===========================================

import sys
import os
import re
import csv
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ログ設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "sales_tracker.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# パス定義
DATA_DIR = PROJECT_ROOT / "data" / "note_sales"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SALES_LOG_FILE = DATA_DIR / "sales_log.csv"
CHARTS_DIR = DATA_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# CSVカラム定義
CSV_COLUMNS = [
    "sale_id",
    "article_id",
    "title",
    "price",
    "timestamp",
    "hour",
    "day_of_week",
    "article_type",
    "title_pattern",
]

# タイトルパターン分類
TITLE_PATTERN_RULES = {
    "括弧付き": r"^【",
    "数字入り": r"\d+",
    "疑問形": r"[？?]$",
    "ハウツー": r"(方法|やり方|入門|始め方)",
    "実績公開": r"(実績|結果|成績|公開)",
    "コード付き": r"(コード|Python|実装)",
    "検証系": r"(検証|テスト|比較)",
    "初心者向け": r"(初心者|入門|基礎)",
}

DAY_NAMES_JP = ["月", "火", "水", "木", "金", "土", "日"]


def _ensure_csv_exists() -> None:
    """CSVファイルが存在しない場合、ヘッダーを書き込んで作成する"""
    if not SALES_LOG_FILE.exists():
        with open(SALES_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        logger.info(f"売上ログCSVを作成しました: {SALES_LOG_FILE}")


def _classify_title_pattern(title: str) -> str:
    """タイトルからパターンを分類する"""
    for pattern_name, regex in TITLE_PATTERN_RULES.items():
        if re.search(regex, title):
            return pattern_name
    return "その他"


def _load_sales_data() -> List[Dict[str, Any]]:
    """売上データを全件読み込む"""
    _ensure_csv_exists()
    data = []
    with open(SALES_LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["price"] = int(row["price"]) if row["price"] else 0
            row["hour"] = int(row["hour"]) if row["hour"] else 0
            data.append(row)
    return data


def log_sale(
    article_id: str,
    title: str,
    price: int,
    timestamp: Optional[str] = None,
    article_type: str = "paid",
) -> Dict[str, Any]:
    """売上を記録する

    Args:
        article_id: note.comの記事ID
        title: 記事タイトル
        price: 販売価格（円）
        timestamp: 売上日時（ISO形式）。Noneの場合は現在時刻
        article_type: 記事タイプ（free/paid/magazine）

    Returns:
        記録された売上データ
    """
    _ensure_csv_exists()

    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromisoformat(timestamp)

    sale_id = f"sale_{dt.strftime('%Y%m%d%H%M%S')}_{article_id[:8]}"
    title_pattern = _classify_title_pattern(title)

    sale_data = {
        "sale_id": sale_id,
        "article_id": article_id,
        "title": title,
        "price": price,
        "timestamp": dt.isoformat(),
        "hour": dt.hour,
        "day_of_week": DAY_NAMES_JP[dt.weekday()],
        "article_type": article_type,
        "title_pattern": title_pattern,
    }

    with open(SALES_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([sale_data[col] for col in CSV_COLUMNS])

    logger.info(f"売上記録: {sale_id} | {title[:30]}... | {price}円")
    return sale_data


def analyze_by_hour() -> Dict[str, Any]:
    """時間帯別の売上分析

    Returns:
        時間帯別分析結果
    """
    logger.info("時間帯別売上分析開始")

    data = _load_sales_data()
    if not data:
        logger.warning("売上データがありません")
        return {"error": "データなし"}

    # 時間帯ごとの集計
    hourly = defaultdict(lambda: {"count": 0, "revenue": 0})
    for row in data:
        h = row["hour"]
        hourly[h]["count"] += 1
        hourly[h]["revenue"] += row["price"]

    # 全時間帯を0で埋める
    result = {}
    for h in range(24):
        result[f"{h:02d}時"] = {
            "件数": hourly[h]["count"],
            "売上": hourly[h]["revenue"],
            "平均単価": round(hourly[h]["revenue"] / hourly[h]["count"]) if hourly[h]["count"] > 0 else 0,
        }

    # ベスト時間帯
    best_hour = max(hourly.keys(), key=lambda h: hourly[h]["revenue"]) if hourly else None

    analysis = {
        "hourly_breakdown": result,
        "best_hour": f"{best_hour:02d}時" if best_hour is not None else "N/A",
        "total_sales": len(data),
        "total_revenue": sum(row["price"] for row in data),
    }

    # グラフ生成
    _generate_hourly_chart(hourly)

    logger.info(f"時間帯別分析完了: ベスト時間帯={analysis['best_hour']}")
    return analysis


def analyze_by_title_pattern() -> Dict[str, Any]:
    """タイトルパターン別の売上分析

    Returns:
        パターン別分析結果
    """
    logger.info("タイトルパターン別売上分析開始")

    data = _load_sales_data()
    if not data:
        logger.warning("売上データがありません")
        return {"error": "データなし"}

    # パターンごとの集計
    pattern_stats = defaultdict(lambda: {"count": 0, "revenue": 0, "prices": []})
    for row in data:
        p = row["title_pattern"]
        pattern_stats[p]["count"] += 1
        pattern_stats[p]["revenue"] += row["price"]
        pattern_stats[p]["prices"].append(row["price"])

    result = {}
    for pattern, stats in pattern_stats.items():
        avg_price = round(stats["revenue"] / stats["count"]) if stats["count"] > 0 else 0
        result[pattern] = {
            "件数": stats["count"],
            "売上合計": stats["revenue"],
            "平均単価": avg_price,
        }

    # ベストパターン
    best_pattern = max(pattern_stats.keys(), key=lambda p: pattern_stats[p]["revenue"]) if pattern_stats else "N/A"

    analysis = {
        "pattern_breakdown": result,
        "best_pattern": best_pattern,
        "total_patterns": len(pattern_stats),
    }

    # グラフ生成
    _generate_pattern_chart(pattern_stats)

    logger.info(f"パターン分析完了: ベストパターン={best_pattern}")
    return analysis


def analyze_by_price() -> Dict[str, Any]:
    """価格帯別のコンバージョン分析

    Returns:
        価格帯別分析結果
    """
    logger.info("価格帯別売上分析開始")

    data = _load_sales_data()
    if not data:
        logger.warning("売上データがありません")
        return {"error": "データなし"}

    # 価格帯定義
    price_ranges = [
        (0, 0, "無料"),
        (1, 300, "100-300円"),
        (301, 500, "301-500円"),
        (501, 1000, "501-1000円"),
        (1001, 2000, "1001-2000円"),
        (2001, 5000, "2001-5000円"),
        (5001, 99999, "5001円以上"),
    ]

    range_stats = defaultdict(lambda: {"count": 0, "revenue": 0})
    for row in data:
        price = row["price"]
        for low, high, label in price_ranges:
            if low <= price <= high:
                range_stats[label]["count"] += 1
                range_stats[label]["revenue"] += price
                break

    result = {}
    for label in [r[2] for r in price_ranges]:
        stats = range_stats[label]
        result[label] = {
            "件数": stats["count"],
            "売上合計": stats["revenue"],
        }

    # ベスト価格帯（売上額ベース）
    best_range = max(range_stats.keys(), key=lambda r: range_stats[r]["revenue"]) if range_stats else "N/A"

    analysis = {
        "price_breakdown": result,
        "best_price_range": best_range,
        "avg_price": round(sum(row["price"] for row in data) / len(data)) if data else 0,
    }

    # グラフ生成
    _generate_price_chart(range_stats, price_ranges)

    logger.info(f"価格分析完了: ベスト価格帯={best_range}")
    return analysis


def generate_sales_report() -> str:
    """週次売上レポートを生成する

    Returns:
        レポートのファイルパス
    """
    logger.info("週次売上レポート生成開始")

    data = _load_sales_data()

    # 直近7日間のデータ
    cutoff = datetime.now() - timedelta(days=7)
    weekly_data = [
        row for row in data
        if datetime.fromisoformat(row["timestamp"]) > cutoff
    ]

    total_revenue = sum(row["price"] for row in weekly_data)
    total_count = len(weekly_data)
    avg_price = round(total_revenue / total_count) if total_count > 0 else 0

    # 日別集計
    daily = defaultdict(lambda: {"count": 0, "revenue": 0})
    for row in weekly_data:
        dt = datetime.fromisoformat(row["timestamp"])
        date_key = dt.strftime("%m/%d(%a)")
        daily[date_key]["count"] += 1
        daily[date_key]["revenue"] += row["price"]

    # レポート生成
    report_lines = [
        "=" * 50,
        f"  note.com 週次売上レポート",
        f"  期間: {(datetime.now() - timedelta(days=7)).strftime('%Y/%m/%d')} - {datetime.now().strftime('%Y/%m/%d')}",
        "=" * 50,
        "",
        f"総売上: {total_revenue:,}円",
        f"売上件数: {total_count}件",
        f"平均単価: {avg_price:,}円",
        "",
        "--- 日別内訳 ---",
    ]

    for date_key, stats in sorted(daily.items()):
        report_lines.append(f"  {date_key}: {stats['count']}件 / {stats['revenue']:,}円")

    report_lines.extend([
        "",
        "--- 全期間累計 ---",
        f"  総売上: {sum(row['price'] for row in data):,}円",
        f"  総件数: {len(data)}件",
        "",
        "=" * 50,
    ])

    report_text = "\n".join(report_lines)

    # レポートファイル保存
    report_file = DATA_DIR / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.txt"
    report_file.write_text(report_text, encoding="utf-8")
    logger.info(f"週次レポート保存: {report_file}")

    # グラフ生成
    _generate_weekly_chart(daily)

    print(report_text)
    return str(report_file)


# =============================================================
# グラフ生成関数
# =============================================================

def _generate_hourly_chart(hourly: Dict[int, Dict[str, int]]) -> None:
    """時間帯別売上グラフを生成する"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # GUIなし
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 日本語フォント設定（利用可能なフォントを探す）
        jp_fonts = [f.name for f in fm.fontManager.ttflist if "gothic" in f.name.lower() or "mincho" in f.name.lower() or "meiryo" in f.name.lower()]
        if jp_fonts:
            plt.rcParams["font.family"] = jp_fonts[0]
        else:
            plt.rcParams["font.family"] = "sans-serif"

        hours = list(range(24))
        counts = [hourly.get(h, {}).get("count", 0) for h in hours]
        revenues = [hourly.get(h, {}).get("revenue", 0) for h in hours]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.bar(hours, counts, color="#4A90D9", alpha=0.8)
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Sales Count")
        ax1.set_title("Sales by Hour of Day")
        ax1.set_xticks(hours)

        ax2.bar(hours, revenues, color="#7B68EE", alpha=0.8)
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("Revenue (JPY)")
        ax2.set_title("Revenue by Hour of Day")
        ax2.set_xticks(hours)

        plt.tight_layout()
        chart_path = CHARTS_DIR / f"hourly_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(str(chart_path), dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"時間帯別グラフ保存: {chart_path}")

    except ImportError:
        logger.warning("matplotlib が見つかりません。グラフ生成をスキップします。")
    except Exception as e:
        logger.error(f"グラフ生成エラー: {e}")


def _generate_pattern_chart(pattern_stats: Dict[str, Dict[str, Any]]) -> None:
    """パターン別売上グラフを生成する"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        jp_fonts = [f.name for f in fm.fontManager.ttflist if "gothic" in f.name.lower() or "meiryo" in f.name.lower()]
        if jp_fonts:
            plt.rcParams["font.family"] = jp_fonts[0]

        patterns = list(pattern_stats.keys())
        counts = [pattern_stats[p]["count"] for p in patterns]
        revenues = [pattern_stats[p]["revenue"] for p in patterns]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.barh(patterns, counts, color="#4A90D9", alpha=0.8)
        ax1.set_xlabel("Sales Count")
        ax1.set_title("Sales by Title Pattern")

        ax2.barh(patterns, revenues, color="#7B68EE", alpha=0.8)
        ax2.set_xlabel("Revenue (JPY)")
        ax2.set_title("Revenue by Title Pattern")

        plt.tight_layout()
        chart_path = CHARTS_DIR / f"pattern_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(str(chart_path), dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"パターン別グラフ保存: {chart_path}")

    except ImportError:
        logger.warning("matplotlib が見つかりません。グラフ生成をスキップします。")
    except Exception as e:
        logger.error(f"グラフ生成エラー: {e}")


def _generate_price_chart(
    range_stats: Dict[str, Dict[str, int]],
    price_ranges: List[Tuple],
) -> None:
    """価格帯別グラフを生成する"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r[2] for r in price_ranges]
        counts = [range_stats[label]["count"] for label in labels]
        revenues = [range_stats[label]["revenue"] for label in labels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.bar(labels, counts, color="#4A90D9", alpha=0.8)
        ax1.set_xlabel("Price Range")
        ax1.set_ylabel("Sales Count")
        ax1.set_title("Sales by Price Range")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(labels, revenues, color="#7B68EE", alpha=0.8)
        ax2.set_xlabel("Price Range")
        ax2.set_ylabel("Revenue (JPY)")
        ax2.set_title("Revenue by Price Range")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        chart_path = CHARTS_DIR / f"price_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(str(chart_path), dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"価格帯別グラフ保存: {chart_path}")

    except ImportError:
        logger.warning("matplotlib が見つかりません。グラフ生成をスキップします。")
    except Exception as e:
        logger.error(f"グラフ生成エラー: {e}")


def _generate_weekly_chart(daily: Dict[str, Dict[str, int]]) -> None:
    """週次推移グラフを生成する"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dates = sorted(daily.keys())
        counts = [daily[d]["count"] for d in dates]
        revenues = [daily[d]["revenue"] for d in dates]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(dates, counts, marker="o", color="#4A90D9", linewidth=2)
        ax1.fill_between(dates, counts, alpha=0.2, color="#4A90D9")
        ax1.set_ylabel("Sales Count")
        ax1.set_title("Weekly Sales Trend")

        ax2.bar(dates, revenues, color="#7B68EE", alpha=0.8)
        ax2.set_ylabel("Revenue (JPY)")
        ax2.set_title("Weekly Revenue")

        plt.tight_layout()
        chart_path = CHARTS_DIR / f"weekly_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(str(chart_path), dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"週次グラフ保存: {chart_path}")

    except ImportError:
        logger.warning("matplotlib が見つかりません。グラフ生成をスキップします。")
    except Exception as e:
        logger.error(f"グラフ生成エラー: {e}")


# =============================================================
# CLI エントリーポイント
# =============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="note.com 売上トラッカー")
    parser.add_argument(
        "--action",
        choices=["log", "hourly", "pattern", "price", "report"],
        default="report",
        help="実行アクション (default: report)",
    )
    parser.add_argument("--article-id", type=str, default="", help="記事ID（log用）")
    parser.add_argument("--title", type=str, default="", help="記事タイトル（log用）")
    parser.add_argument("--price", type=int, default=0, help="販売価格（log用）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  売上トラッカー: action={args.action}")
    logger.info("=" * 60)

    if args.action == "log":
        if not args.article_id or not args.title or args.price <= 0:
            logger.error("--article-id, --title, --price を指定してください")
        else:
            result = log_sale(args.article_id, args.title, args.price)
            logger.info(f"記録完了: {result}")

    elif args.action == "hourly":
        result = analyze_by_hour()
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "pattern":
        result = analyze_by_title_pattern()
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "price":
        result = analyze_by_price()
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "report":
        path = generate_sales_report()
        logger.info(f"レポート: {path}")
