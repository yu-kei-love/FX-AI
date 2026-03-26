# ===========================================
# twitter_bot.py
# Twitter/X 自動投稿ボット
#
# 機能:
#   - 日次予測ツイート投稿
#   - note記事のプロモーション投稿
#   - 週次パフォーマンスレポート投稿
#   - ツイートのスケジュール予約
#   - レート制限・エラーハンドリング
#
# 必要な環境変数:
#   TWITTER_API_KEY, TWITTER_API_SECRET,
#   TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
# ===========================================

import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ログ設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "twitter_bot.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Twitter API 認証情報
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")

# ツイートログ保存先
TWEET_LOG_DIR = PROJECT_ROOT / "data" / "note_sales"
TWEET_LOG_DIR.mkdir(parents=True, exist_ok=True)
TWEET_LOG_FILE = TWEET_LOG_DIR / "tweet_log.json"

# レート制限設定
RATE_LIMIT_TWEETS_PER_15MIN = 50  # Twitter API v2 の制限
RATE_LIMIT_TWEETS_PER_DAY = 300
MIN_TWEET_INTERVAL_SEC = 30  # ツイート間の最小間隔（秒）

# ハッシュタグ定義
HASHTAGS = {
    "general": ["#AI予測", "#投資", "#Python"],
    "fx": ["#FX", "#為替", "#AI予測", "#自動売買"],
    "stock": ["#日経225", "#日本株", "#AI投資", "#株予測"],
    "boat": ["#競艇", "#ボートレース", "#AI予測", "#データ分析"],
    "crypto": ["#暗号通貨", "#ビットコイン", "#AI分析", "#仮想通貨"],
    "note": ["#note", "#AI", "#機械学習", "#投資"],
}


def _get_twitter_client():
    """tweepy クライアントを取得する

    Returns:
        tweepy.Client オブジェクト

    Raises:
        ValueError: API キーが設定されていない場合
        ImportError: tweepy がインストールされていない場合
    """
    if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
        raise ValueError(
            "Twitter API認証情報が設定されていません。\n"
            ".envファイルに以下を設定してください:\n"
            "  TWITTER_API_KEY=...\n"
            "  TWITTER_API_SECRET=...\n"
            "  TWITTER_ACCESS_TOKEN=...\n"
            "  TWITTER_ACCESS_SECRET=..."
        )

    try:
        import tweepy
    except ImportError:
        raise ImportError(
            "tweepy がインストールされていません。\n"
            "pip install tweepy でインストールしてください。"
        )

    client = tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET,
    )
    return client


def _load_tweet_log() -> List[Dict[str, Any]]:
    """ツイートログを読み込む"""
    if TWEET_LOG_FILE.exists():
        try:
            return json.loads(TWEET_LOG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return []
    return []


def _save_tweet_log(log: List[Dict[str, Any]]) -> None:
    """ツイートログを保存する"""
    TWEET_LOG_FILE.write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _check_rate_limit() -> bool:
    """レート制限をチェックする

    Returns:
        True: 投稿可能 / False: 制限に達している
    """
    log = _load_tweet_log()
    now = datetime.now()

    # 直近15分のツイート数
    window_15min = now - timedelta(minutes=15)
    recent_15min = [
        entry for entry in log
        if datetime.fromisoformat(entry["timestamp"]) > window_15min
    ]
    if len(recent_15min) >= RATE_LIMIT_TWEETS_PER_15MIN:
        logger.warning(f"レート制限: 15分間に{len(recent_15min)}件ツイート済み（上限: {RATE_LIMIT_TWEETS_PER_15MIN}）")
        return False

    # 直近24時間のツイート数
    window_24h = now - timedelta(hours=24)
    recent_24h = [
        entry for entry in log
        if datetime.fromisoformat(entry["timestamp"]) > window_24h
    ]
    if len(recent_24h) >= RATE_LIMIT_TWEETS_PER_DAY:
        logger.warning(f"レート制限: 24時間で{len(recent_24h)}件ツイート済み（上限: {RATE_LIMIT_TWEETS_PER_DAY}）")
        return False

    # 最後のツイートからの間隔
    if log:
        last_tweet_time = datetime.fromisoformat(log[-1]["timestamp"])
        elapsed = (now - last_tweet_time).total_seconds()
        if elapsed < MIN_TWEET_INTERVAL_SEC:
            wait_sec = MIN_TWEET_INTERVAL_SEC - elapsed
            logger.info(f"最小間隔待機中: あと{wait_sec:.0f}秒")
            time.sleep(wait_sec)

    return True


def _post_tweet(text: str, dry_run: bool = False) -> Optional[str]:
    """ツイートを投稿する（内部関数）

    Args:
        text: ツイート本文（280文字以内）
        dry_run: True の場合は実際に投稿しない

    Returns:
        投稿成功時のツイートID（dry_runの場合は"DRY_RUN"）
    """
    # 文字数チェック（日本語は全角1文字 = 2カウント扱い）
    if len(text) > 280:
        logger.warning(f"ツイート文字数超過: {len(text)}文字（280文字以内に調整してください）")
        text = text[:277] + "..."

    if not _check_rate_limit():
        logger.error("レート制限に達しています。しばらく待ってから再試行してください。")
        return None

    if dry_run:
        logger.info(f"[DRY RUN] ツイート内容:\n{text}")
        tweet_id = "DRY_RUN"
    else:
        try:
            client = _get_twitter_client()
            response = client.create_tweet(text=text)
            tweet_id = str(response.data["id"])
            logger.info(f"ツイート投稿成功: ID={tweet_id}")
        except Exception as e:
            logger.error(f"ツイート投稿エラー: {e}")
            return None

    # ログに記録
    log = _load_tweet_log()
    log.append({
        "tweet_id": tweet_id,
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
    })
    _save_tweet_log(log)

    return tweet_id


def _build_hashtag_string(categories: List[str]) -> str:
    """カテゴリに応じたハッシュタグ文字列を構築する"""
    tags = set()
    for cat in categories:
        if cat in HASHTAGS:
            tags.update(HASHTAGS[cat])
    # generalは常に含める
    tags.update(HASHTAGS.get("general", []))
    return " ".join(sorted(tags))


# =============================================================
# メイン関数
# =============================================================

def post_prediction_tweet(
    prediction_data: Dict[str, Any],
    dry_run: bool = False,
) -> Optional[str]:
    """日次予測サマリーをツイートする

    Args:
        prediction_data: 予測データ辞書
            例: {"market": "日本株", "direction": "上昇", "confidence": 0.72, "model": "アンサンブル"}
        dry_run: Trueなら投稿せずにログのみ

    Returns:
        ツイートID（失敗時はNone）
    """
    logger.info("予測ツイート投稿開始")

    market = prediction_data.get("market", "マーケット")
    direction = prediction_data.get("direction", "N/A")
    confidence = prediction_data.get("confidence", 0)
    model = prediction_data.get("model", "AIモデル")
    today = datetime.now().strftime("%m/%d")

    # ツイート本文生成
    confidence_pct = f"{confidence * 100:.0f}" if isinstance(confidence, float) else str(confidence)

    text = (
        f"【{today} {market} AI予測】\n\n"
        f"予測方向: {direction}\n"
        f"信頼度: {confidence_pct}%\n"
        f"使用モデル: {model}\n\n"
        f"※AIモデルの予測です。投資は自己責任でお願いします。\n\n"
    )

    # カテゴリ判定してハッシュタグ追加
    categories = ["general"]
    market_lower = market.lower()
    if "株" in market or "stock" in market_lower or "日経" in market:
        categories.append("stock")
    elif "fx" in market_lower or "為替" in market:
        categories.append("fx")
    elif "競艇" in market or "boat" in market_lower:
        categories.append("boat")
    elif "暗号" in market or "crypto" in market_lower or "ビットコイン" in market:
        categories.append("crypto")

    hashtags = _build_hashtag_string(categories)
    text += hashtags

    return _post_tweet(text, dry_run=dry_run)


def post_note_promotion(
    article_title: str,
    url: str,
    dry_run: bool = False,
) -> Optional[str]:
    """note記事のプロモーションツイートを投稿する

    Args:
        article_title: 記事タイトル
        url: 記事のURL
        dry_run: Trueなら投稿せずにログのみ

    Returns:
        ツイートID（失敗時はNone）
    """
    logger.info(f"note記事プロモーション: {article_title}")

    # プロモーション文テンプレート
    import random
    promo_templates = [
        "新記事を公開しました！\n\n📝 {title}\n\n{url}",
        "【新着記事】\n{title}\n\nAIモデルの検証結果を詳しく解説しています。\n{url}",
        "AIで投資予測する方法を記事にまとめました。\n\n{title}\n\n{url}",
        "noteに新しい記事を投稿しました。\n\n{title}\n\nコードも公開中です。\n{url}",
        "検証結果を記事にしました。\n\n{title}\n\n{url}",
    ]

    template = random.choice(promo_templates)
    text = template.format(title=article_title, url=url)

    # ハッシュタグ追加
    hashtags = _build_hashtag_string(["note", "general"])
    text += f"\n\n{hashtags}"

    return _post_tweet(text, dry_run=dry_run)


def post_weekly_performance(
    stats: Dict[str, Any],
    dry_run: bool = False,
) -> Optional[str]:
    """週次パフォーマンスレポートをツイートする

    Args:
        stats: パフォーマンス統計辞書
            例: {
                "period": "3/18-3/24",
                "models": {
                    "日本株": {"win_rate": 0.65, "trades": 20},
                    "FX": {"win_rate": 0.58, "trades": 15},
                },
                "total_trades": 35,
                "overall_win_rate": 0.62,
            }
        dry_run: Trueなら投稿せずにログのみ

    Returns:
        ツイートID（失敗時はNone）
    """
    logger.info("週次パフォーマンスツイート投稿開始")

    period = stats.get("period", "今週")
    total_trades = stats.get("total_trades", 0)
    overall_wr = stats.get("overall_win_rate", 0)
    models = stats.get("models", {})

    text = f"【週次AI予測パフォーマンス {period}】\n\n"

    # 各モデルの成績
    for model_name, model_stats in models.items():
        wr = model_stats.get("win_rate", 0)
        trades = model_stats.get("trades", 0)
        text += f"  {model_name}: 勝率{wr * 100:.0f}% ({trades}トレード)\n"

    text += f"\n総トレード: {total_trades}件\n"
    text += f"総合勝率: {overall_wr * 100:.0f}%\n\n"
    text += "※過去の成績は将来の利益を保証しません。\n\n"

    hashtags = _build_hashtag_string(["general", "stock", "fx"])
    text += hashtags

    return _post_tweet(text, dry_run=dry_run)


def schedule_tweets(
    tweets: List[Dict[str, Any]],
    times: Optional[List[str]] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """ツイートをスケジュールして指定時刻に投稿する

    Args:
        tweets: ツイートデータのリスト
            各要素: {"text": str, "type": str}
        times: 投稿時刻のリスト（"HH:MM"形式）。Noneの場合はデフォルト時刻を使用
        dry_run: Trueなら投稿せずにログのみ

    Returns:
        スケジュール結果のリスト
    """
    logger.info(f"ツイートスケジュール登録: {len(tweets)} 件")

    # デフォルトの最適投稿時刻（日本時間）
    if times is None:
        times = ["07:30", "12:15", "18:00", "21:30"]

    # scheduleライブラリを使ったスケジュール設定
    schedule_results = []

    try:
        import schedule as sched_lib
    except ImportError:
        logger.warning("scheduleライブラリが見つかりません。pip install schedule でインストールしてください。")
        logger.info("即時投稿モードで実行します。")

        # 即時投稿フォールバック
        for i, tweet_data in enumerate(tweets):
            text = tweet_data.get("text", "")
            if text:
                result = _post_tweet(text, dry_run=dry_run)
                schedule_results.append({
                    "index": i,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "tweet_id": result,
                    "scheduled_time": "即時投稿",
                })
                time.sleep(MIN_TWEET_INTERVAL_SEC)

        return schedule_results

    # スケジュール登録
    for i, tweet_data in enumerate(tweets):
        tweet_time = times[i % len(times)]
        text = tweet_data.get("text", "")

        if not text:
            continue

        def make_job(t, dr):
            """クロージャでツイートテキストをキャプチャ"""
            def job():
                _post_tweet(t, dry_run=dr)
            return job

        sched_lib.every().day.at(tweet_time).do(make_job(text, dry_run))

        schedule_results.append({
            "index": i,
            "text": text[:50] + "..." if len(text) > 50 else text,
            "scheduled_time": tweet_time,
            "status": "scheduled",
        })
        logger.info(f"スケジュール登録: {tweet_time} - {text[:40]}...")

    # スケジュール保存（参照用）
    schedule_file = TWEET_LOG_DIR / "scheduled_tweets.json"
    schedule_file.write_text(
        json.dumps(schedule_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"スケジュール保存: {schedule_file}")

    return schedule_results


def run_scheduled_loop(duration_minutes: int = 60) -> None:
    """スケジュール済みツイートの実行ループ

    Args:
        duration_minutes: ループの実行時間（分）
    """
    try:
        import schedule as sched_lib
    except ImportError:
        logger.error("scheduleライブラリが必要です。pip install schedule")
        return

    logger.info(f"スケジュールループ開始: {duration_minutes}分間実行")
    end_time = datetime.now() + timedelta(minutes=duration_minutes)

    while datetime.now() < end_time:
        sched_lib.run_pending()
        time.sleep(10)

    logger.info("スケジュールループ終了")


# =============================================================
# CLI エントリーポイント
# =============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Twitter/X 自動投稿ボット")
    parser.add_argument(
        "--action",
        choices=["prediction", "promotion", "performance", "schedule"],
        default="prediction",
        help="実行アクション (default: prediction)",
    )
    parser.add_argument("--dry-run", action="store_true", help="実際には投稿しない")
    parser.add_argument("--market", type=str, default="日本株", help="予測対象市場")
    parser.add_argument("--title", type=str, default="", help="記事タイトル（promotion用）")
    parser.add_argument("--url", type=str, default="", help="記事URL（promotion用）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  Twitter Bot: action={args.action}, dry_run={args.dry_run}")
    logger.info("=" * 60)

    if args.action == "prediction":
        # サンプル予測データ
        sample_data = {
            "market": args.market,
            "direction": "上昇",
            "confidence": 0.68,
            "model": "5モデルアンサンブル",
        }
        result = post_prediction_tweet(sample_data, dry_run=args.dry_run)
        logger.info(f"結果: {result}")

    elif args.action == "promotion":
        if not args.title or not args.url:
            logger.error("--title と --url を指定してください")
        else:
            result = post_note_promotion(args.title, args.url, dry_run=args.dry_run)
            logger.info(f"結果: {result}")

    elif args.action == "performance":
        # サンプルパフォーマンスデータ
        sample_stats = {
            "period": "3/18-3/24",
            "models": {
                "日本株": {"win_rate": 0.65, "trades": 20},
                "FX": {"win_rate": 0.58, "trades": 15},
                "競艇": {"win_rate": 0.52, "trades": 30},
            },
            "total_trades": 65,
            "overall_win_rate": 0.57,
        }
        result = post_weekly_performance(sample_stats, dry_run=args.dry_run)
        logger.info(f"結果: {result}")

    elif args.action == "schedule":
        # サンプルスケジュール
        sample_tweets = [
            {"text": "おはようございます。本日のAI予測をお届けします。 #AI予測 #投資", "type": "morning"},
            {"text": "午後の市場レビューです。 #AI投資 #日経225", "type": "afternoon"},
        ]
        results = schedule_tweets(sample_tweets, dry_run=args.dry_run)
        for r in results:
            logger.info(f"  {r}")
