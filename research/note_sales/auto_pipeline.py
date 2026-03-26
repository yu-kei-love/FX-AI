# ===========================================
# auto_pipeline.py
# note.com コンテンツ自動化パイプライン（メインオーケストレーター）
#
# 機能:
#   - 日次パイプライン: モデル結果チェック → 記事生成 → ドラフト保存
#   - 週次レビュー: パフォーマンス分析 → レポート生成
#   - ドラフト承認ワークフロー: pending → approved → ツイート通知
#   - 売上データの記録と分析
#   - スケジュール実行（configurable intervals）
#
# パイプライン:
#   a. 新しいモデル結果をチェック
#   b. 結果に基づいて記事ドラフトを生成
#   c. ドラフトをpending/に保存（ユーザーレビュー待ち）
#   d. ユーザー承認後（approved/に移動）、ツイートで告知
#   e. 売上データの記録
#   f. 週次分析レポート生成
# ===========================================

import sys
import os
import json
import shutil
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

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
        logging.FileHandler(str(LOG_DIR / "note_pipeline.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# パス定義
NOTE_SALES_DIR = PROJECT_ROOT / "research" / "note_sales"
DRAFTS_DIR = NOTE_SALES_DIR / "drafts"
PENDING_DIR = DRAFTS_DIR / "pending"
APPROVED_DIR = DRAFTS_DIR / "approved"
DATA_DIR = PROJECT_ROOT / "data"
NOTE_DATA_DIR = DATA_DIR / "note_sales"
PIPELINE_STATE_FILE = NOTE_DATA_DIR / "pipeline_state.json"

# ディレクトリ作成
for d in [PENDING_DIR, APPROVED_DIR, NOTE_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# パイプライン設定
DEFAULT_CONFIG = {
    "daily_article_types": ["daily_report"],  # 日次で生成する記事タイプ
    "free_article_interval_days": 3,          # 無料記事の生成間隔（日）
    "paid_article_interval_days": 7,          # 有料記事の生成間隔（日）
    "auto_tweet_on_approve": True,            # 承認時に自動ツイートするか
    "tweet_dry_run": True,                    # ツイートのドライラン（テスト用）
    "default_free_topics": [
        "AIで日本株を予測してみた結果",
        "Pythonで作るFX自動売買の基礎",
        "競艇AIモデルの検証記録",
        "機械学習アンサンブルの威力を検証",
        "Walk-Forward検証で分かった本当のモデル性能",
        "暗号通貨AI予測は儲かるのか？データで検証",
    ],
    "default_paid_topics": [
        "【完全版】LightGBMで株価予測モデルを構築する方法",
        "【コード公開】FXアンサンブルモデルの実装手順",
        "【実践】競艇AI予測モデルの作り方（全コード付き）",
        "Walk-Forward検証の正しい実装方法【Python】",
        "Kelly基準でベットサイジングを最適化する【理論と実装】",
    ],
}


# ===== Telegram送信 =====
def _send_telegram(text: str, parse_mode: str = None) -> bool:
    """Telegramにメッセージを送信する"""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id_file = PROJECT_ROOT / "data" / "telegram_chat_id.txt"
        if not bot_token or not chat_id_file.exists():
            logger.warning("Telegram設定なし（BOT_TOKENまたはchat_id未設定）")
            return False
        chat_id = int(chat_id_file.read_text().strip())

        import telegram
        async def _send():
            bot = telegram.Bot(token=bot_token)
            # Telegramは4096文字制限。長い場合は分割送信
            max_len = 4000
            for i in range(0, len(text), max_len):
                chunk = text[i:i + max_len]
                await bot.send_message(
                    chat_id=chat_id, text=chunk, parse_mode=parse_mode
                )
        asyncio.run(_send())
        return True
    except Exception as e:
        logger.warning(f"Telegram送信エラー: {e}")
        return False


def send_draft_to_telegram(draft_path: str) -> bool:
    """記事ドラフトをTelegramに送信してレビューを依頼する"""
    path = Path(draft_path)
    if not path.exists():
        logger.error(f"ファイルが見つかりません: {draft_path}")
        return False

    content = path.read_text(encoding="utf-8")
    filename = path.name

    # メタデータがあれば価格情報を取得
    price_info = "無料"
    meta_candidates = [
        path.with_suffix(".meta.json"),
        path.with_name(path.stem + "_meta.json"),
        path.with_name(path.stem + ".json"),
    ]
    for meta_path in meta_candidates:
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                price = meta.get("price", 0)
                if price > 0:
                    price_info = f"{price}円"
            except Exception:
                pass
            break

    header = (
        f"--- note記事レビュー依頼 ---\n"
        f"ファイル: {filename}\n"
        f"価格: {price_info}\n"
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"---\n\n"
    )

    full_text = header + content
    logger.info(f"Telegram送信: {filename}")
    return _send_telegram(full_text)


def send_all_pending_to_telegram() -> int:
    """全ての承認待ちドラフトをTelegramに送信する"""
    count = 0
    for md_file in sorted(PENDING_DIR.glob("*.md")):
        if send_draft_to_telegram(str(md_file)):
            count += 1
    logger.info(f"Telegram送信完了: {count}件")
    return count


def _load_pipeline_state() -> Dict[str, Any]:
    """パイプラインの状態を読み込む"""
    if PIPELINE_STATE_FILE.exists():
        try:
            return json.loads(PIPELINE_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {
        "last_daily_run": None,
        "last_weekly_run": None,
        "last_free_article": None,
        "last_paid_article": None,
        "free_topic_index": 0,
        "paid_topic_index": 0,
        "total_articles_generated": 0,
        "total_articles_approved": 0,
    }


def _save_pipeline_state(state: Dict[str, Any]) -> None:
    """パイプラインの状態を保存する"""
    PIPELINE_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _check_model_results() -> Dict[str, Any]:
    """新しいモデル結果があるかチェックする

    Returns:
        利用可能なモデル結果の辞書
    """
    logger.info("モデル結果チェック中...")
    results = {}

    # FXダッシュボード状態
    fx_path = DATA_DIR / "dashboard_state.joblib"
    if fx_path.exists():
        mod_time = datetime.fromtimestamp(fx_path.stat().st_mtime)
        results["fx"] = {
            "available": True,
            "last_modified": mod_time.isoformat(),
            "path": str(fx_path),
        }
        logger.info(f"  FXモデル: 利用可能 (更新: {mod_time.strftime('%Y-%m-%d %H:%M')})")

    # 日本株スクリーナー
    stock_paths = [
        DATA_DIR / "stock_screener" / "screening_report.json",
        DATA_DIR / "japan_stocks",
    ]
    for sp in stock_paths:
        if sp.exists():
            mod_time = datetime.fromtimestamp(sp.stat().st_mtime)
            results["stocks"] = {
                "available": True,
                "last_modified": mod_time.isoformat(),
                "path": str(sp),
            }
            logger.info(f"  日本株モデル: 利用可能 (更新: {mod_time.strftime('%Y-%m-%d %H:%M')})")
            break

    # 競艇モデル
    boat_paths = [
        DATA_DIR / "boat" / "paper_trade_log.json",
        DATA_DIR / "boat" / "wf_results.json",
    ]
    for bp in boat_paths:
        if bp.exists():
            mod_time = datetime.fromtimestamp(bp.stat().st_mtime)
            results["boat"] = {
                "available": True,
                "last_modified": mod_time.isoformat(),
                "path": str(bp),
            }
            logger.info(f"  競艇モデル: 利用可能 (更新: {mod_time.strftime('%Y-%m-%d %H:%M')})")
            break

    # 暗号通貨モデル
    crypto_path = DATA_DIR / "crypto" / "crypto_model_report.txt"
    if crypto_path.exists():
        mod_time = datetime.fromtimestamp(crypto_path.stat().st_mtime)
        results["crypto"] = {
            "available": True,
            "last_modified": mod_time.isoformat(),
            "path": str(crypto_path),
        }
        logger.info(f"  暗号通貨モデル: 利用可能 (更新: {mod_time.strftime('%Y-%m-%d %H:%M')})")

    # マルチ通貨レポート
    multi_path = DATA_DIR / "multi_currency_report.txt"
    if multi_path.exists():
        mod_time = datetime.fromtimestamp(multi_path.stat().st_mtime)
        results["multi_currency"] = {
            "available": True,
            "last_modified": mod_time.isoformat(),
            "path": str(multi_path),
        }
        logger.info(f"  マルチ通貨レポート: 利用可能 (更新: {mod_time.strftime('%Y-%m-%d %H:%M')})")

    if not results:
        logger.warning("利用可能なモデル結果が見つかりませんでした")

    return results


def run_daily_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """日次パイプラインを実行する

    朝の実行:
    1. モデル結果のチェック
    2. 日次レポートの生成
    3. 定期的に無料/有料記事も生成

    Args:
        config: パイプライン設定（Noneの場合はデフォルト）

    Returns:
        パイプライン実行結果
    """
    if config is None:
        config = DEFAULT_CONFIG

    logger.info("=" * 70)
    logger.info("  日次パイプライン開始")
    logger.info(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    state = _load_pipeline_state()
    result = {
        "timestamp": datetime.now().isoformat(),
        "articles_generated": [],
        "errors": [],
    }

    # Step 1: モデル結果チェック
    logger.info("[Step 1/3] モデル結果チェック")
    model_results = _check_model_results()
    result["model_results_available"] = list(model_results.keys())

    # Step 2: 記事生成
    logger.info("[Step 2/3] 記事生成")

    try:
        from research.note_sales.note_content_generator import (
            generate_daily_report,
            generate_free_article,
            generate_paid_article,
        )

        # 日次レポート（毎日生成）
        if "daily_report" in config.get("daily_article_types", []):
            try:
                path = generate_daily_report()
                result["articles_generated"].append({"type": "daily_report", "path": path})
                state["total_articles_generated"] = state.get("total_articles_generated", 0) + 1
                logger.info(f"日次レポート生成完了: {path}")
            except Exception as e:
                error_msg = f"日次レポート生成エラー: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

        # 無料記事（間隔に応じて生成）
        free_interval = config.get("free_article_interval_days", 3)
        last_free = state.get("last_free_article")
        should_generate_free = True
        if last_free:
            try:
                last_free_dt = datetime.fromisoformat(last_free)
                if (datetime.now() - last_free_dt).days < free_interval:
                    should_generate_free = False
                    logger.info(f"無料記事: 前回から{(datetime.now() - last_free_dt).days}日（間隔: {free_interval}日）→ スキップ")
            except (ValueError, TypeError):
                pass

        if should_generate_free:
            topics = config.get("default_free_topics", DEFAULT_CONFIG["default_free_topics"])
            idx = state.get("free_topic_index", 0) % len(topics)
            topic = topics[idx]
            try:
                path = generate_free_article(topic)
                result["articles_generated"].append({"type": "free", "path": path, "topic": topic})
                state["last_free_article"] = datetime.now().isoformat()
                state["free_topic_index"] = idx + 1
                state["total_articles_generated"] = state.get("total_articles_generated", 0) + 1
                logger.info(f"無料記事生成完了: {path}")
            except Exception as e:
                error_msg = f"無料記事生成エラー: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

        # 有料記事（間隔に応じて生成）
        paid_interval = config.get("paid_article_interval_days", 7)
        last_paid = state.get("last_paid_article")
        should_generate_paid = True
        if last_paid:
            try:
                last_paid_dt = datetime.fromisoformat(last_paid)
                if (datetime.now() - last_paid_dt).days < paid_interval:
                    should_generate_paid = False
                    logger.info(f"有料記事: 前回から{(datetime.now() - last_paid_dt).days}日（間隔: {paid_interval}日）→ スキップ")
            except (ValueError, TypeError):
                pass

        if should_generate_paid:
            topics = config.get("default_paid_topics", DEFAULT_CONFIG["default_paid_topics"])
            idx = state.get("paid_topic_index", 0) % len(topics)
            topic = topics[idx]
            try:
                path = generate_paid_article(topic)
                result["articles_generated"].append({"type": "paid", "path": path, "topic": topic})
                state["last_paid_article"] = datetime.now().isoformat()
                state["paid_topic_index"] = idx + 1
                state["total_articles_generated"] = state.get("total_articles_generated", 0) + 1
                logger.info(f"有料記事生成完了: {path}")
            except Exception as e:
                error_msg = f"有料記事生成エラー: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

    except ImportError as e:
        error_msg = f"コンテンツジェネレーターのインポートエラー: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)

    # Step 3: 状態保存
    logger.info("[Step 3/3] 状態保存")
    state["last_daily_run"] = datetime.now().isoformat()
    _save_pipeline_state(state)

    # サマリー出力
    logger.info("-" * 50)
    logger.info(f"日次パイプライン完了:")
    logger.info(f"  生成記事数: {len(result['articles_generated'])}")
    logger.info(f"  エラー数: {len(result['errors'])}")
    logger.info(f"  利用可能モデル: {', '.join(result['model_results_available'])}")
    logger.info("-" * 50)

    return result


def run_weekly_review() -> Dict[str, Any]:
    """週次パフォーマンスレビューを実行する

    1. 売上分析レポート生成
    2. タイトルパターン分析
    3. 週次パフォーマンスツイート

    Returns:
        週次レビュー結果
    """
    logger.info("=" * 70)
    logger.info("  週次レビュー開始")
    logger.info(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    state = _load_pipeline_state()
    result = {
        "timestamp": datetime.now().isoformat(),
        "reports": [],
        "errors": [],
    }

    # 売上レポート生成
    try:
        from research.note_sales.sales_tracker import (
            generate_sales_report,
            analyze_by_hour,
            analyze_by_title_pattern,
            analyze_by_price,
        )

        report_path = generate_sales_report()
        result["reports"].append({"type": "sales_report", "path": report_path})
        logger.info(f"売上レポート生成完了: {report_path}")

        # 分析実行
        hourly_analysis = analyze_by_hour()
        result["hourly_analysis"] = hourly_analysis

        pattern_analysis = analyze_by_title_pattern()
        result["pattern_analysis"] = pattern_analysis

        price_analysis = analyze_by_price()
        result["price_analysis"] = price_analysis

    except ImportError as e:
        error_msg = f"売上トラッカーのインポートエラー: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
    except Exception as e:
        error_msg = f"売上分析エラー: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)

    # タイトルリサーチ更新
    try:
        from research.note_sales.title_researcher import research_trending_titles
        research_trending_titles()
        logger.info("タイトルリサーチ更新完了")
    except Exception as e:
        logger.warning(f"タイトルリサーチ更新エラー: {e}")

    # 状態保存
    state["last_weekly_run"] = datetime.now().isoformat()
    _save_pipeline_state(state)

    logger.info("-" * 50)
    logger.info(f"週次レビュー完了:")
    logger.info(f"  生成レポート数: {len(result['reports'])}")
    logger.info(f"  エラー数: {len(result['errors'])}")
    logger.info("-" * 50)

    return result


def list_pending_drafts() -> List[Dict[str, Any]]:
    """承認待ちのドラフト一覧を表示する

    Returns:
        ドラフト情報のリスト
    """
    logger.info("承認待ちドラフト一覧取得")
    drafts = []

    for meta_file in sorted(PENDING_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            # 対応する本文ファイルの存在チェック
            content_file = PENDING_DIR / meta.get("filename", "")
            meta["content_exists"] = content_file.exists()
            meta["file_size"] = content_file.stat().st_size if content_file.exists() else 0
            drafts.append(meta)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"メタデータ読み込みエラー: {meta_file} - {e}")

    logger.info(f"承認待ちドラフト: {len(drafts)} 件")
    for d in drafts:
        logger.info(f"  [{d.get('draft_id', '?')}] {d.get('article_type', '?')} - {d.get('created_at', '?')}")

    return drafts


def approve_draft(
    draft_id: str,
    auto_tweet: bool = True,
    tweet_dry_run: bool = True,
    article_url: Optional[str] = None,
) -> Dict[str, Any]:
    """ドラフトを承認して approved/ に移動する

    Args:
        draft_id: ドラフトID
        auto_tweet: 承認時にツイートで告知するか
        tweet_dry_run: ツイートのドライラン
        article_url: 記事のURL（ツイート用）

    Returns:
        承認結果
    """
    logger.info(f"ドラフト承認: {draft_id}")

    # メタデータファイルを探す
    meta_files = list(PENDING_DIR.glob(f"{draft_id}_*_meta.json"))
    if not meta_files:
        logger.error(f"ドラフトが見つかりません: {draft_id}")
        return {"error": f"ドラフトが見つかりません: {draft_id}"}

    meta_file = meta_files[0]
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    content_filename = meta.get("filename", "")
    content_file = PENDING_DIR / content_filename

    if not content_file.exists():
        logger.error(f"コンテンツファイルが見つかりません: {content_file}")
        return {"error": f"コンテンツファイルが見つかりません"}

    # approved/ に移動
    approved_content = APPROVED_DIR / content_filename
    approved_meta = APPROVED_DIR / meta_file.name
    shutil.move(str(content_file), str(approved_content))
    shutil.move(str(meta_file), str(approved_meta))

    # メタデータ更新
    meta["status"] = "approved"
    meta["approved_at"] = datetime.now().isoformat()
    approved_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"ドラフトを承認しました: {draft_id}")
    logger.info(f"  移動先: {approved_content}")

    # 状態更新
    state = _load_pipeline_state()
    state["total_articles_approved"] = state.get("total_articles_approved", 0) + 1
    _save_pipeline_state(state)

    result = {
        "draft_id": draft_id,
        "status": "approved",
        "content_path": str(approved_content),
        "approved_at": meta["approved_at"],
    }

    # 自動ツイート
    if auto_tweet and article_url:
        try:
            from research.note_sales.twitter_bot import post_note_promotion
            tweet_id = post_note_promotion(
                article_title=content_filename.replace(".md", ""),
                url=article_url,
                dry_run=tweet_dry_run,
            )
            result["tweet_id"] = tweet_id
            logger.info(f"プロモーションツイート投稿: {tweet_id}")
        except Exception as e:
            logger.warning(f"ツイート投稿エラー: {e}")
            result["tweet_error"] = str(e)

    return result


def run_scheduled(
    daily_time: str = "07:00",
    weekly_day: str = "monday",
    weekly_time: str = "09:00",
) -> None:
    """スケジュール実行を開始する

    Args:
        daily_time: 日次パイプラインの実行時刻（HH:MM）
        weekly_day: 週次レビューの実行曜日
        weekly_time: 週次レビューの実行時刻（HH:MM）
    """
    try:
        import schedule
    except ImportError:
        logger.error("scheduleライブラリが必要です: pip install schedule")
        return

    import time

    logger.info("=" * 70)
    logger.info("  スケジュール実行モード開始")
    logger.info(f"  日次: 毎日 {daily_time}")
    logger.info(f"  週次: 毎週{weekly_day} {weekly_time}")
    logger.info("=" * 70)

    # 日次パイプライン
    schedule.every().day.at(daily_time).do(run_daily_pipeline)

    # 週次レビュー
    weekday_map = {
        "monday": schedule.every().monday,
        "tuesday": schedule.every().tuesday,
        "wednesday": schedule.every().wednesday,
        "thursday": schedule.every().thursday,
        "friday": schedule.every().friday,
        "saturday": schedule.every().saturday,
        "sunday": schedule.every().sunday,
    }
    weekly_scheduler = weekday_map.get(weekly_day.lower(), schedule.every().monday)
    weekly_scheduler.at(weekly_time).do(run_weekly_review)

    logger.info("スケジュール登録完了。Ctrl+Cで終了します。")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("スケジュール実行を終了しました。")


# =============================================================
# CLI エントリーポイント
# =============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="note.com コンテンツ自動化パイプライン")
    parser.add_argument(
        "--action",
        choices=["daily", "weekly", "list", "approve", "schedule"],
        default="daily",
        help="実行アクション (default: daily)",
    )
    parser.add_argument("--draft-id", type=str, default="", help="承認するドラフトID")
    parser.add_argument("--url", type=str, default="", help="記事URL（承認＋ツイート用）")
    parser.add_argument("--daily-time", type=str, default="07:00", help="日次実行時刻")
    parser.add_argument("--weekly-day", type=str, default="monday", help="週次実行曜日")
    parser.add_argument("--weekly-time", type=str, default="09:00", help="週次実行時刻")
    parser.add_argument("--dry-run", action="store_true", help="ツイートのドライラン")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  note.com パイプライン: action={args.action}")
    logger.info("=" * 60)

    if args.action == "daily":
        result = run_daily_pipeline()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "weekly":
        result = run_weekly_review()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.action == "list":
        drafts = list_pending_drafts()
        if drafts:
            print(f"\n承認待ちドラフト: {len(drafts)} 件\n")
            for d in drafts:
                print(f"  ID: {d.get('draft_id', '?')}")
                print(f"  タイプ: {d.get('article_type', '?')}")
                print(f"  作成日: {d.get('created_at', '?')}")
                print(f"  ファイル: {d.get('filename', '?')}")
                print()
        else:
            print("承認待ちのドラフトはありません。")

    elif args.action == "approve":
        if not args.draft_id:
            logger.error("--draft-id を指定してください")
            # 利用可能なドラフトを表示
            drafts = list_pending_drafts()
            if drafts:
                print("\n利用可能なドラフトID:")
                for d in drafts:
                    print(f"  {d.get('draft_id', '?')}")
        else:
            result = approve_draft(
                args.draft_id,
                auto_tweet=bool(args.url),
                tweet_dry_run=args.dry_run,
                article_url=args.url if args.url else None,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.action == "schedule":
        run_scheduled(
            daily_time=args.daily_time,
            weekly_day=args.weekly_day,
            weekly_time=args.weekly_time,
        )
