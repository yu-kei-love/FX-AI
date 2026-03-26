# ===========================================
# subscription_manager.py
# 競艇AIシグナル - 購読者管理
#
# 機能:
#   - /subscribe <パスワード> で購読開始
#   - /unsubscribe で購読解除
#   - /status で自分の購読状態確認
#   - パスワード認証によるアクセス制御
#   - 購読者リストの管理
#   - Telegram Botのポーリング実行
# ===========================================

import sys
import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
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
        logging.FileHandler(str(LOG_DIR / "subscription_manager.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# パス定義
DATA_DIR = PROJECT_ROOT / "data" / "signal_service"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# 購読者データ管理
# =============================================================

def _load_subscribers() -> Dict[str, Any]:
    """購読者データを読み込む"""
    if SUBSCRIBERS_FILE.exists():
        try:
            return json.loads(SUBSCRIBERS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {"subscribers": {}}


def _save_subscribers(data: Dict[str, Any]) -> None:
    """購読者データを保存する"""
    SUBSCRIBERS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_subscription_password() -> str:
    """購読パスワードを取得する"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    password = os.getenv("SIGNAL_SUBSCRIPTION_PASSWORD", "")
    if not password:
        logger.warning("SIGNAL_SUBSCRIPTION_PASSWORDが設定されていません。.envに追加してください。")
    return password


def add_subscriber(
    user_id: int,
    username: str = "",
    first_name: str = "",
    last_name: str = "",
) -> Dict[str, Any]:
    """購読者を追加する"""
    data = _load_subscribers()
    user_key = str(user_id)

    if user_key in data["subscribers"]:
        # 既存購読者の再有効化
        data["subscribers"][user_key]["active"] = True
        data["subscribers"][user_key]["reactivated_at"] = datetime.now().isoformat()
        logger.info(f"購読者再有効化: {user_id} ({username})")
    else:
        data["subscribers"][user_key] = {
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "active": True,
            "subscribed_at": datetime.now().isoformat(),
            "unsubscribed_at": None,
            "reactivated_at": None,
        }
        logger.info(f"新規購読者追加: {user_id} ({username})")

    _save_subscribers(data)
    return data["subscribers"][user_key]


def remove_subscriber(user_id: int) -> bool:
    """購読者を無効化する（データは保持）"""
    data = _load_subscribers()
    user_key = str(user_id)

    if user_key not in data["subscribers"]:
        return False

    data["subscribers"][user_key]["active"] = False
    data["subscribers"][user_key]["unsubscribed_at"] = datetime.now().isoformat()
    _save_subscribers(data)
    logger.info(f"購読者解除: {user_id}")
    return True


def get_subscriber(user_id: int) -> Optional[Dict[str, Any]]:
    """購読者情報を取得する"""
    data = _load_subscribers()
    return data["subscribers"].get(str(user_id))


def list_subscribers(active_only: bool = True) -> List[Dict[str, Any]]:
    """購読者一覧を取得する"""
    data = _load_subscribers()
    subscribers = list(data["subscribers"].values())
    if active_only:
        subscribers = [s for s in subscribers if s.get("active", False)]
    return subscribers


def get_subscriber_count() -> Dict[str, int]:
    """購読者数を取得する"""
    data = _load_subscribers()
    all_subs = list(data["subscribers"].values())
    active = sum(1 for s in all_subs if s.get("active", False))
    inactive = len(all_subs) - active
    return {"active": active, "inactive": inactive, "total": len(all_subs)}


# =============================================================
# Telegram Bot コマンドハンドラ
# =============================================================

async def handle_subscribe(update, context) -> None:
    """
    /subscribe <パスワード> コマンドの処理。
    正しいパスワードで購読を開始する。
    """
    user = update.effective_user
    args = context.args

    if not args:
        await update.message.reply_text(
            "⚠️ パスワードを入力してください。\n"
            "使い方: /subscribe <パスワード>"
        )
        return

    password = args[0]
    correct_password = _get_subscription_password()

    if not correct_password:
        await update.message.reply_text(
            "⚠️ サービスのパスワードが未設定です。管理者にお問い合わせください。"
        )
        return

    if password != correct_password:
        await update.message.reply_text(
            "❌ パスワードが正しくありません。\n"
            "正しいパスワードを入力してください。"
        )
        logger.warning(f"認証失敗: user_id={user.id}, username={user.username}")
        return

    # 購読追加
    subscriber = add_subscriber(
        user_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
    )

    await update.message.reply_text(
        f"✅ 購読を開始しました！\n\n"
        f"ユーザー: {user.first_name}\n"
        f"登録日: {subscriber.get('subscribed_at', '')[:10]}\n\n"
        f"毎朝、競艇AIの予測シグナルが配信されます。\n"
        f"購読を解除するには /unsubscribe を送信してください。"
    )

    # オーナーに通知
    from research.signal_service.signal_distributor import send_to_owner
    counts = get_subscriber_count()
    send_to_owner(
        f"📢 新規購読者\n"
        f"ユーザー: {user.first_name} (@{user.username})\n"
        f"ID: {user.id}\n"
        f"現在の購読者数: {counts['active']}名"
    )


async def handle_unsubscribe(update, context) -> None:
    """/unsubscribe コマンドの処理。購読を解除する。"""
    user = update.effective_user

    subscriber = get_subscriber(user.id)
    if not subscriber or not subscriber.get("active", False):
        await update.message.reply_text(
            "⚠️ 現在購読していません。"
        )
        return

    remove_subscriber(user.id)

    await update.message.reply_text(
        f"✅ 購読を解除しました。\n\n"
        f"ご利用ありがとうございました。\n"
        f"再度購読するには /subscribe <パスワード> を送信してください。"
    )

    # オーナーに通知
    from research.signal_service.signal_distributor import send_to_owner
    counts = get_subscriber_count()
    send_to_owner(
        f"📢 購読解除\n"
        f"ユーザー: {user.first_name} (@{user.username})\n"
        f"ID: {user.id}\n"
        f"現在の購読者数: {counts['active']}名"
    )


async def handle_status(update, context) -> None:
    """/status コマンドの処理。購読状態を確認する。"""
    user = update.effective_user
    subscriber = get_subscriber(user.id)

    if not subscriber:
        await update.message.reply_text(
            "⚠️ 購読登録がありません。\n"
            "/subscribe <パスワード> で購読を開始してください。"
        )
        return

    active = subscriber.get("active", False)
    status_text = "✅ アクティブ" if active else "❌ 解除済み"
    subscribed_at = subscriber.get("subscribed_at", "不明")[:10]

    # パフォーマンス情報
    from research.signal_service.signal_distributor import get_performance_summary
    monthly = get_performance_summary(period="monthly")

    msg = (
        f"📋 購読状態\n\n"
        f"ステータス: {status_text}\n"
        f"登録日: {subscribed_at}\n\n"
    )

    if monthly["total_bets"] > 0:
        msg += (
            f"📊 今月のシグナル成績\n"
            f"  ベット数: {monthly['total_bets']}\n"
            f"  的中率: {monthly['win_rate']:.1%}\n"
            f"  損益: {monthly['net_profit']:+,.0f}円\n"
            f"  PF: {monthly['pf']:.2f}\n"
        )

    await update.message.reply_text(msg)


async def handle_help(update, context) -> None:
    """/help コマンドの処理。使い方を表示する。"""
    await update.message.reply_text(
        "🏁 競艇AI予測シグナルBot\n\n"
        "【コマンド一覧】\n"
        "/subscribe <パスワード> - シグナル購読を開始\n"
        "/unsubscribe - 購読を解除\n"
        "/status - 購読状態を確認\n"
        "/help - この使い方を表示\n\n"
        "【配信スケジュール】\n"
        "  朝 09:00 - 当日の予測シグナル\n"
        "  夜 21:00 - 当日の結果サマリー\n"
        "  日曜 21:30 - 週次レポート\n\n"
        "ご質問は管理者にお問い合わせください。"
    )


async def handle_subscribers_list(update, context) -> None:
    """
    /subscribers コマンドの処理（オーナーのみ）。
    購読者一覧を表示する。
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    chat_id_file = PROJECT_ROOT / "data" / "telegram_chat_id.txt"

    owner_chat_id = None
    if chat_id_file.exists():
        try:
            owner_chat_id = int(chat_id_file.read_text().strip())
        except (ValueError, TypeError):
            pass

    user = update.effective_user
    if owner_chat_id and user.id != owner_chat_id:
        await update.message.reply_text("⚠️ このコマンドは管理者のみ利用できます。")
        return

    subs = list_subscribers(active_only=False)
    counts = get_subscriber_count()

    if not subs:
        await update.message.reply_text("購読者はいません。")
        return

    lines = [
        f"📋 購読者一覧 (アクティブ: {counts['active']}名 / 全体: {counts['total']}名)",
        "",
    ]

    for s in subs:
        status = "✅" if s.get("active") else "❌"
        name = s.get("first_name", "")
        username = s.get("username", "")
        user_id = s.get("user_id", "")
        sub_date = s.get("subscribed_at", "")[:10]
        lines.append(f"  {status} {name} (@{username}) [ID: {user_id}] - {sub_date}")

    await update.message.reply_text("\n".join(lines))


# =============================================================
# Bot実行
# =============================================================

def run_bot() -> None:
    """
    Telegram Botをポーリングモードで実行する。
    購読管理コマンドを処理する。
    """
    try:
        from telegram.ext import Application, CommandHandler
    except ImportError:
        logger.error("python-telegram-botが必要です: pip install python-telegram-bot")
        return

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKENが.envに設定されていません")
        return

    logger.info("=" * 60)
    logger.info("  購読管理Bot起動")
    logger.info("=" * 60)

    app = Application.builder().token(bot_token).build()

    # コマンドハンドラ登録
    app.add_handler(CommandHandler("subscribe", handle_subscribe))
    app.add_handler(CommandHandler("unsubscribe", handle_unsubscribe))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("start", handle_help))
    app.add_handler(CommandHandler("subscribers", handle_subscribers_list))

    logger.info("Botポーリング開始。Ctrl+Cで終了します。")
    app.run_polling()


# =============================================================
# CLIエントリーポイント
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="競艇AIシグナル - 購読者管理")
    parser.add_argument(
        "--action",
        choices=["bot", "list", "count", "add", "remove"],
        default="bot",
        help="実行アクション (default: bot)",
    )
    parser.add_argument("--user-id", type=int, default=0, help="ユーザーID")
    parser.add_argument("--username", type=str, default="", help="ユーザー名")
    args = parser.parse_args()

    if args.action == "bot":
        run_bot()

    elif args.action == "list":
        subs = list_subscribers(active_only=False)
        counts = get_subscriber_count()
        print(f"購読者数: アクティブ {counts['active']}名 / 全体 {counts['total']}名")
        for s in subs:
            status = "ACTIVE" if s.get("active") else "INACTIVE"
            print(f"  [{status}] {s.get('first_name', '')} (@{s.get('username', '')}) ID: {s.get('user_id', '')}")

    elif args.action == "count":
        counts = get_subscriber_count()
        print(json.dumps(counts, indent=2))

    elif args.action == "add":
        if args.user_id:
            sub = add_subscriber(args.user_id, username=args.username)
            print(f"購読者追加: {json.dumps(sub, ensure_ascii=False, indent=2)}")
        else:
            print("--user-id を指定してください")

    elif args.action == "remove":
        if args.user_id:
            ok = remove_subscriber(args.user_id)
            print(f"購読者解除: {'成功' if ok else '失敗（登録なし）'}")
        else:
            print("--user-id を指定してください")
