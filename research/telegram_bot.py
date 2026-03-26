# ===========================================
# telegram_bot.py
# Telegram Bot: FX/株の予測通知 + 手動承認
#
# 機能:
#   - 予測シグナルをスマホに通知（理由付き）
#   - 承認/却下ボタン
#   - 承認されたら取引を記録（将来的にAPI発注も可能）
#   - /status でモデルの状態確認
#   - /performance で成績確認
# ===========================================

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# 環境変数読み込み
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
LOG_DIR = DATA_DIR / "paper_trade_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHAT_ID_FILE = DATA_DIR / "telegram_chat_id.txt"


# ===== チャットID管理 =====
def save_chat_id(chat_id):
    """チャットIDを保存"""
    CHAT_ID_FILE.write_text(str(chat_id))


def load_chat_id():
    """保存済みチャットIDを読み込み"""
    if CHAT_ID_FILE.exists():
        return int(CHAT_ID_FILE.read_text().strip())
    return None


# ===== コマンドハンドラ =====
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """初回起動: チャットIDを記録"""
    chat_id = update.effective_chat.id
    save_chat_id(chat_id)
    await update.message.reply_text(
        "✅ FX AI Monitor 起動完了\n\n"
        f"Chat ID: {chat_id} (保存済み)\n\n"
        "予測シグナルが発生したら通知します。\n\n"
        "コマンド:\n"
        "/status - モデルの状態確認\n"
        "/performance - トレード成績\n"
        "/help - ヘルプ"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """モデルの現在の状態を表示"""
    log_file = LOG_DIR / "predictions.csv"

    if not log_file.exists():
        await update.message.reply_text("📊 まだ予測ログがありません")
        return

    import pandas as pd
    logs = pd.read_csv(log_file)
    total = len(logs)
    trades = logs[logs["action"].isin(["BUY", "SELL"])]
    skips = logs[logs["action"] == "SKIP"]

    last = logs.iloc[-1] if total > 0 else None
    text = "📊 モデル状態\n"
    text += "━━━━━━━━━━━━━━━━\n"
    text += f"総予測回数: {total}\n"
    text += f"トレード: {len(trades)}回\n"
    text += f"見送り: {len(skips)}回\n"

    if last is not None:
        text += f"\n最新予測:\n"
        text += f"  時刻: {last.get('timestamp', '?')}\n"
        text += f"  判断: {last.get('action', '?')}\n"
        if last.get("reason"):
            text += f"  理由: {last['reason']}\n"

    await update.message.reply_text(text)


async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """トレード成績を表示"""
    log_file = LOG_DIR / "predictions.csv"

    if not log_file.exists():
        await update.message.reply_text("📈 まだトレード履歴がありません")
        return

    import pandas as pd
    logs = pd.read_csv(log_file)
    completed = logs[logs["result"].isin(["WIN", "LOSE"])]

    if completed.empty:
        await update.message.reply_text("📈 まだ結果が確定したトレードがありません\n（4時間後に自動判定されます）")
        return

    wins = (completed["result"] == "WIN").sum()
    total = len(completed)
    net = completed["net_return"].astype(float).sum()

    text = "📈 トレード成績\n"
    text += "━━━━━━━━━━━━━━━━\n"
    text += f"完了: {total}回\n"
    text += f"勝率: {wins}/{total} ({wins/total*100:.1f}%)\n"
    text += f"累積リターン: {net:+.6f}\n"

    # 直近5トレード
    recent = completed.tail(5)
    text += f"\n直近5トレード:\n"
    for _, row in recent.iterrows():
        emoji = "✅" if row["result"] == "WIN" else "❌"
        text += f"  {emoji} {row.get('timestamp', '?')} {row['action']} → {row['result']}\n"

    await update.message.reply_text(text)


async def cmd_picks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """本日の日本株ピックスを表示"""
    picks_file = DATA_DIR / "japan_stocks" / "daily_picks.csv"

    if not picks_file.exists():
        await update.message.reply_text("📋 本日の株ピックスはまだありません")
        return

    import pandas as pd
    df = pd.read_csv(picks_file)

    if df.empty:
        await update.message.reply_text("📋 本日の株ピックスはまだありません")
        return

    text = "📋 本日の日本株ピックス\n"
    text += "━━━━━━━━━━━━━━━━\n"

    for i, row in df.iterrows():
        direction = row.get("direction", "?")
        dir_emoji = "🔵" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        dir_label = "買い" if direction == "BUY" else "売り" if direction == "SELL" else direction
        confidence = row.get("confidence", 0)
        ticker = row.get("ticker", "?")
        name = row.get("name", "")
        reason = row.get("reason", "")

        text += f"\n{dir_emoji} {ticker}"
        if name:
            text += f" ({name})"
        text += f"\n  方向: {dir_label}\n"
        text += f"  自信度: {confidence:.1%}\n" if isinstance(confidence, float) else f"  自信度: {confidence}\n"
        if reason:
            text += f"  理由: {reason}\n"

    await update.message.reply_text(text)


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """リスク管理の状態を表示"""
    risk_file = DATA_DIR / "paper_trade_logs" / "risk_state.json"

    if not risk_file.exists():
        await update.message.reply_text("⚠️ リスク状態ファイルが見つかりません")
        return

    try:
        with open(risk_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        await update.message.reply_text(f"⚠️ リスク状態の読み込みエラー: {str(e)[:100]}")
        return

    balance = state.get("account_balance", "?")
    drawdown = state.get("drawdown", state.get("current_drawdown", 0))
    losing_streak = state.get("losing_streak", 0)
    open_positions = state.get("open_positions", state.get("num_open_positions", 0))
    daily_pnl = state.get("daily_pnl", 0)
    weekly_pnl = state.get("weekly_pnl", 0)

    text = "⚠️ リスク管理状態\n"
    text += "━━━━━━━━━━━━━━━━\n"
    text += f"口座残高: ¥{balance:,.0f}\n" if isinstance(balance, (int, float)) else f"口座残高: {balance}\n"
    text += f"ドローダウン: {drawdown:.2%}\n" if isinstance(drawdown, float) else f"ドローダウン: {drawdown}\n"
    text += f"連敗数: {losing_streak}回\n"
    text += f"オープンポジション: {open_positions}\n"
    text += f"━━━━━━━━━━━━━━━━\n"
    text += f"日次PnL: ¥{daily_pnl:+,.0f}\n" if isinstance(daily_pnl, (int, float)) else f"日次PnL: {daily_pnl}\n"
    text += f"週次PnL: ¥{weekly_pnl:+,.0f}\n" if isinstance(weekly_pnl, (int, float)) else f"週次PnL: {weekly_pnl}\n"

    await update.message.reply_text(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 FX AI Monitor\n\n"
        "/status - モデルの状態確認\n"
        "/performance - トレード成績\n"
        "/picks - 本日の日本株ピックス\n"
        "/risk - リスク管理状態\n"
        "/help - このヘルプ\n\n"
        "シグナル発生時に自動通知します。\n"
        "承認ボタンで手動確認できます。"
    )


# ===== 承認ボタンのコールバック =====
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """承認/却下ボタンが押された時の処理"""
    query = update.callback_query
    await query.answer()

    data = query.data  # "approve_USDJPY_BUY_20260324_1600" or "reject_..."

    parts = data.split("_", 1)
    action = parts[0]

    if action == "approve":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(f"✅ 承認しました。取引を記録します。\n(※現在はペーパートレードモード)")

        # 承認ログを記録
        approval_log = LOG_DIR / "approvals.csv"
        import pandas as pd
        row = {
            "approved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_id": parts[1] if len(parts) > 1 else "",
            "action": "APPROVED",
        }
        df = pd.DataFrame([row])
        if approval_log.exists():
            df.to_csv(approval_log, mode="a", header=False, index=False)
        else:
            df.to_csv(approval_log, index=False)

    elif action == "reject":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("❌ 却下しました。このシグナルはスキップします。")

    elif action == "hold":
        await query.answer("⏸ 保留中。後でもう一度確認できます。")


# ===== AI会話 =====
# 会話履歴（メモリ内、Bot再起動でリセット）
_chat_history = []

SYSTEM_PROMPT = """あなたはFX・株の予測AIモデルのアシスタントです。
ユーザーはスマホからTelegram経由で話しかけています。短く簡潔に答えてください。

あなたが管理しているシステム:
- USD/JPYの5モデルアンサンブル予測（LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees）
- Walk-Forward検証でPF 1.47（自信度70%以上、5人全員一致時）
- ペーパートレード実行中
- 複数通貨ペアの検証も進行中

回答は日本語で、200文字以内を目安にしてください。"""


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """通常のテキストメッセージをAIに転送"""
    if not ANTHROPIC_KEY:
        await update.message.reply_text("AI機能が設定されていません（APIキーなし）")
        return

    user_text = update.message.text
    chat_id = update.effective_chat.id
    saved_id = load_chat_id()

    # 登録済みユーザーのみ
    if saved_id and chat_id != saved_id:
        return

    # 「入力中...」表示
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # 会話履歴に追加
    _chat_history.append({"role": "user", "content": user_text})
    # 直近10往復のみ保持
    if len(_chat_history) > 20:
        _chat_history[:] = _chat_history[-20:]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=_chat_history,
        )
        reply = response.content[0].text
        _chat_history.append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text(f"AI応答エラー: {str(e)[:100]}")


# ===== シグナル送信関数（外部から呼ぶ用） =====
async def send_signal(bot_token, chat_id, signal):
    """予測シグナルを通知する

    signal: dict with keys:
        pair: "USD/JPY"
        action: "BUY" or "SELL"
        price: 158.500
        confidence: 0.732
        agreement: 5
        reasons: ["5モデル全員一致", "トレンド相場", ...]
        category: "FX" or "株"
    """
    from telegram import Bot

    bot = Bot(token=bot_token)

    # カテゴリ絵文字
    cat_emoji = "💱" if signal.get("category") == "FX" else "📊"
    act_emoji = "🔵" if signal["action"] == "BUY" else "🔴"
    direction = "買い" if signal["action"] == "BUY" else "売り"

    # 理由リスト
    reasons_text = "\n".join(f"  • {r}" for r in signal.get("reasons", []))

    signal_id = f"{signal['pair']}_{signal['action']}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    text = (
        f"{cat_emoji} {signal.get('category', 'FX')} シグナル\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"通貨: {signal['pair']}\n"
        f"判断: {act_emoji} {direction} ({signal['action']})\n"
        f"現在値: {signal['price']:.3f}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"理由:\n{reasons_text}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"自信度: {signal['confidence']:.1%}\n"
        f"一致: {signal.get('agreement', '?')}/5人\n"
    )

    keyboard = [
        [
            InlineKeyboardButton("✅ 承認", callback_data=f"approve_{signal_id}"),
            InlineKeyboardButton("❌ 却下", callback_data=f"reject_{signal_id}"),
            InlineKeyboardButton("⏸ 保留", callback_data=f"hold_{signal_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


def send_signal_sync(signal):
    """同期版: paper_trade.pyから呼ぶ用"""
    bot_token = BOT_TOKEN
    chat_id = load_chat_id()
    if not bot_token or not chat_id:
        print("Telegram未設定（スキップ）")
        return
    asyncio.run(send_signal(bot_token, chat_id, signal))


# ===== Bot起動 =====
def run_bot():
    """Telegram Botを起動（常駐）"""
    if not BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN が .env に設定されていません")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("picks", cmd_picks))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram Bot 起動中...")
    print("Ctrl+C で停止")
    app.run_polling()


if __name__ == "__main__":
    run_bot()
