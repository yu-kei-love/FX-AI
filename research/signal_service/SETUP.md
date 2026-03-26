# 競艇AIシグナル配信サービス セットアップガイド

## 1. Telegramチャンネルの作成

### 有料購読者用チャンネルの作成

1. Telegramアプリで「新しいチャンネル」を作成
2. チャンネル名を設定（例: 「競艇AI予測シグナル」）
3. チャンネルを「非公開（Private）」に設定
4. 既存のBotをチャンネルの管理者として追加:
   - チャンネル設定 → 管理者 → 管理者を追加
   - Botのユーザー名を検索して追加
   - 「メッセージを投稿」権限を付与

### チャンネルIDの取得

1. チャンネルにテストメッセージを投稿
2. ブラウザで以下のURLにアクセス:
   ```
   https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
   ```
3. レスポンスの中から `chat.id` を確認（通常 `-100` で始まる数値）
4. または、チャンネルに `@userinfobot` を追加してIDを取得

## 2. 環境変数の設定

`.env` ファイルに以下を追加:

```env
# 既存のBot Token（変更不要）
TELEGRAM_BOT_TOKEN=<既存のトークン>

# 購読者チャンネルID（上記で取得した値）
SIGNAL_CHANNEL_ID=-100XXXXXXXXXX

# 購読パスワード（購読者に共有するコード）
SIGNAL_SUBSCRIPTION_PASSWORD=your_secret_code_here
```

## 3. 依存パッケージのインストール

```bash
pip install python-telegram-bot schedule python-dotenv pandas
```

## 4. サービスの起動

### シグナル配信（スケジュール実行）

```bash
# 自動スケジュール実行（朝9時シグナル、夜9時結果、日曜週次レポート）
python research/signal_service/signal_distributor.py --action schedule

# 時刻をカスタマイズ
python research/signal_service/signal_distributor.py --action schedule \
  --morning-time 08:30 --evening-time 20:30 --weekly-time 21:00

# 手動で朝シグナルを実行
python research/signal_service/signal_distributor.py --action morning

# 手動で結果配信を実行
python research/signal_service/signal_distributor.py --action evening

# 手動で週次レポートを実行
python research/signal_service/signal_distributor.py --action weekly

# テスト（メッセージフォーマット確認）
python research/signal_service/signal_distributor.py --action test

# サービス状態確認
python research/signal_service/signal_distributor.py --action status
```

### 購読管理Bot

```bash
# Botをポーリングモードで起動
python research/signal_service/subscription_manager.py --action bot

# 購読者一覧
python research/signal_service/subscription_manager.py --action list

# 購読者数
python research/signal_service/subscription_manager.py --action count

# 手動で購読者追加
python research/signal_service/subscription_manager.py --action add --user-id 123456789

# 手動で購読者解除
python research/signal_service/subscription_manager.py --action remove --user-id 123456789
```

### 両方を同時に起動する場合

ターミナルを2つ開いて実行:

```bash
# ターミナル1: シグナル配信
python research/signal_service/signal_distributor.py --action schedule

# ターミナル2: 購読管理Bot
python research/signal_service/subscription_manager.py --action bot
```

## 5. 購読パスワードの運用

- 有料プランの購入者にのみパスワードを共有
- 定期的にパスワードを変更する場合は `.env` の `SIGNAL_SUBSCRIPTION_PASSWORD` を更新
- パスワード変更後、既存購読者の再認証は不要（既に登録済み）

## 6. レースデータについて

シグナル生成には当日のレースデータが必要です。
以下のいずれかの形式で配置してください:

- `data/boat/daily/YYYYMMDD.csv` - 当日専用ファイル
- `data/boat/real_race_data.csv` - 全期間データ（当日分を自動抽出）

## 7. データファイル

サービスが自動生成するファイル:

| ファイル | 内容 |
|---------|------|
| `data/signal_service/performance.csv` | 全シグナルの成績記録 |
| `data/signal_service/subscribers.json` | 購読者リスト |
| `data/signal_service/service_state.json` | サービス実行状態 |
| `data/signal_service/daily_signals/` | 日次シグナルのJSON |
| `logs/signal_service.log` | サービスログ |
| `logs/subscription_manager.log` | 購読管理ログ |

## 8. Botコマンド一覧

| コマンド | 説明 | 対象 |
|---------|------|------|
| `/subscribe <パスワード>` | シグナル購読を開始 | 全ユーザー |
| `/unsubscribe` | 購読を解除 | 購読者 |
| `/status` | 購読状態と今月の成績を確認 | 全ユーザー |
| `/help` | 使い方を表示 | 全ユーザー |
| `/subscribers` | 購読者一覧（管理者のみ） | オーナー |
