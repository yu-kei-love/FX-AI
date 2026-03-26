# Live Trade Setup Guide / ライブトレード セットアップガイド

## 1. OANDA デモ口座の作成

1. [OANDA fxTrade Practice](https://www.oanda.com/demo-account/) にアクセス
2. 「デモ口座を開設」をクリック
3. 必要情報（名前、メール、国）を入力して登録
4. メール認証を完了
5. ログイン後、ダッシュボードで「口座番号」を確認（例: `101-001-12345678-001`）

## 2. APIキーの取得

1. OANDA fxTradeにログイン
2. 左メニュー「Manage API Access」または「APIアクセスの管理」を選択
3. 「Generate」をクリックしてAPIトークンを生成
4. 表示されたトークンをコピー（一度しか表示されないので注意）

## 3. .env ファイルの設定

プロジェクトルートの `.env` ファイルに以下を設定:

```
OANDA_API_KEY=ここにAPIトークンを貼り付け
OANDA_ACCOUNT_ID=ここに口座番号を貼り付け
OANDA_ENVIRONMENT=practice
```

- `OANDA_ENVIRONMENT` は必ず `practice` のままにしてください（デモ口座）
- `live` に変更するとリアルマネーで取引されます

## 4. 必要なライブラリのインストール

```bash
pip install oandapyV20
```

## 5. まずドライランで確認

ドライランモードでは予測・ログ・Telegram通知は実行されますが、実際の注文は発行されません。

```bash
# 1回だけ実行（注文なし）
python research/live_trade.py --dry-run

# ログを確認
# data/live_trade_logs/trades.csv にトレード記録
# data/live_trade_logs/live_trade.log にシステムログ
```

問題なければループモードでテスト:

```bash
# 1時間ごとに繰り返し（注文なし）
python research/live_trade.py --loop --dry-run
```

## 6. Practice（デモ）環境で実行

ドライランで問題ないことを確認した後:

```bash
# 1回だけ実行（デモ口座で実際に注文）
python research/live_trade.py

# 1時間ごとに繰り返し（デモ口座で自動取引）
python research/live_trade.py --loop
```

## 7. Live（リアル）環境への切替

**十分なデモ取引の実績を確認してから切り替えてください。**

1. `.env` の `OANDA_ENVIRONMENT` を `live` に変更
2. `OANDA_API_KEY` と `OANDA_ACCOUNT_ID` をリアル口座のものに変更
3. 起動時に10秒の確認待ちが入ります

```bash
# リアル環境（10秒のカウントダウン後に開始）
python research/live_trade.py --loop
```

## 安全機能

| 機能 | 説明 |
|------|------|
| ドライランモード | `--dry-run` で注文を発行せずテスト |
| 日次損失リミット | 口座残高の5%を超える損失で自動停止 |
| 最大ポジション数 | 同時に2ポジションまで |
| キルスイッチ | `data/KILL_SWITCH` ファイルを作成すると即時全停止 |
| 週末停止 | 土日は自動的にスキップ |
| ATRベースSL/TP | 全注文にストップロス・テイクプロフィットを自動設定 |
| Live環境警告 | live環境では起動時に10秒の確認待ち |

### キルスイッチの使い方

緊急停止したい場合:

```bash
# 取引を即時停止
touch data/KILL_SWITCH

# 再開する場合は削除
rm data/KILL_SWITCH
```

## ログファイル

- `data/live_trade_logs/trades.csv` - 全トレード記録
- `data/live_trade_logs/live_trade.log` - システムログ（デバッグ用）

## トラブルシューティング

| 問題 | 対処 |
|------|------|
| `oandapyV20がインストールされていません` | `pip install oandapyV20` を実行 |
| `OANDA_API_KEY が未設定` | `.env` にAPIキーを設定 |
| `OANDA_ACCOUNT_ID が未設定` | `.env` に口座番号を設定 |
| 注文が拒否される | OANDA管理画面で口座残高・証拠金を確認 |
| Telegram通知が来ない | `telegram_bot.py` で `/start` を送ってチャットIDを登録 |
