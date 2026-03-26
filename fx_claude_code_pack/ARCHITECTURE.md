# Architecture

## 1. 全体構造
本システムは以下のレイヤ構成で実装する。

- Layer A: Data
- Layer B: Prediction
- Layer C: Execution
- Layer D: Risk
- Layer E: Reporting / Monetization-ready

初期開発では Layer A〜D を優先し、Layer E はダッシュボードと週次レポート用の最低限構造だけ先に用意する。

## 2. 推奨ディレクトリ構成
```text
fx_ai/
├─ app/
│  └─ dashboard/
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ logs/
├─ models/
├─ notebooks/
├─ reports/
├─ scripts/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ labeling/
│  ├─ models/
│  ├─ validation/
│  ├─ execution/
│  ├─ risk/
│  ├─ reporting/
│  └─ utils/
├─ tests/
├─ .env.example
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

補足:
- 既存リポジトリで作業する場合は、この構成を完全に新設するのではなく、既存構造に合わせて最小変更で寄せる。
- src/ 配下の各ディレクトリには必要に応じて `__init__.py` を作成する。

## 3. レイヤ別責務
### Layer A: Data
責務:
- Phase 1 では OANDA API から価格・スプレッド取得
- Phase 4 以降で FRED / Alpha Vantage / NewsAPI を追加検討
- タイムスタンプ整形
- 欠損・重複処理
- ローカル保存

主な配置:
- src/data/
- data/raw/
- data/interim/

### Layer B: Prediction
責務:
- 特徴量生成
- HMM レジーム判定
- LightGBM 学習と推論
- Triple-Barrier Labeling
- Meta-Labeling

主な配置:
- src/features/
- src/labeling/
- src/models/

### Layer C: Execution
責務:
- 成行前提の執行コスト評価
- スプレッド閾値判定
- 時間帯フィルタ
- 重要指標前後の停止
- No-Trade 判定

主な配置:
- src/execution/

### Layer D: Risk
責務:
- Purged CV
- Walk-Forward
- PBO
- DSR
- 日次損失停止
- 連敗停止
- 異常検知
- 再学習トリガー

主な配置:
- src/validation/
- src/risk/

### Layer E: Reporting / Monetization-ready
責務:
- ダッシュボード表示
- 週次結果集計
- 公開用レポート基礎
- 将来のコンテンツ自動生成への接続余地

主な配置:
- app/dashboard/
- src/reporting/
- reports/

## 4. モジュール設計方針
- 1ファイル1責務を意識する。
- データ取得と学習ロジックを混ぜない。
- 特徴量生成は関数群として分離する。
- 学習器と検証器を分離する。
- バックテストと実運用想定ロジックを混ぜない。
- 設定値は configs/ に寄せる。

## 5. 設定ファイル方針
最低限以下を用意する。
- configs/data.yaml
- configs/model.yaml
- configs/validation.yaml
- configs/execution.yaml
- configs/risk.yaml

## 6. 保存物方針
- 生データ: data/raw/
- 前処理後: data/interim/
- 特徴量済み: data/processed/
- 学習済みモデル: models/
- ログ: logs/
- 評価レポート: reports/

## 7. テスト方針
- utils は単体テストを優先
- 特徴量生成も単体テスト
- データ取得は疎通テスト
- 学習はスモークテスト
- バックテストは最小データで統合テスト

## 8. UI 方針
Streamlit を用いて以下を表示する。
- 価格推移
- レジーム推定結果
- モデル予測結果
- 特徴量重要度
- バックテスト指標
- 停止ログ / 異常検知ログ
