# CHANGELOG - 競輪AIモデル

## [v0.1] - 2026-03-30
### 追加
- model/data_interface.py（sqlite/csv/mockモード対応）
- model/line_predictor.py（ライン予測エンジン・コメント解析・独自スコア）
- model/feature_engine.py（50特徴量・カテゴリA〜H）
- model/prediction_model.py（3段階予測モデル・Purged K-Fold CV）
- model/betting_logic.py（EV計算・Kelly基準・裏切りリスクフィルター）
- model/evaluation.py（ROI・MDD・Sharpe・ライン予測正答率）
- data/bank_master.py（全43会場のバンクデータ）
- scraper/scraper_historical.py（スタブ・実装待ち）
- scraper/scraper_realtime.py（スタブ・実装待ち）
- scraper/scraper_comment.py（コメント解析ロジックのみ実装）

### 未解決
- スクレイピングコードが未実装（データソース選定中）
- バンクデータの一部パラメータが未検証（Perfecta Naviで確認待ち）
- 裏切りリスク（B06）は実データ取得後に実装
- 地元フラグ（B08, H04）は実データ取得後に実装
- 学習・評価はデータが揃ってから実施
