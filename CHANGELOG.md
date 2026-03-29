# CHANGELOG

## [v0.2] - 2026-03-30
### 追加
- ボートレース：data_interface.py（SQLite / CSV / mockモード対応）
- ボートレース：README_for_sale.md（販売用ドキュメント）
- ボートレース：feature_engine.py に DataInterface 統合

## [v0.1] - 2026-03-30
### 追加
- ボートレース：特徴量エンジン74特徴量（feature_engine.py）
- ボートレース：3段階予測モデル（prediction_model.py）
- ボートレース：EV計算・Kelly基準・買いロジック（betting_logic.py）
- ボートレース：評価フレームワーク（evaluation.py）
- ボートレース：スクレイパーのcourse_takenバグ修正
- FX：レジーム検知（regime_detector.py）
- FX：適応戦略・パラメータ切り替え（adaptive_strategy.py）
- FX：スプレッドコスト管理（cost_manager.py）
- 株：ニュース感情分析3段階パイプライン（news_sentiment_model.py）
- 暗号通貨：統計的アービトラージ（stat_arb_model.py）
- 共通：市場異常検知フィルター（market_filter.py）
- 共通：Kelly基準（kelly.py）
- 共通：評価フレームワーク（evaluation.py）

### 未解決
- ボートレース：データ取得中（2年分 / 約35日予定）
- FX：OOS期間（2025-08-25〜）でモデル崩壊中 → 凍結・原因分析中
- FX：Fold3（PF=0.83）の原因分析が未完了
- 競輪：設計段階・コードなし
