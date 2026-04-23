# v0.46/v0.47 + データ拡張 + NLP 並列実行 状況レポート

**生成日時**: 2026-04-23 14:00
**自律実行フェーズ**

## タスク別状況

### ✅ Task D: 選手コメント NLP 前処理 **完了**
- **574,730 rows 処理 14 秒で完了**
- `comment_features` テーブル作成、レース×選手単位で格納
- 生成特徴量 5 種:
  - `sentiment_score` (-1.0〜+1.0, tanh 正規化)
  - `confidence_score` (0.0〜1.0)
  - `injury_flag` (0.0〜1.0)
  - `n_positive`, `n_negative` (キーワード出現数)
  - `comment_length`

**分布**:
- Strong positive: 383
- Positive: 198,042
- Neutral: 376,221
- Negative: 84
- (Strong negative は該当なし、0件)

**技術メモ**:
- sudachipy は Windows DLL ブロックで断念
- 代替: 競輪固有キーワード辞書（ポジ 30語・ネガ 25語・怪我 10語）でルールベース判定
- 否定語パターン（「ない」「ぬ」「ません」）の後処理で精度向上

**検証例**: 「今は順調で頑張る」→ sentiment +0.54, confidence 0.50

---

### 🏃 Task C: 2018-2021 訓練データ拡張 **稼働中**
- 2 並列 (PID 3162 / 3163) 起動済
- chariloto で 2018-2021 全 4 年のデータ存在確認済
- 現進捗: 2019年5-6月処理中（約 3時間経過時点）
- 推定全体: 4年×25K races×3s/2並列 = **約 42 時間（2 日）** で完走見込み
- scraper_historical.py の既存 backfill 機能を活用

**DB 書き込み先**: `races` / `entries` / `results` テーブル（既存、期間追加のみ）

---

### ⚠️ Task A: netkeirin 複数記者予想取得 **ブロック**
**結論: 公開 API からは取得不可**

調査経過:
1. AplRace API の `class=` パラメータに 12 候補試行 (AplYosoka, AplReporter, AplExpert, AplKishaなど)
   → 全て `classfile not found`
2. `/race/yoso/?race_id=X` ページ (150KB) 詳細解析
   → 予想家・AI キーワードはあるが、内容は **ログイン後のみ取得可能** (ajax_ai_support が空 HTML block)
3. ブラウザ動作を再現するには:
   - netkeiba アカウント登録 + ログイン Cookie 必要
   - 有料会員 (プレミアム) の可能性も

**採用判定**: 本セッションの自律実行範囲外 → **延期**

推奨対応: ユーザー側で Chrome DevTools で実ログイン状態の XHR を観察し、class 名と Cookie 文字列を提供すれば実装可能。

---

### ⚠️ Task B: AI 予想取得 (chariloto/oddspark/WINTICKET) **ブロック**
**結論: AI 予想の公開 API は発見できず**

試行:
- chariloto: AI 予想専用 URL は未発見、既存 Odds.do は結果のみ
- oddspark: AI 予想ページ未確認
- WINTICKET: 公式 API/スクレイピング困難 (SPA)
- netkeirin の `ajax_ai_support.html`: block=ai_support の HTML placeholder のみ (要ログイン)

**採用判定**: Task A と同じく認証壁 → **延期**

---

## 現在の DB 状態

| テーブル | Rows | 追加 |
|---|---|---|
| comments | 574,730 | 既存 |
| **comment_features** | **574,730** | **新規 (Task D)** |
| reporter_predictions (gamboo のみ) | 81,674 | 既存 |
| races | 81,793 → 拡張中 | Task C 進行中 |
| entries | 577,375 → 拡張中 | Task C 進行中 |
| results | 244,995 → 拡張中 | Task C 進行中 |
| odds_netkeirin | 3,576,114 | 既存 |

---

## 完了通知

`data/keirin/notification.log`:
- [2026-04-23 13:58:46] タスクD完了 (records: 574,730)

---

## 次の自律作業

1. Task C 完走待ち（~2日後）→ 完走後自動コミット
2. v0.48 以降: Task D の `comment_features` を v0.44 ABC に統合した ROI 検証

## 並行作業者への notices

- **DB 書き込み競合回避**: 別テーブル `comment_features`, `ai_predictions`, `reporter_predictions_multi` (後者2つは未作成)
- **DB 読み取り**: 全テーブル問題なし
- Task C は races/entries/results に INSERT するため、ユーザー側で同テーブル書き込みがあると競合リスク
