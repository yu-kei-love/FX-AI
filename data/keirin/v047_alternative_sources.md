# v0.47 AI予想 代替ソース調査レポート

**調査日時**: 2026-04-23
**目的**: ai.chariloto.com 過去取得不可判明後、代替サイトでの過去 AI 予想取得可否調査

## エグゼクティブサマリ

**判定 C: 部分的可能** — netkeirin「Aiライン極」のみ過去データ取得可能、他 2 サイトは全滅

### 各サイト判定

| サイト | robots.txt | 過去取得 | 認証 | 判定 |
|---|---|---|---|---|
| ai.chariloto.com | 制限なし | ❌ 当日のみ | 不要 | 🟢 今日以降蓄積モード (既進行) |
| **oddspark AI予想** (Profile.do 集約) | 制限なし | 🟡 集約のみ | 不要 | ❌ **個別レース詳細は認証壁** |
| **oddspark AI予想** (RaceYosou.do 個別) | 制限なし | - | **要ログイン** | ❌ ログインページリダイレクト |
| **WINTICKET AI予想** | `/keirin/cups/` 以外 OK | ❌ 404 | 不要 | ❌ 過去保持なし |
| **netkeirin Aiライン極** | 制限なし (Disallow なし) | ✅ **detail ページ取得可** | 🟡 会員有料混在 | 🟡 **取得可能** |

## 詳細調査結果

### 🥇 ai.chariloto.com (過去調査)
- robots.txt: 404 (制限なし)
- 免責事項: スクレイピング禁止明記なし
- **トップページ単一ページ設計、過去アーカイブなし**
- 取得範囲: **今日のみ** → 3-6 ヶ月蓄積モードへ

### 🥈 oddspark AI予想 (今回詳細調査)
- robots.txt: Disallow なし、Noindex のみ (AI予想 URL は Noindex 対象外)
- Profile.do 集約ページ: `?confidence={1,2,3}&jo_code=X&race_date=YYYYMMDD&yosou_model=1`
  - 過去日付で 200 OK (38-43KB) 返却
  - race_date ごとに**異なるレスポンス** (日付パラメータは効いている)
  - ただし表示内容は「本日の推奨レース 1 件」のみ
  - 「集計期間」は現在固定で、個別予想データは直接表示されない
- **個別レース AI予想 URL: `/keirin/yosou/ai/RaceYosou.do`**
  - ログイン必須 (未認証はログインページにリダイレクト)
  - 認証壁のため過去の具体的予想は取得不可

**判定**: 認証なしでは取得不可

### 🥉 WINTICKET AI予想
- robots.txt: `/keirin/cups/` のみ Disallow、AI 予想 URL は対象外
- トップページ: 656KB (SSR)、AI 関連リンク 15 件発見
- 予想 URL パターン: `/keirin/{venue}/predictions/{race_id}/{day}/{race_no}?tipster=dsc-00`
  - 例: `/keirin/maebashi/predictions/2026042222/2/11?tipster=dsc-00`
  - race_id = YYYYMMDD + 2桁venue_code
- **過去日付 URL (2024-01-08) は 404** = **過去データ保持なし**
- `/keirin/predictions/` (今日の一覧) は 337KB で現在予想取得可

**判定**: 過去データなし (今日のみ)

### 🎯 netkeirin Aiライン極 (今回発見)
- robots.txt: Disallow なし (完全クロール許可)
- 予想家プロフィール URL: `/yoso/profile/?id=345`
  - 69KB、AI 3 / 予想家 8 / 的中率 7 / 回収率 11 検出
- **個別予想 detail URL: `/yoso/detail/?id=b1228499_345`**
  - **200 OK で 79KB 返却**
  - **タイトル「【Aiライン極】宇都宮競輪... 2025年5月15日 12R」**
  - **2025年5月15日の予想を取得確認** = **過去データ保持確認** ✅
  - キーワード: AI 2, 予想 69, 期待値 1, 買い目 8
  - 「会員 12」「ログイン 2」「無料 1」 = **一部有料の可能性**

**判定**: 🟡 **取得可能 (ただし有料/無料混在の注意)**

## 課題と懸念

### netkeirin Aiライン極 の取得課題
1. **detail URL の id パターン** が `b1228499_345` のような形式
   - `b` + 8桁 + `_` + 記者ID?
   - 全レース分の id を発見する方法が必要
   - プロフィールページから過去予想リストを辿る必要
2. **有料コンテンツの識別**
   - 「会員 12」「ログイン 2」キーワードで一部有料の兆候
   - 無料部分だけを取得する仕組みが必要
3. **サイト負荷配慮**
   - 過去数年分 × 数十レース/日 = 大量リクエスト
   - delay 5-10 秒 + 並列 1

## 推奨アクション

### 判定 C 戦略（部分的可能）

#### 推奨プラン
1. **netkeirin Aiライン極** から過去予想を取得試行
   - プロフィールページから過去予想リスト (id 一覧) を収集
   - 各 detail page を無料範囲のみ取得
   - 有料表示が出たらスキップ記録
   - delay 10 秒 + 並列 1 (安全側)
   - 深夜 2-5 時実行

2. **ai.chariloto.com** は今日以降蓄積モード並行継続
   - Phase 2 (1 レーステスト) へ進むか判断待ち

3. **oddspark / WINTICKET** は撤退
   - 認証壁・過去データなしで取得不可能

#### リスク管理
- netkeirin の有料コンテンツ判定ロジック必須
  - 「有料」「会員限定」「プレミアム」検出 → スキップ
- サイト構造変更リスク
- 記者 ID のハードコード回避

### 代替アクション（全滅判定の場合）

**判定 B: 全滅** に実質近い（netkeirin も 1 記者のみ確実）ので:
- ユーザーへ提案:
  - チャリロト・運営元に直接問い合わせ (formzu フォーム)
  - 研究目的での過去データ提供依頼
  - netkeirin 有料会員登録で Aiライン極の全予想取得

## 現状の取り組み (v0.44 ABC 以降の改善路線)

AI 予想の過去データが得られなくても、以下は進行中:
1. ✅ **v0.44 ABC**: v1.0 ベース + 記者 Gamboo + 市場人気 = 4 OOS平均 -17% (+13pt 突破)
2. 🏃 **Task C**: 2018-2021 データ拡張 (約 2 日)
3. ✅ **Task D**: 選手コメント NLP (574K rows, 完了)
4. 🏃 **v0.36**: K-Dreams 事前オッズ 3 ヶ月蓄積中

**AI 予想追加による +1 情報源がなくても、モデル改善の打ち手は複数ある**。

## 次のユーザー判断

**選択肢 A**: netkeirin Aiライン極 を試行 (過去データ取得・有料判定込み)
**選択肢 B**: AI 予想路線は撤退、既存 v0.44 ABC を磨く
**選択肢 C**: チャリロト運営に問い合わせ (formzu フォーム)

私の推奨: **B**  
- netkeirin は 1 記者のみで既存 gamboo と似た立ち位置
- Task C + Task D + v0.36 オッズ時系列の方が期待値高い
- 時間対効果で模索より実装を進める方が良い

## 成果物
- `data/keirin/v047_phase1_probe.json` (chariloto top/raceresultyear)
- `data/keirin/v047_top_full.html` (chariloto top full)
- `data/keirin/v047_arch_full.html` (chariloto raceresultyear full)
- `data/keirin/v047_oddspark_sample.html` (oddspark AI predict)
- `data/keirin/v047_oddspark_race_sample.html` (oddspark RaceYosou = ログイン画面)
