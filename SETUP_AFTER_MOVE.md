# 引っ越し後 最初にやること
# このファイルを上から順番に実行するだけでOK

---

## STEP 1：環境確認（5分）

以下がインストールされているか確認する：
```bash
python --version        # 3.9以上
git --version
pip --version
```

不足している場合：
- Python：https://www.python.org/downloads/
- Git：https://git-scm.com/downloads

---

## STEP 2：依存ライブラリのインストール（10分）
```bash
pip install -r requirements.txt
```

requirements.txtが存在しない場合は
以下を手動でインストールする：
```bash
pip install requests
pip install beautifulsoup4
pip install pandas
pip install numpy
pip install lightgbm
pip install scikit-learn
pip install optuna
pip install sqlalchemy
pip install schedule
pip install tqdm
pip install matplotlib
pip install mlflow
```

---

## STEP 3：PCのスリープ設定を無効にする（2分）

Windowsの場合：
設定 → 電源とスリープ → スリープ → なし

理由：
スクレイピング中にスリープすると
データ取得が止まる

---

## STEP 4：ボートレースのデータ取得を再開（1分）
```bash
cd research/boat
python scraper_historical.py --resume
```

--resumeオプションで
前回の続きから自動再開される

取得状況の確認：
```bash
python scraper_historical.py --status
```

出力例：
取得済み：168レース
目標：87,600レース
進捗：0.2%
残り推定：34日

---

## STEP 5：スケジューラーの再設定（3分）

PC再起動後も自動で続きから
再開できるように設定する：
```bash
python setup_scheduler.py --setup-daily
```

設定確認：
タスクスケジューラー →
BoatraceAI\DailyBatch が Ready状態であること

---

## STEP 6：競輪の利用規約確認（30分）

以下のサイトの利用規約を確認する
スクレイピングが許可されているか調べる

確認するサイト：
[ ] Gamboo（gamboo.jp）
    → 利用規約URL：
    → スクレイピング：可 / 不可 / 要確認
    → 個人利用：可 / 不可
    → 商用利用：可 / 不可

[ ] chariloto.com
    → 利用規約URL：
    → スクレイピング：可 / 不可 / 要確認
    → 個人利用：可 / 不可
    → 商用利用：可 / 不可

[ ] Kドリームズ（keirin.kdreams.jp）
    → 利用規約URL：
    → スクレイピング：可 / 不可 / 要確認
    → 個人利用：可 / 不可
    → 商用利用：可 / 不可

[ ] KEIRIN.JP（公式）
    → 利用規約URL：
    → スクレイピング：可 / 不可 / 要確認

確認後：
このファイルの[ ]を[✅]または[❌]に更新する

---

## STEP 7：Perfecta Naviでバンクデータを確認（30分）

URL：
https://www.chariloto.com/perfectanavi/keirin-auto-navi/85/

確認項目：
research/keirin/data/bank_master.py の
以下のパラメータが正しいか照合する：
・みなし直線距離
・カント角度
・バンク特性コメント

間違いがあれば修正してコミットする：
```bash
git add .
git commit -m "bank_master: Perfecta Navi照合・修正"
```

---

## STEP 8：競輪データ取得の開始（利用規約確認後）

STEP 6で利用規約を確認してから実行すること

Gambooがスクレイピング可の場合：
```bash
cd research/keirin
python scraper/scraper_historical.py --start 2024-01-01
```

不可の場合：
charilotoのみを使う設計に変更する

---

## STEP 9：動作確認（10分）

ボートレース：
```bash
cd research/boat
python -c "
from data_interface import DataInterface
from feature_engine import create_features
di = DataInterface(mode='sqlite', db_path='data/boatrace.db')
races, entries, odds = di.get_races('2024-03-28', '2024-03-29'), di.get_entries(), di.get_odds()
print('ボートレース：DB接続OK')
print(f'レース数：{len(races)}')
"
```

競輪（mockモードで確認）：
```bash
cd research/keirin
python -c "
from model.data_interface import DataInterface
from model.feature_engine import create_features
di = DataInterface(mode='mock')
print('競輪：mockモードOK')
"
```

---

## STEP 10：週次確認スケジュール

データ取得中は以下を毎週確認する：

毎週月曜日：
```bash
# ボートレース取得状況
python research/boat/scraper_historical.py --status

# エラーログ確認
tail -50 research/boat/logs/scraper.log
```

確認項目：
[ ] スクレイピングが継続して動いているか
[ ] エラーが増えていないか
[ ] course_takenの分布が正常か
    （1コース：40〜50%が正常）

---

## 完了チェックリスト

[ ] STEP 1：環境確認
[ ] STEP 2：ライブラリインストール
[ ] STEP 3：スリープ無効
[ ] STEP 4：ボートレース取得再開
[ ] STEP 5：スケジューラー設定
[ ] STEP 6：競輪利用規約確認
[ ] STEP 7：バンクデータ照合
[ ] STEP 8：競輪データ取得開始
[ ] STEP 9：動作確認
[ ] STEP 10：週次確認スケジュール登録

全て完了したら：
```bash
git add .
git commit -m "v0.4: 引っ越し後環境セットアップ完了"
git tag -a v0.4 -m "データ取得開始"
```
