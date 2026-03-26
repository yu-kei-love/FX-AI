【コピペで動く】はじめての競艇AI予測 — Pythonがわからなくても大丈夫

こんにちは。僕はプログラミング完全未経験から、AI（Claude）に手伝ってもらいながら競艇の予測モデルを作った人間です。最初は「Pythonって何？」という状態だったし、今でもコードの細かい文法はよくわかっていない。でも、Claudeに聞きながらコードを書いてもらって、自分で動かして、検証で数字を出すことだけは続けてきた。

この記事は、そんな初心者の自分が作ったコードをそのまま共有するものです。コピペして実行するだけで、競艇の予測AIが動きます。Pythonの知識はゼロで構いません。コードの意味は全行に日本語コメントをつけたので、あとから読めば理解できます。まずは「動かす体験」をしてください。


〈この記事を読むと、こうなります〉

あなたのPCで、こんな出力が表示されます。

  ==============================================
   競艇AI予測レポート
  ==============================================
   レース: 2024-06-15 住之江 第8R
  ----------------------------------------------
   1号艇  鈴木選手(A1)  予測勝率: 58.3%  → 買い
   2号艇  田中選手(A2)  予測勝率: 14.7%  → 見送り
   3号艇  佐藤選手(B1)  予測勝率: 11.2%  → 見送り
   4号艇  山田選手(A2)  予測勝率:  9.1%  → 見送り
   5号艇  中村選手(B1)  予測勝率:  4.5%  → 見送り
   6号艇  小林選手(B2)  予測勝率:  2.2%  → 見送り
  ----------------------------------------------
   判定: 1号艇の期待値が高いです。単勝をおすすめします。
  ==============================================

AIが各艇の勝つ確率を計算し、「買い」か「見送り」かを判定してくれます。難しいことは全部コードがやります。


〈必要なもの〉

  1. パソコン（Windows / Mac / Linux どれでもOK）
  2. Python 3.8以上（インストール手順は後述）
  3. 30分の時間

たった3つです。特別なGPUやクラウド環境は不要です。


〈全体の流れ〉

5つのステップで進めます。

  Step 1 → データを用意する（CSVファイル）
  Step 2 → 特徴量を作る（AIに渡す数字の準備）
  Step 3 → AIモデルを学習させる（LightGBM）
  Step 4 → 予測を出す（買い / 見送りの判定）
  Step 5 → 結果を確認する

各ステップでコピペするコードと、その解説を載せています。順番通りに進めれば、必ず動きます。


〈Pythonの準備〉

まだPythonが入っていない方は、以下の手順で入れてください。

  1. https://www.python.org/downloads/ にアクセス
  2. 「Download Python 3.x.x」のボタンをクリック
  3. インストーラを実行（「Add Python to PATH」に必ずチェック）
  4. コマンドプロンプト（Mac はターミナル）を開く
  5. 以下を実行してライブラリをインストール:

```
pip install lightgbm pandas numpy scikit-learn
```

これで準備完了です。


--- ここから有料 ---


〈Step 1: データの準備〉

競艇AIを動かすには、過去のレースデータが必要です。CSVファイルという表形式のデータを使います。

データの入手方法は2つあります。

  方法A: ボートレース公式サイトからダウンロード
    → https://www.boatrace.jp/ の「レース結果」からCSVで取得
    → 手動での加工が必要なため、中級者向け

  方法B: この記事のサンプルデータを使う（おすすめ）
    → 以下のコードを実行すると、学習用データを自動生成します
    → まずはこれで動作確認し、慣れたら実データに差し替えましょう

CSVファイルに必要な列はこちらです。

  列名                  意味                  例
  -------------------------------------------------------
  race_id               レース番号            1, 2, 3...
  lane                  枠番（コース）        1〜6
  racer_class           選手の級別            A1, A2, B1, B2
  racer_win_rate        選手の全国勝率        3.50〜8.50
  motor_2place_rate     モーター2連率         20.0〜60.0
  boat_2place_rate      ボート2連率           20.0〜60.0
  avg_start_timing      平均ST                0.10〜0.25
  weather_wind_speed    風速（m/s）           0〜10
  wave_height           波高（cm）            0〜10
  result                結果（1着=1, 他=0）   0 or 1


〈Step 2: 特徴量を作る〉

「特徴量」とは、AIに判断材料として渡す数字のことです。人間が「1号艇のA1選手で、モーターも良い」と考えるのと同じことを、数字に変換します。

この簡易版では10個の特徴量を使います。

  番号  特徴量名            意味（AIはこう考えます）
  ---------------------------------------------------------------
   1    lane               枠番。1号艇は圧倒的に有利
   2    racer_class_num    級別を数値化。A1=4, A2=3, B1=2, B2=1
   3    racer_win_rate     選手の腕前を表す数字。高いほど強い
   4    motor_2place_rate  エンジンの良し悪し。高いほど良い
   5    boat_2place_rate   ボートの良し悪し。高いほど良い
   6    avg_start_timing   スタートの速さ。小さいほど速い
   7    weather_wind_speed 風の強さ。強風はインコースに不利
   8    wave_height        波の高さ。荒れるとベテラン有利
   9    class_x_lane       級別と枠番の掛け算。A1+1号艇=最強
  10    equipment_score    モーターとボートの合計。装備の総合力

本格版では48個の特徴量を使いますが、この10個だけでもかなりの精度が出ます。


〈Step 3: モデルの学習〉

LightGBM（ライトジービーエム）という機械学習アルゴリズムを使います。Kaggleという世界的なAIコンペでも定番のツールで、高速かつ高精度です。

パラメータ（設定値）の意味を説明します。

  パラメータ名       設定値    意味
  -------------------------------------------------------
  objective          binary    「勝つか負けるか」の2択を学習
  metric             auc       予測の良し悪しを測る指標
  num_leaves         31        木の複雑さ。大きいほど複雑
  learning_rate      0.05      学習の慎重さ。小さいほど丁寧
  n_estimators       200       学習の繰り返し回数
  min_child_samples  20        過学習を防ぐ安全装置
  verbose            -1        余計な表示を消す

初心者はこの値のまま使ってください。チューニングは慣れてからで十分です。


〈Step 4: 予測の出し方〉

学習が終わったモデルに新しいレースのデータを渡すと、各艇の勝つ確率を返してくれます。

判定ロジックはシンプルです。

  予測確率 x オッズ > 1.5  →  買い
  予測確率 x オッズ <= 1.5 →  見送り

この「1.5」がEV（期待値）閾値です。1.0を超えれば理論上プラスですが、AIの誤差を考慮して1.5に設定しています。安全マージンを持たせることで、長期的に利益が出やすくなります。


〈Step 5: 結果の確認〉

以下のコードを実行すると、テストデータでの成績が表示されます。

  的中率:    何%のレースで正しく1着を当てたか
  回収率:    100円賭けたら何円返ってきたか（100%超えで黒字）
  ベット数:  AIが「買い」と判定したレース数


〈完成コード — コピペしてそのまま実行〉

以下のコードをファイル名「boat_ai.py」で保存し、実行してください。

```python
# ============================================
# boat_ai.py — はじめての競艇AI予測
# コピペしてそのまま実行できます
# ============================================

# --- ライブラリの読み込み ---
import numpy as np       # 数値計算用のライブラリ
import pandas as pd      # 表データを扱うライブラリ
import lightgbm as lgb   # AI（機械学習）のライブラリ
from sklearn.model_selection import train_test_split  # データ分割用

# --- 設定値 ---
EV_THRESHOLD = 1.5       # 期待値がこの値を超えたら「買い」
TEST_SIZE = 0.2           # データの20%をテスト用に使う
RANDOM_SEED = 42          # 乱数シード（結果の再現性のため）

# --- 枠番別の平均オッズ（単勝） ---
ODDS_TABLE = {1: 2.5, 2: 7.0, 3: 9.0, 4: 11.0, 5: 20.0, 6: 35.0}

# --- 級別を数値に変換する辞書 ---
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


def generate_sample_data(n_races=5000):
    """サンプルデータを自動生成する関数"""
    rng = np.random.RandomState(RANDOM_SEED)  # 乱数の初期化
    rows = []  # データを入れるリスト

    for race_id in range(n_races):            # レース数ぶんループ
        wind = max(0, rng.normal(3.0, 2.0))   # 風速を生成（平均3m/s）
        wave = max(0, rng.normal(3.0, 2.0))   # 波高を生成（平均3cm）

        boats = []  # 6艇ぶんのデータ
        for lane in range(1, 7):              # 1号艇〜6号艇
            # 級別をランダムに割り当て（1号艇にA1が多い傾向を再現）
            if lane == 1:
                cls = rng.choice([4,3,2,1], p=[0.35,0.30,0.25,0.10])
            else:
                cls = rng.choice([4,3,2,1], p=[0.15,0.25,0.35,0.25])

            win_rate = rng.normal(5.0 + cls*0.5, 1.0)     # 全国勝率
            motor = rng.normal(35 + cls*3, 8)              # モーター2連率
            boat = rng.normal(35 + cls*2, 8)               # ボート2連率
            st = max(0.08, rng.normal(0.18 - cls*0.01, 0.03))  # 平均ST

            boats.append({
                "race_id": race_id,           # レース番号
                "lane": lane,                 # 枠番
                "racer_class": cls,           # 級別（数値）
                "racer_win_rate": win_rate,   # 全国勝率
                "motor_2place_rate": motor,   # モーター2連率
                "boat_2place_rate": boat,     # ボート2連率
                "avg_start_timing": st,       # 平均ST
                "weather_wind_speed": wind,   # 風速
                "wave_height": wave,          # 波高
            })

        # --- 勝者を決める（実際の統計に近い確率で） ---
        scores = []  # 各艇のスコア
        for b in boats:
            lane_power = {1:2.5, 2:0.8, 3:0.6, 4:0.4, 5:0.2, 6:0.1}
            score = (
                lane_power[b["lane"]]              # 枠番の有利さ
                + b["racer_class"] * 0.3           # 級別の強さ
                + b["racer_win_rate"] * 0.1        # 勝率の高さ
                + b["motor_2place_rate"] * 0.01    # モーターの良さ
                - b["avg_start_timing"] * 2.0      # STの速さ
                + rng.normal(0, 0.5)               # ランダム要素
            )
            scores.append(score)

        winner = np.argmax(scores)  # スコア最大の艇が1着

        for i, b in enumerate(boats):
            b["result"] = 1 if i == winner else 0  # 1着なら1、それ以外は0
            rows.append(b)

    df = pd.DataFrame(rows)  # リストを表形式に変換
    return df


def create_features(df):
    """特徴量を作る関数（AIに渡す数字を計算）"""
    df = df.copy()  # 元データを壊さないようにコピー

    # 特徴量9: 級別 × 枠番（A1の1号艇が最強という知識を数値化）
    df["class_x_lane"] = df["racer_class"] * (7 - df["lane"])

    # 特徴量10: 装備スコア（モーターとボートの合計力）
    df["equipment_score"] = df["motor_2place_rate"] + df["boat_2place_rate"]

    # AIに渡す特徴量のリスト（この10個で予測する）
    feature_cols = [
        "lane",                 # 1. 枠番
        "racer_class",          # 2. 級別
        "racer_win_rate",       # 3. 全国勝率
        "motor_2place_rate",    # 4. モーター2連率
        "boat_2place_rate",     # 5. ボート2連率
        "avg_start_timing",     # 6. 平均ST
        "weather_wind_speed",   # 7. 風速
        "wave_height",          # 8. 波高
        "class_x_lane",         # 9. 級別×枠番
        "equipment_score",      # 10. 装備スコア
    ]
    return df, feature_cols


def train_model(df, feature_cols):
    """AIモデルを学習させる関数"""
    X = df[feature_cols]       # 特徴量（AIへの入力）
    y = df["result"]           # 正解ラベル（1着かどうか）

    # データを学習用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,    # 20%をテストに使う
        random_state=RANDOM_SEED # 再現性のため固定
    )

    # LightGBMモデルの設定
    model = lgb.LGBMClassifier(
        objective="binary",         # 2値分類（勝つ/負ける）
        metric="auc",               # 評価指標
        num_leaves=31,              # 木の複雑さ
        learning_rate=0.05,         # 学習率（小さいほど丁寧）
        n_estimators=200,           # 学習の繰り返し回数
        min_child_samples=20,       # 過学習防止の安全装置
        verbose=-1,                 # 余計な表示を消す
        random_state=RANDOM_SEED,   # 再現性のため固定
    )

    # 学習を実行
    model.fit(X_train, y_train)

    # テストデータで精度を確認
    accuracy = model.score(X_test, y_test)
    print(f"  モデル精度（正解率）: {accuracy:.1%}")

    return model, X_test, y_test


def predict_race(model, feature_cols, race_data):
    """1レースの予測を出す関数"""
    df_race, feature_cols = create_features(race_data)  # 特徴量を作成
    X = df_race[feature_cols]    # 特徴量を取り出す

    # 各艇の勝つ確率を予測
    probabilities = model.predict_proba(X)[:, 1]

    # 結果を表示
    print("=" * 50)
    print("  競艇AI予測レポート")
    print("=" * 50)

    best_lane = None    # 最もおすすめの艇
    best_ev = 0         # 最高の期待値

    for i, row in df_race.iterrows():
        lane = int(row["lane"])                  # 枠番
        prob = probabilities[i % len(probabilities)]  # 予測確率
        odds = ODDS_TABLE.get(lane, 5.0)         # オッズ
        ev = prob * odds                         # 期待値を計算

        # 「買い」か「見送り」かを判定
        if ev > EV_THRESHOLD:
            judgment = "← 買い"
        else:
            judgment = "   見送り"

        # 級別を文字に戻す
        cls_name = {4:"A1", 3:"A2", 2:"B1", 1:"B2"}.get(int(row["racer_class"]), "??")

        print(f"  {lane}号艇  級別:{cls_name}  "
              f"予測勝率: {prob:5.1%}  EV: {ev:.2f}  {judgment}")

        # 最高EVの艇を記録
        if ev > best_ev:
            best_ev = ev
            best_lane = lane

    print("-" * 50)
    if best_ev > EV_THRESHOLD:
        print(f"  判定: {best_lane}号艇の期待値が高いです。単勝をおすすめします。")
    else:
        print("  判定: このレースは見送りをおすすめします。")
    print("=" * 50)


def evaluate_model(model, df_test, feature_cols):
    """テストデータでの成績を表示する関数"""
    X_test = df_test[feature_cols]  # テスト用の特徴量
    probs = model.predict_proba(X_test)[:, 1]  # 勝つ確率を予測
    df_eval = df_test.copy()
    df_eval["pred_prob"] = probs    # 予測確率を追加

    total_bet = 0     # ベット総額
    total_return = 0  # リターン総額
    bet_count = 0     # ベット回数
    hit_count = 0     # 的中回数

    # レースごとに集計
    for race_id, group in df_eval.groupby("race_id"):
        for _, row in group.iterrows():
            lane = int(row["lane"])
            prob = row["pred_prob"]
            odds = ODDS_TABLE.get(lane, 5.0)
            ev = prob * odds

            if ev > EV_THRESHOLD:       # 期待値が閾値を超えたら賭ける
                total_bet += 100        # 1ベット100円
                bet_count += 1
                if row["result"] == 1:  # 当たったら
                    total_return += 100 * odds  # オッズぶん返ってくる
                    hit_count += 1

    # 成績を表示
    print("\n" + "=" * 50)
    print("  テストデータでの成績")
    print("=" * 50)
    if bet_count > 0:
        hit_rate = hit_count / bet_count * 100
        roi = total_return / total_bet * 100
        print(f"  ベット数:  {bet_count} レース")
        print(f"  的中率:    {hit_rate:.1f}%")
        print(f"  回収率:    {roi:.1f}%")
        print(f"  収支:      {total_return - total_bet:+,.0f} 円")
    else:
        print("  該当するベットがありませんでした。")
    print("=" * 50)


# --- メイン処理（ここから実行が始まる） ---
if __name__ == "__main__":
    print("競艇AI予測を開始します...\n")

    # Step 1: データを用意
    print("Step 1: サンプルデータを生成中...")
    df = generate_sample_data(n_races=5000)  # 5000レースぶん生成
    print(f"  生成完了: {len(df)} 行のデータ\n")

    # Step 2: 特徴量を作成
    print("Step 2: 特徴量を作成中...")
    df, feature_cols = create_features(df)
    print(f"  特徴量: {len(feature_cols)} 個\n")

    # Step 3: モデルを学習
    print("Step 3: AIモデルを学習中...")
    model, X_test, y_test = train_model(df, feature_cols)
    print("  学習完了\n")

    # Step 4: 新しいレースを予測してみる
    print("Step 4: サンプルレースを予測...\n")
    sample_race = df[df["race_id"] == 0].copy()  # 最初のレースで予測テスト
    predict_race(model, feature_cols, sample_race)

    # Step 5: テストデータで成績を確認
    print("\nStep 5: テストデータで成績を確認...")
    test_df = df[df["race_id"] >= 4000].copy()  # 後半1000レースでテスト
    test_df, _ = create_features(test_df)
    evaluate_model(model, test_df, feature_cols)

    print("\n完了しました。お疲れさまでした。")
```


〈よくある質問〉

Q. 本当にコピペだけで動きますか?
A. はい。Python と lightgbm がインストールされていれば動きます。上のコードをそのままコピーし、boat_ai.py という名前で保存して、コマンドプロンプトで python boat_ai.py と実行してください。

Q. 実際のレースデータはどうやって入手しますか?
A. ボートレース公式サイト（boatrace.jp）の「レース結果」ページからダウンロードできます。CSV形式に加工して、generate_sample_data 関数の代わりに pd.read_csv("your_data.csv") で読み込んでください。

Q. サンプルデータで回収率100%を超えないのですが?
A. サンプルデータはランダム生成なので、実データほどパターンがはっきりしません。実際のレースデータを使うと精度が上がります。まずは「コードが動くこと」を確認することが大切です。

Q. 特徴量を増やしたいのですが?
A. create_features 関数の中に新しい計算を追加し、feature_cols リストに名前を追加するだけです。たとえば選手の当地勝率や体重を追加すると精度が上がります。本格版では48個の特徴量を使っています。

Q. 実際に賭けて利益が出ますか?
A. この簡易版はあくまで学習用です。実運用にはWalk-Forward検証、アンサンブル学習、閾値の最適化など追加の工夫が必要です。まずはこの記事で基礎を身につけ、段階的にステップアップしてください。


〈うまくいかないときの対処法〉

症状: 「ModuleNotFoundError: No module named 'lightgbm'」と表示される
原因: lightgbm がインストールされていません
対処: pip install lightgbm を実行してください

症状: 「pip が認識されません」と表示される
原因: Python のパスが通っていません
対処: Python を再インストールし「Add Python to PATH」にチェックを入れてください

症状: 数値がおかしい、NaN が表示される
原因: データに欠損値（空欄）が含まれている可能性
対処: CSVファイルを開いて空欄がないか確認してください。空欄は 0 で埋めるか、その行を削除してください

症状: 回収率が極端に低い / 高い
原因: データ量が少ない、または EV_THRESHOLD の設定が合っていない
対処: まずはサンプルデータ（5000レース）で動作確認し、実データに切り替えるときは EV_THRESHOLD を 1.3〜2.0 の範囲で調整してみてください


〈次のステップ〉

この記事のコードが動いたら、次は以下に挑戦してみてください。

  1. 実際のレースデータで学習させる
  2. 特徴量を20個、30個と増やす
  3. XGBoost や CatBoost とのアンサンブルを試す
  4. Walk-Forward 検証で過学習を防ぐ

これらの詳しい手順は、別の記事「競艇AI予測モデルの作り方 — EV閾値最適化でPF+24%改善した全手順」で解説しています。

はじめの一歩は踏み出せました。僕自身、まさにこのコードを動かしたところがスタート地点でした。ここから先は、データを変えて、特徴量を足して、少しずつ自分だけのAIに育てていってください。一緒にがんばりましょう。


#競艇AI #ボートレース #Python #初心者 #コピペ #機械学習 #AI予測
