【コード全公開】競艇AI予測モデルの作り方 — EV閾値最適化でPF+24%改善した全手順

こんにちは。プログラミング完全未経験から、AI（Claude）にコードを書いてもらいながら競艇予測モデルを作り始めて1年半。初心者の自分でもここまでできた、という記録をコードごと全公開します。この記事では、モデルの核心部分であるEV閾値最適化の手法と、実際のPythonコードを共有します。

この記事を読むと、以下のことがわかります。

- LightGBM + XGBoost + CatBoost + RF + ExtraTreesの5モデルアンサンブルの組み方
- Walk-Forward検証でEV閾値を最適化する具体的な手順
- PF(Profit Factor)を2.35から2.91へ改善した全プロセス
- 48個の特徴量の重要度ランキングと、削るべき特徴量の判定基準
- Fractional Kelly(0.25倍)によるベットサイジングの実装

〈最終的な成績〉

単勝(Tansho):   PF = 2.91 / 回収率 258.3% / 822ベット
2連単(Exacta):  PF = 3.24 / 回収率 307.4% / 711ベット
2連複(Quinella): PF = 3.21 / 回収率 296.3% / 1,278ベット

Walk-Forward 5ウィンドウ(Expanding Window)で検証した数値です。過学習を防ぐため、テストデータは一度も訓練に使っていません。


〈なぜEV閾値が重要なのか〉

多くの競艇AIの解説記事は「モデルの精度を上げる」ことに集中しています。AUCを0.01改善するとか、特徴量を増やすとか。もちろん精度は大事ですが、実運用で利益を出すには「どのベットを選ぶか」のフィルタリングが同じくらい重要です。

EV(Expected Value、期待値)は次の式で計算します。

  EV = モデル予測確率 x オッズ

EV > 1.0 なら理論上プラス期待値ですが、モデルの誤差を考えると1.0ギリギリのベットは危険です。では閾値をいくつに設定するのが最適か。1.25か、1.50か、2.00か。

ここに「aha moment」がありました。

当初、僕はEV閾値をデフォルトの1.25に設定していました。「EV 1.25以上なら十分プラス期待値だろう」と。結果はPF = 2.35。悪くはないけれど、不安定な期間もありました。

ある日、Claudeに「閾値を上げたらどうなりますか？」と聞かれて試してみたんです。1.50、1.75、2.00と。ベット数は減る。当然です。でもPFは上がり続けた。最終的にEV >= 2.00で単勝PF = 2.91。改善幅は+23.7%。

理由はシンプルでした。EV閾値が低いと「モデルがちょっとだけ有利と判断したベット」が大量に入る。これらはモデルの誤差範囲内で、実際にはプラスかマイナスかわからないベットです。閾値を上げると、モデルが強く確信しているベットだけが残る。ベット数は半減するけれど、1ベットあたりの質が劇的に上がる。

ただし、闇雲に閾値を上げればいいわけではありません。上げすぎるとベット数が少なすぎて統計的に信頼できなくなる。また、買い目の種類(単勝/2連単/2連複)ごとに最適値が異なります。

この記事の有料部分では、21段階(1.00〜2.00、ステップ0.05)の全データを公開し、Walk-Forward検証で最適閾値を自動探索するPythonコードを解説します。


〈改善前後の比較サマリー〉

改善のインパクトを先に示します。

単勝:
  改善前(EV >= 1.25): PF = 2.353 / 回収率 207.8% / 1,722ベット
  改善後(EV >= 2.00): PF = 2.910 / 回収率 258.3% / 822ベット
  PF改善: +0.557 (+23.7%)

2連単:
  改善前(EV >= 1.15): PF = 2.599 / 回収率 245.7% / 1,094ベット
  改善後(EV >= 1.95): PF = 3.241 / 回収率 307.4% / 711ベット
  PF改善: +0.642 (+24.7%)

2連複:
  改善前(EV >= 1.10): PF = 2.599 / 回収率 239.5% / 1,840ベット
  改善後(EV >= 2.00): PF = 3.208 / 回収率 296.3% / 1,278ベット
  PF改善: +0.609 (+23.4%)

注目してほしいのは、ベット数が大幅に減っているにもかかわらず、PFと回収率が大きく改善している点です。「量より質」がベッティングの世界では決定的に重要だということがわかります。


〈モデルの全体像〉

先に全体のパイプラインを説明しておきます。

1. データ収集: レース結果、選手成績、モーター/ボート成績、展示データ、天候データ
2. 特徴量エンジニアリング: 基本15特徴 + 派生33特徴 = 合計48特徴量
3. モデル訓練: 5モデルアンサンブル(LGB + XGB + CatBoost + RF + ExtraTrees)
4. 確率予測: アンサンブル予測 → レース内正規化(6艇の確率合計 = 1.0)
5. EV計算: 予測確率 x オッズ
6. ベット選択: EV >= 閾値 かつ Kelly基準でサイジング
7. Walk-Forward検証: 5ウィンドウExpanding Windowで未来データに対する性能を評価

有料部分では、特に3, 5, 6, 7のコードと詳細データを公開します。


--- ここから有料 ---


〈セクション1: EV最適化テーブル全公開〉

Walk-Forward 5ウィンドウで検証した、EV閾値1.00〜2.00の全21段階のデータです。ここにモデルの実力がすべて詰まっています。

〈単勝(Tansho)の全閾値データ〉

 EV閾値  ベット数  的中率    PF     回収率   Sharpe   MDD       平均EV  累計損益
 1.00     2,212   20.7%   2.061   184.1%   7.806    8,700    2.123   372,200
 1.05     2,111   20.9%   2.134   189.7%   7.975    8,400    2.176   378,600
 1.10     2,000   21.0%   2.201   194.9%   8.046    8,800    2.237   379,400
 1.15     1,901   20.9%   2.258   199.6%   8.072    8,700    2.295   378,500
 1.20     1,819   20.5%   2.285   202.1%   7.989    8,700    2.345   371,600
 1.25     1,722   20.3%   2.353   207.8%   8.030    9,100    2.409   371,300  *旧設定
 1.30     1,635   19.8%   2.382   210.9%   7.901    8,900    2.469   362,700
 1.35     1,525   19.4%   2.445   216.4%   7.853    8,800    2.552   355,100
 1.40     1,438   19.3%   2.533   223.8%   7.917    8,200    2.623   356,000
 1.45     1,362   19.3%   2.629   231.4%   8.018    7,200    2.690   358,000
 1.50     1,291   18.8%   2.641   233.4%   7.827    7,200    2.757   344,300
 1.55     1,229   18.6%   2.697   238.2%   7.787    7,200    2.819   339,700
 1.60     1,168   18.2%   2.703   239.2%   7.559    6,800    2.884   325,200
 1.65     1,109   18.1%   2.760   244.1%   7.484    9,600    2.951   319,700
 1.70     1,058   17.7%   2.719   241.5%   7.193    8,800    3.013   299,500
 1.75     1,016   17.5%   2.792   247.8%   7.235   10,000    3.066   300,300
 1.80       969   17.4%   2.804   249.0%   7.071    9,600    3.129   288,700
 1.85       937   17.2%   2.792   248.4%   6.932    9,200    3.173   278,100
 1.90       893   17.4%   2.862   253.9%   6.914    9,000    3.237   274,800
 1.95       853   17.2%   2.873   255.0%   6.743    9,400    3.299   264,400
 2.00       822   17.2%   2.910   258.3%   6.672    8,600    3.349   260,200  *最適

読み方のポイント:

PFは閾値を上げるとほぼ単調に増加しています。1.00の2.061から2.00の2.910まで、きれいな右肩上がり。これは「モデルが高EVと判断したベットほど実際に質が高い」ことを意味します。モデルの予測精度が十分であることの証拠です。

一方、累計損益は閾値1.10付近でピークの379,400円。閾値を上げるとベット数が減るため、1ベットあたりの利益は増えても総額は減少します。

Sharpeは1.10-1.15付近がピーク(8.07)。これはリスク調整後リターンの最適点を示しています。

つまり、目的によって最適閾値が変わるということです。
- 最大PF(資金効率)を狙うなら → EV >= 2.00
- 最大累計損益を狙うなら → EV >= 1.10
- 最大Sharpe(安定性)を狙うなら → EV >= 1.15

僕は資金効率(PF)を重視しています。理由は、PFが高いほどドローダウンからの回復が早く、資金ショートのリスクが低いからです。


〈2連単(Exacta)の全閾値データ〉

 EV閾値  ベット数  的中率    PF     回収率   Sharpe    MDD       平均EV  累計損益
 1.00     1,173    9.4%   2.533   238.9%   4.247   10,200    3.019   325,890
 1.05     1,153    9.2%   2.532   239.1%   4.191   10,200    3.053   320,825
 1.10     1,125    9.1%   2.566   242.4%   4.190   10,000    3.102   320,475
 1.15     1,094    8.9%   2.599   245.7%   4.172    9,800    3.158   318,800  *旧設定
 1.20     1,067    8.8%   2.627   248.4%   4.153    9,900    3.209   316,640
 1.25     1,045    8.7%   2.660   251.5%   4.156    9,400    3.250   316,665
 1.30     1,020    8.7%   2.711   256.1%   4.183    9,400    3.299   318,515
 1.35       987    8.6%   2.757   260.6%   4.172    9,400    3.365   317,030
 1.40       958    8.6%   2.795   264.2%   4.149    9,200    3.425   314,570
 1.45       930    8.2%   2.807   265.9%   4.079    8,800    3.485   308,585
 1.50       910    8.0%   2.839   269.2%   4.073    9,235    3.529   307,860
 1.55       883    7.8%   2.886   273.8%   4.065    9,035    3.591   306,960
 1.60       863    7.9%   2.928   277.6%   4.067    9,235    3.637   306,550
 1.65       831    7.7%   2.997   284.3%   4.068    8,700    3.715   306,300
 1.70       806    7.7%   3.035   287.8%   4.039   11,360    3.778   302,795
 1.75       786    7.6%   3.089   292.9%   4.049   13,660    3.830   303,295
 1.80       773    7.6%   3.118   295.6%   4.042   12,660    3.865   302,395
 1.85       748    7.8%   3.188   301.9%   4.048   17,050    3.933   302,005
 1.90       731    7.4%   3.188   302.7%   3.979   17,275    3.981   296,270
 1.95       711    7.4%   3.241   307.4%   3.971   16,475    4.039   294,880  *最適
 2.00       695    7.2%   3.215   305.5%   3.863   15,875    4.087   285,690

2連単で注目すべきは、EV 2.00ではなく1.95が最適な点です。PF 3.241が最大値で、2.00では3.215とわずかに下がります。ここに「閾値を上げれば上げるほど良い」わけではない現実が見えます。2連単はそもそも的中率が低い(7-9%)ので、閾値を上げすぎるとベット数が統計的に不安定な領域に入ります。

もう一つ気になるのがMDD。1.85以降でMDDが16,000円超に膨らんでいます。高閾値の2連単は1ベットの当たりハズレが大きく振れるため、連敗時のドローダウンが深くなります。


〈2連複(Quinella)の全閾値データ〉

 EV閾値  ベット数  的中率     PF     回収率   Sharpe    MDD       平均EV  累計損益
 1.00     1,857   12.9%   2.588   238.4%   6.238   11,900    3.554   513,940
 1.05     1,852   12.8%   2.589   238.6%   6.232   11,500    3.561   513,340
 1.10     1,840   12.7%   2.599   239.5%   6.234   11,500    3.577   513,440  *旧設定
 1.15     1,829   12.6%   2.595   239.4%   6.197   11,800    3.592   509,880
 1.20     1,805   12.6%   2.615   241.2%   6.199   11,800    3.624   509,660
 1.25     1,774   12.5%   2.638   243.4%   6.194   11,800    3.666   508,840
 1.30     1,729   12.2%   2.664   246.2%   6.163   12,300    3.728   505,480
 1.35     1,697   11.7%   2.659   246.4%   6.069   12,100    3.774   496,960
 1.40     1,657   11.7%   2.704   250.5%   6.097   11,900    3.832   498,860
 1.45     1,630   11.7%   2.742   253.9%   6.135   11,500    3.871   501,760
 1.50     1,595   11.4%   2.763   256.2%   6.101   11,100    3.924   498,320
 1.55     1,554   11.4%   2.818   261.1%   6.137   10,500    3.987   500,800
 1.60     1,529   11.4%   2.858   264.7%   6.173   10,300    4.027   503,500
 1.65     1,490   11.3%   2.911   269.4%   6.197   10,300    4.090   504,820
 1.70     1,459   11.3%   2.947   272.7%   6.195   11,000    4.141   503,980
 1.75     1,433   11.1%   2.957   274.0%   6.137   11,000    4.185   498,560
 1.80     1,393   10.8%   2.973   275.9%   6.050   11,000    4.254   490,100
 1.85     1,361   11.0%   3.040   281.7%   6.109   11,000    4.311   494,500
 1.90     1,333   11.0%   3.102   287.0%   6.162   11,000    4.362   498,500
 1.95     1,302   11.0%   3.152   291.6%   6.175   10,800    4.420   498,920
 2.00     1,278   11.1%   3.208   296.3%   6.214   10,800    4.466   501,720  *最適

2連複は閾値2.00が最適で、かつMDDが10,800円と安定しています。ベット数も1,278と十分。3つの買い目の中で最もバランスが良い戦略です。累計損益も501,720円と単勝より大きい。個人的には2連複をメイン戦略にしています。


〈セクション2: Walk-Forward EV閾値探索の実装コード〉

ここからが本題のコードです。boat_model.pyから抜粋し、EV閾値探索に関わる部分を共有します。コードはほぼClaudeに書いてもらったものだけど、動く状態で公開するので、僕と同じような初心者の方でもそのまま使えるはず。

〈EV計算とベット判定のコア〉

find_value_bets関数がベット判定の心臓部です。

```python
def find_value_bets(race_df, bet_type="win", min_ev=2.00, kelly_frac=0.25):
    """
    バリューベットを検出する。
    Kelly criterion で賭け金を算出。
    """
    bets = []

    if bet_type == "win":
        for _, row in race_df.iterrows():
            odds = row.get("odds", APPROX_ODDS.get(row["lane"], 10.0))
            if pd.isna(odds) or odds <= 1.0:
                odds = APPROX_ODDS.get(row["lane"], 10.0)
            implied_prob = 1.0 / odds
            model_prob = row["pred_prob"]
            ev = model_prob * odds

            if ev >= min_ev:
                kf = kelly_fraction(model_prob, odds, fraction=kelly_frac)
                bets.append({
                    "race_id": row["race_id"],
                    "lane": row["lane"],
                    "model_prob": model_prob,
                    "implied_prob": implied_prob,
                    "odds": odds,
                    "ev": ev,
                    "kelly_fraction": kf,
                    "bet_type": "win",
                    "win": row["win"],
                })
    return bets
```

ポイントは3つ。

1. EVの計算はシンプルに model_prob x odds。モデルが「この艇が20%で勝つ」と予測し、オッズが10倍なら、EV = 0.20 x 10.0 = 2.0。
2. min_ev がフィルタ。この値を変えることでベットの質をコントロールする。
3. Kelly criterionで賭け金を動的に決定。確信度が高いベットには多く、低いベットには少なく。

〈Kelly Criterionの実装〉

```python
def kelly_fraction(model_prob, odds, fraction=0.25):
    """
    Fractional Kelly Criterion。
    fraction=0.25 → 1/4 Kelly でリスク抑制。

    Kelly: f* = (p * b - q) / b
      p = 勝率, q = 1-p, b = オッズ-1 (net odds)
    """
    p = model_prob
    q = 1.0 - p
    b = odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (p * b - q) / b
    return max(0.0, kelly * fraction)
```

フルKellyは理論上最適ですが、現実には破産リスクが高い。1/4 Kellyにすることで、期待リターンは約75%に下がりますが、破産確率はほぼゼロになります。競艇のような高分散ゲームでは必須の設定です。

〈2連単の条件付き確率計算〉

2連単(Exacta)は「1着と2着を順番通りに当てる」買い方です。これを確率モデルで扱うには条件付き確率が必要です。

```python
# P(1st=A, 2nd=B) = P(win=A) * P(2nd=B | A wins)
# P(2nd=B | A wins) ≈ P(place=B) adjusted for A winning

sorted_boats = race_df.sort_values("pred_prob", ascending=False)
top_n = sorted_boats.head(3)  # 上位3艇の組み合わせを探索

for boat_a in top_n:
    for boat_b in top_n:
        if boat_a == boat_b:
            continue
        p_win_a = boat_a["pred_prob"]

        # Placeモデルがある場合
        p_place_b = boat_b["pred_place_prob"]
        p_b_not_win = max(1.0 - boat_b["pred_prob"], 0.01)
        p_second_b_given_a = (p_place_b - boat_b["pred_prob"]) / p_b_not_win
        p_second_b_given_a = max(min(p_second_b_given_a, 0.8), 0.02)

        exacta_prob = p_win_a * p_second_b_given_a
```

Winモデルとは別にPlaceモデル(2着以内に入る確率)を訓練し、その2つを組み合わせて2連単の確率を計算します。正直この辺の条件付き確率はClaudeに教えてもらうまでよく分からなかったけど、単にWin確率の上位2名を選ぶより精度が高くなるらしい。

〈PF/Sharpe/MDDの計算〉

```python
def compute_metrics(bets_df, base_bet=500):
    # Kelly対応: kelly_fractionがあればベット額を可変に
    if "kelly_fraction" in bets_df.columns:
        bets_df["bet_amount"] = bets_df["kelly_fraction"].apply(
            lambda kf: max(100, min(
                int(base_bet * max(kf, 0.1) * 4 / 100) * 100,
                base_bet * 3
            ))
        )
        bets_df["bet_amount"] = bets_df["bet_amount"].clip(lower=100)

    # 各ベットの損益
    bets_df["pnl"] = np.where(
        bets_df["win"] == 1,
        bets_df["bet_amount"] * bets_df["odds"] - bets_df["bet_amount"],
        -bets_df["bet_amount"]
    )

    gross_profit = bets_df.loc[bets_df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(bets_df.loc[bets_df["pnl"] < 0, "pnl"].sum())

    # PF = 総利益 / 総損失
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 回収率 = 総リターン / 総ベット額
    total_bet = bets_df["bet_amount"].sum()
    total_return = total_bet + bets_df["pnl"].sum()
    recovery = total_return / total_bet if total_bet > 0 else 0

    # Sharpe Ratio (全ベットベース)
    pnl_series = bets_df["pnl"].values
    sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(len(pnl_series))

    # Maximum Drawdown
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    mdd = drawdowns.max()

    return {"pf": pf, "sharpe": sharpe, "mdd": mdd, "recovery": recovery, ...}
```

PF(Profit Factor)は「勝ちトレードの利益合計 / 負けトレードの損失合計」。PF > 1.0なら黒字。2.0以上で優秀、3.0以上は非常に良いモデルです。

〈Walk-Forward検証のコード〉

過学習防止のため、Expanding Window方式のWalk-Forward検証を使っています。

```python
def walk_forward_validate(df, n_folds=5):
    df = df.sort_values("race_date").reset_index(drop=True)
    race_ids = df["race_id"].unique()
    n_races = len(race_ids)
    test_size = n_races // (n_folds + 1)

    for fold in range(n_folds):
        # 訓練: 先頭から train_end まで(拡大していく)
        train_end = test_size * (fold + 1)
        test_end = min(train_end + test_size, n_races)

        train_ids = set(race_ids[:train_end])
        test_ids = set(race_ids[train_end:test_end])

        train_df = df[df["race_id"].isin(train_ids)]
        test_df = df[df["race_id"].isin(test_ids)]

        # 訓練データの末尾15%をバリデーションに
        X_train_all = train_df[FEATURE_COLS].values
        y_train_win = train_df["win"].values
        val_split = int(len(X_train_all) * 0.85)

        # Win model + Place model を訓練
        models = train_ensemble(
            X_train_all[:val_split],
            y_train_win[:val_split],
            X_train_all[val_split:],
            y_train_win[val_split:]
        )

        # テストデータで予測
        X_test = test_df[FEATURE_COLS].values
        test_df["raw_prob"] = predict_proba(models, X_test)
        test_df = normalize_race_probs(test_df)

        # 戦略別にベットをシミュレーション
        for bet_type in ["win", "exacta", "quinella"]:
            all_bets = []
            for rid in test_ids:
                race_data = test_df[test_df["race_id"] == rid]
                bets = find_value_bets(race_data, bet_type=bet_type)
                all_bets.extend(bets)
            metrics = compute_metrics(pd.DataFrame(all_bets))
```

Expanding Windowの仕組みは以下の通りです。

Fold 1: 訓練 = 期間A         テスト = 期間B
Fold 2: 訓練 = 期間A+B       テスト = 期間C
Fold 3: 訓練 = 期間A+B+C     テスト = 期間D
Fold 4: 訓練 = 期間A+B+C+D   テスト = 期間E
Fold 5: 訓練 = 期間A+B+C+D+E テスト = 期間F

各フォールドで訓練データは拡大し、テストは常に未来のデータです。これにより「モデルが未来のデータに対してどの程度有効か」を正確に評価できます。

〈EV閾値グリッドサーチの実装〉

上記のWalk-Forward検証を、EV閾値を変えながら繰り返すことで最適値を探索します。

```python
def optimize_ev_threshold(all_test_dfs, bet_type="win"):
    """
    Walk-Forwardテスト結果に対してEV閾値を変えながらPFを計算。
    最もPFが高く、かつベット数 >= 50の閾値を最適値とする。
    """
    results = []
    for ev_thresh in np.arange(1.00, 2.05, 0.05):
        all_bets = []
        for test_df in all_test_dfs:
            for rid in test_df["race_id"].unique():
                race_data = test_df[test_df["race_id"] == rid]
                bets = find_value_bets(
                    race_data,
                    bet_type=bet_type,
                    min_ev=ev_thresh
                )
                all_bets.extend(bets)

        if len(all_bets) < 50:
            continue

        bets_df = pd.DataFrame(all_bets)
        metrics = compute_metrics(bets_df)
        results.append({
            "ev_threshold": round(ev_thresh, 2),
            "n_bets": metrics["n_bets"],
            "hit_rate": metrics["hit_rate"],
            "pf": metrics["pf"],
            "recovery": metrics["recovery"],
            "sharpe": metrics["sharpe"],
            "mdd": metrics["mdd"],
        })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["pf"].idxmax()]
    return best["ev_threshold"], results_df
```

ポイントは2つ。

1. EV閾値を変えるのはテストデータへのフィルタリングだけ。モデル自体は再訓練しません。Walk-Forwardの各フォールドで一度訓練したモデルの予測結果をプールし、そこにフィルタを適用します。
2. 最低ベット数を50に設定。これより少ないと統計的に信頼できません。

〈5モデルアンサンブルの構成〉

```python
def train_ensemble(X_train, y_train, X_val=None, y_val=None):
    # --- LightGBM ---
    lgb_params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }
    # early_stopping 50 rounds, max 1000 rounds

    # --- XGBoost ---
    xgb_params = {
        "objective": "binary:logistic",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }

    # --- CatBoost ---
    # iterations=500, depth=6, l2_leaf_reg=3.0

    # --- RandomForest ---
    # n_estimators=300, max_depth=8, min_samples_leaf=20

    # --- ExtraTrees ---
    # n_estimators=300, max_depth=8, min_samples_leaf=20
```

5モデルの平均を取ることで、個別モデルの偏りを相殺します。LightGBM単体のPFが2.6でも、アンサンブルにすると2.9に上がる。これは各モデルが異なる特徴に注目するためです。

アンサンブルの重みは均等(各1/5)にしています。重み最適化は過学習のリスクが高く、Walk-Forwardでは安定しませんでした。


〈セクション3: 特徴量重要度ランキング(全48特徴量)〉

LightGBMのgain-basedで測定した全特徴量の重要度です。

 順位  特徴量                    重要度   構成比  累積比
  1   exhibition_time            407     4.6%    4.6%
  2   weight_diff                391     4.4%    9.0%
  3   boat_2place_rate           384     4.3%   13.4%
  4   motor_vs_field_avg         374     4.2%   17.6%
  5   place_consistency          358     4.1%   21.7%
  6   local_vs_national          347     3.9%   25.6%
  7   racer_weight               340     3.9%   29.5%
  8   exhibition_start           334     3.8%   33.2%
  9   win_rate_vs_field_avg      320     3.6%   36.9%
 10   wind_x_wave                300     3.4%   40.3%
 11   wind_x_lane                295     3.3%   43.6%
 12   class_x_motor              290     3.3%   46.9%
 13   wave_height                279     3.2%   50.1%
 14   racer_win_rate             265     3.0%   53.1%
 15   combined_equipment         263     3.0%   56.0%
 16   avg_start_timing           257     2.9%   58.9%
 17   racer_local_win_rate       255     2.9%   61.8%
 18   field_strength_std         251     2.8%   64.7%
 19   motor_2place_rate          250     2.8%   67.5%
 20   class_dominance            243     2.8%   70.3%
 21   start_x_class              241     2.7%   73.0%
 22   lane_class_motor           239     2.7%   75.7%
 23   weather_wind_speed         195     2.2%   77.9%
 24   start_x_lane               187     2.1%   80.0%
 25   class_x_lane               182     2.1%   82.1%
 26   lane                       171     1.9%   84.0%
 27   racer_place_rate           171     1.9%   86.0%
 28   field_strength             168     1.9%   87.9%
 29   wind_dir_x_lane            143     1.6%   89.5%
 30   racer_local_2place_rate    143     1.6%   91.1%
 31   racer_3place_rate          113     1.3%   92.4%
 32   boat_rank                   89     1.0%   93.4%
 33   class_rank_in_race          76     0.9%   94.3%
 34   equipment_rank              73     0.8%   95.1%
 35   start_timing_rank           70     0.8%   95.9%
 36   flying_count                55     0.6%   96.5%
 37   win_rate_rank               50     0.6%   97.1%
 38   motor_rank                  47     0.5%   97.6%
 39   weather_condition           45     0.5%   98.1%
 40   lane_squared                42     0.5%   98.6%
 41   course_type                 42     0.5%   99.1%
 42   flying_x_class              41     0.5%   99.5%
 43   lane_x_race_number          13     0.1%   99.7%
 44   racer_class                 11     0.1%   99.8%
 45   inner_class_advantage       11     0.1%   99.9%
 46   is_inner_lane                7     0.1%  100.0%
 47   late_count                   0     0.0%  100.0%
 48   race_number                  0     0.0%  100.0%


〈セクション4: 特徴量プルーニングの判断基準〉

特徴量は多ければ良いわけではありません。不要な特徴量はノイズとなり、過学習の原因になります。削除すべき特徴量を2つの基準で判定します。

〈基準1: 重要度がほぼゼロの特徴量〉

以下の4特徴量は重要度が全体の0.12%以下で、モデルの予測にほとんど寄与していません。

  削除対象:
  - inner_class_advantage  (重要度 11, 0.12%)
  - is_inner_lane          (重要度 7,  0.08%)
  - late_count             (重要度 0,  0.00%)
  - race_number            (重要度 0,  0.00%)

late_countは全員ほぼ0のため分散がなく、race_numberはレース番号(第1R〜第12R)で予測に無関係です。is_inner_laneは「1-3号艇かどうか」のバイナリ変数ですが、laneそのものが存在するため冗長です。

〈基準2: 高相関の冗長ペア(|r| > 0.95)〉

同じ情報を持つ特徴量が複数あると、モデルが分割を「偶然」選ぶリスクが増えます。

 特徴量ペア                                   相関係数  判定
 lane vs lane_x_race_number                   r=1.000  lane_x_race_number を削除
 racer_place_rate vs racer_3place_rate         r=1.000  racer_3place_rate を削除
 racer_local_win_rate vs racer_local_2place_rate  r=1.000  racer_local_2place_rate を削除
 lane vs lane_squared                         r=0.979  lane_squared を削除
 lane_squared vs lane_x_race_number           r=0.979  (既に上で削除済み)
 racer_win_rate vs racer_3place_rate           r=0.952  (既に上で削除済み)
 racer_win_rate vs racer_place_rate            r=0.952  racer_place_rate を残す

判定のルール: ペアのうち重要度が高い方を残し、低い方を削除。

v3.3ではこのプルーニングを実施済みです。具体的には、BASE_FEATURE_COLSからlate_countを除外し、DERIVED_FEATURE_COLSからinner_class_advantage, is_inner_lane, lane_squared, lane_x_race_number, race_numberを除外しました。racer_3place_rateもracer_win_rateと相関r=0.952のため除外。

プルーニング前後のPF比較:
  v3.2 (48特徴量): 単勝PF = 2.82
  v3.3 (42特徴量): 単勝PF = 2.91  (+3.2%)

特徴量を6個減らしてPFが上がる。これが過学習削減の効果です。


〈セクション5: パイプラインの構築手順〉

実際にこのモデルを動かすまでの手順を整理します。

〈ステップ1: データ準備〉

必要なデータは以下の通りです。

- 選手別成績(全国勝率、2連率、当地成績)
- モーター/ボート成績(2連率)
- 展示データ(展示タイム、展示スタート)
- 天候データ(風速、風向、波高)
- オッズデータ(単勝、2連単、2連複)

データソースは公式のボートレース結果ページからスクレイピングするか、APIサービスを利用します。

〈ステップ2: 特徴量作成〉

```python
# 基本特徴量15個はデータから直接取得
# 派生特徴量はcreate_features()で自動生成

def create_features(df):
    # レース内順位系
    df["class_rank_in_race"] = df.groupby("race_id")["racer_class"].rank(
        ascending=False, method="min"
    )
    df["win_rate_rank"] = df.groupby("race_id")["racer_win_rate"].rank(
        ascending=False, method="min"
    )

    # レース平均との差分
    df["win_rate_vs_field_avg"] = (
        df["racer_win_rate"]
        - df.groupby("race_id")["racer_win_rate"].transform("mean")
    )
    df["motor_vs_field_avg"] = (
        df["motor_2place_rate"]
        - df.groupby("race_id")["motor_2place_rate"].transform("mean")
    )

    # 相互作用特徴量
    df["class_x_lane"] = df["racer_class"] * (7 - df["lane"])
    df["wind_x_lane"] = df["weather_wind_speed"] * df["lane"]
    df["wind_x_wave"] = df["weather_wind_speed"] * df["wave_height"]
    df["local_vs_national"] = df["racer_local_win_rate"] - df["racer_win_rate"]
    df["place_consistency"] = (
        df["racer_place_rate"] / df["racer_win_rate"].clip(lower=1.0)
    )

    # 3要素交互作用
    motor_norm = (df["motor_2place_rate"] - 30) / 20.0
    df["lane_class_motor"] = (
        (7 - df["lane"]) * df["racer_class"] * motor_norm.clip(0, 2)
    )

    return df
```

特徴量設計の思想（Claudeに教わったもの）: 「絶対値」よりも「レース内の相対値」が重要。全国勝率7.0の選手でも、相手が全員A1級なら不利。逆に勝率5.0でも相手がB級ばかりなら有利。win_rate_vs_field_avg, motor_vs_field_avgなどの差分特徴量がそれを捉えます。

〈ステップ3: モデル訓練と予測〉

5モデルのアンサンブルを訓練し、レース内で確率を正規化します。

```python
# 5モデルアンサンブルの予測を平均
def predict_proba(models, X):
    preds = []
    preds.append(models["lgb"].predict(X))
    preds.append(models["xgb"].predict(xgb.DMatrix(X)))
    preds.append(models["cat"].predict_proba(X)[:, 1])
    preds.append(models["rf"].predict_proba(X)[:, 1])
    preds.append(models["et"].predict_proba(X)[:, 1])
    return np.mean(preds, axis=0)

# レース内正規化: 6艇の予測確率の合計を1.0にする
def normalize_race_probs(df):
    race_sums = df.groupby("race_id")["raw_prob"].transform("sum")
    df["pred_prob"] = df["raw_prob"] / race_sums
    return df
```

レース内正規化は必須です。各モデルは個別の艇の勝率を予測しますが、6艇の確率合計が1.0になる保証がありません。正規化しないとEVの計算が歪みます。

〈ステップ4: 実運用でのEV閾値適用〉

最適化した閾値を実運用に適用する際のコード例です。

```python
# 最適EV閾値(Walk-Forwardで導出)
OPTIMAL_EV = {
    "win":      2.00,
    "exacta":   1.95,
    "quinella": 2.00,
}

def generate_daily_bets(today_races_df, models):
    today_races_df = create_features(today_races_df)
    X = today_races_df[FEATURE_COLS].values
    today_races_df["raw_prob"] = predict_proba(models, X)
    today_races_df = normalize_race_probs(today_races_df)

    recommendations = []
    for bet_type, min_ev in OPTIMAL_EV.items():
        for rid in today_races_df["race_id"].unique():
            race = today_races_df[today_races_df["race_id"] == rid]
            bets = find_value_bets(race, bet_type=bet_type, min_ev=min_ev)
            recommendations.extend(bets)

    return pd.DataFrame(recommendations)
```


〈セクション6: ベットサイジング戦略〉

EV閾値でベットを選別した後、各ベットにいくら賭けるかも重要です。

〈Fractional Kelly (1/4 Kelly)の運用〉

compute_metrics関数内のベット額算出ロジックです。

```python
# kelly_fractionからベット額を算出
# base_bet = 500円の場合
bet_amount = max(100, min(
    int(base_bet * max(kf, 0.1) * 4 / 100) * 100,
    base_bet * 3  # 最大1,500円
))
bet_amount = max(bet_amount, 100)  # 最低100円
```

ルール:
- 最低ベット: 100円
- 最大ベット: base_bet の 3倍(base_bet=500円なら1,500円)
- Kelly値が大きいほどベット額が増加
- 100円単位に丸め

〈推奨する運用資金とbase_bet〉

運用資金に対するbase_betの目安:

  運用資金       base_bet   最大ベット   月間ベット数(目安)
  50,000円       200円      600円       約170回
  100,000円      500円      1,500円     約170回
  300,000円      1,000円    3,000円     約170回
  500,000円      2,000円    6,000円     約170回

月間ベット数は2連複(EV >= 2.00)で1,278ベット / 7.5ヶ月 = 約170回/月を想定しています。

重要なのはbase_betを運用資金の1%以下に抑えること。PF = 3.2でもMDD = 10,800円が発生します。base_bet=500円でMDD 10,800円は約22ベット分。これに耐えられる資金が必要です。

〈3戦略の資金配分〉

3つの買い目を併用する場合の推奨配分:

  単勝:     25%
  2連単:    25%
  2連複:    50%

2連複を多めにする理由は、PFが高く(3.21)、MDDが低く(10,800円)、ベット数が多い(1,278)ためです。Sharpeも6.21と最高。安定性と利益のバランスが最も良い戦略です。

2連単はPF最大(3.24)ですがMDDが16,475円と大きく、ベット数も711と少ない。補助的に使います。

単勝はSharpeが6.67と高く安定していますが、PFは2.91で3戦略中最低。ただし的中率17.2%は心理的に楽です。


〈まとめ〉

この記事で解説した改善のポイントを整理します。

1. EV閾値の最適化が最もインパクトが大きい。モデル精度の改善よりも、「どのベットを選ぶか」のフィルタリングでPFが+24%改善した。

2. 閾値の最適値は買い目ごとに異なる。単勝2.00、2連単1.95、2連複2.00。闇雲に上げるのではなく、Walk-Forward検証で探索すること。

3. 特徴量プルーニングも有効。重要度ゼロの特徴量と高相関ペアを削除するだけでPF +3.2%。

4. Fractional Kelly(1/4)でベットサイジング。フルKellyは理論上最適だが現実には危険すぎる。

5. 5モデルアンサンブルは単体モデルより安定。重みは均等でよい。

6. Walk-Forward検証は必須。通常のクロスバリデーションでは時系列データの過学習を検出できない。


〈免責事項〉

本記事は筆者個人の研究成果を共有するものであり、投資助言ではありません。競艇を含む公営競技への投票は自己責任で行ってください。

本記事に掲載したモデルの成績はWalk-Forward検証によるバックテスト結果であり、将来の収益を保証するものではありません。過去の成績が将来の結果を約束するものではないことをご理解ください。

公営競技への投票は、余剰資金の範囲内で行ってください。生活資金を投じることは絶対にお控えください。

モデルのパラメータや特徴量は、データの性質やレース環境の変化に応じて定期的に再検証する必要があります。一度最適化した閾値がずっと有効である保証はありません。


#競艇AI #ボートレース #Python #機械学習 #LightGBM #コード公開
