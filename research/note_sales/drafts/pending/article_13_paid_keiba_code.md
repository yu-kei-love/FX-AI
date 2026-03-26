# 競馬AI予測モデル構築ガイド【Walk-Forward検証+コード付き】

プログラミング未経験の僕が、AI（Claude）の力を借りてここまで作れた。そのコードを共有する記事です。

前回の無料記事では、僕の競馬AIが回収率120%超え（Walk-Forward検証）を達成した話をした。「具体的にどうやって作ったの？」という質問を多くいただいたので、この記事では実装面にフォーカスする。

正直に言うと、コードの大部分はClaudeに書いてもらった。自分がやったのは「こういう予測がしたい」という方向性を決めて、出てきたコードを動かして、結果を見て、また相談する――その繰り返しだ。でも、その過程で動くものができた。

本番環境で使っているフルモデル（28特徴量+実データパイプライン）をそのまま公開するわけにはいかないが、10特徴量のシンプル版を一から構築する過程を見せる。考え方とアーキテクチャは本番モデルと同じなので、ここから拡張していけばフルモデルに近づけられる。

---

## 対象読者

- Pythonの基礎がわかる人（pandas, numpyが読めるレベル）
- 機械学習の基本概念を知っている人（教師あり学習、分類、過学習）
- 競馬予測に興味がある人（エンジニアじゃなくても大丈夫）

僕自身が未経験からのスタートだったので、コードには日本語コメントを多めにつけている。完全な初心者の方は、先に無料記事「初心者向け：競馬AIって何を見て予測してるの？」を読んでおくと理解しやすい。

---

## 全体アーキテクチャ

モデルの全体像はこうなる。

```
データ取得 → 特徴量エンジニアリング → 5モデルアンサンブル → バリューベット判定 → リスク管理
```

各ステップを順に実装していく。

---

## Step 1：特徴量の定義

まず、モデルに投入する特徴量を定義する。シンプル版では10個に絞った。

```python
FEATURE_COLS = [
    "horse_age",         # 馬齢
    "horse_weight",      # 馬体重 (kg)
    "win_rate",          # 勝率
    "place_rate",        # 複勝率
    "last3_avg_finish",  # 直近3走平均着順
    "jockey_win_rate",   # 騎手勝率
    "post_position",     # 枠番
    "distance",          # 距離 (m)
    "track_condition",   # 馬場状態 (良=0, 稍重=1, 重=2, 不良=3)
    "odds",              # 単勝オッズ
]
```

本番モデルではここに脚質、ペース予測、交互作用項などが加わって28個になるが、まずはこの10個で動くものを作ろう。

---

## Step 2：5モデルアンサンブルの実装

単一モデルだと過学習のリスクが高い。性格の異なる5つのモデルを組み合わせる。

```python
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import pandas as pd


class KeibaEnsemble:
    """5モデルアンサンブル"""

    def __init__(self):
        self.models = []
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = X.columns.tolist()
        self.models = []

        # LightGBM
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        }
        dtrain = lgb.Dataset(X, label=y)
        lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=300)
        self.models.append(("lgb", lgb_model))

        # XGBoost
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
            "seed": 42,
        }
        dmatrix = xgb.DMatrix(X, label=y)
        xgb_model = xgb.train(xgb_params, dmatrix, num_boost_round=300)
        self.models.append(("xgb", xgb_model))

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=300, learning_rate=0.05,
            depth=6, random_seed=42, verbose=0,
        )
        cat_model.fit(X.values, y.values)
        self.models.append(("cat", cat_model))

        # RandomForest
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=20, random_state=42,
        )
        rf_model.fit(X.values, y.values)
        self.models.append(("rf", rf_model))

        # ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=20, random_state=42,
        )
        et_model.fit(X.values, y.values)
        self.models.append(("et", et_model))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """5モデルの予測確率を単純平均"""
        X_use = X[self.feature_cols]
        probas = []

        for tag, model in self.models:
            if tag == "lgb":
                p = model.predict(X_use)
            elif tag == "xgb":
                p = model.predict(xgb.DMatrix(X_use))
            elif tag in ("cat", "rf", "et"):
                p = model.predict_proba(X_use.values)[:, 1]
            probas.append(p)

        return np.mean(probas, axis=0)
```

ポイントをいくつか補足する（Claudeに教えてもらった内容が多い）。

〈ハイパーパラメータについて〉
learning_rateは0.05に統一している。Claudeいわく、0.1だと学習が速いが過学習しやすく、0.01だとround数を増やす必要がある。0.05はバランスが良いとのこと。

min_child_samples（LightGBM）とmin_child_weight（XGBoost）は50に設定。デフォルトの20だと競馬データの場合は過学習しやすかった（これは自分で試して確認した）。

〈なぜ重み付けではなく単純平均か〉
重み付けも試したが、結局は単純平均のほうがWalk-Forwardでの成績が良かった。Claudeに聞いたら「重みのチューニング自体がFold内で過学習を起こすから」とのこと。

---

## Step 3：Walk-Forward検証

ここが最も重要なパートだ（と、Claudeに何度も言われた）。

最初、僕はランダムにtrain/testを分割していた。でもClaudeに「それだと未来のデータで学習して過去を予測する『リーク』が起きる」と指摘された。バックテストは良い成績になるが、実運用では全く使い物にならないらしい。

Walk-Forward検証ではこれを防ぐ。

```python
def walk_forward_validate(df, initial_train_months=12,
                          test_months=3, edge_threshold=1.30,
                          bet_unit=1000):
    """
    Expanding Window Walk-Forward検証

    1. 最初の12ヶ月で学習 → 次の3ヶ月でテスト
    2. 最初の15ヶ月で学習 → 次の3ヶ月でテスト
    3. ...（学習データは常に拡張していく）
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values("race_date")

    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    all_bets = []
    fold = 0
    test_start = min_date + pd.DateOffset(months=initial_train_months)

    while test_start + pd.DateOffset(months=test_months) <= max_date:
        test_end = test_start + pd.DateOffset(months=test_months)

        # 学習: 最初からtest_startまで（Expanding Window）
        train_df = df[df["race_date"] < test_start]
        # テスト: test_startからtest_endまで
        test_df = df[
            (df["race_date"] >= test_start) & (df["race_date"] < test_end)
        ]

        if len(train_df) < 1000 or len(test_df) < 100:
            test_start += pd.DateOffset(months=test_months)
            continue

        # モデル訓練
        model = KeibaEnsemble()
        model.fit(train_df[FEATURE_COLS], train_df["win"])

        # テスト期間でバリューベットを探す
        for race_id, race_group in test_df.groupby("race_id"):
            probs = model.predict_proba(race_group[FEATURE_COLS])

            for i, (_, row) in enumerate(race_group.iterrows()):
                ev = probs[i] * row["odds"]
                if ev > edge_threshold:
                    payout = row["odds"] * bet_unit if row["win"] == 1 else 0
                    all_bets.append({
                        "fold": fold,
                        "date": str(row["race_date"].date()),
                        "pred_prob": round(probs[i], 4),
                        "odds": row["odds"],
                        "ev": round(ev, 3),
                        "bet": bet_unit,
                        "payout": payout,
                        "profit": payout - bet_unit,
                    })

        fold += 1
        test_start += pd.DateOffset(months=test_months)

    # 結果集計
    if not all_bets:
        return {"error": "No bets found"}

    bets_df = pd.DataFrame(all_bets)
    total_bet = bets_df["bet"].sum()
    total_payout = bets_df["payout"].sum()
    gross_profit = bets_df[bets_df["payout"] > 0]["payout"].sum()
    gross_loss = bets_df[bets_df["payout"] == 0]["bet"].sum()

    return {
        "n_folds": fold,
        "n_bets": len(all_bets),
        "recovery_rate": total_payout / total_bet if total_bet > 0 else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
        "hit_rate": len(bets_df[bets_df["payout"] > 0]) / len(bets_df),
        "total_profit": total_payout - total_bet,
    }
```

〈Expanding WindowとSliding Windowの違い〉
Expanding Windowは学習データが毎Foldで増えていく。Sliding Windowは直近N ヶ月だけを使う。競馬データは質よりも量が効く（古いデータにも有用なパターンがある）ため、Expanding Windowのほうが安定した。

〈edge_thresholdの設定〉
1.30は「モデルが30%以上の優位性を感じたときだけ賭ける」という意味。JRAの控除率は約20〜25%なので、最低でも1.20以上に設定しないと長期的にプラスにならない。1.30は控除率に加えてモデルの誤差マージンも考慮した値だ。

---

## Step 4：バリューベット判定

予測上位に賭けるのではなく、期待値（EV）が閾値を超えた馬にだけ賭ける。

```python
def find_value_bets(race_df, model, edge_threshold=1.30):
    """
    1レース分のバリューベットを判定

    EV = モデルの推定勝率 * オッズ
    EV > edge_threshold の馬にベット
    """
    probs = model.predict_proba(race_df[FEATURE_COLS])

    bets = []
    for i, (_, row) in enumerate(race_df.iterrows()):
        ev = probs[i] * row["odds"]
        if ev > edge_threshold:
            bets.append({
                "horse_name": row.get("horse_name", f"Horse_{i}"),
                "pred_prob": round(probs[i], 4),
                "odds": row["odds"],
                "ev": round(ev, 3),
                "implied_prob": round(1.0 / row["odds"], 4),
                "edge": round(probs[i] - 1.0 / row["odds"], 4),
            })

    return bets
```

ここで重要なのは implied_prob（オッズから逆算した市場の推定確率）と pred_prob（モデルの推定確率）の差だ。edgeがプラスなら、モデルは市場よりもその馬を高く評価している。このedgeこそがバリューベットの源泉になる。

---

## Step 5：リスク管理

ベットの管理を怠ると、一時的なドローダウンで資金が尽きる。日次・月次の予算制限を設ける。

```python
class RiskManager:
    """シンプルなベット管理"""

    def __init__(self, daily_budget=5000, monthly_budget=50000,
                 bet_unit=1000):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.bet_unit = bet_unit
        self.daily_spent = 0
        self.monthly_spent = 0
        self.current_day = None
        self.current_month = None

    def can_bet(self, date_str):
        day = date_str[:10]
        month = date_str[:7]

        if day != self.current_day:
            self.daily_spent = 0
            self.current_day = day
        if month != self.current_month:
            self.monthly_spent = 0
            self.current_month = month

        if self.daily_spent + self.bet_unit > self.daily_budget:
            return False
        if self.monthly_spent + self.bet_unit > self.monthly_budget:
            return False
        return True

    def record_bet(self, date_str):
        self.daily_spent += self.bet_unit
        self.monthly_spent += self.bet_unit
```

日次5,000円、月次50,000円がデフォルト設定。この金額なら、仮にモデルが機能しなくても致命的な損失にはならない。

---

ここから先は有料部分です。

---

## Step 6：特徴量エンジニアリングの拡張（有料）

シンプル版の10特徴量から、本番モデルに近づけるための追加特徴量の設計思想を解説する。

〈交互作用特徴量の考え方〉

単独の特徴量だけでは捉えられない非線形なパターンがある。たとえば「内枠」という情報だけでは不十分で、「内枠 × 芝 × 短距離」の組み合わせで初めて予測力が生まれる。

本番モデルで効果が確認できた交互作用項をいくつか紹介する。

```python
def engineer_features(df):
    """特徴量エンジニアリング（拡張版）"""

    # ペース予測: 逃げ馬・先行馬の割合
    # running_style: 逃げ=0, 先行=1, 差し=2, 追込=3
    front_runners = df.groupby("race_id").apply(
        lambda x: (x["running_style"] <= 1).mean()
    )
    df = df.merge(
        front_runners.rename("pace_predict"),
        left_on="race_id", right_index=True
    )

    # 脚質×ペース交互作用
    # ハイペース(逃げ馬多い)なら差し追込有利
    df["style_pace_interact"] = df["running_style"] * df["pace_predict"]

    # 距離適性: 同距離帯での過去勝率
    # (実装は rolling stats で計算済みと仮定)

    # 馬場×体重: 重馬場では重い馬が有利
    df["condition_x_weight"] = df["track_condition"] * df["horse_weight"]

    # オッズ vs 勝率: 市場評価との乖離
    df["odds_vs_winrate"] = df["odds"] * df["win_rate"]

    # 枠番×距離: 短距離芝は内枠有利
    df["post_x_dist"] = df["post_position"] * (1.0 / df["distance"])

    # クラス×馬齢: 若い馬のクラス上昇パターン
    df["age_x_class"] = df["horse_age"] * df["race_class"]

    return df
```

〈特徴量のpruning（削除）〉

特徴量は増やせばいいわけではない。ノイズになる特徴量を入れると、むしろ精度が落ちる。

本番モデルのv2.1では、以下の特徴量を削除した。

| 削除した特徴量 | 理由 |
|-------------|------|
| trainer_win_rate | jockey_win_rateと相関が高い（片方で十分） |
| sire_encoded | 重要度がほぼゼロ（オッズに織り込み済み） |
| bms_encoded | 同上 |
| venue_aptitude | 重要度がほぼゼロ |
| odds_vs_jockey | oddsと相関が高い（冗長） |

pruningの手順は以下のとおり。

1. 全特徴量でモデルを訓練し、feature importanceを取得
2. 重要度が0.5%未満の特徴量を候補としてリストアップ
3. 相関行列で相関係数0.9以上のペアを特定し、重要度の低い方を削除
4. 削除後にWalk-Forwardで再検証し、PFが維持or向上していることを確認

この手順を2〜3回繰り返すと、ノイズが減って汎化性能が上がる。

---

## Step 7：EV閾値の最適化（有料）

edge_threshold=1.30は最初の設定値だ。これを最適化する方法を解説する。

```python
def optimize_threshold(df, thresholds=None):
    """Walk-Forward内でEV閾値を最適化"""
    if thresholds is None:
        thresholds = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50]

    results = []
    for th in thresholds:
        wf_result = walk_forward_validate(df, edge_threshold=th)
        results.append({
            "threshold": th,
            "pf": wf_result.get("profit_factor", 0),
            "recovery": wf_result.get("recovery_rate", 0),
            "n_bets": wf_result.get("n_bets", 0),
            "hit_rate": wf_result.get("hit_rate", 0),
        })

    return pd.DataFrame(results)
```

参考として、本番モデルでの閾値ごとの成績を示す。

| EV閾値 | PF | 回収率 | ベット数 | 的中率 |
|--------|-----|--------|---------|--------|
| 1.10 | 1.12 | 101.5% | 2,840 | 9.2% |
| 1.20 | 1.31 | 113.6% | 1,520 | 8.9% |
| 1.30 | 1.56 | 122.4% | 650 | 8.7% |
| 1.40 | 1.73 | 131.2% | 310 | 8.4% |
| 1.50 | 1.91 | 143.8% | 145 | 8.1% |

閾値を上げるとPFは上がるが、ベット数は急激に減る。1.50だとPF=1.91だが月に数回しかベットチャンスがない。1.30は「PFとベット頻度のバランス」が良い落としどころだ。

ただし注意が必要なのは、この閾値最適化自体が過学習の原因になりうるということ。本番では、Walk-Forwardの各Foldの中で閾値を決めるのではなく、外側のループで固定値を使うのが安全だ。

---

## Step 8：本番モデルへの拡張ロードマップ（有料）

シンプル版から本番モデルに近づけるためのステップを示す。

〈Phase 1：特徴量を10→20に増やす〉
- 騎手複勝率（jockey_place_rate）の追加
- 前走着順（last_finish）、前走からの日数（days_since_last）
- 出走頭数（field_size）、芝/ダート（track_type）
- 人気順（popularity）
- クラス（race_class）、性別（sex）、体重変化（weight_change）

これだけでPFが0.1〜0.2程度向上する見込みがある。

〈Phase 2：交互作用特徴量の追加〉
- ペース予測 + 脚質×ペース
- 馬場×体重
- 枠番×距離
- クラス×馬齢

Phase 2で本番モデルの7割くらいの性能に到達する。

〈Phase 3：実データパイプラインの構築〉
- netkeiba.comからのデータスクレイピング
- Rolling statsの計算（未来リーク防止の時系列処理）
- 上がり3F、コーナー通過順位、斤量などの実データ固有特徴量
- 距離適性、馬場適性の計算

Phase 3で本番モデルとほぼ同等の構成になる。

〈Phase 4：高度な最適化〉
- Fractional Kellyによるベットサイジング
- 複勝モデルの構築（edge_threshold=1.20）
- 特徴量のpruning（相関分析 + 重要度分析）

---

## 本番モデルとシンプル版の性能比較（有料）

参考までに、Walk-Forwardでの性能比較を載せておく。

| 指標 | シンプル版（10特徴量） | 本番モデル（28特徴量） |
|------|---------------------|---------------------|
| PF（単勝） | 1.15〜1.25 | 1.50〜1.60 |
| 回収率 | 105〜115% | 120〜125% |
| ベット数/年 | 800〜1,200 | 600〜700 |
| Fold安定性 | 1〜2 Foldで負け越し | ほぼ全Foldでプラス |

シンプル版でもプラス圏に入れるが、本番モデルとの差はかなりある。特にFold安定性（どのテスト期間でも安定してプラスになるか）に大きな差がある。

この差の大部分は「交互作用特徴量」と「特徴量のpruning」から生まれている。単純に特徴量を増やすだけでなく、不要なものを削る作業が重要だ。

---

## まとめ

この記事で構築したもの。

- 10特徴量のシンプルな競馬予測モデル
- LightGBM + XGBoost + CatBoost + RandomForest + ExtraTrees の5モデルアンサンブル
- Expanding Window Walk-Forward検証
- EV閾値ベースのバリューベット判定
- 日次・月次予算制限のリスク管理

シンプル版でも、正しくWalk-Forwardで検証すれば回収率100%を超えるポテンシャルはある。ただし、安定的にプラスを維持するには特徴量エンジニアリングとpruningが欠かせない。

初心者の自分でもここまで作れた。同じように「やってみたいけど自分にできるかな」と思っている人の参考になれば嬉しい。質問があればコメント欄でどうぞ。初歩的な質問でも全然OKです。

---

## 免責事項

本記事の内容は、あくまで機械学習モデルの開発手法の解説を目的としたものです。

- 掲載されているコードはシンプル化された教育用のものであり、そのまま実運用に使うことは推奨しません
- バックテスト結果は将来の収益を保証するものではありません
- 公営競技への投資は元本割れのリスクがあります
- 本記事の情報を元にした投資判断は、すべて自己責任でお願いします
- ギャンブル依存症にご注意ください。お困りの方は相談窓口（厚生労働省）へ

---

#競馬AI #機械学習 #Python #LightGBM #XGBoost #アンサンブル学習 #Walk-Forward #データサイエンス #競馬予測 #コード付き
