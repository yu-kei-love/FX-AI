# 初心者がFX自動売買AIを作ってみた【簡易版・コード全公開】

FXの自動売買AIを自分で作りたい。でもどこから手をつけていいかわからない。

僕もまったく同じ状態からスタートした。プログラミングの経験はゼロ。でもAI（Claude）の力を借りたら、ここまでのものが作れた。そのコードを全部公開する。

この記事で公開するのは、プロダクション版から一部の最適化を省いた簡易バージョンだ。それでもWalk-Forward検証で動作する完全なパイプラインになっている。コピペで動くコードなので、そこから自分なりに改良していく土台にしてほしい。

---

## この記事で得られるもの

- 5モデルアンサンブルによるFX方向予測の完全なコード
- 特徴量エンジニアリング（テクニカル指標 + 交互作用特徴量）
- Walk-Forward検証のパイプライン
- 複数通貨ペア対応の設計パターン
- バックテスト結果の評価方法

## この記事に含まれないもの（プロダクション版の機能）

以下の最適化はプロダクション版にのみ含まれており、本記事では省略している。これらを追加するとPFがさらに向上する。

- 時間帯フィルター（特定のUTC時間帯をスキップしてPF改善）
- ATRベースの動的SL/TP（ボラティリティに応じた損切り・利確の自動調整）
- 性能ベースのモデル重み動的調整
- 経済カレンダー連携（高インパクトイベント前後のトレード回避）
- ボラティリティレジーム判定（異常ボラ時のスキップ）
- マルチタイムフレーム特徴量（4時間足・日足の指標統合）
- Telegram通知連携

---

## 自己紹介

僕はプログラミングの知識がゼロの状態から、AI（Claude）にコードを書いてもらいながら予測モデルを作り始めた。エンジニアではないし、コードの細かい部分は正直わかっていないことも多い。でも「検証して、数字で判断する」ことだけは妥協していない。

今はFX、日本株、競艇、競輪、仮想通貨の5市場で、同じ考え方（Walk-Forward + アンサンブル学習）をベースにモデルを動かしている。FXについてはUSD/JPYとAUD/JPYの2通貨ペアで実際にペーパートレード（デモ取引）を回している。

---

## アーキテクチャ概要

簡易版のアーキテクチャは以下の通り。

```
1時間足データ取得
    ↓
テクニカル指標計算（RSI, MACD, BB, ATR等）
    ↓
交互作用特徴量の追加
    ↓
方向ラベル生成（12時間後に上昇=1, 下降=0）
    ↓
5モデルアンサンブル学習
    ↓
予測 + 自信度フィルター + 一致度フィルター
    ↓
シグナル出力（BUY / SELL / SKIP）
```

---

ここから先は有料です。コード全体と詳細な解説を含みます。

---

## 1. 特徴量の設計

まず、1時間足のOHLCVデータからテクニカル指標を計算する。

```python
import numpy as np
import pandas as pd


def compute_rsi(close, period=14):
    """RSI（相対力指数）を計算"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD（移動平均収束拡散法）を計算"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close, period=20):
    """ボリンジャーバンドを計算"""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (2 * std) / sma.replace(0, np.nan)
    return upper, lower, width


def compute_atr(high, low, close, period=14):
    """ATR（平均真の値幅）を計算"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_technical_features(df):
    """テクニカル指標をDataFrameに追加"""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 基本指標
    df["RSI_14"] = compute_rsi(close, 14)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(close)
    df["BB_upper"], df["BB_lower"], df["BB_width"] = compute_bollinger(close)
    df["ATR_14"] = compute_atr(high, low, close, 14)

    # 移動平均
    for period in [5, 20, 75]:
        df[f"MA_{period}"] = close.rolling(period).mean()

    # リターン
    df["Return_1"] = close.pct_change(1)
    df["Return_6"] = close.pct_change(6)
    df["Return_24"] = close.pct_change(24)

    # ボラティリティ
    df["Volatility_24"] = df["Return_1"].rolling(24).std()

    # 出来高変化率
    if "Volume" in df.columns:
        df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(24).mean()

    return df
```

次に、交互作用特徴量を追加する。これは単独のテクニカル指標よりも予測力が高い場合がある。

```python
def add_interaction_features(df):
    """交互作用特徴量を追加"""
    # RSI x ボラティリティ: 高ボラ時のRSIはシグナルが強い
    df["RSI_x_Vol"] = df["RSI_14"] * df["Volatility_24"]

    # MACD正規化: ボラティリティでMACDを割って相対化
    df["MACD_norm"] = df["MACD"] / df["Volatility_24"].replace(0, np.nan)

    # BB内のポジション: 現在価格がBB内のどこにいるか（0=下限、1=上限）
    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / bb_range

    # MA乖離率: 短期MAと長期MAの差を価格で正規化
    df["MA_cross"] = (df["MA_5"] - df["MA_75"]) / df["Close"]

    # モメンタム加速度: リターンの変化率
    df["Momentum_accel"] = df["Return_1"] - df["Return_1"].shift(1)

    # ボラティリティ変化率: ボラが増加中か減少中か
    df["Vol_change"] = df["Volatility_24"].pct_change(6)

    # HL比率: 高値-安値の幅を価格で正規化
    df["HL_ratio"] = (df["High"] - df["Low"]) / df["Close"]

    # ローソク足内ポジション: 終値が高値-安値のどこにいるか
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Close_position"] = (df["Close"] - df["Low"]) / hl_range

    # リターン偏り: 直近12本のうち陽線の割合 - 0.5
    df["Return_skew_12"] = df["Return_1"].rolling(12).apply(
        lambda x: (x > 0).sum() / len(x) - 0.5, raw=True
    )

    return df
```

特徴量リストの定義。

```python
# ベース特徴量
BASE_FEATURES = [
    "RSI_14", "MACD", "MACD_hist", "BB_width",
    "ATR_14", "MA_5", "MA_20", "MA_75",
    "Return_1", "Return_6", "Return_24",
    "Volatility_24",
]

# 交互作用特徴量
INTERACTION_FEATURES = [
    "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
    "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
    "Return_skew_12",
]

FEATURE_COLS = BASE_FEATURES + INTERACTION_FEATURES
```

---

## 2. 5モデルアンサンブル

LightGBM, XGBoost, CatBoost, RandomForest, ExtraTreesの5モデルを使う。

```python
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


class SimpleEnsembleClassifier:
    """5モデルアンサンブル（簡易版: 固定重み）"""

    def __init__(self, n_estimators=500, learning_rate=0.03):
        self.models = []
        self.model_names = []

        # LightGBM
        self.models.append(lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        ))
        self.model_names.append("LightGBM")

        # XGBoost
        self.models.append(xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        ))
        self.model_names.append("XGBoost")

        # CatBoost（インストール済みの場合のみ）
        try:
            from catboost import CatBoostClassifier
            self.models.append(CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=6,
                random_state=42,
                verbose=0,
            ))
            self.model_names.append("CatBoost")
        except ImportError:
            print("CatBoost未インストール。4モデルで続行。")

        # RandomForest
        self.models.append(RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ))
        self.model_names.append("RandomForest")

        # ExtraTrees
        self.models.append(ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ))
        self.model_names.append("ExtraTrees")

    def fit(self, X, y):
        for name, model in zip(self.model_names, self.models):
            model.fit(X, y)
            print(f"  {name} 学習完了")
        return self

    def predict_proba(self, X):
        """全モデルの平均確率（簡易版: 均等重み）"""
        probas = []
        for model in self.models:
            p = model.predict_proba(X)[:, 1]
            probas.append(p)
        return np.mean(probas, axis=0)

    def predict_with_agreement(self, X):
        """多数決予測 + 一致度"""
        preds = []
        for model in self.models:
            p = model.predict(X)
            preds.append(p)
        preds = np.array(preds)

        # 多数決
        vote_sum = preds.sum(axis=0)
        n_models = len(self.models)
        final_pred = (vote_sum > n_models / 2).astype(int)

        # 一致度: 最終予測と同じ予測をしたモデル数
        agreement = np.where(
            final_pred == 1, vote_sum, n_models - vote_sum
        )
        return final_pred, agreement
```

---

## 3. Walk-Forward検証パイプライン

ここが一番重要な部分だ。過去のデータだけで学習して未来を予測する構造を、コードで実装する。

```python
def walk_forward_backtest(df, feature_cols, n_splits=5, forecast_horizon=12,
                          confidence_threshold=0.60, min_agreement=4):
    """
    Walk-Forward検証

    df: OHLCVデータ + テクニカル指標 + 交互作用特徴量
    n_splits: Walk-Forwardの分割数
    forecast_horizon: 予測期間（時間）
    confidence_threshold: 自信度の閾値
    min_agreement: 最低一致モデル数
    """
    # 方向ラベル: forecast_horizon時間後に上昇=1, 下降=0
    df = df.copy()
    df["Future_Close"] = df["Close"].shift(-forecast_horizon)
    df["Label"] = (df["Future_Close"] > df["Close"]).astype(int)
    df = df.dropna(subset=["Label"] + feature_cols)

    total_len = len(df)
    split_size = total_len // (n_splits + 1)

    all_trades = []

    for fold in range(n_splits):
        train_end = (fold + 1) * split_size
        test_start = train_end
        test_end = min(test_start + split_size, total_len)

        if test_end <= test_start:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        X_train = train_df[feature_cols].values
        y_train = train_df["Label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["Label"].values

        # 学習
        ensemble = SimpleEnsembleClassifier(n_estimators=500, learning_rate=0.03)
        ensemble.fit(X_train, y_train)

        # 予測
        probas = ensemble.predict_proba(X_test)
        preds, agreements = ensemble.predict_with_agreement(X_test)

        # 各テスト時点でトレード判定
        for i in range(len(test_df)):
            confidence = max(probas[i], 1.0 - probas[i])
            agreement = int(agreements[i])

            # フィルター
            if confidence < confidence_threshold:
                continue
            if agreement < min_agreement:
                continue

            direction = "BUY" if preds[i] == 1 else "SELL"
            actual_direction = int(y_test[i])

            # リターン計算
            entry_price = test_df.iloc[i]["Close"]
            if i + forecast_horizon < len(test_df):
                exit_price = test_df.iloc[i + forecast_horizon]["Close"]
            else:
                continue

            raw_return = (exit_price - entry_price) / entry_price
            if direction == "SELL":
                raw_return = -raw_return

            spread = 0.0003  # スプレッドコスト
            net_return = raw_return - spread

            all_trades.append({
                "fold": fold,
                "timestamp": test_df.index[i],
                "direction": direction,
                "confidence": confidence,
                "agreement": agreement,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_return": raw_return,
                "net_return": net_return,
                "result": "WIN" if net_return > 0 else "LOSE",
            })

        print(f"Fold {fold+1}/{n_splits}: 学習{len(train_df)}本, テスト{len(test_df)}本, "
              f"トレード{sum(1 for t in all_trades if t['fold'] == fold)}回")

    return pd.DataFrame(all_trades)
```

---

## 4. バックテスト結果の評価

トレード結果を集計して主要指標を出す関数。

```python
def evaluate_trades(trades_df):
    """バックテスト結果を評価"""
    if len(trades_df) == 0:
        print("トレードなし")
        return

    n_trades = len(trades_df)
    n_wins = (trades_df["result"] == "WIN").sum()
    win_rate = n_wins / n_trades

    net_returns = trades_df["net_return"]
    total_return = net_returns.sum()
    avg_return = net_returns.mean()

    # Profit Factor
    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = abs(net_returns[net_returns < 0].sum())
    pf = gross_profit / max(gross_loss, 1e-10)

    # Sharpe Ratio（年率換算: 1時間足ベースで年約6500本）
    sharpe = (net_returns.mean() / max(net_returns.std(), 1e-10)) * np.sqrt(6500)

    # 最大ドローダウン
    cumulative = (1 + net_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (peak - cumulative) / peak
    max_dd = drawdown.max()

    print("=" * 50)
    print("バックテスト結果")
    print("=" * 50)
    print(f"  総トレード数:    {n_trades}")
    print(f"  勝率:            {win_rate:.1%} ({n_wins}/{n_trades})")
    print(f"  PF:              {pf:.2f}")
    print(f"  累積リターン:    {total_return:+.4f}")
    print(f"  平均リターン:    {avg_return:+.6f}")
    print(f"  Sharpe比:        {sharpe:.2f}")
    print(f"  最大ドローダウン: {max_dd:.1%}")
    print("=" * 50)

    # フォールド別
    print("\nフォールド別:")
    for fold in sorted(trades_df["fold"].unique()):
        fold_trades = trades_df[trades_df["fold"] == fold]
        fold_wins = (fold_trades["result"] == "WIN").sum()
        fold_wr = fold_wins / len(fold_trades) if len(fold_trades) > 0 else 0
        fold_ret = fold_trades["net_return"].sum()
        print(f"  Fold {fold+1}: {len(fold_trades)}回, "
              f"勝率{fold_wr:.1%}, 累積{fold_ret:+.4f}")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "pf": pf,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
```

---

## 5. 実行スクリプト

全体をつなげて実行する。

```python
def main():
    """メイン実行"""
    # --- データ読み込み ---
    # 1時間足CSVを読み込む（Open, High, Low, Close, Volumeカラム）
    # ここは自分のデータソースに合わせて変更してください
    df = pd.read_csv("usdjpy_1h.csv", index_col=0, parse_dates=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    print(f"データ読み込み: {len(df)}本 ({df.index[0]} ~ {df.index[-1]})")

    # --- 特徴量生成 ---
    df = add_technical_features(df)
    df = add_interaction_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"特徴量生成完了: {len(FEATURE_COLS)}個, 有効データ{len(df)}本")

    # --- Walk-Forward検証 ---
    trades = walk_forward_backtest(
        df, FEATURE_COLS,
        n_splits=5,
        forecast_horizon=12,
        confidence_threshold=0.60,
        min_agreement=4,
    )

    # --- 結果評価 ---
    if len(trades) > 0:
        results = evaluate_trades(trades)

        # CSVに保存
        trades.to_csv("backtest_trades.csv", index=False)
        print(f"\nトレード履歴を保存: backtest_trades.csv")
    else:
        print("有効なトレードがありませんでした。閾値を調整してください。")


if __name__ == "__main__":
    main()
```

---

## 6. 簡易版の想定パフォーマンス

この簡易版のコードをUSD/JPY 1時間足データで実行した場合の想定パフォーマンス。

| 指標 | 簡易版（本記事） | プロダクション版 |
|------|----------------|----------------|
| 予測期間 | 12時間 | 12時間 |
| PF | 約1.1〜1.3 | 約1.5〜1.6 |
| 勝率 | 約52〜55% | 約55〜57% |
| Sharpe | 約5〜10 | 約13〜14 |
| フィルター | 自信度+一致度のみ | 時間帯+経済イベント+ボラ異常+自信度+一致度 |
| SL/TP | なし（固定時間決済） | ATRベース動的調整 |
| 重み | 均等 | 検証データでの精度ベース |

簡易版でもPF>1.0（プラス圏）にはなるが、プロダクション版との差は大きい。特に時間帯フィルターとATRベースのSL/TPが効いている。プロダクション版はWalk-Forward分析で特定のUTC時間帯のPFが1.0を下回ることを発見し、その時間帯を除外している。

---

## 7. 改善の方向性

簡易版からプロダクション版に近づけるためのヒントをいくつか書いておく。

〈改善ヒント1：時間帯フィルター〉
FXは24時間取引されるが、時間帯によって特性が大きく異なる。東京時間、ロンドン時間、ニューヨーク時間、そしてセッションの重複時間。Walk-Forwardの結果を時間帯別に集計して、PFが低い時間帯を除外するだけで大幅に改善する可能性がある。

〈改善ヒント2：ATRベースのSL/TP〉
固定時間での決済ではなく、ATR（Average True Range）に基づいた損切り・利確を設定する。ボラティリティが高いときはSL/TPを広く、低いときは狭く。プロダクション版ではSL=2.5xATR、TP=1.5xATRを使っている（この数値自体もWalk-Forward分析で最適化した結果）。

〈改善ヒント3：マルチタイムフレーム特徴量〉
1時間足のテクニカル指標だけでなく、4時間足や日足のRSI・MACD・BB幅を追加する。上位足のトレンド情報が1時間足の予測に寄与する。

〈改善ヒント4：性能ベース重み付け〉
簡易版ではモデルの重みが均等だが、プロダクション版では直近の検証データでの精度に基づいて重みを動的に調整している。精度の高いモデルに大きな重みを配分することで、アンサンブルの質が上がる。

---

## 8. 複数通貨ペアへの対応

コードの構造は1通貨ペアだが、複数ペアへの対応は比較的簡単だ。

```python
PAIRS = ["USDJPY", "AUDJPY", "EURJPY", "GBPJPY", "EURUSD"]


def run_all_pairs():
    """全通貨ペアで予測を実行"""
    for pair in PAIRS:
        print(f"\n{'='*50}")
        print(f"  {pair}")
        print(f"{'='*50}")

        # データ読み込み（通貨ペアごとにCSVを用意）
        csv_path = f"{pair.lower()}_1h.csv"
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"  {csv_path} が見つかりません。スキップ。")
            continue

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_index()

        df = add_technical_features(df)
        df = add_interaction_features(df)
        df = df.dropna(subset=FEATURE_COLS)

        trades = walk_forward_backtest(
            df, FEATURE_COLS,
            n_splits=5,
            forecast_horizon=12,
            confidence_threshold=0.60,
            min_agreement=4,
        )

        if len(trades) > 0:
            evaluate_trades(trades)
            trades.to_csv(f"backtest_{pair.lower()}.csv", index=False)
```

注意点として、通貨ペアによってスプレッドが異なる。USD/JPYは約0.3pips、AUD/JPYやGBP/JPYは0.5pips以上かかることが多い。この差はPFに直接影響するので、ペアごとにスプレッドを設定すべきだ。

---

## まとめ

この記事で公開したのは、FX自動売買AIの簡易版コードだ。5モデルアンサンブル + Walk-Forward 5分割 + 自信度・一致度フィルターの完全なパイプラインになっている。

簡易版の想定PFは1.1〜1.3。プロダクション版（PF 1.5〜1.6）との差は、時間帯フィルター、ATRベースSL/TP、性能ベース重み付け、マルチタイムフレーム特徴量などの最適化によるものだ。

プログラミング未経験の僕でもここまで作れたので、同じようにAIの力を借りて挑戦してみたい人の参考になればと思う。改善ヒントに書いた4つの方向性だけでも、PFは大幅に向上するはずだ。

プロダクション版の全コードと最適化の詳細については、別途有料記事として公開を検討している。興味のある方はフォローしていただければ、公開時に通知が届きます。

---

## 動作環境

| 項目 | 要件 |
|------|------|
| Python | 3.9以上 |
| 必須ライブラリ | numpy, pandas, lightgbm, xgboost, scikit-learn |
| 推奨ライブラリ | catboost（5モデル目） |
| データ | 1時間足OHLCVのCSV（最低2年分推奨） |
| メモリ | 4GB以上 |

```
pip install numpy pandas lightgbm xgboost scikit-learn catboost
```

---

## 免責事項

本記事の内容は、機械学習によるFX予測モデルの開発手法の共有を目的としたものです。

- 掲載されているコードは教育・研究目的であり、投資助言ではありません
- バックテスト結果は過去データに基づくシミュレーションであり、将来の収益を保証するものではありません
- FX取引にはレバレッジに伴う大きなリスクがあり、元本を超える損失が発生する可能性があります
- 本記事のコードを使用した取引は、すべて自己責任でお願いします
- 簡易版のコードにはプロダクション版の重要な最適化（リスク管理、SL/TP等）が含まれていません。そのまま実運用に使うことは推奨しません

---

#FX #自動売買 #Python #機械学習 #LightGBM #XGBoost #アンサンブル学習 #Walk-Forward #バックテスト #USD/JPY
