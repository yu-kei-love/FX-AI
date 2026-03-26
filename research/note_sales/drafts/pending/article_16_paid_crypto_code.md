# BTC予測AIの作り方【テクニカル+オンチェーン・コード付き】

この記事では、BTC（ビットコイン）の価格予測AIを実際にPythonで実装する方法を共有する。プログラミング未経験の僕でもここまでできた、そのコードを公開するという趣旨だ。

テクニカル指標とオンチェーンデータを組み合わせた「ハイブリッドモデル」の簡易版（10特徴量）を、ゼロから構築していく。

僕がAI（Claude）と一緒に作って本番環境で動かしているモデルは54特徴量・3層アンサンブル（LightGBM+XGBoost、LSTM-Attention、Simplified TFT → メタアンサンブル）という構成だが、この記事ではその核となるアイデアを10特徴量・LightGBM単体で再現する。コアの考え方は同じなので、ここを理解すればフルモデルへの拡張もやりやすいと思う。

---

## 対象読者

- Pythonを少し触ったことがある人（pandas、numpyが使える程度）
- 仮想通貨のチャートを見たことがある人
- 機械学習について少しでも調べたことがある人

完全な初心者の方は、別の記事（初心者向け：仮想通貨×AIって実際どうなの？）を先に読んでもらうと理解しやすいと思う。僕自身もプログラミング未経験からスタートしたので、できるだけわかりやすく書いたつもりだ。

---

## 全体像

モデルの構造はシンプルだ。

```
データ取得 → 特徴量生成（10個） → ラベル生成 → LightGBMで学習 → Walk-Forward検証
```

フルモデルとの対応関係はこうなる。

| 項目          | この記事の簡易版    | フルモデル（本番）         |
|--------------|-----------------|------------------------|
| 特徴量数       | 10個             | 54個                    |
| モデル         | LightGBM単体     | LightGBM+XGBoost, LSTM-Attention, TFT → メタアンサンブル |
| タイムフレーム   | 1時間足のみ        | 1時間足 + 4時間足 + 日足    |
| オンチェーン     | Funding Rate     | Funding Rate, OI変化, ETH相関 |
| 検証           | 単純Walk-Forward  | Purged Walk-Forward（gap付き） |

---

## Step 1：環境構築

必要なライブラリをインストールする。

```python
pip install pandas numpy lightgbm scikit-learn
```

以下のimportを冒頭に記述する。

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
```

---

## Step 2：データの準備

BTC/USDTの1時間足OHLCVデータが必要だ。Binance APIやCryptoDataDownloadなどから取得できる。

この記事ではCSV形式を前提とする。カラムはOpen, High, Low, Close, Volume、インデックスはDatetimeIndex。

```python
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    # 欠損値の前方補完
    df = df.ffill().dropna()
    return df
```

もしオンチェーンデータ（Funding Rate）も含むCSVを用意できるなら、同じDataFrameにfunding_rateカラムとして含めておく。取得方法は後述する。

---

ここから先は有料パートです。

---

〈有料パート〉

## Step 3：特徴量の生成（10個の厳選特徴量）

フルモデルでは54個の特徴量を使っているが、重要度分析の結果から上位10個を厳選した。この10個だけでもフルモデルの7割程度の性能を再現できる。

```python
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI（相対力指数）を計算"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(close: pd.Series) -> tuple:
    """MACDラインとヒストグラムを計算"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """ATR（真の値幅）を計算"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def compute_bollinger_width(close: pd.Series, period: int = 20,
                            std_mult: float = 2.0) -> pd.Series:
    """ボリンジャーバンド幅を計算"""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return (upper - lower) / sma.replace(0, np.nan)


def garman_klass_vol(open_: pd.Series, high: pd.Series,
                     low: pd.Series, close: pd.Series,
                     window: int = 24) -> pd.Series:
    """Garman-Klassボラティリティ（OHLCを全部使う推定量）"""
    log_hl = np.log(high / low.replace(0, np.nan)) ** 2
    log_co = np.log(close / open_.replace(0, np.nan)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean())
```

特徴量をまとめて生成する関数を用意する。

```python
def build_features(df: pd.DataFrame) -> tuple:
    """10個の特徴量を生成して返す"""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    # 1. RSI
    df["RSI_14"] = compute_rsi(close, 14)

    # 2. MACD（ヒストグラムのみ。MACD本線との相関が高いため）
    _, macd_hist = compute_macd(close)
    df["MACD_hist"] = macd_hist

    # 3. ボリンジャーバンド幅
    df["BB_width"] = compute_bollinger_width(close)

    # 4. ATR（ボラティリティ指標）
    df["ATR_14"] = compute_atr(high, low, close, 14)

    # 5. Garman-Klassボラティリティ
    df["GK_vol"] = garman_klass_vol(open_, high, low, close, 24)

    # 6. 出来高/移動平均比率
    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # 7. RSI × ボラティリティ（交互作用特徴量）
    df["RSI_x_Vol"] = df["RSI_14"] * df["GK_vol"]

    # 8. MACD / ATR（スケール不変モメンタム）
    macd_line, _ = compute_macd(close)
    df["MACD_norm"] = macd_line / df["ATR_14"].replace(0, np.nan)

    # 9. 12時間リターン（ラグ特徴量）
    df["return_lag_12"] = close.pct_change(12)

    # 10. Funding Rate（オンチェーン。データがなければ0で埋める）
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0

    feature_cols = [
        "RSI_14", "MACD_hist", "BB_width", "ATR_14", "GK_vol",
        "volume_ratio", "RSI_x_Vol", "MACD_norm", "return_lag_12",
        "funding_rate"
    ]

    df = df.dropna(subset=feature_cols)
    return df, feature_cols
```

### なぜこの10個なのか

フルモデルの特徴量重要度分析を見て（ここはAIに出力してもらった）、以下の基準で選定した。

| 特徴量          | カテゴリ    | 選定理由                                     |
|----------------|-----------|---------------------------------------------|
| RSI_14         | テクニカル   | 買われすぎ/売られすぎの定番指標。安定して上位         |
| MACD_hist      | テクニカル   | モメンタムの変化を捉える。Signal Lineとの冗長性を排除 |
| BB_width       | テクニカル   | ボラティリティの拡大/縮小を検知                     |
| ATR_14         | テクニカル   | 絶対的なボラティリティ水準                         |
| GK_vol         | ボラティリティ | OHLCを全活用する効率的な推定量。重要度が一貫して高い   |
| volume_ratio   | ボリューム   | 出来高の異常を検知。急騰/急落の前兆                  |
| RSI_x_Vol      | 交互作用    | 「高RSI＋高ボラ」の組み合わせは単独指標より予測力が高い |
| MACD_norm      | 交互作用    | ATRで正規化することでスケール不変に                  |
| return_lag_12  | ラグ        | 12時間前のリターン。短すぎ（1h）はノイズ、長すぎ（24h）は鮮度落ち |
| funding_rate   | オンチェーン  | 先物市場のポジション偏りを反映。仮想通貨固有の情報源    |

---

## Step 4：ラベルの生成

予測ターゲット（ラベル）は「12時間後に価格が上がるか下がるか」の二値分類だ。

ただし、取引コスト以下の小さな値動きはノイズなので除外する（デッドゾーン）。

```python
FORECAST_HORIZON = 12  # 12時間先を予測
TRANSACTION_COST = 0.002  # 0.2%（往復手数料）

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """12時間後のリターンでラベルを生成。デッドゾーン付き"""
    df = df.copy()
    future_return = df["Close"].shift(-FORECAST_HORIZON) / df["Close"] - 1.0
    df["future_return"] = future_return

    # デッドゾーン：取引コスト以下の動きはラベルなし（NaN）
    df["label"] = np.nan
    df.loc[future_return > TRANSACTION_COST, "label"] = 1.0   # 上昇
    df.loc[future_return < -TRANSACTION_COST, "label"] = 0.0  # 下降

    # 異常ボラティリティ期間も除外（上位2%）
    rolling_std = df["Close"].pct_change().rolling(24).std()
    extreme = rolling_std > rolling_std.quantile(0.98)
    df.loc[extreme, "label"] = np.nan

    return df
```

### デッドゾーンの意義

これがないとどうなるか。12時間後に+0.01%しか動かなかったケースも「上昇」としてラベル付けされる。しかし、実際には取引コスト0.2%に負けるため、この取引は赤字だ。

デッドゾーンを設けることで、「取引して意味のある動き」だけを学習対象にできる。フルモデルでもこの設計を採用しており、PFの改善に大きく寄与した。

---

## Step 5：Walk-Forward検証

時系列データの検証で最も重要なルール：未来のデータで学習してはいけない。

これを守るための手法がWalk-Forward検証だ。

```python
def walk_forward_split(n_samples: int, n_splits: int = 5,
                       min_train_ratio: float = 0.3):
    """
    Expanding Window Walk-Forward
    学習データは徐々に増えていき、テストデータは固定サイズ
    """
    min_train = int(n_samples * min_train_ratio)
    remaining = n_samples - min_train
    fold_size = remaining // n_splits

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        if test_end <= test_start:
            continue

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))

        yield train_idx, test_idx
```

フルモデルではtrain/testの間にPurge Gap（60時間の空白）を設けてラベルのリーク（先読み）を防止しているが、簡易版ではまずこの基本形で動かすことを優先する。

### Purge Gapとは

予測ホライズンが12時間の場合、訓練データの末尾12時間分のラベルには「テストデータの価格情報」が含まれている（12時間先の価格で計算しているため）。Purge Gapはこの汚染を防ぐための空白期間だ。

フルモデルでは予測ホライズンの5倍（60時間）をgapとして確保している。簡易版でも精度を上げたい場合は以下のように修正する。

```python
# Purge Gap付きバージョン（精度向上したい場合）
PURGE_GAP = FORECAST_HORIZON * 5  # 60時間

# walk_forward_split の中で
# test_start = train_end + PURGE_GAP
# とするだけでOK
```

---

## Step 6：LightGBMで学習

いよいよモデルの学習だ。

```python
def train_lgb(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: list) -> lgb.Booster:
    """LightGBMの学習"""
    lgb_train = lgb.Dataset(X_train, label=y_train,
                            feature_name=feature_names)
    lgb_val = lgb.Dataset(X_val, label=y_val,
                          feature_name=feature_names, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.01,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "max_depth": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200),
        ],
    )
    return model
```

### パラメータの解説

| パラメータ        | 値     | 意味                                     |
|-----------------|-------|------------------------------------------|
| num_leaves      | 31    | 決定木の葉の数。大きいと過学習リスク            |
| learning_rate   | 0.01  | 学習率。小さいほど慎重に学習（ただし遅い）        |
| feature_fraction| 0.6   | 各ラウンドで使う特徴量の割合。ランダム性の確保     |
| bagging_fraction| 0.6   | 各ラウンドで使うサンプルの割合                  |
| min_child_samples| 100  | 葉ノードの最小サンプル数。過学習防止の重要パラメータ |
| lambda_l1/l2    | 0.5/2.0| L1/L2正則化。大きいほど過学習を抑制            |
| max_depth       | 5     | 木の最大深さ。浅くして汎化性能を確保             |

フルモデルではこのパラメータをそのまま使っている。仮想通貨のようにノイズが多い市場では、過学習を抑制する方向に振るのが鉄則だ。learning_rate=0.1にして高速に学習するより、0.01で2000ラウンド回したほうが汎化性能が高い傾向がある。

---

## Step 7：評価指標の計算

利益を出せるかどうかは、Accuracy（精度）ではなくProfit Factor（PF）で判断する。

```python
def compute_metrics(predictions: list) -> dict:
    """
    predictions: list of dict, 各要素は
      {"pred_proba": float, "label": int, "future_return": float}
    """
    if not predictions:
        return {"pf": 0, "accuracy": 0, "trade_count": 0}

    gains = 0.0
    losses = 0.0
    correct = 0
    total = 0
    trade_count = 0

    for p in predictions:
        prob = p["pred_proba"]
        label = p["label"]
        ret = p["future_return"]

        # Confidence Threshold: 60%以上でのみ取引
        if prob < 0.60 and prob > 0.40:
            continue

        trade_count += 1
        total += 1

        # 予測に基づくリターン
        if prob >= 0.60:  # Long
            trade_return = ret - TRANSACTION_COST
        else:  # Short (prob <= 0.40)
            trade_return = -ret - TRANSACTION_COST

        if trade_return > 0:
            gains += trade_return
            correct += 1
        else:
            losses += abs(trade_return)

    pf = gains / losses if losses > 0 else float("inf")
    accuracy = correct / total if total > 0 else 0

    return {
        "pf": round(pf, 4),
        "accuracy": round(accuracy, 4),
        "trade_count": trade_count,
        "total_gain": round(gains, 6),
        "total_loss": round(losses, 6),
    }
```

### PFとAccuracyの違い

Accuracy 60%でもPFが1.0を割ることはある。逆にAccuracy 45%でもPFが1.5を超えることもある。

重要なのは「当たる回数」ではなく「当たったときの利益と外れたときの損失の比率」だ。PF > 1.0であれば長期的にプラス。PF > 1.5なら良好。PF > 2.0なら優秀。

---

## Step 8：すべてを組み合わせて実行

```python
def run_walk_forward(data_path: str):
    """Walk-Forward検証のメインループ"""
    # データ読み込み
    df = load_data(data_path)
    df, feature_cols = build_features(df)
    df = generate_labels(df)

    # ラベルありのデータだけ使う
    valid_mask = df["label"].notna()
    df_valid = df[valid_mask].copy()

    X = df_valid[feature_cols].values
    y = df_valid["label"].values
    returns = df_valid["future_return"].values

    all_predictions = []

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_split(len(X), n_splits=5)
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        returns_test = returns[test_idx]

        # 訓練データの後半20%をバリデーションに使う
        val_split = int(len(X_train) * 0.8)
        X_tr = X_train[:val_split]
        y_tr = y_train[:val_split]
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]

        # 学習
        model = train_lgb(X_tr, y_tr, X_val, y_val, feature_cols)

        # 予測
        pred_proba = model.predict(X_test)

        # 結果収集
        for j in range(len(X_test)):
            all_predictions.append({
                "pred_proba": pred_proba[j],
                "label": int(y_test[j]),
                "future_return": returns_test[j],
            })

        # フォールドごとの結果
        fold_metrics = compute_metrics(all_predictions[-len(X_test):])
        print(f"Fold {fold_i + 1}: PF={fold_metrics['pf']:.2f}  "
              f"Acc={fold_metrics['accuracy']:.2%}  "
              f"Trades={fold_metrics['trade_count']}")

        # 特徴量重要度の表示
        importance = model.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance
        }).sort_values("importance", ascending=False)
        print(f"  Top 5 features: {imp_df.head().to_string(index=False)}")

    # 全体の結果
    overall = compute_metrics(all_predictions)
    print(f"\nOverall: PF={overall['pf']:.2f}  "
          f"Acc={overall['accuracy']:.2%}  "
          f"Trades={overall['trade_count']}")

    return overall
```

---

## Funding Rateの取得方法

オンチェーンデータの中でも、Funding Rateは最も取得しやすく、予測力も高い。

Binance APIで取得する例を示す。

```python
import requests

def fetch_funding_rate(symbol: str = "BTCUSDT",
                       limit: int = 1000) -> pd.DataFrame:
    """Binance先物のFunding Rateを取得"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    resp = requests.get(url, params=params)
    data = resp.json()

    df = pd.DataFrame(data)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.set_index("fundingTime")
    df = df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})

    # 8時間ごとのデータを1時間足にリサンプル（前方補完）
    df = df.resample("1h").ffill()
    return df
```

Funding Rateは通常8時間ごとに更新される。正の値は「ロングがショートに支払う」（ロング過多）、負の値は「ショートがロングに支払う」（ショート過多）を意味する。

極端に正のFunding Rateは「ロングが過熱している＝反落リスクが高い」というシグナルになりうる。

---

## フルモデルへの拡張ポイント

この簡易版からフルモデルに近づけるための拡張ポイントを挙げておく。

### 拡張1：マルチタイムフレーム

1時間足だけでなく、4時間足と日足でもテクニカル指標を計算する。上位足のトレンドを捉えることで、「短期的には下がっているが、日足レベルでは上昇トレンド」のような情報を取り込める。

フルモデルでは、4時間足と日足それぞれでRSI、MACD、MACD_hist、BB幅、ATR、ADXの6指標を追加している（計12個の追加特徴量）。

### 拡張2：LSTM-Attentionの追加

LightGBMはテーブルデータとして各時点を独立に扱う。しかし時系列データには「順序」に意味がある。直近168時間（1週間分）のシーケンスを入力するLSTM-Attentionを追加することで、時系列パターンを捕捉できる。

フルモデルのLSTM-Attentionは2層LSTM（hidden_size=128）＋Self-Attentionの構成だ。Attentionにより「1週間の中で特にどの時点が重要か」を動的に学習する。

### 拡張3：Simplified TFTの追加

Temporal Fusion Transformer（TFT）はGoogleが提案した時系列予測用のTransformerだ。Variable Selection Networkにより「いまどの特徴量が重要か」を時点ごとに動的に判断する。

フルモデルでは簡略化版を実装している。完全なTFTは複雑すぎるため、Variable Selection + Multi-Head Attention + GRN（Gated Residual Network）のコア部分だけを採用。

### 拡張4：メタアンサンブル

3つのサブモデルの予測確率を、ロジスティック回帰で最適統合する。単純平均よりも、各モデルの得意/不得意を学習して重み付けするほうが性能が良い。

### 拡張5：Profit-Weighted Loss

フルモデルのLightGBMは、カスタム損失関数を使っている。大きなリターンが期待できるケースの誤りをより強くペナルティする。これにより、モデルは「大きく動くケース」の予測精度を優先的に向上させる。

---

## よくある失敗と対策

### 失敗1：リーク（未来情報の漏洩）

最も深刻な失敗。ラベル生成時にshift(-12)を忘れたり、Walk-Forwardを使わず全データでtrain/test splitしたりすると、バックテストは驚異的な成績を出すが、実運用では全く使えない。

対策：Walk-Forwardを必ず使う。Purge Gapも設ける。

### 失敗2：過学習

10特徴量ならリスクは低いが、特徴量を増やしすぎると起きる。バックテストでは良い成績なのに、新しいデータでは全くダメ。

対策：min_child_samples=100、max_depth=5、正則化を強めに。

### 失敗3：取引コストの無視

コスト0%でバックテストすると「PF=2.0」に見えても、0.2%のコストを入れたら「PF=0.8」になることはザラにある。

対策：ラベル生成時にデッドゾーンを設ける。評価時に必ずコストを引く。

### 失敗4：データの偏り

ブルマーケットだけのデータで学習すると「常にロング」のモデルができる。

対策：ベアマーケットも含む十分な期間（最低2年）のデータを使う。

---

## まとめ

この記事で構築した簡易版モデルの構成をまとめる。

| 項目           | 内容                           |
|---------------|-------------------------------|
| 入力データ      | BTC/USDT 1時間足 OHLCV + Funding Rate |
| 特徴量          | 10個（テクニカル7 + ボラティリティ交互作用2 + オンチェーン1） |
| ラベル          | 12時間後の方向（デッドゾーン±0.2%）  |
| モデル          | LightGBM（正則化強め）            |
| 検証            | 5-fold Walk-Forward            |
| 取引フィルタ     | Confidence > 60%でのみエントリー   |

ここから先は、マルチタイムフレーム、LSTM-Attention、TFT、メタアンサンブルを追加していくことで、フルモデルに近づけることができる。

僕みたいなプログラミング未経験者でもここまで作れたので、同じように挑戦してみたい人の参考になればと思う。フルモデルの詳細や、実運用に向けたデータパイプラインの構築については、今後の記事で共有していく予定だ。

---

## 免責事項

本記事の内容は、あくまで機械学習モデルの開発手法の共有を目的としたものです。

- 掲載されているコードを使用した投資結果については一切の責任を負いません
- バックテスト結果は将来の収益を保証しません
- 仮想通貨への投資は元本割れのリスクがあります
- 本記事の情報を元にした投資判断は、すべて自己責任でお願いします
- コードはあくまで教育目的であり、本番環境での使用には十分なテストが必要です

---

#BTC予測 #Python #LightGBM #機械学習 #仮想通貨AI #オンチェーン #Walk-Forward #コード付き #Funding_Rate #データサイエンス
