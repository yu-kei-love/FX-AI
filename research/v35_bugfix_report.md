# v3.5 バグ修正レポート — AI予測モデルの「カンニング」問題

日付: 2026-03-27

## 発見の経緯

仮想通貨モデルの再学習結果が異常に良かった：
- Profit Factor: 4.0〜6.4（プロのヘッジファンドで1.5〜2.0が優秀）
- Sharpe Ratio: 7.5〜12.7（プロで2.0〜3.0が優秀）
- 勝率: 80〜84%
- Total Return: Fold 4で+16,864%（1年弱で資金168倍）

これは明らかに現実離れしており、コードを精査した結果、3つの致命的バグを発見。

---

## バグ1: マルチタイムフレーム特徴量の未来データ漏洩（最も深刻）

### 問題
4時間足・日足のテクニカル指標を1時間足データに結合する際、**まだ確定していないローソク足の情報が混入**していた。

例: 10:00の時点で予測を行う際、08:00〜12:00の4時間足（まだ12:00まで確定しない）のRSI・MACD・BB幅が使われていた。つまり「未来の値動きを見てから予測する」というカンニング状態。

### 修正前のコード
```python
# research/common/features.py (FXモデル共通)
for col in ["RSI_4h", "MACD_hist_4h", "BB_width_4h"]:
    df[col] = df_4h[col].reindex(df.index, method="ffill")

# research/crypto/hybrid_model.py (仮想通貨モデル)
feat = feat.reindex(df.index, method="ffill")
```

### 修正後のコード
```python
# research/common/features.py
for col in ["RSI_4h", "MACD_hist_4h", "BB_width_4h"]:
    shifted = df_4h[col].shift(1)  # 1本前の確定済みデータのみ使用
    df[col] = shifted.reindex(df.index, method="ffill")

# research/crypto/hybrid_model.py
feat = feat.shift(1).reindex(df.index, method="ffill")
```

### 影響範囲
- FXモデル（USDJPY, AUDJPY）: 7つのMTF特徴量すべて
- 仮想通貨モデル: 4h, daily のRSI, MACD, BB, ATR, ADX

---

## バグ2: 連続損失フィルターによるPF水増し（仮想通貨モデル）

### 問題
3連敗後にトレードを一時停止するフィルターがあったが、**停止中の負けトレードがPF計算から除外**されていた。損失が人為的に減るためPFが膨張。

さらに、停止中でも「もし勝っていたら」カウンターをリセットする非対称ロジックがあり、実際より楽観的な結果になっていた。

### 修正前のコード
```python
# 連続損失フィルター
strategy_returns = np.zeros_like(raw_returns)
consec_losses = 0
for i in range(len(raw_returns)):
    if consec_losses >= 3:
        strategy_returns[i] = 0.0  # skip trade
        if raw_returns[i] > 0:  # would have won, reset counter
            consec_losses = 0
    else:
        strategy_returns[i] = raw_returns[i]
        ...

# PFは strategy_returns（フィルター後）で計算 → 損失が消える
gains = strategy_returns[strategy_returns > 0].sum()
losses = abs(strategy_returns[strategy_returns < 0].sum())
```

### 修正後
連続損失フィルターを完全に除去。全トレードの生リターンでPF/Sharpeを計算。
```python
strategy_returns = direction * traded_returns - TRANSACTION_COST
# フィルターなし、全トレードで評価
```

---

## バグ3: Sharpe Ratio の年率換算が27倍に膨張（仮想通貨モデル）

### 問題
12時間足のリターンに対して `sqrt(730)` （≈27倍）をかけて年率換算していた。
計算式自体は数学的に間違いではないが、トレード頻度を考慮していなかったため、実効的なSharpeが大幅に水増しされていた。

### 修正前
```python
periods_per_year = 8760 / FORECAST_HORIZON  # = 730
sharpe = mean / std * np.sqrt(730)  # 27倍の増幅
```

### 修正後
```python
trade_ratio = len(strategy_returns) / max(len(predictions), 1)
effective_trades_per_year = trade_ratio * (8760 / FORECAST_HORIZON)
sharpe = mean / std * np.sqrt(effective_trades_per_year)
```

---

## バグ4: Platt Calibration のテストデータ漏洩（仮想通貨モデル）

### 問題
予測確率のキャリブレーション（Platt Scaling）に**テストセットの正解ラベル**を使っていた。
テストデータの答えを見てから確率を調整するのは、バックテストの信頼性を損なう。

### 修正前
```python
cal_labels = y_test_final[-len(final_pred):]  # テストの正解ラベル
platt_lr.fit(cal_pred.reshape(-1, 1), cal_labels)
```

### 修正後
```python
cal_labels_val = y_val_final[-len(cal_pred_val):]  # バリデーションのラベルのみ
platt_lr.fit(cal_pred_val.reshape(-1, 1), cal_labels_val)
```

---

## 修正前後の比較（仮想通貨モデル Fold 1）

| 指標 | 修正前（カンニングあり） | 修正後（v3.5） |
|------|------------------------|----------------|
| Profit Factor | 6.41 | 2.83 |
| Sharpe Ratio | 7.52 | 1.53 |
| 勝率 | 81.9% | 85.7% |
| トレード数 | 206 | 14 |
| Total Return | +495% | +3.6% |

修正後の数字は現実的。Sharpe 1.53はプロのファンドでも十分な水準だが、
トレード数14はサンプル不足のため、全Fold合計での判断が必要。

---

## 影響を受けたモデル

| モデル | バグ1 (MTF漏洩) | バグ2 (損失フィルター) | バグ3 (Sharpe) | バグ4 (Calibration) |
|--------|:---:|:---:|:---:|:---:|
| FX (USDJPY/AUDJPY) | 該当 | - | - | - |
| 仮想通貨 (BTC) | 該当 | 該当 | 該当 | 該当 |
| ボートレース | - | - | - | - |
| 競馬 | 要確認 | - | - | - |
| 競輪 | 要確認 | - | - | - |

ボートレースは時系列リサンプルを使わないため、未来漏洩の問題なし。

---

## 教訓

1. バックテストの数字が良すぎるときは、まず疑え
2. マルチタイムフレーム分析は未来漏洩の温床 — 必ず確定済みデータのみ使う
3. バックテスト内でトレードをフィルタリングする機能は、PF/Sharpeを歪める
4. テストデータの正解ラベルは、評価以外に絶対使わない
5. どんなに優秀に見える結果も、ペーパートレードで実証するまで信用しない

---

## 修正済みファイル一覧

- `research/common/features.py` — MTF shift(1) 追加
- `research/crypto/hybrid_model.py` — 連続損失フィルター除去、Sharpe修正、Calibration修正、MTF shift(1) 追加
