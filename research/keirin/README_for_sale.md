# 競輪AI予測モデル 設計書

## このコードについて

予測・評価のロジック部分のみです。
データ取得（スクレイピング）は含まれません。

---

## 含まれるファイル

```
model/
├── data_interface.py   データ取得の抽象化層（sqlite/csv/mockモード）
├── line_predictor.py   ライン予測エンジン（競輪の核心）
├── feature_engine.py   50特徴量の計算エンジン
├── prediction_model.py 3段階予測モデル
├── betting_logic.py    EV計算・Kelly基準・フィルター
└── evaluation.py       評価フレームワーク

data/
└── bank_master.py      全43会場のバンクデータ
```

## 含まれないもの

- スクレイピングコード
- 実際のレースデータ
- 学習済みモデルの重み

---

## 動作確認方法（データなし）

```python
import sys
sys.path.insert(0, "model")
sys.path.insert(0, "data")

from data_interface import DataInterface

# mockモードで動作確認（評価・学習には使用禁止）
di = DataInterface(mode="mock")
races   = di.get_races()
entries = di.get_entries()
lines   = di.get_lines()
print(f"races: {len(races)}, entries: {len(entries)}, lines: {len(lines)}")
```

---

## 競輪AIの核心：ライン予測

競輪の特徴は「ライン戦術」。選手が地区別に連携して走るため、
**どの選手がどのラインで走るかを予測することが最重要課題**です。

```python
from line_predictor import LinePredictor

predictor = LinePredictor()

# 3つの情報源を統合
line_probs = predictor.predict_lines(
    entries_df,
    comments_df=comments_df,     # 選手コメント（最高精度）
    reporter_lines=reporter_dict, # 記者予想（高精度）
)
# → {"3-7-4": 0.85, "9-2": 0.75, ...}
```

### 情報源の重み

| 情報源 | 重み | 精度 |
|--------|------|------|
| 選手コメント | 0.6 | 最高（直接の意思表示） |
| 記者予想 | 0.3 | 高（取材ベース） |
| 独自スコア | 0.1 | 中（地区・期別データ） |

---

## 特徴量（50個）

| カテゴリ | 内容 | 数 | 重要度 |
|----------|------|-----|--------|
| A | 選手個人（勝率・脚質・ギア倍数・バック回数） | 12 | 高 |
| **B** | **ライン（位置・人数・信頼度・バック合計）** | **8** | **最重要** |
| C | バンク×脚質（周長・直線・カント・有利度） | 6 | 高 |
| D | 展開予測（逃げ/捲り/差し/荒れリスク） | 4 | 独自 |
| E | 風・天候（屋内は0固定） | 5 | 中 |
| F | オッズ（確定値・変化率・急変フラグ） | 6 | 中 |
| G | レース構成（グレード・レースタイプ） | 5 | 低 |
| H | レース内相対（得点順位・バック順位） | 4 | 中 |

### 最重要特徴量：バック回数（back_count）

バック回数 = 1周目のバックストレートを先頭で通過した回数
→ 逃げ戦術の積極性を示す最重要指標

---

## バンクマスタ（全43会場）

```python
from bank_master import get_bank_info, get_style_advantage

# 会場情報取得
info = get_bank_info("前橋")
# → {"length": 335, "is_dome": True, "escape_rate": 0.32, ...}

# 戦法別有利度
adv = get_style_advantage("平塚", wind_speed=5.0, wind_direction=180)
# → {"escape": 0.30, "makuri": 0.75, "sashi": 0.60}
```

### 屋内会場（前橋・小倉）

風の影響ゼロ。モデル内で `wind_speed = 0` に自動固定されます。

---

## 3段階予測モデル

```
Stage1: 各選手の1着確率
  LightGBM + NN + ロジスティック回帰 → スタッキング
       ↓
Stage2: 条件付き2着確率
       ↓
Stage3: 3連単の全通り確率計算
  7人出走 → 210通り
  8人出走 → 336通り
  9人出走 → 504通り
```

---

## EV計算と買い条件

```python
from betting_logic import calc_ev, kelly_bet

# 控除率25%を必ず考慮
EV = calc_ev(prob=0.05, odds=30.0)  # → 1.125

# EV ≥ 1.1 のみ購入候補
# 予測可能性スコア ≥ 60 のみ購入候補
```

### フィルター一覧

| フィルター | 処理 |
|-----------|------|
| オッズ急上昇（≥20%） | 買い取り消し |
| オッズ急上昇（10〜20%） | 信頼度50%カット |
| 裏切りリスク（≥20%） | 買い取り消し（競輪固有） |
| 裏切りリスク（10〜20%） | 信頼度40%カット（競輪固有） |
| ラインconfidence低（<0.5） | 予測スコア-30点 |
| 単騎3人以上 | 予測スコア-20点 |
| 強風（≥4m/s） | 予測スコア-15点 |

---

## ペーパートレード移行条件

実データで以下を全て満たすまで移行しない：

| 指標 | 条件 |
|------|------|
| ROI | > 0% |
| 最大MDD | < 30% |
| 検証期間 | ≥ 3ヶ月 |
| キャリブレーション誤差 | < 10% |

---

## 免責事項

- 本コードは個人の学習・研究目的です
- 収益を保証するものではありません
- ギャンブルは適切な範囲で楽しんでください
- 過去の成績は将来の利益を保証しません
