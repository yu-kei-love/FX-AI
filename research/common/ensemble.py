"""
アンサンブル（多数決）モデルモジュール

「アンサンブル」= 複数のモデルを組み合わせて予測する方法

1人の専門家より5人の多数決の方が安定する、という考え方。
それぞれ得意分野が違うモデルを組み合わせることで、
1つのモデルが間違えても他がカバーできる。

使うモデル（5人）:
  - LightGBM:     勾配ブースティング。葉っぱ優先で深く掘る
  - XGBoost:      勾配ブースティング。階層優先で広く見る
  - CatBoost:     勾配ブースティング。カテゴリ変数に強い。独自の順序付きブースティング
  - RandomForest: ランダムな決定木300本の多数決。過学習しにくい
  - ExtraTrees:   RandomForestの亜種。分割点をランダムに選ぶ。さらに多様性が高い

5人のモデルが「買い」「売り」を投票する。
  - 5人一致（全員同意見）→ 最高の自信度
  - 4人一致 → 高い自信度
  - 3人一致 → 通常の自信度
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


class EnsembleClassifier:
    """5つのモデルの多数決で予測するクラス"""

    def __init__(self, lgb_params=None, n_estimators=200, learning_rate=0.05):
        self.lgb_params = lgb_params or {}
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        # 5つのモデルを用意
        self.model_lgb = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            verbosity=-1,
            **self.lgb_params,
        )
        self.model_xgb = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model_cat = CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        self.model_rf = RandomForestClassifier(
            n_estimators=min(n_estimators, 300),
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        self.model_et = ExtraTreesClassifier(
            n_estimators=min(n_estimators, 300),
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

        self.models = [
            self.model_lgb,
            self.model_xgb,
            self.model_cat,
            self.model_rf,
            self.model_et,
        ]
        self.model_names = [
            "LightGBM", "XGBoost", "CatBoost", "RandomForest", "ExtraTrees",
        ]

    def fit(self, X, y):
        """5つのモデルを同じデータで学習する"""
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """多数決で予測する（過半数が一致した方向）"""
        predictions = np.array([m.predict(X) for m in self.models])
        vote = predictions.mean(axis=0)
        return (vote > 0.5).astype(int)

    def predict_proba(self, X):
        """5つのモデルの確率の平均を返す"""
        probas = np.array([m.predict_proba(X) for m in self.models])
        return probas.mean(axis=0)

    def predict_with_agreement(self, X):
        """予測 + 一致度を返す

        Returns:
            predictions: 多数決の予測結果 (0 or 1)
            agreement: 一致した人数 (3, 4, or 5)
        """
        predictions = np.array([m.predict(X) for m in self.models])
        vote_sum = predictions.sum(axis=0)  # 0~5: 「買い」に投票した人数
        final_pred = (vote_sum >= 3).astype(int)  # 3人以上が買い → 買い

        # 一致人数: 最終予測と同じ方向に投票した人数
        agreement = np.where(
            final_pred == 1,
            vote_sum,           # 買い予測 → 買いに投票した人数
            5 - vote_sum,       # 売り予測 → 売りに投票した人数
        )
        return final_pred, agreement
