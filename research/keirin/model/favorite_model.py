# ===========================================
# model/favorite_model.py
# 本命狙い特化モデル (v0.38)
#
# 設計: 本命的中を重視したサンプル重み
#   payout < 1,000円  (本命決着)       : weight 2.0
#   1000 <= payout < 10000 (中穴)       : weight 1.0
#   payout >= 10,000円 (穴)              : weight 0.3
#
# 目的: A_本命 パターン (odds<10) のROIを改善
# ===========================================

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


class FavoriteStage1Model:
    """本命重視 Stage1 (LightGBM 単体)"""

    PARAMS = {
        "objective":          "binary",
        "metric":             "auc",
        "learning_rate":      0.05,
        "num_leaves":         31,
        "lambda_l1":          0.01,
        "lambda_l2":          0.01,
        "min_child_samples":  20,
        "feature_fraction":   0.9,
        "n_estimators":       500,
        "verbosity":          -1,
        "random_state":       42,
        "n_jobs":             -1,
    }

    def __init__(self):
        if not _HAS_LGB:
            raise ImportError("lightgbm が必要")
        self.model = lgb.LGBMClassifier(**self.PARAMS)
        self.is_fitted = False

    def fit(self, X, y, payouts):
        """
        Parameters:
            X: 特徴量 DataFrame
            y: 1着=1, その他=0
            payouts: 各エントリーに対応する 3連単払戻額 Series (円)
                     当該レースの当たり combo payout を全選手に broadcast してOK
        """
        weights = np.ones(len(y), dtype=float)
        pa = np.asarray(payouts, dtype=float)
        weights[pa < 1000] = 2.0
        weights[pa >= 10000] = 0.3
        self.model.fit(X.values, y.values, sample_weight=weights)
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("fit() 未実行")
        return self.model.predict_proba(X.values)[:, 1]

    def get_feature_importance(self, feature_names):
        imp = self.model.feature_importances_
        pairs = list(zip(feature_names, imp))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
