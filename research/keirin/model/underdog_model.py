# ===========================================
# model/underdog_model.py
# 穴狙い特化モデル (v0.38)
#
# 設計: 穴的中を重視したサンプル重み
#   payout >= 10,000円 (穴)   : weight 2.5
#   1000 <= payout < 10000   : weight 1.0
#   payout < 1000円 (本命)     : weight 0.3
#
# 目的: C_穴 パターン (odds>=100) のROIを改善
# ===========================================

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


class UnderdogStage1Model:
    """穴重視 Stage1 (LightGBM 単体)"""

    PARAMS = {
        "objective":          "binary",
        "metric":             "auc",
        "learning_rate":      0.03,
        "num_leaves":         63,
        "lambda_l1":          1.0,
        "lambda_l2":          1.0,
        "min_child_samples":  5,
        "feature_fraction":   0.7,
        "n_estimators":       800,
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
        weights = np.ones(len(y), dtype=float)
        pa = np.asarray(payouts, dtype=float)
        weights[pa >= 10000] = 2.5
        weights[pa < 1000] = 0.3
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
