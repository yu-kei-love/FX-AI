# ===========================================
# model/stage2_model.py
# Stage2: 1着条件付き 2着確率モデル
#
# 現状の Stage1 は「1着確率」しか予測せず、2着・3着は均等仮定。
# Stage2 では「1着が X だった場合の Y の 2着確率」を学習する。
#
# 特徴量:
#   - Y の 61 特徴量 (FEATURE_NAMES)
#   - Y の Stage1 予測確率
#   - X (仮定 1着) の Stage1 予測確率
#   - X-Y の勾配 (delta_grade_score, delta_elo_rating)
#
# 学習データ:
#   1 レース × (N-1) 行 (actual 1着 X 固定、それ以外が Y 候補)
#   target = 1 if Y == actual 2着 else 0
#
# 予測:
#   P(2着=Y | 1着=X) を各 (X,Y) ペアで計算
#
# 注意: data-leak 回避のため、Stage1 の in-sample 予測を使う
#       Stage1Model 内部の stacking と同じ方針
# ===========================================

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


STAGE2_EXTRA_FEATURES = [
    "stage1_prob_self",
    "stage1_prob_fixed",       # 仮定 1着の Stage1 prob
    "delta_stage1_prob",        # self - fixed
    "delta_grade_score",
    "delta_elo_rating",
    "delta_recent_trend",
]


class Stage2Model:
    """条件付き2着確率モデル (LightGBM 単体)"""

    PARAMS = {
        "objective":         "binary",
        "metric":            "auc",
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         5,
        "lambda_l1":         0.1,
        "lambda_l2":         0.1,
        "min_child_samples": 20,
        "feature_fraction":  0.85,
        "bagging_fraction":  0.85,
        "bagging_freq":      5,
        "n_estimators":      500,
        "verbose":           -1,
        "random_state":      42,
        "n_jobs":            -1,
    }

    def __init__(self):
        if not _HAS_LGB:
            raise ImportError("lightgbm が必要")
        self.model = lgb.LGBMClassifier(**self.PARAMS)
        self.is_fitted = False

    def fit(self, X_df, y):
        """
        Parameters:
            X_df: DataFrame of shape (n_pairs, n_feat)
                  列 = 基本特徴量 + STAGE2_EXTRA_FEATURES
            y:    1 if (row's candidate) == actual 2着 else 0
        """
        self.feature_names = list(X_df.columns)
        self.model.fit(X_df.values, np.asarray(y))
        self.is_fitted = True

    def predict_proba(self, X_df):
        if not self.is_fitted:
            raise RuntimeError("fit() 未実行")
        return self.model.predict_proba(X_df.values)[:, 1]

    def get_feature_importance(self):
        imp = self.model.feature_importances_
        pairs = list(zip(self.feature_names, imp))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
