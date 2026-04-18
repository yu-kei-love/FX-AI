# ===========================================
# model/prediction_model.py
# 競輪AI - 3段階予測モデル
#
# ボートレースと同じ3段階構造。
# 競輪固有の差：
#   - 出走人数が7〜9人（ボートは6人固定）
#   - 3連単の組み合わせが最大504通り（ボートは120通り）
#   - ラインの信頼度を特徴量として明示的に使う
#
# 注意：データがない状態でもコードを完成させた。
#       学習・評価はデータが揃ってから行う。
# ===========================================

import itertools
from typing import Optional

import numpy as np
import pandas as pd

# LightGBM/sklearn は実行時にのみ必要。importエラーを防ぐためtry/exceptで囲む
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from feature_engine import FEATURE_NAMES


# =============================================================
# Stage1：各選手の1着確率
# =============================================================

class Stage1Model:
    """
    各選手の1着確率を予測する。

    アンサンブル：LightGBM + SimpleNN + ロジスティック回帰
    → スタッキングで統合
    """

    LGB_PARAMS = {
        "objective":   "binary",
        "metric":      "auc",
        "num_leaves":  63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, tuned_params: dict = None, tuned_label: str = None):
        """
        Parameters:
            tuned_params: LGBMClassifier に渡す override パラメータ
                          （指定時は LGB_PARAMS より優先）
            tuned_label: "normal"/"midnight" を指定すると
                         models/best_params_{label}.json を自動読み込み
        """
        if not _HAS_LGB or not _HAS_SKLEARN:
            raise ImportError("lightgbm と scikit-learn が必要です")

        # tuned_label 指定で JSON から読み込み
        if tuned_label and not tuned_params:
            import os
            import json as _json
            _here = os.path.dirname(os.path.abspath(__file__))
            _json_path = os.path.join(
                _here, "..", "models", f"best_params_{tuned_label}.json"
            )
            if os.path.exists(_json_path):
                with open(_json_path, "r", encoding="utf-8") as _f:
                    tuned_params = _json.load(_f).get("best_params", {})

        # パラメータ決定: LGB_PARAMS + tuned_params (override)
        lgb_params = dict(self.LGB_PARAMS)
        if tuned_params:
            lgb_params.update(tuned_params)
        self._lgb_params_used = lgb_params
        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.nn_model  = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            random_state=42,
        )
        self.lr_model  = LogisticRegression(max_iter=500, random_state=42)
        self.scaler    = StandardScaler()
        self.meta_lr   = LogisticRegression(max_iter=200)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        学習する。

        Parameters:
            X: 特徴量DataFrame（行=選手×レース）
            y: 1着=1, それ以外=0
        """
        X_np = X[FEATURE_NAMES].fillna(0).values
        X_scaled = self.scaler.fit_transform(X_np)

        self.lgb_model.fit(X_np, y)
        self.nn_model.fit(X_scaled, y)
        self.lr_model.fit(X_scaled, y)

        # スタッキング用のメタ特徴量
        meta_X = np.column_stack([
            self.lgb_model.predict_proba(X_np)[:, 1],
            self.nn_model.predict_proba(X_scaled)[:, 1],
            self.lr_model.predict_proba(X_scaled)[:, 1],
        ])
        self.meta_lr.fit(meta_X, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """1着確率を予測する（0〜1）"""
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。fit()を先に呼んでください。")
        X_np     = X[FEATURE_NAMES].fillna(0).values
        X_scaled = self.scaler.transform(X_np)

        meta_X = np.column_stack([
            self.lgb_model.predict_proba(X_np)[:, 1],
            self.nn_model.predict_proba(X_scaled)[:, 1],
            self.lr_model.predict_proba(X_scaled)[:, 1],
        ])
        return self.meta_lr.predict_proba(meta_X)[:, 1]

    def get_feature_importance(self):
        """
        LightGBMの特徴量重要度を返す。

        Returns:
            list[(str, float)]: [(特徴量名, スコア), ...] 降順ソート済み
        """
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません")
        importance = self.lgb_model.feature_importances_
        pairs = list(zip(FEATURE_NAMES, importance))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


# =============================================================
# Stage2：条件付き2着確率
# =============================================================

class Stage2Model:
    """
    条件付き2着確率を予測する。

    「選手Bが2着になる確率 | 選手Aが1着のとき」
    → ライン構成を考慮して補正する
    """

    LGB_PARAMS = {
        "objective":    "binary",
        "metric":       "auc",
        "num_leaves":   31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "verbose": -1,
    }

    def __init__(self):
        if not _HAS_LGB:
            raise ImportError("lightgbm が必要です")
        self.models = {}  # 1着選手ごとに別モデルを持つ（マルチモデル）
        self.shared_model = lgb.LGBMClassifier(**self.LGB_PARAMS)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        学習する。

        Parameters:
            X: 特徴量DataFrame（B-01_line_position など条件付き特徴量を含む）
            y: 2着=1, それ以外=0
        """
        X_np = X[FEATURE_NAMES].fillna(0).values
        self.shared_model.fit(X_np, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """条件付き2着確率を返す"""
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。fit()を先に呼んでください。")
        X_np = X[FEATURE_NAMES].fillna(0).values
        return self.shared_model.predict_proba(X_np)[:, 1]


# =============================================================
# 3連単確率の計算
# =============================================================

def calc_trifecta_probs(
    stage1_probs: dict,
    stage2_probs: dict,
    n_entries: int,
) -> dict:
    """
    全通りの3連単確率を計算する。

    出走人数別の組み合わせ数：
    - 7人：7 × 6 × 5 = 210通り
    - 8人：8 × 7 × 6 = 336通り
    - 9人：9 × 8 × 7 = 504通り

    Parameters:
        stage1_probs: {car_no: 1着確率}
        stage2_probs: {(1着car_no, 2着car_no): 条件付き2着確率}
        n_entries   : 出走人数

    Returns:
        {(1st, 2nd, 3rd): 確率} の dict
    """
    car_nos = list(stage1_probs.keys())

    # 出走人数の検証
    if n_entries not in (7, 8, 9):
        raise ValueError(f"n_entriesは7〜9人です: {n_entries}")

    trifecta_probs = {}

    for combo in itertools.permutations(car_nos, 3):
        first, second, third = combo

        # P(1st=A) × P(2nd=B|1st=A) × P(3rd=C|1st=A,2nd=B)
        p1 = stage1_probs.get(first, 0.0)
        p2_given_p1 = stage2_probs.get((first, second), 0.0)
        # 3着は残りの確率から簡略計算（完全な条件付き確率モデルは実データで改善）
        remaining_prob = 1.0 - p1 - p2_given_p1 * p1
        p3_given_p1p2  = stage2_probs.get((second, third), 0.0) * remaining_prob

        trifecta_probs[combo] = p1 * p2_given_p1 * p3_given_p1p2

    # 確率の合計が1になるよう正規化
    total = sum(trifecta_probs.values())
    if total > 0:
        trifecta_probs = {k: v / total for k, v in trifecta_probs.items()}

    return trifecta_probs


def _normalize_within_race(probs: dict) -> dict:
    """レース内の確率の合計を1に正規化する"""
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: v / total for k, v in probs.items()}


# =============================================================
# Purged K-Fold CV（時系列対応）
# =============================================================

def purged_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_splits: int = 5,
    gap_days: int = 7,
):
    """
    Purged K-Fold Cross Validation（時系列安全版）。

    未来リークを防ぐため：
    1. 時系列順にデータを分割する
    2. トレーニング末尾〜テスト先頭の間に gap_days 分の空白を設ける

    Parameters:
        X         : 特徴量DataFrame
        y         : ターゲット
        dates     : 各行の日付（"YYYYMMDD" 形式の文字列）
        n_splits  : 分割数
        gap_days  : ギャップ日数（未来リーク防止）

    Yields:
        (train_idx, test_idx)
    """
    dates_sorted = pd.to_datetime(dates, format="%Y%m%d")
    sorted_indices = np.argsort(dates_sorted.values)

    fold_size = len(sorted_indices) // n_splits

    for fold in range(n_splits):
        # テスト範囲
        test_start  = fold * fold_size
        test_end    = (fold + 1) * fold_size if fold < n_splits - 1 else len(sorted_indices)
        test_idx    = sorted_indices[test_start:test_end]
        test_dates  = dates_sorted.iloc[test_idx]

        # ギャップを設けてトレーニングデータを選択
        cutoff_date = test_dates.min() - pd.Timedelta(days=gap_days)
        train_idx   = sorted_indices[dates_sorted.iloc[sorted_indices] < cutoff_date]

        if len(train_idx) == 0:
            continue

        yield list(train_idx), list(test_idx)


# =============================================================
# ウォークフォワード評価
# =============================================================

def walk_forward_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_class,
    n_splits: int = 5,
) -> list:
    """
    ウォークフォワード評価を実行する。

    Parameters:
        X          : 特徴量DataFrame
        y          : ターゲット
        dates      : 日付Series
        model_class: Stage1Model または Stage2Model クラス
        n_splits   : 分割数

    Returns:
        results: [{"fold": int, "auc": float, "n_train": int, "n_test": int}]
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("scikit-learn が必要です")

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        purged_kfold_cv(X, y, dates, n_splits=n_splits)
    ):
        model = model_class()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict_proba(X.iloc[test_idx])
        auc = roc_auc_score(y.iloc[test_idx], y_pred)
        results.append({
            "fold":    fold_idx + 1,
            "auc":     round(float(auc), 4),
            "n_train": len(train_idx),
            "n_test":  len(test_idx),
        })

    return results
