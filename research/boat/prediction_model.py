# ===========================================
# prediction_model.py
# ボートレースAI - 3段階予測モデル
#
# STAGE1: 各艇の1着確率（LightGBM + NN + LR スタッキング）
# STAGE2: 条件付き2着確率（1着がX艇のとき2着がY艇になる確率）
# STAGE3: 3連単全120通りの確率計算
#
# 注意：データがない状態でもコードを完成させた。
#       学習・評価はデータが揃ってから行う。
#       合成データは絶対に使わない。
# ===========================================

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations

# 必要なライブラリは学習時にインポート（データなし環境での
# インポートエラーを避けるためtry/exceptで囲む）
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================
# STAGE1：1着確率予測
# =============================================================

class Stage1Model:
    """
    各艇の1着確率を予測するモデル。
    LightGBM + NN + ロジスティック回帰をスタッキングで統合。
    """

    def __init__(self):
        self.lgbm_model  = None
        self.nn_model    = None
        self.lr_model    = None
        self.lr_scaler   = None   # LR・NNは正規化が必要
        self.meta_model  = None   # スタッキング用メタモデル
        self.feature_names = None

        # LightGBMのハイパーパラメータ（未検証 → グリッドサーチ必要）
        self.lgbm_params = {
            "objective":       "binary",
            "metric":          "binary_logloss",
            "boosting_type":   "gbdt",
            "num_leaves":      31,      # 未検証
            "max_depth":       -1,
            "learning_rate":   0.05,    # 未検証
            "n_estimators":    500,     # 未検証
            "min_child_samples": 20,    # 未検証
            "subsample":       0.8,     # 未検証
            "colsample_bytree": 0.8,    # 未検証
            "reg_alpha":       0.1,     # 未検証
            "reg_lambda":      0.1,     # 未検証
            "random_state":    42,
            "n_jobs":          -1,
            "verbose":         -1,
        }

    def train(self, X_train, y_train, dates_train):
        """
        スタッキングで3モデルを統合する。

        手順：
        1. Purged K-Fold CVでOOF予測を生成
        2. OOF予測でメタモデルを学習

        Parameters:
            X_train     : 特徴量 DataFrame（shape: n_entries × n_features）
            y_train     : ラベル（1着=1, それ以外=0）
            dates_train : 日付配列（Purged K-Fold用）
        """
        assert HAS_LGB,     "lightgbmが必要です: pip install lightgbm"
        assert HAS_SKLEARN, "scikit-learnが必要です: pip install scikit-learn"

        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.float32)

        # Purged K-Fold CV でOOF予測を生成
        folds = list(purged_kfold_cv(X, y, dates_train, n_splits=5, gap_days=7))

        oof_lgbm = np.zeros(len(X))
        oof_nn   = np.zeros(len(X))
        oof_lr   = np.zeros(len(X))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # --- LightGBM ---
            lgbm = lgb.LGBMClassifier(**self.lgbm_params)
            lgbm.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof_lgbm[val_idx] = lgbm.predict_proba(X_val)[:, 1]

            # --- NN ---
            if HAS_TORCH:
                scaler_nn = StandardScaler()
                X_tr_sc = scaler_nn.fit_transform(X_tr)
                X_val_sc = scaler_nn.transform(X_val)
                nn_model = _SimpleNN(X_tr.shape[1])
                oof_nn[val_idx] = _train_nn(nn_model, X_tr_sc, y_tr, X_val_sc)

            # --- ロジスティック回帰 ---
            scaler_lr = StandardScaler()
            X_tr_sc = scaler_lr.fit_transform(X_tr)
            X_val_sc = scaler_lr.transform(X_val)
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr.fit(X_tr_sc, y_tr)
            oof_lr[val_idx] = lr.predict_proba(X_val_sc)[:, 1]

            print(f"  Fold {fold_idx+1}/5 完了")

        # 全データで各モデルを再学習（推論時に使う）
        self.lgbm_model = lgb.LGBMClassifier(**self.lgbm_params)
        self.lgbm_model.fit(X, y)

        if HAS_TORCH:
            self.lr_scaler = StandardScaler()
            X_sc = self.lr_scaler.fit_transform(X)
            self.nn_model = _SimpleNN(X.shape[1])
            _train_nn(self.nn_model, X_sc, y, X_sc)  # 全データで再学習

        self.lr_scaler = StandardScaler()
        X_sc = self.lr_scaler.fit_transform(X)
        self.lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.lr_model.fit(X_sc, y)

        # メタモデルの学習（OOF予測を入力として）
        meta_X = np.column_stack([oof_lgbm, oof_nn, oof_lr])
        self.meta_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
        )
        self.meta_model.fit(meta_X, y)

        print("STAGE1 学習完了")

    def predict_proba(self, X, race_ids=None):
        """
        各艇の1着確率を返す。
        6艇の確率の合計が1.0になるように正規化。

        Parameters:
            X        : 特徴量（shape: n_entries × n_features）
            race_ids : レースIDの配列（正規化用・Noneの場合は全体で正規化）

        Returns:
            probs: 各艇の1着確率（shape: n_entries）
        """
        assert self.meta_model is not None, "先にtrain()を実行してください"

        X_arr = np.array(X, dtype=np.float32)

        # 各モデルで予測
        p_lgbm = self.lgbm_model.predict_proba(X_arr)[:, 1]

        if self.nn_model and HAS_TORCH:
            X_sc = self.lr_scaler.transform(X_arr)
            p_nn = _predict_nn(self.nn_model, X_sc)
        else:
            p_nn = p_lgbm  # NNなしの場合はLightGBMで代替

        X_sc = self.lr_scaler.transform(X_arr)
        p_lr = self.lr_model.predict_proba(X_sc)[:, 1]

        # メタモデルで統合
        meta_X = np.column_stack([p_lgbm, p_nn, p_lr])
        raw_probs = self.meta_model.predict_proba(meta_X)[:, 1]

        # レース内で確率の合計が1.0になるように正規化
        if race_ids is not None:
            probs = _normalize_within_race(raw_probs, race_ids)
        else:
            probs = raw_probs / raw_probs.sum()

        return probs


# =============================================================
# STAGE2：条件付き2着確率予測
# =============================================================

class Stage2Model:
    """
    「1着がX艇のとき、Y艇が2着になる確率」を予測する。
    STAGE1の結果を特徴量に追加して学習する。
    """

    def __init__(self):
        self.model = None
        self.lgbm_params = {
            "objective":     "binary",
            "metric":        "binary_logloss",
            "num_leaves":    31,      # 未検証
            "learning_rate": 0.05,   # 未検証
            "n_estimators":  300,    # 未検証
            "random_state":  42,
            "n_jobs":        -1,
            "verbose":       -1,
        }

    def train(self, X_train, y_train, stage1_preds, dates_train):
        """
        STAGE1の結果を特徴量に追加して2着確率を学習する。

        Parameters:
            X_train      : 特徴量（1着候補・2着候補のペア）
            y_train      : 2着ラベル（2着=1）
            stage1_preds : STAGE1の1着確率予測
            dates_train  : 日付配列
        """
        assert HAS_LGB, "lightgbmが必要です"

        # STAGE1の予測値を特徴量に追加
        X = np.column_stack([np.array(X_train, dtype=np.float32), stage1_preds])

        self.model = lgb.LGBMClassifier(**self.lgbm_params)
        self.model.fit(X, np.array(y_train, dtype=np.float32))
        print("STAGE2 学習完了")

    def predict_proba(self, X, stage1_preds):
        """
        条件付き2着確率を返す。

        Parameters:
            X            : 特徴量
            stage1_preds : STAGE1の1着確率

        Returns:
            probs: 条件付き2着確率
        """
        assert self.model is not None, "先にtrain()を実行してください"
        X_arr = np.column_stack([np.array(X, dtype=np.float32), stage1_preds])
        return self.model.predict_proba(X_arr)[:, 1]


# =============================================================
# STAGE3：3連単確率計算
# =============================================================

def calc_trifecta_probs(stage1_probs, stage2_probs, lanes=None):
    """
    全120通りの3連単確率を計算する。

    P(X→Y→Z) = P(1着=X) × P(2着=Y|1着=X) × P(3着=Z|1着=X,2着=Y)

    Parameters:
        stage1_probs : dict {lane: 1着確率}
        stage2_probs : dict {(1着lane, 2着lane): 条件付き2着確率}
        lanes        : 艇番リスト（デフォルト: [1,2,3,4,5,6]）

    Returns:
        dict: {"1-2-3": 0.063, "1-3-2": 0.047, ...}
    """
    if lanes is None:
        lanes = [1, 2, 3, 4, 5, 6]

    trifecta_probs = {}

    for combo in permutations(lanes, 3):
        first, second, third = combo

        p_first  = stage1_probs.get(first, 0.0)
        p_second = stage2_probs.get((first, second), 0.0)

        # 3着確率：残り4艇から均等に（簡易計算）
        # 精度向上のためSTAGE3モデルを追加実装することを推奨
        remaining_lanes = [l for l in lanes if l not in (first, second)]
        if third in remaining_lanes:
            p_third = 1.0 / len(remaining_lanes)  # 暫定：均等分配（未検証）
        else:
            p_third = 0.0

        p_combo = p_first * p_second * p_third
        key = f"{first}-{second}-{third}"
        trifecta_probs[key] = p_combo

    # 合計が1.0になるように正規化
    total = sum(trifecta_probs.values())
    if total > 0:
        trifecta_probs = {k: v / total for k, v in trifecta_probs.items()}

    return trifecta_probs


# =============================================================
# バリデーション：Purged K-Fold CV
# =============================================================

def purged_kfold_cv(X, y, dates, n_splits=5, gap_days=7):
    """
    Purged K-Fold CV。
    時系列を守りながら、リーク防止のためギャップを設ける。

    Parameters:
        X        : 特徴量
        y        : ラベル
        dates    : 日付配列（YYYYMMDD形式の文字列 or datetime）
        n_splits : 分割数
        gap_days : トレーニング末尾とバリデーション先頭のギャップ（日数）

    Yields:
        (train_indices, val_indices): インデックスのタプル
    """
    dates_arr = pd.to_datetime(
        pd.Series(dates).astype(str), format="%Y%m%d", errors="coerce"
    ).values

    sorted_idx = np.argsort(dates_arr)
    sorted_dates = dates_arr[sorted_idx]
    n = len(X)
    fold_size = n // n_splits

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end   = (fold + 1) * fold_size if fold < n_splits - 1 else n

        val_idx_sorted  = sorted_idx[val_start:val_end]
        val_date_start  = sorted_dates[val_start]

        # ギャップ: バリデーション開始日からgap_days以上前をtrainに使う
        gap = np.timedelta64(gap_days, "D")
        train_mask = sorted_dates < (val_date_start - gap)
        train_idx_sorted = sorted_idx[train_mask]

        if len(train_idx_sorted) < 100:
            continue  # trainが少なすぎる場合はスキップ

        yield train_idx_sorted, val_idx_sorted


# =============================================================
# ニューラルネット（シンプル）
# =============================================================

class _SimpleNN(object if not HAS_TORCH else nn.Module):
    """
    シンプルな2〜3層NN。
    PyTorchが利用可能な場合のみ機能する。
    """

    def __init__(self, input_dim, hidden_dim=128):
        if not HAS_TORCH:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _train_nn(model, X_train, y_train, X_val, epochs=50, lr=1e-3, batch_size=512):
    """NNを学習してX_valの予測値を返す。"""
    if not HAS_TORCH:
        return np.zeros(len(X_val))

    import torch
    import torch.optim as optim

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_t), batch_size):
            X_batch = X_t[i:i+batch_size]
            y_batch = y_t[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_v).squeeze().numpy()

    return val_pred


def _predict_nn(model, X):
    """学習済みNNで推論する。"""
    if not HAS_TORCH:
        return np.zeros(len(X))

    import torch
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_t).squeeze().numpy()
    return pred


# =============================================================
# ユーティリティ
# =============================================================

def _normalize_within_race(probs, race_ids):
    """レースID内で確率の合計が1.0になるように正規化する。"""
    probs = np.array(probs, dtype=np.float64)
    race_ids = np.array(race_ids)

    for rid in np.unique(race_ids):
        mask = race_ids == rid
        total = probs[mask].sum()
        if total > 0:
            probs[mask] /= total

    return probs
