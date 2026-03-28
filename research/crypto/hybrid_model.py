"""
Hybrid Cryptocurrency Prediction Model  v3.3 pruned
=====================================================
3-layer ensemble: Feature Engineering -> Sub-Models (LightGBM/XGBoost, LSTM-Attention, TFT) -> Meta-Ensemble

Usage:
    # Training
    results = train_and_evaluate("path/to/btc_1h.csv")

    # Prediction
    signal, confidence, position_size = predict(latest_df)
"""

import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crypto_hybrid")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TRANSACTION_COST = 0.002  # 0.2% round-trip (stricter: spread + fees)
FORECAST_HORIZON = 12  # hours (was 4; 12h was the breakthrough for FX model)
CONFIDENCE_THRESHOLD = 0.60  # lowered from 0.75 to get more trades
SEQUENCE_LENGTH = 168  # 1 week of lookback for neural models (was 48)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================================
# Layer 1: Feature Engineering
# ============================================================================

# ---- Technical Indicators ----

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = _sma(close, period)
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, width


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = compute_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def compute_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad).replace(0, np.nan)


def compute_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + mfr))


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


# ---- Volatility Estimators ----

def realized_volatility(close: pd.Series, window: int = 24) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(window)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
    log_hl = np.log(high / low.replace(0, np.nan)) ** 2
    return np.sqrt(log_hl.rolling(window).mean() / (4 * np.log(2)))


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24
) -> pd.Series:
    log_hl = np.log(high / low.replace(0, np.nan)) ** 2
    log_co = np.log(close / open_.replace(0, np.nan)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean())


# ---- Multi-Timeframe Resampling ----

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h data to a higher timeframe."""
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    resampled = df.resample(rule).agg(agg).dropna()
    return resampled


def add_multi_timeframe_features(
    df: pd.DataFrame, suffix: str, rule: str
) -> pd.DataFrame:
    """Compute indicators on a resampled timeframe and merge back to 1h."""
    resampled = _resample_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]], rule)
    close = resampled["Close"]
    high = resampled["High"]
    low = resampled["Low"]
    volume = resampled["Volume"]

    feat = pd.DataFrame(index=resampled.index)
    feat[f"RSI_{suffix}"] = compute_rsi(close, 14)
    macd_l, macd_s, macd_h = compute_macd(close)
    feat[f"MACD_{suffix}"] = macd_l
    feat[f"MACD_hist_{suffix}"] = macd_h
    _, _, bb_w = compute_bollinger_bands(close)
    feat[f"BB_width_{suffix}"] = bb_w
    feat[f"ATR_{suffix}"] = compute_atr(high, low, close, 14)
    feat[f"ADX_{suffix}"] = compute_adx(high, low, close, 14)

    # v3.5 修正: 未確定ローソク足の未来漏洩を防止するため、1本前にshift
    # 現在進行中のリサンプル足はまだ確定していないので、確定済みの前回値を使う
    feat = feat.shift(1).reindex(df.index, method="ffill")
    for col in feat.columns:
        df[col] = feat[col]
    return df


# ---- Full Feature Pipeline ----

FEATURE_COLS: List[str] = []  # populated by build_features


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the complete feature set for crypto prediction.
    Expects df with columns: Open, High, Low, Close, Volume and a DatetimeIndex.
    Returns (df_with_features, feature_column_names).
    """
    df = df.copy()
    feature_names: List[str] = []

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    # --- 1h Technical Indicators ---
    df["RSI_14"] = compute_rsi(close, 14)
    macd_l, macd_s, macd_h = compute_macd(close)
    df["MACD"] = macd_l
    df["MACD_signal"] = macd_s
    df["MACD_hist"] = macd_h
    bb_u, bb_l, bb_w = compute_bollinger_bands(close)
    df["BB_upper"] = bb_u
    df["BB_lower"] = bb_l
    df["BB_width"] = bb_w
    df["ATR_14"] = compute_atr(high, low, close, 14)
    df["ADX_14"] = compute_adx(high, low, close, 14)
    stoch_k, stoch_d = compute_stochastic(high, low, close, 14, 3)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d
    df["Williams_R"] = compute_williams_r(high, low, close, 14)
    df["CCI"] = compute_cci(high, low, close, 20)
    df["MFI"] = compute_mfi(high, low, close, volume, 14)
    df["OBV"] = compute_obv(close, volume)
    # Normalize OBV to percentage changes for stationarity
    df["OBV_pct"] = df["OBV"].pct_change(1)

    # v3.3 pruned: Removed MACD_signal (correlated w/ MACD), BB_upper/BB_lower
    #   (keep BB_width + BB_position), Williams_R (near-zero importance)
    feature_names.extend([
        "RSI_14", "MACD", "MACD_hist",
        "BB_width",
        "ATR_14", "ADX_14", "Stoch_K", "Stoch_D",
        "CCI", "MFI", "OBV_pct",
    ])

    # --- Multi-timeframe (4h, 1d) ---
    df = add_multi_timeframe_features(df, "4h", "4h")
    df = add_multi_timeframe_features(df, "1d", "1D")
    feature_names.extend([
        "RSI_4h", "MACD_4h", "MACD_hist_4h", "BB_width_4h", "ATR_4h", "ADX_4h",
        "RSI_1d", "MACD_1d", "MACD_hist_1d", "BB_width_1d", "ATR_1d", "ADX_1d",
    ])

    # --- Volume Profile ---
    vol_ma20 = volume.rolling(20).mean()
    df["volume_ma_ratio"] = volume / vol_ma20.replace(0, np.nan)
    df["volume_spike"] = (volume > 2 * vol_ma20).astype(int)
    # Approximate buy/sell using close vs open
    df["buy_sell_volume_ratio"] = np.where(
        close >= open_,
        volume,
        -volume,
    )
    df["buy_sell_volume_ratio"] = (
        df["buy_sell_volume_ratio"].rolling(14).sum()
        / volume.rolling(14).sum().replace(0, np.nan)
    )
    # v3.3 pruned: volume_spike removed (near-zero importance)
    feature_names.extend(["volume_ma_ratio", "buy_sell_volume_ratio"])

    # --- Volatility ---
    df["realized_vol"] = realized_volatility(close, 24)
    df["parkinson_vol"] = parkinson_volatility(high, low, 24)
    df["garman_klass_vol"] = garman_klass_volatility(open_, high, low, close, 24)
    feature_names.extend(["realized_vol", "parkinson_vol", "garman_klass_vol"])

    # --- Crypto-Specific (filled from merged auxiliary data if available) ---
    # v3.3 pruned: fear_greed_index removed (near-zero importance)
    crypto_features = ["funding_rate", "funding_rate_ma24", "open_interest_change"]
    for col in crypto_features:
        if col in df.columns and df[col].notna().sum() > 100:
            feature_names.append(col)
        elif col not in df.columns:
            df[col] = 0.0  # neutral default when data unavailable
    # Still compute fear_greed_index column if present, just don't use as feature
    if "fear_greed_index" not in df.columns:
        df["fear_greed_index"] = 0.0

    # --- Cross-Asset ---
    if "ETH_Close" in df.columns:
        df["eth_btc_corr"] = (
            df["Close"].pct_change().rolling(24).corr(df["ETH_Close"].pct_change())
        )
        df["eth_momentum"] = df["ETH_Close"].pct_change(12)
        feature_names.extend(["eth_btc_corr", "eth_momentum"])

    # --- Lag Features ---
    # v3.3 pruned: return_lag_1, return_lag_3 removed (near-zero importance)
    for lag in [2, 6, 12, 24]:
        col_name = f"return_lag_{lag}"
        df[col_name] = close.pct_change(lag)
        feature_names.append(col_name)
    # Still compute pruned lags for potential downstream use
    for lag in [1, 3]:
        df[f"return_lag_{lag}"] = close.pct_change(lag)

    # --- Time Features (cyclic encoding for 24/7 market) ---
    hour = df.index.hour
    dow = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feature_names.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])

    # --- Moving Average ratios ---
    for w in [5, 10, 20, 50]:
        col_name = f"ma_ratio_{w}"
        df[col_name] = close / _sma(close, w).replace(0, np.nan) - 1.0
        feature_names.append(col_name)

    # --- Interaction Features (from FX v3 — improved PF from 1.03 to 1.14) ---
    # RSI x Volatility: high RSI in high vol = stronger signal
    df["RSI_x_Vol"] = df["RSI_14"] * df["realized_vol"]
    # MACD normalized by ATR: scale-invariant momentum
    df["MACD_norm"] = df["MACD"] / df["ATR_14"].replace(0, np.nan)
    # Bollinger Band position: where price sits within the bands (0=lower, 1=upper)
    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_position"] = (close - df["BB_lower"]) / bb_range
    # MA cross: short vs long MA ratio
    ma_short = _sma(close, 10)
    ma_long = _sma(close, 50)
    df["MA_cross"] = ma_short / ma_long.replace(0, np.nan) - 1.0
    # Momentum acceleration: 2nd derivative of price
    mom_12 = close.pct_change(12)
    df["Momentum_accel"] = mom_12 - mom_12.shift(12)
    # Volume change rate
    df["Vol_change"] = volume.pct_change(12)
    # High-Low ratio: intrabar volatility proxy
    df["HL_ratio"] = (high - low) / close.replace(0, np.nan)
    # Close position within bar: buying/selling pressure proxy
    hl_range = (high - low).replace(0, np.nan)
    df["Close_position"] = (close - low) / hl_range
    # Return skewness over 12 periods
    df["Return_skew_12"] = close.pct_change(1).rolling(12).apply(
        lambda x: x.skew() if len(x) > 2 else 0.0, raw=False
    )
    feature_names.extend([
        "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
        "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
        "Return_skew_12",
    ])

    global FEATURE_COLS
    FEATURE_COLS = feature_names
    return df, feature_names


# ---- Label Generation ----

def generate_labels(df: pd.DataFrame, horizon: int = FORECAST_HORIZON, cost: float = TRANSACTION_COST) -> pd.DataFrame:
    """Generate binary label with dead-zone: skip ambiguous moves near zero."""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
    df["future_return"] = future_return
    # Dead-zone: only label clear moves (above cost = UP, below -cost = DOWN, else NaN)
    df["label"] = np.nan
    df.loc[future_return > cost, "label"] = 1.0
    df.loc[future_return < -cost, "label"] = 0.0
    # Add volatility filter: skip extremely volatile periods (>5 std)
    rolling_std = df["Close"].pct_change().rolling(24).std()
    extreme_vol = rolling_std > rolling_std.quantile(0.98)
    df.loc[extreme_vol, "label"] = np.nan
    return df


# ============================================================================
# Layer 2: Sub-Models
# ============================================================================

# ---------------------------------------------------------------------------
# Model A: LightGBM + XGBoost Ensemble
# ---------------------------------------------------------------------------

def _profit_weighted_logloss(y_pred: np.ndarray, dtrain: lgb.Dataset):
    """
    Custom LightGBM objective: profit-weighted binary cross-entropy.
    Penalizes confident wrong predictions more when the actual return is large.
    This encourages the model to be correct on high-magnitude moves.
    """
    y_true = dtrain.get_label()
    returns = dtrain.get_weight()  # We'll pass |future_return| as weight

    # Sigmoid to convert raw predictions to probabilities
    preds = 1.0 / (1.0 + np.exp(-y_pred))
    preds = np.clip(preds, 1e-7, 1.0 - 1e-7)

    # Profit-weighted gradient: scale by return magnitude
    # Larger returns get more weight so the model focuses on big moves
    weight = 1.0 + np.abs(returns) * 10.0  # amplify return signal

    # Standard logloss gradient and hessian, scaled by weight
    grad = weight * (preds - y_true)
    hess = weight * preds * (1.0 - preds)

    return grad, hess


def _profit_weighted_eval(y_pred: np.ndarray, dtrain: lgb.Dataset):
    """Custom eval metric: profit-weighted accuracy."""
    y_true = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-y_pred))
    pred_labels = (preds > 0.5).astype(int)
    correct = (pred_labels == y_true).astype(float)
    returns = dtrain.get_weight()
    # Weight accuracy by return magnitude
    weight = 1.0 + np.abs(returns) * 10.0
    weighted_acc = (correct * weight).sum() / weight.sum()
    return "profit_weighted_acc", weighted_acc, True  # True = higher is better


class GBMEnsemble:
    """Gradient boosting ensemble (LightGBM + XGBoost)."""

    def __init__(self):
        self.lgb_model: Optional[lgb.Booster] = None
        self.xgb_model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self._used_custom_obj: bool = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        returns_train: Optional[np.ndarray] = None,
        returns_val: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = feature_names
        self._used_custom_obj = returns_train is not None and len(returns_train) == len(y_train)
        logger.info("Training LightGBM...")

        # Use profit-weighted objective if returns are available
        use_custom_obj = returns_train is not None and len(returns_train) == len(y_train)
        if use_custom_obj:
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=np.abs(returns_train), feature_name=feature_names)
            lgb_val = lgb.Dataset(X_val, label=y_val, weight=np.abs(returns_val) if returns_val is not None else None, feature_name=feature_names, reference=lgb_train)
            logger.info("Using profit-weighted custom loss function")
        else:
            lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=lgb_train)

        lgb_params = {
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
            "seed": SEED,
        }
        if use_custom_obj:
            lgb_params["objective"] = _profit_weighted_logloss
            lgb_params["metric"] = "None"
        else:
            lgb_params["objective"] = "binary"
            lgb_params["metric"] = "binary_logloss"

        train_kwargs = {
            "params": lgb_params,
            "train_set": lgb_train,
            "num_boost_round": 2000,
            "valid_sets": [lgb_val],
            "callbacks": [
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200),
            ],
        }
        if use_custom_obj:
            train_kwargs["feval"] = _profit_weighted_eval

        self.lgb_model = lgb.train(**train_kwargs)
        logger.info(f"LightGBM best iteration: {self.lgb_model.best_iteration}")

        logger.info("Training XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,  # reduced from 6
            "learning_rate": 0.01,  # slower learning
            "subsample": 0.6,  # more aggressive subsampling
            "colsample_bytree": 0.6,
            "min_child_weight": 100,  # increased from 50
            "reg_alpha": 0.5,  # stronger L1
            "reg_lambda": 2.0,  # stronger L2
            "gamma": 0.1,  # minimum loss reduction for split
            "seed": SEED,
            "verbosity": 0,
        }

        self.xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=200,
        )
        logger.info(f"XGBoost best iteration: {self.xgb_model.best_iteration}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lgb_raw = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)
        # If custom objective was used, raw output needs sigmoid
        if self._used_custom_obj:
            lgb_pred = 1.0 / (1.0 + np.exp(-lgb_raw))
        else:
            lgb_pred = lgb_raw
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        xgb_pred = self.xgb_model.predict(dmat, iteration_range=(0, self.xgb_model.best_iteration))
        return 0.5 * (lgb_pred + xgb_pred)

    def save(self, path: Path) -> None:
        self.lgb_model.save_model(str(path / "lgb_model.txt"))
        self.xgb_model.save_model(str(path / "xgb_model.json"))
        with open(path / "gbm_feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)
        with open(path / "gbm_config.pkl", "wb") as f:
            pickle.dump({"used_custom_obj": self._used_custom_obj}, f)

    def load(self, path: Path) -> None:
        self.lgb_model = lgb.Booster(model_file=str(path / "lgb_model.txt"))
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(path / "xgb_model.json"))
        with open(path / "gbm_feature_names.pkl", "rb") as f:
            self.feature_names = pickle.load(f)
        config_path = path / "gbm_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)
            self._used_custom_obj = config.get("used_custom_obj", False)
        else:
            # Legacy models without config: assume custom obj was used
            self._used_custom_obj = True
            logger.warning("No gbm_config.pkl found, assuming custom objective was used")


# ---------------------------------------------------------------------------
# Model B: BiLSTM with Self-Attention (PyTorch)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over sequence dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V)
        return context


class LSTMAttentionModel(nn.Module):
    """2-layer BiLSTM with self-attention for sequence classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_dim * 2)  # *2 for bidirectional
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        attn_out = self.attention(lstm_out)  # (batch, seq_len, hidden*2)
        attn_out = self.layer_norm(attn_out + lstm_out)  # residual connection

        # Pool over sequence: use last timestep + mean
        last_hidden = attn_out[:, -1, :]
        mean_hidden = attn_out.mean(dim=1)
        pooled = last_hidden + mean_hidden  # (batch, hidden*2)

        out = self.dropout(F.relu(self.fc1(pooled)))
        out = torch.sigmoid(self.fc2(out)).squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# Model C: Simplified Temporal Fusion Transformer (PyTorch)
# ---------------------------------------------------------------------------

class VariableSelectionNetwork(nn.Module):
    """Selects and weights input features via a gating mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.flattened_grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        weights = self.gate(x)  # per-feature importance
        weighted = x * weights
        transformed = self.flattened_grn(weighted)
        return transformed


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) block from TFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate_fc = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h_out = self.fc2(h)
        gate = torch.sigmoid(self.gate_fc(h))
        out = self.layer_norm(gate * h_out + residual)
        return out


class SimplifiedTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer:
    Variable Selection -> LSTM Encoder -> Multi-Head Attention -> Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variable selection
        selected = self.vsn(x)  # (batch, seq, hidden)

        # LSTM encoding
        lstm_out, _ = self.lstm(selected)  # (batch, seq, hidden)

        # Self-attention with residual
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        # GRN post-processing
        grn_out = self.grn(attn_out)  # (batch, seq, hidden)

        # Pool: last timestep
        final = grn_out[:, -1, :]  # (batch, hidden)
        return self.output_fc(final).squeeze(-1)


# ---------------------------------------------------------------------------
# Sequence Dataset for PyTorch models
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Creates sliding-window sequences for LSTM / TFT input."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val


# ---------------------------------------------------------------------------
# Neural Model Trainer
# ---------------------------------------------------------------------------

class NeuralTrainer:
    """Handles training loop with early stopping for both LSTM and TFT models."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        self.criterion = nn.BCELoss()
        self.best_val_loss = float("inf")
        self.best_state_dict = None
        self.patience_counter = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            self.optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 15,
    ) -> None:
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % 10 == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.2e}"
                )

            if self.patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.model.to(DEVICE)

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            pred = self.model(X_batch)
            preds.append(pred.cpu().numpy())
        return np.concatenate(preds)


# ============================================================================
# Layer 3: Meta-Ensemble
# ============================================================================

class MetaEnsemble:
    """Stacking ensemble: learns optimal combination of sub-model predictions."""

    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.weights: Optional[np.ndarray] = None  # fallback simple average weights
        self.use_lr = True

    def train(self, preds_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        """
        preds_dict: {"gbm": proba_array, "lstm": proba_array, "tft": proba_array}
        """
        X_meta = np.column_stack([preds_dict[k] for k in sorted(preds_dict.keys())])
        try:
            self.meta_model.fit(X_meta, y_true)
            self.use_lr = True
            logger.info(f"Meta-ensemble LR coefficients: {self.meta_model.coef_}")
        except Exception as e:
            logger.warning(f"LR meta-model failed ({e}), using equal weights")
            self.use_lr = False
            self.weights = np.ones(len(preds_dict)) / len(preds_dict)

    def predict_proba(self, preds_dict: Dict[str, np.ndarray]) -> np.ndarray:
        X_meta = np.column_stack([preds_dict[k] for k in sorted(preds_dict.keys())])
        if self.use_lr:
            return self.meta_model.predict_proba(X_meta)[:, 1]
        else:
            return X_meta @ self.weights

    def save(self, path: Path) -> None:
        with open(path / "meta_ensemble.pkl", "wb") as f:
            pickle.dump({"model": self.meta_model, "use_lr": self.use_lr, "weights": self.weights}, f)

    def load(self, path: Path) -> None:
        with open(path / "meta_ensemble.pkl", "rb") as f:
            data = pickle.load(f)
        self.meta_model = data["model"]
        self.use_lr = data["use_lr"]
        self.weights = data["weights"]


# ============================================================================
# Risk Management
# ============================================================================

@dataclass
class RiskManager:
    """Position sizing and risk controls for crypto trading."""

    max_position_pct: float = 0.05       # 5% max
    stop_loss_pct: float = 0.02          # 2% stop loss per trade
    max_daily_drawdown: float = 0.05     # 5% max daily drawdown
    max_consecutive_losses: int = 3      # pause after 3 consecutive losses
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    is_paused: bool = False
    trades_today: List[float] = field(default_factory=list)

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion for optimal position sizing."""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        b = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / b
        # Half-Kelly for safety
        return max(0.0, kelly * 0.5)

    def compute_position_size(
        self, confidence: float, portfolio_value: float, win_rate: float = 0.55,
        avg_win: float = 0.015, avg_loss: float = 0.01,
        vol_scalar: float = 1.0,
    ) -> float:
        """Compute position size with Kelly + risk limits + volatility scaling.

        Args:
            vol_scalar: multiplier in (0, 1] that shrinks the position when
                        current volatility exceeds the historical median.
        """
        if self.is_paused:
            logger.warning("Trading paused due to consecutive losses or daily drawdown")
            return 0.0

        kelly = self.kelly_fraction(win_rate, avg_win, avg_loss)

        # Scale by confidence above threshold
        confidence_scale = (confidence - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD)
        confidence_scale = np.clip(confidence_scale, 0.0, 1.0)

        raw_size = kelly * confidence_scale * portfolio_value

        # Apply adaptive volatility scalar (reduce size in high-vol regimes)
        raw_size *= np.clip(vol_scalar, 0.05, 1.0)

        # Cap at max position
        max_size = self.max_position_pct * portfolio_value
        position_size = min(raw_size, max_size)

        return round(position_size, 2)

    def update_after_trade(self, pnl_pct: float) -> None:
        """Update risk state after a trade."""
        self.trades_today.append(pnl_pct)
        self.daily_pnl += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check pause conditions
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_paused = True
            logger.warning(f"Trading PAUSED: {self.consecutive_losses} consecutive losses")

        if self.daily_pnl < -self.max_daily_drawdown:
            self.is_paused = True
            logger.warning(f"Trading PAUSED: daily drawdown {self.daily_pnl:.2%} exceeds limit")

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each day)."""
        self.daily_pnl = 0.0
        self.trades_today = []
        if self.consecutive_losses < self.max_consecutive_losses:
            self.is_paused = False


# ============================================================================
# Walk-Forward Validation with Purged K-Fold
# ============================================================================

class PurgedWalkForward:
    """
    Walk-forward validation with purging to prevent look-ahead bias.
    Expanding window: train grows, validation & test are fixed-size.
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_gap: int = FORECAST_HORIZON * 5,  # gap between train and val (5x for safety)
    ):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_gap = purge_gap

    def split(self, n_samples: int):
        """Yield (train_idx, val_idx, test_idx) for each fold."""
        min_train = int(n_samples * 0.3)
        remaining = n_samples - min_train
        fold_size = remaining // self.n_splits

        for i in range(self.n_splits):
            train_end = min_train + i * fold_size
            val_start = train_end + self.purge_gap
            val_size = int(fold_size * self.val_ratio / (self.val_ratio + self.test_ratio))
            test_start = val_start + val_size + self.purge_gap
            test_end = min(test_start + fold_size - val_size, n_samples)

            if test_end <= test_start or val_start + val_size > n_samples:
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_start + val_size)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, val_idx, test_idx


# ============================================================================
# Backtesting & Metrics
# ============================================================================

def compute_trading_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict:
    """Compute classification + trading metrics."""
    # Only trade when confidence > threshold
    trade_mask = (predictions > threshold) | (predictions < (1 - threshold))
    traded_preds = (predictions[trade_mask] > 0.5).astype(int)
    traded_labels = labels[trade_mask]
    traded_returns = returns[trade_mask]

    # Classification metrics on traded signals
    if len(traded_preds) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
                "sharpe": 0, "per_trade_sharpe": 0, "profit_factor": 0,
                "max_drawdown": 0, "trade_count": 0, "total_bars": len(predictions),
                "win_rate": 0, "total_return": 0}

    acc = accuracy_score(traded_labels, traded_preds)
    prec = precision_score(traded_labels, traded_preds, zero_division=0)
    rec = recall_score(traded_labels, traded_preds, zero_division=0)
    f1 = f1_score(traded_labels, traded_preds, zero_division=0)

    # v3.5 修正: 連続損失フィルター除去（バックテスト結果を歪めるため）
    # 全トレードの生リターンでPF/Sharpeを計算する
    direction = np.where(predictions[trade_mask] > 0.5, 1, -1)
    strategy_returns = direction * traded_returns - TRANSACTION_COST

    cumulative = (1 + strategy_returns).cumprod()

    # Portfolio-level Sharpe: include all bars (0 return for non-traded)
    all_bar_returns = np.zeros(len(predictions))
    traded_indices = np.where(trade_mask)[0]
    for i, idx in enumerate(traded_indices):
        all_bar_returns[idx] = strategy_returns[i]

    # v3.5 修正: Sharpe年率換算 — トレード頻度ベースで正しく計算
    # 実際のトレード数/全バー数 × 年間バー数 = 年間実効トレード数
    trade_ratio = len(strategy_returns) / max(len(predictions), 1)
    effective_trades_per_year = trade_ratio * (8760 / FORECAST_HORIZON)
    sharpe = 0.0
    if strategy_returns.std() > 0 and len(strategy_returns) > 1:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(effective_trades_per_year)

    per_trade_sharpe = sharpe  # 同一（フィルター除去後は区別不要）

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_dd = drawdown.max() if len(drawdown) > 0 else 0.0

    win_rate = (strategy_returns > 0).mean() if len(strategy_returns) > 0 else 0.0

    # Profit Factor（生データで計算）
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "sharpe": round(sharpe, 4),
        "per_trade_sharpe": round(per_trade_sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 4),
        "trade_count": int(trade_mask.sum()),
        "total_bars": len(predictions),
        "win_rate": round(win_rate, 4),
        "total_return": round(float(cumulative[-1] - 1), 4) if len(cumulative) > 0 else 0.0,
    }


# ============================================================================
# Full Training Pipeline
# ============================================================================

class HybridCryptoModel:
    """
    Complete hybrid crypto prediction system.
    Orchestrates feature engineering, sub-models, and meta-ensemble.
    """

    def __init__(self):
        self.gbm = GBMEnsemble()
        self.lstm_model: Optional[LSTMAttentionModel] = None
        self.tft_model: Optional[SimplifiedTFT] = None
        self.meta = MetaEnsemble()
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.risk_manager = RiskManager()
        self._calibrator = None
        self._is_trained = False

    def _merge_auxiliary_data(self, df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
        """Merge Fear&Greed, funding rate, and ETH data into the main BTC dataframe."""
        # --- Fear & Greed Index (daily → forward-fill to hourly) ---
        fg_path = data_dir / "fear_greed.csv"
        if fg_path.exists():
            try:
                fg = pd.read_csv(fg_path, parse_dates=["timestamp"])
                fg.index = pd.to_datetime(fg["timestamp"], utc=True)
                fg = fg[["fear_greed_value"]].rename(columns={"fear_greed_value": "fear_greed_index"})
                fg = fg.reindex(df.index, method="ffill", tolerance=pd.Timedelta("48h"))
                df["fear_greed_index"] = fg["fear_greed_index"]
                n_filled = df["fear_greed_index"].notna().sum()
                logger.info(f"Merged Fear&Greed index: {n_filled}/{len(df)} rows filled")
            except Exception as e:
                logger.warning(f"Failed to merge Fear&Greed data: {e}")

        # --- Funding Rate (8h → forward-fill to hourly) ---
        fr_path = data_dir / "funding_rate.csv"
        if fr_path.exists():
            try:
                fr = pd.read_csv(fr_path)
                fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True, format="mixed")
                fr.index = pd.to_datetime(fr["timestamp"], utc=True)
                fr = fr[["funding_rate"]]
                fr = fr.reindex(df.index, method="ffill", tolerance=pd.Timedelta("24h"))
                df["funding_rate"] = fr["funding_rate"]
                # Rolling 24h average funding rate for trend detection
                df["funding_rate_ma24"] = df["funding_rate"].rolling(24, min_periods=1).mean()
                n_filled = df["funding_rate"].notna().sum()
                logger.info(f"Merged funding rate: {n_filled}/{len(df)} rows filled")
            except Exception as e:
                logger.warning(f"Failed to merge funding rate data: {e}")

        # --- Open Interest (if available) ---
        oi_path = data_dir / "open_interest.csv"
        if oi_path.exists():
            try:
                oi = pd.read_csv(oi_path, parse_dates=["timestamp"])
                oi.index = pd.to_datetime(oi["timestamp"], utc=True)
                if "open_interest" in oi.columns:
                    oi_series = oi["open_interest"].reindex(df.index, method="ffill", tolerance=pd.Timedelta("4h"))
                    df["open_interest_change"] = oi_series.pct_change(1)
                    n_filled = df["open_interest_change"].notna().sum()
                    logger.info(f"Merged open interest: {n_filled}/{len(df)} rows filled")
            except Exception as e:
                logger.warning(f"Failed to merge open interest data: {e}")

        # --- ETH cross-asset data ---
        eth_path = data_dir / "eth_1h.csv"
        if eth_path.exists():
            try:
                eth = pd.read_csv(eth_path, parse_dates=[0])
                date_col = [c for c in eth.columns if "time" in c.lower()][0]
                eth.index = pd.to_datetime(eth[date_col], utc=True)
                close_col = [c for c in eth.columns if c.lower() == "close"][0]
                df["ETH_Close"] = eth[close_col].reindex(df.index, method="ffill", tolerance=pd.Timedelta("2h"))
                n_filled = df["ETH_Close"].notna().sum()
                logger.info(f"Merged ETH close: {n_filled}/{len(df)} rows filled")
            except Exception as e:
                logger.warning(f"Failed to merge ETH data: {e}")

        return df

    def _prepare_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data with all features and labels."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=["Timestamp"] if "Timestamp" in pd.read_csv(data_path, nrows=1).columns else [0])

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if date_col:
                df.index = pd.to_datetime(df[date_col[0]])
                df = df.drop(columns=date_col)
            else:
                df.index = pd.to_datetime(df.iloc[:, 0])
                df = df.iloc[:, 1:]

        # Ensure required columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                # Try case-insensitive match
                matches = [c for c in df.columns if c.lower() == col.lower()]
                if matches:
                    df = df.rename(columns={matches[0]: col})
                else:
                    raise ValueError(f"Missing required column: {col}")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        logger.info(f"Data shape: {df.shape}, range: {df.index[0]} to {df.index[-1]}")

        # --- Merge auxiliary data (Fear&Greed, funding rate, ETH) ---
        data_dir = Path(data_path).parent
        df = self._merge_auxiliary_data(df, data_dir)

        # Build features
        df, self.feature_cols = build_features(df)
        df = generate_labels(df)

        # Drop rows with NaN in features or label
        df = df.dropna(subset=self.feature_cols + ["label", "future_return"])
        logger.info(f"After feature engineering: {len(df)} rows, {len(self.feature_cols)} features")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

        return df

    def _train_neural_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train LSTM and TFT models, return validation predictions."""
        batch_size = 256

        # --- LSTM ---
        logger.info("Training BiLSTM-Attention model...")
        self.lstm_model = LSTMAttentionModel(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.5)

        train_ds = SequenceDataset(X_train, y_train, SEQUENCE_LENGTH)
        val_ds = SequenceDataset(X_val, y_val, SEQUENCE_LENGTH)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        lstm_trainer = NeuralTrainer(self.lstm_model, lr=1e-3, weight_decay=1e-4)
        lstm_trainer.fit(train_loader, val_loader, max_epochs=100, patience=15)
        lstm_val_pred = lstm_trainer.predict_proba(val_loader)

        # --- TFT ---
        logger.info("Training Simplified TFT model...")
        self.tft_model = SimplifiedTFT(input_dim=input_dim, hidden_dim=64, num_heads=4, dropout=0.4)

        tft_trainer = NeuralTrainer(self.tft_model, lr=1e-3, weight_decay=1e-4)
        tft_trainer.fit(train_loader, val_loader, max_epochs=100, patience=15)
        tft_val_pred = tft_trainer.predict_proba(val_loader)

        return lstm_val_pred, tft_val_pred

    def _predict_neural(self, X: np.ndarray, y_dummy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from trained neural models on arbitrary data."""
        batch_size = 256
        ds = SequenceDataset(X, y_dummy, SEQUENCE_LENGTH)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # LSTM
        self.lstm_model.eval()
        lstm_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                pred = self.lstm_model(X_batch)
                lstm_preds.append(pred.cpu().numpy())
        lstm_preds = np.concatenate(lstm_preds)

        # TFT
        self.tft_model.eval()
        tft_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                pred = self.tft_model(X_batch)
                tft_preds.append(pred.cpu().numpy())
        tft_preds = np.concatenate(tft_preds)

        return lstm_preds, tft_preds

    def train_and_evaluate(self, data_path: str) -> Dict:
        """
        Full training + walk-forward evaluation pipeline.
        Returns dict with all metrics across folds.
        """
        set_seed()
        df = self._prepare_data(data_path)

        X_all = df[self.feature_cols].values
        y_all = df["label"].values
        returns_all = df["future_return"].values

        # Scale features
        self.scaler.fit(X_all)
        X_scaled = self.scaler.transform(X_all)

        # Walk-forward splits
        wf = PurgedWalkForward(n_splits=5, purge_gap=FORECAST_HORIZON * 5)
        all_fold_metrics = []

        for fold_i, (train_idx, val_idx, test_idx) in enumerate(wf.split(len(X_scaled))):
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk-Forward Fold {fold_i + 1}")
            logger.info(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
            logger.info(f"{'='*60}")

            X_train, y_train = X_scaled[train_idx], y_all[train_idx]
            X_val, y_val = X_scaled[val_idx], y_all[val_idx]
            X_test, y_test = X_scaled[test_idx], y_all[test_idx]
            returns_train = returns_all[train_idx]
            returns_val = returns_all[val_idx]
            returns_test = returns_all[test_idx]

            # --- Model A: GBM ---
            self.gbm.train(X_train, y_train, X_val, y_val, self.feature_cols,
                           returns_train=returns_train, returns_val=returns_val)
            gbm_val_pred = self.gbm.predict_proba(X_val)
            gbm_test_pred = self.gbm.predict_proba(X_test)

            # --- Models B & C: Neural ---
            input_dim = X_train.shape[1]

            # For neural models, we need enough data for sequences
            if len(X_val) > SEQUENCE_LENGTH and len(X_train) > SEQUENCE_LENGTH:
                lstm_val_pred, tft_val_pred = self._train_neural_models(
                    X_train, y_train, X_val, y_val, input_dim
                )
                lstm_test_pred, tft_test_pred = self._predict_neural(
                    X_test, y_test
                )

                # Align lengths: neural models produce fewer predictions due to sequence windowing
                n_neural_val = len(lstm_val_pred)
                n_neural_test = len(lstm_test_pred)

                # Trim GBM preds to match neural output (they lose SEQUENCE_LENGTH from the start)
                gbm_val_trimmed = gbm_val_pred[-n_neural_val:] if n_neural_val < len(gbm_val_pred) else gbm_val_pred
                gbm_test_trimmed = gbm_test_pred[-n_neural_test:] if n_neural_test < len(gbm_test_pred) else gbm_test_pred
                y_val_trimmed = y_val[-n_neural_val:] if n_neural_val < len(y_val) else y_val
                y_test_trimmed = y_test[-n_neural_test:] if n_neural_test < len(y_test) else y_test
                returns_test_trimmed = returns_test[-n_neural_test:] if n_neural_test < len(returns_test) else returns_test

                # Ensure all arrays have the same length
                min_val = min(len(gbm_val_trimmed), len(lstm_val_pred), len(tft_val_pred), len(y_val_trimmed))
                min_test = min(len(gbm_test_trimmed), len(lstm_test_pred), len(tft_test_pred), len(y_test_trimmed))

                # --- Meta-Ensemble ---
                val_preds = {
                    "gbm": gbm_val_trimmed[:min_val],
                    "lstm": lstm_val_pred[:min_val],
                    "tft": tft_val_pred[:min_val],
                }
                self.meta.train(val_preds, y_val_trimmed[:min_val])

                test_preds = {
                    "gbm": gbm_test_trimmed[:min_test],
                    "lstm": lstm_test_pred[:min_test],
                    "tft": tft_test_pred[:min_test],
                }
                ensemble_pred = self.meta.predict_proba(test_preds)
                fold_metrics = compute_trading_metrics(
                    ensemble_pred, y_test_trimmed[:min_test], returns_test_trimmed[:min_test]
                )
            else:
                # Not enough data for sequences — use GBM only
                logger.warning("Insufficient data for neural models in this fold, using GBM only")
                ensemble_pred = gbm_test_pred
                fold_metrics = compute_trading_metrics(ensemble_pred, y_test, returns_test)

            fold_metrics["fold"] = fold_i + 1
            all_fold_metrics.append(fold_metrics)
            logger.info(f"Fold {fold_i + 1} results: {fold_metrics}")

        # --- Final model: retrain on all data (train+val) for deployment ---
        logger.info("\n" + "=" * 60)
        logger.info("Final retraining on full dataset for deployment...")
        logger.info("=" * 60)

        split_70 = int(len(X_scaled) * 0.70)
        split_85 = int(len(X_scaled) * 0.85)

        X_train_final = X_scaled[:split_70]
        y_train_final = y_all[:split_70]
        X_val_final = X_scaled[split_70:split_85]
        y_val_final = y_all[split_70:split_85]
        X_test_final = X_scaled[split_85:]
        y_test_final = y_all[split_85:]
        returns_train_final = returns_all[:split_70]
        returns_val_final = returns_all[split_70:split_85]
        returns_test_final = returns_all[split_85:]

        # GBM
        self.gbm.train(X_train_final, y_train_final, X_val_final, y_val_final, self.feature_cols,
                       returns_train=returns_train_final, returns_val=returns_val_final)
        gbm_val_final = self.gbm.predict_proba(X_val_final)
        gbm_test_final = self.gbm.predict_proba(X_test_final)

        # Neural
        input_dim = X_train_final.shape[1]
        if len(X_val_final) > SEQUENCE_LENGTH and len(X_train_final) > SEQUENCE_LENGTH:
            lstm_val_f, tft_val_f = self._train_neural_models(
                X_train_final, y_train_final, X_val_final, y_val_final, input_dim
            )
            lstm_test_f, tft_test_f = self._predict_neural(X_test_final, y_test_final)

            n_nv = len(lstm_val_f)
            n_nt = len(lstm_test_f)
            min_v = min(len(gbm_val_final[-n_nv:]), len(lstm_val_f), len(tft_val_f))
            min_t = min(len(gbm_test_final[-n_nt:]), len(lstm_test_f), len(tft_test_f))

            val_preds_final = {
                "gbm": gbm_val_final[-n_nv:][:min_v],
                "lstm": lstm_val_f[:min_v],
                "tft": tft_val_f[:min_v],
            }
            self.meta.train(val_preds_final, y_val_final[-n_nv:][:min_v])

            test_preds_final = {
                "gbm": gbm_test_final[-n_nt:][:min_t],
                "lstm": lstm_test_f[:min_t],
                "tft": tft_test_f[:min_t],
            }
            final_pred = self.meta.predict_proba(test_preds_final)
            final_metrics = compute_trading_metrics(
                final_pred, y_test_final[-n_nt:][:min_t], returns_test_final[-n_nt:][:min_t]
            )
        else:
            final_pred = gbm_test_final
            final_metrics = compute_trading_metrics(final_pred, y_test_final, returns_test_final)

        # Calibrate predictions using Platt scaling (LogisticRegression)
        # v3.5 修正: テストデータではなくバリデーションデータでキャリブレーション
        # テストラベルを使うと未来情報の漏洩になる
        if len(X_val_final) > 0:
            # バリデーションセットの予測値でキャリブレーション
            cal_pred_val = gbm_val_final  # GBMのバリデーション予測を使用
            cal_labels_val = y_val_final[-len(cal_pred_val):]
            if len(cal_pred_val) == len(cal_labels_val):
                try:
                    platt_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                    platt_lr.fit(cal_pred_val.reshape(-1, 1), cal_labels_val)
                    self._calibrator = platt_lr
                    logger.info(f"Platt calibration fitted on val: range [{cal_pred_val.min():.4f}, {cal_pred_val.max():.4f}]")
                    calibrated = platt_lr.predict_proba(final_pred.reshape(-1, 1))[:, 1]
                    logger.info(f"Calibrated pred range [{calibrated.min():.4f}, {calibrated.max():.4f}]")
                    logger.info(f"Calibrated mean: {calibrated.mean():.4f}, <0.5: {(calibrated < 0.5).sum()}/{len(calibrated)}")
                except Exception as e:
                    logger.warning(f"Platt calibration failed: {e}")
                    self._calibrator = None
            else:
                self._calibrator = None
        else:
            self._calibrator = None

        # Save models
        self.save(MODEL_DIR)
        self._is_trained = True

        # Summary
        avg_metrics = {}
        all_keys = set()
        for m in all_fold_metrics:
            all_keys.update(m.keys())
        all_keys.discard("fold")
        for key in sorted(all_keys):
            vals = [m.get(key, 0) for m in all_fold_metrics]
            avg_metrics[f"avg_{key}"] = round(np.mean(vals), 4)
            avg_metrics[f"std_{key}"] = round(np.std(vals), 4)

        results = {
            "walk_forward_folds": all_fold_metrics,
            "walk_forward_averages": avg_metrics,
            "final_holdout_metrics": final_metrics,
            "feature_count": len(self.feature_cols),
            "feature_names": self.feature_cols,
            "data_points": len(df),
            "device": str(DEVICE),
        }

        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Walk-Forward Average Metrics: {avg_metrics}")
        logger.info(f"Final Hold-out Metrics: {final_metrics}")
        logger.info(f"Models saved to: {MODEL_DIR}")

        return results

    def predict(
        self, data_path_or_df, portfolio_value: float = 10000.0
    ) -> Tuple[str, float, float]:
        """
        Generate a trading signal using the SAME pipeline as training.

        Args:
            data_path_or_df: path to full CSV file (str/Path) or DataFrame with full history
            portfolio_value: current portfolio value for position sizing

        Returns:
            (signal, confidence, position_size)
            signal: "BUY", "SELL", or "HOLD"
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train_and_evaluate() first or load().")

        # --- Use _prepare_data() for consistent feature computation ---
        if isinstance(data_path_or_df, (str, Path)):
            # File path: use exact same pipeline as training
            df = self._prepare_data(str(data_path_or_df))
        elif isinstance(data_path_or_df, pd.DataFrame):
            # DataFrame: save to temp file and use _prepare_data()
            # This ensures identical feature computation as training
            import tempfile
            latest_df = data_path_or_df.copy()

            # Normalize column names
            col_map = {}
            for col in latest_df.columns:
                if col.lower() in ("open", "high", "low", "close", "volume"):
                    col_map[col] = col.capitalize()
            if col_map:
                latest_df = latest_df.rename(columns=col_map)

            # Ensure timestamp column exists
            if not isinstance(latest_df.index, pd.DatetimeIndex):
                date_col = [c for c in latest_df.columns if "time" in c.lower() or "date" in c.lower()]
                if date_col:
                    latest_df.index = pd.to_datetime(latest_df[date_col[0]])
                    latest_df = latest_df.drop(columns=date_col)

            # Save to temp CSV and process through _prepare_data()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
                tmp_path = f.name
                # Reset index to write timestamp as column
                save_df = latest_df[["Open", "High", "Low", "Close", "Volume"]].copy()
                save_df.index.name = "Timestamp"
                save_df.to_csv(f)

            try:
                df = self._prepare_data(tmp_path)
            finally:
                import os
                os.unlink(tmp_path)
        else:
            raise TypeError(f"Expected str, Path, or DataFrame, got {type(data_path_or_df)}")

        # Check we have enough data after feature computation
        if len(df) < SEQUENCE_LENGTH + 1:
            logger.warning(f"Need at least {SEQUENCE_LENGTH + 1} rows after feature computation, got {len(df)}")
            return "HOLD", 0.0, 0.0

        # Verify all required features exist (no zero-filling!)
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing features after _prepare_data(): {missing}. Cannot predict.")
            return "HOLD", 0.0, 0.0

        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        # GBM prediction (last row)
        gbm_pred = self.gbm.predict_proba(X_scaled[-1:])

        # Neural predictions (need sequence)
        if self.lstm_model is not None and self.tft_model is not None:
            x_seq = torch.FloatTensor(X_scaled[-SEQUENCE_LENGTH:]).unsqueeze(0).to(DEVICE)

            self.lstm_model.eval()
            self.tft_model.eval()
            with torch.no_grad():
                lstm_pred = self.lstm_model(x_seq).cpu().numpy()
                tft_pred = self.tft_model(x_seq).cpu().numpy()

            preds = {
                "gbm": gbm_pred,
                "lstm": lstm_pred,
                "tft": tft_pred,
            }
            confidence = float(self.meta.predict_proba(preds)[0])
            logger.debug(f"Meta-ensemble prediction: gbm={gbm_pred[0]:.4f}, lstm={lstm_pred[0]:.4f}, tft={tft_pred[0]:.4f} -> meta={confidence:.4f}")
        else:
            confidence = float(gbm_pred[0])
            logger.warning("Neural models not loaded, using GBM-only prediction")

        # Apply Platt scaling calibration if available
        if hasattr(self, '_calibrator') and self._calibrator is not None:
            raw_confidence = confidence
            try:
                confidence = float(self._calibrator.predict_proba([[confidence]])[0, 1])
                logger.debug(f"Platt calibration: {raw_confidence:.4f} -> {confidence:.4f}")
            except Exception as e:
                logger.warning(f"Platt calibration failed: {e}")

        # Determine signal
        if confidence > CONFIDENCE_THRESHOLD:
            signal = "BUY"
        elif confidence < (1 - CONFIDENCE_THRESHOLD):
            signal = "SELL"
        else:
            signal = "HOLD"

        # --- Adaptive volatility scaling for position sizing ---
        # Use realized_vol column (rolling 24h std of log returns) that
        # _prepare_data() already computes.  target_vol = median of the
        # full history available; vol_scalar = min(1, target / current).
        vol_scalar = 1.0
        if "realized_vol" in df.columns:
            vol_series = df["realized_vol"].dropna()
            if len(vol_series) >= 2:
                target_vol = float(vol_series.median())
                current_vol = float(vol_series.iloc[-1])
                if current_vol > 0:
                    vol_scalar = min(1.0, target_vol / current_vol)
                    logger.info(
                        f"Vol-scaling: current_vol={current_vol:.6f}, "
                        f"target_vol={target_vol:.6f}, vol_scalar={vol_scalar:.4f}"
                    )

        # Position sizing
        if signal == "HOLD":
            position_size = 0.0
        else:
            position_size = self.risk_manager.compute_position_size(
                max(confidence, 1 - confidence), portfolio_value,
                vol_scalar=vol_scalar,
            )

        logger.info(f"Prediction: signal={signal}, confidence={confidence:.4f}, position_size={position_size:.2f}")
        return signal, round(confidence, 4), position_size

    def save(self, path: Path) -> None:
        """Save all model components."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.gbm.save(path)
        self.meta.save(path)

        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(path / "feature_cols.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)

        if hasattr(self, '_calibrator') and self._calibrator is not None:
            with open(path / "calibrator.pkl", "wb") as f:
                pickle.dump(self._calibrator, f)

        if self.lstm_model is not None:
            torch.save(self.lstm_model.state_dict(), path / "lstm_model.pt")
        if self.tft_model is not None:
            torch.save(self.tft_model.state_dict(), path / "tft_model.pt")

        # Save model architecture params for loading
        meta_info = {
            "input_dim": len(self.feature_cols),
            "lstm_hidden": 128,
            "lstm_layers": 2,
            "tft_hidden": 64,
            "tft_heads": 4,
            "sequence_length": SEQUENCE_LENGTH,
        }
        with open(path / "meta_info.pkl", "wb") as f:
            pickle.dump(meta_info, f)

        logger.info(f"All models saved to {path}")

    def load(self, path: Path) -> None:
        """Load all model components."""
        path = Path(path)

        self.gbm.load(path)
        self.meta.load(path)

        with open(path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(path / "feature_cols.pkl", "rb") as f:
            self.feature_cols = pickle.load(f)

        with open(path / "meta_info.pkl", "rb") as f:
            meta_info = pickle.load(f)

        input_dim = meta_info["input_dim"]

        if (path / "lstm_model.pt").exists():
            self.lstm_model = LSTMAttentionModel(
                input_dim=input_dim,
                hidden_dim=meta_info["lstm_hidden"],
                num_layers=meta_info["lstm_layers"],
            ).to(DEVICE)
            self.lstm_model.load_state_dict(torch.load(path / "lstm_model.pt", map_location=DEVICE))
            self.lstm_model.eval()

        if (path / "tft_model.pt").exists():
            self.tft_model = SimplifiedTFT(
                input_dim=input_dim,
                hidden_dim=meta_info["tft_hidden"],
                num_heads=meta_info["tft_heads"],
            ).to(DEVICE)
            self.tft_model.load_state_dict(torch.load(path / "tft_model.pt", map_location=DEVICE))
            self.tft_model.eval()

        # Load calibrator if available
        cal_path = path / "calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                self._calibrator = pickle.load(f)
            logger.info("Loaded prediction calibrator")
        else:
            self._calibrator = None

        self._is_trained = True
        logger.info(f"All models loaded from {path}")


# ============================================================================
# Public API
# ============================================================================

_model_instance: Optional[HybridCryptoModel] = None


def get_model() -> HybridCryptoModel:
    """Get or create the singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = HybridCryptoModel()
    return _model_instance


def train_and_evaluate(data_path: str) -> Dict:
    """
    Train the full hybrid model and evaluate with walk-forward validation.

    Args:
        data_path: path to CSV with columns [Timestamp, Open, High, Low, Close, Volume]

    Returns:
        dict with walk-forward metrics, final holdout metrics, feature importance
    """
    model = get_model()
    return model.train_and_evaluate(data_path)


def predict(latest_data: pd.DataFrame, portfolio_value: float = 10000.0) -> Tuple[str, float, float]:
    """
    Generate prediction from the latest market data.

    Args:
        latest_data: DataFrame with >= 49 rows of OHLCV data (1h candles)
        portfolio_value: current portfolio value

    Returns:
        (signal, confidence, position_size)
        signal: "BUY" / "SELL" / "HOLD"
        confidence: float 0-1
        position_size: dollar amount to allocate
    """
    model = get_model()
    return model.predict(latest_data, portfolio_value)


def load_model(model_dir: str = None) -> HybridCryptoModel:
    """Load a previously trained model from disk."""
    model = get_model()
    path = Path(model_dir) if model_dir else MODEL_DIR
    model.load(path)
    return model


# ============================================================================
# Walk-Forward Expanding Window Validation (standalone, like FX v3)
# ============================================================================

def walk_forward_expanding(data_path: str, min_train_hours: int = 4320,
                           test_hours: int = 720) -> Dict:
    """
    Expanding-window walk-forward validation (matches FX v3 14_main_system.py).

    Each window:
      - Train on all data up to window boundary (expanding)
      - Test on next `test_hours` hours
      - Move forward by `test_hours`

    Computes PF and Sharpe on ALL bars (not just traded bars) per the
    feedback_sharpe_calculation.md lesson.
    """
    logger.info("=" * 60)
    logger.info("EXPANDING WINDOW WALK-FORWARD VALIDATION")
    logger.info(f"  min_train={min_train_hours}h, test={test_hours}h")
    logger.info("=" * 60)

    # Prepare data once
    model = HybridCryptoModel()
    df = model._prepare_data(data_path)
    feature_cols = model.feature_cols

    X_all = df[feature_cols].values
    y_all = df["label"].values
    returns_all = df["future_return"].values
    n_total = len(X_all)

    # Generate expanding window splits
    splits = []
    step_size = test_hours
    pos = min_train_hours
    while pos + step_size <= n_total:
        train_end = pos
        test_start = pos
        test_end = min(pos + step_size, n_total)
        splits.append((train_end, test_start, test_end))
        pos += step_size

    logger.info(f"Total data points: {n_total}, windows: {len(splits)}")

    all_window_metrics = []
    all_trade_returns = []  # collect ALL bar returns across windows for overall stats

    for w_i, (train_end, test_start, test_end) in enumerate(splits):
        logger.info(f"\n--- Window {w_i + 1}/{len(splits)}: train=[0:{train_end}], test=[{test_start}:{test_end}] ---")

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        returns_train = returns_all[:train_end]
        X_test = X_all[test_start:test_end]
        y_test = y_all[test_start:test_end]
        returns_test = returns_all[test_start:test_end]

        if len(X_train) < 500 or len(X_test) < 10:
            logger.warning(f"Skipping window {w_i + 1}: insufficient data")
            continue

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Split train into train/val (last 15% of train for early stopping)
        val_split = int(len(X_train_s) * 0.85)
        X_tr = X_train_s[:val_split]
        y_tr = y_train[:val_split]
        X_va = X_train_s[val_split:]
        y_va = y_train[val_split:]
        ret_tr = returns_train[:val_split]
        ret_va = returns_train[val_split:]

        # Train GBM only (faster for walk-forward; neural models too slow per window)
        gbm = GBMEnsemble()
        try:
            gbm.train(X_tr, y_tr, X_va, y_va, feature_cols,
                       returns_train=ret_tr, returns_val=ret_va)
        except Exception as e:
            logger.warning(f"Window {w_i + 1} GBM training failed: {e}")
            continue

        # Predict on test
        test_pred = gbm.predict_proba(X_test_s)

        # Compute metrics (PF and Sharpe on ALL bars)
        metrics = compute_trading_metrics(test_pred, y_test, returns_test)
        metrics["window"] = w_i + 1
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        all_window_metrics.append(metrics)

        # Collect per-bar returns for overall computation
        trade_mask = (test_pred > CONFIDENCE_THRESHOLD) | (test_pred < (1 - CONFIDENCE_THRESHOLD))
        direction = np.where(test_pred > 0.5, 1, -1)
        bar_returns = np.zeros(len(test_pred))
        bar_returns[trade_mask] = direction[trade_mask] * returns_test[trade_mask] - TRANSACTION_COST
        all_trade_returns.extend(bar_returns.tolist())

        logger.info(
            f"  Window {w_i + 1}: PF={metrics['profit_factor']:.2f}  "
            f"Sharpe={metrics['sharpe']:.2f}  trades={metrics['trade_count']}  "
            f"WR={metrics['win_rate']:.2%}  return={metrics['total_return']:.4f}"
        )

    # --- Overall summary across all windows ---
    if not all_window_metrics:
        logger.error("No valid windows. Check data size.")
        return {}

    all_bar_arr = np.array(all_trade_returns)
    periods_per_year = 8760 / FORECAST_HORIZON

    # Overall PF on ALL bars
    total_gains = all_bar_arr[all_bar_arr > 0].sum()
    total_losses = abs(all_bar_arr[all_bar_arr < 0].sum())
    overall_pf = total_gains / total_losses if total_losses > 0 else float('inf')

    # Overall Sharpe on ALL bars
    overall_sharpe = 0.0
    if all_bar_arr.std() > 0:
        overall_sharpe = all_bar_arr.mean() / all_bar_arr.std() * np.sqrt(periods_per_year)

    # Cumulative return
    cumulative = (1 + all_bar_arr).cumprod()
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    dd = (peak - cumulative) / np.where(peak > 0, peak, 1)
    max_dd = dd.max()

    # Average per-window metrics
    avg_pf = np.mean([m['profit_factor'] for m in all_window_metrics if m['profit_factor'] < 100])
    avg_sharpe = np.mean([m['sharpe'] for m in all_window_metrics])
    avg_wr = np.mean([m['win_rate'] for m in all_window_metrics if m['trade_count'] > 0])
    total_trades = sum(m['trade_count'] for m in all_window_metrics)

    print("\n" + "=" * 60)
    print("EXPANDING WINDOW WALK-FORWARD RESULTS")
    print("=" * 60)
    print(f"  Windows:          {len(all_window_metrics)}")
    print(f"  Total trades:     {total_trades}")
    print(f"  Total bars:       {len(all_bar_arr)}")
    print(f"  Overall PF:       {overall_pf:.2f}")
    print(f"  Overall Sharpe:   {overall_sharpe:.2f}  (on ALL bars)")
    print(f"  Total return:     {total_return * 100:+.2f}%")
    print(f"  Max drawdown:     {max_dd * 100:.2f}%")
    print(f"  Avg win rate:     {avg_wr * 100:.1f}%")
    print(f"  Avg PF/window:    {avg_pf:.2f}")
    print(f"  Avg Sharpe/window:{avg_sharpe:.2f}")
    print("=" * 60)

    print("\nPer-window breakdown:")
    for m in all_window_metrics:
        print(
            f"  W{m['window']:2d}: PF={m['profit_factor']:6.2f}  "
            f"Sharpe={m['sharpe']:6.2f}  trades={m['trade_count']:3d}  "
            f"WR={m['win_rate']:.2%}  return={m['total_return']:+.4f}"
        )

    return {
        "windows": all_window_metrics,
        "overall_pf": round(overall_pf, 4),
        "overall_sharpe": round(overall_sharpe, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "total_trades": total_trades,
        "total_bars": len(all_bar_arr),
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python hybrid_model.py <path_to_csv>           # Train + evaluate")
        print("  python hybrid_model.py <path_to_csv> --wf      # Expanding walk-forward only")
        print("  CSV must have columns: Timestamp, Open, High, Low, Close, Volume")
        sys.exit(1)

    data_path = sys.argv[1]

    if "--wf" in sys.argv:
        # Expanding window walk-forward validation only
        wf_results = walk_forward_expanding(data_path)
    else:
        # Full training pipeline
        results = train_and_evaluate(data_path)

        print("\n" + "=" * 60)
        print("WALK-FORWARD RESULTS")
        print("=" * 60)
        for fold in results["walk_forward_folds"]:
            print(f"  Fold {fold['fold']}: acc={fold['accuracy']:.4f}  f1={fold['f1']:.4f}  "
                  f"sharpe={fold['sharpe']:.4f}  PF={fold['profit_factor']:.2f}  "
                  f"maxDD={fold['max_drawdown']:.4f}  trades={fold['trade_count']}")

        print("\nAverages:")
        for k, v in results["walk_forward_averages"].items():
            print(f"  {k}: {v}")

        print(f"\nFinal hold-out: {results['final_holdout_metrics']}")
