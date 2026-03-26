"""
特徴量生成モジュール
テクニカル指標・HMMレジーム・派生特徴量の計算
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from numpy.linalg import LinAlgError


# テクニカル指標の特徴量カラム一覧（基本セット）
# 注: 時間帯セッション特徴量・他通貨ペア特徴量はWalk-Forwardで効果なし（PF改善なし）のため不採用
# v3.3 pruned: Removed near-zero importance (Regime, Regime_changed) and
#   correlated redundants (MA_5, MA_75 -> keep MA_25; BB_upper, BB_lower -> keep BB_width;
#   MACD_signal -> keep MACD)
FEATURE_COLS = [
    "RSI_14", "MACD", "MACD_hist",
    "BB_width",
    "MA_25",
    "Return_1", "Return_3", "Return_6", "Return_24",
    "Volatility_24", "Hour", "DayOfWeek",
    "Regime_duration",
    # v3.4 マルチタイムフレーム特徴量: PF 0.95→1.93, WinRate +8.6%
    "RSI_4h", "MACD_hist_4h", "BB_width_4h",
    "RSI_daily", "MACD_hist_daily", "BB_width_daily",
    "Daily_trend_dir",
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を計算してDataFrameに追加する"""
    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss.replace(0, np.nan))
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, np.nan)

    # Moving Averages
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_25"] = df["Close"].rolling(25).mean()
    df["MA_75"] = df["Close"].rolling(75).mean()

    # Returns & Volatility
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_6"] = df["Close"].pct_change(6)
    df["Return_24"] = df["Close"].pct_change(24)
    df["Volatility_24"] = df["Return_1"].rolling(24).std()

    # Time features
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek

    # 周期的エンコーディング（23時→0時のジャンプをなくす）
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DoW_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
    df["DoW_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 5)

    # FX市場セッション（UTCベース）
    h = df["Hour"]
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)    # 東京 00-09 UTC (9-18 JST)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)  # ロンドン 07-16 UTC
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)     # NY 13-22 UTC
    df["Session_overlap"] = (                                   # ロンドン-NY重複（最も活発）
        (h >= 13) & (h < 16)
    ).astype(int)

    # 曜日効果
    df["IsFriday"] = (df["DayOfWeek"] == 4).astype(int)
    df["IsMonday"] = (df["DayOfWeek"] == 0).astype(int)
    df["IsMonthEnd"] = (df.index.is_month_end | (df.index + pd.Timedelta(hours=24)).is_month_end).astype(int)

    # Derived
    df["Abs_ret_1h"] = df["Return_1"].abs()

    return df


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """4時間足・日足のテクニカル指標を1時間足データから計算して追加する。
    v3.4: PF 0.95→1.93, WinRate +8.6%"""
    # 4時間足リサンプル
    df_4h = df[["Open", "High", "Low", "Close"]].resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    # 4h RSI
    delta_4h = df_4h["Close"].diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0.0).rolling(14).mean()
    loss_4h = (-delta_4h).where(delta_4h < 0, 0.0).rolling(14).mean()
    rs_4h = np.where(loss_4h == 0, np.inf, gain_4h / loss_4h.replace(0, np.nan))
    df_4h["RSI_4h"] = 100.0 - (100.0 / (1.0 + rs_4h))

    # 4h MACD histogram
    ema12_4h = df_4h["Close"].ewm(span=12, adjust=False).mean()
    ema26_4h = df_4h["Close"].ewm(span=26, adjust=False).mean()
    macd_4h = ema12_4h - ema26_4h
    signal_4h = macd_4h.ewm(span=9, adjust=False).mean()
    df_4h["MACD_hist_4h"] = macd_4h - signal_4h

    # 4h BB width
    sma20_4h = df_4h["Close"].rolling(20).mean()
    std20_4h = df_4h["Close"].rolling(20).std()
    df_4h["BB_width_4h"] = (4 * std20_4h) / sma20_4h.replace(0, np.nan)

    # 日足リサンプル
    df_d = df[["Open", "High", "Low", "Close"]].resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    # Daily RSI
    delta_d = df_d["Close"].diff()
    gain_d = delta_d.where(delta_d > 0, 0.0).rolling(14).mean()
    loss_d = (-delta_d).where(delta_d < 0, 0.0).rolling(14).mean()
    rs_d = np.where(loss_d == 0, np.inf, gain_d / loss_d.replace(0, np.nan))
    df_d["RSI_daily"] = 100.0 - (100.0 / (1.0 + rs_d))

    # Daily MACD histogram
    ema12_d = df_d["Close"].ewm(span=12, adjust=False).mean()
    ema26_d = df_d["Close"].ewm(span=26, adjust=False).mean()
    macd_d = ema12_d - ema26_d
    signal_d = macd_d.ewm(span=9, adjust=False).mean()
    df_d["MACD_hist_daily"] = macd_d - signal_d

    # Daily BB width
    sma20_d = df_d["Close"].rolling(20).mean()
    std20_d = df_d["Close"].rolling(20).std()
    df_d["BB_width_daily"] = (4 * std20_d) / sma20_d.replace(0, np.nan)

    # Daily trend direction
    df_d["Daily_trend_dir"] = (df_d["Close"] > df_d["Close"].rolling(20).mean()).astype(float)

    # マージ（forward-fill）
    for col in ["RSI_4h", "MACD_hist_4h", "BB_width_4h"]:
        df[col] = df_4h[col].reindex(df.index, method="ffill")
    for col in ["RSI_daily", "MACD_hist_daily", "BB_width_daily", "Daily_trend_dir"]:
        df[col] = df_d[col].reindex(df.index, method="ffill")

    return df


def fit_hmm(df: pd.DataFrame, n_components: int = 3) -> GaussianHMM:
    """HMMを学習して返す。Return・Volatilityカラムが必要。"""
    df["Return"] = df["Close"].pct_change(24)
    df["Volatility"] = df["Return"].rolling(24).std()
    df_clean = df.dropna(subset=["Return", "Volatility"])

    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(df_clean[["Return", "Volatility"]].values)

    best_model = None
    best_score = -np.inf
    for k in range(5):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=200,
                random_state=42 + k,
            )
            model.fit(X_hmm)
            score = model.score(X_hmm)
            if score > best_score:
                best_score = score
                best_model = model
        except (LinAlgError, ValueError):
            continue

    if best_model is None:
        raise RuntimeError(f"HMM学習に失敗しました（{n_components}状態）")

    return best_model, scaler, df_clean.index


def add_regime_features(df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """HMMレジーム分類を実行し、レジーム関連特徴量を追加する（全データで学習 — 注意: lookahead bias あり）"""
    model_hmm, scaler, clean_idx = fit_hmm(df, n_components)

    df_clean = df.loc[clean_idx]
    X_hmm = scaler.transform(df_clean[["Return", "Volatility"]].values)
    states = model_hmm.predict(X_hmm)

    df["Regime"] = np.nan
    df.loc[clean_idx, "Regime"] = states
    df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)
    df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
    regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
    df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1
    df["Regime_changes_3h"] = (
        df["Regime"].diff().fillna(0) != 0
    ).astype(int).rolling(3).sum().fillna(0)

    return df


def add_regime_features_wf(df: pd.DataFrame, train_end_idx: int, n_components: int = 3) -> pd.DataFrame:
    """Walk-Forward用: trainデータのみでHMMを学習し、全データに適用する（lookahead bias なし）

    Parameters:
        df: 全データ（Return, Volatility カラムが必要）
        train_end_idx: 学習データの最終インデックス位置（exclusive）
        n_components: HMM状態数
    """
    # trainデータのみでHMM学習
    df_train = df.iloc[:train_end_idx].copy()
    if "Return" not in df_train.columns:
        df_train["Return"] = df_train["Close"].pct_change(24)
        df_train["Volatility"] = df_train["Return"].rolling(24).std()
    df_clean_train = df_train.dropna(subset=["Return", "Volatility"])

    scaler = StandardScaler()
    X_hmm_train = scaler.fit_transform(df_clean_train[["Return", "Volatility"]].values)

    best_model = None
    best_score = -np.inf
    for k in range(5):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=200,
                random_state=42 + k,
            )
            model.fit(X_hmm_train)
            score = model.score(X_hmm_train)
            if score > best_score:
                best_score = score
                best_model = model
        except (LinAlgError, ValueError):
            continue

    if best_model is None:
        # フォールバック: 全部レジーム0
        df["Regime"] = 0
        df["Regime_changed"] = 0
        df["Regime_duration"] = 1
        df["Regime_changes_3h"] = 0
        return df

    # 全データにReturn/Volatilityを計算（まだ無い場合）
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change(24)
        df["Volatility"] = df["Return"].rolling(24).std()
    df_clean = df.dropna(subset=["Return", "Volatility"])

    # train scalerで全データを変換 → trainで学習済みHMMで予測
    X_hmm_all = scaler.transform(df_clean[["Return", "Volatility"]].values)
    states = best_model.predict(X_hmm_all)

    df["Regime"] = np.nan
    df.loc[df_clean.index, "Regime"] = states
    df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)
    df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
    regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
    df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1
    df["Regime_changes_3h"] = (
        df["Regime"].diff().fillna(0) != 0
    ).astype(int).rolling(3).sum().fillna(0)

    return df


def prepare_dataset(df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """特徴量とラベルを計算し、欠損値を除去して返す"""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = add_technical_features(df)
    df = add_regime_features(df)

    # ラベル（4時間後の方向）
    df["Close_4h_later"] = df["Close"].shift(-4)
    df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)
    df["Return_4h"] = (df["Close_4h_later"] - df["Close"]) / df["Close"]

    df = df.dropna(subset=feature_cols + ["Label", "Return_4h"])
    return df
