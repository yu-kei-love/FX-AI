# ===========================================
# stock_improvement.py
# 日本株予測モデル v3.4 → v4.0 改善実験
#
# テスト項目:
#   1. テクニカル指標追加 (RSI, MACD, BB for target stock)
#   2. ハイパーパラメータ最適化 (n_estimators, lr, max_depth)
#   3. アンサンブル重みパワー最適化 (accuracy^N)
#   4. ボラティリティレジーム別フィルタリング
#   5. 曜日フィルタリング
#   6. Confidence/Agreement閾値スイープ
#   7. 全改善組み合わせ最終比較
#
# Walk-Forward backtestでベースライン vs 改善版を比較
# ===========================================

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.common.ensemble import EnsembleClassifier

DATA_DIR = PROJECT_ROOT / "data" / "japan_stocks"
RESULTS_PATH = Path(__file__).resolve().parent / "improvement_results.txt"

# Walk-Forward設定 (japan_stock_model.py v3.4と同一)
TRAIN_DAYS = 252
TEST_DAYS = 63
BASELINE_CONF_THRESHOLD = 0.60
BASELINE_AGREE_THRESHOLD = 4

# 米国指数定義
US_INDICES = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW",
    "^VIX": "VIX",
}

US_SECTOR_ETFS = {
    "QQQ": "US_Tech",
    "XLF": "US_Finance",
    "XLE": "US_Energy",
    "XLV": "US_Healthcare",
}

# セクター定義
SECTORS = {
    "semiconductor": {
        "tickers": ["8035.T", "6857.T", "6723.T"],
        "names": ["Tokyo_Electron", "Advantest", "Renesas"],
        "us_proxy": "US_Tech",
    },
    "auto": {
        "tickers": ["7203.T", "7267.T", "6902.T"],
        "names": ["Toyota", "Honda", "Denso"],
        "us_proxy": None,
    },
    "tech": {
        "tickers": ["6758.T", "9984.T", "7974.T", "6501.T"],
        "names": ["Sony", "SoftBank_Group", "Nintendo", "Hitachi"],
        "us_proxy": "US_Tech",
    },
    "finance": {
        "tickers": ["8306.T", "8316.T"],
        "names": ["MUFG", "SMFG"],
        "us_proxy": "US_Finance",
    },
}

STOCK_SECTORS = {
    "8035.T": "semiconductor", "6857.T": "semiconductor", "6723.T": "semiconductor",
    "4063.T": "semiconductor",
    "7203.T": "auto", "7267.T": "auto", "6902.T": "auto",
    "6758.T": "tech", "9984.T": "tech", "7974.T": "tech",
    "6501.T": "tech", "9433.T": "tech",
    "8306.T": "finance", "8316.T": "finance",
}

results_log = []


def log(msg):
    """結果ログに追記しつつprintもする"""
    print(msg)
    results_log.append(msg)


# =============================================================
# データ読み込み
# =============================================================
def load_all_data():
    """保存済みCSVから全データを読み込む"""
    all_data = {}
    name_to_ticker = {
        "SP500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "VIX": "^VIX",
        "Nikkei225": "^N225", "TOPIX_ETF": "1306.T",
        "Toyota": "7203.T", "SoftBank_Group": "9984.T", "Sony": "6758.T",
        "MUFG": "8306.T", "Renesas": "6723.T", "Tokyo_Electron": "8035.T",
        "Advantest": "6857.T", "Nintendo": "7974.T", "Honda": "7267.T",
        "SMFG": "8316.T", "KDDI": "9433.T", "Shin_Etsu": "4063.T",
        "Hitachi": "6501.T", "Denso": "6902.T",
        "US_Tech": "QQQ", "US_Finance": "XLF",
        "US_Energy": "XLE", "US_Healthcare": "XLV",
        "USDJPY_daily": "USDJPY",
    }
    for csv_path in DATA_DIR.glob("*.csv"):
        name = csv_path.stem
        if name == "daily_picks":
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            ticker = name_to_ticker.get(name)
            if ticker:
                all_data[ticker] = df
            all_data[name] = df
        except Exception as e:
            print(f"  [WARN] {name} load error: {e}")
    return all_data


# =============================================================
# ベースライン特徴量生成 (japan_stock_model.py v3.4 完全再現)
# =============================================================
def make_baseline_features(all_data, target_ticker):
    """v3.4ベースライン特徴量を完全再現"""
    if target_ticker not in all_data:
        return None, None, None

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)
    features = pd.DataFrame(index=target_df.index)

    # 米国指数リターン (SP500は除外、NASDAQと相関高い)
    for ticker, name in US_INDICES.items():
        if ticker in all_data and ticker != "^VIX" and ticker != "^GSPC":
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_returns = us_df["Close"].pct_change()
            us_returns.name = f"{name}_Return"
            features = features.join(us_returns, how="left")
            us_ret_5d = us_df["Close"].pct_change(5)
            us_ret_5d.name = f"{name}_Return_5d"
            features = features.join(us_ret_5d, how="left")

    # 米国セクターETFリターン
    for ticker, name in US_SECTOR_ETFS.items():
        if ticker in all_data:
            sec_df = all_data[ticker].copy()
            sec_df.index = pd.to_datetime(sec_df.index)
            sec_returns = sec_df["Close"].pct_change()
            sec_returns.name = f"{name}_Return"
            features = features.join(sec_returns, how="left")

    # VIX
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_level = vix_df["Close"].copy()
        vix_level.name = "VIX_Level"
        features = features.join(vix_level, how="left")
        vix_change = vix_df["Close"].pct_change()
        vix_change.name = "VIX_Change"
        features = features.join(vix_change, how="left")

    # USD/JPY
    usdjpy_name = "USDJPY_daily"
    if usdjpy_name in all_data:
        usdjpy_df = all_data[usdjpy_name].copy()
        usdjpy_df.index = pd.to_datetime(usdjpy_df.index)
        close_col = "USDJPY_Close" if "USDJPY_Close" in usdjpy_df.columns else "Close"
        fx_change = usdjpy_df[close_col].pct_change()
        fx_change.name = "USDJPY_Change"
        features = features.join(fx_change, how="left")
        fx_change_5d = usdjpy_df[close_col].pct_change(5)
        fx_change_5d.name = "USDJPY_Change_5d"
        features = features.join(fx_change_5d, how="left")

    # 対象銘柄のモメンタム
    for period in [1, 5, 20]:
        ret = target_df["Close"].pct_change(period)
        ret.name = f"Target_Return_{period}d"
        features = features.join(ret, how="left")

    # 出来高
    if "Volume" in target_df.columns:
        vol_change = target_df["Volume"].pct_change()
        vol_change.name = "Volume_Change"
        features = features.join(vol_change, how="left")
        vol_ratio = target_df["Volume"] / target_df["Volume"].rolling(20).mean()
        vol_ratio.name = "Volume_Ratio"
        features = features.join(vol_ratio, how="left")

    # 米国指数との相関
    target_returns = target_df["Close"].pct_change()
    for ticker, name in US_INDICES.items():
        if ticker in all_data and ticker != "^VIX":
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_ret = us_df["Close"].pct_change()
            combined = pd.DataFrame({"target": target_returns, "us": us_ret}).dropna()
            if len(combined) > 20:
                rolling_corr = combined["target"].rolling(20).corr(combined["us"])
                rolling_corr.name = f"Corr_{name}_20d"
                features = features.join(rolling_corr, how="left")

    # VIXレジーム
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_close = vix_df["Close"].reindex(features.index, method="ffill")
        vix_pct = vix_close.rolling(120, min_periods=20).rank(pct=True)
        features["VIX_Percentile"] = vix_pct
        vix_of_vix = vix_close.rolling(20, min_periods=5).std()
        features["VIX_of_VIX"] = vix_of_vix
        features["VIX_Return_5d"] = vix_close.pct_change(5)

    # 週次モメンタム
    target_close = target_df["Close"]
    daily_up = (target_close.pct_change() > 0).astype(float)
    features["RSI_Proxy_10d"] = daily_up.rolling(10, min_periods=5).mean()
    features["Cumulative_Return_10d"] = target_close.pct_change(10)
    features["Target_Volatility_20d"] = target_close.pct_change().rolling(20, min_periods=5).std()

    # カレンダー
    features["DayOfWeek"] = features.index.dayofweek
    features["Month"] = features.index.month

    # インタラクション
    for ticker, name in US_INDICES.items():
        corr_col = f"Corr_{name}_20d"
        if corr_col in features.columns:
            features[f"CorrMom_{name}_5d"] = features[corr_col].diff(5)

    if "VIX_Level" in features.columns:
        features["VIX_Regime"] = np.where(
            features["VIX_Level"] < 15, 0,
            np.where(features["VIX_Level"] < 25, 1, 2)
        )
        if "NASDAQ_Return" in features.columns:
            features["VIX_x_NASDAQ"] = features["VIX_Regime"] * features["NASDAQ_Return"]

    if "US_Tech_Return" in features.columns and "US_Finance_Return" in features.columns:
        features["TechFinance_Spread"] = features["US_Tech_Return"] - features["US_Finance_Return"]
        features["TechFinance_Spread_5d"] = features["TechFinance_Spread"].rolling(5).sum()

    if "US_Energy_Return" in features.columns and "US_Tech_Return" in features.columns:
        features["EnergyTech_Spread"] = features["US_Energy_Return"] - features["US_Tech_Return"]

    if "VIX_Percentile" in features.columns and "Target_Return_5d" in features.columns:
        features["VIXPct_x_Mom5d"] = features["VIX_Percentile"] * features["Target_Return_5d"]

    # ターゲット
    future_return = target_df["Close"].pct_change().shift(-1)
    y = (future_return > 0).astype(int)
    y.name = "Target"

    # 欠損値処理
    features = features.ffill()
    common_idx = features.index.intersection(y.dropna().index)
    features = features.loc[common_idx]
    y = y.loc[common_idx]
    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]
    y = y.loc[valid_mask]

    return features, y, features.columns.tolist()


# =============================================================
# 改善1: テクニカル指標特徴量 (RSI, MACD, BB for target stock)
# =============================================================
def add_technical_indicators(features, all_data, target_ticker):
    """対象銘柄にテクニカル指標を追加"""
    if target_ticker not in all_data:
        return features

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)
    close = target_df["Close"]

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "RSI_14"
    features = features.join(rsi, how="left")

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    macd_hist.name = "MACD_Hist"
    features = features.join(macd_hist, how="left")

    # Bollinger Band width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = (2 * std20 * 2) / sma20.replace(0, np.nan)
    bb_width.name = "BB_Width"
    features = features.join(bb_width, how="left")

    # BB position: where close is relative to bands
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos.name = "BB_Position"
    features = features.join(bb_pos, how="left")

    # ATR (Average True Range) - volatility measure
    if all(c in target_df.columns for c in ["High", "Low", "Close"]):
        high = target_df["High"]
        low = target_df["Low"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = atr / close  # ATR as percentage of price
        atr_pct.name = "ATR_Pct_14"
        features = features.join(atr_pct, how="left")

    # Stochastic Oscillator %K
    if all(c in target_df.columns for c in ["High", "Low", "Close"]):
        low14 = target_df["Low"].rolling(14).min()
        high14 = target_df["High"].rolling(14).max()
        stoch_k = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
        stoch_k.name = "Stochastic_K"
        features = features.join(stoch_k, how="left")

    # Rate of change (ROC) 10-day
    roc = close.pct_change(10) * 100
    roc.name = "ROC_10"
    features = features.join(roc, how="left")

    # MA crossover signals
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma_cross = (ma5 - ma20) / close
    ma_cross.name = "MA_Cross_5_20"
    features = features.join(ma_cross, how="left")

    return features


# =============================================================
# 改善2: 追加のクロスマーケット特徴量
# =============================================================
def add_cross_market_features(features, all_data, target_ticker):
    """追加のクロスマーケット特徴量"""
    if target_ticker not in all_data:
        return features

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)

    # Overnight gap features (from v3.4)
    if "Open" in target_df.columns:
        overnight_gap = (target_df["Open"] - target_df["Close"].shift(1)) / target_df["Close"].shift(1)
        overnight_gap.name = "Overnight_Gap"
        features = features.join(overnight_gap, how="left")

        gap_direction = np.sign(overnight_gap)
        gap_direction.name = "Gap_Direction"
        features = features.join(gap_direction, how="left")

        gap_abs = overnight_gap.abs()
        gap_abs.name = "Gap_Abs"
        features = features.join(gap_abs, how="left")

    # US market momentum (3-day, not just 1-day and 5-day)
    for ticker, name in [("^IXIC", "NASDAQ"), ("^DJI", "DOW")]:
        if ticker in all_data:
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_ret_3d = us_df["Close"].pct_change(3)
            us_ret_3d.name = f"{name}_Return_3d"
            features = features.join(us_ret_3d, how="left")

    # Intraday range features (from v3.4 futures proxy)
    if all(c in target_df.columns for c in ["High", "Low", "Close"]):
        intraday_range = (target_df["High"] - target_df["Low"]) / target_df["Close"]
        intraday_range.name = "Intraday_Range"
        features = features.join(intraday_range, how="left")

        range_ma = intraday_range.rolling(20).mean()
        range_ratio = intraday_range / range_ma.replace(0, np.nan)
        range_ratio.name = "Range_vs_MA20"
        features = features.join(range_ratio, how="left")

    # Close position within day's range
    if all(c in target_df.columns for c in ["High", "Low", "Close"]):
        close_position = (target_df["Close"] - target_df["Low"]) / (target_df["High"] - target_df["Low"]).replace(0, np.nan)
        close_position.name = "Close_Position"
        features = features.join(close_position, how="left")

    # VIX term structure proxy & acceleration
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_close = vix_df["Close"].reindex(features.index, method="ffill")
        vix_ma20 = vix_close.rolling(20).mean()
        features["VIX_TermStructure_Proxy"] = vix_close / vix_ma20.replace(0, np.nan)
        features["VIX_Acceleration"] = vix_close.pct_change().diff()

    # Volume-range divergence
    if "Volume" in target_df.columns and "Intraday_Range" in features.columns:
        vol_z = (target_df["Volume"] - target_df["Volume"].rolling(20).mean()) / target_df["Volume"].rolling(20).std().replace(0, np.nan)
        range_z = (features["Intraday_Range"] - features["Intraday_Range"].rolling(20).mean()) / features["Intraday_Range"].rolling(20).std().replace(0, np.nan)
        features["Vol_Range_Divergence"] = vol_z.reindex(features.index) - range_z

    # Sector momentum (if applicable)
    sector_name = STOCK_SECTORS.get(target_ticker)
    if sector_name and sector_name in SECTORS:
        sector = SECTORS[sector_name]
        sector_returns = []
        for tick in sector["tickers"]:
            if tick != target_ticker and tick in all_data:
                df = all_data[tick].copy()
                df.index = pd.to_datetime(df.index)
                ret = df["Close"].pct_change()
                sector_returns.append(ret)
        if sector_returns:
            sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)
            sector_avg.name = f"Sector_AvgReturn"
            features = features.join(sector_avg, how="left")

    return features


# =============================================================
# 改善3: 追加インタラクション特徴量
# =============================================================
def add_extra_interactions(features):
    """追加のインタラクション特徴量"""
    # RSI x VIX regime
    if "RSI_14" in features.columns and "VIX_Regime" in features.columns:
        features["RSI_x_VIXRegime"] = features["RSI_14"] * features["VIX_Regime"]

    # MACD momentum x NASDAQ
    if "MACD_Hist" in features.columns and "NASDAQ_Return" in features.columns:
        features["MACD_x_NASDAQ"] = np.sign(features["MACD_Hist"]) * features["NASDAQ_Return"]

    # Gap x VIX (large gaps in high VIX = different signal)
    if "Overnight_Gap" in features.columns and "VIX_Level" in features.columns:
        features["Gap_x_VIX"] = features["Overnight_Gap"] * features["VIX_Level"]

    # BB position x momentum
    if "BB_Position" in features.columns and "Target_Return_5d" in features.columns:
        features["BB_x_Mom5d"] = features["BB_Position"] * features["Target_Return_5d"]

    # Volume spike x price direction
    if "Volume_Ratio" in features.columns and "Target_Return_1d" in features.columns:
        features["VolSpike_x_Return"] = features["Volume_Ratio"] * features["Target_Return_1d"]

    return features


# =============================================================
# メトリクス計算
# =============================================================
def compute_metrics(df, conf_threshold, agree_threshold, total_bars):
    """DataFrameからメトリクスを計算"""
    high_conf = df[
        (df["confidence"] >= conf_threshold) &
        (df["agreement"] >= agree_threshold)
    ].copy()

    trade_count = len(high_conf)
    if trade_count == 0:
        return {
            "accuracy": 0, "pf": 0, "sharpe": 0, "mdd": 0,
            "win_rate": 0, "trade_count": 0, "total_bars": total_bars,
            "trade_ratio": 0,
        }

    accuracy = high_conf["correct"].mean()
    high_conf["pnl"] = np.where(high_conf["prediction"] == high_conf["actual"], 1.0, -1.0)

    gross_profit = high_conf.loc[high_conf["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(high_conf.loc[high_conf["pnl"] < 0, "pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = (high_conf["pnl"] > 0).mean()

    all_returns = np.zeros(total_bars)
    trade_indices = np.linspace(0, total_bars - 1, trade_count, dtype=int)
    all_returns[trade_indices] = high_conf["pnl"].values
    sharpe = (all_returns.mean() / all_returns.std() * np.sqrt(252)) if all_returns.std() > 0 else 0

    cumulative = np.cumsum(high_conf["pnl"].values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    mdd = drawdown.max() if len(drawdown) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "win_rate": round(win_rate, 4),
        "trade_count": trade_count,
        "total_bars": total_bars,
        "trade_ratio": round(trade_count / total_bars, 4),
    }


# =============================================================
# Walk-Forward テストエンジン
# =============================================================
def run_walk_forward(X, y, n_estimators=200, learning_rate=0.05, max_depth=6,
                     weight_power=3, conf_threshold=0.60, agree_threshold=4,
                     day_filter=None, vix_filter=None):
    """Walk-Forward検証を実行

    Args:
        X, y: features and target
        n_estimators, learning_rate, max_depth: model hyperparameters
        weight_power: power for accuracy-based ensemble weighting
        conf_threshold, agree_threshold: signal filtering thresholds
        day_filter: list of allowed dayofweek values (0-4), None = all days
        vix_filter: tuple (min_vix, max_vix) to filter VIX regime, None = no filter
    """
    total_len = len(X)
    if total_len < TRAIN_DAYS + TEST_DAYS:
        return None

    all_results = []
    fold = 0
    train_end = TRAIN_DAYS

    while train_end + TEST_DAYS <= total_len:
        fold += 1
        test_end = min(train_end + TEST_DAYS, total_len)

        val_size = max(20, int(train_end * 0.1))
        X_train = X.iloc[0:train_end - val_size]
        y_train = y.iloc[0:train_end - val_size]
        X_val = X.iloc[train_end - val_size:train_end]
        y_val = y.iloc[train_end - val_size:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        # Build model with given hyperparameters
        model = EnsembleClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        # Override max_depth for all sub-models
        if max_depth != 6:
            model.model_lgb.max_depth = max_depth
            model.model_xgb.max_depth = max_depth
            model.model_cat.set_params(depth=min(max_depth, 10))
            model.model_rf.max_depth = max_depth + 2
            model.model_et.max_depth = max_depth + 2

        model.fit(X_train.values, y_train.values)

        # Performance-based weights
        weights = []
        for sub_model in model.models:
            preds = sub_model.predict(X_val.values)
            acc = (preds == y_val.values).mean()
            weights.append(acc)
        weights = np.array(weights)
        weights = weights ** weight_power
        weights = weights / weights.sum()

        # Weighted prediction
        probas = np.array([m.predict_proba(X_test.values)[:, 1] for m in model.models])
        weighted_proba = (probas * weights[:, None]).sum(axis=0)
        preds = (weighted_proba >= 0.5).astype(int)

        individual_preds = np.array([m.predict(X_test.values) for m in model.models])
        vote_sum = individual_preds.sum(axis=0)
        agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
        confidence = np.where(preds == 1, weighted_proba, 1.0 - weighted_proba)

        for i in range(len(X_test)):
            result_entry = {
                "date": X_test.index[i],
                "prediction": preds[i],
                "agreement": int(agreement[i]),
                "confidence": confidence[i],
                "actual": y_test.iloc[i],
                "correct": int(preds[i] == y_test.iloc[i]),
                "day_of_week": X_test.index[i].dayofweek,
            }
            # Store VIX level if available
            if "VIX_Level" in X_test.columns:
                result_entry["vix_level"] = X_test.iloc[i]["VIX_Level"]
            all_results.append(result_entry)

        train_end += TEST_DAYS

    if not all_results:
        return None

    df = pd.DataFrame(all_results)

    # Apply day filter
    if day_filter is not None:
        df = df[df["day_of_week"].isin(day_filter)]

    # Apply VIX filter
    if vix_filter is not None and "vix_level" in df.columns:
        min_vix, max_vix = vix_filter
        df = df[(df["vix_level"] >= min_vix) & (df["vix_level"] <= max_vix)]

    if len(df) == 0:
        return None

    metrics = compute_metrics(df, conf_threshold, agree_threshold, total_len)
    metrics["df"] = df
    return metrics


# =============================================================
# TEST 1: テクニカル指標特徴量の効果
# =============================================================
def test_technical_indicators(all_data):
    log("\n" + "=" * 60)
    log("[TEST 1] テクニカル指標特徴量の効果")
    log("=" * 60)

    test_tickers = [
        ("^N225", "Nikkei225"),
        ("8035.T", "Tokyo_Electron"),
        ("7203.T", "Toyota"),
        ("8306.T", "MUFG"),
        ("6758.T", "Sony"),
    ]

    results = {}
    for ticker, name in test_tickers:
        log(f"\n  {ticker} ({name}):")

        # Baseline
        X_base, y_base, _ = make_baseline_features(all_data, ticker)
        if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
            log(f"    [SKIP] データ不足")
            continue

        baseline = run_walk_forward(X_base, y_base)
        if baseline is None:
            log(f"    [SKIP] WF失敗")
            continue

        # + Technical indicators
        X_tech = X_base.copy()
        X_tech = add_technical_indicators(X_tech, all_data, ticker)
        X_tech = X_tech.ffill()
        valid = X_tech.notna().all(axis=1)
        X_tech = X_tech.loc[valid]
        y_tech = y_base.loc[valid]

        tech_result = run_walk_forward(X_tech, y_tech)
        if tech_result is None:
            log(f"    [SKIP] Tech WF失敗")
            continue

        delta_pf = tech_result["pf"] - baseline["pf"]
        log(f"    Baseline:     Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
            f"Sharpe={baseline['sharpe']:.2f} Trades={baseline['trade_count']}")
        log(f"    +Technical:   Acc={tech_result['accuracy']:.1%} PF={tech_result['pf']:.2f} "
            f"Sharpe={tech_result['sharpe']:.2f} Trades={tech_result['trade_count']}")
        log(f"    Delta PF:     {delta_pf:+.2f}")

        results[ticker] = {"baseline": baseline, "tech": tech_result}

    return results


# =============================================================
# TEST 2: クロスマーケット + インタラクション特徴量
# =============================================================
def test_cross_market_features(all_data):
    log("\n" + "=" * 60)
    log("[TEST 2] クロスマーケット + インタラクション特徴量")
    log("=" * 60)

    test_tickers = [
        ("^N225", "Nikkei225"),
        ("8035.T", "Tokyo_Electron"),
        ("7203.T", "Toyota"),
        ("8306.T", "MUFG"),
        ("6758.T", "Sony"),
    ]

    results = {}
    for ticker, name in test_tickers:
        log(f"\n  {ticker} ({name}):")

        X_base, y_base, _ = make_baseline_features(all_data, ticker)
        if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
            log(f"    [SKIP] データ不足")
            continue

        baseline = run_walk_forward(X_base, y_base)
        if baseline is None:
            log(f"    [SKIP] WF失敗")
            continue

        # + Cross-market features
        X_cross = X_base.copy()
        X_cross = add_cross_market_features(X_cross, all_data, ticker)
        X_cross = X_cross.ffill()
        valid = X_cross.notna().all(axis=1)
        X_cross = X_cross.loc[valid]
        y_cross = y_base.loc[valid]

        cross_result = run_walk_forward(X_cross, y_cross)
        if cross_result is None:
            log(f"    [SKIP] Cross WF失敗")
            continue

        delta_pf = cross_result["pf"] - baseline["pf"]
        log(f"    Baseline:      Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
            f"Sharpe={baseline['sharpe']:.2f} Trades={baseline['trade_count']}")
        log(f"    +CrossMarket:  Acc={cross_result['accuracy']:.1%} PF={cross_result['pf']:.2f} "
            f"Sharpe={cross_result['sharpe']:.2f} Trades={cross_result['trade_count']}")
        log(f"    Delta PF:      {delta_pf:+.2f}")

        results[ticker] = {"baseline": baseline, "cross": cross_result}

    return results


# =============================================================
# TEST 3: ハイパーパラメータ最適化
# =============================================================
def test_hyperparameters(all_data):
    log("\n" + "=" * 60)
    log("[TEST 3] ハイパーパラメータ最適化")
    log("=" * 60)

    # Use Nikkei225 as reference
    ticker = "^N225"
    X_base, y_base, _ = make_baseline_features(all_data, ticker)
    if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
        log("  [SKIP] データ不足")
        return None

    # Baseline
    baseline = run_walk_forward(X_base, y_base)
    if baseline is None:
        log("  [SKIP] WF失敗")
        return None

    log(f"\n  Baseline (n=200, lr=0.05, depth=6, power=3):")
    log(f"    Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
        f"Sharpe={baseline['sharpe']:.2f} Trades={baseline['trade_count']}")

    # Grid search
    param_grid = [
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 6, "weight_power": 3},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "weight_power": 3},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 6, "weight_power": 3},
        {"n_estimators": 200, "learning_rate": 0.08, "max_depth": 6, "weight_power": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "weight_power": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 8, "weight_power": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "weight_power": 2},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "weight_power": 4},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "weight_power": 5},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6, "weight_power": 3},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "weight_power": 4},
    ]

    best_score = baseline["pf"]
    best_params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "weight_power": 3}

    log(f"\n  {'n_est':>5} {'lr':>5} {'depth':>5} {'power':>5} | {'Acc':>6} {'PF':>6} {'Sharpe':>6} {'Trades':>6}")
    log(f"  {'-'*60}")

    for params in param_grid:
        result = run_walk_forward(
            X_base, y_base,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            weight_power=params["weight_power"],
        )
        if result is None:
            continue

        log(f"  {params['n_estimators']:>5} {params['learning_rate']:>5} "
            f"{params['max_depth']:>5} {params['weight_power']:>5} | "
            f"{result['accuracy']:.1%} {result['pf']:5.2f} "
            f"{result['sharpe']:5.2f} {result['trade_count']:>6}")

        # Score = PF * min(1, trades/20)
        trade_penalty = min(1.0, result["trade_count"] / 20)
        score = result["pf"] * trade_penalty
        baseline_score = best_score * min(1.0, baseline["trade_count"] / 20)

        if score > baseline_score:
            best_score = result["pf"]
            best_params = params.copy()

    log(f"\n  Best params: n_est={best_params['n_estimators']}, "
        f"lr={best_params['learning_rate']}, depth={best_params['max_depth']}, "
        f"power={best_params['weight_power']}")

    return best_params


# =============================================================
# TEST 4: Confidence/Agreement閾値スイープ
# =============================================================
def test_threshold_sweep(all_data):
    log("\n" + "=" * 60)
    log("[TEST 4] Confidence/Agreement 閾値スイープ")
    log("=" * 60)

    ticker = "^N225"
    X_base, y_base, _ = make_baseline_features(all_data, ticker)
    if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
        log("  [SKIP] データ不足")
        return None

    # Run WF once and sweep thresholds on results
    result = run_walk_forward(X_base, y_base, conf_threshold=0.50, agree_threshold=3)
    if result is None:
        log("  [SKIP] WF失敗")
        return None

    wf_df = result["df"]
    total_len = len(X_base)

    log(f"\n  {'Conf':>5} {'Agree':>5} | {'Acc':>6} {'PF':>6} {'Sharpe':>6} {'Trades':>6} {'WinRate':>7}")
    log(f"  {'-'*60}")

    best_score = -1
    best_params = (0.60, 4)

    for conf_t in np.arange(0.52, 0.76, 0.02):
        for agree_t in [3, 4, 5]:
            metrics = compute_metrics(wf_df, conf_t, agree_t, total_len)
            if metrics["trade_count"] < 5:
                continue

            trade_penalty = min(1.0, metrics["trade_count"] / 20)
            score = metrics["pf"] * trade_penalty

            marker = " <-- current" if (abs(conf_t - 0.60) < 0.01 and agree_t == 4) else ""
            log(f"  {conf_t:.2f}  {agree_t:>5} | "
                f"{metrics['accuracy']:.1%}  "
                f"{metrics['pf']:5.2f}  "
                f"{metrics['sharpe']:5.2f}  "
                f"{metrics['trade_count']:5d}   "
                f"{metrics['win_rate']:.1%}{marker}")

            if score > best_score:
                best_score = score
                best_params = (round(conf_t, 2), agree_t)

    log(f"\n  Best thresholds: Conf={best_params[0]:.2f}, Agree={best_params[1]}")
    log(f"  (Current: Conf={BASELINE_CONF_THRESHOLD}, Agree={BASELINE_AGREE_THRESHOLD})")

    return best_params


# =============================================================
# TEST 5: ボラティリティレジーム別フィルタリング
# =============================================================
def test_volatility_filter(all_data):
    log("\n" + "=" * 60)
    log("[TEST 5] ボラティリティレジーム別フィルタリング")
    log("=" * 60)

    ticker = "^N225"
    X_base, y_base, _ = make_baseline_features(all_data, ticker)
    if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
        log("  [SKIP] データ不足")
        return None

    # Run with low threshold to get all results
    result = run_walk_forward(X_base, y_base, conf_threshold=0.50, agree_threshold=3)
    if result is None:
        log("  [SKIP] WF失敗")
        return None

    wf_df = result["df"]
    total_len = len(X_base)

    if "vix_level" not in wf_df.columns:
        log("  [SKIP] VIX data not available in results")
        return None

    # Analyze by VIX regime
    vix_regimes = [
        ("Low VIX (<15)", 0, 15),
        ("Mid VIX (15-20)", 15, 20),
        ("High-Mid VIX (20-25)", 20, 25),
        ("High VIX (25-30)", 25, 30),
        ("Very High VIX (>30)", 30, 100),
    ]

    log(f"\n  {'Regime':>25} | {'Acc':>6} {'PF':>6} {'Trades':>6} {'WinRate':>7}")
    log(f"  {'-'*60}")

    best_regime = None
    best_regime_pf = 0

    for regime_name, vmin, vmax in vix_regimes:
        regime_df = wf_df[(wf_df["vix_level"] >= vmin) & (wf_df["vix_level"] < vmax)]
        if len(regime_df) < 5:
            log(f"  {regime_name:>25} | insufficient data ({len(regime_df)} bars)")
            continue

        metrics = compute_metrics(regime_df, BASELINE_CONF_THRESHOLD, BASELINE_AGREE_THRESHOLD, len(regime_df))

        log(f"  {regime_name:>25} | "
            f"{metrics['accuracy']:.1%}  "
            f"{metrics['pf']:5.2f}  "
            f"{metrics['trade_count']:5d}   "
            f"{metrics['win_rate']:.1%}")

        if metrics["trade_count"] >= 5 and metrics["pf"] > best_regime_pf:
            best_regime_pf = metrics["pf"]
            best_regime = (vmin, vmax)

    if best_regime:
        log(f"\n  Best VIX regime: {best_regime[0]}-{best_regime[1]} (PF={best_regime_pf:.2f})")

    return best_regime


# =============================================================
# TEST 6: 曜日フィルタリング
# =============================================================
def test_day_filter(all_data):
    log("\n" + "=" * 60)
    log("[TEST 6] 曜日別分析 + フィルタリング")
    log("=" * 60)

    ticker = "^N225"
    X_base, y_base, _ = make_baseline_features(all_data, ticker)
    if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
        log("  [SKIP] データ不足")
        return None

    result = run_walk_forward(X_base, y_base, conf_threshold=0.50, agree_threshold=3)
    if result is None:
        log("  [SKIP] WF失敗")
        return None

    wf_df = result["df"]
    total_len = len(X_base)
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

    log(f"\n  All predictions by day:")
    log(f"  {'Day':>6} | {'Samples':>7} {'Acc':>6} {'UpRatio':>7}")
    log(f"  {'-'*40}")

    for dow in range(5):
        day_data = wf_df[wf_df["day_of_week"] == dow]
        if len(day_data) == 0:
            continue
        acc = day_data["correct"].mean()
        up_ratio = day_data["actual"].mean()
        log(f"  {day_names[dow]:>6} | {len(day_data):>7} {acc:>6.1%} {up_ratio:>7.1%}")

    # High confidence by day
    high_conf = wf_df[
        (wf_df["confidence"] >= BASELINE_CONF_THRESHOLD) &
        (wf_df["agreement"] >= BASELINE_AGREE_THRESHOLD)
    ]

    log(f"\n  High confidence signals by day (Conf>={BASELINE_CONF_THRESHOLD}, Agree>={BASELINE_AGREE_THRESHOLD}):")
    log(f"  {'Day':>6} | {'Signals':>7} {'Acc':>6} {'PF':>6}")
    log(f"  {'-'*40}")

    good_days = []
    for dow in range(5):
        day_data = high_conf[high_conf["day_of_week"] == dow]
        if len(day_data) < 3:
            log(f"  {day_names[dow]:>6} | {len(day_data):>7}  insufficient")
            continue

        acc = day_data["correct"].mean()
        pnl = np.where(day_data["prediction"] == day_data["actual"], 1.0, -1.0)
        gp = pnl[pnl > 0].sum()
        gl = abs(pnl[pnl < 0].sum())
        pf = gp / gl if gl > 0 else float("inf")

        log(f"  {day_names[dow]:>6} | {len(day_data):>7} {acc:>6.1%} {pf:>6.2f}")

        if pf >= 1.0:
            good_days.append(dow)

    if good_days:
        log(f"\n  Profitable days: {[day_names[d] for d in good_days]}")
    else:
        log(f"\n  No consistently profitable days found, keeping all days")
        good_days = None

    return good_days


# =============================================================
# TEST 7: 全改善組み合わせ vs ベースライン (最終比較)
# =============================================================
def test_combined(all_data, best_hyperparams=None, best_thresholds=None,
                  best_vix_regime=None, best_days=None):
    log("\n" + "=" * 60)
    log("[TEST 7] 全改善組み合わせ vs ベースライン (最終比較)")
    log("=" * 60)

    hp = best_hyperparams or {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "weight_power": 3}
    conf_t = best_thresholds[0] if best_thresholds else BASELINE_CONF_THRESHOLD
    agree_t = best_thresholds[1] if best_thresholds else BASELINE_AGREE_THRESHOLD

    log(f"\n  Hyperparams: n_est={hp['n_estimators']}, lr={hp['learning_rate']}, "
        f"depth={hp['max_depth']}, power={hp['weight_power']}")
    log(f"  Thresholds: Conf={conf_t}, Agree={agree_t}")
    if best_vix_regime:
        log(f"  VIX filter: {best_vix_regime[0]}-{best_vix_regime[1]}")
    if best_days:
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        log(f"  Day filter: {[day_names[d] for d in best_days]}")

    test_tickers = [
        ("^N225", "Nikkei225"),
        ("8035.T", "Tokyo_Electron"),
        ("6857.T", "Advantest"),
        ("7203.T", "Toyota"),
        ("6758.T", "Sony"),
        ("8306.T", "MUFG"),
        ("7974.T", "Nintendo"),
        ("7267.T", "Honda"),
    ]

    summary = []

    for ticker, name in test_tickers:
        log(f"\n  {ticker} ({name}):")

        # Baseline (v3.4 settings)
        X_base, y_base, _ = make_baseline_features(all_data, ticker)
        if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
            log(f"    [SKIP] データ不足")
            continue

        baseline = run_walk_forward(X_base, y_base)
        if baseline is None:
            log(f"    [SKIP] ベースラインWF失敗")
            continue

        # Improved: baseline + technical + cross-market + interactions
        X_imp = X_base.copy()
        X_imp = add_technical_indicators(X_imp, all_data, ticker)
        X_imp = add_cross_market_features(X_imp, all_data, ticker)
        X_imp = add_extra_interactions(X_imp)
        X_imp = X_imp.ffill()
        valid = X_imp.notna().all(axis=1)
        X_imp = X_imp.loc[valid]
        y_imp = y_base.loc[valid]

        improved = run_walk_forward(
            X_imp, y_imp,
            n_estimators=hp["n_estimators"],
            learning_rate=hp["learning_rate"],
            max_depth=hp["max_depth"],
            weight_power=hp["weight_power"],
            conf_threshold=conf_t,
            agree_threshold=agree_t,
            day_filter=best_days,
            vix_filter=best_vix_regime,
        )

        if improved is None:
            log(f"    [SKIP] 改善版WF失敗")
            continue

        delta_pf = improved["pf"] - baseline["pf"]
        delta_sharpe = improved["sharpe"] - baseline["sharpe"]
        delta_acc = improved["accuracy"] - baseline["accuracy"]

        log(f"    Baseline:   Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
            f"Sharpe={baseline['sharpe']:.2f} MDD={baseline['mdd']:.0f} Trades={baseline['trade_count']}")
        log(f"    Improved:   Acc={improved['accuracy']:.1%} PF={improved['pf']:.2f} "
            f"Sharpe={improved['sharpe']:.2f} MDD={improved['mdd']:.0f} Trades={improved['trade_count']}")
        log(f"    Delta:      Acc={delta_acc:+.1%} PF={delta_pf:+.2f} Sharpe={delta_sharpe:+.2f}")

        improved_flag = "YES" if (delta_pf > 0.05 or delta_acc > 0.01) else "NO"
        log(f"    Improved?:  {improved_flag}")

        summary.append({
            "ticker": ticker,
            "name": name,
            "baseline_acc": baseline["accuracy"],
            "improved_acc": improved["accuracy"],
            "baseline_pf": baseline["pf"],
            "improved_pf": improved["pf"],
            "baseline_sharpe": baseline["sharpe"],
            "improved_sharpe": improved["sharpe"],
            "baseline_trades": baseline["trade_count"],
            "improved_trades": improved["trade_count"],
            "delta_pf": delta_pf,
            "delta_sharpe": delta_sharpe,
            "improved": improved_flag,
        })

    # Summary
    if summary:
        log("\n" + "=" * 60)
        log("FINAL SUMMARY")
        log("=" * 60)

        improved_count = sum(1 for s in summary if s["improved"] == "YES")
        avg_delta_pf = np.mean([s["delta_pf"] for s in summary])
        avg_delta_sharpe = np.mean([s["delta_sharpe"] for s in summary])
        avg_baseline_pf = np.mean([s["baseline_pf"] for s in summary])
        avg_improved_pf = np.mean([s["improved_pf"] for s in summary])

        log(f"\n  Stocks tested:    {len(summary)}")
        log(f"  Improved:         {improved_count} / {len(summary)}")
        log(f"  Avg Baseline PF:  {avg_baseline_pf:.2f}")
        log(f"  Avg Improved PF:  {avg_improved_pf:.2f}")
        log(f"  Avg Delta PF:     {avg_delta_pf:+.2f}")
        log(f"  Avg Delta Sharpe: {avg_delta_sharpe:+.2f}")

        log(f"\n  Per-stock results:")
        log(f"  {'Ticker':>10} {'Name':>18} {'Base PF':>8} {'Impr PF':>8} {'Delta':>7} {'Improved':>9}")
        log(f"  {'-'*70}")
        for s in summary:
            log(f"  {s['ticker']:>10} {s['name']:>18} {s['baseline_pf']:>8.2f} "
                f"{s['improved_pf']:>8.2f} {s['delta_pf']:>+7.2f} {s['improved']:>9}")

        if improved_count >= len(summary) * 0.5:
            log(f"\n  CONCLUSION: Improvements are significant ({improved_count}/{len(summary)} stocks improved)")
            log(f"  RECOMMENDATION: Update japan_stock_model.py to v4.0")
        else:
            log(f"\n  CONCLUSION: Improvements are limited ({improved_count}/{len(summary)} stocks improved)")
            log(f"  RECOMMENDATION: Keep current v3.4 model, selectively apply improvements")

    return summary


# =============================================================
# メイン
# =============================================================
def main():
    log("=" * 60)
    log(f"Stock Model Improvement Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"Baseline: japan_stock_model.py v3.4")
    log("=" * 60)

    # Load data
    log("\n[LOAD] Loading data...")
    all_data = load_all_data()
    log(f"  Loaded: {len(all_data)} datasets")

    # Run all tests
    test_technical_indicators(all_data)
    test_cross_market_features(all_data)
    best_hyperparams = test_hyperparameters(all_data)
    best_thresholds = test_threshold_sweep(all_data)
    best_vix_regime = test_volatility_filter(all_data)
    best_days = test_day_filter(all_data)

    # Combined final comparison
    summary = test_combined(
        all_data,
        best_hyperparams=best_hyperparams,
        best_thresholds=best_thresholds,
        best_vix_regime=best_vix_regime,
        best_days=best_days,
    )

    # Save results
    log(f"\nResults saved to: {RESULTS_PATH}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results_log))

    return summary


if __name__ == "__main__":
    main()
