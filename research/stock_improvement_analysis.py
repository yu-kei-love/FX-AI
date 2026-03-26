# ===========================================
# stock_improvement_analysis.py
# 日本株予測モデル改善分析
#
# テスト項目:
#   1. セクター別モデル (半導体, 自動車, テック, 金融)
#   2. オーバーナイトギャップ特徴量 (米国終値 vs 日本始値)
#   3. 先物・オプションデータ特徴量
#   4. Confidence閾値最適化
#   5. 曜日別分析
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.common.ensemble import EnsembleClassifier

DATA_DIR = PROJECT_ROOT / "data" / "japan_stocks"
RESULTS_PATH = Path(__file__).resolve().parent / "stock_improvement_results.txt"

# Walk-Forward設定 (japan_stock_model.pyと同一)
TRAIN_DAYS = 252
TEST_DAYS = 63
BASELINE_CONF_THRESHOLD = 0.60
BASELINE_AGREE_THRESHOLD = 4

# セクター定義
SECTORS = {
    "semiconductor": {
        "tickers": ["8035.T", "6857.T", "6723.T"],
        "names": ["Tokyo_Electron", "Advantest", "Renesas"],
        "us_proxy": "US_Tech",   # QQQ
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
    for csv_path in DATA_DIR.glob("*.csv"):
        name = csv_path.stem
        if name == "daily_picks":
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            # ティッカーを逆引き
            ticker = _name_to_ticker(name)
            if ticker:
                all_data[ticker] = df
            all_data[name] = df  # 名前でもアクセス可能に
        except Exception as e:
            print(f"  [WARN] {name} 読み込みエラー: {e}")
    return all_data


def _name_to_ticker(name):
    """名前からティッカーへの逆引き"""
    all_maps = {
        "SP500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "VIX": "^VIX",
        "Nikkei225": "^N225", "TOPIX_ETF": "1306.T",
        "Toyota": "7203.T", "SoftBank_Group": "9984.T", "Sony": "6758.T",
        "MUFG": "8306.T", "Renesas": "6723.T", "Tokyo_Electron": "8035.T",
        "Advantest": "6857.T", "Nintendo": "7974.T", "Honda": "7267.T",
        "SMFG": "8316.T", "KDDI": "9433.T", "Shin_Etsu": "4063.T",
        "Hitachi": "6501.T", "Denso": "6902.T",
        "US_Tech": "QQQ", "US_Finance": "XLF",
        "US_Energy": "XLE", "US_Healthcare": "XLV",
    }
    return all_maps.get(name)


# =============================================================
# ベースライン特徴量生成 (japan_stock_model.pyのmake_features再現)
# =============================================================
def make_baseline_features(all_data, target_ticker):
    """ベースライン特徴量 = 現行japan_stock_model.pyと同一ロジック"""
    if target_ticker not in all_data:
        return None, None, None

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)
    features = pd.DataFrame(index=target_df.index)

    # 米国指数リターン
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

    # 相関
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
# 改善版特徴量: オーバーナイトギャップ + 先物/オプション proxy
# =============================================================
def add_overnight_gap_features(features, all_data, target_ticker):
    """米国終値 vs 日本始値のギャップ特徴量を追加"""
    if target_ticker not in all_data:
        return features

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)

    # 日本市場の前日終値 vs 当日始値 (=オーバーナイトギャップ)
    if "Open" in target_df.columns:
        overnight_gap = (target_df["Open"] - target_df["Close"].shift(1)) / target_df["Close"].shift(1)
        overnight_gap.name = "Overnight_Gap"
        features = features.join(overnight_gap, how="left")

        # ギャップの方向 (正=ギャップアップ, 負=ギャップダウン)
        gap_direction = np.sign(overnight_gap)
        gap_direction.name = "Gap_Direction"
        features = features.join(gap_direction, how="left")

        # ギャップサイズの絶対値
        gap_abs = overnight_gap.abs()
        gap_abs.name = "Gap_Abs"
        features = features.join(gap_abs, how="left")

    # 米国S&P500終値 vs 日本始値の乖離 (クロスマーケットギャップ)
    if "^GSPC" in all_data and "Open" in target_df.columns:
        sp500_df = all_data["^GSPC"].copy()
        sp500_df.index = pd.to_datetime(sp500_df.index)
        sp500_ret = sp500_df["Close"].pct_change()
        sp500_ret.name = "SP500_Return_for_gap"
        # 米国リターンと日本ギャップの差 = サプライズ度
        combined = features.join(sp500_ret, how="left")
        if "Overnight_Gap" in features.columns and "SP500_Return_for_gap" in combined.columns:
            features["Gap_vs_SP500"] = features["Overnight_Gap"] - combined["SP500_Return_for_gap"].ffill()

    # NASDAQ終値リターン vs 日本ギャップ (テック系)
    if "^IXIC" in all_data and "Overnight_Gap" in features.columns:
        nasdaq_df = all_data["^IXIC"].copy()
        nasdaq_df.index = pd.to_datetime(nasdaq_df.index)
        nasdaq_ret = nasdaq_df["Close"].pct_change()
        nasdaq_ret_aligned = nasdaq_ret.reindex(features.index, method="ffill")
        features["Gap_vs_NASDAQ"] = features["Overnight_Gap"] - nasdaq_ret_aligned

    return features


def add_futures_proxy_features(features, all_data, target_ticker):
    """先物/オプションデータのプロキシ特徴量を追加

    実際の先物データがない場合、プロキシとして:
    - 日中レンジ / ボラティリティ比率
    - VIXの term structure proxy
    - 出来高/ボラティリティの乖離
    """
    if target_ticker not in all_data:
        return features

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)

    # 日中レンジ比率 (High-Low)/Close
    if all(c in target_df.columns for c in ["High", "Low", "Close"]):
        intraday_range = (target_df["High"] - target_df["Low"]) / target_df["Close"]
        intraday_range.name = "Intraday_Range"
        features = features.join(intraday_range, how="left")

        # レンジの20日平均との乖離 (= 異常ボラティリティ検知)
        range_ma = intraday_range.rolling(20).mean()
        range_ratio = intraday_range / range_ma
        range_ratio.name = "Range_vs_MA20"
        features = features.join(range_ratio, how="left")

    # VIX term structure proxy: VIX水準 vs VIX 20日MA
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_close = vix_df["Close"].reindex(features.index, method="ffill")
        vix_ma20 = vix_close.rolling(20).mean()
        features["VIX_TermStructure_Proxy"] = vix_close / vix_ma20

        # VIX速度 (VIXの変化速度 = 恐怖の加速度)
        features["VIX_Acceleration"] = vix_close.pct_change().diff()

    # 出来高ボラティリティ乖離: 出来高急増 but 値動き小 = 転換点サイン
    if "Volume" in target_df.columns and "Intraday_Range" in features.columns:
        vol_z = (target_df["Volume"] - target_df["Volume"].rolling(20).mean()) / target_df["Volume"].rolling(20).std()
        range_z = (features["Intraday_Range"] - features["Intraday_Range"].rolling(20).mean()) / features["Intraday_Range"].rolling(20).std()
        features["Vol_Range_Divergence"] = vol_z.reindex(features.index) - range_z

    # 終値位置: (Close - Low) / (High - Low) = 日中での終値の位置
    if all(c in target_df.columns for c in ["High", "Low", "Close"]):
        close_position = (target_df["Close"] - target_df["Low"]) / (target_df["High"] - target_df["Low"])
        close_position.name = "Close_Position"
        features = features.join(close_position, how="left")

    return features


def add_sector_specific_features(features, all_data, target_ticker, sector_name):
    """セクター固有の特徴量を追加"""
    sector = SECTORS.get(sector_name)
    if not sector:
        return features

    # 同セクター他銘柄の平均リターン (セクターモメンタム)
    sector_returns = []
    for tick in sector["tickers"]:
        if tick != target_ticker and tick in all_data:
            df = all_data[tick].copy()
            df.index = pd.to_datetime(df.index)
            ret = df["Close"].pct_change()
            sector_returns.append(ret)

    if sector_returns:
        sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)
        sector_avg.name = f"Sector_{sector_name}_AvgReturn"
        features = features.join(sector_avg, how="left")

        # セクター平均の5日リターン
        sector_avg_5d = pd.concat(sector_returns, axis=1).mean(axis=1).rolling(5).sum()
        sector_avg_5d.name = f"Sector_{sector_name}_AvgReturn_5d"
        features = features.join(sector_avg_5d, how="left")

    # 米国プロキシとの相関強度
    us_proxy = sector.get("us_proxy")
    if us_proxy and us_proxy in all_data:
        proxy_df = all_data[us_proxy].copy()
        proxy_df.index = pd.to_datetime(proxy_df.index)
        proxy_ret = proxy_df["Close"].pct_change()
        proxy_ret.name = f"USProxy_{sector_name}_Return"
        features = features.join(proxy_ret, how="left")

        # 米国プロキシの5日リターン
        proxy_ret_5d = proxy_df["Close"].pct_change(5)
        proxy_ret_5d.name = f"USProxy_{sector_name}_Return_5d"
        features = features.join(proxy_ret_5d, how="left")

    return features


# =============================================================
# Walk-Forward エンジン
# =============================================================
def walk_forward_test(X, y, conf_threshold=0.60, agree_threshold=4, label="baseline"):
    """Walk-Forward検証を実行し、メトリクスを返す"""
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

        model = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train.values, y_train.values)

        # 性能ベース重み計算
        weights = []
        for sub_model in model.models:
            preds = sub_model.predict(X_val.values)
            acc = (preds == y_val.values).mean()
            weights.append(acc)
        weights = np.array(weights)
        weights = weights ** 3
        weights = weights / weights.sum()

        # 重み付き予測
        probas = np.array([m.predict_proba(X_test.values)[:, 1] for m in model.models])
        weighted_proba = (probas * weights[:, None]).sum(axis=0)
        preds = (weighted_proba >= 0.5).astype(int)

        individual_preds = np.array([m.predict(X_test.values) for m in model.models])
        vote_sum = individual_preds.sum(axis=0)
        agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)
        confidence = np.where(preds == 1, weighted_proba, 1.0 - weighted_proba)

        for i in range(len(X_test)):
            all_results.append({
                "date": X_test.index[i],
                "prediction": preds[i],
                "agreement": int(agreement[i]),
                "confidence": confidence[i],
                "actual": y_test.iloc[i],
                "correct": int(preds[i] == y_test.iloc[i]),
                "day_of_week": X_test.index[i].dayofweek,
            })

        train_end += TEST_DAYS

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    return compute_metrics_from_df(df, conf_threshold, agree_threshold, total_len)


def compute_metrics_from_df(df, conf_threshold, agree_threshold, total_bars):
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
            "trade_ratio": 0, "df": df,
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
        "df": df,
    }


# =============================================================
# テスト1: セクター別モデル
# =============================================================
def test_sector_models(all_data):
    """セクター別にモデルを構築し、汎用モデルと比較"""
    log("\n" + "=" * 60)
    log("[TEST 1] セクター別モデル vs 汎用モデル")
    log("=" * 60)

    for sector_name, sector_info in SECTORS.items():
        log(f"\n--- セクター: {sector_name} ---")

        for ticker in sector_info["tickers"]:
            name = sector_info["names"][sector_info["tickers"].index(ticker)]
            log(f"\n  銘柄: {ticker} ({name})")

            # ベースライン
            X_base, y_base, _ = make_baseline_features(all_data, ticker)
            if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
                log(f"    [SKIP] データ不足")
                continue

            baseline = walk_forward_test(X_base, y_base, label="baseline")
            if baseline is None:
                log(f"    [SKIP] WF失敗")
                continue

            # セクター特化版
            X_sector, y_sector, _ = make_baseline_features(all_data, ticker)
            X_sector = add_sector_specific_features(X_sector, all_data, ticker, sector_name)
            X_sector = X_sector.ffill()
            valid = X_sector.notna().all(axis=1)
            X_sector = X_sector.loc[valid]
            y_sector = y_sector.loc[valid]

            sector_result = walk_forward_test(X_sector, y_sector, label="sector")

            if sector_result is None:
                log(f"    [SKIP] セクターWF失敗")
                continue

            delta_acc = sector_result["accuracy"] - baseline["accuracy"]
            delta_pf = sector_result["pf"] - baseline["pf"]

            log(f"    ベースライン:    Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
                f"Sharpe={baseline['sharpe']:.2f} Trades={baseline['trade_count']}")
            log(f"    セクター特化:    Acc={sector_result['accuracy']:.1%} PF={sector_result['pf']:.2f} "
                f"Sharpe={sector_result['sharpe']:.2f} Trades={sector_result['trade_count']}")
            log(f"    差分:            Acc={delta_acc:+.1%} PF={delta_pf:+.2f}")


# =============================================================
# テスト2: オーバーナイトギャップ特徴量
# =============================================================
def test_overnight_gap(all_data):
    """オーバーナイトギャップ特徴量の効果を検証"""
    log("\n" + "=" * 60)
    log("[TEST 2] オーバーナイトギャップ特徴量")
    log("=" * 60)

    test_tickers = [("^N225", "Nikkei225"), ("8035.T", "Tokyo_Electron"),
                    ("7203.T", "Toyota"), ("8306.T", "MUFG")]

    for ticker, name in test_tickers:
        log(f"\n  銘柄: {ticker} ({name})")

        X_base, y_base, _ = make_baseline_features(all_data, ticker)
        if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
            log(f"    [SKIP] データ不足")
            continue

        baseline = walk_forward_test(X_base, y_base, label="baseline")
        if baseline is None:
            log(f"    [SKIP] WF失敗")
            continue

        # オーバーナイトギャップ追加版
        X_gap, y_gap, _ = make_baseline_features(all_data, ticker)
        X_gap = add_overnight_gap_features(X_gap, all_data, ticker)
        X_gap = X_gap.ffill()
        valid = X_gap.notna().all(axis=1)
        X_gap = X_gap.loc[valid]
        y_gap = y_gap.loc[valid]

        gap_result = walk_forward_test(X_gap, y_gap, label="gap")
        if gap_result is None:
            log(f"    [SKIP] ギャップWF失敗")
            continue

        delta_acc = gap_result["accuracy"] - baseline["accuracy"]
        delta_pf = gap_result["pf"] - baseline["pf"]

        log(f"    ベースライン:     Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
            f"Sharpe={baseline['sharpe']:.2f} Trades={baseline['trade_count']}")
        log(f"    +ギャップ特徴量:  Acc={gap_result['accuracy']:.1%} PF={gap_result['pf']:.2f} "
            f"Sharpe={gap_result['sharpe']:.2f} Trades={gap_result['trade_count']}")
        log(f"    差分:             Acc={delta_acc:+.1%} PF={delta_pf:+.2f}")


# =============================================================
# テスト3: 先物/オプション proxy 特徴量
# =============================================================
def test_futures_proxy(all_data):
    """先物/オプションのプロキシ特徴量の効果を検証"""
    log("\n" + "=" * 60)
    log("[TEST 3] 先物/オプション Proxy 特徴量")
    log("=" * 60)

    test_tickers = [("^N225", "Nikkei225"), ("6857.T", "Advantest"),
                    ("6758.T", "Sony"), ("8316.T", "SMFG")]

    for ticker, name in test_tickers:
        log(f"\n  銘柄: {ticker} ({name})")

        X_base, y_base, _ = make_baseline_features(all_data, ticker)
        if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
            log(f"    [SKIP] データ不足")
            continue

        baseline = walk_forward_test(X_base, y_base, label="baseline")
        if baseline is None:
            log(f"    [SKIP] WF失敗")
            continue

        # 先物proxy追加版
        X_fut, y_fut, _ = make_baseline_features(all_data, ticker)
        X_fut = add_futures_proxy_features(X_fut, all_data, ticker)
        X_fut = X_fut.ffill()
        valid = X_fut.notna().all(axis=1)
        X_fut = X_fut.loc[valid]
        y_fut = y_fut.loc[valid]

        fut_result = walk_forward_test(X_fut, y_fut, label="futures_proxy")
        if fut_result is None:
            log(f"    [SKIP] 先物proxyWF失敗")
            continue

        delta_acc = fut_result["accuracy"] - baseline["accuracy"]
        delta_pf = fut_result["pf"] - baseline["pf"]

        log(f"    ベースライン:     Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
            f"Sharpe={baseline['sharpe']:.2f} Trades={baseline['trade_count']}")
        log(f"    +先物Proxy:       Acc={fut_result['accuracy']:.1%} PF={fut_result['pf']:.2f} "
            f"Sharpe={fut_result['sharpe']:.2f} Trades={fut_result['trade_count']}")
        log(f"    差分:             Acc={delta_acc:+.1%} PF={delta_pf:+.2f}")


# =============================================================
# テスト4: Confidence閾値最適化
# =============================================================
def test_confidence_threshold(all_data):
    """Confidence閾値とAgreement閾値の最適化"""
    log("\n" + "=" * 60)
    log("[TEST 4] Confidence閾値最適化")
    log("=" * 60)

    # Nikkei225 + 全改善特徴量で閾値をスイープ
    ticker = "^N225"
    X_full, y_full, _ = make_baseline_features(all_data, ticker)
    if X_full is None:
        log("  [SKIP] Nikkei225データなし")
        return

    X_full = add_overnight_gap_features(X_full, all_data, ticker)
    X_full = add_futures_proxy_features(X_full, all_data, ticker)
    X_full = X_full.ffill()
    valid = X_full.notna().all(axis=1)
    X_full = X_full.loc[valid]
    y_full = y_full.loc[valid]

    if len(X_full) < TRAIN_DAYS + TEST_DAYS:
        log("  [SKIP] データ不足")
        return

    # まずWFを1回だけ実行して全結果を取得
    total_len = len(X_full)
    all_wf_results = []
    fold = 0
    train_end = TRAIN_DAYS

    while train_end + TEST_DAYS <= total_len:
        fold += 1
        test_end = min(train_end + TEST_DAYS, total_len)

        val_size = max(20, int(train_end * 0.1))
        X_train = X_full.iloc[0:train_end - val_size]
        y_train = y_full.iloc[0:train_end - val_size]
        X_val = X_full.iloc[train_end - val_size:train_end]
        y_val = y_full.iloc[train_end - val_size:train_end]
        X_test = X_full.iloc[train_end:test_end]
        y_test = y_full.iloc[train_end:test_end]

        model = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train.values, y_train.values)

        weights = []
        for sub_model in model.models:
            preds = sub_model.predict(X_val.values)
            acc = (preds == y_val.values).mean()
            weights.append(acc)
        weights = np.array(weights) ** 3
        weights = weights / weights.sum()

        probas = np.array([m.predict_proba(X_test.values)[:, 1] for m in model.models])
        weighted_proba = (probas * weights[:, None]).sum(axis=0)
        preds_arr = (weighted_proba >= 0.5).astype(int)

        individual_preds = np.array([m.predict(X_test.values) for m in model.models])
        vote_sum = individual_preds.sum(axis=0)
        agreement = np.where(preds_arr == 1, vote_sum, 5 - vote_sum)
        confidence = np.where(preds_arr == 1, weighted_proba, 1.0 - weighted_proba)

        for i in range(len(X_test)):
            all_wf_results.append({
                "date": X_test.index[i],
                "prediction": preds_arr[i],
                "agreement": int(agreement[i]),
                "confidence": confidence[i],
                "actual": y_test.iloc[i],
                "correct": int(preds_arr[i] == y_test.iloc[i]),
                "day_of_week": X_test.index[i].dayofweek,
            })

        train_end += TEST_DAYS

    wf_df = pd.DataFrame(all_wf_results)

    # 閾値スイープ
    log(f"\n  日経225 (改善版特徴量) - 閾値スイープ:")
    log(f"  {'Conf':>5} {'Agree':>5} | {'Acc':>6} {'PF':>6} {'Sharpe':>6} {'Trades':>6} {'WinRate':>7}")
    log(f"  {'-'*55}")

    best_score = -1
    best_params = (0.60, 4)

    for conf_t in np.arange(0.52, 0.76, 0.02):
        for agree_t in [3, 4, 5]:
            metrics = compute_metrics_from_df(wf_df, conf_t, agree_t, total_len)
            if metrics["trade_count"] < 10:
                continue

            # スコア = PF * min(1, trades/30) ... トレード数が少なすぎるとペナルティ
            trade_penalty = min(1.0, metrics["trade_count"] / 30)
            score = metrics["pf"] * trade_penalty

            log(f"  {conf_t:.2f}  {agree_t:>5} | "
                f"{metrics['accuracy']:.1%}  "
                f"{metrics['pf']:5.2f}  "
                f"{metrics['sharpe']:5.2f}  "
                f"{metrics['trade_count']:5d}   "
                f"{metrics['win_rate']:.1%}")

            if score > best_score:
                best_score = score
                best_params = (conf_t, agree_t)

    log(f"\n  最適閾値: Confidence={best_params[0]:.2f}, Agreement={best_params[1]}")
    log(f"  (現行: Confidence={BASELINE_CONF_THRESHOLD}, Agreement={BASELINE_AGREE_THRESHOLD})")

    return best_params


# =============================================================
# テスト5: 曜日別分析
# =============================================================
def test_day_of_week(all_data):
    """曜日別の予測精度を分析"""
    log("\n" + "=" * 60)
    log("[TEST 5] 曜日別分析")
    log("=" * 60)

    ticker = "^N225"
    X_base, y_base, _ = make_baseline_features(all_data, ticker)
    if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
        log("  [SKIP] データ不足")
        return

    result = walk_forward_test(X_base, y_base, label="dow_analysis")
    if result is None:
        log("  [SKIP] WF失敗")
        return

    wf_df = result["df"]
    day_names = {0: "月曜", 1: "火曜", 2: "水曜", 3: "木曜", 4: "金曜"}

    log(f"\n  日経225 - 曜日別 全予測:")
    log(f"  {'曜日':>6} | {'サンプル':>6} {'正解率':>6} {'上昇割合':>6}")
    log(f"  {'-'*40}")

    for dow in range(5):
        day_data = wf_df[wf_df["day_of_week"] == dow]
        if len(day_data) == 0:
            continue
        acc = day_data["correct"].mean()
        up_ratio = day_data["actual"].mean()
        log(f"  {day_names[dow]:>6} | {len(day_data):>6} {acc:>6.1%} {up_ratio:>6.1%}")

    # 高信頼度のみ
    high_conf = wf_df[
        (wf_df["confidence"] >= BASELINE_CONF_THRESHOLD) &
        (wf_df["agreement"] >= BASELINE_AGREE_THRESHOLD)
    ]

    log(f"\n  日経225 - 曜日別 高信頼度シグナル (Conf>={BASELINE_CONF_THRESHOLD}, Agree>={BASELINE_AGREE_THRESHOLD}):")
    log(f"  {'曜日':>6} | {'シグナル':>6} {'正解率':>6} {'PF':>6}")
    log(f"  {'-'*40}")

    best_day = None
    best_day_acc = 0

    for dow in range(5):
        day_data = high_conf[high_conf["day_of_week"] == dow]
        if len(day_data) < 3:
            log(f"  {day_names[dow]:>6} | {len(day_data):>6}  データ不足")
            continue

        acc = day_data["correct"].mean()
        pnl = np.where(day_data["prediction"] == day_data["actual"], 1.0, -1.0)
        gp = pnl[pnl > 0].sum()
        gl = abs(pnl[pnl < 0].sum())
        pf = gp / gl if gl > 0 else float("inf")

        log(f"  {day_names[dow]:>6} | {len(day_data):>6} {acc:>6.1%} {pf:>6.2f}")

        if acc > best_day_acc:
            best_day_acc = acc
            best_day = dow

    if best_day is not None:
        log(f"\n  最も予測精度が高い曜日: {day_names[best_day]} ({best_day_acc:.1%})")


# =============================================================
# テスト6: 全改善組み合わせ vs ベースライン
# =============================================================
def test_combined_improvement(all_data, best_conf=None):
    """全改善を組み合わせたモデル vs ベースラインの最終比較"""
    log("\n" + "=" * 60)
    log("[TEST 6] 全改善組み合わせ vs ベースライン (最終比較)")
    log("=" * 60)

    conf_t = best_conf[0] if best_conf else BASELINE_CONF_THRESHOLD
    agree_t = best_conf[1] if best_conf else BASELINE_AGREE_THRESHOLD

    test_tickers = [
        ("^N225", "Nikkei225", None),
        ("8035.T", "Tokyo_Electron", "semiconductor"),
        ("6857.T", "Advantest", "semiconductor"),
        ("7203.T", "Toyota", "auto"),
        ("6758.T", "Sony", "tech"),
        ("8306.T", "MUFG", "finance"),
        ("7974.T", "Nintendo", "tech"),
        ("7267.T", "Honda", "auto"),
    ]

    summary = []

    for ticker, name, sector in test_tickers:
        log(f"\n  {ticker} ({name}):")

        # ベースライン
        X_base, y_base, _ = make_baseline_features(all_data, ticker)
        if X_base is None or len(X_base) < TRAIN_DAYS + TEST_DAYS:
            log(f"    [SKIP] データ不足")
            continue

        baseline = walk_forward_test(X_base, y_base,
                                     conf_threshold=BASELINE_CONF_THRESHOLD,
                                     agree_threshold=BASELINE_AGREE_THRESHOLD,
                                     label="baseline")
        if baseline is None:
            log(f"    [SKIP] ベースラインWF失敗")
            continue

        # 改善版 (ギャップ + 先物proxy + セクター)
        X_imp, y_imp, _ = make_baseline_features(all_data, ticker)
        X_imp = add_overnight_gap_features(X_imp, all_data, ticker)
        X_imp = add_futures_proxy_features(X_imp, all_data, ticker)
        if sector:
            X_imp = add_sector_specific_features(X_imp, all_data, ticker, sector)
        X_imp = X_imp.ffill()
        valid = X_imp.notna().all(axis=1)
        X_imp = X_imp.loc[valid]
        y_imp = y_imp.loc[valid]

        improved = walk_forward_test(X_imp, y_imp,
                                     conf_threshold=conf_t,
                                     agree_threshold=agree_t,
                                     label="improved")
        if improved is None:
            log(f"    [SKIP] 改善版WF失敗")
            continue

        delta_acc = improved["accuracy"] - baseline["accuracy"]
        delta_pf = improved["pf"] - baseline["pf"]
        delta_sharpe = improved["sharpe"] - baseline["sharpe"]

        log(f"    ベースライン:  Acc={baseline['accuracy']:.1%} PF={baseline['pf']:.2f} "
            f"Sharpe={baseline['sharpe']:.2f} MDD={baseline['mdd']:.0f} Trades={baseline['trade_count']}")
        log(f"    改善版:        Acc={improved['accuracy']:.1%} PF={improved['pf']:.2f} "
            f"Sharpe={improved['sharpe']:.2f} MDD={improved['mdd']:.0f} Trades={improved['trade_count']}")
        log(f"    差分:          Acc={delta_acc:+.1%} PF={delta_pf:+.2f} Sharpe={delta_sharpe:+.2f}")

        improved_flag = "YES" if (delta_pf > 0.05 or delta_acc > 0.01) else "NO"
        log(f"    改善判定:      {improved_flag}")

        summary.append({
            "ticker": ticker,
            "name": name,
            "baseline_acc": baseline["accuracy"],
            "improved_acc": improved["accuracy"],
            "baseline_pf": baseline["pf"],
            "improved_pf": improved["pf"],
            "baseline_sharpe": baseline["sharpe"],
            "improved_sharpe": improved["sharpe"],
            "delta_acc": delta_acc,
            "delta_pf": delta_pf,
            "delta_sharpe": delta_sharpe,
            "improved": improved_flag,
        })

    # サマリー
    if summary:
        log("\n" + "=" * 60)
        log("最終サマリー")
        log("=" * 60)

        improved_count = sum(1 for s in summary if s["improved"] == "YES")
        avg_delta_acc = np.mean([s["delta_acc"] for s in summary])
        avg_delta_pf = np.mean([s["delta_pf"] for s in summary])
        avg_delta_sharpe = np.mean([s["delta_sharpe"] for s in summary])

        log(f"  テスト銘柄数: {len(summary)}")
        log(f"  改善銘柄数:   {improved_count} / {len(summary)}")
        log(f"  平均Acc差:    {avg_delta_acc:+.2%}")
        log(f"  平均PF差:     {avg_delta_pf:+.2f}")
        log(f"  平均Sharpe差: {avg_delta_sharpe:+.2f}")

        if conf_t != BASELINE_CONF_THRESHOLD or agree_t != BASELINE_AGREE_THRESHOLD:
            log(f"  最適閾値:     Conf={conf_t:.2f}, Agree={agree_t}")

        if improved_count >= len(summary) * 0.5:
            log(f"\n  結論: 改善は有意 (過半数の銘柄で改善) → japan_stock_model.py更新推奨")
        else:
            log(f"\n  結論: 改善は限定的 (過半数で未改善) → 現行モデル維持推奨")

    return summary


# =============================================================
# メイン
# =============================================================
def main():
    log("=" * 60)
    log(f"日本株予測モデル改善分析 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 60)

    # データ読み込み
    log("\n[LOAD] データ読み込み中...")
    all_data = load_all_data()
    log(f"  読み込み完了: {len(all_data)}データセット")

    # テスト実行
    test_sector_models(all_data)
    test_overnight_gap(all_data)
    test_futures_proxy(all_data)
    best_params = test_confidence_threshold(all_data)
    test_day_of_week(all_data)
    summary = test_combined_improvement(all_data, best_params)

    # 結果保存
    log(f"\n結果保存: {RESULTS_PATH}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results_log))

    return summary


if __name__ == "__main__":
    main()
