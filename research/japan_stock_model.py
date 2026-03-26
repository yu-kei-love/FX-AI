# ===========================================
# japan_stock_model.py  v3.4
# 日本株予測モデル: 米国→日本 相関戦略
#
# v3.4 改善点:
#   - オーバーナイトギャップ特徴量（米国終値 vs 日本始値）
#   - 先物/オプション Proxy 特徴量（日中レンジ、終値位置、VIX加速度）
#   - セクター特化特徴量（同セクター銘柄モメンタム）
#
# v3.0 改善点:
#   - 性能ベースアンサンブル重み付け（accuracy^3）
#   - ボラティリティレジーム特徴量（VIX_percentile, VIX_of_VIX）
#   - 週次モメンタム特徴量（10d RSI proxy, 累積リターン）
#   - VIXレジーム適応閾値
#   - 米国セクター追加（XLE, XLV）
#
# パイプライン:
#   1. 米国指数・日本株のデータ取得
#   2. 特徴量生成（米国リターン、VIX、為替、モメンタム、ギャップ等）
#   3. 5モデルアンサンブルで予測（性能ベース重み付け）
#   4. Walk-Forward検証
#   5. スクリーニング → Telegram通知
# ===========================================

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# プロジェクトルートをパスに追加（共通モジュール読み込み用）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.common.ensemble import EnsembleClassifier

warnings.filterwarnings("ignore")


# =============================================================
# PF / Sharpe / MDD メトリクス計算（全バーベース）
# =============================================================
def compute_trading_metrics(results_df, all_bars_count=None):
    """Walk-Forward結果からPF, Sharpe, MDDを計算する

    重要: Sharpeは「全バー」ベースで計算する（トレードバーだけだと過大評価になる）

    Args:
        results_df: prediction, actual, confidence, agreement列を持つDataFrame
        all_bars_count: 全バー数（Sharpe計算用）。Noneの場合はlen(results_df)を使用

    Returns:
        dict: {pf, sharpe, mdd, win_rate, trade_count, ...}
    """
    if results_df is None or len(results_df) == 0:
        return {"pf": 0, "sharpe": 0, "mdd": 0, "win_rate": 0, "trade_count": 0}

    # 高信頼度シグナルのみ抽出
    trades = results_df[
        (results_df["confidence"] >= CONFIDENCE_THRESHOLD) &
        (results_df["agreement"] >= AGREEMENT_THRESHOLD)
    ].copy()

    trade_count = len(trades)
    if trade_count == 0:
        return {"pf": 0, "sharpe": 0, "mdd": 0, "win_rate": 0, "trade_count": 0}

    # リターン計算: 予測方向にポジションを取った場合の損益
    # actual=1(上昇), prediction=1(買い) → +1 (正解)
    # actual=0(下落), prediction=1(買い) → -1 (不正解)
    # actual=0(下落), prediction=0(売り) → +1 (正解)
    # actual=1(上昇), prediction=0(売り) → -1 (不正解)
    trades["pnl"] = np.where(trades["prediction"] == trades["actual"], 1.0, -1.0)

    # Profit Factor
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win Rate
    win_rate = (trades["pnl"] > 0).mean()

    # Sharpe Ratio（全バーベース: トレードしないバーは0リターンとして計算）
    total_bars = all_bars_count if all_bars_count else len(results_df)
    # 全バーのリターン配列を作成（トレードしないバーは0）
    all_returns = np.zeros(total_bars)
    # トレードしたバーにPnLを割り当て（均等配分）
    trade_indices = np.linspace(0, total_bars - 1, trade_count, dtype=int)
    all_returns[trade_indices] = trades["pnl"].values
    sharpe = (all_returns.mean() / all_returns.std() * np.sqrt(252)) if all_returns.std() > 0 else 0

    # Maximum Drawdown
    cumulative = np.cumsum(trades["pnl"].values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    mdd = drawdown.max() if len(drawdown) > 0 else 0

    return {
        "pf": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "win_rate": round(win_rate, 4),
        "trade_count": trade_count,
        "gross_profit": round(gross_profit, 1),
        "gross_loss": round(gross_loss, 1),
    }


# ===== 定数 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "japan_stocks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 米国指数ティッカー
US_INDICES = {
    "^GSPC": "SP500",       # S&P 500
    "^IXIC": "NASDAQ",      # ナスダック
    "^DJI": "DOW",          # ダウ平均
    "^VIX": "VIX",          # 恐怖指数
}

# 米国セクターETF
US_SECTOR_ETFS = {
    "QQQ": "US_Tech",       # ナスダック100 ETF（テック系）
    "XLF": "US_Finance",    # 金融セクターETF
    "XLE": "US_Energy",     # エネルギーセクターETF
    "XLV": "US_Healthcare", # ヘルスケアセクターETF
}

# 日本株ティッカー
JP_STOCKS = {
    "^N225": "Nikkei225",           # 日経225
    "1306.T": "TOPIX_ETF",         # TOPIX連動ETF
    "7203.T": "Toyota",            # トヨタ自動車
    "9984.T": "SoftBank_Group",    # ソフトバンクグループ
    "6758.T": "Sony",              # ソニー
    "8306.T": "MUFG",              # 三菱UFJ
    "6723.T": "Renesas",           # ルネサスエレクトロニクス
    "8035.T": "Tokyo_Electron",    # 東京エレクトロン
    "6857.T": "Advantest",         # アドバンテスト
    "7974.T": "Nintendo",          # 任天堂
    "7267.T": "Honda",             # ホンダ
    "8316.T": "SMFG",              # 三井住友FG
    "9433.T": "KDDI",              # KDDI
    "4063.T": "Shin_Etsu",        # 信越化学工業
    "6501.T": "Hitachi",           # 日立製作所
    "6902.T": "Denso",             # デンソー
}

# スクリーニング対象（個別株のみ、指数・ETFは除く）
SCREENING_TARGETS = [
    "7203.T", "9984.T", "6758.T", "8306.T",
    "6723.T", "8035.T", "6857.T", "7974.T", "7267.T",
    "8316.T", "9433.T", "4063.T", "6501.T", "6902.T",
]

# Walk-Forward設定
TRAIN_DAYS = 252    # 約1年（営業日）
TEST_DAYS = 63      # 約3ヶ月（営業日）

# 予測閾値（FX v3と同じ基準）
CONFIDENCE_THRESHOLD = 0.60
AGREEMENT_THRESHOLD = 4

# ニュースキーワード → 関連銘柄マッピング（将来のニュース統合用）
NEWS_KEYWORD_MAP = {
    "大谷": ["7832.T"],                        # バンダイナムコ（大谷関連グッズ）
    "トヨタ": ["7203.T"],                       # トヨタ自動車
    "ソフトバンク": ["9984.T", "9434.T"],        # ソフトバンクG、ソフトバンク
    "ソニー": ["6758.T"],                       # ソニー
    "任天堂": ["7974.T"],                       # 任天堂
    "半導体": ["6857.T", "8035.T", "6723.T"],   # アドバンテスト、東京エレクトロン、ルネサス
    "AI": ["9984.T", "6758.T", "4689.T"],       # ソフトバンクG、ソニー、Zホールディングス
    "円安": ["7203.T", "6758.T", "7267.T"],     # 輸出関連（トヨタ、ソニー、ホンダ）
    "円高": ["9983.T", "3382.T"],               # 内需関連（ファーストリテイリング、セブン＆アイ）
    "利上げ": ["8306.T", "8316.T", "8411.T"],   # 銀行（MUFG、三井住友、みずほ）
    "原油": ["5020.T", "1605.T"],               # ENEOS、INPEX
    "EV": ["7203.T", "7267.T", "6752.T"],       # トヨタ、ホンダ、パナソニック
}

# セクター分類（v3.4: セクター特化特徴量用）
STOCK_SECTORS = {
    "8035.T": "semiconductor",  # 東京エレクトロン
    "6857.T": "semiconductor",  # アドバンテスト
    "6723.T": "semiconductor",  # ルネサス
    "4063.T": "semiconductor",  # 信越化学（半導体材料）
    "7203.T": "auto",           # トヨタ
    "7267.T": "auto",           # ホンダ
    "6902.T": "auto",           # デンソー
    "6758.T": "tech",           # ソニー
    "9984.T": "tech",           # ソフトバンクG
    "7974.T": "tech",           # 任天堂
    "6501.T": "tech",           # 日立
    "9433.T": "tech",           # KDDI
    "8306.T": "finance",        # MUFG
    "8316.T": "finance",        # 三井住友FG
}

SECTOR_US_PROXY = {
    "semiconductor": "QQQ",     # ナスダック100 ETF
    "auto": None,
    "tech": "QQQ",
    "finance": "XLF",
}


# =============================================================
# 1. データ取得
# =============================================================
def download_data(period="2y"):
    """米国指数・日本株のデータをダウンロードして保存する

    Args:
        period: 取得期間（デフォルト: 2年）

    Returns:
        dict: ティッカー → DataFrameの辞書
    """
    print("=" * 50)
    print("[DL] データダウンロード開始")
    print("=" * 50)

    all_data = {}

    # 全ティッカーをまとめてダウンロード
    all_tickers = {**US_INDICES, **US_SECTOR_ETFS, **JP_STOCKS}

    for ticker, name in all_tickers.items():
        try:
            print(f"  ダウンロード中: {ticker} ({name})...", end=" ")
            df = yf.download(ticker, period=period, interval="1d", progress=False)

            if df.empty:
                print("[NG] データなし（スキップ）")
                continue

            # マルチカラムの場合はフラットにする
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 保存
            save_path = DATA_DIR / f"{name}.csv"
            df.to_csv(save_path)
            all_data[ticker] = df
            print(f"[OK] {len(df)}行")

        except Exception as e:
            print(f"[NG] エラー: {e}")

    # USD/JPYデータも読み込み（1時間足から日足に変換）
    usdjpy_path = PROJECT_ROOT / "data" / "usdjpy_1h.csv"
    if usdjpy_path.exists():
        try:
            usdjpy = pd.read_csv(usdjpy_path, header=[0, 1], index_col=0)
            usdjpy.columns = usdjpy.columns.get_level_values(0)
            usdjpy.index = pd.to_datetime(usdjpy.index, utc=True)
            # 日足に変換（最後の終値を使う）
            usdjpy_daily = usdjpy["Close"].resample("1D").last().dropna()
            usdjpy_daily.index = usdjpy_daily.index.tz_localize(None)
            usdjpy_daily = usdjpy_daily.to_frame(name="USDJPY_Close")
            usdjpy_daily.to_csv(DATA_DIR / "USDJPY_daily.csv")
            all_data["USDJPY"] = usdjpy_daily
            print(f"  USD/JPY日足: [OK] {len(usdjpy_daily)}行")
        except Exception as e:
            print(f"  USD/JPY読み込みエラー: {e}")

    print(f"\n合計: {len(all_data)}銘柄のデータ取得完了")
    return all_data


# =============================================================
# 2. 特徴量生成
# =============================================================
def make_features(all_data, target_ticker="^N225"):
    """予測用の特徴量を生成する

    米国市場のオーバーナイトリターンをメイン特徴量とし、
    VIX、為替、モメンタム等を組み合わせる。

    Args:
        all_data: download_data()の戻り値（ティッカー→DataFrame辞書）
        target_ticker: 予測対象のティッカー

    Returns:
        X: 特徴量DataFrame
        y: ターゲット（翌日上昇=1, 下落=0）
        feature_names: 特徴量名リスト
    """
    print(f"\n[DATA] 特徴量生成: {target_ticker}")

    # 対象銘柄のデータ確認
    if target_ticker not in all_data:
        print(f"  [NG] {target_ticker}のデータがありません")
        return None, None, None

    target_df = all_data[target_ticker].copy()
    target_df.index = pd.to_datetime(target_df.index)

    # ===== 特徴量格納用 =====
    features = pd.DataFrame(index=target_df.index)

    # ----- 米国指数のオーバーナイトリターン -----
    # 米国の終値変化率が、翌日の日本市場に影響する
    # v3.3 pruned: SP500_Return/SP500_Return_5d removed (correlated with NASDAQ, keep NASDAQ)
    for ticker, name in US_INDICES.items():
        if ticker in all_data and ticker != "^VIX" and ticker != "^GSPC":
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_returns = us_df["Close"].pct_change()
            us_returns.name = f"{name}_Return"
            # 日付で結合（米国と日本の営業日のズレを吸収）
            features = features.join(us_returns, how="left")

            # 5日リターンも追加
            us_ret_5d = us_df["Close"].pct_change(5)
            us_ret_5d.name = f"{name}_Return_5d"
            features = features.join(us_ret_5d, how="left")

    # ----- 米国セクターETFリターン -----
    for ticker, name in US_SECTOR_ETFS.items():
        if ticker in all_data:
            sec_df = all_data[ticker].copy()
            sec_df.index = pd.to_datetime(sec_df.index)
            sec_returns = sec_df["Close"].pct_change()
            sec_returns.name = f"{name}_Return"
            features = features.join(sec_returns, how="left")

    # ----- VIX（恐怖指数） -----
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)

        # VIXの水準
        vix_level = vix_df["Close"].copy()
        vix_level.name = "VIX_Level"
        features = features.join(vix_level, how="left")

        # VIXの変化
        vix_change = vix_df["Close"].pct_change()
        vix_change.name = "VIX_Change"
        features = features.join(vix_change, how="left")

        # v3.3 pruned: VIX_High removed (near-zero importance)

    # ----- USD/JPY為替変化 -----
    if "USDJPY" in all_data:
        usdjpy_df = all_data["USDJPY"].copy()
        usdjpy_df.index = pd.to_datetime(usdjpy_df.index)

        fx_change = usdjpy_df["USDJPY_Close"].pct_change()
        fx_change.name = "USDJPY_Change"
        features = features.join(fx_change, how="left")

        fx_change_5d = usdjpy_df["USDJPY_Close"].pct_change(5)
        fx_change_5d.name = "USDJPY_Change_5d"
        features = features.join(fx_change_5d, how="left")

    # ----- 対象銘柄のモメンタム -----
    # 自分自身の過去リターン（1日、5日、20日）
    for period in [1, 5, 20]:
        ret = target_df["Close"].pct_change(period)
        ret.name = f"Target_Return_{period}d"
        features = features.join(ret, how="left")

    # ----- 出来高変化 -----
    if "Volume" in target_df.columns:
        vol_change = target_df["Volume"].pct_change()
        vol_change.name = "Volume_Change"
        features = features.join(vol_change, how="left")

        # 出来高の20日平均との比率
        vol_ratio = target_df["Volume"] / target_df["Volume"].rolling(20).mean()
        vol_ratio.name = "Volume_Ratio"
        features = features.join(vol_ratio, how="left")

    # ----- 米国指数との相関（ローリング20日） -----
    target_returns = target_df["Close"].pct_change()
    for ticker, name in US_INDICES.items():
        if ticker in all_data and ticker != "^VIX":
            us_df = all_data[ticker].copy()
            us_df.index = pd.to_datetime(us_df.index)
            us_ret = us_df["Close"].pct_change()

            # 日付で整列してから相関を計算
            combined = pd.DataFrame({
                "target": target_returns,
                "us": us_ret,
            }).dropna()
            if len(combined) > 20:
                rolling_corr = combined["target"].rolling(20).corr(combined["us"])
                rolling_corr.name = f"Corr_{name}_20d"
                features = features.join(rolling_corr, how="left")

    # ----- ボラティリティレジーム特徴量 (v3.0) -----
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_close = vix_df["Close"].reindex(features.index, method="ffill")

        # VIXのパーセンタイル（過去120日での順位）
        vix_pct = vix_close.rolling(120, min_periods=20).rank(pct=True)
        vix_pct.name = "VIX_Percentile"
        features["VIX_Percentile"] = vix_pct

        # VIXのボラティリティ（VIXの変動性 = 恐怖の恐怖）
        vix_of_vix = vix_close.rolling(20, min_periods=5).std()
        vix_of_vix.name = "VIX_of_VIX"
        features["VIX_of_VIX"] = vix_of_vix

        # VIX変化率5日（短期トレンド）
        vix_ret_5d = vix_close.pct_change(5)
        features["VIX_Return_5d"] = vix_ret_5d

    # ----- 週次モメンタム特徴量 (v3.0) -----
    target_close = target_df["Close"]
    # 10日RSI proxy（上昇日数の割合）
    daily_up = (target_close.pct_change() > 0).astype(float)
    features["RSI_Proxy_10d"] = daily_up.rolling(10, min_periods=5).mean()

    # v3.3 pruned: Cumulative_Return_5d removed (near-zero importance)
    # 累積リターン（2w only）
    features["Cumulative_Return_10d"] = target_close.pct_change(10)

    # ボラティリティ（20日）
    features["Target_Volatility_20d"] = target_close.pct_change().rolling(20, min_periods=5).std()

    # ----- カレンダー特徴量 -----
    features["DayOfWeek"] = features.index.dayofweek    # 0=月曜, 4=金曜
    features["Month"] = features.index.month
    # v3.3 pruned: IsMonday, IsFriday removed (near-zero importance)

    # ----- インタラクション特徴量 -----

    # 1) US-JP相関モメンタム: 相関の変化率（相関が強まっている/弱まっている）
    for ticker, name in US_INDICES.items():
        corr_col = f"Corr_{name}_20d"
        if corr_col in features.columns:
            # 相関の5日変化（相関モメンタム）
            corr_mom = features[corr_col].diff(5)
            features[f"CorrMom_{name}_5d"] = corr_mom

    # 2) VIXレジーム × 米国リターンのインタラクション
    if "VIX_Level" in features.columns:
        # VIXレジーム: Low(<15), Mid(15-25), High(>25)
        features["VIX_Regime"] = np.where(
            features["VIX_Level"] < 15, 0,
            np.where(features["VIX_Level"] < 25, 1, 2)
        )
        # v3.3 pruned: VIX_x_SP500 removed (SP500 features pruned)
        # VIXレジーム × NASDAQリターンのインタラクション
        if "NASDAQ_Return" in features.columns:
            features["VIX_x_NASDAQ"] = features["VIX_Regime"] * features["NASDAQ_Return"]

    # 3) セクターローテーションシグナル: テック vs 金融の相対強度
    if "US_Tech_Return" in features.columns and "US_Finance_Return" in features.columns:
        # テック-金融のスプレッド（正=テック優勢、負=金融優勢）
        features["TechFinance_Spread"] = features["US_Tech_Return"] - features["US_Finance_Return"]
        # 5日累積スプレッド
        features["TechFinance_Spread_5d"] = features["TechFinance_Spread"].rolling(5).sum()

    # 4) エネルギー vs テック（リスクオン/オフ指標） (v3.0)
    if "US_Energy_Return" in features.columns and "US_Tech_Return" in features.columns:
        features["EnergyTech_Spread"] = features["US_Energy_Return"] - features["US_Tech_Return"]

    # 5) VIXパーセンタイル × モメンタムのインタラクション (v3.0)
    if "VIX_Percentile" in features.columns and "Target_Return_5d" in features.columns:
        features["VIXPct_x_Mom5d"] = features["VIX_Percentile"] * features["Target_Return_5d"]

    # ----- ターゲット: 翌日上昇=1, 下落=0 -----
    future_return = target_df["Close"].pct_change().shift(-1)
    y = (future_return > 0).astype(int)
    y.name = "Target"

    # ----- 欠損値処理 -----
    # 前方補完してからNaN行を削除
    features = features.ffill()

    # yとfeaturesの共通インデックスのみ使う
    common_idx = features.index.intersection(y.dropna().index)
    features = features.loc[common_idx]
    y = y.loc[common_idx]

    # まだNaNが残る行は削除
    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]
    y = y.loc[valid_mask]

    feature_names = features.columns.tolist()
    print(f"  特徴量数: {len(feature_names)}")
    print(f"  データ行数: {len(features)}")
    print(f"  ターゲット分布: UP={y.sum()}, DOWN={len(y) - y.sum()}")

    return features, y, feature_names


# =============================================================
# 3. 学習・予測（Walk-Forward検証）
# =============================================================
def _compute_model_weights(model, X_val, y_val):
    """各サブモデルの精度に基づく重みを計算する (v3.0)

    accuracy^3 で重み付け → 高精度モデルの影響力を増大
    """
    weights = []
    for sub_model in model.models:
        preds = sub_model.predict(X_val)
        acc = (preds == y_val).mean()
        weights.append(acc)
    weights = np.array(weights)
    weights = weights ** 3  # 精度の3乗で差を強調
    weights = weights / weights.sum()
    return weights


def _weighted_predict(model, X, weights):
    """重み付きアンサンブル予測 (v3.0)"""
    probas = np.array([m.predict_proba(X)[:, 1] for m in model.models])
    weighted_proba = (probas * weights[:, None]).sum(axis=0)
    preds = (weighted_proba >= 0.5).astype(int)

    # 一致度: 各モデルの予測方向カウント
    individual_preds = np.array([m.predict(X) for m in model.models])
    vote_sum = individual_preds.sum(axis=0)
    agreement = np.where(preds == 1, vote_sum, 5 - vote_sum)

    # confidence: 重み付き確率の予測方向側
    confidence = np.where(preds == 1, weighted_proba, 1.0 - weighted_proba)

    return preds, agreement.astype(int), confidence


def train_and_predict(X, y, feature_names):
    """Expanding Window Walk-Forward方式で学習・予測する (v3.0)

    v3.0: 性能ベースアンサンブル重み付け + VIXレジーム適応閾値

    Args:
        X: 特徴量DataFrame
        y: ターゲットSeries
        feature_names: 特徴量名リスト

    Returns:
        results: Walk-Forwardの各期間の結果リスト
        latest_prediction: 最新の予測結果（直近の予測に使用）
    """
    print("\n[AI] Expanding Window Walk-Forward学習・予測 (v3.0)")

    results = []
    latest_prediction = None

    total_len = len(X)
    if total_len < TRAIN_DAYS + TEST_DAYS:
        print(f"  [NG] データ不足: {total_len}行 < {TRAIN_DAYS + TEST_DAYS}行（必要最低限）")
        return results, latest_prediction

    # Expanding Window: 訓練開始は常に0、訓練終了をスライド
    fold = 0
    train_end = TRAIN_DAYS
    while train_end + TEST_DAYS <= total_len:
        fold += 1
        test_end = min(train_end + TEST_DAYS, total_len)

        # 訓練データは常に最初から（expanding）
        # v3.0: 訓練の末尾10%をバリデーションに使い、重みを計算
        val_size = max(20, int(train_end * 0.1))
        X_train = X.iloc[0:train_end - val_size]
        y_train = y.iloc[0:train_end - val_size]
        X_val = X.iloc[train_end - val_size:train_end]
        y_val = y.iloc[train_end - val_size:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        # アンサンブルモデル学習
        model = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train.values, y_train.values)

        # v3.0: 性能ベース重み計算
        weights = _compute_model_weights(model, X_val.values, y_val.values)

        # v3.0: 重み付き予測
        predictions, agreements, confidences = _weighted_predict(
            model, X_test.values, weights
        )

        # 各予測の詳細を記録
        fold_results = []
        for i in range(len(X_test)):
            pred = predictions[i]
            agree = agreements[i]
            conf = confidences[i]
            actual = y_test.iloc[i]
            date = X_test.index[i]

            fold_results.append({
                "date": date,
                "prediction": pred,        # 1=上昇予測, 0=下落予測
                "agreement": agree,         # 一致人数（3～5）
                "confidence": conf,         # 自信度
                "actual": actual,           # 実際の結果
                "correct": int(pred == actual),
            })

        fold_df = pd.DataFrame(fold_results)

        # 高信頼度（閾値超え）のみの成績
        high_conf = fold_df[
            (fold_df["confidence"] >= CONFIDENCE_THRESHOLD) &
            (fold_df["agreement"] >= AGREEMENT_THRESHOLD)
        ]

        accuracy_all = fold_df["correct"].mean() if len(fold_df) > 0 else 0
        accuracy_high = high_conf["correct"].mean() if len(high_conf) > 0 else 0

        print(f"  Fold {fold}: "
              f"全体正解率={accuracy_all:.1%} ({len(fold_df)}件), "
              f"高信頼度={accuracy_high:.1%} ({len(high_conf)}件) "
              f"重み=[{', '.join(f'{w:.2f}' for w in weights)}]")

        results.extend(fold_results)

        # 次のウィンドウへ（expanding: train_endをスライド）
        train_end += TEST_DAYS

    # 最新データでの予測（全データで学習 = expanding方式）
    if total_len >= TRAIN_DAYS + 1:
        latest_train_end = total_len - 1

        # v3.0: バリデーション分割
        val_size = max(20, int(latest_train_end * 0.1))
        X_lt = X.iloc[0:latest_train_end - val_size]
        y_lt = y.iloc[0:latest_train_end - val_size]
        X_lv = X.iloc[latest_train_end - val_size:latest_train_end]
        y_lv = y.iloc[latest_train_end - val_size:latest_train_end]
        X_latest = X.iloc[-1:]

        model = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_lt.values, y_lt.values)

        weights = _compute_model_weights(model, X_lv.values, y_lv.values)
        pred, agree, conf = _weighted_predict(model, X_latest.values, weights)

        latest_prediction = {
            "date": X.index[-1],
            "prediction": int(pred[0]),
            "agreement": int(agree[0]),
            "confidence": float(conf[0]),
            "direction": "UP" if pred[0] == 1 else "DOWN",
        }

        print(f"\n  [PIN] 最新予測: {latest_prediction['direction']} "
              f"(自信度={conf[0]:.1%}, 一致={agree[0]}/5)")

    return results, latest_prediction


# =============================================================
# 4. スクリーニング（銘柄選定）
# =============================================================
def screen_stocks(all_data, top_n=3):
    """全対象銘柄に対して予測を実行し、上位を返す

    Args:
        all_data: ダウンロードデータ
        top_n: 上位何銘柄を返すか

    Returns:
        picks: 推奨銘柄リスト（信頼度順）
    """
    print("\n" + "=" * 50)
    print("[SEARCH] 銘柄スクリーニング")
    print("=" * 50)

    candidates = []

    for ticker in SCREENING_TARGETS:
        name = JP_STOCKS.get(ticker, ticker)
        print(f"\n--- {ticker} ({name}) ---")

        try:
            # 特徴量生成
            X, y, feature_names = make_features(all_data, target_ticker=ticker)
            if X is None or len(X) < TRAIN_DAYS + TEST_DAYS:
                print(f"  [WARN] データ不足でスキップ")
                continue

            # Walk-Forward予測
            results, latest_pred = train_and_predict(X, y, feature_names)

            if latest_pred is None:
                print(f"  [WARN] 最新予測なし")
                continue

            # Walk-Forwardの全体成績を計算
            results_df = pd.DataFrame(results)
            high_conf = results_df[
                (results_df["confidence"] >= CONFIDENCE_THRESHOLD) &
                (results_df["agreement"] >= AGREEMENT_THRESHOLD)
            ]
            wf_accuracy = high_conf["correct"].mean() if len(high_conf) > 0 else 0
            wf_count = len(high_conf)

            # PF/Sharpe/MDD メトリクス計算（全バーベース）
            metrics = compute_trading_metrics(results_df, all_bars_count=len(results_df))
            print(f"  [DATA] PF={metrics['pf']:.2f} Sharpe={metrics['sharpe']:.2f} "
                  f"MDD={metrics['mdd']:.0f} WinRate={metrics['win_rate']:.1%} "
                  f"Trades={metrics['trade_count']}")

            # 理由を生成
            reasons = _generate_reasons(latest_pred, wf_accuracy, wf_count, all_data, ticker)

            candidates.append({
                "ticker": ticker,
                "name": name,
                "direction": latest_pred["direction"],
                "confidence": latest_pred["confidence"],
                "agreement": latest_pred["agreement"],
                "wf_accuracy": wf_accuracy,
                "wf_high_conf_count": wf_count,
                "pf": metrics["pf"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "win_rate": metrics["win_rate"],
                "trade_count": metrics["trade_count"],
                "reasons": reasons,
                "date": latest_pred["date"],
            })

        except Exception as e:
            print(f"  [NG] エラー: {e}")
            import traceback
            traceback.print_exc()

    # 信頼度 × 一致度でソート
    candidates.sort(key=lambda x: (x["agreement"], x["confidence"]), reverse=True)

    # 高信頼度の銘柄のみ返す
    picks = [c for c in candidates
             if c["confidence"] >= CONFIDENCE_THRESHOLD
             and c["agreement"] >= AGREEMENT_THRESHOLD][:top_n]

    print(f"\n[LIST] 推奨銘柄: {len(picks)}件 / {len(candidates)}件中")
    for p in picks:
        print(f"  {p['ticker']} ({p['name']}): "
              f"{p['direction']} 自信度={p['confidence']:.1%} "
              f"一致={p['agreement']}/5 WF正解率={p['wf_accuracy']:.1%} "
              f"PF={p.get('pf', 0):.2f} Sharpe={p.get('sharpe', 0):.2f} "
              f"MDD={p.get('mdd', 0):.0f}")

    return picks


def _generate_reasons(latest_pred, wf_accuracy, wf_count, all_data, ticker):
    """予測の理由リストを生成する"""
    reasons = []

    # 一致度
    agree = latest_pred["agreement"]
    if agree == 5:
        reasons.append("5モデル全員一致（最高信頼度）")
    elif agree == 4:
        reasons.append("4/5モデル一致（高信頼度）")

    # Walk-Forward成績
    if wf_count > 0:
        reasons.append(f"Walk-Forward検証: 正解率{wf_accuracy:.1%}（高信頼{wf_count}件）")

    # VIXの状態
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"]
        vix_last = vix_df["Close"].iloc[-1]
        if vix_last > 30:
            reasons.append(f"[WARN] VIX={vix_last:.1f}（恐怖指数高い → リスク大）")
        elif vix_last < 15:
            reasons.append(f"VIX={vix_last:.1f}（市場は安定）")

    # 米国市場の前日の動き
    if "^GSPC" in all_data:
        sp500_ret = all_data["^GSPC"]["Close"].pct_change().iloc[-1]
        direction_jp = "上昇" if sp500_ret > 0 else "下落"
        reasons.append(f"前日S&P500: {sp500_ret:+.2%}（{direction_jp}）")

    # 対象銘柄のモメンタム
    if ticker in all_data:
        target_ret_5d = all_data[ticker]["Close"].pct_change(5).iloc[-1]
        if not np.isnan(target_ret_5d):
            momentum = "上昇トレンド" if target_ret_5d > 0 else "下落トレンド"
            reasons.append(f"5日モメンタム: {target_ret_5d:+.2%}（{momentum}）")

    return reasons


# =============================================================
# 5. ニュースキーワードチェック（将来のニュース統合用）
# =============================================================
def check_news_keywords(headlines):
    """ニュースのヘッドラインからキーワードを検出し、関連銘柄を返す

    将来、Twitter/ニュースAPIと連携する際のプレースホルダー。

    Args:
        headlines: ニュースヘッドラインのリスト（日本語文字列）

    Returns:
        matches: [{"keyword": str, "stocks": list, "headline": str}, ...]
    """
    matches = []

    for headline in headlines:
        for keyword, stocks in NEWS_KEYWORD_MAP.items():
            if keyword in headline:
                matches.append({
                    "keyword": keyword,
                    "stocks": stocks,
                    "headline": headline,
                })

    if matches:
        print(f"\n[NEWS] ニュースキーワード検出: {len(matches)}件")
        for m in matches:
            print(f"  「{m['keyword']}」→ {m['stocks']} ({m['headline'][:30]}...)")

    return matches


# =============================================================
# 6. Telegram通知
# =============================================================
def send_stock_signal(stock_code, direction, confidence, agreement, reasons):
    """日本株のシグナルをTelegramに送信する

    telegram_bot.pyのsend_signal_sync()を利用する。

    Args:
        stock_code: ティッカー（例: "7203.T"）
        direction: "BUY" or "SELL"
        confidence: 自信度（0.0～1.0）
        agreement: 一致人数（3～5）
        reasons: 理由リスト
    """
    try:
        from research.telegram_bot import send_signal_sync

        name = JP_STOCKS.get(stock_code, stock_code)

        signal = {
            "pair": f"{stock_code} ({name})",
            "action": direction,
            "price": 0.0,       # 株価は別途取得が必要
            "confidence": confidence,
            "agreement": agreement,
            "reasons": reasons,
            "category": "株",
        }

        # 直近の株価を取得してセット
        try:
            ticker_data = yf.download(stock_code, period="1d", progress=False)
            if not ticker_data.empty:
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = ticker_data.columns.get_level_values(0)
                signal["price"] = float(ticker_data["Close"].iloc[-1])
        except Exception:
            pass  # 株価取得失敗は無視（通知は送る）

        send_signal_sync(signal)
        print(f"  📤 Telegram送信完了: {stock_code} ({name}) → {direction}")

    except ImportError:
        print("  [WARN] telegram_bot.pyが読み込めません（Telegram通知スキップ）")
    except Exception as e:
        print(f"  [WARN] Telegram送信エラー: {e}")


# =============================================================
# 7. メインパイプライン
# =============================================================
def run_pipeline():
    """全パイプラインを実行する

    1. データダウンロード
    2. 銘柄スクリーニング
    3. 上位銘柄をTelegram通知
    4. 結果をCSV保存
    """
    print("=" * 60)
    print(f"[JP] 日本株予測パイプライン 実行開始: {datetime.now()}")
    print("=" * 60)

    # ----- 1. データ取得 -----
    all_data = download_data(period="2y")

    if len(all_data) < 3:
        print("[NG] 十分なデータが取得できませんでした。ネットワーク接続を確認してください。")
        return

    # ----- 2. スクリーニング -----
    picks = screen_stocks(all_data, top_n=3)

    # ----- 3. Telegram通知 -----
    if picks:
        print("\n📤 Telegram通知送信中...")
        for pick in picks:
            direction = "BUY" if pick["direction"] == "UP" else "SELL"
            send_stock_signal(
                stock_code=pick["ticker"],
                direction=direction,
                confidence=pick["confidence"],
                agreement=pick["agreement"],
                reasons=pick["reasons"],
            )
    else:
        print("\n[WARN] 高信頼度のシグナルなし（閾値: 自信度≥60%, 一致≥4/5）")
        print("  → 無理にトレードしないのが正解です！")

    # ----- 4. 結果保存 -----
    save_path = DATA_DIR / "daily_picks.csv"
    if picks:
        picks_df = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": p["ticker"],
            "name": p["name"],
            "direction": p["direction"],
            "confidence": p["confidence"],
            "agreement": p["agreement"],
            "wf_accuracy": p["wf_accuracy"],
            "pf": p.get("pf", 0),
            "sharpe": p.get("sharpe", 0),
            "mdd": p.get("mdd", 0),
            "win_rate": p.get("win_rate", 0),
            "trade_count": p.get("trade_count", 0),
            "reasons": " / ".join(p["reasons"]),
        } for p in picks])

        # 既存ファイルがあれば追記
        if save_path.exists():
            existing = pd.read_csv(save_path)
            picks_df = pd.concat([existing, picks_df], ignore_index=True)

        picks_df.to_csv(save_path, index=False)
        print(f"\n💾 結果保存: {save_path}")
    else:
        # シグナルなしの記録も残す
        no_signal = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": "N/A",
            "name": "シグナルなし",
            "direction": "HOLD",
            "confidence": 0.0,
            "agreement": 0,
            "wf_accuracy": 0.0,
            "reasons": "高信頼度シグナルなし",
        }])
        if save_path.exists():
            existing = pd.read_csv(save_path)
            no_signal = pd.concat([existing, no_signal], ignore_index=True)
        no_signal.to_csv(save_path, index=False)

    # ----- 日経225全体の予測も実施 -----
    print("\n" + "=" * 50)
    print("📈 日経225全体の予測")
    print("=" * 50)

    X_nikkei, y_nikkei, feat_nikkei = make_features(all_data, target_ticker="^N225")
    if X_nikkei is not None and len(X_nikkei) >= TRAIN_DAYS + TEST_DAYS:
        results_nikkei, latest_nikkei = train_and_predict(X_nikkei, y_nikkei, feat_nikkei)

        if latest_nikkei:
            print(f"\n  日経225 明日の予測: {latest_nikkei['direction']} "
                  f"（自信度={latest_nikkei['confidence']:.1%}, "
                  f"一致={latest_nikkei['agreement']}/5）")

            # Walk-Forwardの総合成績
            if results_nikkei:
                res_df = pd.DataFrame(results_nikkei)
                total_acc = res_df["correct"].mean()
                high_conf = res_df[
                    (res_df["confidence"] >= CONFIDENCE_THRESHOLD) &
                    (res_df["agreement"] >= AGREEMENT_THRESHOLD)
                ]
                high_acc = high_conf["correct"].mean() if len(high_conf) > 0 else 0
                nikkei_metrics = compute_trading_metrics(res_df, all_bars_count=len(res_df))
                print(f"  Walk-Forward総合: 全体正解率={total_acc:.1%}, "
                      f"高信頼度正解率={high_acc:.1%}（{len(high_conf)}件）")
                print(f"  [DATA] PF={nikkei_metrics['pf']:.2f} "
                      f"Sharpe={nikkei_metrics['sharpe']:.2f} "
                      f"MDD={nikkei_metrics['mdd']:.0f} "
                      f"Trades={nikkei_metrics['trade_count']}")

    print("\n" + "=" * 60)
    print(f"[OK] パイプライン完了: {datetime.now()}")
    print("=" * 60)


# =============================================================
# 実行
# =============================================================
if __name__ == "__main__":
    run_pipeline()
