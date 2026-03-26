# ===========================================
# multi_currency_pipeline.py
# 複数通貨ペアの自動学習・最適化・検証パイプライン
#
# USD/JPYと同じ手法を全通貨ペアに適用:
#   1. 価格データ + ファンダメンタルデータ収集
#   2. 特徴量生成（テクニカル + ファンダメンタル）
#   3. 5モデルアンサンブル + Walk-Forward検証
#   4. 閾値最適化（自信度 × 一致人数）
#   5. 利益が出るペアのみペーパートレード対象
# ===========================================

import sys
import warnings
import zipfile
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.common.validation import walk_forward_splits, compute_metrics
from research.common.ensemble import EnsembleClassifier

DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()

# ===== 通貨ペア定義 =====
PAIRS = {
    "usdjpy": {
        "ticker": "USDJPY=X",
        "spread": 0.0003,
        "cot_name": "JAPANESE YEN",
        "rate_tickers": ["^TNX"],  # US 10Y
        "commodity": None,
        "description": "ドル円（現行モデル）",
    },
    "cadjpy": {
        "ticker": "CADJPY=X",
        "spread": 0.0004,
        "cot_name": "CANADIAN DOLLAR",
        "rate_tickers": ["^TNX"],
        "commodity": "CL=F",  # WTI Crude Oil
        "description": "カナダドル円（原油連動）",
    },
    "eurjpy": {
        "ticker": "EURJPY=X",
        "spread": 0.0003,
        "cot_name": "EURO FX",
        "rate_tickers": ["^TNX"],
        "commodity": None,
        "description": "ユーロ円",
    },
    "gbpjpy": {
        "ticker": "GBPJPY=X",
        "spread": 0.0005,
        "cot_name": "BRITISH POUND STERLING",
        "rate_tickers": ["^TNX"],
        "commodity": None,
        "description": "ポンド円（高ボラ）",
    },
    "eurusd": {
        "ticker": "EURUSD=X",
        "spread": 0.0002,
        "cot_name": "EURO FX",
        "rate_tickers": ["^TNX"],
        "commodity": None,
        "description": "ユーロドル（世界最大）",
    },
    "gbpusd": {
        "ticker": "GBPUSD=X",
        "spread": 0.0003,
        "cot_name": "BRITISH POUND STERLING",
        "rate_tickers": ["^TNX"],
        "commodity": None,
        "description": "ポンドドル",
    },
    "audjpy": {
        "ticker": "AUDJPY=X",
        "spread": 0.0004,
        "cot_name": "AUSTRALIAN DOLLAR",
        "rate_tickers": ["^TNX"],
        "commodity": "GC=F",  # Gold
        "description": "豪ドル円（資源国）",
    },
    "nzdjpy": {
        "ticker": "NZDJPY=X",
        "spread": 0.0005,
        "cot_name": "NEW ZEALAND DOLLAR",
        "rate_tickers": ["^TNX"],
        "commodity": None,
        "description": "NZドル円",
    },
}


# ===== データ収集 =====
def download_price_data(pair_id, pair_info):
    """1時間足の価格データをダウンロード"""
    import yfinance as yf
    print(f"  {pair_id}: 価格データ取得中...")
    df = yf.download(pair_info["ticker"], period="2y", interval="1h", progress=False)
    if hasattr(df.columns, "droplevel"):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def download_commodity_data(ticker):
    """コモディティデータをダウンロード"""
    import yfinance as yf
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if hasattr(df.columns, "droplevel"):
        df.columns = df.columns.droplevel(1)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df[["Close"]].dropna().sort_index()


def download_rate_data():
    """米国10年債利回りをダウンロード"""
    import yfinance as yf
    df = yf.download("^TNX", period="2y", interval="1d", progress=False)
    if hasattr(df.columns, "droplevel"):
        df.columns = df.columns.droplevel(1)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df[["Close"]].rename(columns={"Close": "rate_us_10y"}).dropna().sort_index()


def fetch_cot_multi():
    """CFTC COTデータを複数通貨分まとめて取得"""
    print("  COTデータ取得中...")
    all_data = []
    for year in range(2024, datetime.now().year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for fn in zf.namelist():
                    if fn.endswith(".txt"):
                        df = pd.read_csv(zf.open(fn))
                        all_data.append(df)
        except Exception:
            continue

    if not all_data:
        return {}

    df_all = pd.concat(all_data, ignore_index=True)

    results = {}
    for pair_id, pair_info in PAIRS.items():
        cot_name = pair_info["cot_name"]
        mask = df_all["Market_and_Exchange_Names"].str.contains(cot_name, case=False, na=False)
        df_cot = df_all[mask].copy()
        if df_cot.empty:
            continue

        df_cot["Date"] = pd.to_datetime(df_cot["Report_Date_as_YYYY-MM-DD"])
        df_cot = df_cot.sort_values("Date").drop_duplicates("Date", keep="last")
        df_cot = df_cot.set_index("Date")

        # ポジションデータの抽出
        long_cols = [c for c in df_cot.columns if "Asset_Mgr" in c and "Long" in c and "All" in c]
        short_cols = [c for c in df_cot.columns if "Asset_Mgr" in c and "Short" in c and "All" in c]
        dealer_long = [c for c in df_cot.columns if "Dealer" in c and "Long" in c and "All" in c]
        dealer_short = [c for c in df_cot.columns if "Dealer" in c and "Short" in c and "All" in c]
        oi_col = [c for c in df_cot.columns if "Open_Interest" in c and "All" in c]

        if not long_cols or not short_cols:
            continue

        spec_long = pd.to_numeric(df_cot[long_cols[0]], errors="coerce").fillna(0)
        spec_short = pd.to_numeric(df_cot[short_cols[0]], errors="coerce").fillna(0)
        comm_long = pd.to_numeric(df_cot[dealer_long[0]], errors="coerce").fillna(0) if dealer_long else 0
        comm_short = pd.to_numeric(df_cot[dealer_short[0]], errors="coerce").fillna(0) if dealer_short else 0
        oi = pd.to_numeric(df_cot[oi_col[0]], errors="coerce").fillna(1) if oi_col else pd.Series(1, index=df_cot.index)

        features = pd.DataFrame(index=df_cot.index)
        spec_net = spec_long - spec_short
        comm_net = comm_long - comm_short
        features["cot_spec_net_norm"] = spec_net / oi.replace(0, np.nan)
        features["cot_spec_net_change"] = spec_net.diff()
        features["cot_comm_net_norm"] = comm_net / oi.replace(0, np.nan)
        features["cot_spec_zscore"] = (spec_net - spec_net.rolling(26).mean()) / spec_net.rolling(26).std().replace(0, np.nan)
        features["cot_oi_change"] = oi.pct_change()
        features = features.fillna(0)

        results[pair_id] = features

    return results


# ===== 特徴量生成 =====
def make_technical_features(df):
    """テクニカル指標を計算"""
    close = df["Close"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, np.nan)

    # MA
    df["MA_5"] = close.rolling(5).mean()
    df["MA_25"] = close.rolling(25).mean()
    df["MA_75"] = close.rolling(75).mean()

    # Returns & Volatility
    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Return_6"] = close.pct_change(6)
    df["Return_24"] = close.pct_change(24)
    df["Volatility_24"] = df["Return_1"].rolling(24).std()

    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek

    return df


TECH_COLS = [
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "MA_5", "MA_25", "MA_75",
    "Return_1", "Return_3", "Return_6", "Return_24",
    "Volatility_24", "Hour", "DayOfWeek",
]

COT_COLS = [
    "cot_spec_net_norm", "cot_spec_net_change", "cot_comm_net_norm",
    "cot_spec_zscore", "cot_oi_change",
]

RATE_COLS = ["rate_us_10y", "rate_trend"]
COMMODITY_COLS = ["commodity_ret24", "commodity_ma_ratio"]


def add_rate_features(df, df_rate):
    """金利特徴量を追加"""
    if df_rate is None or df_rate.empty:
        df["rate_us_10y"] = 0.0
        df["rate_trend"] = 0
        return df
    ma20 = df_rate["rate_us_10y"].rolling(20).mean()
    df_rate["rate_trend"] = (df_rate["rate_us_10y"] > ma20).astype(int)
    df_rate_reindex = df_rate.reindex(df.index, method="ffill")
    df["rate_us_10y"] = df_rate_reindex["rate_us_10y"].fillna(0)
    df["rate_trend"] = df_rate_reindex["rate_trend"].fillna(0).astype(int)
    return df


def add_commodity_features(df, df_commodity):
    """コモディティ特徴量を追加"""
    if df_commodity is None or df_commodity.empty:
        df["commodity_ret24"] = 0.0
        df["commodity_ma_ratio"] = 0.0
        return df
    c = df_commodity["Close"]
    df_commodity["commodity_ret24"] = c.pct_change(1)  # daily return
    ma20 = c.rolling(20).mean()
    df_commodity["commodity_ma_ratio"] = (c - ma20) / ma20.replace(0, np.nan)
    reindex = df_commodity[["commodity_ret24", "commodity_ma_ratio"]].reindex(df.index, method="ffill")
    df["commodity_ret24"] = reindex["commodity_ret24"].fillna(0)
    df["commodity_ma_ratio"] = reindex["commodity_ma_ratio"].fillna(0)
    return df


def add_cot_features(df, cot_features):
    """COT特徴量を追加"""
    if cot_features is None or cot_features.empty:
        for col in COT_COLS:
            df[col] = 0.0
        return df
    reindex = cot_features.reindex(df.index, method="ffill")
    for col in COT_COLS:
        df[col] = reindex[col].fillna(0) if col in reindex.columns else 0.0
    return df


# ===== 検証 =====
def run_walkforward(df, feat_cols, spread, conf_thresh=0.70, min_agree=5):
    """Walk-Forward検証を実行"""
    y = df["Label"].values
    ret = df["Return_4h"].values
    n = len(df)
    splits = walk_forward_splits(n, 4320, 720)

    if not splits:
        return None

    all_ret = []
    for tr, te in splits:
        model = EnsembleClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(df[feat_cols].iloc[tr], y[tr])
        p = model.predict_proba(df[feat_cols].iloc[te])[:, 1]
        _, ag = model.predict_with_agreement(df[feat_cols].iloc[te])

        for i in range(len(te)):
            c = max(p[i], 1 - p[i])
            if c >= conf_thresh and ag[i] >= min_agree:
                d = 1.0 if p[i] > 0.5 else -1.0
                all_ret.append(ret[te[i]] * d - spread)

    if not all_ret:
        return None
    return compute_metrics(np.array(all_ret))


def optimize_thresholds(df, feat_cols, spread):
    """閾値のグリッドサーチ"""
    best = None
    best_config = None

    for conf in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for agree in [3, 4, 5]:
            m = run_walkforward(df, feat_cols, spread, conf, agree)
            if m and m["n_trades"] >= 20:  # 最低20トレード
                if best is None or m["pf"] > best["pf"]:
                    best = m
                    best_config = {"conf": conf, "agree": agree}

    return best, best_config


# ===== メインパイプライン =====
def run_pipeline():
    """全通貨ペアのパイプラインを実行"""
    print("=" * 70)
    print("Multi-Currency Pipeline")
    print("=" * 70)

    # 1. 共通データ取得
    print("\n[1/4] データ収集")
    df_rate = download_rate_data()
    print(f"  金利データ: {len(df_rate)} rows")

    cot_all = fetch_cot_multi()
    print(f"  COTデータ: {len(cot_all)} pairs")

    # 2. 各通貨ペアを処理
    results = {}

    for pair_id, pair_info in PAIRS.items():
        if pair_id == "usdjpy":
            continue  # USD/JPYは既に最適化済み

        print(f"\n{'='*50}")
        print(f"[{pair_id.upper()}] {pair_info['description']}")
        print(f"{'='*50}")

        # 価格データ
        df = download_price_data(pair_id, pair_info)
        if len(df) < 5000:
            print(f"  データ不足: {len(df)} rows. スキップ")
            continue
        print(f"  価格: {len(df)} rows ({df.index.min()} ~ {df.index.max()})")

        # テクニカル指標
        df = make_technical_features(df)

        # ファンダメンタル
        df = add_rate_features(df, df_rate)
        df = add_cot_features(df, cot_all.get(pair_id))

        # コモディティ
        if pair_info["commodity"]:
            print(f"  コモディティ({pair_info['commodity']})取得中...")
            df_commodity = download_commodity_data(pair_info["commodity"])
            df = add_commodity_features(df, df_commodity)
            print(f"  コモディティ: {len(df_commodity)} rows")
        else:
            df = add_commodity_features(df, None)

        # ラベル
        df["Close_4h"] = df["Close"].shift(-4)
        df["Label"] = (df["Close_4h"] > df["Close"]).astype(int)
        df["Return_4h"] = (df["Close_4h"] - df["Close"]) / df["Close"]

        # 特徴量カラム
        feat_cols = TECH_COLS + RATE_COLS + COT_COLS + COMMODITY_COLS
        feat_cols = [c for c in feat_cols if c in df.columns]

        df = df.dropna(subset=feat_cols + ["Label", "Return_4h"])
        print(f"  学習データ: {len(df)} rows, {len(feat_cols)} features")

        if len(df) < 5000:
            print(f"  dropna後データ不足. スキップ")
            continue

        # 閾値最適化
        print(f"  Walk-Forward + 閾値最適化中...")
        best_metrics, best_config = optimize_thresholds(df, feat_cols, pair_info["spread"])

        if best_metrics is None:
            print(f"  トレードなし. スキップ")
            results[pair_id] = {"status": "NO_TRADES"}
            continue

        results[pair_id] = {
            "status": "OK",
            "description": pair_info["description"],
            "spread": pair_info["spread"],
            "data_rows": len(df),
            "features": len(feat_cols),
            "best_config": best_config,
            "pf": best_metrics["pf"],
            "win_rate": best_metrics["win_rate"],
            "exp_value": best_metrics["exp_value_net"],
            "n_trades": best_metrics["n_trades"],
            "sharpe": best_metrics["sharpe"],
            "profitable": best_metrics["exp_value_net"] > 0,
        }

        print(f"  Best: conf>={best_config['conf']:.2f}, agree>={best_config['agree']}")
        print(f"  PF={best_metrics['pf']:.2f}, Win={best_metrics['win_rate']:.1f}%, "
              f"Exp={best_metrics['exp_value_net']:+.6f}, N={best_metrics['n_trades']}")

        if best_metrics["exp_value_net"] > 0:
            print(f"  ★ 手数料後プラス！ペーパートレード候補")

    # 3. USD/JPYの結果を追加
    results["usdjpy"] = {
        "status": "OK",
        "description": "ドル円（現行モデル）",
        "spread": 0.0003,
        "best_config": {"conf": 0.70, "agree": 5},
        "pf": 1.47,
        "win_rate": 52.9,
        "exp_value": 0.000065,
        "n_trades": 244,
        "profitable": True,
    }

    return results


def generate_report(results):
    """結果レポートを生成"""
    report = []
    report.append("=" * 70)
    report.append("MULTI-CURRENCY ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)
    report.append("")
    report.append("Model: 5-Model Ensemble (LightGBM, XGBoost, CatBoost, RF, ExtraTrees)")
    report.append("Validation: Walk-Forward (4320 train / 720 test)")
    report.append("Features: Technical + US Rate + COT Positions + Commodity (where applicable)")
    report.append("")

    # Ranking
    ranked = [(k, v) for k, v in results.items() if v.get("status") == "OK"]
    ranked.sort(key=lambda x: -x[1].get("pf", 0))

    report.append("-" * 70)
    report.append(f"{'Pair':>8} {'PF':>6} {'Win%':>6} {'Exp(net)':>12} {'Trades':>7} {'Conf':>5} {'Agree':>5} {'Status':>10}")
    report.append("-" * 70)

    profitable_pairs = []
    for pair_id, r in ranked:
        cfg = r.get("best_config", {})
        status = "PROFIT" if r.get("profitable") else "LOSS"
        report.append(
            f"{pair_id:>8} {r['pf']:>6.2f} {r['win_rate']:>5.1f}% "
            f"{r['exp_value']:>+12.6f} {r['n_trades']:>7d} "
            f"{cfg.get('conf', 0):>5.2f} {cfg.get('agree', 0):>5d} "
            f"{'*** ' + status if r.get('profitable') else status:>10}"
        )
        if r.get("profitable"):
            profitable_pairs.append(pair_id)

    report.append("-" * 70)
    report.append("")

    # Summary
    report.append("=== SUMMARY ===")
    report.append(f"Total pairs tested: {len(ranked)}")
    report.append(f"Profitable pairs: {len(profitable_pairs)}")
    if profitable_pairs:
        report.append(f"Paper trade candidates: {', '.join(profitable_pairs)}")
    else:
        report.append("No profitable pairs found (besides USD/JPY)")
    report.append("")

    # Profit estimates for profitable pairs
    if profitable_pairs:
        report.append("=== PROFIT ESTIMATES (100万円, leverage 25x) ===")
        report.append("")
        for pair_id in profitable_pairs:
            r = results[pair_id]
            capital = 1_000_000
            position = capital * 25
            monthly_trades = r["n_trades"] / 14  # ~14 months of test data
            monthly_return = r["exp_value"] * monthly_trades
            monthly_profit = monthly_return * position
            annual_profit = monthly_profit * 12
            report.append(f"  {pair_id.upper()}: {r['description']}")
            report.append(f"    PF={r['pf']:.2f}, Monthly trades ~{monthly_trades:.0f}")
            report.append(f"    Monthly profit: {monthly_profit:+,.0f} yen")
            report.append(f"    Annual profit:  {annual_profit:+,.0f} yen")
            report.append(f"    Annual ROI:     {annual_profit/capital*100:+.1f}%")
            report.append("")

    # Disclaimer
    report.append("=== IMPORTANT ===")
    report.append("- All results are BACKTEST estimates. Real trading may differ.")
    report.append("- Paper trade for 200+ trades before real money.")
    report.append("- Past performance does not guarantee future results.")

    return "\n".join(report)


if __name__ == "__main__":
    results = run_pipeline()

    report = generate_report(results)
    print("\n\n")
    print(report)

    # Save report
    report_path = DATA_DIR / "multi_currency_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
