# ===========================================
# 20_ab_test.py
# 正解率低下の原因を特定する A/B テストスクリプト
# 実行: python research/20_ab_test.py
# ===========================================

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# ----- 共通: データ読み込み & 基本特徴量作成（14_main_system と同等の前処理） -----
script_dir = Path(__file__).resolve().parent
data_dir = (script_dir / ".." / "data").resolve()
data_path = (data_dir / "usdjpy_1h.csv").resolve()

print("データ読み込み中:", data_path)
df = pd.read_csv(
    data_path,
    skiprows=3,
    names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Datetime"] = df["Datetime"].astype(str).str.slice(0, 19)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime", "Close"])
df = df.set_index("Datetime")
df = df.sort_index()

# ----- HMM レジーム（3状態, 標準化＋diag 共分散） -----
df["Return"] = df["Close"].pct_change(24)
df["Volatility"] = df["Return"].rolling(24).std()
df_clean = df.dropna(subset=["Return", "Volatility"])
X_hmm_raw = df_clean[["Return", "Volatility"]].values

scaler_hmm = StandardScaler()
X_hmm = scaler_hmm.fit_transform(X_hmm_raw)

def _fit_hmm_stable(X: np.ndarray, n_components: int = 3):
    """HMM を複数初期化して最良モデルを選ぶ（3レジーム専用）"""
    from numpy.linalg import LinAlgError

    best_model = None
    best_score = -np.inf
    for k in range(5):  # n_init=5 相当
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=200,
                random_state=42 + k,
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except (LinAlgError, ValueError):
            continue
    if best_model is None:
        raise RuntimeError("HMM 学習に失敗しました（3状態）。")
    return best_model

print("HMM（3レジーム）を学習中...")
model_hmm = _fit_hmm_stable(X_hmm, n_components=3)
df["Regime"] = np.nan
df.loc[df_clean.index, "Regime"] = model_hmm.predict(X_hmm)
df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)

# ----- 共通のテクニカル指標（14_main_system と同じ） -----
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss.replace(0, np.nan))
df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
df["BB_upper"] = sma20 + 2 * std20
df["BB_lower"] = sma20 - 2 * std20
df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, np.nan)
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_25"] = df["Close"].rolling(25).mean()
df["MA_75"] = df["Close"].rolling(75).mean()
df["Return_1"] = df["Close"].pct_change(1)
df["Return_3"] = df["Close"].pct_change(3)
df["Return_6"] = df["Close"].pct_change(6)
df["Return_24"] = df["Close"].pct_change(24)
df["Volatility_24"] = df["Return_1"].rolling(24).std()
df["Hour"] = df.index.hour
df["DayOfWeek"] = df.index.dayofweek
df["Close_4h_later"] = df["Close"].shift(-4)
df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)
df["Return_4h"] = (df["Close_4h_later"] - df["Close"]) / df["Close"]


# ----- 金利・日足トレンド特徴量（パターンBのみで使用） -----
def add_macro_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """金利データと日足トレンドを df にマージ（14_main_system と同様）"""
    df = df_in.copy()

    # 金利・金利差
    rate_us_10y = pd.Series(index=df.index, dtype=float)
    rate_diff = pd.Series(index=df.index, dtype=float)
    rate_trend = pd.Series(index=df.index, dtype=int)
    try:
        us_rate_path = (data_dir / "us_10y_rate.csv").resolve()
        diff_rate_path = (data_dir / "rate_diff.csv").resolve()
        df_us = pd.read_csv(us_rate_path, parse_dates=["Date"])
        df_us = df_us.set_index("Date").sort_index()
        df_diff = pd.read_csv(diff_rate_path, parse_dates=["Date"])
        df_diff = df_diff.set_index("Date").sort_index()
        df_rates = df_us.join(df_diff[["rate_diff"]], how="outer").sort_index()
        df_rates["ma20"] = df_rates["rate_us_10y"].rolling(20).mean()
        df_rates["rate_trend"] = (df_rates["rate_us_10y"] > df_rates["ma20"]).astype(int)
        df_rates = df_rates.reindex(df.index, method="ffill")
        rate_us_10y = df_rates["rate_us_10y"]
        rate_diff = df_rates["rate_diff"]
        rate_trend = df_rates["rate_trend"]
    except Exception:
        rate_us_10y = pd.Series(0.0, index=df.index)
        rate_diff = pd.Series(0.0, index=df.index)
        rate_trend = pd.Series(0, index=df.index)

    # 日足トレンド
    trend_200d = pd.Series(index=df.index, dtype=int)
    trend_strength = pd.Series(index=df.index, dtype=float)
    try:
        daily_path = (data_dir / "usdjpy_1d.csv").resolve()
        df_d = pd.read_csv(daily_path)
        df_d["Date"] = pd.to_datetime(df_d["Date"])
        df_d = df_d.set_index("Date").sort_index()
        close_d = pd.to_numeric(df_d["Close"], errors="coerce")
        ma200 = close_d.rolling(200).mean()
        trend_flag = (close_d > ma200).astype(int)
        strength = (close_d - ma200) / ma200.replace(0, pd.NA)
        df_trend_d = pd.DataFrame(
            {
                "trend_200d": trend_flag.fillna(0).astype(int),
                "trend_strength": strength.fillna(0.0),
            },
            index=close_d.index,
        )
        df_trend_h = df_trend_d.reindex(df.index, method="ffill")
        trend_200d = df_trend_h["trend_200d"]
        trend_strength = df_trend_h["trend_strength"]
    except Exception:
        trend_200d = pd.Series(0, index=df.index)
        trend_strength = pd.Series(0.0, index=df.index)

    df["rate_us_10y"] = rate_us_10y
    df["rate_diff"] = rate_diff
    df["rate_trend"] = rate_trend
    df["trend_200d"] = trend_200d
    df["trend_strength"] = trend_strength
    return df


# ----- 評価指標の計算ヘルパー（方向予測ベース） -----
def evaluate_directional_model(y_true: np.ndarray, y_pred: np.ndarray, ret4: np.ndarray) -> dict:
    """
    方向予測（0/1）に基づき、4時間リターンから PF, MDD, Sharpe 等を計算。
    ポジションサイズは常に 1.0 とする。
    """
    # 正解率
    mask_valid = ~np.isnan(y_pred)
    if mask_valid.sum() == 0:
        acc = float("nan")
    else:
        acc = (y_pred[mask_valid] == y_true[mask_valid]).mean()

    # 方向ベース損益
    trade_returns = []
    for i in range(len(y_pred)):
        if i >= len(ret4):
            break
        d = y_pred[i]
        if d not in (0, 1):
            continue
        direction_mult = 1.0 if d == 1 else -1.0
        trade_returns.append(ret4[i] * direction_mult)

    if not trade_returns:
        return {
            "accuracy": acc,
            "pf": float("nan"),
            "mdd": float("nan"),
            "sharpe": float("nan"),
        }

    tr = np.array(trade_returns, dtype=float)
    # PF
    profit = tr[tr > 0].sum()
    loss = tr[tr < 0].sum()
    pf = (profit / -loss) if loss < 0 else (float("inf") if profit > 0 else float("nan"))

    # MDD（初期資金1.0）
    initial_capital = 1.0
    equity_curve = initial_capital + np.cumsum(tr)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100.0
    mdd = float(drawdown.max()) if len(drawdown) > 0 else float("nan")

    # Sharpe（年率換算, 1時間足）
    mean_r = float(tr.mean())
    std_r = float(tr.std(ddof=1)) if len(tr) > 1 else float("nan")
    sharpe = (mean_r / std_r * math.sqrt(24 * 365)) if std_r and std_r > 0 else float("nan")

    return {
        "accuracy": acc,
        "pf": pf,
        "mdd": mdd,
        "sharpe": sharpe,
    }


# ----- パターンA: シンプル版（原点回帰） -----
def run_pattern_a(df_base: pd.DataFrame):
    """
    パターンA: シンプル版
    ・特徴量: RSI・MACD・BB・MA・Return・Volatility・Hour・DayOfWeek・Regime
    ・HMMレジーム: 3状態（上で計算済み）
    ・サンプルウェイトなし
    ・金利・日足トレンドなし
    ・LightGBM デフォルトパラメータ
    ・4時間後の方向（Label）を予測
    """
    df = df_base.copy()
    feature_cols_a = [
        "RSI_14", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_lower", "BB_width",
        "MA_5", "MA_25", "MA_75",
        "Return_1", "Return_3", "Return_6", "Return_24",
        "Volatility_24", "Hour", "DayOfWeek",
        "Regime",
    ]
    df = df.dropna(subset=feature_cols_a + ["Label", "Return_4h"])

    X = df[feature_cols_a].values
    y = df["Label"].values.astype(int)
    ret4 = df["Return_4h"].values

    n_total = len(df)
    split_idx = int(n_total * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ret4_test = ret4[split_idx:]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    metrics = evaluate_directional_model(y_test, y_pred, ret4_test)
    return metrics


# ----- パターンB: 現在の設計（best_params.json + 全特徴量 + サンプルウェイト） -----
def run_pattern_b(df_base: pd.DataFrame):
    """
    パターンB: 現在の設計に近い構成
    ・data/best_params.json のパラメータを使用（存在する場合）
    ・全特徴量（金利・日足トレンド含む）
    ・サンプルウェイトあり（期間別重み）
    ・4時間後の方向（Label）を予測
    """
    df = add_macro_features(df_base.copy())

    # best_params.json の読み込み（なければデフォルト）
    best_params_path = (data_dir / "best_params.json").resolve()
    if best_params_path.exists():
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_cfg = json.load(f)
        tb = best_cfg.get("triple_barrier", {})
        wm = best_cfg.get("wait_mode", {})
        ml = best_cfg.get("meta_labeling", {})
        lg = best_cfg.get("lgbm", {})
        sw = best_cfg.get("sample_weights", {})
        print("パターンB: best_params.json のパラメータを使用します。")
    else:
        tb = {}
        wm = {}
        ml = {}
        lg = {}
        sw = {}
        print("パターンB: best_params.json がないためデフォルトパラメータを使用します。")

    # サンプルウェイト設定（期間別）
    ref_ts = df.index.max()
    months_diff = []
    for ts in df.index:
        m = (ref_ts.year - ts.year) * 12 + (ref_ts.month - ts.month)
        months_diff.append(m)
    months_diff = np.array(months_diff)

    w_recent = float(sw.get("weight_recent_6m", 1.0))
    w_6_12 = float(sw.get("weight_6_12m", 0.7))
    w_12_18 = float(sw.get("weight_12_18m", 0.4))
    w_over_18 = float(sw.get("weight_over_18m", 0.2))

    weights = np.where(
        months_diff < 6,
        w_recent,
        np.where(
            months_diff < 12,
            w_6_12,
            np.where(
                months_diff < 18,
                w_12_18,
                w_over_18,
            ),
        ),
    )

    # 特徴量（14_main_system と同等の全セット）
    feature_cols_b = [
        "RSI_14", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_lower", "BB_width",
        "MA_5", "MA_25", "MA_75",
        "Return_1", "Return_3", "Return_6", "Return_24",
        "Volatility_24", "Hour", "DayOfWeek",
        "Regime",
        "rate_us_10y", "rate_diff", "rate_trend",
        "trend_200d", "trend_strength",
    ]
    df = df.dropna(subset=feature_cols_b + ["Label", "Return_4h"])

    X = df[feature_cols_b].values
    y = df["Label"].values.astype(int)
    ret4 = df["Return_4h"].values
    w_all = weights[: len(df)]

    n_total = len(df)
    split_idx = int(n_total * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ret4_test = ret4[split_idx:]
    w_train = w_all[:split_idx]

    n_estimators = int(lg.get("n_estimators", 200))
    learning_rate = float(lg.get("learning_rate", 0.05))
    max_depth = int(lg.get("max_depth", 6))

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    metrics = evaluate_directional_model(y_test, y_pred, ret4_test)
    return metrics


def main():
    # パターンA: シンプル版
    print("\n===== パターンA: シンプル版（原点回帰） =====")
    metrics_a = run_pattern_a(df)
    print("正解率: {:.4f} ({:.2f}%)".format(
        metrics_a["accuracy"], metrics_a["accuracy"] * 100 if not math.isnan(metrics_a["accuracy"]) else float("nan")
    ))
    print("PF: {:.2f}".format(metrics_a["pf"]))
    print("MDD: {:.2f}%".format(metrics_a["mdd"]))
    print("Sharpe: {:.2f}".format(metrics_a["sharpe"]))

    # パターンB: 現在の設計
    print("\n===== パターンB: 現在の設計 =====")
    metrics_b = run_pattern_b(df)
    print("正解率: {:.4f} ({:.2f}%)".format(
        metrics_b["accuracy"], metrics_b["accuracy"] * 100 if not math.isnan(metrics_b["accuracy"]) else float("nan")
    ))
    print("PF: {:.2f}".format(metrics_b["pf"]))
    print("MDD: {:.2f}%".format(metrics_b["mdd"]))
    print("Sharpe: {:.2f}".format(metrics_b["sharpe"]))

    # 簡易比較
    print("\n===== A/B 比較サマリー =====")
    print("正解率: A={:.2f}% / B={:.2f}%".format(
        metrics_a["accuracy"] * 100 if not math.isnan(metrics_a["accuracy"]) else float("nan"),
        metrics_b["accuracy"] * 100 if not math.isnan(metrics_b["accuracy"]) else float("nan"),
    ))
    print("PF:     A={:.2f}  / B={:.2f}".format(metrics_a["pf"], metrics_b["pf"]))
    print("MDD:    A={:.2f}% / B={:.2f}%".format(metrics_a["mdd"], metrics_b["mdd"]))
    print("Sharpe: A={:.2f}  / B={:.2f}".format(metrics_a["sharpe"], metrics_b["sharpe"]))


if __name__ == "__main__":
    main()

