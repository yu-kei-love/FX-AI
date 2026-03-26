# ===========================================
# 12_news_features.py
# GDELTからニュース取得→感情スコアを特徴量に追加
# 経済指標前後は「待機条件」として扱う（予測しない）
# 11の最適パラメータで再学習し、54.74%と比較
# ===========================================

from datetime import datetime, timedelta, date
from calendar import monthrange
from urllib.parse import quote
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from hmmlearn.hmm import GaussianHMM

# TextBlob で感情スコア（-1〜1）
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# ----- 経済指標イベント生成（固定スケジュール） -----
def generate_economic_events(start_dt, end_dt):
    events = []
    y, m = start_dt.year, start_dt.month
    end_y, end_m = end_dt.year, end_dt.month
    while (y, m) <= (end_y, end_m):
        for d in range(1, 8):
            try:
                cand = date(y, m, d)
                if cand.weekday() == 4:
                    events.append((datetime(y, m, d, 14, 0, 0), 3))
                    break
            except ValueError:
                pass
        wed_count = 0
        for d in range(1, monthrange(y, m)[1] + 1):
            try:
                cand = date(y, m, d)
                if cand.weekday() == 2:
                    wed_count += 1
                    if wed_count == 3:
                        events.append((datetime(y, m, d, 14, 0, 0), 3))
                        break
            except ValueError:
                pass
        try:
            events.append((datetime(y, m, 15, 14, 0, 0), 2))
        except ValueError:
            pass
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return events


def build_economic_wait_flag(dt_series, events, hours_before=2, hours_after=1, min_importance=2):
    """重要度 min_importance 以上の発表の前 hours_before 時間・後 hours_after 時間は待機"""
    dt_arr = pd.to_datetime(dt_series)
    wait = np.zeros(len(dt_arr), dtype=bool)
    for i, t in enumerate(dt_arr):
        t = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        if t.tzinfo:
            t = t.replace(tzinfo=None)
        for ev_dt, imp in events:
            if imp < min_importance:
                continue
            if ev_dt.tzinfo:
                ev_dt = ev_dt.replace(tzinfo=None)
            delta_h = (ev_dt - t).total_seconds() / 3600
            if 0 <= delta_h <= hours_before or -hours_after <= delta_h < 0:
                wait[i] = True
                break
    return wait


# ----- GDELT API でニュース取得（APIキー不要） -----
GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = "forex USD JPY"
MAX_RECORDS = 250


def fetch_gdelt_month(start_dt, end_dt):
    """1ヶ月分のGDELT記事を取得。失敗時は None"""
    try:
        import requests
        start_str = start_dt.strftime("%Y%m%d%H%M%S")
        end_str = end_dt.strftime("%Y%m%d%H%M%S")
        params = {
            "query": QUERY,
            "mode": "artlist",
            "maxrecords": MAX_RECORDS,
            "startdatetime": start_str,
            "enddatetime": end_str,
            "format": "json",
        }
        url = GDELT_BASE + "?" + "&".join("{}={}".format(k, quote(str(v))) for k, v in params.items())
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if data is None:
            return None
        # レスポンスが配列の場合と {"articles": [...]} の両対応
        if isinstance(data, list):
            return data
        return data.get("articles", []) or []
    except Exception:
        return None


def fetch_gdelt_past_two_years_monthly(end_dt):
    """過去2年分を1ヶ月ごとに分割してGDELTから取得。各記事は (datetime, title) のリストで返す"""
    all_articles = []
    cur = end_dt - timedelta(days=365 * 2)
    while cur < end_dt:
        month_end = cur + timedelta(days=32)
        month_end = month_end.replace(day=1) - timedelta(seconds=1)
        if month_end > end_dt:
            month_end = end_dt
        articles = fetch_gdelt_month(cur, month_end)
        if articles:
            for a in articles:
                title = a.get("title") or a.get("snippet") or ""
                if not isinstance(title, str) or not title.strip():
                    continue
                dt_str = a.get("seendate") or a.get("date") or a.get("publishedat")
                if not dt_str:
                    continue
                try:
                    dt = pd.to_datetime(dt_str)
                    if dt.tzinfo:
                        dt = dt.tz_localize(None)
                    all_articles.append((dt, title))
                except Exception:
                    continue
        cur = month_end + timedelta(seconds=1)
    return all_articles


def build_sentiment_features_from_gdelt(df_index, end_dt):
    """
    GDELT過去2年を1ヶ月ずつ取得→TextBlobで感情スコア→1時間平均・24hMA・変化速度を計算。
    取得失敗やニュースがない時間は0で埋める。
    """
    df_idx = df_index.tz_localize(None) if df_index.tz is not None else df_index
    n = len(df_idx)
    sentiment_hourly = np.zeros(n)
    if not HAS_TEXTBLOB:
        return sentiment_hourly, np.zeros(n), np.zeros(n)

    articles = fetch_gdelt_past_two_years_monthly(end_dt)
    if not articles:
        return sentiment_hourly, np.zeros(n), np.zeros(n)

    rows = []
    for dt, title in articles:
        try:
            pol = TextBlob(str(title)).sentiment.polarity
            rows.append({"datetime": dt.floor("h") if hasattr(dt, "floor") else dt, "sentiment": pol})
        except Exception:
            continue
    if not rows:
        return sentiment_hourly, np.zeros(n), np.zeros(n)

    news_df = pd.DataFrame(rows)
    hourly = news_df.groupby("datetime")["sentiment"].mean()
    df_floor = pd.to_datetime(df_idx).floor("h")
    sentiment_hourly = hourly.reindex(df_floor).fillna(0).values
    sentiment_24h_ma = pd.Series(sentiment_hourly).rolling(24, min_periods=1).mean().values
    sentiment_velocity = np.zeros(n)
    sentiment_velocity[1:] = np.diff(sentiment_24h_ma)
    return sentiment_hourly, sentiment_24h_ma, sentiment_velocity


# ----- データ読み込み（11と同じベース） -----
script_dir = Path(__file__).resolve().parent
data_path = (script_dir / ".." / "data" / "usdjpy_1h.csv").resolve()
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

# HMM・特徴量（11と同じ）
df["Return"] = df["Close"].pct_change(24)
df["Volatility"] = df["Return"].rolling(24).std()
df_clean = df.dropna(subset=["Return", "Volatility"])
X_hmm = df_clean[["Return", "Volatility"]].values
model_hmm = GaussianHMM(
    n_components=3, covariance_type="full", n_iter=100, random_state=42,
)
model_hmm.fit(X_hmm)
states = model_hmm.predict(X_hmm)
df["Regime"] = np.nan
df.loc[df_clean.index, "Regime"] = states
df["Regime"] = df["Regime"].ffill().fillna(0).astype(int)
df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1

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
ret_1h = df["Close"].pct_change(1)
df["Volatility_24"] = ret_1h.rolling(24).std()
df["Hour"] = df.index.hour
df["DayOfWeek"] = df.index.dayofweek
df["Close_4h_later"] = df["Close"].shift(-4)
df["Label"] = (df["Close_4h_later"] > df["Close"]).astype(int)
regime_changes_rolling = (df["Regime"].diff().fillna(0) != 0).astype(int).rolling(3).sum()
df["Regime_changes_3h"] = regime_changes_rolling.fillna(0)
df["Abs_ret_1h"] = df["Return_1"].abs()

# ----- GDELT ニュース感情特徴量（失敗時は0で埋めるフォールバック） -----
df["sentiment_hourly"] = 0.0
df["sentiment_24h_ma"] = 0.0
df["sentiment_velocity"] = 0.0
end_dt = df.index.max()
if hasattr(end_dt, "to_pydatetime"):
    end_dt = end_dt.to_pydatetime()
if end_dt.tzinfo:
    end_dt = end_dt.replace(tzinfo=None)
print("GDELT からニュースを取得中（過去2年・1ヶ月ごと）...")
try:
    sh, s24, sv = build_sentiment_features_from_gdelt(df.index, end_dt)
    df["sentiment_hourly"] = sh
    df["sentiment_24h_ma"] = s24
    df["sentiment_velocity"] = sv
    if np.any(sh != 0) or np.any(s24 != 0):
        print("GDELT ニュース感情特徴量を取得しました。")
    else:
        print("GDELT で記事が得られなかったため、感情スコアは0で埋めました。")
except Exception as e:
    print("GDELT 取得失敗のため、感情スコアを0で埋めて続行します: {}".format(e))

# ----- 経済指標は「待機条件」のみ（特徴量には入れない） -----
events = generate_economic_events(
    df.index.min().to_pydatetime() if hasattr(df.index.min(), "to_pydatetime") else df.index.min(),
    df.index.max().to_pydatetime() if hasattr(df.index.max(), "to_pydatetime") else df.index.max(),
)
df["wait_economic"] = build_economic_wait_flag(
    df.index, events, hours_before=2, hours_after=1, min_importance=2
)

# ベース特徴量＋GDELT感情（経済指標は特徴量にしない）
feature_cols = [
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "MA_5", "MA_25", "MA_75",
    "Return_1", "Return_3", "Return_6", "Return_24",
    "Volatility_24", "Hour", "DayOfWeek",
    "Regime", "Regime_changed", "Regime_duration",
    "sentiment_hourly", "sentiment_24h_ma", "sentiment_velocity",
]
df = df.dropna(subset=[c for c in feature_cols if c in df.columns] + ["Label"])
for c in feature_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

X = df[feature_cols]
y_direction = df["Label"]
close_arr = df["Close"].values
n_total = len(df)

# ----- 11の最適パラメータでTriple-Barrier・学習/テスト分割 -----
BARRIER_UP, BARRIER_DOWN, BARRIER_T = 0.005, -0.003, 24
y_triple = np.full(n_total, np.nan)
for i in range(n_total - BARRIER_T):
    c0 = close_arr[i]
    label = 2
    for t in range(1, BARRIER_T + 1):
        ret = (close_arr[i + t] - c0) / c0
        if ret >= BARRIER_UP:
            label = 1
            break
        if ret <= BARRIER_DOWN:
            label = 0
            break
    y_triple[i] = label
df["Label_triple"] = y_triple

split_idx = int(n_total * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train_dir = y_direction.iloc[:split_idx]
y_test_dir = y_direction.iloc[split_idx:]
regime_test = df["Regime"].iloc[split_idx:].values
vol_test = df["Volatility_24"].iloc[split_idx:].values
rc_test = df["Regime_changes_3h"].iloc[split_idx:].values
abs_test = df["Abs_ret_1h"].iloc[split_idx:].values
wait_economic_test = df["wait_economic"].iloc[split_idx:].values
hist_avg_vol = df["Volatility_24"].iloc[:split_idx].mean()
hist_avg_abs = df["Abs_ret_1h"].iloc[:split_idx].mean()
if hist_avg_abs <= 0:
    hist_avg_abs = 1e-8

# モデルA（11パラメータ）
mask_a = ((df["Label_triple"] == 0) | (df["Label_triple"] == 1)).iloc[:split_idx]
X_a = X_train[mask_a]
y_a = df.loc[X_a.index, "Label_triple"].astype(int)
model_a = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_a.fit(X_a, y_a)

# モデルB（Meta-Labeling、採用率40%）
model_b_p = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_b_p.fit(X_train, y_train_dir)
X_train_meta = X_train.copy()
X_train_meta["primary_proba"] = model_b_p.predict_proba(X_train)[:, 1]
y_meta = (model_b_p.predict(X_train) == y_train_dir.values).astype(int)
model_b_s = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1,
)
model_b_s.fit(X_train_meta, y_meta)
X_test_meta = X_test.copy()
X_test_meta["primary_proba"] = model_b_p.predict_proba(X_test)[:, 1]
proba_adopt = model_b_s.predict_proba(X_test_meta)[:, 1]
thresh = np.percentile(model_b_s.predict_proba(X_train_meta)[:, 1], 60)
adopt_b = (proba_adopt >= thresh)

# モデルC（待機: 2倍・2回・3倍）
cond1 = vol_test > (2 * hist_avg_vol)
cond2 = rc_test >= 2
cond3 = abs_test > (3 * hist_avg_abs)
wait_c = cond1 | cond2 | cond3

pred_a = model_a.predict(X_test)
pred_b = model_b_p.predict(X_test)
y_test_values = y_test_dir.values
pred_final = np.full(len(y_test_values), np.nan)
for i in range(len(y_test_values)):
    if wait_economic_test[i]:
        continue
    r = regime_test[i]
    if r == 0:
        pred_final[i] = pred_a[i]
    elif r == 1:
        if adopt_b[i]:
            pred_final[i] = pred_b[i]
    else:
        if not wait_c[i]:
            pred_final[i] = pred_a[i]

mask_p = ~np.isnan(pred_final)
n_pred = mask_p.sum()
acc = (pred_final[mask_p] == y_test_values[mask_p]).mean() if n_pred > 0 else 0.0
acc_11 = 0.5474
diff = acc - acc_11

print("\n【GDELTニュース感情＋経済指標待機条件の結果】")
print("  予測サンプル数: {} / {}".format(int(n_pred), len(y_test_values)))
print("  全体の正解率: {:.4f} ({:.2f}%)".format(acc, acc * 100))
print("  11の結果（54.74%）との差: {:.4f} ({:+.2f}%)".format(diff, diff * 100))

# ----- MLflow（実験名: fx_ai_phase4） -----
mlflow.set_experiment("fx_ai_phase4")
with mlflow.start_run():
    mlflow.log_param("script", "12_news_features")
    mlflow.log_param("news_source", "gdelt")
    mlflow.log_param("economic_as_wait", True)
    mlflow.log_param("wait_hours_before", 2)
    mlflow.log_param("wait_hours_after", 1)
    mlflow.log_param("wait_min_importance", 2)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("accuracy_diff_vs_11", diff)
    mlflow.log_metric("n_predictions", int(n_pred))
    mlflow.log_metric("n_economic_wait", int(wait_economic_test.sum()))
    mlflow.sklearn.log_model(model_a, "model_a")
    mlflow.sklearn.log_model(model_b_p, "model_b_primary")
    mlflow.sklearn.log_model(model_b_s, "model_b_secondary")
    print("\nMLflow に記録しました（実験名: fx_ai_phase4）")
