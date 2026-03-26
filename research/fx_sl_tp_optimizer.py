# ===========================================
# fx_sl_tp_optimizer.py
# ATRベースのSL/TP最適化スクリプト
#
# Walk-Forward検証（拡張ウィンドウ、最大8フォールド）で
# ATR倍率ベースのSL/TP組み合わせ25通りを評価し、
# 最適な組み合わせを特定する。
#
# v3.2 params: n_estimators=500, learning_rate=0.03
# ===========================================

import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# 共通モジュール
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.common.data_loader import load_usdjpy_1h, add_rate_features, add_daily_trend_features
from research.common.features import add_technical_features, FEATURE_COLS
from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits

# ===================== 設定 =====================
FORECAST_HORIZON = 12
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4
N_ESTIMATORS = 500
LEARNING_RATE = 0.03
ATR_PERIOD = 14
MAX_HOLD_BARS = 24  # 最大保有時間（バー数）
SPREAD = 0.0003  # スプレッドコスト（USD/JPY）
MAX_FOLDS = 8

SL_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]


# ===================== ヘルパー関数 =====================

def compute_atr(df, period=14):
    """ATR(Average True Range)を計算する"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, min_periods=period).mean()
    return atr


def _resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


def _compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line


def _compute_bb_width(close, period=20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std) / sma.replace(0, np.nan)


def add_multi_timeframe_features(df):
    for rule, suffix in [("4h", "4h"), ("1D", "daily")]:
        resampled = _resample_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]], rule)
        close = resampled["Close"]
        feat = pd.DataFrame(index=resampled.index)
        feat[f"RSI_{suffix}"] = _compute_rsi(close, 14)
        feat[f"MACD_{suffix}"] = _compute_macd(close)
        feat[f"BB_width_{suffix}"] = _compute_bb_width(close)
        feat = feat.reindex(df.index, method="ffill")
        for col in feat.columns:
            df[col] = feat[col]
    return df


def add_volatility_regime_features(df):
    vol = df["Volatility_24"]
    df["Vol_percentile"] = vol.rolling(720, min_periods=72).apply(
        lambda x: (x[-1] >= x).sum() / len(x), raw=True
    )
    df["Vol_of_vol"] = vol.rolling(120, min_periods=24).std()
    return df


def add_calendar_awareness_features(df):
    h = df.index.hour
    dow = df.index.dayofweek
    df["Hour_x_DoW"] = h * 10 + dow
    df["Session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
    df["Session_london"] = ((h >= 7) & (h < 16)).astype(int)
    df["Session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    df["Session_overlap"] = ((h >= 13) & (h < 16)).astype(int)
    return df


def prepare_full_data():
    """USDJPY 1hデータを読み込み、全特徴量を生成する"""
    print("データ読み込み中...")
    df = load_usdjpy_1h()
    df = add_technical_features(df)
    df = add_rate_features(df)
    df = add_daily_trend_features(df)
    df = add_multi_timeframe_features(df)

    # 交互作用特徴量
    df["Return"] = df["Close"].pct_change(24)
    df["Volatility"] = df["Return"].rolling(24).std()
    df["RSI_x_Vol"] = df["RSI_14"] * df["Volatility_24"]
    df["MACD_norm"] = df["MACD"] / df["Volatility_24"].replace(0, np.nan)
    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / bb_range
    df["MA_cross"] = (df["MA_5"] - df["MA_75"]) / df["Close"]
    df["Momentum_accel"] = df["Return_1"] - df["Return_1"].shift(1)
    df["Vol_change"] = df["Volatility_24"].pct_change(6)
    df["HL_ratio"] = (df["High"] - df["Low"]) / df["Close"]
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Close_position"] = (df["Close"] - df["Low"]) / hl_range
    df["Return_skew_12"] = df["Return_1"].rolling(12).apply(
        lambda x: (x > 0).sum() / len(x) - 0.5, raw=True
    )

    df = add_volatility_regime_features(df)
    df = add_calendar_awareness_features(df)

    # ATR計算
    df["ATR"] = compute_atr(df, ATR_PERIOD)

    # 特徴量カラムリスト
    interaction_cols = [
        "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
        "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
        "Return_skew_12",
    ]
    mtf_cols = [
        "RSI_4h", "MACD_4h", "BB_width_4h",
        "RSI_daily", "MACD_daily", "BB_width_daily",
    ]
    vol_regime_cols = ["Vol_percentile", "Vol_of_vol"]
    calendar_cols = ["Hour_x_DoW", "Session_tokyo", "Session_london", "Session_ny", "Session_overlap"]
    base_feature_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]
    feature_cols = base_feature_cols + interaction_cols + mtf_cols + vol_regime_cols + calendar_cols

    df = df.dropna(subset=feature_cols + ["ATR"])
    print(f"データ準備完了: {len(df)}本 ({df.index[0]} ~ {df.index[-1]})")
    return df, feature_cols


def simulate_trades_with_sl_tp(
    df, entry_indices, directions, confidences, agreements,
    sl_mult, tp_mult, max_hold=MAX_HOLD_BARS,
):
    """
    ATRベースのSL/TPでトレードをシミュレーションする。

    各エントリーバーについて、後続のバーをたどり：
      - SLヒット → 損切り（SL価格で決済）
      - TPヒット → 利確（TP価格で決済）
      - max_hold到達 → 現在価格で決済（時間切れ）
    を判定してリターンを計算する。

    Returns:
        list of dict: 各トレードの結果
    """
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df["ATR"].values
    n = len(df)

    trades = []
    for i, entry_idx in enumerate(entry_indices):
        if entry_idx >= n - 1:
            continue

        entry_price = closes[entry_idx]
        atr_val = atrs[entry_idx]
        direction = directions[i]  # 1=BUY, 0=SELL

        sl_distance = atr_val * sl_mult
        tp_distance = atr_val * tp_mult

        if direction == 1:  # BUY
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # SELL
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        exit_price = None
        exit_reason = None
        exit_bar = None

        # 後続バーを順にチェック
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            bar_high = highs[j]
            bar_low = lows[j]
            bar_close = closes[j]

            if direction == 1:  # BUY
                # SLチェック（最優先）
                if bar_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    exit_bar = j
                    break
                # TPチェック
                if bar_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    exit_bar = j
                    break
            else:  # SELL
                # SLチェック
                if bar_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    exit_bar = j
                    break
                # TPチェック
                if bar_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    exit_bar = j
                    break

        # 時間切れ決済
        if exit_price is None:
            exit_idx = min(entry_idx + max_hold, n - 1)
            exit_price = closes[exit_idx]
            exit_reason = "TIME"
            exit_bar = exit_idx

        # リターン計算
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price

        net_return = raw_return - SPREAD

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_bar,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "atr": atr_val,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "raw_return": raw_return,
            "net_return": net_return,
        })

    return trades


def compute_trade_metrics(trades):
    """トレードリストから評価指標を計算する"""
    if not trades:
        return {
            "pf": float("nan"), "win_rate": float("nan"),
            "avg_win_loss_ratio": float("nan"), "mdd": float("nan"),
            "n_trades": 0, "avg_return": float("nan"),
            "total_return": float("nan"),
            "sl_count": 0, "tp_count": 0, "time_count": 0,
        }

    returns = np.array([t["net_return"] for t in trades])
    n_trades = len(returns)

    # PF
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_sum = wins.sum() if len(wins) > 0 else 0
    loss_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = profit_sum / loss_sum if loss_sum > 0 else (float("inf") if profit_sum > 0 else float("nan"))

    # Win rate
    win_rate = len(wins) / n_trades * 100.0

    # Avg win/loss ratio (payoff ratio)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss_val = abs(losses.mean()) if len(losses) > 0 else 0
    avg_win_loss_ratio = avg_win / avg_loss_val if avg_loss_val > 0 else float("nan")

    # MDD
    equity = 1.0 + np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100.0
    mdd = float(drawdown.max())

    # Exit reason counts
    sl_count = sum(1 for t in trades if t["exit_reason"] == "SL")
    tp_count = sum(1 for t in trades if t["exit_reason"] == "TP")
    time_count = sum(1 for t in trades if t["exit_reason"] == "TIME")

    return {
        "pf": pf,
        "win_rate": win_rate,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "mdd": mdd,
        "n_trades": n_trades,
        "avg_return": float(returns.mean()),
        "total_return": float(returns.sum()),
        "sl_count": sl_count,
        "tp_count": tp_count,
        "time_count": time_count,
    }


def run_walk_forward_optimization(df, feature_cols):
    """
    Walk-Forward検証（拡張ウィンドウ）でATRベースSL/TP組み合わせを評価する。
    """
    n = len(df)

    # 12h方向ラベル
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df = df.dropna(subset=["Label"])
    n = len(df)

    # Walk-Forward分割（拡張ウィンドウ）
    min_train = 4320  # 6ヶ月
    test_size = 720   # 1ヶ月
    splits = walk_forward_splits(n, min_train, test_size)

    # 最大8フォールドに制限（速度）
    if len(splits) > MAX_FOLDS:
        step = len(splits) / MAX_FOLDS
        selected = [splits[int(i * step)] for i in range(MAX_FOLDS)]
        splits = selected

    print(f"Walk-Forward分割: {len(splits)}フォールド")
    print(f"  学習開始: {min_train}本, テスト: {test_size}本/フォールド")

    # 全組み合わせ
    combos = list(itertools.product(SL_MULTIPLIERS, TP_MULTIPLIERS))
    print(f"SL/TP組み合わせ: {len(combos)}通り")

    # 結果格納
    combo_results = {combo: [] for combo in combos}

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_start = time.time()
        print(f"\n--- Fold {fold_idx+1}/{len(splits)} ---")
        print(f"  学習: {len(train_idx)}本, テスト: {len(test_idx)}本")

        # 学習データ
        X_train = df.iloc[train_idx][feature_cols].values
        y_train = df.iloc[train_idx]["Label"].values

        # テストデータ
        X_test = df.iloc[test_idx][feature_cols].values

        # アンサンブル学習
        ensemble = EnsembleClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        ensemble.fit(X_train, y_train)

        # テストデータで予測
        preds, agreement = ensemble.predict_with_agreement(X_test)
        proba = ensemble.predict_proba(X_test)[:, 1]
        confidence = np.maximum(proba, 1.0 - proba)

        # フィルター適用
        trade_mask = (confidence >= CONFIDENCE_THRESHOLD) & (agreement >= MIN_AGREEMENT)
        trade_indices_in_test = np.where(trade_mask)[0]

        if len(trade_indices_in_test) == 0:
            print(f"  トレードなし（フィルター通過0件）")
            continue

        # テストデータ中の実インデックス
        actual_indices = test_idx[trade_indices_in_test]
        directions = preds[trade_indices_in_test]
        confs = confidence[trade_indices_in_test]
        agrees = agreement[trade_indices_in_test]

        print(f"  トレード候補: {len(actual_indices)}件")

        # 各SL/TP組み合わせでシミュレーション
        for sl_mult, tp_mult in combos:
            trades = simulate_trades_with_sl_tp(
                df, actual_indices, directions, confs, agrees,
                sl_mult, tp_mult,
            )
            metrics = compute_trade_metrics(trades)
            combo_results[(sl_mult, tp_mult)].append(metrics)

        elapsed = time.time() - fold_start
        print(f"  完了: {elapsed:.1f}秒")

    return combo_results


def aggregate_results(combo_results):
    """全フォールドの結果を集計する"""
    summary = []

    for (sl_mult, tp_mult), fold_metrics_list in combo_results.items():
        if not fold_metrics_list or all(m["n_trades"] == 0 for m in fold_metrics_list):
            summary.append({
                "sl_mult": sl_mult, "tp_mult": tp_mult,
                "avg_pf": float("nan"), "avg_win_rate": float("nan"),
                "avg_win_loss_ratio": float("nan"), "avg_mdd": float("nan"),
                "total_trades": 0, "avg_return": float("nan"),
                "total_return": float("nan"),
                "avg_sl_pct": float("nan"), "avg_tp_pct": float("nan"),
                "avg_time_pct": float("nan"),
            })
            continue

        # トレードがある fold だけ集計
        valid = [m for m in fold_metrics_list if m["n_trades"] > 0]
        if not valid:
            summary.append({
                "sl_mult": sl_mult, "tp_mult": tp_mult,
                "avg_pf": float("nan"), "avg_win_rate": float("nan"),
                "avg_win_loss_ratio": float("nan"), "avg_mdd": float("nan"),
                "total_trades": 0, "avg_return": float("nan"),
                "total_return": float("nan"),
                "avg_sl_pct": float("nan"), "avg_tp_pct": float("nan"),
                "avg_time_pct": float("nan"),
            })
            continue

        total_trades = sum(m["n_trades"] for m in valid)

        # トレード数で加重平均
        w_pf = np.average([m["pf"] for m in valid if not np.isnan(m["pf"]) and not np.isinf(m["pf"])],
                          weights=[m["n_trades"] for m in valid if not np.isnan(m["pf"]) and not np.isinf(m["pf"])]) \
            if any(not np.isnan(m["pf"]) and not np.isinf(m["pf"]) for m in valid) else float("nan")
        w_wr = np.average([m["win_rate"] for m in valid], weights=[m["n_trades"] for m in valid])
        w_wlr = np.average(
            [m["avg_win_loss_ratio"] for m in valid if not np.isnan(m["avg_win_loss_ratio"])],
            weights=[m["n_trades"] for m in valid if not np.isnan(m["avg_win_loss_ratio"])]
        ) if any(not np.isnan(m["avg_win_loss_ratio"]) for m in valid) else float("nan")
        max_mdd = max(m["mdd"] for m in valid)
        total_return = sum(m["total_return"] for m in valid)
        avg_return = total_return / total_trades if total_trades > 0 else 0

        # Exit reason breakdown
        total_sl = sum(m["sl_count"] for m in valid)
        total_tp = sum(m["tp_count"] for m in valid)
        total_time = sum(m["time_count"] for m in valid)
        sl_pct = total_sl / total_trades * 100 if total_trades > 0 else 0
        tp_pct = total_tp / total_trades * 100 if total_trades > 0 else 0
        time_pct = total_time / total_trades * 100 if total_trades > 0 else 0

        summary.append({
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "avg_pf": w_pf,
            "avg_win_rate": w_wr,
            "avg_win_loss_ratio": w_wlr,
            "avg_mdd": max_mdd,
            "total_trades": total_trades,
            "avg_return": avg_return,
            "total_return": total_return,
            "avg_sl_pct": sl_pct,
            "avg_tp_pct": tp_pct,
            "avg_time_pct": time_pct,
        })

    return pd.DataFrame(summary)


def find_optimal(df_summary):
    """最適なSL/TP組み合わせを選択する（PFベース、MDD制約あり）"""
    # NaN行を除外
    valid = df_summary.dropna(subset=["avg_pf"])
    valid = valid[valid["total_trades"] >= 10]  # 最低10トレード

    if valid.empty:
        return None

    # PFが最大の組み合わせ（MDD 30%以下の制約付き）
    constrained = valid[valid["avg_mdd"] <= 30.0]
    if constrained.empty:
        constrained = valid  # 制約緩和

    best_idx = constrained["avg_pf"].idxmax()
    return constrained.loc[best_idx]


def main():
    start_time = time.time()

    # データ準備
    df, feature_cols = prepare_full_data()

    # Walk-Forward最適化実行
    combo_results = run_walk_forward_optimization(df, feature_cols)

    # 結果集計
    df_summary = aggregate_results(combo_results)
    df_summary = df_summary.sort_values("avg_pf", ascending=False)

    # 最適組み合わせ
    best = find_optimal(df_summary)

    elapsed = time.time() - start_time

    # ===== 結果出力 =====
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("ATR-Based SL/TP Optimization Results")
    output_lines.append("=" * 70)
    output_lines.append(f"Data: USDJPY 1H, {len(df)} bars")
    output_lines.append(f"Model: v3.2 Ensemble (n_est={N_ESTIMATORS}, lr={LEARNING_RATE})")
    output_lines.append(f"ATR Period: {ATR_PERIOD}")
    output_lines.append(f"Forecast Horizon: {FORECAST_HORIZON}h")
    output_lines.append(f"Filters: confidence>={CONFIDENCE_THRESHOLD}, agreement>={MIN_AGREEMENT}")
    output_lines.append(f"Max Hold: {MAX_HOLD_BARS} bars")
    output_lines.append(f"Max Folds: {MAX_FOLDS}")
    output_lines.append(f"Spread: {SPREAD}")
    output_lines.append(f"Total time: {elapsed:.0f}s")
    output_lines.append("")

    # 全組み合わせの結果テーブル
    output_lines.append("-" * 70)
    output_lines.append(f"{'SL':>4s} {'TP':>4s} | {'PF':>6s} {'WinR%':>6s} {'W/L':>5s} {'MDD%':>6s} | {'Trades':>6s} {'TotRet%':>8s} | {'SL%':>5s} {'TP%':>5s} {'TM%':>5s}")
    output_lines.append("-" * 70)

    for _, row in df_summary.iterrows():
        pf_str = f"{row['avg_pf']:.2f}" if not np.isnan(row['avg_pf']) else "  N/A"
        wr_str = f"{row['avg_win_rate']:.1f}" if not np.isnan(row['avg_win_rate']) else " N/A"
        wl_str = f"{row['avg_win_loss_ratio']:.2f}" if not np.isnan(row['avg_win_loss_ratio']) else " N/A"
        mdd_str = f"{row['avg_mdd']:.1f}" if not np.isnan(row['avg_mdd']) else " N/A"
        ret_str = f"{row['total_return']*100:.2f}" if not np.isnan(row['total_return']) else "  N/A"
        sl_str = f"{row['avg_sl_pct']:.0f}" if not np.isnan(row['avg_sl_pct']) else " N/A"
        tp_str = f"{row['avg_tp_pct']:.0f}" if not np.isnan(row['avg_tp_pct']) else " N/A"
        tm_str = f"{row['avg_time_pct']:.0f}" if not np.isnan(row['avg_time_pct']) else " N/A"

        output_lines.append(
            f"{row['sl_mult']:>4.1f} {row['tp_mult']:>4.1f} | {pf_str:>6s} {wr_str:>6s} {wl_str:>5s} {mdd_str:>6s} | {int(row['total_trades']):>6d} {ret_str:>8s} | {sl_str:>5s} {tp_str:>5s} {tm_str:>5s}"
        )

    output_lines.append("-" * 70)
    output_lines.append("")

    # 最適結果
    output_lines.append("=" * 70)
    output_lines.append("OPTIMAL SL/TP COMBINATION")
    output_lines.append("=" * 70)
    if best is not None:
        output_lines.append(f"  SL Multiplier: {best['sl_mult']:.1f} x ATR")
        output_lines.append(f"  TP Multiplier: {best['tp_mult']:.1f} x ATR")
        output_lines.append(f"  Risk/Reward:   1:{best['tp_mult']/best['sl_mult']:.2f}")
        output_lines.append(f"  Profit Factor: {best['avg_pf']:.2f}")
        output_lines.append(f"  Win Rate:      {best['avg_win_rate']:.1f}%")
        output_lines.append(f"  W/L Ratio:     {best['avg_win_loss_ratio']:.2f}")
        output_lines.append(f"  Max Drawdown:  {best['avg_mdd']:.1f}%")
        output_lines.append(f"  Total Trades:  {int(best['total_trades'])}")
        output_lines.append(f"  Total Return:  {best['total_return']*100:.2f}%")
        output_lines.append(f"  Avg Return:    {best['avg_return']*100:.4f}%")
        output_lines.append(f"  Exit Breakdown: SL={best['avg_sl_pct']:.0f}% TP={best['avg_tp_pct']:.0f}% TIME={best['avg_time_pct']:.0f}%")
    else:
        output_lines.append("  No valid combination found (insufficient trades)")
    output_lines.append("=" * 70)

    # 現在の設定との比較
    output_lines.append("")
    output_lines.append("Current paper_trade.py settings: SL=1.5xATR, TP=2.0xATR")
    if best is not None:
        current_key = (1.5, 2.0)
        current_row = df_summary[(df_summary["sl_mult"] == 1.5) & (df_summary["tp_mult"] == 2.0)]
        if not current_row.empty:
            cur = current_row.iloc[0]
            output_lines.append(f"  Current PF:  {cur['avg_pf']:.2f}  vs  Optimal PF:  {best['avg_pf']:.2f}")
            output_lines.append(f"  Current WR:  {cur['avg_win_rate']:.1f}%  vs  Optimal WR:  {best['avg_win_rate']:.1f}%")
            output_lines.append(f"  Current MDD: {cur['avg_mdd']:.1f}%  vs  Optimal MDD: {best['avg_mdd']:.1f}%")

    # 出力
    result_text = "\n".join(output_lines)
    print(result_text)

    # ファイル保存
    result_path = Path(__file__).resolve().parent / "fx_sl_tp_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"\n結果保存: {result_path}")

    return df_summary, best


if __name__ == "__main__":
    df_summary, best = main()
