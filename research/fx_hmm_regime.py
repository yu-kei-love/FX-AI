# ===========================================
# fx_hmm_regime.py
# HMM (Hidden Markov Model) ベースのマーケットレジーム検出
#
# Walk-Forward検証で3つの戦略を比較:
#   1. Baseline: レジームなし
#   2. Regime as Feature: レジームを特徴量として追加
#   3. Regime as Filter: choppy レジーム中のトレードをスキップ
#
# 結果: research/fx_hmm_regime_results.txt に保存
# ===========================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from numpy.linalg import LinAlgError

warnings.filterwarnings("ignore")

# パス設定
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from research.common.data_loader import load_usdjpy_1h
from research.common.features import add_technical_features, FEATURE_COLS
from research.common.ensemble import EnsembleClassifier


# ===========================================
# HMM レジーム検出
# ===========================================

def fit_hmm_on_train(df_train, n_components=3):
    """trainデータのみでHMMを学習する (lookahead bias なし)

    Returns:
        model: 学習済みGaussianHMM (or None if failed)
        scaler: 学習済みStandardScaler
        regime_labels: dict mapping state_id -> label name
    """
    ret = df_train["Close"].pct_change(24)
    vol = ret.rolling(24).std()
    hmm_df = pd.DataFrame({"Return": ret, "Volatility": vol}, index=df_train.index)
    hmm_df = hmm_df.dropna()

    if len(hmm_df) < 100:
        return None, None, None

    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(hmm_df[["Return", "Volatility"]].values)

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
        return None, None, None

    # レジームのラベル付け: 平均リターンで分類
    # state with highest mean return -> trending_up
    # state with lowest mean return -> trending_down
    # remaining state -> choppy/ranging
    states = best_model.predict(X_hmm)
    state_returns = {}
    for s in range(n_components):
        mask = states == s
        if mask.sum() > 0:
            state_returns[s] = hmm_df["Return"].values[mask].mean()
        else:
            state_returns[s] = 0.0

    sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s])
    regime_labels = {
        sorted_states[0]: "trending_down",
        sorted_states[-1]: "trending_up",
    }
    for s in sorted_states[1:-1]:
        regime_labels[s] = "choppy"

    return best_model, scaler, regime_labels


def predict_regimes(df, model, scaler, regime_labels):
    """全データにレジームを割り当てる

    Returns:
        regime_series: pd.Series of regime state IDs (int)
        regime_name_series: pd.Series of regime names (str)
        choppy_mask: pd.Series (bool) - True if choppy regime
    """
    ret = df["Close"].pct_change(24)
    vol = ret.rolling(24).std()
    hmm_df = pd.DataFrame({"Return": ret, "Volatility": vol}, index=df.index)
    hmm_clean = hmm_df.dropna()

    if len(hmm_clean) == 0 or model is None:
        regime = pd.Series(0, index=df.index)
        regime_name = pd.Series("choppy", index=df.index)
        choppy = pd.Series(True, index=df.index)
        return regime, regime_name, choppy

    X = scaler.transform(hmm_clean[["Return", "Volatility"]].values)
    states = model.predict(X)

    regime = pd.Series(np.nan, index=df.index)
    regime.loc[hmm_clean.index] = states
    regime = regime.ffill().fillna(0).astype(int)

    regime_name = regime.map(regime_labels).fillna("choppy")
    choppy = regime_name == "choppy"

    return regime, regime_name, choppy


def add_regime_derived_features(df, regime):
    """レジーム関連の派生特徴量を追加"""
    df = df.copy()
    df["Regime"] = regime.values
    df["Regime_changed"] = (df["Regime"] != df["Regime"].shift(1)).astype(int)
    regime_grp = (df["Regime"] != df["Regime"].shift(1)).cumsum()
    df["Regime_duration"] = df.groupby(regime_grp).cumcount() + 1
    df["Regime_changes_3h"] = (
        df["Regime"].diff().fillna(0) != 0
    ).astype(int).rolling(3).sum().fillna(0)
    return df


# ===========================================
# データ準備
# ===========================================

def prepare_base_data():
    """基本データと特徴量を準備"""
    print("データ読み込み中...")
    df = load_usdjpy_1h()
    df = add_technical_features(df)

    # interaction features (paper_trade.py と同じ)
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

    # ラベル: 12時間後の方向
    FORECAST_HORIZON = 12
    df["Close_Nh_later"] = df["Close"].shift(-FORECAST_HORIZON)
    df["Label"] = (df["Close_Nh_later"] > df["Close"]).astype(int)
    df["Return_Nh"] = (df["Close_Nh_later"] - df["Close"]) / df["Close"]

    # Base feature columns (no regime)
    base_feature_cols = [c for c in FEATURE_COLS if not c.startswith("Regime")]
    interaction_cols = [
        "RSI_x_Vol", "MACD_norm", "BB_position", "MA_cross",
        "Momentum_accel", "Vol_change", "HL_ratio", "Close_position",
        "Return_skew_12",
    ]
    feature_cols = base_feature_cols + interaction_cols

    df = df.dropna(subset=feature_cols + ["Label", "Return_Nh"])
    print(f"データ準備完了: {len(df)} 行, 期間: {df.index[0]} ~ {df.index[-1]}")
    return df, feature_cols


# ===========================================
# Walk-Forward 検証
# ===========================================

TRAIN_SIZE = 2000    # 学習期間 (約3ヶ月)
TEST_SIZE = 500      # テスト期間 (約3週間)
STEP_SIZE = 500      # ステップ
FORECAST_HORIZON = 12
CONFIDENCE_THRESHOLD = 0.60
MIN_AGREEMENT = 4


def compute_pf_and_metrics(returns):
    """Profit Factor と主要メトリクスを計算"""
    if len(returns) == 0:
        return {"pf": 0.0, "sharpe": 0.0, "win_rate": 0.0, "n_trades": 0, "total_return": 0.0}

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    n = len(returns)
    win_rate = len(wins) / n if n > 0 else 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252 * 24 / FORECAST_HORIZON)) if std_ret > 0 else 0.0
    total_ret = returns.sum()

    return {
        "pf": pf,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_trades": n,
        "total_return": total_ret,
    }


def run_walk_forward(df, feature_cols, mode="baseline"):
    """Walk-Forward検証を実行

    mode:
        "baseline" - レジームなし
        "regime_feature" - レジームを特徴量として追加
        "regime_filter" - choppy レジーム中のトレードをスキップ
    """
    n = len(df)
    n_windows = max(0, (n - TRAIN_SIZE - TEST_SIZE) // STEP_SIZE + 1)

    all_results = []
    window_metrics = []

    print(f"\n{'='*60}")
    print(f"Walk-Forward: {mode}")
    print(f"  学習: {TRAIN_SIZE}h, テスト: {TEST_SIZE}h, ステップ: {STEP_SIZE}h")
    print(f"  ウィンドウ数: {n_windows}")
    print(f"{'='*60}")

    for k in range(n_windows):
        train_start = k * STEP_SIZE
        train_end = train_start + TRAIN_SIZE
        test_start = train_end
        test_end = min(test_start + TEST_SIZE, n)

        if test_end - test_start < TEST_SIZE:
            break

        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]

        # ----- HMM学習 (trainのみ) -----
        hmm_model, hmm_scaler, regime_labels = None, None, None
        if mode in ("regime_feature", "regime_filter"):
            hmm_model, hmm_scaler, regime_labels = fit_hmm_on_train(df_train)

        # ----- 特徴量の準備 -----
        if mode == "regime_feature" and hmm_model is not None:
            # trainデータにレジーム特徴量追加
            regime_train, _, _ = predict_regimes(df_train, hmm_model, hmm_scaler, regime_labels)
            df_train_feat = add_regime_derived_features(df_train, regime_train)

            regime_test, _, choppy_test = predict_regimes(df_test, hmm_model, hmm_scaler, regime_labels)
            df_test_feat = add_regime_derived_features(df_test, regime_test)

            regime_cols = ["Regime", "Regime_duration", "Regime_changes_3h"]
            feat_cols_this = feature_cols + regime_cols
        elif mode == "regime_filter" and hmm_model is not None:
            df_train_feat = df_train
            df_test_feat = df_test
            feat_cols_this = feature_cols

            _, _, choppy_test = predict_regimes(df_test, hmm_model, hmm_scaler, regime_labels)
        else:
            df_train_feat = df_train
            df_test_feat = df_test
            feat_cols_this = feature_cols
            choppy_test = pd.Series(False, index=df_test.index)

        # ----- 有効カラムチェック -----
        valid_cols = [c for c in feat_cols_this if c in df_train_feat.columns and c in df_test_feat.columns]

        X_train = df_train_feat[valid_cols].values
        y_train = df_train_feat["Label"].values
        X_test = df_test_feat[valid_cols].values
        y_test = df_test_feat["Label"].values
        returns_test = df_test_feat["Return_Nh"].values

        # ----- アンサンブル学習 -----
        ensemble = EnsembleClassifier(n_estimators=500, learning_rate=0.03)
        ensemble.fit(
            pd.DataFrame(X_train, columns=valid_cols),
            pd.Series(y_train),
        )

        # ----- 予測 -----
        X_test_df = pd.DataFrame(X_test, columns=valid_cols)
        preds, agreement = ensemble.predict_with_agreement(X_test_df)
        probas = ensemble.predict_proba(X_test_df)

        # confidence filter
        confidence = np.maximum(probas[:, 1], 1.0 - probas[:, 1])
        conf_mask = confidence >= CONFIDENCE_THRESHOLD
        agree_mask = agreement >= MIN_AGREEMENT

        trade_mask = conf_mask & agree_mask

        # regime filter (only for regime_filter mode)
        if mode == "regime_filter" and hmm_model is not None:
            regime_trade_mask = ~choppy_test.values
            trade_mask = trade_mask & regime_trade_mask

        # direction-adjusted returns
        direction = np.where(preds == 1, 1.0, -1.0)
        adjusted_returns = direction * returns_test

        # only count trades that pass filter
        trade_returns = adjusted_returns[trade_mask]

        metrics = compute_pf_and_metrics(pd.Series(trade_returns))
        window_metrics.append(metrics)

        all_results.append({
            "window": k + 1,
            "test_start": df_test.index[0],
            "test_end": df_test.index[-1],
            **metrics,
        })

        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float('inf') else "inf"
        print(f"  Window {k+1:3d}: "
              f"{df_test.index[0].strftime('%Y-%m-%d')} ~ {df_test.index[-1].strftime('%Y-%m-%d')} | "
              f"PF={pf_str:>6s} | Win={metrics['win_rate']:.1%} | "
              f"N={metrics['n_trades']:4d} | Sharpe={metrics['sharpe']:.2f}")

    # ----- 全体集計 -----
    if not window_metrics:
        print("  !! データ不足でウィンドウなし")
        return {"mode": mode, "windows": [], "summary": {}}

    avg_pf = np.mean([m["pf"] for m in window_metrics if m["pf"] != float("inf")])
    avg_sharpe = np.mean([m["sharpe"] for m in window_metrics])
    avg_win = np.mean([m["win_rate"] for m in window_metrics])
    total_trades = sum(m["n_trades"] for m in window_metrics)
    total_ret = sum(m["total_return"] for m in window_metrics)

    summary = {
        "avg_pf": avg_pf,
        "avg_sharpe": avg_sharpe,
        "avg_win_rate": avg_win,
        "total_trades": total_trades,
        "total_return": total_ret,
        "n_windows": len(window_metrics),
    }

    print(f"\n  --- {mode} 全体サマリー ---")
    print(f"  平均PF: {avg_pf:.3f}")
    print(f"  平均Sharpe: {avg_sharpe:.2f}")
    print(f"  平均Win率: {avg_win:.1%}")
    print(f"  総トレード数: {total_trades}")
    print(f"  総リターン: {total_ret:.4f}")

    return {"mode": mode, "windows": all_results, "summary": summary}


# ===========================================
# レジーム分析 (可視化用情報)
# ===========================================

def analyze_regimes(df):
    """全データでHMMフィット → レジーム特性を分析して表示"""
    print("\n" + "=" * 60)
    print("HMM レジーム分析 (全データ)")
    print("=" * 60)

    model, scaler, labels = fit_hmm_on_train(df)
    if model is None:
        print("  HMM学習に失敗しました")
        return

    regime, regime_name, choppy = predict_regimes(df, model, scaler, labels)

    for state_id, label in labels.items():
        mask = regime == state_id
        count = mask.sum()
        pct = count / len(df) * 100
        avg_ret = df.loc[mask, "Return_Nh"].mean() if "Return_Nh" in df.columns else 0
        avg_vol = df.loc[mask, "Volatility_24"].mean() if "Volatility_24" in df.columns else 0
        print(f"  State {state_id} ({label:15s}): "
              f"{count:5d} bars ({pct:5.1f}%) | "
              f"Avg 12h Return: {avg_ret:+.5f} | "
              f"Avg Vol: {avg_vol:.6f}")

    # レジーム遷移行列
    print("\n  レジーム遷移行列 (train):")
    trans = model.transmat_
    header = "      " + "  ".join([f"  S{i}" for i in range(3)])
    print(header)
    for i in range(3):
        row = "  S{}: ".format(i) + "  ".join([f"{trans[i,j]:.2f}" for j in range(3)])
        print(row)

    return regime, regime_name, choppy


# ===========================================
# メイン
# ===========================================

def main():
    df, feature_cols = prepare_base_data()

    # レジーム分析
    analyze_regimes(df)

    # Walk-Forward 3モード比較
    result_baseline = run_walk_forward(df, feature_cols, mode="baseline")
    result_regime_feat = run_walk_forward(df, feature_cols, mode="regime_feature")
    result_regime_filter = run_walk_forward(df, feature_cols, mode="regime_filter")

    # ===========================================
    # 結果比較テーブル
    # ===========================================
    print("\n" + "=" * 80)
    print("Walk-Forward 比較サマリー")
    print("=" * 80)

    header = f"{'Mode':<25s} | {'Avg PF':>8s} | {'Avg Sharpe':>10s} | {'Win Rate':>8s} | {'Trades':>7s} | {'Total Ret':>10s}"
    print(header)
    print("-" * 80)

    for result in [result_baseline, result_regime_feat, result_regime_filter]:
        s = result["summary"]
        if not s:
            continue
        mode = result["mode"]
        pf_str = f"{s['avg_pf']:.3f}"
        print(f"{mode:<25s} | {pf_str:>8s} | {s['avg_sharpe']:>10.2f} | "
              f"{s['avg_win_rate']:>7.1%} | {s['total_trades']:>7d} | "
              f"{s['total_return']:>10.4f}")

    # ===========================================
    # 判定: 改善があるか？
    # ===========================================
    baseline_pf = result_baseline["summary"].get("avg_pf", 0)
    feat_pf = result_regime_feat["summary"].get("avg_pf", 0)
    filter_pf = result_regime_filter["summary"].get("avg_pf", 0)

    baseline_sharpe = result_baseline["summary"].get("avg_sharpe", 0)
    feat_sharpe = result_regime_feat["summary"].get("avg_sharpe", 0)
    filter_sharpe = result_regime_filter["summary"].get("avg_sharpe", 0)

    improvements = []
    if feat_pf > baseline_pf * 1.05 and feat_sharpe > baseline_sharpe:
        improvements.append("regime_feature")
    if filter_pf > baseline_pf * 1.05 and filter_sharpe > baseline_sharpe:
        improvements.append("regime_filter")

    print("\n" + "=" * 80)
    if improvements:
        print(f"改善が確認された戦略: {', '.join(improvements)}")
        for imp in improvements:
            if imp == "regime_feature":
                print(f"  regime_feature: PF {baseline_pf:.3f} -> {feat_pf:.3f} "
                      f"(+{(feat_pf/baseline_pf - 1)*100:.1f}%), "
                      f"Sharpe {baseline_sharpe:.2f} -> {feat_sharpe:.2f}")
            elif imp == "regime_filter":
                print(f"  regime_filter: PF {baseline_pf:.3f} -> {filter_pf:.3f} "
                      f"(+{(filter_pf/baseline_pf - 1)*100:.1f}%), "
                      f"Sharpe {baseline_sharpe:.2f} -> {filter_sharpe:.2f}")
    else:
        print("HMMレジームによる明確な改善なし。Baselineを維持。")
        print(f"  Baseline:       PF={baseline_pf:.3f}, Sharpe={baseline_sharpe:.2f}")
        print(f"  Regime Feature: PF={feat_pf:.3f}, Sharpe={feat_sharpe:.2f}")
        print(f"  Regime Filter:  PF={filter_pf:.3f}, Sharpe={filter_sharpe:.2f}")

    # ===========================================
    # 結果をファイルに保存
    # ===========================================
    output_path = script_dir / "fx_hmm_regime_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("HMM Regime Detection - Walk-Forward Validation Results\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Data: {df.index[0]} ~ {df.index[-1]} ({len(df)} bars)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Train Size: {TRAIN_SIZE}h\n")
        f.write(f"  Test Size:  {TEST_SIZE}h\n")
        f.write(f"  Step Size:  {STEP_SIZE}h\n")
        f.write(f"  Forecast Horizon: {FORECAST_HORIZON}h\n")
        f.write(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"  Min Agreement: {MIN_AGREEMENT}/5\n")
        f.write(f"  HMM States: 3 (trending_up, trending_down, choppy)\n\n")

        f.write("-" * 80 + "\n")
        f.write("Walk-Forward Comparison Summary\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mode':<25s} | {'Avg PF':>8s} | {'Avg Sharpe':>10s} | "
                f"{'Win Rate':>8s} | {'Trades':>7s} | {'Total Ret':>10s}\n")
        f.write("-" * 80 + "\n")

        for result in [result_baseline, result_regime_feat, result_regime_filter]:
            s = result["summary"]
            if not s:
                continue
            mode = result["mode"]
            pf_str = f"{s['avg_pf']:.3f}"
            f.write(f"{mode:<25s} | {pf_str:>8s} | {s['avg_sharpe']:>10.2f} | "
                    f"{s['avg_win_rate']:>7.1%} | {s['total_trades']:>7d} | "
                    f"{s['total_return']:>10.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Per-Window Details\n")
        f.write("-" * 80 + "\n")

        for result in [result_baseline, result_regime_feat, result_regime_filter]:
            f.write(f"\n[{result['mode']}]\n")
            for w in result["windows"]:
                pf_str = f"{w['pf']:.2f}" if w['pf'] != float('inf') else "inf"
                f.write(f"  Window {w['window']:3d}: "
                        f"{w['test_start'].strftime('%Y-%m-%d')} ~ "
                        f"{w['test_end'].strftime('%Y-%m-%d')} | "
                        f"PF={pf_str:>6s} | Win={w['win_rate']:.1%} | "
                        f"N={w['n_trades']:4d} | Sharpe={w['sharpe']:.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Conclusion\n")
        f.write("=" * 80 + "\n")
        if improvements:
            f.write(f"Improvement confirmed for: {', '.join(improvements)}\n")
            for imp in improvements:
                if imp == "regime_feature":
                    f.write(f"  regime_feature: PF {baseline_pf:.3f} -> {feat_pf:.3f} "
                            f"(+{(feat_pf/baseline_pf - 1)*100:.1f}%), "
                            f"Sharpe {baseline_sharpe:.2f} -> {feat_sharpe:.2f}\n")
                elif imp == "regime_filter":
                    f.write(f"  regime_filter: PF {baseline_pf:.3f} -> {filter_pf:.3f} "
                            f"(+{(filter_pf/baseline_pf - 1)*100:.1f}%), "
                            f"Sharpe {baseline_sharpe:.2f} -> {filter_sharpe:.2f}\n")
            f.write("\nRecommendation: Apply the improved strategy to production.\n")
        else:
            f.write("No clear improvement from HMM regime detection.\n")
            f.write(f"  Baseline:       PF={baseline_pf:.3f}, Sharpe={baseline_sharpe:.2f}\n")
            f.write(f"  Regime Feature: PF={feat_pf:.3f}, Sharpe={feat_sharpe:.2f}\n")
            f.write(f"  Regime Filter:  PF={filter_pf:.3f}, Sharpe={filter_sharpe:.2f}\n")
            f.write("\nRecommendation: Keep baseline (no regime). "
                    "HMM regime features already pruned in v3.3.\n")

    print(f"\n結果を保存しました: {output_path}")

    return {
        "baseline": result_baseline,
        "regime_feature": result_regime_feat,
        "regime_filter": result_regime_filter,
        "improvements": improvements,
    }


if __name__ == "__main__":
    results = main()
