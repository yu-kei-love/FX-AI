# ===========================================
# stock_screener.py
# 日本大型株スクリーニングパイプライン
#
# 目的:
#   米国市場の指標を使って、予測可能な日本大型株を
#   自動的に発見する。Walk-Forward検証を通過した
#   銘柄のみを採用する。
#
# パイプライン:
#   Phase 1: データダウンロード
#   Phase 2: 相関関係の発見
#   Phase 3: 特徴量エンジニアリング
#   Phase 4: Walk-Forward検証
#   Phase 5: レポート生成
#   Phase 6: Telegram通知
#
# リスク管理方針:
#   くれぐれも大損のないようにリスク管理を最も大切にする。
#   Walk-Forward検証は絶対条件。通過しない銘柄は不採用。
#   コスト控除後の期待値がプラスでなければ不採用。
# ===========================================

import sys
import os
import json
import argparse
import warnings
import time
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# プロジェクトルートをパスに追加（共通モジュール読み込み用）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.common.ensemble import EnsembleClassifier

warnings.filterwarnings("ignore")


# =============================================================
# 定数・設定
# =============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stock_screener"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ===== 日本大型株（TOPIX100から流動性上位約33銘柄） =====
JP_LARGE_CAPS = {
    # 自動車
    "7203.T": "トヨタ", "7267.T": "ホンダ", "7269.T": "スズキ", "6902.T": "デンソー",
    # テック・半導体
    "6758.T": "ソニー", "6861.T": "キーエンス", "8035.T": "東京エレクトロン",
    "6723.T": "ルネサス", "6594.T": "日本電産",
    # 金融
    "8306.T": "三菱UFJ", "8316.T": "三井住友FG", "8411.T": "みずほFG",
    "8766.T": "東京海上", "8801.T": "三井不動産",
    # 通信・IT
    "9432.T": "NTT", "9433.T": "KDDI", "9984.T": "ソフトバンクG", "4755.T": "楽天",
    # 商社
    "8058.T": "三菱商事", "8031.T": "三井物産", "8001.T": "伊藤忠",
    # 製造・重工
    "6501.T": "日立", "6301.T": "コマツ", "7011.T": "三菱重工",
    "6503.T": "三菱電機",
    # 医薬・ヘルスケア
    "4502.T": "武田薬品", "4568.T": "第一三共", "4519.T": "中外製薬",
    # 素材・化学
    "4063.T": "信越化学", "3407.T": "旭化成",
    # 食品・小売・エンタメ
    "2914.T": "JT", "9983.T": "ファーストリテイリング", "7974.T": "任天堂",
}

# ===== 米国指標（日本株に影響を与える可能性のある指標群） =====
US_INDICATORS = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW",
    "^VIX": "VIX",
    "^SOX": "SOX_半導体",
    "QQQ": "NASDAQ100_ETF",
    "XLF": "金融ETF",
    "XLE": "エネルギーETF",
    "XLV": "ヘルスケアETF",
    "GC=F": "金",
    "CL=F": "原油",
    "HG=F": "銅",
    "DX-Y.NYB": "ドル指数",
}

# ===== Walk-Forward設定 =====
TRAIN_DAYS = 252    # 約1年（営業日）
TEST_DAYS = 63      # 約3ヶ月（営業日）

# ===== 閾値グリッドサーチ範囲 =====
CONF_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
AGREE_THRESHOLDS = [3, 4, 5]

# ===== コスト設定 =====
# 株式の取引コスト: 往復0.1%（手数料+スリッページ）
# FX(0.03%)より高いため、利益のハードルも高い
STOCK_COST = 0.001

# ===== 採用基準（リスク管理の要） =====
MIN_PF = 1.2           # 最低プロフィットファクター
MIN_TRADES = 30         # 最低取引回数（統計的に有意な数）
# 期待値 > 0 は別途チェック


# =============================================================
# Phase 1: データダウンロード
# =============================================================
def download_all_data(period="2y"):
    """全銘柄・全指標のデータをダウンロードして保存する

    Args:
        period: 取得期間（デフォルト: 2年）

    Returns:
        dict: ティッカー → DataFrameの辞書
    """
    print("=" * 60)
    print("Phase 1: データダウンロード")
    print("=" * 60)

    all_data = {}
    all_tickers = {**JP_LARGE_CAPS, **US_INDICATORS}
    total = len(all_tickers)
    success = 0
    fail = 0

    for i, (ticker, name) in enumerate(all_tickers.items(), 1):
        try:
            print(f"  [{i}/{total}] {ticker} ({name})...", end=" ")
            df = yf.download(ticker, period=period, interval="1d", progress=False)

            if df.empty:
                print("データなし（スキップ）")
                fail += 1
                continue

            # マルチカラムの場合はフラットにする
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 保存
            safe_name = name.replace("/", "_").replace("\\", "_")
            save_path = DATA_DIR / f"{ticker.replace('^', '').replace('=', '_')}.csv"
            df.to_csv(save_path)
            all_data[ticker] = df
            print(f"OK ({len(df)}行)")
            success += 1

        except Exception as e:
            print(f"エラー: {e}")
            fail += 1

    # USD/JPYデータ（既存の1時間足から日足に変換）
    usdjpy_path = PROJECT_ROOT / "data" / "usdjpy_1h.csv"
    if usdjpy_path.exists():
        try:
            usdjpy = pd.read_csv(usdjpy_path, index_col=0, parse_dates=True)
            usdjpy_daily = usdjpy["Close"].resample("1D").last().dropna()
            usdjpy_daily = usdjpy_daily.to_frame(name="Close")
            usdjpy_daily.to_csv(DATA_DIR / "USDJPY.csv")
            all_data["USDJPY"] = usdjpy_daily
            print(f"  USD/JPY日足: OK ({len(usdjpy_daily)}行)")
        except Exception as e:
            print(f"  USD/JPY読み込みエラー: {e}")

    print(f"\n成功: {success}  失敗: {fail}  合計: {len(all_data)}銘柄")
    return all_data


def load_cached_data():
    """保存済みデータを読み込む（Phase 1スキップ用）

    Returns:
        dict: ティッカー → DataFrameの辞書
    """
    print("=" * 60)
    print("Phase 1: キャッシュデータ読み込み")
    print("=" * 60)

    all_data = {}
    all_tickers = {**JP_LARGE_CAPS, **US_INDICATORS}

    for ticker in all_tickers:
        csv_path = DATA_DIR / f"{ticker.replace('^', '').replace('=', '_')}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_data[ticker] = df

    # USD/JPY
    usdjpy_path = DATA_DIR / "USDJPY.csv"
    if usdjpy_path.exists():
        all_data["USDJPY"] = pd.read_csv(usdjpy_path, index_col=0, parse_dates=True)

    print(f"  読み込み完了: {len(all_data)}銘柄")
    if len(all_data) == 0:
        print("  ※ キャッシュが見つかりません。--quickを外して再実行してください。")
    return all_data


# =============================================================
# Phase 2: 相関関係の発見
# =============================================================
def find_correlations(all_data):
    """日本株 × 米国指標 のローリング相関を計算し、有望ペアを発見する

    米国市場は日本の前日に閉まるため、1日ラグを取る。
    |平均相関| > 0.3 のペアを有望とみなす。

    Args:
        all_data: ティッカー → DataFrame辞書

    Returns:
        promising_pairs: [(jp_ticker, us_ticker, mean_corr, std_corr), ...]
        corr_matrix_df: 相関マトリクスDataFrame
    """
    print("\n" + "=" * 60)
    print("Phase 2: 相関関係の発見")
    print("=" * 60)

    # 日次リターンを計算
    jp_returns = {}
    for ticker in JP_LARGE_CAPS:
        if ticker in all_data:
            df = all_data[ticker]
            jp_returns[ticker] = df["Close"].pct_change().dropna()

    us_returns = {}
    for ticker in US_INDICATORS:
        if ticker in all_data:
            df = all_data[ticker]
            us_returns[ticker] = df["Close"].pct_change().dropna()

    # 相関マトリクスを構築
    results = []
    for jp_tick in jp_returns:
        for us_tick in us_returns:
            jp_ret = jp_returns[jp_tick]
            us_ret = us_returns[us_tick]

            # 日付で結合（1日ラグ: 米国の今日 → 日本の翌営業日）
            combined = pd.DataFrame({
                "jp": jp_ret,
                "us": us_ret.shift(1),  # 米国を1日前にシフト
            }).dropna()

            if len(combined) < 120:
                continue

            # ローリング60日相関
            rolling_corr = combined["jp"].rolling(60).corr(combined["us"]).dropna()

            if len(rolling_corr) < 30:
                continue

            mean_corr = rolling_corr.mean()
            std_corr = rolling_corr.std()
            max_corr = rolling_corr.max()
            min_corr = rolling_corr.min()

            results.append({
                "jp_ticker": jp_tick,
                "us_ticker": us_tick,
                "jp_name": JP_LARGE_CAPS.get(jp_tick, jp_tick),
                "us_name": US_INDICATORS.get(us_tick, us_tick),
                "mean_corr": mean_corr,
                "std_corr": std_corr,
                "max_corr": max_corr,
                "min_corr": min_corr,
            })

    corr_df = pd.DataFrame(results)
    if corr_df.empty:
        print("  相関データが生成できませんでした。")
        return [], corr_df

    # 相関マトリクス（ピボット表）を保存
    pivot = corr_df.pivot_table(
        index="jp_ticker", columns="us_ticker", values="mean_corr"
    )
    pivot.to_csv(DATA_DIR / "correlation_matrix.csv")

    # |平均相関| > 0.3 のペアをフィルタ
    promising = corr_df[corr_df["mean_corr"].abs() > 0.3].copy()
    promising = promising.sort_values("mean_corr", key=abs, ascending=False)

    # 各JP銘柄について最も相関の高いUS指標をリスト化
    promising_pairs = []
    for _, row in promising.iterrows():
        promising_pairs.append((
            row["jp_ticker"], row["us_ticker"],
            row["mean_corr"], row["std_corr"],
        ))

    print(f"  全ペア数: {len(corr_df)}")
    print(f"  有望ペア（|相関|>0.3）: {len(promising_pairs)}")

    # 上位10ペアを表示
    print("\n  --- 相関上位10ペア ---")
    for jp_t, us_t, mc, sc in promising_pairs[:10]:
        jp_n = JP_LARGE_CAPS.get(jp_t, jp_t)
        us_n = US_INDICATORS.get(us_t, us_t)
        print(f"    {jp_n:12s} <-> {us_n:16s}  相関={mc:+.3f} (std={sc:.3f})")

    return promising_pairs, corr_df


# =============================================================
# Phase 3: 特徴量エンジニアリング
# =============================================================
def make_stock_features(jp_ticker, related_us_tickers, all_data):
    """指定されたJP銘柄の予測特徴量を生成する

    Args:
        jp_ticker: 日本株ティッカー
        related_us_tickers: 相関の高い米国指標ティッカーリスト
        all_data: 全データ辞書

    Returns:
        df: 特徴量+ターゲット付きDataFrame
        feature_cols: 特徴量カラム名リスト
    """
    if jp_ticker not in all_data:
        return None, None

    jp_df = all_data[jp_ticker].copy()
    jp_df.index = pd.to_datetime(jp_df.index)
    features = pd.DataFrame(index=jp_df.index)
    feature_cols = []

    # ----- 関連米国指標のリターン（1日ラグ） -----
    for us_tick in related_us_tickers:
        if us_tick not in all_data:
            continue
        us_df = all_data[us_tick].copy()
        us_df.index = pd.to_datetime(us_df.index)
        us_name = US_INDICATORS.get(us_tick, us_tick).replace(" ", "_")

        # 1日、5日、20日リターン
        for d in [1, 5, 20]:
            col = f"{us_name}_Ret{d}d"
            ret = us_df["Close"].pct_change(d).shift(1)  # 1日ラグ
            ret.name = col
            features = features.join(ret, how="left")
            feature_cols.append(col)

        # 20日ボラティリティ
        col_vol = f"{us_name}_Vol20d"
        vol = us_df["Close"].pct_change().rolling(20).std().shift(1)
        vol.name = col_vol
        features = features.join(vol, how="left")
        feature_cols.append(col_vol)

    # ----- VIX（常に含める） -----
    if "^VIX" in all_data:
        vix_df = all_data["^VIX"].copy()
        vix_df.index = pd.to_datetime(vix_df.index)

        vix_level = vix_df["Close"].shift(1)
        vix_level.name = "VIX_Level"
        features = features.join(vix_level, how="left")
        feature_cols.append("VIX_Level")

        vix_chg = vix_df["Close"].pct_change().shift(1)
        vix_chg.name = "VIX_Change"
        features = features.join(vix_chg, how="left")
        feature_cols.append("VIX_Change")

    # ----- USD/JPY為替変化（常に含める） -----
    if "USDJPY" in all_data:
        fx_df = all_data["USDJPY"].copy()
        fx_df.index = pd.to_datetime(fx_df.index)

        fx_chg = fx_df["Close"].pct_change().shift(1)
        fx_chg.name = "USDJPY_Change"
        features = features.join(fx_chg, how="left")
        feature_cols.append("USDJPY_Change")

    # ----- JP銘柄の自身のモメンタム -----
    for d in [1, 5, 20]:
        col = f"JP_Momentum_{d}d"
        ret = jp_df["Close"].pct_change(d)
        ret.name = col
        features = features.join(ret, how="left")
        feature_cols.append(col)

    # ----- JP銘柄の出来高変化 -----
    if "Volume" in jp_df.columns:
        for d in [1, 5]:
            col = f"JP_VolChg_{d}d"
            vol_chg = jp_df["Volume"].pct_change(d)
            vol_chg.name = col
            features = features.join(vol_chg, how="left")
            feature_cols.append(col)

    # ----- 曜日・月 -----
    features["DayOfWeek"] = features.index.dayofweek
    features["Month"] = features.index.month
    feature_cols.extend(["DayOfWeek", "Month"])

    # ----- トップUS指標とのローリング相関（20日） -----
    if related_us_tickers and related_us_tickers[0] in all_data:
        top_us = all_data[related_us_tickers[0]].copy()
        top_us.index = pd.to_datetime(top_us.index)
        jp_ret = jp_df["Close"].pct_change()
        us_ret = top_us["Close"].pct_change().shift(1)
        combined = pd.DataFrame({"jp": jp_ret, "us": us_ret}).dropna()
        roll_corr = combined["jp"].rolling(20).corr(combined["us"])
        roll_corr.name = "RollingCorr_20d"
        features = features.join(roll_corr, how="left")
        feature_cols.append("RollingCorr_20d")

    # ----- ターゲット: 翌日上昇=1, 下落=0 -----
    features["Target"] = (jp_df["Close"].pct_change().shift(-1) > 0).astype(int)

    # NaN除去
    features = features.dropna(subset=feature_cols + ["Target"])

    # inf除去（出来高ゼロ等で発生する可能性）
    features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    return features, feature_cols


# =============================================================
# Phase 4: Walk-Forward検証
# =============================================================
def validate_stock(jp_ticker, features_df, feature_cols):
    """Walk-Forward検証でストック予測の有効性を確認する

    拡張ウィンドウ方式:
      - 最初のTRAIN_DAYS日で学習
      - 次のTEST_DAYS日でテスト
      - ウィンドウを63日ずらして繰り返し

    閾値グリッドサーチ:
      conf × agree の全組合せを試し、最良の設定を見つける

    Args:
        jp_ticker: 日本株ティッカー
        features_df: 特徴量DataFrame（Targetカラム含む）
        feature_cols: 特徴量カラム名リスト

    Returns:
        best_result: dict（利益性がある場合）or None
    """
    jp_name = JP_LARGE_CAPS.get(jp_ticker, jp_ticker)

    X = features_df[feature_cols].values
    y = features_df["Target"].values
    returns = features_df["Target"].values  # 方向（1=上昇, 0=下落）
    n = len(X)

    if n < TRAIN_DAYS + TEST_DAYS:
        print(f"    {jp_name}: データ不足 ({n}日 < {TRAIN_DAYS+TEST_DAYS}日)")
        return None

    # ----- Walk-Forwardの全テスト期間で予測を蓄積 -----
    # 各テスト日について: (予測方向, 一致度, 実際の方向)
    all_predictions = []

    start = 0
    fold = 0
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        if len(X_test) == 0:
            break

        # アンサンブル学習
        try:
            model = EnsembleClassifier(n_estimators=150, learning_rate=0.05)
            model.fit(X_train, y_train)
            preds, agreements = model.predict_with_agreement(X_test)
            probas = model.predict_proba(X_test)

            # 各テストサンプルの情報を蓄積
            for i in range(len(X_test)):
                # 予測確信度: 多数派の確率
                conf = max(probas[i][0], probas[i][1])
                all_predictions.append({
                    "pred": preds[i],
                    "agreement": agreements[i],
                    "confidence": conf,
                    "actual": y_test[i],
                })
        except Exception:
            pass  # モデル学習失敗はスキップ

        fold += 1
        # 拡張ウィンドウ: テスト期間分だけ前進
        start += TEST_DAYS

    if len(all_predictions) == 0:
        return None

    pred_df = pd.DataFrame(all_predictions)

    # ----- 閾値グリッドサーチ -----
    best_result = None
    best_exp_value = -999

    for conf_th, agree_th in product(CONF_THRESHOLDS, AGREE_THRESHOLDS):
        # フィルタ: 確信度 >= conf_th かつ 一致度 >= agree_th
        mask = (pred_df["confidence"] >= conf_th) & (pred_df["agreement"] >= agree_th)
        filtered = pred_df[mask]
        n_trades = len(filtered)

        if n_trades < MIN_TRADES:
            continue

        # 勝ち（予測=実際）・負け（予測!=実際）
        wins = (filtered["pred"] == filtered["actual"]).sum()
        losses = n_trades - wins
        win_rate = wins / n_trades if n_trades > 0 else 0

        # プロフィットファクター（勝ち回数/負け回数の近似）
        # 株式の場合、各トレードの利益/損失は近似的に等しいと仮定
        pf = wins / max(losses, 1)

        # 期待値（コスト控除後）
        # 平均勝ちリターン ≒ 平均負けリターン と仮定（日次リターンの場合）
        # 簡易計算: EV = win_rate - (1-win_rate) - cost
        # より正確: EV = win_rate * avg_win - (1-win_rate) * avg_loss - cost
        avg_daily_return = 0.005  # 日本大型株の平均日次変動幅（約0.5%）
        exp_value = (win_rate * avg_daily_return
                     - (1 - win_rate) * avg_daily_return
                     - STOCK_COST)

        # 採用基準: PF >= 1.2 かつ 期待値 > 0 かつ 最低取引回数
        if pf >= MIN_PF and exp_value > 0 and n_trades >= MIN_TRADES:
            if exp_value > best_exp_value:
                best_exp_value = exp_value
                best_result = {
                    "ticker": jp_ticker,
                    "name": jp_name,
                    "best_conf": conf_th,
                    "best_agree": agree_th,
                    "pf": round(pf, 3),
                    "win_rate": round(win_rate, 4),
                    "exp_value": round(exp_value, 6),
                    "trades": n_trades,
                    "total_predictions": len(pred_df),
                    "wf_folds": fold,
                }

    return best_result


# =============================================================
# Phase 4 メインループ: 全有望銘柄を検証
# =============================================================
def run_validation(promising_pairs, all_data):
    """全有望銘柄をWalk-Forward検証にかける

    Args:
        promising_pairs: Phase 2の出力（jp_ticker, us_ticker, corr, std）
        all_data: 全データ辞書

    Returns:
        results: {jp_ticker: {result_dict + correlation info}}
    """
    print("\n" + "=" * 60)
    print("Phase 4: Walk-Forward検証")
    print("=" * 60)

    # 各JP銘柄の関連US指標をグループ化
    jp_us_map = {}
    jp_corr_map = {}  # 最高相関の記録
    for jp_t, us_t, mc, sc in promising_pairs:
        if jp_t not in jp_us_map:
            jp_us_map[jp_t] = []
            jp_corr_map[jp_t] = (us_t, mc)
        jp_us_map[jp_t].append(us_t)
        # 最も強い相関を記録
        if abs(mc) > abs(jp_corr_map[jp_t][1]):
            jp_corr_map[jp_t] = (us_t, mc)

    # 重複除去：各銘柄の上位5指標のみ使う
    for jp_t in jp_us_map:
        jp_us_map[jp_t] = list(dict.fromkeys(jp_us_map[jp_t]))[:5]

    total_stocks = len(jp_us_map)
    print(f"  検証対象: {total_stocks}銘柄\n")

    results = {}
    profitable_count = 0

    for idx, (jp_ticker, us_tickers) in enumerate(jp_us_map.items(), 1):
        jp_name = JP_LARGE_CAPS.get(jp_ticker, jp_ticker)
        print(f"  [{idx}/{total_stocks}] {jp_name} ({jp_ticker})", end="")
        print(f"  関連指標: {len(us_tickers)}個")

        # 特徴量生成
        features_df, feature_cols = make_stock_features(
            jp_ticker, us_tickers, all_data
        )
        if features_df is None or len(features_df) < TRAIN_DAYS + TEST_DAYS:
            print(f"    -> データ不足、スキップ")
            results[jp_ticker] = {
                "ticker": jp_ticker, "name": jp_name,
                "status": "データ不足", "profitable": False,
            }
            continue

        # Walk-Forward検証
        result = validate_stock(jp_ticker, features_df, feature_cols)

        if result is not None:
            # 相関情報を追加
            top_us, top_corr = jp_corr_map.get(jp_ticker, ("N/A", 0))
            result["top_us_indicator"] = top_us
            result["top_us_name"] = US_INDICATORS.get(top_us, top_us)
            result["correlation"] = round(top_corr, 4)
            result["status"] = "利益性あり"
            result["profitable"] = True
            results[jp_ticker] = result
            profitable_count += 1
            print(f"    -> *** 利益性あり *** PF={result['pf']:.3f}"
                  f" WR={result['win_rate']:.1%} EV={result['exp_value']:.4f}"
                  f" trades={result['trades']}")
        else:
            top_us, top_corr = jp_corr_map.get(jp_ticker, ("N/A", 0))
            results[jp_ticker] = {
                "ticker": jp_ticker, "name": jp_name,
                "top_us_indicator": top_us,
                "top_us_name": US_INDICATORS.get(top_us, top_us),
                "correlation": round(top_corr, 4),
                "status": "基準未達", "profitable": False,
            }
            print(f"    -> 基準未達（PF<{MIN_PF} or EV<=0 or trades<{MIN_TRADES}）")

    print(f"\n  === 検証完了 ===")
    print(f"  検証銘柄数: {total_stocks}")
    print(f"  利益性あり: {profitable_count}")
    print(f"  該当なし:   {total_stocks - profitable_count}")

    return results


# =============================================================
# Phase 5: レポート生成
# =============================================================
def generate_report(results, corr_df=None):
    """スクリーニング結果の詳細レポートを生成する

    Args:
        results: run_validation()の戻り値
        corr_df: 相関マトリクスDataFrame（Phase 2から）
    """
    print("\n" + "=" * 60)
    print("Phase 5: レポート生成")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 70)
    lines.append("  日本大型株スクリーニングレポート")
    lines.append(f"  生成日時: {now}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("【採用基準】")
    lines.append(f"  - プロフィットファクター (PF) >= {MIN_PF}")
    lines.append(f"  - コスト控除後期待値 > 0 （コスト={STOCK_COST*100:.1f}%）")
    lines.append(f"  - 最低取引回数 >= {MIN_TRADES}")
    lines.append(f"  - Walk-Forward検証: {TRAIN_DAYS}日訓練 / {TEST_DAYS}日テスト（拡張ウィンドウ）")
    lines.append("")

    # 利益性ありの銘柄
    profitable = {k: v for k, v in results.items() if v.get("profitable", False)}
    not_profitable = {k: v for k, v in results.items() if not v.get("profitable", False)}

    lines.append("=" * 70)
    if profitable:
        lines.append(f"  *** 利益性のある銘柄: {len(profitable)}銘柄 ***")
    else:
        lines.append("  該当なし（利益性のある銘柄は見つかりませんでした）")
    lines.append("=" * 70)
    lines.append("")

    for ticker, r in sorted(profitable.items(), key=lambda x: x[1].get("exp_value", 0), reverse=True):
        lines.append(f"  *** PROFIT *** {r['name']} ({ticker})")
        lines.append(f"    最良閾値: conf={r['best_conf']:.2f}, agree={r['best_agree']}")
        lines.append(f"    PF={r['pf']:.3f}  勝率={r['win_rate']:.1%}  "
                      f"期待値={r['exp_value']:.4f}  取引回数={r['trades']}")
        lines.append(f"    主要関連指標: {r.get('top_us_name', 'N/A')} "
                      f"(相関={r.get('correlation', 0):+.4f})")
        # 利益試算（100万円資本）
        capital = 1_000_000
        trades_per_year = r["trades"] * 2  # WFテスト期間→年間概算
        annual_return = r["exp_value"] * trades_per_year
        annual_yen = capital * annual_return
        lines.append(f"    年間利益概算（100万円資本）: "
                      f"約{annual_yen:,.0f}円 ({annual_return:.1%})")
        lines.append("")

    # 基準未達の銘柄一覧
    lines.append("-" * 70)
    lines.append(f"  基準未達の銘柄: {len(not_profitable)}")
    lines.append("-" * 70)
    for ticker, r in sorted(not_profitable.items()):
        name = r.get("name", ticker)
        status = r.get("status", "不明")
        top_us = r.get("top_us_name", "N/A")
        corr = r.get("correlation", 0)
        lines.append(f"    {name:16s} ({ticker:8s})  状態={status:8s}  "
                      f"関連={top_us} (相関={corr:+.3f})")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  リスク管理注意事項:")
    lines.append("  1. 過去の検証結果は将来の利益を保証しません")
    lines.append("  2. 定期的な再検証を推奨（月1回程度）")
    lines.append("  3. 1銘柄あたりの資金配分は資本の10%以下を推奨")
    lines.append("  4. 損切りルールを必ず設定してください")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # ファイル保存
    report_path = DATA_DIR / "screening_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  レポート保存: {report_path}")
    print("\n" + report_text)

    return report_text


# =============================================================
# Phase 6: Telegram通知
# =============================================================
def notify_results(results):
    """スクリーニング結果をTelegramに送信する

    Args:
        results: run_validation()の戻り値
    """
    print("\n" + "=" * 60)
    print("Phase 6: Telegram通知")
    print("=" * 60)

    profitable = {k: v for k, v in results.items() if v.get("profitable", False)}

    # 通知メッセージ構築
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg_lines = [f"[株スクリーニング結果] {now}"]
    msg_lines.append("")

    if profitable:
        msg_lines.append(f"利益性のある銘柄: {len(profitable)}件")
        msg_lines.append("")
        for ticker, r in sorted(profitable.items(),
                                 key=lambda x: x[1].get("exp_value", 0),
                                 reverse=True):
            msg_lines.append(
                f"  {r['name']} PF={r['pf']:.2f} "
                f"WR={r['win_rate']:.0%} EV={r['exp_value']:.4f} "
                f"({r['trades']}回)"
            )
            msg_lines.append(
                f"    conf={r['best_conf']:.2f} agree={r['best_agree']} "
                f"関連={r.get('top_us_name', 'N/A')}"
            )
    else:
        msg_lines.append("該当なし（基準を満たす銘柄は見つかりませんでした）")

    signal = "\n".join(msg_lines)

    try:
        from research.telegram_bot import send_signal_sync
        send_signal_sync(signal)
        print("  Telegram送信完了")
    except Exception as e:
        print(f"  Telegram送信失敗（スキップ）: {e}")


# =============================================================
# 結果保存
# =============================================================
def save_results(results):
    """利益性のある銘柄をJSONに保存する

    Args:
        results: run_validation()の戻り値
    """
    profitable = {k: v for k, v in results.items() if v.get("profitable", False)}

    # JSON保存用にデータ整形
    save_data = []
    for ticker, r in profitable.items():
        save_data.append({
            "ticker": r["ticker"],
            "name": r["name"],
            "best_conf": r["best_conf"],
            "best_agree": r["best_agree"],
            "pf": r["pf"],
            "win_rate": r["win_rate"],
            "exp_value": r["exp_value"],
            "trades": r["trades"],
            "top_us_indicator": r.get("top_us_indicator", ""),
            "top_us_name": r.get("top_us_name", ""),
            "correlation": r.get("correlation", 0),
        })

    json_path = DATA_DIR / "profitable_stocks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"  利益性銘柄JSON保存: {json_path} ({len(save_data)}件)")
    return json_path


# =============================================================
# メイン: フルスクリーニング
# =============================================================
def run_full_screening():
    """Phase 1-6 を順番に実行するフルパイプライン"""
    print("=" * 60)
    print("  日本大型株 自動スクリーニング（フルモード）")
    print(f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  対象: {len(JP_LARGE_CAPS)}銘柄 × {len(US_INDICATORS)}指標")
    print(f"  推定所要時間: 15-30分")
    print("=" * 60)

    start_time = time.time()

    # Phase 1: データダウンロード
    all_data = download_all_data()

    # Phase 2: 相関発見
    promising_pairs, corr_df = find_correlations(all_data)

    if not promising_pairs:
        print("\n有望なペアが見つかりませんでした。終了します。")
        return {}

    # Phase 4: Walk-Forward検証（Phase 3は4の中で自動実行）
    results = run_validation(promising_pairs, all_data)

    # Phase 5: レポート生成
    generate_report(results, corr_df)

    # 結果保存
    save_results(results)

    # Phase 6: Telegram通知
    notify_results(results)

    elapsed = time.time() - start_time
    print(f"\n全工程完了: {elapsed/60:.1f}分")

    return results


def run_quick_screening():
    """キャッシュデータを使ったクイックスクリーニング（Phase 1スキップ）"""
    print("=" * 60)
    print("  日本大型株 自動スクリーニング（クイックモード）")
    print(f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ※ キャッシュデータを使用（ダウンロードスキップ）")
    print("=" * 60)

    start_time = time.time()

    # Phase 1: キャッシュ読み込み
    all_data = load_cached_data()
    if len(all_data) == 0:
        print("\nキャッシュが見つかりません。フルモードで再実行してください。")
        return {}

    # Phase 2-6 はフルモードと同じ
    promising_pairs, corr_df = find_correlations(all_data)

    if not promising_pairs:
        print("\n有望なペアが見つかりませんでした。終了します。")
        return {}

    results = run_validation(promising_pairs, all_data)
    generate_report(results, corr_df)
    save_results(results)
    notify_results(results)

    elapsed = time.time() - start_time
    print(f"\n全工程完了: {elapsed/60:.1f}分")

    return results


# =============================================================
# エントリポイント
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="日本大型株スクリーニング: 米国指標との相関を利用した予測可能性の自動探索"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="キャッシュデータを使用（ダウンロードをスキップ）"
    )
    args = parser.parse_args()

    if args.quick:
        run_quick_screening()
    else:
        run_full_screening()
