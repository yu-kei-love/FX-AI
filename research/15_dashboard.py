# ===========================================
# 15_dashboard.py
# Streamlit でシステム状態をリアルタイム表示するダッシュボード
# 実行: streamlit run research/15_dashboard.py
# 必要: pip install streamlit plotly
# 事前に research/14_main_system.py を実行し data/dashboard_state.joblib を生成すること
# ===========================================

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# 日本語フォント・表示用
st.set_page_config(page_title="FX-AI ダッシュボード", layout="wide", initial_sidebar_state="auto")

# 状態ファイルのパス（data/dashboard_state.joblib）
script_dir = Path(__file__).resolve().parent
state_path = script_dir.parent / "data" / "dashboard_state.joblib"

def load_state():
    """14_main_system.py が保存したダッシュボード用状態を読み込む"""
    if not state_path.exists():
        return None
    import joblib
    return joblib.load(state_path)

def build_cumulative_return_series(signals, ret4_test):
    """累積リターン推移を計算（時系列リストを返す）"""
    series = []
    cum = 0.0
    for i in range(min(len(signals), len(ret4_test))):
        if i >= len(signals):
            break
        s = signals[i]
        pos = s.get("position_size", 0) or 0
        dr = s.get("direction", "なし")
        if pos == 0 or dr == "なし":
            series.append(cum)
            continue
        direction_mult = 1.0 if dr == "買い" else -1.0
        cum += ret4_test[i] * direction_mult * pos
        series.append(cum)
    return series

def build_rolling_accuracy_series(preds, y_test, window=100):
    """直近 window 件の正解率推移を計算（有効予測のみ）"""
    n = len(preds)
    y_test = np.asarray(y_test)
    acc_series = []
    for i in range(n):
        start = max(0, i - window + 1)
        valid = []
        for j in range(start, i + 1):
            if j < len(preds) and isinstance(preds[j], (int, float, np.integer)) and preds[j] in (0, 1):
                valid.append((int(preds[j]), y_test[j]))
        if len(valid) >= 1:
            pred_v = np.array([v[0] for v in valid])
            actual_v = np.array([v[1] for v in valid])
            acc_series.append((i, np.mean(pred_v == actual_v)))
        else:
            acc_series.append((i, np.nan))
    return acc_series

def main():
    state = load_state()

    # ----- ① ヘッダー -----
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.title("FX-AI システムダッシュボード")
    with col2:
        st.metric("現在時刻", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with col3:
        status = "稼働中" if state else "停止中"
        st.metric("システム状態", status)

    if state is None:
        st.warning("状態ファイルが見つかりません。先に `research/14_main_system.py` を実行してからダッシュボードを開いてください。")
        st.info("30秒後に再読み込みします…")
        time.sleep(30)
        st.rerun()
        return

    df = state["df"]
    signals = state["signals"]
    preds = state["preds"]
    y_test = state["y_test"]
    ret4_test = state["ret4_test"]
    split_idx = state["split_idx"]
    n_test = state["n_test"]

    # ----- ② 現在のシグナル（直近1件） -----
    st.subheader("現在のシグナル")
    if signals:
        s = signals[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("現在のレジーム", s.get("regime", "-"))
        c2.metric("現在のモード", s.get("mode", "-"))
        c3.metric("方向", s.get("direction", "-"))
        c4.metric("ポジションサイズ", s.get("position_size", 0))
        c5.metric("信頼度", s.get("confidence", 0))
        reasons = s.get("reason", [])
        if isinstance(reasons, list):
            st.caption("理由: " + " / ".join(reasons))
        else:
            st.caption("理由: " + str(reasons))
    else:
        st.info("シグナルがありません。")

    # ----- ③ パフォーマンスグラフ -----
    st.subheader("パフォーマンスグラフ")

    cum_series = build_cumulative_return_series(signals, ret4_test)
    if cum_series:
        # 累積リターン推移（x はテスト期間のインデックス）
        x_idx = list(range(len(cum_series)))
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=x_idx, y=cum_series, mode="lines", name="累積リターン"))
        fig_cum.update_layout(title="累積リターンの推移", xaxis_title="ステップ", yaxis_title="累積リターン", height=300)
        st.plotly_chart(fig_cum, use_container_width=True)

    acc_series = build_rolling_accuracy_series(preds, y_test, window=100)
    if acc_series:
        x_acc = [a[0] for a in acc_series]
        y_acc = [a[1] for a in acc_series]
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=x_acc, y=y_acc, mode="lines", name="直近100件正解率"))
        fig_acc.update_layout(title="直近100件の正解率推移", xaxis_title="ステップ", yaxis_title="正解率", height=300)
        st.plotly_chart(fig_acc, use_container_width=True)

    # ----- ④ レジーム分布（直近のレジーム割合・円グラフ） -----
    st.subheader("レジーム分布")
    regime_names = ["トレンド", "レンジ", "高ボラ"]
    # テスト期間の直近分のレジームを使用
    df_test = df.iloc[split_idx:]
    if len(df_test) > 0:
        regime_counts = df_test["Regime"].value_counts().sort_index()
        labels = [regime_names[int(k)] if k in (0, 1, 2) else f"レジーム{k}" for k in regime_counts.index]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=regime_counts.values, hole=0.4)])
        fig_pie.update_layout(title="直近のレジーム割合", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("レジームデータがありません。")

    # ----- ⑤ システムログ（直近10件のシグナル履歴） -----
    st.subheader("システムログ（直近10件のシグナル履歴）")
    recent = signals[-10:] if len(signals) >= 10 else signals
    if recent:
        log_data = []
        for s in reversed(recent):
            log_data.append({
                "日時": s.get("timestamp", "-"),
                "レジーム": s.get("regime", "-"),
                "モード": s.get("mode", "-"),
                "方向": s.get("direction", "-"),
                "ポジション": s.get("position_size", 0),
                "信頼度": s.get("confidence", 0),
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
    else:
        st.info("シグナル履歴がありません。")

    # 30秒ごとに自動更新
    st.caption("最終更新: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " — 30秒後に自動更新します。")
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()
