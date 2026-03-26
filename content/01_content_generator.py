# ===========================================
# 01_content_generator.py
# FX AIシステムの学習結果を Claude API で記事に変換する
# 実行: python content/01_content_generator.py
# ===========================================

import os
import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# .env の読み込み（ANTHROPIC_API_KEY 用）
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import joblib
import numpy as np

# 免責事項テンプレート（全記事に必ず含める）
DISCLAIMER = """本記事は教育・記録目的のコンテンツです。
投資助言・売買推奨ではありません。
掲載されている検証結果は過去データに
基づくものであり、将来の利益を保証する
ものではありません。"""


def load_dashboard_state():
    """data/dashboard_state.joblib からダッシュボード状態を読み込む"""
    state_path = project_root / "data" / "dashboard_state.joblib"
    if not state_path.exists():
        raise FileNotFoundError(
            "data/dashboard_state.joblib が見つかりません。"
            "先に research/14_main_system.py を実行してください。"
        )
    return joblib.load(state_path)


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
            acc_series.append(float(np.mean(pred_v == actual_v)))
        else:
            acc_series.append(np.nan)
    return acc_series


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


def summarize_state(state):
    """
    ダッシュボード状態から記事生成用の要約テキストを組み立てる。
    直近の正解率推移・レジーム分布・累積リターン・直近10件シグナルを含める。
    """
    df = state["df"]
    signals = state["signals"]
    preds = state["preds"]
    y_test = state["y_test"]
    ret4_test = state["ret4_test"]
    split_idx = state["split_idx"]

    # 直近の正解率推移（直近100件窓の正解率の概要）
    acc_series = build_rolling_accuracy_series(preds, y_test, window=100)
    acc_valid = [a for a in acc_series if not np.isnan(a)]
    if acc_valid:
        acc_recent = acc_valid[-50:]  # 直近50ポイント
        acc_summary = (
            f"直近の正解率推移（100件窓）: ステップ数 {len(acc_series)} 件。"
            f"直近付近の正解率はおおむね {min(acc_recent):.2%} ～ {max(acc_recent):.2%} の範囲。"
        )
    else:
        acc_summary = "直近の正解率推移: 有効な予測が少ないため要約なし。"

    # レジーム分布（テスト期間）
    df_test = df.iloc[split_idx:]
    regime_names = ["トレンド", "レンジ", "高ボラ"]
    regime_counts = df_test["Regime"].value_counts().sort_index()
    regime_parts = []
    for k in regime_counts.index:
        name = regime_names[int(k)] if k in (0, 1, 2) else f"レジーム{k}"
        pct = 100 * regime_counts[k] / len(df_test)
        regime_parts.append(f"{name}: {pct:.1f}%")
    regime_summary = "レジーム分布（テスト期間）: " + ", ".join(regime_parts)

    # 累積リターン
    cum_series = build_cumulative_return_series(signals, ret4_test)
    if cum_series:
        cum_final = cum_series[-1]
        cum_summary = f"累積リターン: テスト期間を通した累積リターンは {cum_final:.4f}（数値は記事にそのまま使わず、傾向のみ参照すること）。"
    else:
        cum_summary = "累積リターン: データなし。"

    # 直近10件のシグナル（表形式のテキスト）
    recent = signals[-10:] if len(signals) >= 10 else signals
    signal_lines = []
    for s in recent:
        ts = s.get("timestamp", "-")
        regime = s.get("regime", "-")
        mode = s.get("mode", "-")
        direction = s.get("direction", "-")
        pos = s.get("position_size", 0)
        conf = s.get("confidence", 0)
        signal_lines.append(f"  {ts} | レジーム:{regime} モード:{mode} 方向:{direction} ポジション:{pos} 信頼度:{conf}")
    signals_summary = "直近10件のシグナル:\n" + "\n".join(signal_lines) if signal_lines else "直近のシグナル: なし。"

    return (
        f"{acc_summary}\n\n"
        f"{regime_summary}\n\n"
        f"{cum_summary}\n\n"
        f"{signals_summary}"
    )


def call_claude(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Claude API を呼び出し、生成テキストを返す"""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # content は list of content blocks
    if message.content and len(message.content) > 0:
        return message.content[0].text
    return ""


def generate_x_post(data_summary: str, api_key: str) -> str:
    """X（Twitter）投稿用の短文（140文字以内）を生成"""
    system = (
        "あなたはFX・AIシステムの運営者向けに、SNS投稿文を書くライターです。"
        "核心的な数値は含めず、興味を引く表現にしてください。"
        "140文字以内で、AIの今週の成績を報告する投稿にしてください。"
        "ハッシュタグは必要に応じて1〜2個まで。"
    )
    user = (
        "以下の学習結果の要約を踏まえ、X（旧Twitter）用の投稿文を1本だけ書いてください。"
        "140文字以内に収め、数値は具体的に書かず、興味を引く言い回しにしてください。\n\n"
        "【要約】\n" + data_summary
    )
    body = call_claude(system, user, api_key).strip()
    return body + "\n\n" + DISCLAIMER


def generate_note_article(data_summary: str, api_key: str) -> str:
    """note用の長文（タイトル・無料部分・有料予告・免責）を生成"""
    system = (
        "あなたはFX・AIシステムの検証結果を記事にするライターです。"
        "note用の長文を作成します。"
        "構成: タイトル、無料部分（概要・気づき）、有料部分の予告（「詳細は有料記事で」と明記）。"
        "投資の勧誘や具体的な数値の誇張は避けてください。"
    )
    user = (
        "以下の学習結果の要約を踏まえ、note用の記事を書いてください。\n"
        "必ず次の構成にしてください。\n"
        "1. タイトル（1行）\n"
        "2. 無料部分: 概要と気づき（数段落）\n"
        "3. 有料部分の予告: 「詳細な数値・シグナル一覧・今後の方針は有料記事で」と明記した1〜2段落\n"
        "免責事項は別途付与するので、本文には含めないでください。\n\n"
        "【要約】\n" + data_summary
    )
    body = call_claude(system, user, api_key).strip()
    return body + "\n\n" + DISCLAIMER


def generate_substack_article(data_summary: str, api_key: str) -> str:
    """Substack用のメール形式（件名・本文・免責）を生成"""
    system = (
        "あなたはFX・AIシステムの検証結果をメールマガジン用に書くライターです。"
        "Substack用のメール形式で、件名と本文を出力してください。"
        "noteより少し詳しく、読者が続きを読みたくなるように書いてください。"
        "投資の勧誘は避けてください。"
    )
    user = (
        "以下の学習結果の要約を踏まえ、Substack用のメールを書いてください。\n"
        "次の形式で出力してください。\n"
        "【件名】\n（1行の件名）\n\n"
        "【本文】\n（数段落の本文。noteよりやや詳しめ）\n"
        "免責事項は別途付与するので、本文には含めないでください。\n\n"
        "【要約】\n" + data_summary
    )
    body = call_claude(system, user, api_key).strip()
    return body + "\n\n" + DISCLAIMER


def save_generated(out_dir: Path, date_prefix: str, kind: str, content: str) -> Path:
    """生成テキストを content/generated/YYYY-MM-DD_[種類].txt で保存"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date_prefix}_{kind}.txt"
    path.write_text(content, encoding="utf-8")
    return path


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key.strip() == "your_anthropic_key_here":
        print("エラー: .env に ANTHROPIC_API_KEY を設定してください。")
        sys.exit(1)

    # 入力データの読み込み
    state = load_dashboard_state()
    data_summary = summarize_state(state)

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    out_dir = script_dir / "generated"

    print("記事を生成しています...")

    # ① X投稿用（140文字以内）
    x_content = generate_x_post(data_summary, api_key)
    save_generated(out_dir, date_prefix, "x", x_content)
    print(f"  保存: {date_prefix}_x.txt")

    # ② note用（長文・下書き）
    note_content = generate_note_article(data_summary, api_key)
    save_generated(out_dir, date_prefix, "note", note_content)
    print(f"  保存: {date_prefix}_note.txt")

    # ③ Substack用（メール形式）
    substack_content = generate_substack_article(data_summary, api_key)
    save_generated(out_dir, date_prefix, "substack", substack_content)
    print(f"  保存: {date_prefix}_substack.txt")

    print("完了。content/generated/ を確認してください。")


if __name__ == "__main__":
    main()
