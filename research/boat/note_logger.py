# ===========================================
# note_logger.py
# ボートレースAI - noteコンテンツ自動記録
#
# 設計思想：
#   開発と並行してnoteのネタを自動で蓄積する。
#   「気づいたときに書く」では続かないので
#   システムが自動で記録→記事を書くときに振り返るだけ。
#
# 使い方:
#   from research.boat.note_logger import log_note_content, NoteCategory
#
#   log_note_content(
#       title="...",
#       category=NoteCategory.FAILURE,
#       what_happened="...",
#       why_it_matters="...",
#       priority="high"
#   )
# ===========================================

import json
from enum import Enum
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTE_LOG_FILE = PROJECT_ROOT / "data" / "boat" / "note_content_log.json"
NOTE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class NoteCategory(Enum):
    """noteのネタカテゴリ（売れやすい順）"""
    FAILURE     = "失敗談"          # 最も売れる：共感が高い
    MONEY       = "収益・損益記録"   # 2位：数字の公開が注目される
    DISCOVERY   = "意外な発見"       # 3位：専門性が伝わる
    IMPROVEMENT = "精度改善"         # 4位：改善前後の数字で示す
    DESIGN      = "設計の意思決定"   # 5位：技術者向けに刺さる
    DATA        = "データの特徴"     # 分析系読者向け
    MILESTONE   = "マイルストーン"   # 進捗報告


def log_note_content(
    title,
    category,
    what_happened,
    why_it_matters,
    what_changed=None,
    metrics_before=None,
    metrics_after=None,
    code_snippet=None,
    priority="medium",
    status="未執筆",
):
    """
    noteのネタを自動記録する。

    Parameters:
        title          : 記事タイトル案
        category       : NoteCategory
        what_happened  : 何が起きたか（1〜3行）
        why_it_matters : なぜ重要か（読者への価値）
        what_changed   : 何を変えたか
        metrics_before : 変更前の指標 dict
        metrics_after  : 変更後の指標 dict
        code_snippet   : 関連コード（あれば）
        priority       : "high"/"medium"/"low"
        status         : "未執筆"/"執筆中"/"公開済み"
    """
    # カテゴリをEnumかstrで受け付ける
    cat_value = category.value if isinstance(category, NoteCategory) else str(category)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "category": cat_value,
        "priority": priority,
        "status": status,
        "content": {
            "what_happened": what_happened,
            "why_it_matters": why_it_matters,
            "what_changed": what_changed,
            "metrics": {
                "before": metrics_before,
                "after": metrics_after,
            },
            "code_snippet": code_snippet,
        },
    }

    # 既存ログ読み込み
    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)

    # 優先度が高い場合はターミナルに通知
    if priority == "high":
        print(f"\n📝 NOTE CONTENT LOGGED [{cat_value}]")
        print(f"  タイトル案: {title}")
        print(f"  内容: {what_happened}")
        print(f"  → {NOTE_LOG_FILE.name} に記録しました\n")


def check_metric_change(old_metrics, new_metrics, experiment_name=""):
    """
    指標の変化を検知して自動記録する。

    Parameters:
        old_metrics (dict): 変更前の指標（roi, mdd 等を含む）
        new_metrics (dict): 変更後の指標
        experiment_name (str): 実験名
    """
    roi_old = old_metrics.get("roi", 0)
    roi_new = new_metrics.get("roi", 0)
    mdd_new = new_metrics.get("mdd", 0)
    pf_old  = old_metrics.get("pf", 0)
    pf_new  = new_metrics.get("pf", 0)

    roi_change = roi_new - roi_old
    pf_change  = pf_new - pf_old

    # ROI 5%以上改善
    if roi_change >= 5:
        log_note_content(
            title=f"{experiment_name}でROIが{roi_change:.1f}%改善した話",
            category=NoteCategory.IMPROVEMENT,
            what_happened=f"ROI {roi_old:.1f}% → {roi_new:.1f}%（+{roi_change:.1f}%）",
            why_it_matters="どの特徴量・設計変更が最も効いたかを具体的に示せる",
            metrics_before=old_metrics,
            metrics_after=new_metrics,
            priority="high",
        )

    # ROI 5%以上悪化
    if roi_change <= -5:
        log_note_content(
            title=f"変更したら逆に悪化した話：{experiment_name}（原因と対処法）",
            category=NoteCategory.FAILURE,
            what_happened=f"ROI {roi_old:.1f}% → {roi_new:.1f}%（{roi_change:.1f}%）",
            why_it_matters="失敗談は最も読まれる。原因究明の過程が価値になる",
            metrics_before=old_metrics,
            metrics_after=new_metrics,
            priority="high",
        )

    # PF 0.1以上改善
    if pf_change >= 0.1:
        log_note_content(
            title=f"PFが{pf_old:.2f}→{pf_new:.2f}に改善：{experiment_name}",
            category=NoteCategory.IMPROVEMENT,
            what_happened=f"PF {pf_old:.2f} → {pf_new:.2f}（+{pf_change:.2f}）",
            why_it_matters="PFの改善は期待値プラスへの具体的な進捗として示せる",
            metrics_before=old_metrics,
            metrics_after=new_metrics,
            priority="medium",
        )

    # 最大ドローダウン 20%以上
    if mdd_new >= 20:
        log_note_content(
            title=f"最大ドローダウン{mdd_new:.1f}%：何が起きたか",
            category=NoteCategory.FAILURE,
            what_happened=f"MDDが{mdd_new:.1f}%に達した（{experiment_name}）",
            why_it_matters="リスク管理の重要性をリアルに伝えられる",
            metrics_before=old_metrics,
            metrics_after=new_metrics,
            priority="high",
        )


def log_data_discovery(title, description, priority="medium"):
    """データを取って初めてわかった事実を記録する。"""
    log_note_content(
        title=f"データを取ってみたら意外な事実がわかった：{title}",
        category=NoteCategory.DISCOVERY,
        what_happened=description,
        why_it_matters="実際のデータを見ないとわからない事実は読者の関心が高い",
        priority=priority,
    )


def log_bug(bug_description, impact, fix):
    """バグ・設計ミスを記録する。"""
    log_note_content(
        title=f"重大なバグを発見：{bug_description}",
        category=NoteCategory.FAILURE,
        what_happened=bug_description,
        why_it_matters=f"影響：{impact}。同じミスをする人を救える記事になる",
        what_changed=fix,
        priority="high",
    )


def log_daily_pnl(date, pnl_data):
    """ペーパートレードの日次損益を記録する。"""
    today_roi = pnl_data.get("today_roi", 0)

    if isinstance(date, str):
        d = datetime.strptime(date, "%Y%m%d")
    else:
        d = date

    # 月初に前月分を記録
    if d.day == 1:
        log_note_content(
            title=f"{d.strftime('%Y年%m月')}のペーパートレード全記録",
            category=NoteCategory.MONEY,
            what_happened=f"月間ROI: {pnl_data.get('monthly_roi', 0):.1f}%",
            why_it_matters="収益記録は最も注目される。良くても悪くても正直に公開する",
            metrics_after=pnl_data,
            priority="high",
        )

    # 大きな当たり（1日ROI 50%以上）
    if today_roi >= 50:
        log_note_content(
            title=f"1日でROI {today_roi:.0f}%：何が起きたか全解説",
            category=NoteCategory.MONEY,
            what_happened=f"本日の収益：{pnl_data.get('today_profit', 0):,}円",
            why_it_matters="大きな当たりの詳細分析は最も読まれる",
            metrics_after=pnl_data,
            priority="high",
        )


def generate_note_report(priority_filter=None, status_filter="未執筆"):
    """
    蓄積されたnoteネタを優先度・カテゴリ別に整理して表示する。

    Parameters:
        priority_filter (str|None): "high"/"medium"/"low" でフィルタ
        status_filter   (str|None): "未執筆"/"執筆中"/"公開済み" でフィルタ
    """
    logs = _load_logs()

    filtered = logs
    if status_filter:
        filtered = [l for l in filtered if l.get("status") == status_filter]
    if priority_filter:
        filtered = [l for l in filtered if l.get("priority") == priority_filter]

    # 優先度順ソート
    priority_order = {"high": 0, "medium": 1, "low": 2}
    filtered.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

    print("=" * 60)
    print(f"📝 noteネタ一覧 ({len(filtered)}件)")
    if status_filter:
        print(f"   フィルタ: status={status_filter}", end="")
    if priority_filter:
        print(f" priority={priority_filter}", end="")
    print()
    print("=" * 60)

    for i, log in enumerate(filtered, 1):
        priority_mark = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(
            log.get("priority", "low"), "⚪"
        )
        print(f"\n{i}. {priority_mark} [{log['category']}]")
        print(f"   タイトル: {log['title']}")
        print(f"   記録日時: {log['timestamp'][:10]}")
        content = log.get("content", {})
        print(f"   内容: {content.get('what_happened', '')}")
        metrics = content.get("metrics", {})
        if metrics.get("before") or metrics.get("after"):
            print(f"   指標前: {metrics.get('before')}")
            print(f"   指標後: {metrics.get('after')}")

    high_count   = sum(1 for l in filtered if l.get("priority") == "high")
    medium_count = sum(1 for l in filtered if l.get("priority") == "medium")
    print(f"\n合計: {len(filtered)}件 (高={high_count}, 中={medium_count})")


def mark_as_written(title_keyword):
    """タイトルにキーワードを含む記録を「公開済み」に更新する。"""
    logs = _load_logs()
    updated = 0
    for log in logs:
        if title_keyword in log.get("title", ""):
            log["status"] = "公開済み"
            updated += 1
    _save_logs(logs)
    print(f"{updated}件を「公開済み」に更新しました")


def _load_logs():
    """ログファイルを読み込む。存在しない場合は空リストを返す。"""
    try:
        with open(NOTE_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_logs(logs):
    """ログファイルに書き込む。"""
    with open(NOTE_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def init_note_log():
    """初期登録済みnoteネタを投入する（開発開始時点で確定している高価値コンテンツ）。"""
    existing = _load_logs()
    existing_titles = {l["title"] for l in existing}

    initial_entries = [
        {
            "title": "なぜボートレースはAI予測の穴場なのか：競馬・競輪と徹底比較",
            "category": NoteCategory.DISCOVERY.value,
            "priority": "high",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "競馬に比べてファン層が小さくAI参入者が少ない",
                "why_it_matters": "障壁を突破できれば競合ほぼゼロの状態で戦える",
                "key_points": [
                    "競馬に比べてファン層が小さくAI参入者が少ない",
                    "公式APIがなくスクレイピングが必要→ここで諦める人が多い",
                    "進入コースの概念が難解→理解せずに諦める人が多い",
                    "予想師文化が強くAIという発想自体が少ない",
                    "障壁を突破できれば競合ほぼゼロの状態で戦える",
                ],
                "what_changed": None,
                "metrics": {"before": None, "after": None},
                "code_snippet": None,
            },
        },
        {
            "title": "ボートレースAIの致命的な落とし穴：進入コースを1ヶ月無視していた話",
            "category": NoteCategory.FAILURE.value,
            "priority": "high",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "艇番（枠番）≠実際の進入コース。1コース勝率55%・6コース勝率4%という最重要情報を無視していた",
                "why_it_matters": "全体の30〜40%のレースで誤った情報をモデルに食わせていた。同じミスをする人を救える記事になる",
                "key_points": [
                    "艇番（枠番）≠実際の進入コース",
                    "1コース勝率55%・6コース勝率4%という最重要情報を無視していた",
                    "全体の30〜40%のレースで誤った情報をモデルに食わせていた",
                    "スクレイピングでcourse_takenを取得して解決",
                ],
                "what_changed": None,
                "metrics": {"before": None, "after": None},
                "code_snippet": None,
            },
        },
        {
            "title": "合成データで回収率300%を出した話：なぜこれが最悪の罠なのか",
            "category": NoteCategory.FAILURE.value,
            "priority": "high",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "合成データ10,000レースで単勝回収率324%・Sharpe比20.38を達成。実データで評価し直したら大幅に下落",
                "why_it_matters": "合成データはカンニングと同じ構造。Sharpe比20はあり得ない数字（金融業界で2.0でも天才）",
                "key_points": [
                    "合成データはカンニングと同じ構造",
                    "Sharpe比20はあり得ない数字",
                    "実データで評価しない限りモデルの本当の実力はわからない",
                    "合成データを完全削除して実データのみで再評価",
                ],
                "what_changed": None,
                "metrics": {
                    "before": {"roi": 324, "sharpe": 20.38},
                    "after": None,  # 実データ評価後に記入
                },
                "code_snippet": None,
            },
        },
        {
            "title": "EVの計算式が3週間間違っていた：控除率を忘れていた話",
            "category": NoteCategory.FAILURE.value,
            "priority": "high",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "EV = 予測確率 × オッズ で計算していた。控除率約25%を考慮していなかった",
                "why_it_matters": "全ての期待値が過大評価されていた。約3週間分の計算が誤り。同じミスをする人を救える",
                "what_changed": "EV = 予測確率 × オッズ × 0.75 に修正。閾値も合わせて見直し",
                "metrics": {"before": None, "after": None},
                "code_snippet": None,
            },
        },
        {
            "title": "オッズが直前に急上昇した艇は買うな：市場の知恵に謙虚になる設計",
            "category": NoteCategory.DESIGN.value,
            "priority": "high",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "オッズ急上昇フィルターを実装した",
                "why_it_matters": "「市場が知っていて自分が知らない情報」への謙虚さという設計思想を伝えられる",
                "what_changed": "15分前→1分前で10%以上上昇で信頼度50%カット、20%以上で買いキャンセル",
                "metrics": {"before": None, "after": None},
                "code_snippet": None,
            },
        },
        {
            "title": "似たモデルを5つ並べても意味がない：アンサンブルの正しい理解",
            "category": NoteCategory.FAILURE.value,
            "priority": "medium",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "LightGBM・XGBoost・CatBoost・RF・ExtraTreesを5つアンサンブル→計算時間5倍、精度ほぼ変わらず",
                "why_it_matters": "多様性のないアンサンブルは実質1つと同じ。仕組みが全く違うモデルを組み合わせることが重要",
                "what_changed": "LightGBM + ニューラルネット + ロジスティック回帰のスタッキングに変更",
                "metrics": {"before": None, "after": None},
                "code_snippet": None,
            },
        },
        {
            "title": "オッズは確定値だけ見ても意味がない：7回取得して初めてわかること",
            "category": NoteCategory.DESIGN.value,
            "priority": "medium",
            "status": "未執筆",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "what_happened": "オッズを2時間前から1分前まで7タイミングで取得する設計を実装した",
                "why_it_matters": "確定値だけ見るのは試合終了後にスコアを見るのと同じ。変化率・加速度が重要な特徴量になる",
                "what_changed": "timing='120min'/'60min'/'30min'/'15min'/'5min'/'1min'/'final' で7回保存",
                "metrics": {"before": None, "after": None},
                "code_snippet": None,
            },
        },
    ]

    added = 0
    for entry in initial_entries:
        if entry["title"] not in existing_titles:
            existing.append(entry)
            added += 1

    if added > 0:
        _save_logs(existing)
        print(f"初期ネタ {added}件 を登録しました")
    else:
        print("初期ネタは既に登録済みです")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_note_log()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        generate_note_report(priority_filter="high")
    else:
        init_note_log()
        generate_note_report()
