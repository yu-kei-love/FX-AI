# ===========================================
# title_researcher.py
# note.com のトレンドタイトルリサーチ
#
# 機能:
#   - 人気投資/AI系note記事のタイトル調査
#   - クリックされやすいタイトル候補の生成
#   - タイトルパターンの分析（数字、疑問形、括弧等）
#   - リサーチ結果を title_research.json に保存
# ===========================================

import sys
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import Counter

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ログ設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "title_research.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# パス定義
NOTE_SALES_DIR = PROJECT_ROOT / "research" / "note_sales"
RESEARCH_FILE = NOTE_SALES_DIR / "title_research.json"

# =============================================================
# タイトルパターン定義
# note.comで売れるタイトルのパターン
# =============================================================
TITLE_PATTERNS = {
    "数字入り": r"\d+",
    "疑問形": r"[？?]",
    "括弧_隅付き": r"[【】\[\]]",
    "括弧_丸": r"[（）()]",
    "感嘆符": r"[！!]",
    "ネガティブ訴求": r"(失敗|損|注意|危険|やめ|間違|後悔|落とし穴)",
    "ポジティブ訴求": r"(成功|勝|利益|稼|儲|最強|最高|完全)",
    "ハウツー": r"(方法|やり方|手順|ステップ|入門|始め方|コツ|テクニック)",
    "リスト形式": r"(\d+選|\d+つ|\d+個)",
    "限定感": r"(秘密|極意|裏|本音|真実|暴露|完全版)",
    "初心者向け": r"(初心者|入門|基礎|はじめて|ゼロから|0から)",
    "実績アピール": r"(\d+万円|\d+%|実績|結果|証拠|公開)",
    "時事性": r"(2024|2025|2026|最新|今|速報)",
    "AI関連": r"(AI|人工知能|機械学習|ChatGPT|Python|自動化)",
}

# タイトルテンプレート集
TITLE_TEMPLATES = {
    "stock_prediction": [
        "【AI株予測】{topic}を機械学習で分析した結果",
        "Pythonで{topic}を予測するAIモデルを作ってみた",
        "【検証】AIは{topic}を予測できるのか？{n}ヶ月の成績を公開",
        "{topic}をAIで予測して{result}だった話",
        "【完全版】{topic}のAI予測モデル構築ガイド",
        "【初心者向け】{topic}のAI分析入門",
        "プロが教える{topic}のAI予測手法{n}選",
        "【実績公開】{topic}AI予測の勝率{n}%を達成した方法",
        "なぜ{topic}の予測にAIが有効なのか？データで解説",
        "【無料公開】{topic}AI予測の基本フレームワーク",
    ],
    "fx_trading": [
        "【FX AI】{topic}の自動売買システムを作った結果",
        "FXでAIを使って{topic}を検証した{n}ヶ月の記録",
        "【Python】{topic}のFX予測モデル完全実装ガイド",
        "AIトレーダーが{topic}について語る本音",
        "【検証済】{topic}のFX戦略で利益は出るのか？",
        "FX初心者がAIで{topic}を始めた結果",
        "Walk-Forward検証で{topic}の本当の実力を測る",
        "【リスク管理】{topic}で大損しないための{n}つのルール",
        "FXのAI予測で{topic}はどこまで有効か",
        "【コード公開】{topic}のFX予測モデルをPythonで実装",
    ],
    "boat_race": [
        "【競艇AI】{topic}で勝率を上げる方法",
        "AIが競艇の{topic}を分析した驚きの結果",
        "【検証】競艇AI予測の{n}レース分の成績を公開",
        "Pythonで競艇の{topic}予測モデルを作ってみた",
        "【競艇攻略】{topic}をAIで分析する手法",
        "5モデルアンサンブルで競艇{topic}の的中率が上がった話",
        "Kelly基準で{topic}の最適ベットサイズを計算する",
        "【EV計算】{topic}で期待値プラスの賭け方",
        "競艇AI予測で{topic}に注目すべき{n}つの理由",
        "【データ分析】{topic}から見る競艇の傾向",
    ],
    "crypto": [
        "【暗号通貨AI】{topic}の予測モデルを構築",
        "ビットコインの{topic}をAIで予測した結果",
        "【検証】暗号通貨AI予測は{topic}で使えるのか？",
        "Pythonで{topic}の暗号通貨予測を実装する方法",
        "オンチェーンデータ×AIで{topic}を分析",
        "【テクニカル分析】{topic}にAIを活用する方法",
        "暗号通貨市場で{topic}を活かす{n}つの戦略",
        "DeFi時代の{topic}AI予測モデル",
        "【初心者向け】{topic}の暗号通貨AI分析入門",
        "仮想通貨の{topic}をPythonで検証してみた",
    ],
    "general_ai": [
        "【AI投資】{topic}で月{n}万円の不労所得は可能か",
        "ChatGPTと{topic}を組み合わせた投資戦略",
        "AIで{topic}を自動化した話【Python】",
        "2026年版・{topic}のAI活用ガイド",
        "なぜ{topic}にAIを使うべきなのか【データで検証】",
        "【後悔しない】{topic}のAI投資で気をつけること{n}選",
        "個人投資家が{topic}でAIを使うメリットとデメリット",
        "【体験談】{topic}のAI予測で分かったこと",
        "Pythonで始める{topic}のAI分析【環境構築から】",
        "【まとめ】{topic}のAI投資に必要な知識全部",
    ],
}


def research_trending_titles() -> List[Dict[str, Any]]:
    """人気の投資/AI系note記事のタイトルをリサーチする

    注意: 実際のスクレイピングは行わず、テンプレートベースのリサーチデータを
    生成します。Web検索を利用する場合は別途APIキーが必要です。

    Returns:
        トレンドタイトル情報のリスト
    """
    logger.info("トレンドタイトルリサーチ開始")

    # リサーチカテゴリ
    categories = [
        "AI 投資 予測",
        "Python 株 自動売買",
        "機械学習 FX トレード",
        "競艇 AI 予測",
        "暗号通貨 AI 分析",
        "LightGBM 株価予測",
        "note 投資 人気記事",
    ]

    # 効果的なタイトルパターンの知識ベース
    trending_data = [
        {
            "category": "AI投資",
            "sample_titles": [
                "【2026最新】AIで株価予測する方法を完全解説",
                "ChatGPTに株の銘柄選びをさせてみた結果がすごかった",
                "【実績公開】AI自動売買で月10万円稼いだ3ヶ月の記録",
                "初心者がPythonで投資AIを作るまでの全手順",
                "【無料】AIトレードの始め方ガイド2026年版",
            ],
            "avg_engagement": "high",
            "key_elements": ["具体的な数字", "実績", "最新年度", "初心者向け"],
        },
        {
            "category": "Python金融",
            "sample_titles": [
                "Pythonで日経平均を予測するAIを作ってみた【コード全公開】",
                "【LightGBM】株価予測モデルの作り方を1から解説",
                "FXの自動売買botをPythonで作る方法【初心者OK】",
                "Walk-Forward検証とは？投資AIの正しい評価方法",
                "【コピペOK】5分で動く株価予測AIのPythonコード",
            ],
            "avg_engagement": "high",
            "key_elements": ["コード提供", "初心者歓迎", "具体的手順"],
        },
        {
            "category": "競艇AI",
            "sample_titles": [
                "AIで競艇予測したら回収率120%だった話",
                "【競艇AI】Pythonで勝てる予測モデルを作る方法",
                "5つのAIモデルをアンサンブルして競艇予測の精度を上げた",
                "競艇データをPythonでスクレイピングして分析する方法",
                "【検証】AI競艇予測は本当に儲かるのか？100レースの結果",
            ],
            "avg_engagement": "medium",
            "key_elements": ["具体的成績", "検証結果", "実践的"],
        },
    ]

    # リサーチ結果を保存
    research_result = {
        "research_date": datetime.now().isoformat(),
        "categories": categories,
        "trending_data": trending_data,
        "analysis": analyze_title_patterns(
            [t for d in trending_data for t in d["sample_titles"]]
        ),
    }

    RESEARCH_FILE.write_text(
        json.dumps(research_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"リサーチ結果を保存しました: {RESEARCH_FILE}")

    return trending_data


def generate_title_candidates(topic: str, n: int = 10) -> List[str]:
    """クリックされやすいタイトル候補を生成する

    Args:
        topic: 記事のトピック
        n: 生成するタイトル数（デフォルト: 10）

    Returns:
        タイトル候補のリスト
    """
    logger.info(f"タイトル候補生成: topic={topic}, n={n}")

    candidates = []
    import random

    # 全カテゴリのテンプレートを統合
    all_templates = []
    for category_templates in TITLE_TEMPLATES.values():
        all_templates.extend(category_templates)

    # テンプレートからタイトル生成
    random.shuffle(all_templates)
    for template in all_templates[:n * 2]:
        try:
            title = template.format(
                topic=topic,
                n=random.choice([3, 5, 7, 10, 30, 50, 100]),
                result=random.choice(["驚きの結果", "意外な結果", "想像以上の成果"]),
            )
            candidates.append(title)
        except (KeyError, IndexError):
            continue

    # 重複除去して上位n件を返す
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    result = unique_candidates[:n]

    # スコアリング（パターンマッチ数でソート）
    scored = []
    for title in result:
        score = 0
        for pattern_name, pattern in TITLE_PATTERNS.items():
            if re.search(pattern, title):
                score += 1
        scored.append((title, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    final = [t for t, s in scored[:n]]

    logger.info(f"タイトル候補 {len(final)} 件生成完了")
    for i, title in enumerate(final, 1):
        logger.info(f"  {i}. {title}")

    return final


def analyze_title_patterns(titles: List[str]) -> Dict[str, Any]:
    """タイトルパターンを分析する

    どのパターン要素が多く使われているかを分析する。

    Args:
        titles: 分析対象のタイトルリスト

    Returns:
        パターン分析結果の辞書
    """
    logger.info(f"タイトルパターン分析: {len(titles)} 件")

    if not titles:
        return {"error": "分析対象のタイトルがありません"}

    # 各パターンの出現回数
    pattern_counts = {}
    for pattern_name, pattern in TITLE_PATTERNS.items():
        count = sum(1 for t in titles if re.search(pattern, t))
        pattern_counts[pattern_name] = {
            "count": count,
            "percentage": round(count / len(titles) * 100, 1),
        }

    # タイトル長の分析
    lengths = [len(t) for t in titles]
    length_stats = {
        "平均文字数": round(sum(lengths) / len(lengths), 1),
        "最短": min(lengths),
        "最長": max(lengths),
        "推奨範囲": "25-45文字",
    }

    # 先頭パターン分析
    first_chars = Counter()
    for t in titles:
        if t.startswith("【"):
            bracket_end = t.find("】")
            if bracket_end > 0:
                first_chars[t[: bracket_end + 1]] += 1
        elif t[0].isdigit():
            first_chars["数字で開始"] += 1
        else:
            first_chars["通常テキスト"] += 1

    # 結果をまとめる
    analysis = {
        "total_titles": len(titles),
        "pattern_analysis": pattern_counts,
        "length_statistics": length_stats,
        "first_char_patterns": dict(first_chars.most_common(10)),
        "recommendations": [
            "【】で囲んだカテゴリ表示は注目を集めやすい",
            "具体的な数字（3選、5つのコツ等）を含めると効果的",
            "「初心者向け」「完全版」等の限定ワードが有効",
            "疑問形のタイトルは読者の興味を引く",
            "タイトル長は30-40文字が最適",
            "年度（2026年）を入れると最新感が出る",
        ],
        "analyzed_at": datetime.now().isoformat(),
    }

    logger.info("パターン分析完了:")
    for name, data in pattern_counts.items():
        if data["count"] > 0:
            logger.info(f"  {name}: {data['count']}件 ({data['percentage']}%)")

    return analysis


# =============================================================
# CLI エントリーポイント
# =============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="note.com タイトルリサーチ")
    parser.add_argument(
        "--action",
        choices=["research", "generate", "analyze"],
        default="research",
        help="実行アクション (default: research)",
    )
    parser.add_argument("--topic", type=str, default="AI株価予測", help="トピック")
    parser.add_argument("--n", type=int, default=10, help="タイトル候補数")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  タイトルリサーチ: action={args.action}")
    logger.info("=" * 60)

    if args.action == "research":
        data = research_trending_titles()
        logger.info(f"リサーチ完了: {len(data)} カテゴリ")

    elif args.action == "generate":
        titles = generate_title_candidates(args.topic, args.n)
        logger.info(f"タイトル候補 {len(titles)} 件生成")

    elif args.action == "analyze":
        # 既存のリサーチデータから分析
        if RESEARCH_FILE.exists():
            data = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
            all_titles = []
            for td in data.get("trending_data", []):
                all_titles.extend(td.get("sample_titles", []))
            result = analyze_title_patterns(all_titles)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            logger.warning("リサーチデータがありません。先に --action research を実行してください。")
