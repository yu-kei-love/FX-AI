# ===========================================
# scraper/scraper_comment.py
# 競輪 - 選手コメントスクレイパー
#
# 取得タイミング：前日夜（本番前日20時以降）
#
# 選手コメントはライン予測の最重要情報源：
# 「○○選手を追います」→ ライン構成の直接的な証拠
#
# 注意：このファイルはスクレイピングコードのため
#       note販売パッケージには含めない
# ===========================================

import time
import random
import sqlite3
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

SLEEP_MIN = 1.0
SLEEP_MAX = 2.0

# コメント取得ソース（優先順位順）
COMMENT_SOURCES = [
    "gamboo",       # 競輪ギャンブー（記者予想あり）
    "chariloto",    # チャリロト公式
    "keirin_official",  # 各競輪場公式
]


def _polite_sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def fetch_race_comments(race_id: str, source: str = "gamboo") -> list:
    """
    レースの選手コメントを取得する。

    Parameters:
        race_id : レースID
        source  : "gamboo" / "chariloto" / "keirin_official"

    Returns:
        [{"car_no": int, "comment_text": str, "comment_date": str}]
    """
    raise NotImplementedError(
        f"fetch_race_comments({source}): 実装待ち（スクレイピング先調査後に実装）\n"
        "選手コメントのURL構造を確認してから実装すること"
    )


def fetch_reporter_predictions(race_id: str) -> dict:
    """
    記者予想（ライン予想）を取得する。

    Returns:
        {"3-7-4": 0.9, "9-2": 0.7}  # ライン文字列: 信頼度
    """
    raise NotImplementedError(
        "fetch_reporter_predictions: 実装待ち（gamboo/チャリロトの記者予想ページ調査後に実装）"
    )


def parse_comment_for_lines(comment_text: str) -> dict:
    """
    コメントテキストからライン情報を抽出する。

    解析ルール：
    「○○選手を追います」→ {"role": "follow", "target": "○○"}
    「先行します」「自力で行きます」→ {"role": "lead", "target": None}
    「番手で勝負します」→ {"role": "second", "target": None}
    「単騎で行きます」→ {"role": "single", "target": None}

    Parameters:
        comment_text: 選手コメントの文字列

    Returns:
        {"role": str, "target": str|None, "confidence": float}
    """
    patterns = {
        "lead": [
            r"(先行|自力|前から|逃げ|前で)",
            r"(先手で|先頭で)",
        ],
        "follow": [
            r"(\S+)選手(を|の後ろ|の番手)(を?追|につ)",
            r"(\S+)番(の後ろ|番手|を追)",
            r"(番手で|マークで|後ろにつ)",
        ],
        "single": [
            r"(単騎|一人で|単独で)",
        ],
    }

    for role, role_patterns in patterns.items():
        for pattern in role_patterns:
            m = re.search(pattern, comment_text)
            if m:
                target = None
                if role == "follow" and m.lastindex and m.lastindex >= 1:
                    raw = m.group(1)
                    # 数字だけなら車番、それ以外は選手名
                    target = raw if raw.strip() else None
                return {
                    "role":       role,
                    "target":     target,
                    "confidence": 0.80 if role in ("lead", "single") else 0.70,
                }

    # パターンに合致しない → 不明
    return {"role": "unknown", "target": None, "confidence": 0.30}


def save_comments_to_db(comments: list, race_id: str):
    """
    コメントをDBに保存する。

    Parameters:
        comments : [{"car_no": int, "comment_text": str, "comment_date": str, "source": str}]
        race_id  : レースID
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DBが見つかりません: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    for c in comments:
        comment_id = f"{race_id}_c{c['car_no']:02d}_{c.get('source', 'unknown')}"
        cur.execute("""
            INSERT OR REPLACE INTO comments
            (comment_id, race_id, car_no, comment_text, comment_date, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            comment_id,
            race_id,
            c["car_no"],
            c["comment_text"],
            c.get("comment_date", datetime.now().strftime("%Y-%m-%d")),
            c.get("source", "unknown"),
        ))

    conn.commit()
    conn.close()


def run_comment_scraping(race_ids: list):
    """
    レース一覧のコメントを一括取得してDBに保存する。

    前日夜に実行する想定。
    """
    print(f"コメント取得開始: {len(race_ids)}レース")

    for race_id in race_ids:
        for source in COMMENT_SOURCES:
            try:
                comments = fetch_race_comments(race_id, source)
                save_comments_to_db(comments, race_id)
                print(f"  [{race_id}] {source} コメント保存完了")
                _polite_sleep()
                break  # 1つのソースで取れたら次のレースへ
            except NotImplementedError as e:
                print(f"  未実装: {e}")
                break
            except Exception as e:
                print(f"  エラー [{race_id}/{source}]: {e}")
                continue


if __name__ == "__main__":
    # コメント解析テスト
    test_comments = [
        "○○選手を追います。しっかり番手で勝負したい",
        "先行して自力で勝ちたい",
        "単騎になりますが、自分の競走をします",
        "番手で勝負します",
        "前に出て先行する予定です",
    ]
    print("コメント解析テスト:")
    for comment in test_comments:
        result = parse_comment_for_lines(comment)
        print(f"  '{comment[:20]}...' → {result}")
