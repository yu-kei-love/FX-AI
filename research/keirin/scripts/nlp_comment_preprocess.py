# ===========================================
# scripts/nlp_comment_preprocess.py
# v0.46: 選手コメント NLP 前処理 (Task D)
#
# 574K コメントから以下の特徴量を生成:
#   - sentiment_score: -1.0 〜 +1.0
#   - confidence_score: 0.0 〜 1.0 (自信の強さ)
#   - keyword_features: JSON {positive, negative, injury, top_candidate, ...}
#
# 手法: 競輪固有キーワード辞書によるルールベース判定
#       (sudachipy Windows DLL ブロック回避のため)
# ===========================================

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
from feature_engine import DB_PATH

REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"
PROGRESS_LOG = REPORT_DIR / "parallel_progress.log"
NOTIFICATION_LOG = REPORT_DIR / "notification.log"

# 競輪固有感情辞書
POSITIVE_KEYWORDS = {
    # 強ポジ (+1.0)
    "絶好調": 1.0, "絶好": 1.0, "最高": 1.0, "快調": 1.0,
    "勝てる": 1.0, "勝機": 1.0, "自信": 1.0, "楽勝": 1.0,
    "復活": 1.0, "復調": 0.8, "絶対": 0.8,
    # 中ポジ (+0.6)
    "好調": 0.6, "調子": 0.5, "上向き": 0.6, "余裕": 0.6,
    "狙える": 0.7, "狙う": 0.5, "勝負": 0.6, "頑張": 0.3,
    "しっかり": 0.4, "力": 0.3, "問題ない": 0.6,
    # 軽ポジ (+0.3)
    "悪くない": 0.4, "いい": 0.3, "良い": 0.3, "まずまず": 0.3,
    "動ける": 0.4, "仕上がった": 0.5, "万全": 0.8,
}

NEGATIVE_KEYWORDS = {
    # 強ネガ (-1.0)
    "絶望": -1.0, "最悪": -1.0, "無理": -1.0, "勝負にならない": -1.0,
    "厳しい": -0.7, "きつい": -0.6, "苦しい": -0.7,
    # 中ネガ (-0.5)
    "不調": -0.7, "悪い": -0.5, "ダメ": -0.7, "だめ": -0.7,
    "疲れ": -0.5, "疲労": -0.5, "痛い": -0.5,
    "出遅れ": -0.4, "乗れない": -0.6, "合わない": -0.4,
    # 軽ネガ (-0.3)
    "難しい": -0.4, "わからない": -0.2, "不安": -0.5,
    "心配": -0.4, "微妙": -0.3, "まだ": -0.1,
}

INJURY_KEYWORDS = {
    "怪我": 1.0, "けが": 1.0, "負傷": 1.0, "故障": 1.0,
    "痛み": 0.7, "違和感": 0.6, "張り": 0.4,
    "腰": 0.3, "膝": 0.3, "足": 0.2,
}

CONFIDENCE_HIGH = ["絶対", "必ず", "自信", "間違いな", "確実", "任せ", "勝てる", "絶好調"]
CONFIDENCE_LOW = ["どうか", "わからな", "たぶん", "かもしれな", "かな",
                  "どうだろう", "難しい"]


def score_comment(text):
    """コメント文から各種スコアを計算"""
    if not isinstance(text, str) or not text.strip():
        return {
            "sentiment_score": 0.0,
            "confidence_score": 0.0,
            "injury_flag": 0.0,
            "n_positive": 0,
            "n_negative": 0,
            "length": 0,
        }

    pos_score = 0.0
    n_pos = 0
    for kw, w in POSITIVE_KEYWORDS.items():
        cnt = text.count(kw)
        if cnt > 0:
            pos_score += w * cnt
            n_pos += cnt

    neg_score = 0.0
    n_neg = 0
    for kw, w in NEGATIVE_KEYWORDS.items():
        cnt = text.count(kw)
        if cnt > 0:
            neg_score += w * cnt  # w は負値
            n_neg += cnt

    injury = 0.0
    for kw, w in INJURY_KEYWORDS.items():
        if kw in text:
            injury = max(injury, w)

    total = pos_score + neg_score  # 符号付き
    # 正規化: tanh で -1〜1 に
    import math
    sentiment = math.tanh(total / 2.0)

    # 信頼度
    conf_high = sum(text.count(kw) for kw in CONFIDENCE_HIGH)
    conf_low = sum(text.count(kw) for kw in CONFIDENCE_LOW)
    if conf_high + conf_low > 0:
        confidence = (conf_high - conf_low) / (conf_high + conf_low + 1) * 0.5 + 0.5
    else:
        confidence = 0.5

    # 否定語チェック (「勝てない」「調子悪くない」等)
    negation_patterns = [r"ない\b", r"ぬ\b", r"ません"]
    has_negation = any(re.search(p, text) for p in negation_patterns)
    if has_negation and sentiment > 0:
        sentiment *= 0.5  # 弱める

    return {
        "sentiment_score": round(sentiment, 4),
        "confidence_score": round(confidence, 4),
        "injury_flag": round(injury, 4),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "length": len(text),
    }


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [TaskD-NLP] {msg}"
    print(line)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    t0 = time.time()
    log("=== 選手コメント NLP 前処理 開始 ===")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    # テーブル作成
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS comment_features (
            race_id           TEXT NOT NULL,
            sha_ban           INTEGER NOT NULL,
            sentiment_score   REAL,
            confidence_score  REAL,
            injury_flag       REAL,
            n_positive        INTEGER,
            n_negative        INTEGER,
            comment_length    INTEGER,
            processed_at      TEXT,
            PRIMARY KEY (race_id, sha_ban)
        );
        CREATE INDEX IF NOT EXISTS idx_cf_race ON comment_features(race_id);
    """)
    conn.commit()

    # 未処理コメントを取得
    already_done = set(
        (row[0], row[1]) for row in cur.execute(
            "SELECT race_id, sha_ban FROM comment_features"
        ).fetchall()
    )
    log(f"既存処理済: {len(already_done):,}")

    cur.execute("""
        SELECT race_id, sha_ban, comment_text
        FROM comments
        WHERE comment_text IS NOT NULL
    """)
    rows = cur.fetchall()
    log(f"コメント総数: {len(rows):,}")

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    n_new = 0
    n_skip = 0
    batch = []
    BATCH_SIZE = 5000
    for i, (race_id, sha_ban, text) in enumerate(rows):
        if (race_id, sha_ban) in already_done:
            n_skip += 1
            continue
        try:
            sha_ban = int(sha_ban)
        except (ValueError, TypeError):
            continue
        s = score_comment(text)
        batch.append((
            race_id, sha_ban,
            s["sentiment_score"], s["confidence_score"], s["injury_flag"],
            s["n_positive"], s["n_negative"], s["length"], now,
        ))
        if len(batch) >= BATCH_SIZE:
            cur.executemany("""
                INSERT OR REPLACE INTO comment_features
                    (race_id, sha_ban, sentiment_score, confidence_score,
                     injury_flag, n_positive, n_negative,
                     comment_length, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            conn.commit()
            n_new += len(batch)
            batch = []
            if i % 50000 == 0 and i > 0:
                log(f"進捗 {i:,}/{len(rows):,} ({i/len(rows)*100:.1f}%), 新規 {n_new:,}")

    if batch:
        cur.executemany("""
            INSERT OR REPLACE INTO comment_features
                (race_id, sha_ban, sentiment_score, confidence_score,
                 injury_flag, n_positive, n_negative,
                 comment_length, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
        n_new += len(batch)

    conn.close()

    elapsed = time.time() - t0
    log(f"=== 完了 新規 {n_new:,}, スキップ {n_skip:,}, 所要 {elapsed/60:.1f}分 ===")
    # 通知
    with open(NOTIFICATION_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now}] タスクD完了 (records: {n_new:,})\n")


if __name__ == "__main__":
    main()
