# ===========================================
# scripts/build_elo_cache.py
# 競輪AI - Elo レーティング事前計算
#
# 全レースを時系列順に処理して拡張Eloを計算し
# elo_cache テーブルに保存する。
#
# 使い方:
#   cd research/keirin
#   python scripts/build_elo_cache.py
#
# 設計:
#   初期値: 1500, K係数: 32
#   1着選手は全敗者に勝ったとして更新
#   2着選手は1着以外の敗者に勝ったとして更新
#   （北海道大学の拡張Eloに準拠）
#   ※レース前のEloをas_of_dateに記録（データリーク防止）
# ===========================================

import sqlite3
import time
from itertools import groupby
from operator import itemgetter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT.parent.parent / "data" / "keirin" / "keirin.db"

K = 32.0
INITIAL = 1500.0


def build_elo_cache(db_path=None):
    """全レースのEloを計算してelo_cacheに保存する"""
    db = Path(db_path) if db_path else DB_PATH
    print(f"DB: {db}")

    conn = sqlite3.connect(str(db), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # elo_cache テーブル作成
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elo_cache (
            senshu_name TEXT NOT NULL,
            as_of_date  TEXT NOT NULL,
            elo_rating  REAL NOT NULL,
            PRIMARY KEY (senshu_name, as_of_date)
        )
    """)

    # 既存キャッシュをクリア（全件再計算）
    conn.execute("DELETE FROM elo_cache")
    conn.commit()
    print("既存キャッシュをクリア")

    # 全レース結果を時系列順に読み込み
    print("全レース結果を読み込み中...")
    t0 = time.time()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.race_id, rc.race_date, r.senshu_name, r.rank
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.senshu_name IS NOT NULL AND r.rank IS NOT NULL
        ORDER BY rc.race_date, r.race_id, r.rank
    """)
    all_results = cur.fetchall()
    elapsed = time.time() - t0
    print(f"  {len(all_results):,} 行読み込み ({elapsed:.1f}秒)")

    if not all_results:
        print("レースデータなし")
        conn.close()
        return

    # 現在のElo値
    current_elo = {}  # senshu_name → float

    # キャッシュ用バッファ
    cache_buffer = []  # [(senshu_name, as_of_date, elo_rating), ...]
    BATCH_SIZE = 10000

    # レースごとにグループ化して処理
    print("Elo計算中...")
    t0 = time.time()
    race_count = 0

    for race_id, race_group in groupby(all_results, key=itemgetter(0)):
        race_rows = list(race_group)
        if not race_rows:
            continue

        race_date = race_rows[0][1]
        race_count += 1

        # 出走者と着順
        participants = []
        for _, _, sname, rank in race_rows:
            if sname:
                participants.append((sname, rank))

        # レース前のEloをキャッシュに記録（データリーク防止）
        for sname, _ in participants:
            if sname not in current_elo:
                current_elo[sname] = INITIAL
            cache_buffer.append((
                sname, race_date, current_elo[sname]
            ))

        # 拡張Elo更新
        n = len(participants)
        if n < 2:
            continue

        elo_deltas = {sname: 0.0 for sname, _ in participants}

        for i in range(n):
            for j in range(i + 1, n):
                name_i, rank_i = participants[i]
                name_j, rank_j = participants[j]

                elo_i = current_elo.get(name_i, INITIAL)
                elo_j = current_elo.get(name_j, INITIAL)

                # 期待勝率
                exp_i = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))

                # 実際の結果
                if rank_i < rank_j:
                    actual_i = 1.0
                elif rank_i > rank_j:
                    actual_i = 0.0
                else:
                    actual_i = 0.5

                # K係数をペア数で調整
                k_adj = K / max(n - 1, 1)
                delta = k_adj * (actual_i - exp_i)
                elo_deltas[name_i] += delta
                elo_deltas[name_j] -= delta

        # Elo更新
        for sname, delta in elo_deltas.items():
            current_elo[sname] = current_elo.get(sname, INITIAL) + delta

        # バッチ書き込み
        if len(cache_buffer) >= BATCH_SIZE:
            conn.executemany(
                "INSERT OR REPLACE INTO elo_cache "
                "(senshu_name, as_of_date, elo_rating) VALUES (?, ?, ?)",
                cache_buffer,
            )
            conn.commit()
            cache_buffer.clear()

        if race_count % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {race_count:,} レース処理 ({elapsed:.1f}秒)")

    # 残りバッファを書き込み
    if cache_buffer:
        conn.executemany(
            "INSERT OR REPLACE INTO elo_cache "
            "(senshu_name, as_of_date, elo_rating) VALUES (?, ?, ?)",
            cache_buffer,
        )
        conn.commit()

    elapsed = time.time() - t0
    print(f"\n完了: {race_count:,} レース, "
          f"{len(current_elo):,} 選手 ({elapsed:.1f}秒)")

    # 統計
    cur.execute("SELECT COUNT(*) FROM elo_cache")
    cache_count = cur.fetchone()[0]
    cur.execute("SELECT MIN(elo_rating), MAX(elo_rating), AVG(elo_rating) "
                "FROM elo_cache")
    elo_min, elo_max, elo_avg = cur.fetchone()
    print(f"elo_cache: {cache_count:,} 行")
    print(f"Elo範囲: {elo_min:.1f} 〜 {elo_max:.1f} (平均: {elo_avg:.1f})")

    conn.close()


if __name__ == "__main__":
    build_elo_cache()
