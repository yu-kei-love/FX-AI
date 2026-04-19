# ===========================================
# scripts/check_oddspark_progress.py
# オッズパーク取得進捗確認
#
# 使い方:
#   python scripts/check_oddspark_progress.py
# ===========================================

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
from feature_engine import DB_PATH


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    print("=" * 60)
    print("  オッズパーク取得進捗")
    print("=" * 60)

    # 総 races (2022-2024)
    total_races = cur.execute("""
        SELECT COUNT(*) FROM races
        WHERE race_date BETWEEN '20220101' AND '20241231'
    """).fetchone()[0]

    # 取得済み races
    done_races = cur.execute("""
        SELECT COUNT(DISTINCT race_id) FROM odds_trifecta_final
    """).fetchone()[0]

    # 総オッズ
    total_odds = cur.execute(
        "SELECT COUNT(*) FROM odds_trifecta_final"
    ).fetchone()[0]

    progress_pct = done_races / total_races * 100 if total_races > 0 else 0
    print(f"\n  対象レース: {total_races:,}")
    print(f"  取得済み:   {done_races:,} ({progress_pct:.2f}%)")
    print(f"  オッズ件数: {total_odds:,}")
    if done_races > 0:
        print(f"  平均odds/race: {total_odds/done_races:.1f}")

    # 1時間ごとの取得件数
    now = datetime.now()
    print(f"\n=== 直近6時間の取得件数 ===")
    for h in range(6, 0, -1):
        t_lo = (now - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
        t_hi = (now - timedelta(hours=h-1)).strftime("%Y-%m-%d %H:%M:%S")
        cnt = cur.execute("""
            SELECT COUNT(DISTINCT race_id) FROM odds_trifecta_final
            WHERE fetched_at >= ? AND fetched_at < ?
        """, (t_lo, t_hi)).fetchone()[0]
        print(f"  {t_lo[11:16]} - {t_hi[11:16]} : {cnt:,} races")

    # 残り時間推定
    last_1h = cur.execute("""
        SELECT COUNT(DISTINCT race_id) FROM odds_trifecta_final
        WHERE fetched_at >= ?
    """, ((now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),)).fetchone()[0]
    remaining = total_races - done_races
    if last_1h > 0:
        eta_hours = remaining / last_1h
        print(f"\n  直近1h速度: {last_1h:,} races/hour")
        print(f"  残り: {remaining:,} races")
        print(f"  推定残り時間: {eta_hours:.1f} hours")

    # エラーログ
    failed_log = PROJECT_DIR / "scraper" / "failed_oddspark.log"
    if failed_log.exists():
        print(f"\n=== failed_oddspark.log 末尾 5 行 ===")
        try:
            with failed_log.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-5:]
            for line in lines:
                print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  読めず: {e}")
    else:
        print(f"\n  failed log なし")

    # 各 parallel の進捗
    p1_log = PROJECT_DIR.parent.parent / "data" / "keirin" / "oddspark_p1.log"
    p2_log = PROJECT_DIR.parent.parent / "data" / "keirin" / "oddspark_p2.log"
    for name, lp in [("P1", p1_log), ("P2", p2_log)]:
        if lp.exists():
            try:
                with lp.open("r", encoding="utf-8", errors="replace") as f:
                    tail = f.readlines()[-1:]
                print(f"\n  [{name}] {tail[0].rstrip() if tail else ''}")
            except Exception:
                pass

    conn.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
