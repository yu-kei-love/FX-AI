# ===========================================
# scripts/check_odds_coverage.py
# 競輪オッズ収集状況の確認スクリプト
#
# 使い方:
#   cd research/keirin
#   python scripts/check_odds_coverage.py
#
# 出力:
#   - 過去7日間の収集レース数（日別）
#   - スナップショット分数別カバレッジ (60/30/10/0分前)
#   - 券種別（3連単・2車単）カバレッジ
#   - 最新 odds_errors.log の末尾10行
# ===========================================

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))

from feature_engine import DB_PATH

ERRLOG = PROJECT_DIR.parent.parent / "data" / "keirin" / "odds_errors.log"


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    print("=" * 60)
    print("  競輪オッズ収集カバレッジ")
    print("=" * 60)

    # 1. 過去7日間の日別収集
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    today_str = today.strftime("%Y%m%d")
    week_str = week_ago.strftime("%Y%m%d")

    print(f"\n=== 過去7日間 (race_date) ===")
    rows = cur.execute("""
        SELECT r.race_date,
               COUNT(DISTINCT oh.race_id) AS n_races,
               COUNT(*)                    AS n_rows
        FROM odds_history oh
        JOIN races r ON oh.race_id = r.race_id
        WHERE r.race_date >= ?
        GROUP BY r.race_date
        ORDER BY r.race_date DESC
    """, (week_str,)).fetchall()
    if not rows:
        print("  (収集データなし)")
    else:
        print(f"  {'日付':>10}  {'レース数':>8}  {'レコード数':>10}")
        for race_date, n_races, n_rows in rows:
            print(f"  {race_date:>10}  {n_races:>8,}  {n_rows:>10,}")

    # 2. 分数別スナップショット
    print(f"\n=== スナップショット分数別 (過去7日) ===")
    rows = cur.execute("""
        SELECT oh.minutes_before,
               COUNT(DISTINCT oh.race_id) AS n_races,
               COUNT(*)                    AS n_rows
        FROM odds_history oh
        JOIN races r ON oh.race_id = r.race_id
        WHERE r.race_date >= ?
        GROUP BY oh.minutes_before
        ORDER BY oh.minutes_before DESC
    """, (week_str,)).fetchall()
    if not rows:
        print("  (データなし)")
    else:
        print(f"  {'分前':>6}  {'レース数':>8}  {'レコード数':>10}")
        for m, n_races, n_rows in rows:
            print(f"  {m:>5}分  {n_races:>8,}  {n_rows:>10,}")

    # 3. 券種別
    print(f"\n=== 券種別 (過去7日) ===")
    rows = cur.execute("""
        SELECT oh.odds_type,
               COUNT(DISTINCT oh.race_id) AS n_races,
               COUNT(*)                    AS n_rows
        FROM odds_history oh
        JOIN races r ON oh.race_id = r.race_id
        WHERE r.race_date >= ?
        GROUP BY oh.odds_type
        ORDER BY oh.odds_type
    """, (week_str,)).fetchall()
    if not rows:
        print("  (データなし)")
    else:
        print(f"  {'券種':>10}  {'レース数':>8}  {'レコード数':>10}")
        for t, n_races, n_rows in rows:
            print(f"  {t:>10}  {n_races:>8,}  {n_rows:>10,}")

    # 4. 完全カバレッジ率（4スナップショット全部 取れたレース / 期間全レース）
    print(f"\n=== 完全カバレッジ率 (過去7日) ===")
    total_races = cur.execute("""
        SELECT COUNT(*) FROM races
        WHERE race_date >= ? AND race_date <= ?
    """, (week_str, today_str)).fetchone()[0]

    full_cov = cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT oh.race_id
            FROM odds_history oh
            JOIN races r ON oh.race_id = r.race_id
            WHERE r.race_date >= ? AND r.race_date <= ?
            GROUP BY oh.race_id, oh.odds_type
            HAVING COUNT(DISTINCT oh.minutes_before) = 4
        )
    """, (week_str, today_str)).fetchone()[0]

    print(f"  期間内レース数: {total_races:,}")
    print(f"  4スナップショット完備: {full_cov:,} "
          f"({full_cov/total_races*100:.1f}%)"
          if total_races > 0 else "  期間内レースなし")

    # 5. odds_errors.log 末尾
    print(f"\n=== odds_errors.log (末尾10行) ===")
    if ERRLOG.exists():
        try:
            with ERRLOG.open("r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-10:]
            for line in tail:
                print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  読み込みエラー: {e}")
    else:
        print(f"  ログファイルなし: {ERRLOG}")

    conn.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
