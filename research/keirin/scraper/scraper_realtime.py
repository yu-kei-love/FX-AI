# ===========================================
# scraper/scraper_realtime.py
# 競輪 - リアルタイムデータスクレイパー
#
# 取得タイミング：
#   2時間前 → オッズ取得
#   60分前  → オッズ取得
#   30分前  → オッズ取得
#   15分前  → オッズ取得 + ライン更新
#   5分前   → オッズ取得（急変監視）
#   1分前   → オッズ取得（最終確認）
#   確定    → 確定オッズ取得
#
# 注意：このファイルはスクレイピングコードのため
#       note販売パッケージには含めない
# ===========================================

import time
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keirin" / "keirin.db"

SLEEP_MIN = 1.0
SLEEP_MAX = 2.0

ODDS_TIMINGS = ["120min", "60min", "30min", "15min", "5min", "1min", "final"]


def _polite_sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def fetch_current_odds(race_id: str, timing: str) -> dict:
    """
    現在のオッズを取得してDBに保存する。

    Parameters:
        race_id : レースID
        timing  : "120min"/"60min"/"30min"/"15min"/"5min"/"1min"/"final"

    Returns:
        {car_no: {"win_odds": float, "trifecta_odds": float}}
    """
    raise NotImplementedError(
        "fetch_current_odds: 実装待ち（データソース選定後に実装）"
    )


def fetch_current_lines(race_id: str) -> list:
    """
    現在のライン情報を取得する（15分前に更新）。

    記者予想や最新の選手コメントを参照する。

    Returns:
        [{"line_str": "3-7-4", "confidence": 0.80, "source": "gamboo"}]
    """
    raise NotImplementedError(
        "fetch_current_lines: 実装待ち（データソース選定後に実装）"
    )


def detect_odds_surge(
    race_id: str,
    car_no: int,
    prev_timing: str = "15min",
    curr_timing: str = "1min",
) -> dict:
    """
    オッズ急変を検出する。

    ボートレースと同じ設計：
    - ≥20%上昇 → 買いシグナル取り消し
    - 10〜20%上昇 → 信頼度50%カット
    - 5〜10%上昇  → 信頼度20%カット

    Returns:
        {"change_rate": float, "action": "cancel"|"reduce_50"|"reduce_20"|"ok"}
    """
    if not DB_PATH.exists():
        return {"change_rate": 0.0, "action": "ok"}

    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    cur.execute(
        "SELECT win_odds FROM odds WHERE race_id=? AND car_no=? AND timing=?",
        (race_id, car_no, prev_timing),
    )
    prev_row = cur.fetchone()

    cur.execute(
        "SELECT win_odds FROM odds WHERE race_id=? AND car_no=? AND timing=?",
        (race_id, car_no, curr_timing),
    )
    curr_row = cur.fetchone()
    conn.close()

    if not prev_row or not curr_row or prev_row[0] <= 0:
        return {"change_rate": 0.0, "action": "ok"}

    change_rate = (curr_row[0] - prev_row[0]) / prev_row[0]

    if change_rate >= 0.20:
        action = "cancel"
    elif change_rate >= 0.10:
        action = "reduce_50"
    elif change_rate >= 0.05:
        action = "reduce_20"
    else:
        action = "ok"

    return {"change_rate": round(change_rate, 4), "action": action}


def run_realtime_monitoring(race_id: str):
    """
    1レース分のリアルタイム監視を実行する。

    各タイミングでオッズを取得し、急変を監視する。
    """
    print(f"[MONITOR] {race_id} のリアルタイム監視開始")

    for timing in ODDS_TIMINGS:
        try:
            fetch_current_odds(race_id, timing)
            if timing == "15min":
                fetch_current_lines(race_id)
            print(f"  [{timing}] オッズ取得完了")
        except NotImplementedError as e:
            print(f"  未実装: {e}")
            break
        except Exception as e:
            print(f"  エラー [{timing}]: {e}")

        _polite_sleep()


if __name__ == "__main__":
    print("リアルタイムスクレイパー - 動作確認")
    print("※ fetch_current_odds は実装待ち")
