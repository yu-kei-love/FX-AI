# ===========================================
# setup_scheduler.py
# Windowsタスクスケジューラ 自動設定スクリプト
#
# 処理内容：
#   1. 毎朝08:00 → scraper_daily.py（出走表一括取得）
#   2. 当日の各レース × 7タイミング → scraper_realtime.py を自動登録
#   3. 翌日分のタスクを毎日23:00に設定
#
# 実行方法：
#   python research/boat/setup_scheduler.py --setup-daily
#     → 毎朝08:00の日次バッチとセットアップタスクを登録
#
#   python research/boat/setup_scheduler.py --setup-today
#     → 今日のレース分のリアルタイムタスクを今すぐ登録
#
# 注意: 管理者権限が必要です（schtasks コマンドを使用）
# ===========================================

import re
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "data" / "boat" / "setup_scheduler.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# Python実行ファイルのパス
PYTHON_EXE = sys.executable

# タスク名のプレフィックス
TASK_PREFIX = "BoatraceAI"

# scraper_realtime.pyの相対パス
REALTIME_SCRIPT = PROJECT_ROOT / "research" / "boat" / "scraper_realtime.py"
DAILY_SCRIPT    = PROJECT_ROOT / "research" / "boat" / "scraper_daily.py"
SETUP_SCRIPT    = PROJECT_ROOT / "research" / "boat" / "setup_scheduler.py"

# タイミング別のレース締め切り前の分数（マイナスが締め切り前）
TIMING_OFFSETS = {
    "120min": -120,
    "60min":  -60,
    "30min":  -30,
    "15min":  -15,
    "5min":   -5,
    "1min":   -1,
    "final":  +5,    # 締め切り後5分（結果確定後）
}


# =============================================================
# Windowsタスクスケジューラ操作
# =============================================================

def run_schtasks(args_list):
    """schtasksコマンドを実行する。"""
    cmd = ["schtasks"] + args_list
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            logger.warning(f"schtasks失敗: {' '.join(cmd)}\n{result.stderr}")
            return False
        return True
    except FileNotFoundError:
        logger.error("schtasksコマンドが見つかりません（Windows専用）")
        return False


def create_task(task_name, command, run_time, description=""):
    """
    Windowsタスクスケジューラにタスクを登録する。

    Parameters:
        task_name (str): タスク名
        command   (str): 実行コマンド
        run_time  (str): 実行時刻 "HH:MM"
        description (str): 説明
    """
    # 既存タスクを削除
    run_schtasks(["/Delete", "/TN", task_name, "/F"])

    # 新規登録
    success = run_schtasks([
        "/Create",
        "/TN", task_name,
        "/TR", command,
        "/SC", "ONCE",
        "/ST", run_time,
        "/F",
    ])

    if success:
        logger.info(f"  タスク登録: {task_name} @ {run_time}")
    else:
        logger.error(f"  タスク登録失敗: {task_name}")

    return success


def create_daily_task(task_name, command, run_time):
    """毎日繰り返すタスクを登録する。"""
    # 既存タスクを削除
    run_schtasks(["/Delete", "/TN", task_name, "/F"])

    success = run_schtasks([
        "/Create",
        "/TN", task_name,
        "/TR", command,
        "/SC", "DAILY",
        "/ST", run_time,
        "/F",
    ])

    if success:
        logger.info(f"  日次タスク登録: {task_name} @ {run_time} (毎日)")
    else:
        logger.error(f"  日次タスク登録失敗: {task_name}")

    return success


def delete_task(task_name):
    """タスクを削除する。"""
    return run_schtasks(["/Delete", "/TN", task_name, "/F"])


def list_tasks(prefix=TASK_PREFIX):
    """登録済みのBoatraceAI関連タスクを一覧表示する。"""
    result = subprocess.run(
        ["schtasks", "/Query", "/FO", "LIST", "/V"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    lines = result.stdout.split("\n")
    tasks = []
    current_task = {}
    for line in lines:
        if line.startswith("タスク名:") or line.startswith("TaskName:"):
            name = line.split(":", 1)[1].strip()
            if prefix in name:
                current_task = {"name": name}
        elif ("次回の実行時刻:" in line or "Next Run Time:" in line) and current_task:
            current_task["next_run"] = line.split(":", 1)[1].strip()
        elif ("状態:" in line or "Status:" in line) and current_task:
            current_task["status"] = line.split(":", 1)[1].strip()
            tasks.append(current_task)
            current_task = {}

    if tasks:
        print(f"\n登録済み {prefix} タスク ({len(tasks)}件):")
        for t in tasks:
            print(f"  {t.get('name', '?')}")
            print(f"    次回実行: {t.get('next_run', '?')}")
            print(f"    状態: {t.get('status', '?')}")
    else:
        print(f"{prefix} タスクは登録されていません")

    return tasks


# =============================================================
# レーススケジュール取得
# =============================================================

def get_race_schedule_from_db(date_str, db_path=None):
    """
    DBから当日のレーススケジュール（時刻）を取得する。

    Returns:
        list[dict]: [{venue_code, venue_id, race_no, race_time}, ...]
    """
    from research.boat.db_manager import get_conn, DB_PATH
    path = db_path or DB_PATH

    try:
        with get_conn(path) as conn:
            rows = conn.execute("""
                SELECT r.venue_id, r.race_no, r.race_time
                FROM races r
                WHERE r.date = ?
                AND r.race_time IS NOT NULL
                ORDER BY r.race_time, r.venue_id, r.race_no
            """, (date_str,)).fetchall()
    except Exception as e:
        logger.warning(f"DBからスケジュール取得失敗: {e}")
        return []

    schedule = []
    for row in rows:
        venue_id = row[0]
        race_no  = row[1]
        race_time = row[2]  # "HH:MM"

        # venue_code は venue_id から逆引き
        venue_code = str(venue_id).zfill(2)

        schedule.append({
            "venue_code": venue_code,
            "venue_id": venue_id,
            "race_no": race_no,
            "race_time": race_time,
        })

    return schedule


def parse_race_time(race_time_str, date_str):
    """
    レース時刻文字列をdatetimeオブジェクトに変換する。

    Parameters:
        race_time_str (str): "HH:MM"
        date_str (str): "YYYYMMDD"

    Returns:
        datetime
    """
    dt = datetime.strptime(date_str + " " + race_time_str, "%Y%m%d %H:%M")
    return dt


# =============================================================
# セットアップ処理
# =============================================================

def setup_daily_tasks():
    """
    常時起動タスクを登録する：
    1. 毎朝08:00: scraper_daily.py
    2. 毎日23:00: setup_scheduler.py --setup-today（翌日分を設定）
    """
    logger.info("=== 日次・セットアップタスク登録 ===")

    # 1. 毎朝08:00 出走表バッチ
    daily_cmd = f'"{PYTHON_EXE}" "{DAILY_SCRIPT}"'
    create_daily_task(
        task_name=f"{TASK_PREFIX}\\DailyBatch",
        command=daily_cmd,
        run_time="08:00",
    )

    # 2. 毎日23:00 翌日のリアルタイムタスク設定
    setup_cmd = f'"{PYTHON_EXE}" "{SETUP_SCRIPT}" --setup-today --tomorrow'
    create_daily_task(
        task_name=f"{TASK_PREFIX}\\SetupScheduler",
        command=setup_cmd,
        run_time="23:00",
    )

    logger.info("=== 日次タスク登録完了 ===")
    logger.info("  毎朝08:00: 出走表バッチ")
    logger.info("  毎日23:00: 翌日のリアルタイムタスク設定")


def setup_realtime_tasks(date_str=None, db_path=None):
    """
    指定日のレース × 7タイミング のリアルタイムタスクを登録する。

    Parameters:
        date_str (str|None): YYYYMMDD（Noneなら今日）
        db_path  (Path|None): DBパス
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    logger.info(f"=== リアルタイムタスク登録 {date_str} ===")

    schedule = get_race_schedule_from_db(date_str, db_path)

    if not schedule:
        logger.warning(f"DBにレーススケジュールがありません ({date_str})")
        logger.warning("先に scraper_daily.py を実行してください")
        return 0

    registered = 0
    skipped = 0
    now = datetime.now()

    for race_info in schedule:
        venue_code = race_info["venue_code"]
        race_no    = race_info["race_no"]
        race_time_str = race_info["race_time"]

        if not race_time_str:
            continue

        try:
            deadline_dt = parse_race_time(race_time_str, date_str)
        except ValueError:
            logger.warning(f"時刻パース失敗: {race_time_str}")
            continue

        for timing, offset_min in TIMING_OFFSETS.items():
            # タスク実行時刻を計算
            task_dt = deadline_dt + timedelta(minutes=offset_min)

            # 過去のタスクはスキップ
            if task_dt <= now:
                skipped += 1
                continue

            task_time_str = task_dt.strftime("%H:%M")
            task_name = (
                f"{TASK_PREFIX}\\{date_str}_{venue_code}"
                f"_R{race_no:02d}_{timing}"
            )

            cmd = (
                f'"{PYTHON_EXE}" "{REALTIME_SCRIPT}" '
                f'--venue {venue_code} --race {race_no} '
                f'--timing {timing} --date {date_str}'
            )

            success = create_task(
                task_name=task_name,
                command=cmd,
                run_time=task_time_str,
            )

            if success:
                registered += 1

    logger.info(f"=== リアルタイムタスク登録完了 ===")
    logger.info(f"  登録: {registered}件, スキップ（過去）: {skipped}件")
    return registered


def cleanup_old_tasks(days_back=3):
    """
    数日前のリアルタイムタスクを削除する（タスクスケジューラの整理）。
    """
    logger.info(f"=== 古いタスクの削除 ({days_back}日前以前) ===")

    for i in range(days_back, 30):
        old_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        # 日付パターンに一致するタスクを削除
        result = subprocess.run(
            ["schtasks", "/Query", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        for line in result.stdout.split("\n"):
            if old_date in line and TASK_PREFIX in line:
                # タスク名を抽出
                parts = line.split(",")
                if parts:
                    task_name = parts[0].strip('"')
                    if delete_task(task_name):
                        logger.info(f"  削除: {task_name}")


# =============================================================
# メイン
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ボートレースAI タスクスケジューラ設定"
    )
    parser.add_argument("--setup-daily", action="store_true",
                        help="日次バッチとセットアップタスクを登録（初回のみ実行）")
    parser.add_argument("--setup-today", action="store_true",
                        help="今日のリアルタイムタスクを登録")
    parser.add_argument("--tomorrow", action="store_true",
                        help="--setup-today と組み合わせて翌日分を設定")
    parser.add_argument("--list", action="store_true",
                        help="登録済みタスクを一覧表示")
    parser.add_argument("--cleanup", action="store_true",
                        help="古いタスクを削除")
    parser.add_argument("--date", default=None,
                        help="対象日 YYYYMMDD")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    if args.cleanup:
        cleanup_old_tasks()
        return

    if args.setup_daily:
        setup_daily_tasks()
        return

    if args.setup_today:
        if args.tomorrow:
            target = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        elif args.date:
            target = args.date
        else:
            target = datetime.now().strftime("%Y%m%d")
        setup_realtime_tasks(target)
        return

    # 引数なしの場合: セットアップガイドを表示
    print("""
ボートレースAI タスクスケジューラ設定

【初回セットアップ（一度だけ実行）】
  python research/boat/setup_scheduler.py --setup-daily

  → 以下を毎日自動実行するよう登録されます：
    08:00: 出走表バッチ（scraper_daily.py）
    23:00: 翌日のリアルタイムタスク設定

【手動で今日分のリアルタイムタスクを登録】
  python research/boat/setup_scheduler.py --setup-today

【登録済みタスクの確認】
  python research/boat/setup_scheduler.py --list

【古いタスクの削除】
  python research/boat/setup_scheduler.py --cleanup

注意: Windowsの管理者権限で実行してください
    """)


if __name__ == "__main__":
    main()
