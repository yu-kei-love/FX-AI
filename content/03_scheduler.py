# ===========================================
# 03_scheduler.py
# データ取得・モデル再学習・記事生成・下書き保存・Optuna最適化を自動スケジュールで実行する
# 実行: python content/03_scheduler.py
# 停止: Ctrl+C
# ===========================================

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import schedule

# プロジェクトルート（実行時のカレントディレクトリをプロジェクトルートにしたいため）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# ログファイル・各種パス
LOG_FILE = script_dir / "logs" / "scheduler_log.txt"
ACCURACY_HISTORY_FILE = script_dir / "logs" / "accuracy_history.csv"
GENERATED_DIR = script_dir / "generated"
X_DRAFT_DIR = script_dir / "x_draft"


def ensure_dir(path: Path) -> None:
    """任意のディレクトリが存在しなければ作成する"""
    path.mkdir(parents=True, exist_ok=True)


def ensure_log_dir() -> None:
    """ログ用ディレクトリが存在しなければ作成する"""
    ensure_dir(LOG_FILE.parent)


def scheduler_log(message: str) -> None:
    """スケジューラの開始・完了・エラーを logs/scheduler_log.txt に追記する"""
    ensure_log_dir()
    line = f"[{datetime.now().isoformat()}] {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(message)


def run_script(rel_path: str, job_name: str) -> bool:
    """
    指定したスクリプトをサブプロセスで実行する。
    成功で True、失敗で False。エラーが出ても例外は握りつぶし、ログに記録する。
    """
    scheduler_log(f"開始: {job_name} ({rel_path})")
    try:
        result = subprocess.run(
            [sys.executable, rel_path],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600,
            encoding="utf-8",
        )
        if result.returncode == 0:
            scheduler_log(f"完了: {job_name}")
            return True
        else:
            err = (result.stderr or result.stdout or "")[:500]
            scheduler_log(f"失敗: {job_name} (exit code {result.returncode}) - {err}")
            return False
    except subprocess.TimeoutExpired:
        scheduler_log(f"失敗: {job_name} (タイムアウト)")
        return False
    except Exception as e:
        scheduler_log(f"失敗: {job_name} - {e}")
        return False


def run_script_capture_stdout(rel_path: str, job_name: str) -> Tuple[bool, str]:
    """
    stdout を呼び出し元で解析したい場合のラッパー。
    成功/失敗と stdout を返す。
    """
    scheduler_log(f"開始: {job_name} ({rel_path})")
    try:
        result = subprocess.run(
            [sys.executable, rel_path],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=7200,
            encoding="utf-8",
        )
        stdout = result.stdout or ""
        if result.returncode == 0:
            scheduler_log(f"完了: {job_name}")
            return True, stdout
        else:
            err = (result.stderr or stdout or "")[:500]
            scheduler_log(f"失敗: {job_name} (exit code {result.returncode}) - {err}")
            return False, stdout
    except subprocess.TimeoutExpired:
        scheduler_log(f"失敗: {job_name} (タイムアウト)")
        return False, ""
    except Exception as e:
        scheduler_log(f"失敗: {job_name} - {e}")
        return False, ""


def read_last_accuracy() -> Optional[float]:
    """accuracy_history.csv から直近の精度を読み取る（なければ None）"""
    if not ACCURACY_HISTORY_FILE.exists():
        return None
    try:
        last_line = None
        with open(ACCURACY_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line or last_line.startswith("date,accuracy, status"):
            return None
        parts = [p.strip() for p in last_line.split(",")]
        if len(parts) < 2:
            return None
        return float(parts[1])
    except Exception:
        return None


def append_accuracy_history(date_str: str, accuracy: Optional[float], status: str, reason: str) -> None:
    """精度履歴を accuracy_history.csv に追記する（日付・精度・採用/スキップ理由）"""
    ensure_log_dir()
    if not ACCURACY_HISTORY_FILE.exists():
        # ヘッダー行を追加
        with open(ACCURACY_HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write("date,accuracy,status,reason\n")
    acc_str = "" if accuracy is None else f"{accuracy:.6f}"
    with open(ACCURACY_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{date_str},{acc_str},{status},{reason}\n")


def parse_accuracy_from_stdout(stdout: str) -> Optional[float]:
    """14_main_system.py の標準出力から「正解率」をパースする"""
    import re

    for line in stdout.splitlines():
        if "正解率" in line:
            # 例: "  正解率: 0.5123 (51.23%)"
            m = re.search(r"正解率:\\s*([0-9.]+)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
    return None


def job_daily_data_fetch() -> None:
    """毎日 00:00 実行: 新しいドル円データを取得（research/01_data_fetch.py）"""
    run_script("research/01_data_fetch.py", "データ取得（01_data_fetch.py）")


def job_daily_retrain() -> None:
    """
    毎日 00:30 実行:
    ・モデルを再学習（research/14_main_system.py）
    ・新旧モデルの精度を比較し、2%以上悪化したら「旧モデルを維持」として記録
    ・Purged CVスコア（ここではテスト正解率を代理指標とする）が48%未満ならスキップとして記録
    """
    today = datetime.now().strftime("%Y-%m-%d")
    old_acc = read_last_accuracy()

    success, stdout = run_script_capture_stdout(
        "research/14_main_system.py",
        "モデル再学習（14_main_system.py）",
    )
    if not success:
        reason = "再学習スクリプトがエラー終了したためスキップ扱い。"
        scheduler_log(f"再学習スキップ理由: {reason}")
        append_accuracy_history(today, None, "error", reason)
        return

    new_acc = parse_accuracy_from_stdout(stdout)
    if new_acc is None:
        reason = "正解率を標準出力から取得できなかったためスキップ扱い。"
        scheduler_log(f"再学習スキップ理由: {reason}")
        append_accuracy_history(today, None, "parse_error", reason)
        return

    # Purged CV スコアの代理としてテスト正解率を使用し、0.48 未満ならスキップ
    if new_acc < 0.48:
        reason = f"Purged CVスコア（テスト正解率）{new_acc:.4f} が 0.48 を下回ったため再学習結果を採用しない。"
        scheduler_log(f"再学習スキップ理由: {reason}")
        append_accuracy_history(today, new_acc, "skipped_purged_cv_below_0_48", reason)
        return

    # 旧モデルと比較して 2%以上悪化していないかを判定
    if old_acc is not None and new_acc < old_acc - 0.02:
        reason = (
            f"新モデル精度 {new_acc:.4f} が旧モデル {old_acc:.4f} より 2%以上悪化しているため "
            "旧モデルを維持（新モデルは採用しない）。"
        )
        scheduler_log(f"再学習スキップ理由: {reason}")
        append_accuracy_history(today, new_acc, "skipped_worse_than_old", reason)
        return

    # 採用
    reason = "新モデルの精度が条件を満たしたため採用。"
    scheduler_log(f"再学習採用: {reason} (accuracy={new_acc:.4f})")
    append_accuracy_history(today, new_acc, "adopted", reason)


def get_latest_generated_date() -> Optional[str]:
    """
    content/generated/ 内の YYYY-MM-DD_*.txt から
    最新の日付を1つ返す。該当ファイルがなければ None。
    """
    import re

    if not GENERATED_DIR.exists():
        return None
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_(note|substack|x)\.txt$")
    dates = []
    for f in GENERATED_DIR.iterdir():
        if not f.is_file():
            continue
        m = date_pattern.match(f.name)
        if m:
            dates.append(m.group(1))
    return max(dates) if dates else None


def load_generated_files(date_str: str) -> dict:
    """
    指定日付の generated ファイルを読み込む。
    返り値: {\"note\": str or None, \"substack\": str or None, \"x\": str or None}
    """
    out = {\"note\": None, \"substack\": None, \"x\": None}
    for key, suffix in [(\"note\", \"note\"), (\"substack\", \"substack\"), (\"x\", \"x\")]:
        path = GENERATED_DIR / f\"{date_str}_{suffix}.txt\"
        if path.exists():
            try:
                out[key] = path.read_text(encoding=\"utf-8\")
            except Exception as e:
                scheduler_log(f\"generated 読み込みエラー ({path}): {e}\")
                out[key] = None
    return out


def post_wordpress_draft(title: str, content: str, url: str, user: str, password: str) -> bool:
    """
    WordPress REST API で下書き投稿を作成する。
    成功で True、失敗で False。
    """
    try:
        import base64
        import urllib.request
        import json

        endpoint = url.rstrip(\"/\") + \"/wp-json/wp/v2/posts\"
        credentials = base64.b64encode(f\"{user}:{password}\".encode()).decode()
        body = json.dumps({\"title\": title, \"content\": content, \"status\": \"draft\"}).encode(\"utf-8\")

        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                \"Authorization\": f\"Basic {credentials}\",
                \"Content-Type\": \"application/json\",\"
            },
            method=\"POST\",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.getcode() < 300
    except Exception as e:
        scheduler_log(f\"WordPress エラー: {e}\")
        return False


def job_daily_x_only() -> None:
    """
    毎日 01:30 実行:
    ・X投稿文のみ自動生成・保存する（短文のみ）
    """
    # まず記事を生成（X用短文を含む）
    run_script(\"content/01_content_generator.py\", \"X投稿文生成（01_content_generator.py）\")

    date_str = get_latest_generated_date()
    if not date_str:
        scheduler_log(\"X: generated ファイルが見つからないためスキップ\")
        return

    files = load_generated_files(date_str)
    x_content = files.get(\"x\")
    if not x_content:
        scheduler_log(\"X: 投稿文ファイルがないためスキップ\")
        return

    ensure_dir(X_DRAFT_DIR)
    out_path = X_DRAFT_DIR / f\"{date_str}_x_draft.txt\"
    try:
        out_path.write_text(x_content, encoding=\"utf-8\")
        scheduler_log(f\"X: 投稿文を保存しました {out_path}\")
    except Exception as e:
        scheduler_log(f\"X: 投稿文保存エラー {e}\")


def job_daily_wordpress_seo() -> None:
    """
    毎日 02:00 実行:
    ・WordPress用SEO記事を自動生成・下書き投稿する
      （現状は note 記事を元に下書き投稿）
    """
    # note を含む記事を生成
    run_script(\"content/01_content_generator.py\", \"WordPress用記事生成（01_content_generator.py）\")

    date_str = get_latest_generated_date()
    if not date_str:
        scheduler_log(\"WordPress: generated ファイルが見つからないためスキップ\")
        return

    files = load_generated_files(date_str)
    note_content = files.get(\"note\")
    if not note_content:
        scheduler_log(\"WordPress: note ファイルがないためスキップ\")
        return

    import os

    url = (os.environ.get(\"WORDPRESS_URL\") or \"\").strip()
    user = (os.environ.get(\"WORDPRESS_USER\") or \"\").strip()
    password = (os.environ.get(\"WORDPRESS_PASSWORD\") or \"\").strip()

    if not url or not user or not password:
        scheduler_log(\"WordPress: 環境変数未設定のためスキップ (WORDPRESS_URL / USER / PASSWORD)\")
        return

    lines = note_content.strip().splitlines()
    title = lines[0].strip() if lines else f\"FX-AI SEO記事 {date_str}\"
    content = \"\\n\".join(lines[1:]).strip() if len(lines) > 1 else note_content

    if post_wordpress_draft(title, content, url, user, password):
        scheduler_log(f\"WordPress: SEO記事を下書き投稿しました（日付: {date_str}）\")
    else:
        scheduler_log(f\"WordPress: SEO記事の下書き投稿に失敗しました（日付: {date_str}）\")


def job_daily_stock_paper_trade() -> None:
    """毎日 07:30 実行: 日本株ペーパートレード v2（research/paper_trade_stocks.py）
    米国市場終了後・日本市場開場前に予測を実行する。
    """
    run_script("research/paper_trade_stocks.py", "日本株ペーパートレード v2（paper_trade_stocks.py）")


def job_weekly_optuna_and_long_articles() -> None:
    """
    毎週月曜 03:00 実行:
    ・Optunaでパラメータ最適化（research/11_optuna_optimize.py）
    ・note用長文記事を自動生成・保存
    ・Substack原稿を自動生成・保存
      （content/01_content_generator.py が generated/ に保存する）
    """
    run_script(\"research/11_optuna_optimize.py\", \"Optuna最適化（11_optuna_optimize.py）\")
    run_script(\"content/01_content_generator.py\", \"長文記事・Substack原稿生成（01_content_generator.py）\")


def main() -> None:
    # スケジュール登録
    schedule.every().day.at("00:00").do(job_daily_data_fetch)
    schedule.every().day.at("00:30").do(job_daily_retrain)
    schedule.every().day.at("01:30").do(job_daily_x_only)
    schedule.every().day.at("02:00").do(job_daily_wordpress_seo)
    schedule.every().day.at("07:30").do(job_daily_stock_paper_trade)
    schedule.every().monday.at("03:00").do(job_weekly_optuna_and_long_articles)

    scheduler_log("スケジューラを起動しました。")
    print("スケジューラ起動中（Ctrl+C で停止）")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler_log("スケジューラを手動停止しました。")
        print("\nスケジューラを停止しました。")


if __name__ == "__main__":
    main()
