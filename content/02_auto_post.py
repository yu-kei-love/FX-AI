# ===========================================
# 02_auto_post.py
# 生成された記事を各媒体に自動投稿（または下書き保存）する
# 実行: python content/02_auto_post.py
# ===========================================

import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# プロジェクトルートで .env を読み込む
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
load_dotenv(project_root / ".env")

# パス定義
GENERATED_DIR = script_dir / "generated"
SUBSTACK_DRAFT_DIR = script_dir / "substack_draft"
X_DRAFT_DIR = script_dir / "x_draft"
LOG_FILE = script_dir / "logs" / "post_log.txt"


def get_latest_generated_date():
    """
    content/generated/ 内の YYYY-MM-DD_*.txt から
    最新の日付を1つ返す。該当ファイルがなければ None。
    """
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_(note|substack|x)\.txt$")
    dates = []
    if not GENERATED_DIR.exists():
        return None
    for f in GENERATED_DIR.iterdir():
        if not f.is_file():
            continue
        m = date_pattern.match(f.name)
        if m:
            dates.append(m.group(1))
    return max(dates) if dates else None


def load_generated_files(date_str):
    """
    指定日付の generated ファイルを読み込む。
    返り値: {"note": str or None, "substack": str or None, "x": str or None}
    """
    out = {"note": None, "substack": None, "x": None}
    for key, suffix in [("note", "note"), ("substack", "substack"), ("x", "x")]:
        path = GENERATED_DIR / f"{date_str}_{suffix}.txt"
        if path.exists():
            try:
                out[key] = path.read_text(encoding="utf-8")
            except Exception:
                out[key] = None
    return out


def ensure_dir(path):
    """ディレクトリが存在しなければ作成する"""
    path.mkdir(parents=True, exist_ok=True)


def log(message: str):
    """投稿結果を content/logs/post_log.txt に追記する"""
    ensure_dir(LOG_FILE.parent)
    line = f"[{datetime.now().isoformat()}] {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(message)


def post_wordpress_draft(title: str, content: str, url: str, user: str, password: str) -> bool:
    """
    WordPress REST API で下書き投稿を作成する。
    成功で True、失敗で False。
    """
    try:
        import base64
        import urllib.request
        import json

        endpoint = url.rstrip("/") + "/wp-json/wp/v2/posts"
        credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
        body = json.dumps({"title": title, "content": content, "status": "draft"}).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if 200 <= resp.getcode() < 300:
                return True
            return False
    except Exception as e:
        log(f"WordPress エラー: {e}")
        return False


def run_wordpress(files: dict, date_str: str) -> None:
    """WordPress に note 記事を下書き投稿する。未設定ならスキップ。"""
    url = (os.environ.get("WORDPRESS_URL") or "").strip()
    user = (os.environ.get("WORDPRESS_USER") or "").strip()
    password = (os.environ.get("WORDPRESS_PASSWORD") or "").strip()

    if not url or not user or not password:
        log("WordPress: 未設定のためスキップ（WORDPRESS_URL / USER / PASSWORD）")
        return

    note_content = files.get("note")
    if not note_content:
        log("WordPress: note ファイルがないためスキップ")
        return

    # 1行目をタイトル、2行目以降を本文とする
    lines = note_content.strip().splitlines()
    title = lines[0].strip() if lines else f"FX-AI 検証レポート {date_str}"
    content = "\n".join(lines[1:]).strip() if len(lines) > 1 else note_content

    if post_wordpress_draft(title, content, url, user, password):
        log(f"WordPress: 下書き投稿しました（日付: {date_str}）")
    else:
        log(f"WordPress: 下書き投稿に失敗しました（日付: {date_str}）")


def run_substack_draft(files: dict, date_str: str) -> None:
    """Substack メール原稿を content/substack_draft/ に保存する。"""
    try:
        ensure_dir(SUBSTACK_DRAFT_DIR)
        content = files.get("substack")
        if not content:
            log("Substack: 原稿なしのためスキップ")
            return
        path = SUBSTACK_DRAFT_DIR / f"{date_str}_substack_draft.txt"
        path.write_text(content, encoding="utf-8")
        log(f"Substack: 原稿を保存しました {path}")
    except Exception as e:
        log(f"Substack エラー: {e}")


def run_x_draft(files: dict, date_str: str) -> None:
    """X（Twitter）投稿文を content/x_draft/ に保存する。"""
    try:
        ensure_dir(X_DRAFT_DIR)
        content = files.get("x")
        if not content:
            log("X: 投稿文なしのためスキップ")
            return
        path = X_DRAFT_DIR / f"{date_str}_x_draft.txt"
        path.write_text(content, encoding="utf-8")
        log(f"X: 投稿文を保存しました {path}")
    except Exception as e:
        log(f"X エラー: {e}")


def main():
    date_str = get_latest_generated_date()
    if not date_str:
        log("対象なし: content/generated/ に YYYY-MM-DD_note/substack/x.txt がありません")
        return

    files = load_generated_files(date_str)
    log(f"--- 自動投稿開始（日付: {date_str}）---")

    # 各媒体を順に実行（エラーが出ても他を続行）
    run_wordpress(files, date_str)
    run_substack_draft(files, date_str)
    run_x_draft(files, date_str)

    log("--- 自動投稿処理完了 ---")


if __name__ == "__main__":
    main()
