# ===========================================
# scraper/scraper_chariloto_ai_probe.py
# v0.47 Phase 1: ai.chariloto.com 構造調査
#
# 目的: 最小リクエスト(2-3件)でページ構造・データ取得方法を確認
#
# 安全設計:
#   - delay 10秒以上 (調査段階なのでさらに保守的に)
#   - 並列 1
#   - User-Agent 連絡先付き
#   - Referer 設定
#   - API 呼び出しをキャプチャして JSON 取得ルート特定を優先
# ===========================================

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

UA = ("KeirinResearchBot/1.0 (personal research, "
      "contact: momochimax326@gmail.com)")


def probe():
    out = {
        "top_page": {},
        "api_calls": [],
        "archive_check": {},
        "html_samples": {},
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=UA,
            extra_http_headers={
                "Accept-Language": "ja,en;q=0.9",
            },
        )
        page = context.new_page()

        api_calls = []

        def on_response(response):
            try:
                u = response.url
                # 自分ドメイン内 + API or data 系
                if ("ai.chariloto.com" in u or "chariloto" in u) and \
                   any(k in u.lower() for k in ["api", ".json", "race", "odds",
                                                 "predict", "yoso", "ai_"]):
                    try:
                        body = response.text() if response.status == 200 else ""
                    except Exception:
                        body = ""
                    ct = response.headers.get("content-type", "")
                    api_calls.append({
                        "url": u,
                        "status": response.status,
                        "content_type": ct,
                        "body_size": len(body),
                        "body_sample": body[:800] if body else "",
                    })
            except Exception:
                pass

        page.on("response", on_response)

        # 1. トップページ
        print("[1] TOP page fetch...")
        try:
            page.goto("https://ai.chariloto.com/", timeout=30000,
                      wait_until="networkidle")
            time.sleep(10)  # 追加 10 秒 (JS 完全描画待ち、サーバー休憩兼ねる)
            out["top_page"]["title"] = page.title()
            out["top_page"]["url"] = page.url
            html = page.content()
            out["top_page"]["html_size"] = len(html)
            out["html_samples"]["top"] = html[:5000]
            # 印・期待値系の DOM セレクタ探索
            for kw in ["★", "◎", "○", "▲", "注", "△",
                       "期待値", "推奨", "買い目"]:
                count = html.count(kw)
                if count > 0:
                    out["top_page"].setdefault("keyword_counts", {})[kw] = count
        except Exception as e:
            out["top_page"]["error"] = str(e)

        # 10秒待機
        time.sleep(10)

        # 2. 過去レース系 URL パターン試行 (1件のみ)
        print("[2] Archive URL probe...")
        out["archive_check"]["tested_urls"] = []
        archive_urls = [
            "https://ai.chariloto.com/raceresultyear",  # About page から判明
        ]
        for u in archive_urls:
            try:
                page.goto(u, timeout=30000, wait_until="networkidle")
                time.sleep(10)
                title = page.title()
                html = page.content()
                # 過去年度リンクを探す
                year_links = []
                for y in range(2020, 2026):
                    if str(y) in html:
                        year_links.append(y)
                out["archive_check"]["tested_urls"].append({
                    "url": u, "title": title, "html_size": len(html),
                    "years_detected": year_links,
                })
                out["html_samples"]["archive"] = html[:5000]
            except Exception as e:
                out["archive_check"]["tested_urls"].append({
                    "url": u, "error": str(e),
                })

        out["api_calls"] = api_calls
        browser.close()

    return out


def main():
    print("=" * 60)
    print("  v0.47 Phase 1: ai.chariloto.com 構造調査")
    print("=" * 60)
    t0 = time.time()
    result = probe()
    elapsed = time.time() - t0
    print(f"\n調査完了: {elapsed:.1f}秒")

    # 結果保存
    out_json = REPORT_DIR / "v047_phase1_probe.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON保存: {out_json}")

    # 要約
    print(f"\n=== Top page ===")
    print(f"  title: {result['top_page'].get('title', '-')}")
    print(f"  html_size: {result['top_page'].get('html_size', 0):,}")
    if result['top_page'].get('keyword_counts'):
        print(f"  keywords: {result['top_page']['keyword_counts']}")

    print(f"\n=== Archive page ===")
    for item in result['archive_check']['tested_urls']:
        if 'error' in item:
            print(f"  {item['url']}: ERROR {item['error']}")
        else:
            print(f"  {item['url']}: {item.get('title','')}")
            print(f"    size={item.get('html_size',0):,}, years={item.get('years_detected',[])}")

    print(f"\n=== API calls captured ({len(result['api_calls'])}) ===")
    for a in result['api_calls'][:20]:
        print(f"  [{a['status']}] {a['content_type'][:30]:>30} ({a['body_size']}b) {a['url'][:100]}")


if __name__ == "__main__":
    main()
