# Phase 1 再調査: HTML full dump + 予想データ構造解析
import json, time, re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

UA = ("KeirinResearchBot/1.0 (personal research, "
      "contact: momochimax326@gmail.com)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=UA,
            extra_http_headers={"Accept-Language": "ja,en;q=0.9"},
        )
        page = ctx.new_page()

        # 1. トップページ: full HTML を取得、予想データ要素を CSS セレクタで抽出
        print("[1] Top page with full HTML dump...")
        page.goto("https://ai.chariloto.com/", timeout=30000,
                  wait_until="networkidle")
        time.sleep(10)
        html_top = page.content()
        (REPORT_DIR / "v047_top_full.html").write_text(html_top, encoding="utf-8")
        print(f"  size: {len(html_top):,}")

        # 予想データ要素: star/mark/expected_value の CSS セレクタを探索
        # 典型的な Next.js の element
        try:
            # ★ を含む要素
            star_elems = page.query_selector_all("text=★")
            print(f"  ★ elements: {len(star_elems)}")
            for i, el in enumerate(star_elems[:5]):
                txt = el.text_content()
                html_ctx = el.evaluate("(el) => el.outerHTML")
                print(f"    [{i}] text='{txt}' html={html_ctx[:200]}")
        except Exception as e:
            print(f"  selector err: {e}")

        time.sleep(10)

        # 2. 過去レースリンク(raceresultyear)探索: year link をクリックして年度別取得
        print("\n[2] /raceresultyear full HTML...")
        page.goto("https://ai.chariloto.com/raceresultyear", timeout=30000,
                  wait_until="networkidle")
        time.sleep(10)
        html_arch = page.content()
        (REPORT_DIR / "v047_arch_full.html").write_text(html_arch, encoding="utf-8")
        print(f"  size: {len(html_arch):,}")

        # href 内の URL パターン抽出
        hrefs = re.findall(r'href="(/[^"#]+)"', html_arch)
        internal_hrefs = sorted(set(hrefs))
        print(f"  internal hrefs ({len(internal_hrefs)}):")
        for h in internal_hrefs[:40]:
            print(f"    {h}")

        browser.close()

    # トップ HTML から構造分析
    print("\n[ANALYSIS] Top page structure")
    # ★ 周辺を HTML タグ構造で
    for m in list(re.finditer(r'★', html_top))[:3]:
        idx = m.start()
        # 前 300, 後 500 で HTML snippet
        snippet = html_top[max(0, idx-300):idx+500]
        print(f"\n  ★@{idx}:")
        print(f"    {snippet[:600]}")


if __name__ == "__main__":
    main()
