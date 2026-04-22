# ===========================================
# v0.44: 記者予想 (Gamboo) の的中率分析
#
# reporter_predictions の predicted_line:
#   "4-2,1,3-5" 形式
#   - 先頭 "4" = 本命 1着
#   - "-" 以降 "2,1,3" = 2着候補
#   - 次の "-" 以降 "5" = 3着候補
#   (複数 "," で並列候補)
#
# 評価項目:
#   - 本命 1着的中率
#   - 本命 3着以内率
#   - 推奨ライン (先頭 3 候補) 内の 1着的中率
#   - トップ 2-3-4 通りの的中率
# ===========================================

import json
import re
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "model"))
from feature_engine import DB_PATH

REPORT_DIR = PROJECT_DIR.parent.parent / "data" / "keirin"


def parse_line(line_str):
    """
    '4-2,1,3-5' → {
        'honmei': 4,
        'ni_chaku_candidates': [2, 1, 3],
        'san_chaku_candidates': [5],
    }
    '4=2-1,3' など形式バリエーションもあり、'-' で分割
    """
    if not isinstance(line_str, str) or not line_str:
        return None
    # 全角/半角統一
    s = line_str.strip()
    # 区切り "-" で分割 → 1着/2着/3着
    # "," 内は 複数候補
    parts = re.split(r"[-=]", s)
    if not parts:
        return None
    try:
        # 1着: 先頭部の最初の数字
        first_part = parts[0].strip()
        nums_first = [int(x) for x in re.findall(r"\d+", first_part)]
        if not nums_first:
            return None
        honmei = nums_first[0]
        # 2着候補
        ni = []
        if len(parts) >= 2:
            ni = [int(x) for x in re.findall(r"\d+", parts[1])]
        # 3着候補
        san = []
        if len(parts) >= 3:
            san = [int(x) for x in re.findall(r"\d+", parts[2])]
        return {
            "honmei": honmei,
            "ni": ni,
            "san": san,
            "recommended_set": [honmei] + ni + san,  # 全候補
        }
    except (ValueError, IndexError):
        return None


def analyze():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 予想と結果を JOIN
    rows = cur.execute("""
        SELECT rp.race_id, rp.predicted_line,
               r1.sha_ban AS sha_1,
               r2.sha_ban AS sha_2,
               r3.sha_ban AS sha_3,
               rc.is_midnight
        FROM reporter_predictions rp
        JOIN results r1 ON rp.race_id = r1.race_id AND r1.rank = 1
        LEFT JOIN results r2 ON rp.race_id = r2.race_id AND r2.rank = 2
        LEFT JOIN results r3 ON rp.race_id = r3.race_id AND r3.rank = 3
        JOIN races rc ON rp.race_id = rc.race_id
        WHERE rp.reporter_name = 'gamboo'
          AND rp.predicted_line IS NOT NULL
    """).fetchall()
    conn.close()

    print(f"Gamboo 予想 × 結果付きレース: {len(rows):,}")

    n_total = 0
    n_honmei_1st = 0
    n_honmei_top3 = 0
    n_recset_1st = 0
    n_recset_top3 = 0
    n_top3_1st = 0
    n_top3_top3 = 0
    n_top4_1st = 0
    n_ni_top2 = 0    # 2着候補に 実 2着が含まれるか
    parse_fail = 0

    # midnight 分離
    normal_n = 0
    normal_honmei_1st = 0
    mid_n = 0
    mid_honmei_1st = 0

    for race_id, line, s1, s2, s3, is_mid in rows:
        parsed = parse_line(line)
        if parsed is None:
            parse_fail += 1
            continue
        n_total += 1
        honmei = parsed["honmei"]
        rec_set = set(parsed["recommended_set"])
        top3 = [honmei] + parsed["ni"][:1] + parsed["san"][:1]
        top4 = rec_set  # 全候補 (typically ~5-7)

        actuals = [x for x in [s1, s2, s3] if x is not None]

        # honmei
        if honmei == s1:
            n_honmei_1st += 1
        if honmei in actuals:
            n_honmei_top3 += 1
        # rec_set (all candidates)
        if s1 in rec_set:
            n_recset_1st += 1
        if any(a in rec_set for a in actuals):
            n_recset_top3 += 1
        # top3 推奨
        if s1 in top3:
            n_top3_1st += 1
        # 2 着候補の 2 着的中
        if s2 is not None and s2 in parsed["ni"]:
            n_ni_top2 += 1

        # midnight 分離
        if is_mid == 1:
            mid_n += 1
            if honmei == s1:
                mid_honmei_1st += 1
        else:
            normal_n += 1
            if honmei == s1:
                normal_honmei_1st += 1

    if n_total == 0:
        print("データなし")
        return

    print(f"\n=== 解析対象 {n_total:,} レース (parse 失敗 {parse_fail}) ===")
    print(f"\n本命 (先頭 1車):")
    print(f"  1着的中: {n_honmei_1st:,} / {n_total:,} = {n_honmei_1st/n_total*100:.2f}%")
    print(f"  3着以内: {n_honmei_top3:,} / {n_total:,} = {n_honmei_top3/n_total*100:.2f}%")

    print(f"\n全推奨ライン (本命+2着候補+3着候補、typically 5-7 車):")
    print(f"  1着含有率: {n_recset_1st:,} / {n_total:,} = {n_recset_1st/n_total*100:.2f}%")
    print(f"  3着以内含有率: {n_recset_top3:,} / {n_total:,} = {n_recset_top3/n_total*100:.2f}%")

    print(f"\n2着候補 (先頭)")
    print(f"  実 2着含有率: {n_ni_top2:,} / {n_total:,} = {n_ni_top2/n_total*100:.2f}%")

    print(f"\n=== 通常 vs ミッドナイト ===")
    if normal_n > 0:
        print(f"  通常 本命1着: {normal_honmei_1st:,}/{normal_n:,} = {normal_honmei_1st/normal_n*100:.2f}%")
    if mid_n > 0:
        print(f"  ミッドナイト 本命1着: {mid_honmei_1st:,}/{mid_n:,} = {mid_honmei_1st/mid_n*100:.2f}%")

    # ベースライン比較
    print(f"\n=== 無作為基準比較 (7車立て=14.29%, 9車立て=11.11%) ===")
    honmei_1st_rate = n_honmei_1st / n_total * 100
    baseline_est = 14.3  # 混合 7車想定
    print(f"  本命 1着率 {honmei_1st_rate:.2f}% vs {baseline_est}% → "
          f"{honmei_1st_rate - baseline_est:+.2f}pt")

    # 保存
    out = {
        "total_races": n_total,
        "parse_fail": parse_fail,
        "honmei_1st_rate":     n_honmei_1st / n_total * 100,
        "honmei_top3_rate":    n_honmei_top3 / n_total * 100,
        "rec_set_1st_rate":    n_recset_1st / n_total * 100,
        "rec_set_top3_rate":   n_recset_top3 / n_total * 100,
        "ni_candidate_top2":   n_ni_top2 / n_total * 100,
        "normal_n": normal_n,
        "normal_honmei_1st_rate": (normal_honmei_1st / normal_n * 100) if normal_n else 0,
        "mid_n": mid_n,
        "mid_honmei_1st_rate":    (mid_honmei_1st / mid_n * 100) if mid_n else 0,
    }
    out_path = REPORT_DIR / "v044_mark_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    analyze()
