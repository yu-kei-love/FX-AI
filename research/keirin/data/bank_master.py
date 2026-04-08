# ===========================================
# bank_master.py
# 競輪 全43会場 バンクデータ（固定値）
#
# 参照元：keirin.netkeiba.com/race/course/ 各会場ページ
#         （2026-03-31 照合・修正）
#
# 修正サマリー（Perfecta Navi照合）：
#   周長の大幅修正：宇都宮 333→500, 大宮 400→500, 平塚 500→400,
#                   静岡 500→400, 大垣 333→400, 奈良 400→333,
#                   小松島 333→400, 高知 400→500,
#                   小倉 333→400, 別府 333→400
#   廃止会場を削除：郡山（2013年廃止）・浜松（2022年廃止）・福岡（2011年廃止）
#   新規追加：弥彦, 四日市
#   改修中：千葉（500m→250m木製バンク工事中・2026年時点で無開催）
#
# ※ cant は一部会場のみ公式値、未記載会場は未検証（None）
# ※ escape_rate は過去データから推計（未検証）
# ※ wind_impact は会場の地理的特性に基づく主観的スコア（1.0=標準）
#
# 会場数：43（北日本3・関東8・南関東7・中部7・北信越1・
#             近畿4・中国3・四国4・九州6 ＋千葉は改修中）
# ===========================================


BANK_MASTER = {
    # ===== 北日本地区（3会場） =====
    "函館": {
        "venue_id": "01",
        "district": "北日本",
        "prefecture": "北海道",
        "length": 400,
        "straight": 51.3,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.2,  # 海沿い・海風あり
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",  # 標準バンク・脚質差なし
    },
    "青森": {
        "venue_id": "02",
        "district": "北日本",
        "prefecture": "青森県",
        "length": 400,
        "straight": 58.9,   # netkeiba確認
        "cant": None,        # 未取得（「ややきつめ」との記述）
        "is_dome": False,
        "wind_impact": 1.2,  # 強風多い
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",  # 直線長め・後方有利
    },
    "いわき平": {
        "venue_id": "03",
        "district": "北日本",
        "prefecture": "福島県",
        "length": 400,
        "straight": 62.7,   # netkeiba確認（400m中第3位の直線長さ）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,  # ポリカーボネート囲いで風が逃げにくい
        "escape_rate": 0.26,  # 未検証
        "style_bias": "makuri",  # 直線長め・追込有利
    },
    # ===== 関東地区（8会場） =====
    "弥彦": {
        "venue_id": "04",
        "district": "北日本",  # 新潟は北日本地区に分類
        "prefecture": "新潟県",
        "length": 400,
        "straight": 63.1,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",  # 直線長め
    },
    "前橋": {
        "venue_id": "05",
        "district": "関東",
        "prefecture": "群馬県",
        "length": 335,
        "straight": 46.7,   # netkeiba確認（旧値43.0から修正）
        "cant": 36.0,        # netkeiba確認「36度・国内トップクラス」
        "is_dome": True,     # 屋内（確認済み）
        "wind_impact": 0.0,  # 屋内なので風影響なし
        "escape_rate": 0.32,
        "style_bias": "escape",  # 短バンク・カント急
    },
    "取手": {
        "venue_id": "06",
        "district": "関東",
        "prefecture": "茨城県",
        "length": 400,
        "straight": 54.8,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",  # 「クセのない標準的バンク・力勝負」
    },
    "宇都宮": {
        "venue_id": "07",
        "district": "関東",
        "prefecture": "栃木県",
        "length": 500,       # 修正: 333→500（雷神バンク）
        "straight": 63.3,   # netkeiba確認
        "cant": None,        # 未取得（「フラットバンク」との記述）
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.24,  # 未検証（500mは逃げ不利）
        "style_bias": "sashi",  # 500m・直線長め・追込有利
    },
    "大宮": {
        "venue_id": "08",
        "district": "関東",
        "prefecture": "埼玉県",
        "length": 500,       # 修正: 400→500
        "straight": 66.7,   # netkeiba確認
        "cant": None,        # 未取得（「深い」との記述）
        "is_dome": False,
        "wind_impact": 1.0,  # 季節により風向変化あり
        "escape_rate": 0.23,  # 未検証
        "style_bias": "sashi",  # 「直線長くカント深い・追込有利」
    },
    "西武園": {
        "venue_id": "09",
        "district": "関東",
        "prefecture": "埼玉県",
        "length": 400,
        "straight": 47.6,   # netkeiba確認（旧値55.0から修正）
        "cant": None,        # 未取得（「カント緩め」との記述）
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.31,  # 未検証
        "style_bias": "escape",  # 「333mバンクに近い性格・逃げ有利」（明示確認）
    },
    "京王閣": {
        "venue_id": "10",
        "district": "関東",
        "prefecture": "東京都",
        "length": 400,
        "straight": 51.5,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 0.9,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "立川": {
        "venue_id": "11",
        "district": "関東",
        "prefecture": "東京都",
        "length": 400,
        "straight": 58.0,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 南関東地区（7会場） =====
    "松戸": {
        "venue_id": "12",
        "district": "南関東",
        "prefecture": "千葉県",
        "length": 333,
        "straight": 38.2,   # netkeiba確認（旧値と一致）
        "cant": 29.7,        # 旧値維持（未検証）
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    "千葉": {
        "venue_id": "13",
        "district": "南関東",
        "prefecture": "千葉県",
        # 改修中（2026年時点）：500m→250m木製バンク「千葉公園ドーム」建設中
        # 通常の競輪レースは開催されていない
        "length": 250,       # 改修後の予定（未確定）
        "straight": None,    # 未定
        "cant": None,        # 未定
        "is_dome": True,     # 改修後はドーム型
        "wind_impact": 0.0,
        "escape_rate": None,
        "style_bias": "unknown",  # 改修中のためデータなし
    },
    "川崎": {
        "venue_id": "14",
        "district": "南関東",
        "prefecture": "神奈川県",
        "length": 400,
        "straight": 58.0,   # netkeiba確認（旧値52.5から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "平塚": {
        "venue_id": "15",
        "district": "南関東",
        "prefecture": "神奈川県",
        "length": 400,       # 修正: 500→400（湘南バンク）
        "straight": 54.2,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.3,  # 海沿い・川からの強風
        "escape_rate": 0.26,  # 未検証
        "style_bias": "makuri",  # 「捲り・先まくり有利」
    },
    "小田原": {
        "venue_id": "16",
        "district": "南関東",
        "prefecture": "神奈川県",
        "length": 333,
        "straight": 36.1,   # netkeiba確認（旧値42.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "伊東": {
        "venue_id": "17",
        "district": "南関東",
        "prefecture": "静岡県",
        "length": 333,
        "straight": 46.6,   # netkeiba確認（旧値44.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    "静岡": {
        "venue_id": "18",
        "district": "中部",
        "prefecture": "静岡県",
        "length": 400,       # 修正: 500→400
        "straight": 56.4,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 中部地区（7会場） =====
    "豊橋": {
        "venue_id": "19",
        "district": "中部",
        "prefecture": "愛知県",
        "length": 400,
        "straight": 60.3,   # netkeiba確認（旧値55.5から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "名古屋": {
        "venue_id": "20",
        "district": "中部",
        "prefecture": "愛知県",
        "length": 400,
        "straight": 58.8,   # netkeiba確認（旧値57.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    "岐阜": {
        "venue_id": "21",
        "district": "中部",
        "prefecture": "岐阜県",
        "length": 400,
        "straight": 59.3,   # netkeiba確認（旧値53.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "大垣": {
        "venue_id": "22",
        "district": "中部",
        "prefecture": "岐阜県",
        "length": 400,       # 修正: 333→400
        "straight": 56.0,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "松阪": {
        "venue_id": "23",
        "district": "中部",
        "prefecture": "三重県",
        "length": 400,
        "straight": 61.5,   # netkeiba確認（旧値53.5から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "四日市": {
        "venue_id": "24",
        "district": "中部",
        "prefecture": "三重県",
        "length": 400,       # 新規追加
        "straight": 62.4,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 北信越地区（1会場） =====
    "富山": {
        "venue_id": "25",
        "district": "北信越",
        "prefecture": "富山県",
        "length": 333,
        "straight": 43.0,   # netkeiba確認（旧値44.0とほぼ一致）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "福井": {
        "venue_id": "26",
        "district": "北信越",
        "prefecture": "福井県",
        "length": 400,
        "straight": 52.8,   # netkeiba確認（旧値55.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 近畿地区（4会場） =====
    "奈良": {
        "venue_id": "27",
        "district": "近畿",
        "prefecture": "奈良県",
        "length": 333,       # 修正: 400→333
        "straight": 38.0,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.31,  # 未検証（333mなので逃げ有利）
        "style_bias": "escape",
    },
    "向日町": {
        "venue_id": "28",
        "district": "近畿",
        "prefecture": "京都府",
        "length": 400,
        "straight": 47.3,   # netkeiba確認（旧値54.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 0.9,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "和歌山": {
        "venue_id": "29",
        "district": "近畿",
        "prefecture": "和歌山県",
        "length": 400,
        "straight": 59.9,   # netkeiba確認（旧値54.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "岸和田": {
        "venue_id": "30",
        "district": "近畿",
        "prefecture": "大阪府",
        "length": 400,
        "straight": 56.7,   # netkeiba確認（旧値55.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 中国地区（3会場） =====
    "玉野": {
        "venue_id": "31",
        "district": "中国",
        "prefecture": "岡山県",
        "length": 400,
        "straight": 47.9,   # netkeiba確認（旧値54.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.2,  # 海沿い
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "広島": {
        "venue_id": "32",
        "district": "中国",
        "prefecture": "広島県",
        "length": 400,
        "straight": 57.9,   # netkeiba確認（旧値55.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    "防府": {
        "venue_id": "33",
        "district": "中国",
        "prefecture": "山口県",
        "length": 333,
        "straight": 42.5,   # netkeiba確認（旧値43.0とほぼ一致）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,  # 山に囲まれ風の影響あり
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    # ===== 四国地区（4会場） =====
    "高松": {
        "venue_id": "34",
        "district": "四国",
        "prefecture": "香川県",
        "length": 400,
        "straight": 54.8,   # netkeiba確認（旧値54.0とほぼ一致）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "小松島": {
        "venue_id": "35",
        "district": "四国",
        "prefecture": "徳島県",
        "length": 400,       # 修正: 333→400
        "straight": 55.5,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    "高知": {
        "venue_id": "36",
        "district": "四国",
        "prefecture": "高知県",
        "length": 500,       # 修正: 400→500
        "straight": 52.0,   # netkeiba確認
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.24,  # 未検証（500mは逃げ不利）
        "style_bias": "sashi",
    },
    "松山": {
        "venue_id": "37",
        "district": "四国",
        "prefecture": "愛媛県",
        "length": 400,
        "straight": 58.6,   # netkeiba確認（旧値53.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 九州地区（6会場） =====
    "小倉": {
        "venue_id": "38",
        "district": "九州",
        "prefecture": "福岡県",
        "length": 400,       # 修正: 333→400（ミスドームバンク）
        "straight": 56.9,   # netkeiba確認
        "cant": None,        # 未取得（旧値37.0は誤り）
        "is_dome": True,     # 屋内（確認済み・無風）
        "wind_impact": 0.0,  # 屋内なので風影響なし
        "escape_rate": 0.30,  # 未検証
        "style_bias": "escape",  # 屋内・高速バンク
    },
    "久留米": {
        "venue_id": "39",
        "district": "九州",
        "prefecture": "福岡県",
        "length": 400,
        "straight": 50.7,   # netkeiba確認（旧値54.5から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "武雄": {
        "venue_id": "40",
        "district": "九州",
        "prefecture": "佐賀県",
        "length": 400,
        "straight": 64.4,   # netkeiba確認（旧値53.0から修正）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.26,  # 未検証（直線長い）
        "style_bias": "makuri",
    },
    "佐世保": {
        "venue_id": "41",
        "district": "九州",
        "prefecture": "長崎県",
        "length": 400,
        "straight": 40.2,   # netkeiba確認（400m中最短直線）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.33,  # 未検証（直線最短→逃げ有利）
        "style_bias": "escape",  # 「400m中最短直線・逃げ有利」（明示確認）
    },
    "別府": {
        "venue_id": "42",
        "district": "九州",
        "prefecture": "大分県",
        "length": 400,       # 修正: 333→400
        "straight": 60.0,   # netkeiba確認（59.96m）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "熊本": {
        "venue_id": "43",
        "district": "九州",
        "prefecture": "熊本県",
        "length": 400,
        "straight": 60.3,   # netkeiba確認（2024-07再開）
        "cant": None,        # 未取得
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
}

assert len(BANK_MASTER) == 43, f"会場数が{len(BANK_MASTER)}です（43であるべき）"

# 地区ごとのライン組成親和性（同地区同士はライン組みやすい）
DISTRICT_LINE_AFFINITY = {
    "北日本":  {"北日本": 0.75, "関東": 0.30},
    "関東":    {"関東": 0.75, "北日本": 0.25, "南関東": 0.20},
    "南関東":  {"南関東": 0.75, "関東": 0.25},
    "中部":    {"中部": 0.75, "北信越": 0.30},
    "北信越":  {"北信越": 0.75, "中部": 0.30},
    "近畿":    {"近畿": 0.80},
    "中国":    {"中国": 0.80, "四国": 0.20},
    "四国":    {"四国": 0.80, "中国": 0.20},
    "九州":    {"九州": 0.85},
}


def get_bank_info(venue_name: str) -> dict | None:
    """会場名からバンク情報を取得する"""
    return BANK_MASTER.get(venue_name)


def get_style_advantage(
    venue_name: str,
    wind_speed: float,
    wind_direction: float,  # 度数法 0=北,90=東,180=南,270=西
) -> dict:
    """
    バンク特性と風を組み合わせた戦法別有利度スコアを返す。

    バック向かい風の判定：
    - 風向180度±45度をバック向かい風として扱う（簡略化）

    Parameters:
        venue_name      : 会場名
        wind_speed      : 風速（m/s）
        wind_direction  : 風向（度）

    Returns:
        dict: {"escape": 0.8, "makuri": 0.5, "sashi": 0.3}
    """
    bank = get_bank_info(venue_name)
    if bank is None:
        return {"escape": 0.50, "makuri": 0.50, "sashi": 0.50}

    # 千葉は改修中で開催なし
    if bank.get("style_bias") == "unknown":
        return {"escape": 0.50, "makuri": 0.50, "sashi": 0.50}

    # 屋内会場は風影響なし
    effective_wind = 0.0 if bank["is_dome"] else wind_speed
    is_back_headwind = False
    if effective_wind >= 3.0:
        back_wind_angle = abs(wind_direction - 180.0)
        is_back_headwind = back_wind_angle <= 45.0

    # バンク周長によるベーススコア
    # ≤335m：直線が短く逃げ有利
    # 400m：バランス型
    # 500m：直線が長く捲り・差し有利
    if bank["length"] <= 335:
        base_escape = 0.75
        base_makuri = 0.45
        base_sashi  = 0.35
    elif bank["length"] == 400:
        base_escape = 0.60
        base_makuri = 0.55
        base_sashi  = 0.50
    else:  # 500m
        base_escape = 0.45
        base_makuri = 0.65
        base_sashi  = 0.60

    # カントがきつい（≥34度）→ 捲り有利
    if bank.get("cant") and bank["cant"] >= 34.0:
        base_makuri += 0.10

    # バック向かい風補正（風速4m/s以上で効果が大きくなる）
    wind_escape_penalty = 0.0
    wind_makuri_bonus   = 0.0
    if is_back_headwind and effective_wind >= 4.0:
        factor = min(effective_wind / 4.0, 2.0)
        wind_escape_penalty = 0.15 * factor
        wind_makuri_bonus   = 0.10 * factor

    escape = max(0.0, min(1.0, base_escape - wind_escape_penalty))
    makuri = max(0.0, min(1.0, base_makuri + wind_makuri_bonus))
    sashi  = max(0.0, min(1.0, base_sashi))

    return {
        "escape": round(escape, 3),
        "makuri": round(makuri, 3),
        "sashi":  round(sashi, 3),
    }
