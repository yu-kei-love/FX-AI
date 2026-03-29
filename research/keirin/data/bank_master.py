# ===========================================
# bank_master.py
# 競輪 全43会場 バンクデータ（固定値）
#
# 参照元：Perfecta Navi、各会場公式情報
# ※ escape_rate は過去データから推計（未検証）
# ※ straight・cant は公式値。一部近似値あり（未検証マークあり）
#
# 会場数：43（北日本4・関東7・南関東5・中部7・北信越2・
#             近畿4・中国3・四国4・九州7）
# ===========================================


BANK_MASTER = {
    # ===== 北日本地区（4会場） =====
    "函館": {
        "venue_id": "01",
        "district": "北日本",
        "length": 400,
        "straight": 56.4,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "青森": {
        "venue_id": "02",
        "district": "北日本",
        "length": 400,
        "straight": 51.2,   # 未検証
        "cant": 32.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.2,  # 強風多い
        "escape_rate": 0.27,  # 未検証
        "style_bias": "escape",
    },
    "いわき平": {
        "venue_id": "03",
        "district": "北日本",
        "length": 400,
        "straight": 56.0,   # 未検証
        "cant": 30.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "郡山": {
        "venue_id": "04",
        "district": "北日本",
        "length": 400,
        "straight": 58.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 関東地区（7会場） =====
    "前橋": {
        "venue_id": "05",
        "district": "関東",
        "length": 335,
        "straight": 43.0,
        "cant": 36.0,
        "is_dome": True,    # 屋内
        "wind_impact": 0.0,  # 屋内なので風影響なし
        "escape_rate": 0.32,
        "style_bias": "escape",
    },
    "取手": {
        "venue_id": "06",
        "district": "関東",
        "length": 400,
        "straight": 53.0,   # 未検証
        "cant": 30.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "宇都宮": {
        "venue_id": "07",
        "district": "関東",
        "length": 333,
        "straight": 43.5,   # 未検証
        "cant": 33.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "大宮": {
        "venue_id": "08",
        "district": "関東",
        "length": 400,
        "straight": 52.0,   # 未検証
        "cant": 30.2,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "西武園": {
        "venue_id": "09",
        "district": "関東",
        "length": 400,
        "straight": 55.0,   # 未検証
        "cant": 31.5,        # 未検証
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "京王閣": {
        "venue_id": "10",
        "district": "関東",
        "length": 400,
        "straight": 54.2,   # 未検証
        "cant": 32.0,        # 未検証
        "is_dome": False,
        "wind_impact": 0.9,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "立川": {
        "venue_id": "11",
        "district": "関東",
        "length": 400,
        "straight": 55.4,   # 未検証
        "cant": 30.8,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 南関東地区（5会場） =====
    "松戸": {
        "venue_id": "12",
        "district": "南関東",
        "length": 333,
        "straight": 38.2,
        "cant": 29.7,
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    "川崎": {
        "venue_id": "13",
        "district": "南関東",
        "length": 400,
        "straight": 52.5,   # 未検証
        "cant": 32.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "平塚": {
        "venue_id": "14",
        "district": "南関東",
        "length": 500,
        "straight": 70.0,   # 未検証
        "cant": 24.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.3,  # 海沿い・強風
        "escape_rate": 0.24,
        "style_bias": "makuri",
    },
    "小田原": {
        "venue_id": "15",
        "district": "南関東",
        "length": 333,
        "straight": 42.0,   # 未検証
        "cant": 34.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "伊東": {
        "venue_id": "16",
        "district": "南関東",
        "length": 333,
        "straight": 44.0,   # 未検証
        "cant": 32.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    # ===== 中部地区（7会場） =====
    "静岡": {
        "venue_id": "17",
        "district": "中部",
        "length": 500,
        "straight": 68.0,   # 未検証
        "cant": 26.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.25,
        "style_bias": "makuri",
    },
    "浜松": {
        "venue_id": "18",
        "district": "中部",
        "length": 400,
        "straight": 54.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "豊橋": {
        "venue_id": "19",
        "district": "中部",
        "length": 400,
        "straight": 55.5,   # 未検証
        "cant": 30.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "名古屋": {
        "venue_id": "20",
        "district": "中部",
        "length": 400,
        "straight": 57.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "岐阜": {
        "venue_id": "21",
        "district": "中部",
        "length": 400,
        "straight": 53.0,   # 未検証
        "cant": 32.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "大垣": {
        "venue_id": "22",
        "district": "中部",
        "length": 333,
        "straight": 42.0,   # 未検証
        "cant": 35.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "松阪": {
        "venue_id": "23",
        "district": "中部",
        "length": 400,
        "straight": 53.5,   # 未検証
        "cant": 31.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 北信越地区（2会場） =====
    "富山": {
        "venue_id": "24",
        "district": "北信越",
        "length": 333,
        "straight": 44.0,   # 未検証
        "cant": 33.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "福井": {
        "venue_id": "25",
        "district": "北信越",
        "length": 400,
        "straight": 55.0,   # 未検証
        "cant": 30.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 近畿地区（4会場） =====
    "奈良": {
        "venue_id": "26",
        "district": "近畿",
        "length": 400,
        "straight": 56.0,   # 未検証
        "cant": 30.0,        # 未検証
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "和歌山": {
        "venue_id": "27",
        "district": "近畿",
        "length": 400,
        "straight": 54.0,   # 未検証
        "cant": 32.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "岸和田": {
        "venue_id": "28",
        "district": "近畿",
        "length": 400,
        "straight": 55.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "向日町": {
        "venue_id": "29",
        "district": "近畿",
        "length": 400,
        "straight": 54.0,   # 未検証
        "cant": 32.0,        # 未検証
        "is_dome": False,
        "wind_impact": 0.9,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    # ===== 中国地区（3会場） =====
    "玉野": {
        "venue_id": "30",
        "district": "中国",
        "length": 400,
        "straight": 54.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.2,  # 海沿い
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "広島": {
        "venue_id": "31",
        "district": "中国",
        "length": 400,
        "straight": 55.0,   # 未検証
        "cant": 30.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "防府": {
        "venue_id": "32",
        "district": "中国",
        "length": 333,
        "straight": 43.0,   # 未検証
        "cant": 34.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    # ===== 四国地区（4会場） =====
    "高松": {
        "venue_id": "33",
        "district": "四国",
        "length": 400,
        "straight": 54.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "小松島": {
        "venue_id": "34",
        "district": "四国",
        "length": 333,
        "straight": 42.0,   # 未検証
        "cant": 33.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "高知": {
        "venue_id": "35",
        "district": "四国",
        "length": 400,
        "straight": 55.0,   # 未検証
        "cant": 30.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "松山": {
        "venue_id": "36",
        "district": "四国",
        "length": 400,
        "straight": 53.0,   # 未検証
        "cant": 31.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 九州地区（7会場） =====
    "福岡": {
        "venue_id": "37",
        "district": "九州",
        "length": 500,
        "straight": 66.0,   # 未検証
        "cant": 25.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.25,
        "style_bias": "makuri",
    },
    "小倉": {
        "venue_id": "38",
        "district": "九州",
        "length": 333,
        "straight": 40.5,
        "cant": 37.0,
        "is_dome": True,    # 屋内
        "wind_impact": 0.0,  # 屋内なので風影響なし
        "escape_rate": 0.33,
        "style_bias": "escape",
    },
    "久留米": {
        "venue_id": "39",
        "district": "九州",
        "length": 400,
        "straight": 54.5,   # 未検証
        "cant": 30.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "武雄": {
        "venue_id": "40",
        "district": "九州",
        "length": 400,
        "straight": 53.0,   # 未検証
        "cant": 31.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "佐世保": {
        "venue_id": "41",
        "district": "九州",
        "length": 400,
        "straight": 52.5,   # 未検証
        "cant": 31.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "別府": {
        "venue_id": "42",
        "district": "九州",
        "length": 333,
        "straight": 44.0,   # 未検証
        "cant": 33.0,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "熊本": {
        "venue_id": "43",
        "district": "九州",
        "length": 400,
        "straight": 54.0,   # 未検証
        "cant": 30.5,        # 未検証
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
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

    # 屋内会場は風影響なし
    effective_wind = 0.0 if bank["is_dome"] else wind_speed
    is_back_headwind = False
    if effective_wind >= 3.0:
        back_wind_angle = abs(wind_direction - 180.0)
        is_back_headwind = back_wind_angle <= 45.0

    # バンク周長によるベーススコア
    # 333m：直線が短く逃げ有利
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
    if bank["cant"] >= 34.0:
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


def get_all_venue_names() -> list:
    """全43会場名のリストを返す"""
    return list(BANK_MASTER.keys())


def get_venues_by_district(district: str) -> list:
    """地区名から会場名リストを返す"""
    return [name for name, info in BANK_MASTER.items()
            if info.get("district") == district]


def get_dome_venues() -> list:
    """屋内会場のリストを返す（前橋・小倉）"""
    return [name for name, info in BANK_MASTER.items()
            if info.get("is_dome")]
