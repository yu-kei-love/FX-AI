# ===========================================
# bank_master.py
# 競輪 全43会場 バンクデータ（固定値）
#
# 参照元：keirin.netkeiba.com/race/course/ 各会場ページ
#         Perfecta Navi・KEIRIN.JP 公式データ
#         （2026-04-08 全場照合・カント角度追加）
#         Wikipedia・LotoPlace・公式サイト
#         （2026-04-08 要確認5場を照合・全43場完了）
#
# 修正サマリー：
#   2026-03-31: 周長の大幅修正（宇都宮333→500 他多数）
#               廃止会場削除（郡山・浜松・福岡）
#               新規追加（弥彦・四日市）
#   2026-04-08: Perfecta Navi照合
#               カント角度を全場追加（36場確定・5場要確認・千葉改修中・高知追加）
#               みなし直線修正：前橋 46.7→46.6, 別府 60.0→59.9
#               松戸カント精度修正: 29.7→29.74
#   2026-04-08: 要確認5場を追加照合（Wikipedia・LotoPlace）
#               宇都宮 cant=25.79, 大宮 cant=26.28, 高知 cant=24.50(確認済み)
#               熊本 straight=69.5, cant=29.74（改修工事中だが記録保持）
#               奈良 cant=33.0, 小田原 cant=35.0（暫定値・要実測確認）
#   2026-04-11: jka_code フィールドを全43場に追加
#               （chariloto.com/keirin/results/ から実値を取得して設定）
#               scraper_historical.py の URL生成で使用
#
# ※ cant は度数の小数点表記に統一（例：32°00'29" → 32.01）
# ※ escape_rate は過去データから推計（未検証）
# ※ wind_impact は会場の地理的特性に基づく主観的スコア（1.0=標準）
#
# 会場数：43（北日本3・関東8・南関東7・中部7・北信越2・
#             近畿4・中国3・四国4・九州6 ＋千葉は改修中）
# ===========================================


BANK_MASTER = {
    # ===== 北日本地区（3会場） =====
    "函館": {
        "venue_id": "01",
        "jka_code": "11",
        "district": "北日本",
        "prefecture": "北海道",
        "length": 400,
        "straight": 51.3,   # Perfecta Navi照合済み
        "cant": 30.61,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.2,  # 海沿い・海風あり
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "青森": {
        "venue_id": "02",
        "jka_code": "12",
        "district": "北日本",
        "prefecture": "青森県",
        "length": 400,
        "straight": 58.9,   # Perfecta Navi照合済み
        "cant": 32.25,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.2,  # 強風多い
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",  # 直線長め・後方有利
    },
    "いわき平": {
        "venue_id": "03",
        "jka_code": "13",
        "district": "北日本",
        "prefecture": "福島県",
        "length": 400,
        "straight": 62.7,   # Perfecta Navi照合済み（400m中第3位の直線長さ）
        "cant": 32.92,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,  # ポリカーボネート囲いで風が逃げにくい
        "escape_rate": 0.26,  # 未検証
        "style_bias": "makuri",  # 直線長め・追込有利
    },
    # ===== 関東地区（8会場） =====
    "弥彦": {
        "venue_id": "04",
        "jka_code": "21",
        "district": "北日本",  # 新潟は北日本地区に分類
        "prefecture": "新潟県",
        "length": 400,
        "straight": 63.1,   # Perfecta Navi照合済み
        "cant": 32.40,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",  # 直線長め
    },
    "前橋": {
        "venue_id": "05",
        "jka_code": "22",
        "district": "関東",
        "prefecture": "群馬県",
        "length": 335,
        "straight": 46.6,   # Perfecta Navi照合済み（旧値46.7から修正）
        "cant": 36.00,       # Perfecta Navi照合済み（国内トップクラス）
        "is_dome": True,     # 屋内（確認済み）
        "wind_impact": 0.0,  # 屋内なので風影響なし
        "escape_rate": 0.32,
        "style_bias": "escape",  # 短バンク・カント急
    },
    "取手": {
        "venue_id": "06",
        "jka_code": "23",
        "district": "関東",
        "prefecture": "茨城県",
        "length": 400,
        "straight": 54.8,   # Perfecta Navi照合済み
        "cant": 31.51,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",  # 「クセのない標準的バンク・力勝負」
    },
    "宇都宮": {
        "venue_id": "07",
        "jka_code": "24",
        "district": "関東",
        "prefecture": "栃木県",
        "length": 500,       # 修正: 333→500（雷神バンク）
        "straight": 63.3,   # Perfecta Navi照合済み
        "cant": 25.79,       # Wikipedia・公式サイト確認済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.24,  # 未検証（500mは逃げ不利）
        "style_bias": "sashi",  # 500m・直線長め・追込有利
    },
    "大宮": {
        "venue_id": "08",
        "jka_code": "25",
        "district": "関東",
        "prefecture": "埼玉県",
        "length": 500,       # 修正: 400→500
        "straight": 66.7,   # Perfecta Navi照合済み
        "cant": 26.28,       # LotoPlace確認済み
        "is_dome": False,
        "wind_impact": 1.0,  # 季節により風向変化あり
        "escape_rate": 0.23,  # 未検証
        "style_bias": "sashi",  # 「直線長くカント深い・追込有利」
    },
    "西武園": {
        "venue_id": "09",
        "jka_code": "26",
        "district": "関東",
        "prefecture": "埼玉県",
        "length": 400,
        "straight": 47.6,   # Perfecta Navi照合済み
        "cant": 29.45,       # Perfecta Navi照合済み（カント緩め）
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.31,  # 未検証
        "style_bias": "escape",  # 「333mバンクに近い性格・逃げ有利」
    },
    "京王閣": {
        "venue_id": "10",
        "jka_code": "27",
        "district": "関東",
        "prefecture": "東京都",
        "length": 400,
        "straight": 51.5,   # Perfecta Navi照合済み
        "cant": 32.17,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 0.9,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "立川": {
        "venue_id": "11",
        "jka_code": "28",
        "district": "関東",
        "prefecture": "東京都",
        "length": 400,
        "straight": 58.0,   # Perfecta Navi照合済み
        "cant": 31.22,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 南関東地区（7会場） =====
    "松戸": {
        "venue_id": "12",
        "jka_code": "31",
        "district": "南関東",
        "prefecture": "千葉県",
        "length": 333,
        "straight": 38.2,   # Perfecta Navi照合済み
        "cant": 29.74,       # Perfecta Navi照合済み（旧値29.7を精度修正）
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    "千葉": {
        "venue_id": "13",
        "jka_code": "32",
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
        "jka_code": "34",
        "district": "南関東",
        "prefecture": "神奈川県",
        "length": 400,
        "straight": 58.0,   # Perfecta Navi照合済み
        "cant": 32.17,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "平塚": {
        "venue_id": "15",
        "jka_code": "35",
        "district": "南関東",
        "prefecture": "神奈川県",
        "length": 400,       # 修正: 500→400（湘南バンク）
        "straight": 54.2,   # Perfecta Navi照合済み
        "cant": 31.64,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.3,  # 海沿い・川からの強風
        "escape_rate": 0.26,  # 未検証
        "style_bias": "makuri",  # 「捲り・先まくり有利」
    },
    "小田原": {
        "venue_id": "16",
        "jka_code": "36",
        "district": "南関東",
        "prefecture": "神奈川県",
        "length": 333,
        "straight": 36.1,   # Perfecta Navi照合済み
        "cant": 35.00,       # 暫定値・要実測確認（「全国屈指の傾斜」記述から推定）
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "伊東": {
        "venue_id": "17",
        "jka_code": "37",
        "district": "南関東",
        "prefecture": "静岡県",
        "length": 333,
        "straight": 46.6,   # Perfecta Navi照合済み
        "cant": 34.69,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    "静岡": {
        "venue_id": "18",
        "jka_code": "38",
        "district": "中部",
        "prefecture": "静岡県",
        "length": 400,       # 修正: 500→400
        "straight": 56.4,   # Perfecta Navi照合済み
        "cant": 30.72,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.2,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 中部地区（7会場） =====
    "豊橋": {
        "venue_id": "19",
        "jka_code": "45",
        "district": "中部",
        "prefecture": "愛知県",
        "length": 400,
        "straight": 60.3,   # Perfecta Navi照合済み
        "cant": 33.84,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "名古屋": {
        "venue_id": "20",
        "jka_code": "42",
        "district": "中部",
        "prefecture": "愛知県",
        "length": 400,
        "straight": 58.8,   # Perfecta Navi照合済み
        "cant": 34.03,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    "岐阜": {
        "venue_id": "21",
        "jka_code": "43",
        "district": "中部",
        "prefecture": "岐阜県",
        "length": 400,
        "straight": 59.3,   # Perfecta Navi照合済み
        "cant": 33.25,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "大垣": {
        "venue_id": "22",
        "jka_code": "44",
        "district": "中部",
        "prefecture": "岐阜県",
        "length": 400,       # 修正: 333→400
        "straight": 56.0,   # Perfecta Navi照合済み
        "cant": 30.62,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "松阪": {
        "venue_id": "23",
        "jka_code": "47",
        "district": "中部",
        "prefecture": "三重県",
        "length": 400,
        "straight": 61.5,   # Perfecta Navi照合済み
        "cant": 34.42,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "四日市": {
        "venue_id": "24",
        "jka_code": "48",
        "district": "中部",
        "prefecture": "三重県",
        "length": 400,
        "straight": 62.4,   # Perfecta Navi照合済み
        "cant": 32.25,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 北信越地区（2会場） =====
    "富山": {
        "venue_id": "25",
        "jka_code": "46",
        "district": "北信越",
        "prefecture": "富山県",
        "length": 333,
        "straight": 43.0,   # Perfecta Navi照合済み
        "cant": 33.69,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.31,
        "style_bias": "escape",
    },
    "福井": {
        "venue_id": "26",
        "jka_code": "51",
        "district": "北信越",
        "prefecture": "福井県",
        "length": 400,
        "straight": 52.8,   # Perfecta Navi照合済み
        "cant": 31.48,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    # ===== 近畿地区（4会場） =====
    "奈良": {
        "venue_id": "27",
        "jka_code": "53",
        "district": "近畿",
        "prefecture": "奈良県",
        "length": 333,       # 修正: 400→333
        "straight": 38.0,   # Perfecta Navi照合済み
        "cant": 33.00,       # 暫定値・要実測確認（333m平均的な値）
        "is_dome": False,
        "wind_impact": 0.8,
        "escape_rate": 0.31,  # 未検証（333mなので逃げ有利）
        "style_bias": "escape",
    },
    "向日町": {
        "venue_id": "28",
        "jka_code": "54",
        "district": "近畿",
        "prefecture": "京都府",
        "length": 400,
        "straight": 47.3,   # Perfecta Navi照合済み
        "cant": 30.49,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 0.9,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "和歌山": {
        "venue_id": "29",
        "jka_code": "55",
        "district": "近畿",
        "prefecture": "和歌山県",
        "length": 400,
        "straight": 59.9,   # Perfecta Navi照合済み
        "cant": 32.25,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "岸和田": {
        "venue_id": "30",
        "jka_code": "56",
        "district": "近畿",
        "prefecture": "大阪府",
        "length": 400,
        "straight": 56.7,   # Perfecta Navi照合済み
        "cant": 30.93,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 中国地区（3会場） =====
    "玉野": {
        "venue_id": "31",
        "jka_code": "61",
        "district": "中国",
        "prefecture": "岡山県",
        "length": 400,
        "straight": 47.9,   # Perfecta Navi照合済み
        "cant": 30.62,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.2,  # 海沿い
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "広島": {
        "venue_id": "32",
        "jka_code": "62",
        "district": "中国",
        "prefecture": "広島県",
        "length": 400,
        "straight": 57.9,   # Perfecta Navi照合済み
        "cant": 30.79,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    "防府": {
        "venue_id": "33",
        "jka_code": "63",
        "district": "中国",
        "prefecture": "山口県",
        "length": 333,
        "straight": 42.5,   # Perfecta Navi照合済み
        "cant": 34.69,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,  # 山に囲まれ風の影響あり
        "escape_rate": 0.30,
        "style_bias": "escape",
    },
    # ===== 四国地区（4会場） =====
    "高松": {
        "venue_id": "34",
        "jka_code": "71",
        "district": "四国",
        "prefecture": "香川県",
        "length": 400,
        "straight": 54.8,   # Perfecta Navi照合済み
        "cant": 33.26,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "escape",
    },
    "小松島": {
        "venue_id": "35",
        "jka_code": "73",
        "district": "四国",
        "prefecture": "徳島県",
        "length": 400,       # 修正: 333→400
        "straight": 55.5,   # Perfecta Navi照合済み
        "cant": 29.77,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    "高知": {
        "venue_id": "36",
        "jka_code": "74",
        "district": "四国",
        "prefecture": "高知県",
        "length": 500,       # 修正: 400→500
        "straight": 52.0,   # Perfecta Navi照合済み
        "cant": 24.50,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.24,  # 未検証（500mは逃げ不利）
        "style_bias": "sashi",
    },
    "松山": {
        "venue_id": "37",
        "jka_code": "75",
        "district": "四国",
        "prefecture": "愛媛県",
        "length": 400,
        "straight": 58.6,   # Perfecta Navi照合済み
        "cant": 34.03,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.28,  # 未検証
        "style_bias": "makuri",
    },
    # ===== 九州地区（6会場） =====
    "小倉": {
        "venue_id": "38",
        "jka_code": "81",
        "district": "九州",
        "prefecture": "福岡県",
        "length": 400,       # 修正: 333→400（ミスドームバンク）
        "straight": 56.9,   # Perfecta Navi照合済み
        "cant": 34.03,       # Perfecta Navi照合済み（旧値37.0は誤りで削除済み）
        "is_dome": True,     # 屋内（確認済み・無風）
        "wind_impact": 0.0,  # 屋内なので風影響なし
        "escape_rate": 0.30,  # 未検証
        "style_bias": "escape",  # 屋内・高速バンク
    },
    "久留米": {
        "venue_id": "39",
        "jka_code": "83",
        "district": "九州",
        "prefecture": "福岡県",
        "length": 400,
        "straight": 50.7,   # Perfecta Navi照合済み
        "cant": 31.48,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.29,  # 未検証
        "style_bias": "escape",
    },
    "武雄": {
        "venue_id": "40",
        "jka_code": "84",
        "district": "九州",
        "prefecture": "佐賀県",
        "length": 400,
        "straight": 64.4,   # Perfecta Navi照合済み
        "cant": 32.01,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.26,  # 未検証（直線長い）
        "style_bias": "makuri",
    },
    "佐世保": {
        "venue_id": "41",
        "jka_code": "85",
        "district": "九州",
        "prefecture": "長崎県",
        "length": 400,
        "straight": 40.2,   # Perfecta Navi照合済み（400m中最短直線）
        "cant": 31.48,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.1,
        "escape_rate": 0.33,  # 未検証（直線最短→逃げ有利）
        "style_bias": "escape",  # 「400m中最短直線・逃げ有利」
    },
    "別府": {
        "venue_id": "42",
        "jka_code": "86",
        "district": "九州",
        "prefecture": "大分県",
        "length": 400,       # 修正: 333→400
        "straight": 59.9,   # Perfecta Navi照合済み（旧値60.0から修正）
        "cant": 33.69,       # Perfecta Navi照合済み
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.27,  # 未検証
        "style_bias": "makuri",
    },
    "熊本": {
        "venue_id": "43",
        "jka_code": "87",
        "district": "九州",
        "prefecture": "熊本県",
        "length": 500,       # LotoPlace確認済み（500mバンク）
        "straight": 69.5,   # LotoPlace確認済み（旧値60.3から修正）
        "cant": 29.74,       # LotoPlace確認済み
        # 改修工事中（2025年時点）だがバンクデータは記録として保持
        "is_dome": False,
        "wind_impact": 1.0,
        "escape_rate": 0.24,  # 未検証（500mは逃げ不利）
        "style_bias": "sashi",  # 500m・追込有利
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
