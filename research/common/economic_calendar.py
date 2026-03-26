"""
経済イベントカレンダー管理モジュール

高インパクト経済イベント（FOMC, NFP, CPI, BOJ会合など）の前後で
トレードを自動停止するためのカレンダー機能を提供する。

タイムゾーンはJST（Asia/Tokyo, UTC+9）を基準とする。
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import csv
import os

# ─── タイムゾーン定義 ───
JST = timezone(timedelta(hours=9))

# ─── 2026年の高インパクト経済イベント一覧 ───
# フォーマット: (イベント名, JST日時, 通貨ペアへの影響)
# 日時は発表時刻（JST）を記載。後で更新可能。

# FOMC: 年8回、通常水曜日 14:00 ET → 翌木曜日 03:00 JST
_FOMC_2026 = [
    datetime(2026, 1, 29, 3, 0, tzinfo=JST),   # 1月会合
    datetime(2026, 3, 19, 3, 0, tzinfo=JST),   # 3月会合
    datetime(2026, 5, 7, 3, 0, tzinfo=JST),    # 5月会合
    datetime(2026, 6, 18, 3, 0, tzinfo=JST),   # 6月会合
    datetime(2026, 7, 30, 3, 0, tzinfo=JST),   # 7月会合
    datetime(2026, 9, 17, 3, 0, tzinfo=JST),   # 9月会合
    datetime(2026, 11, 5, 3, 0, tzinfo=JST),   # 11月会合
    datetime(2026, 12, 17, 3, 0, tzinfo=JST),  # 12月会合
]

# 米国雇用統計（NFP）: 毎月第1金曜日 8:30 ET → 21:30 JST
_NFP_2026 = [
    datetime(2026, 1, 9, 22, 30, tzinfo=JST),  # 1月（冬時間）
    datetime(2026, 2, 6, 22, 30, tzinfo=JST),  # 2月（冬時間）
    datetime(2026, 3, 6, 22, 30, tzinfo=JST),  # 3月（冬時間）
    datetime(2026, 4, 3, 21, 30, tzinfo=JST),  # 4月（夏時間）
    datetime(2026, 5, 8, 21, 30, tzinfo=JST),  # 5月
    datetime(2026, 6, 5, 21, 30, tzinfo=JST),  # 6月
    datetime(2026, 7, 2, 21, 30, tzinfo=JST),  # 7月
    datetime(2026, 8, 7, 21, 30, tzinfo=JST),  # 8月
    datetime(2026, 9, 4, 21, 30, tzinfo=JST),  # 9月
    datetime(2026, 10, 2, 21, 30, tzinfo=JST), # 10月
    datetime(2026, 11, 6, 22, 30, tzinfo=JST), # 11月（冬時間）
    datetime(2026, 12, 4, 22, 30, tzinfo=JST), # 12月（冬時間）
]

# 米国CPI（消費者物価指数）: 毎月中旬 8:30 ET → 21:30/22:30 JST
_CPI_2026 = [
    datetime(2026, 1, 14, 22, 30, tzinfo=JST),  # 1月（冬時間）
    datetime(2026, 2, 11, 22, 30, tzinfo=JST),  # 2月
    datetime(2026, 3, 11, 22, 30, tzinfo=JST),  # 3月
    datetime(2026, 4, 14, 21, 30, tzinfo=JST),  # 4月（夏時間）
    datetime(2026, 5, 12, 21, 30, tzinfo=JST),  # 5月
    datetime(2026, 6, 10, 21, 30, tzinfo=JST),  # 6月
    datetime(2026, 7, 14, 21, 30, tzinfo=JST),  # 7月
    datetime(2026, 8, 12, 21, 30, tzinfo=JST),  # 8月
    datetime(2026, 9, 11, 21, 30, tzinfo=JST),  # 9月
    datetime(2026, 10, 13, 21, 30, tzinfo=JST), # 10月
    datetime(2026, 11, 12, 22, 30, tzinfo=JST), # 11月（冬時間）
    datetime(2026, 12, 10, 22, 30, tzinfo=JST), # 12月
]

# BOJ金融政策決定会合: 年8回、通常2日間の最終日に発表（時刻不定、概ね12:00 JST前後）
_BOJ_2026 = [
    datetime(2026, 1, 24, 12, 0, tzinfo=JST),
    datetime(2026, 3, 19, 12, 0, tzinfo=JST),
    datetime(2026, 4, 28, 12, 0, tzinfo=JST),
    datetime(2026, 6, 18, 12, 0, tzinfo=JST),
    datetime(2026, 7, 30, 12, 0, tzinfo=JST),
    datetime(2026, 9, 17, 12, 0, tzinfo=JST),
    datetime(2026, 10, 29, 12, 0, tzinfo=JST),
    datetime(2026, 12, 18, 12, 0, tzinfo=JST),
]

# ECB金融政策決定: 年8回、通常木曜日 13:45 CET → 21:45 JST
_ECB_2026 = [
    datetime(2026, 1, 22, 22, 15, tzinfo=JST),  # 冬時間 (CET+8h)
    datetime(2026, 3, 5, 22, 15, tzinfo=JST),
    datetime(2026, 4, 16, 21, 15, tzinfo=JST),   # 夏時間 (CEST+7h)
    datetime(2026, 6, 4, 21, 15, tzinfo=JST),
    datetime(2026, 7, 16, 21, 15, tzinfo=JST),
    datetime(2026, 9, 10, 21, 15, tzinfo=JST),
    datetime(2026, 10, 29, 22, 15, tzinfo=JST),  # 冬時間
    datetime(2026, 12, 10, 22, 15, tzinfo=JST),
]

# 米国GDP速報値: 四半期末の月末 8:30 ET → 21:30/22:30 JST
_GDP_2026 = [
    datetime(2026, 1, 29, 22, 30, tzinfo=JST),  # Q4 2025速報（冬時間）
    datetime(2026, 4, 29, 21, 30, tzinfo=JST),  # Q1 2026速報（夏時間）
    datetime(2026, 7, 29, 21, 30, tzinfo=JST),  # Q2 2026速報
    datetime(2026, 10, 29, 22, 30, tzinfo=JST), # Q3 2026速報（冬時間）
]


def _build_event_list() -> List[Dict]:
    """
    全ハードコードイベントを統一フォーマットの辞書リストに変換する。

    Returns:
        イベント辞書のリスト。各辞書は以下のキーを持つ:
            - name: イベント名
            - datetime_jst: JST日時 (datetime)
            - impact: 影響度 ("high")
            - source: "hardcoded" or "custom"
    """
    events = []

    for dt in _FOMC_2026:
        events.append({
            "name": "FOMC政策金利発表",
            "datetime_jst": dt,
            "impact": "high",
            "source": "hardcoded",
        })

    for dt in _NFP_2026:
        events.append({
            "name": "米国雇用統計(NFP)",
            "datetime_jst": dt,
            "impact": "high",
            "source": "hardcoded",
        })

    for dt in _CPI_2026:
        events.append({
            "name": "米国消費者物価指数(CPI)",
            "datetime_jst": dt,
            "impact": "high",
            "source": "hardcoded",
        })

    for dt in _BOJ_2026:
        events.append({
            "name": "BOJ金融政策決定会合",
            "datetime_jst": dt,
            "impact": "high",
            "source": "hardcoded",
        })

    for dt in _ECB_2026:
        events.append({
            "name": "ECB政策金利発表",
            "datetime_jst": dt,
            "impact": "high",
            "source": "hardcoded",
        })

    for dt in _GDP_2026:
        events.append({
            "name": "米国GDP速報値",
            "datetime_jst": dt,
            "impact": "high",
            "source": "hardcoded",
        })

    # 日時順にソート
    events.sort(key=lambda e: e["datetime_jst"])
    return events


# モジュールレベルのイベントリスト（ハードコード分）
_EVENTS: List[Dict] = _build_event_list()

# カスタムイベント（CSV読み込み分）を追加格納するリスト
_CUSTOM_EVENTS: List[Dict] = []


def _get_all_events() -> List[Dict]:
    """ハードコード＋カスタムイベントを統合して日時順で返す。"""
    combined = _EVENTS + _CUSTOM_EVENTS
    combined.sort(key=lambda e: e["datetime_jst"])
    return combined


def is_safe_to_trade(
    current_time: datetime,
    buffer_hours_before: float = 4.0,
    buffer_hours_after: float = 2.0,
) -> Tuple[bool, Optional[str]]:
    """
    現在時刻が高インパクトイベントのバッファゾーン内かどうかを判定する。

    Args:
        current_time: 判定対象の日時（timezone-awareであること。naiveの場合JSTとみなす）
        buffer_hours_before: イベント前の取引停止時間（デフォルト4時間）
        buffer_hours_after: イベント後の取引停止時間（デフォルト2時間）

    Returns:
        (is_safe, event_name):
            - is_safe: True = 取引可能、False = 取引停止推奨
            - event_name: 取引停止の原因イベント名（安全な場合はNone）
    """
    # naive datetimeの場合はJSTとみなす
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=JST)

    before_delta = timedelta(hours=buffer_hours_before)
    after_delta = timedelta(hours=buffer_hours_after)

    for event in _get_all_events():
        event_dt = event["datetime_jst"]
        # バッファゾーン: [event_dt - before, event_dt + after]
        if (event_dt - before_delta) <= current_time <= (event_dt + after_delta):
            return (False, event["name"])

    return (True, None)


def get_upcoming_events(
    current_time: datetime,
    days_ahead: int = 7,
) -> List[Dict]:
    """
    指定期間内の今後のイベント一覧を返す。

    Args:
        current_time: 基準日時（timezone-awareであること。naiveの場合JSTとみなす）
        days_ahead: 何日先までのイベントを取得するか（デフォルト7日）

    Returns:
        イベント辞書のリスト（日時順）。各辞書のキー:
            - name: イベント名
            - datetime_jst: JST日時
            - impact: 影響度
            - source: "hardcoded" or "custom"
            - hours_until: 現在時刻からの残り時間（小数）
    """
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=JST)

    cutoff = current_time + timedelta(days=days_ahead)
    upcoming = []

    for event in _get_all_events():
        event_dt = event["datetime_jst"]
        if current_time <= event_dt <= cutoff:
            hours_until = (event_dt - current_time).total_seconds() / 3600.0
            upcoming.append({
                **event,
                "hours_until": round(hours_until, 1),
            })

    return upcoming


def load_custom_events(csv_path: str) -> int:
    """
    CSVファイルからカスタムイベントを読み込んで追加する。

    CSVフォーマット（ヘッダー行必須）:
        name,datetime_jst,impact
        例: "RBA政策金利発表,2026-04-07 12:30,high"

    datetime_jstは "YYYY-MM-DD HH:MM" 形式（JSTとして解釈）。
    impactは "high", "medium", "low" のいずれか。

    Args:
        csv_path: CSVファイルのパス

    Returns:
        読み込んだイベント数

    Raises:
        FileNotFoundError: CSVファイルが見つからない場合
        ValueError: CSVのフォーマットが不正な場合
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"カスタムイベントCSVが見つかりません: {csv_path}")

    loaded_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 必須カラムの確認
        required_cols = {"name", "datetime_jst", "impact"}
        if reader.fieldnames is None:
            raise ValueError("CSVにヘッダー行がありません")
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSVに必須カラムがありません: {missing}")

        for row_num, row in enumerate(reader, start=2):
            try:
                dt = datetime.strptime(row["datetime_jst"].strip(), "%Y-%m-%d %H:%M")
                dt = dt.replace(tzinfo=JST)
            except ValueError:
                raise ValueError(
                    f"行{row_num}: datetime_jstのフォーマットが不正です "
                    f"(期待: 'YYYY-MM-DD HH:MM', 実際: '{row['datetime_jst']}')"
                )

            impact = row["impact"].strip().lower()
            if impact not in ("high", "medium", "low"):
                raise ValueError(
                    f"行{row_num}: impactは 'high', 'medium', 'low' のいずれかを指定してください "
                    f"(実際: '{row['impact']}')"
                )

            _CUSTOM_EVENTS.append({
                "name": row["name"].strip(),
                "datetime_jst": dt,
                "impact": impact,
                "source": "custom",
            })
            loaded_count += 1

    return loaded_count


def clear_custom_events() -> None:
    """読み込み済みのカスタムイベントをすべてクリアする。"""
    _CUSTOM_EVENTS.clear()


# ─── 動作確認用 ───
if __name__ == "__main__":
    now = datetime.now(tz=JST)
    print(f"現在時刻 (JST): {now.strftime('%Y-%m-%d %H:%M')}")
    print()

    # 取引可否チェック
    safe, event_name = is_safe_to_trade(now)
    if safe:
        print("✔ 現在、取引可能です。")
    else:
        print(f"✘ 取引停止推奨: {event_name} の前後バッファゾーン内です。")
    print()

    # 今後7日間のイベント
    upcoming = get_upcoming_events(now, days_ahead=7)
    if upcoming:
        print(f"--- 今後7日間のイベント ({len(upcoming)}件) ---")
        for ev in upcoming:
            dt_str = ev["datetime_jst"].strftime("%Y-%m-%d %H:%M JST")
            print(f"  {dt_str}  {ev['name']}  (あと{ev['hours_until']}時間)")
    else:
        print("今後7日間に高インパクトイベントはありません。")
