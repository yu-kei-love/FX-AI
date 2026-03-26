"""
CFTC（米商品先物取引委員会）ポジションデータモジュール

CFTCは毎週火曜日時点の先物ポジションデータを金曜日に公開する。
「Commitments of Traders (COT)」レポートと呼ばれ、以下の3グループの建玉を公開：

  - Large Speculators（大口投機筋）: ヘッジファンドなど。トレンドを追う傾向
  - Commercials（商業筋/ヘッジャー）: 実需企業。逆張り傾向（本業のリスクヘッジ）
  - Small Traders（小口トレーダー）: 個人投資家など。遅れて動く傾向

なぜ使うのか:
  - 大口投機筋のネットポジション（買い - 売り）が増えている
    → 「プロが円安方向に賭けている」と解釈できる
  - 商業筋が逆に動いている
    → 「実需企業がヘッジを増やしている」= 相場転換の兆候かもしれない
  - ポジションの急変は相場の転換点を示唆することが多い

データソース: CFTC公式サイト（無料・合法）
  - 日本円先物のCOTレポート（週次データ）
"""

import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "data").resolve()

# CFTC COTレポートのURL（年ごとのCSV）
# "Futures Only" の "Financial" レポートを使用
# Contract code 099741 = Japanese Yen (CME)
CFTC_BASE_URL = "https://www.cftc.gov/files/dea/history"
JPY_CONTRACT_CODE = "099741"  # Japanese Yen futures (CME)


def _download_cot_year(year: int) -> pd.DataFrame:
    """1年分のCOTデータをCFTCからダウンロードする"""
    # Financial Futures レポート（通貨先物を含む）
    url = f"{CFTC_BASE_URL}/fut_fin_txt_{year}.zip"
    print(f"  CFTC COT {year}年データをダウンロード中... ({url})")

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    # ZIPファイルからCSVを読み込む
    with ZipFile(io.BytesIO(r.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".txt") or n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"ZIP内にCSV/TXTファイルが見つかりません: {zf.namelist()}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f)

    return df


def _filter_jpy(df: pd.DataFrame) -> pd.DataFrame:
    """日本円先物のデータだけを抽出する"""
    # CFTC_Contract_Market_Code列でフィルタリング
    # カラム名はレポートによって異なる場合がある
    code_col = None
    for candidate in ["CFTC_Contract_Market_Code", "CFTC Contract Market Code",
                       "cftc_contract_market_code"]:
        if candidate in df.columns:
            code_col = candidate
            break

    if code_col is None:
        # Market_and_Exchange_Names列で "JAPANESE YEN" を検索
        name_col = None
        for candidate in ["Market_and_Exchange_Names", "Market and Exchange Names"]:
            if candidate in df.columns:
                name_col = candidate
                break
        if name_col is None:
            raise ValueError(f"JPYデータのフィルタリングに使えるカラムが見つかりません: {df.columns.tolist()[:10]}")
        mask = df[name_col].str.contains("JAPANESE YEN", case=False, na=False)
    else:
        mask = df[code_col].astype(str).str.strip() == JPY_CONTRACT_CODE

    jpy_df = df[mask].copy()
    if jpy_df.empty:
        print(f"  警告: 日本円先物のデータが見つかりません")
    return jpy_df


def _extract_positions(df: pd.DataFrame) -> pd.DataFrame:
    """COTデータからポジション情報を抽出する

    Financial Futures レポートのカラム構造:
    - Asset_Mgr_Positions_Long/Short_All  → 資産運用会社（投機筋として扱う）
    - Lev_Money_Positions_Long/Short_All  → レバレッジドマネー（ヘッジファンド）
    - Dealer_Positions_Long/Short_All     → ディーラー（商業筋として扱う）
    - Open_Interest_All                   → 総建玉
    """
    # 日付
    date_col = "Report_Date_as_YYYY-MM-DD"
    if date_col not in df.columns:
        for col in df.columns:
            if "date" in col.lower() and "yyyy" in col.lower():
                date_col = col
                break

    result = pd.DataFrame()
    result["Date"] = pd.to_datetime(df[date_col])

    # Asset Manager = 投機筋（spec）
    if "Asset_Mgr_Positions_Long_All" in df.columns:
        result["spec_long"] = pd.to_numeric(df["Asset_Mgr_Positions_Long_All"], errors="coerce")
        result["spec_short"] = pd.to_numeric(df["Asset_Mgr_Positions_Short_All"], errors="coerce")
        result["spec_net"] = result["spec_long"] - result["spec_short"]
    else:
        result["spec_net"] = 0

    # Dealer = 商業筋（comm）
    if "Dealer_Positions_Long_All" in df.columns:
        result["comm_long"] = pd.to_numeric(df["Dealer_Positions_Long_All"], errors="coerce")
        result["comm_short"] = pd.to_numeric(df["Dealer_Positions_Short_All"], errors="coerce")
        result["comm_net"] = result["comm_long"] - result["comm_short"]
    else:
        result["comm_net"] = 0

    # Leveraged Money = ヘッジファンド
    if "Lev_Money_Positions_Long_All" in df.columns:
        result["lev_long"] = pd.to_numeric(df["Lev_Money_Positions_Long_All"], errors="coerce")
        result["lev_short"] = pd.to_numeric(df["Lev_Money_Positions_Short_All"], errors="coerce")
        result["lev_net"] = result["lev_long"] - result["lev_short"]

    # 総建玉
    if "Open_Interest_All" in df.columns:
        result["open_interest"] = pd.to_numeric(df["Open_Interest_All"], errors="coerce")

    result = result.set_index("Date").sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result


def fetch_cot_data(start_year: int = 2020) -> pd.DataFrame:
    """複数年のCOTデータを取得して結合する"""
    current_year = pd.Timestamp.now().year
    all_data = []

    for year in range(start_year, current_year + 1):
        try:
            raw = _download_cot_year(year)
            jpy = _filter_jpy(raw)
            if jpy.empty:
                continue
            positions = _extract_positions(jpy)
            all_data.append(positions)
            print(f"  {year}年: {len(positions)}週分取得")
        except Exception as e:
            print(f"  {year}年: 取得失敗 ({e})")

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def compute_cot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    COTデータから特徴量を計算する

    特徴量:
    1. spec_net_norm: 投機筋ネットポジション（正規化）
       → プラス = 円買い（円高予想）が多い、マイナス = 円売り（円安予想）が多い
    2. spec_net_change: 投機筋ネットポジションの前週比変化
       → 急に増えた = 新しいトレンドの始まり？
    3. comm_net_norm: 商業筋ネットポジション（正規化）
       → 投機筋と逆方向なら相場転換のサイン？
    4. spec_net_zscore: 投機筋ネットポジションのZスコア（過去26週の中での位置）
       → +2以上 = 極端に偏っている = 反転リスク大
    5. oi_change: 総建玉の変化率
       → 建玉増加+トレンド = トレンド継続、建玉減少 = トレンド弱まり
    """
    result = pd.DataFrame(index=df.index)

    # 投機筋ネットポジション（OIで正規化）
    if "open_interest" in df.columns and "spec_net" in df.columns:
        oi = df["open_interest"].replace(0, np.nan)
        result["cot_spec_net_norm"] = df["spec_net"] / oi
    elif "spec_net" in df.columns:
        # OIがない場合は絶対値の移動平均で正規化
        rolling_abs = df["spec_net"].abs().rolling(26, min_periods=4).mean().replace(0, np.nan)
        result["cot_spec_net_norm"] = df["spec_net"] / rolling_abs
    else:
        result["cot_spec_net_norm"] = 0.0

    # 投機筋ネットポジションの前週比変化
    if "spec_net" in df.columns:
        result["cot_spec_net_change"] = df["spec_net"].diff()
        # 変化も正規化
        rolling_abs_change = result["cot_spec_net_change"].abs().rolling(26, min_periods=4).mean().replace(0, np.nan)
        result["cot_spec_net_change"] = result["cot_spec_net_change"] / rolling_abs_change
    else:
        result["cot_spec_net_change"] = 0.0

    # 商業筋ネットポジション（正規化）
    if "comm_net" in df.columns:
        if "open_interest" in df.columns:
            result["cot_comm_net_norm"] = df["comm_net"] / oi
        else:
            rolling_abs = df["comm_net"].abs().rolling(26, min_periods=4).mean().replace(0, np.nan)
            result["cot_comm_net_norm"] = df["comm_net"] / rolling_abs
    else:
        result["cot_comm_net_norm"] = 0.0

    # Zスコア（過去26週 = 約半年の中での位置）
    if "spec_net" in df.columns:
        rolling_mean = df["spec_net"].rolling(26, min_periods=8).mean()
        rolling_std = df["spec_net"].rolling(26, min_periods=8).std().replace(0, np.nan)
        result["cot_spec_zscore"] = (df["spec_net"] - rolling_mean) / rolling_std
    else:
        result["cot_spec_zscore"] = 0.0

    # 総建玉の変化率
    if "open_interest" in df.columns:
        result["cot_oi_change"] = df["open_interest"].pct_change()
    else:
        result["cot_oi_change"] = 0.0

    return result.fillna(0.0)


# 特徴量カラム名（外部から参照用）
COT_FEATURE_COLS = [
    "cot_spec_net_norm",
    "cot_spec_net_change",
    "cot_comm_net_norm",
    "cot_spec_zscore",
    "cot_oi_change",
]


def save_cot_data() -> pd.DataFrame:
    """COTデータを取得してCSVに保存する"""
    print("CFTC COTデータを取得中...")
    raw = fetch_cot_data(start_year=2020)
    if raw.empty:
        print("データが取得できませんでした")
        return pd.DataFrame()

    features = compute_cot_features(raw)
    # 生データと特徴量を結合して保存
    combined = pd.concat([raw, features], axis=1)
    path = DATA_DIR / "cftc_cot.csv"
    combined.to_csv(path)
    print(f"保存完了: {path} ({len(combined)}行)")
    return combined


def load_cot_data() -> pd.DataFrame:
    """保存済みのCOTデータを読み込む"""
    path = DATA_DIR / "cftc_cot.csv"
    if not path.exists():
        return save_cot_data()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def add_cot_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    1時間足のDataFrameにCFTCポジション特徴量を追加する

    週次データを1時間足に展開する方法:
    - COTレポートは毎週金曜に公開される（火曜時点のデータ）
    - 公開後、次の公開まで同じ値を使う（前方補完 = forward fill）
    - つまり「直近のCOTデータがどうだったか」を常に参照できる
    """
    try:
        df_cot = load_cot_data()
    except Exception as e:
        print(f"CFTC COTデータの読み込み失敗: {e}")
        for col in COT_FEATURE_COLS:
            df_hourly[col] = 0.0
        return df_hourly

    # 特徴量カラムだけ取り出す
    cot_cols = [c for c in COT_FEATURE_COLS if c in df_cot.columns]
    if not cot_cols:
        # 特徴量がない場合は生データから再計算
        features = compute_cot_features(df_cot)
        for col in COT_FEATURE_COLS:
            if col in features.columns:
                df_cot[col] = features[col]
        cot_cols = [c for c in COT_FEATURE_COLS if c in df_cot.columns]

    # 週次データを1時間足のインデックスに展開（前方補完）
    for col in cot_cols:
        series = df_cot[col].dropna()
        if series.empty:
            df_hourly[col] = 0.0
            continue
        df_hourly[col] = series.reindex(df_hourly.index, method="ffill").fillna(0.0)

    # 足りないカラムはゼロ埋め
    for col in COT_FEATURE_COLS:
        if col not in df_hourly.columns:
            df_hourly[col] = 0.0

    return df_hourly


# 直接実行時はデータ取得＆確認
if __name__ == "__main__":
    combined = save_cot_data()
    if not combined.empty:
        print(f"\n【最新のCOTデータ】")
        print(f"最新日: {combined.index[-1]}")
        for col in COT_FEATURE_COLS:
            if col in combined.columns:
                val = combined[col].iloc[-1]
                print(f"  {col}: {val:+.4f}")

        # 投機筋の生データも表示
        if "spec_net" in combined.columns:
            print(f"\n  投機筋ネットポジション: {combined['spec_net'].iloc[-1]:,.0f}枚")
        if "comm_net" in combined.columns:
            print(f"  商業筋ネットポジション: {combined['comm_net'].iloc[-1]:,.0f}枚")
        if "open_interest" in combined.columns:
            print(f"  総建玉: {combined['open_interest'].iloc[-1]:,.0f}枚")
