"""
自動リサーチパイプライン
======================
新しい特徴量候補をWalk-Forward検証で自動テストし、
パフォーマンスが改善する特徴量のみを採用する。

安全方針:
  - 信頼済みAPI（FRED, NewsAPI, yfinance）のみ使用
  - 任意のURL・外部コード実行・ファイルダウンロード禁止
  - 書き込みはdata/ディレクトリのみ
  - モデル変更にはWalk-Forward検証必須（PF > 1.3 かつ期待値 > 0）
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from research.common.data_loader import load_usdjpy_1h
from research.common.features import FEATURE_COLS, prepare_dataset
from research.common.ensemble import EnsembleClassifier
from research.common.validation import walk_forward_splits, compute_metrics

# ===== 環境変数 =====
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ===== ログ設定 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ===== Walk-Forward設定 =====
MIN_TRAIN_SIZE = 4320   # 6ヶ月分（30*24*6）
TEST_SIZE = 720         # 1ヶ月分（30*24）
CONF_THRESHOLD = 0.70   # 信頼度しきい値
AGREE_THRESHOLD = 5     # 一致度しきい値

# ===== 採用基準 =====
MIN_PF = 1.3            # 最低プロフィットファクター
MIN_IMPROVEMENT = 0.05  # 最低改善率（5%）
REQUEST_TIMEOUT = 30    # APIリクエストタイムアウト（秒）


# =========================================================
# 1. 候補特徴量リスト
# =========================================================

CANDIDATE_FEATURES = [
    {
        "name": "VIX_change",
        "source": "yfinance",
        "ticker": "^VIX",
        "description": "VIX恐怖指数の変化率 - 市場の恐怖モメンタム",
    },
    {
        "name": "US10Y_change",
        "source": "fred",
        "series_id": "DGS10",
        "description": "米国10年債利回りの変化 - 金利動向",
    },
    {
        "name": "Yield_spread_2y10y",
        "source": "fred_spread",
        "series_ids": ["DGS2", "DGS10"],
        "description": "米国2年-10年利回りスプレッド - 景気後退指標",
    },
    {
        "name": "Gold_change",
        "source": "yfinance",
        "ticker": "GC=F",
        "description": "金価格の変化率 - 安全資産フロー",
    },
    {
        "name": "Oil_change",
        "source": "yfinance",
        "ticker": "CL=F",
        "description": "原油価格の変化率 - コモディティ動向",
    },
    {
        "name": "News_sentiment",
        "source": "newsapi",
        "query": "USD JPY forex",
        "description": "ニュースセンチメントスコア - 市場心理",
    },
    {
        "name": "DXY_change",
        "source": "yfinance",
        "ticker": "DX-Y.NYB",
        "description": "ドルインデックスの変化率 - ドル全体の強弱",
    },
    {
        "name": "SP500_overnight",
        "source": "yfinance",
        "ticker": "^GSPC",
        "description": "S&P500のオーバーナイトリターン - リスクオン/オフ",
    },
]


# =========================================================
# 2. データ取得関数（安全なAPI呼び出しのみ）
# =========================================================

def fetch_fred_series(series_id: str, start_date: str = None) -> pd.Series:
    """
    FRED APIから経済指標データを取得する

    安全: api.stlouisfed.org のみにアクセス
    Returns: pd.Series（日付インデックス）or None（失敗時）
    """
    import requests

    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY が未設定です")
        return None

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "asc",
        }
        if start_date:
            params["observation_start"] = start_date

        logger.info(f"FRED API取得中: {series_id}")
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"FRED {series_id}: データなし")
            return None

        dates = []
        values = []
        for obs in observations:
            if obs["value"] != ".":
                dates.append(pd.Timestamp(obs["date"]))
                values.append(float(obs["value"]))

        series = pd.Series(values, index=dates, name=series_id)
        logger.info(f"FRED {series_id}: {len(series)}件取得完了")
        return series

    except Exception as e:
        logger.error(f"FRED {series_id} 取得エラー: {e}")
        return None


def fetch_yfinance_series(ticker: str, period: str = "5y",
                          column: str = "Close") -> pd.Series:
    """
    yfinanceから株価・指数データを取得する

    安全: yfinanceライブラリ経由のみ
    Returns: pd.Series（日付インデックス）or None（失敗時）
    """
    try:
        import yfinance as yf

        logger.info(f"yfinance取得中: {ticker}")
        data = yf.download(ticker, period=period, progress=False, timeout=REQUEST_TIMEOUT)

        if data is None or data.empty:
            logger.warning(f"yfinance {ticker}: データなし")
            return None

        # マルチカラムの場合の対応
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if column not in data.columns:
            logger.warning(f"yfinance {ticker}: '{column}'カラムが見つかりません")
            return None

        series = data[column].dropna()
        series.name = ticker
        logger.info(f"yfinance {ticker}: {len(series)}件取得完了")
        return series

    except Exception as e:
        logger.error(f"yfinance {ticker} 取得エラー: {e}")
        return None


def fetch_news_sentiment(query: str, days_back: int = 30) -> pd.Series:
    """
    NewsAPIからニュースを取得し、簡易センチメントスコアを算出する

    方法: ヘッドラインのポジティブ/ネガティブ単語をカウント
    安全: newsapi.org/v2/everything のみにアクセス
    Returns: pd.Series（日付インデックス、-1〜+1）or None（失敗時）
    """
    import requests

    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY が未設定です")
        return None

    # ポジティブ・ネガティブ単語辞書（FX関連）
    positive_words = {
        "rally", "surge", "gain", "rise", "bullish", "strong",
        "recovery", "boost", "advance", "optimism", "growth",
        "upbeat", "positive", "soar", "climb", "strengthen",
    }
    negative_words = {
        "fall", "drop", "decline", "crash", "bearish", "weak",
        "recession", "plunge", "slump", "fear", "risk",
        "downbeat", "negative", "tumble", "collapse", "weaken",
    }

    try:
        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": 100,
        }

        logger.info(f"NewsAPI取得中: query='{query}', {days_back}日分")
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            logger.warning("NewsAPI: 記事が見つかりません")
            return None

        # 日別にセンチメントを集計
        daily_scores = {}
        for article in articles:
            pub_date_str = article.get("publishedAt", "")
            if not pub_date_str:
                continue

            pub_date = pd.Timestamp(pub_date_str[:10])
            title = (article.get("title") or "").lower()
            desc = (article.get("description") or "").lower()
            text = title + " " + desc

            words = set(text.split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            total = pos_count + neg_count

            score = (pos_count - neg_count) / total if total > 0 else 0.0

            if pub_date not in daily_scores:
                daily_scores[pub_date] = []
            daily_scores[pub_date].append(score)

        # 日別平均
        dates = sorted(daily_scores.keys())
        avg_scores = [np.mean(daily_scores[d]) for d in dates]

        series = pd.Series(avg_scores, index=dates, name="sentiment")
        logger.info(f"NewsAPI: {len(series)}日分のセンチメント算出完了")
        return series

    except Exception as e:
        logger.error(f"NewsAPI取得エラー: {e}")
        return None


# =========================================================
# 3. 特徴量エンジニアリング
# =========================================================

def create_candidate_feature(df: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    """
    候補特徴量をDataFrameに追加する

    日足データ → 1時間足にforward-fillで展開
    変化率（pct_change）に変換して特徴量化

    Parameters:
        df: ベースの1時間足DataFrame（DatetimeIndex）
        candidate: 候補設定dict

    Returns:
        新しい特徴量カラムが追加されたDataFrame（失敗時はNone）
    """
    name = candidate["name"]
    source = candidate["source"]
    series = None

    try:
        # --- データ取得 ---
        if source == "yfinance":
            series = fetch_yfinance_series(candidate["ticker"])

        elif source == "fred":
            series = fetch_fred_series(candidate["series_id"])

        elif source == "fred_spread":
            # 2つのFREDシリーズの差分（スプレッド）
            ids = candidate["series_ids"]
            s1 = fetch_fred_series(ids[0])  # 短期（2Y）
            s2 = fetch_fred_series(ids[1])  # 長期（10Y）
            if s1 is not None and s2 is not None:
                combined = pd.DataFrame({"short": s1, "long": s2}).dropna()
                series = combined["long"] - combined["short"]
                series.name = name

        elif source == "newsapi":
            series = fetch_news_sentiment(candidate["query"])

        else:
            logger.warning(f"未対応のソース: {source}")
            return None

        if series is None:
            logger.warning(f"特徴量 '{name}': データ取得失敗")
            return None

        # --- 1時間足に展開（forward-fill） ---
        # 日足・非規則データ → 1時間足のインデックスに合わせる
        series.index = pd.DatetimeIndex(series.index)
        series = series[~series.index.duplicated(keep="last")]
        series = series.reindex(df.index, method="ffill")

        # --- 特徴量化 ---
        if source == "newsapi":
            # センチメントはそのまま使用（既に-1〜+1のスコア）
            df[name] = series.fillna(0.0)
        elif source == "fred_spread":
            # スプレッドはそのまま使用
            df[name] = series.fillna(method="ffill").fillna(0.0)
        else:
            # 価格系は変化率に変換
            df[name] = series.pct_change().fillna(0.0)

        # NaN残りを0で埋め
        df[name] = df[name].fillna(0.0)

        # inf対策
        df[name] = df[name].replace([np.inf, -np.inf], 0.0)

        logger.info(f"特徴量 '{name}' 作成完了 (非ゼロ: {(df[name] != 0).sum()}件)")
        return df

    except Exception as e:
        logger.error(f"特徴量 '{name}' 作成エラー: {e}")
        return None


# =========================================================
# 4. Walk-Forward検証
# =========================================================

def _run_walk_forward(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Walk-Forward検証を実行し、評価指標を返す

    Parameters:
        df: prepare_dataset済みのDataFrame
        feature_cols: 使用する特徴量カラムリスト

    Returns:
        compute_metricsの結果dict
    """
    # 使用可能な特徴量のみフィルタ
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logger.warning(f"欠損特徴量（スキップ）: {missing}")

    if not available_cols:
        logger.error("使用可能な特徴量がありません")
        return compute_metrics(np.array([]))

    X = df[available_cols].values
    y = df["Label"].values
    returns_4h = df["Return_4h"].values

    splits = walk_forward_splits(
        n_total=len(X),
        min_train_size=MIN_TRAIN_SIZE,
        test_size=TEST_SIZE,
    )

    if not splits:
        logger.error("Walk-Forward分割が生成できません（データ不足）")
        return compute_metrics(np.array([]))

    all_trade_returns = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        r_test = returns_4h[test_idx]

        # アンサンブル学習
        model = EnsembleClassifier()
        model.fit(X_train, y_train)

        # 予測 + フィルタリング
        preds, agreement = model.predict_with_agreement(X_test)
        probas = model.predict_proba(X_test)
        confidence = probas.max(axis=1)

        # 信頼度 >= 0.70 かつ 全員一致(5) のトレードのみ
        mask = (confidence >= CONF_THRESHOLD) & (agreement >= AGREE_THRESHOLD)
        directions = np.where(preds == 1, 1.0, -1.0)
        trade_returns = (directions * r_test)[mask]

        all_trade_returns.extend(trade_returns.tolist())

    return compute_metrics(np.array(all_trade_returns))


def test_feature(df: pd.DataFrame, base_feature_cols: list,
                 new_feature_name: str) -> dict:
    """
    新しい特徴量をWalk-Forward検証でテストする

    ベースライン（既存特徴量のみ）と新特徴量追加時を比較し、
    採用可否を判定する。

    Parameters:
        df: prepare_dataset済みのDataFrame（新特徴量カラム追加済み）
        base_feature_cols: 既存の特徴量カラムリスト
        new_feature_name: テスト対象の新特徴量名

    Returns:
        dict: {
            feature_name, pf_baseline, pf_with_feature,
            improvement, ev_baseline, ev_with_feature,
            accepted, reason
        }
    """
    logger.info(f"=== Walk-Forward検証開始: {new_feature_name} ===")

    # ベースライン
    logger.info("ベースライン（既存特徴量のみ）を検証中...")
    metrics_base = _run_walk_forward(df, base_feature_cols)
    pf_base = metrics_base["pf"]
    ev_base = metrics_base["exp_value_net"]

    # 新特徴量追加
    extended_cols = base_feature_cols + [new_feature_name]
    logger.info(f"新特徴量追加版を検証中: +{new_feature_name}")
    metrics_new = _run_walk_forward(df, extended_cols)
    pf_new = metrics_new["pf"]
    ev_new = metrics_new["exp_value_net"]

    # 改善率計算
    if np.isnan(pf_base) or pf_base == 0:
        improvement = 0.0
    else:
        improvement = (pf_new - pf_base) / pf_base

    # 採用判定
    accepted = False
    reason = ""

    if np.isnan(pf_new):
        reason = "PFが計算不能（トレード数不足）"
    elif pf_new < MIN_PF:
        reason = f"PF {pf_new:.2f} < 基準 {MIN_PF}"
    elif ev_new <= 0:
        reason = f"期待値 {ev_new:.6f} <= 0"
    elif improvement < MIN_IMPROVEMENT:
        reason = f"改善率 {improvement:.1%} < 基準 {MIN_IMPROVEMENT:.0%}"
    else:
        accepted = True
        reason = f"採用OK: PF {pf_base:.2f} → {pf_new:.2f} (+{improvement:.1%})"

    result = {
        "feature_name": new_feature_name,
        "pf_baseline": round(pf_base, 4) if not np.isnan(pf_base) else None,
        "pf_with_feature": round(pf_new, 4) if not np.isnan(pf_new) else None,
        "improvement": round(improvement, 4),
        "ev_baseline": round(ev_base, 6) if not np.isnan(ev_base) else None,
        "ev_with_feature": round(ev_new, 6) if not np.isnan(ev_new) else None,
        "n_trades_baseline": metrics_base["n_trades"],
        "n_trades_with_feature": metrics_new["n_trades"],
        "win_rate_baseline": round(metrics_base["win_rate"], 2) if not np.isnan(metrics_base["win_rate"]) else None,
        "win_rate_with_feature": round(metrics_new["win_rate"], 2) if not np.isnan(metrics_new["win_rate"]) else None,
        "accepted": accepted,
        "reason": reason,
    }

    status = "採用" if accepted else "不採用"
    logger.info(f"結果: [{status}] {reason}")
    return result


# =========================================================
# 5. メインパイプライン
# =========================================================

def send_telegram_message(text: str):
    """
    Telegram経由でレポートを送信する

    安全: api.telegram.org のみにアクセス
    """
    import requests as req

    chat_id_file = DATA_DIR / "telegram_chat_id.txt"
    if not TELEGRAM_BOT_TOKEN or not chat_id_file.exists():
        logger.warning("Telegram未設定（送信スキップ）")
        return

    chat_id = chat_id_file.read_text().strip()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Telegramメッセージ上限（4096文字）
    if len(text) > 4000:
        text = text[:4000] + "\n...(省略)"

    try:
        resp = req.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        logger.info("Telegramレポート送信完了")
    except Exception as e:
        logger.error(f"Telegram送信エラー: {e}")


def run_research(pair: str = "USDJPY") -> list:
    """
    自動リサーチパイプラインを実行する

    手順:
      1. ベースデータ読み込み
      2. 各候補特徴量について:
         a. データ取得
         b. 特徴量作成
         c. Walk-Forward検証
      3. レポート生成・保存・送信
      4. 採用特徴量リストを返却

    Returns:
        採用された特徴量名のリスト
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"自動リサーチパイプライン開始: {pair}")
    logger.info(f"候補特徴量数: {len(CANDIDATE_FEATURES)}")
    logger.info("=" * 60)

    # --- 1. データ読み込み ---
    logger.info("ベースデータ読み込み中...")
    df = load_usdjpy_1h()
    df = prepare_dataset(df, FEATURE_COLS)
    logger.info(f"データ読み込み完了: {len(df)}行, 期間: {df.index[0]} ~ {df.index[-1]}")

    # --- 2. 各候補を検証 ---
    results = []
    accepted_features = []

    for i, candidate in enumerate(CANDIDATE_FEATURES, 1):
        name = candidate["name"]
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{i}/{len(CANDIDATE_FEATURES)}] 検証中: {name}")
        logger.info(f"  説明: {candidate['description']}")
        logger.info(f"  ソース: {candidate['source']}")

        # データフレームのコピーで作業（他の候補に影響しないように）
        df_test = df.copy()

        # 特徴量作成
        df_result = create_candidate_feature(df_test, candidate)
        if df_result is None:
            result = {
                "feature_name": name,
                "accepted": False,
                "reason": "データ取得または特徴量作成失敗",
                "pf_baseline": None,
                "pf_with_feature": None,
                "improvement": 0.0,
            }
            results.append(result)
            logger.warning(f"  → スキップ: データ取得失敗")
            continue

        # NaN行を落とす
        df_result = df_result.dropna(subset=FEATURE_COLS + [name, "Label", "Return_4h"])

        # Walk-Forward検証
        result = test_feature(df_result, list(FEATURE_COLS), name)
        results.append(result)

        if result["accepted"]:
            accepted_features.append(name)

        # API負荷軽減のため少し待機
        time.sleep(1)

    # --- 3. レポート生成 ---
    elapsed = time.time() - start_time
    report = _generate_report(pair, results, accepted_features, elapsed)

    # コンソール出力
    print("\n" + report)

    # Telegram送信
    send_telegram_message(report)

    # CSV保存
    _save_results(results)

    logger.info(f"\nパイプライン完了: {elapsed:.0f}秒")
    logger.info(f"採用された特徴量: {accepted_features if accepted_features else 'なし'}")

    return accepted_features


def _generate_report(pair: str, results: list, accepted: list,
                     elapsed: float) -> str:
    """テキストレポートを生成する"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"<b>自動リサーチレポート</b>",
        f"日時: {now}",
        f"通貨ペア: {pair}",
        f"検証時間: {elapsed:.0f}秒",
        f"候補数: {len(results)}",
        f"採用数: {len(accepted)}",
        "",
        "━━━ 検証結果一覧 ━━━",
    ]

    for r in results:
        status = "OK" if r["accepted"] else "NG"
        pf_base = f"{r['pf_baseline']:.2f}" if r.get("pf_baseline") is not None else "N/A"
        pf_new = f"{r['pf_with_feature']:.2f}" if r.get("pf_with_feature") is not None else "N/A"
        imp = f"{r.get('improvement', 0):.1%}"

        lines.append(f"\n[{status}] {r['feature_name']}")
        lines.append(f"  PF: {pf_base} → {pf_new} ({imp})")
        lines.append(f"  理由: {r.get('reason', '')}")

    if accepted:
        lines.append(f"\n━━━ 採用された特徴量 ━━━")
        for name in accepted:
            lines.append(f"  + {name}")
    else:
        lines.append(f"\n今回は採用される特徴量はありませんでした。")

    return "\n".join(lines)


def _save_results(results: list):
    """検証結果をCSVに保存する"""
    try:
        filepath = DATA_DIR / "research_results.csv"
        df_results = pd.DataFrame(results)
        df_results["timestamp"] = datetime.now().isoformat()

        # 既存ファイルがあれば追記
        if filepath.exists():
            df_existing = pd.read_csv(filepath)
            df_results = pd.concat([df_existing, df_results], ignore_index=True)

        df_results.to_csv(filepath, index=False)
        logger.info(f"検証結果を保存: {filepath}")
    except Exception as e:
        logger.error(f"結果保存エラー: {e}")


# =========================================================
# 6. 自動採用関数（安全策: JSONに保存のみ）
# =========================================================

def adopt_features(accepted_features: list):
    """
    採用された特徴量をJSONファイルに保存する

    安全策: コードファイルは自動変更しない。
    人間がレビューしてから手動でFEATURE_COLSに追加する。

    Parameters:
        accepted_features: 採用された特徴量名のリスト
    """
    if not accepted_features:
        logger.info("採用された特徴量はありません（保存スキップ）")
        return

    filepath = DATA_DIR / "adopted_features.json"

    # 既存データ読み込み
    existing = {}
    if filepath.exists():
        try:
            existing = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # 更新
    history = existing.get("history", [])
    history.append({
        "timestamp": datetime.now().isoformat(),
        "features": accepted_features,
    })

    # 全採用済み特徴量（重複除去）
    all_adopted = list(set(existing.get("all_adopted", []) + accepted_features))

    output = {
        "all_adopted": sorted(all_adopted),
        "latest_run": datetime.now().isoformat(),
        "latest_features": accepted_features,
        "history": history,
    }

    filepath.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"採用特徴量を保存: {filepath}")

    # 手動採用の手順を表示
    print("\n" + "=" * 50)
    print("【手動採用の手順】")
    print("=" * 50)
    print("以下の特徴量がWalk-Forward検証に合格しました。")
    print("手動でコードに組み込んでください:\n")
    for feat in accepted_features:
        print(f"  1. research/common/features.py の FEATURE_COLS に '{feat}' を追加")
        print(f"  2. データ取得ロジックをpaper_trade.pyに組み込む")
        print(f"  3. 再度Walk-Forward検証を実行して確認")
        print()
    print(f"詳細: {filepath}")
    print("=" * 50)


# =========================================================
# 7. スケジューラ統合
# =========================================================

def run_daily_research():
    """
    日次リサーチを実行する

    1日1回呼び出し、全候補を検証してレポート送信
    """
    logger.info("=" * 60)
    logger.info("日次自動リサーチ実行")
    logger.info(f"実行日時: {datetime.now()}")
    logger.info("=" * 60)

    try:
        accepted = run_research(pair="USDJPY")

        if accepted:
            adopt_features(accepted)
            summary = f"日次リサーチ完了: {len(accepted)}件の新特徴量を採用候補として保存"
        else:
            summary = "日次リサーチ完了: 採用基準を満たす特徴量はありませんでした"

        logger.info(summary)
        send_telegram_message(f"<b>{summary}</b>")

    except Exception as e:
        error_msg = f"日次リサーチでエラー発生: {e}"
        logger.error(error_msg, exc_info=True)
        send_telegram_message(f"<b>エラー</b>\n{error_msg}")


# =========================================================
# メイン実行
# =========================================================

if __name__ == "__main__":
    logger.info("自動リサーチパイプラインを起動します")
    accepted = run_research(pair="USDJPY")

    if accepted:
        adopt_features(accepted)
        logger.info(f"完了: {len(accepted)}件の特徴量が採用候補に")
    else:
        logger.info("完了: 今回は採用候補なし")
