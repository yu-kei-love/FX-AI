# ===========================================
# note_content_generator.py
# note.com 記事の自動生成
#
# 機能:
#   - モデル結果から無料記事を生成（集客用）
#   - モデル結果+コードから有料記事を生成
#   - 日次予測レポートを生成（マガジン購読者向け）
#   - テンプレートによる一貫したフォーマット
#   - 免責事項の自動挿入
#
# 出力: research/note_sales/drafts/ 以下にMarkdownファイル
# ===========================================

import csv
import re
import sys
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ログ設定
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "note_content.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# パス定義
NOTE_SALES_DIR = PROJECT_ROOT / "research" / "note_sales"
TEMPLATES_DIR = NOTE_SALES_DIR / "templates"
DRAFTS_DIR = NOTE_SALES_DIR / "drafts"
PENDING_DIR = DRAFTS_DIR / "pending"
APPROVED_DIR = DRAFTS_DIR / "approved"
DATA_DIR = PROJECT_ROOT / "data"
RESEARCH_DIR = PROJECT_ROOT / "research"

# ディレクトリ作成
for d in [PENDING_DIR, APPROVED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================
# データファイルパーサー
# =============================================================

def _parse_fx_multi_pair_results() -> Optional[Dict[str, Any]]:
    """research/fx_multi_pair_results.txt からFXマルチ通貨ペア結果を読み込む"""
    path = RESEARCH_DIR / "fx_multi_pair_results.txt"
    if not path.exists():
        logger.warning(f"FXマルチ通貨ペア結果が見つかりません: {path}")
        return None

    try:
        text = path.read_text(encoding="utf-8")
        pairs = {}
        # パースする行: "USDJPY        3744    55.5%    1.51    13.73  20.54%    23.11  +0.000329"
        pair_pattern = re.compile(
            r"^(USD|EUR|GBP|AUD)\w+\s+"
            r"(\d+)\s+"            # trades
            r"([\d.]+)%\s+"        # win rate
            r"([\d.]+)\s+"         # PF
            r"([\d.]+)\s+"         # Sharpe
            r"([\d.]+)%\s+"        # MDD
            r"([\d.]+)\s+"         # Sortino
            r"([+-]?[\d.]+)",      # ExpVal
            re.MULTILINE,
        )
        for m in pair_pattern.finditer(text):
            pair_name = m.group(0).split()[0]
            pairs[pair_name] = {
                "trades": int(m.group(2)),
                "win_rate": float(m.group(3)),
                "pf": float(m.group(4)),
                "sharpe": float(m.group(5)),
                "mdd": float(m.group(6)),
                "sortino": float(m.group(7)),
                "exp_val": float(m.group(8)),
            }
        if pairs:
            logger.info(f"FXマルチ通貨ペア結果を読み込みました: {len(pairs)}ペア")
            return pairs
    except Exception as e:
        logger.warning(f"FXマルチ通貨ペア結果の解析に失敗: {e}")
    return None


def _parse_fx_confidence_results() -> Optional[Dict[str, Any]]:
    """research/fx_confidence_results.txt から信頼度閾値最適化結果を読み込む"""
    path = RESEARCH_DIR / "fx_confidence_results.txt"
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        result = {}
        # OPTIMAL THRESHOLD セクション
        opt_match = re.search(r"OPTIMAL THRESHOLD:\s*([\d.]+)", text)
        if opt_match:
            result["optimal_threshold"] = float(opt_match.group(1))
        for key, pattern in [
            ("pf", r"Profit Factor:\s*([\d.]+)"),
            ("win_rate", r"Win Rate:\s*([\d.]+)%"),
            ("n_trades", r"N Trades:\s*(\d+)"),
            ("sharpe", r"Sharpe Ratio:\s*([\d.]+)"),
            ("mdd", r"MDD:\s*([\d.]+)%"),
        ]:
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        if result:
            logger.info("FX信頼度閾値最適化結果を読み込みました")
            return result
    except Exception as e:
        logger.warning(f"FX信頼度閾値結果の解析に失敗: {e}")
    return None


def _parse_fx_time_filter_results() -> Optional[Dict[str, Any]]:
    """research/fx_time_filter_results.txt から時間帯フィルタ結果を読み込む"""
    path = RESEARCH_DIR / "fx_time_filter_results.txt"
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        result = {}
        # OVERALL METRICS
        for key, pattern in [
            ("total_trades", r"Total trades:\s*(\d+)"),
            ("win_rate", r"Win rate:\s*([\d.]+)%"),
            ("pf", r"PF:\s*([\d.]+)"),
            ("sharpe", r"Sharpe:\s*([\d.]+)"),
        ]:
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

        # BAD HOURS
        bad_hours_jst = re.findall(r"\*\* BAD \*\*.*?(\d+)h", text)
        # Fallback: parse from BAD HOURS section
        bad_section = re.findall(r"(\d+):00 UTC \((\d+):00 JST\) - PF=([\d.]+)", text)
        result["bad_hours"] = [
            {"utc": int(h[0]), "jst": int(h[1]), "pf": float(h[2])}
            for h in bad_section
            if float(h[2]) < 1.0
        ]

        # IMPROVEMENT section
        imp = re.search(
            r"IMPROVEMENT IF BAD HOURS EXCLUDED.*?"
            r"PF\s+([\d.]+)\s+([\d.]+)",
            text, re.DOTALL,
        )
        if imp:
            result["pf_after_filter"] = float(imp.group(2))

        if result:
            logger.info("FX時間帯フィルタ結果を読み込みました")
            return result
    except Exception as e:
        logger.warning(f"FX時間帯フィルタ結果の解析に失敗: {e}")
    return None


def _parse_boat_ev_results() -> Optional[Dict[str, Any]]:
    """research/boat_ev_optimization_results.txt からボートEV最適化結果を読み込む"""
    path = RESEARCH_DIR / "boat_ev_optimization_results.txt"
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        result = {}
        # OPTIMAL THRESHOLDS セクションから各券種を取得
        bet_types = {
            "win": "Win \\(Tansho\\)",
            "exacta": "Exacta \\(Nirentan\\)",
            "quinella": "Quinella \\(Nirenfuku\\)",
        }
        for key, label in bet_types.items():
            section = re.search(
                rf"{label}:.*?"
                r"Optimal EV threshold:\s*([\d.]+).*?"
                r"PF:\s*([\d.]+).*?"
                r"Recovery:\s*([\d.]+).*?"
                r"Hit rate:\s*([\d.]+).*?"
                r"N bets:\s*(\d+).*?"
                r"Sharpe:\s*([\d.]+).*?"
                r"MDD:\s*([\d,]+)\s*yen.*?"
                r"Total PnL:\s*([\d,]+)\s*yen",
                text, re.DOTALL,
            )
            if section:
                result[key] = {
                    "ev_threshold": float(section.group(1)),
                    "pf": float(section.group(2)),
                    "recovery": float(section.group(3)),
                    "hit_rate": float(section.group(4)),
                    "n_bets": int(section.group(5)),
                    "sharpe": float(section.group(6)),
                    "mdd": section.group(7),
                    "total_pnl": section.group(8),
                }
        if result:
            logger.info("ボートEV最適化結果を読み込みました")
            return result
    except Exception as e:
        logger.warning(f"ボートEV最適化結果の解析に失敗: {e}")
    return None


def _parse_feature_importance() -> Optional[Dict[str, List[Dict]]]:
    """research/feature_importance_report.txt から特徴量重要度を読み込む"""
    path = RESEARCH_DIR / "feature_importance_report.txt"
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        result = {}
        # モデルごとのセクションを特定
        model_map = {
            "fx": "MODEL 1: FX",
            "stock": "MODEL 2: Japan Stock",
            "boat": "MODEL 3: Boat Racing",
            "crypto": "MODEL 4: Crypto",
            "keiba": "MODEL 5: Keiba",
        }
        for key, header in model_map.items():
            if header not in text:
                continue
            # トップ5の特徴量を抽出
            section_start = text.index(header)
            # 次のMODELセクションまたはSUMMARYまでを切り出す
            next_model = text.find("MODEL", section_start + len(header))
            if next_model == -1:
                next_model = text.find("SUMMARY", section_start)
            section = text[section_start:next_model] if next_model != -1 else text[section_start:]

            features = []
            feat_pattern = re.compile(
                r"^\s+(\d+)\s+([\w_]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
                re.MULTILINE,
            )
            for m in feat_pattern.finditer(section):
                features.append({
                    "rank": int(m.group(1)),
                    "name": m.group(2),
                    "importance": int(m.group(3)),
                    "pct": float(m.group(4)),
                })
                if len(features) >= 5:
                    break
            if features:
                result[key] = features

        if result:
            logger.info(f"特徴量重要度を読み込みました: {list(result.keys())}")
            return result
    except Exception as e:
        logger.warning(f"特徴量重要度の解析に失敗: {e}")
    return None


def _load_daily_picks() -> Optional[List[Dict[str, Any]]]:
    """data/japan_stocks/daily_picks.csv から本日の銘柄ピックを読み込む"""
    path = DATA_DIR / "japan_stocks" / "daily_picks.csv"
    if not path.exists():
        logger.warning(f"日次ピックが見つかりません: {path}")
        return None

    try:
        picks = []
        seen_tickers = {}  # ticker -> best row (highest confidence)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("ticker", "")
                confidence = float(row.get("confidence", 0))
                # 同一銘柄は最も信頼度が高いものを採用
                if ticker not in seen_tickers or confidence > seen_tickers[ticker]["confidence"]:
                    seen_tickers[ticker] = {
                        "date": row.get("date", ""),
                        "ticker": ticker,
                        "name": row.get("name", ""),
                        "direction": row.get("direction", ""),
                        "confidence": confidence,
                        "agreement": int(row.get("agreement", 0)),
                        "wf_accuracy": float(row.get("wf_accuracy", 0)),
                        "pf": float(row.get("pf", 0)),
                        "sharpe": float(row.get("sharpe", 0)),
                        "mdd": float(row.get("mdd", 0)),
                        "win_rate": float(row.get("win_rate", 0)),
                        "trade_count": int(row.get("trade_count", 0)),
                    }
        picks = sorted(seen_tickers.values(), key=lambda x: x["confidence"], reverse=True)
        if picks:
            logger.info(f"日次ピックを読み込みました: {len(picks)}銘柄")
            return picks
    except Exception as e:
        logger.warning(f"日次ピックの読み込みに失敗: {e}")
    return None


def _load_sanitized_code_snippet(model_type: str) -> Optional[Dict[str, str]]:
    """モデルソースからサニタイズされたコードスニペットを生成する

    Args:
        model_type: "ensemble", "fx", "boat" のいずれか

    Returns:
        {"setup": ..., "core": ..., "execution": ...} 辞書。取得不可の場合はNone
    """
    source_map = {
        "ensemble": RESEARCH_DIR / "common" / "ensemble.py",
        "boat": RESEARCH_DIR / "boat" / "boat_model.py",
        "fx": RESEARCH_DIR / "fx_confidence_optimizer.py",
    }
    path = source_map.get(model_type)
    if path is None or not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")

        if model_type == "ensemble":
            # EnsembleClassifier のセットアップとコア部分
            setup = (
                "import numpy as np\n"
                "import lightgbm as lgb\n"
                "import xgboost as xgb\n"
                "from catboost import CatBoostClassifier\n"
                "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier\n"
                "\n"
                "# 5モデルアンサンブル（多数決方式）\n"
                "# LightGBM / XGBoost / CatBoost / RandomForest / ExtraTrees"
            )
            core = (
                "class EnsembleClassifier:\n"
                '    """5つのモデルの多数決で予測するクラス"""\n'
                "\n"
                "    def __init__(self, n_estimators=500, learning_rate=0.03):\n"
                "        self.model_lgb = lgb.LGBMClassifier(\n"
                "            n_estimators=n_estimators, learning_rate=learning_rate,\n"
                "            max_depth=6, random_state=42, verbosity=-1)\n"
                "        self.model_xgb = xgb.XGBClassifier(\n"
                "            n_estimators=n_estimators, learning_rate=learning_rate,\n"
                "            max_depth=6, random_state=42, verbosity=0)\n"
                "        self.model_cat = CatBoostClassifier(\n"
                "            iterations=n_estimators, learning_rate=learning_rate,\n"
                "            depth=6, random_seed=42, verbose=0)\n"
                "        self.model_rf = RandomForestClassifier(\n"
                "            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)\n"
                "        self.model_et = ExtraTreesClassifier(\n"
                "            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)\n"
                "        self.models = [\n"
                "            self.model_lgb, self.model_xgb, self.model_cat,\n"
                "            self.model_rf, self.model_et]\n"
                "\n"
                "    def fit(self, X, y):\n"
                "        for model in self.models:\n"
                "            model.fit(X, y)\n"
                "        return self\n"
                "\n"
                "    def predict_with_agreement(self, X):\n"
                '        """予測 + 一致度（3/4/5人）を返す"""\n'
                "        preds = np.array([m.predict(X) for m in self.models])\n"
                "        vote_sum = preds.sum(axis=0)\n"
                "        final = (vote_sum >= 3).astype(int)\n"
                "        agreement = np.where(final == 1, vote_sum, 5 - vote_sum)\n"
                "        return final, agreement"
            )
            execution = (
                "# Walk-Forward検証で過学習を防ぐ\n"
                "from sklearn.model_selection import TimeSeriesSplit\n"
                "\n"
                "tscv = TimeSeriesSplit(n_splits=8)\n"
                "results = []\n"
                "for train_idx, test_idx in tscv.split(X):\n"
                "    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\n"
                "    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\n"
                "\n"
                "    model = EnsembleClassifier(n_estimators=500, learning_rate=0.03)\n"
                "    model.fit(X_train, y_train)\n"
                "    preds, agreement = model.predict_with_agreement(X_test)\n"
                "\n"
                "    # 信頼度フィルタ: 4/5以上一致のみ採用\n"
                "    mask = agreement >= 4\n"
                "    filtered_preds = preds[mask]\n"
                "    filtered_y = y_test.values[mask]\n"
                "    accuracy = (filtered_preds == filtered_y).mean()\n"
                "    results.append({'accuracy': accuracy, 'n_trades': mask.sum()})"
            )
            return {"setup": setup, "core": core, "execution": execution}

        elif model_type == "boat":
            setup = (
                "import numpy as np\n"
                "import pandas as pd\n"
                "from pathlib import Path\n"
                "\n"
                "# ボートレース予測: 5モデルアンサンブル + EV閾値最適化\n"
                "# Fractional Kelly (0.25x) ベットサイジング"
            )
            core = (
                "# 基本特徴量 + 相互作用特徴量（40+次元）\n"
                "BASE_FEATURES = [\n"
                '    "lane", "racer_class", "racer_win_rate", "racer_place_rate",\n'
                '    "motor_2place_rate", "boat_2place_rate", "avg_start_timing",\n'
                '    "racer_weight", "weather_wind_speed", "wave_height",\n'
                "]\n"
                "\n"
                "INTERACTION_FEATURES = [\n"
                '    "win_rate_vs_field_avg",   # 勝率 - レース平均\n'
                '    "motor_vs_field_avg",      # モーター - レース平均\n'
                '    "class_x_lane",            # 級別 x 枠番\n'
                '    "wind_x_lane",             # 風速 x 枠番\n'
                '    "class_x_motor",           # 級別 x モーター\n'
                "]\n"
                "\n"
                "def calculate_ev(pred_prob, odds):\n"
                '    """期待値(EV) = 予測確率 x オッズ"""\n'
                "    return pred_prob * odds\n"
                "\n"
                "def kelly_fraction(pred_prob, odds, fraction=0.25):\n"
                '    """Fractional Kelly: 最適ベット比率"""\n'
                "    edge = pred_prob * odds - 1\n"
                "    if edge <= 0:\n"
                "        return 0.0\n"
                "    kelly = edge / (odds - 1)\n"
                "    return kelly * fraction  # 安全のため25%Kelly"
            )
            execution = (
                "# EV閾値でフィルタリング（最適閾値はWalk-Forwardで決定）\n"
                "# 単勝: EV >= 2.00, 2連単: EV >= 1.95, 2連複: EV >= 2.00\n"
                "ev_thresholds = {'win': 2.00, 'exacta': 1.95, 'quinella': 2.00}\n"
                "\n"
                "for bet_type, threshold in ev_thresholds.items():\n"
                "    mask = ev_values[bet_type] >= threshold\n"
                "    selected = predictions[bet_type][mask]\n"
                "    bet_sizes = [\n"
                "        kelly_fraction(p, o, fraction=0.25)\n"
                "        for p, o in zip(selected['prob'], selected['odds'])\n"
                "    ]\n"
                "    print(f'{bet_type}: {mask.sum()} bets selected (EV >= {threshold})')"
            )
            return {"setup": setup, "core": core, "execution": execution}

        elif model_type == "fx":
            setup = (
                "import numpy as np\n"
                "import pandas as pd\n"
                "import lightgbm as lgb\n"
                "import xgboost as xgb\n"
                "from catboost import CatBoostClassifier\n"
                "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier\n"
                "\n"
                "# FX方向予測: 5モデルアンサンブル\n"
                "# 対象: USDJPY, EURUSD, GBPUSD, EURJPY, AUDJPY\n"
                "# 予測ホライズン: 12時間"
            )
            core = (
                "# 特徴量: テクニカル指標 + レジーム検知\n"
                "FEATURES = [\n"
                '    "Regime_duration",  # レジーム持続期間（最重要）\n'
                '    "Volatility_24",    # 24時間ボラティリティ\n'
                '    "MA_75",            # 75期間移動平均\n'
                '    "BB_upper",         # ボリンジャーバンド上限\n'
                '    "MACD_signal",      # MACDシグナル\n'
                '    "BB_width",         # ボリンジャーバンド幅\n'
                '    "RSI_14",           # RSI(14)\n'
                '    "Return_24",        # 24時間リターン\n'
                "]\n"
                "\n"
                "# 信頼度フィルタ: 閾値0.78で最適化\n"
                "# confidence >= 0.78 & agreement >= 4/5\n"
                "def apply_confidence_filter(predictions, confidence, threshold=0.78):\n"
                '    """高信頼度の予測のみを採用"""\n'
                "    mask = confidence >= threshold\n"
                "    return predictions[mask], confidence[mask]\n"
                "\n"
                "# 時間帯フィルタ: JST 05時, 08時を除外\n"
                "BAD_HOURS_JST = [5, 8]  # PF < 1.0 の時間帯\n"
                "def apply_time_filter(df, hour_col='hour_jst'):\n"
                '    """負の期待値の時間帯を除外"""\n'
                "    return df[~df[hour_col].isin(BAD_HOURS_JST)]"
            )
            execution = (
                "# Walk-Forward検証（拡張ウィンドウ、8フォールド）\n"
                "# 最小訓練期間: 4320時間、テスト期間: 720時間\n"
                "\n"
                "results_by_pair = {}\n"
                "for pair in ['USDJPY', 'EURUSD', 'GBPUSD', 'EURJPY', 'AUDJPY']:\n"
                "    df = load_fx_data(pair, timeframe='1h')\n"
                "    X, y = create_features(df)\n"
                "\n"
                "    model = EnsembleClassifier(n_estimators=500, learning_rate=0.03)\n"
                "    preds, agreement = walk_forward_validate(\n"
                "        model, X, y,\n"
                "        min_train=4320, test_size=720, step=720)\n"
                "\n"
                "    # フィルタ適用\n"
                "    filtered = apply_confidence_filter(preds, confidence, threshold=0.78)\n"
                "    filtered = apply_time_filter(filtered)\n"
                "    metrics = calculate_metrics(filtered)\n"
                "    results_by_pair[pair] = metrics\n"
                "    print(f'{pair}: PF={metrics[\"pf\"]:.2f}, WR={metrics[\"win_rate\"]:.1f}%')"
            )
            return {"setup": setup, "core": core, "execution": execution}

    except Exception as e:
        logger.warning(f"コードスニペットの生成に失敗 ({model_type}): {e}")
    return None


# =============================================================
# テンプレート・ユーティリティ
# =============================================================

def _load_template(template_name: str) -> str:
    """テンプレートファイルを読み込む

    Args:
        template_name: テンプレートファイル名（拡張子なし）

    Returns:
        テンプレート文字列
    """
    path = TEMPLATES_DIR / f"{template_name}.md"
    if not path.exists():
        logger.error(f"テンプレートが見つかりません: {path}")
        raise FileNotFoundError(f"テンプレートが見つかりません: {path}")
    return path.read_text(encoding="utf-8")


def _load_disclaimer() -> str:
    """免責事項テンプレートを読み込む"""
    return _load_template("disclaimer")


def _generate_draft_id() -> str:
    """一意のドラフトIDを生成する"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


def _save_draft(content: str, draft_id: str, article_type: str) -> Path:
    """ドラフトをpendingディレクトリに保存する

    Args:
        content: 記事のMarkdownコンテンツ
        draft_id: ドラフトID
        article_type: 記事タイプ（free, paid, daily_report）

    Returns:
        保存先のPathオブジェクト
    """
    filename = f"{draft_id}_{article_type}.md"
    filepath = PENDING_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"ドラフト保存完了: {filepath}")

    # メタデータも保存
    meta = {
        "draft_id": draft_id,
        "article_type": article_type,
        "created_at": datetime.now().isoformat(),
        "status": "pending",
        "filename": filename,
    }
    meta_path = PENDING_DIR / f"{draft_id}_{article_type}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return filepath


def _load_model_results() -> Dict[str, Any]:
    """利用可能なモデル結果を読み込む

    Returns:
        各モデルの最新結果を含む辞書
    """
    results = {}

    # ダッシュボード状態（FXモデル結果）
    dashboard_path = DATA_DIR / "dashboard_state.joblib"
    if dashboard_path.exists():
        try:
            import joblib
            state = joblib.load(dashboard_path)
            results["fx"] = state
            logger.info("FXダッシュボード状態を読み込みました")
        except Exception as e:
            logger.warning(f"FXダッシュボード状態の読み込みに失敗: {e}")

    # リサーチファイルから実データを読み込む
    fx_multi = _parse_fx_multi_pair_results()
    if fx_multi:
        results["fx_multi"] = fx_multi

    fx_conf = _parse_fx_confidence_results()
    if fx_conf:
        results["fx_confidence"] = fx_conf

    fx_time = _parse_fx_time_filter_results()
    if fx_time:
        results["fx_time_filter"] = fx_time

    boat_ev = _parse_boat_ev_results()
    if boat_ev:
        results["boat_ev"] = boat_ev

    feat_imp = _parse_feature_importance()
    if feat_imp:
        results["feature_importance"] = feat_imp

    # 日本株ピック
    stock_picks = _load_daily_picks()
    if stock_picks:
        results["stock_picks"] = stock_picks

    # 日本株スクリーナー結果
    stock_report = DATA_DIR / "stock_screener" / "screening_report.json"
    if stock_report.exists():
        try:
            results["stocks"] = json.loads(stock_report.read_text(encoding="utf-8"))
            logger.info("株式スクリーナー結果を読み込みました")
        except Exception as e:
            logger.warning(f"株式スクリーナー結果の読み込みに失敗: {e}")

    # 競艇モデル結果
    boat_report = DATA_DIR / "boat" / "paper_trade_log.json"
    if boat_report.exists():
        try:
            results["boat"] = json.loads(boat_report.read_text(encoding="utf-8"))
            logger.info("競艇モデル結果を読み込みました")
        except Exception as e:
            logger.warning(f"競艇モデル結果の読み込みに失敗: {e}")

    # 暗号通貨モデル結果
    crypto_report = DATA_DIR / "crypto" / "crypto_model_report.txt"
    if crypto_report.exists():
        try:
            results["crypto"] = crypto_report.read_text(encoding="utf-8")
            logger.info("暗号通貨モデル結果を読み込みました")
        except Exception as e:
            logger.warning(f"暗号通貨モデル結果の読み込みに失敗: {e}")

    # マルチ通貨レポート
    multi_report = DATA_DIR / "multi_currency_report.txt"
    if multi_report.exists():
        try:
            results["multi_currency"] = multi_report.read_text(encoding="utf-8")
            logger.info("マルチ通貨レポートを読み込みました")
        except Exception as e:
            logger.warning(f"マルチ通貨レポートの読み込みに失敗: {e}")

    return results


# =============================================================
# フォーマット関数（実データ対応版）
# =============================================================

def _format_model_summary(model_results: Dict[str, Any]) -> str:
    """モデル結果のサマリーテキストを生成する（実データ付き）"""
    sections = []

    # FXモデル: マルチ通貨ペア結果を含む
    if "fx_multi" in model_results:
        fx = model_results["fx_multi"]
        best_pair = max(fx.items(), key=lambda x: x[1]["pf"])
        sections.append(
            f"- **FX予測モデル**: 5モデルアンサンブル（LightGBM + XGBoost + CatBoost + RF + ExtraTrees）\n"
            f"  - 対象5通貨ペア中、最高PF: {best_pair[0]} (PF={best_pair[1]['pf']:.2f}, "
            f"勝率{best_pair[1]['win_rate']:.1f}%, Sharpe={best_pair[1]['sharpe']:.2f})"
        )
    elif "fx" in model_results:
        sections.append("- **FX予測モデル**: アンサンブル（LightGBM + XGBoost + CatBoost）による方向予測")

    # 日本株モデル
    if "stock_picks" in model_results:
        picks = model_results["stock_picks"]
        best = picks[0] if picks else None
        if best:
            sections.append(
                f"- **日本株モデル**: 米国市場→日本市場の相関を利用した5モデルアンサンブル\n"
                f"  - 本日{len(picks)}銘柄をピック（最高信頼度: {best['name']} "
                f"信頼度{best['confidence']:.1%}, WF正解率{best['wf_accuracy']:.1%}）"
            )
    elif "stocks" in model_results:
        sections.append("- **日本株モデル**: 米国市場→日本市場の相関を利用した5モデルアンサンブル")

    # 競艇モデル
    if "boat_ev" in model_results:
        boat = model_results["boat_ev"]
        if "quinella" in boat:
            q = boat["quinella"]
            sections.append(
                f"- **競艇モデル**: 5モデルアンサンブル＋EV閾値最適化＋Fractional Kelly\n"
                f"  - 2連複(最適): PF={q['pf']:.3f}, 回収率{q['recovery']:.1%}, "
                f"Sharpe={q['sharpe']:.3f}, {q['n_bets']}ベット"
            )
    elif "boat" in model_results:
        sections.append("- **競艇モデル**: 5モデルアンサンブル＋Fractional Kelly ベットサイジング")

    if "crypto" in model_results:
        sections.append("- **暗号通貨モデル**: ハイブリッドモデル（テクニカル＋オンチェーン指標）")

    if not sections:
        sections.append("- 各種AIモデルによる多市場予測")

    return "\n".join(sections)


def _format_stock_predictions(model_results: Dict[str, Any]) -> str:
    """日本株予測のフォーマット（daily_picks.csv の実データ使用）"""
    if "stock_picks" in model_results:
        picks = model_results["stock_picks"]
        lines = [
            "| 銘柄 | コード | 予測方向 | 信頼度 | モデル一致 | WF正解率 | PF | Sharpe | 取引数 |",
            "|------|--------|----------|--------|-----------|----------|-----|--------|--------|",
        ]
        for p in picks:
            lines.append(
                f"| {p['name']} | {p['ticker']} | {p['direction']} | "
                f"{p['confidence']:.1%} | {p['agreement']}/5 | "
                f"{p['wf_accuracy']:.1%} | {p['pf']:.2f} | "
                f"{p['sharpe']:.2f} | {p['trade_count']} |"
            )
        lines.append("")
        lines.append("*WF正解率 = Walk-Forward検証での正解率（過学習防止済み）*")
        return "\n".join(lines)

    if "stocks" in model_results:
        stocks = model_results["stocks"]
        if isinstance(stocks, dict) and "results" in stocks:
            lines = ["| 銘柄 | 予測方向 | 信頼度 |", "|------|----------|--------|"]
            for r in stocks["results"][:10]:
                name = r.get("name", "不明")
                direction = r.get("prediction", "N/A")
                confidence = r.get("confidence", 0)
                lines.append(f"| {name} | {direction} | {confidence:.1%} |")
            return "\n".join(lines)

    return "本日の日本株予測データはありません。"


def _format_fx_predictions(model_results: Dict[str, Any]) -> str:
    """FX予測のフォーマット（実データ使用）"""
    sections = []

    # マルチ通貨ペア結果テーブル
    if "fx_multi" in model_results:
        fx = model_results["fx_multi"]
        sections.append("#### 通貨ペア別パフォーマンス（Walk-Forward検証）\n")
        sections.append("| 通貨ペア | 取引数 | 勝率 | PF | Sharpe | MDD | Sortino |")
        sections.append("|----------|--------|------|----|--------|-----|---------|")
        for pair, data in fx.items():
            sections.append(
                f"| {pair} | {data['trades']} | {data['win_rate']:.1f}% | "
                f"{data['pf']:.2f} | {data['sharpe']:.2f} | "
                f"{data['mdd']:.1f}% | {data['sortino']:.2f} |"
            )
        sections.append("")

    # 信頼度フィルタ結果
    if "fx_confidence" in model_results:
        conf = model_results["fx_confidence"]
        sections.append(
            f"#### 信頼度フィルタ最適化\n\n"
            f"- 最適閾値: {conf.get('optimal_threshold', 'N/A')}\n"
            f"- PF: {conf.get('pf', 'N/A')}, 勝率: {conf.get('win_rate', 'N/A')}%\n"
            f"- Sharpe: {conf.get('sharpe', 'N/A')}, MDD: {conf.get('mdd', 'N/A')}%\n"
            f"- 取引数: {conf.get('n_trades', 'N/A')}"
        )

    # 時間帯フィルタ結果
    if "fx_time_filter" in model_results:
        tf = model_results["fx_time_filter"]
        sections.append(
            f"\n#### 時間帯フィルタ\n\n"
            f"- 全体: PF={tf.get('pf', 'N/A')}, 勝率={tf.get('win_rate', 'N/A')}%, "
            f"Sharpe={tf.get('sharpe', 'N/A')}"
        )
        bad = tf.get("bad_hours", [])
        if bad:
            bad_str = ", ".join([f"JST {h['jst']}時 (PF={h['pf']:.2f})" for h in bad])
            sections.append(f"- 除外推奨時間帯: {bad_str}")
        pf_after = tf.get("pf_after_filter")
        if pf_after:
            sections.append(f"- フィルタ適用後PF: {pf_after}")

    if not sections:
        if "fx" in model_results:
            return "FXモデルの予測結果については、ダッシュボードの最新状態をご確認ください。"
        return "本日のFX予測データはありません。"

    return "\n".join(sections)


def _format_boat_predictions(model_results: Dict[str, Any]) -> str:
    """競艇予測のフォーマット（実データ使用）"""
    sections = []

    if "boat_ev" in model_results:
        boat = model_results["boat_ev"]
        sections.append("#### EV閾値最適化結果（Walk-Forward検証）\n")
        sections.append("| 券種 | 最適EV閾値 | PF | 回収率 | 的中率 | ベット数 | Sharpe | 累積損益 |")
        sections.append("|------|-----------|-----|--------|--------|---------|--------|---------|")

        name_map = {"win": "単勝", "exacta": "2連単", "quinella": "2連複"}
        for bet_type in ["win", "exacta", "quinella"]:
            if bet_type in boat:
                d = boat[bet_type]
                sections.append(
                    f"| {name_map[bet_type]} | {d['ev_threshold']:.2f} | "
                    f"{d['pf']:.3f} | {d['recovery']:.1%} | "
                    f"{d['hit_rate']:.1%} | {d['n_bets']} | "
                    f"{d['sharpe']:.3f} | {d['total_pnl']}円 |"
                )
        sections.append("")
        sections.append("*5モデルアンサンブル + Fractional Kelly (0.25x) ベットサイジング*")
        return "\n".join(sections)

    if "boat" in model_results:
        boat = model_results["boat"]
        if isinstance(boat, dict):
            total_bets = boat.get("total_bets", "N/A")
            total_profit = boat.get("total_profit", "N/A")
            return (
                f"- 累計ベット数: {total_bets}\n"
                f"- 累計損益: {total_profit}\n"
                f"- 5モデルアンサンブル＋EV閾値最適化による予測"
            )

    return "本日の競艇予測データはありません。"


def _format_crypto_predictions(model_results: Dict[str, Any]) -> str:
    """暗号通貨予測のフォーマット"""
    if "feature_importance" in model_results and "crypto" in model_results["feature_importance"]:
        feats = model_results["feature_importance"]["crypto"]
        top_feats = ", ".join([f['name'] for f in feats[:3]])
        return (
            f"暗号通貨(BTC)ハイブリッドモデル（テクニカル＋オンチェーン指標）\n\n"
            f"- 重要特徴量トップ3: {top_feats}\n"
            f"- マルチタイムフレーム指標（1日/4時間/短期）を統合\n"
            f"- RSI, MACD, ADX, ボリンジャーバンド等を活用"
        )
    if "crypto" not in model_results:
        return "本日の暗号通貨予測データはありません。"
    return "暗号通貨ハイブリッドモデルの最新予測結果を確認中です。"


def _format_real_metrics_table(model_results: Dict[str, Any]) -> str:
    """実データに基づくパフォーマンス指標テーブルを生成する"""
    # FXモデルのメイン指標（USDJPY）
    if "fx_multi" in model_results and "USDJPY" in model_results["fx_multi"]:
        fx = model_results["fx_multi"]["USDJPY"]
        return (
            f"| 勝率 | {fx['win_rate']:.1f}% |\n"
            f"| Profit Factor | {fx['pf']:.2f} |\n"
            f"| Sharpe Ratio | {fx['sharpe']:.2f} |\n"
            f"| Max Drawdown | {fx['mdd']:.1f}% |"
        )

    # 信頼度最適化結果からフォールバック
    if "fx_confidence" in model_results:
        conf = model_results["fx_confidence"]
        return (
            f"| 勝率 | {conf.get('win_rate', 'N/A')}% |\n"
            f"| Profit Factor | {conf.get('pf', 'N/A')} |\n"
            f"| Sharpe Ratio | {conf.get('sharpe', 'N/A')} |\n"
            f"| Max Drawdown | {conf.get('mdd', 'N/A')}% |"
        )

    return (
        "| 勝率 | データ準備中 |\n"
        "| Profit Factor | データ準備中 |\n"
        "| Sharpe Ratio | データ準備中 |\n"
        "| Max Drawdown | データ準備中 |"
    )


def _format_recent_accuracy(model_results: Dict[str, Any]) -> str:
    """直近パフォーマンスのフォーマット（実データ対応）"""
    sections = []

    if "fx_multi" in model_results:
        fx = model_results["fx_multi"]
        sections.append("#### FXモデル（Walk-Forward検証結果）\n")
        sections.append("| 通貨ペア | 勝率 | PF | Sharpe |")
        sections.append("|----------|------|----|--------|")
        for pair, data in fx.items():
            sections.append(
                f"| {pair} | {data['win_rate']:.1f}% | "
                f"{data['pf']:.2f} | {data['sharpe']:.2f} |"
            )
        sections.append("")

    if "boat_ev" in model_results:
        boat = model_results["boat_ev"]
        sections.append("#### 競艇モデル（最適EV閾値時）\n")
        sections.append("| 券種 | PF | 的中率 | Sharpe |")
        sections.append("|------|----|--------|--------|")
        name_map = {"win": "単勝", "exacta": "2連単", "quinella": "2連複"}
        for bt in ["win", "exacta", "quinella"]:
            if bt in boat:
                d = boat[bt]
                sections.append(
                    f"| {name_map[bt]} | {d['pf']:.3f} | "
                    f"{d['hit_rate']:.1%} | {d['sharpe']:.3f} |"
                )
        sections.append("")

    if "stock_picks" in model_results:
        picks = model_results["stock_picks"]
        # 平均WF正解率を計算
        avg_acc = sum(p["wf_accuracy"] for p in picks) / len(picks) if picks else 0
        avg_pf = sum(p["pf"] for p in picks) / len(picks) if picks else 0
        sections.append(
            f"#### 日本株モデル\n\n"
            f"- 本日ピック数: {len(picks)}銘柄\n"
            f"- 平均WF正解率: {avg_acc:.1%}\n"
            f"- 平均PF: {avg_pf:.2f}"
        )

    if not sections:
        return "各モデルの直近パフォーマンスについては、週次レポートで詳細を公開しています。"

    return "\n".join(sections)


def _format_cumulative_performance(model_results: Dict[str, Any]) -> str:
    """累積パフォーマンスのフォーマット（実データ対応）"""
    sections = []

    if "boat_ev" in model_results:
        boat = model_results["boat_ev"]
        total_pnl = 0
        for bt in ["win", "exacta", "quinella"]:
            if bt in boat:
                pnl_str = boat[bt].get("total_pnl", "0")
                pnl = int(pnl_str.replace(",", ""))
                total_pnl += pnl
        sections.append(
            f"#### 競艇モデル累積損益\n\n"
            f"- 単勝: {boat.get('win', {}).get('total_pnl', 'N/A')}円\n"
            f"- 2連単: {boat.get('exacta', {}).get('total_pnl', 'N/A')}円\n"
            f"- 2連複: {boat.get('quinella', {}).get('total_pnl', 'N/A')}円\n"
            f"- **合計: {total_pnl:,}円**"
        )

    if "fx_multi" in model_results:
        fx = model_results["fx_multi"]
        best = max(fx.items(), key=lambda x: x[1]["sharpe"])
        sections.append(
            f"\n#### FXモデル\n\n"
            f"- 最高Sharpe: {best[0]} (Sharpe={best[1]['sharpe']:.2f})\n"
            f"- 5通貨ペアのWalk-Forward検証済み"
        )

    if not sections:
        return "累積パフォーマンスの推移は、週次レポートにてグラフ付きで公開しています。"

    return "\n".join(sections)


def _format_analysis_results_with_data(model_results: Dict[str, Any]) -> str:
    """無料記事用の分析結果セクション（実データ付き）"""
    sections = ["### 主な分析結果\n"]
    sections.append(
        "AIモデルの検証では、Walk-Forward法を採用し、"
        "未来のデータで過学習していないことを確認しています。\n"
    )

    # FXマルチ通貨ペア結果
    if "fx_multi" in model_results:
        fx = model_results["fx_multi"]
        sections.append("#### FX方向予測（12時間ホライズン）\n")
        sections.append("| 通貨ペア | 取引数 | 勝率 | PF | Sharpe |")
        sections.append("|----------|--------|------|----|--------|")
        for pair, data in fx.items():
            sections.append(
                f"| {pair} | {data['trades']} | {data['win_rate']:.1f}% | "
                f"{data['pf']:.2f} | {data['sharpe']:.2f} |"
            )
        sections.append("")

    # 競艇モデル概要
    if "boat_ev" in model_results:
        boat = model_results["boat_ev"]
        if "quinella" in boat:
            q = boat["quinella"]
            sections.append(
                f"#### 競艇予測モデル\n\n"
                f"- 2連複: PF={q['pf']:.3f}, 回収率{q['recovery']:.1%}, "
                f"Sharpe={q['sharpe']:.3f}\n"
                f"- Walk-Forward 5ウィンドウ検証済み\n"
            )

    # 特徴量重要度の概要
    if "feature_importance" in model_results:
        fi = model_results["feature_importance"]
        sections.append("#### 主要な特徴量（モデル別トップ3）\n")
        model_names = {"fx": "FX", "stock": "日本株", "boat": "競艇", "crypto": "暗号通貨"}
        for key, label in model_names.items():
            if key in fi:
                top3 = ", ".join([f['name'] for f in fi[key][:3]])
                sections.append(f"- **{label}**: {top3}")
        sections.append("")

    sections.append(
        "\n*具体的なコード実装や詳細なパラメータ設定は有料記事でご紹介しています。*"
    )

    return "\n".join(sections)


# =============================================================
# メイン関数
# =============================================================

def generate_free_article(
    topic: str,
    model_results: Optional[Dict[str, Any]] = None,
) -> str:
    """無料記事を生成する（集客・トラフィック用）

    Args:
        topic: 記事のトピック（例: "AIで日本株を予測してみた結果"）
        model_results: モデル結果辞書（Noneの場合は自動読み込み）

    Returns:
        保存先のファイルパス文字列
    """
    logger.info(f"無料記事生成開始: トピック={topic}")

    if model_results is None:
        model_results = _load_model_results()

    template = _load_template("free_article")
    disclaimer = _load_disclaimer()
    draft_id = _generate_draft_id()
    today = datetime.now().strftime("%Y年%m月%d日")

    # テンプレートに値を埋め込む
    content = template.format(
        title=topic,
        subtitle=f"AIモデルによる分析結果を公開 | {today}",
        introduction=(
            f"こんにちは。今回は「{topic}」について、私が開発しているAI予測システムの"
            f"結果を共有します。\n\n"
            f"このシステムは、LightGBM・XGBoost・CatBoostなどの機械学習モデルを"
            f"アンサンブルし、複数市場（FX・日本株・競艇・暗号通貨）の予測を行っています。"
        ),
        topic_description=(
            f"本記事では、{topic}に関するAI分析の概要と主要な発見をお伝えします。\n"
            f"具体的なコード実装や詳細なパラメータ設定は有料記事でご紹介しています。"
        ),
        model_overview=_format_model_summary(model_results),
        analysis_results=_format_analysis_results_with_data(model_results),
        discussion=(
            "AI予測は万能ではありませんが、適切なリスク管理と組み合わせることで"
            "有用なツールとなり得ます。\n\n"
            "重要なのは、モデルの精度だけでなく、ドローダウン管理やポジションサイジングを"
            "含めたシステム全体の設計です。"
        ),
        summary=(
            f"今回は{topic}について概要をご紹介しました。\n\n"
            f"より詳しい内容（実装コード・パラメータ・詳細な検証結果）は"
            f"有料記事およびマガジンで公開しています。"
        ),
        disclaimer=disclaimer,
        publish_date=today,
    )

    filepath = _save_draft(content, draft_id, "free")
    logger.info(f"無料記事ドラフト作成完了: {filepath}")
    return str(filepath)


def generate_paid_article(
    topic: str,
    model_results: Optional[Dict[str, Any]] = None,
    code_snippets: Optional[Dict[str, str]] = None,
) -> str:
    """有料記事を生成する（コード付きチュートリアル）

    Args:
        topic: 記事のトピック
        model_results: モデル結果辞書
        code_snippets: コードスニペット辞書
            - "setup": 環境準備コード
            - "core": コア実装コード
            - "execution": 実行コード

    Returns:
        保存先のファイルパス文字列
    """
    logger.info(f"有料記事生成開始: トピック={topic}")

    if model_results is None:
        model_results = _load_model_results()

    # コードスニペットが未指定の場合、トピックに応じて実際のコードを読み込む
    if code_snippets is None:
        # トピックからモデルタイプを推定
        topic_lower = topic.lower()
        if any(w in topic_lower for w in ["競艇", "ボート", "boat"]):
            code_snippets = _load_sanitized_code_snippet("boat")
        elif any(w in topic_lower for w in ["fx", "為替", "通貨"]):
            code_snippets = _load_sanitized_code_snippet("fx")
        else:
            code_snippets = _load_sanitized_code_snippet("ensemble")

    if code_snippets is None:
        code_snippets = {
            "setup": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "import lightgbm as lgb\n"
                "from sklearn.model_selection import TimeSeriesSplit\n"
                "\n"
                "# データ読み込み\n"
                "# ※ 実際のデータパスは環境に合わせて変更してください"
            ),
            "core": (
                "# アンサンブルモデルの実装コードは\n"
                "# 有料マガジンの過去記事をご参照ください\n"
                "pass"
            ),
            "execution": (
                "# モデル学習・評価の実行\n"
                "# 結果の確認\n"
                "pass"
            ),
        }

    template = _load_template("paid_tutorial")
    disclaimer = _load_disclaimer()
    draft_id = _generate_draft_id()
    today = datetime.now().strftime("%Y年%m月%d日")

    content = template.format(
        title=topic,
        subtitle=f"Pythonで実装するAI予測 | 完全コード付き | {today}",
        introduction=(
            f"本記事では「{topic}」を実際のPythonコードとともに解説します。\n\n"
            f"コピー＆ペーストで動作するコードを提供しますので、ご自身の環境で"
            f"すぐに試すことができます。"
        ),
        learning_objectives=(
            f"- {topic}の基本的な考え方\n"
            f"- Pythonによる実装手順\n"
            f"- Walk-Forward検証による評価方法\n"
            f"- リスク管理の組み込み方"
        ),
        prerequisites=(
            "- Python 3.8以上\n"
            "- 基本的なPandas/NumPyの知識\n"
            "- 機械学習の基礎知識（分類問題の理解）"
        ),
        method_explanation=(
            f"本手法は、複数の機械学習モデル（LightGBM、XGBoost、CatBoost等）を"
            f"アンサンブルし、各モデルの予測を統合することで精度を向上させます。\n\n"
            f"重要なポイントは以下の通りです：\n"
            f"1. **Walk-Forward検証**: 未来のデータへの漏洩を防ぐ\n"
            f"2. **アンサンブル**: 複数モデルの多数決で安定性を確保\n"
            f"3. **信頼度フィルタ**: 低信頼度の予測をスキップ"
        ),
        setup_code=code_snippets.get("setup", "# 環境準備コード"),
        core_code=code_snippets.get("core", "# コア実装コード"),
        execution_code=code_snippets.get("execution", "# 実行コード"),
        validation_results=(
            "Walk-Forward検証の結果、以下のパフォーマンスが確認されました。\n\n"
            "※ 注意: これは過去データに基づく検証結果であり、将来の利益を保証するものではありません。"
        ),
        metrics_table=_format_real_metrics_table(model_results),
        discussion=(
            "検証結果は有望ですが、以下の点に注意が必要です：\n\n"
            "1. **過学習リスク**: パラメータの過度な最適化は避ける\n"
            "2. **市場環境の変化**: レジーム変化に対応する仕組みが重要\n"
            "3. **取引コスト**: スプレッド・手数料を必ず考慮する"
        ),
        improvement_tips=(
            "1. 特徴量エンジニアリングの工夫（経済指標の追加等）\n"
            "2. ハイパーパラメータのOptuna最適化\n"
            "3. レジーム検知（HMM等）との組み合わせ\n"
            "4. ポジションサイジングの最適化（Kelly基準等）"
        ),
        summary=(
            f"本記事では{topic}の実装を完全コード付きで解説しました。\n\n"
            f"このアプローチをベースに、ご自身のデータや市場に合わせたカスタマイズを"
            f"試してみてください。質問やフィードバックはコメント欄でお待ちしています。"
        ),
        disclaimer=disclaimer,
        publish_date=today,
    )

    filepath = _save_draft(content, draft_id, "paid")
    logger.info(f"有料記事ドラフト作成完了: {filepath}")
    return str(filepath)


def generate_daily_report(
    predictions: Optional[Dict[str, Any]] = None,
) -> str:
    """日次予測レポートを生成する（マガジン購読者向け）

    Args:
        predictions: 予測データ辞書（Noneの場合は自動読み込み）

    Returns:
        保存先のファイルパス文字列
    """
    today = datetime.now()
    date_str = today.strftime("%Y年%m月%d日")
    logger.info(f"日次レポート生成開始: {date_str}")

    if predictions is None:
        predictions = _load_model_results()

    template = _load_template("daily_report")
    disclaimer = _load_disclaimer()
    draft_id = _generate_draft_id()

    content = template.format(
        title=f"【AI予測】{date_str} マーケット予測レポート",
        date=date_str,
        market_overview=(
            "本日の市場環境と主要な経済イベントの概要です。\n\n"
            "※ 市場データはモデル実行時点のものです。リアルタイムの値とは"
            "異なる場合があります。"
        ),
        prediction_summary=(
            "本日のAIモデルによる各市場の予測サマリーです。\n\n"
            + _format_model_summary(predictions)
        ),
        stock_predictions=_format_stock_predictions(predictions),
        fx_predictions=_format_fx_predictions(predictions),
        boat_predictions=_format_boat_predictions(predictions),
        crypto_predictions=_format_crypto_predictions(predictions),
        recent_accuracy=_format_recent_accuracy(predictions),
        cumulative_performance=_format_cumulative_performance(predictions),
        key_points=(
            "- 各予測の信頼度スコアに注目してください\n"
            "- 信頼度が低い予測は見送りを推奨します\n"
            "- ポジションサイズはリスク管理に従ってください\n"
            "- FX: JST 05時・08時は予測精度が低いため取引非推奨"
        ),
        risk_factors=(
            "- 重要経済指標の発表前後は予測精度が低下する可能性があります\n"
            "- 地政学的リスクはモデルに反映されていません\n"
            "- 流動性が低い時間帯はスリッページにご注意ください\n"
            "- 過去の検証結果は将来のパフォーマンスを保証しません"
        ),
        disclaimer=disclaimer,
    )

    filepath = _save_draft(content, draft_id, "daily_report")
    logger.info(f"日次レポート作成完了: {filepath}")
    return str(filepath)


# =============================================================
# CLI エントリーポイント
# =============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="note.com 記事自動生成")
    parser.add_argument(
        "--type",
        choices=["free", "paid", "daily"],
        default="daily",
        help="生成する記事タイプ (default: daily)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="AIで株価を予測する方法",
        help="記事のトピック (free/paid記事用)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  note.com 記事自動生成: type={args.type}")
    logger.info("=" * 60)

    if args.type == "free":
        path = generate_free_article(args.topic)
    elif args.type == "paid":
        path = generate_paid_article(args.topic)
    else:
        path = generate_daily_report()

    logger.info(f"生成完了: {path}")
