# ===========================================
# news_sentiment_model.py
# 株プロジェクト - ニュース感情分析モデル
#
# 設計方針：
#   - 「ニュースが出た後の方向性予測」のみ対象
#   - 「ニュースが出る前の予測」は不可能
#   - 判断できない場合は必ずスキップする
#   - 3段階パイプライン（速度優先→精度優先→文脈理解）
#
# 注意：実装・コードのみ完成。
#       データ取得・学習はボートレース安定後。
# ===========================================

import re
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum


# =============================================================
# データクラス・定数
# =============================================================

class SentimentLabel(Enum):
    POSITIVE   = "positive"
    NEGATIVE   = "negative"
    NEUTRAL    = "neutral"
    SKIP       = "skip"  # 判断不可能 → スキップ


@dataclass
class NewsItem:
    """ニュース1件の表現。"""
    text:      str
    source:    str          # "twitter" / "newsapi" / "nikkei"
    timestamp: str          # ISO形式
    tickers:   List[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """感情分析結果。"""
    news:                   NewsItem
    sentiment:              SentimentLabel
    score:                  float           # -1.0〜+1.0
    confidence:             float           # 0〜1
    stage_used:             int             # 1=ルールベース / 2=FinBERT / 3=Claude
    affected_tickers:       List[str] = field(default_factory=list)
    skip_reason:            str = ""
    needs_context_analysis: bool = False


# ルールベース辞書（日本語）
POSITIVE_WORDS = {
    "好決算":    1.0, "上方修正":   0.8, "自社株買い": 0.8,
    "増配":      0.7, "新製品":     0.5, "提携":       0.3,
    "成長":      0.4, "最高益":     1.0, "黒字転換":   0.9,
    "買収":      0.4, "特許":       0.5,
}

NEGATIVE_WORDS = {
    "下方修正":  -1.0, "リコール":  -0.8, "赤字":      -0.9,
    "減配":      -0.7, "訴訟":      -0.6, "不正":      -0.9,
    "倒産":      -1.0, "業績悪化":  -0.8, "CEO辞任":   -0.7,
    "規制":      -0.5,
}

# 文脈分析が必要なキーワード（Claude APIに回す）
CONTEXT_REQUIRED_PATTERNS = [
    r"CEO.{0,10}辞任",
    r"買収.{0,15}提携",
    r"規制.{0,20}発表",
    r"後任",
    r"条件付き",
]


# =============================================================
# STEP1：ニュース収集
# =============================================================

class NewsCollector:
    """
    複数ソースからニュースを収集する。

    ソースの優先順位：
    X（Twitter） → 速報性最高
    NewsAPI      → 精度高め
    日経RSS      → 日本語強い
    """

    def collect_realtime(self, keywords: List[str]) -> List[NewsItem]:
        """
        リアルタイムニュースを収集する。

        監視キーワード：
        ・企業名・銘柄コード
        ・イベントキーワード（決算・リコール等）
        ・マクロキーワード（利上げ・円高等）

        Parameters:
            keywords: 監視するキーワードのリスト

        Returns:
            news_items: 収集したニュースのリスト
        """
        # 実装はAPIキー取得後
        raise NotImplementedError(
            "実装はAPIキー取得後。\n"
            "必要なAPIキー: TWITTER_BEARER_TOKEN / NEWSAPI_KEY"
        )

    def collect_historical(
        self,
        start_date: str,
        end_date:   str,
        tickers:    List[str] = None,
    ) -> List[NewsItem]:
        """
        過去データを一括取得する。
        ボートレース・FXが安定してから実装する。
        """
        raise NotImplementedError("実装はボートレース・FXが安定後")


# =============================================================
# STEP2：銘柄特定
# =============================================================

class StockMapper:
    """ニュースから影響を受ける銘柄を特定する。"""

    def __init__(self):
        # セクターマップ（代表例）
        self.sector_map = {
            "cybersecurity": ["CRWD", "PANW", "ZS", "OKTA"],
            "semiconductor":  ["6723.T", "6762.T", "8035.T"],
            "ev":             ["7203.T", "7201.T", "7261.T"],
            "bank_jp":        ["8306.T", "8316.T", "8411.T"],
            "trading_co":     ["8001.T", "8002.T", "8031.T"],
        }

        # 逆相関マップ（一方が上がれば他方が下がる関係）
        self.inverse_map = {
            "ai_disrupts_security": {
                "down": ["cybersecurity"],
                "up":   ["ai_infrastructure"],
            },
            "rate_hike": {
                "up":   ["bank_jp"],
                "down": ["growth_stock"],
            },
            "yen_strength": {
                "up":   ["import_heavy"],
                "down": ["export_heavy"],
            },
        }

    def map_news_to_stocks(self, news_text: str) -> dict:
        """
        ニュースから直接・間接・逆影響銘柄を特定する。
        NER（固有表現認識）+ 辞書のハイブリッド。

        Parameters:
            news_text: ニューステキスト

        Returns:
            {
                "direct":  ["7203.T"],     # 直接言及されている銘柄
                "sector":  ["7201.T"],     # セクター関連
                "inverse": ["8035.T"],     # 逆相関銘柄
            }
        """
        result = {"direct": [], "sector": [], "inverse": []}

        # 銘柄コードの直接検出（例: 7203.T や 7203）
        ticker_pattern = re.compile(r"\b(\d{4}(?:\.T)?)\b")
        direct_tickers = ticker_pattern.findall(news_text)
        result["direct"] = list(set(direct_tickers))

        # セクターキーワードによる間接検出
        for sector, tickers in self.sector_map.items():
            if sector.replace("_", "").lower() in news_text.lower():
                result["sector"].extend(tickers)

        # 逆相関マップの適用
        for event, mapping in self.inverse_map.items():
            event_key = event.replace("_", " ").lower()
            if event_key in news_text.lower():
                for direction, sectors in mapping.items():
                    for sec in sectors:
                        if sec in self.sector_map:
                            result["inverse"].extend(self.sector_map[sec])

        # 重複除去
        result["sector"]  = list(set(result["sector"]))
        result["inverse"] = list(set(result["inverse"]))

        return result


# =============================================================
# STEP3：感情分析パイプライン
# =============================================================

class SentimentAnalyzer:
    """3段階の感情分析パイプライン。"""

    def __init__(self):
        self._finbert_model = None   # 遅延ロード
        self._stock_mapper  = StockMapper()

    def analyze(self, news: NewsItem) -> SentimentResult:
        """
        3段階パイプラインで感情分析する。

        第1段階: ルールベース（数秒・速報性優先）
        第2段階: FinBERT（数十秒・英語強い）
        第3段階: Claude API（文脈理解・複雑ケースのみ）
        """
        # 文脈分析が必要か先にチェック
        needs_context = self._needs_context_analysis(news.text)

        # 第1段階: ルールベース
        stage1 = self.analyze_rule_based(news.text)

        if needs_context:
            # 複雑なケース → Claude APIへ
            return self.analyze_claude_api(news, stage1)

        if stage1.confidence >= 0.8:
            # 十分な確信度 → ルールベースで確定
            result = SentimentResult(
                news=news,
                sentiment=stage1.sentiment,
                score=stage1.score,
                confidence=stage1.confidence,
                stage_used=1,
                needs_context_analysis=False,
            )
        else:
            # 確信度が低い → FinBERT（第2段階）へ
            result = self.analyze_finbert(news, stage1)

        # 銘柄特定
        result.affected_tickers = (
            news.tickers or
            self._stock_mapper.map_news_to_stocks(news.text).get("direct", [])
        )

        return result

    def analyze_rule_based(self, text: str) -> "Stage1Result":
        """
        第1段階: ルールベース辞書（数秒・速報性優先）。

        複雑なケース（CEO辞任等）は
        needs_context_analysis=Trueフラグを立てる。
        """
        score = 0.0
        hit_count = 0

        for word, value in POSITIVE_WORDS.items():
            if word in text:
                score += value
                hit_count += 1

        for word, value in NEGATIVE_WORDS.items():
            if word in text:
                score += value
                hit_count += 1

        # スコアを-1〜+1に正規化
        if hit_count > 0:
            score = max(-1.0, min(1.0, score / hit_count))

        # 感情ラベルを決定
        if score >= 0.3:
            sentiment = SentimentLabel.POSITIVE
        elif score <= -0.3:
            sentiment = SentimentLabel.NEGATIVE
        elif abs(score) < 0.1 and hit_count == 0:
            sentiment = SentimentLabel.SKIP
        else:
            sentiment = SentimentLabel.NEUTRAL

        confidence = min(0.9, 0.5 + abs(score) * 0.4)

        return _Stage1Result(
            sentiment=sentiment,
            score=score,
            confidence=confidence,
        )

    def analyze_finbert(
        self,
        news:   NewsItem,
        stage1: "_Stage1Result" = None,
    ) -> SentimentResult:
        """
        第2段階: FinBERT（数十秒・英語強い）。

        Note: 実装はボートレース安定後（要GPU/transformers）
        """
        try:
            from transformers import pipeline
            if self._finbert_model is None:
                self._finbert_model = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    return_all_scores=True,
                )
            result = self._finbert_model(news.text[:512])[0]
            scores = {r["label"]: r["score"] for r in result}

            best = max(scores, key=scores.get)
            confidence = scores[best]

            label_map = {"positive": SentimentLabel.POSITIVE,
                         "negative": SentimentLabel.NEGATIVE,
                         "neutral":  SentimentLabel.NEUTRAL}
            sentiment = label_map.get(best, SentimentLabel.NEUTRAL)
            score = scores.get("positive", 0.0) - scores.get("negative", 0.0)

        except Exception:
            # FinBERT利用不可の場合はStage1結果を流用
            if stage1:
                sentiment  = stage1.sentiment
                score      = stage1.score
                confidence = stage1.confidence * 0.8
            else:
                sentiment  = SentimentLabel.SKIP
                score      = 0.0
                confidence = 0.0

        return SentimentResult(
            news=news,
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            stage_used=2,
        )

    def analyze_claude_api(
        self,
        news:   NewsItem,
        stage1: "_Stage1Result" = None,
    ) -> SentimentResult:
        """
        第3段階: Claude API（文脈理解・複雑ケースのみ）。
        コスト削減のため最小限に抑える。
        """
        try:
            import anthropic
            client = anthropic.Anthropic()

            prompt = (
                f"以下のニュースは株価にポジティブかネガティブか判定してください。\n"
                f"ニュース: {news.text}\n\n"
                f"以下のJSON形式で回答してください:\n"
                f'{{"sentiment": "positive/negative/neutral/skip", '
                f'"score": -1.0〜1.0, "confidence": 0〜1, '
                f'"reason": "理由", "affected_tickers": ["銘柄コード"]}}'
            )

            message = client.messages.create(
                model="claude-haiku-4-5-20251001",  # コスト削減のためHaikuを使用
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            raw = message.content[0].text.strip()
            # JSONブロックを抽出
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError("JSON解析失敗")

            label_map = {
                "positive": SentimentLabel.POSITIVE,
                "negative": SentimentLabel.NEGATIVE,
                "neutral":  SentimentLabel.NEUTRAL,
                "skip":     SentimentLabel.SKIP,
            }

            return SentimentResult(
                news=news,
                sentiment=label_map.get(parsed.get("sentiment", "skip"), SentimentLabel.SKIP),
                score=float(parsed.get("score", 0.0)),
                confidence=float(parsed.get("confidence", 0.5)),
                stage_used=3,
                affected_tickers=parsed.get("affected_tickers", []),
            )

        except Exception as e:
            # Claude API利用不可の場合はスキップ
            return SentimentResult(
                news=news,
                sentiment=SentimentLabel.SKIP,
                score=0.0,
                confidence=0.0,
                stage_used=3,
                skip_reason=f"Claude API利用不可: {e}",
            )

    def _needs_context_analysis(self, text: str) -> bool:
        """
        文脈分析が必要か判定する。

        以下の場合はClaude APIに回す：
        ・CEO辞任（後任評価が必要）
        ・買収・提携（条件次第で逆反応）
        ・規制発表（対象範囲が重要）
        ・第1・2段階の結果が矛盾している
        """
        for pattern in CONTEXT_REQUIRED_PATTERNS:
            if re.search(pattern, text):
                return True
        return False


# =============================================================
# 出来高急増フィルター
# =============================================================

def apply_volume_surge_filter(signal: dict, volume_history: list) -> dict:
    """
    ボートレースのオッズ急上昇フィルターと同じ思想。
    「市場が知っていて自分が知らない情報」に謙虚になる。

    直前の出来高が通常の3倍以上 → 信頼度を下げる。

    Parameters:
        signal         : 買いシグナル dict
        volume_history : 出来高履歴リスト（最新が末尾）

    Returns:
        signal（更新済み）
    """
    if not volume_history or len(volume_history) < 5:
        return signal

    import numpy as np
    baseline = float(np.median(volume_history[:-1]))
    current  = float(volume_history[-1])

    if baseline <= 0:
        return signal

    ratio = current / baseline

    if ratio >= 6.0:
        signal["cancelled"]    = True
        signal["confidence"]   = 0.0
        signal["cancel_reason"] = f"出来高{ratio:.1f}倍（急増）→ 完全キャンセル"
    elif ratio >= 3.0:
        signal["confidence"] *= 0.5
        signal["cancel_reason"] = f"出来高{ratio:.1f}倍 → 信頼度50%"
    elif ratio >= 2.0:
        signal["confidence"] *= 0.8
        signal["cancel_reason"] = f"出来高{ratio:.1f}倍 → 信頼度80%"

    return signal


# =============================================================
# 統合予測クラス
# =============================================================

class StockPredictor:
    """感情スコア + テクニカル + マクロ指標を統合してLightGBMで方向性予測。"""

    def __init__(self):
        self.model   = None
        self.feature_names = None

    def predict(
        self,
        news_data:  dict,
        price_data: dict,
        macro_data: dict,
    ) -> dict:
        """
        方向性を予測する。

        Parameters:
            news_data  : {"sentiment_score": float, "confidence": float, ...}
            price_data : {"rsi": float, "macd": float, "volume_ratio": float, ...}
            macro_data : {"jpy_usd": float, "vix": float, ...}

        Returns:
            {
                "ticker":    str,
                "direction": "up" / "down" / "skip",
                "confidence": float,
                "est_change": float,  # 推定変動幅（%）
            }
        """
        assert self.model is not None, (
            "先にモデルを学習してください。"
            "学習はボートレース・FXが安定後。"
        )

        import numpy as np
        features = self._build_features(news_data, price_data, macro_data)
        prob = self.model.predict_proba([features])[0]

        if max(prob) < 0.6:
            direction = "skip"
        elif prob[1] > prob[0]:
            direction = "up"
        else:
            direction = "down"

        return {
            "ticker":     news_data.get("ticker", ""),
            "direction":  direction,
            "confidence": float(max(prob)),
            "est_change": 0.0,  # 未実装 → データが揃ってから追加
        }

    def _build_features(
        self,
        news_data:  dict,
        price_data: dict,
        macro_data: dict,
    ) -> list:
        """特徴量を組み立てる。"""
        return [
            news_data.get("sentiment_score",   0.0),
            news_data.get("confidence",         0.0),
            news_data.get("stage_used",         1),
            price_data.get("rsi",              50.0),
            price_data.get("macd",              0.0),
            price_data.get("volume_ratio",      1.0),
            macro_data.get("jpy_usd",           0.0),
            macro_data.get("vix",              20.0),
        ]


# =============================================================
# 内部用データクラス
# =============================================================

@dataclass
class _Stage1Result:
    """ルールベース分析の内部結果。"""
    sentiment:  SentimentLabel
    score:      float
    confidence: float
