# ===========================================
# model/line_predictor.py
# 競輪AI - ライン予測エンジン（核心部分）
#
# 3つの情報源を統合してライン確率を計算する：
#   1. 選手コメント（最高精度・前日夜に取得）
#   2. 記者予想（高精度・外部データ）
#   3. 独自ラインスコア（地区×期別×過去実績）
#
# 注意：データがない状態でもコードを完成させた。
#       動作確認・学習はデータが揃ってから行う。
# ===========================================

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# コメントからライン意図を読み取るパターン
_FOLLOW_PATTERNS = [
    r"(\S+)選手(を|の後ろ|の番手)(を|で)?追",
    r"(\S+)選手(の後ろ|の番手)で",
    r"(\S+)番の?(後ろ|番手)",
    r"番手で",  # 特定名前なし
]

_LEAD_PATTERNS = [
    r"(先行|自力|前で|前から|逃げ)",
    r"(先頭で|先手で)",
]

_SECOND_PATTERNS = [
    r"(番手で|2番手で|後ろに?つ)",
    r"(マーク|マークで|ついて行)",
]

_SINGLE_PATTERNS = [
    r"(単騎|一人で|自分で|単独)",
]


@dataclass
class LinePrediction:
    """ライン予測の中間結果"""
    car_no: int
    line_cars: list        # ライン構成の車番リスト（先頭から順に）
    position: str          # 先頭/番手/3番手/単騎
    confidence: float      # 0〜1
    source: str            # comment/reporter/score


@dataclass
class RaceLinePrediction:
    """1レース分のライン予測結果"""
    race_id: str
    line_probs: dict       # {"3-7-4": 0.85, "9-2": 0.75, ...}
    predictability_score: float  # 0〜100
    skip_recommended: bool
    skip_reason: str


class LinePredictor:
    """
    ライン予測エンジン。

    3つの情報源を統合してライン確率を予測する。
    コメント > 記者予想 > 独自スコアの優先順位で重み付け。
    """

    # 統合時の重み
    WEIGHT_COMMENT   = 0.6
    WEIGHT_REPORTER  = 0.3
    WEIGHT_SCORE     = 0.1

    # コメントなし時の重み
    WEIGHT_NO_COMMENT_REPORTER = 0.5
    WEIGHT_NO_COMMENT_SCORE    = 0.5

    # 予測可能性スコアのペナルティ
    PENALTY_LOW_CONFIDENCE  = 30
    PENALTY_BETRAYAL_HIGH   = 20
    PENALTY_MANY_SINGLES    = 20  # 単騎3人以上
    PENALTY_STRONG_WIND     = 15  # 風速4m/s以上
    PENALTY_RAIN            = 10
    PENALTY_GENERAL_GRADE   = 5   # F1/F2

    SKIP_THRESHOLD = 60  # 予測可能性スコアがこれ以下は見送り推奨

    def predict_lines(self,
                      entries_df: pd.DataFrame,
                      comments_df: pd.DataFrame = None,
                      reporter_lines: dict = None) -> dict:
        """
        ライン構成の確率を予測する。

        Parameters:
            entries_df    : 出走情報（car_no, district, style 等を含む）
            comments_df   : 選手コメント（省略可）
            reporter_lines: 記者予想 {"3-7-4": 0.9, ...}（省略可）

        Returns:
            line_probs: {"3-7-4": 0.85, "9-2": 0.75, ...}
        """
        # ステップ1：コメントからライン推定
        comment_lines = self._predict_from_comments(entries_df, comments_df)

        # ステップ2：記者予想を参照
        reporter_result = self._use_reporter_lines(reporter_lines)

        # ステップ3：独自ラインスコアで補完
        score_lines = self._predict_from_score(entries_df)

        # ステップ4：3つを統合して最終確率を計算
        return self._integrate_predictions(
            comment_lines, reporter_result, score_lines
        )

    # =============================================================
    # ステップ1：コメント解析
    # =============================================================

    def _predict_from_comments(self,
                                entries_df: pd.DataFrame,
                                comments_df: pd.DataFrame = None
                                ) -> dict:
        """
        選手コメントからライン予測する。

        解析ルール：
        - 「◯◯選手を追います」→ 追う側・追われる側をラインとして記録
        - 「先行します」「自力で行きます」→ ライン先頭候補
        - 「番手で勝負します」→ ライン2番手候補
        - コメントがない選手 → 地区情報と独自スコアで補完・confidence を下げる

        Returns:
            {"3-7-4": confidence} 形式の dict
        """
        if comments_df is None or len(comments_df) == 0:
            return {}

        car_to_comment = {}
        for _, row in comments_df.iterrows():
            car_to_comment[int(row["car_no"])] = str(row.get("comment_text", ""))

        car_to_district = {}
        for _, row in entries_df.iterrows():
            car_to_district[int(row["car_no"])] = str(row.get("district", ""))

        # 各選手のインテント解析
        intents = {}  # car_no → {"role": "lead"|"follow"|"single", "target_name": str|None}
        for car_no, comment in car_to_comment.items():
            role, target = self._parse_comment_intent(comment)
            intents[car_no] = {"role": role, "target_name": target}

        # フォロー関係からライン候補を構築
        follow_map = {}  # 追われる選手名 → 追う選手番号のリスト
        for car_no, intent in intents.items():
            if intent["role"] == "follow" and intent["target_name"]:
                target = intent["target_name"]
                if target not in follow_map:
                    follow_map[target] = []
                follow_map[target].append(car_no)

        # ライン構成の推定
        lines = {}
        used_cars = set()

        # 先頭候補を特定して、それを追う選手をラインとして組む
        for car_no, intent in intents.items():
            if intent["role"] == "lead" and car_no not in used_cars:
                line = [car_no]
                used_cars.add(car_no)
                # この選手を追うコメントをした選手を追加
                for follower_car, follower_intent in intents.items():
                    if (follower_car not in used_cars
                            and follower_intent["role"] == "follow"):
                        line.append(follower_car)
                        used_cars.add(follower_car)
                        if len(line) >= 3:
                            break
                line_str = "-".join(str(c) for c in line)
                lines[line_str] = 0.80  # コメント由来は高信頼度

        # 単騎宣言
        for car_no, intent in intents.items():
            if intent["role"] == "single" and car_no not in used_cars:
                lines[str(car_no)] = 0.80
                used_cars.add(car_no)

        # コメントなし選手は地区ベースで補完（confidence低め）
        no_comment_cars = [
            int(row["car_no"]) for _, row in entries_df.iterrows()
            if int(row["car_no"]) not in car_to_comment
        ]
        if no_comment_cars:
            fallback = self._predict_from_score(
                entries_df[entries_df["car_no"].isin(no_comment_cars)]
            )
            for line_str, conf in fallback.items():
                if line_str not in lines:
                    lines[line_str] = conf * 0.6  # confidenceを下げる

        return lines

    def _parse_comment_intent(self, comment: str):
        """
        コメントからインテント（役割・対象）を解析する。

        Returns:
            (role, target_name)
            role: "lead" / "follow" / "single" / "unknown"
            target_name: str または None
        """
        if not comment:
            return "unknown", None

        # 先頭宣言
        for pattern in _LEAD_PATTERNS:
            if re.search(pattern, comment):
                return "lead", None

        # 単騎宣言
        for pattern in _SINGLE_PATTERNS:
            if re.search(pattern, comment):
                return "single", None

        # 追走宣言（ターゲット名を取得）
        for pattern in _FOLLOW_PATTERNS:
            m = re.search(pattern, comment)
            if m:
                # グループ1があれば選手名またはターゲット番号
                target = m.group(1) if m.lastindex and m.lastindex >= 1 else None
                return "follow", target

        # 番手宣言（ターゲット不明）
        for pattern in _SECOND_PATTERNS:
            if re.search(pattern, comment):
                return "follow", None

        return "unknown", None

    # =============================================================
    # ステップ2：記者予想の参照
    # =============================================================

    def _use_reporter_lines(self,
                             reporter_lines: dict = None) -> dict:
        """
        記者予想をそのまま返す（形式変換のみ）。

        Parameters:
            reporter_lines: {"3-7-4": 0.9, "9-2": 0.8, ...} または None

        Returns:
            同形式の dict
        """
        if not reporter_lines:
            return {}
        # 値が float でなければ無視する
        return {k: float(v) for k, v in reporter_lines.items()
                if isinstance(v, (int, float))}

    # =============================================================
    # ステップ3：独自ラインスコア
    # =============================================================

    def _predict_from_score(self,
                             entries_df: pd.DataFrame,
                             historical_pairs: dict = None) -> dict:
        """
        独自ラインスコアでライン構成を予測する。

        pair_score(A, B) の計算：
        - 同じ地区 → ベーススコア 0.7
        - 同期（期別が同じ） → +0.1
        - 過去の組成実績あり → 実績値を使う
        - 地区が違う → ベーススコア 0.05

        Returns:
            {"3-7-4": confidence} 形式の dict
        """
        if entries_df is None or len(entries_df) == 0:
            return {}

        car_nos  = list(entries_df["car_no"].astype(int))
        district = dict(zip(entries_df["car_no"].astype(int),
                            entries_df["district"].astype(str)))
        term_map = {}
        if "term" in entries_df.columns:
            term_map = dict(zip(entries_df["car_no"].astype(int),
                                entries_df["term"].astype(int)))

        # 地区ごとにグループ化
        district_groups = defaultdict(list)
        for car_no in car_nos:
            district_groups[district[car_no]].append(car_no)

        lines = {}
        used = set()

        for dist, group_cars in district_groups.items():
            if len(group_cars) < 2:
                # 単騎候補
                car_no = group_cars[0]
                if car_no not in used:
                    lines[str(car_no)] = 0.45
                    used.add(car_no)
                continue

            # 地区内でライン候補を構築
            # バック回数が多い選手を先頭候補にする
            back_counts = {}
            if "back_count" in entries_df.columns:
                for car_no in group_cars:
                    row = entries_df[entries_df["car_no"] == car_no]
                    if len(row) > 0:
                        back_counts[car_no] = int(row["back_count"].iloc[0])
            if back_counts:
                group_cars_sorted = sorted(group_cars,
                                           key=lambda c: back_counts.get(c, 0),
                                           reverse=True)
            else:
                group_cars_sorted = group_cars

            # 2〜3車ラインを構築
            for i in range(0, len(group_cars_sorted), 3):
                chunk = group_cars_sorted[i:i + 3]
                if all(c not in used for c in chunk):
                    # ペアスコア計算
                    score = 0.70  # 同地区ベース
                    if len(chunk) >= 2 and term_map:
                        if term_map.get(chunk[0]) == term_map.get(chunk[1]):
                            score += 0.10  # 同期ボーナス
                    if historical_pairs:
                        pair_key = (chunk[0], chunk[1])
                        if pair_key in historical_pairs:
                            score = historical_pairs[pair_key]

                    line_str = "-".join(str(c) for c in chunk)
                    lines[line_str] = min(score, 1.0)
                    for c in chunk:
                        used.add(c)

        # 残り（どのラインにも入らなかった選手）は単騎
        for car_no in car_nos:
            if car_no not in used:
                lines[str(car_no)] = 0.35
                used.add(car_no)

        return lines

    # =============================================================
    # ステップ4：統合
    # =============================================================

    def _integrate_predictions(self,
                                comment_lines: dict,
                                reporter_result: dict,
                                score_lines: dict) -> dict:
        """
        3つの予測を重み付き平均で統合する。

        重み：
        - コメントあり  → コメント0.6・記者0.3・スコア0.1
        - コメントなし  → 記者0.5・スコア0.5
        - 両方なし      → スコアのみ（confidence 0.3まで下げる）
        """
        has_comment  = bool(comment_lines)
        has_reporter = bool(reporter_result)

        all_lines = set(comment_lines) | set(reporter_result) | set(score_lines)
        result = {}

        for line_str in all_lines:
            c = comment_lines.get(line_str, 0.0)
            r = reporter_result.get(line_str, 0.0)
            s = score_lines.get(line_str, 0.0)

            if has_comment and has_reporter:
                conf = (c * self.WEIGHT_COMMENT
                        + r * self.WEIGHT_REPORTER
                        + s * self.WEIGHT_SCORE)
            elif has_comment:
                conf = (c * (self.WEIGHT_COMMENT + self.WEIGHT_REPORTER / 2)
                        + s * (self.WEIGHT_SCORE + self.WEIGHT_REPORTER / 2))
            elif has_reporter:
                conf = (r * self.WEIGHT_NO_COMMENT_REPORTER
                        + s * self.WEIGHT_NO_COMMENT_SCORE)
            else:
                # 両方なし → スコアのみ・confidence上限0.3
                conf = min(s, 0.30)

            result[line_str] = round(min(conf, 1.0), 4)

        # confidence が 0 のラインは除外
        return {k: v for k, v in result.items() if v > 0.0}

    # =============================================================
    # 補助メソッド
    # =============================================================

    def calc_betrayal_risk(self,
                           car_no: int,
                           historical_results: pd.DataFrame) -> float:
        """
        裏切りリスクを計算する（0〜1）。

        裏切り率 = コメントと違う行動をした回数 ÷ 総レース数

        高リスク選手：裏切り率 > 10%
        → ラインの信頼度を下げる
        → 予測可能性スコアを下げる

        Parameters:
            car_no              : 車番
            historical_results  : 過去のレース結果DataFrame
                                  （actual_line, comment_line カラム必要）

        Returns:
            betrayal_rate: 0〜1
        """
        if historical_results is None or len(historical_results) == 0:
            return 0.0  # データなし → リスク不明（0として扱う）

        racer_data = historical_results[
            historical_results["car_no"] == car_no
        ]
        if len(racer_data) == 0:
            return 0.0

        # actual_line と comment_line が両方ある行を対象にする
        valid = racer_data.dropna(subset=["actual_line", "comment_line"])
        if len(valid) == 0:
            return 0.0

        betrayals = (valid["actual_line"] != valid["comment_line"]).sum()
        return round(float(betrayals) / len(valid), 4)

    def calc_reporter_accuracy(self,
                               reporter_predictions: pd.DataFrame,
                               actual_results: pd.DataFrame) -> dict:
        """
        記者予想の正答率を計算する（グレード別・会場別）。

        Returns:
            {
                "accuracy_by_grade": {"G1": 0.65, "F1": 0.45, ...},
                "accuracy_by_venue": {"前橋": 0.70, "平塚": 0.50, ...},
                "overall": 0.55,
            }
        """
        if (reporter_predictions is None or len(reporter_predictions) == 0
                or actual_results is None or len(actual_results) == 0):
            return {"accuracy_by_grade": {}, "accuracy_by_venue": {}, "overall": 0.0}

        merged = reporter_predictions.merge(
            actual_results[["race_id", "grade", "venue_name", "actual_line"]],
            on="race_id",
            how="inner",
        )
        if len(merged) == 0:
            return {"accuracy_by_grade": {}, "accuracy_by_venue": {}, "overall": 0.0}

        merged["correct"] = (merged["predicted_line"] == merged["actual_line"])

        acc_grade = merged.groupby("grade")["correct"].mean().to_dict()
        acc_venue = merged.groupby("venue_name")["correct"].mean().to_dict()
        overall   = float(merged["correct"].mean())

        return {
            "accuracy_by_grade": {k: round(v, 4) for k, v in acc_grade.items()},
            "accuracy_by_venue": {k: round(v, 4) for k, v in acc_venue.items()},
            "overall": round(overall, 4),
        }

    def calc_predictability_score(self,
                                   entries_df: pd.DataFrame,
                                   line_probs: dict,
                                   wind_speed: float,
                                   is_rain: bool,
                                   grade: str = "F1",
                                   historical_results: pd.DataFrame = None
                                   ) -> tuple:
        """
        レースの予測可能性スコアを計算する（0〜100）。

        Returns:
            (score, skip_recommended, reason)
        """
        score = 100

        # ライン信頼度が低い（全ラインの平均 < 0.5）
        if line_probs:
            avg_conf = np.mean(list(line_probs.values()))
            if avg_conf < 0.50:
                score -= self.PENALTY_LOW_CONFIDENCE

        # 裏切り率が高い選手がいる（> 10%）
        if historical_results is not None:
            for _, row in entries_df.iterrows():
                risk = self.calc_betrayal_risk(
                    int(row["car_no"]), historical_results
                )
                if risk > 0.10:
                    score -= self.PENALTY_BETRAYAL_HIGH
                    break  # 1人でもいれば1回だけペナルティ

        # 単騎が3人以上
        single_count = sum(
            1 for k in line_probs if "-" not in k
        )
        if single_count >= 3:
            score -= self.PENALTY_MANY_SINGLES

        # 風速4m/s以上
        if wind_speed >= 4.0:
            score -= self.PENALTY_STRONG_WIND

        # 雨
        if is_rain:
            score -= self.PENALTY_RAIN

        # グレードが一般戦（F1/F2）
        if grade in ("F1", "F2"):
            score -= self.PENALTY_GENERAL_GRADE

        score = max(0, min(100, score))
        skip = score < self.SKIP_THRESHOLD

        reasons = []
        if skip:
            if line_probs and avg_conf < 0.50:
                reasons.append(f"ライン信頼度低（平均{avg_conf:.2f}）")
            if single_count >= 3:
                reasons.append(f"単騎{single_count}人")
            if wind_speed >= 4.0:
                reasons.append(f"強風（{wind_speed}m/s）")
            if is_rain:
                reasons.append("雨")

        reason_str = "・".join(reasons) if reasons else ""

        return score, skip, reason_str
