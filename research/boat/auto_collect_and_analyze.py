# ===========================================
# auto_collect_and_analyze.py
# ボートレースデータの自動収集 → 自動解析パイプライン v3
#
# v3 高速化:
#   - 3会場同時並列収集（concurrent.futures）
#   - オッズは初回スキップ（データ量優先）
#   - スリープ短縮（0.3-0.8秒）
#   - タイムアウト15秒、リトライ1回
#   - アンサンブルモデル（LGB + XGB）
# ===========================================

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# === 設定 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "boat"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR / "auto_pipeline.log"
REPORT_FILE = DATA_DIR / "real_data_report.txt"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 並列数（サーバーに配慮しつつ高速化）
MAX_WORKERS = 3


def _collect_venue(venue, start, end):
    """1会場分の収集（スレッドで実行）"""
    from research.boat.fetch_real_data import fetch_historical_data
    try:
        fetch_historical_data(start, end, venues=[venue], skip_odds=True)
        return venue, True
    except Exception as e:
        logger.warning(f"会場{venue} 収集エラー: {e}")
        return venue, False


def phase1_collect():
    """Phase 1: 並列データ収集（主要8会場 × 半年、オッズスキップ）"""
    import pandas as pd

    csv_path = DATA_DIR / "real_race_data.csv"

    existing_count = 0
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
            existing_count = len(df)
        except Exception:
            pass

    logger.info(f"=== Phase 1: 並列データ収集開始（既存: {existing_count}レース）===")
    logger.info(f"並列数: {MAX_WORKERS}会場同時")

    today = datetime.now()

    # 半年分を2期間に分割
    periods = [
        ((today - timedelta(days=180)).strftime("%Y%m%d"),
         (today - timedelta(days=90)).strftime("%Y%m%d")),
        ((today - timedelta(days=90)).strftime("%Y%m%d"),
         today.strftime("%Y%m%d")),
    ]

    # 主要8会場
    priority_venues = ["04", "24", "12", "01", "21", "15", "07", "17"]
    # 残り16会場
    other_venues = [
        "02", "03", "05", "06", "08", "09", "10", "11",
        "13", "14", "16", "18", "19", "20", "22", "23",
    ]

    # 並列で収集（主要会場 → 残り会場）
    for label, venues in [("主要8会場", priority_venues), ("残り16会場", other_venues)]:
        logger.info(f"\n--- {label} 収集開始 ---")
        for start, end in periods:
            # MAX_WORKERS会場を同時に収集
            tasks = [(v, start, end) for v in venues]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_collect_venue, v, s, e): v
                    for v, s, e in tasks
                }
                for future in as_completed(futures):
                    venue, success = future.result()
                    status = "完了" if success else "エラー"
                    logger.info(f"  会場{venue}: {status}")

            # 進捗表示
            if csv_path.exists():
                df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
                n_venues = df["venue_code"].nunique()
                logger.info(f"  累計: {len(df)}レース ({n_venues}会場) [{start}〜{end}]")

    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"venue_code": str, "date": str})
        return len(df)
    return 0


def phase2_analyze():
    """Phase 2: 改良版モデル学習と評価"""
    import pandas as pd
    import numpy as np
    from research.boat.fetch_real_data import convert_to_model_format

    logger.info("=== Phase 2: 改良版モデル学習・評価 ===")

    df = convert_to_model_format()
    if df is None or df.empty:
        logger.error("データ変換失敗")
        return

    n_races = df["race_id"].nunique()
    logger.info(f"学習データ: {n_races}レース × 6艇 = {len(df)}行")

    # === レポート開始 ===
    lines = []
    lines.append("=" * 70)
    lines.append(f"  ボートレース実データ解析レポート v3")
    lines.append(f"  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  データ: {n_races}レース")
    if "venue_code" in df.columns:
        n_v = df["venue_code"].nunique()
        lines.append(f"  会場数: {n_v}")
    lines.append("=" * 70)

    # 枠番別統計
    lines.append("\n【枠番別勝率・2着率】")
    for lane in range(1, 7):
        ld = df[df["lane"] == lane]
        wr = ld["win"].mean()
        pr = ld["place_top2"].mean()
        lines.append(f"  {lane}号艇: 勝率={wr:.1%}  2着率={pr:.1%}")

    # === 拡張特徴量 ===
    features = [
        "lane", "racer_class", "racer_win_rate", "racer_place_rate",
        "racer_3place_rate", "racer_local_win_rate", "racer_local_2place_rate",
        "motor_2place_rate", "boat_2place_rate", "avg_start_timing",
        "flying_count", "late_count", "weight",
        "weather_wind_speed", "weather_condition", "wave_height", "course_type",
    ]

    available = [f for f in features if f in df.columns]
    lines.append(f"\n【使用特徴量】{len(available)}個: {', '.join(available)}")

    X = df[available].copy().fillna(0)
    y_win = df["win"].copy()

    # Walk-Forward: 時系列70/30
    race_ids = sorted(df["race_id"].unique())
    split_idx = int(len(race_ids) * 0.7)
    train_races = set(race_ids[:split_idx])
    test_races = set(race_ids[split_idx:])

    train_mask = df["race_id"].isin(train_races)
    test_mask = df["race_id"].isin(test_races)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_win, y_test_win = y_win[train_mask], y_win[test_mask]

    lines.append(f"\n【モデル学習】")
    lines.append(f"  学習: {len(X_train)}行 ({len(train_races)}レース)")
    lines.append(f"  テスト: {len(X_test)}行 ({len(test_races)}レース)")

    try:
        import lightgbm as lgb
        import xgboost as xgb

        # === LightGBM ===
        lgb_win = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
        )
        lgb_win.fit(X_train, y_train_win)
        lgb_proba = lgb_win.predict_proba(X_test)[:, 1]

        # === XGBoost ===
        xgb_win = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, verbosity=0, use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_win.fit(X_train, y_train_win)
        xgb_proba = xgb_win.predict_proba(X_test)[:, 1]

        # === アンサンブル ===
        ensemble_proba = (lgb_proba + xgb_proba) / 2.0

        test_df = df[test_mask].copy()
        test_df["pred_win_lgb"] = lgb_proba
        test_df["pred_win_xgb"] = xgb_proba
        test_df["pred_win"] = ensemble_proba

        lines.append(f"\n{'='*70}")
        lines.append(f"  バックテスト結果")
        lines.append(f"{'='*70}")

        # --- 戦略A: 最高確率単勝 ---
        total_bet_a, total_return_a, correct_a, races_a = 0, 0, 0, 0
        for race_id in test_races:
            race = test_df[test_df["race_id"] == race_id]
            if len(race) != 6:
                continue
            races_a += 1
            best_idx = race["pred_win"].idxmax()
            if race.loc[best_idx, "win"] == 1:
                correct_a += 1
                pred = max(race.loc[best_idx, "pred_win"], 0.05)
                est_odds = min(1.0 / pred, 50.0)
                total_return_a += 100 * est_odds
            total_bet_a += 100

        if races_a > 0:
            acc_a = correct_a / races_a
            roi_a = total_return_a / total_bet_a
            lines.append(f"\n  戦略A（単勝・最高確率）:")
            lines.append(f"    レース数: {races_a}")
            lines.append(f"    的中率: {acc_a:.1%} ({correct_a}/{races_a})")
            lines.append(f"    投資: {total_bet_a:,}円 → 回収: {total_return_a:,.0f}円")
            lines.append(f"    回収率: {roi_a:.1%}")

        # --- 戦略B: 高確信度のみ ---
        total_bet_b, total_return_b, correct_b, bets_b = 0, 0, 0, 0
        for race_id in test_races:
            race = test_df[test_df["race_id"] == race_id]
            if len(race) != 6:
                continue
            best_idx = race["pred_win"].idxmax()
            pred = race.loc[best_idx, "pred_win"]
            if pred >= 0.50:
                bets_b += 1
                total_bet_b += 100
                if race.loc[best_idx, "win"] == 1:
                    correct_b += 1
                    est_odds = min(1.0 / max(pred, 0.05), 50.0)
                    total_return_b += 100 * est_odds

        if bets_b > 0:
            acc_b = correct_b / bets_b
            roi_b = total_return_b / total_bet_b
            lines.append(f"\n  戦略B（高確信度 ≥50%）:")
            lines.append(f"    ベット数: {bets_b}/{races_a}レース（選択率{bets_b/races_a:.0%}）")
            lines.append(f"    的中率: {acc_b:.1%} ({correct_b}/{bets_b})")
            lines.append(f"    投資: {total_bet_b:,}円 → 回収: {total_return_b:,.0f}円")
            lines.append(f"    回収率: {roi_b:.1%}")

        # --- 戦略C: オッズ連動バリューベッティング ---
        has_odds = "odds" in test_df.columns and test_df["odds"].notna().sum() > 0
        if has_odds:
            total_bet_c, total_return_c, correct_c, bets_c = 0, 0, 0, 0
            for race_id in test_races:
                race = test_df[test_df["race_id"] == race_id]
                if len(race) != 6:
                    continue
                for _, row in race.iterrows():
                    if pd.isna(row.get("odds")) or row["odds"] <= 0:
                        continue
                    pred = row["pred_win"]
                    actual_odds = row["odds"]
                    expected_value = pred * actual_odds * 0.75
                    if expected_value > 1.20:
                        bets_c += 1
                        total_bet_c += 100
                        if row["win"] == 1:
                            correct_c += 1
                            total_return_c += 100 * actual_odds

            if bets_c > 0:
                roi_c = total_return_c / total_bet_c
                lines.append(f"\n  戦略C（オッズ連動バリュー EV>1.2）:")
                lines.append(f"    ベット数: {bets_c}")
                lines.append(f"    的中数: {correct_c}")
                lines.append(f"    投資: {total_bet_c:,}円 → 回収: {total_return_c:,.0f}円")
                lines.append(f"    回収率: {roi_c:.1%}")
            else:
                lines.append(f"\n  戦略C: EV>1.2の機会なし")
        else:
            lines.append(f"\n  戦略C: オッズデータなし（次回収集時に取得）")

        # --- 特徴量重要度 ---
        lines.append(f"\n【特徴量重要度（LightGBM）】")
        importance = dict(zip(available, lgb_win.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * int(imp / max(importance.values()) * 20)
            lines.append(f"  {feat:25s} {imp:>4d} {bar}")

        # --- モデル比較 ---
        lines.append(f"\n【モデル別的中率】")
        for name, proba in [("LightGBM", lgb_proba), ("XGBoost", xgb_proba), ("Ensemble", ensemble_proba)]:
            test_df_tmp = df[test_mask].copy()
            test_df_tmp["p"] = proba
            correct = 0
            total = 0
            for race_id in test_races:
                race = test_df_tmp[test_df_tmp["race_id"] == race_id]
                if len(race) != 6:
                    continue
                total += 1
                if race.loc[race["p"].idxmax(), "win"] == 1:
                    correct += 1
            if total > 0:
                lines.append(f"  {name:12s}: {correct}/{total} = {correct/total:.1%}")

    except Exception as e:
        lines.append(f"\nモデル学習エラー: {e}")
        import traceback
        lines.append(traceback.format_exc())

    lines.append(f"\n{'='*70}")
    lines.append(f"  改善方針:")
    lines.append(f"  - データ量増加で精度向上（現在{n_races}→目標5000+レース）")
    lines.append(f"  - オッズデータ追加収集でバリューベッティング検証")
    lines.append(f"  - 会場別モデル（インが強い場/弱い場）")
    lines.append(f"{'='*70}")

    report_text = "\n".join(lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    logger.info(f"レポート保存: {REPORT_FILE}")
    logger.info("\n" + report_text)


def main():
    logger.info("=" * 70)
    logger.info("  ボートレース自動収集・解析パイプライン v3（並列高速版）")
    logger.info(f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  並列数: {MAX_WORKERS}会場同時")
    logger.info("=" * 70)

    start_time = time.time()

    # Phase 1
    total_races = phase1_collect()
    elapsed = (time.time() - start_time) / 60
    logger.info(f"収集完了: {total_races}レース（{elapsed:.0f}分経過）")

    # Phase 2
    if total_races >= 100:
        phase2_analyze()
    else:
        logger.info(f"データ不足（{total_races}レース）")

    elapsed = (time.time() - start_time) / 60
    logger.info(f"全工程完了: {elapsed:.0f}分")


if __name__ == "__main__":
    main()
