# ===========================================
# auto_pipeline.py
# 暗号通貨AI予測: 自動パイプライン
# データ収集 → 特徴量生成 → モデル学習 → 評価レポート
# ===========================================

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "crypto"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = DATA_DIR / "crypto_model_report.txt"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "crypto_pipeline.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def phase1_collect():
    """Phase 1: データ収集"""
    logger.info("=" * 70)
    logger.info("  Phase 1: 暗号通貨データ収集")
    logger.info("=" * 70)

    try:
        from research.crypto.data_collector import main as collect_main
        collect_main()
        logger.info("Phase 1 完了: データ収集成功")
        return True
    except Exception as e:
        logger.error(f"Phase 1 エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def phase2_train():
    """Phase 2: モデル学習・評価"""
    logger.info("=" * 70)
    logger.info("  Phase 2: ハイブリッドモデル学習・評価")
    logger.info("=" * 70)

    btc_csv = DATA_DIR / "btc_1h.csv"
    if not btc_csv.exists():
        logger.error(f"BTC 1hデータが見つかりません: {btc_csv}")
        return False

    try:
        from research.crypto.hybrid_model import train_and_evaluate
        results = train_and_evaluate(str(btc_csv))

        # レポート生成
        lines = []
        lines.append("=" * 70)
        lines.append(f"  暗号通貨AI予測モデル 評価レポート")
        lines.append(f"  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        if results and isinstance(results, dict):
            # Walk-Forward結果
            if "walk_forward" in results:
                wf = results["walk_forward"]
                lines.append(f"\n【Walk-Forward検証】")
                for key, val in wf.items():
                    if isinstance(val, float):
                        lines.append(f"  {key}: {val:.4f}")
                    else:
                        lines.append(f"  {key}: {val}")

            # Holdout結果
            if "holdout" in results:
                ho = results["holdout"]
                lines.append(f"\n【Holdout テスト結果】")
                for key, val in ho.items():
                    if isinstance(val, float):
                        lines.append(f"  {key}: {val:.4f}")
                    else:
                        lines.append(f"  {key}: {val}")

            # モデル別結果
            if "sub_models" in results:
                lines.append(f"\n【サブモデル別成績】")
                for name, metrics in results["sub_models"].items():
                    lines.append(f"  {name}:")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            lines.append(f"    {k}: {v:.4f}")
                        else:
                            lines.append(f"    {k}: {v}")
        else:
            lines.append(f"\n結果: {results}")

        lines.append(f"\n{'='*70}")

        report_text = "\n".join(lines)
        REPORT_FILE.write_text(report_text, encoding="utf-8")
        logger.info(f"レポート保存: {REPORT_FILE}")
        logger.info("\n" + report_text)

        return True

    except Exception as e:
        logger.error(f"Phase 2 エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # エラーレポートも保存
        error_report = f"""{'='*70}
  暗号通貨AI予測モデル - エラーレポート
  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

エラー: {e}

{traceback.format_exc()}

対応方針:
  - データ量を確認（最低1000時間足が必要）
  - 特徴量のNaN率を確認
  - メモリ使用量を確認（LSTMは大量メモリを使用）
{'='*70}
"""
        REPORT_FILE.write_text(error_report, encoding="utf-8")
        return False


def main():
    logger.info("=" * 70)
    logger.info("  暗号通貨AI自動パイプライン開始")
    logger.info(f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    start = time.time()

    # Phase 1: データ収集
    if not phase1_collect():
        logger.error("データ収集に失敗。パイプライン中断。")
        return

    elapsed = (time.time() - start) / 60
    logger.info(f"データ収集完了: {elapsed:.1f}分経過")

    # Phase 2: モデル学習
    if not phase2_train():
        logger.warning("モデル学習でエラーが発生。レポートを確認してください。")

    elapsed = (time.time() - start) / 60
    logger.info(f"全工程完了: {elapsed:.1f}分")
    logger.info(f"レポート: {REPORT_FILE}")


if __name__ == "__main__":
    main()
