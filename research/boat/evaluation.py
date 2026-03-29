# ===========================================
# evaluation.py
# ボートレースAI - 評価フレームワーク
#
# 設計方針：
#   - ROI・MDD・Sharpe比など全指標を会場別・グレード別に分解する
#   - Sharpe比のann_factorは固定値を使わない（FXモデルの失敗を繰り返さない）
#   - データが入ったら即実行できる状態
#   - ペーパートレード移行判定を自動化する
# ===========================================

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from note_logger import log_note_content, NoteCategory
    HAS_NOTE_LOGGER = True
except ImportError:
    HAS_NOTE_LOGGER = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR   = PROJECT_ROOT / "data" / "boat" / "reports"


# =============================================================
# 基本評価指標
# =============================================================

def calc_roi(total_payout: float, total_investment: float) -> float:
    """
    ROI = (総払い戻し - 総投資額) / 総投資額 × 100

    Parameters:
        total_payout     : 総払い戻し（円）
        total_investment : 総投資額（円）

    Returns:
        roi: ROI（%）
    """
    if total_investment <= 0:
        return 0.0
    return (total_payout - total_investment) / total_investment * 100.0


def calc_max_drawdown(capital_history: list) -> float:
    """
    最大ドローダウン = ピークからの最大下落率（%）

    Parameters:
        capital_history: 資金推移のリスト

    Returns:
        max_dd: 最大ドローダウン（%）
    """
    arr = np.array(capital_history, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    peak = np.maximum.accumulate(arr)
    # ピークが0のケースを回避
    drawdowns = np.where(peak > 0, (peak - arr) / peak, 0.0)
    return float(drawdowns.max() * 100.0)


def calc_sharpe(returns: list, trades_per_day: float = None) -> float:
    """
    Sharpe比を計算する。

    注意：ann_factorは実際のトレード頻度から計算する。
    固定値（252など）を使わない（FXモデルの失敗を繰り返さない）。

    ann_factor = √(252 × trades_per_day)

    Parameters:
        returns        : 1トレードあたりのリターン率リスト
        trades_per_day : 1日あたりの平均トレード数（Noneの場合は年率換算しない）

    Returns:
        sharpe: Sharpe比
    """
    arr = np.array(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    mean_r = arr.mean()
    std_r  = arr.std(ddof=1)

    if std_r == 0:
        return 0.0

    # ann_factorを実際のトレード頻度から計算（固定値を使わない）
    if trades_per_day and trades_per_day > 0:
        ann_factor = np.sqrt(252.0 * trades_per_day)
    else:
        # トレード頻度不明の場合はトレード単位のSharpe（年率換算なし）
        ann_factor = np.sqrt(len(arr))

    return float(mean_r / std_r * ann_factor)


def calc_ev_realization_rate(
    predicted_evs: list,
    actual_returns: list,
) -> float:
    """
    EV実現率：モデルが予測したEVが実際に実現しているか確認する。

    ズレが大きい場合はモデルの確率推定に問題がある。

    Parameters:
        predicted_evs  : モデルが予測したEVのリスト
        actual_returns : 実際のリターン（的中=オッズ、外れ=0）のリスト

    Returns:
        realization_rate: EV実現率（1.0が理想）
    """
    if not predicted_evs or not actual_returns:
        return 0.0

    predicted = np.array(predicted_evs, dtype=np.float64)
    actual    = np.array(actual_returns, dtype=np.float64)

    mean_predicted_ev = predicted.mean()
    mean_actual_ev    = actual.mean()

    if mean_predicted_ev <= 0:
        return 0.0

    return float(mean_actual_ev / mean_predicted_ev)


def calc_calibration(
    predicted_probs: list,
    actual_results: list,
    n_bins: int = 10,
) -> dict:
    """
    カリブレーション：「モデルが30%と言ったら実際に30%当たるか」

    Parameters:
        predicted_probs : モデルの予測確率リスト
        actual_results  : 実際の的中（1=的中/0=外れ）リスト
        n_bins          : ビン数（デフォルト10→10%刻み）

    Returns:
        calibration_result: {
            "bins": [(下限, 上限), ...],
            "predicted_mean": [...],
            "actual_rate": [...],
            "max_error": float,
            "is_well_calibrated": bool  # 最大ズレ < 10%ならTrue
        }
    """
    preds  = np.array(predicted_probs, dtype=np.float64)
    actual = np.array(actual_results,  dtype=np.float64)

    bins_data = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (preds >= low) & (preds < high)
        count = mask.sum()

        if count == 0:
            bins_data.append({
                "range":    (round(low, 2), round(high, 2)),
                "count":    0,
                "predicted": None,
                "actual":   None,
                "error":    None,
            })
            continue

        pred_mean  = float(preds[mask].mean())
        actual_rate = float(actual[mask].mean())
        bins_data.append({
            "range":    (round(low, 2), round(high, 2)),
            "count":    int(count),
            "predicted": round(pred_mean, 4),
            "actual":   round(actual_rate, 4),
            "error":    round(abs(pred_mean - actual_rate), 4),
        })

    errors = [b["error"] for b in bins_data if b["error"] is not None]
    max_error = float(max(errors)) if errors else 0.0

    return {
        "bins":               bins_data,
        "max_error":          round(max_error, 4),
        "is_well_calibrated": max_error < 0.10,  # 10%未満ならOK
    }


# =============================================================
# 分解評価
# =============================================================

def calc_breakdown_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    全体のROIだけでなく分解して見る。

    分解軸：
    ・会場別
    ・グレード別（SG/G1/G2/G3/一般）
    ・レース番号別（1R〜12R）
    ・進入コース別（1着艇のコース）

    Parameters:
        results_df: 列 [race_id, venue_id, grade, race_no, course_taken_1st,
                        investment, payout, is_hit]

    Returns:
        breakdown_df: 分解結果のDataFrame
    """
    def _calc_group_metrics(grp):
        roi   = calc_roi(grp["payout"].sum(), grp["investment"].sum())
        n     = len(grp)
        hits  = grp["is_hit"].sum()
        return pd.Series({
            "n_bets":      n,
            "hit_count":   hits,
            "hit_rate":    hits / n if n > 0 else 0.0,
            "total_invest": grp["investment"].sum(),
            "total_payout": grp["payout"].sum(),
            "roi":         roi,
        })

    breakdown_list = []

    # 会場別
    if "venue_id" in results_df.columns:
        v_bd = results_df.groupby("venue_id").apply(_calc_group_metrics).reset_index()
        v_bd.insert(0, "axis", "venue")
        v_bd.rename(columns={"venue_id": "group_key"}, inplace=True)
        breakdown_list.append(v_bd)

    # グレード別
    if "grade" in results_df.columns:
        g_bd = results_df.groupby("grade").apply(_calc_group_metrics).reset_index()
        g_bd.insert(0, "axis", "grade")
        g_bd.rename(columns={"grade": "group_key"}, inplace=True)
        breakdown_list.append(g_bd)

    # レース番号別
    if "race_no" in results_df.columns:
        r_bd = results_df.groupby("race_no").apply(_calc_group_metrics).reset_index()
        r_bd.insert(0, "axis", "race_no")
        r_bd.rename(columns={"race_no": "group_key"}, inplace=True)
        breakdown_list.append(r_bd)

    # 1着艇の進入コース別
    if "course_taken_1st" in results_df.columns:
        c_bd = results_df.groupby("course_taken_1st").apply(_calc_group_metrics).reset_index()
        c_bd.insert(0, "axis", "course_1st")
        c_bd.rename(columns={"course_taken_1st": "group_key"}, inplace=True)
        breakdown_list.append(c_bd)

    if not breakdown_list:
        return pd.DataFrame()

    return pd.concat(breakdown_list, ignore_index=True)


# =============================================================
# ペーパートレード移行判定
# =============================================================

def check_paper_trade_ready(metrics: dict) -> bool:
    """
    以下を全て満たすかチェックする。

    ペーパートレード移行条件（CLAUDE.mdより）：
    ① ROI > 0%
    ② 最大MDD < 30%
    ③ カリブレーションのズレ < 10%
    ④ 3ヶ月以上のペーパートレード（データ量チェック）

    Parameters:
        metrics: {
            "roi": float,
            "max_drawdown": float,
            "calibration_max_error": float,
            "n_months": float,
        }

    Returns:
        True: 全条件を満たす（移行OK）
        False: 条件未達
    """
    checks = {
        "ROI > 0%":              metrics.get("roi", -1)          >  0.0,
        "最大MDD < 30%":          metrics.get("max_drawdown", 100) < 30.0,
        "カリブレーションズレ < 10%": metrics.get("calibration_max_error", 1.0) < 0.10,
        "3ヶ月以上のデータ":       metrics.get("n_months", 0)      >= 3.0,
    }

    print("=== ペーパートレード移行判定 ===")
    all_pass = True
    for cond, passed in checks.items():
        status = "✓ OK" if passed else "✗ NG"
        print(f"  {status}  {cond}: {metrics.get(cond.split()[0].lower(), '?')}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n→ 全条件クリア！ペーパートレードへ移行可能です。")
    else:
        print("\n→ 条件未達。引き続き検証を継続してください。")

    return all_pass


# =============================================================
# 週次・月次レポート
# =============================================================

def generate_weekly_report(results_df: pd.DataFrame, save_path: str = None) -> str:
    """
    毎週月曜日に自動実行する週次レポートを生成する。

    Returns:
        report_text: レポート文字列
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or str(
        REPORT_DIR / f"weekly_{datetime.now().strftime('%Y%m%d')}.txt"
    )

    if results_df.empty:
        report = "データなし（データが揃ってから実行してください）"
        Path(save_path).write_text(report, encoding="utf-8")
        return report

    total_invest  = results_df["investment"].sum()
    total_payout  = results_df["payout"].sum()
    roi           = calc_roi(total_payout, total_invest)
    hit_rate      = results_df["is_hit"].mean() if "is_hit" in results_df.columns else 0.0

    capital_hist  = results_df["capital"].tolist() if "capital" in results_df.columns else []
    max_dd        = calc_max_drawdown(capital_hist) if capital_hist else 0.0

    lines = [
        f"=== 週次レポート {datetime.now().strftime('%Y-%m-%d')} ===",
        f"投資額合計: {total_invest:,.0f}円",
        f"払戻合計:   {total_payout:,.0f}円",
        f"ROI:       {roi:+.2f}%",
        f"的中率:    {hit_rate:.2%}",
        f"最大DD:    {max_dd:.2f}%",
        "",
    ]

    # 分解評価
    breakdown = calc_breakdown_metrics(results_df)
    if not breakdown.empty:
        lines.append("--- 会場別 ---")
        venue_bd = breakdown[breakdown["axis"] == "venue"]
        for _, row in venue_bd.iterrows():
            lines.append(
                f"  会場{row['group_key']}: ROI={row['roi']:+.1f}% "
                f"({row['n_bets']}件)"
            )

    report = "\n".join(lines)
    Path(save_path).write_text(report, encoding="utf-8")
    print(f"週次レポート保存: {save_path}")
    return report


def generate_monthly_report(results_df: pd.DataFrame, save_path: str = None) -> str:
    """
    毎月1日に自動実行する月次レポートを生成する。

    Returns:
        report_text: レポート文字列
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or str(
        REPORT_DIR / f"monthly_{datetime.now().strftime('%Y%m')}.txt"
    )

    if results_df.empty:
        report = "データなし（データが揃ってから実行してください）"
        Path(save_path).write_text(report, encoding="utf-8")
        return report

    # 月次では分解評価を全軸で実施
    total_invest = results_df["investment"].sum()
    total_payout = results_df["payout"].sum()
    roi          = calc_roi(total_payout, total_invest)

    lines = [
        f"=== 月次レポート {datetime.now().strftime('%Y年%m月')} ===",
        f"投資額合計: {total_invest:,.0f}円",
        f"払戻合計:   {total_payout:,.0f}円",
        f"ROI:       {roi:+.2f}%",
        "",
    ]

    breakdown = calc_breakdown_metrics(results_df)
    if not breakdown.empty:
        for axis in ["venue", "grade", "race_no", "course_1st"]:
            ax_data = breakdown[breakdown["axis"] == axis]
            if ax_data.empty:
                continue
            label_map = {
                "venue": "会場別", "grade": "グレード別",
                "race_no": "レース番号別", "course_1st": "1着コース別",
            }
            lines.append(f"--- {label_map.get(axis, axis)} ---")
            for _, row in ax_data.iterrows():
                lines.append(
                    f"  {row['group_key']}: ROI={row['roi']:+.1f}% "
                    f"的中率={row['hit_rate']:.1%} ({row['n_bets']}件)"
                )
            lines.append("")

    report = "\n".join(lines)
    Path(save_path).write_text(report, encoding="utf-8")
    print(f"月次レポート保存: {save_path}")
    return report


# =============================================================
# note記録との連携
# =============================================================

def log_significant_events(metrics_before: dict, metrics_after: dict) -> None:
    """
    ROIが5%以上変化したときnote_logger.pyを呼び出して記録する。

    Parameters:
        metrics_before : 変化前の評価指標
        metrics_after  : 変化後の評価指標
    """
    if not HAS_NOTE_LOGGER:
        return

    roi_change = metrics_after.get("roi", 0) - metrics_before.get("roi", 0)

    if abs(roi_change) >= 5.0:
        log_note_content(
            title=f"ROIが{roi_change:+.1f}%変化：原因と詳細",
            category=NoteCategory.IMPROVEMENT if roi_change > 0 else NoteCategory.FAILURE,
            what_happened=(
                f"ROI {metrics_before.get('roi', 0):.1f}% "
                f"→ {metrics_after.get('roi', 0):.1f}%"
            ),
            why_it_matters="精度の変化は読者に最も刺さるコンテンツ",
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            priority="high",
        )


# =============================================================
# 全指標をまとめて計算するユーティリティ
# =============================================================

def calc_all_metrics(
    results_df: pd.DataFrame,
    predicted_probs: list = None,
    actual_results: list  = None,
) -> dict:
    """
    全評価指標をまとめて計算する。

    Parameters:
        results_df      : 取引結果DataFrame
        predicted_probs : カリブレーション用の予測確率リスト
        actual_results  : カリブレーション用の的中結果リスト

    Returns:
        metrics: 全指標のdict
    """
    if results_df.empty:
        return {"error": "データなし"}

    total_invest  = results_df["investment"].sum()
    total_payout  = results_df["payout"].sum()
    roi           = calc_roi(total_payout, total_invest)

    capital_hist  = results_df["capital"].tolist() if "capital" in results_df.columns else []
    max_dd        = calc_max_drawdown(capital_hist) if capital_hist else 0.0

    # Sharpe比（実際のトレード頻度から計算）
    if "return_rate" in results_df.columns and "date" in results_df.columns:
        n_days = results_df["date"].nunique()
        n_trades = len(results_df)
        tpd = n_trades / max(n_days, 1)
        sharpe = calc_sharpe(results_df["return_rate"].tolist(), trades_per_day=tpd)
    else:
        sharpe = 0.0

    # EV実現率
    ev_real = 0.0
    if "predicted_ev" in results_df.columns and "actual_return" in results_df.columns:
        ev_real = calc_ev_realization_rate(
            results_df["predicted_ev"].tolist(),
            results_df["actual_return"].tolist(),
        )

    # カリブレーション
    calibration = {}
    if predicted_probs and actual_results:
        calibration = calc_calibration(predicted_probs, actual_results)

    # 期間
    if "date" in results_df.columns:
        dates = pd.to_datetime(results_df["date"].astype(str), format="%Y%m%d", errors="coerce")
        n_days   = (dates.max() - dates.min()).days if not dates.isna().all() else 0
        n_months = n_days / 30.0
    else:
        n_months = 0.0

    metrics = {
        "roi":                    round(roi, 4),
        "max_drawdown":           round(max_dd, 4),
        "sharpe":                 round(sharpe, 4),
        "ev_realization_rate":    round(ev_real, 4),
        "calibration_max_error":  calibration.get("max_error", None),
        "n_months":               round(n_months, 1),
        "n_trades":               len(results_df),
        "total_investment":       total_invest,
        "total_payout":           total_payout,
    }

    return metrics
