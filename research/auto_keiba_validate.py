"""
競馬データが12ヶ月分溜まったら自動でWalk-Forward検証を実行する。
スクレイパーと並行して起動し、30分ごとにデータ量をチェック。
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CHECK_INTERVAL = 1800  # 30分ごとにチェック
MIN_MONTHS = 12  # 最低12ヶ月分のデータが必要


def check_and_run():
    from research.keiba.keiba_model import (
        load_real_data, walk_forward_validate, walk_forward_validate_place,
        generate_report
    )
    import pandas as pd

    while True:
        print(f"\n[{datetime.now():%H:%M}] データ量チェック...")
        df = load_real_data()
        if df.empty:
            print("  データなし。30分後に再チェック。")
            time.sleep(CHECK_INTERVAL)
            continue

        dates = pd.to_datetime(df["race_date"])
        months_span = (dates.max() - dates.min()).days / 30.0
        n_races = df["race_id"].nunique()
        print(f"  {len(df)}行, {n_races}レース, {months_span:.1f}ヶ月分")
        print(f"  期間: {dates.min().date()} ~ {dates.max().date()}")

        if months_span < MIN_MONTHS:
            remaining = MIN_MONTHS - months_span
            print(f"  あと{remaining:.1f}ヶ月分必要。30分後に再チェック。")
            time.sleep(CHECK_INTERVAL)
            continue

        # 12ヶ月以上のデータがある！検証実行
        print(f"\n{'='*60}")
        print(f"  12ヶ月以上のデータ確認！Walk-Forward検証開始")
        print(f"{'='*60}")

        print("\n[単勝] edge=1.10")
        win_r = walk_forward_validate(
            df, initial_train_months=6, test_months=2,
            edge_threshold=1.10, bet_unit=1000
        )
        print(f"  PF={win_r['overall_profit_factor']:.2f}")
        print(f"  RR={win_r['overall_recovery_rate']:.1%}")
        print(f"  Bets={win_r['total_n_bets']}")
        print(f"  Hit={win_r['overall_hit_rate']:.1%}")

        print("\n[複勝] edge=1.05")
        place_r = walk_forward_validate_place(
            df, initial_train_months=6, test_months=2,
            edge_threshold=1.05, bet_unit=1000
        )
        print(f"  PF={place_r['overall_profit_factor']:.2f}")
        print(f"  RR={place_r['overall_recovery_rate']:.1%}")
        print(f"  Bets={place_r['total_n_bets']}")
        print(f"  Hit={place_r['overall_hit_rate']:.1%}")

        # レポート保存
        report = generate_report(win_r, place_r)
        report_path = Path(__file__).resolve().parent.parent / "data" / "keiba" / "keiba_report_real.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  レポート保存: {report_path}")

        print("\n検証完了！このスクリプトは終了します。")
        break


if __name__ == "__main__":
    check_and_run()
