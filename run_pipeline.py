"""
SRM / Trade Promotion Optimisation - end-to-end pipeline runner.
Runs every stage in order (ingest -> treat -> model -> baseline -> insights ->
optimise -> QC), the same sequence a scheduled Azure Data Factory job would run on
Databricks, then builds the charts and the HTML dashboard.

Usage:  python run_pipeline.py
"""
import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import s00_generate_sample_data as s00
import s01_union_map as s01
import s02_treatment as s02
import s03_modelling as s03
import s04_baseline as s04
import s05_insights as s05
import s06_optimiser as s06
import s07_quality_checks as s07


def main():
    t0 = time.time()
    print("=" * 70)
    print("SRM / TRADE PROMOTION OPTIMISATION  -  END TO END PIPELINE")
    print("=" * 70)
    s00.generate()
    s01.run()
    s02.run()
    s03.run()
    s04.run()
    s05.run()
    s06.run()
    s07.run()
    # extras
    try:
        import viz, report
        viz.build_all()
        report.build_dashboard()
    except Exception as e:
        print("[extras] charts/dashboard skipped:", e)
    print("=" * 70)
    print(f"DONE in {time.time()-t0:.1f}s. See ./outputs for results, charts and dashboard.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
