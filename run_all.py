"""
Run both workstreams: SRM / trade promotion optimisation and marketing mix modelling.
Usage:  python run_all.py
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script):
    print(f"\n{'#'*70}\n# RUNNING {script}\n{'#'*70}")
    subprocess.run([sys.executable, script], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    run("run_pipeline.py")        # SRM / trade promotion optimisation (CTA x PPG x week)
    run("mmm/run_mmm.py")         # marketing mix modelling (retailer x promo group x week)
    print("\nALL DONE - SRM + MMM complete.")
    print("  SRM dashboard : outputs/dashboard.html")
    print("  MMM dashboard : mmm/outputs/mmm_dashboard.html")
