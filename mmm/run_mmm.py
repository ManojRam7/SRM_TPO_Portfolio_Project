"""
MMM pipeline runner (marketing mix modelling, retailer-level workstream).
Runs: data gen -> prep -> elastic-net model -> contribution -> ROI/scenario
-> validation -> charts + dashboard.

Usage:  python mmm/run_mmm.py
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
import pandas as pd
import config_mmm as M
import m00_generate_mmm_data as m00
import m01_data_prep as m01
import m02_mmm_model as m02
import m03_contribution as m03
import m04_roi_scenario as m04
import viz_mmm, report_mmm


def validate():
    s = pd.read_csv(M.MMM_OUT / "mmm_model_summary.csv")
    g = pd.read_csv(M.MMM_DATA / "ground_truth_mmm.csv")
    r = s.groupby("promo_group").agg(recovered=("price_elasticity", "mean"),
                                     r2=("r2", "mean"), mape=("mape", "mean")).reset_index()
    r = r.merge(g[["promo_group", "true_e_price"]], on="promo_group")
    r["abs_error"] = (r.recovered - r.true_e_price).abs().round(3)
    r["pct_error"] = (r.abs_error / r.true_e_price.abs() * 100).round(1)
    r = r.rename(columns={"true_e_price": "true"}).round({"recovered": 2, "true": 2, "r2": 3, "mape": 1})
    r.to_csv(M.MMM_OUT / "mmm_validation.csv", index=False)
    print(f"[validate] MMM elasticity recovery: mean abs error {r.abs_error.mean():.3f} | "
          f"{(r.pct_error <= 20).sum()}/{len(r)} within 20%")


def main():
    t0 = time.time()
    print("=" * 64); print("MARKETING MIX MODELLING (MMM) - PIPELINE"); print("=" * 64)
    m00.generate(); m01.run(); m02.run(); m03.run(); m04.run()
    validate()
    viz_mmm.build_all(); report_mmm.build_dashboard()
    print("=" * 64); print(f"MMM DONE in {time.time()-t0:.1f}s -> mmm/outputs/mmm_dashboard.html"); print("=" * 64)


if __name__ == "__main__":
    main()
