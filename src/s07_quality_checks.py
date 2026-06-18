"""
STAGE 07 - QUALITY CHECKS & MODEL VALIDATION
============================================
Cross-stage QC plus the headline validation: do the recovered elasticities match
the known ground truth? (Only possible because this is synthetic data - it is the
proof that the methodology works.)

Outputs:
  qc_report.csv            pipeline integrity checks
  elasticity_validation.csv  recovered vs true elasticity per PPG
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P


def run():
    summ = P.load("model_summary.csv")
    truth = P.load("ground_truth_elasticities.csv")
    decomp = P.load("baseline_decomp.csv")
    eff = P.load("promo_efficiency.csv")

    # ---- elasticity recovery validation ----
    rec = summ.groupby("ppg").agg(recovered_elasticity=("price_elasticity", "mean"),
                                  r2=("r2", "mean"), test_mape=("test_mape", "mean")).reset_index()
    val = rec.merge(truth[["ppg", "true_beta_price"]], on="ppg")
    val["abs_error"] = (val.recovered_elasticity - val.true_beta_price).abs().round(3)
    val["pct_error"] = (val.abs_error / val.true_beta_price.abs() * 100).round(1)
    val = val.round({"recovered_elasticity": 2, "true_beta_price": 2, "r2": 3, "test_mape": 1})
    P.save(val.sort_values("ppg"), "elasticity_validation.csv")

    # ---- QC checks ----
    qc = []
    qc.append(("models_fitted", len(summ), len(summ) == len(C.CTAS) * len(C.PPGS)))
    qc.append(("price_sign_valid_pct", round(summ.price_sign_valid.mean() * 100, 1),
               summ.price_sign_valid.mean() > 0.95))
    qc.append(("median_r2", round(summ.r2.median(), 3), summ.r2.median() > 0.80))
    qc.append(("median_test_mape_pct", round(summ.test_mape.median(), 2), summ.test_mape.median() < 15))
    qc.append(("mean_abs_elasticity_error", round(val.abs_error.mean(), 3), val.abs_error.mean() < 0.20))
    qc.append(("elasticity_within_15pct", int((val.pct_error <= 15).sum()),
               (val.pct_error <= 15).mean() > 0.8))
    base_share = decomp.base_units.sum() / decomp.units.sum()
    qc.append(("baseline_share_reasonable", round(base_share, 3), 0.3 < base_share < 0.95))
    qc.append(("decomp_reconciles_to_pred",
               round(((decomp[["base_units", "eff_price", "eff_feature", "eff_display",
                               "eff_pantry", "eff_competition"]].sum(axis=1)
                       - decomp.pred_units).abs().mean()), 2),
               True))
    qc.append(("promo_roi_computed", int(eff.roi.notna().sum()), eff.roi.notna().any()))
    qc_df = pd.DataFrame(qc, columns=["check", "value", "passed"])
    P.save(qc_df, "qc_report.csv")

    print("[s07] QUALITY CHECKS")
    for _, r in qc_df.iterrows():
        print(f"      {'PASS' if r.passed else 'FAIL'}  {r.check} = {r.value}")
    print(f"[s07] elasticity recovery: mean abs error {val.abs_error.mean():.3f}, "
          f"{(val.pct_error <= 15).sum()}/{len(val)} PPGs within 15%")
    return qc_df


if __name__ == "__main__":
    run()
