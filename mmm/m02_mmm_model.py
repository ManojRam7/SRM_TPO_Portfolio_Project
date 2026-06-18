"""
MMM STAGE 02 - ELASTIC-NET LOG-LOG MODEL (per Retailer x Promo Group)
Same approach as R glmnet / cv.glmnet, written in numpy. Fits a regularised log-log demand
model, selects features (incl. acting-item cross-prices), and relaxed-refits to get
unbiased elasticities. Coefficient on log_price IS the price elasticity.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
import config_mmm as M
import mmm_common as MC
from m01_data_prep import ACT_SLOTS

FEATURES = ["log_price", "catalogue", "display", "si_log", "trend", "tpr_lag1"] + ACT_SLOTS


def run():
    ads = pd.read_csv(M.MMM_OUT / "mmm_model_ads.csv")
    feats_all = [f for f in FEATURES if f in ads.columns]
    summ, coefs = [], []
    for rpg, g in ads.groupby("retailer_pg"):
        g = g.reset_index(drop=True)
        if len(g) < 40:
            continue
        res = MC.fit_elastic_net(g, feats_all, ycol="log_qty")
        c = res["coef"]
        pred = c.get("intercept", 0) + sum(c.get(f, 0) * g[f].fillna(0) for f in feats_all)
        summ.append({
            "retailer_pg": rpg, "retailer": g.retailer.iloc[0], "promo_group": g.promo_group.iloc[0],
            "lambda": res["lambda"], "r2": round(res["r2"], 4),
            "mape": round(MC.mape(g.units, np.expm1(pred)), 2),
            "price_elasticity": round(c.get("log_price", np.nan), 4),
            "b_catalogue": round(c.get("catalogue", np.nan), 4),
            "b_display": round(c.get("display", np.nan), 4),
            "n_selected": len(res["selected"]),
            "selected": ";".join(res["selected"]),
        })
        row = {"retailer_pg": rpg}; row.update({k: round(v, 6) for k, v in c.items()})
        coefs.append(row)
    s = pd.DataFrame(summ); cf = pd.DataFrame(coefs).fillna(0.0)
    s.to_csv(M.MMM_OUT / "mmm_model_summary.csv", index=False)
    cf.to_csv(M.MMM_OUT / "mmm_coefficients.csv", index=False)
    print(f"[m02] MMM models fitted: {len(s)} | median R2: {s.r2.median():.3f} | "
          f"median MAPE: {s.mape.median():.1f}% | mean elasticity: {s.price_elasticity.mean():.2f}")
    return s


if __name__ == "__main__":
    run()
