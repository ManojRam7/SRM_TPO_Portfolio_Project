"""
MMM STAGE 03 - CONTRIBUTION DECOMPOSITION
Baseline = predict with promo levers off
(price->base, catalogue/display/pantry = 0); incremental = actual - baseline,
decomposed (log-additive, exact) into price, catalogue, display, pantry, competition.
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
from m01_data_prep import ACT_SLOTS


def run():
    ads = pd.read_csv(M.MMM_OUT / "mmm_model_ads.csv")
    coef = pd.read_csv(M.MMM_OUT / "mmm_coefficients.csv").set_index("retailer_pg")
    rows = []
    for rpg, g in ads.groupby("retailer_pg"):
        if rpg not in coef.index:
            continue
        c = coef.loc[rpg].to_dict()
        g = g.reset_index(drop=True); n = len(g)
        logp = np.log(g.avg_price.values); logb = np.log(g.base_price.values)

        def T(name, vec):
            return c.get(name, 0.0) * vec if name in c else np.zeros(n)

        acts = [s for s in ACT_SLOTS if s in g.columns and c.get(s, 0.0) != 0.0]
        comp_act = np.zeros(n); comp_base = np.zeros(n)
        for s in acts:
            v = g[s].fillna(g[s].median()).values
            comp_act += c.get(s, 0.0) * v
            comp_base += c.get(s, 0.0) * np.nanmedian(v)

        k0 = (c.get("intercept", 0.0) + T("log_price", logb) + T("si_log", g.si_log.values)
              + T("trend", g.trend.values) + comp_base)
        base = np.exp(k0); cum = k0.copy(); contrib = {}
        for name, add in [("price", T("log_price", logp) - T("log_price", logb)),
                          ("catalogue", T("catalogue", g.catalogue.values)),
                          ("display", T("display", g.display.values)),
                          ("pantry", T("tpr_lag1", g.tpr_lag1.values)),
                          ("competition", comp_act - comp_base)]:
            new = cum + add; contrib[name] = np.exp(new) - np.exp(cum); cum = new
        pred = np.exp(cum)
        rows.append(pd.DataFrame({
            "retailer": g.retailer, "promo_group": g.promo_group, "retailer_pg": rpg,
            "week_ending": g.week_ending, "units": g.units, "base_units": base.round(1),
            "incr_units": (pred - base).round(1), "eff_price": contrib["price"].round(1),
            "eff_catalogue": contrib["catalogue"].round(1), "eff_display": contrib["display"].round(1),
            "eff_pantry": contrib["pantry"].round(1), "eff_competition": contrib["competition"].round(1),
            "avg_price": g.avg_price, "base_price": g.base_price, "tpr_discount": g.tpr_discount,
            "catalogue": g.catalogue, "display": g.display}))
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(M.MMM_OUT / "mmm_contribution.csv", index=False)
    print(f"[m03] contribution rows: {len(d):,} | baseline share: {d.base_units.sum()/d.units.sum()*100:.1f}%")
    return d


if __name__ == "__main__":
    run()
