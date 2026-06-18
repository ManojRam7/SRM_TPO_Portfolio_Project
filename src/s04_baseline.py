"""
STAGE 04 - BASELINE & INCREMENTAL DECOMPOSITION
===============================================
Use the fitted coefficients to compute, per CTA-PPG-week:
  * baseline units  = expected demand with NO own promotion (price at base,
    feature/display/pantry = 0), keeping seasonality, trend and holidays.
  * incremental     = actual - baseline, decomposed (log-additive, exact) into
    price, feature, display, pantry and competition effects.

Method: predict twice per week, once as observed and once with the promo levers
zeroed, and exp() both back to units.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P


def run():
    ads = P.load("model_ads.csv")
    coef = P.load("model_coefficients.csv").set_index("cta_ppg")
    coef_cols = [c for c in coef.columns]

    rows = []
    for cp, g in ads.groupby("cta_ppg"):
        if cp not in coef.index:
            continue
        c = coef.loc[cp].to_dict()
        g = g.reset_index(drop=True)
        log_price = np.log(g.avg_price.values)
        log_base = np.log(g.base_price.values)
        n = len(g)

        def term(name, vec):
            return c.get(name, 0.0) * vec if name in c else np.zeros(n)

        # competitor slots: use group-median log price as the 'base' competitor level
        comp_present = [s for s in P.COMP_SLOTS if s in g.columns and c.get(s, 0.0) != 0.0]
        comp_actual = np.zeros(n)
        comp_base = np.zeros(n)
        for s in comp_present:
            v = g[s].fillna(g[s].median()).values
            comp_actual += c.get(s, 0.0) * v
            comp_base += c.get(s, 0.0) * np.nanmedian(v)

        holidays = [h for h in P.HOLIDAYS if h in g.columns]
        k0 = (c.get("intercept", 0.0)
              + term("log_price", log_base)
              + term("si_log", g.si_log.values)
              + term("trend", g.trend.values)
              + sum(term(h, g[h].values) for h in holidays)
              + comp_base)

        base_units = np.exp(k0)
        cum = k0.copy()
        contribs = {}
        for name, add in [
            ("price", term("log_price", log_price) - term("log_price", log_base)),
            ("feature", term("feat_acv", g.feat_acv.values)),
            ("display", term("disp_acv", g.disp_acv.values)),
            ("pantry", term("pantry_lag", g.pantry_lag.values)),
            ("competition", comp_actual - comp_base),
        ]:
            new = cum + add
            contribs[name] = np.exp(new) - np.exp(cum)
            cum = new
        pred_units = np.exp(cum)

        out = pd.DataFrame({
            "cta": g.cta, "ppg": g.ppg, "cta_ppg": cp, "week_ending": g.week_ending,
            "units": g.units, "pred_units": pred_units.round(1), "base_units": base_units.round(1),
            "incr_units": (pred_units - base_units).round(1),
            "eff_price": contribs["price"].round(1), "eff_feature": contribs["feature"].round(1),
            "eff_display": contribs["display"].round(1), "eff_pantry": contribs["pantry"].round(1),
            "eff_competition": contribs["competition"].round(1),
            "avg_price": g.avg_price, "base_price": g.base_price, "discount_depth": g.discount_depth,
        })
        rows.append(out)

    decomp = pd.concat(rows, ignore_index=True)
    P.save(decomp, "baseline_decomp.csv")
    tot = decomp.units.sum()
    base = decomp.base_units.sum()
    print(f"[s04] decomposed {len(decomp):,} rows | baseline share: {base/tot*100:.1f}% | "
          f"incremental share: {(1-base/tot)*100:.1f}%")
    return decomp


if __name__ == "__main__":
    run()
