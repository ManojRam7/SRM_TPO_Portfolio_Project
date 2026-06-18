"""
MMM STAGE 04 - ROI MAPPING & SCENARIO SIMULATION
  * ROI: trade expenditure (TE) from trade terms -> promotion ROI per Retailer x PG.
  * Scenarios: use model coefficients to simulate alternative promo plans and pick the
    spend allocation that maximises profit (the 'optimise marketing spend' deliverable).
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
    contrib = pd.read_csv(M.MMM_OUT / "mmm_contribution.csv")
    roi_terms = pd.read_csv(M.MMM_DATA / "roi_terms.csv").set_index("promo_group")

    # ---------- ROI mapping (per Retailer x PG) ----------
    c = contrib.merge(roi_terms.reset_index().drop(columns=["base_price"]), on="promo_group", how="left")
    c["TE_per_unit"] = (c.list_price * c.std_terms + c.settlement_discount + c.deferred_deal).round(3)
    c["own_incr"] = (c.eff_price + c.eff_catalogue + c.eff_display + c.eff_pantry).clip(lower=0)
    c["trade_spend"] = (((c.base_price - c.avg_price) + c.TE_per_unit) * c.units).clip(lower=0)
    c["incr_profit"] = c.own_incr * (c.avg_price - c.cogs_per_unit)
    c["is_promo"] = (c.tpr_discount > 1).astype(int)
    roi = c.groupby(["retailer", "promo_group"]).agg(
        promo_weeks=("is_promo", "sum"), incr_units=("own_incr", "sum"),
        trade_spend=("trade_spend", "sum"), incr_profit=("incr_profit", "sum")).reset_index()
    roi["roi"] = (roi.incr_profit / roi.trade_spend.replace(0, np.nan)).round(2)
    for col in ["incr_units", "trade_spend", "incr_profit"]:
        roi[col] = roi[col].round(0)
    roi = roi.sort_values("roi", ascending=False)
    roi.to_csv(M.MMM_OUT / "mmm_roi.csv", index=False)

    # ---------- Scenario simulation ----------
    cogs = roi_terms["cogs_per_unit"].to_dict(); te_map = (roi_terms.list_price * roi_terms.std_terms
        + roi_terms.settlement_discount + roi_terms.deferred_deal).to_dict()
    scen_rows = []
    for rpg, g in ads.groupby("retailer_pg"):
        if rpg not in coef.index:
            continue
        cf = coef.loc[rpg].to_dict(); g = g.reset_index(drop=True); n = len(g)
        pg = g.promo_group.iloc[0]; base = g.base_price.values
        cg = cogs.get(pg, base[0] * (1 - M.GMARGIN)); te = te_map.get(pg, 0.0)
        acts = [s for s in ACT_SLOTS if s in g.columns and cf.get(s, 0.0) != 0.0]
        A = (cf.get("intercept", 0.0) + cf.get("si_log", 0.0) * g.si_log.values
             + cf.get("trend", 0.0) * g.trend.values
             + sum(cf.get(s, 0.0) * g[s].fillna(g[s].median()).values for s in acts))
        e = cf.get("log_price", -2.0); bc = cf.get("catalogue", 0.0); bd = cf.get("display", 0.0)

        CAT_COST, DISP_COST = 0.08, 0.05    # catalogue/display cost as a fraction of base price/unit

        def sim(price, cat, disp):
            q = np.exp(A + e * np.log(price) + bc * cat + bd * disp)
            cat_cost = CAT_COST * base * cat * q
            disp_cost = DISP_COST * base * disp * q
            profit = (q * (price - cg - te) - cat_cost - disp_cost).sum()
            spend = (((base - price) + te) * q + cat_cost + disp_cost).sum()
            return profit, spend

        cur_p, cur_s = sim(g.avg_price.values, g.catalogue.values, g.display.values)
        promo = (g.tpr_discount.values > 1)
        scenarios = {
            "Current": (cur_p, cur_s),
            "Shallower TPR (cap 15%)": sim(np.maximum(base * 0.85, g.avg_price.values), g.catalogue.values, g.display.values),
            "Catalogue-led (10% + feature)": sim(np.where(promo, base * 0.90, base),
                                                 np.where(promo, 1, g.catalogue.values), g.display.values),
            "Trim deep promos (>=25%)": sim(np.where(g.tpr_discount.values >= 25, base, g.avg_price.values),
                                            g.catalogue.values, g.display.values),
        }
        best = max(scenarios, key=lambda k: scenarios[k][0])
        for name, (p, s) in scenarios.items():
            scen_rows.append({"retailer_pg": rpg, "retailer": g.retailer.iloc[0], "promo_group": pg,
                              "scenario": name, "profit": round(p, 0), "trade_spend": round(s, 0),
                              "vs_current_pct": round((p / cur_p - 1) * 100, 1) if cur_p else 0,
                              "recommended": int(name == best)})
    scen = pd.DataFrame(scen_rows)
    scen.to_csv(M.MMM_OUT / "mmm_scenarios.csv", index=False)

    cur = scen[scen.scenario == "Current"].profit.sum()
    best = scen[scen.recommended == 1].profit.sum()
    focal_pgs = [p[0] for p in M.PROMO_GROUPS if p[1] == M.FOCAL_MANUFACTURER]
    print(f"[m04] ROI ranked {len(roi)} retailer-PGs (avg {M.FOCAL_MANUFACTURER} ROI "
          f"{roi[roi.promo_group.isin(focal_pgs)].roi.mean():.2f})")
    print(f"[m04] scenario optimisation: best-plan profit uplift {(best/cur-1)*100:.1f}% vs current")
    return scen


if __name__ == "__main__":
    run()
