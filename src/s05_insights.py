"""
STAGE 05 - INSIGHTS GENERATION (waterfall, promo efficiency, price ladder, ROI)
===============================================================================
Turns the decomposition + financials into the business-facing views:
  * waterfall_summary.csv : sales-driver decomposition (base + levers)
  * promo_efficiency.csv  : per-PPG incremental volume, trade spend, ROI ranking
  * price_ladder.csv      : price vs market share by category
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P


def run():
    d = P.load("baseline_decomp.csv")
    fin = P.load("financials.csv")
    ads = P.load("ads_mapped.csv")[["cta", "ppg", "week_ending", "manufacturer", "brand", "category"]]
    ads["week_ending"] = pd.to_datetime(ads["week_ending"])
    d["week_ending"] = pd.to_datetime(d["week_ending"])
    d = d.merge(fin, on="ppg", how="left").merge(ads, on=["cta", "ppg", "week_ending"], how="left")

    # ---------- 1. Waterfall (sales-driver decomposition) ----------
    drivers = ["base_units", "eff_price", "eff_feature", "eff_display", "eff_pantry", "eff_competition"]
    wf = d.groupby("ppg")[drivers + ["units"]].sum().reset_index()
    total = d[drivers + ["units"]].sum()
    total["ppg"] = "TOTAL"
    wf = pd.concat([wf, pd.DataFrame([total])], ignore_index=True)
    for c in drivers + ["units"]:
        wf[c] = wf[c].round(0)
    wf = wf.rename(columns={"base_units": "Base", "eff_price": "Price", "eff_feature": "Feature",
                            "eff_display": "Display", "eff_pantry": "Pantry",
                            "eff_competition": "Competition", "units": "Actual"})
    P.save(wf, "waterfall_summary.csv")

    # ---------- 2. Promo efficiency & ROI (per PPG) ----------
    d["own_incr"] = (d.eff_price + d.eff_feature + d.eff_display + d.eff_pantry).clip(lower=0)
    d["unit_margin"] = d.avg_price - d.cogs_per_unit          # gross profit per unit at promo price
    d["trade_spend"] = ((d.base_price - d.avg_price) * d.units).clip(lower=0)
    # promotion ROI = gross profit earned on INCREMENTAL units / trade spend (standard CPG metric)
    d["incr_profit"] = d.own_incr * d.unit_margin
    d["is_promo"] = (d.discount_depth > 0.01).astype(int)

    eff = d.groupby(["ppg", "manufacturer", "brand", "category"]).agg(
        promo_weeks=("is_promo", "sum"),
        total_units=("units", "sum"),
        incremental_units=("own_incr", "sum"),
        trade_spend=("trade_spend", "sum"),
        incremental_profit=("incr_profit", "sum"),
    ).reset_index()
    eff["roi"] = (eff.incremental_profit / eff.trade_spend.replace(0, np.nan)).round(2)
    eff["lift_pct"] = (eff.incremental_units / (eff.total_units - eff.incremental_units) * 100).round(1)
    eff["rev_per_trade_dollar"] = ((eff.incremental_units *
                                    d.groupby("ppg").avg_price.mean().reindex(eff.ppg).values)
                                   / eff.trade_spend.replace(0, np.nan)).round(2)
    eff = eff.sort_values("roi", ascending=False)
    for c in ["total_units", "incremental_units", "trade_spend", "incremental_profit"]:
        eff[c] = eff[c].round(0)
    P.save(eff, "promo_efficiency.csv")

    # ---------- 3. Price ladder (price vs share by category) ----------
    pl = d.groupby(["category", "ppg", "manufacturer", "brand"]).agg(
        avg_price=("avg_price", "mean"), avg_base_price=("base_price", "mean"),
        avg_discount=("discount_depth", "mean"), units=("units", "sum")).reset_index()
    pl["market_share_pct"] = (pl.units / pl.groupby("category").units.transform("sum") * 100).round(1)
    pl["avg_price"] = pl.avg_price.round(2)
    pl["avg_discount_pct"] = (pl.avg_discount * 100).round(1)
    pl = pl.sort_values(["category", "avg_price"])
    P.save(pl[["category", "ppg", "brand", "manufacturer", "avg_price", "avg_base_price",
               "avg_discount_pct", "units", "market_share_pct"]], "price_ladder.csv")

    focal = eff[eff.manufacturer == C.FOCAL_MANUFACTURER]
    print(f"[s05] waterfall + promo efficiency + price ladder written")
    print(f"[s05] {C.FOCAL_MANUFACTURER} PPGs avg promo ROI: {focal.roi.mean():.2f} | best: {focal.iloc[0].ppg} "
          f"(ROI {focal.iloc[0].roi:.2f})")
    return eff


if __name__ == "__main__":
    run()
