"""
MMM STAGE 00 - SAMPLE DATA  (Retailer x Promo Group x Week)
Weekly retailer-level chocolate sales for two fictional grocery chains. Synthetic,
with known coefficients so the model's recovery can be measured.
Outputs: mmm/data/{mmm_raw.csv, roi_terms.csv, ground_truth_mmm.csv, ground_truth_cross_mmm.csv}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
import config as C
import config_mmm as M


def generate():
    rng = np.random.default_rng(C.RANDOM_SEED)
    weeks = pd.date_range(C.START_DATE, periods=C.N_WEEKS, freq="7D")
    woy = weeks.isocalendar().week.astype(int).values
    pg = pd.DataFrame(M.PROMO_GROUPS, columns=M.PG_COLS)

    rows = []
    for ret in M.RETAILERS:
        rmult = rng.uniform(0.9, 1.3)
        for _, p in pg.iterrows():
            base = p.base_price
            promo = rng.random(C.N_WEEKS) < 0.30
            tpr = np.where(promo, rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.35], C.N_WEEKS), 0.0)
            price = np.round(base * (1 - tpr), 2)
            catalogue = ((rng.random(C.N_WEEKS) < M.CATALOGUE_RATE) & promo).astype(int)
            display = ((rng.random(C.N_WEEKS) < M.DISPLAY_RATE) & promo).astype(int)
            # southern-hemisphere chocolate seasonality: Easter and Christmas peaks, softer in the hot Dec-Feb summer
            si = 1.0 + 0.16 * np.cos(2 * np.pi * woy / 52.0) + 0.10 * ((weeks.month == 4)).astype(int)
            sub = pd.DataFrame({
                "retailer": ret, "promo_group": p.promo_group, "manufacturer": p.manufacturer,
                "brand": p.brand, "segment": p.segment, "week_ending": weeks,
                "t": np.arange(C.N_WEEKS), "base_price": base, "avg_price": price,
                "tpr_discount": (tpr * 100).round(1), "catalogue": catalogue, "display": display,
                "si": si.round(3), "_rmult": rmult, "_base_qty": p.base_qty,
                "_e": p.e_price, "_bcat": p.b_catalogue, "_bdisp": p.b_display,
            })
            rows.append(sub)
    panel = pd.concat(rows, ignore_index=True).sort_values(["retailer", "promo_group", "t"]).reset_index(drop=True)
    panel["tpr_lag1"] = panel.groupby(["retailer", "promo_group"])["tpr_discount"].shift(1).fillna(0)

    price_wide = panel.pivot_table(index=["retailer", "week_ending"], columns="promo_group",
                                   values="avg_price", aggfunc="first")
    base_wide = panel.pivot_table(index=["retailer", "week_ending"], columns="promo_group",
                                  values="base_price", aggfunc="first")
    cross = {}
    for t, a, b in M.CROSS_TRUTH:
        cross.setdefault(t, []).append((a, b))

    logp = np.log(panel.avg_price.values); logb = np.log(panel.base_price.values)
    mu = (np.log(panel._base_qty.values * panel._rmult.values)
          + panel._e.values * (logp - logb)
          + panel._bcat.values * panel.catalogue.values
          + panel._bdisp.values * panel.display.values
          + 1.0 * np.log(panel.si.values)
          + M.TREND_BETA * panel.t.values
          + M.PANTRY_BETA * panel.tpr_lag1.values)
    for tgt, acts in cross.items():
        idx = panel.index[panel.promo_group.values == tgt]
        if len(idx) == 0:
            continue
        ck = list(zip(panel.loc[idx, "retailer"], panel.loc[idx, "week_ending"]))
        for a, b in acts:
            if a in price_wide.columns:
                mu[idx] += b * (np.log(price_wide.loc[ck, a].values) - np.log(base_wide.loc[ck, a].values))
    qty = np.exp(mu + rng.normal(0, M.NOISE_SD, len(panel)))
    panel["units"] = np.maximum(np.round(qty), 1).astype(int)
    panel["value_sales"] = (panel.units * panel.avg_price).round(2)

    raw = panel[["retailer", "promo_group", "manufacturer", "brand", "segment", "week_ending",
                 "units", "value_sales", "avg_price", "base_price", "tpr_discount",
                 "catalogue", "display", "si"]]
    raw.to_csv(M.MMM_DATA / "mmm_raw.csv", index=False)

    # ROI trade terms
    roi = pg[["promo_group", "base_price"]].copy()
    roi["list_price"] = (roi.base_price * M.LIST_UPLIFT).round(2)
    roi["cogs_per_unit"] = (roi.base_price * (1 - M.GMARGIN)).round(3)
    roi["std_terms"] = M.TRADE_TERMS["std_terms"]
    roi["settlement_discount"] = M.TRADE_TERMS["settlement_discount"]
    roi["deferred_deal"] = M.TRADE_TERMS["deferred_deal"]
    roi.to_csv(M.MMM_DATA / "roi_terms.csv", index=False)

    pg[["promo_group", "e_price", "b_catalogue", "b_display"]].rename(
        columns={"e_price": "true_e_price", "b_catalogue": "true_b_catalogue",
                 "b_display": "true_b_display"}).to_csv(M.MMM_DATA / "ground_truth_mmm.csv", index=False)
    pd.DataFrame(M.CROSS_TRUTH, columns=["promo_group", "acting_item", "true_beta"]).to_csv(
        M.MMM_DATA / "ground_truth_cross_mmm.csv", index=False)

    print(f"[m00] MMM raw rows: {len(raw):,} ({raw.promo_group.nunique()} PGs x {len(M.RETAILERS)} retailers x {C.N_WEEKS} wks)")
    return raw


if __name__ == "__main__":
    generate()
