"""
STAGE 00 - SAMPLE DATA GENERATION
=================================
Creates realistic retail scanner data at CTA x PPG x Week with a known log-log
data-generating process. The *true* elasticities are saved so later stages can
prove the model recovers them.

Real scanner and loyalty-card data is licensed and cannot be shared, so the whole
project runs on this synthetic panel. Brands and manufacturers are fictional.

Outputs (./data):
  scanner_raw.csv              raw weekly scanner facts (the 'Extract' in ELT)
  financials.csv               list price, gMargin, COGS per PPG
  competitor_map.csv           inter/intra competitor mapping per PPG (+ distractors)
  trade_rate.csv               trade-spend rate per PPG
  ground_truth_elasticities.csv   true own-price/feature/display betas
  ground_truth_cross.csv          true cross-price betas
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C


def _calendar():
    weeks = pd.date_range(C.START_DATE, periods=C.N_WEEKS, freq="7D")
    cal = pd.DataFrame({"week_ending": weeks})
    cal["t"] = np.arange(C.N_WEEKS)
    cal["month"] = cal["week_ending"].dt.month
    cal["woy"] = cal["week_ending"].dt.isocalendar().week.astype(int)
    # Holiday flags (approximate US calendar)
    cal["Halloween"] = ((cal.month == 10) & (cal.week_ending.dt.day >= 18)).astype(int)
    cal["Christmas"] = ((cal.month == 12) & (cal.week_ending.dt.day >= 10) & (cal.week_ending.dt.day <= 26)).astype(int)
    cal["Easter"] = ((cal.month == 4) & (cal.week_ending.dt.day <= 18)).astype(int)
    cal["Valentine"] = ((cal.month == 2) & (cal.week_ending.dt.day >= 6) & (cal.week_ending.dt.day <= 16)).astype(int)
    cal["SummerDip"] = (cal.month.isin([7, 8])).astype(int)
    return cal


def _seasonality_index(cal, category):
    """Smooth seasonal multiplier (>0). log(SI) is a model feature."""
    phase = 2 * np.pi * (cal["woy"] / 52.0)
    if category == "Chocolate":
        si = 1.0 + 0.18 * np.cos(phase) + 0.10 * cal["Christmas"] + 0.06 * cal["Easter"]
    elif category == "Fruity":
        si = 1.0 + 0.20 * cal["Halloween"] + 0.10 * cal["Easter"] + 0.05 * np.sin(phase)
    else:  # Gum
        si = 1.0 + 0.05 * np.sin(phase) + 0.04 * (cal["month"].isin([1, 6])).astype(int)
    return np.clip(si, 0.6, 1.8)


def generate():
    rng = np.random.default_rng(C.RANDOM_SEED)
    cal = _calendar()
    ppg_df = pd.DataFrame(C.PPGS, columns=C.PPG_COLUMNS)

    # ---- 1. Build the price/promo panel (CTA x PPG x Week) ----
    rows = []
    for cta in C.CTAS:
        cta_mult = rng.uniform(0.8, 1.2)             # market size multiplier
        for _, p in ppg_df.iterrows():
            base = p.base_price
            # constant base price per PPG (base-price changes would normally get
            # their own base-price elasticity model; kept constant for clean recovery)
            bp = np.full(C.N_WEEKS, base)
            # promo schedule: ~32% of weeks promoted
            promo = rng.random(C.N_WEEKS) < 0.32
            depth = np.where(promo, rng.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.35], C.N_WEEKS), 0.0)
            price = np.round(bp * (1 - depth), 2)
            feat = np.where(promo, rng.uniform(0.15, 0.85, C.N_WEEKS), rng.uniform(0.0, 0.05, C.N_WEEKS))
            disp = np.where(promo, rng.uniform(0.10, 0.75, C.N_WEEKS), rng.uniform(0.0, 0.04, C.N_WEEKS))
            si = _seasonality_index(cal, p.category).values
            sub = pd.DataFrame({
                "cta": cta, "ppg": p.ppg, "manufacturer": p.manufacturer, "brand": p.brand,
                "category": p.category, "segment": p.segment,
                "week_ending": cal.week_ending.values, "t": cal.t.values,
                "base_price": bp, "avg_price": price, "discount_depth": depth,
                "feat_acv": feat.round(3), "disp_acv": disp.round(3), "si": si.round(3),
                "_cta_mult": cta_mult, "_base_units": p.base_units,
                "_beta_price": p.beta_price, "_beta_feat": p.beta_feat, "_beta_disp": p.beta_disp,
            })
            for h in C.HOLIDAY_UPLIFT:
                sub[h] = cal[h].values
            rows.append(sub)
    panel = pd.concat(rows, ignore_index=True)

    # ---- 2. Wide competitor price lookup per (cta, week) for cross-price terms ----
    price_wide = panel.pivot_table(index=["cta", "week_ending"], columns="ppg",
                                   values="avg_price", aggfunc="first")
    base_wide = panel.pivot_table(index=["cta", "week_ending"], columns="ppg",
                                  values="base_price", aggfunc="first")
    cross_by_target = {}
    for tgt, act, beta in C.CROSS_PRICE_TRUTH:
        cross_by_target.setdefault(tgt, []).append((act, beta))

    # ---- 3. Generate units from the log-log DGP ----
    panel = panel.sort_values(["cta", "ppg", "t"]).reset_index(drop=True)
    panel["disc_lag1"] = panel.groupby(["cta", "ppg"])["discount_depth"].shift(1).fillna(0.0) * 100

    key = list(zip(panel.cta, panel.week_ending))
    log_price = np.log(panel.avg_price.values)
    log_base = np.log(panel.base_price.values)
    mu = (np.log(panel._base_units.values * panel._cta_mult.values)
          + panel._beta_price.values * (log_price - log_base)
          + panel._beta_feat.values * panel.feat_acv.values
          + panel._beta_disp.values * panel.disp_acv.values
          + 1.0 * np.log(panel.si.values)                       # true SI elasticity = 1.0
          + C.TREND_BETA * panel.t.values
          + C.PANTRY_BETA * panel.disc_lag1.values)
    for h, up in C.HOLIDAY_UPLIFT.items():
        mu += up * panel[h].values
    # cross-price contributions
    for tgt, actors in cross_by_target.items():
        m = (panel.ppg.values == tgt)
        if not m.any():
            continue
        idx = panel.index[m]
        ck = list(zip(panel.loc[idx, "cta"], panel.loc[idx, "week_ending"]))
        for act, beta in actors:
            if act in price_wide.columns:
                cp = price_wide.loc[ck, act].values
                cbp = base_wide.loc[ck, act].values
                mu[idx] += beta * (np.log(cp) - np.log(cbp))
    noise = rng.normal(0, C.NOISE_SD, len(panel))
    units = np.exp(mu + noise)
    panel["units"] = np.maximum(np.round(units), 1).astype(int)
    panel["dollar_sales"] = (panel.units * panel.avg_price).round(2)

    # ---- 4. Write the raw scanner extract ----
    raw_cols = ["cta", "ppg", "manufacturer", "brand", "category", "segment",
                "week_ending", "units", "dollar_sales", "avg_price", "base_price",
                "feat_acv", "disp_acv"]
    raw = panel[raw_cols].copy()
    raw.to_csv(C.DATA / "scanner_raw.csv", index=False)

    # ---- 5. Financials ----
    fin = ppg_df[["ppg", "base_price"]].copy()
    fin["list_price"] = (fin.base_price * C.LIST_PRICE_UPLIFT).round(2)   # wholesale ref (display only)
    fin["gmargin"] = C.GMARGIN
    # COGS as a fraction of the base SHELF price so base margin == gMargin (42%);
    # this keeps promo economics coherent (a discount < gMargin can still profit).
    fin["cogs_per_unit"] = (fin.base_price * (1 - C.GMARGIN)).round(3)
    fin.drop(columns=["base_price"]).to_csv(C.DATA / "financials.csv", index=False)

    tr = ppg_df[["ppg"]].copy()
    tr["trade_rate"] = C.TRADE_RATE
    tr.to_csv(C.DATA / "trade_rate.csv", index=False)

    # ---- 6. Competitor mapping (true actors + 1 distractor each) ----
    all_ppgs = ppg_df.ppg.tolist()
    man = dict(zip(ppg_df.ppg, ppg_df.manufacturer))
    map_rows = []
    for tgt, actors in cross_by_target.items():
        for act, _ in actors:
            rel = "INTRA" if man[act] == man.get(tgt) else "INTER"
            map_rows.append((tgt, act, rel, "true"))
        # add a distractor actor (no real relationship) for selection to reject
        cand = [q for q in all_ppgs if q != tgt and q not in [a for a, _ in actors]]
        dist = rng.choice(cand)
        rel = "INTRA" if man[dist] == man.get(tgt) else "INTER"
        map_rows.append((tgt, dist, rel, "distractor"))
    pd.DataFrame(map_rows, columns=["ppg", "comp_ppg", "relationship", "_origin"]).to_csv(
        C.DATA / "competitor_map.csv", index=False)

    # ---- 7. Ground truth ----
    ppg_df[["ppg", "beta_price", "beta_feat", "beta_disp"]].rename(
        columns={"beta_price": "true_beta_price", "beta_feat": "true_beta_feat",
                 "beta_disp": "true_beta_disp"}).to_csv(
        C.DATA / "ground_truth_elasticities.csv", index=False)
    pd.DataFrame(C.CROSS_PRICE_TRUTH, columns=["ppg", "comp_ppg", "true_beta_cross"]).to_csv(
        C.DATA / "ground_truth_cross.csv", index=False)

    print(f"[s00] scanner_raw rows: {len(raw):,}  ({raw.ppg.nunique()} PPGs x {len(C.CTAS)} CTAs x {C.N_WEEKS} weeks)")
    print(f"[s00] avg weekly units: {raw.units.mean():,.0f}   total $ sales: {raw.dollar_sales.sum():,.0f}")
    print(f"[s00] wrote raw, financials, trade_rate, competitor_map, ground_truth to {C.DATA}")
    return raw


if __name__ == "__main__":
    generate()
