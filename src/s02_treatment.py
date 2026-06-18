"""
STAGE 02 - DATA TREATMENT & FEATURE ENGINEERING
===============================================
Build the modelling features and competitor (inter/intra) cross-price columns,
then create the model-ready ADS.

Features created (log-log demand model):
  log_units (DV), log_price (own-price elasticity), feat_acv, disp_acv,
  si_log (seasonality), trend, pantry_lag (lagged discount), holiday flags,
  INTRA/INTER competitor log-price slots, time-based train/test flag.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P


def run():
    ads = P.load("ads_mapped.csv")
    ads["week_ending"] = pd.to_datetime(ads["week_ending"])
    cmap = P.load("competitor_map.csv")

    # ---- core transforms ----
    ads["log_units"] = np.log1p(ads["units"])
    ads["log_price"] = np.log(ads["avg_price"])
    ads["si_log"] = np.log(ads["si"]) if "si" in ads.columns else 0.0
    # recompute SI if not carried (raw didn't keep it) -> approximate from base seasonality
    if "si" not in ads.columns:
        ads["si_log"] = 0.0
    ads["trend"] = ads.groupby("cta_ppg").cumcount()
    ads["discount_depth"] = (ads["base_price"] - ads["avg_price"]) / ads["base_price"]
    ads["discount_depth"] = ads["discount_depth"].clip(lower=0)
    ads = ads.sort_values(["cta_ppg", "week_ending"]).reset_index(drop=True)
    ads["pantry_lag"] = ads.groupby("cta_ppg")["discount_depth"].shift(1).fillna(0) * 100

    # holiday flags from the calendar
    we = ads["week_ending"]
    ads["Halloween"] = ((we.dt.month == 10) & (we.dt.day >= 18)).astype(int)
    ads["Christmas"] = ((we.dt.month == 12) & (we.dt.day.between(10, 26))).astype(int)
    ads["Easter"] = ((we.dt.month == 4) & (we.dt.day <= 18)).astype(int)
    ads["Valentine"] = ((we.dt.month == 2) & (we.dt.day.between(6, 16))).astype(int)
    ads["SummerDip"] = (we.dt.month.isin([7, 8])).astype(int)

    # si_log: rebuild from seasonality proxy so the feature exists for modelling
    phase = 2 * np.pi * (we.dt.isocalendar().week.astype(int) / 52.0)
    cat = ads["category"]
    si = np.where(cat == "Chocolate", 1 + 0.18*np.cos(phase) + 0.10*ads.Christmas + 0.06*ads.Easter,
         np.where(cat == "Fruity",   1 + 0.20*ads.Halloween + 0.10*ads.Easter + 0.05*np.sin(phase),
                                      1 + 0.05*np.sin(phase)))
    ads["si_log"] = np.log(np.clip(si, 0.6, 1.8))

    # ---- competitor cross-price slots ----
    price_wide = ads.pivot_table(index=["cta", "week_ending"], columns="ppg",
                                 values="avg_price", aggfunc="first")
    for slot in P.COMP_SLOTS:
        ads[slot] = np.nan
    # assign actors into INTRA_/INTER_ slots per target PPG
    for ppg, grp in cmap.groupby("ppg"):
        intra = grp[grp.relationship == "INTRA"]["comp_ppg"].tolist()
        inter = grp[grp.relationship == "INTER"]["comp_ppg"].tolist()
        mask = ads["ppg"] == ppg
        if not mask.any():
            continue
        sub_idx = ads.index[mask]
        keys = list(zip(ads.loc[sub_idx, "cta"], ads.loc[sub_idx, "week_ending"]))
        for slot, actor in list(zip(P.INTRA_SLOTS, intra)) + list(zip(P.INTER_SLOTS, inter)):
            if actor in price_wide.columns:
                ads.loc[sub_idx, slot] = np.log(price_wide.loc[keys, actor].values)
    # drop fully-empty slot columns to keep ADS clean
    for slot in P.COMP_SLOTS:
        if ads[slot].isna().all():
            ads.drop(columns=[slot], inplace=True)

    # ---- time-based train/test flag (last TEST_FRACTION of weeks per CTA-PPG) ----
    cutoff_rank = int(C.N_WEEKS * (1 - C.TEST_FRACTION))
    ads["test_flag"] = (ads.groupby("cta_ppg").cumcount() >= cutoff_rank).astype(int)

    P.save(ads, "model_ads.csv")
    present = [s for s in P.COMP_SLOTS if s in ads.columns]
    print(f"[s02] model ADS rows: {len(ads):,} | competitor slots present: {len(present)} | "
          f"train rows: {(ads.test_flag==0).sum():,} test rows: {(ads.test_flag==1).sum():,}")
    return ads


if __name__ == "__main__":
    run()
