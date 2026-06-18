"""
MMM STAGE 01 - DATA PREP & FEATURE ENGINEERING
Builds log-log features and competitor cross-price (acting-item) columns, with
acting items selected by correlation with the target's volume.
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

ACT_SLOTS = ["ACT_1_logprice", "ACT_2_logprice", "ACT_3_logprice"]


def run():
    raw = pd.read_csv(M.MMM_DATA / "mmm_raw.csv")
    raw["week_ending"] = pd.to_datetime(raw["week_ending"])
    # QC
    assert raw.duplicated(["retailer", "promo_group", "week_ending"]).sum() == 0
    assert raw[["units", "avg_price"]].le(0).sum().sum() == 0

    raw["retailer_pg"] = raw["retailer"] + "|" + raw["promo_group"]
    raw["log_qty"] = np.log1p(raw["units"])
    raw["log_price"] = np.log(raw["avg_price"])
    raw["si_log"] = np.log(raw["si"].clip(lower=0.5))
    raw = raw.sort_values(["retailer_pg", "week_ending"]).reset_index(drop=True)
    raw["trend"] = raw.groupby("retailer_pg").cumcount()
    # tpr_lag1 already in % from raw 'tpr_discount'; rebuild lag here for safety
    raw["tpr_lag1"] = raw.groupby("retailer_pg")["tpr_discount"].shift(1).fillna(0)

    # acting-item (cross-price) selection per promo group (pooled), attach per retailer-week
    price_wide = raw.pivot_table(index=["retailer", "week_ending"], columns="promo_group",
                                 values="avg_price", aggfunc="first")
    for s in ACT_SLOTS:
        raw[s] = np.nan
    all_pg = raw.promo_group.unique().tolist()
    selection = {}
    for tgt in all_pg:
        cands = [c for c in all_pg if c != tgt]
        actors = MC.acting_item_filter(raw, tgt, cands)[:3]
        selection[tgt] = actors
        mask = raw.promo_group == tgt
        keys = list(zip(raw.loc[mask, "retailer"], raw.loc[mask, "week_ending"]))
        for slot, act in zip(ACT_SLOTS, actors):
            if act in price_wide.columns:
                raw.loc[mask, slot] = np.log(price_wide.loc[keys, act].values)
    for s in ACT_SLOTS:
        if raw[s].isna().all():
            raw.drop(columns=[s], inplace=True)

    raw.to_csv(M.MMM_OUT / "mmm_model_ads.csv", index=False)
    pd.DataFrame([(k, ";".join(v)) for k, v in selection.items()],
                 columns=["promo_group", "acting_items"]).to_csv(M.MMM_OUT / "mmm_acting_items.csv", index=False)
    print(f"[m01] MMM model ADS rows: {len(raw):,} | acting-item slots: {sum(s in raw.columns for s in ACT_SLOTS)}")
    return raw


if __name__ == "__main__":
    run()
