"""
STAGE 01 - UNION & MAPPING
==========================
Append/union the raw extracts, run raw-data quality checks, and map dimension
keys + financials onto the Analytical Data Set (ADS). Checks: row count,
duplicate keys, critical nulls, non-positive units or prices.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P


def run():
    raw = P.load("scanner_raw.csv")
    fin = P.load("financials.csv")

    # ---- Raw-data QC ----
    qc = []
    qc.append(("row_count", len(raw), len(raw) > 0))
    key_dupes = raw.duplicated(["cta", "ppg", "week_ending"]).sum()
    qc.append(("duplicate_keys", key_dupes, key_dupes == 0))
    nulls = int(raw[["units", "avg_price", "base_price"]].isna().sum().sum())
    qc.append(("critical_nulls", nulls, nulls == 0))
    neg = int((raw[["units", "avg_price"]] <= 0).sum().sum())
    qc.append(("non_positive_units_price", neg, neg == 0))
    qc_df = pd.DataFrame(qc, columns=["check", "value", "passed"])
    P.save(qc_df, "qc_raw_data.csv")

    # ---- Dimension key + financial mapping ----
    ads = raw.merge(fin, on="ppg", how="left")
    ads["cta_ppg"] = ads["cta"] + "|" + ads["ppg"]          # the CTA-PPG modelling key
    ads["week_ending"] = pd.to_datetime(ads["week_ending"])
    ads = ads.sort_values(["cta", "ppg", "week_ending"]).reset_index(drop=True)

    P.save(ads, "ads_mapped.csv")
    print(f"[s01] union+map ADS rows: {len(ads):,} | CTA-PPG combos: {ads.cta_ppg.nunique()} "
          f"| QC passed: {qc_df.passed.all()}")
    return ads


if __name__ == "__main__":
    run()
