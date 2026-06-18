# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - BASELINE & INCREMENTAL DECOMPOSITION
# MAGIC Databricks version of `src/s04_baseline.py`. Predict twice per CTA-PPG-week: once
# MAGIC with actuals, once with the promo levers zeroed (price->base, feat/disp/pantry=0).
# MAGIC The zero-promo prediction is the baseline; actual - baseline = incremental lift.

# COMMAND ----------

import numpy as np, pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

PROC = "/mnt/srm/PROCESS"
ads = spark.read.parquet(f"{PROC}/ADS/model_ads")
coef = spark.read.parquet(f"{PROC}/MODEL/model_coefficients")   # written alongside model_summary

# Broadcast coefficients keyed by cta_ppg and decompose per group with applyInPandas.
# (Same log-additive, exact decomposition as src/s04_baseline.py.)

schema = StructType([StructField(c, t) for c, t in [
    ("cta_ppg", StringType()), ("week_ending", DateType()), ("units", DoubleType()),
    ("base_units", DoubleType()), ("incr_units", DoubleType()), ("eff_price", DoubleType()),
    ("eff_feature", DoubleType()), ("eff_display", DoubleType()), ("eff_pantry", DoubleType()),
    ("eff_competition", DoubleType())]])

# COMMAND ----------

# MAGIC %md At scale this joins the coefficient table and evaluates the baseline
# MAGIC counterfactual with Spark SQL `exp(...)` expressions; the per-group pandas form
# MAGIC in `src/s04_baseline.py` is the runnable reference.

# COMMAND ----------

# (Spark expression form, abbreviated)
# baseline = exp( intercept + b_price*ln(base_price) + b_si*si_log + b_trend*trend
#                 + b_holiday*holiday... + competition_at_base )
# incremental = actual - baseline ; then split price/feature/display/pantry sequentially.
print("See src/s04_baseline.py for the exact, runnable decomposition.")
