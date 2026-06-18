# Databricks notebook source
# MAGIC %md
# MAGIC # 99 - QUALITY CHECKS
# MAGIC Model QC gates. Runs after each stage in the Data Factory pipeline; a failed
# MAGIC assert stops the run before results reach the reporting layer.

# COMMAND ----------

from pyspark.sql import functions as F
PROC = "/mnt/srm/PROCESS"
summ = spark.read.parquet(f"{PROC}/MODEL/model_summary")

# COMMAND ----------

# DBTITLE 1,Model QC gates
stats = summ.agg(
    F.count("*").alias("models"),
    F.round(F.avg("price_sign_valid") * 100, 1).alias("valid_sign_pct"),
    F.round(F.expr("percentile(r2, 0.5)"), 3).alias("median_r2"),
    F.round(F.expr("percentile(test_mape, 0.5)"), 2).alias("median_mape"),
).collect()[0]
print(stats)
assert stats["valid_sign_pct"] >= 95, "too many counter-intuitive price signs"
assert stats["median_r2"] >= 0.80, "median R2 below threshold"
assert stats["median_mape"] <= 15, "median test MAPE above threshold"
print("ALL MODEL QC GATES PASSED")

# COMMAND ----------

# MAGIC %md Unit tests: VIF flags collinear pairs, percentile
# MAGIC bounds (5-95), discount-type classification, competitor mapping completeness.
# MAGIC See `tests/` in the repo root for the runnable pytest equivalents.
