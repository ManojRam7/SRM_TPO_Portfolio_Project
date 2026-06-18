# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - INSIGHTS (Waterfall, Promo Efficiency, Price Ladder)
# MAGIC Databricks version of `src/s05_insights.py`. Aggregates the decomposition +
# MAGIC financials into the business-facing views that feed Power BI / Azure SQL.

# COMMAND ----------

from pyspark.sql import functions as F
PROC = "/mnt/srm/PROCESS"; OUT = "/mnt/srm/OUTPUT"
d = spark.read.parquet(f"{PROC}/MODEL/baseline_decomp")
fin = spark.read.parquet(f"{PROC}/ADS/ads_mapped").select("ppg", "cogs_per_unit").dropDuplicates()
d = d.join(F.broadcast(fin), "ppg", "left")

# COMMAND ----------

# DBTITLE 1,Waterfall (sales-driver decomposition)
drivers = ["base_units", "eff_price", "eff_feature", "eff_display", "eff_pantry", "eff_competition", "units"]
waterfall = d.groupBy("ppg").agg(*[F.round(F.sum(c), 0).alias(c) for c in drivers])
waterfall.write.mode("overwrite").parquet(f"{OUT}/WATERFALL")

# COMMAND ----------

# DBTITLE 1,Promo efficiency & ROI (per PPG)
d = (d.withColumn("own_incr", F.greatest(F.lit(0.0), F.col("eff_price") + F.col("eff_feature") + F.col("eff_display") + F.col("eff_pantry")))
       .withColumn("unit_margin", F.col("avg_price") - F.col("cogs_per_unit"))
       .withColumn("trade_spend", F.greatest(F.lit(0.0), (F.col("base_price") - F.col("avg_price")) * F.col("units")))
       .withColumn("incr_profit", F.col("own_incr") * F.col("unit_margin")))
eff = (d.groupBy("ppg", "manufacturer").agg(
            F.sum((F.col("discount_depth") > 0.01).cast("int")).alias("promo_weeks"),
            F.sum("own_incr").alias("incremental_units"),
            F.sum("trade_spend").alias("trade_spend"),
            F.sum("incr_profit").alias("incremental_profit"))
        .withColumn("roi", F.round(F.col("incremental_profit") / F.col("trade_spend"), 2)))
eff.write.mode("overwrite").parquet(f"{OUT}/PROMO_EFFICIENCY")
display(eff.orderBy(F.desc("roi")))
