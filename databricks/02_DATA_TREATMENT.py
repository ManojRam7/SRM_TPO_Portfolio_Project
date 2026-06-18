# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - DATA TREATMENT & FEATURE ENGINEERING
# MAGIC Databricks version of `src/s02_treatment.py`. Builds the log-log modelling
# MAGIC features and the inter/intra competitor cross-price columns using Spark
# MAGIC window functions and a self-join.

# COMMAND ----------

from pyspark.sql import functions as F, Window

PROC = "/mnt/srm/PROCESS"
RAW = "/mnt/srm/RAW"
ads = spark.read.parquet(f"{PROC}/ADS/ads_mapped")
cmap = spark.read.option("header", True).csv(f"{RAW}/FILES/competitor_map.csv")

# COMMAND ----------

# DBTITLE 1,Core transforms (log-log features, discount, pantry lag, holidays)
w = Window.partitionBy("cta_ppg").orderBy("week_ending")
ads = (ads
       .withColumn("log_units", F.log1p("units"))
       .withColumn("log_price", F.log("avg_price"))
       .withColumn("discount_depth", F.greatest(F.lit(0.0), (F.col("base_price") - F.col("avg_price")) / F.col("base_price")))
       .withColumn("trend", F.row_number().over(w) - 1)
       .withColumn("pantry_lag", F.coalesce(F.lag("discount_depth").over(w), F.lit(0.0)) * 100)
       .withColumn("mth", F.month("week_ending")).withColumn("dy", F.dayofmonth("week_ending"))
       .withColumn("Halloween", ((F.col("mth") == 10) & (F.col("dy") >= 18)).cast("int"))
       .withColumn("Christmas", ((F.col("mth") == 12) & (F.col("dy").between(10, 26))).cast("int"))
       .withColumn("Easter", ((F.col("mth") == 4) & (F.col("dy") <= 18)).cast("int"))
       .withColumn("Valentine", ((F.col("mth") == 2) & (F.col("dy").between(6, 16))).cast("int"))
       .withColumn("SummerDip", F.col("mth").isin(7, 8).cast("int"))
       .withColumn("woy", F.weekofyear("week_ending"))
       )
# seasonality index (category specific) -> si_log
phase = 2 * 3.141592653589793 * (F.col("woy") / 52.0)
si = (F.when(F.col("category") == "Chocolate", 1 + 0.18 * F.cos(phase) + 0.10 * F.col("Christmas") + 0.06 * F.col("Easter"))
       .when(F.col("category") == "Fruity", 1 + 0.20 * F.col("Halloween") + 0.10 * F.col("Easter") + 0.05 * F.sin(phase))
       .otherwise(1 + 0.05 * F.sin(phase)))
ads = ads.withColumn("si_log", F.log(F.greatest(F.lit(0.6), F.least(F.lit(1.8), si))))

# COMMAND ----------

# DBTITLE 1,Competitor cross-price slots
# self-join prices on (cta, week_ending) to attach each mapped actor's log price
prices = ads.select(F.col("cta"), F.col("week_ending"),
                    F.col("ppg").alias("comp_ppg"), F.col("avg_price").alias("comp_price"))
# rank actors within target x relationship to assign INTRA_1.. / INTER_1.. slots
wm = Window.partitionBy("ppg", "relationship").orderBy("comp_ppg")
cmap = cmap.withColumn("slot", F.concat_ws("_", "relationship", "COMP", F.row_number().over(wm), F.lit("logprice")))

linked = (ads.select("cta_ppg", "cta", "week_ending", "ppg")
          .join(cmap.select("ppg", "comp_ppg", "slot"), on="ppg", how="left")
          .join(prices, on=["cta", "week_ending", "comp_ppg"], how="left")
          .withColumn("comp_logprice", F.log("comp_price"))
          .groupBy("cta_ppg", "cta", "week_ending")
          .pivot("slot").agg(F.first("comp_logprice")))

ads = ads.join(linked, on=["cta_ppg", "cta", "week_ending"], how="left")

# COMMAND ----------

# DBTITLE 1,Time-based train/test split (last 20% of weeks per CTA-PPG)
n_weeks = ads.select("week_ending").distinct().count()
cutoff = int(n_weeks * 0.80)
ads = ads.withColumn("test_flag", (F.col("trend") >= cutoff).cast("int"))

ads.write.mode("overwrite").parquet(f"{PROC}/ADS/model_ads")
print("model ADS rows:", ads.count())
