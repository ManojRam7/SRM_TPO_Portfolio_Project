# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - UNION & MAPPING
# MAGIC Databricks version of `src/s01_union_map.py`. Reads the raw scanner extract
# MAGIC and financials from the data lake, builds one ADS, runs raw-data quality
# MAGIC checks, and maps dimension keys + financials.
# MAGIC
# MAGIC | Input | Path (ADLS) |
# MAGIC |---|---|
# MAGIC | Scanner raw | `/mnt/srm/RAW/SCANNER/scanner_raw` |
# MAGIC | Financials  | `/mnt/srm/RAW/FILES/financials` |
# MAGIC | Output ADS  | `/mnt/srm/PROCESS/ADS/ads_mapped` |

# COMMAND ----------

from pyspark.sql import functions as F

RAW = "/mnt/srm/RAW"
PROC = "/mnt/srm/PROCESS"

# COMMAND ----------

# DBTITLE 1,Read raw extracts
scanner = spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW}/SCANNER/scanner_raw.csv")
fin = spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW}/FILES/financials.csv")

# COMMAND ----------

# DBTITLE 1,Raw-data quality checks
dupes = scanner.groupBy("cta", "ppg", "week_ending").count().filter("count > 1").count()
nulls = scanner.filter("units is null or avg_price is null or base_price is null").count()
assert dupes == 0, f"duplicate CTA-PPG-week keys: {dupes}"
assert nulls == 0, f"null critical fields: {nulls}"
print("raw rows:", scanner.count())

# COMMAND ----------

# DBTITLE 1,Dimension key + financial mapping
ads = (scanner
       .join(F.broadcast(fin), on="ppg", how="left")
       .withColumn("cta_ppg", F.concat_ws("|", "cta", "ppg"))
       .withColumn("week_ending", F.to_date("week_ending")))

# COMMAND ----------

ads.write.mode("overwrite").parquet(f"{PROC}/ADS/ads_mapped")
print("union+map complete:", ads.count(), "rows |", ads.select("cta_ppg").distinct().count(), "CTA-PPG")
