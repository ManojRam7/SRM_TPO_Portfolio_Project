# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - LOG-LOG STEPWISE REGRESSION (per CTA-PPG)
# MAGIC Databricks version of `src/s03_modelling.py`. In SparkR this would be `gapply`
# MAGIC with per-group `lm()` + `stepAIC()`; here it is the PySpark equivalent,
# MAGIC `groupBy("cta_ppg").applyInPandas(...)`,
# MAGIC which ships each CTA-PPG group to a worker and fits one model with the same
# MAGIC guards: stepwise (AIC), VIF pruning, economic SIGN CONSTRAINTS, train/test.
# MAGIC The coefficient on `log_price` is the price elasticity.

# COMMAND ----------

import numpy as np, pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

PROC = "/mnt/srm/PROCESS"
ads = spark.read.parquet(f"{PROC}/ADS/model_ads")

PRIMARY = ["log_price", "feat_acv", "disp_acv", "si_log", "trend", "pantry_lag"]
HOLIDAYS = ["Halloween", "Christmas", "Easter", "Valentine", "SummerDip"]
COMP = [f"{r}_COMP_{i}_logprice" for r in ("INTRA", "INTER") for i in (1, 2, 3)]
SIGN = {"log_price": -1, "feat_acv": 1, "disp_acv": 1, "si_log": 1, "trend": 1, "pantry_lag": -1}

out_schema = StructType([
    StructField("cta_ppg", StringType()), StructField("n_train", IntegerType()),
    StructField("r2", DoubleType()), StructField("test_mape", DoubleType()),
    StructField("price_elasticity", DoubleType()), StructField("beta_feat", DoubleType()),
    StructField("beta_disp", DoubleType()), StructField("price_sign_valid", IntegerType()),
    StructField("selected_features", StringType()),
])

# COMMAND ----------

def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid); tss = float(((y - y.mean()) ** 2).sum())
    n, k = X.shape
    return beta, (1 - rss / tss if tss > 0 else 0.0), n * np.log(rss / n + 1e-12) + 2 * k

def _design(df, feats):
    return np.column_stack([np.ones(len(df))] + [df[f].fillna(0).values for f in feats]), df["log_units"].values

def fit_group(pdf: pd.DataFrame) -> pd.DataFrame:
    cp = pdf["cta_ppg"].iloc[0]
    tr = pdf[pdf.test_flag == 0]; te = pdf[pdf.test_flag == 1]
    feats = [f for f in PRIMARY + HOLIDAYS + COMP if f in pdf.columns and tr[f].fillna(0).nunique() > 1]
    if len(tr) < 30 or "log_price" not in feats:
        return pd.DataFrame([[cp, len(tr), 0., None, None, None, None, 0, ""]], columns=out_schema.names)

    # VIF pruning of competitor terms
    comp = [f for f in feats if f in COMP]
    while comp:
        M = tr[feats].fillna(0).values; vif = {}
        for j, c in enumerate(feats):
            others = [i for i in range(M.shape[1]) if i != j]
            Xo = np.column_stack([np.ones(len(M)), M[:, others]])
            _, r2o, _ = _ols(Xo, M[:, j]); vif[c] = 1 / max(1e-6, 1 - r2o)
        worst = max(comp, key=lambda c: vif[c])
        if vif[worst] >= 8: feats.remove(worst); comp.remove(worst)
        else: break

    # backward stepwise by AIC
    X, y = _design(tr, feats); _, _, best = _ols(X, y); improved = True
    while improved and len(feats) > 1:
        improved = False
        for f in list(feats):
            Xt, yt = _design(tr, [c for c in feats if c != f]); _, _, aic = _ols(Xt, yt)
            if aic < best - 2: best, feats, improved = aic, [c for c in feats if c != f], True; break

    # economic sign constraints (drop wrong-signed non-price terms, refit)
    for _ in range(12):
        X, y = _design(tr, feats); beta, r2, _ = _ols(X, y)
        bad = [n for n, b in zip(feats, beta[1:])
               if n != "log_price" and SIGN.get(n) and np.sign(b) != SIGN[n] and abs(b) > 1e-6]
        if not bad: break
        feats = [f for f in feats if f not in bad]

    X, y = _design(tr, feats); beta, r2, _ = _ols(X, y); coef = dict(zip(["intercept"] + feats, beta))
    mape = None
    if len(te):
        Xte = np.column_stack([np.ones(len(te))] + [te[f].fillna(0).values for f in feats])
        pred = np.expm1(Xte @ beta); a = te["units"].values; m = a != 0
        mape = float(np.mean(np.abs((a[m] - pred[m]) / a[m])) * 100)
    pe = coef.get("log_price", np.nan)
    return pd.DataFrame([[cp, len(tr), round(r2, 4), mape, round(pe, 4),
                          round(coef.get("feat_acv", 0), 4), round(coef.get("disp_acv", 0), 4),
                          int(-10 <= pe <= 0), ";".join(feats)]], columns=out_schema.names)

# COMMAND ----------

model_summary = ads.groupBy("cta_ppg").applyInPandas(fit_group, schema=out_schema)
model_summary.write.mode("overwrite").parquet(f"{PROC}/MODEL/model_summary")
display(model_summary.orderBy("price_elasticity"))
