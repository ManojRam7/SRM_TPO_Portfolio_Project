# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - TRADE PROMOTION OPTIMISER (Google OR-Tools)
# MAGIC For each CTA-PPG it solves a binary program: assign one offer per week to
# MAGIC maximise the profit pool, subject to guardrails (margin floor, trade-spend
# MAGIC cap, minimum average unit price, minimum no-promo weeks), using
# MAGIC `ortools.linear_solver.pywraplp` with the CBC solver.
# MAGIC
# MAGIC `value[i][j]` / `margin[i][j]` come from the demand model (offer i in week j).

# COMMAND ----------

import numpy as np, pandas as pd
from ortools.linear_solver import pywraplp
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

PROC = "/mnt/srm/PROCESS"; OUT = "/mnt/srm/OUTPUT"
opt_input = spark.read.parquet(f"{PROC}/MODEL/optimiser_input")   # per CTA-PPG-week-offer: value, margin, spend

schema = StructType([StructField("cta_ppg", StringType()), StructField("week", IntegerType()),
                     StructField("offer_index", IntegerType()), StructField("chosen", IntegerType())])

# COMMAND ----------

def optimise(pdf: pd.DataFrame) -> pd.DataFrame:
    weeks = sorted(pdf.week.unique()); offers = sorted(pdf.offer_index.unique())
    nW, nO = len(weeks), len(offers)
    val = pdf.pivot(index="week", columns="offer_index", values="value").values     # nW x nO
    margin = pdf.pivot(index="week", columns="offer_index", values="margin").values
    spend = pdf.pivot(index="week", columns="offer_index", values="spend").values
    no_promo = offers.index(0)                                                       # index of 0% offer

    solver = pywraplp.Solver.CreateSolver("CBC")
    x = {(i, j): solver.IntVar(0, 1, f"x_{i}_{j}") for i in range(nO) for j in range(nW)}

    # objective: maximise total profit pool
    solver.Maximize(solver.Sum(val[j][i] * x[i, j] for i in range(nO) for j in range(nW)))

    # exactly one offer per week
    for j in range(nW):
        solver.Add(solver.Sum(x[i, j] for i in range(nO)) == 1)

    # guardrails
    MARGIN_FLOOR = 0.0; TRADE_RATE = 0.18; MIN_NO_PROMO = 30
    solver.Add(solver.Sum(margin[j][i] * x[i, j] for i in range(nO) for j in range(nW)) >= MARGIN_FLOOR)
    gsv = solver.Sum(val[j][i] * x[i, j] for i in range(nO) for j in range(nW))
    solver.Add(solver.Sum(spend[j][i] * x[i, j] for i in range(nO) for j in range(nW)) <= TRADE_RATE * gsv)
    solver.Add(solver.Sum(x[no_promo, j] for j in range(nW)) >= MIN_NO_PROMO)

    solver.Solve()
    rows = [[pdf.cta_ppg.iloc[0], weeks[j], offers[i], int(x[i, j].solution_value())]
            for i in range(nO) for j in range(nW) if x[i, j].solution_value() > 0.5]
    return pd.DataFrame(rows, columns=schema.names)

# COMMAND ----------

plan = opt_input.groupBy("cta_ppg").applyInPandas(optimise, schema=schema)
plan.write.mode("overwrite").parquet(f"{OUT}/OPTIMISER_PLAN")
print("optimiser complete")
# NOTE: a runnable numpy version (no OR-Tools dependency) is in src/s06_optimiser.py
