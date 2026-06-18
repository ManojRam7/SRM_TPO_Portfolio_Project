############################################################################
## Scenario_Creation.R - simulate alternative promo plans and compare profit
## R equivalent of the scenario half of m04_roi_scenario.py. Uses the fitted
## coefficients to predict volume and profit under each candidate plan.
############################################################################
rm(list = ls())
library(data.table)
source("Functions.R")

CAT_COST <- 0.08; DISP_COST <- 0.05     # catalogue/display cost as a share of base price per unit

## A: non-promo part of the linear predictor (intercept, seasonality, trend, acting items)
simulate <- function(A, e, bc, bd, base, price, cat, disp, cogs, te) {
  q <- exp(A + e * log(price) + bc * cat + bd * disp)
  cat_cost  <- CAT_COST  * base * cat  * q
  disp_cost <- DISP_COST * base * disp * q
  list(profit = sum(q * (price - cogs - te) - cat_cost - disp_cost),
       spend  = sum((base - price + te) * q + cat_cost + disp_cost))
}

scenarios <- list(
  Current          = function(d) d,
  Shallower_TPR    = function(d) d[, avg_price := pmax(base_price * 0.85, avg_price)],
  Catalogue_led    = function(d) d[tpr_discount > 1, `:=`(avg_price = base_price * 0.90, catalogue = 1L)],
  Trim_deep_promos = function(d) d[tpr_discount >= 25, avg_price := base_price]
)

## Example for one retailer x promo group, given its coefficient vector `cf`
## (named: intercept, log_price, catalogue, display, si_log, trend, ACT_*):
run_scenarios <- function(d, cf, cogs, te) {
  acts <- intersect(grep("^ACT_", names(cf), value = TRUE), names(d))
  A <- cf[["intercept"]] + cf[["si_log"]] * d$si_log + cf[["trend"]] * d$trend
  for (a in acts) A <- A + cf[[a]] * fifelse(is.na(d[[a]]), median(d[[a]], na.rm = TRUE), d[[a]])
  res <- lapply(names(scenarios), function(s) {
    x <- scenarios[[s]](copy(d))
    r <- simulate(A, cf[["log_price"]], cf[["catalogue"]], cf[["display"]],
                  x$base_price, x$avg_price, x$catalogue, x$display, cogs, te)
    data.table(scenario = s, profit = round(r$profit), trade_spend = round(r$spend))
  })
  out <- rbindlist(res)
  out[, recommended := profit == max(profit)]
  out
}
