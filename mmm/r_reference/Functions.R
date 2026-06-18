############################################################################
## Functions.R - shared helpers for the R version of the MMM workstream
## Log features, pantry lags and acting-item (cross-price) selection.
## Run the scripts from this folder after `python mmm/run_mmm.py` has written
## ../data/mmm_raw.csv and ../data/roi_terms.csv.
############################################################################
library(data.table)

## Log transforms for the DV (quantity) and IVs (price, seasonality) + pantry lag.
add_log_features <- function(dt) {
  dt <- copy(dt)
  dt[, week_ending := as.IDate(week_ending)]
  setorder(dt, retailer, promo_group, week_ending)
  dt[, `:=`(
    log_qty   = log1p(units),
    log_price = log(avg_price),
    si_log    = log(pmax(si, 0.5))
  )]
  dt[, trend    := seq_len(.N) - 1L, by = .(retailer, promo_group)]
  dt[, tpr_lag1 := shift(tpr_discount, 1, fill = 0, type = "lag"), by = .(retailer, promo_group)]
  dt
}

## Rank candidate acting items for a target promo group by |correlation| between
## the actor's log price and the target's log volume (matched on retailer-week).
acting_item_filter <- function(dt, target, candidates, cutoff = 0.05) {
  tgt <- dt[promo_group == target, .(retailer, week_ending, y = log_qty)]
  out <- rbindlist(lapply(candidates, function(a) {
    act <- dt[promo_group == a, .(retailer, week_ending, p = log_price)]
    m <- merge(tgt, act, by = c("retailer", "week_ending"))
    if (nrow(m) <= 10) return(NULL)
    data.table(item = a, value = cor(m$y, m$p, use = "pairwise.complete.obs"))
  }))
  if (!nrow(out)) return(character(0))
  out <- out[abs(value) >= cutoff][order(-abs(value))]
  out$item
}

## Attach up to three acting-item log-price columns (ACT_1..3_logprice) to a target.
attach_acting_items <- function(dt, target, actors) {
  base <- dt[promo_group == target]
  for (k in seq_along(actors)) {
    act <- dt[promo_group == actors[k], .(retailer, week_ending, v = log_price)]
    setnames(act, "v", paste0("ACT_", k, "_logprice"))
    base <- merge(base, act, by = c("retailer", "week_ending"), all.x = TRUE)
  }
  base
}

mape <- function(actual, pred) {
  keep <- actual != 0
  mean(abs((actual[keep] - pred[keep]) / actual[keep])) * 100
}
