############################################################################
## Contribution_Calculation.R - baseline vs incremental decomposition
## R equivalent of m03_contribution.py. Predict with actual inputs, then again
## with the promo levers switched off (price at base, catalogue/display/pantry = 0);
## the gap is the promotion-driven volume.
############################################################################
library(data.table)

## fit: an lm() from Modelling_Data.R;  d: the design data for one retailer x promo group
contribution <- function(fit, d) {
  pred_vol <- expm1(predict(fit, newdata = d))

  d_base <- copy(d)
  d_base[, log_price := log(base_price)]
  for (col in c("catalogue", "display", "tpr_lag1")) {
    if (col %in% names(d_base)) set(d_base, j = col, value = 0)
  }
  base_vol <- expm1(predict(fit, newdata = d_base))

  data.table(retailer = d$retailer, promo_group = d$promo_group,
             week_ending = d$week_ending, units = d$units,
             pred_units = round(pred_vol, 1), base_units = round(base_vol, 1),
             incr_units = round(pred_vol - base_vol, 1))
}
