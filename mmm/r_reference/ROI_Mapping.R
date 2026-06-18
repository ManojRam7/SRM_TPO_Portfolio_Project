############################################################################
## ROI_Mapping.R - trade expenditure (TE) and promotion ROI per retailer x PG
## R equivalent of the ROI half of m04_roi_scenario.py.
##   TE per unit  = list price x standard terms + settlement discount + deferred deal
##   trade spend  = (base price - promo price + TE) x units
##   ROI          = incremental profit / trade spend
############################################################################
rm(list = ls())
library(data.table)

contrib <- fread("../outputs/mmm_contribution.csv")
terms   <- fread("../data/roi_terms.csv")
d <- merge(contrib, terms[, !"base_price"], by = "promo_group", all.x = TRUE)

d[, TE_per_unit  := list_price * std_terms + settlement_discount + deferred_deal]
d[, own_incr     := pmax(eff_price + eff_catalogue + eff_display + eff_pantry, 0)]
d[, trade_spend  := pmax((base_price - avg_price + TE_per_unit) * units, 0)]
d[, incr_profit  := own_incr * (avg_price - cogs_per_unit)]
d[, is_promo     := as.integer(tpr_discount > 1)]

roi <- d[, .(promo_weeks = sum(is_promo), incr_units = round(sum(own_incr)),
             trade_spend = round(sum(trade_spend)), incr_profit = round(sum(incr_profit))),
         by = .(retailer, promo_group)]
roi[, roi := round(incr_profit / trade_spend, 2)]
fwrite(roi[order(-roi)], "../outputs/r_mmm_roi.csv")
