############################################################################
## Modelling_Data.R - elastic-net log-log model per retailer x promo group
## R equivalent of m01_data_prep.py + m02_mmm_model.py (glmnet / cv.glmnet).
############################################################################
rm(list = ls())
library(data.table); library(glmnet)
source("Functions.R")
set.seed(1234)

raw <- fread("../data/mmm_raw.csv")
dt  <- add_log_features(raw)

## Acting items are chosen per promo group on the pooled data, then attached per retailer-week.
groups <- unique(dt$promo_group)
actors <- setNames(lapply(groups, function(g)
  head(acting_item_filter(dt, g, setdiff(groups, g)), 3)), groups)

results <- list(); coefs <- list()
for (g in groups) {
  d_all <- attach_acting_items(dt, g, actors[[g]])
  for (r in unique(d_all$retailer)) {
    d <- d_all[retailer == r]
    feats <- c("log_price", "catalogue", "display", "si_log", "trend", "tpr_lag1",
               grep("^ACT_", names(d), value = TRUE))
    feats <- feats[sapply(feats, function(f) uniqueN(d[[f]]) > 1)]
    X <- as.matrix(d[, ..feats]); X[is.na(X)] <- 0
    y <- d$log_qty

    ## elastic net (alpha = 0.8), lambda chosen by 5-fold CV
    cvfit <- cv.glmnet(X, y, alpha = 0.8, nfolds = 5)
    b     <- coef(cvfit, s = "lambda.min")
    sel   <- setdiff(rownames(b)[as.vector(b != 0)], "(Intercept)")

    ## relaxed refit: OLS on the selected features removes shrinkage bias
    fit  <- lm(reformulate(sel, response = "y"), data = data.frame(y = y, X))
    pred <- expm1(fitted(fit))
    key  <- paste(r, g, sep = "|")
    results[[key]] <- data.table(retailer_pg = key, retailer = r, promo_group = g,
                                 n = nrow(d), r2 = summary(fit)$r.squared,
                                 mape = mape(d$units, pred),
                                 price_elasticity = unname(coef(fit)["log_price"]),
                                 lambda = cvfit$lambda.min,
                                 selected = paste(sel, collapse = ";"))
    coefs[[key]] <- data.table(retailer_pg = key, term = names(coef(fit)), estimate = coef(fit))
  }
}
fwrite(rbindlist(results), "../outputs/r_mmm_model_summary.csv")
fwrite(rbindlist(coefs),   "../outputs/r_mmm_coefficients.csv")
