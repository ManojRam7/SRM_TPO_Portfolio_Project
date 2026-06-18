# Marketing Mix Modelling (retailer level)

Weekly retailer x promo group models for two grocery chains (Freshway and Northmart, both
fictional). The approach follows the SRM module (log-log demand) but uses regularised regression,
automatic cross-price selection, ROI from trade terms and scenario simulation.

## Run

```bash
python mmm/run_mmm.py
open mmm/outputs/mmm_dashboard.html
```

## Pipeline

| Stage | File | Purpose |
|---|---|---|
| 00 | `m00_generate_mmm_data.py` | Synthetic retailer x promo group x week data with known coefficients |
| 01 | `m01_data_prep.py` | Log features, pantry lag and acting-item (cross-price) selection |
| 02 | `m02_mmm_model.py` | Elastic-net log-log model with a relaxed OLS refit, per retailer and promo group |
| 03 | `m03_contribution.py` | Baseline vs incremental contribution |
| 04 | `m04_roi_scenario.py` | ROI from trade terms and scenario simulation |
| | `mmm_common.py` | Elastic net (coordinate descent + CV) and the acting-item filter |
| | `viz_mmm.py`, `report_mmm.py` | Charts and dashboard |

## Method

- **Elastic net** (alpha 0.8, lambda chosen by 5-fold CV) selects features and shrinks correlated
  promo and cross-price terms. A **relaxed OLS refit** on the selected set removes the shrinkage
  bias, so the price coefficients are usable as elasticities.
- **Acting items**: candidate competitor or own-portfolio groups are ranked by the correlation of
  their price with the target's volume, and the top three are kept.
- **ROI**: trade expenditure per unit = list price x standard terms + settlement discount +
  deferred deal; ROI = incremental profit / trade spend.
- **Scenarios**: Current, Shallower TPR (cap 15%), Catalogue-led (10% + feature) and Trim deep
  promos (>= 25%) are simulated with the fitted coefficients; catalogue and display carry a cost.

## Results (latest run)

| Metric | Result |
|---|---|
| Models | 14 (7 promo groups x 2 retailers) |
| Median R² / MAPE | 0.976 / 5.5% |
| Elasticity recovery | mean absolute error 0.034; 7 of 7 within 20% |
| Baseline share of volume | 76% |
| Best scenario vs current plan | +25.6% modelled profit (catalogue-led for 10 groups, trim deep promos for 4) |

## R version

`r_reference/` has the same steps in R: `Functions.R` (features, acting items),
`Modelling_Data.R` (glmnet + relaxed refit), `Contribution_Calculation.R`, `ROI_Mapping.R` and
`Scenario_Creation.R`. They read the CSVs written by the Python pipeline.
