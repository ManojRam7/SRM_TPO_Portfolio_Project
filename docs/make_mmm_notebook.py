"""Builds notebooks/MMM_Walkthrough.ipynb (valid nbformat 4 JSON)."""
import json
from pathlib import Path

def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in l]}
def code(*l): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [x + "\n" for x in l]}

cells = [
 md("# Marketing Mix Modelling (MMM): Walkthrough",
    "", "Retailer × Promo Group × Week. Elastic-net log-log models with acting-item cross-price",
    "selection, contribution decomposition, ROI mapping and scenario simulation.",
    "Synthetic data with known coefficients (so recovery can be proved)."),
 code("import sys, os", "sys.path.append(os.path.abspath('../mmm'))",
      "sys.path.append(os.path.abspath('..')); sys.path.append(os.path.abspath('../src'))",
      "import pandas as pd, config_mmm as M; pd.set_option('display.max_columns', 30)"),
 md("## 1 and 2. Generate data and prepare features (incl. acting-item cross-price selection)"),
 code("import m00_generate_mmm_data as m00, m01_data_prep as m01",
      "m00.generate(); ads = m01.run()",
      "pd.read_csv(M.MMM_OUT/'mmm_acting_items.csv')"),
 md("## 3. Elastic-net log-log model (per retailer × PG) + relaxed OLS"),
 code("import m02_mmm_model as m02",
      "s = m02.run()",
      "s[['retailer','promo_group','lambda','r2','mape','price_elasticity','n_selected']]"),
 md("### Recovered vs TRUE elasticity"),
 code("g = pd.read_csv(M.MMM_DATA/'ground_truth_mmm.csv')",
      "r = s.groupby('promo_group').price_elasticity.mean().reset_index().merge(g, on='promo_group')",
      "r['abs_err'] = (r.price_elasticity - r.true_e_price).abs(); r.round(3)"),
 md("## 4. Contribution decomposition (baseline vs incremental)"),
 code("import m03_contribution as m03", "d = m03.run(); d.head()"),
 md("## 5. ROI mapping + scenario simulation (optimise spend allocation)"),
 code("import m04_roi_scenario as m04", "scen = m04.run()",
      "pd.read_csv(M.MMM_OUT/'mmm_roi.csv')[['retailer','promo_group','promo_weeks','roi']]"),
 code("rec = scen[scen.recommended==1][['retailer','promo_group','scenario','vs_current_pct']]",
      "rec.head(14)"),
 md("## 6. Charts & dashboard"),
 code("import viz_mmm, report_mmm", "viz_mmm.build_all(); report_mmm.build_dashboard()",
      "print('Open mmm/outputs/mmm_dashboard.html')"),
 md("---", "**Takeaway:** elastic-net recovers the known elasticities, contributions split sales into",
    "baseline vs promo drivers, ROI ranks promotions, and scenario simulation finds the spend",
    "allocation that lifts profit; the MMM counterpart of the SRM optimiser."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "MMM_Walkthrough.ipynb"
out.write_text(json.dumps(nb, indent=1)); print("wrote", out)
