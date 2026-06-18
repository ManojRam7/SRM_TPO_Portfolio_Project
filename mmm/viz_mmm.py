"""MMM charts -> mmm/outputs/charts/*.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
import config_mmm as M

NAVY="#1F3364"; TEAL="#006D77"; AMBER="#E0A500"; GREEN="#1E7A46"; RED="#B23A48"; GREY="#8A8F98"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.color": "#eef0f3", "figure.dpi": 120})


def recovery():
    s = pd.read_csv(M.MMM_OUT / "mmm_model_summary.csv")
    g = pd.read_csv(M.MMM_DATA / "ground_truth_mmm.csv")
    r = s.groupby("promo_group").price_elasticity.mean().reset_index().merge(g, on="promo_group")
    fig, ax = plt.subplots(figsize=(6.2, 5))
    lim = [r.true_e_price.min()-0.2, r.true_e_price.max()+0.2]
    ax.plot(lim, lim, "--", color=GREY, lw=1, label="perfect (y=x)")
    ax.scatter(r.true_e_price, r.price_elasticity, s=70, color=TEAL, edgecolor="white", zorder=3)
    for _, x in r.iterrows():
        ax.annotate(x.promo_group.replace("_", " ").title(), (x.true_e_price, x.price_elasticity),
                    fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#444")
    err = (r.price_elasticity - r.true_e_price).abs().mean()
    ax.set_xlabel("TRUE elasticity"); ax.set_ylabel("RECOVERED (elastic-net + relaxed OLS)")
    ax.set_title(f"MMM recovers known elasticities\n(mean abs error {err:.3f})", color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(M.MMM_CHARTS / "mmm_elasticity_recovery.png"); plt.close(fig)


def contribution():
    d = pd.read_csv(M.MMM_OUT / "mmm_contribution.csv")
    steps = {"Base": d.base_units.sum(), "Price": d.eff_price.sum(), "Catalogue": d.eff_catalogue.sum(),
             "Display": d.eff_display.sum(), "Pantry": d.eff_pantry.sum(), "Competition": d.eff_competition.sum()}
    fig, ax = plt.subplots(figsize=(7.4, 4.6)); cum = 0
    for i, (k, v) in enumerate(steps.items()):
        col = NAVY if k == "Base" else (GREEN if v >= 0 else RED)
        ax.bar(i, v, bottom=cum if k != "Base" else 0, color=col, edgecolor="white"); cum += v
    ax.bar(len(steps), d.units.sum(), color="#2E5984", edgecolor="white")
    ax.set_xticks(range(len(steps)+1)); ax.set_xticklabels(list(steps)+["Actual"], rotation=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_title("MMM contribution decomposition (all promo groups)", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(M.MMM_CHARTS / "mmm_contribution.png"); plt.close(fig)


def roi_chart():
    r = pd.read_csv(M.MMM_OUT / "mmm_roi.csv")
    r["label"] = r.retailer + " | " + r.promo_group.str.replace("_", " ").str.title()
    r = r.sort_values("roi")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = [GREEN if x >= 1 else (AMBER if x >= 0.18 else RED) for x in r.roi]
    ax.barh(r.label, r.roi, color=colors, edgecolor="white")
    ax.axvline(1.0, color=GREY, ls="--", lw=1)
    ax.set_xlabel("Promotion ROI (incremental profit / trade spend)")
    ax.set_title("MMM promotion ROI by retailer x promo group", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(M.MMM_CHARTS / "mmm_roi.png"); plt.close(fig)


def scenarios():
    s = pd.read_csv(M.MMM_OUT / "mmm_scenarios.csv")
    agg = s.groupby("scenario").profit.sum() / 1e6
    agg = agg.reindex(["Current", "Shallower TPR (cap 15%)", "Catalogue-led (10% + feature)", "Trim deep promos (>=25%)"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = [GREY] + [TEAL]*3
    ax.bar(range(len(agg)), agg.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(agg))); ax.set_xticklabels([x.replace(" (", "\n(") for x in agg.index], fontsize=8)
    ax.set_ylabel("Total profit ($M)")
    ax.set_title("Scenario simulation: total profit by spend-allocation plan", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(M.MMM_CHARTS / "mmm_scenarios.png"); plt.close(fig)


def build_all():
    recovery(); contribution(); roi_chart(); scenarios()
    print(f"[viz_mmm] 4 charts -> {M.MMM_CHARTS}")


if __name__ == "__main__":
    build_all()
