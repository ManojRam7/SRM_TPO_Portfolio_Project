"""Charts for the SRM project (matplotlib). Saved to outputs/charts/*.png."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C

NAVY = "#1F3364"; BLUE = "#2E5984"; TEAL = "#006D77"; AMBER = "#E0A500"
GREEN = "#1E7A46"; RED = "#B23A48"; GREY = "#8A8F98"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc",
                     "axes.grid": True, "grid.color": "#eef0f3", "figure.dpi": 120})


def _load(name):
    return pd.read_csv(C.OUT / name)


def chart_elasticity_recovery():
    v = _load("elasticity_validation.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lim = [min(v.true_beta_price.min(), v.recovered_elasticity.min()) - 0.2,
           max(v.true_beta_price.max(), v.recovered_elasticity.max()) + 0.2]
    ax.plot(lim, lim, "--", color=GREY, lw=1, label="perfect recovery (y=x)")
    ax.scatter(v.true_beta_price, v.recovered_elasticity, s=70, color=TEAL, zorder=3, edgecolor="white")
    for _, r in v.iterrows():
        ax.annotate(r.ppg.replace("_", " ").title(), (r.true_beta_price, r.recovered_elasticity),
                    fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#444")
    ax.set_xlabel("TRUE price elasticity (data-generating)")
    ax.set_ylabel("RECOVERED elasticity (model)")
    ax.set_title("Model recovers the known elasticities\n(mean abs error %.3f)" % v.abs_error.mean(),
                 color=NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(C.CHARTS / "elasticity_recovery.png"); plt.close(fig)


def chart_waterfall():
    wf = _load("waterfall_summary.csv")
    row = wf[wf.ppg == "TOTAL"].iloc[0]
    steps = ["Base", "Price", "Feature", "Display", "Pantry", "Competition"]
    vals = [row[s] for s in steps]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cum = 0;
    for i, (s, v) in enumerate(zip(steps, vals)):
        color = NAVY if s == "Base" else (GREEN if v >= 0 else RED)
        ax.bar(i, v, bottom=cum if s != "Base" else 0, color=color, edgecolor="white")
        cum += v
    ax.bar(len(steps), row["Actual"], color=BLUE, edgecolor="white")
    ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels(steps + ["Actual"], rotation=20)
    ax.set_ylabel("Units (millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_title("Sales-driver waterfall (all PPGs): baseline vs promotional lift",
                 color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(C.CHARTS / "waterfall.png"); plt.close(fig)


def chart_promo_roi():
    e = _load("promo_efficiency.csv")
    e = e[e.manufacturer == C.FOCAL_MANUFACTURER].sort_values("roi")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = [GREEN if r >= 1 else (AMBER if r >= 0.4 else RED) for r in e.roi]
    ax.barh(e.ppg.str.replace("_", " ").str.title(), e.roi, color=colors, edgecolor="white")
    ax.axvline(1.0, color=GREY, ls="--", lw=1)
    ax.set_xlabel("Promotion ROI (incremental gross profit / trade spend)")
    ax.set_title(f"Promotion efficiency by PPG ({C.FOCAL_MANUFACTURER.title()})", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(C.CHARTS / "promo_roi.png"); plt.close(fig)


def chart_optimiser():
    o = _load("optimiser_reco.csv")
    agg = o.groupby("ppg")[["current_profit", "optimised_profit"]].sum()
    agg = agg.sort_values("optimised_profit", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(agg)); h = 0.38
    ax.barh(y - h/2, agg.current_profit/1e6, h, label="Current plan", color=GREY)
    ax.barh(y + h/2, agg.optimised_profit/1e6, h, label="Optimised plan", color=TEAL)
    ax.set_yticks(y); ax.set_yticklabels(agg.index.str.replace("_", " ").str.title())
    ax.set_xlabel("Profit pool ($M)")
    up = (o.optimised_profit.sum()/o.current_profit.sum()-1)*100
    ax.set_title(f"Optimiser uplift: +{up:.1f}% profit at lower trade spend", color=NAVY, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.CHARTS / "optimiser_uplift.png"); plt.close(fig)


def chart_baseline_trend():
    d = _load("baseline_decomp.csv")
    d["week_ending"] = pd.to_datetime(d["week_ending"])
    g = d.groupby("week_ending")[["base_units", "incr_units"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.stackplot(g.week_ending, g.base_units, g.incr_units.clip(lower=0),
                 labels=["Baseline", "Incremental (promo)"], colors=[NAVY, AMBER], alpha=0.9)
    ax.set_ylabel("Units / week"); ax.legend(loc="upper left", frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.set_title("Baseline vs incremental volume over time (all PPGs)", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(C.CHARTS / "baseline_trend.png"); plt.close(fig)


def build_all():
    chart_elasticity_recovery(); chart_waterfall(); chart_promo_roi()
    chart_optimiser(); chart_baseline_trend()
    print(f"[viz] 5 charts written to {C.CHARTS}")


if __name__ == "__main__":
    build_all()
