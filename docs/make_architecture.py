"""Render docs/architecture.png - the Azure/Databricks ELT architecture."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY="#1F3364"; BLUE="#2E5984"; TEAL="#006D77"; AMBER="#E0A500"; GREY="#5b6470"; LIGHT="#EAF1F7"
fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

def box(x, y, w, h, text, fc=LIGHT, ec=BLUE, tc=NAVY, fs=9, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, c=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color=c))

ax.text(6, 6.7, "SRM / Trade Promotion Optimisation: Azure ELT Architecture",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

# Sources
box(0.2, 4.2, 2.2, 1.9, "DATA SOURCES\n\nScanner (POS) data\nLoyalty-card panel\nPromo calendar\nFinancials / Trade rate",
    fc="#fff", ec=GREY, tc=GREY, fs=8.5)
# ADF ingest
box(2.9, 4.7, 1.8, 1.0, "Azure Data\nFactory\n(ingest + orchestrate)", fc=LIGHT, ec=BLUE, fs=8.5)
# Data Lake
box(2.9, 3.1, 1.8, 1.0, "Azure Data Lake\n(ADLS)\nraw / processed", fc=LIGHT, ec=BLUE, fs=8.5)
# Databricks block
box(5.1, 2.7, 4.4, 3.2, "", fc="#F7FAFD", ec=TEAL)
ax.text(7.3, 5.6, "Azure Databricks  (PySpark + SparkR)", ha="center", fontsize=9.5,
        fontweight="bold", color=TEAL)
stages = ["01  Union & Map", "02  Treatment & Features", "03  Log-log Stepwise Model",
          "04  Baseline / Incremental", "05  Insights (Waterfall, ROI)", "06  Optimiser (OR-Tools)"]
for i, s in enumerate(stages):
    col = i % 2; row = i // 2
    box(5.25 + col*2.15, 4.6 - row*0.75, 2.0, 0.6, s, fc="#fff", ec=TEAL, tc=NAVY, fs=7.6)
# Serving
box(9.9, 4.5, 1.9, 1.2, "Azure SQL DB\n(curated serving)", fc=LIGHT, ec=BLUE, fs=8.5)
box(9.9, 2.9, 1.9, 1.2, "Power BI\nSRM Insights\nDashboards", fc=LIGHT, ec=AMBER, tc="#9A5200", fs=8.5)
# DevOps band
box(2.9, 1.2, 8.9, 0.9, "Azure DevOps  CI/CD  (YAML dev->prod)   +   Git   +   QC gates & unit tests",
    fc="#F2F4F7", ec=GREY, tc=GREY, fs=9)

arrow(2.4, 5.2, 2.9, 5.2)                 # sources -> ADF
arrow(3.8, 4.7, 3.8, 4.1)                 # ADF -> lake
arrow(4.7, 3.6, 5.1, 3.9)                 # lake -> databricks
arrow(9.5, 4.3, 9.9, 4.9)                 # databricks -> SQL
arrow(10.85, 4.5, 10.85, 4.1)            # SQL -> Power BI
arrow(3.8, 5.7, 3.8, 5.7)
ax.text(6.0, 0.6, "Unit of analysis: CTA x PPG x Week   |   3 years of weekly scanner data",
        ha="center", fontsize=8.5, color=GREY, style="italic")

out = Path(__file__).resolve().parent / "architecture.png"
fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
