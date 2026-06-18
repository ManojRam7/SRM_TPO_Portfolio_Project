"""Builds mmm/outputs/mmm_dashboard.html (self-contained)."""
import sys, base64
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
import config_mmm as M


def _img(name):
    p = M.MMM_CHARTS / name
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="width:100%;border-radius:8px"/>' if p.exists() else ""


def _table(df):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>" for _, r in df.iterrows())
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def build_dashboard():
    s = pd.read_csv(M.MMM_OUT / "mmm_model_summary.csv")
    val = pd.read_csv(M.MMM_OUT / "mmm_validation.csv")
    roi = pd.read_csv(M.MMM_OUT / "mmm_roi.csv")
    scen = pd.read_csv(M.MMM_OUT / "mmm_scenarios.csv")
    cur = scen[scen.scenario == "Current"].profit.sum()
    best = scen[scen.recommended == 1].profit.sum()
    uplift = (best / cur - 1) * 100

    def card(v, l, sub=""):
        return f'<div class="card"><div class="num">{v}</div><div class="lbl">{l}</div><div class="sub">{sub}</div></div>'
    kpis = "".join([
        card(len(s), "models fitted", "retailer x PG"),
        card(f"{s.r2.median():.3f}", "median R&sup2;", "elastic-net"),
        card(f"{s.mape.median():.1f}%", "median MAPE", "fit"),
        card(f"{val.abs_error.mean():.3f}", "elasticity error", "recovered vs true"),
        card(f"+{uplift:.0f}%", "scenario uplift", "best vs current"),
    ])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>MMM Results</title><style>
 body{{font-family:Segoe UI,Calibri,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}}
 .hero{{background:linear-gradient(135deg,#006D77,#1F3364);color:#fff;padding:26px 34px}}
 .hero h1{{margin:0 0 6px;font-size:22px}} .wrap{{max-width:1080px;margin:0 auto;padding:22px 34px}}
 .cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:-40px 0 18px}}
 .card{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 2px 10px rgba(0,0,0,.06);text-align:center}}
 .card .num{{font-size:21px;font-weight:700;color:#006D77}} .card .lbl{{font-size:11px;margin-top:3px}} .card .sub{{font-size:9px;color:#888}}
 h2{{color:#1F3364;border-bottom:2px solid #e3e8ef;padding-bottom:6px;margin-top:28px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff;border-radius:8px;overflow:hidden}}
 th{{background:#006D77;color:#fff;padding:7px 9px;text-align:left;font-size:11px}}
 td{{padding:6px 9px;border-bottom:1px solid #eef0f3}} tr:nth-child(even) td{{background:#f8fafc}}
 .note{{font-size:11px;color:#777;margin-top:6px}}</style></head><body>
<div class="hero"><h1>Marketing Mix Modelling (MMM): Results Dashboard</h1>
<p>Retailer-level confectionery panel &middot; Retailer &times; Promo Group &times; Week &middot; elastic-net log-log &rarr; contribution &rarr; ROI &rarr; scenarios</p></div>
<div class="wrap"><div class="cards">{kpis}</div>
 <h2>1. Elasticity recovery (elastic-net + relaxed OLS)</h2>
 <div class="grid2"><div>{_img('mmm_elasticity_recovery.png')}</div>
   <div>{_table(val.round(3))}</div></div>
 <h2>2. Contribution decomposition</h2>
 {_img('mmm_contribution.png')}
 <h2>3. Promotion ROI</h2>
 <div class="grid2"><div>{_img('mmm_roi.png')}</div>
   <div>{_table(roi[['retailer','promo_group','promo_weeks','roi']])}</div></div>
 <h2>4. Scenario simulation: optimise spend allocation</h2>
 <div class="grid2"><div>{_img('mmm_scenarios.png')}</div>
   <div>{_table(scen[scen.recommended==1][['retailer','promo_group','scenario','vs_current_pct']].rename(columns={'vs_current_pct':'uplift %'}))}</div></div>
 <p class="note">Best per-PG spend allocation improves modelled profit by ~{uplift:.0f}% vs the current plan. Synthetic data; all retailers and brands are fictional.</p>
</div></body></html>"""
    (M.MMM_OUT / "mmm_dashboard.html").write_text(html, encoding="utf-8")
    print(f"[report_mmm] dashboard -> {M.MMM_OUT/'mmm_dashboard.html'}")


if __name__ == "__main__":
    build_dashboard()
