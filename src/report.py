"""Builds a single self-contained outputs/dashboard.html (charts embedded as base64)."""
import sys, base64
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C


def _img(name):
    p = C.CHARTS / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px"/>'


def _table(df, n=None, money=()):
    df = df.head(n) if n else df
    d = df.copy()
    for c in money:
        if c in d.columns:
            d[c] = d[c].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    rows = ""
    for _, r in d.iterrows():
        rows += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in d.columns) + "</tr>"
    return f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def build_dashboard():
    summ = pd.read_csv(C.OUT / "model_summary.csv")
    val = pd.read_csv(C.OUT / "elasticity_validation.csv")
    eff = pd.read_csv(C.OUT / "promo_efficiency.csv")
    o = pd.read_csv(C.OUT / "optimiser_reco.csv")
    qc = pd.read_csv(C.OUT / "qc_report.csv")

    uplift = (o.optimised_profit.sum() / o.current_profit.sum() - 1) * 100
    spend_cut = (1 - o.optimised_trade_spend.sum() / o.current_trade_spend.sum()) * 100
    focal = eff[eff.manufacturer == C.FOCAL_MANUFACTURER].copy()

    def card(v, label, sub=""):
        return f'<div class="card"><div class="num">{v}</div><div class="lbl">{label}</div><div class="sub">{sub}</div></div>'

    kpis = "".join([
        card(f"{len(summ)}", "models fitted", "one per CTA-PPG"),
        card(f"{summ.r2.median():.3f}", "median R&sup2;", "training fit"),
        card(f"{summ.test_mape.median():.1f}%", "median test MAPE", "out-of-sample"),
        card(f"{val.abs_error.mean():.3f}", "elasticity error", "recovered vs true"),
        card(f"+{uplift:.1f}%", "profit uplift", "optimiser vs current"),
        card(f"-{spend_cut:.0f}%", "trade spend", "more efficient"),
    ])

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SRM / Trade Promotion Optimisation: Results Dashboard</title>
<style>
 body{{font-family:Segoe UI,Calibri,Arial,sans-serif;margin:0;background:#f4f6f9;color:#222}}
 .hero{{background:linear-gradient(135deg,#1F3364,#2E5984);color:#fff;padding:28px 36px}}
 .hero h1{{margin:0 0 6px;font-size:24px}} .hero p{{margin:0;opacity:.85}}
 .wrap{{max-width:1100px;margin:0 auto;padding:24px 36px}}
 .cards{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:-44px 0 20px}}
 .card{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 2px 10px rgba(0,0,0,.06);text-align:center}}
 .card .num{{font-size:22px;font-weight:700;color:#1F3364}} .card .lbl{{font-size:11px;color:#333;margin-top:3px}}
 .card .sub{{font-size:9px;color:#888}}
 h2{{color:#1F3364;border-bottom:2px solid #e3e8ef;padding-bottom:6px;margin-top:30px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
 table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff;border-radius:8px;overflow:hidden}}
 th{{background:#1F3364;color:#fff;padding:7px 9px;text-align:left;font-size:11px}}
 td{{padding:6px 9px;border-bottom:1px solid #eef0f3}} tr:nth-child(even) td{{background:#f8fafc}}
 .note{{font-size:11px;color:#777;margin-top:6px}}
 .pass{{color:#1E7A46;font-weight:700}} .fail{{color:#B23A48;font-weight:700}}
</style></head><body>
<div class="hero"><h1>SRM &amp; Trade Promotion Optimisation: Results Dashboard</h1>
<p>Synthetic US confectionery panel &middot; CTA &times; PPG &times; week &middot; log-log elasticity modelling &rarr; baseline &rarr; optimiser</p></div>
<div class="wrap">
 <div class="cards">{kpis}</div>

 <h2>1. Does the model work? Elasticity recovery</h2>
 <div class="grid2">
   <div>{_img('elasticity_recovery.png')}</div>
   <div>{_table(val.rename(columns={'recovered_elasticity':'recovered','true_beta_price':'true','abs_error':'abs_err','pct_error':'pct_err%','test_mape':'mape%'}))}</div>
 </div>
 <p class="note">Because the data has known elasticities baked in, we can check that the pipeline recovers them: mean absolute error {val.abs_error.mean():.3f}, every PPG within {val.pct_error.max():.0f}% of truth.</p>

 <h2>2. Where do sales come from? Baseline vs incremental</h2>
 <div class="grid2"><div>{_img('waterfall.png')}</div><div>{_img('baseline_trend.png')}</div></div>

 <h2>3. Which promotions pay back? Promo efficiency ({C.FOCAL_MANUFACTURER.title()})</h2>
 <div class="grid2">
   <div>{_img('promo_roi.png')}</div>
   <div>{_table(focal[['ppg','promo_weeks','lift_pct','roi']].rename(columns={'promo_weeks':'promo wks','lift_pct':'lift %'}))}</div>
 </div>

 <h2>4. What should we do? Optimiser recommendation</h2>
 <div class="grid2">
   <div>{_img('optimiser_uplift.png')}</div>
   <div>{_table(o.groupby('ppg')[['current_profit','optimised_profit','current_trade_spend','optimised_trade_spend']].sum().reset_index().assign(uplift_pct=lambda x:((x.optimised_profit/x.current_profit-1)*100).round(1)), money=('current_profit','optimised_profit','current_trade_spend','optimised_trade_spend'))}</div>
 </div>
 <p class="note">Optimiser lifts the modelled profit pool by <b>+{uplift:.1f}%</b> while cutting trade spend <b>{spend_cut:.0f}%</b>, by reallocating the promo budget to the highest-return weeks and trimming over-deep discounts, within margin/AUP/no-promo guardrails.</p>

 <h2>5. Quality checks</h2>
 {_table(qc.assign(passed=qc.passed.map(lambda b: '<span class="pass">PASS</span>' if b else '<span class="fail">FAIL</span>')))}
 <p class="note">Generated by run_pipeline.py &middot; synthetic data; all brands and manufacturers are fictional.</p>
</div></body></html>"""
    out = C.OUT / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] dashboard -> {out}")


if __name__ == "__main__":
    build_dashboard()
