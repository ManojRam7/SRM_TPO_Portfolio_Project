"""
STAGE 06 - TRADE PROMOTION OPTIMISER
====================================
For each CTA-PPG, choose a discount depth per week (from a candidate offer grid)
that MAXIMISES the annual profit pool, subject to business guardrails:
  * trade-spend cap        (total discount $ <= trade_rate x GSV)
  * minimum no-promo weeks  (protect base price / avoid over-promotion)
  * minimum average unit price (AUP floor)
  * maximum discount depth

The problem is naturally a binary program (one depth per week), and
databricks/06_OPTIMISER.py solves it that way with Google OR-Tools. OR-Tools is
not installable in every environment, so this local version uses a greedy
Lagrangian-style solver that respects the same constraints and uses the SAME
model-predicted demand. Plans are scored against the current (actual)
plan using the same model, so the uplift isolates the optimisation effect.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P


def _units(A, c_price, c_feat, c_disp, base, depth, feat_p, disp_p):
    price = base * (1 - depth)
    promo = (depth > 0).astype(float)
    return np.exp(A + c_price * np.log(price) + c_feat * feat_p * promo + c_disp * disp_p * promo)


def _profit(units, base, depth, cogs):
    return units * (base * (1 - depth) - cogs)


def _optimise_group(g, c, cogs):
    g = g.reset_index(drop=True)
    n = len(g)
    base = g.base_price.values
    feat_p = g.loc[g.discount_depth > 0, "feat_acv"].mean() if (g.discount_depth > 0).any() else 0.4
    disp_p = g.loc[g.discount_depth > 0, "disp_acv"].mean() if (g.discount_depth > 0).any() else 0.3
    feat_p = 0.4 if np.isnan(feat_p) else feat_p
    disp_p = 0.3 if np.isnan(disp_p) else disp_p

    holidays = [h for h in P.HOLIDAYS if h in g.columns]
    comp = np.zeros(n)
    for s in P.COMP_SLOTS:
        if s in g.columns and c.get(s, 0.0) != 0.0:
            comp += c.get(s, 0.0) * g[s].fillna(g[s].median()).values
    A = (c.get("intercept", 0.0) + c.get("si_log", 0.0) * g.si_log.values
         + c.get("trend", 0.0) * g.trend.values
         + sum(c.get(h, 0.0) * g[h].values for h in holidays) + comp)
    c_price, c_feat, c_disp = c.get("log_price", -2.0), c.get("feat_acv", 0.0), c.get("disp_acv", 0.0)

    grid = np.array(C.OFFER_GRID)
    # profit & units for every (week, candidate depth)
    U = np.vstack([_units(A, c_price, c_feat, c_disp, base, np.full(n, d), feat_p, disp_p) for d in grid]).T
    PR = np.vstack([_profit(U[:, k], base, grid[k], cogs) for k in range(len(grid))]).T  # n x grid

    # ---- current (actual) plan scored with the SAME model ----
    cur_depth = g.discount_depth.values
    cur_units = _units(A, c_price, c_feat, c_disp, base, cur_depth, feat_p, disp_p)
    cur_profit = _profit(cur_units, base, cur_depth, cogs)
    cur_spend = base * cur_depth * cur_units

    def spend_of(ch):
        return base * grid[ch] * U[np.arange(n), ch]

    def gsv_of(ch):
        return U[np.arange(n), ch] * base

    # ---- choose promo weeks within a frequency budget (best weeks first) ----
    # Reallocate the SAME number of promo weeks to the weeks/depths with the
    # highest profit gain, instead of promoting everywhere.
    base_profit = PR[:, 0]
    best_k = PR[:, 1:].argmax(axis=1) + 1                       # best non-zero depth per week
    best_gain = PR[np.arange(n), best_k] - base_profit
    budget = max(0, min(int((cur_depth > 0).sum()), n - C.MIN_NO_PROMO_WEEKS))
    choice = np.zeros(n, dtype=int)                            # start every week at base price
    order = np.argsort(-best_gain)
    promo_weeks = [i for i in order if best_gain[i] > 0][:budget]
    for i in promo_weeks:
        choice[i] = best_k[i]

    # ---- guardrails: trade-spend cap & AUP floor -> trim depth greedily ----
    def violates():
        spend = spend_of(choice).sum()
        cap = C.TRADE_RATE * gsv_of(choice).sum()
        avg_price = (base * (1 - grid[choice])).mean()
        return spend > cap or avg_price < C.MIN_AUP_FRACTION * base.mean()

    guard = 0
    while violates() and guard < 5 * n:
        guard += 1
        cands = [i for i in range(n) if choice[i] > 0]
        if not cands:
            break
        best_i, best_score = None, None
        for i in cands:
            k = choice[i]
            dloss = PR[i, k] - PR[i, k - 1]
            dspend = (base[i] * grid[k] * U[i, k]) - (base[i] * grid[k - 1] * U[i, k - 1])
            score = dloss / (abs(dspend) + 1e-6)
            if best_score is None or score < best_score:
                best_score, best_i = score, i
        choice[best_i] -= 1

    opt_depth = grid[choice]
    opt_units = U[np.arange(n), choice]
    opt_profit = PR[np.arange(n), choice]
    opt_spend = base * opt_depth * opt_units

    return {
        "cta_ppg": g.cta_ppg.iloc[0], "cta": g.cta.iloc[0], "ppg": g.ppg.iloc[0],
        "current_profit": round(cur_profit.sum(), 0), "optimised_profit": round(opt_profit.sum(), 0),
        "uplift_pct": round((opt_profit.sum() / cur_profit.sum() - 1) * 100, 1) if cur_profit.sum() > 0 else 0,
        "current_promo_weeks": int((cur_depth > 0).sum()), "optimised_promo_weeks": int((opt_depth > 0).sum()),
        "current_avg_depth_pct": round(cur_depth[cur_depth > 0].mean() * 100, 1) if (cur_depth > 0).any() else 0,
        "optimised_avg_depth_pct": round(opt_depth[opt_depth > 0].mean() * 100, 1) if (opt_depth > 0).any() else 0,
        "current_trade_spend": round(cur_spend.sum(), 0), "optimised_trade_spend": round(opt_spend.sum(), 0),
    }, pd.DataFrame({"cta_ppg": g.cta_ppg.iloc[0], "week_ending": g.week_ending,
                     "current_depth": cur_depth, "optimised_depth": opt_depth})


def run():
    ads = P.load("model_ads.csv")
    coef = P.load("model_coefficients.csv").set_index("cta_ppg")
    fin = P.load("financials.csv").set_index("ppg")["cogs_per_unit"].to_dict()

    recos, plans = [], []
    for cp, g in ads.groupby("cta_ppg"):
        if cp not in coef.index:
            continue
        c = coef.loc[cp].to_dict()
        cogs = fin.get(g.ppg.iloc[0], g.base_price.iloc[0] * (1 - C.GMARGIN))
        # only optimise the focal manufacturer's PPGs; competitors are context
        if g.manufacturer.iloc[0] != C.FOCAL_MANUFACTURER:
            continue
        rec, plan = _optimise_group(g, c, cogs)
        recos.append(rec)
        plans.append(plan)

    reco = pd.DataFrame(recos).sort_values("uplift_pct", ascending=False)
    P.save(reco, "optimiser_reco.csv")
    P.save(pd.concat(plans, ignore_index=True), "optimiser_plan.csv")

    tot_cur = reco.current_profit.sum()
    tot_opt = reco.optimised_profit.sum()
    print(f"[s06] optimised {len(reco)} {C.FOCAL_MANUFACTURER} CTA-PPG plans | "
          f"profit uplift: {(tot_opt/tot_cur-1)*100:.1f}% | "
          f"trade spend {reco.current_trade_spend.sum():,.0f} -> {reco.optimised_trade_spend.sum():,.0f}")
    return reco


if __name__ == "__main__":
    run()
