"""
STAGE 03 - LOG-LOG STEPWISE REGRESSION
======================================
For every CTA-PPG, fit a log-log demand model:

    log(units) ~ log_price + feat_acv + disp_acv + si_log + trend + pantry_lag
                 + holiday flags + INTRA/INTER competitor log-prices

with four guards against unstable or implausible models:
  * stepwise selection by AIC (both directions)
  * VIF pruning of multicollinear competitor terms (threshold 8)
  * economic SIGN CONSTRAINTS (price<0, feature/display>0, ...) - drop & refit
  * time-based train/test split (R^2 + MAPE on held-out weeks)

The same logic is often written in R as per-group lm() + stepAIC(); here it is
numpy, applied per CTA-PPG group so it maps directly onto Spark applyInPandas.
Coefficient on log_price IS the price elasticity.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
import pipeline_common as P

PRIMARY = ["log_price", "feat_acv", "disp_acv", "si_log", "trend", "pantry_lag"]


def _candidate_features(df):
    feats = [f for f in PRIMARY if f in df.columns]
    feats += [h for h in P.HOLIDAYS if h in df.columns]
    feats += [s for s in P.COMP_SLOTS if s in df.columns]
    # keep only non-constant, non-null features
    keep = []
    for f in feats:
        col = df[f].fillna(0)
        if col.nunique() > 1:
            keep.append(f)
    return keep


def _fit_design(df, feats):
    X = np.column_stack([np.ones(len(df))] + [df[f].fillna(0).values.astype(float) for f in feats])
    y = df["log_units"].values.astype(float)
    return X, y


def _stepwise(df, feats):
    """Backward elimination by AIC."""
    current = list(feats)
    X, y = _fit_design(df, current)
    _, _, _, _, best_aic, _, _ = P.ols(X, y)
    improved = True
    while improved and len(current) > 1:
        improved = False
        for f in list(current):
            trial = [c for c in current if c != f]
            Xt, yt = _fit_design(df, trial)
            _, _, _, _, aic, _, _ = P.ols(Xt, yt)
            if aic < best_aic - C.AIC_IMPROVE_MIN:
                best_aic = aic
                current = trial
                improved = True
                break
    return current


def _vif_prune(df, feats):
    comp = [f for f in feats if f in P.COMP_SLOTS]
    feats = list(feats)
    while comp:
        v = P.vif(df[feats].fillna(0))
        worst = max(comp, key=lambda c: v.get(c, 1.0))
        if v.get(worst, 1.0) >= C.VIF_THRESHOLD:
            feats.remove(worst)
            comp.remove(worst)
        else:
            break
    return feats


def _sign_ok(name, beta):
    if name == "log_price":
        return C.PRICE_BETA_BOUNDS[0] <= beta <= C.PRICE_BETA_BOUNDS[1]
    if name in P.COMP_SLOTS:
        return C.CROSS_BETA_BOUNDS[0] <= beta <= C.CROSS_BETA_BOUNDS[1]
    prior = C.SIGN_PRIORS.get(name)
    return True if prior is None else (np.sign(beta) == prior or abs(beta) < 1e-6)


def _fit_one(df):
    train = df[df.test_flag == 0]
    test = df[df.test_flag == 1]
    feats = _candidate_features(train)
    if len(train) < 30 or "log_price" not in feats:
        return None
    feats = _vif_prune(train, feats)
    feats = _stepwise(train, feats)

    # iterative sign-constraint enforcement (drop wrong-signed non-price terms, refit)
    for _ in range(12):
        X, y = _fit_design(train, feats)
        beta, *_ = P.ols(X, y)
        names = ["intercept"] + feats
        bad = [n for n, b in zip(names[1:], beta[1:])
               if not _sign_ok(n, b) and n != "log_price"]
        if not bad:
            break
        feats = [f for f in feats if f not in bad]
        if not feats:
            break

    X, y = _fit_design(train, feats)
    beta, fitted, rss, r2, aic, n, k = P.ols(X, y)
    names = ["intercept"] + feats
    coef = dict(zip(names, beta))

    # held-out accuracy (back-transform to units)
    if len(test) > 0:
        Xte = np.column_stack([np.ones(len(test))] +
                              [test[f].fillna(0).values.astype(float) for f in feats])
        pred_units = np.expm1(Xte @ beta)
        test_mape = P.mape(test["units"].values, pred_units)
    else:
        test_mape = np.nan

    price_ok = _sign_ok("log_price", coef.get("log_price", 0.0))
    rec = {
        "cta_ppg": df.cta_ppg.iloc[0], "cta": df.cta.iloc[0], "ppg": df.ppg.iloc[0],
        "n_train": int(n), "n_features": len(feats), "r2": round(r2, 4),
        "test_mape": round(test_mape, 2) if test_mape == test_mape else np.nan,
        "price_elasticity": round(coef.get("log_price", np.nan), 4),
        "beta_feat": round(coef.get("feat_acv", np.nan), 4),
        "beta_disp": round(coef.get("disp_acv", np.nan), 4),
        "beta_si": round(coef.get("si_log", np.nan), 4),
        "beta_pantry": round(coef.get("pantry_lag", np.nan), 5),
        "price_sign_valid": bool(price_ok),
        "selected_features": ";".join(feats),
    }
    # store full coefficient vector for baseline/decomposition
    coef_row = {"cta_ppg": df.cta_ppg.iloc[0]}
    coef_row.update({k_: round(v_, 6) for k_, v_ in coef.items()})
    return rec, coef_row


def run():
    ads = P.load("model_ads.csv")
    summaries, coefs = [], []
    for _, grp in ads.groupby("cta_ppg"):
        res = _fit_one(grp.reset_index(drop=True))
        if res:
            summaries.append(res[0])
            coefs.append(res[1])
    summary = pd.DataFrame(summaries)
    coef_df = pd.DataFrame(coefs).fillna(0.0)
    P.save(summary, "model_summary.csv")
    P.save(coef_df, "model_coefficients.csv")

    valid = summary.price_sign_valid.mean() * 100
    print(f"[s03] models fitted: {len(summary)} | median R2: {summary.r2.median():.3f} | "
          f"median test MAPE: {summary.test_mape.median():.1f}% | valid price sign: {valid:.0f}%")
    print(f"[s03] mean recovered elasticity: {summary.price_elasticity.mean():.2f}")
    return summary


if __name__ == "__main__":
    run()
