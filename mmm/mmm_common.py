"""
MMM shared methods (pure numpy):
  * elastic-net via coordinate descent + k-fold CV (same idea as R glmnet / cv.glmnet)
  * relaxed refit (OLS on selected features -> unbiased elasticities)
  * acting-item (cross-price) selection by correlation
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "src"))
import config_mmm as M


def _soft(z, g):
    return np.sign(z) * max(abs(z) - g, 0.0)


def _standardize(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _cd(Xs, yc, alpha, lam, max_iter=400, tol=1e-7):
    n, p = Xs.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        old = beta.copy()
        for j in range(p):
            r = yc - Xs @ beta + Xs[:, j] * beta[j]
            rho = Xs[:, j] @ r / n
            beta[j] = _soft(rho, lam * alpha) / (1.0 + lam * (1 - alpha))
        if np.max(np.abs(beta - old)) < tol:
            break
    return beta


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid); tss = float(((y - y.mean()) ** 2).sum())
    return beta, (1 - rss / tss if tss > 0 else 0.0)


def fit_elastic_net(df, feats, ycol="log_qty", alpha=None, lambdas=None, relaxed=None):
    """cv.glmnet-style: CV over lambda, fit at lambda.min, optional relaxed OLS refit."""
    alpha = M.ENET_ALPHA if alpha is None else alpha
    lambdas = M.ENET_LAMBDAS if lambdas is None else lambdas
    relaxed = M.RELAXED_REFIT if relaxed is None else relaxed
    feats = [f for f in feats if f in df.columns and df[f].fillna(0).nunique() > 1]
    X = df[feats].fillna(0).values.astype(float)
    y = df[ycol].values.astype(float)
    Xs, mu, sd = _standardize(X)
    yc = y - y.mean()

    # 5-fold CV to pick lambda.min
    n = len(y); idx = np.arange(n); rng = np.random.default_rng(1234); rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    best_lam, best_mse = lambdas[0], np.inf
    for lam in lambdas:
        mses = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            b = _cd(Xs[tr], yc[tr], alpha, lam)
            pred = Xs[f] @ b + y.mean()
            mses.append(np.mean((y[f] - pred) ** 2))
        if np.mean(mses) < best_mse:
            best_mse, best_lam = np.mean(mses), lam

    beta_s = _cd(Xs, yc, alpha, best_lam)
    selected = [feats[j] for j in range(len(feats)) if abs(beta_s[j]) > 1e-8]

    if relaxed and selected:
        Xr = np.column_stack([np.ones(n)] + [df[f].fillna(0).values for f in selected])
        b, r2 = _ols(Xr, y)
        coef = dict(zip(["intercept"] + selected, b))
    else:
        beta_o = beta_s / sd
        intercept = y.mean() - (beta_o * mu).sum()
        coef = {"intercept": intercept}
        coef.update({feats[j]: beta_o[j] for j in range(len(feats)) if abs(beta_s[j]) > 1e-8})
        pred = X @ beta_o + intercept
        r2 = 1 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)
    return {"coef": coef, "selected": selected, "lambda": best_lam, "r2": r2, "alpha": alpha}


def acting_item_filter(df, target, candidates, qty_col="log_qty",
                       price_col="log_price", cutoff=None):
    """Rank cross-price 'acting items' by |corr| with the target's volume."""
    cutoff = M.ACTING_ITEM_CORR_CUTOFF if cutoff is None else cutoff
    tgt = df[df.promo_group == target][["week_ending", qty_col]].rename(columns={qty_col: "y"})
    out = []
    for act in candidates:
        a = df[df.promo_group == act][["week_ending", price_col]].rename(columns={price_col: "p"})
        m = tgt.merge(a, on="week_ending", how="inner").dropna()
        if len(m) > 10:
            c = np.corrcoef(m.y, m.p)[0, 1]
            if abs(c) >= cutoff:
                out.append((act, round(float(c), 3)))
    return [a for a, _ in sorted(out, key=lambda t: -abs(t[1]))]


def mape(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float); m = a != 0
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else np.nan
