"""Shared helpers: OLS, AIC, VIF, IO. Pure numpy/pandas (no statsmodels needed)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C

INTRA_SLOTS = [f"INTRA_COMP_{i}_logprice" for i in range(1, 4)]
INTER_SLOTS = [f"INTER_COMP_{i}_logprice" for i in range(1, 4)]
COMP_SLOTS = INTRA_SLOTS + INTER_SLOTS
HOLIDAYS = list(C.HOLIDAY_UPLIFT.keys())


def load(name):
    return pd.read_csv(C.DATA / name) if (C.DATA / name).exists() else pd.read_csv(C.OUT / name)


def save(df, name, out=True):
    path = (C.OUT if out else C.DATA) / name
    df.to_csv(path, index=False)
    return path


def ols(X, y):
    """Ordinary least squares via lstsq. X already includes intercept col.
    Returns beta, fitted, rss, r2, aic, n, k."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else 0.0
    n, k = X.shape
    aic = n * np.log(rss / n + 1e-12) + 2 * k
    return beta, fitted, rss, r2, aic, n, k


def vif(df_feats):
    """VIF for each column = 1/(1-R^2) regressing it on the others."""
    cols = list(df_feats.columns)
    out = {}
    M = df_feats.values.astype(float)
    for j, c in enumerate(cols):
        others = [i for i in range(M.shape[1]) if i != j]
        if not others:
            out[c] = 1.0
            continue
        Xo = np.column_stack([np.ones(M.shape[0]), M[:, others]])
        yo = M[:, j]
        try:
            _, _, _, r2, *_ = ols(Xo, yo)
            out[c] = 1.0 / max(1e-6, (1 - r2))
        except Exception:
            out[c] = 1.0
    return out


def mape(actual, pred):
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = a != 0
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else np.nan
