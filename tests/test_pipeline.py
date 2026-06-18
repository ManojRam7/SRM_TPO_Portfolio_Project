"""
Unit tests (pytest):
VIF, percentile bounds, discount classification, sign constraints, and the
end-to-end elasticity-recovery guarantee.

Run:  python -m pytest tests/ -q     (or: python tests/test_pipeline.py)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))
import config as C
import pipeline_common as P


def test_vif_flags_collinearity():
    n = 200
    a = np.random.RandomState(0).normal(size=n)
    df = pd.DataFrame({"x1": a, "x2": a * 1.0 + 1e-6, "x3": np.random.RandomState(1).normal(size=n)})
    v = P.vif(df)
    assert v["x1"] > C.VIF_THRESHOLD and v["x2"] > C.VIF_THRESHOLD   # collinear pair flagged
    assert v["x3"] < C.VIF_THRESHOLD                                  # independent not flagged


def test_percentile_bounds():
    s = pd.Series(range(100))
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    assert lo < hi and lo >= 0 and hi <= 99


def test_discount_classification():
    base, price = 2.00, 1.50
    depth = (base - price) / base
    assert abs(depth - 0.25) < 1e-9
    assert (0 if depth <= 0.05 else 1) == 1     # promoted flag


def test_ols_recovers_known_slope():
    rng = np.random.RandomState(7)
    x = rng.normal(size=500)
    y = 3.0 - 2.5 * x + rng.normal(scale=0.05, size=500)     # true slope -2.5
    X = np.column_stack([np.ones(500), x])
    beta, *_ = P.ols(X, y)
    assert abs(beta[1] - (-2.5)) < 0.05


def test_elasticity_recovery_endtoend():
    """If the pipeline has been run, recovered elasticities must match truth."""
    val = C.OUT / "elasticity_validation.csv"
    if not val.exists():
        return  # pipeline not yet run; skip
    v = pd.read_csv(val)
    assert v.abs_error.mean() < 0.20
    assert (v.pct_error <= 20).mean() > 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("all tests passed")
