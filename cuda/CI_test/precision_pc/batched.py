"""Batched partial correlation in NumPy.

`engine.evaluate` loops over tests in Python. Measured against the same algorithm batched,
that loop costs **63-71x** at every problem size tried, and every laptop-side sweep in this
study has paid it.

The design constraint is not speed, it is **bitwise identity**. `code_digest` covers this
file, so a change that moved any result by one ulp would invalidate every cell already
computed and force the E0.1 gate to be re-established -- for a speedup. So the fast path
uses exactly the same LAPACK entry points as the scalar routes, on stacked arrays:
`np.linalg.cholesky`, `np.linalg.solve` and `np.linalg.inv` are all gufuncs that call the
same per-matrix kernel the scalar code calls, in the same order, on the same bytes.

What is NOT reimplemented is just as important. A hand-rolled batched Cholesky -- an
explicit i/j loop vectorised across the batch -- would be faster still, and would change
the summation order, and would therefore change results in the last ulp. That is the E0.3
"route matters" axis, and it is not a change to make silently in the name of throughput.

Degenerate elements are the other half. NumPy's stacked `cholesky` and `inv` raise for the
WHOLE batch if any element fails, with no per-element `info` (unlike torch's
`cholesky_ex`). Rather than guess which failed, the batch falls back to the scalar routes
for that bucket, which keeps the answer exactly right and gives up the speedup only where
the data is degenerate. `fallbacks` reports how often that happened, so the cost is
visible instead of mysterious.
"""

from __future__ import annotations

import numpy as np

from . import parcorr

#: Routes with a batched fast path. Others fall back to the scalar loop entirely.
BATCHABLE = ("inv", "chol")


def gather_submatrices(C: np.ndarray, index: np.ndarray) -> np.ndarray:
    """[M, m] variable indices -> [M, m, m] submatrices, no Python loop."""
    return C[index[:, :, None], index[:, None, :]]


def _rho_from_inverse_entries(p00, p01, p11):
    """rho = -P01 / sqrt(P00 * P11), with the same guards as the scalar routes."""
    den = p00 * p11
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = -p01 / np.sqrt(den)
    bad = ~np.isfinite(den) | (den <= 0)
    return rho, bad


def batch_chol(sub: np.ndarray, rcond: float = 1e-15):
    """The Cholesky route, batched. Returns (rho, status, kappa, min_diag) or None.

    None means at least one element is not positive definite, and the caller must use the
    scalar route for the whole bucket: NumPy's stacked cholesky raises for the batch
    without saying which element failed.
    """
    if not np.all(np.isfinite(sub)):
        return None
    try:
        L = np.linalg.cholesky(sub)
    except np.linalg.LinAlgError:
        return None

    m = sub.shape[1]
    d = np.abs(np.diagonal(L, axis1=1, axis2=2))
    dmin, dmax = d.min(axis=1), d.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(dmin > 0, (dmax / dmin) ** 2, np.inf)

    e = np.zeros((sub.shape[0], m, 2), dtype=np.float64)
    e[:, 0, 0] = 1.0
    e[:, 1, 1] = 1.0
    try:
        y = np.linalg.solve(L, e)
        x = np.linalg.solve(np.swapaxes(L, 1, 2), y)
    except np.linalg.LinAlgError:
        return None

    rho, bad = _rho_from_inverse_entries(x[:, 0, 0], x[:, 0, 1], x[:, 1, 1])
    status = _classify(rho, bad)
    return rho, status, kappa, dmin


def batch_inv(sub: np.ndarray, rcond: float = 1e-15):
    """The explicit-inverse route (what cuPC does), batched.

    Returns None when any element would take the pseudo-inverse branch. That branch is a
    CODE PATH change, not a perturbation -- E2.4 measures how often it fires -- so it is
    resolved by the scalar route rather than approximated here.
    """
    if not np.all(np.isfinite(sub)):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.linalg.cond(sub)
    if not np.all(np.isfinite(cond)) or np.any(cond > 1.0 / rcond):
        return None
    try:
        P = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        return None

    rho, bad = _rho_from_inverse_entries(P[:, 0, 0], P[:, 0, 1], P[:, 1, 1])
    status = _classify(rho, bad)

    # kappa_proxy and min_diag come from the same Cholesky the scalar route computes for
    # diagnostics, so the two paths report identical numbers.
    try:
        L = np.linalg.cholesky(sub)
    except np.linalg.LinAlgError:
        return None
    d = np.abs(np.diagonal(L, axis1=1, axis2=2))
    dmin, dmax = d.min(axis=1), d.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(dmin > 0, (dmax / dmin) ** 2, np.inf)
    return rho, status, kappa, dmin


def _classify(rho: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Status codes, in the same order of precedence as `parcorr._classify_rho`."""
    status = np.full(rho.shape, parcorr.OK, dtype=np.uint8)
    a = np.abs(rho)
    status = np.where(a > 1.0, parcorr.RHO_GT1, status)
    status = np.where(a == 1.0, parcorr.RHO_UNIT, status)
    status = np.where(~np.isfinite(rho) | bad, parcorr.NONFINITE, status)
    return status.astype(np.uint8)


_ROUTES = {"chol": batch_chol, "inv": batch_inv}


def evaluate_bucket(sub: np.ndarray, variant: str, rcond: float = 1e-15):
    """Try the batched route for one uniform-shape bucket. None means fall back."""
    fn = _ROUTES.get(variant)
    return None if fn is None else fn(sub, rcond=rcond)
