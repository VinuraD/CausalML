"""Partial correlation, three ways, each reporting how it went.

Convention: `sub` is the correlation submatrix over the ordered set {i, j} | S, so the
pair under test is at (0, 1) and the conditioning set occupies 2..m-1.

Every route returns the same 4-tuple:

    (rho, status, kappa_proxy, min_diag)

`status` is the load-bearing part and the reason these do not simply return a float.
At BF16 and FP8, elementwise rounding destroys positive-definiteness, Cholesky fails,
|rho| saturates to exactly 1.0, and the statistic becomes inf or NaN. A NaN statistic
compared against a threshold evaluates to False, i.e. "not rejected", i.e. *the same
decision as a well-behaved test that was not rejected either* -- so it would be counted
as "no flip". That biases the flip rate downward, systematically, in precisely the
formats this paper is about, and would manufacture a reassuring negative result. Carrying
a mandatory status makes that impossible to do by accident: `metrics.flip_counts` refuses
to fold a non-OK test into the "same decision" bucket.

`kappa_proxy` rides along on the Cholesky that has already been computed. np.linalg.cond
runs an SVD, which across millions of tests would dominate runtime; the ratio of the
largest to smallest diagonal entry of the Cholesky factor, squared, tracks kappa closely
enough to threshold on and costs nothing extra. E1.2 calibrates it against exact cond on
a subsample.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ status codes
# uint8, stored in every per-test row. Values are frozen: they appear in Parquet files.
OK = np.uint8(0)
RHO_UNIT = np.uint8(1)  # |rho| == 1 exactly -> stat = inf (the FP8 saturation case)
RHO_GT1 = np.uint8(2)  # |rho| > 1 -> the submatrix was not a valid correlation matrix
CHOL_FAIL = np.uint8(3)  # not positive definite
PINV_FALLBACK = np.uint8(4)  # singular; Moore-Penrose pseudo-inverse used instead
NONFINITE = np.uint8(5)  # NaN or inf appeared in the inputs or the solve
SINGULAR = np.uint8(6)  # detected singular and no fallback was permitted

STATUS_NAMES = {
    int(OK): "OK",
    int(RHO_UNIT): "RHO_UNIT",
    int(RHO_GT1): "RHO_GT1",
    int(CHOL_FAIL): "CHOL_FAIL",
    int(PINV_FALLBACK): "PINV_FALLBACK",
    int(NONFINITE): "NONFINITE",
    int(SINGULAR): "SINGULAR",
}

VARIANTS = ("inv", "chol", "regression")

_NAN = float("nan")


def _classify_rho(rho: float, status):
    """Fold the value of rho into the status. |rho| == 1 is not an error, but it is not OK."""
    if not np.isfinite(rho):
        return rho, NONFINITE
    a = abs(rho)
    if a > 1.0:
        return rho, RHO_GT1
    if a == 1.0:
        return rho, RHO_UNIT
    return rho, status


def cond_proxy_from_chol(L: np.ndarray) -> float:
    """Condition-number proxy: (max diag L / min diag L) ** 2.

    L is already computed by the Cholesky route, so this is free. It is a proxy, not the
    truth -- the calibration curve and the range over which it is monotone in the real
    cond are E1.2's job.
    """
    d = np.abs(np.diag(L))
    mn = d.min()
    if mn <= 0.0 or not np.isfinite(mn):
        return float("inf")
    return float((d.max() / mn) ** 2)


# ------------------------------------------------------- 1. explicit inverse (cuPC)
def parcorr_inv(sub: np.ndarray, rcond: float = 1e-15, allow_pinv: bool = True):
    """rho from the precision matrix: rho = -P[0,1] / sqrt(P[0,0] * P[1,1]).

    This is what cuPC does. The singularity fallback is a BRANCH, not a perturbation:
    reduced precision can change which code path executes, and E2.4 measures how often.
    """
    sub = np.ascontiguousarray(sub, dtype=np.float64)
    if not np.all(np.isfinite(sub)):
        return _NAN, NONFINITE, float("inf"), _NAN

    kappa, min_diag = _chol_diagnostics(sub)

    used_pinv = False
    try:
        if np.linalg.cond(sub) > 1.0 / rcond:
            raise np.linalg.LinAlgError("ill-conditioned beyond rcond")
        P = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        if not allow_pinv:
            return _NAN, SINGULAR, kappa, min_diag
        P = np.linalg.pinv(sub, rcond=rcond)
        used_pinv = True

    den = P[0, 0] * P[1, 1]
    if not np.isfinite(den) or den <= 0.0:
        return _NAN, NONFINITE, kappa, min_diag
    rho = float(-P[0, 1] / np.sqrt(den))
    rho, status = _classify_rho(rho, PINV_FALLBACK if used_pinv else OK)
    return rho, status, kappa, min_diag


def _chol_diagnostics(sub: np.ndarray):
    """(kappa_proxy, min_diag) from a Cholesky attempt, without raising."""
    try:
        L = np.linalg.cholesky(sub)
    except np.linalg.LinAlgError:
        return float("inf"), _NAN
    d = np.abs(np.diag(L))
    return cond_proxy_from_chol(L), float(d.min())


# ------------------------------------------------------------- 2. Cholesky solve
def parcorr_chol(sub: np.ndarray, rcond: float = 1e-15, allow_pinv: bool = True):
    """Same quantity without ever forming the full inverse.

    Only three entries of inv(sub) are needed -- (0,0), (0,1), (1,1) -- so solve
    sub @ x = e0 and sub @ x = e1 through the Cholesky factor. Numerically the best
    behaved of the three routes at high kappa, which is itself an E0.3 finding: the
    spread between ROUTES may exceed the spread between dtypes.
    """
    sub = np.ascontiguousarray(sub, dtype=np.float64)
    if not np.all(np.isfinite(sub)):
        return _NAN, NONFINITE, float("inf"), _NAN

    try:
        L = np.linalg.cholesky(sub)
    except np.linalg.LinAlgError:
        if not allow_pinv:
            return _NAN, CHOL_FAIL, float("inf"), _NAN
        rho, status, kappa, min_diag = parcorr_inv(sub, rcond=rcond, allow_pinv=True)
        # Report the real reason: Cholesky failed, and the fallback rescued it.
        return rho, CHOL_FAIL if status == OK else status, kappa, min_diag

    kappa = cond_proxy_from_chol(L)
    min_diag = float(np.abs(np.diag(L)).min())

    e = np.zeros((sub.shape[0], 2), dtype=np.float64)
    e[0, 0] = 1.0
    e[1, 1] = 1.0
    try:
        y = np.linalg.solve(L, e)
        x = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        return _NAN, SINGULAR, kappa, min_diag

    p00, p01, p11 = x[0, 0], x[0, 1], x[1, 1]
    den = p00 * p11
    if not np.isfinite(den) or den <= 0.0:
        return _NAN, NONFINITE, kappa, min_diag
    rho, status = _classify_rho(float(-p01 / np.sqrt(den)), OK)
    return rho, status, kappa, min_diag


# --------------------------------------------------------- 3. regression residuals
def parcorr_regression(sub: np.ndarray, rcond: float = 1e-15, allow_pinv: bool = True):
    """rho as the correlation of the residuals of i and j after regressing both on S.

    Algebraically identical to the other two, numerically a third path, and the one a
    statistician would write. For |S| = 0 it degenerates to the marginal correlation.
    """
    sub = np.ascontiguousarray(sub, dtype=np.float64)
    if not np.all(np.isfinite(sub)):
        return _NAN, NONFINITE, float("inf"), _NAN

    kappa, min_diag = _chol_diagnostics(sub)
    m = sub.shape[0]
    if m == 2:
        return _classify_rho(float(sub[0, 1]), OK) + (kappa, min_diag)

    Css = sub[2:, 2:]
    Cis = sub[0, 2:]
    Cjs = sub[1, 2:]
    try:
        if np.linalg.cond(Css) > 1.0 / rcond:
            raise np.linalg.LinAlgError("ill-conditioned beyond rcond")
        bi = np.linalg.solve(Css, Cis)
        bj = np.linalg.solve(Css, Cjs)
        status = OK
    except np.linalg.LinAlgError:
        if not allow_pinv:
            return _NAN, SINGULAR, kappa, min_diag
        Pss = np.linalg.pinv(Css, rcond=rcond)
        bi = Pss @ Cis
        bj = Pss @ Cjs
        status = PINV_FALLBACK

    # Residual (co)variances in correlation units.
    vii = 1.0 - Cis @ bi
    vjj = 1.0 - Cjs @ bj
    vij = sub[0, 1] - Cis @ bj
    den = vii * vjj
    if not np.isfinite(den) or den <= 0.0:
        return _NAN, NONFINITE, kappa, min_diag
    rho, status = _classify_rho(float(vij / np.sqrt(den)), status)
    return rho, status, kappa, min_diag


ROUTES = {
    "inv": parcorr_inv,
    "chol": parcorr_chol,
    "regression": parcorr_regression,
}


def get(variant: str):
    if variant not in ROUTES:
        raise ValueError("unknown parcorr variant " + repr(variant) + "; have " + repr(VARIANTS))
    return ROUTES[variant]
