"""Part 1 - Fisher-z, and your first compile.

Given to you:  fisher_z_numpy (baseline), fisher_z (njit, naive log form).
Yours:         fisher_z_log1p, and the error study of Exercise 1c.
"""

import math

import numpy as np
from numba import njit
from scipy.stats import norm
from math import log1p
from mpmath import mp, mpf, log


# ---------------------------------------------------------------- 1.1 baseline
def fisher_z_numpy(r, n, k):
    """Two-sided p-value for H0: rho == 0, via the Fisher z transform.

    Works on scalars and arrays. This is the reference the njit version must match.
    """
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(n - k - 3) * np.abs(z)
    return 2 * (1 - norm.cdf(stat))


# ------------------------------------------------------------- 1.2 njit version
@njit(cache=True)
def fisher_z(r, n, k):
    """Same p-value, compiled. scipy.stats does not exist inside nopython mode,
    so the normal CDF is expressed through erfc:  2*(1 - Phi(x)) == erfc(x/sqrt(2)).
    """
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    stat = math.sqrt(n - k - 3.0) * abs(z)
    return math.erfc(stat / math.sqrt(2.0))


# ------------------------------------------------------- 1.3 the log1p variant
@njit(cache=True)
def fisher_z_log1p(r, n, k):
    """EXERCISE 1c. Identical to `fisher_z` except for the z transform, which
    must use math.log1p:

        z = 0.5 * (log1p(r) - log1p(-r))

    Hint on why: log(1 + r) loses the information in r's low bits when r is close
    to 1, because 1 + r is rounded before the log is taken. log1p is designed to
    keep it. Everything else in this function stays the same.
    """
    # raise NotImplementedError("Exercise 1c: implement the log1p form")
    z=0.5*(log1p(r)-log1p(-r))
    stat=math.sqrt(n-k-3.0)*abs(z)
    return math.erfc(stat/math.sqrt(2.0))


def z_reference(r, dps: int = 50):
    """EXERCISE 1c. High-precision reference for the z transform, 0.5*log((1+r)/(1-r)).

    numpy has no float128 on Windows, so use mpmath:

        from mpmath import mp, mpf, log
        
        ...

    Return a plain Python float (the mpmath value cast back down) so the caller
    can compute relative errors against the float64 implementations.
    """
    mp.dps = dps
    rfunc = np.frompyfunc(mpf, 1, 1)
    r_mpf = rfunc(r)
    arr=mpf(0.5) * (mp.log(mpf(1) + r_mpf) - mp.log(mpf(1) - r_mpf))
    return float(arr)
    # raise NotImplementedError("Exercise 1c: implement the mpmath reference")


def z_naive(r: float) -> float:
    """float64 naive transform, plain Python - for the error plot only."""
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def z_log1p(r: float) -> float:
    """float64 log1p transform, plain Python - for the error plot only.

    EXERCISE 1c: one line, same formula as fisher_z_log1p uses.
    """
    return 0.5 * (log1p(r) - log1p(-r))
