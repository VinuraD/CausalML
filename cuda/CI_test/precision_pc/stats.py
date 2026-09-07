"""The Fisher-z conditional independence test.

The central design decision of this module, and the reason it exists rather than a
one-line helper: **the harness stores the test statistic, never the p-value.**

Three consequences, all load-bearing:

1. `2 * (1 - Phi(x))` is exactly 0.0 for x above about 8.3 in float64. A p-value that
   has flushed to zero can never flip, which biases every measured flip rate DOWNWARD --
   in the low-precision formats the paper is about. Storing `stat` eliminates the
   underflow structurally rather than by remembering to use the erfc form. (The erfc form
   is still what `pvalue` uses, and it survives to stat ~ 37; `log10p` goes further.)

2. The Fisher-z statistic is standard normal under the null, so the post-hoc threshold is
   a single constant, `crit(alpha) = Phi^-1(1 - alpha/2)`, INDEPENDENT of n and of |S|:

       p < alpha   <==>   stat > crit(alpha)

   exactly, for every test, at every level, at any alpha. That equivalence is what makes
   the "record everything, threshold afterwards" design work at all, and it is asserted
   in `tests/test_stats.py`.

3. Comparing `stat` rather than `p` keeps the resolution where the decision is made. Two
   tests whose p-values are both 0.0 may have very different statistics.

The z-transform variant and the normal-CDF implementation are named knobs rather than
hard-coded calls, because E0.3 needs them as experimental factors: the proposal expects
the spread between algorithmic variants at FP64 to be LARGER than the spread between
dtypes, and that cannot be measured if the variant is baked in.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc, erfcinv, ndtr

# Variants exposed as factors for E0.3.
Z_TRANSFORMS = ("log1p", "naive")
CDF_IMPLS = ("erfc", "ndtr")


# ------------------------------------------------------------------ z transform
def z_log1p(r):
    """atanh(r) via log1p. Accurate as |r| -> 1, which is where lagged variables live."""
    r = np.asarray(r, dtype=np.float64)
    return 0.5 * (np.log1p(r) - np.log1p(-r))


def z_naive(r):
    """atanh(r) as 0.5 * log((1+r)/(1-r)).

    The textbook form, and what a naive GPU port writes. It loses accuracy as |r| -> 1
    because (1 + r) and (1 - r) are each rounded before the division. E0.3 exists to
    measure how much of an apparent "precision effect" is really this formula effect.
    """
    r = np.asarray(r, dtype=np.float64)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


_Z = {"log1p": z_log1p, "naive": z_naive}


def z_transform(r, variant: str = "log1p"):
    if variant not in _Z:
        raise ValueError("unknown z transform " + repr(variant) + "; have " + repr(Z_TRANSFORMS))
    return _Z[variant](r)


# ------------------------------------------------------------------- statistic
def fisher_stat(rho, n: int, k: int, variant: str = "log1p"):
    """The Fisher-z test statistic sqrt(n - |S| - 3) * |z(rho)|.

    Returns +inf where |rho| >= 1 (the saturation case: FP8 rounds any |r| >= 0.96875 to
    exactly 1.0). +inf is the honest value -- it means "infinitely significant" -- and it
    compares correctly against any threshold. NaN input propagates to NaN, which the
    engine records as a status rather than silently treating as "not rejected".
    """
    rho = np.asarray(rho, dtype=np.float64)
    scale = n - k - 3.0
    if np.any(scale <= 0):
        raise ValueError(
            "n - |S| - 3 = " + repr(float(np.min(scale))) + " is not positive; "
            "the Fisher-z scaling is undefined (n too small for this conditioning set)"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = z_transform(rho, variant)
    return np.sqrt(scale) * np.abs(z)


# ------------------------------------------------------------------- threshold
def crit(alpha: float) -> float:
    """Two-sided critical value Phi^-1(1 - alpha/2). One constant for every test.

    Uses erfcinv rather than scipy.stats.norm.ppf so the inverse and the forward tail
    (`pvalue`, which uses erfc) come from the same function family and agree exactly at
    the boundary.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1), got " + repr(alpha))
    return float(math.sqrt(2.0) * erfcinv(alpha))


def rejects(stat, alpha: float):
    """Reject the null (declare DEPENDENT, keep the edge) iff stat > crit(alpha).

    NaN propagates as False here, which is why the engine must carry a status column:
    a NaN statistic is "undefined", not "independent", and conflating the two is the
    single most likely way this harness produces a wrong result.
    """
    return np.asarray(stat, dtype=np.float64) > crit(alpha)


# --------------------------------------------------------------------- p-value
def pvalue(stat, impl: str = "erfc"):
    """Two-sided p-value. For reporting and plots only -- never for decisions.

    erfc: p = erfc(stat / sqrt(2)). Survives to stat ~ 37 before underflowing.
          (scipy.special.erfc flushes subnormals, so it gives up a little earlier
          than math.erfc, which reaches ~40. Use log10p beyond that.)
    ndtr: p = 2 * (1 - Phi(stat)). Flushes to exactly 0.0 above stat ~ 8.3. Kept as a
          selectable implementation precisely so E0.3 can demonstrate the underflow, and
          so a reviewer can see the harness knows the difference.
    """
    s = np.asarray(stat, dtype=np.float64)
    if impl == "erfc":
        return erfc(s / math.sqrt(2.0))
    if impl == "ndtr":
        return 2.0 * (1.0 - ndtr(s))
    raise ValueError("unknown cdf impl " + repr(impl) + "; have " + repr(CDF_IMPLS))


def log10p(stat):
    """log10 of the two-sided p-value, computed without ever forming the p-value.

    Needed for plotting the deep tail, where p itself underflows even in the erfc form.
    Uses the asymptotic expansion log(erfc(x)) ~ -x^2 - log(x*sqrt(pi)) for large x.
    """
    s = np.asarray(stat, dtype=np.float64)
    out = np.empty_like(s)
    small = s < 25.0
    out[small] = np.log10(erfc(s[small] / math.sqrt(2.0)))
    x = s[~small] / math.sqrt(2.0)
    out[~small] = (-(x**2) - np.log(x * math.sqrt(math.pi))) / math.log(10.0)
    return out
