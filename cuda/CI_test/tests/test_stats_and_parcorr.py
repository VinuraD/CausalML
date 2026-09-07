"""Steps 2, 4, 6: the statistic, the threshold equivalence, and the three parcorr routes."""

import math

import numpy as np
import pytest

from precision_pc import corr, parcorr, stats


# ------------------------------------------------------------------- the statistic
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
def test_threshold_equivalence_is_exact(alpha):
    """p < alpha  <==>  stat > crit(alpha), for a single constant crit.

    This is the property the whole record-everything-threshold-later design rests on.
    It is what makes alpha a post-hoc knob rather than a re-run, and it holds
    independently of n and |S| because the Fisher-z statistic is standard normal.
    """
    s = np.random.default_rng(0).uniform(0.0, 12.0, 200_000)
    assert np.array_equal(stats.pvalue(s) < alpha, s > stats.crit(alpha))


def test_erfc_survives_where_ndtr_underflows():
    """The reason the harness stores `stat` and never `p`.

    2*(1 - Phi(x)) is exactly 0.0 above x ~ 8.3. A p-value of 0 can never flip, so
    storing p would bias every flip rate downward -- in the low-precision formats the
    paper is about.
    """
    assert stats.pvalue(8.3, impl="ndtr") == 0.0
    assert stats.pvalue(8.3, impl="erfc") > 0.0
    assert stats.pvalue(37.0, impl="erfc") > 0.0
    # Past ~37.5 even erfc flushes to zero in scipy, which is why log10p exists.
    assert stats.log10p(np.array([60.0]))[0] < -700


def test_log10p_extends_past_float_underflow():
    lp = stats.log10p(np.array([8.0, 30.0, 100.0]))
    assert lp[0] < -14 and lp[1] < -190 and lp[2] < -2000
    assert np.all(np.diff(lp) < 0)


def test_saturated_correlation_gives_infinite_statistic():
    """|rho| == 1 is what FP8 does to a strongly correlated pair. inf is the honest
    answer -- it compares correctly against any threshold -- and it must not be NaN."""
    s = stats.fisher_stat(np.array([1.0, -1.0]), 200, 1)
    assert np.all(np.isinf(s)) and np.all(s > 0)


def test_log1p_beats_naive_z_near_unit_correlation():
    """E0.3 in miniature: the FORMULA matters before the dtype does."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    r = 1.0 - 1e-13
    ref = float(mp.atanh(mp.mpf(r)))
    e_log1p = abs(float(stats.z_log1p(r)) - ref) / abs(ref)
    e_naive = abs(float(stats.z_naive(r)) - ref) / abs(ref)
    assert e_log1p <= e_naive


def test_fisher_stat_rejects_insufficient_samples():
    with pytest.raises(ValueError, match="not positive"):
        stats.fisher_stat(np.array([0.5]), n=5, k=3)


# --------------------------------------------------------------- partial correlation
def _well_conditioned(seed=0, m=4):
    rng = np.random.default_rng(seed)
    C, _ = corr.with_condition_number(m, 10.0, rng)
    return C


@pytest.mark.parametrize("seed", range(5))
def test_three_routes_agree_when_well_conditioned(seed):
    C = _well_conditioned(seed)
    vals = [parcorr.get(v)(C)[0] for v in parcorr.VARIANTS]
    for v in vals:
        assert abs(v - vals[0]) < 1e-12


def test_routes_agree_with_high_precision_reference():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    C = _well_conditioned(1)
    P = mp.matrix(C.tolist()) ** -1
    ref = float(-P[0, 1] / mp.sqrt(P[0, 0] * P[1, 1]))
    for v in parcorr.VARIANTS:
        assert abs(parcorr.get(v)(C)[0] - ref) < 1e-10


def test_singular_matrix_reports_status_instead_of_crashing():
    """A perfectly collinear pair. Every route must return a status, not raise and not
    silently produce a number that looks fine."""
    S = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
    for v in parcorr.VARIANTS:
        rho, status, kappa, min_diag = parcorr.get(v)(S)
        assert status != parcorr.OK, "%s reported OK on a singular matrix" % v


def test_nonfinite_input_is_caught():
    S = np.array([[1.0, np.nan], [np.nan, 1.0]])
    for v in parcorr.VARIANTS:
        rho, status, _, _ = parcorr.get(v)(S)
        assert status == parcorr.NONFINITE
        assert np.isnan(rho)


def test_kappa_proxy_tracks_exact_condition_number():
    """E1.2: the proxy rides on the Cholesky that is computed anyway. np.linalg.cond runs
    an SVD, which across millions of tests would dominate runtime. What matters is not a
    tight correlation but MONOTONICITY over the range that gets thresholded."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(7)
    proxy, exact = [], []
    for _ in range(400):
        kappa = 10 ** rng.uniform(0, 8)
        C, realised = corr.with_condition_number(4, kappa, rng)
        p = parcorr.parcorr_chol(C)[2]
        if np.isfinite(p) and p > 0:
            proxy.append(p)
            exact.append(realised)
    rho = spearmanr(np.log10(proxy), np.log10(exact)).statistic
    assert rho > 0.9, "kappa proxy is not monotone in the truth (spearman=%.3f)" % rho


def test_pinv_fallback_is_a_recorded_branch():
    """E2.4: reduced precision can change WHICH CODE PATH runs, not merely a value.
    The fallback must be visible in the status, never silent."""
    rng = np.random.default_rng(8)
    C, _ = corr.with_condition_number(4, 1e17, rng)
    rho, status, _, _ = parcorr.parcorr_inv(C, rcond=1e-15, allow_pinv=True)
    rho2, status2, _, _ = parcorr.parcorr_inv(C, rcond=1e-15, allow_pinv=False)
    assert status in (parcorr.PINV_FALLBACK, parcorr.RHO_UNIT, parcorr.RHO_GT1, parcorr.NONFINITE)
    assert status2 in (parcorr.SINGULAR, parcorr.RHO_UNIT, parcorr.RHO_GT1, parcorr.NONFINITE)
