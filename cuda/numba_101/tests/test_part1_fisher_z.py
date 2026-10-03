"""Acceptance tests for Part 1. Red until you implement the log1p route."""

import numpy as np
import pytest
from scipy.special import erfc
from scipy.stats import norm

from numba101.part1_fisher_z import (
    fisher_z,
    fisher_z_log1p,
    fisher_z_numpy,
    z_log1p,
    z_naive,
    z_reference,
)

RS = [0.0, 0.2, 0.5, 0.8]


def test_the_identity_itself():
    """2*(1 - Phi(x)) == erfc(x / sqrt(2)), where float64 can represent both."""
    x = np.linspace(0.0, 5.0, 51)
    np.testing.assert_allclose(2 * (1 - norm.cdf(x)), erfc(x / np.sqrt(2.0)), rtol=1e-9)


@pytest.mark.parametrize("r", RS)
def test_njit_matches_the_numpy_baseline(r):
    """Agreement where `1 - cdf` has not yet lost its low bits to cancellation."""
    assert fisher_z(r, 20, 0) == pytest.approx(float(fisher_z_numpy(r, 20, 0)), rel=1e-11)


def test_erfc_form_survives_where_the_baseline_underflows():
    """The reason the njit version is not merely a faster baseline: `1 - cdf`
    saturates at exactly 0.0 near x = 8.3, erfc does not. A p-value that has
    been flushed to zero cannot flip - so this changes what a flip rate means."""
    assert float(fisher_z_numpy(0.9, 500, 0)) == 0.0
    assert fisher_z(0.9, 500, 0) > 0.0


@pytest.mark.parametrize("r", RS + [0.95, 0.999])
def test_log1p_agrees_at_moderate_r(r):
    """Away from r = 1 the two formulations must be indistinguishable."""
    assert fisher_z_log1p(r, 500, 0) == pytest.approx(fisher_z(r, 500, 0), rel=1e-12, abs=1e-300)


def test_log1p_is_never_less_accurate_near_one():
    """E0.3: at r -> 1, log1p must not lose to the naive form against a
    high-precision reference. If this fails, either the reference or the log1p
    route is wrong - do not average the failure away."""
    worse = []
    for r in 1.0 - np.logspace(-4, -8, 20):
        ref = z_reference(float(r))
        e_naive = abs(z_naive(float(r)) - ref) / abs(ref)
        e_log1p = abs(z_log1p(float(r)) - ref) / abs(ref)
        if e_log1p > e_naive:
            worse.append((r, e_naive, e_log1p))
    assert not worse, f"log1p was less accurate at {worse}"


def test_compiled_in_nopython_mode():
    """A silent fallback to object mode would make every timing meaningless."""
    fisher_z(0.5, 100, 0)
    assert fisher_z.nopython_signatures
