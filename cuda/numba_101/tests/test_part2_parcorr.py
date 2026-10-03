"""Acceptance tests for Part 2."""

import numpy as np
import pytest

from numba101.part2_parcorr import (
    corr_with_condition_number,
    parcorr_chol,
    parcorr_inv,
    parcorr_recursive_1,
    parcorr_reference,
)


@pytest.fixture
def well_conditioned():
    rng = np.random.default_rng(7)
    C, _ = corr_with_condition_number(3, 5.0, rng)
    return np.ascontiguousarray(C)


def test_generator_returns_valid_correlation_matrix():
    rng = np.random.default_rng(3)
    C, kappa = corr_with_condition_number(4, 1e3, rng)
    assert C.shape == (4, 4)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)
    np.testing.assert_allclose(C, C.T, atol=1e-15)
    assert np.linalg.eigvalsh(C).min() > 0, "must be positive definite"
    assert np.isfinite(kappa) and kappa > 1.0


def test_three_routes_agree(well_conditioned):
    """Exercise 2a: ~1e-12 on well-conditioned input."""
    C = well_conditioned
    a = parcorr_inv(C)
    b = parcorr_chol(C)
    c = parcorr_recursive_1(C[0, 1], C[0, 2], C[1, 2])
    assert b == pytest.approx(a, abs=1e-12)
    assert c == pytest.approx(a, abs=1e-12)


def test_independence_gives_zero_partial_correlation():
    """If i and j correlate only through k, rho(i,j|k) must be 0."""
    r_ik, r_jk = 0.6, 0.5
    C = np.array([[1.0, r_ik * r_jk, r_ik],
                  [r_ik * r_jk, 1.0, r_jk],
                  [r_ik, r_jk, 1.0]])
    assert parcorr_inv(np.ascontiguousarray(C)) == pytest.approx(0.0, abs=1e-14)
    assert parcorr_chol(np.ascontiguousarray(C)) == pytest.approx(0.0, abs=1e-14)
    assert parcorr_recursive_1(r_ik * r_jk, r_ik, r_jk) == pytest.approx(0.0, abs=1e-14)


def test_reference_is_tighter_than_float64(well_conditioned):
    """The mpmath reference must be stable when you raise the precision."""
    C = well_conditioned
    assert parcorr_reference(C, dps=60) == pytest.approx(parcorr_reference(C, dps=120), abs=1e-25)
