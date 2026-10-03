"""Acceptance tests for Part 4 - including E0.1, the gate on everything else."""

import numpy as np
import pytest

from numba101.part4_skeleton import flip_rate, level0, level0_numpy


def _corr(N=60, n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N))
    return np.ascontiguousarray(np.corrcoef(X, rowvar=False))


def test_alpha_one_removes_nothing():
    """Exercise 4b: the pairing device only works if the test set is fixed."""
    C = _corr()
    adj, pvals = level0(C, 500, 1.0)
    N = C.shape[0]
    assert adj.sum() == N * (N - 1)
    assert np.all(np.diag(adj) == 0)
    iu = np.triu_indices(N, 1)
    assert np.all(np.isfinite(pvals[iu]))
    np.testing.assert_array_equal(pvals[iu], pvals.T[iu])


def test_numpy_and_njit_level0_agree():
    """Exercise 4a: same answers, different speed."""
    C = _corr()
    adj_a, p_a = level0(C, 500, 0.05)
    adj_b, p_b = level0_numpy(C, 500, 0.05)
    np.testing.assert_array_equal(adj_a, adj_b)
    iu = np.triu_indices(C.shape[0], 1)
    np.testing.assert_allclose(p_a[iu], p_b[iu], rtol=1e-12, atol=1e-300)


def test_e01_noise_floor_is_exactly_zero():
    """E0.1. Not 'approximately zero'. Exactly 0.0, twice in a row, or the whole
    study rests on nothing."""
    C = _corr()
    _, p1 = level0(C, 500, 1.0)
    _, p2 = level0(C, 500, 1.0)
    assert flip_rate(p1, p2, 0.01) == 0.0
    assert flip_rate(p1, p1, 0.05) == 0.0


def test_flip_rate_counts_upper_triangle_only():
    """Two tests out of three flipped -> 2/3, regardless of the diagonal."""
    a = np.array([[1.0, 0.001, 0.001], [0.001, 1.0, 0.5], [0.001, 0.5, 1.0]])
    b = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert flip_rate(a, b, 0.01) == pytest.approx(2.0 / 3.0)
