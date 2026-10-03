"""Acceptance tests for Part 5."""

import numpy as np
import pytest

from numba101.part2_parcorr import corr_with_condition_number
from numba101.part5_cond import calibrate, cond_proxy_chol


def test_proxy_is_one_for_the_identity():
    assert cond_proxy_chol(np.ascontiguousarray(np.eye(4))) == pytest.approx(1.0, rel=1e-12)


def test_proxy_tracks_the_truth_monotonically():
    """Not equality - the proxy is a proxy. It must order matrices the same way
    np.linalg.cond does, over the kappa range you intend to use it in."""
    rng = np.random.default_rng(11)
    pairs = []
    for kappa in np.logspace(0, 6, 30):
        C, exact = corr_with_condition_number(4, kappa, rng)
        pairs.append((cond_proxy_chol(np.ascontiguousarray(C)), exact))
    proxy, exact = map(np.array, zip(*pairs))
    rho = np.corrcoef(np.log10(proxy), np.log10(exact))[0, 1]
    assert rho > 0.9, f"log-log correlation only {rho:.3f} - the proxy is not usable"


def test_calibrate_returns_paired_arrays():
    proxy, exact = calibrate(m=3, n_matrices=200, seed=0)
    assert proxy.shape == exact.shape == (200,)
    assert np.all(np.isfinite(proxy)) and np.all(np.isfinite(exact))
