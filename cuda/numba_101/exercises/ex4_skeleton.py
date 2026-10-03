"""Exercises 4a, 4b, 4c.  Run:  python exercises/ex4_skeleton.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numba101  # noqa: E402,F401
import numpy as np

from numba101.part4_skeleton import flip_rate, level0, level0_numpy


def random_corr(N, n_samples, seed=0):
    """Sample correlation matrix from n_samples draws of N independent variables.

    Independent by construction, so the true level-0 answer is 'no edges' and
    every rejection you see is a false positive - useful for sanity checks.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, N))
    return np.corrcoef(X, rowvar=False)


def ex4a():
    """njit level 0 vs vectorised NumPy level 0."""
    print("--- 4a: level 0 timing -------------------------------------------")
    # TODO(you): warm up level0 on a tiny matrix first (that call compiles it),
    # then time both versions at N = 500, 1000, 2000 with n = 1000 samples.
    # Predict the winner before you run it, then explain what you actually got.
    raise NotImplementedError("Exercise 4a")


def ex4b():
    """The alpha = 1.0 device."""
    print("--- 4b: alpha = 1.0 paired harness -------------------------------")
    corr = random_corr(200, 1000, seed=1)
    adj, pvals = level0(np.ascontiguousarray(corr), 1000, 1.0)
    assert adj.sum() == 200 * 199, "at alpha = 1.0 nothing may be removed"
    print(f"kept {adj.sum() // 2} of {200 * 199 // 2} edges, stored "
          f"{np.isfinite(pvals).sum() // 2} p-values")
    # TODO(you): build numba101.part4_skeleton.paired_pvalues on top of this -
    # one p-value matrix per precision format, all from the same test set.
    raise NotImplementedError("Exercise 4b")


def ex4c():
    """E0.1: the noise floor must be exactly zero."""
    print("--- 4c: flip rate noise floor (E0.1) -----------------------------")
    corr = np.ascontiguousarray(random_corr(200, 1000, seed=1))
    _, p1 = level0(corr, 1000, 1.0)
    _, p2 = level0(corr, 1000, 1.0)
    fr = flip_rate(p1, p2, 0.01)
    print(f"FP64 vs FP64 flip rate = {fr!r}")
    assert fr == 0.0, "E0.1 FAILED - nothing measured after this means anything"
    print("E0.1 passes. Record the numba/numpy versions and the thread pinning")
    print("from numba101.env.report() in the lab book alongside this result.")


if __name__ == "__main__":
    ex4c()  # run the noise floor first - it gates everything else
    ex4a()
    ex4b()
