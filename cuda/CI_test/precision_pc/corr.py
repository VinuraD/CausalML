"""Correlation matrix construction, with the rounding conventions made explicit.

Every decision in this module is one a reviewer will challenge, so each is a named
argument with a recorded default rather than an implicit choice buried in a formula.

The important one is exact symmetry. `np.corrcoef` does not guarantee that C[i, j] and
C[j, i] are bitwise equal -- they come from different reduction orders and can differ in
the last ulp. That matters more than it sounds: if the matrix is only approximately
symmetric, then elementwise rounding to a low-precision format can round the two
triangles DIFFERENTLY, Cholesky then fails on a matrix that should have been fine, and
the harness measures its own bug instead of the format. Symmetrising exactly once, here,
means every rung of the ladder inherits exact symmetry for free.
"""

from __future__ import annotations

import numpy as np

from . import checks

#: Where in the pipeline the low-precision rounding is applied. Recorded per cell.
ROUND_STAGES = ("corr", "data", "both")


def symmetrise(C: np.ndarray) -> np.ndarray:
    """Force exact bitwise symmetry by mirroring the upper triangle onto the lower.

    Mirroring, not averaging: (C + C.T) / 2 is symmetric in exact arithmetic but the
    division reintroduces rounding, and it changes every entry rather than half of them.
    Copying one triangle is exact and touches nothing it does not have to.
    """
    C = np.array(C, dtype=np.float64, copy=True, order="C")
    iu = np.triu_indices_from(C, k=1)
    C[(iu[1], iu[0])] = C[iu]
    return C


def correlation(
    X: np.ndarray,
    *,
    keep_unit_diagonal: bool = True,
    clip_to_unit: bool = True,
) -> np.ndarray:
    """Sample correlation matrix of columns of X, exactly symmetric, unit diagonal.

    Parameters
    ----------
    keep_unit_diagonal
        Force the diagonal to exactly 1.0. It should already be 1.0 up to rounding;
        forcing it removes a source of Cholesky failure that has nothing to do with the
        experiment.
    clip_to_unit
        Clamp off-diagonal entries into [-1, 1]. Floating-point error can push a sample
        correlation a few ulp outside, which makes atanh return NaN for a reason that is
        not the format under test.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (observations x variables), got shape " + repr(X.shape))
    C = np.corrcoef(X, rowvar=False)
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    C = symmetrise(C)
    if clip_to_unit:
        np.clip(C, -1.0, 1.0, out=C)
    if keep_unit_diagonal:
        np.fill_diagonal(C, 1.0)
    C = np.ascontiguousarray(C)
    checks.correlation_matrix(C, "corr.correlation")
    return C


def submatrix(C: np.ndarray, i: int, j: int, S) -> np.ndarray:
    """The correlation submatrix over the ordered set {i, j} | S.

    Convention used everywhere in this harness: the pair under test is at indices (0, 1)
    and the conditioning set occupies 2..m-1. `np.ascontiguousarray` is not optional --
    fancy indexing gives a non-contiguous view and Numba's typing rejects it.
    """
    idx = np.empty(2 + len(S), dtype=np.intp)
    idx[0] = i
    idx[1] = j
    idx[2:] = S
    return np.ascontiguousarray(C[np.ix_(idx, idx)])


def with_condition_number(m: int, kappa: float, rng: np.random.Generator):
    """Random m x m correlation matrix with approximately the requested condition number.

    Build from a chosen eigenvalue spectrum, then normalise to unit diagonal:

        lambda = logspace(0, log10 kappa, m), rescaled to sum to m
        A = Q diag(lambda) Q.T  with Q a random orthogonal matrix
        C = D A D              with D = diag(1 / sqrt(diag A))

    The normalisation in the last step CHANGES the spectrum, so the realised condition
    number is not the one that was asked for. Both are returned, and every plot must use
    the realised value -- that is one of the recorded reviewer-facing decisions.
    """
    if m < 2:
        raise ValueError("m must be at least 2, got " + repr(m))
    lam = np.logspace(0.0, np.log10(kappa), m)
    lam = lam * (m / lam.sum())
    # QR of a Gaussian matrix, with the sign convention fixed so the result is uniform
    # on O(m) (the raw QR from LAPACK is not).
    A0 = rng.standard_normal((m, m))
    Q, R = np.linalg.qr(A0)
    Q = Q * np.sign(np.diag(R))
    A = (Q * lam) @ Q.T
    d = 1.0 / np.sqrt(np.diag(A))
    C = d[:, None] * A * d[None, :]
    C = symmetrise(C)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)
    C = np.ascontiguousarray(C)
    checks.correlation_matrix(C, "corr.with_condition_number")
    return C, float(np.linalg.cond(C))
