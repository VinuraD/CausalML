"""Stage-boundary assertions.

Every one of these corresponds to a way a result could be silently wrong rather than
loudly broken. The proposal names two of them explicitly (NEP 50 dtype promotion; exact
symmetry under rounding); the rest are the invariants the rest of the harness relies on.

These raise. They are not warnings, and they are not recovered from by substituting a
default -- a silently coerced dtype or a silently unsymmetric matrix produces a number
that looks fine and is wrong.
"""

from __future__ import annotations

import numpy as np


class CheckFailed(AssertionError):
    """A stage-boundary invariant was violated."""


def dtype_is(a: np.ndarray, expected, where: str) -> None:
    """NumPy 2.x changed promotion rules (NEP 50). Assert dtypes at every boundary."""
    if a.dtype != np.dtype(expected):
        raise CheckFailed(
            f"{where}: dtype is {a.dtype}, expected {np.dtype(expected)}. "
            "NumPy 2.x promotion (NEP 50) can silently change this."
        )


def is_finite(a: np.ndarray, where: str) -> None:
    if not np.all(np.isfinite(a)):
        n = int((~np.isfinite(a)).sum())
        raise CheckFailed(f"{where}: {n} non-finite value(s) of {a.size}")


def is_square(a: np.ndarray, where: str) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise CheckFailed(f"{where}: expected a square 2-D array, got shape {a.shape}")


def is_bitwise_symmetric(a: np.ndarray, where: str) -> None:
    """Bitwise, not `allclose`.

    Elementwise rounding preserves symmetry only if the matrix was EXACTLY symmetric to
    begin with: round(x) == round(x) trivially, but round(x) != round(y) for x != y that
    merely agree to 1e-16. Enforcing exact symmetry once, here, means every format in the
    ladder inherits it for free and Cholesky never fails for a reason we invented.
    """
    is_square(a, where)
    if a.tobytes() != np.ascontiguousarray(a.T).tobytes():
        d = np.abs(a - a.T)
        raise CheckFailed(
            f"{where}: matrix is not BITWISE symmetric (max |C - C.T| = {d.max():.3e}). "
            "Symmetrise exactly before rounding."
        )


def has_unit_diagonal(a: np.ndarray, where: str) -> None:
    """Exactly 1.0, not approximately. 1.0 is representable in every format in the ladder."""
    is_square(a, where)
    d = np.diag(a)
    if not np.all(d == 1.0):
        bad = int((d != 1.0).sum())
        raise CheckFailed(
            f"{where}: {bad} diagonal entr(ies) are not exactly 1.0 "
            f"(max deviation {np.max(np.abs(d - 1.0)):.3e})"
        )


def in_unit_interval(a: np.ndarray, where: str, tol: float = 0.0) -> None:
    """Correlation entries must lie in [-1, 1]. Rounding can push them outside."""
    mx = float(np.nanmax(np.abs(a)))
    if mx > 1.0 + tol:
        raise CheckFailed(f"{where}: max |entry| = {mx!r} exceeds 1 (tol={tol})")


def correlation_matrix(a: np.ndarray, where: str, dtype=np.float64) -> None:
    """The full contract for a correlation matrix entering the engine."""
    dtype_is(a, dtype, where)
    is_square(a, where)
    is_bitwise_symmetric(a, where)
    has_unit_diagonal(a, where)
    in_unit_interval(a, where)
