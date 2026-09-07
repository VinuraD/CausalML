"""Evaluate a fixed TestSet under one (format, parcorr route, z-transform) combination.

This is the unit of work the whole harness is built around. One call = one CELL, and a
cell is exactly reproducible from its key: the same TestSet digest, correlation matrix,
format and variants must produce bitwise-identical output, in this process or any other.
`tests/test_engine.py` asserts that across two separate interpreters.

The engine deliberately does NOT threshold. It returns `stat`, and thresholds are applied
afterwards by `metrics`. That is the alpha = 1.0 device from the SSDBM paper generalised:
record everything, decide later, so a change of alpha never requires recomputation and
never changes which tests were run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import checks, corr, formats, parcorr, stats
from .testset import TestSet


@dataclass(frozen=True)
class CellResult:
    """Per-test output for one cell. Arrays are parallel to the TestSet."""

    testset_digest: str
    rho: np.ndarray  # float64[T]
    stat: np.ndarray  # float64[T]
    status: np.ndarray  # uint8[T]
    kappa_proxy: np.ndarray  # float64[T]
    min_diag: np.ndarray  # float64[T]
    n_obs: int

    def __len__(self) -> int:
        return int(self.stat.shape[0])

    def status_counts(self) -> dict:
        out = {}
        for code, name in parcorr.STATUS_NAMES.items():
            c = int((self.status == np.uint8(code)).sum())
            if c:
                out[name] = c
        return out

    @property
    def n_ok(self) -> int:
        return int((self.status == parcorr.OK).sum())

    @property
    def n_undefined(self) -> int:
        """Tests whose decision is not well defined: NaN statistic.

        Distinct from "not rejected". Conflating the two is the single most likely way
        this harness produces a wrong scientific result, so it gets its own counter and
        `metrics.flip_counts` refuses to silently absorb it.
        """
        return int(np.isnan(self.stat).sum())


def evaluate(
    C: np.ndarray,
    ts: TestSet,
    n_obs: int,
    *,
    fmt: str = "float64",
    variant: str = "inv",
    ztransform: str = "log1p",
    rcond: float = 1e-15,
    allow_pinv: bool = True,
    keep_unit_diagonal: bool = True,
    check: bool = True,
) -> CellResult:
    """Run every test in `ts` against `C`, rounded to `fmt`.

    `C` is rounded ONCE, here, rather than per submatrix. Rounding is elementwise, so
    extract-then-round and round-then-extract are identical -- but rounding once means the
    symmetry and unit-diagonal conventions are decided in a single place and every test
    inherits them, instead of the convention drifting to per-test scope.
    """
    C = np.ascontiguousarray(C, dtype=np.float64)
    if check:
        checks.correlation_matrix(C, "engine.evaluate(input C)")
    if C.shape[0] != ts.n_vars:
        raise ValueError(
            "C is %dx%d but the TestSet is over %d variables"
            % (C.shape[0], C.shape[1], ts.n_vars)
        )

    Cr = formats.round_corr(C, fmt, keep_unit_diagonal=keep_unit_diagonal)
    checks.dtype_is(Cr, np.float64, "engine.evaluate(rounded C)")

    route = parcorr.get(variant)
    T = len(ts)
    rho = np.empty(T, dtype=np.float64)
    status = np.empty(T, dtype=np.uint8)
    kappa = np.empty(T, dtype=np.float64)
    mindiag = np.empty(T, dtype=np.float64)

    for t in range(T):
        S = ts.conditioning_set(t)
        sub = corr.submatrix(Cr, int(ts.i[t]), int(ts.j[t]), S)
        r, st, kp, md = route(sub, rcond=rcond, allow_pinv=allow_pinv)
        rho[t] = r
        status[t] = st
        kappa[t] = kp
        mindiag[t] = md

    # Vectorised statistic. Levels differ, so the sqrt(n - |S| - 3) scale is per test.
    lvl = ts.level.astype(np.float64)
    scale = n_obs - lvl - 3.0
    if np.any(scale <= 0):
        raise ValueError(
            "n_obs=%d is too small for level %d: n - |S| - 3 <= 0"
            % (n_obs, int(lvl[np.argmin(scale)]))
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = stats.z_transform(rho, ztransform)
        stat = np.sqrt(scale) * np.abs(z)

    return CellResult(
        testset_digest=ts.digest(),
        rho=rho,
        stat=stat,
        status=status,
        kappa_proxy=kappa,
        min_diag=mindiag,
        n_obs=int(n_obs),
    )
