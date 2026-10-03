"""Environment doctor. Run this first:  python check_env.py

Verifies the four things that silently break this lesson: thread pinning,
nopython compilation, the on-disk compile cache, and the ml_dtypes formats.
"""

import numba101  # noqa: F401  -- must precede numpy
import math
import time

import numpy as np
from numba import njit

import ml_dtypes
import numba
import scipy


@njit(cache=True)
def _warm(r):
    return math.erfc(0.5 * math.log1p(r) / math.sqrt(2.0))


def main():
    ok = True
    print("versions")
    print(f"  numpy {np.__version__}   numba {numba.__version__}   "
          f"scipy {scipy.__version__}   ml_dtypes {ml_dtypes.__version__}")

    print("thread pinning (E0.1 precondition)")
    for var, val in numba101.env.report().items():
        print(f"  {var:<24} {val}")
        ok &= val == "1"

    print("nopython compilation")
    t0 = time.perf_counter()
    _warm(0.5)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(100_000):
        _warm(0.5)
    t_run = time.perf_counter() - t0
    print(f"  first call (compile): {t_compile * 1e3:8.1f} ms")
    print(f"  100k calls after:     {t_run * 1e3:8.1f} ms  "
          f"({t_run / 100_000 * 1e9:.0f} ns/call)")
    print(f"  nopython signatures:  {_warm.nopython_signatures}")
    ok &= bool(_warm.nopython_signatures)

    print("njit + np.linalg (needed for Part 2)")

    @njit(cache=True)
    def _inv(a):
        return np.linalg.inv(a)

    @njit(cache=True)
    def _chol(a):
        return np.linalg.cholesky(a)

    A = np.ascontiguousarray(np.eye(3) * 2.0)
    print(f"  inv ok: {_inv(A)[0, 0] == 0.5}   cholesky ok: {_chol(A)[0, 0] == math.sqrt(2.0)}")

    print("compile cache")
    import pathlib
    cached = list(pathlib.Path(__file__).parent.rglob("__pycache__/*.nbi"))
    print(f"  {len(cached)} cache index file(s) written "
          f"(cache=True only works for functions defined in a FILE, not in a REPL)")

    print("low-precision formats")
    for name in ("float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"):
        dt = getattr(np, name, None) or getattr(ml_dtypes, name)
        print(f"  0.1 -> {name:<14} {float(np.array(0.1).astype(dt).astype(np.float64))!r}")

    print("high-precision reference")
    from mpmath import mp, mpf
    mp.dps = 50
    print(f"  mpmath dps=50: log(3) = {mp.log(mpf(3))}")
    print(f"  numpy float128 available: {hasattr(np, 'float128')} "
          f"(False on Windows - use mpmath)")

    print()
    print("ALL GOOD - start with exercises/ex1_fisher_z.py" if ok else
          "PROBLEMS ABOVE - fix before starting")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
