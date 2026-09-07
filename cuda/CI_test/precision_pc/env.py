"""Thread pinning and determinism control. Imported before numpy, always.

Why this is not hygiene but experiment E0.1: multithreaded BLAS chooses its reduction
order at runtime, so the same FP64 computation can give bit-different results on two
runs of the same binary. If the pinning is skipped, the noise floor of E0.1 is not zero
and every precision number measured afterwards is polluted -- it would be measuring
non-determinism, not precision.

The distinction this module insists on: setting an environment variable is a statement of
INTENT. `verify()` collects EVIDENCE, by asking the loaded BLAS libraries how many threads
they will actually use. Those differ whenever numpy was imported first, whenever a library
ignores the variable it was supposed to read, and whenever a cluster job inherited a
different setting. Only the evidence gates an experiment.
"""

from __future__ import annotations

import os
import sys
import warnings

#: Every thread-count variable that any BLAS/threading backend in this stack reads.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)

#: Set alongside the thread counts: dict/set iteration order can feed accumulation order.
DETERMINISM_VARS = {"PYTHONHASHSEED": "0"}

_pinned = False


def pin(check_import_order: bool = True) -> None:
    """Set every thread-count variable to 1. Call before importing numpy.

    If numpy is already imported the variables are still set, but a RuntimeWarning is
    raised: the BLAS backend reads them once, at load time, so the pinning has NOT taken
    effect for this process. `pytest.ini` promotes that warning to an error.
    """
    global _pinned
    if check_import_order and "numpy" in sys.modules and not _pinned:
        warnings.warn(
            "numpy was imported before precision_pc.env.pin(); BLAS thread counts are "
            "fixed at numpy import time, so this process is NOT pinned and the E0.1 "
            "noise floor cannot be trusted. Move `import precision_pc` to the top.",
            RuntimeWarning,
            stacklevel=2,
        )
    for var in THREAD_VARS:
        os.environ[var] = "1"
    for var, val in DETERMINISM_VARS.items():
        os.environ.setdefault(var, val)
    _pinned = True


def report() -> dict:
    """The thread-related environment as declared right now (intent, not evidence)."""
    out = {var: os.environ.get(var) for var in THREAD_VARS}
    out.update({var: os.environ.get(var) for var in DETERMINISM_VARS})
    return out


def threadpools() -> list:
    """What the loaded native libraries actually report (evidence, not intent).

    Returns a list of {user_api, internal_api, prefix, filepath, version, num_threads}.
    Empty list means threadpoolctl is unavailable, which is itself a failure to verify --
    callers must not read that as "verified".
    """
    try:
        import threadpoolctl
    except ImportError:
        return []
    return threadpoolctl.threadpool_info()


def verify() -> tuple[bool, list[str]]:
    """Check the pinning actually took. Returns (ok, list of human-readable problems).

    This is the check that gates an experiment. It is deliberately stricter than
    `report()`: a variable set to "1" that some library ignored still fails here.
    """
    problems: list[str] = []

    for var in THREAD_VARS:
        val = os.environ.get(var)
        if val != "1":
            problems.append(f"{var} is {val!r}, expected '1'")

    pools = threadpools()
    if not pools:
        problems.append(
            "threadpoolctl reported no native thread pools; cannot verify BLAS threading. "
            "Install threadpoolctl (it is a hard dependency) -- environment variables "
            "alone are not evidence."
        )
    for p in pools:
        n = p.get("num_threads")
        if n != 1:
            problems.append(
                f"{p.get('internal_api')} ({p.get('prefix')}) reports num_threads={n}, "
                f"expected 1 -- {p.get('filepath')}"
            )

    return (not problems), problems
