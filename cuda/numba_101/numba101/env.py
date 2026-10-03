"""Single-thread pinning. MUST be imported before numpy, in every script.

Why this is not just hygiene: multithreaded BLAS chooses its reduction order
at runtime, so the same FP64 computation can give bit-different results on two
runs of the same binary. If you skip this, the noise floor of experiment E0.1
is not zero and every precision number you measure afterwards is polluted.

Usage (first two lines of any script):

    from numba101 import env; env.pin()   # or simply `import numba101`
    import numpy as np
"""

import os
import sys
import warnings

_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_pinned = False


def pin(check_import_order: bool = True) -> None:
    """Set every BLAS thread-count variable to 1.

    If numpy was already imported, the variables are set anyway but a warning is
    raised: the BLAS backend reads them once, at load time, so the pinning will
    not have taken effect for this process.
    """
    global _pinned
    if check_import_order and "numpy" in sys.modules and not _pinned:
        warnings.warn(
            "numpy was imported before numba101.env.pin(); BLAS thread counts "
            "are fixed at numpy import time, so this process is NOT pinned. "
            "Move the env import to the top of your script.",
            RuntimeWarning,
            stacklevel=2,
        )
    for var in _VARS:
        os.environ[var] = "1"
    _pinned = True


def report() -> dict:
    """Return the thread-related environment as seen right now (for the lab book)."""
    return {var: os.environ.get(var) for var in _VARS}
