"""Part 5 - Cheap condition numbers.

np.linalg.cond runs an SVD. Across millions of CI tests it dominates runtime, so
E1.2 needs a proxy that rides along on work you are already doing.

All yours.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def cond_proxy_chol(sub):
    """EXERCISE 5a. Condition-number proxy from a Cholesky factor.

    You are already computing L in parcorr_chol. The diagonal of L carries scale
    information: the ratio of its largest to smallest entry, squared, tracks the
    condition number of the matrix.

    Return that ratio squared. Do the Cholesky inside this function (or take L as
    an argument if you would rather not repeat the factorisation - decide which,
    and say why, when you get to Exercise 5c's timing).
    """
    raise NotImplementedError("Exercise 5a: Cholesky diagonal-ratio proxy")


def calibrate(m: int, n_matrices: int = 10_000, seed: int = 0):
    """EXERCISE 5b. Calibration curve: proxy vs exact np.linalg.cond.

    Generate n_matrices correlation submatrices spanning a wide kappa range
    (numba101.part2_parcorr.corr_with_condition_number), record both numbers, and
    report the relationship. Log-log is the space to look at it in.

    The useful summary is not "correlation = 0.98". It is: over what kappa range
    is the proxy monotone in the truth, and how large is the spread at fixed
    proxy value? That spread is what decides whether you can threshold on it.

    Returns (proxy_values, exact_values) as two float64 arrays.
    """
    raise NotImplementedError("Exercise 5b: calibration curve")
