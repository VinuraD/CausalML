"""Exercises 3a, 3b, 3c.  Run:  python exercises/ex3_precision.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numba101  # noqa: E402,F401
import numpy as np

from numba101.part2_parcorr import corr_with_condition_number, parcorr_inv
from numba101.part3_precision import (
    FORMATS,
    digits_surviving,
    parcorr_across_formats,
    round_matrix,
    round_to,
)


def ex3a():
    """What survives a round trip through each format."""
    print("--- 3a: 0.1 through the precision ladder --------------------------")
    x = np.float64(0.1)
    for name, dt in FORMATS.items():
        y = float(round_to(x, dt))
        print(f"{name:<12} {y!r:<24} digits={digits_surviving(float(x), dt)}")
    print("Explain each digit count from the mantissa width of the format.\n")


def ex3b():
    """rho under rounding, swept over condition number."""
    print("--- 3b: rho vs format vs kappa (E1.1) ----------------------------")
    # TODO(you):
    #   For kappa in logspace(0, 8, 25), draw 3x3 correlation matrices, then
    #   parcorr_across_formats(C, parcorr_inv). Plot |rho_fmt - rho_float64|
    #   against realised kappa, one curve per format.
    #   Save to figures/e11_rho_error.png.
    raise NotImplementedError("Exercise 3b")


def ex3c():
    """The decision-level question - the one the paper actually asks."""
    print("--- 3c: flip rate vs kappa (E1.1) --------------------------------")
    # TODO(you):
    #   1. Convert each rho from 3b into a p-value with fisher_z(rho, n, k=1).
    #   2. At alpha = 0.01, does the accept/reject decision change relative to
    #      float64? Aggregate over the matrices at each kappa: that fraction is
    #      the flip rate.
    #   3. Plot flip rate vs kappa, one curve per format, and save to
    #      figures/e11_flip_rate.png.
    #
    #   Report the flip rate, NOT the numerical error. A 1e-3 error in rho that
    #   never crosses the threshold costs the algorithm nothing; a 1e-9 error
    #   that does costs it an edge. Once you have this curve for one matrix size
    #   you own the measurement apparatus for E1.1 and E1.2 - the rest is scale.
    #
    #   Also vary n (the sample size). n enters the test statistic through
    #   sqrt(n - k - 3), so it moves every p-value toward 0 as n grows. Flip rate
    #   at n = 100 and at n = 10000 are different experiments; say which you ran.
    raise NotImplementedError("Exercise 3c")


if __name__ == "__main__":
    ex3a()
    ex3b()
    ex3c()
