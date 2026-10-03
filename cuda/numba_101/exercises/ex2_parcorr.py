"""Exercises 2a, 2b.  Run:  python exercises/ex2_parcorr.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numba101  # noqa: E402,F401
import numpy as np

from numba101.part2_parcorr import (
    corr_with_condition_number,
    parcorr_chol,
    parcorr_inv,
    parcorr_recursive_1,
    parcorr_reference,
)


def ex2a():
    """All three routes agree to ~1e-12 on well-conditioned input."""
    print("--- 2a: three routes agree ---------------------------------------")
    rng = np.random.default_rng(0)
    C, kappa = corr_with_condition_number(3, 10.0, rng)
    a = parcorr_inv(np.ascontiguousarray(C))
    b = parcorr_chol(np.ascontiguousarray(C))
    c = parcorr_recursive_1(C[0, 1], C[0, 2], C[1, 2])
    print(f"kappa={kappa:.3g}  inv={a:.15f}  chol={b:.15f}  recursive={c:.15f}")
    assert abs(a - b) < 1e-12 and abs(a - c) < 1e-12, "fix this before moving on"


def ex2b():
    """The important one: error vs condition number, three routes."""
    print("--- 2b: conditioning sweep (E0.3) --------------------------------")
    # TODO(you):
    #   1. Sweep kappa over logspace(0, 8, 25). At each kappa draw several
    #      matrices (say 50) with different seeds - one sample per kappa is noise.
    #   2. For each matrix compute all three rho values plus parcorr_reference.
    #   3. Plot median |relative error| vs REALISED kappa, three curves, log-log.
    #      Save to figures/e03_conditioning.png.
    #   4. Expected ordering: recursive degrades first, explicit inverse next,
    #      Cholesky holds longest. Check whether you actually see that.
    #
    # The finding to write down: compare the SPREAD between these three curves
    # against the spread you get in Part 3 from changing dtype. If the method
    # matters more than the dtype, that reframes the whole study - and it is
    # exactly why E0.3 exists as a control.
    raise NotImplementedError("Exercise 2b")


if __name__ == "__main__":
    ex2a()
    ex2b()
