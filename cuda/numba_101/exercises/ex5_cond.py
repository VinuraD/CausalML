"""Exercises 5a, 5b, 5c.  Run:  python exercises/ex5_cond.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numba101  # noqa: E402,F401
import numpy as np

from numba101.part2_parcorr import corr_with_condition_number
from numba101.part5_cond import calibrate, cond_proxy_chol


def ex5ab():
    print("--- 5a/5b: proxy calibration -------------------------------------")
    proxy, exact = calibrate(m=4, n_matrices=10_000, seed=0)
    # TODO(you): log-log scatter of proxy vs exact, save to
    # figures/e12_cond_calibration.png. Report the kappa range over which the
    # proxy is monotone in the truth, and the spread at fixed proxy value.
    raise NotImplementedError("Exercise 5b: the plot and the written finding")


def ex5c():
    print("--- 5c: proxy vs np.linalg.cond timing ---------------------------")
    # TODO(you): time cond_proxy_chol against np.linalg.cond over 100k small
    # matrices. Warm up the njit function first. The ratio is the one-line
    # footnote that justifies the design choice in E1.2.
    raise NotImplementedError("Exercise 5c")


if __name__ == "__main__":
    ex5ab()
    ex5c()
