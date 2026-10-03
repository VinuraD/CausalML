"""Exercises 1a, 1b, 1c.  Run:  python exercises/ex1_fisher_z.py"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numba101  # noqa: E402,F401  -- pins BLAS threads, MUST come before numpy
import numpy as np
from scipy.special import erfc
from scipy.stats import norm

import seaborn as sns

import matplotlib.pyplot as plt 

from numba101.part1_fisher_z import (
    fisher_z,
    fisher_z_log1p,
    fisher_z_numpy,
    z_log1p,
    z_naive,
    z_reference,
)
sns.set_style('whitegrid')

def ex1a():
    """Verify 2*(1 - Phi(x)) == erfc(x / sqrt(2)), and find where the baseline
    stops being able to represent its own answer."""
    print("--- 1a: identity check -------------------------------------------")
    x = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0])
    lhs = 2 * (1 - norm.cdf(x))
    rhs = erfc(x / np.sqrt(2.0))
    print(f"{'x':>5} {'2*(1-Phi(x))':>24} {'erfc(x/sqrt2)':>24}")
    for xi, a, b in zip(x, lhs, rhs):
        print(f"{xi:5.1f} {a:24.17e} {b:24.17e}")
    print()
    print("Two things to write down before moving on:")
    print("  1. Where does the factor 2 go? Why is ONE erfc call enough for a")
    print("     TWO-sided p-value?")
    print("  2. The identity is exact in real arithmetic, but the left column")
    print("     hits exactly 0.0 around x ~ 8.3 while the right column keeps")
    print("     going. Explain why - it is not the normal CDF that fails - and")
    print("     decide what that means for a study whose whole subject is")
    print("     precision. (Hint: what is the smallest number y for which")
    print("     1 - y != 1 in float64?)")
    print()
    print("Now the p-values themselves, at n = 20, k = 0:")
    for r in [0.0, 0.2, 0.5, 0.8, 0.9]:
        a = float(fisher_z_numpy(r, 20, 0))
        b = fisher_z(float(r), 20, 0)
        print(f"  r={r:<5} scipy={a:.17e}  erfc={b:.17e}  rel={abs(a - b) / a:.2e}")
    print()


def ex1b():
    """Time 1e6 scalar calls, njit vs numpy. Warm up first."""
    print("--- 1b: scalar timing --------------------------------------------")
    # TODO(you):
    #   1. Warm up: call fisher_z once and time THAT separately. The first call
    #      compiles; report the compile time as its own number.
    #   2. Time 1e6 scalar calls of each version (use time.perf_counter).
    #   3. Then time ONE vectorised fisher_z_numpy call on an array of 1e6 values.
    #   4. Explain the two results. Which cost is per-element and which is
    #      per-call? That distinction is the whole point of this exercise.
    start=time.perf_counter()
    fisher_z(0.5,20,0)
    end=time.perf_counter()
    print(f"Compile time for fisher_z: {end-start:.6f} seconds")

    r=np.random.rand(int(1e6))
    start=time.perf_counter()
    for i in r:
        fisher_z(i,20,0)
    end=time.perf_counter()
    print(f"Time for 1e6 scalar calls of fisher_z: {end-start:.6f} seconds")

    start=time.perf_counter()
    fisher_z_numpy(r,20,0)
    end=time.perf_counter()
    print(f"Time for vectorized call of fisher_z_numpy: {end-start:.6f} seconds")
    return
    # raise NotImplementedError("Exercise 1b")


def ex1c():
    """E0.3: naive log vs log1p against a high-precision reference."""
    print("--- 1c: log1p vs naive (E0.3) ------------------------------------")
    rs = 1.0 - np.logspace(-1, -6, 60)  # 0.9 -> 0.999999
    # TODO(you):
    #   1. For each r: ref = z_reference(r); compute z_naive(r) and z_log1p(r).
    #   2. Relative error of each against ref.
    #   3. Plot both on a log-log axis (x = 1 - r is the readable x axis) and
    #      save to figures/e03_log1p.png.
    #   4. Repeat at the p-value level with fisher_z vs fisher_z_log1p: does the
    #      z-transform error survive into the p-value, or does erfc's saturation
    #      hide it? The answer decides whether E0.3 matters downstream.
    # raise NotImplementedError("Exercise 1c")
    z_ref=np.array([z_reference(r) for r in rs])
    z_naive_vals=np.array([z_naive(r) for r in rs])
    z_log1p_vals=np.array([z_log1p(r) for r in rs])
    print('Relative error of z_naive vs z_reference:', np.sum(np.abs(z_naive_vals-z_ref)))
    print('Relative error of z_log1p vs z_reference:', np.sum(np.abs(z_log1p_vals-z_ref)))
    ax1=sns.scatterplot(x=1-rs,y=np.abs(z_naive_vals-z_ref),label='z_naive',color='blue')
    ax2=sns.scatterplot(x=1-rs,y=np.abs(z_log1p_vals-z_ref),label='z_log1p',color='orange')
    ax1.set(xscale='log',yscale='log')
    ax2.set(xscale='log',yscale='log')
    plt.xlabel('1 - r')
    plt.ylabel('Relative error')
    plt.savefig('figures/e03_log1p.png')

    z_ref=np.array([fisher_z_numpy(r,60,0) for r in rs]) 
    z_naive_vals=np.array([fisher_z(r,60,0) for r in rs])
    z_log1p_vals=np.array([fisher_z_log1p(r,60,0) for r in rs])
    print('Relative error of fisher_z vs z_reference:', np.sum(np.abs(z_naive_vals-z_ref)))
    print('Relative error of fisher_z_log1p vs z_reference:', np.sum(np.abs(z_log1p_vals-z_ref)))

    return



if __name__ == "__main__":
    ex1a()
    ex1b()
    ex1c()
