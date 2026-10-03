# Lesson 1 — Numba, by building the CI test

**Time:** ~2 hours. **Prerequisites:** Python, NumPy, and the partial-correlation formula. **No GPU required.**

Every exercise builds a piece you will actually use in E0.1–E1.2. Nothing here is a toy.

---

## Setup

```bash
pip install numba numpy scipy ml_dtypes
```

Put this at the very top of every script, **before importing NumPy**:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
```

This is not optional hygiene — it's experiment E0.1. Multithreaded BLAS varies floating-point reduction order between runs, so without it, two identical FP64 runs can disagree and you'll misattribute that to precision.

---

## Part 0 — The mental model (10 min, no code)

Numba compiles a Python function to machine code the first time you call it, specialised to the *types* of the arguments. Three consequences that explain almost every problem you'll hit:

1. **Compilation happens on first call**, so timing that call measures the compiler, not your code. Always call once to warm up before benchmarking.
2. **Only a subset of Python works** in `nopython` mode. Lists of arrays, dicts, and most object manipulation don't. Arrays, scalars, tuples, loops, and `math` functions do.
3. **Loops are fast, and vectorising is often unnecessary.** This inverts your NumPy instincts. In Numba you write the explicit `for` loop you'd have been taught to avoid.

Always use `@njit`, never bare `@jit`. `@njit` fails loudly if it can't compile; `@jit` silently falls back to slow object mode, and you won't notice.

---

## Part 1 — Fisher-z, and your first compile (20 min)

### 1.1 The baseline

```python
import numpy as np
from scipy.stats import norm

def fisher_z_numpy(r, n, k):
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(n - k - 3) * np.abs(z)
    return 2 * (1 - norm.cdf(stat))
```

### 1.2 The Numba version

`scipy.stats` doesn't work inside `@njit`. You need `math.erfc`, since the standard normal CDF is `Φ(x) = 0.5 * erfc(-x / √2)`.

```python
from numba import njit
import math

@njit(cache=True)
def fisher_z(r, n, k):
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    stat = math.sqrt(n - k - 3.0) * abs(z)
    return math.erfc(stat / math.sqrt(2.0))
```

**Exercise 1a.** Verify the identity `2 * (1 - Φ(x)) == erfc(x / √2)` numerically. Convince yourself the two-sided p-value collapses to a single `erfc` call.

**Exercise 1b.** Time both over 10⁶ scalar calls. Warm up first. Expect Numba to win by a large margin on scalars and NumPy to win on large vectorised arrays — understand *why* before moving on.

### 1.3 The first real finding

Rewrite the transform using `math.log1p`:

```python
z = 0.5 * (math.log1p(r) - math.log1p(-r))
```

**Exercise 1c.** For `r` from 0.9 to 0.999999, compare both formulations against a high-precision reference (`mpmath`, or `float128` if your platform has it). Plot relative error against `r`.

> This is **experiment E0.3**, and it's worth doing carefully. The GPU implementations use `log1p()` and `normcdf()` from the CUDA Math API. If naive `log((1+r)/(1-r))` diverges from `log1p` at high correlation — exactly where near-collinear lagged variables live — then part of what looks like a precision effect is actually a *formula* effect. You need to separate these before you can attribute anything to dtype.

---

## Part 2 — Partial correlation three ways (30 min)

For conditioning set `S`, build the correlation submatrix over `{i, j} ∪ S` and extract ρ(i,j|S).

### 2.1 Explicit inverse (what cuPC does)

```python
@njit(cache=True)
def parcorr_inv(sub):
    P = np.linalg.inv(sub)
    return -P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])
```

`np.linalg.inv` works in `@njit` on contiguous 2-D float arrays. Note that later, in CUDA kernels, it will **not** — you'll hand-write the small-matrix solve. Getting the CPU version right first gives you a reference to test against.

### 2.2 Cholesky solve

Write `parcorr_chol` using `np.linalg.cholesky` and forward/back substitution rather than forming the inverse.

### 2.3 Recursive formula

For `|S| = 1` there's a closed form. Implement it directly from the three pairwise correlations.

**Exercise 2a.** All three must agree to ~1e-12 on well-conditioned input. If they don't, fix it now.

**Exercise 2b — the important one.** Generate submatrices at controlled condition numbers (build from a chosen eigenvalue spectrum: `Q @ diag(λ) @ Q.T` with `Q` orthogonal, then normalise to unit diagonal). Sweep κ from 10⁰ to 10⁸ and plot each method's error against a `float128` or `mpmath` reference.

You should see the recursive formula degrade first, the explicit inverse next, and Cholesky hold longest. **Note the ordering — it may matter more than the choice of dtype**, which is precisely why E0.3 exists as a control.

---

## Part 3 — Simulating reduced precision (25 min)

This is the core trick that frees the whole study from needing a GPU.

```python
import ml_dtypes  # gives bfloat16, float8_e4m3fn, float8_e5m2 as numpy dtypes

def round_to(x, dtype):
    """Round to target format, return in float64 for computation."""
    return x.astype(dtype).astype(np.float64)
```

**Exercise 3a.** Round `np.float64(0.1)` through float32, bfloat16, and float8_e4m3fn. Print each with `repr` at full precision. Count how many decimal digits survive each.

**Exercise 3b.** Take a 3×3 correlation submatrix, round it to each format, invert in float64, and compare the resulting partial correlation to the unrounded result. Sweep κ.

**Exercise 3c — the decision-level question.** Convert those ρ differences into p-values and ask: at α = 0.01, does the decision change? Report the *flip rate*, not the numerical error.

> **This is the study in miniature.** If you can produce a flip-rate-versus-κ curve for one matrix size, you have the measurement apparatus for E1.1 and E1.2. Everything after this is scaling it up.

**Design note.** You are rounding the *inputs* and computing in float64. This models a machine whose storage is low-precision but whose accumulator is wide — which is how FP8 tensor-core matmul actually behaves. Modelling low-precision *accumulation* is a different experiment; note the distinction in your lab book now, because a reviewer will ask which one you measured.

---

## Part 4 — The skeleton loop (30 min)

Level 0 of PC-stable: test every pair for marginal independence.

```python
@njit(cache=True)
def level0(corr, n, alpha):
    N = corr.shape[0]
    adj = np.ones((N, N), dtype=np.uint8)
    np.fill_diagonal(adj, 0)
    pvals = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            p = fisher_z(corr[i, j], n, 0)
            pvals[i, j] = p
            pvals[j, i] = p
            if p > alpha:          # independent -> remove edge
                adj[i, j] = 0
                adj[j, i] = 0
    return adj, pvals
```

Note that `fisher_z` is called from inside another `@njit` function — Numba inlines it, with no Python call overhead. This composability is the main reason to prefer `@njit` over hand-vectorising.

**Exercise 4a.** Time against a NumPy-vectorised level 0 at N = 500, 1000, 2000.

**Exercise 4b — build the harness you'll actually use.** Note that `level0` returns **all p-values**, not just the adjacency matrix. That's deliberate. Run at α = 1.0 (nothing removed), store every p-value, and apply thresholds afterwards.

This is the α = 1.0 device from the SSDBM paper, repurposed. It gives you an exactly paired comparison across precisions — without it, a flip at level 0 changes the *test set* at level 1 and your pairing falls apart.

**Exercise 4c.** Write `flip_rate(pvals_a, pvals_b, alpha)` returning the fraction of tests where the two precisions disagree on the accept/reject decision. Confirm it returns exactly 0.0 for FP64 vs FP64 with identical input. **That's E0.1, and it must pass before anything else means anything.**

---

## Part 5 — Cheap condition numbers (15 min)

`np.linalg.cond` runs an SVD. Across millions of tests it will dominate your runtime.

**Exercise 5a.** Implement a proxy: since you're already doing a Cholesky, use the ratio of largest to smallest diagonal entry of the factor, squared.

**Exercise 5b.** Correlate the proxy against exact `cond` on 10,000 random submatrices at varying κ. Establish the calibration curve.

**Exercise 5c.** Time both. This ratio justifies the design choice in E1.2, and it's a one-line footnote in the paper that reviewers will appreciate.

---

## Checkpoint

By the end you should have, as working code:

- A Numba Fisher-z with the `log1p` fix and an error characterisation (E0.3)
- Three partial-correlation routes plus their conditioning behaviour (E0.3)
- A precision-rounding harness covering FP32 → FP8 (E1.1)
- A flip-rate function with a verified zero noise floor (E0.1)
- A cheap κ estimator with a calibration curve (E1.2)

That is Tier 0 and most of Tier 1, built while learning the tool.

---

## Learning sources, in the order I'd use them

**Primary — the Numba docs.** Start with "A ~5 minute guide to Numba," then the User Manual sections on `@njit`, supported Python features, and supported NumPy features. The supported-features pages are the ones you'll return to constantly; bookmark them. Then read "Troubleshooting and tips," specifically the part on typing errors — Numba's error messages are notoriously long, and ten minutes learning to read them saves hours.

**For the GPU part later — the Numba CUDA docs**, plus the official *CUDA C++ Programming Guide* chapters 1–5 for the execution model. Numba's CUDA target uses the same concepts with Python syntax, so the NVIDIA guide is still the right conceptual reference.

**Book — Gorelick & Ozsvald, *High Performance Python* (2nd ed., O'Reilly).** The profiling chapters are the valuable part; treat the Numba chapter as an introduction rather than a reference.

**Course — NVIDIA DLI, "Fundamentals of Accelerated Computing with CUDA Python."** Paid, and specifically Numba CUDA. Worth it if you prefer structured material, skippable if you're comfortable with docs.

**Code to read — the cuPC source** (`LIS-Laboratory/cupc` on GitHub). Not Numba, but it is the single most relevant codebase to this project. Read its CI-test kernel after Part 2 and you'll recognise everything.

**One caution.** Much Numba material online is about speeding up data-science loops. Your workload is small-matrix linear algebra with data-dependent control flow, which is a different regime. Trust measurement over blog-post rules of thumb.

---

## Lesson 2 preview

Batching: turning the `(i, j, S)` loop into an array of test descriptors so that all tests of a given `|S|` are processed with uniform shapes. That restructuring is what makes the eventual GPU port straightforward — and it's worth doing on CPU first, where debugging is easy.
