# numba_101 — Lesson 1 workspace

Scaffold for [`../numba_lesson_01.md`](../numba_lesson_01.md): *Numba, by building the CI test*.

Everything the lesson hands you verbatim is already here and working. Everything
the lesson asks you to derive is a function with a docstring, the hints you need,
and a `NotImplementedError`. The test suite is the spec: **the lesson is finished
when `pytest` is green.**

---

## 0. Environment

Uses the existing `causal` conda env (Python 3.12). `ml_dtypes` and `pytest` have
been added to it; numba 0.67, numpy 2.4.6 and scipy 1.15 were already present.

```powershell
conda activate causal
cd D:\Github\CausalML\cuda\numba_101
python check_env.py
```

`check_env.py` verifies the four things that silently break this lesson:

| check | why it matters |
|---|---|
| BLAS pinned to 1 thread | multithreaded reductions change FP64 results between runs, so the E0.1 noise floor is not zero |
| `nopython_signatures` non-empty | `@jit` falling back to object mode looks like working code and is ~100x slower |
| `.nbi` cache files written | `cache=True` only works for functions defined in a **file**, never in a REPL or `python -c` |
| `ml_dtypes` formats round-trip | the whole precision study depends on these dtypes existing |

Two platform facts this env confirms, both worth knowing now:

- **`np.float128` does not exist on Windows.** Use `mpmath` for every
  high-precision reference in this lesson. `check_env.py` prints this.
- **First-call compile cost is real** — around 12 s for the first `@njit`
  function in a fresh process, then ~250 ns/call. Warm up before timing, always.

## 1. Layout

```
check_env.py            environment doctor - run this first
numba101/               the library you are building (import this package FIRST)
  env.py                BLAS thread pinning; warns if numpy beat it to the import
  part1_fisher_z.py     Fisher-z, erfc form, log1p variant           [1c yours]
  part2_parcorr.py      partial correlation three ways               [2.2, 2.3, 2b yours]
  part3_precision.py    the FP32 -> FP8 rounding harness             [3a, 3b yours]
  part4_skeleton.py     PC-stable level 0, flip rate, alpha=1 device [4a, 4b, 4c yours]
  part5_cond.py         cheap condition-number proxy                 [5a, 5b yours]
exercises/ex*.py        runnable drivers, one per Part - the TODOs live here too
tests/                  the acceptance criteria (this is the "CI test" of the title)
figures/                where the plots go
LAB_BOOK.md             findings, decisions, and the things a reviewer will ask about
```

## 2. Working order

```powershell
python check_env.py                              # green before anything else
python -m pytest                                 # see the whole spec, all red
python exercises\ex1_fisher_z.py                 # then work Part by Part
python -m pytest tests\test_part1_fisher_z.py    # ...until that Part is green
```

Do the Parts in order — Part 4 imports Part 1, Part 5 imports Part 2, and
Exercise 4c is the gate everything downstream rests on.

## 3. The one rule about import order

```python
import numba101   # pins OMP/OPENBLAS/MKL to 1 thread
import numpy as np
```

Never the other way round. BLAS reads its thread count once, at load time, so
setting the variables after numpy is imported does nothing. `env.pin()` raises a
`RuntimeWarning` if you get this wrong, and `pytest.ini` promotes that warning to
an error so it cannot pass silently.

## 4. Progress → experiment map

| Exercise | Delivers | Experiment |
|---|---|---|
| 1c | `log1p` Fisher-z + error characterisation | E0.3 |
| 2a, 2b | three partial-correlation routes + conditioning behaviour | E0.3 |
| 3a, 3b | FP32 → FP8 rounding harness | E1.1 |
| 3c | flip rate vs κ | E1.1 |
| 4b | α = 1.0 paired-comparison device | E1.1 |
| 4c | flip rate with a **verified zero** noise floor | **E0.1 — gates everything** |
| 5a–5c | cheap κ estimator + calibration curve | E1.2 |

That is Tier 0 and most of Tier 1, built while learning the tool.

## 5. Numba gotchas you will hit, in the order you will hit them

1. **The first call compiles.** Time it separately; it is not your code's speed.
2. **`cache=True` needs a file.** In a notebook or `python -c` it raises
   `cannot cache function ... no locator available`. That is expected, not a bug.
3. **Read typing errors from the bottom.** Numba's tracebacks are enormous; the
   last few lines name the unsupported operation. Ten minutes learning to read
   them saves hours.
4. **`np.linalg.inv` / `cholesky` need contiguous float64 2-D arrays.** Slicing a
   submatrix out of a big correlation matrix usually gives a non-contiguous view;
   `np.ascontiguousarray` it, or Numba's typing will reject it.
5. **Never bare `@jit`.** It falls back to object mode silently. Always `@njit`.

## 6. Reference material

Ordered as in the lesson: the Numba docs (5-minute guide → `@njit` → supported
Python/NumPy features → troubleshooting), then the Numba CUDA docs plus the *CUDA
C++ Programming Guide* ch. 1–5 for Lesson 2, Gorelick & Ozsvald's *High
Performance Python* for the profiling chapters, and the cuPC source
(`LIS-Laboratory/cupc`) — read its CI-test kernel after Part 2.

**One caution from the lesson, repeated because it matters:** most Numba material
online is about speeding up data-science loops. This workload is small-matrix
linear algebra with data-dependent control flow, a different regime. Trust
measurement over blog-post rules of thumb.
