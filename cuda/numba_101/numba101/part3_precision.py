"""Part 3 - Simulating reduced precision without a GPU.

Design note, copy this into your lab book now: you round the INPUTS and compute
in float64. That models a machine with low-precision storage and a wide
accumulator, which is how FP8 tensor-core matmul actually behaves. Simulating
low-precision ACCUMULATION is a different experiment. A reviewer will ask which
one you measured.

Given to you:  FORMATS, round_to.
Yours:         round_matrix, parcorr_across_formats, flip_rate helpers in part4.
"""

import ml_dtypes
import numpy as np

#: The precision ladder for E1.1. Keys are the names used in plots and tables.
FORMATS = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float8_e4m3": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
}


def round_to(x, dtype):
    """Round to the target storage format, return float64 for computation."""
    return np.asarray(x).astype(dtype).astype(np.float64)


def round_matrix(sub, dtype, keep_unit_diagonal: bool = True):
    """EXERCISE 3b. Round a correlation submatrix to `dtype`.

    Two decisions you have to make explicitly, because they change the answer:

      * The diagonal of a correlation matrix is exactly 1.0, which is
        representable in every format here. Rounding it is a no-op - but if you
        ever rescale, it is not. `keep_unit_diagonal` forces the issue: decide
        which convention you are measuring and say so in the lab book.
      * Rounding entrywise can break symmetry only if you round the two triangles
        differently. Round one triangle and mirror it, so the matrix stays exactly
        symmetric and Cholesky still applies. Otherwise you are measuring your own
        bug instead of the format.

    Returns a float64 array holding representable-in-`dtype` values.
    """
    raise NotImplementedError("Exercise 3b: matrix rounding with the above two rules")


def parcorr_across_formats(sub, parcorr_fn, formats=None):
    """EXERCISE 3b. Round `sub` to each format, run `parcorr_fn` in float64 on the
    rounded matrix, and return {format_name: rho}.

    Keep the float64 entry in the dict - it is the unrounded baseline every other
    row is compared against.
    """
    raise NotImplementedError("Exercise 3b: sweep the format ladder")


def digits_surviving(value: float, dtype) -> int:
    """EXERCISE 3a. How many decimal digits of `value` survive a round trip
    through `dtype`? Compare repr(value) with repr(round_to(value, dtype)) and
    count matching leading significant digits. Sanity check: float32 should give
    you about 7, bfloat16 about 3, float8_e4m3 about 2 - and you should be able
    to explain each from the mantissa width.
    """
    raise NotImplementedError("Exercise 3a: count surviving digits")
