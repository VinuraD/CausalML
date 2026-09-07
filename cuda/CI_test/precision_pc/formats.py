"""The precision ladder.

Design note, recorded here because a reviewer will ask which experiment was run: these
functions round the INPUTS to a target storage format and return float64, so all
subsequent arithmetic happens in float64. That models a machine with low-precision
STORAGE and a wide accumulator, which is how FP8 tensor-core matmul actually behaves.
Simulating low-precision ACCUMULATION is a different experiment and is not what this
module does. The `round_stage` column records the distinction in every result row.

The ladder is not one-dimensional -- it is a (mantissa, exponent) grid:

    tf32 = (10, 8)    float16 = (10, 5)    bfloat16 = (7, 8)
    e4m3 = ( 3, 4)    e5m2    = ( 2, 5)

float16 is kept even though no GPU uses it for this workload, because tf32-vs-float16
isolates EXPONENT RANGE at fixed mantissa, and bfloat16-vs-tf32 isolates MANTISSA at
fixed range. Since correlations live in [-1, 1], exponent range should be nearly
irrelevant -- so tf32 and float16 must agree on correlation inputs, and any divergence
between them is a bug in this module, not a finding. `tests/test_formats.py` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import ml_dtypes
import numpy as np

# --------------------------------------------------------------------------- TF32
# TF32 is sign(1) + exponent(8, same as fp32) + explicit mantissa(10, same as fp16),
# held in a 32-bit register with the low 13 mantissa bits forced to zero. There is no
# numpy or ml_dtypes dtype for it, which is why the ladder cannot be a {name: dtype} dict.

_TF32_MASK = np.uint32(0xFFFFE000)  # keep sign + exponent + 10 mantissa bits
_TF32_HALF = np.uint32(0x00001000)  # 2**12, the tie point of the 13 dropped bits


def to_tf32(x, mode: str = "rne") -> np.ndarray:
    """Round to the nearest TF32 value, returned as float64.

    Two-step on purpose: float64 -> float32 (hardware RNE) -> drop 13 mantissa bits. On
    real hardware a tensor core's input IS an fp32 register, so the double rounding is
    the physically faithful chain, not an approximation of a direct float64 -> tf32 round.
    Recorded as a decision.

    `mode` exists because PTX exposes both cvt.rn.tf32.f32 (nearest-even) and
    cvt.rna.tf32.f32 (nearest, ties away), and which one a given toolkit emits has varied
    across ISA versions. Until this is calibrated against a real TF32 device, every TF32
    number this harness produces is provisional. The local Pascal GPU has no TF32 at all,
    so the calibration has to wait for cluster access.

    Add-then-mask implements round-to-nearest-even: adding (2**12 - 1 + lsb) and then
    truncating carries exactly when the dropped part exceeds half, or equals half and the
    retained LSB is 1.
    """
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32)
    if mode == "rne":
        lsb = (u >> np.uint32(13)) & np.uint32(1)
        bias = (_TF32_HALF - np.uint32(1)) + lsb
    elif mode == "rna":
        bias = np.broadcast_to(_TF32_HALF, u.shape)
    elif mode == "trunc":
        bias = np.broadcast_to(np.uint32(0), u.shape)
    else:
        raise ValueError("unknown tf32 rounding mode " + repr(mode))
    r = ((u + bias) & _TF32_MASK).view(np.float32)
    # The NaN guard is NOT optional. A NaN whose payload lives only in the low 13 bits,
    # e.g. 0x7F800001, becomes 0x7F801000 after the add and 0x7F800000 == +Inf after the
    # mask: a silent NaN -> Inf. BF16 and FP8 correlation matrices do produce NaNs, and a
    # NaN that has quietly become Inf is exactly the class of bug that biases a flip rate
    # downward. Overflow of finite values needs no guard: 0x7F7FFFFF + 0x1000 masks to
    # +Inf, which is the correct IEEE round-to-nearest overflow result.
    out = np.where(np.isnan(a), a, r)
    # A signalling-NaN payload sets the invalid flag on the widening cast. The value is
    # preserved as a NaN either way, which is all that matters here.
    with np.errstate(invalid="ignore"):
        return np.asarray(out, dtype=np.float64)


def _via_dtype(dt) -> Callable[[np.ndarray], np.ndarray]:
    def _round(x):
        return np.asarray(np.asarray(x).astype(dt), dtype=np.float64)

    return _round


# ---------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Format:
    """One rung of the ladder."""

    name: str
    mantissa_bits: int  # explicit mantissa bits, excluding the implicit leading 1
    exponent_bits: int
    round: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    eps: float
    max_finite: float
    #: Smallest |r| < 1 that this format rounds to exactly 1.0, which makes atanh
    #: infinite. For E4M3 this is 0.96875: in a phi -> 1 VAR regime that is the DOMINANT
    #: behaviour, not a rounding perturbation, so FP8 must be reported through the status
    #: breakdown rather than as a flip rate. None means it cannot happen.
    saturates_at: Optional[float]

    @property
    def decimal_digits(self) -> float:
        """Significant decimal digits, for the digits-lost ~ log10(kappa) heuristic (C2)."""
        return (self.mantissa_bits + 1) * float(np.log10(2.0))


def _saturation_point(round_fn) -> Optional[float]:
    """Smallest |r| < 1 the format rounds to exactly 1.0, by bisection."""
    just_below_one = np.nextafter(1.0, 0.0)
    if float(round_fn(np.array([just_below_one]))[0]) < 1.0:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(round_fn(np.array([mid]))[0]) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def _make(name, mant, expo, round_fn, eps, mx) -> Format:
    return Format(
        name=name,
        mantissa_bits=mant,
        exponent_bits=expo,
        round=round_fn,
        eps=float(eps),
        max_finite=float(mx),
        saturates_at=_saturation_point(round_fn),
    )


FORMATS: dict[str, Format] = {}

for _name, _dt, _mant, _expo in [
    ("float64", np.float64, 52, 11),
    ("float32", np.float32, 23, 8),
    ("float16", np.float16, 10, 5),
    ("bfloat16", ml_dtypes.bfloat16, 7, 8),
    ("float8_e4m3", ml_dtypes.float8_e4m3fn, 3, 4),
    ("float8_e5m2", ml_dtypes.float8_e5m2, 2, 5),
]:
    _fi = np.finfo(_dt) if _dt in (np.float64, np.float32, np.float16) else ml_dtypes.finfo(_dt)
    FORMATS[_name] = _make(_name, _mant, _expo, _via_dtype(_dt), _fi.eps, _fi.max)

# tf32 has no dtype, so its eps and max come from its (10, 8) shape directly.
FORMATS["tf32"] = _make(
    "tf32", 10, 8, to_tf32, 2.0**-10, float(np.finfo(np.float32).max)
)

#: Canonical order for tables and plots: widest mantissa first.
LADDER = [
    "float64",
    "float32",
    "tf32",
    "float16",
    "bfloat16",
    "float8_e4m3",
    "float8_e5m2",
]


def get(name: str) -> Format:
    if name not in FORMATS:
        raise KeyError("unknown format " + repr(name) + "; have " + repr(sorted(FORMATS)))
    return FORMATS[name]


def round_corr(C: np.ndarray, fmt, keep_unit_diagonal: bool = True) -> np.ndarray:
    """Round a whole correlation matrix once, preserving exact symmetry.

    Rounding is elementwise, so extract-then-round and round-then-extract are identical.
    Rounding the FULL matrix once per format -- rather than each submatrix per test -- is
    therefore free correctness: the symmetry and unit-diagonal conventions are decided in
    one place and every test inherits them, instead of the convention drifting to per-test
    scope where it can differ between levels.

    Symmetry is preserved by construction: the input is asserted bitwise symmetric (see
    `checks.correlation_matrix`), and an elementwise map of a bitwise-symmetric matrix is
    bitwise symmetric. No triangle-mirroring needed.
    """
    f = fmt if isinstance(fmt, Format) else get(fmt)
    out = np.ascontiguousarray(
        f.round(np.asarray(C, dtype=np.float64)), dtype=np.float64
    )
    if keep_unit_diagonal:
        # 1.0 is exactly representable in every format here, so this is a no-op for the
        # ladder as it stands. It is kept explicit because it stops being a no-op the
        # moment anything rescales, and the lab book has to record which convention ran.
        np.fill_diagonal(out, 1.0)
    return out
