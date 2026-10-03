"""Acceptance tests for Part 3."""

import numpy as np
import pytest

from numba101.part2_parcorr import parcorr_inv
from numba101.part3_precision import (
    FORMATS,
    digits_surviving,
    parcorr_across_formats,
    round_matrix,
    round_to,
)


def test_round_to_is_a_noop_for_float64():
    x = np.float64(0.1)
    assert float(round_to(x, np.float64)) == float(x)


def test_rounding_loses_precision_monotonically():
    """Each rung of the ladder must lose at least as much as the one above."""
    x = 0.1
    errs = [abs(float(round_to(np.float64(x), dt)) - x) for dt in FORMATS.values()]
    assert errs == sorted(errs), f"ladder is out of order: {errs}"


def test_digits_surviving_matches_mantissa_width():
    assert digits_surviving(0.1, np.float32) >= 6
    assert 2 <= digits_surviving(0.1, FORMATS["bfloat16"]) <= 4
    assert digits_surviving(0.1, FORMATS["float8_e4m3"]) <= 2


def test_round_matrix_preserves_symmetry_and_diagonal():
    """If rounding breaks symmetry you are measuring your own bug."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    R = round_matrix(C, FORMATS["bfloat16"])
    np.testing.assert_array_equal(R, R.T)
    np.testing.assert_allclose(np.diag(R), 1.0, atol=0)
    assert R.dtype == np.float64, "compute in float64; only the storage is narrow"


def test_sweep_returns_every_format_and_an_unrounded_baseline():
    C = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    out = parcorr_across_formats(np.ascontiguousarray(C), parcorr_inv)
    assert set(out) == set(FORMATS)
    assert out["float64"] == pytest.approx(parcorr_inv(np.ascontiguousarray(C)), abs=1e-15)
