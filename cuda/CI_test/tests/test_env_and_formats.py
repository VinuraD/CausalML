"""Steps 1 and 3: the pinning is real, and the format ladder is correct."""

import numpy as np
import pytest

from precision_pc import env, formats


# --------------------------------------------------------------------------- env
def test_thread_vars_are_pinned():
    for var, val in env.report().items():
        if var in env.THREAD_VARS:
            assert val == "1", "%s is %r" % (var, val)


def test_pinning_is_verified_not_merely_declared():
    """Environment variables are intent. threadpoolctl is evidence.

    A test that only checked os.environ would pass on a machine where numpy was imported
    first and the pinning silently did nothing -- which is precisely the failure that
    makes the E0.1 noise floor non-zero.
    """
    ok, problems = env.verify()
    assert ok, "thread pinning not verified: %r" % problems
    pools = env.threadpools()
    assert pools, "threadpoolctl reported no pools; cannot verify BLAS threading"
    assert all(p["num_threads"] == 1 for p in pools)


# ------------------------------------------------------------------------ tf32
@pytest.mark.parametrize("mode", ["rne", "rna", "trunc"])
def test_tf32_output_is_representable(mode):
    """Every result must have its low 13 mantissa bits clear."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 100_000)
    y = formats.to_tf32(x, mode=mode).astype(np.float32)
    assert np.all((y.view(np.uint32) & np.uint32(0x1FFF)) == 0)


def test_tf32_is_idempotent():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 50_000)
    once = formats.to_tf32(x)
    assert np.array_equal(formats.to_tf32(once), once)


def test_tf32_is_monotone():
    x = np.sort(np.random.default_rng(2).uniform(-1, 1, 50_000))
    y = formats.to_tf32(x)
    assert np.all(np.diff(y) >= 0)


def test_tf32_relative_error_within_half_ulp():
    """10 explicit mantissa bits => relative error at most 2**-11."""
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 200_000)
    y = formats.to_tf32(x)
    rel = np.abs(y - x) / np.abs(x)
    assert rel.max() <= 2.0**-11


def test_tf32_preserves_nan_payload_instead_of_making_inf():
    """The regression this guard exists for.

    A NaN whose payload lives only in the low 13 bits (0x7F800001) becomes 0x7F801000
    after the rounding add and 0x7F800000 == +Inf after the mask. Silent NaN -> Inf, in
    exactly the formats that produce NaNs. Without the guard this test fails.
    """
    nan_payload = np.array([0x7F800001, 0xFF800001], dtype=np.uint32).view(np.float32)
    assert np.all(np.isnan(nan_payload))
    out = formats.to_tf32(nan_payload)
    assert np.all(np.isnan(out)), "NaN silently became %r" % out


def test_tf32_preserves_infinities_and_signed_zero():
    x = np.array([np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
    y = formats.to_tf32(x).astype(np.float32)
    assert y[0] == np.inf and y[1] == -np.inf
    assert np.signbit(y[3]) and not np.signbit(y[2])


def test_tf32_overflows_to_inf_like_ieee():
    biggest = np.array([np.finfo(np.float32).max], dtype=np.float32)
    assert np.isinf(formats.to_tf32(biggest)[0])


# ------------------------------------------------------------------- the ladder
def test_tf32_and_float16_agree_on_correlation_inputs():
    """Both have a 10-bit mantissa, and float16's narrower exponent does not bind on
    [-1, 1] except for correlations below its subnormal threshold. Divergence beyond a
    handful of double-rounding ties is a BUG in this module, not a finding."""
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 500_000)
    a = formats.get("tf32").round(x)
    b = formats.get("float16").round(x)
    differ = np.mean(a != b)
    assert differ < 1e-3, "tf32 and float16 differ on %.4f%% of inputs" % (100 * differ)


def test_saturation_points_match_format_arithmetic():
    """E4M3 has 3 mantissa bits, so spacing in [0.5, 1) is 2**-4 and anything above
    1 - 2**-5 rounds to exactly 1.0. That saturation, not rounding noise, is what FP8
    actually does to a strongly correlated pair."""
    # ml_dtypes converts float64 -> float8 via float32, so the realised boundary sits
    # one float32 ulp below the exact midpoint. The scientific claim is the midpoint.
    assert formats.get("float8_e4m3").saturates_at == pytest.approx(0.96875, abs=1e-6)
    assert formats.get("float8_e5m2").saturates_at == pytest.approx(0.9375, abs=1e-6)
    assert formats.get("float64").saturates_at is None

    e4m3 = formats.get("float8_e4m3").round
    assert e4m3(np.array([0.97]))[0] == 1.0
    assert e4m3(np.array([0.96]))[0] < 1.0


@pytest.mark.parametrize("name", formats.LADDER)
def test_rounding_keeps_correlations_in_range(name):
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, 20_000)
    y = formats.get(name).round(x)
    assert np.all(np.abs(y) <= 1.0)


@pytest.mark.parametrize("name", formats.LADDER)
def test_round_corr_preserves_exact_symmetry_and_unit_diagonal(name):
    from precision_pc import checks, corr

    rng = np.random.default_rng(6)
    C = corr.correlation(rng.standard_normal((200, 12)))
    R = formats.round_corr(C, name)
    checks.is_bitwise_symmetric(R, "rounded to " + name)
    checks.has_unit_diagonal(R, "rounded to " + name)


def test_mantissa_ordering_is_monotone_in_precision():
    """The ladder must be ordered by decreasing precision, or every plot is misleading."""
    digits = [formats.get(n).decimal_digits for n in formats.LADDER]
    assert digits == sorted(digits, reverse=True)
