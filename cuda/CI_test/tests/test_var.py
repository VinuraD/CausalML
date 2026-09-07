"""Tier 3: the VAR generator, and the C3 mechanism it exists to demonstrate."""

import numpy as np
import pytest

from precision_pc import corr, engine, formats, parcorr
from precision_pc import testset as T
from precision_pc.datasets import var as V


# ------------------------------------------------------------------ the machinery
@pytest.mark.parametrize("radius", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("lags", [1, 2, 3])
def test_spectral_radius_is_an_exact_knob(radius, lags):
    """Scaling A_i -> s**(i+1) A_i maps every companion eigenvalue lambda -> s*lambda,
    which is what makes "how near a unit root" a setting rather than a search."""
    rng = np.random.default_rng(0)
    A = V.random_var_coefs(3, lags, radius, rng)
    assert V.spectral_radius(A) == pytest.approx(radius, abs=1e-9)


def test_nonstationary_coefficients_are_refused():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="stationarity"):
        V.random_var_coefs(2, 1, 1.0, rng)
    with pytest.raises(ValueError, match="stationarity"):
        V.design(n_obs=100, phi=1.0)


def test_embedding_shape_and_effective_sample_size():
    """Embedding costs observations: n_eff = n - tau_max + 1. The Fisher-z scaling uses
    n_eff, so getting this wrong shifts every statistic."""
    X = np.arange(100, dtype=np.float64).reshape(50, 2)
    D = V.embed(X, 4)
    assert D.shape == (50 - 4 + 1, 2 * 4)
    C, n_eff, _ = V.design(n_obs=500, tau_max=6, n_series=3, seed=0)
    assert C.shape == (18, 18)
    assert n_eff == 500 - 6 + 1


def test_embedding_columns_are_the_right_lags():
    """Column (s * tau_max + l) must be series s at lag l, or the conditioning-set
    semantics are silently wrong."""
    x = np.arange(20, dtype=np.float64).reshape(20, 1)
    D = V.embed(x, 3)
    # lag 0 is the most recent; lag l is l steps earlier.
    assert np.array_equal(D[:, 0], x[2:, 0])
    assert np.array_equal(D[:, 1], x[1:-1, 0])
    assert np.array_equal(D[:, 2], x[:-2, 0])


def test_design_is_reproducible_from_seed():
    a = V.design(n_obs=800, phi=0.9, tau_max=4, n_series=2, seed=7)[0]
    b = V.design(n_obs=800, phi=0.9, tau_max=4, n_series=2, seed=7)[0]
    assert a.tobytes() == b.tobytes()
    c = V.design(n_obs=800, phi=0.9, tau_max=4, n_series=2, seed=8)[0]
    assert a.tobytes() != c.tobytes()


def test_burn_in_reaches_stationarity():
    """Without a burn-in the first observations are non-stationary, which would change
    the correlation matrix for reasons unrelated to the experiment."""
    rng = np.random.default_rng(0)
    X = V.simulate_ar1(4000, 0.95, rng)
    first, last = X[:1000].var(), X[-1000:].var()
    assert 0.5 < first / last < 2.0


# --------------------------------------------------------------- the C3 mechanism
def test_conditioning_increases_monotonically_with_phi():
    """Contribution C3. Lagged variables of an autocorrelated series are near-collinear
    BY CONSTRUCTION, so kappa must rise with phi independently of the sample size."""
    kappas = []
    for phi in (0.0, 0.3, 0.6, 0.9, 0.95, 0.99):
        C, _, _ = V.design(n_obs=4000, phi=phi, tau_max=5, n_series=1, seed=1)
        kappas.append(np.linalg.cond(C))
    assert kappas == sorted(kappas), "kappa is not monotone in phi: %r" % kappas
    assert kappas[-1] / kappas[0] > 100


def test_theoretical_toeplitz_matches_the_simulation():
    """The noise-free reference isolates conditioning caused BY CONSTRUCTION from the
    sampling variability of a finite realisation."""
    phi, tau = 0.95, 5
    C, _, _ = V.design(n_obs=200_000, phi=phi, tau_max=tau, n_series=1, seed=2)
    ref = V.theoretical_ar1_corr(tau, phi)
    assert np.max(np.abs(C - ref)) < 0.02
    assert np.linalg.cond(C) == pytest.approx(np.linalg.cond(ref), rel=0.15)


def test_ar1_does_not_reach_the_fp32_failure_regime():
    """The calibration that must be stated rather than assumed.

    kappa(3x3) ~ 1/(1-phi), so even at the proposal's most extreme phi = 0.99 the
    submatrices a CI test inverts sit near 4e2 -- three orders of magnitude below FP32's
    onset at ~1e8. An AR(1) exercises TF32/BF16/FP8 and leaves FP32 clean, and any claim
    that "FP32 is safe for time series" based on this generator alone would be an
    artefact of the instrument, not a finding.
    """
    ref = V.theoretical_ar1_corr(3, 0.99)
    assert np.linalg.cond(ref) < 1e4
    ref6 = V.theoretical_ar1_corr(3, 0.999999)
    assert np.linalg.cond(ref6) < 1e8


def test_common_factor_beats_ar1_at_equal_phi():
    """The reason `structure` exists: cross-sectional collinearity stacks on top of the
    temporal kind, so a factor model reaches worse conditioning at the same phi."""
    Ca, _, _ = V.design(n_obs=4000, phi=0.9, tau_max=3, n_series=4, structure="ar1", seed=3)
    Cf, _, _ = V.design(n_obs=4000, phi=0.9, tau_max=3, n_series=4, structure="factor", seed=3)
    assert np.linalg.cond(Cf) > np.linalg.cond(Ca)


def test_oversampling_increases_persistence():
    """Observing the SAME process more finely makes adjacent samples nearly identical --
    a realistic route to ill-conditioning that has nothing to do with a unit root."""
    C1, _, _ = V.design(n_obs=4000, phi=0.9, tau_max=4, sampling_ratio=1, seed=4)
    C8, _, _ = V.design(n_obs=4000, phi=0.9, tau_max=4, sampling_ratio=8, seed=4)
    assert np.linalg.cond(C8) > np.linalg.cond(C1)


# ------------------------------------------------------- the FP8 saturation prediction
def test_fp8_saturates_lagged_correlations_at_high_phi():
    """Pre-registered: E4M3 rounds any |r| >= 0.96875 to exactly 1.0. At phi = 0.99 the
    lag-1/2/3 correlations are 0.99/0.98/0.97, all above that, so FP8's behaviour in the
    flagship time-series regime is DEGENERACY, not rounding perturbation. At phi = 0.95
    the lag-1 correlation is 0.95, below the threshold, so it stays clean.

    The transition is sharp and mechanistic, which is what makes it a real prediction.
    """
    e4m3 = formats.get("float8_e4m3")
    assert e4m3.saturates_at == pytest.approx(0.96875, abs=1e-6)

    ts = T.level0(4)
    clean = V.design(n_obs=4000, phi=0.95, tau_max=4, n_series=1, seed=5)
    hot = V.design(n_obs=4000, phi=0.99, tau_max=4, n_series=1, seed=5)

    r_clean = engine.evaluate(clean[0], ts, clean[1], fmt="float8_e4m3")
    r_hot = engine.evaluate(hot[0], ts, hot[1], fmt="float8_e4m3")

    frac_ok_clean = r_clean.n_ok / len(r_clean)
    frac_ok_hot = r_hot.n_ok / len(r_hot)
    assert frac_ok_clean > frac_ok_hot, (
        "expected FP8 to degrade between phi=0.95 and phi=0.99: %.2f vs %.2f"
        % (frac_ok_clean, frac_ok_hot)
    )


def test_degenerate_fp8_output_is_never_silently_agreement():
    """The wild instance of failure mode A. At phi=0.99 most E4M3 tests are undefined,
    and counting them as 'not rejected' would understate the effect enormously."""
    C, n_eff, _ = V.design(n_obs=4000, phi=0.99, tau_max=4, n_series=3, seed=6)
    ts = T.from_adjacency(
        np.ones((C.shape[0],) * 2, dtype=bool) ^ np.eye(C.shape[0], dtype=bool), 1
    )
    res = engine.evaluate(C, ts, n_eff, fmt="float8_e4m3")
    counts = res.status_counts()
    assert counts.get("OK", 0) < len(res)
    assert res.n_undefined > 0

    from precision_pc import metrics

    ref = engine.evaluate(C, ts, n_eff, fmt="float64")
    with pytest.raises(ValueError, match="undefined"):
        metrics.flip_counts(ref, res, 0.01, undefined_policy="error")
