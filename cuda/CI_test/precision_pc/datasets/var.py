"""Vector autoregressive processes and lag embedding -- the Tier 3 instrument.

Why this module is the mechanism of contribution C3, and not merely another dataset:
lagged variables of an autocorrelated series are near-collinear BY CONSTRUCTION. For an
AR(1), corr(X_t, X_{t-k}) = phi^k, so the design matrix over lags is Toeplitz and its
conditioning degrades as phi -> 1 regardless of how many observations there are. That is
the claim: ill-conditioning arises independently of the sample-to-dimension ratio that
governs the classical analysis.

**Where the ill-conditioning lands is what matters.** A CI test never inverts the whole
correlation matrix -- it inverts the 3x3 or 4x4 submatrix over {i, j} | S. A synthetic
matrix built to a requested condition number spreads its ill-conditioning across the full
spectrum, and the small submatrices stay well conditioned (measured: parent at 5e11, the
inverted submatrices topping out near 1e7). Lagged variables are different in kind,
because the near-collinearity sits between SPECIFIC PAIRS and therefore inside exactly
the small blocks the algorithm touches.

**A calibration that must be stated plainly.** For a plain AR(1), kappa of the 3x3
submatrix scales as roughly 1/(1 - phi):

    phi = 0.95 -> 8.4e1     phi = 0.999    -> 4.5e3
    phi = 0.99 -> 4.4e2     phi = 0.999999 -> 4.5e6

FP32's flip onset sits near kappa ~ 1e8, which would need phi ~ 0.99999996 -- an
essentially perfect random walk, not a plausible telemetry series. So an AR(1) at the
proposal's most extreme phi = 0.99 exercises TF32, BF16 and FP8 comfortably and leaves
FP32 clean. That is why this module offers more than AR(1): a general VAR whose companion
spectral radius is the knob, a common-factor structure that stacks CROSS-SECTIONAL
collinearity on top of the temporal kind, and oversampling. Those compound in ways one
AR(1) lag structure cannot.

**Two statistical caveats to record rather than hide.**

1. Embedding at tau_max costs observations: the effective sample size is n - tau_max + 1,
   and the Fisher-z scaling uses sqrt(n_eff - |S| - 3). `design()` returns n_eff, and it
   is what the engine must be given.
2. Observations of an autocorrelated series are not independent, so the Fisher-z null is
   not exactly standard normal and the nominal alpha is not the true type-I rate. This
   experiment measures whether PRECISION changes a decision, and the decision is made by
   Fisher-z either way -- so the comparison remains valid even though the test itself is
   mis-calibrated. Do not report these p-values as evidence about the graph.
"""

from __future__ import annotations

import numpy as np

from .. import corr

STRUCTURES = ("ar1", "var", "factor")


# ------------------------------------------------------------------ coefficient sets
def companion(A: np.ndarray) -> np.ndarray:
    """Companion matrix of a VAR(p) with coefficient stack A of shape (p, k, k).

    The process is stationary iff every eigenvalue of this matrix lies strictly inside
    the unit circle, so its spectral radius is the single knob for "how near a unit root".
    """
    p, k, _ = A.shape
    C = np.zeros((k * p, k * p), dtype=np.float64)
    C[:k] = np.hstack([A[i] for i in range(p)])
    if p > 1:
        C[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return C


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(companion(A)))))


def random_var_coefs(k: int, p: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Random VAR(p) coefficients rescaled to an exact companion spectral radius.

    Scaling A_i -> s**(i+1) * A_i maps every companion eigenvalue lambda -> s * lambda,
    which is what makes the radius an exact knob rather than a target to search for.
    """
    if not 0.0 <= radius < 1.0:
        raise ValueError("spectral radius must be in [0, 1) for stationarity, got %r" % radius)
    A = rng.standard_normal((p, k, k)) / np.sqrt(k * p)
    r = spectral_radius(A)
    if r == 0.0:
        return A
    s = radius / r
    return np.array([A[i] * s ** (i + 1) for i in range(p)])


# -------------------------------------------------------------------- the processes
def simulate_var(
    n_obs: int, A: np.ndarray, rng: np.random.Generator, *, burn_in: int = 500, sigma: float = 1.0
) -> np.ndarray:
    """Simulate a VAR(p). Returns (n_obs, k).

    The burn-in is not optional: starting at zero makes the first few hundred
    observations non-stationary, and at phi -> 1 the relaxation time is long. A
    non-stationary prefix would change the correlation matrix for reasons unrelated to
    the experiment.
    """
    p, k, _ = A.shape
    total = n_obs + burn_in
    X = np.zeros((total, k), dtype=np.float64)
    E = rng.standard_normal((total, k)) * sigma
    for t in range(p, total):
        acc = E[t].copy()
        for i in range(p):
            acc += A[i] @ X[t - i - 1]
        X[t] = acc
    out = X[burn_in:]
    if not np.all(np.isfinite(out)):
        raise ValueError(
            "VAR simulation diverged (spectral radius %.6f). Use radius < 1."
            % spectral_radius(A)
        )
    return out


def simulate_ar1(
    n_obs: int, phi: float, rng: np.random.Generator, *, n_series: int = 1, burn_in: int = 500
) -> np.ndarray:
    """Independent AR(1) series. The canonical C3 instrument."""
    A = np.zeros((1, n_series, n_series))
    np.fill_diagonal(A[0], phi)
    return simulate_var(n_obs, A, rng, burn_in=burn_in)


def simulate_factor(
    n_obs: int,
    phi: float,
    rng: np.random.Generator,
    *,
    n_series: int = 4,
    loading: float = 0.95,
    burn_in: int = 500,
) -> np.ndarray:
    """Series driven by a shared autoregressive factor.

    This stacks CROSS-SECTIONAL collinearity on top of the temporal kind: every series is
    `loading` times a common AR(1) factor plus idiosyncratic noise, so distinct series at
    the same lag are near-collinear too. Real telemetry looks far more like this than like
    independent AR(1)s, and it reaches worse conditioning at a given phi.
    """
    f = simulate_ar1(n_obs, phi, rng, n_series=1, burn_in=burn_in)[:, 0]
    idio = rng.standard_normal((n_obs, n_series)) * np.sqrt(max(1.0 - loading**2, 1e-12))
    return loading * f[:, None] + idio


# ------------------------------------------------------------------- lag embedding
def embed(X: np.ndarray, tau_max: int) -> np.ndarray:
    """Time-delay embedding: columns are each series at lags 0 .. tau_max - 1.

    Variable ordering is (series, lag) in C order, so for k series the design has
    k * tau_max columns and column (s * tau_max + l) is series s at lag l.
    """
    if tau_max < 1:
        raise ValueError("tau_max must be at least 1, got %r" % tau_max)
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    n_eff = n - tau_max + 1
    if n_eff < 4:
        raise ValueError(
            "embedding at tau_max=%d leaves only %d effective observations" % (tau_max, n_eff)
        )
    cols = [X[tau_max - 1 - l : n - l, s] for s in range(k) for l in range(tau_max)]
    return np.ascontiguousarray(np.column_stack(cols))


# ------------------------------------------------------------------------ front door
def design(
    *,
    n_obs: int,
    phi: float = 0.9,
    tau_max: int = 3,
    n_series: int = 1,
    structure: str = "ar1",
    lags: int = 1,
    loading: float = 0.95,
    sampling_ratio: int = 1,
    seed: int = 0,
    burn_in: int = 500,
):
    """Build a lag-embedded correlation matrix.

    Returns (C, n_eff, dataset_id).

    `sampling_ratio` > 1 models observing the SAME process more finely: the per-step
    persistence becomes phi ** (1 / ratio), so adjacent samples are more nearly identical.
    Oversampled telemetry is a realistic route to ill-conditioning that has nothing to do
    with the underlying process being close to a unit root.
    """
    if structure not in STRUCTURES:
        raise ValueError("unknown structure %r; have %r" % (structure, STRUCTURES))
    if not 0.0 <= phi < 1.0:
        raise ValueError("phi must be in [0, 1) for stationarity, got %r" % phi)
    if sampling_ratio < 1:
        raise ValueError("sampling_ratio must be >= 1, got %r" % sampling_ratio)

    phi_eff = phi ** (1.0 / sampling_ratio) if sampling_ratio > 1 else phi
    rng = np.random.default_rng(seed)

    if structure == "ar1":
        X = simulate_ar1(n_obs, phi_eff, rng, n_series=n_series, burn_in=burn_in)
    elif structure == "factor":
        X = simulate_factor(
            n_obs, phi_eff, rng, n_series=n_series, loading=loading, burn_in=burn_in
        )
    else:  # general VAR, spectral radius = phi_eff
        A = random_var_coefs(n_series, lags, phi_eff, rng)
        X = simulate_var(n_obs, A, rng, burn_in=burn_in)

    D = embed(X, tau_max)
    C = corr.correlation(D)
    n_eff = D.shape[0]
    dsid = "var:%s:phi=%g:tau=%d:k=%d:p=%d:os=%d" % (
        structure, phi, tau_max, n_series, lags, sampling_ratio
    )
    return C, n_eff, dsid


def theoretical_ar1_corr(tau_max: int, phi: float) -> np.ndarray:
    """The exact Toeplitz correlation matrix of an AR(1) lag embedding.

    Useful as a noise-free reference: it isolates the conditioning that autocorrelation
    causes BY CONSTRUCTION from the sampling variability of any finite realisation.
    """
    return np.array(
        [[phi ** abs(i - j) for j in range(tau_max)] for i in range(tau_max)], dtype=np.float64
    )
