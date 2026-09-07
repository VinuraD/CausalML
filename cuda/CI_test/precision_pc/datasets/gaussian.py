"""Random DAGs and linear-Gaussian SCMs.

Needed for Tier 2 and E4.1: measuring the structural Hamming distance between graphs
recovered at different precisions is only interesting if the graph has structure, and
measuring SHD *against ground truth* needs a known DAG.

Two sampling modes, and the distinction matters for attribution:

`population`
    The exact correlation matrix implied by the SCM, Sigma = (I-B)^-T Omega (I-B)^-1,
    with no sampling noise at all. FP64 then recovers the true skeleton essentially
    perfectly, so any deviation at lower precision is attributable to precision alone.
    This is the clean instrument for E2.1's amplification measurement.

`sample`
    A finite draw of `n_obs` observations. Realistic, but FP64 itself now makes errors,
    so precision effects are mixed with sampling variability. Needed for E4.1, where the
    question is whether low precision is HARMFUL relative to truth rather than merely
    different from FP64.

`n_obs` is still meaningful in population mode: it enters the Fisher-z scaling through
sqrt(n - |S| - 3) and therefore sets where the decision threshold falls, even when the
correlation matrix carries no sampling error.
"""

from __future__ import annotations

import numpy as np

from .. import checks, corr


def random_dag(n_vars: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Random DAG adjacency, `B[i, j] = True` meaning i -> j.

    Drawn in a random topological order so the result is acyclic by construction: edges
    only ever run from an earlier position to a later one.
    """
    order = rng.permutation(n_vars)
    A = np.zeros((n_vars, n_vars), dtype=bool)
    for a in range(n_vars):
        for b in range(a + 1, n_vars):
            if rng.random() < edge_prob:
                A[order[a], order[b]] = True
    return A


def random_weights(
    dag: np.ndarray, rng: np.random.Generator, low: float = 0.3, high: float = 0.9
) -> np.ndarray:
    """Edge weights with magnitude bounded away from zero.

    Weights near zero would make an edge undetectable at any precision, which shows up as
    a recovery failure rather than a precision effect and muddies E4.1.
    """
    W = np.zeros(dag.shape, dtype=np.float64)
    idx = np.argwhere(dag)
    mag = rng.uniform(low, high, len(idx))
    sign = rng.choice([-1.0, 1.0], len(idx))
    W[idx[:, 0], idx[:, 1]] = mag * sign
    return W


def population_correlation(W: np.ndarray, noise_var: np.ndarray | None = None) -> np.ndarray:
    """Exact correlation implied by X = W^T X + e.

    Sigma = (I - W)^-T diag(noise) (I - W)^-1, then normalised to unit diagonal.
    """
    n = W.shape[0]
    if noise_var is None:
        noise_var = np.ones(n, dtype=np.float64)
    M = np.linalg.inv(np.eye(n) - W)
    Sigma = M.T @ np.diag(noise_var) @ M
    d = 1.0 / np.sqrt(np.diag(Sigma))
    C = d[:, None] * Sigma * d[None, :]
    C = corr.symmetrise(C)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)
    C = np.ascontiguousarray(C)
    checks.correlation_matrix(C, "gaussian.population_correlation")
    return C


def simulate(W: np.ndarray, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_obs observations from the linear-Gaussian SCM."""
    n = W.shape[0]
    E = rng.standard_normal((n_obs, n))
    return E @ np.linalg.inv(np.eye(n) - W)


def design(
    *,
    n_vars: int = 12,
    n_obs: int = 1000,
    edge_prob: float = 0.25,
    sampling: str = "population",
    seed: int = 0,
    weight_low: float = 0.3,
    weight_high: float = 0.9,
):
    """Build a linear-Gaussian instance.

    Returns (C, n_obs, dataset_id, true_skeleton, true_dag).

    Both the skeleton and the directed DAG are returned because they score different
    things: the skeleton is the ground truth for an undirected recovery (E4.1), and the
    DAG is what the TRUE CPDAG is computed from, which is the only fair target for a
    recovered CPDAG -- a DAG's orientation is not identifiable beyond its Markov
    equivalence class, so scoring arrows against the DAG itself would penalise a correct
    answer.
    """
    if sampling not in ("population", "sample"):
        raise ValueError("sampling must be 'population' or 'sample', got %r" % sampling)
    rng = np.random.default_rng(seed)
    dag = random_dag(n_vars, edge_prob, rng)
    W = random_weights(dag, rng, weight_low, weight_high)

    if sampling == "population":
        C = population_correlation(W)
    else:
        C = corr.correlation(simulate(W, n_obs, rng))

    skel = dag | dag.T
    dsid = "gaussian:%s:N=%d:p=%g:n=%d:seed=%d" % (sampling, n_vars, edge_prob, n_obs, seed)
    return C, n_obs, dsid, skel, dag
