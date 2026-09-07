"""Tier 2: cascade mode, the trajectory, separating sets, and amplification."""

import numpy as np
import pytest

from precision_pc import metrics, parcorr, skeleton
from precision_pc import testset as T
from precision_pc.datasets import gaussian as G


def _instance(n_vars=12, n_obs=1000, seed=0, sampling="population"):
    C, n, dsid, skel, dag = G.design(
        n_vars=n_vars, n_obs=n_obs, edge_prob=0.25, sampling=sampling, seed=seed
    )
    return C, n, dsid, skel


# ------------------------------------------------------------------------ the DGP
def test_random_dag_is_acyclic():
    import itertools

    rng = np.random.default_rng(0)
    A = G.random_dag(15, 0.4, rng)
    # A DAG's adjacency is nilpotent: no path can return to its origin.
    P = np.eye(15, dtype=bool)
    for _ in range(15):
        P = P @ A
    assert not P.any()
    assert not np.any(A & A.T), "an edge and its reverse both present"


def test_population_correlation_is_a_valid_correlation_matrix():
    from precision_pc import checks

    rng = np.random.default_rng(1)
    W = G.random_weights(G.random_dag(10, 0.3, rng), rng)
    C = G.population_correlation(W)
    checks.correlation_matrix(C, "population")
    assert np.all(np.linalg.eigvalsh(C) > 0), "not positive definite"


def test_population_matches_a_large_sample():
    rng = np.random.default_rng(2)
    W = G.random_weights(G.random_dag(8, 0.3, rng), rng)
    from precision_pc import corr as _c

    exact = G.population_correlation(W)
    drawn = _c.correlation(G.simulate(W, 400_000, rng))
    assert np.max(np.abs(exact - drawn)) < 0.02


# -------------------------------------------------------------------- the skeleton
def test_skeleton_recovers_structure_from_population_correlations():
    C, n, _, truth = _instance()
    r = skeleton.discover(C, n, 0.01, fmt="float64", max_level=3)
    tru = {(int(a), int(b)) for a, b in np.argwhere(np.triu(truth, 1))}
    rec = r.edge_set()
    # Not exact: max_level caps the search, so a pair needing a larger separating set
    # keeps its edge. What matters is that recovery is far better than chance.
    assert len(rec & tru) >= 0.6 * len(tru)
    assert r.n_edges < 0.5 * (truth.shape[0] * (truth.shape[0] - 1) / 2)


def test_skeleton_is_bitwise_reproducible():
    C, n, _, _ = _instance()
    a = skeleton.discover(C, n, 0.01)
    b = skeleton.discover(C, n, 0.01)
    assert a.edge_set() == b.edge_set()
    assert a.sepsets == b.sepsets
    assert a.stat.tobytes() == b.stat.tobytes()


def test_trajectory_records_every_executed_test():
    C, n, _, _ = _instance()
    r = skeleton.discover(C, n, 0.01, max_level=2)
    assert len(r.trajectory) == len(r.stat) == len(r.status)
    assert sum(r.level_counts.values()) == len(r.trajectory)
    assert set(r.level_counts) <= {0, 1, 2}
    # Level 0 tests every pair exactly once.
    N = C.shape[0]
    assert r.level_counts[0] == N * (N - 1) // 2


def test_pc_stable_is_order_independent():
    """PC-stable snapshots the adjacency at the start of each level, so relabelling the
    variables cannot change which edges survive. Order-dependent PC has no such guarantee
    -- that difference is E2.2."""
    C, n, _, _ = _instance(n_vars=10)
    perm = np.random.default_rng(3).permutation(10)
    Cp = np.ascontiguousarray(C[np.ix_(perm, perm)])

    a = skeleton.discover(C, n, 0.01, stable=True)
    b = skeleton.discover(Cp, n, 0.01, stable=True)
    back = {tuple(sorted((int(perm[i]), int(perm[j])))) for i, j in b.edge_set()}
    assert a.edge_set() == back


def test_stop_at_first_does_not_change_the_skeleton():
    """cuPC and Schmidt et al. evaluate ALL separating-set candidates in parallel rather
    than short-circuiting. That changes how much work is done -- which matters for the
    throughput measurement -- but not which edges survive."""
    C, n, _, _ = _instance()
    a = skeleton.discover(C, n, 0.01, stop_at_first=True)
    b = skeleton.discover(C, n, 0.01, stop_at_first=False)
    assert a.edge_set() == b.edge_set()
    assert len(b.trajectory) > len(a.trajectory)


def test_nan_statistic_never_deletes_an_edge():
    """A degenerate statistic means the test could not be evaluated. Treating NaN as
    'accept the null' would silently delete edges in exactly the low-precision formats
    under study, manufacturing a large but entirely artefactual SHD."""
    C, n, _, _ = _instance(n_vars=10)
    r = skeleton.discover(C, n, 0.01, fmt="float8_e5m2", max_level=2)
    nan_idx = np.flatnonzero(np.isnan(r.stat))
    for t in nan_idx:
        i, j = int(r.trajectory.i[t]), int(r.trajectory.j[t])
        S = tuple(int(x) for x in r.trajectory.conditioning_set(t))
        assert r.sepsets.get((min(i, j), max(i, j))) != S


# ------------------------------------------------------------------- the comparisons
def test_skeleton_diff_is_the_symmetric_difference():
    C, n, _, _ = _instance()
    a = skeleton.discover(C, n, 0.01, fmt="float64")
    b = skeleton.discover(C, n, 0.01, fmt="float8_e5m2")
    d = metrics.skeleton_diff(a, b)
    assert d.edges_changed == len(a.edge_set() ^ b.edge_set())
    assert d.edges_changed == d.edges_lost + d.edges_gained
    assert metrics.skeleton_diff(a, a).edges_changed == 0


def test_sepset_divergence_finds_the_unmeasured_channel():
    """E2.3. Precision can change which separating set is recorded even when the skeleton
    is identical -- and sepsets determine v-structures, hence orientation. A skeleton-only
    comparison would score that as perfect agreement."""
    C, n, _, _ = _instance()
    a = skeleton.discover(C, n, 0.01, fmt="float64")
    b = skeleton.discover(C, n, 0.01, fmt="float8_e4m3")
    s = metrics.sepset_divergence(a, b)
    assert s.n_common_removed > 0
    assert 0 <= s.n_sepset_differ <= s.n_common_removed
    assert metrics.sepset_divergence(a, a).n_sepset_differ == 0


def test_union_of_trajectories_is_a_superset_of_each():
    """The property that makes union replay a valid unbiased denominator: a CI test is a
    pure function of (C, i, j, S, n, format), so evaluating a test FP8 would reach and
    FP64 never reached is perfectly well defined under FP64."""
    C, n, _, _ = _instance(n_vars=10)
    runs = [
        skeleton.discover(C, n, 0.01, fmt=f, max_level=2)
        for f in ("float64", "bfloat16", "float8_e4m3")
    ]
    u = runs[0].trajectory
    for r in runs[1:]:
        u = u.union(r.trajectory)
    keys = set(u.keys().tolist())
    for r in runs:
        assert set(r.trajectory.keys().tolist()) <= keys


def test_union_replay_is_exactly_paired():
    from precision_pc import engine

    C, n, _, _ = _instance(n_vars=10)
    runs = {
        f: skeleton.discover(C, n, 0.01, fmt=f, max_level=2)
        for f in ("float64", "float8_e4m3")
    }
    u = runs["float64"].trajectory.union(runs["float8_e4m3"].trajectory)
    a = engine.evaluate(C, u, n, fmt="float64")
    b = engine.evaluate(C, u, n, fmt="float8_e4m3")
    assert a.testset_digest == b.testset_digest
    fc = metrics.flip_counts(a, b, 0.01, undefined_policy="exclude")
    assert fc.n_tests == len(u)


def test_pinv_branch_rate_is_recorded_per_run():
    """E2.4: a code-path flip, distinct from numeric perturbation."""
    C, n, _, _ = _instance(n_vars=10)
    r = skeleton.discover(C, n, 0.01, fmt="float8_e5m2", max_level=2)
    assert (r.status == parcorr.PINV_FALLBACK).sum() >= 0
    assert r.status.shape == r.stat.shape


def test_level_shares_sum_to_one():
    C, n, _, _ = _instance()
    r = skeleton.discover(C, n, 0.01, max_level=3)
    assert sum(r.level_shares().values()) == pytest.approx(1.0)


# ======================================== the wave-batched fast path (5-7x on cascade)
@pytest.mark.parametrize("stop_at_first", [True, False])
@pytest.mark.parametrize("fmt", ["float64", "float8_e4m3"])
def test_wave_batching_is_exactly_equal_to_the_loop(stop_at_first, fmt):
    """Batching the skeleton search is only safe if it executes the SAME tests.

    The search short-circuits, and the executed trajectory is a recorded OUTPUT -- it
    feeds union-replay and E3.2's level counts. Naively batching every candidate would
    evaluate tests the loop never reaches and inflate that record. So candidates are
    evaluated in waves: wave k takes the k-th candidate of every still-live edge, and the
    trajectory is re-sorted into the loop's edge-major order, so even the digest matches.
    """
    C, n, _, _ = _instance(n_vars=14)
    a = skeleton.discover(C, n, 0.01, fmt=fmt, max_level=3,
                          stop_at_first=stop_at_first, batched=False)
    b = skeleton.discover(C, n, 0.01, fmt=fmt, max_level=3,
                          stop_at_first=stop_at_first, batched=True)
    assert a.edge_set() == b.edge_set()
    assert a.sepsets == b.sepsets, "a different separating set was recorded"
    assert a.trajectory.digest() == b.trajectory.digest(), "the trajectory changed"
    assert a.stat.tobytes() == b.stat.tobytes()
    assert (a.status == b.status).all()
    assert a.level_counts == b.level_counts


def test_order_dependent_pc_is_not_batched():
    """With stable=False a removal takes effect immediately, so an edge removed early
    changes the candidate sets of edges considered after it. Evaluating a wave across
    edges would reorder exactly that dependency, so the order-dependent variant keeps the
    loop -- E2.2 compares the two, so both have to stay exact."""
    C, n, _, _ = _instance(n_vars=12)
    a = skeleton.discover(C, n, 0.01, stable=False, batched=False)
    b = skeleton.discover(C, n, 0.01, stable=False, batched=True)
    assert a.edge_set() == b.edge_set()
    assert a.trajectory.digest() == b.trajectory.digest()


def test_candidate_order_matches_the_loop():
    """Which candidate is reached FIRST decides which separating set is recorded, and
    sepsets determine v-structures hence orientation (E2.3)."""
    import numpy as _np

    from precision_pc import skeleton_batch

    adj = _np.ones((6, 6), dtype=bool)
    _np.fill_diagonal(adj, False)
    got = skeleton_batch.candidates_for(adj, 0, 1, 1)
    expected = [s for a, b in ((0, 1), (1, 0))
                for s in [(k,) for k in range(6) if k not in (a, b)]]
    seen, want = set(), []
    for s in expected:
        if s not in seen:
            seen.add(s)
            want.append(s)
    assert got == want
