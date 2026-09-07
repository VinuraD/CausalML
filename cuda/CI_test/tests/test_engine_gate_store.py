"""Steps 5, 7, 8, 9: pairing, determinism, the gate, and the store.

These are the tests that stop the harness producing a WRONG result rather than a broken
one, so each names the failure mode it prevents.
"""

import numpy as np
import pytest

from precision_pc import corr, engine, gate, metrics, parcorr, store
from precision_pc import testset as T


def _setup(n_vars=20, n_obs=200, seed=0):
    rng = np.random.default_rng(seed)
    C = corr.correlation(rng.standard_normal((n_obs, n_vars)))
    return C, T.level0(n_vars), n_obs


# ------------------------------------------------------------------------- pairing
def test_testset_digest_is_content_addressed():
    a = T.level0(10)
    b = T.level0(10)
    c = T.level0(11)
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_metrics_refuses_to_compare_different_test_sets():
    """Failure mode B: comparing test #7134 against a DIFFERENT test #7134 manufactures
    flips out of nothing and leaves no trace in the output. The digest check is the only
    thing standing between the harness and a spectacular false positive."""
    C, ts, n = _setup()
    r1 = engine.evaluate(C, ts, n)
    ts2 = T.level0(ts.n_vars)
    object.__setattr__(ts2, "i", ts2.i[::-1].copy())  # same length, different tests
    r2 = engine.evaluate(C, ts2, n)
    with pytest.raises(metrics.PairingError):
        metrics.flip_counts(r1, r2, 0.01)


def test_testset_roundtrips_through_parquet(tmp_path):
    ts = T.from_adjacency(np.ones((8, 8), dtype=bool) ^ np.eye(8, dtype=bool), 2)
    p = ts.to_parquet(tmp_path / "ts.parquet")
    back = T.TestSet.from_parquet(p)
    assert back.digest() == ts.digest()
    assert np.array_equal(back.S, ts.S)


def test_union_is_a_superset_of_both():
    """`union` mode is what removes the trajectory bias instead of merely reporting it."""
    a = T.level0(6)
    b = T.from_adjacency(np.ones((6, 6), dtype=bool) ^ np.eye(6, dtype=bool), 1)
    u = a.union(b)
    assert len(u) == len(a) + len(b)
    assert u.jaccard(a) < 1.0
    assert set(a.keys().tolist()) <= set(u.keys().tolist())
    assert set(b.keys().tolist()) <= set(u.keys().tolist())


# -------------------------------------------------------------------- determinism
def test_engine_is_bitwise_reproducible():
    C, ts, n = _setup()
    a = engine.evaluate(C, ts, n)
    b = engine.evaluate(C, ts, n)
    assert metrics.is_bitwise_identical(a, b)
    assert a.stat.tobytes() == b.stat.tobytes()


def test_e01_noise_floor_is_exactly_zero():
    """Gate G0. Not approximately zero. Exactly 0.0, or nothing downstream means anything."""
    C, ts, n = _setup()
    a = engine.evaluate(C, ts, n, fmt="float64")
    b = engine.evaluate(C, ts, n, fmt="float64")
    fc = metrics.flip_counts(a, b, 0.01, undefined_policy="exclude")
    assert fc.flip_rate == 0.0
    assert fc.n_flip == 0


def test_gate_passes_and_crosses_a_process_boundary():
    rec = gate.run(seed=0, n_vars=16, n_obs=200, cross_process=True, anticontrol=True)
    assert rec.passed
    assert rec.bitwise_identical
    assert rec.cross_process
    assert "cross-process: bitwise=True" in rec.detail


def test_gate_has_demonstrated_power():
    """E0.1b, the anti-control. A gate that can only pass is not a gate.

    An in-process double-run of deterministic code is bitwise identical whether or not
    the harness is correctly pinned, so without this the gate would pass on a broken
    apparatus and "zero flips at FP32" would be published as a scientific negative when
    it is really an instrument that cannot measure.
    """
    assert gate.run_anticontrol(seed=0, n_vars=16, n_obs=200) is True


# ------------------------------------------------------------- undefined != same
def test_undefined_decisions_cannot_be_silently_counted_as_agreement():
    """Failure mode A, the one that would manufacture a reassuring negative result.

    NaN > threshold is False, i.e. "not rejected", i.e. the same decision as a
    well-behaved test that was also not rejected. Folding those together biases the flip
    rate downward in exactly the formats the paper is about.
    """
    C, ts, n = _setup(n_vars=10)
    ref = engine.evaluate(C, ts, n)
    bad = np.array(ref.stat, copy=True)
    bad[:5] = np.nan
    var = engine.CellResult(
        testset_digest=ref.testset_digest, rho=ref.rho, stat=bad,
        status=ref.status, kappa_proxy=ref.kappa_proxy,
        min_diag=ref.min_diag, n_obs=ref.n_obs,
    )
    with pytest.raises(ValueError, match="undefined"):
        metrics.flip_counts(ref, var, 0.01, undefined_policy="error")

    fc = metrics.flip_counts(ref, var, 0.01, undefined_policy="exclude")
    assert fc.n_undefined == 5
    assert fc.n_compared == len(ref) - 5


def test_fp8_saturation_shows_up_as_status_not_as_agreement():
    """The concrete instance of the same failure mode: E4M3 rounds any |r| >= 0.96875 to
    exactly 1.0, and in a strongly-correlated regime that is the DOMINANT behaviour."""
    rng = np.random.default_rng(3)
    C, _ = corr.with_condition_number(6, 1e8, rng)
    ts = T.from_adjacency(np.ones((6, 6), dtype=bool) ^ np.eye(6, dtype=bool), 2)
    res = engine.evaluate(C, ts, 200, fmt="float8_e4m3")
    counts = res.status_counts()
    assert counts.get("OK", 0) < len(res), "expected degenerate outputs at e4m3, got %r" % counts


# ---------------------------------------------------------------- amplification
def test_amplification_is_dimensionless_and_undefined_at_zero_flips():
    """SHD / flip_rate has units of edges and moves with N for no scientific reason.
    The quantity C1 actually claims is edges changed PER FLIPPED DECISION."""
    a = metrics.Amplification(n_flipped_tests=10, edges_changed=30, shd=30, flip_rate=0.001)
    assert a.amplification == 3.0
    zero = metrics.Amplification(n_flipped_tests=0, edges_changed=0, shd=0, flip_rate=0.0)
    assert zero.amplification is None, "zero flips means UNDEFINED, not infinite or zero"


# ----------------------------------------------------------------------- the store
def test_cell_digest_requires_the_full_key():
    with pytest.raises(ValueError, match="missing"):
        store.cell_digest({"format": "float32"})


def test_cell_digest_changes_with_code_digest():
    """Without the code digest in the key, editing parcorr.py silently reuses stale
    shards computed by the old routine."""
    base = dict(
        experiment="e", mode="replay", dataset_id="d", testset_digest="t", seed=0,
        format="float32", parcorr_variant="inv", ztransform_variant="log1p",
        code_digest="aaa",
    )
    other = dict(base, code_digest="bbb")
    assert store.cell_digest(base) != store.cell_digest(other)


def test_schema_violation_fails_loudly(tmp_path):
    """Silent column drift is what makes a multi-shard dataset unreadable weeks later."""
    import pyarrow as pa

    from precision_pc import schema

    st = store.RunStore(tmp_path, "r")
    st.init()
    complete = {
        "cell_digest": ["x"],
        "level": np.array([0], dtype=np.int8),
        "log10kappa_bin": np.array([1], dtype=np.int8),
        "count": np.array([1], dtype=np.uint64),
    }
    with pytest.raises(ValueError, match="unexpected column"):
        st.write_shard("hist", "abc", pa.table(dict(complete, surprise=[1])))
    with pytest.raises(ValueError, match="missing column"):
        st.write_shard("hist", "abc", pa.table({"cell_digest": ["x"]}))
    assert schema.SCHEMA_VERSION >= 1


def test_shard_existence_means_done(tmp_path):
    import pyarrow as pa

    st = store.RunStore(tmp_path, "r")
    st.init()
    assert not st.is_done("deadbeef")
    st.write_shard(
        "hist", "deadbeef",
        pa.table({
            "cell_digest": ["deadbeef"], "level": np.array([0], dtype=np.int8),
            "log10kappa_bin": np.array([1], dtype=np.int8),
            "count": np.array([5], dtype=np.uint64),
        }),
    )
    assert not st.is_done("deadbeef"), "only the `cells` shard marks a cell done"


def test_subsample_never_drops_a_flipped_or_degenerate_test():
    """The retained-unconditionally rule. Sampling the numerator would be indefensible."""
    C, ts, n = _setup(n_vars=30)
    res = engine.evaluate(C, ts, n, fmt="bfloat16")
    flipped = np.zeros(len(res), dtype=bool)
    flipped[[1, 7, 11]] = True
    keep = store.sample_mask(res, alphas=[0.01], rate=0.001, seed=0, flipped=flipped)
    assert keep[[1, 7, 11]].all()
    assert keep[res.status != parcorr.OK].all()
    assert keep[np.isnan(res.stat)].all()
