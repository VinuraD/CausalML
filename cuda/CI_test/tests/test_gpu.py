"""The GPU engine. Skips cleanly on a machine with no CUDA device.

The contract these tests defend: a GPU cell and a CPU cell must be directly comparable,
which means the GPU path has to reproduce the CPU *status* semantics exactly. A GPU run
that quietly returned `status=OK` for a failed factorisation would recreate failure-mode A
-- degenerate output counted as agreement -- on new hardware, and every flip rate computed
against it would be biased downward.
"""

from __future__ import annotations

import numpy as np
import pytest

from precision_pc import corr, engine, gpu, metrics, parcorr
from precision_pc import testset as T

needs_gpu = pytest.mark.skipif(not gpu.available(), reason="no CUDA device")


def _instance(n_vars=16, n_obs=400, level=1, seed=0):
    rng = np.random.default_rng(seed)
    C = corr.correlation(rng.standard_normal((n_obs, n_vars)))
    adj = np.ones((n_vars, n_vars), dtype=bool)
    np.fill_diagonal(adj, False)
    return C, T.from_adjacency(adj, level), n_obs


# ------------------------------------------------------------------ host-side logic
def test_bucketing_partitions_every_test_exactly_once():
    """Uniform shapes per batch are what make the GPU path viable, and they address the
    warp-divergence trap: a batch mixing conditioning-set sizes leaves most lanes idle."""
    _, ts, _ = _instance(n_vars=10, level=2)
    buckets = gpu.bucket_by_level(ts)
    pos = np.concatenate([b.positions for b in buckets])
    assert sorted(pos.tolist()) == list(range(len(ts)))
    for b in buckets:
        assert b.index.shape == (len(b.positions), b.level + 2)
        # Column 0 is i, column 1 is j, the rest are S -- the engine's convention.
        assert np.array_equal(b.index[:, 0], ts.i[b.positions])
        assert np.array_equal(b.index[:, 1], ts.j[b.positions])


def test_available_and_device_info_never_raise_without_torch():
    info = gpu.device_info()
    assert isinstance(info, dict)
    assert isinstance(gpu.available(), bool)


def test_compute_dtype_is_validated():
    C, ts, n = _instance(n_vars=6)
    with pytest.raises(ValueError, match="compute_dtype"):
        gpu.evaluate(C, ts, n, compute_dtype="float999")


# ------------------------------------------------------------------- on the device
@needs_gpu
def test_fp64_gpu_matches_the_cpu_engine():
    """Same algorithm, same answers. The Cholesky route is the fair comparison: it is what
    the GPU path implements."""
    C, ts, n = _instance()
    cpu = engine.evaluate(C, ts, n, fmt="float64", variant="chol")
    g = gpu.evaluate(C, ts, n, fmt="float64", compute_dtype="float64")
    assert g.testset_digest == cpu.testset_digest
    assert np.nanmax(np.abs(cpu.rho - g.rho)) < 1e-12
    assert (cpu.status == g.status).all()
    assert metrics.flip_counts(cpu, g, 0.01, undefined_policy="exclude").n_flip == 0


@needs_gpu
def test_degenerate_input_reports_status_not_garbage():
    """The trap this module exists to avoid: `torch.linalg.inv` returns garbage rather
    than raising for singular batch elements, per element. `cholesky_ex` reports it."""
    n_vars = 6
    rng = np.random.default_rng(0)
    C = corr.correlation(rng.standard_normal((300, n_vars)))
    C = np.array(C)
    C[1, 0] = C[0, 1] = 1.0  # exact duplicate pair -> singular submatrices
    C = corr.symmetrise(C)
    np.fill_diagonal(C, 1.0)
    adj = np.ones((n_vars, n_vars), dtype=bool)
    np.fill_diagonal(adj, False)
    ts = T.from_adjacency(adj, 1)
    g = gpu.evaluate(C, ts, 300, fmt="float64", compute_dtype="float64")
    assert (g.status != parcorr.OK).any(), "a singular batch reported OK everywhere"
    bad = g.status != parcorr.OK
    assert np.isnan(g.stat[bad]).any() or np.isinf(g.stat[bad]).any()


@needs_gpu
def test_simulated_and_real_single_precision_agree():
    """The validation the whole CPU study rests on.

    Tier 1-3 SIMULATE low precision by rounding inputs and accumulating in float64. If
    that differs from actually computing in float32, every flip rate measured on the
    laptop describes a machine that does not exist.
    """
    rng = np.random.default_rng(3)
    C, _ = corr.with_condition_number(8, 1e6, rng)
    adj = np.ones((8, 8), dtype=bool)
    np.fill_diagonal(adj, False)
    ts = T.from_adjacency(adj, 2)

    ref = gpu.evaluate(C, ts, 200, fmt="float64", compute_dtype="float64")
    simulated = gpu.evaluate(C, ts, 200, fmt="float32", compute_dtype="float64")
    real = gpu.evaluate(C, ts, 200, fmt="float64", compute_dtype="float32")

    d_sim = np.nanmax(np.abs(simulated.rho - ref.rho))
    d_real = np.nanmax(np.abs(real.rho - ref.rho))
    # Same order of magnitude is the claim; they are different machines, not the same one.
    assert 0.1 < d_sim / d_real < 10.0, "simulation is not faithful: %.2e vs %.2e" % (d_sim, d_real)


@needs_gpu
def test_tf32_flags_are_pinned_and_reported():
    """PyTorch has changed its TF32 default across versions. An unrecorded flag turns TF32
    from a level of the ladder into an invisible contaminant of every float32 result."""
    flags = gpu.set_determinism(allow_tf32=False)
    assert flags["allow_tf32_matmul"] is False
    assert flags["float32_matmul_precision"] == "highest"
    flags = gpu.set_determinism(allow_tf32=True)
    assert flags["allow_tf32_matmul"] is True
    gpu.set_determinism(allow_tf32=False)  # leave the process as we found it


@needs_gpu
def test_gpu_is_bitwise_reproducible():
    """Cross-device comparison is meaningless if the device is not reproducible against
    itself first. This is the device-side analogue of the E0.1 gate."""
    C, ts, n = _instance(n_vars=12)
    a = gpu.evaluate(C, ts, n, compute_dtype="float64")
    b = gpu.evaluate(C, ts, n, compute_dtype="float64")
    assert metrics.is_bitwise_identical(a, b)


@needs_gpu
def test_batching_does_not_change_the_answer():
    """Batch size is a throughput knob; it must not be a numerical one."""
    C, ts, n = _instance(n_vars=14)
    whole = gpu.evaluate(C, ts, n, compute_dtype="float64")
    split = gpu.evaluate(C, ts, n, compute_dtype="float64", batch_size=97)
    assert metrics.is_bitwise_identical(whole, split)
