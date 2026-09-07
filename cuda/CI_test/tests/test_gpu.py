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


# ============================================== Tier 5: calibration and the GPU job
def test_tf32_tie_cases_actually_contain_ties():
    """A calibration built from random floats would pass under EITHER rounding rule and
    prove nothing. Ties -- values whose dropped 13 mantissa bits are exactly 0x1000 -- are
    the only place nearest-even and nearest-away differ."""
    x = gpu.tf32_tie_cases(1024)
    dropped = x.view(np.uint32) & np.uint32(0x1FFF)
    assert int((dropped == np.uint32(0x1000)).sum()) > 500, "no ties in the sample"
    assert np.all(np.isfinite(x))


def test_calibration_refuses_on_pre_ampere_rather_than_guessing():
    """TF32 tensor cores need sm_80+. On older silicon the result would be a fallback,
    not a TF32 measurement, so the calibration must decline rather than report a number."""
    out = gpu.calibrate_tf32()
    assert isinstance(out, dict)
    if not out.get("ran"):
        assert "reason" in out
    else:
        assert out["device_mode"] in ("rne", "rna", "trunc")
        assert "assumption_holds" in out


def test_throughput_reports_a_cpu_baseline():
    """The CPU baseline must be the batched implementation, not a per-test loop -- the
    paper plan calls a strawman baseline out by name, and measuring against the loop once
    produced a 75x figure that a fair comparison turned into 0.56x."""
    rows = gpu.throughput(n_vars=12, level=1, reps=1, dtypes=("float64",))
    cpu = [r for r in rows if r["backend"] == "cpu_batched_numpy"]
    assert len(cpu) == 1
    assert cpu[0]["tests_per_s"] > 0


def test_fp64_fp32_ratio_is_none_without_both():
    assert gpu.fp64_fp32_ratio([{"backend": "gpu", "dtype": "float64", "seconds": 1.0}]) is None


def test_gpu_sbatch_renders_and_charges_the_gpu_account(tmp_path):
    """Alliance splits the allocation. Charging the GPU job to the _cpu association is
    rejected at sbatch time -- after the queue accepted it, and after a second factor."""
    import re

    from precision_pc.hpc.config import Config, HpcConfig
    from precision_pc.hpc.slurm import SlurmBackend

    cfg = Config(hpc=HpcConfig(host="h", user="u", account="def-x_cpu",
                               gpu_account="def-x_gpu", remote_root="/r", gpus="h100:1"))
    be = SlurmBackend(cfg)
    be._root = "/r"
    t = be.render_sbatch("tier5", "abc", template="gpu_job.sbatch")
    assert not re.findall(r"@[A-Z_]+@", t)
    assert "\r" not in t
    assert "#SBATCH --account=def-x_gpu" in t
    assert "#SBATCH --gpus-per-node=h100:1" in t
    # ... while the CPU array template still charges the CPU association.
    assert "#SBATCH --account=def-x_cpu" in be.render_sbatch("run", "abc")


def test_gpu_job_script_ships():
    from precision_pc.hpc import slurm

    for name in ("gpu_job.sbatch", "gpu_job.py"):
        p = slurm.REMOTE_DIR / name
        assert p.exists(), "%s missing" % name
    assert b"\r" not in (slurm.REMOTE_DIR / "gpu_job.sbatch").read_bytes()
