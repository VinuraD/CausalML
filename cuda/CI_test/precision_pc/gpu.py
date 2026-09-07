"""Batched conditional-independence tests on the GPU.

This is Exercise A from the learning path and paper-plan gate **G0**: *throughput vs batch
size, GPU against NumPy -- find the crossover; kill if the GPU never wins at realistic
batch sizes.*

Two things make this more than a port of `engine.py`.

**It separates storage precision from compute precision.** The CPU path *simulates* a
low-precision machine by rounding the inputs and accumulating in float64 -- low-precision
storage with a wide accumulator, which is how FP8 tensor-core matmul behaves. The GPU can
do the other thing too: actually compute in float32 or bfloat16. Having both means the
simulation can be **validated rather than assumed**, which matters because every Tier 1-3
number in this study came from the simulated path. `round_stage` records which was run.

**It must reproduce the CPU status contract exactly**, or it reintroduces the failure this
harness exists to prevent. From the learning path's own list of traps:

> `torch.linalg.inv` on singular matrices returns garbage rather than raising, per batch
> element. Use `cholesky_ex` and check the `info` tensor, or you'll silently emit false
> edges.

So every batch goes through `torch.linalg.cholesky_ex`, `info != 0` becomes `CHOL_FAIL`,
and the result carries the same `(rho, stat, status, kappa_proxy, min_diag)` as the CPU
engine. A GPU run that quietly returned `status=OK` for a failed factorisation would
recreate failure-mode A -- degenerate output counted as agreement -- on new hardware.

A note on TF32 that the code cannot enforce: TF32 is a **tensor-core matmul** mode. This
workload is batched Cholesky and triangular solves, which go through cuSOLVER, and
NVIDIA's documentation states cuSOLVER "always uses the precision defined by the API
call" -- TF32 is exposed there only through the iterative-refinement solvers, which this
does not use. So on this path TF32 is expected to be a *storage* format, not a compute
one, and `tf32_engaged()` records the flag state either way so the claim is checkable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import formats, parcorr, stats
from .engine import CellResult
from .testset import TestSet

#: Compute precisions the device path can actually use, as opposed to simulate.
COMPUTE_DTYPES = ("float64", "float32", "bfloat16", "float16")


def available() -> bool:
    """True when a CUDA device is usable. Never imports torch unless it is installed."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 -- a broken driver must not crash a CPU-only run
        return False


def device_info() -> dict:
    """Everything about the device that belongs in a manifest."""
    try:
        import torch
    except ImportError:
        return {"torch": None}
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = list(cap)
        # TF32 and bf16 tensor cores arrived with Ampere (sm_80). On anything older the
        # formats may still "work" via emulation or silently fall back, so a run on such a
        # device cannot be presented as a TF32 measurement.
        info["has_tf32_hardware"] = cap[0] >= 8
        info["has_bf16_hardware"] = cap[0] >= 8
        props = torch.cuda.get_device_properties(0)
        info["sm_count"] = props.multi_processor_count
        info["total_memory_mb"] = props.total_memory // (1024 * 1024)
    return info


def set_determinism(allow_tf32: bool = False) -> dict:
    """Pin the flags that decide reproducibility, and report what was pinned.

    TF32 is off by default here, deliberately. PyTorch has changed its default across
    versions, and an unrecorded flag turns TF32 from an experimental level of the ladder
    into an invisible contaminant of every float32 result.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("highest" if not allow_tf32 else "high")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 -- older torch, or an op with no deterministic kernel
        pass
    return {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _torch_dtype(name: str):
    import torch

    return {
        "float64": torch.float64,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


@dataclass(frozen=True)
class Bucket:
    """Tests sharing a conditioning-set size, so the batch has one uniform shape."""

    level: int
    positions: np.ndarray  # int64[M] -- index back into the TestSet
    index: np.ndarray  # int64[M, level + 2] -- the variables of each submatrix


def bucket_by_level(ts: TestSet) -> list:
    """Group tests by |S|.

    Uniform shapes per batch are what make the GPU path viable at all, and they directly
    address the warp-divergence trap: different pairs terminate at different conditioning
    sizes, so a mixed batch would leave most lanes idle. `TestSet` already stores `level`,
    so this costs nothing.
    """
    out = []
    lv = np.asarray(ts.level, dtype=np.int64)
    for level in sorted(set(int(x) for x in lv)):
        pos = np.flatnonzero(lv == level).astype(np.int64)
        idx = np.empty((pos.size, level + 2), dtype=np.int64)
        idx[:, 0] = ts.i[pos]
        idx[:, 1] = ts.j[pos]
        if level:
            idx[:, 2:] = ts.S[pos, :level]
        out.append(Bucket(level=level, positions=pos, index=idx))
    return out


def evaluate(
    C: np.ndarray,
    ts: TestSet,
    n_obs: int,
    *,
    fmt: str = "float64",
    compute_dtype: str = "float64",
    ztransform: str = "log1p",
    device: str = "cuda",
    allow_tf32: bool = False,
    batch_size: Optional[int] = None,
    keep_unit_diagonal: bool = True,
) -> CellResult:
    """Run every test in `ts` on the device, returning the same CellResult as the CPU path.

    `fmt` rounds the *inputs* (storage precision, comparable with the CPU simulation);
    `compute_dtype` sets the precision the arithmetic actually runs in. Setting
    `fmt="float64", compute_dtype="float32"` measures real single-precision computation;
    `fmt="float32", compute_dtype="float64"` reproduces what the CPU path simulates.
    """
    import torch

    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError("compute_dtype must be one of %r" % (COMPUTE_DTYPES,))
    set_determinism(allow_tf32=allow_tf32)

    # Round on the host, in float64, so storage precision means exactly what it means on
    # the CPU path and the two are comparable.
    Cr = formats.round_corr(np.asarray(C, dtype=np.float64), fmt,
                            keep_unit_diagonal=keep_unit_diagonal)
    dev = torch.device(device)
    dt = _torch_dtype(compute_dtype)
    Cd = torch.as_tensor(Cr, dtype=dt, device=dev)

    T = len(ts)
    rho = np.full(T, np.nan, dtype=np.float64)
    status = np.full(T, parcorr.NONFINITE, dtype=np.uint8)
    kappa = np.full(T, np.inf, dtype=np.float64)
    mindiag = np.full(T, np.nan, dtype=np.float64)

    for b in bucket_by_level(ts):
        m = b.level + 2
        idx_all = torch.as_tensor(b.index, dtype=torch.long, device=dev)
        step = batch_size or idx_all.shape[0] or 1
        for s in range(0, idx_all.shape[0], step):
            idx = idx_all[s : s + step]
            if idx.shape[0] == 0:
                continue
            # Gather [M, m, m] submatrices from the resident correlation tensor. Advanced
            # indexing on both axes at once, so nothing round-trips to the host.
            sub = Cd[idx[:, :, None], idx[:, None, :]]

            r, st, kp, md = _parcorr_batch(sub, m)

            pos = b.positions[s : s + step]
            rho[pos] = r
            status[pos] = st
            kappa[pos] = kp
            mindiag[pos] = md

    # The statistic is computed on the HOST in float64, always. The decision is a
    # threshold crossing, and computing it at reduced precision would confound "the
    # partial correlation moved" with "the transform moved" -- E0.3's whole point.
    lvl = ts.level.astype(np.float64)
    scale = n_obs - lvl - 3.0
    if np.any(scale <= 0):
        raise ValueError("n_obs=%d is too small for level %d" % (n_obs, int(lvl.max())))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = stats.z_transform(rho, ztransform)
        stat = np.sqrt(scale) * np.abs(z)

    return CellResult(
        testset_digest=ts.digest(),
        rho=rho,
        stat=stat,
        status=status,
        kappa_proxy=kappa,
        min_diag=mindiag,
        n_obs=int(n_obs),
    )


def _parcorr_batch(sub, m: int):
    """Batched partial correlation via Cholesky, with per-element status.

    `cholesky_ex` rather than `inv` or plain `cholesky`: it reports failure per batch
    element in an `info` tensor instead of raising for the whole batch or -- worse --
    returning garbage for the failed elements while looking successful.
    """
    import torch

    M = sub.shape[0]
    finite = torch.isfinite(sub).all(dim=(1, 2))

    L, info = torch.linalg.cholesky_ex(sub)
    okc = (info == 0) & finite

    diag = torch.diagonal(L, dim1=1, dim2=2).abs()
    dmin = diag.amin(dim=1)
    dmax = diag.amax(dim=1)
    # Same proxy as parcorr.cond_proxy_from_chol, so CPU and GPU kappas are comparable.
    kappa = torch.where(dmin > 0, (dmax / dmin) ** 2, torch.full_like(dmin, float("inf")))

    # Two right-hand sides give the three entries of the inverse that rho needs.
    e = torch.zeros((M, m, 2), dtype=sub.dtype, device=sub.device)
    e[:, 0, 0] = 1
    e[:, 1, 1] = 1
    Lsafe = torch.where(okc[:, None, None], L, torch.eye(m, dtype=sub.dtype, device=sub.device))
    x = torch.cholesky_solve(e, Lsafe)

    p00, p01, p11 = x[:, 0, 0], x[:, 0, 1], x[:, 1, 1]
    den = p00 * p11
    good = okc & (den > 0) & torch.isfinite(den)
    r = torch.where(good, -p01 / torch.sqrt(den.clamp_min(1e-300)), torch.full_like(den, float("nan")))

    # Status, in the same order of precedence as the CPU routes.
    st = torch.full((M,), int(parcorr.OK), dtype=torch.uint8, device=sub.device)
    st = torch.where(~finite, torch.tensor(int(parcorr.NONFINITE), dtype=torch.uint8, device=sub.device), st)
    st = torch.where(finite & (info != 0), torch.tensor(int(parcorr.CHOL_FAIL), dtype=torch.uint8, device=sub.device), st)
    st = torch.where(okc & ~good, torch.tensor(int(parcorr.NONFINITE), dtype=torch.uint8, device=sub.device), st)
    ar = r.abs()
    st = torch.where(good & (ar == 1), torch.tensor(int(parcorr.RHO_UNIT), dtype=torch.uint8, device=sub.device), st)
    st = torch.where(good & (ar > 1), torch.tensor(int(parcorr.RHO_GT1), dtype=torch.uint8, device=sub.device), st)

    return (
        r.double().cpu().numpy(),
        st.cpu().numpy().astype(np.uint8),
        kappa.double().cpu().numpy(),
        dmin.double().cpu().numpy(),
    )
