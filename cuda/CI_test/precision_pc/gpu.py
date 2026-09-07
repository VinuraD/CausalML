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


# ============================================================== TF32 calibration (B3)
#: Values chosen to sit exactly on TF32 rounding ties, where `rne` and `rna` differ.
#: A tie is a value whose dropped 13 mantissa bits are exactly 0x1000.
def tf32_tie_cases(n: int = 4096) -> np.ndarray:
    """float32 values lying exactly on a TF32 rounding tie, plus ordinary values.

    Ties are the only place round-to-nearest-EVEN and round-to-nearest-AWAY differ, so a
    calibration built from random floats would pass under either rule and prove nothing.
    """
    rng = np.random.default_rng(0)
    # Construct exact ties: arbitrary sign/exponent/high-mantissa, dropped bits = 0x1000.
    hi = rng.integers(0, 1 << 19, size=n, dtype=np.uint32) << np.uint32(13)
    expo = rng.integers(100, 150, size=n, dtype=np.uint32) << np.uint32(23)
    sign = rng.integers(0, 2, size=n, dtype=np.uint32) << np.uint32(31)
    ties = (sign | expo | (hi & np.uint32(0x007FE000)) | np.uint32(0x1000)).view(np.float32)
    ordinary = rng.uniform(-1, 1, n).astype(np.float32)
    x = np.concatenate([ties, ordinary])
    return x[np.isfinite(x)]


def calibrate_tf32(x: Optional[np.ndarray] = None) -> dict:
    """Compare this harness's simulated TF32 against the device's real one.

    `LAB_BOOK` §7 records that PTX exposes both `cvt.rn.tf32.f32` (nearest-even) and
    `cvt.rna.tf32.f32` (nearest, ties away), that which one a toolkit emits has varied,
    and that `formats.to_tf32` therefore ASSUMES `rne`. Every TF32 number in this study is
    provisional until that assumption is checked against hardware.

    The device's rounding is extracted by multiplying by the identity with TF32 enabled:
    the tensor core converts each input to TF32 before the multiply, and multiplying by
    one leaves exactly that rounded value.
    """
    import torch

    info = device_info()
    if not info.get("cuda_available"):
        return {"ran": False, "reason": "no CUDA device"}
    if not info.get("has_tf32_hardware"):
        return {
            "ran": False,
            "reason": "device is sm_%d%d; TF32 tensor cores need sm_80+"
            % tuple(info["capability"]),
            "device": info.get("device_name"),
        }

    x = tf32_tie_cases() if x is None else np.asarray(x, dtype=np.float32)
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        t = torch.as_tensor(x, dtype=torch.float32, device="cuda")
        eye = torch.eye(t.numel(), dtype=torch.float32, device="cuda")
        device_rounded = (t.unsqueeze(0) @ eye).squeeze(0).cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

    out = {
        "ran": True,
        "device": info.get("device_name"),
        "capability": info.get("capability"),
        "n_values": int(x.size),
        "n_ties": int(((x.view(np.uint32) & np.uint32(0x1FFF)) == np.uint32(0x1000)).sum()),
    }
    for mode in ("rne", "rna", "trunc"):
        ours = formats.to_tf32(x, mode=mode).astype(np.float32)
        agree = int(np.sum(ours == device_rounded))
        out["match_" + mode] = agree
        out["frac_" + mode] = float(agree / x.size)
    best = max(("rne", "rna", "trunc"), key=lambda m: out["frac_" + m])
    out["device_mode"] = best
    out["exact"] = out["frac_" + best] == 1.0
    out["harness_assumes"] = "rne"
    out["assumption_holds"] = bool(best == "rne" and out["exact"])
    return out


# ================================================================= E5.1 throughput
def throughput(
    n_vars: int = 60,
    n_obs: int = 500,
    level: int = 1,
    dtypes=("float64", "float32"),
    batch_sizes=(None,),
    reps: int = 3,
    include_cpu: bool = True,
) -> List[dict]:
    """E5.1. Per-format throughput, plus the batched-NumPy baseline on the same machine.

    Two quantities are reported and must never be conflated. The **within-device
    FP64:FP32 ratio** is architecture-dependent and normalised, so it is meaningful even
    on a small part -- it is what supports "FP64 is nearly free on a datacenter card and
    crippling on a consumer one". **Absolute throughput** needs a real card and says
    nothing on a 2-SM laptop GPU.

    Timings are reported with and without host-device transfer, following the SSDBM
    reporting precedent.
    """
    import time

    import torch

    from . import corr as _corr
    from . import engine as _engine
    from . import testset as _T

    rng = np.random.default_rng(0)
    C = _corr.correlation(rng.standard_normal((n_obs, n_vars)))
    adj = np.ones((n_vars, n_vars), dtype=bool)
    np.fill_diagonal(adj, False)
    ts = _T.from_adjacency(adj, level)
    rows: List[dict] = []

    def timeit(fn):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    if include_cpu:
        t = timeit(lambda: _engine.evaluate(C, ts, n_obs, fmt="float64", variant="chol"))
        rows.append({
            "backend": "cpu_batched_numpy", "dtype": "float64", "batch_size": None,
            "n_tests": len(ts), "seconds": t, "tests_per_s": len(ts) / t,
        })

    if available():
        for dt in dtypes:
            for bs in batch_sizes:
                try:
                    t = timeit(lambda: evaluate(C, ts, n_obs, compute_dtype=dt, batch_size=bs))
                except Exception as exc:  # noqa: BLE001 -- e.g. bf16 on pre-Ampere
                    rows.append({"backend": "gpu", "dtype": dt, "batch_size": bs,
                                 "n_tests": len(ts), "error": repr(exc)[:200]})
                    continue
                rows.append({
                    "backend": "gpu", "dtype": dt, "batch_size": bs,
                    "n_tests": len(ts), "seconds": t, "tests_per_s": len(ts) / t,
                })
    return rows


def fp64_fp32_ratio(rows: List[dict]) -> Optional[float]:
    """The normalised, architecture-dependent number E5.1 actually needs.

    Meaningful even on a small device, because it divides out the SM count.
    """
    g = {r["dtype"]: r for r in rows if r.get("backend") == "gpu" and "seconds" in r}
    if "float64" not in g or "float32" not in g:
        return None
    return g["float64"]["seconds"] / g["float32"]["seconds"]
