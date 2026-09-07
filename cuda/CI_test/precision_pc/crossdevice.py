"""E0.2, restructured: cross-device agreement, decomposed into attributable arms.

The proposal frames E0.2 as "GPU and CPU may disagree at identical precision, because the
GPU path uses CUDA's `log1p()` and `normcdf()` while the CPU reference uses R's `pnorm`",
and expects the effect at alpha = 0.01 "where implementations diverge most".

**That framing is predisposed to a null, and the reason is worth stating.** At alpha =
0.01 the critical value is 2.5758, and every credible normal-CDF implementation agrees
there to about one ulp -- CUDA's `normcdf` is documented at 5 ulp, `log1p` at 1 ulp, `log`
at 1 ulp. Implementations diverge meaningfully only for arguments above roughly 8, where
p < 1e-15. **The region where implementations disagree is disjoint from the region where
decisions are marginal.** Run as specified, E0.2 measures a difference that cannot reach a
threshold.

What *can* move a decision at alpha = 0.01 is the other half of the pipeline: different
summation order in the small solve, and FMA contraction, perturb rho at the 1e-16 level,
and `atanh` amplifies that by 1/(1 - r^2), which diverges as |r| -> 1. That is largest
exactly in the phi -> 1 time-series regime this study already generates.

So the experiment splits in two, and each arm holds the other fixed:

    arm A   transcendental implementation, with rho held IDENTICAL
    arm B   linear algebra (device, route, summation order), with the transform held
            identical -- host-side float64 log1p in every case

`gpu.evaluate` was written for arm B: it computes rho on the device but the statistic on
the host in float64, so a CPU/GPU comparison isolates the linear algebra and cannot be
contaminated by a difference in the transform.

Both arms are reported against a measured **cross-machine floor**: two CPUs (x86_64 Linux
vs AMD64 Windows, different BLAS, different numpy) already disagree by up to 7.2e-13
relative with zero decision flips. Nothing at or below that is evidence about a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import corr, engine, parcorr, stats
from .testset import TestSet

#: Measured on 2340 level-0 tests, Nibi vs this laptop, both FP64. See LAB_BOOK.
CROSS_MACHINE_FLOOR_REL = 7.2e-13


# ------------------------------------------------------------------------- arm A
def transcendental_arm(rho: np.ndarray, n_obs: int, k: int = 0) -> Dict[str, np.ndarray]:
    """Vary the transform implementation with rho held identical.

    Returns each implementation's statistic on the same inputs, so the caller can ask the
    only question that matters: does the disagreement reach a decision threshold, or does
    it live in a tail no threshold visits?
    """
    rho = np.asarray(rho, dtype=np.float64)
    scale = math.sqrt(n_obs - k - 3.0)
    out = {
        "log1p_numpy": scale * np.abs(stats.z_log1p(rho)),
        "naive_numpy": scale * np.abs(stats.z_naive(rho)),
        "atanh_numpy": scale * np.abs(np.arctanh(np.clip(rho, -1 + 1e-300, 1 - 1e-300))),
    }
    try:  # a genuine second implementation of the same functions
        import torch

        t = torch.as_tensor(rho, dtype=torch.float64)
        out["log1p_torch_cpu"] = (
            scale * (0.5 * (torch.log1p(t) - torch.log1p(-t))).abs()
        ).numpy()
        if torch.cuda.is_available():
            g = t.cuda()
            out["log1p_cuda"] = (
                scale * (0.5 * (torch.log1p(g) - torch.log1p(-g))).abs()
            ).cpu().numpy()
            out["atanh_cuda"] = (scale * torch.atanh(g).abs()).cpu().numpy()
    except ImportError:
        pass
    return out


def tail_profile(alphas=(0.001, 0.01, 0.05), stat_grid=None) -> List[dict]:
    """Where does transcendental disagreement live, relative to where decisions are made?

    This is the quantitative form of the objection to E0.2 as specified.
    """
    if stat_grid is None:
        stat_grid = np.array([1.0, 2.0, 2.5758, 3.29, 5.0, 8.0, 10.0, 20.0, 37.0])
    rows = []
    for s in np.asarray(stat_grid, dtype=np.float64):
        p_erfc = float(stats.pvalue(s, impl="erfc"))
        p_ndtr = float(stats.pvalue(s, impl="ndtr"))
        rel = abs(p_erfc - p_ndtr) / p_erfc if p_erfc > 0 else float("nan")
        rows.append(
            {
                "stat": float(s),
                "p_erfc": p_erfc,
                "p_ndtr": p_ndtr,
                "rel_disagreement": rel,
                "is_a_decision_boundary": bool(
                    any(abs(s - stats.crit(a)) < 1e-3 for a in alphas)
                ),
            }
        )
    return rows


# ------------------------------------------------------------------------- arm B
@dataclass(frozen=True)
class ArmBResult:
    backend: str
    max_abs_drho: float
    max_rel_drho: float
    max_abs_dstat: float
    flips: Dict[float, int]
    n_tests: int
    amplification_observed: float  # max d(stat)/d(rho), against the analytic prediction
    amplification_predicted: float  # max sqrt(n-k-3) / (1 - r^2)


def linalg_arm(
    C: np.ndarray,
    ts: TestSet,
    n_obs: int,
    *,
    reference: str = "chol",
    backends=("inv", "regression", "gpu"),
    alphas=(0.001, 0.01, 0.05),
) -> List[ArmBResult]:
    """Vary how rho is computed, with the transform held identical.

    Every backend's statistic is formed the same way -- host-side float64 `log1p` -- so
    any difference here is linear algebra: summation order, FMA contraction, the device.
    """
    ref = engine.evaluate(C, ts, n_obs, fmt="float64", variant=reference)
    lvl = ts.level.astype(np.float64)
    scale = np.sqrt(n_obs - lvl - 3.0)
    out = []
    for b in backends:
        if b == "gpu":
            from . import gpu

            if not gpu.available():
                continue
            var = gpu.evaluate(C, ts, n_obs, fmt="float64", compute_dtype="float64")
        else:
            var = engine.evaluate(C, ts, n_obs, fmt="float64", variant=b)

        m = np.isfinite(ref.rho) & np.isfinite(var.rho)
        drho = np.abs(ref.rho - var.rho)
        dstat = np.abs(ref.stat - var.stat)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = drho / np.maximum(np.abs(ref.rho), 1e-300)
            # atanh's derivative is 1/(1-r^2): the analytic amplification of a rho
            # perturbation into the statistic, and the reason |r| -> 1 is the danger zone.
            predicted = scale / np.maximum(1.0 - ref.rho**2, 1e-300)
            observed = np.where(drho > 0, dstat / np.maximum(drho, 1e-300), 0.0)

        flips = {}
        for a in alphas:
            c = stats.crit(a)
            flips[a] = int(((ref.stat > c) != (var.stat > c))[m].sum())

        out.append(
            ArmBResult(
                backend=b,
                max_abs_drho=float(np.nanmax(drho[m])) if m.any() else float("nan"),
                max_rel_drho=float(np.nanmax(rel[m])) if m.any() else float("nan"),
                max_abs_dstat=float(np.nanmax(dstat[m])) if m.any() else float("nan"),
                flips=flips,
                n_tests=int(m.sum()),
                amplification_observed=float(np.nanmax(observed[m])) if m.any() else 0.0,
                amplification_predicted=float(np.nanmax(predicted[m])) if m.any() else 0.0,
            )
        )
    return out


def near_unit_correlation_sweep(n_obs: int = 4000, k: int = 1) -> List[dict]:
    """How far a 1-ulp perturbation of rho travels, as |r| -> 1.

    The mechanism behind arm B: the statistic's sensitivity to rho is
    sqrt(n - k - 3) / (1 - r^2), so a perturbation invisible at r = 0.5 becomes large
    near r = 1. This is why the phi -> 1 regime is where a cross-device difference could
    plausibly reach a decision, and why alpha = 0.01 in the *tail* was the wrong place to
    look for one.
    """
    scale = math.sqrt(n_obs - k - 3.0)
    rows = []
    for r in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1 - 1e-6, 1 - 1e-8):
        one_ulp = np.spacing(r) if r else np.spacing(1.0)
        z0 = float(stats.z_log1p(np.float64(r)))
        z1 = float(stats.z_log1p(np.float64(r + one_ulp)))
        rows.append(
            {
                "r": r,
                "sensitivity": scale / (1.0 - r * r),
                "stat": scale * abs(z0),
                "dstat_per_ulp": scale * abs(z1 - z0),
            }
        )
    return rows
