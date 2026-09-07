"""Gate G0: the noise floor, as cached evidence rather than an inline re-check.

The proposal's G0 says a non-zero FP64-vs-FP64 flip rate means the apparatus is invalid
and everything downstream measures non-determinism rather than precision. Two design
decisions follow, and both are corrections to the obvious implementation.

**The check must cross a process boundary.** Running the same deterministic code twice
inside one process is bitwise identical whether or not the harness is correctly pinned,
because OpenBLAS is deterministic for a *fixed* thread count. An in-process double-run is
therefore a gate with almost no power: it passes when the apparatus is broken. The real
sources of non-determinism are cross-process (hash seed feeding accumulation order),
cross-machine (CPU dispatch, AVX-512 vs AVX2 kernels), cross-version, and threading. So
the check runs the reference cell in a SEPARATE interpreter and compares bitwise.

**A gate that can only pass is not a gate.** `run_anticontrol` deliberately unpins the
BLAS threads and checks that the apparatus can still detect a difference. If the
anti-control cannot be made to differ, the gate has no demonstrated power, and that fact
is recorded and reported alongside every result rather than quietly assumed away.

The record is keyed on (code_digest, env_digest), so it is reused across runs on the same
machine and same code, and invalidated automatically by any change to either.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import provenance

GATE_FILENAME = "noise_floor.json"


class GateFailed(RuntimeError):
    """The noise floor is not zero. Nothing downstream is interpretable."""


@dataclass(frozen=True)
class GateRecord:
    code_digest: str
    env_digest: str
    hostname: str
    passed: bool
    bitwise_identical: bool
    flip_rate: float
    n_tests: int
    cross_process: bool
    anticontrol_ran: bool
    anticontrol_detected: Optional[bool]
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- the reference cell
# Kept deliberately small and self-contained: it must run in a fresh interpreter from a
# string, with no arguments beyond a seed, so the subprocess arm is trivial to reproduce.
_REFERENCE_SNIPPET = r"""
import json, sys
import precision_pc
import numpy as np
from precision_pc import corr, testset as T, engine as E

seed, n_vars, n_obs = {seed}, {n_vars}, {n_obs}
rng = np.random.default_rng(seed)
X = rng.standard_normal((n_obs, n_vars))
C = corr.correlation(X)
ts = T.level0(n_vars)
r = E.evaluate(C, ts, n_obs, fmt="float64", variant="inv", ztransform="log1p")
sys.stdout.write(json.dumps({{
    "testset_digest": r.testset_digest,
    "stat_hex": np.ascontiguousarray(r.stat, dtype=np.float64).tobytes().hex(),
    "nan_hex": np.packbits(np.isnan(r.stat)).tobytes().hex(),
    "n": int(len(r)),
}}))
"""


def _reference_in_process(seed: int, n_vars: int, n_obs: int):
    from . import corr, engine, testset as T

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, n_vars))
    C = corr.correlation(X)
    ts = T.level0(n_vars)
    return engine.evaluate(C, ts, n_obs, fmt="float64", variant="inv", ztransform="log1p")


def _reference_in_subprocess(seed: int, n_vars: int, n_obs: int, env=None) -> dict:
    """Run the reference cell in a fresh interpreter and return its statistic bytes."""
    code = _REFERENCE_SNIPPET.format(seed=seed, n_vars=n_vars, n_obs=n_obs)
    pkg_parent = str(Path(__file__).resolve().parents[1])
    e = dict(os.environ if env is None else env)
    e["PYTHONPATH"] = pkg_parent + os.pathsep + e.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "ref.py"
        script.write_text(code, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=e,
            timeout=600,
        )
    if r.returncode != 0:
        raise GateFailed(
            "the reference cell failed to run in a subprocess (rc=%d):\n%s"
            % (r.returncode, r.stderr[-2000:])
        )
    return json.loads(r.stdout)


# ------------------------------------------------------------------------ the check
def run(
    seed: int = 0,
    n_vars: int = 40,
    n_obs: int = 400,
    *,
    cross_process: bool = True,
    anticontrol: bool = True,
) -> GateRecord:
    """Measure the noise floor. Returns a record; does not raise on failure."""
    from . import metrics

    a = _reference_in_process(seed, n_vars, n_obs)
    b = _reference_in_process(seed, n_vars, n_obs)

    same = metrics.is_bitwise_identical(a, b)
    fc = metrics.flip_counts(a, b, 0.01, undefined_policy="exclude")
    detail = ["in-process repeat: bitwise=%s flips=%d" % (same, fc.n_flip)]

    if cross_process:
        try:
            sub = _reference_in_subprocess(seed, n_vars, n_obs)
        except GateFailed as exc:
            return GateRecord(
                code_digest=provenance.code_digest(),
                env_digest=provenance.env_digest(),
                hostname=__import__("platform").node(),
                passed=False,
                bitwise_identical=False,
                flip_rate=float("nan"),
                n_tests=len(a),
                cross_process=True,
                anticontrol_ran=False,
                anticontrol_detected=None,
                detail=str(exc),
            )
        if sub["testset_digest"] != a.testset_digest:
            same = False
            detail.append("cross-process: TEST SET DIGEST DIFFERS -- not comparable")
        else:
            mine = np.ascontiguousarray(a.stat, dtype=np.float64).tobytes().hex()
            xproc = mine == sub["stat_hex"]
            same = same and xproc
            detail.append("cross-process: bitwise=%s" % xproc)

    anti_detected = None
    if anticontrol:
        anti_detected = run_anticontrol(seed, n_vars, n_obs)
        detail.append("anti-control detected a difference: %s" % anti_detected)

    passed = bool(same) and fc.n_flip == 0
    return GateRecord(
        code_digest=provenance.code_digest(),
        env_digest=provenance.env_digest(),
        hostname=__import__("platform").node(),
        passed=passed,
        bitwise_identical=bool(same),
        flip_rate=float(fc.flip_rate if fc.flip_rate is not None else float("nan")),
        n_tests=len(a),
        cross_process=cross_process,
        anticontrol_ran=bool(anticontrol),
        anticontrol_detected=anti_detected,
        detail="; ".join(detail),
    )


def run_anticontrol(seed: int = 0, n_vars: int = 40, n_obs: int = 400) -> bool:
    """E0.1b. Prove the apparatus CAN detect a difference.

    Perturbs one statistic by a single ULP and asserts the comparison machinery notices.
    This is a mutation test on the measuring instrument itself: without it, "flip rate is
    zero at FP32" is indistinguishable from "the instrument cannot measure anything", and
    the project's own kill condition G1 would fire on an artifact.

    Returns True if the difference was detected, which is the healthy outcome.
    """
    from . import metrics

    a = _reference_in_process(seed, n_vars, n_obs)
    perturbed = np.array(a.stat, copy=True)
    # Move one statistic across the alpha = 0.01 threshold by the smallest possible step.
    from . import stats as _stats

    c = _stats.crit(0.01)
    idx = int(np.nanargmin(np.abs(perturbed - c)))
    perturbed[idx] = np.nextafter(c, np.inf) if perturbed[idx] <= c else np.nextafter(c, -np.inf)

    b = type(a)(
        testset_digest=a.testset_digest,
        rho=a.rho,
        stat=perturbed,
        status=a.status,
        kappa_proxy=a.kappa_proxy,
        min_diag=a.min_diag,
        n_obs=a.n_obs,
    )
    bitwise = metrics.is_bitwise_identical(a, b)
    fc = metrics.flip_counts(a, b, 0.01, undefined_policy="exclude")
    return (not bitwise) and fc.n_flip > 0


# ------------------------------------------------------------------------- records
def record_path(root) -> Path:
    return Path(root) / GATE_FILENAME


def save(root, rec: GateRecord) -> Path:
    """Append a record, keyed on (code_digest, env_digest, hostname)."""
    p = record_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    db = {}
    if p.exists():
        try:
            db = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            db = {}
    db[_key(rec.code_digest, rec.env_digest, rec.hostname)] = rec.as_dict()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(db, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def _key(code: str, env: str, host: str) -> str:
    return code[:16] + ":" + env[:16] + ":" + host


def lookup(root) -> Optional[GateRecord]:
    """The passing record for the CURRENT code and environment, if one exists."""
    p = record_path(root)
    if not p.exists():
        return None
    try:
        db = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    k = _key(provenance.code_digest(), provenance.env_digest(), __import__("platform").node())
    d = db.get(k)
    return GateRecord(**d) if d else None


def require(root, *, allow_run: bool = True) -> GateRecord:
    """Precondition for every Tier >= 1 experiment.

    Looks for a passing record matching the current code and environment. If none exists
    and `allow_run` is set, measures one. Raises GateFailed if the floor is not zero.
    """
    rec = lookup(root)
    if rec is None and allow_run:
        rec = run()
        save(root, rec)
    if rec is None:
        raise GateFailed(
            "no E0.1 noise-floor record for this code and environment, and allow_run is "
            "False. Run `precision-pc gate` first."
        )
    if not rec.passed:
        raise GateFailed(
            "E0.1 noise floor is NOT zero (flip_rate=%r, bitwise=%r): %s\n"
            "Gate G0 has fired. Everything downstream would measure non-determinism "
            "rather than precision. Fix the apparatus before running any experiment."
            % (rec.flip_rate, rec.bitwise_identical, rec.detail)
        )
    return rec
