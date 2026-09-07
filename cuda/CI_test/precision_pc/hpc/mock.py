"""A fake cluster in a temporary directory.

The point of this backend is that **it runs the real worker**. Everything above it -- the
claim protocol, the submit/poll/fetch loop, the shard layout, the completeness check --
is exercised end to end on a laptop, with no cluster, no queue wait and no second factor.
The only thing left untested is SSH transport and SLURM itself.

It also spawns real subprocesses rather than calling the worker in-process, because the
concurrency is the part most likely to be wrong: two workers racing for the same cell is
exactly what `os.link` claiming exists to handle, and an in-process fake would never
produce that race.

What this backend does NOT establish, stated plainly so nobody reads more into a green
run than is there: it says nothing about module loading, wheel availability on compute
nodes, `$SLURM_TMPDIR`, array-size limits, or whether the cluster's BLAS behaves like the
laptop's. Those are what `preflight` and a first small real submission are for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import Backend, register
from .errors import ResultsIncomplete

_WORKER = Path(__file__).resolve().parent / "remote" / "worker.py"


@register("mock")
class MockBackend(Backend):
    """Runs workers as local subprocesses against a local 'remote root'."""

    def __init__(self, cfg):
        self.cfg = cfg
        root = getattr(cfg.hpc, "remote_root", "") or ""
        self.root = Path(os.path.expanduser(root or ".mock-cluster")).resolve()
        self.n_workers = 2
        self._procs: dict = {}

    # ------------------------------------------------------------------ lifecycle
    def preflight(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        return {
            "backend": "mock",
            "root": str(self.root),
            "python": sys.executable,
            "note": "no cluster contacted; this backend fakes one on the local filesystem",
        }

    def stage(self, *, force: bool = False) -> dict:
        (self.root / "code").mkdir(parents=True, exist_ok=True)
        return {"staged": True, "root": str(self.root), "note": "mock: nothing to install"}

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def submit(self, config_path, run_id: str) -> dict:
        """Start `n_workers` real worker subprocesses against a shared run directory."""
        rd = self.run_dir(run_id)
        (rd / "logs").mkdir(parents=True, exist_ok=True)
        (rd / "claims").mkdir(parents=True, exist_ok=True)
        cfg_dst = rd / "config.yaml"
        cfg_dst.write_bytes(Path(config_path).read_bytes())

        repo = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")

        procs = []
        for i in range(self.n_workers):
            e = dict(env)
            e["SLURM_JOB_ID"] = "mock%s" % run_id
            e["SLURM_ARRAY_TASK_ID"] = str(i)
            log = open(rd / "logs" / ("worker-%d.out" % i), "wb")
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(_WORKER), str(rd), str(cfg_dst), "0"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=e,
                )
            )
        self._procs[run_id] = procs
        return {"jobid": "mock-" + run_id, "workers": len(procs), "run_dir": str(rd)}

    # ------------------------------------------------------------------- progress
    def status(self, run_id: str) -> dict:
        rd = self.run_dir(run_id)
        procs = self._procs.get(run_id, [])
        alive = sum(1 for p in procs if p.poll() is None)
        done = self._count_shards(rd)
        state = "running" if alive else ("done" if done else "pending")
        return {
            "state": state,
            "jobid": "mock-" + run_id,
            "workers_alive": alive,
            "done": done,
            "run_dir": str(rd),
            "reason": "",
        }

    @staticmethod
    def _count_shards(rd: Path) -> int:
        n = 0
        for t in ("cells", "graphs"):
            d = rd / t
            if d.exists():
                n += len(list(d.glob("*.parquet")))
        return n

    def wait(self, run_id: str, timeout: float = 600.0) -> dict:
        """Convenience for tests. A real backend must never block like this."""
        t0 = time.time()
        for p in self._procs.get(run_id, []):
            remaining = max(timeout - (time.time() - t0), 1)
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()
        return self.status(run_id)

    # ---------------------------------------------------------------------- fetch
    def fetch(self, run_id: str, dest, *, require_complete: bool = False,
              expected: Optional[int] = None) -> dict:
        """Copy shards out of the fake cluster into a local run directory."""
        import shutil

        rd = self.run_dir(run_id)
        dest = Path(dest)
        copied = 0
        for table in ("cells", "flips", "hist", "tests", "graphs", "levels", "cpdag"):
            src = rd / table
            if not src.exists():
                continue
            (dest / table).mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.parquet"):
                shutil.copy2(f, dest / table / f.name)
                copied += 1
        for extra in ("manifest.json", "config.yaml"):
            if (rd / extra).exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rd / extra, dest / extra)
        ts = rd / "testsets"
        if ts.exists():
            (dest / "testsets").mkdir(parents=True, exist_ok=True)
            for f in ts.glob("*.parquet"):
                shutil.copy2(f, dest / "testsets" / f.name)

        got = self._count_shards(rd)
        if require_complete and expected is not None and got < expected:
            raise ResultsIncomplete(
                "fetched %d of %d expected cells. Partial results are not returned as if "
                "they were complete: a missing row is indistinguishable from a computed "
                "one and every aggregate would be wrong by an unknown amount."
                % (got, expected)
            )
        return {"files": copied, "cells": got, "dest": str(dest)}

    def failures(self, run_id: str) -> list:
        """Worker summaries, so a failed cell is visible without reading every log."""
        rd = self.run_dir(run_id) / "workers"
        out = []
        if rd.exists():
            for f in sorted(rd.glob("*.json")):
                try:
                    out.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return out

    def close(self) -> None:
        for procs in self._procs.values():
            for p in procs:
                if p.poll() is None:
                    p.kill()
        self._procs.clear()
