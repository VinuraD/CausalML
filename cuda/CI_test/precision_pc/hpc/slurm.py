"""The SLURM backend.

Everything here is shaped by one constraint: **each connection may cost an interactive
second factor.** So the design minimises connections rather than bytes.

    preflight   1 connection
    stage       1 connection (streamed; stage.sh takes minutes)
    submit      2 connections -- push one tar, then one script that untars and sbatches
    status      1 connection  -- a single _probe() returns everything
    fetch       2 connections -- tar the shards remotely, pull the tar

A naive implementation would spend five connections on submit alone (mkdir, three
uploads, sbatch), i.e. five second-factor prompts for one job.

**Source-of-truth order, enforced here and not merely documented:**

    output shards  >  sacct / squeue  >  local .state/jobs.json

The queue forgets: `squeue` lists only pending and running jobs, `sacct` is subject to
retention. So "is this cell done?" is answered by counting shards, and the job id cached
locally is only ever a hint used to look up state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

from .. import atomic
from . import Backend, register
from .config import REDACTOR
from .errors import (
    HpcError,
    JobFailed,
    JobPending,
    ProvenanceMismatch,
    ResultsIncomplete,
    StagingIncomplete,
)
from .ssh import Ssh, lf

log = logging.getLogger("precision_pc.hpc")

_HERE = Path(__file__).resolve().parent
REMOTE_DIR = _HERE / "remote"
PKG_ROOT = _HERE.parent  # precision_pc/

#: SLURM states that mean the job will never produce more output.
DEAD = {
    "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
    "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "SPECIAL_EXIT", "REVOKED",
}
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"}

#: Only these characters may appear in a remote path we interpolate unquoted.
_SAFE_PATH = re.compile(r"^[~$A-Za-z0-9_/.\-]+$")

#: Placeholder SHAPE, not a bare "@": SLURM's own --signal=B:TERM@90 contains one.
_TOKEN = re.compile(r"@[A-Z_]+@")

RESULT_TABLES = ("cells", "flips", "hist", "tests", "graphs", "levels", "cpdag")


def _safe(path: str, what: str) -> str:
    if not _SAFE_PATH.match(path):
        raise HpcError("refusing to interpolate unsafe %s into a remote command: %r" % (what, path))
    return path


@register("slurm")
class SlurmBackend(Backend):
    """Drives a SLURM cluster over one persistent SSH connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.hpc = cfg.hpc
        self.ssh = Ssh(cfg.hpc)
        self._root: Optional[str] = None  # resolved absolute remote_root
        self._state_dir = Path(".state")

    # ------------------------------------------------------------------ identity
    @staticmethod
    def run_digest(config: dict, code_digest: str) -> str:
        """Identity of a run: its resolved config AND the code that would execute it.

        Both, deliberately. Without the config two runs collide; without the code digest
        an edit to `parcorr.py` silently reuses shards produced by the previous routine.
        """
        from .. import provenance

        h = hashlib.sha256()
        h.update(provenance.canonical_json(config).encode())
        h.update(b"|")
        h.update(code_digest.encode())
        return h.hexdigest()

    def job_name(self, run_id: str, digest: str) -> str:
        # SLURM job names are truncated in squeue output; keep it short and greppable.
        return "ppc-%s-%s" % (run_id[:12], digest[:8])

    def remote_run_dir(self, run_id: str) -> str:
        return "%s/runs/%s" % (self._require_root(), run_id)

    def _require_root(self) -> str:
        if self._root is None:
            raise HpcError("remote root not resolved; call preflight() first")
        return self._root

    # ----------------------------------------------------------------- preflight
    def preflight(self) -> dict:
        """One connection. Proves auth, resolves the remote root, reports the environment.

        Resolves `~` and `$SCRATCH` to an absolute path ONCE, here, because SLURM
        expands neither inside `#SBATCH` directives -- an `--output` of `~/x/...` creates
        a directory literally named `~`. Uses logical `pwd` rather than `pwd -P`: the
        result must be absolute, but resolving the ~/scratch symlink to its physical
        target risks a path that is not mounted the same way on the compute nodes.
        """
        raw = _safe(self.hpc.remote_root, "remote_root")
        script = "\n".join([
            "set -u",
            'echo "LOGIN=$(hostname -s)"',
            'echo "USER=$(whoami)"',
            'echo "SCRATCH=${SCRATCH:-}"',
            "mkdir -p %s 2>/dev/null || true" % raw,
            # `pwd`, not `pwd -P`. The path must be ABSOLUTE, because SLURM expands
            # neither ~ nor $VARS inside #SBATCH directives -- an --output of ~/x/...
            # creates a directory literally named "~". But it must NOT be symlink-
            # resolved: on Alliance clusters ~/scratch is a symlink, and the physical
            # /scratch/... target is not guaranteed to be mounted identically on the
            # compute nodes. Logical pwd gives an absolute path that still travels
            # through the documented location.
            "ROOT=$(cd %s 2>/dev/null && pwd)" % raw,
            'echo "ROOT=${ROOT:-}"',
            'if [ -n "${ROOT:-}" ]; then mkdir -p "$ROOT/runs" "$ROOT/code"; fi',
            'echo "VENV=$( [ -d "${ROOT:-}/venv" ] && echo yes || echo no )"',
            'echo "SBATCH=$(command -v sbatch >/dev/null 2>&1 && echo yes || echo no)"',
            'echo "SQUEUE=$(command -v squeue >/dev/null 2>&1 && echo yes || echo no)"',
            'echo "ACCOUNTS=$(sacctmgr -nP show assoc user=$USER format=Account 2>/dev/null | sort -u | tr "\n" "," )"',
            'echo "DEFACCT=$(sacctmgr -nP show user $USER format=DefaultAccount 2>/dev/null | tail -1)"',
            'echo "PARTITIONS=$(sinfo -h -o %P 2>/dev/null | sort -u | tr "\n" "," )"',
            # Below: facts the mock cluster cannot know. Each would otherwise surface
            # as a failure minutes into staging, or after a queue wait.
            'echo "MAXARRAY=$(scontrol show config 2>/dev/null | grep -i MaxArraySize | tr -d " " | cut -d= -f2)"',
            'echo "PYMODULES=$(module -t avail python 2>&1 | grep -E "^python/3" | sort -u | tr "\n" "," )"',
            'if command -v avail_wheels >/dev/null 2>&1; then',
            '  echo "WHEELS=$(avail_wheels numpy scipy numba pyarrow pandas ml_dtypes threadpoolctl 2>/dev/null | base64 -w0)"',
            'else echo "WHEELS="; fi',
            'echo "QUOTA=$(diskusage_report 2>/dev/null | base64 -w0)"',
        ])
        # scrub=False: this output is PARSED. remote_root is a registered redaction value,
        # so scrubbing would turn ROOT into "***" and every derived path would be wrong.
        rc, out = self.ssh.run_script(script, scrub=False)
        if rc != 0:
            raise HpcError("preflight failed (rc=%d): %s" % (rc, REDACTOR.scrub(out)[-800:]))
        env = dict(
            line.split("=", 1) for line in out.splitlines() if "=" in line and line[0].isupper()
        )
        root = env.get("ROOT", "")
        if not root:
            raise HpcError(
                "could not resolve remote_root %r on the cluster -- does it exist and is it "
                "writable?" % self.hpc.remote_root
            )
        self._root = root
        REDACTOR.add(root)
        if env.get("SBATCH") != "yes":
            raise HpcError("no `sbatch` on the remote PATH -- is this a SLURM cluster?")
        accounts = [a for a in (env.get("ACCOUNTS") or "").split(",") if a]
        return {
            "backend": "slurm",
            "login_node": env.get("LOGIN"),
            "root": root,
            "venv_present": env.get("VENV") == "yes",
            "accounts": accounts,
            "default_account": (env.get("DEFACCT") or "").strip(),
            "configured_account_valid": (self.hpc.account in accounts) if accounts else None,
            "partitions": [p for p in (env.get("PARTITIONS") or "").split(",") if p],
            "max_array_size": (env.get("MAXARRAY") or "?").strip(),
            "python_modules": [m for m in (env.get("PYMODULES") or "").split(",") if m],
            "wheels": _b64(env.get("WHEELS", "")),
            "quota": _b64(env.get("QUOTA", "")),
        }

    # --------------------------------------------------------------------- stage
    def stage(self, *, force: bool = False) -> dict:
        """Run stage.sh on the LOGIN node, streamed.

        Login nodes have outbound internet; compute nodes do not. So the venv and any
        wheels missing from the cluster index must be resolved here, once.
        """
        if self._root is None:
            self.preflight()
        root = self._require_root()
        self._push_code()
        script = (REMOTE_DIR / "stage.sh").read_text(encoding="utf-8")
        rc, out = self.ssh.run_script(
            script, [root, self.hpc.modules, "force" if force else ""], scrub=False
        )
        text = REDACTOR.scrub(out)
        if rc != 0:
            raise StagingIncomplete("stage.sh failed (rc=%d):\n%s" % (rc, text[-3000:]))
        return {"staged": True, "root": root, "log": text}

    def _push_code(self) -> str:
        """Tar the package and push it in one connection."""
        root = self._require_root()
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "code.tar"
            with tarfile.open(bundle, "w") as tf:
                for f in sorted(PKG_ROOT.rglob("*.py")):
                    if "__pycache__" in f.parts:
                        continue
                    tf.add(f, arcname="precision_pc/" + str(f.relative_to(PKG_ROOT)).replace("\\", "/"),
                           filter=_tar_reset)
                for extra in sorted(REMOTE_DIR.glob("*.sbatch")) + sorted(REMOTE_DIR.glob("*.sh")):
                    tf.add(extra, arcname="precision_pc/hpc/remote/" + extra.name, filter=_tar_reset)
            staging = "%s/.ppc-code.tar" % root
            self.ssh.push(bundle, staging)
        rc, out = self.ssh.run_script(
            "set -eu\nmkdir -p '%s/code'\ntar xf '%s' -C '%s/code'\nrm -f '%s'\n"
            % (root, staging, root, staging),
            scrub=False,
        )
        if rc != 0:
            raise HpcError("failed to unpack code on the cluster: %s" % REDACTOR.scrub(out)[-500:])
        return root + "/code"

    # -------------------------------------------------------------------- submit
    def render_sbatch(self, run_id: str, digest: str, *, max_seconds: float = 0.0,
                      template: str = "array.sbatch") -> str:
        """Substitute the tokens and refuse anything that would fail on the cluster."""
        root = self._require_root()
        rundir = self.remote_run_dir(run_id)
        t = (REMOTE_DIR / template).read_text(encoding="utf-8")
        repl = {
            "@JOBNAME@": self.job_name(run_id, digest),
            # Alliance splits the allocation: a GPU job must be charged to the _gpu
            # association. Charging the wrong one is rejected at sbatch time, after
            # the queue wait and another second factor.
            "@ACCOUNT@": self.hpc.account_for(gpu="gpu" in template),
            "@GPU_ACCOUNT@": self.hpc.account_for(gpu=True),
            "@ARRAY@": self.hpc.array,
            "@CPUS@": str(self.hpc.cpus),
            "@MEM@": self.hpc.mem,
            "@TIME@": self.hpc.time,
            "@PARTITION@": ("#SBATCH --partition=" + self.hpc.partition) if self.hpc.partition else "",
            "@RUNDIR@": rundir,
            "@MODULES@": self.hpc.modules,
            "@REMOTE_ROOT@": root,
            "@CODEDIR@": root + "/code",
            "@MAXSECONDS@": str(int(max_seconds)),
            "@GPU_DIRECTIVE@": self.hpc.gpu_directive(),
            "@GPU_PARTITION@": ("#SBATCH --partition=" + self.hpc.gpu_partition)
            if self.hpc.gpu_partition else "",
            "@GPU_TIME@": self.hpc.gpu_time,
        }
        for k, v in repl.items():
            t = t.replace(k, v)
        left = sorted(set(_TOKEN.findall(t)))
        if left:
            raise HpcError("unsubstituted placeholder(s) in %s: %r" % (template, left))
        if "\r" in t:
            raise HpcError(
                "rendered %s contains a carriage return; SLURM rejects DOS line breaks "
                "outright (`Batch script contains DOS line breaks`)" % template
            )
        return t

    def submit(self, config_path, run_id: str, *, max_seconds: float = 0.0) -> dict:
        """Two connections. Idempotent: reattaches to an already-queued job."""
        from .. import provenance, runner

        if self._root is None:
            self.preflight()
        root = self._require_root()
        cfg = runner.load_config(config_path)
        digest = self.run_digest(cfg, provenance.code_digest())
        name = self.job_name(run_id, digest)
        rundir = self.remote_run_dir(run_id)

        self._push_code()

        sbatch = self.render_sbatch(run_id, digest, max_seconds=max_seconds)
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "job.tar"
            sb = Path(td) / "array.sbatch"
            sb.write_bytes(lf(sbatch))
            cf = Path(td) / "config.yaml"
            cf.write_bytes(lf(Path(config_path).read_text(encoding="utf-8")))
            with tarfile.open(bundle, "w") as tf:
                tf.add(sb, arcname="array.sbatch", filter=_tar_reset)
                tf.add(cf, arcname="config.yaml", filter=_tar_reset)
            staging = "%s/.ppc-job-%s.tar" % (root, digest[:12])
            self.ssh.push(bundle, staging)  # connection 1

        script = "\n".join([
            "set -eu",
            'JD="%s"' % rundir,
            'mkdir -p "$JD/logs" "$JD/claims" "$JD/workers"',
            "tar xf '%s' -C \"$JD\"" % staging,
            "rm -f '%s'" % staging,
            # The remote QUEUE is the source of truth for reattachment, not a local cache.
            'J=$(squeue -h -u "$USER" -n "%s" -o %%i 2>/dev/null | head -1)' % name,
            'if [ -z "$J" ]; then',
            '  J=$(cd "$JD" && sbatch --parsable array.sbatch); echo "NEW $J"',
            'else echo "REATTACH $J"; fi',
        ])
        rc, out = self.ssh.run_script(script, scrub=False)  # connection 2
        if rc != 0:
            raise JobFailed("sbatch failed (rc=%d): %s" % (rc, REDACTOR.scrub(out)[-1500:]))
        parts = out.strip().split()
        jobid = parts[-1] if parts else ""
        # An array submission returns "12345"; strip any "_0" suffix defensively.
        if not jobid.split("_")[0].isdigit():
            raise JobFailed("could not read a job id from sbatch output: %r" % parts[:6])
        action = "reattached" if "REATTACH" in out else "submitted"
        self._save_state(run_id, digest, jobid)
        return {"jobid": jobid, "action": action, "job_name": name,
                "run_dir": rundir, "digest": digest}

    # -------------------------------------------------------------------- status
    def status(self, run_id: str) -> dict:
        """One connection returning everything. Never blocks."""
        if self._root is None:
            self.preflight()
        rundir = self.remote_run_dir(run_id)
        cached = self._load_state(run_id)
        jobid = (cached or {}).get("jobid", "")

        counts = " ".join(
            'echo "N_%s=$(ls "$JD/%s"/*.parquet 2>/dev/null | wc -l)";' % (t.upper(), t)
            for t in RESULT_TABLES
        )
        script = "\n".join([
            "set -u",
            'JD="%s"' % rundir,
            'if [ ! -d "$JD" ]; then echo "MISSING=yes"; exit 0; fi',
            'echo "MISSING=no"',
            counts,
            'echo "N_CLAIMS=$(ls "$JD"/claims/*.claim 2>/dev/null | wc -l)"',
            'echo "N_WORKERS=$(ls "$JD"/workers/*.json 2>/dev/null | wc -l)"',
            'JOB="%s"' % jobid,
            'if [ -n "$JOB" ]; then',
            # squeue first (live), sacct second (historical) -- the queue forgets.
            '  S=$(squeue -h -j "$JOB" -o "%T|%r|%S" 2>/dev/null | head -1)',
            '  [ -z "$S" ] && S=$(sacct -n -X -j "$JOB" -o State,ExitCode 2>/dev/null | head -1)',
            '  echo "STATE=$S"',
            'else echo "STATE="; fi',
            # grep-then-tail, not tail: a Python traceback prints its cause in the MIDDLE
            # and the last lines are often unrelated teardown noise.
            'LOGF=$(ls -1t "$JD"/logs/*.out 2>/dev/null | head -1)',
            'if [ -n "${LOGF:-}" ]; then',
            '  { echo "== matched =="; ',
            '    grep -n -i -m 40 -E "error|traceback|cannot|not found|denied|oom|killed|GATE FAILED|FAILED" "$LOGF" 2>/dev/null;',
            '    echo; echo "== tail =="; tail -n 30 "$LOGF"; } | base64 -w0 > "$JD/.logcap" 2>/dev/null',
            '  echo "LOG=$(cat "$JD/.logcap" 2>/dev/null || true)"; rm -f "$JD/.logcap"',
            'else echo "LOG="; fi',
        ])
        rc, out = self.ssh.run_script(script, scrub=False)
        if rc != 0:
            raise HpcError("status probe failed (rc=%d): %s" % (rc, REDACTOR.scrub(out)[-500:]))
        env = dict(
            line.split("=", 1) for line in out.splitlines() if "=" in line and line[0].isupper()
        )
        if env.get("MISSING") == "yes":
            return {"state": "absent", "done": 0, "jobid": jobid, "run_dir": rundir,
                    "reason": "no run directory on the cluster; submit first"}

        done = sum(int(env.get("N_%s" % t.upper(), 0) or 0) for t in ("cells", "graphs"))
        raw_state = (env.get("STATE") or "").strip()
        sfields = raw_state.split("|")
        sched = sfields[0].strip() if sfields and sfields[0] else ""
        base = sched.split()[0] if sched else ""
        reason = sfields[1].strip() if len(sfields) > 1 else ""
        start = sfields[2].strip() if len(sfields) > 2 else ""

        if base in LIVE:
            state = "running" if base == "RUNNING" else "pending"
        elif base in DEAD:
            state = "failed"
        elif base.startswith("COMPLETED"):
            state = "done"
        else:
            state = "unknown" if not base else base.lower()

        return {
            "state": state,
            "scheduler_state": base,
            "jobid": jobid,
            "done": done,
            "claims": int(env.get("N_CLAIMS", 0) or 0),
            "workers_reported": int(env.get("N_WORKERS", 0) or 0),
            "shards": {t: int(env.get("N_%s" % t.upper(), 0) or 0) for t in RESULT_TABLES},
            "reason": reason,
            "estimated_start": start,
            "log": _b64(env.get("LOG", "")),
            "run_dir": rundir,
        }

    def wait_or_raise(self, run_id: str) -> dict:
        """Return the status when complete; raise JobPending / JobFailed otherwise.

        This is the control-flow shape the CLI maps onto exit codes: pending is not an
        error, it is "come back later", and it must be distinguishable from failure.
        """
        st = self.status(run_id)
        if st["state"] in ("done",):
            return st
        if st["state"] == "failed":
            raise JobFailed(
                "job %s ended in state %s after producing %d cell(s).\n%s"
                % (st["jobid"], st["scheduler_state"], st["done"], _excerpt(st["log"]))
            )
        raise JobPending(
            "job %s is %s%s%s (%d cell(s) done so far)"
            % (st["jobid"], st["state"],
               " -- " + st["reason"] if st["reason"] else "",
               ", est. start " + st["estimated_start"] if st["estimated_start"] else "",
               st["done"])
        )

    # --------------------------------------------------------------------- fetch
    def fetch(self, run_id: str, dest, *, require_complete: bool = False,
              expected: Optional[int] = None, allow_stale: bool = False) -> dict:
        """Tar the shards remotely, pull once, extract, verify provenance."""
        from .. import provenance

        if self._root is None:
            self.preflight()
        rundir = self.remote_run_dir(run_id)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        tables = " ".join(RESULT_TABLES)
        script = "\n".join([
            "set -eu",
            'JD="%s"' % rundir,
            'cd "$JD"',
            "F=''",
            "for t in %s testsets; do [ -d \"$t\" ] && F=\"$F $t\"; done" % tables,
            'for f in manifest.json config.yaml; do [ -f "$f" ] && F="$F $f"; done',
            'if [ -z "$F" ]; then echo "EMPTY=yes"; exit 0; fi',
            'echo "EMPTY=no"',
            'tar cf "$JD/.results.tar" $F',
            'echo "BYTES=$(stat -c %s "$JD/.results.tar" 2>/dev/null || echo 0)"',
        ])
        rc, out = self.ssh.run_script(script, scrub=False)
        if rc != 0:
            raise HpcError("failed to pack results (rc=%d): %s" % (rc, REDACTOR.scrub(out)[-500:]))
        if "EMPTY=yes" in out:
            return {"files": 0, "cells": 0, "dest": str(dest), "note": "nothing to fetch yet"}

        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "results.tar"
            self.ssh.pull(rundir + "/.results.tar", local)
            self.ssh.run_script("rm -f '%s/.results.tar'" % rundir, scrub=False)
            with tarfile.open(local) as tf:
                try:
                    tf.extractall(td, filter="data")  # refuses absolute paths and ".."
                except TypeError:  # python < 3.12
                    tf.extractall(td)

            staged = Path(td)
            # Provenance check BEFORE anything is merged. Shards from a different code
            # digest would build a table whose rows came from two different programs --
            # no checksum downstream would ever catch that.
            remote_manifest = staged / "manifest.json"
            if remote_manifest.exists():
                m = json.loads(remote_manifest.read_text(encoding="utf-8"))
                mine = provenance.code_digest()
                theirs = m.get("code_digest", "")
                if theirs and theirs != mine and not allow_stale:
                    raise ProvenanceMismatch(
                        "remote results were produced by code_digest %s but this tree is %s. "
                        "Merging them would build a table whose rows came from two different "
                        "programs. Re-stage and resubmit, or pass allow_stale=True if you are "
                        "certain the difference cannot affect these cells."
                        % (theirs[:16], mine[:16])
                    )

                # The code digest is not enough. A cell digest covers the CODE but not the
                # ENVIRONMENT, so the same config built by the same source on two machines
                # produces the SAME shard filename with different contents -- and the
                # cluster and the laptop differ in every package version. Copying one over
                # the other would silently build a table whose rows were computed under two
                # different numpy builds, which no checksum downstream could detect.
                existing = dest / "manifest.json"
                if existing.exists() and not allow_stale:
                    prev = json.loads(existing.read_text(encoding="utf-8"))
                    prev_env = prev.get("env_digest", "")
                    new_env = m.get("env_digest", "")
                    has_shards = any(
                        (dest / t).is_dir() and any((dest / t).glob("*.parquet"))
                        for t in RESULT_TABLES
                    )
                    if has_shards and prev_env and new_env and prev_env != new_env:
                        raise ProvenanceMismatch(
                            "%s already holds shards computed under env_digest %s, but these "
                            "were computed under %s. Cell digests cover the code, not the "
                            "environment, so these shards share filenames while holding "
                            "results from two different package stacks -- merging them would "
                            "produce a table nobody could interpret. Fetch into a separate "
                            "run directory instead."
                            % (dest, prev_env[:16], new_env[:16])
                        )

            copied = 0
            for table in RESULT_TABLES + ("testsets",):
                src = staged / table
                if not src.is_dir():
                    continue
                (dest / table).mkdir(parents=True, exist_ok=True)
                for f in src.glob("*.parquet"):
                    shutil.copy2(f, dest / table / f.name)
                    copied += 1
            for extra in ("manifest.json", "config.yaml"):
                if (staged / extra).exists():
                    shutil.copy2(staged / extra, dest / extra)

        got = sum(len(list((dest / t).glob("*.parquet"))) for t in ("cells", "graphs")
                  if (dest / t).exists())
        if require_complete and expected is not None and got < expected:
            raise ResultsIncomplete(
                "fetched %d of %d expected cells. A partial table is not returned as if it "
                "were complete: a missing row is indistinguishable from a computed one."
                % (got, expected)
            )
        return {"files": copied, "cells": got, "dest": str(dest)}

    def worker_reports(self, run_id: str) -> list:
        """Per-worker summaries, so a failed cell is visible without reading every log."""
        rundir = self.remote_run_dir(run_id)
        rc, out = self.ssh.run_script(
            'for f in "%s"/workers/*.json; do [ -f "$f" ] && base64 -w0 "$f" && echo; done'
            % rundir,
            scrub=False,
        )
        reports = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                reports.append(json.loads(base64.b64decode(line)))
            except Exception:
                pass
        return reports

    # --------------------------------------------------------------- local cache
    def _state_file(self) -> Path:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        return self._state_dir / "jobs.json"

    def _save_state(self, run_id: str, digest: str, jobid: str) -> None:
        p = self._state_file()
        db = {}
        if p.exists():
            try:
                db = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                db = {}
        db[run_id] = {"jobid": jobid, "digest": digest, "t": time.time()}
        atomic.write_text(p, json.dumps(db, indent=2))

    def _load_state(self, run_id: str) -> Optional[dict]:
        p = self._state_file()
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8")).get(run_id)
        except Exception:
            return None

    def forget(self, run_id: str) -> None:
        """Drop the cached job id.

        The escape hatch for a fix that lives OUTSIDE the run digest -- more walltime, a
        larger array, a corrected module string. When the fix is inside the digest, the
        digest changes on its own and this is unnecessary.
        """
        p = self._state_file()
        if not p.exists():
            return
        try:
            db = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        db.pop(run_id, None)
        atomic.write_text(p, json.dumps(db, indent=2))

    def cancel(self, run_id: str) -> dict:
        cached = self._load_state(run_id) or {}
        jobid = cached.get("jobid")
        if not jobid:
            return {"cancelled": False, "reason": "no cached job id for %r" % run_id}
        rc, out = self.ssh.run_script("scancel '%s' && echo CANCELLED" % jobid, scrub=False)
        return {"cancelled": rc == 0 and "CANCELLED" in out, "jobid": jobid}

    def close(self) -> None:
        self.ssh.close()


# --------------------------------------------------------------------- helpers
def _tar_reset(ti):
    """Normalise tar metadata: reproducible bundles, no clock-skew warnings on extract."""
    ti.mtime, ti.uid, ti.gid = 0, 0, 0
    ti.uname = ti.gname = ""
    return ti


def _b64(s: str) -> str:
    if not s:
        return ""
    try:
        return base64.b64decode(s).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _excerpt(log: str, n: int = 2000) -> str:
    return log[-n:] if log else "(no log captured)"
