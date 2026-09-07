"""Remote entrypoint. One array task runs one copy of this.

    python worker.py <run_dir> <config.yaml> [max_seconds]

The loop is: claim a cell, compute it, write its shard, repeat. Nothing is partitioned in
advance -- see `claim.py` for why -- so every task runs the same code and the number of
tasks is a pure throughput knob.

THIS SCRIPT REPORTS, IT DOES NOT ABORT. A worker that dies on one bad cell wastes the
whole allocation and the queue wait that bought it. Every cell is wrapped: a failure
releases the claim, records the traceback, and moves to the next cell. The run ends with
a summary written to disk, so there is always something to read even when nothing worked.
"""

import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path


def _pin_threads():
    """Pin BLAS before numpy loads. On a cluster this is not merely about determinism.

    Each array task gets --cpus-per-task=1, but OpenBLAS sizes its pool from the number
    of cores it can SEE, not the number it was allocated. With 16 concurrent tasks on a
    64-core node, an unpinned BLAS would spawn 64 threads each: ~1000 threads fighting
    over 16 cores. That destroys throughput AND breaks the reduction-order determinism
    that E0.1 rests on, so every result from the run would be uninterpretable.
    """
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[v] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _isolate_caches(task_id: str):
    """Give this task its own scratch for every cache that would otherwise race.

    $SLURM_TMPDIR is an Alliance/Compute-Canada convention, not standard SLURM, so fall
    back through $TMPDIR to /tmp rather than assuming it exists.
    """
    base = None
    for var in ("SLURM_TMPDIR", "TMPDIR"):
        v = os.environ.get(var)
        if v and os.path.isdir(v):
            base = v
            break
    base = base or "/tmp"
    scratch = os.path.join(base, "precision-pc-" + task_id)
    os.makedirs(scratch, exist_ok=True)
    # numba writes .nbi/.nbc next to the source by default; concurrent tasks on a shared
    # filesystem race on those writes.
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(scratch, "numba")
    os.environ["MPLCONFIGDIR"] = os.path.join(scratch, "mpl")  # font cache races
    os.environ["XDG_CACHE_HOME"] = os.path.join(scratch, "xdg")
    os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(scratch, "pycache")
    os.environ["TMPDIR"] = scratch
    for k in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX"):
        os.makedirs(os.environ[k], exist_ok=True)
    return scratch


def main(argv):
    if len(argv) < 3:
        print("usage: worker.py <run_dir> <config.yaml> [max_seconds]", file=sys.stderr)
        return 2

    run_dir = Path(argv[1])
    config_path = Path(argv[2])
    max_seconds = float(argv[3]) if len(argv) > 3 else 0.0

    job = os.environ.get("SLURM_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    task_id = "%s_%s_%s" % (job, task, uuid.uuid4().hex[:6])

    _pin_threads()
    scratch = _isolate_caches(task_id)

    import precision_pc  # noqa: F401  -- pins again, and before numpy
    from precision_pc import runner, store
    from precision_pc.hpc.claim import Claimer, next_cell

    t0 = time.time()
    print("[worker] %s starting in %s" % (task_id, scratch), flush=True)

    ok, problems = precision_pc.env.verify()
    print("[worker] thread pinning verified: %s" % ok, flush=True)
    for p in problems:
        print("[worker]   PROBLEM: %s" % p, flush=True)

    cfg = runner.load_config(config_path)
    run_id = run_dir.name
    st = store.RunStore(run_dir.parent, run_id)
    st.init(config=cfg, extra={"run_id": run_id, "worker": task_id})

    marker = "graphs" if cfg["mode"] == "cascade" else "cells"
    claimer = Claimer(run_dir / "claims", task_id)
    reclaim_after = float(cfg.get("reclaim_after_s", 3600))

    if cfg["mode"] == "cascade":
        from precision_pc import cascade

        digests = [
            store.cell_digest(
                cascade.cell_key(cfg, dsid, seed, fmt, stable, __import__(
                    "precision_pc.provenance", fromlist=["code_digest"]).code_digest())
            )
            for seed in cfg["seeds"]
            for dsid in [cascade.dataset(cfg, int(seed))[2]]
            for stable in cfg["stable_variants"]
            for fmt in cfg["formats"]
        ]
    else:
        digests = [c.digest for c in runner.enumerate_cells(cfg)]

    print("[worker] %d cells in this config" % len(digests), flush=True)

    done_n = fail_n = 0
    failures = []
    longest = 0.0

    while True:
        if max_seconds and (time.time() - t0) + longest > max_seconds:
            print("[worker] stopping: not enough walltime left for another cell", flush=True)
            break
        d = next_cell(
            digests,
            claimer=claimer,
            is_done=lambda x: st.is_done(x, marker),
            reclaim_after_s=reclaim_after,
        )
        if d is None:
            print("[worker] no work left", flush=True)
            break

        c0 = time.time()
        try:
            _run_one(cfg, run_dir, run_id, d, marker, st)
            dt = time.time() - c0
            longest = max(longest, dt)
            done_n += 1
            print("[worker] done %s in %.1fs (%d)" % (d[:12], dt, done_n), flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_n += 1
            claimer.release(d)  # let another worker retry it
            failures.append({"cell": d, "error": repr(exc)[:400],
                             "traceback": traceback.format_exc()[-2000:]})
            print("[worker] FAILED %s: %r" % (d[:12], exc), flush=True)
            traceback.print_exc()

    claimer.cleanup()
    summary = {
        "worker": task_id,
        "elapsed_s": time.time() - t0,
        "cells_done": done_n,
        "cells_failed": fail_n,
        "failures": failures,
        "pinned": ok,
    }
    outdir = run_dir / "workers"
    outdir.mkdir(parents=True, exist_ok=True)
    from precision_pc import atomic

    atomic.write_text(outdir / (task_id + ".json"), json.dumps(summary, indent=2))
    print("[worker] complete: %d done, %d failed" % (done_n, fail_n), flush=True)
    # Exit 0 even with failures: a non-zero exit makes SLURM mark the array task FAILED,
    # which the client reads as "no results" rather than "results you will not like".
    return 0


def _run_one(cfg, run_dir, run_id, digest, marker, st):
    """Compute exactly the cell named by `digest`, and nothing else."""
    from precision_pc import runner

    if cfg["mode"] == "cascade":
        from precision_pc import cascade

        sub = dict(cfg)
        # Narrow to the one seed/format this cell needs. The reference format must stay in
        # the list: cascade compares against it and the union arm needs its trajectory.
        for seed in cfg["seeds"]:
            for stable in cfg["stable_variants"]:
                dsid = cascade.dataset(cfg, int(seed))[2]
                for fmt in cfg["formats"]:
                    from precision_pc import provenance, store as _s

                    k = cascade.cell_key(cfg, dsid, seed, fmt, stable, provenance.code_digest())
                    if _s.cell_digest(k) == digest:
                        sub["seeds"] = [seed]
                        sub["stable_variants"] = [stable]
                        sub["formats"] = sorted({fmt, cfg["reference"]["format"]})
                        cascade.run(sub, run_dir.parent, run_id)
                        return
        raise KeyError("cell %s not found in this config" % digest)

    matched = [c for c in runner.enumerate_cells(cfg) if c.digest == digest]
    if not matched:
        raise KeyError("cell %s not found in this config" % digest)
    c = matched[0]
    sub = dict(cfg)
    sub["seeds"] = [c.seed]
    sub["formats"] = sorted({c.fmt, cfg["reference"]["format"]})
    sub["parcorr_variants"] = sorted({c.parcorr_variant, cfg["reference"]["parcorr_variant"]})
    sub["ztransform_variants"] = sorted(
        {c.ztransform_variant, cfg["reference"]["ztransform_variant"]}
    )
    runner.run_config(sub, run_dir.parent, run_id)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
