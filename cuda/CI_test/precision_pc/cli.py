"""Command line interface.

    python -m precision_pc doctor              # local only, no computation
    python -m precision_pc gate                # measure the E0.1 noise floor
    python -m precision_pc run configs/x.yaml  # execute a config
    python -m precision_pc report <run_id>     # summarise a finished run
    python -m precision_pc cells configs/x.yaml    # list cell digests  (HPC fan-out)
    python -m precision_pc run-cell configs/x.yaml <digest>   # one cell (HPC fan-out)

`cells` and `run-cell` are the entire interface the cluster fan-out needs from the SCIENCE
side: a work-stealing array task loops over cells, claims one, computes it. That works
because cells are content-addressed and a shard's existence means done -- no completion
protocol, no partitioning, no shared state.

The `hpc` group is the TRANSPORT around that, and it does define a Backend -- not because
the fan-out needs the abstraction, but because a second implementation of it (`mock`) is
what lets the whole submit -> poll -> fetch loop be tested on a laptop with no cluster.

Exit codes: 0 ok, 1 local/config error, 2 a gate or check failed, 3 work still pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path("runs")


def _fmt_bool(b) -> str:
    return "yes" if b else "NO"


# ---------------------------------------------------------------------------- doctor
def cmd_doctor(a) -> int:
    import precision_pc
    from precision_pc import env, formats, provenance, stats

    ok = True
    print("versions")
    info = provenance.env_info()
    print("  python %s   %s" % (info["python"], info["platform"]))
    for k, v in info["versions"].items():
        print("  %-10s %s" % (k, v))

    print("\nthread pinning (the E0.1 precondition)")
    pinned, problems = env.verify()
    for var, val in env.report().items():
        print("  %-24s %s" % (var, val))
    for p in info["blas"]["threadpools"]:
        print("  %-24s num_threads=%s  (%s)" % (p["internal_api"], p["num_threads"], p["version"]))
    if problems:
        ok = False
        for p in problems:
            print("  PROBLEM: %s" % p)
    print("  verified: %s" % _fmt_bool(pinned))
    if not pinned:
        print("  fix: run via `python -m precision_pc`, or `import precision_pc` before numpy")

    print("\nBLAS")
    print("  %s %s" % (info["blas"]["vendor"], info["blas"]["version"]))

    print("\nprecision ladder")
    for name in formats.LADDER:
        f = formats.get(name)
        sat = "never" if f.saturates_at is None else ("|r| >= %.6f" % f.saturates_at)
        print(
            "  %-13s mantissa=%2d exp=%2d  digits=%5.2f  saturates: %s"
            % (name, f.mantissa_bits, f.exponent_bits, f.decimal_digits, sat)
        )

    print("\nthreshold equivalence (p < alpha  <=>  stat > crit)")
    import numpy as np

    rng = np.random.default_rng(0)
    s = rng.uniform(0.0, 8.0, 50_000)
    for alpha in (0.001, 0.01, 0.05):
        good = bool(np.all((stats.pvalue(s) < alpha) == (s > stats.crit(alpha))))
        ok &= good
        print("  alpha=%-6s crit=%.6f  holds: %s" % (alpha, stats.crit(alpha), _fmt_bool(good)))

    print("\nprovenance")
    print("  code_digest %s" % provenance.code_digest()[:16])
    print("  env_digest  %s" % provenance.env_digest()[:16])
    g = provenance.git_info()
    print("  git %s%s" % ((g["sha"] or "?")[:12], " (DIRTY)" if g["dirty"] else ""))

    print("\n%s" % ("ALL GOOD" if ok else "PROBLEMS ABOVE -- fix before running experiments"))
    return 0 if ok else 2


# ------------------------------------------------------------------------------ gate
def cmd_gate(a) -> int:
    from precision_pc import gate

    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    print("measuring the E0.1 noise floor (FP64 vs FP64, cross-process)...")
    rec = gate.run(n_vars=a.n_vars, n_obs=a.n_obs, cross_process=not a.no_cross_process)
    gate.save(root, rec)
    print("  bitwise identical  : %s" % _fmt_bool(rec.bitwise_identical))
    print("  flip rate          : %r" % rec.flip_rate)
    print("  tests              : %d" % rec.n_tests)
    print("  anti-control (E0.1b): %s" % _fmt_bool(rec.anticontrol_detected))
    print("  detail             : %s" % rec.detail)
    if not rec.passed:
        print("\nGATE G0 HAS FIRED. The noise floor is not zero, so everything downstream")
        print("would measure non-determinism rather than precision. Stop and fix this.")
        return 2
    if rec.anticontrol_ran and not rec.anticontrol_detected:
        print("\nWARNING: the anti-control did NOT detect an injected difference.")
        print("The gate passed, but it has no demonstrated power -- a null result from")
        print("this apparatus would be uninterpretable. Investigate before proceeding.")
        return 2
    print("\ngate passed and has demonstrated power")
    return 0


# ------------------------------------------------------------------------------- run
def cmd_run(a) -> int:
    from precision_pc import runner

    cfg = runner.load_config(a.config)
    run_id = a.run_id or Path(a.config).stem
    print("run_id     : %s" % run_id)
    print("experiment : %s   mode: %s" % (cfg["experiment"], cfg["mode"]))
    print("factors    : %d format(s) x %d parcorr x %d ztransform x %d seed(s)" % (
        len(cfg["formats"]), len(cfg["parcorr_variants"]),
        len(cfg["ztransform_variants"]), len(cfg["seeds"]),
    ))

    if cfg["mode"] == "cascade":
        from precision_pc import cascade

        def cprog(fmt, seed, stable, n):
            print("  [%3d] %-13s seed=%-3d stable=%s" % (n, fmt, seed, stable))

        out = cascade.run(cfg, a.root, run_id, progress=cprog)
        print("%(cells_new)d new, %(cells_skipped)d already done" % out)
        return 0

    def progress(cell, n, total):
        print("  [%3d] %-13s seed=%-3d %s" % (n, cell.fmt, cell.seed, cell.digest[:12]))

    out = runner.run_config(cfg, a.root, run_id, progress=progress)
    print("\n%(cells_new)d new, %(cells_skipped)d already done, %(cells_total)d total" % out)
    return 0


# ----------------------------------------------------------------------------- cells
def cmd_cells(a) -> int:
    from precision_pc import runner, store

    cfg = runner.load_config(a.config)
    st = store.RunStore(a.root, a.run_id or Path(a.config).stem)
    done = st.done_cells()
    n_pending = 0
    for c in runner.enumerate_cells(cfg):
        state = "done" if c.digest in done else "pending"
        if state == "pending":
            n_pending += 1
        if not a.pending_only or state == "pending":
            print("%s\t%s\t%s\tseed=%d\t%s" % (c.digest, state, c.fmt, c.seed, c.parcorr_variant))
    return 3 if n_pending else 0


def cmd_run_cell(a) -> int:
    """Compute exactly one cell. The remote entrypoint for an array task."""
    from precision_pc import runner

    cfg = runner.load_config(a.config)
    cfg = dict(cfg)
    target = a.digest
    matched = [c for c in runner.enumerate_cells(cfg) if c.digest == target]
    if not matched:
        print("no cell with digest %s in this config" % target, file=sys.stderr)
        return 1
    c = matched[0]
    cfg["seeds"] = [c.seed]
    cfg["formats"] = sorted({c.fmt, cfg["reference"]["format"]})
    cfg["parcorr_variants"] = sorted({c.parcorr_variant, cfg["reference"]["parcorr_variant"]})
    cfg["ztransform_variants"] = sorted(
        {c.ztransform_variant, cfg["reference"]["ztransform_variant"]}
    )
    out = runner.run_config(cfg, a.root, a.run_id or Path(a.config).stem)
    print(json.dumps(out))
    return 0


# ---------------------------------------------------------------------------- report
def cmd_report(a) -> int:
    from precision_pc import store

    st = store.RunStore(a.root, a.run_id)
    if not st.manifest_path().exists():
        print("no run at %s" % st.dir, file=sys.stderr)
        return 1
    cells = st.read("cells")
    flips = st.read("flips")
    if cells.empty:
        print("run %s has no completed cells" % a.run_id)
        return 3

    import pandas as pd

    m = st.manifest()
    print("run        : %s" % a.run_id)
    print("code       : %s" % m["code_digest"][:16])
    print("noise floor: %s" % _fmt_bool(bool(cells["noise_floor_ok"].all())))
    print("cells      : %d" % len(cells))

    j = flips.merge(
        cells[["cell_digest", "format", "parcorr_variant", "ztransform_variant", "seed", "level"]],
        on=["cell_digest", "level"],
        how="left",
    )
    j["flip_rate"] = j["n_flip"] / j["n_compared"].replace(0, pd.NA)

    print("\nflip rate by format and alpha (mean over seeds, +- SE across seeds)")
    print("  tests share submatrices, so a binomial CI would be far too narrow;")
    print("  the independent replicate is the SEED.")
    for level in sorted(j["level"].unique()):
        sub = j[j["level"] == level]
        print("\n  level %d" % level)
        piv = sub.groupby(["format", "alpha"])["flip_rate"].agg(["mean", "sem", "count"])
        for (fmt, alpha), row in piv.iterrows():
            se = 0.0 if pd.isna(row["sem"]) else row["sem"]
            print(
                "    %-13s alpha=%-6s %8.4f%%  +- %.4f%%  (n_seeds=%d)"
                % (fmt, alpha, 100 * row["mean"], 100 * se, int(row["count"]))
            )

    print("\nstatus breakdown (degenerate outputs, per format)")
    agg = cells.groupby("format")[
        ["n_tests", "n_ok", "n_rho_unit", "n_rho_gt1", "n_chol_fail", "n_pinv", "n_undefined"]
    ].sum()
    for fmt, row in agg.iterrows():
        extra = {
            k: int(row[k])
            for k in ("n_rho_unit", "n_rho_gt1", "n_chol_fail", "n_pinv", "n_undefined")
            if row[k]
        }
        print("  %-13s n=%-8d ok=%-8d %s" % (fmt, row["n_tests"], row["n_ok"], extra or ""))
    return 0


# ------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    import precision_pc  # noqa: F401  -- pins BLAS before numpy loads

    ap = argparse.ArgumentParser(prog="python -m precision_pc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="run directory root")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check the environment; no computation, no network")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("gate", help="measure the E0.1 noise floor and record it")
    p.add_argument("--n-vars", type=int, default=40)
    p.add_argument("--n-obs", type=int, default=400)
    p.add_argument("--no-cross-process", action="store_true",
                   help="skip the subprocess arm (much weaker; not recommended)")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("run", help="execute a config")
    p.add_argument("config")
    p.add_argument("--run-id", default=None)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("cells", help="list cell digests and their state")
    p.add_argument("config")
    p.add_argument("--run-id", default=None)
    p.add_argument("--pending-only", action="store_true")
    p.set_defaults(fn=cmd_cells)

    p = sub.add_parser("run-cell", help="compute exactly one cell by digest")
    p.add_argument("config")
    p.add_argument("digest")
    p.add_argument("--run-id", default=None)
    p.set_defaults(fn=cmd_run_cell)

    p = sub.add_parser("report", help="summarise a finished run")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_report)

    # Imported lazily: the core commands must work on a machine with no cluster config,
    # no ssh client and no network, so the transport is never pulled in for `doctor`.
    from precision_pc.hpc import cli_hpc

    cli_hpc.add_subparsers(sub)

    a = ap.parse_args(argv)

    # An unhandled traceback out of the cluster path is the most likely place for a
    # hostname or account to leak into a log, so scrub them before anything can print.
    try:
        from precision_pc.hpc.config import install_excepthook

        install_excepthook()
    except Exception:  # noqa: BLE001 -- the cluster layer is optional
        pass

    try:
        return a.fn(a)
    except Exception as exc:  # noqa: BLE001
        # Order matters, and so does the guard. The cluster commands deliberately need
        # nothing from the scientific stack -- `hpc doctor` and `hpc stage` must work on
        # an interpreter with no numpy. Importing `gate` here (it imports numpy at module
        # scope) made the HANDLER raise while reporting a failure, burying the real error
        # under a ModuleNotFoundError. So: classify HPC errors first, and never let a
        # missing optional import turn a clean message into a second traceback.
        try:
            from precision_pc.hpc.errors import (
                AuthRequired, ConfigInvalid, ConfigMissing, HpcError,
                JobFailed, JobPending, ProvenanceMismatch, ResultsIncomplete,
            )
        except Exception:  # noqa: BLE001
            HpcError = ()  # type: ignore

        if HpcError:
            # JobPending is CONTROL FLOW, not failure: "still queued" has to be
            # distinguishable from "broken" by a script that polls.
            if isinstance(exc, JobPending):
                print("pending: %s" % exc)
                return 3
            if isinstance(exc, (ConfigMissing, ConfigInvalid, AuthRequired)):
                print("error: %s" % exc, file=sys.stderr)
                return 1
            if isinstance(exc, (JobFailed, ProvenanceMismatch, ResultsIncomplete)):
                print("failed: %s" % exc, file=sys.stderr)
                return 2
            if isinstance(exc, HpcError):
                print("error: %s" % exc, file=sys.stderr)
                return 1

        try:
            from precision_pc.gate import GateFailed
        except ImportError:
            pass  # no scientific stack here, so it cannot have been a gate failure
        else:
            if isinstance(exc, GateFailed):
                print("gate failed: %s" % exc, file=sys.stderr)
                return 2
        print("error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
