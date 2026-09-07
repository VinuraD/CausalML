"""The `hpc` subcommand group.

Kept out of `precision_pc/cli.py` so that the core CLI never imports the cluster layer:
`doctor`, `gate` and `run` must work on a machine with no config, no ssh client and no
network, and an eager import would drag in tomllib and the whole transport for nothing.

Command shapes follow one rule: **announce the cost before paying it.** Anything that
opens a connection says so first, because on an MFA cluster a connection means the user
has to pick up their phone.

    hpc init        write a config template            no network
    hpc doctor      check everything checkable locally  no network
    hpc preflight   prove auth, resolve the root        exactly 1 connection
    hpc stage       build the remote venv               1 connection, several minutes
    hpc submit      push code + config, sbatch          2 connections
    hpc status      progress, queue state, log excerpt  1 connection
    hpc fetch       pull result shards                  2 connections
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ok(b) -> str:
    return "yes" if b else "NO"


def _load(a):
    from .config import load_config

    return load_config(getattr(a, "config_path", None))


def _backend(a):
    from . import make_backend

    cfg = _load(a)
    name = getattr(a, "backend", None) or cfg.backend
    return make_backend(name, cfg), cfg


# ------------------------------------------------------------------------- init
def cmd_init(a) -> int:
    from .config import HOME_CONFIG, write_template

    path = Path(a.path) if a.path else HOME_CONFIG
    p = write_template(path, force=a.force)
    print("wrote %s" % p)
    print()
    print("Fill in the four REQUIRED fields under [hpc]: host, user, account, remote_root.")
    print("This file holds no password and no token -- authentication is your SSH key plus")
    print("whatever second factor the cluster asks for, typed at a terminal.")
    return 0


# ----------------------------------------------------------------------- doctor
def cmd_doctor(a) -> int:
    """Everything checkable without touching the network."""
    import os

    from .config import ENV_VAR, HOME_CONFIG, default_config_path, warn_if_world_readable
    from .errors import ConfigInvalid, ConfigMissing
    from .ssh import find_ssh, supports_multiplexing

    ok = True
    print("config")
    p = default_config_path()
    print("  $%-28s %s" % (ENV_VAR, os.environ.get(ENV_VAR) or "(unset)"))
    print("  %-29s %s" % ("home location", HOME_CONFIG))
    if p is None:
        print("  found: NO -- run: python -m precision_pc hpc init")
        return 2
    print("  found: %s" % p)
    warn_if_world_readable(p)
    try:
        cfg = _load(a)
        print("  parsed: yes   backend=%s" % cfg.backend)
        for f in ("host", "user", "account", "remote_root"):
            v = getattr(cfg.hpc, f)
            print("  %-12s %s" % (f, "set" if v else "MISSING"))
            ok &= bool(v)
    except (ConfigMissing, ConfigInvalid) as exc:
        print("  parsed: NO -- %s" % exc)
        return 2

    print("\nssh")
    try:
        exe = find_ssh(cfg.hpc.ssh_exe)
        print("  client        %s" % exe)
        print("  multiplexing  %s" % _ok(supports_multiplexing(exe)))
    except Exception as exc:  # noqa: BLE001
        print("  client        NOT FOUND (%s)" % exc)
        ok = False
        exe = None
    key = os.path.expanduser(cfg.hpc.key_path)
    kh = os.path.expanduser(cfg.hpc.known_hosts)
    print("  key           %s  (%s)" % (_ok(os.path.exists(key)), key))
    print("  known_hosts   %s  (%s)" % (_ok(os.path.exists(kh)), kh))
    if not os.path.exists(kh):
        print("                strict host-key checking will fall back to your default file")

    print("\ninteractivity")
    tty = sys.stdin.isatty()
    print("  stdin is a TTY %s" % _ok(tty))
    if not tty:
        print("                 a second factor cannot be typed here, so any command that")
        print("                 opens a connection will refuse rather than hang")

    print("\nscheduler settings")
    for f in ("partition", "cpus", "mem", "time", "array", "modules"):
        print("  %-12s %r" % (f, getattr(cfg.hpc, f)))
    if cfg.hpc.cpus != 1:
        print("  WARNING: cpus != 1. Each task pins BLAS to one thread; more is wasted, and")
        print("           if pinning ever fails concurrent tasks oversubscribe the node and")
        print("           break the determinism E0.1 rests on.")
        ok = False

    print("\n%s" % ("ALL GOOD" if ok else "PROBLEMS ABOVE"))
    return 0 if ok else 2


# -------------------------------------------------------------------- preflight
def cmd_preflight(a) -> int:
    be, cfg = _backend(a)
    print("connecting to the cluster -- expect a second-factor prompt now")
    try:
        info = be.preflight()
    finally:
        if not a.keep_open:
            be.close()

    print("\ncluster")
    for k in ("backend", "login_node", "root", "venv_present", "max_array_size"):
        print("  %-18s %s" % (k, info.get(k)))

    print("\naccounting")
    accts = info.get("accounts") or []
    print("  %-18s %s" % ("accounts", ", ".join(accts) if accts else "(none reported)"))
    print("  %-18s %s" % ("default", info.get("default_account") or "(none)"))
    valid = info.get("configured_account_valid")
    print("  %-18s %s" % ("configured account",
                          {True: "valid", False: "NOT IN THE LIST", None: "could not verify"}[valid]))
    if valid is False:
        print("    The account in your config is not among your associations. sbatch would")
        print("    reject the job after the queue accepted it -- fix hpc.account first.")

    print("\npython modules")
    mods = info.get("python_modules") or []
    print("  %s" % (", ".join(mods) if mods else "(none matched python/3*)"))
    want = cfg.hpc.modules.split()[0] if cfg.hpc.modules else ""
    if want and mods and want not in mods:
        print("  WARNING: configured module %r is not in that list; `hpc stage` would fail" % want)

    if info.get("wheels"):
        print("\nwheelhouse (compute nodes have no internet, so anything missing here must")
        print("be downloaded on the login node by stage.sh)")
        for line in info["wheels"].splitlines()[:14]:
            print("  %s" % line)

    if info.get("quota"):
        print("\nquota")
        for line in info["quota"].splitlines()[:10]:
            print("  %s" % line)

    parts = info.get("partitions") or []
    print("\npartitions  %d available" % len(parts))
    if not cfg.hpc.partition:
        print("  partition is unset, which is correct on Alliance clusters: SLURM routes by")
        print("  the requested walltime, and naming a bucket by hand only narrows the pool.")
    return 0


def cmd_stage(a) -> int:
    be, cfg = _backend(a)
    print("staging on a login node -- this takes several minutes")
    try:
        out = be.stage(force=a.force)
    finally:
        be.close()
    print(out.get("log", ""))
    print("staged at %s" % out.get("root"))
    return 0


# ----------------------------------------------------------------- submit/status
def cmd_submit(a) -> int:
    be, cfg = _backend(a)
    run_id = a.run_id or Path(a.config).stem
    try:
        out = be.submit(a.config, run_id, max_seconds=a.max_seconds)
    finally:
        be.close()
    print("%s job %s (%s)" % (out["action"], out["jobid"], out["job_name"]))
    print("  run dir %s" % out["run_dir"])
    print("  digest  %s" % out["digest"][:16])
    print("\nCheck progress with: python -m precision_pc hpc status %s" % run_id)
    return 0


def cmd_status(a) -> int:
    from .errors import JobFailed, JobPending

    be, cfg = _backend(a)
    try:
        st = be.status(a.run_id)
    finally:
        be.close()
    print("state       %s%s" % (st["state"],
                                (" (%s)" % st.get("scheduler_state")) if st.get("scheduler_state") else ""))
    print("jobid       %s" % st.get("jobid"))
    print("cells done  %s" % st.get("done"))
    if st.get("claims") is not None:
        print("claims      %s" % st["claims"])
    if st.get("shards"):
        print("shards      %s" % {k: v for k, v in st["shards"].items() if v})
    if st.get("reason"):
        print("reason      %s" % st["reason"])
    if st.get("estimated_start"):
        print("est. start  %s" % st["estimated_start"])
    if a.log and st.get("log"):
        print("\n--- log ---\n%s" % st["log"])

    if st["state"] == "failed":
        print("\nJOB FAILED. Re-run with --log for the diagnostic excerpt.", file=sys.stderr)
        return 2
    if st["state"] in ("pending", "running", "absent"):
        return 3  # still pending -- distinct from failure, so it is scriptable
    return 0


def cmd_fetch(a) -> int:
    be, cfg = _backend(a)
    dest = Path(a.root) / a.run_id
    try:
        out = be.fetch(a.run_id, dest, allow_stale=a.allow_stale)
    finally:
        be.close()
    print("fetched %d file(s), %d cell(s) -> %s" % (out["files"], out["cells"], out["dest"]))
    if out.get("note"):
        print("  %s" % out["note"])
    print("\nSummarise with: python -m precision_pc report %s" % a.run_id)
    return 0


def cmd_reports(a) -> int:
    be, cfg = _backend(a)
    try:
        reps = be.worker_reports(a.run_id)
    finally:
        be.close()
    if not reps:
        print("no worker reports yet")
        return 3
    for r in reps:
        print("worker %-24s done=%-4d failed=%-3d pinned=%s  %.0fs"
              % (r.get("worker", "?")[:24], r.get("cells_done", 0),
                 r.get("cells_failed", 0), r.get("pinned"), r.get("elapsed_s", 0)))
        for f in r.get("failures", [])[:3]:
            print("    %s  %s" % (f.get("cell", "")[:12], f.get("error", "")[:160]))
    return 0


# ------------------------------------------------------------------ registration
def add_subparsers(sub) -> None:
    """Attach the `hpc` group to the main parser."""
    hp = sub.add_parser("hpc", help="cluster fan-out: init, doctor, preflight, stage, submit, status, fetch")
    hp.add_argument("--config-path", default=None, help="cluster config TOML (default: ~/.precision-pc/hpc.toml)")
    hp.add_argument("--backend", default=None, help="override the backend (slurm | mock)")
    hsub = hp.add_subparsers(dest="hpc_cmd", required=True)

    p = hsub.add_parser("init", help="write a config template (no network)")
    p.add_argument("--path", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = hsub.add_parser("doctor", help="check everything checkable locally (no network)")
    p.set_defaults(fn=cmd_doctor)

    p = hsub.add_parser("preflight", help="prove authentication; exactly one connection")
    p.add_argument("--keep-open", action="store_true")
    p.set_defaults(fn=cmd_preflight)

    p = hsub.add_parser("stage", help="build the remote venv on a login node")
    p.add_argument("--force", action="store_true", help="rebuild the venv from scratch")
    p.set_defaults(fn=cmd_stage)

    p = hsub.add_parser("submit", help="push code and config, submit the array job")
    p.add_argument("config")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="stop claiming new cells once this much walltime is used")
    p.set_defaults(fn=cmd_submit)

    p = hsub.add_parser("status", help="progress and queue state; exit 3 while pending")
    p.add_argument("run_id")
    p.add_argument("--log", action="store_true", help="print the diagnostic log excerpt")
    p.set_defaults(fn=cmd_status)

    p = hsub.add_parser("fetch", help="pull result shards into the local run directory")
    p.add_argument("run_id")
    p.add_argument("--allow-stale", action="store_true",
                   help="merge results even if the remote code digest differs (unsafe)")
    p.set_defaults(fn=cmd_fetch)

    p = hsub.add_parser("reports", help="per-worker summaries, including failed cells")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_reports)
