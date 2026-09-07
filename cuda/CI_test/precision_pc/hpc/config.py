"""Cluster configuration.

This file holds NO password and NO token. Authentication is an SSH key plus whatever
second factor the cluster demands, typed at a console. `_build` refuses a config
containing a password or token field rather than helpfully storing it.

The primary location is OUTSIDE the working tree (`~/.precision-pc/hpc.toml`), because a
file outside the repository cannot be committed even by accident -- strictly stronger
than a .gitignore rule.

Redaction covers three sinks: log records, exception text, and captured subprocess output.
The one rule that matters, learned the hard way in the reference implementation: **never
scrub output your own code parses.** Redaction exists to keep identifiers out of logs and
artifacts, not to corrupt the program's own inputs. `remote_root` is a registered value,
so scrubbing `pwd -P` output turns a path into `***` and path resolution fails after a
successful login. Callers that PRINT must scrub at the print site.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .errors import ConfigInvalid, ConfigMissing

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

log = logging.getLogger("precision_pc.hpc")

ENV_VAR = "PRECISION_PC_HPC_CONFIG"
HOME_CONFIG = Path.home() / ".precision-pc" / "hpc.toml"


# ------------------------------------------------------------------- redaction
class Secret:
    """Deliberately NOT a str subclass.

    A str subclass silently survives f-strings, `%s`, `json.dumps`, concatenation and
    `.format()`, so it leaks everywhere while looking safe. This one has to be unwrapped
    on purpose.
    """

    __slots__ = ("_v",)

    def __init__(self, value: str):
        self._v = value
        REDACTOR.add(value)

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "Secret(***)"

    def __str__(self) -> str:
        return "***"


class Redactor(logging.Filter):
    """One registry, three sinks: logs, exceptions, captured output."""

    def __init__(self):
        super().__init__()
        self._values: set = set()

    def add(self, value) -> None:
        # Short values would mangle unrelated text ("nb" appearing inside "numba").
        if value and isinstance(value, str) and len(value) >= 6:
            self._values.add(value)

    def scrub(self, text: str) -> str:
        if not text:
            return text
        for v in sorted(self._values, key=len, reverse=True):  # longest first
            text = text.replace(v, "***")
        return text

    def filter(self, record) -> bool:
        record.msg = self.scrub(str(record.msg))
        record.args = ()
        return True


REDACTOR = Redactor()
log.addFilter(REDACTOR)


def install_excepthook() -> None:
    """Scrub tracebacks. An unhandled exception is the most likely leak path."""
    import traceback

    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(REDACTOR.scrub(text))

    if prev is not hook:
        sys.excepthook = hook


# ---------------------------------------------------------------------- schema
@dataclass
class HpcConfig:
    host: str = ""
    user: str = ""
    account: str = ""
    remote_root: str = ""

    #: Alliance clusters split the allocation by resource type -- `def-<pi>_cpu` and
    #: `def-<pi>_gpu` are separate associations, and charging a CPU array job to the GPU
    #: account is rejected at sbatch time. Empty means "same as `account`", which is
    #: correct on clusters that do not split.
    gpu_account: str = ""

    ssh_exe: str = ""
    key_path: str = "~/.ssh/id_ed25519"
    known_hosts: str = "~/.ssh/known_hosts"

    # Scheduler shape. Tiers 1-3 are CPU-only and embarrassingly parallel; Tier 5 is the
    # only part that needs a GPU, and it is submitted as a separate job.
    partition: str = ""
    cpus: int = 1  # MUST be 1: see the note in array.sbatch about BLAS oversubscription
    mem: str = "4000M"
    time: str = "0-03:00"
    array: str = "0-15%8"
    modules: str = "python/3.12"

    # GPU job (Tier 5 only)
    gpu_partition: str = ""
    gpus: str = "h100:1"
    gpu_flavour: str = "gpus-per-node"  # gres | gpus-per-node | gpus-per-task
    gpu_time: str = "0-01:00"

    reclaim_after_s: int = 3600
    max_wait_s: int = 60

    def account_for(self, gpu: bool) -> str:
        """The association to charge. Falls back to `account` when no split exists."""
        if gpu and self.gpu_account:
            return self.gpu_account
        return self.account

    def gpu_directive(self) -> str:
        if self.gpu_flavour == "gres":
            return "#SBATCH --gres=gpu:" + self.gpus
        if self.gpu_flavour == "gpus-per-node":
            return "#SBATCH --gpus-per-node=" + self.gpus
        return "#SBATCH --gpus-per-task=" + self.gpus


@dataclass
class Config:
    backend: str = "slurm"
    hpc: HpcConfig = field(default_factory=HpcConfig)
    source: Optional[Path] = None

    def redacted(self) -> dict:
        """A dict safe to print AND to embed in result metadata."""
        d = dataclasses.asdict(self.hpc)
        for k in ("host", "user", "account", "remote_root", "key_path"):
            if d.get(k):
                d[k] = "***"
        return {"backend": self.backend, "hpc": d, "source": str(self.source)}


# ----------------------------------------------------------------------- load
def default_config_path() -> Optional[Path]:
    """Search order: $PRECISION_PC_HPC_CONFIG, then the home location."""
    env = os.environ.get(ENV_VAR)
    if env and Path(env).exists():
        return Path(env)
    if HOME_CONFIG.exists():
        return HOME_CONFIG
    return None


def _build(cls, data: dict, section: str, required=()):
    """Strict construction. Unknown keys are FATAL.

    A typo'd key would otherwise silently take its default -- and the defaults here
    (`cpus`, `modules`, `array`) are exactly the ones that decide whether the run is
    correct and whether it oversubscribes the node.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigInvalid(
            "[%s] has unknown key(s) %r; known keys are %r"
            % (section, sorted(unknown), sorted(known))
        )
    for k in ("password", "token", "secret"):
        if k in data:
            raise ConfigInvalid(
                "[%s] contains a %r. This client never stores one: authentication is an "
                "SSH key plus a second factor typed at a terminal." % (section, k)
            )
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise ConfigInvalid("[%s] is missing required key(s) %r" % (section, missing))
    return cls(**data)


def load_config(path=None) -> Config:
    p = Path(path) if path else default_config_path()
    if p is None or not p.exists():
        raise ConfigMissing(
            "no cluster config found. Looked at $%s and %s. "
            "Run: python -m precision_pc hpc init" % (ENV_VAR, HOME_CONFIG)
        )
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    hpc = _build(
        HpcConfig,
        raw.get("hpc", {}),
        "hpc",
        required=("host", "user", "account", "remote_root"),
    )
    cfg = Config(backend=raw.get("backend", "slurm"), hpc=hpc, source=p)

    # Register identifiers at load time so call sites never have to remember.
    for v in (hpc.host, hpc.user, hpc.account, hpc.remote_root, hpc.key_path):
        REDACTOR.add(v)

    if hpc.cpus != 1:
        log.warning(
            "hpc.cpus=%d. Each array task pins BLAS to one thread, so more than one CPU "
            "per task is wasted -- and if the pinning ever fails, concurrent tasks will "
            "oversubscribe the node and break the determinism E0.1 rests on.",
            hpc.cpus,
        )
    return cfg


def warn_if_world_readable(path: Path) -> None:
    """Warn, never crash. On Windows os.chmod only toggles the read-only bit."""
    try:
        if os.name == "nt":
            return
        mode = path.stat().st_mode
        if mode & 0o077:
            log.warning(
                "%s is group/world readable. Fix: chmod 600 %s", path, path
            )
    except OSError:
        pass


TEMPLATE = '''# precision_pc cluster configuration
#
# This file holds NO password and NO token. Authentication is your SSH key plus the
# cluster's second factor, typed at a terminal when a connection opens.
#
# Fill in the four REQUIRED fields under [hpc]. Everything else has a working default.

backend = "slurm"          # slurm | mock

[hpc]
host        = ""           # REQUIRED, e.g. "nibi.alliancecan.ca"
user        = ""           # REQUIRED
account     = ""           # REQUIRED. On Alliance this is the CPU association,
                           # e.g. "def-yourpi_cpu" -- check `hpc preflight` output.
remote_root = ""           # REQUIRED, e.g. "~/scratch/precision-pc"

key_path    = "~/.ssh/id_ed25519"
known_hosts = "~/.ssh/known_hosts"

# --- CPU array job: Tiers 1-3. This is where nearly all the work is. ---------
partition = ""             # "" -> let SLURM choose
cpus      = 1              # keep at 1; every task pins BLAS to a single thread
mem       = "4000M"
time      = "0-03:00"
array     = "0-15%8"       # 16 workers, at most 8 running at once
modules   = "python/3.12"

# --- GPU job: Tier 5 throughput only -----------------------------------------
# Alliance splits the allocation: a GPU job must be charged to the _gpu association,
# not the _cpu one. Leave empty on clusters that do not split.
gpu_account   = ""
gpu_partition = ""
gpus          = "h100:1"
gpu_flavour   = "gpus-per-node"   # gres | gpus-per-node | gpus-per-task
gpu_time      = "0-01:00"

# A claim older than this with no output shard is assumed dead and may be stolen.
# Must exceed the longest plausible single-cell runtime.
reclaim_after_s = 3600
'''


def write_template(path: Path, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise ConfigInvalid("%s already exists; pass --force to overwrite" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    try:
        if os.name != "nt":
            os.chmod(path, 0o600)
    except OSError:
        pass
    return path
