"""Cluster fan-out.

The contract, written down before any code, because everything else follows from it:

    A cell is independently computable from (resolved config, cell digest), its output
    shard is immutable, and the shard's EXISTENCE is the only evidence it is done.

Given that, fan-out needs no clever abstraction: workers pull cells from a shared queue,
write shards, and stop. There is no partitioning to get right, no completion protocol, and
no state that can disagree with the filesystem.

The source-of-truth hierarchy is deliberate and ordered:

    output shards  >  sacct / squeue  >  local jobs.json cache

The queue forgets -- `squeue` shows only pending and running jobs, `sacct` is subject to
retention limits -- and the local cache is only ever a hint. "Is this cell done?" is
answered by looking for its shard, always.
"""

from __future__ import annotations

import abc
import importlib
from typing import Dict, Optional


class Backend(abc.ABC):
    """What a cluster must be able to do. Deliberately small."""

    name: str = "?"

    @abc.abstractmethod
    def preflight(self) -> dict:
        """Prove we can reach the cluster and report its environment. One connection."""

    @abc.abstractmethod
    def stage(self, *, force: bool = False, with_torch: bool = False) -> dict:
        """Prepare the remote environment. Login node only; needs internet."""

    @abc.abstractmethod
    def submit(self, config_path, run_id: str) -> dict:
        """Push code + config and submit the array. Idempotent: reattaches if already queued."""

    @abc.abstractmethod
    def status(self, run_id: str) -> dict:
        """Machine-readable progress: {'state', 'done', 'total', 'jobid', 'reason', ...}.

        Must NOT block. A queued array can sit for hours; a client that blocks on that
        cannot be closed and reopened.
        """

    @abc.abstractmethod
    def fetch(self, run_id: str, dest) -> dict:
        """Bring result shards back. Partial results are allowed but must be reported."""

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Register a backend.

    The isinstance check is not ceremony. A decorator binds to whatever definition follows
    it, so inserting a helper function between `@register("slurm")` and the class silently
    registers the HELPER, and the failure surfaces much later as the backend constructor
    being handed a Config it does not understand. Refuse at import time instead.
    """

    def deco(cls):
        if not (isinstance(cls, type) and issubclass(cls, Backend)):
            raise TypeError(
                "@register(%r) must decorate a Backend subclass, got %r" % (name, cls)
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _load_builtin(name: str) -> None:
    try:
        importlib.import_module("." + name, __package__)
    except ImportError:
        pass


def make_backend(name: str, cfg) -> Backend:
    if name not in _REGISTRY:
        _load_builtin(name)
    if name not in _REGISTRY:
        raise KeyError("unknown backend %r; have %r" % (name, sorted(_REGISTRY)))
    return _REGISTRY[name](cfg)


def available() -> list:
    for n in ("slurm", "mock"):
        _load_builtin(n)
    return sorted(_REGISTRY)
