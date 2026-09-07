"""Error taxonomy for the cluster layer.

Split deliberately into three tiers, because they call for different responses:

  configuration   the user must fix something locally before anything can run
  job lifecycle   the cluster said something; usually wait, sometimes resubmit
  correctness     a result would be WRONG. Never recovered from, never defaulted.

The third tier is the one that matters. Every class in it corresponds to a way the
numbers could come back silently incorrect, so they are raised and never warned.
"""

from __future__ import annotations


class HpcError(Exception):
    """Base. __str__ is scrubbed of host/user/account identifiers."""

    def __str__(self) -> str:
        from .config import REDACTOR

        return REDACTOR.scrub(super().__str__())


# ------------------------------------------------------------------ configuration
class ConfigMissing(HpcError):
    """No cluster config found. Fix: `python -m precision_pc hpc init`."""


class ConfigInvalid(HpcError):
    """The config exists but is unusable -- unknown key, missing field, or a secret."""


class AuthRequired(HpcError):
    """Authentication needs a human at a terminal (a second factor typed in).

    Raised rather than attempted, because a headless retry loop against an MFA prompt
    burns the user's tokens and never succeeds.
    """


# ----------------------------------------------------------------- job lifecycle
class JobPending(HpcError):
    """Work is queued or running. This is CONTROL FLOW, not a failure.

    The submit/poll/fetch loop is expressed by raising this: a call either returns
    results or says "come back later". A queued job can sit for hours, so nothing in
    this client blocks waiting for one.
    """


class JobFailed(HpcError):
    """The job reached a terminal state without producing complete results."""


class StagingIncomplete(HpcError):
    """The remote environment has not been prepared (no venv, no wheelhouse, no code)."""


# -------------------------------------------------------------------- correctness
class ProvenanceMismatch(HpcError):
    """Remote results were produced by different code or a different config.

    Shards are keyed by a digest that covers the code, so this means the remote tree was
    not the tree that was submitted. Merging those results into one table would produce a
    dataset whose rows came from two different programs, which no checksum would catch.
    """


class GateNotEstablished(HpcError):
    """No passing E0.1 noise-floor record for the remote environment.

    The gate is keyed on (code_digest, env_digest, hostname), and a compute node is a
    different environment from the laptop -- different CPU, possibly different BLAS
    kernels. A noise floor measured locally says nothing about the cluster, so results
    computed there are not interpretable until the gate has been run THERE.
    """


class ResultsIncomplete(HpcError):
    """Some cells are missing and the caller asked for a complete set.

    Never silently return a partial table: a row that was quietly absent is
    indistinguishable from one that was computed, and every aggregate downstream would
    be wrong by an unknown amount.
    """
