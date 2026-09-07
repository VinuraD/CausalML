"""Cascade mode: free-running skeleton discovery, recording what it actually did.

This is the half of the harness that `replay` mode cannot provide. Replay evaluates a
FIXED set of tests under every format, which gives an exactly paired per-test flip rate.
Cascade lets each format run its own search, so the path can diverge -- and the whole
point of contribution C1 is that a small per-test flip rate is amplified by that path
dependence into a much larger difference in the recovered graph.

Three things are recorded that a plain PC implementation throws away, each because an
experiment needs it:

`trajectory`
    Exactly which (i, j, S) tests this run executed, as a TestSet. This is what makes
    `union` replay mode possible: the union of every format's trajectory is a superset of
    all of them, and evaluating it under each format removes the trajectory bias instead
    of merely reporting it. It is also E3.2's measurement -- the per-level test counts.

`sepsets`
    Which separating set was recorded for each removed edge. Separating sets determine
    v-structures and hence orientation, so precision can change the CPDAG even when the
    skeleton is identical. E2.3 measures that channel, which nobody has measured.

`status` per test
    Including which tests took the pseudo-inverse branch. That is a CODE PATH flip, not a
    numeric perturbation, and E2.4 sweeps the rcond threshold to measure how often
    reduced precision changes which branch executes.

Two algorithm variants, because E2.2 asks whether order-independence dampens
amplification:

`stable=True`   PC-stable (Colombo & Maathuis 2014). The adjacency used to form
                conditioning sets is snapshotted at the START of each level and edge
                removals are applied at the END, so the result does not depend on the
                order in which edges are visited.
`stable=False`  Order-dependent PC. Removals take effect immediately, so an edge removed
                early changes the candidate sets of every edge considered after it.

`stop_at_first` mirrors a real GPU implementation. cuPC and Schmidt et al. evaluate ALL
separating-set candidates in parallel rather than short-circuiting at the first
acceptance, which changes the number of tests executed (and therefore the throughput
measurement) without changing the skeleton. It does not change which sepset is recorded,
because both rules keep the lowest-index accepting candidate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import batched as batched_mod
from . import checks, corr, formats, parcorr, skeleton_batch, stats, testset


@dataclass
class SkeletonResult:
    """Everything one cascade run did."""

    adj: np.ndarray  # bool[N, N], undirected skeleton
    sepsets: Dict[Tuple[int, int], Tuple[int, ...]]
    trajectory: testset.TestSet
    stat: np.ndarray  # float64[T], parallel to the trajectory
    status: np.ndarray  # uint8[T]
    kappa_proxy: np.ndarray  # float64[T]
    min_diag: np.ndarray  # float64[T]
    n_obs: int
    alpha: float
    max_level_reached: int
    level_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return int(np.triu(self.adj, 1).sum())

    def edge_set(self) -> set:
        iu = np.argwhere(np.triu(self.adj, 1))
        return {(int(a), int(b)) for a, b in iu}

    def level_shares(self) -> Dict[int, float]:
        """Fraction of executed tests at each level -- E3.2's quantity."""
        tot = sum(self.level_counts.values()) or 1
        return {k: v / tot for k, v in sorted(self.level_counts.items())}


def _candidate_sets(adj_ref: np.ndarray, a: int, b: int, level: int):
    """Conditioning-set candidates drawn from adj(a) \\ {b}, in a deterministic order.

    Sorted by variable index, matching the convention Schmidt et al. use to preserve
    candidate ordering by position in the correlation matrix. Determinism here is what
    makes the recorded sepset reproducible.
    """
    nb = [k for k in range(adj_ref.shape[0]) if k != b and k != a and adj_ref[a, k]]
    if len(nb) < level:
        return
    for S in itertools.combinations(sorted(nb), level):
        yield S


def discover(
    C: np.ndarray,
    n_obs: int,
    alpha: float,
    *,
    fmt: str = "float64",
    variant: str = "inv",
    ztransform: str = "log1p",
    max_level: int = 3,
    stable: bool = True,
    stop_at_first: bool = True,
    rcond: float = 1e-15,
    allow_pinv: bool = True,
    keep_unit_diagonal: bool = True,
    check: bool = True,
    batched: bool = True,
) -> SkeletonResult:
    """Run PC skeleton discovery at one precision, recording the whole trajectory."""
    C = np.ascontiguousarray(C, dtype=np.float64)
    if check:
        checks.correlation_matrix(C, "skeleton.discover(input C)")

    Cr = formats.round_corr(C, fmt, keep_unit_diagonal=keep_unit_diagonal)
    route = parcorr.get(variant)
    crit = stats.crit(alpha)

    N = C.shape[0]
    adj = np.ones((N, N), dtype=bool)
    np.fill_diagonal(adj, False)
    sepsets: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    tri, st_l, status_l, kap_l, md_l = [], [], [], [], []
    level_counts: Dict[int, int] = {}
    max_level_reached = -1

    def run_test(i: int, j: int, S: Tuple[int, ...]) -> float:
        """Execute one CI test, record it in the trajectory, return the statistic."""
        sub = corr.submatrix(Cr, i, j, np.asarray(S, dtype=np.int64))
        rho, status, kappa, mind = route(sub, rcond=rcond, allow_pinv=allow_pinv)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = stats.z_transform(np.float64(rho), ztransform)
            s = float(np.sqrt(n_obs - len(S) - 3.0) * np.abs(z))
        tri.append((i, j, S))
        st_l.append(s)
        status_l.append(status)
        kap_l.append(kappa)
        md_l.append(mind)
        level_counts[len(S)] = level_counts.get(len(S), 0) + 1
        return s

    for level in range(0, max_level + 1):
        if n_obs - level - 3 <= 0:
            break
        # PC-stable snapshots the adjacency; order-dependent PC reads the live one.
        adj_ref = adj.copy() if stable else adj

        # Wave-batched fast path. Only for PC-stable: with stable=False a removal takes
        # effect immediately, so an edge removed early changes the candidate sets of
        # edges considered after it, and evaluating a wave across edges would reorder
        # exactly that dependency. E2.2 compares the two variants, so both stay exact.
        if batched and stable and variant in batched_mod.BATCHABLE:
            recs, removals, any_cand = skeleton_batch.run_level(
                Cr, adj, adj_ref, level,
                n_obs=n_obs, crit=crit, variant=variant, ztransform=ztransform,
                rcond=rcond, allow_pinv=allow_pinv, stop_at_first=stop_at_first,
            )
            for (i, j, S, sv, stv, kpv, mdv) in recs:
                tri.append((i, j, S))
                st_l.append(sv)
                status_l.append(stv)
                kap_l.append(kpv)
                md_l.append(mdv)
                level_counts[len(S)] = level_counts.get(len(S), 0) + 1
            for (i, j), S in removals:
                sepsets[(i, j)] = tuple(S)
                adj[i, j] = adj[j, i] = False
            if any_cand:
                max_level_reached = level
                continue
            break
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i, j]]
        if not edges:
            break

        # Is any conditioning set of this size even available? If not, we are done.
        any_candidate = False
        pending_removals = []

        for i, j in edges:
            if not stable and not adj[i, j]:
                continue  # already removed earlier in this same level
            seen = set()
            found: Optional[Tuple[int, ...]] = None
            for a, b in ((i, j), (j, i)):
                for S in _candidate_sets(adj_ref, a, b, level):
                    if S in seen:
                        continue
                    seen.add(S)
                    any_candidate = True
                    s = run_test(i, j, S)
                    # Accept the null (independent) => this S separates i and j.
                    # NaN is NOT acceptance: a degenerate statistic means the test could
                    # not be evaluated, and treating it as independence would silently
                    # delete edges in exactly the low-precision formats under study.
                    if not np.isnan(s) and s <= crit:
                        if found is None:
                            found = S
                        if stop_at_first:
                            break
                if found is not None and stop_at_first:
                    break
            if found is not None:
                sepsets[(i, j)] = tuple(found)
                if stable:
                    pending_removals.append((i, j))
                else:
                    adj[i, j] = adj[j, i] = False

        if stable:
            for i, j in pending_removals:
                adj[i, j] = adj[j, i] = False

        if any_candidate:
            max_level_reached = level
        else:
            break

    ts = testset.from_triples(tri, N) if tri else testset.level0(N)
    return SkeletonResult(
        adj=adj,
        sepsets=sepsets,
        trajectory=ts,
        stat=np.asarray(st_l, dtype=np.float64),
        status=np.asarray(status_l, dtype=np.uint8),
        kappa_proxy=np.asarray(kap_l, dtype=np.float64),
        min_diag=np.asarray(md_l, dtype=np.float64),
        n_obs=int(n_obs),
        alpha=float(alpha),
        max_level_reached=max_level_reached,
        level_counts=level_counts,
    )
