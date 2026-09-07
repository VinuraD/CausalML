"""Wave-batched level processing for PC-stable skeleton discovery.

`skeleton.discover` evaluates one CI test at a time. Cascade mode drives it, so the
amplification and time-series experiments pay that loop in full -- `e2_1_amplification`
(210 cells) takes minutes where the replay configs now take seconds.

The difficulty is that the search short-circuits: an edge stops testing candidates as
soon as one separates it. Naively batching every candidate of every edge would evaluate
tests the loop never reaches, and the executed trajectory is a **recorded output** -- it
feeds union-replay and E3.2's level counts -- so inflating it would change results, not
just timing.

So this batches in **waves**. Wave k evaluates the k-th candidate of every edge still
live, in one batched call. Edges that accept drop out before wave k+1. Exactly the same
tests are executed as the loop, in the same quantity; only the *order* of evaluation
differs, and the recorded trajectory is re-sorted into the loop's edge-major order before
being returned, so even the digest matches.

**Only PC-stable is batched.** With `stable=False` a removal takes effect immediately, so
an edge removed early changes the candidate sets of edges considered after it -- and
evaluating a wave across edges would reorder exactly that dependency. E2.2 compares the
two variants, so both have to stay exact; the order-dependent path keeps the loop.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import batched as batched_mod
from . import parcorr, stats


def candidates_for(adj_ref: np.ndarray, i: int, j: int, level: int) -> List[Tuple[int, ...]]:
    """Every conditioning set the loop would try for edge (i, j), in the same order.

    The loop iterates `((i, j), (j, i))` and, for each, subsets of that endpoint's
    neighbourhood in sorted order, skipping duplicates. Reproduced exactly here, because
    which candidate is reached FIRST decides which separating set gets recorded -- and
    sepsets determine v-structures, hence orientation (E2.3).
    """
    out: List[Tuple[int, ...]] = []
    seen = set()
    n = adj_ref.shape[0]
    for a, b in ((i, j), (j, i)):
        nb = [k for k in range(n) if k != b and k != a and adj_ref[a, k]]
        if len(nb) < level:
            continue
        for S in itertools.combinations(sorted(nb), level):
            if S in seen:
                continue
            seen.add(S)
            out.append(S)
    return out


def _evaluate(Cr, edges, sets, level, n_obs, variant, ztransform, rcond, allow_pinv):
    """One wave: evaluate `sets[e]` for every edge in `edges`, batched where possible."""
    m = level + 2
    idx = np.empty((len(edges), m), dtype=np.int64)
    for r, ((i, j), S) in enumerate(zip(edges, sets)):
        idx[r, 0] = i
        idx[r, 1] = j
        if level:
            idx[r, 2:] = S
    sub = batched_mod.gather_submatrices(Cr, idx)

    got = batched_mod.evaluate_bucket(sub, variant, rcond=rcond)
    if got is None:
        # A degenerate element somewhere in the wave. NumPy's stacked cholesky/inv raise
        # for the whole batch without saying which, so fall back element by element --
        # exact, and only as slow as the degenerate case deserves.
        route = parcorr.get(variant)
        rho = np.empty(len(edges)); status = np.empty(len(edges), dtype=np.uint8)
        kap = np.empty(len(edges)); md = np.empty(len(edges))
        for r in range(len(edges)):
            rho[r], status[r], kap[r], md[r] = route(
                np.ascontiguousarray(sub[r]), rcond=rcond, allow_pinv=allow_pinv
            )
    else:
        rho, status, kap, md = got

    with np.errstate(divide="ignore", invalid="ignore"):
        z = stats.z_transform(rho, ztransform)
        stat = np.sqrt(n_obs - level - 3.0) * np.abs(z)
    return rho, stat, status, kap, md


def run_level(
    Cr: np.ndarray,
    adj: np.ndarray,
    adj_ref: np.ndarray,
    level: int,
    *,
    n_obs: int,
    crit: float,
    variant: str,
    ztransform: str,
    rcond: float,
    allow_pinv: bool,
    stop_at_first: bool,
):
    """Process one level of PC-stable, batching across edges.

    Returns (records, removals, any_candidate) where `records` is in the loop's
    edge-major order: a list of (i, j, S, stat, status, kappa, min_diag) per executed test.
    """
    n = adj.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    plans = {e: candidates_for(adj_ref, e[0], e[1], level) for e in edges}
    any_candidate = any(plans[e] for e in edges)
    if not any_candidate:
        return [], [], False

    # Per-edge records, so the loop's edge-major ordering can be restored at the end even
    # though evaluation proceeds wave by wave.
    per_edge: Dict[Tuple[int, int], list] = {e: [] for e in edges}
    found: Dict[Tuple[int, int], Optional[Tuple[int, ...]]] = {e: None for e in edges}
    live = [e for e in edges if plans[e]]
    wave = 0

    while live:
        batch = [e for e in live if wave < len(plans[e])]
        if not batch:
            break
        sets = [plans[e][wave] for e in batch]
        rho, stat, status, kap, md = _evaluate(
            Cr, batch, sets, level, n_obs, variant, ztransform, rcond, allow_pinv
        )
        still: List[Tuple[int, int]] = []
        for r, e in enumerate(batch):
            per_edge[e].append((e[0], e[1], sets[r], float(stat[r]), np.uint8(status[r]),
                                float(kap[r]), float(md[r])))
            # NaN is NOT acceptance: a degenerate statistic means the test could not be
            # evaluated, and treating it as independence would delete edges in exactly the
            # low-precision formats under study.
            accepted = (not np.isnan(stat[r])) and stat[r] <= crit
            if accepted and found[e] is None:
                found[e] = sets[r]
            if not (accepted and stop_at_first):
                still.append(e)
        live = still
        wave += 1

    records = []
    for e in edges:
        records.extend(per_edge[e])
    removals = [(e, found[e]) for e in edges if found[e] is not None]
    return records, removals, True
