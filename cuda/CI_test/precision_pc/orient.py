"""V-structures, Meek's rules, and CPDAG comparison.

This module exists to turn the separating-set divergence measured by E2.3 into a number a
reader can act on. The skeleton comparison already showed that precision barely moves the
undirected graph -- but separating sets determine v-structures, v-structures propagate
through Meek's rules, and the *orientations* are what a practitioner actually reads off a
causal graph. A run whose skeleton is bit-for-bit identical can still disagree about
almost every arrow.

Representation. `A[i, j] = True` means "the edge between i and j exists, and an arrowhead
at j is permitted":

    A[i, j] and A[j, i]          undirected   i --- j
    A[i, j] and not A[j, i]      directed     i --> j
    neither                      no edge

Orienting i --> j is therefore `A[j, i] = False` -- it removes a possibility rather than
adding one, which is exactly what the orientation rules do.

Two honesty notes that belong in the write-up:

1. **Conflicting v-structures are counted, not silently resolved.** When two unshielded
   triples demand opposite arrowheads on the same edge, the PC algorithm as usually
   implemented lets whichever is processed last win, and the result is no longer a valid
   CPDAG. Precision can change how many such conflicts occur, so `n_conflicts` is
   reported rather than hidden -- a graph with conflicts is a different kind of object
   from one without, and averaging SHD over both without saying so would be misleading.

2. **Meek R4 is off by default.** R1-R3 are sufficient to reach the CPDAG when the input
   is a skeleton plus sepsets from PC with no background knowledge; R4 only becomes
   necessary once background knowledge is present. It is implemented and switchable so
   the claim can be checked rather than asserted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np


@dataclass
class CPDAG:
    """A completed partially directed acyclic graph."""

    A: np.ndarray  # bool[N, N]
    vstructures: Set[Tuple[int, int, int]]  # (i, k, j) with i < j, colliding at k
    n_conflicts: int = 0

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    def has_edge(self, i: int, j: int) -> bool:
        return bool(self.A[i, j] or self.A[j, i])

    def is_directed(self, i: int, j: int) -> bool:
        """True iff i --> j."""
        return bool(self.A[i, j] and not self.A[j, i])

    def is_undirected(self, i: int, j: int) -> bool:
        return bool(self.A[i, j] and self.A[j, i])

    def skeleton(self) -> np.ndarray:
        return self.A | self.A.T

    def directed_edges(self) -> Set[Tuple[int, int]]:
        return {
            (i, j)
            for i in range(self.n)
            for j in range(self.n)
            if i != j and self.is_directed(i, j)
        }

    def undirected_edges(self) -> Set[Tuple[int, int]]:
        return {
            (i, j)
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if self.is_undirected(i, j)
        }

    def edge_set(self) -> Set[Tuple[int, int]]:
        s = self.skeleton()
        return {(int(a), int(b)) for a, b in np.argwhere(np.triu(s, 1))}

    @property
    def n_directed(self) -> int:
        return len(self.directed_edges())

    @property
    def n_undirected(self) -> int:
        return len(self.undirected_edges())

    def orientation_of(self, i: int, j: int) -> str:
        """'none', 'undirected', 'ij' (i->j) or 'ji' (j->i)."""
        if not self.has_edge(i, j):
            return "none"
        if self.is_undirected(i, j):
            return "undirected"
        return "ij" if self.is_directed(i, j) else "ji"


# ------------------------------------------------------------------- v-structures
def find_vstructures(adj: np.ndarray, sepsets: Dict[Tuple[int, int], Tuple[int, ...]]):
    """Unshielded colliders implied by the skeleton and its separating sets.

    For every unshielded triple i - k - j (i and j adjacent to k, not to each other),
    k is a collider iff k is NOT in the set that separated i and j.

    A pair with no recorded sepset is skipped rather than guessed at: it means the edge
    was never removed, so there is no separating set and the triple is not unshielded in
    the sense the rule requires.
    """
    n = adj.shape[0]
    out = set()
    for k in range(n):
        nb = [x for x in range(n) if adj[x, k]]
        for i, j in itertools.combinations(sorted(nb), 2):
            if adj[i, j]:
                continue  # shielded
            S = sepsets.get((min(i, j), max(i, j)))
            if S is None:
                continue
            if k not in S:
                out.add((min(i, j), k, max(i, j)))
    return out


# --------------------------------------------------------------------- Meek rules
def _apply_meek(A: np.ndarray, use_r4: bool = False, max_iter: int = 1000) -> np.ndarray:
    """Apply Meek's orientation rules to a fixpoint.

    R1  a -> b,  b - c,  a,c non-adjacent          =>  b -> c
    R2  a -> c -> b,  a - b                        =>  a -> b
    R3  a - b, a - c, a - d, c -> b, d -> b,
        c,d non-adjacent                           =>  a -> b
    R4  a - b, a - c, c -> d, d -> b,
        c,b non-adjacent                           =>  a -> b   (background knowledge only)
    """
    n = A.shape[0]

    def adjacent(x, y):
        return A[x, y] or A[y, x]

    def undirected(x, y):
        return A[x, y] and A[y, x]

    def directed(x, y):
        return A[x, y] and not A[y, x]

    for _ in range(max_iter):
        changed = False

        # R1
        for b in range(n):
            for c in range(n):
                if b == c or not undirected(b, c):
                    continue
                for a in range(n):
                    if a in (b, c):
                        continue
                    if directed(a, b) and not adjacent(a, c):
                        A[c, b] = False
                        changed = True
                        break

        # R2
        for a in range(n):
            for b in range(n):
                if a == b or not undirected(a, b):
                    continue
                for c in range(n):
                    if c in (a, b):
                        continue
                    if directed(a, c) and directed(c, b):
                        A[b, a] = False
                        changed = True
                        break

        # R3
        for a in range(n):
            for b in range(n):
                if a == b or not undirected(a, b):
                    continue
                cands = [
                    c
                    for c in range(n)
                    if c not in (a, b) and undirected(a, c) and directed(c, b)
                ]
                done = False
                for c, d in itertools.combinations(cands, 2):
                    if not adjacent(c, d):
                        A[b, a] = False
                        changed = True
                        done = True
                        break
                if done:
                    break

        # R4
        if use_r4:
            for a in range(n):
                for b in range(n):
                    if a == b or not undirected(a, b):
                        continue
                    done = False
                    for c in range(n):
                        if c in (a, b) or not undirected(a, c):
                            continue
                        for d in range(n):
                            if d in (a, b, c):
                                continue
                            if directed(c, d) and directed(d, b) and not adjacent(c, b):
                                A[b, a] = False
                                changed = True
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break

        if not changed:
            break
    return A


def to_cpdag(
    adj: np.ndarray,
    sepsets: Dict[Tuple[int, int], Tuple[int, ...]],
    *,
    apply_meek: bool = True,
    use_r4: bool = False,
) -> CPDAG:
    """Skeleton + separating sets -> CPDAG."""
    adj = np.asarray(adj, dtype=bool)
    A = adj.copy()
    np.fill_diagonal(A, False)

    vs = find_vstructures(adj, sepsets)

    # Orient every collider, counting the edges where two colliders disagree. Whichever is
    # applied last wins, exactly as a standard implementation behaves -- but the count is
    # reported so a graph built on conflicts is not mistaken for a clean one.
    # Count distinct CONFLICTED EDGES, not conflict events: one edge demanded in both
    # directions by several colliders is one broken edge, and counting the events instead
    # would inflate the number by however many triples happen to touch it.
    conflicted: Set[Tuple[int, int]] = set()
    oriented: Set[Tuple[int, int]] = set()
    for i, k, j in sorted(vs):
        for a in (i, j):
            if (k, a) in oriented:  # k was already required to be a tail on this edge
                conflicted.add((min(a, k), max(a, k)))
            A[k, a] = False
            oriented.add((a, k))
    conflicts = len(conflicted)

    if apply_meek:
        A = _apply_meek(A, use_r4=use_r4)
    return CPDAG(A=A, vstructures=vs, n_conflicts=conflicts)


def from_skeleton_result(result, **kw) -> CPDAG:
    """Convenience: build the CPDAG of a `skeleton.SkeletonResult`."""
    return to_cpdag(result.adj, result.sepsets, **kw)


def dag_to_cpdag(dag: np.ndarray, **kw) -> CPDAG:
    """The CPDAG of a known DAG -- the ground truth a recovered CPDAG is scored against.

    The separating sets are read off the DAG by d-separation of each non-adjacent pair,
    which is what makes the resulting v-structures the true ones.
    """
    dag = np.asarray(dag, dtype=bool)
    n = dag.shape[0]
    skel = dag | dag.T
    sep: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    for i, j in itertools.combinations(range(n), 2):
        if skel[i, j]:
            continue
        # Parents of i and j always d-separate a non-adjacent pair in a DAG.
        S = tuple(sorted(set(np.flatnonzero(dag[:, i])) | set(np.flatnonzero(dag[:, j]))))
        sep[(i, j)] = S
    return to_cpdag(skel, sep, **kw)


# ------------------------------------------------------------------- comparison
@dataclass(frozen=True)
class CpdagDiff:
    n_pairs_differ: int  # CPDAG SHD
    skeleton_differ: int  # pairs where edge presence differs
    orientation_differ: int  # pairs present in BOTH but oriented differently
    n_common_edges: int
    n_vstruct_ref: int
    n_vstruct_var: int
    n_vstruct_differ: int
    n_directed_ref: int
    n_directed_var: int

    @property
    def orientation_rate(self) -> Optional[float]:
        """Fraction of shared edges whose orientation disagrees.

        This is the number the separating-set divergence was pointing at: it is defined
        only on edges both graphs recovered, so it isolates the ORIENTATION channel from
        skeleton differences entirely.
        """
        if self.n_common_edges == 0:
            return None
        return self.orientation_differ / self.n_common_edges

    def as_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        d["orientation_rate"] = self.orientation_rate
        return d


def cpdag_diff(ref: CPDAG, var: CPDAG) -> CpdagDiff:
    """Structural Hamming distance between two CPDAGs, decomposed.

    Each unordered pair contributes at most 1: either the edge is present in one and not
    the other (a skeleton difference), or it is present in both with different
    orientation (an orientation difference). Reporting the two parts separately is what
    makes "the skeleton is robust but the orientation is not" a measurable claim rather
    than an impression.
    """
    n = ref.n
    skel_d = orient_d = common = 0
    for i, j in itertools.combinations(range(n), 2):
        a, b = ref.orientation_of(i, j), var.orientation_of(i, j)
        if (a == "none") != (b == "none"):
            skel_d += 1
        elif a != "none":
            common += 1
            if a != b:
                orient_d += 1
    return CpdagDiff(
        n_pairs_differ=skel_d + orient_d,
        skeleton_differ=skel_d,
        orientation_differ=orient_d,
        n_common_edges=common,
        n_vstruct_ref=len(ref.vstructures),
        n_vstruct_var=len(var.vstructures),
        n_vstruct_differ=len(ref.vstructures ^ var.vstructures),
        n_directed_ref=ref.n_directed,
        n_directed_var=var.n_directed,
    )
