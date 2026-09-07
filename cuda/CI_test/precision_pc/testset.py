"""The test-descriptor batch: a content-addressed, immutable list of CI tests.

Structure of arrays, with uniform shapes per level:

    i: int32[T]   j: int32[T]   S: int32[T, Lmax] (padded -1)   level: int8[T]

Two reasons it is shaped this way. First, it is what makes the eventual CUDA port
mechanical -- all tests of a given |S| have identical shapes and can be dispatched as one
batch. Second, and more immediately important, it makes the PAIRING exact.

Pairing is where this harness is most likely to produce a spectacular false positive.
Replay joins per-format results to a reference by position. If two test sets are ever
mixed -- a re-run at a slightly different alpha, a stale shard from before a code change,
a set-iteration-order difference -- then test #7134 in one run is compared against a
DIFFERENT test #7134 in another, and flips are manufactured out of nothing, leaving no
trace in the output. The defence is that a TestSet is content-addressed: its digest
covers the descriptors themselves, every result records the digest it was computed
against, and `metrics` refuses to compare two cells whose digests differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import provenance


@dataclass(frozen=True)
class TestSet:
    """An ordered, immutable list of conditional independence tests."""

    i: np.ndarray  # int32[T]
    j: np.ndarray  # int32[T]
    S: np.ndarray  # int32[T, Lmax], -1 padded
    level: np.ndarray  # int8[T], == number of non-negative entries in each row of S
    n_vars: int

    def __post_init__(self):
        T = self.i.shape[0]
        if not (self.j.shape[0] == T and self.S.shape[0] == T and self.level.shape[0] == T):
            raise ValueError("TestSet arrays disagree on length")
        if self.i.dtype != np.int32 or self.j.dtype != np.int32:
            raise ValueError("i and j must be int32")
        if self.S.dtype != np.int32:
            raise ValueError("S must be int32")
        if self.level.dtype != np.int8:
            raise ValueError("level must be int8")

    def __len__(self) -> int:
        return int(self.i.shape[0])

    @property
    def lmax(self) -> int:
        return int(self.S.shape[1])

    def conditioning_set(self, t: int) -> np.ndarray:
        row = self.S[t]
        return row[row >= 0]

    # ------------------------------------------------------------------ identity
    def digest(self) -> str:
        """Content address. Covers the descriptors and n_vars, nothing else.

        Deliberately NOT a function of alpha, format or code version: the same set of
        (i, j, S) triples evaluated under two formats must share a digest, or replay
        could never join them.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(b"testset-v1")
        h.update(np.int64(self.n_vars).tobytes())
        h.update(np.int64(len(self)).tobytes())
        h.update(np.int64(self.lmax).tobytes())
        for a in (self.i, self.j, self.S, self.level):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    # -------------------------------------------------------------------- set ops
    def keys(self) -> np.ndarray:
        """One hashable key per test, for set operations. Order-insensitive within S."""
        T = len(self)
        out = np.empty(T, dtype=object)
        for t in range(T):
            a, b = int(self.i[t]), int(self.j[t])
            lo, hi = (a, b) if a < b else (b, a)
            out[t] = (lo, hi, tuple(sorted(int(x) for x in self.conditioning_set(t))))
        return out

    def union(self, other: "TestSet") -> "TestSet":
        """Union of two test sets, in a deterministic order.

        This is what makes `union` replay mode possible. A CI test is a pure function of
        (C, i, j, S, n, format, variant) -- the execution trajectory decides only which
        tests get ASKED, never what they return -- so evaluating a test that FP8 would
        reach and FP64 never reached is perfectly well defined under FP64. Taking the
        union over every format's trajectory removes the trajectory bias entirely,
        instead of merely reporting it.
        """
        if self.n_vars != other.n_vars:
            raise ValueError("cannot union test sets over different variable counts")
        seen = {}
        for ts in (self, other):
            ks = ts.keys()
            for t in range(len(ts)):
                seen.setdefault(ks[t], (int(ts.i[t]), int(ts.j[t]), ts.conditioning_set(t)))
        items = [seen[k] for k in sorted(seen, key=lambda k: (len(k[2]), k[0], k[1], k[2]))]
        return from_triples(items, self.n_vars)

    def jaccard(self, other: "TestSet") -> float:
        a = set(self.keys().tolist())
        b = set(other.keys().tolist())
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    def index_of(self, other: "TestSet") -> np.ndarray:
        """Position of each of `other`'s tests within self; -1 where absent."""
        pos = {k: t for t, k in enumerate(self.keys().tolist())}
        return np.array([pos.get(k, -1) for k in other.keys().tolist()], dtype=np.int64)

    # ---------------------------------------------------------------- persistence
    def to_parquet(self, path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = {
            "test_idx": pa.array(np.arange(len(self), dtype=np.int32)),
            "i": pa.array(self.i),
            "j": pa.array(self.j),
            "level": pa.array(self.level),
        }
        for c in range(self.lmax):
            cols["s%d" % c] = pa.array(np.ascontiguousarray(self.S[:, c]))
        table = pa.table(cols)
        table = table.replace_schema_metadata(
            {
                b"testset_digest": self.digest().encode(),
                b"n_vars": str(self.n_vars).encode(),
                b"lmax": str(self.lmax).encode(),
            }
        )
        # Unique temp name: several array tasks write the same test-set artifact
        # concurrently. See precision_pc/atomic.py.
        from . import atomic

        tmp = atomic.tmp_sibling(path)
        pq.write_table(table, tmp, compression="zstd")
        return atomic.finalize(tmp, path)

    @staticmethod
    def from_parquet(path) -> "TestSet":
        import pyarrow.parquet as pq

        t = pq.read_table(path)
        meta = t.schema.metadata or {}
        n_vars = int(meta[b"n_vars"])
        lmax = int(meta[b"lmax"])
        i = t["i"].to_numpy(zero_copy_only=False).astype(np.int32)
        j = t["j"].to_numpy(zero_copy_only=False).astype(np.int32)
        level = t["level"].to_numpy(zero_copy_only=False).astype(np.int8)
        S = np.empty((len(i), lmax), dtype=np.int32)
        for c in range(lmax):
            S[:, c] = t["s%d" % c].to_numpy(zero_copy_only=False).astype(np.int32)
        ts = TestSet(i=i, j=j, S=S, level=level, n_vars=n_vars)
        want = meta.get(b"testset_digest")
        if want is not None and ts.digest() != want.decode():
            raise ValueError(
                "testset digest mismatch on read: file claims "
                + want.decode()[:12]
                + " but content hashes to "
                + ts.digest()[:12]
            )
        return ts


# ------------------------------------------------------------------ constructors
def from_triples(triples, n_vars: int) -> TestSet:
    """Build from an iterable of (i, j, S)."""
    triples = list(triples)
    T = len(triples)
    lmax = max((len(s) for _, _, s in triples), default=0)
    lmax = max(lmax, 1)  # keep at least one column so the Parquet schema is stable
    i = np.empty(T, dtype=np.int32)
    j = np.empty(T, dtype=np.int32)
    S = np.full((T, lmax), -1, dtype=np.int32)
    level = np.empty(T, dtype=np.int8)
    for t, (a, b, s) in enumerate(triples):
        i[t] = a
        j[t] = b
        s = np.asarray(sorted(int(x) for x in s), dtype=np.int32)
        S[t, : len(s)] = s
        level[t] = len(s)
    return TestSet(i=i, j=j, S=S, level=level, n_vars=n_vars)


def level0(n_vars: int) -> TestSet:
    """Every unordered pair, no conditioning. The marginal-independence pass."""
    iu = np.triu_indices(n_vars, 1)
    T = iu[0].size
    return TestSet(
        i=iu[0].astype(np.int32),
        j=iu[1].astype(np.int32),
        S=np.full((T, 1), -1, dtype=np.int32),
        level=np.zeros(T, dtype=np.int8),
        n_vars=n_vars,
    )


def from_adjacency(adj: np.ndarray, level_size: int) -> TestSet:
    """Every (i, j, S) a PC-stable pass at this level would test, given an adjacency.

    S ranges over subsets of size `level_size` of adj(i) \\ {j}. This is the level-`k`
    candidate set of PC-stable, which fixes the adjacency at the START of the level --
    that is exactly what makes PC-stable order-independent.
    """
    import itertools

    n = adj.shape[0]
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a, b]:
                continue
            nb_a = [k for k in range(n) if k != b and adj[a, k]]
            nb_b = [k for k in range(n) if k != a and adj[b, k]]
            for src in (nb_a, nb_b):
                if len(src) >= level_size:
                    for s in itertools.combinations(sorted(src), level_size):
                        out.append((a, b, s))
    # Deduplicate: the same (a, b, S) can arise from both neighbourhoods.
    seen = {}
    for a, b, s in out:
        seen.setdefault((a, b, tuple(s)), (a, b, s))
    items = [seen[k] for k in sorted(seen)]
    return from_triples(items, n)
