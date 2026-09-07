"""Run directory layout, content-addressed shards, atomic writes.

The single property everything else depends on: **a shard's existence means that cell is
done.** That makes resume free, makes the HPC fan-out a work-stealing loop rather than a
static partition, and means the answer to "is this cell finished?" never depends on a
scheduler, a database or a lock.

Layout:

    runs/<run_id>/
      manifest.json                     provenance for the whole run
      testsets/<testset_digest>.parquet immutable, format-invariant descriptors
      cells/<cell_digest>.parquet       one row per (cell, level), exact counters
      flips/<cell_digest>.parquet       one row per (cell, level, alpha)
      hist/<cell_digest>.parquet        log10 kappa coverage
      tests/<cell_digest>.parquet       per-test rows (possibly a sample)

Writes go to a `.tmp` sibling and are moved into place with `os.replace`, which is atomic
within a volume on both POSIX and Windows. A killed process therefore leaves either a
complete shard or no shard, never a half-written one that `already_done` would count.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import atomic, provenance, schema

TABLES = ("cells", "flips", "hist", "tests", "graphs", "levels", "cpdag")


def cell_digest(key: dict) -> str:
    """Content address of a cell.

    The key MUST include the experiment, the mode, the testset digest and the code
    digest, not just (dataset, format, seed). Without the experiment and mode, two
    different experiments collide on one digest; without the code digest, an edit to
    `parcorr.py` silently reuses stale shards computed by the old routine.
    """
    required = {
        "experiment",
        "mode",
        "dataset_id",
        "testset_digest",
        "seed",
        "format",
        "parcorr_variant",
        "ztransform_variant",
        "code_digest",
    }
    missing = required - set(key)
    if missing:
        raise ValueError("cell key is missing %r" % sorted(missing))
    return hashlib.sha256(provenance.canonical_json(key).encode()).hexdigest()


class RunStore:
    """A single run directory."""

    def __init__(self, root, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.dir = self.root / run_id

    # ------------------------------------------------------------------- layout
    def table_dir(self, table: str) -> Path:
        if table not in TABLES:
            raise KeyError("unknown table %r; have %r" % (table, TABLES))
        return self.dir / table

    def shard_path(self, table: str, cell_digest: str) -> Path:
        return self.table_dir(table) / (cell_digest + ".parquet")

    def testset_path(self, testset_digest: str) -> Path:
        return self.dir / "testsets" / (testset_digest + ".parquet")

    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def init(self, config: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
        for t in TABLES:
            self.table_dir(t).mkdir(parents=True, exist_ok=True)
        (self.dir / "testsets").mkdir(parents=True, exist_ok=True)
        return provenance.write_manifest(self.manifest_path(), config, extra)

    def manifest(self) -> dict:
        p = self.manifest_path()
        if not p.exists():
            raise FileNotFoundError("no manifest at %s; call init() first" % p)
        return json.loads(p.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------- resume
    def is_done(self, cell_digest: str, table: str = "cells") -> bool:
        """A cell is done iff its marker shard exists. Nothing else is consulted.

        Cascade mode marks completion with a `graphs` shard, since it writes no `cells`."""
        return self.shard_path(table, cell_digest).exists()

    def done_cells(self, table: str = "cells") -> set:
        d = self.table_dir(table)
        if not d.exists():
            return set()
        return {p.stem for p in d.glob("*.parquet")}

    # --------------------------------------------------------------------- write
    def write_shard(self, table: str, cell_digest: str, rows) -> Path:
        """Validate against the declared schema, then write atomically."""
        tbl = rows if isinstance(rows, pa.Table) else pa.table(rows)
        tbl = schema.validate(tbl, table)
        path = self.shard_path(table, cell_digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # Already written, by an earlier run or by a concurrent worker. The path is
            # content-addressed -- the digest covers the code, the config and the cell --
            # so an existing shard holds equivalent bytes and rewriting it would only
            # create a race. A code change produces a different digest, so this can never
            # silently reuse a stale result.
            return path
        tmp = atomic.tmp_sibling(path)
        pq.write_table(tbl, tmp, compression="zstd")
        return atomic.finalize(tmp, path)

    def write_testset(self, ts) -> Path:
        """Immutable and content-addressed: if it already exists, it is already correct."""
        path = self.testset_path(ts.digest())
        if path.exists():
            return path
        return ts.to_parquet(path)

    def read_testset(self, testset_digest: str):
        from .testset import TestSet

        return TestSet.from_parquet(self.testset_path(testset_digest))

    # ---------------------------------------------------------------------- read
    def read(self, table: str):
        """Read every shard of a table as one pandas DataFrame.

        Reads with the DECLARED schema, so a shard written by an older code version with
        a different column set fails loudly here instead of being unified with nulls.
        """
        import pyarrow.dataset as ds

        d = self.table_dir(table)
        files = sorted(d.glob("*.parquet"))
        if not files:
            return schema.SCHEMAS[table].empty_table().to_pandas()
        dataset = ds.dataset([str(f) for f in files], schema=schema.SCHEMAS[table])
        return dataset.to_table().to_pandas()


# ------------------------------------------------------------------- subsampling
def sample_mask(
    result,
    *,
    alphas: Iterable[float],
    rate: float,
    seed: int,
    flipped: Optional[np.ndarray] = None,
    band_width: float = 10.0,
) -> np.ndarray:
    """Which per-test rows to keep.

    Retained unconditionally, because these are the rows every downstream question is
    about and they are tiny by construction:

      1. every test that flipped relative to the reference -- the numerator, never sampled;
      2. every test whose status is not OK -- the degenerate cases;
      3. every test in the near-threshold band, the only tests that CAN flip;

    plus a deterministic systematic sample of the remainder at `rate`. Systematic (every
    k-th) rather than Bernoulli so the kept count is exact and reproducible from the seed
    without storing the mask.
    """
    from . import parcorr, stats

    s = np.asarray(result.stat, dtype=np.float64)
    keep = np.zeros(s.shape, dtype=bool)

    keep |= np.asarray(result.status, dtype=np.uint8) != parcorr.OK
    keep |= np.isnan(s)
    if flipped is not None:
        keep |= np.asarray(flipped, dtype=bool)

    alphas = list(alphas)
    if alphas:
        lo = stats.crit(min(max(alphas) * band_width, 0.999999))
        hi = stats.crit(min(alphas) / band_width)
        with np.errstate(invalid="ignore"):
            keep |= (s >= lo) & (s <= hi)

    if rate >= 1.0:
        return np.ones_like(keep)
    rest = np.flatnonzero(~keep)
    if rest.size and rate > 0.0:
        step = max(int(round(1.0 / rate)), 1)
        offset = int(np.random.default_rng(seed).integers(0, step)) if step > 1 else 0
        keep[rest[offset::step]] = True
    return keep
