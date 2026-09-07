"""Explicit Parquet schemas.

Without this module the harness works fine for a fortnight and then breaks silently: a
column gets added in one experiment and not another, `pyarrow.dataset` unifies the
schemas by filling nulls, and an aggregate quietly averages over rows that were never
computed. Declaring the schemas as objects, validating on every write, and versioning
them means that failure becomes an exception at write time instead of a wrong number at
analysis time.

Four tables, each answering a different question:

    testset  which tests exist            written once, format-invariant, content-addressed
    tests    what happened to each test   per cell, optionally subsampled
    cells    what happened to each cell   per (cell, level), EXACT counters
    flips    decisions at each alpha      per (cell, level, alpha), long format

The hard rule that makes subsampling safe: **every reported rate comes from the exact
counters in `cells` and `flips`, never from counting rows in `tests`.** The per-test table
is for diagnostics and regression, and may be a sample; the counters are always a census.
"""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- tests
# Per CI test, per cell. `test_idx` joins positionally to the testset artifact, which is
# why the descriptor columns (i, j, S) are NOT repeated here.
#
# Note what is stored: `stat`, not `p`. A p-value computed as 2*(1 - Phi(x)) is exactly
# 0.0 above x ~ 8.3, and a zero p-value can never flip, which biases every flip rate
# downward. Storing the statistic removes the underflow structurally. The threshold is a
# single constant per alpha, so nothing is lost.
TESTS = pa.schema(
    [
        pa.field("test_idx", pa.int32(), nullable=False),
        pa.field("rho", pa.float64(), nullable=False),
        pa.field("stat", pa.float64(), nullable=False),
        pa.field("status", pa.uint8(), nullable=False),
        pa.field("kappa_proxy", pa.float32(), nullable=False),
        pa.field("min_diag", pa.float32(), nullable=True),
        pa.field("kappa_exact", pa.float32(), nullable=True),  # subsample only (E1.2)
        pa.field("sampled", pa.bool_(), nullable=False),
    ]
)

# --------------------------------------------------------------------------- cells
# One row per (cell, level). Counters are uint64 and EXACT -- a census even when the
# per-test rows are a sample.
CELLS = pa.schema(
    [
        # identity
        pa.field("cell_digest", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),  # replay | cascade | union
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("testset_digest", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        # the factors
        pa.field("format", pa.string(), nullable=False),
        pa.field("parcorr_variant", pa.string(), nullable=False),
        pa.field("ztransform_variant", pa.string(), nullable=False),
        pa.field("cdf_impl", pa.string(), nullable=False),
        pa.field("round_stage", pa.string(), nullable=False),
        pa.field("keep_unit_diag", pa.bool_(), nullable=False),
        pa.field("allow_pinv", pa.bool_(), nullable=False),
        pa.field("rcond", pa.float64(), nullable=False),
        pa.field("linalg_impl", pa.string(), nullable=False),  # numpy | numba
        # the setting
        pa.field("n_obs", pa.int32(), nullable=False),
        pa.field("n_vars", pa.int32(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        # exact counters
        pa.field("n_tests", pa.uint64(), nullable=False),
        pa.field("n_ok", pa.uint64(), nullable=False),
        pa.field("n_rho_unit", pa.uint64(), nullable=False),
        pa.field("n_rho_gt1", pa.uint64(), nullable=False),
        pa.field("n_chol_fail", pa.uint64(), nullable=False),
        pa.field("n_pinv", pa.uint64(), nullable=False),
        pa.field("n_nonfinite", pa.uint64(), nullable=False),
        pa.field("n_singular", pa.uint64(), nullable=False),
        pa.field("n_undefined", pa.uint64(), nullable=False),
        # numerics vs the reference cell
        pa.field("max_abs_dstat", pa.float64(), nullable=True),
        pa.field("mean_abs_dstat", pa.float64(), nullable=True),
        pa.field("max_abs_drho", pa.float64(), nullable=True),
        # provenance, stamped so a Parquet file is self-describing in isolation
        pa.field("code_digest", pa.string(), nullable=False),
        pa.field("config_digest", pa.string(), nullable=True),
        pa.field("env_digest", pa.string(), nullable=False),
        pa.field("noise_floor_ok", pa.bool_(), nullable=False),
        pa.field("sampling_rate", pa.float64(), nullable=False),
        pa.field("sampling_seed", pa.int64(), nullable=False),
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("t_wall_s", pa.float64(), nullable=True),
    ]
)

# --------------------------------------------------------------------------- flips
# Long format. alpha is a VALUE, never part of a column name -- otherwise adding an alpha
# changes the schema and every previously written file becomes unreadable as one dataset.
FLIPS = pa.schema(
    [
        pa.field("cell_digest", pa.string(), nullable=False),
        pa.field("ref_cell_digest", pa.string(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("alpha", pa.float64(), nullable=False),
        pa.field("n_tests", pa.uint64(), nullable=False),
        pa.field("n_compared", pa.uint64(), nullable=False),
        pa.field("n_flip", pa.uint64(), nullable=False),
        pa.field("n_undefined", pa.uint64(), nullable=False),
        pa.field("n_dep_to_indep", pa.uint64(), nullable=False),
        pa.field("n_indep_to_dep", pa.uint64(), nullable=False),
        pa.field("n_reject_ref", pa.uint64(), nullable=False),
        pa.field("n_reject_var", pa.uint64(), nullable=False),
        pa.field("bitwise_identical", pa.bool_(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------- hist
# The structure that lets E1.2 scale without per-test rows: flip counts binned by
# log10(kappa). Summing `count` over a cell must reproduce `cells.n_tests` exactly.
HIST = pa.schema(
    [
        pa.field("cell_digest", pa.string(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("log10kappa_bin", pa.int8(), nullable=False),
        pa.field("count", pa.uint64(), nullable=False),
    ]
)

# -------------------------------------------------------------------------- graphs
# Cascade mode: one row per (format cell, alpha). This is where contribution C1 lives.
#
# The numerator and denominator of amplification come from DIFFERENT arms and are stored
# side by side so that is impossible to forget: `edges_changed` from the free-running
# cascade, `n_union_flip` from replaying the UNION of every format's trajectory (the only
# test set on which a flip rate is exactly paired and unbiased by any one trajectory).
GRAPHS = pa.schema(
    [
        pa.field("cell_digest", pa.string(), nullable=False),
        pa.field("ref_cell_digest", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("format", pa.string(), nullable=False),
        pa.field("parcorr_variant", pa.string(), nullable=False),
        pa.field("ztransform_variant", pa.string(), nullable=False),
        pa.field("stable", pa.bool_(), nullable=False),
        pa.field("stop_at_first", pa.bool_(), nullable=False),
        pa.field("alpha", pa.float64(), nullable=False),
        pa.field("max_level", pa.int8(), nullable=False),
        # skeleton, precision vs precision
        pa.field("n_edges_ref", pa.int32(), nullable=False),
        pa.field("n_edges_var", pa.int32(), nullable=False),
        pa.field("edges_changed", pa.int32(), nullable=False),
        pa.field("edges_lost", pa.int32(), nullable=False),
        pa.field("edges_gained", pa.int32(), nullable=False),
        # skeleton vs ground truth (E4.1) -- null when the DGP has no known truth
        pa.field("shd_truth_ref", pa.int32(), nullable=True),
        pa.field("shd_truth_var", pa.int32(), nullable=True),
        # separating sets (E2.3): the channel that changes the CPDAG with an identical skeleton
        pa.field("n_sepset_common", pa.int32(), nullable=False),
        pa.field("n_sepset_differ", pa.int32(), nullable=False),
        # trajectory divergence -- the empirical bound on replay-mode bias
        pa.field("traj_jaccard", pa.float64(), nullable=False),
        pa.field("n_traj_ref", pa.int32(), nullable=False),
        pa.field("n_traj_var", pa.int32(), nullable=False),
        # the paired denominator, from union replay
        pa.field("n_union_tests", pa.int32(), nullable=False),
        pa.field("n_union_compared", pa.int32(), nullable=False),
        pa.field("n_union_flip", pa.int32(), nullable=False),
        pa.field("n_union_undefined", pa.int32(), nullable=False),
        # code-path flips (E2.4), distinct from numeric perturbation
        pa.field("n_pinv_ref", pa.int32(), nullable=False),
        pa.field("n_pinv_var", pa.int32(), nullable=False),
        pa.field("max_level_ref", pa.int8(), nullable=False),
        pa.field("max_level_var", pa.int8(), nullable=False),
        pa.field("code_digest", pa.string(), nullable=False),
        pa.field("config_digest", pa.string(), nullable=True),
        pa.field("env_digest", pa.string(), nullable=False),
        pa.field("noise_floor_ok", pa.bool_(), nullable=False),
        pa.field("schema_version", pa.int16(), nullable=False),
    ]
)

# -------------------------------------------------------------------------- levels
# E3.2: how the executed work is distributed across conditioning-set sizes. Only cascade
# mode can answer this -- under the alpha = 1.0 device the per-level counts are fixed by
# construction and cannot shift.
LEVELS = pa.schema(
    [
        pa.field("cell_digest", pa.string(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("n_tests", pa.uint64(), nullable=False),
    ]
)

# --------------------------------------------------------------------------- cpdag
# E2.3 at the level a practitioner actually reads. A separate table rather than extra
# columns on `graphs`, so adding it does not invalidate runs already on disk.
#
# `identical_skeleton` is the column that isolates the channel: filtering to rows where
# it is true removes every skeleton difference, so whatever orientation disagreement
# remains is attributable to separating sets alone.
CPDAG = pa.schema(
    [
        pa.field("cell_digest", pa.string(), nullable=False),
        pa.field("ref_cell_digest", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("experiment", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("format", pa.string(), nullable=False),
        pa.field("alpha", pa.float64(), nullable=False),
        pa.field("identical_skeleton", pa.bool_(), nullable=False),
        # the decomposition: SHD = skeleton_differ + orientation_differ
        pa.field("cpdag_shd", pa.int32(), nullable=False),
        pa.field("skeleton_differ", pa.int32(), nullable=False),
        pa.field("orientation_differ", pa.int32(), nullable=False),
        pa.field("n_common_edges", pa.int32(), nullable=False),
        # v-structures, the mechanism connecting sepsets to orientation
        pa.field("n_vstruct_ref", pa.int32(), nullable=False),
        pa.field("n_vstruct_var", pa.int32(), nullable=False),
        pa.field("n_vstruct_differ", pa.int32(), nullable=False),
        pa.field("n_directed_ref", pa.int32(), nullable=False),
        pa.field("n_directed_var", pa.int32(), nullable=False),
        pa.field("n_undirected_ref", pa.int32(), nullable=False),
        pa.field("n_undirected_var", pa.int32(), nullable=False),
        # conflicting colliders: a graph with these is not a valid CPDAG at all
        pa.field("n_conflicts_ref", pa.int32(), nullable=False),
        pa.field("n_conflicts_var", pa.int32(), nullable=False),
        # against ground truth, when the DGP has one (E4.1 at the CPDAG level)
        pa.field("cpdag_shd_truth_ref", pa.int32(), nullable=True),
        pa.field("cpdag_shd_truth_var", pa.int32(), nullable=True),
        pa.field("code_digest", pa.string(), nullable=False),
        pa.field("schema_version", pa.int16(), nullable=False),
    ]
)

SCHEMAS = {
    "tests": TESTS,
    "cells": CELLS,
    "flips": FLIPS,
    "hist": HIST,
    "graphs": GRAPHS,
    "levels": LEVELS,
    "cpdag": CPDAG,
}


def validate(table: pa.Table, name: str) -> pa.Table:
    """Cast a table to its declared schema, failing loudly on any mismatch."""
    if name not in SCHEMAS:
        raise KeyError("unknown table %r; have %r" % (name, sorted(SCHEMAS)))
    want = SCHEMAS[name]
    have = set(table.schema.names)
    need = set(want.names)
    missing = need - have
    extra = have - need
    if missing:
        raise ValueError("table %r is missing column(s) %r" % (name, sorted(missing)))
    if extra:
        raise ValueError(
            "table %r has unexpected column(s) %r. Add them to schema.py and bump "
            "SCHEMA_VERSION rather than writing them ad hoc -- silent column drift is "
            "what makes a multi-shard dataset unreadable weeks later." % (name, sorted(extra))
        )
    return table.select(want.names).cast(want)
