"""Config -> cells -> shards.

A run is a full factorial over the factors named in the config. Each combination is a
CELL, content-addressed by its key, computed independently, and written as its own shard.
Nothing in this module knows about SLURM, and that is the point: because cells are
independently computable and a shard's existence means "done", fanning this out across a
cluster later is a work-stealing loop over `enumerate_cells`, not a new architecture.

The reference cell of every run is float64 with the config's reference variants. Every
other cell is compared against it, which is why `run_config` computes it first and keeps
it in memory for the duration.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa

from . import corr, engine, gate, metrics, parcorr, provenance, schema, stats, store, testset


# --------------------------------------------------------------------------- config
DEFAULTS = {
    "experiment": "e1_1_flip_rate",
    "mode": "replay",
    "n_vars": 40,
    "n_obs": 400,
    "levels": [0],
    "seeds": [0],
    "formats": ["float64", "float32", "tf32", "float16", "bfloat16", "float8_e4m3"],
    "parcorr_variants": ["inv"],
    "ztransform_variants": ["log1p"],
    "alphas": [0.001, 0.01, 0.05],
    "dataset": {"kind": "gaussian_iid"},
    "reference": {
        "format": "float64",
        "parcorr_variant": "inv",
        "ztransform_variant": "log1p",
    },
    "rcond": 1e-15,
    "allow_pinv": True,
    "keep_unit_diag": True,
    "round_stage": "corr",
    "cdf_impl": "erfc",
    "linalg_impl": "numpy",
    "per_test_storage": "subsample",  # full | subsample | none
    "sampling_rate": 0.02,
    "sampling_seed": 12345,
    "undefined_policy": "exclude",
    "require_gate": True,
    # cascade mode only
    "alpha_run": 0.01,
    "max_level": 3,
    "stable_variants": [True],
    "stop_at_first": True,
    # Cluster fan-out. Lives here rather than only in HpcConfig because
    # `load_config` rejects unknown keys, so a value that never appears in DEFAULTS
    # can never reach the remote worker -- it would silently keep the default.
    "reclaim_after_s": 3600,
}


def load_config(path) -> dict:
    """Load a flat YAML config, merged onto DEFAULTS.

    Flat and fully resolved on purpose: no includes, no inheritance, no templating. The
    moment a config can reference another one, the digest needs a canonicalisation rule,
    and that is a bug farm. `--set key=value` produces a resolved config; the RESOLVED
    form is what gets digested and stored.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            "config has unknown key(s) %r. A typo'd key would otherwise silently take "
            "its default, and the defaults here are the determinism-critical ones."
            % sorted(unknown)
        )
    return cfg


# -------------------------------------------------------------------------- datasets
def build_dataset(cfg: dict, seed: int):
    """Return (C, n_obs, dataset_id). The correlation matrix is the unit of input."""
    spec = cfg["dataset"]
    kind = spec.get("kind", "gaussian_iid")
    n_vars, n_obs = int(cfg["n_vars"]), int(cfg["n_obs"])
    rng = np.random.default_rng(seed)

    if kind == "gaussian_iid":
        X = rng.standard_normal((n_obs, n_vars))
        return corr.correlation(X), n_obs, "gaussian_iid:%d:%d" % (n_vars, n_obs)

    if kind == "kappa":
        kappa = float(spec.get("kappa", 1e6))
        C, realised = corr.with_condition_number(n_vars, kappa, rng)
        return C, n_obs, "kappa:%g:%d" % (kappa, n_vars)

    if kind == "var":
        # Tier 3. n_vars is DERIVED here (n_series * tau_max), not configured, and the
        # effective sample size is n_obs - tau_max + 1 because embedding costs
        # observations. Both are returned so the Fisher-z scaling uses the right n.
        from .datasets import var as _var

        C, n_eff, dsid = _var.design(
            n_obs=n_obs,
            phi=float(spec.get("phi", 0.9)),
            tau_max=int(spec.get("tau_max", 3)),
            n_series=int(spec.get("n_series", 1)),
            structure=spec.get("structure", "ar1"),
            lags=int(spec.get("lags", 1)),
            loading=float(spec.get("loading", 0.95)),
            sampling_ratio=int(spec.get("sampling_ratio", 1)),
            seed=seed,
        )
        return C, n_eff, dsid + ":seed=%d" % seed

    raise ValueError("unknown dataset kind %r" % kind)


def build_testset(cfg: dict, C: np.ndarray) -> testset.TestSet:
    levels = list(cfg["levels"])
    if levels == [0]:
        return testset.level0(C.shape[0])
    # Levels >= 1 need an adjacency to draw conditioning sets from. At this milestone the
    # complete graph is used, which is the alpha = 1.0 device: it fixes the test set so
    # the comparison is exactly paired. Its known bias -- conditioning sets drawn from the
    # complete graph are better conditioned than those a real run at alpha = 0.01 would
    # encounter -- is why `mode` is recorded on every row.
    n = C.shape[0]
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    parts = [testset.level0(n)] if 0 in levels else []
    for lv in sorted(x for x in levels if x > 0):
        parts.append(testset.from_adjacency(adj, lv))
    out = parts[0]
    for p in parts[1:]:
        out = out.union(p)
    return out


# ----------------------------------------------------------------------------- cells
@dataclass(frozen=True)
class Cell:
    key: dict
    digest: str
    seed: int
    fmt: str
    parcorr_variant: str
    ztransform_variant: str
    is_reference: bool


def enumerate_cells(cfg: dict, code_digest: Optional[str] = None) -> Iterator[Cell]:
    """Every (seed, format, parcorr variant, z-transform) combination in the config."""
    code_digest = code_digest or provenance.code_digest()
    ref = cfg["reference"]
    for seed in cfg["seeds"]:
        C, n_obs, dataset_id = build_dataset(cfg, int(seed))
        ts = build_testset(cfg, C)
        td = ts.digest()
        for fmt, pv, zv in itertools.product(
            cfg["formats"], cfg["parcorr_variants"], cfg["ztransform_variants"]
        ):
            key = {
                "experiment": cfg["experiment"],
                "mode": cfg["mode"],
                "dataset_id": dataset_id,
                "testset_digest": td,
                "seed": int(seed),
                "format": fmt,
                "parcorr_variant": pv,
                "ztransform_variant": zv,
                "round_stage": cfg["round_stage"],
                "rcond": cfg["rcond"],
                "allow_pinv": cfg["allow_pinv"],
                "keep_unit_diag": cfg["keep_unit_diag"],
                "linalg_impl": cfg["linalg_impl"],
                "n_obs": int(n_obs),
                "code_digest": code_digest,
            }
            yield Cell(
                key=key,
                digest=store.cell_digest(key),
                seed=int(seed),
                fmt=fmt,
                parcorr_variant=pv,
                ztransform_variant=zv,
                is_reference=(
                    fmt == ref["format"]
                    and pv == ref["parcorr_variant"]
                    and zv == ref["ztransform_variant"]
                ),
            )


# ------------------------------------------------------------------------ execution
def _status_count(res, code) -> int:
    return int((res.status == code).sum())


def _cell_rows(cfg, cell, res, ref_res, run_id, code_digest, config_digest, env_digest,
               noise_floor_ok, ts, wall_s):
    """The `cells` table rows for one cell: one per level present in the test set."""
    lv = np.asarray(ts.level, dtype=np.int8)
    rows = []
    for level in sorted(set(int(x) for x in lv)):
        m = lv == level
        d_stat = d_rho = None
        if ref_res is not None:
            with np.errstate(invalid="ignore"):
                ds = np.abs(res.stat[m] - ref_res.stat[m])
                dr = np.abs(res.rho[m] - ref_res.rho[m])
            d_stat = ds[np.isfinite(ds)]
            d_rho = dr[np.isfinite(dr)]
        rows.append(
            {
                "cell_digest": cell.digest,
                "run_id": run_id,
                "experiment": cfg["experiment"],
                "mode": cfg["mode"],
                "dataset_id": cell.key["dataset_id"],
                "testset_digest": res.testset_digest,
                "seed": np.int32(cell.seed),
                "format": cell.fmt,
                "parcorr_variant": cell.parcorr_variant,
                "ztransform_variant": cell.ztransform_variant,
                "cdf_impl": cfg["cdf_impl"],
                "round_stage": cfg["round_stage"],
                "keep_unit_diag": bool(cfg["keep_unit_diag"]),
                "allow_pinv": bool(cfg["allow_pinv"]),
                "rcond": float(cfg["rcond"]),
                "linalg_impl": cfg["linalg_impl"],
                "n_obs": np.int32(res.n_obs),
                "n_vars": np.int32(ts.n_vars),
                "level": np.int8(level),
                "n_tests": np.uint64(int(m.sum())),
                "n_ok": np.uint64(_status_count_masked(res, parcorr.OK, m)),
                "n_rho_unit": np.uint64(_status_count_masked(res, parcorr.RHO_UNIT, m)),
                "n_rho_gt1": np.uint64(_status_count_masked(res, parcorr.RHO_GT1, m)),
                "n_chol_fail": np.uint64(_status_count_masked(res, parcorr.CHOL_FAIL, m)),
                "n_pinv": np.uint64(_status_count_masked(res, parcorr.PINV_FALLBACK, m)),
                "n_nonfinite": np.uint64(_status_count_masked(res, parcorr.NONFINITE, m)),
                "n_singular": np.uint64(_status_count_masked(res, parcorr.SINGULAR, m)),
                "n_undefined": np.uint64(int(np.isnan(res.stat[m]).sum())),
                "max_abs_dstat": None if d_stat is None or not d_stat.size else float(d_stat.max()),
                "mean_abs_dstat": None if d_stat is None or not d_stat.size else float(d_stat.mean()),
                "max_abs_drho": None if d_rho is None or not d_rho.size else float(d_rho.max()),
                "code_digest": code_digest,
                "config_digest": config_digest,
                "env_digest": env_digest,
                "noise_floor_ok": bool(noise_floor_ok),
                "sampling_rate": float(cfg["sampling_rate"]),
                "sampling_seed": np.int64(cfg["sampling_seed"]),
                "schema_version": np.int16(schema.SCHEMA_VERSION),
                "t_wall_s": float(wall_s),
            }
        )
    return rows


def _status_count_masked(res, code, m) -> int:
    return int((res.status[m] == code).sum())


def _flip_rows(cfg, cell, res, ref_res, ref_digest, ts):
    rows = []
    lv = np.asarray(ts.level, dtype=np.int8)
    for level in sorted(set(int(x) for x in lv)):
        m = lv == level
        sub_ref = _slice(ref_res, m)
        sub_var = _slice(res, m)
        for alpha in cfg["alphas"]:
            fc = metrics.flip_counts(
                sub_ref, sub_var, float(alpha), undefined_policy=cfg["undefined_policy"]
            )
            rows.append(
                {
                    "cell_digest": cell.digest,
                    "ref_cell_digest": ref_digest,
                    "level": np.int8(level),
                    "alpha": float(alpha),
                    "n_tests": np.uint64(fc.n_tests),
                    "n_compared": np.uint64(fc.n_compared),
                    "n_flip": np.uint64(fc.n_flip),
                    "n_undefined": np.uint64(fc.n_undefined),
                    "n_dep_to_indep": np.uint64(fc.n_dep_to_indep),
                    "n_indep_to_dep": np.uint64(fc.n_indep_to_dep),
                    "n_reject_ref": np.uint64(fc.n_reject_ref),
                    "n_reject_var": np.uint64(fc.n_reject_var),
                    "bitwise_identical": bool(metrics.is_bitwise_identical(sub_ref, sub_var)),
                }
            )
    return rows


def _slice(res, m):
    return engine.CellResult(
        testset_digest=res.testset_digest,
        rho=res.rho[m],
        stat=res.stat[m],
        status=res.status[m],
        kappa_proxy=res.kappa_proxy[m],
        min_diag=res.min_diag[m],
        n_obs=res.n_obs,
    )


def _hist_rows(cell, res, ts):
    rows = []
    lv = np.asarray(ts.level, dtype=np.int8)
    for level in sorted(set(int(x) for x in lv)):
        m = lv == level
        counts, edges = metrics.kappa_histogram(_slice(res, m))
        for b, c in enumerate(counts):
            if c:
                rows.append(
                    {
                        "cell_digest": cell.digest,
                        "level": np.int8(level),
                        "log10kappa_bin": np.int8(int(edges[b])),
                        "count": np.uint64(int(c)),
                    }
                )
    return rows


def _test_rows(cfg, res, flipped):
    mode = cfg["per_test_storage"]
    T = len(res)
    if mode == "none":
        keep = np.zeros(T, dtype=bool)
    elif mode == "full":
        keep = np.ones(T, dtype=bool)
    else:
        keep = store.sample_mask(
            res,
            alphas=cfg["alphas"],
            rate=float(cfg["sampling_rate"]),
            seed=int(cfg["sampling_seed"]),
            flipped=flipped,
        )
    idx = np.flatnonzero(keep)
    return {
        "test_idx": idx.astype(np.int32),
        "rho": res.rho[idx],
        "stat": res.stat[idx],
        "status": res.status[idx],
        "kappa_proxy": res.kappa_proxy[idx].astype(np.float32),
        "min_diag": res.min_diag[idx].astype(np.float32),
        "kappa_exact": np.full(idx.size, np.nan, dtype=np.float32),
        "sampled": np.ones(idx.size, dtype=bool),
    }


def run_config(cfg: dict, root, run_id: str, *, progress=None) -> dict:
    """Execute every cell of a config, skipping any whose shard already exists."""
    code_digest = provenance.code_digest()
    env_digest = provenance.env_digest()
    config_digest = provenance.digest_obj(cfg)

    st = store.RunStore(root, run_id)
    st.init(config=cfg, extra={"run_id": run_id})

    # `noise_floor_ok` is stamped into every result row so a Parquet file is
    # self-describing in isolation. It must therefore never claim a gate that did not
    # run: when the gate is not required, the flag reflects whether a PASSING record
    # already exists for this code and environment, not an optimistic default.
    if cfg["require_gate"]:
        noise_ok = gate.require(Path(root)).passed
    else:
        rec = gate.lookup(Path(root))
        noise_ok = bool(rec is not None and rec.passed)

    done = st.done_cells()
    n_done = n_new = 0

    # Group by seed so the dataset and reference cell are built once per seed.
    cells = list(enumerate_cells(cfg, code_digest))
    by_seed: dict = {}
    for c in cells:
        by_seed.setdefault(c.seed, []).append(c)

    for seed, group in sorted(by_seed.items()):
        C, n_obs, _ = build_dataset(cfg, seed)
        ts = build_testset(cfg, C)
        st.write_testset(ts)

        ref_cell = next((c for c in group if c.is_reference), None)
        if ref_cell is None:
            raise ValueError(
                "config has no reference cell: reference=%r is not in the factor grid"
                % (cfg["reference"],)
            )
        ref_res = engine.evaluate(
            C, ts, n_obs,
            fmt=ref_cell.fmt, variant=ref_cell.parcorr_variant,
            ztransform=ref_cell.ztransform_variant,
            rcond=cfg["rcond"], allow_pinv=cfg["allow_pinv"],
            keep_unit_diagonal=cfg["keep_unit_diag"],
        )

        for cell in group:
            if cell.digest in done:
                n_done += 1
                continue
            t0 = time.perf_counter()
            res = ref_res if cell.is_reference else engine.evaluate(
                C, ts, n_obs,
                fmt=cell.fmt, variant=cell.parcorr_variant,
                ztransform=cell.ztransform_variant,
                rcond=cfg["rcond"], allow_pinv=cfg["allow_pinv"],
                keep_unit_diagonal=cfg["keep_unit_diag"],
            )
            wall = time.perf_counter() - t0

            flip_rows = _flip_rows(cfg, cell, res, ref_res, ref_cell.digest, ts)

            # Tests that flipped at ANY alpha are retained unconditionally by the
            # subsampler: they are the numerator of every rate this run reports, and
            # sampling the numerator would be indefensible.
            flipped = np.zeros(len(res), dtype=bool)
            for alpha in cfg["alphas"]:
                c = stats.crit(float(alpha))
                with np.errstate(invalid="ignore"):
                    flipped |= (ref_res.stat > c) != (res.stat > c)

            st.write_shard("flips", cell.digest, _table(_rows_to_cols(flip_rows), "flips"))
            st.write_shard("hist", cell.digest, _table(_rows_to_cols(_hist_rows(cell, res, ts)), "hist"))
            st.write_shard("tests", cell.digest, _table(_test_rows(cfg, res, flipped), "tests"))
            st.write_shard(
                "cells",
                cell.digest,
                _table(
                    _rows_to_cols(
                        _cell_rows(
                            cfg, cell, res, None if cell.is_reference else ref_res,
                            run_id, code_digest, config_digest, env_digest,
                            noise_ok, ts, wall,
                        )
                    ),
                    "cells",
                ),
            )
            n_new += 1
            if progress:
                progress(cell, n_new, len(cells))

    return {"run_id": run_id, "cells_total": len(cells), "cells_new": n_new, "cells_skipped": n_done}


def _rows_to_cols(rows):
    """List-of-dicts -> dict-of-lists, preserving key order and tolerating empties."""
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: [r[k] for r in rows] for k in keys}


def _table(cols: dict, name: str) -> pa.Table:
    """Build a pyarrow table, producing a correctly-typed EMPTY table when there are no rows.

    `pa.table({})` has no schema, so an empty shard written that way would be unreadable
    as part of the dataset. An empty-but-typed shard still means "this cell is done".
    """
    if not cols:
        return schema.SCHEMAS[name].empty_table()
    return pa.table(cols)
