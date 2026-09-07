"""Cascade mode driver: E2.1-E2.4, E3.2 and E4.1 all fall out of one loop.

Two arms, deliberately separate, computed on different test sets and stored side by side
so that the distinction is impossible to forget:

    cascade arm   each format runs its own free search   -> edges_changed   (NUMERATOR)
    union arm     every format replays the UNION of all  -> n_union_flip    (DENOMINATOR)
                  trajectories

A flip rate is not even *defined* in cascade mode, because different formats execute
different numbers of tests. That is exactly why the denominator has to come from a fixed
test set, and why the union is the right one: it is a superset of every format's
trajectory, so unlike a plain FP64 replay it does not privilege the reference's path.
Amplification is then `edges_changed / n_union_flip` -- dimensionless, and what
contribution C1 actually claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import engine, gate, metrics, orient, parcorr, provenance, schema, skeleton, store
from .runner import _rows_to_cols, _table, build_dataset


def dataset(cfg: dict, seed: int):
    """Like `runner.build_dataset`, but also returns ground truth when the DGP has one."""
    spec = cfg["dataset"]
    if spec.get("kind") == "dag":
        from .datasets import gaussian as g

        return g.design(
            n_vars=int(cfg["n_vars"]),
            n_obs=int(cfg["n_obs"]),
            edge_prob=float(spec.get("edge_prob", 0.25)),
            sampling=spec.get("sampling", "population"),
            seed=seed,
        )
    C, n_obs, dsid = build_dataset(cfg, seed)
    return C, n_obs, dsid, None, None


def cell_key(cfg, dsid, seed, fmt, stable, code_digest) -> dict:
    ref = cfg["reference"]
    return {
        "experiment": cfg["experiment"],
        "mode": "cascade",
        "dataset_id": dsid,
        # The trajectory is an OUTPUT in cascade mode, not an input, so there is no
        # incoming testset digest to key on.
        "testset_digest": "cascade",
        "seed": int(seed),
        "format": fmt,
        "parcorr_variant": ref["parcorr_variant"],
        "ztransform_variant": ref["ztransform_variant"],
        "stable": bool(stable),
        "stop_at_first": bool(cfg["stop_at_first"]),
        "alpha_run": float(cfg["alpha_run"]),
        "max_level": int(cfg["max_level"]),
        "rcond": cfg["rcond"],
        "allow_pinv": cfg["allow_pinv"],
        "code_digest": code_digest,
    }


def run(cfg: dict, root, run_id: str, *, progress=None) -> dict:
    code_digest = provenance.code_digest()
    env_digest = provenance.env_digest()
    config_digest = provenance.digest_obj(cfg)

    st = store.RunStore(root, run_id)
    st.init(config=cfg, extra={"run_id": run_id})

    if cfg["require_gate"]:
        noise_ok = gate.require(Path(root)).passed
    else:
        rec = gate.lookup(Path(root))
        noise_ok = bool(rec is not None and rec.passed)

    done = st.done_cells("graphs")
    ref = cfg["reference"]
    n_new = n_skip = 0

    for seed in cfg["seeds"]:
        C, n_obs, dsid, truth, truth_dag = dataset(cfg, int(seed))
        for stable in cfg["stable_variants"]:
            # --- cascade arm -------------------------------------------------
            runs = {
                fmt: skeleton.discover(
                    C, n_obs, float(cfg["alpha_run"]),
                    fmt=fmt,
                    variant=ref["parcorr_variant"],
                    ztransform=ref["ztransform_variant"],
                    max_level=int(cfg["max_level"]),
                    stable=bool(stable),
                    stop_at_first=bool(cfg["stop_at_first"]),
                    rcond=cfg["rcond"],
                    allow_pinv=cfg["allow_pinv"],
                    keep_unit_diagonal=cfg["keep_unit_diag"],
                )
                for fmt in cfg["formats"]
            }

            # --- union arm ---------------------------------------------------
            union = None
            for fmt in cfg["formats"]:
                t = runs[fmt].trajectory
                union = t if union is None else union.union(t)
            st.write_testset(union)
            replay = {
                fmt: engine.evaluate(
                    C, union, n_obs, fmt=fmt,
                    variant=ref["parcorr_variant"],
                    ztransform=ref["ztransform_variant"],
                    rcond=cfg["rcond"],
                    allow_pinv=cfg["allow_pinv"],
                    keep_unit_diagonal=cfg["keep_unit_diag"],
                )
                for fmt in cfg["formats"]
            }

            rfmt = ref["format"]
            rdig = store.cell_digest(cell_key(cfg, dsid, seed, rfmt, stable, code_digest))
            rrun = runs[rfmt]

            # --- orientation arm ---------------------------------------------
            # Separating sets determine v-structures, which propagate through Meek's
            # rules. This is what turns the sepset divergence into a number about arrows.
            cpdags = {f: orient.from_skeleton_result(runs[f]) for f in cfg["formats"]}
            truth_cpdag = None if truth is None else orient.dag_to_cpdag(truth_dag)
            rcp = cpdags[rfmt]

            for fmt in cfg["formats"]:
                dig = store.cell_digest(cell_key(cfg, dsid, seed, fmt, stable, code_digest))
                if dig in done:
                    n_skip += 1
                    continue
                vrun = runs[fmt]

                sd = metrics.skeleton_diff(rrun, vrun)
                sep = metrics.sepset_divergence(rrun, vrun)
                fc = metrics.flip_counts(
                    replay[rfmt], replay[fmt], float(cfg["alpha_run"]),
                    undefined_policy=cfg["undefined_policy"],
                )

                row = {
                    "cell_digest": dig,
                    "ref_cell_digest": rdig,
                    "run_id": run_id,
                    "experiment": cfg["experiment"],
                    "dataset_id": dsid,
                    "seed": np.int32(seed),
                    "format": fmt,
                    "parcorr_variant": ref["parcorr_variant"],
                    "ztransform_variant": ref["ztransform_variant"],
                    "stable": bool(stable),
                    "stop_at_first": bool(cfg["stop_at_first"]),
                    "alpha": float(cfg["alpha_run"]),
                    "max_level": np.int8(cfg["max_level"]),
                    "n_edges_ref": np.int32(sd.n_edges_ref),
                    "n_edges_var": np.int32(sd.n_edges_var),
                    "edges_changed": np.int32(sd.edges_changed),
                    "edges_lost": np.int32(sd.edges_lost),
                    "edges_gained": np.int32(sd.edges_gained),
                    "shd_truth_ref": (
                        None if truth is None else np.int32(metrics.shd_vs_truth(rrun, truth))
                    ),
                    "shd_truth_var": (
                        None if truth is None else np.int32(metrics.shd_vs_truth(vrun, truth))
                    ),
                    "n_sepset_common": np.int32(sep.n_common_removed),
                    "n_sepset_differ": np.int32(sep.n_sepset_differ),
                    "traj_jaccard": float(metrics.trajectory_jaccard(rrun, vrun)),
                    "n_traj_ref": np.int32(len(rrun.trajectory)),
                    "n_traj_var": np.int32(len(vrun.trajectory)),
                    "n_union_tests": np.int32(fc.n_tests),
                    "n_union_compared": np.int32(fc.n_compared),
                    "n_union_flip": np.int32(fc.n_flip),
                    "n_union_undefined": np.int32(fc.n_undefined),
                    "n_pinv_ref": np.int32((rrun.status == parcorr.PINV_FALLBACK).sum()),
                    "n_pinv_var": np.int32((vrun.status == parcorr.PINV_FALLBACK).sum()),
                    "max_level_ref": np.int8(rrun.max_level_reached),
                    "max_level_var": np.int8(vrun.max_level_reached),
                    "code_digest": code_digest,
                    "config_digest": config_digest,
                    "env_digest": env_digest,
                    "noise_floor_ok": bool(noise_ok),
                    "schema_version": np.int16(schema.SCHEMA_VERSION),
                }
                lv_rows = [
                    {
                        "cell_digest": dig,
                        "level": np.int8(lv),
                        "n_tests": np.uint64(cnt),
                    }
                    for lv, cnt in sorted(vrun.level_counts.items())
                ]

                vcp = cpdags[fmt]
                cd = orient.cpdag_diff(rcp, vcp)
                cpdag_row = {
                    "cell_digest": dig, "ref_cell_digest": rdig, "run_id": run_id,
                    "experiment": cfg["experiment"], "dataset_id": dsid,
                    "seed": np.int32(seed), "format": fmt,
                    "alpha": float(cfg["alpha_run"]),
                    "identical_skeleton": bool(cd.skeleton_differ == 0),
                    "cpdag_shd": np.int32(cd.n_pairs_differ),
                    "skeleton_differ": np.int32(cd.skeleton_differ),
                    "orientation_differ": np.int32(cd.orientation_differ),
                    "n_common_edges": np.int32(cd.n_common_edges),
                    "n_vstruct_ref": np.int32(cd.n_vstruct_ref),
                    "n_vstruct_var": np.int32(cd.n_vstruct_var),
                    "n_vstruct_differ": np.int32(cd.n_vstruct_differ),
                    "n_directed_ref": np.int32(cd.n_directed_ref),
                    "n_directed_var": np.int32(cd.n_directed_var),
                    "n_undirected_ref": np.int32(rcp.n_undirected),
                    "n_undirected_var": np.int32(vcp.n_undirected),
                    "n_conflicts_ref": np.int32(rcp.n_conflicts),
                    "n_conflicts_var": np.int32(vcp.n_conflicts),
                    "cpdag_shd_truth_ref": (
                        None if truth_cpdag is None
                        else np.int32(orient.cpdag_diff(truth_cpdag, rcp).n_pairs_differ)
                    ),
                    "cpdag_shd_truth_var": (
                        None if truth_cpdag is None
                        else np.int32(orient.cpdag_diff(truth_cpdag, vcp).n_pairs_differ)
                    ),
                    "code_digest": code_digest,
                    "schema_version": np.int16(schema.SCHEMA_VERSION),
                }
                st.write_shard("cpdag", dig, _table(_rows_to_cols([cpdag_row]), "cpdag"))
                st.write_shard("levels", dig, _table(_rows_to_cols(lv_rows), "levels"))
                # `graphs` is written LAST: its existence is what marks the cell done.
                st.write_shard("graphs", dig, _table(_rows_to_cols([row]), "graphs"))
                n_new += 1
                if progress:
                    progress(fmt, int(seed), bool(stable), n_new)

    return {"run_id": run_id, "cells_new": n_new, "cells_skipped": n_skip}
