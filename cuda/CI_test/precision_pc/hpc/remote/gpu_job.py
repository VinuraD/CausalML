"""Tier 5 remote entrypoint: TF32 calibration and E5.1 throughput.

    python gpu_job.py <run_dir>

THIS JOB REPORTS, IT DOES NOT ABORT. A GPU allocation costs a queue wait and an
interactive second factor to obtain; a job that dies on one bad measurement wastes both
and returns nothing to explain why. Every section is wrapped, and a report is written
whatever happens -- even one that only says what failed.

It exits 0 even on failure, deliberately: a non-zero exit makes SLURM mark the job FAILED,
which the client reads as "no results" rather than "results you will not like".
"""

import json
import os
import sys
import traceback
from pathlib import Path


def _pin():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _section(report, name, fn):
    """Run one measurement; record its result or its traceback, never both missing."""
    try:
        report[name] = fn()
    except Exception as exc:  # noqa: BLE001
        report[name] = {"error": repr(exc)[:500], "traceback": traceback.format_exc()[-3000:]}
        print("[gpu] section %s FAILED: %r" % (name, exc), flush=True)
        traceback.print_exc()


def main(argv):
    if len(argv) < 2:
        print("usage: gpu_job.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)

    _pin()
    import precision_pc  # noqa: F401 -- pins before numpy
    from precision_pc import atomic, gpu, provenance

    report = {"provenance": provenance.manifest(), "device": gpu.device_info()}
    print("[gpu] device: %s" % report["device"].get("device_name"), flush=True)
    print("[gpu] capability: %s  SMs: %s" % (report["device"].get("capability"),
                                             report["device"].get("sm_count")), flush=True)

    # --- B3: is the harness's simulated TF32 the same rounding the hardware does? ------
    # Every TF32 number this study has produced is provisional until this passes: PTX
    # exposes both cvt.rn (nearest-even) and cvt.rna (ties away), and `formats.to_tf32`
    # assumes the former.
    _section(report, "tf32_calibration", gpu.calibrate_tf32)
    c = report.get("tf32_calibration", {})
    if c.get("ran"):
        print("[gpu] TF32: device rounds %s; harness assumes rne; assumption holds: %s"
              % (c.get("device_mode"), c.get("assumption_holds")), flush=True)
    else:
        print("[gpu] TF32 calibration skipped: %s" % c.get("reason"), flush=True)

    # --- B5: E5.1 throughput ----------------------------------------------------------
    # Two quantities, never conflated: the within-device FP64:FP32 RATIO is normalised and
    # architecture-dependent, so it is the number that supports "FP64 is nearly free on a
    # datacenter card"; ABSOLUTE throughput needs a real card and is what a practitioner
    # budgets against.
    def _throughput():
        rows = []
        for n_vars, level in ((30, 1), (60, 1), (90, 1), (30, 2)):
            r = gpu.throughput(
                n_vars=n_vars, n_obs=1000, level=level,
                dtypes=("float64", "float32", "bfloat16"),
                batch_sizes=(None, 8192, 65536),
                reps=5,
            )
            for row in r:
                row["n_vars"] = n_vars
                row["level"] = level
            rows.extend(r)
        return {"rows": rows, "fp64_fp32_ratio": gpu.fp64_fp32_ratio(rows)}

    _section(report, "throughput", _throughput)

    # --- the device must be reproducible against ITSELF before any cross-device claim --
    def _determinism():
        import numpy as np

        from precision_pc import corr, metrics
        from precision_pc import testset as T

        rng = np.random.default_rng(0)
        C = corr.correlation(rng.standard_normal((500, 24)))
        adj = np.ones((24, 24), dtype=bool)
        np.fill_diagonal(adj, False)
        ts = T.from_adjacency(adj, 1)
        a = gpu.evaluate(C, ts, 500, compute_dtype="float64")
        b = gpu.evaluate(C, ts, 500, compute_dtype="float64")
        return {
            "n_tests": len(ts),
            "bitwise_identical": bool(metrics.is_bitwise_identical(a, b)),
            "flips": metrics.flip_counts(a, b, 0.01, undefined_policy="exclude").n_flip,
        }

    _section(report, "device_determinism", _determinism)

    # --- E0.2 arm B on real datacenter silicon ----------------------------------------
    def _crossdevice():
        import numpy as np

        from precision_pc import corr, crossdevice as X
        from precision_pc import testset as T
        from precision_pc.datasets import var as V

        out = {}
        rng = np.random.default_rng(0)
        C = corr.correlation(rng.standard_normal((500, 16)))
        adj = np.ones((16, 16), dtype=bool)
        np.fill_diagonal(adj, False)
        out["iid"] = [vars(r) for r in X.linalg_arm(C, T.from_adjacency(adj, 1), 500)]
        Cv, nv, _ = V.design(n_obs=4000, phi=0.99, tau_max=4, n_series=4, seed=0)
        N = Cv.shape[0]
        av = np.ones((N, N), dtype=bool)
        np.fill_diagonal(av, False)
        out["var_phi099"] = [vars(r) for r in X.linalg_arm(Cv, T.from_adjacency(av, 1), nv)]
        return out

    _section(report, "crossdevice_armB", _crossdevice)

    out = run_dir / "gpu_report.json"
    atomic.write_text(out, json.dumps(report, indent=2, default=str))
    print("[gpu] wrote %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
