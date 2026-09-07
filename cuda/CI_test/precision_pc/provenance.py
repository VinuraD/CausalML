"""Run provenance: what ran, on what, from which code.

The proposal's section 5.1 ends with "fix seeds, library versions, and driver versions;
log them alongside every reported number". This module is that instruction made
structural, so it cannot be forgotten for one run and remembered for the next.

Three digests, each answering a different question:

    code_digest    did the source change?          -> invalidates cached results
    config_digest  did the experiment change?      -> names the run
    env_digest     did the machine or stack change? -> invalidates the E0.1 gate record

They are separate because they invalidate different things. A new numpy version must
re-open the noise-floor question without forcing a recompute of unrelated cells; an edit
to `parcorr.py` must invalidate the cells but not the gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent


def _sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, stable float repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest_obj(obj: Any) -> str:
    return _sha256_of(canonical_json(obj))


# ------------------------------------------------------------------- code digest
def code_digest() -> str:
    """SHA-256 over every .py file in the package, by relative path and content.

    Hashing the source rather than trusting the git SHA is deliberate: the working tree
    is usually dirty during development, and a result computed from uncommitted code must
    not silently claim the provenance of the last commit.
    """
    h = hashlib.sha256()
    for p in sorted(_PKG.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(_PKG)).replace(os.sep, "/").encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()


# -------------------------------------------------------------------- git status
def git_info(cwd: Path | None = None) -> dict:
    cwd = cwd or _PKG
    def _run(args):
        try:
            r = subprocess.run(
                args, cwd=str(cwd), capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "sha": sha,
        "dirty": None if status is None else bool(status.strip()),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


# ------------------------------------------------------------------ environment
def _version(mod_name: str):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def blas_info() -> dict:
    """Which BLAS numpy is actually linked against, and how many threads it will use."""
    import numpy as np

    out: dict = {"vendor": None, "version": None, "threadpools": []}
    try:
        cfg = np.show_config("dicts") or {}
        blas = (cfg.get("Build Dependencies") or {}).get("blas") or {}
        out["vendor"] = blas.get("name")
        out["version"] = blas.get("version")
    except Exception:
        pass
    from . import env as _env

    out["threadpools"] = [
        {
            "internal_api": p.get("internal_api"),
            "prefix": p.get("prefix"),
            "version": p.get("version"),
            "num_threads": p.get("num_threads"),
        }
        for p in _env.threadpools()
    ]
    return out


def gpu_info() -> dict:
    """TF32 flag state and device identity. Never imports torch unless it is already loaded.

    Importing torch costs seconds and initialises CUDA, which the core FP-simulation path
    must never pay for. But when torch IS in play, its TF32 flags have to be recorded with
    every result: PyTorch may silently use TF32 for float32 matmul, the defaults have
    changed across versions, and an unrecorded flag turns TF32 from an experimental level
    into an invisible contaminant.
    """
    if "torch" not in sys.modules:
        return {"torch_loaded": False}
    try:
        import torch

        info = {
            "torch_loaded": True,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
        return info
    except Exception as exc:
        return {"torch_loaded": True, "error": repr(exc)}


def env_info() -> dict:
    from . import env as _env

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "versions": {
            name: _version(name)
            for name in ("numpy", "scipy", "numba", "ml_dtypes", "pandas", "pyarrow")
        },
        "blas": blas_info(),
        "threads": _env.report(),
        "gpu": gpu_info(),
    }


def env_digest() -> str:
    """Digest of the parts of the environment that can change a floating-point result.

    Deliberately excludes the hostname and the platform string: two identical stacks on
    two machines of the same architecture should share a gate record. It DOES include the
    BLAS vendor, version and thread counts, and the library versions.
    """
    info = env_info()
    material = {
        "python": info["python"],
        "machine": info["machine"],
        "versions": info["versions"],
        "blas_vendor": info["blas"]["vendor"],
        "blas_version": info["blas"]["version"],
        "threads": info["threads"],
        "threadpools": [
            {"api": p["internal_api"], "n": p["num_threads"]}
            for p in info["blas"]["threadpools"]
        ],
    }
    return digest_obj(material)


# ---------------------------------------------------------------------- manifest
def manifest(config: dict | None = None, extra: dict | None = None) -> dict:
    """The full provenance record written beside every run."""
    import datetime as _dt

    m = {
        "schema_version": 1,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "code_digest": code_digest(),
        "env_digest": env_digest(),
        "git": git_info(),
        "env": env_info(),
    }
    if config is not None:
        m["config"] = config
        m["config_digest"] = digest_obj(config)
    if extra:
        m.update(extra)
    return m


def write_manifest(path, config: dict | None = None, extra: dict | None = None) -> dict:
    """Write the manifest atomically. Returns what was written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = manifest(config, extra)
    # Every array task calls RunStore.init(), so this is written concurrently.
    from . import atomic

    atomic.write_text(path, json.dumps(m, indent=2, default=str))
    return m
