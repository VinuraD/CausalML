#!/bin/bash
# Run ONCE on a LOGIN NODE, and again whenever dependencies change.
#
#   bash stage.sh <remote_root> <modules>
#
# Compute nodes have no outbound internet; login nodes do. So everything the jobs need
# must be resolved here first:
#
#   <remote_root>/venv/          the environment every array task activates
#   <remote_root>/wheelhouse/    wheels for anything missing from the cluster's own index
#   <remote_root>/numba-cache/   a pre-warmed JIT cache, copied node-local by each task
#   <remote_root>/code/          the precision_pc source tree (pushed separately)
#
# Re-running is safe: the venv is reused unless --force is passed.
set -uo pipefail

ROOT=${1:?usage: stage.sh <remote_root> <modules>}
MODULES=${2:?}
FORCE=${3:-}

# A leading ~ arrives here as a LITERAL character: the path came in as an argv element and
# is used as "$ROOT", and bash only expands tildes on unquoted literals at parse time.
# Left alone it silently creates a directory actually named "~" under $HOME.
case "$ROOT" in
  "~")   ROOT="$HOME" ;;
  "~/"*) ROOT="$HOME/${ROOT#\~/}" ;;
esac

mkdir -p "$ROOT" || { echo "[stage] cannot write $ROOT" >&2; exit 1; }
# Logical pwd, matching slurm.py preflight. Absolute -- SLURM expands neither ~ nor
# $VARS in #SBATCH directives -- but NOT symlink-resolved: on Alliance ~/scratch is a
# symlink, and the physical target is not guaranteed to be mounted the same way on the
# compute nodes. Resolving it here also made the venv path disagree with the one
# array.sbatch bakes in.
ROOT=$(cd "$ROOT" && pwd)
echo "[stage] root    : $ROOT"
echo "[stage] modules : $MODULES"

module load $MODULES || { echo "[stage] module load failed" >&2; exit 1; }

VENV="$ROOT/venv"
if [ -n "$FORCE" ] && [ -d "$VENV" ]; then
  echo "[stage] --force: removing existing venv"
  rm -rf "$VENV"
fi

if [ -d "$VENV" ]; then
  echo "[stage] reusing existing venv"
else
  echo "[stage] building venv"
  if command -v virtualenv >/dev/null 2>&1; then
    # --system-site-packages so packages provided by a MODULE (Alliance ships
    # pyarrow via `module load arrow`, and blocks the pip route with a dummy
    # pyarrow==9999 wheel) are visible inside the venv. Without it the venv is
    # isolated and the module might as well not be loaded.
    virtualenv --no-download --system-site-packages "$VENV" || exit 1
  else
    python -m venv --system-site-packages "$VENV" || exit 1
  fi
fi
source "$VENV/bin/activate"

# Prefer the cluster's own wheelhouse (--no-index). Alliance clusters carry numpy, scipy,
# numba and pyarrow; ml_dtypes and threadpoolctl are the likely gaps, so fall back to
echo "[stage] venv python: $(python -V 2>&1)  ($(command -v python))"
# downloading them HERE, where there is internet, into a local wheelhouse the compute
# nodes can install from offline.
WHEELS="$ROOT/wheelhouse"
mkdir -p "$WHEELS"
pip install --no-index --upgrade pip >/dev/null 2>&1 || true

# Import name != distribution name for two of these.
modname () { case "$1" in PyYAML) echo yaml ;; *) echo "$1" ;; esac; }

PKGS="numpy scipy numba pyarrow pandas PyYAML threadpoolctl ml_dtypes"
MISSING=""
for p in $PKGS; do
  m=$(modname "$p")

  # A package may already be importable because a MODULE provided it -- Alliance ships
  # pyarrow that way and publishes a dummy `pyarrow==9999` wheel to block the pip route.
  # Check before installing, or the 9999 sentinel is mistaken for a real failure.
  if python -c "import $m" >/dev/null 2>&1; then
    echo "[stage] $p        <- already importable (module-provided)"
    continue
  fi

  if pip install --no-index "$p" >/dev/null 2>&1 && python -c "import $m" >/dev/null 2>&1; then
    echo "[stage] $p        <- cluster wheelhouse"
    continue
  fi

  echo "[stage] $p        not in cluster index; downloading from PyPI"
  LOG="$WHEELS/.$p.log"
  # Alliance pins pip to their own wheelhouse via PIP_CONFIG_FILE *and* a constraints
  # file, and ships a patched pip. Clear both for the FALLBACK only -- the --no-index
  # attempt above must still prefer the cluster's curated wheels. --only-binary because
  # building pyarrow from source needs CMake and Arrow C++.
  if env -u PIP_CONFIG_FILE -u PIP_CONSTRAINT pip download --dest "$WHEELS"        --only-binary=:all: --index-url https://pypi.org/simple "$p" >"$LOG" 2>&1      && pip install --no-index --find-links "$WHEELS" "$p" >>"$LOG" 2>&1; then
    echo "[stage] $p        <- PyPI wheelhouse"
    continue
  fi

  echo "[stage] $p        FAILED to install" >&2
  echo "[stage]   --- why $p failed (last 12 lines) ---" >&2
  tail -n 12 "$LOG" 2>/dev/null | sed "s/^/[stage]   /" >&2
  echo "[stage]   --- cluster index ---" >&2
  avail_wheels "$p" 2>&1 | head -6 | sed "s/^/[stage]   /" >&2 || true
  echo "[stage]   --- is a module available? ---" >&2
  module -t spider "$p" 2>&1 | head -6 | sed "s/^/[stage]   /" >&2 || true
  echo "[stage]   --- pip constraints in effect ---" >&2
  env | grep -E "^PIP_" | sed "s/^/[stage]   /" >&2 || true
  MISSING="$MISSING $p"
done

echo "[stage] installed versions:"
python - <<'PY'
for m in ("numpy","scipy","numba","pyarrow","pandas","yaml","threadpoolctl","ml_dtypes"):
    try:
        mod = __import__(m)
        print("   %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("   %-14s MISSING (%s)" % (m, type(e).__name__))
PY

pip freeze > "$ROOT/requirements.lock" 2>/dev/null || true

# Pre-warm the numba JIT cache once, so each array task copies files instead of compiling.
CODE="$ROOT/code"
if [ -d "$CODE/precision_pc" ]; then
  echo "[stage] warming the numba cache"
  mkdir -p "$ROOT/numba-cache"
  ( cd "$CODE" && NUMBA_CACHE_DIR="$ROOT/numba-cache" PYTHONPATH="$CODE" \
      python -c "import precision_pc, numpy as np; from precision_pc import corr, testset, engine;
C = corr.correlation(np.random.default_rng(0).standard_normal((200, 8)));
engine.evaluate(C, testset.level0(8), 200)" >/dev/null 2>&1 ) \
    && echo "[stage] numba cache warmed" || echo "[stage] cache warm skipped"
else
  echo "[stage] no code at $CODE yet -- push it, then re-run stage.sh to warm the cache"
fi

if [ -n "$MISSING" ]; then
  echo >&2
  echo "[stage] STAGING INCOMPLETE -- could not install:$MISSING" >&2
  echo "[stage] Every array task would fail with an ImportError, after a queue wait." >&2
  exit 1
fi

echo
echo "[stage] ================ done ================"
echo "[stage] venv         : $VENV"
echo "[stage] wheelhouse   : $WHEELS"
echo "[stage] resolved root: $ROOT"
echo "[stage] paste this into ~/.precision-pc/hpc.toml if it differs:"
echo "[stage]   remote_root = \"$ROOT\""
