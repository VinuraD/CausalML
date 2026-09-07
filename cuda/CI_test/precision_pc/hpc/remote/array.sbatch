#!/bin/bash
# Template. Tokens of the form AT-NAME-AT are substituted client-side before submission.
#
# One ARRAY of identical workers, each pulling cells from a shared queue -- not a static
# partition. Cell costs vary by orders of magnitude (an FP8 cell hits the pseudo-inverse
# fallback on most tests), so a fixed k::N split would leave most tasks idle. See claim.py.
#SBATCH --job-name=@JOBNAME@
#SBATCH --account=@ACCOUNT@
#SBATCH --array=@ARRAY@
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=@CPUS@
#SBATCH --mem=@MEM@
#SBATCH --time=@TIME@
@PARTITION@
#SBATCH --output=@RUNDIR@/logs/slurm-%A_%a.out
#SBATCH --signal=B:TERM@90
set -euo pipefail

echo "[job] array task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on $(hostname -s) at $(date -u +%FT%TZ)"

# Thread pinning is set here AND again inside worker.py. Here because the environment must
# be right before python starts; there because a worker may be run outside SLURM. With N
# concurrent tasks on a many-core node, an unpinned OpenBLAS sizes its pool from the cores
# it can SEE rather than the ones it was allocated, oversubscribes the node by ~50x, and
# destroys the reduction-order determinism that experiment E0.1 depends on.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONHASHSEED=0

# Compute nodes have no outbound internet. Everything must resolve locally.
export PIP_NO_INDEX=1

module load @MODULES@

# Node-local scratch for every cache that would otherwise race across array tasks.
# $SLURM_TMPDIR is an Alliance convention, not standard SLURM; fall back if absent.
SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ppc-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH"
export NUMBA_CACHE_DIR="$SCRATCH/numba" MPLCONFIGDIR="$SCRATCH/mpl"
export XDG_CACHE_HOME="$SCRATCH/xdg" PYTHONPYCACHEPREFIX="$SCRATCH/pycache"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"

# The venv is built ONCE by stage.sh on a login node and lives on the shared filesystem.
# Rebuilding it per task would mean N tasks each unpacking thousands of small wheel files
# onto a shared filesystem simultaneously -- slow, and antisocial on a shared cluster.
source "@REMOTE_ROOT@/venv/bin/activate"
export PYTHONPATH="@CODEDIR@:${PYTHONPATH:-}"

python -c "import numpy, precision_pc; print('[job] numpy', numpy.__version__, '| precision_pc', precision_pc.__version__)"

# Warm the numba cache from the shared copy if stage.sh built one, so each task pays a
# file copy (milliseconds) instead of a JIT compile (~12s measured).
if [ -d "@REMOTE_ROOT@/numba-cache" ]; then
  cp -r "@REMOTE_ROOT@/numba-cache/." "$NUMBA_CACHE_DIR/" 2>/dev/null || true
fi

# The E0.1 noise floor is a property of the ENVIRONMENT, and a compute node is not the
# laptop: different CPU, possibly different BLAS kernels. A floor measured locally says
# nothing here, so establish it on the node before computing anything.
python -m precision_pc --root "@RUNDIR@/.." gate --n-vars 24 --n-obs 250 || {
  echo "[job] GATE FAILED on this node -- refusing to compute results that could not be interpreted" >&2
  exit 1
}

python "@CODEDIR@/precision_pc/hpc/remote/worker.py" "@RUNDIR@" "@RUNDIR@/config.yaml" "@MAXSECONDS@"
echo "[job] complete $(date -u +%FT%TZ)"
