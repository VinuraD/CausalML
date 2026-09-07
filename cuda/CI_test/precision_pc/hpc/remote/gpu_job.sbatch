#!/bin/bash
# Tier 5 GPU job: TF32 calibration and E5.1 throughput. Tokens of the form AT-NAME-AT are
# substituted client-side before submission.
#
# ONE task, one GPU, no array. Unlike the CPU fan-out this is not embarrassingly parallel:
# it is a measurement of one device, and running several would only add contention.
#
# The account is the _gpu association. On Alliance the allocation is split, and charging a
# GPU job to the _cpu account is rejected at sbatch time -- after the queue accepted it.
#SBATCH --job-name=@JOBNAME@
#SBATCH --account=@GPU_ACCOUNT@
#SBATCH --ntasks=1
@GPU_DIRECTIVE@
#SBATCH --cpus-per-task=@CPUS@
#SBATCH --mem=@MEM@
#SBATCH --time=@GPU_TIME@
@GPU_PARTITION@
#SBATCH --output=@RUNDIR@/logs/gpu-%j.out
#SBATCH --signal=B:TERM@90
set -euo pipefail

echo "[gpu] job $SLURM_JOB_ID on $(hostname -s) at $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONHASHSEED=0

# TF32 is the whole point of the calibration, so its state must be explicit rather than
# inherited. NVIDIA_TF32_OVERRIDE is the library-wide switch (cuBLAS, cuDNN, cuSOLVER);
# leaving it unset means the answer depends on library defaults that have changed across
# CUDA versions. The calibration turns TF32 on deliberately, per call, from Python.
unset NVIDIA_TF32_OVERRIDE

SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ppc-gpu-${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH"
export NUMBA_CACHE_DIR="$SCRATCH/numba" MPLCONFIGDIR="$SCRATCH/mpl"
export XDG_CACHE_HOME="$SCRATCH/xdg" PYTHONPYCACHEPREFIX="$SCRATCH/pycache"
export CUDA_CACHE_PATH="$SCRATCH/nv"   # PTX JIT cache; shared-FS default races
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$CUDA_CACHE_PATH"

module load @MODULES@
source "@REMOTE_ROOT@/venv/bin/activate"
export PYTHONPATH="@CODEDIR@:${PYTHONPATH:-}"

python -c "import torch; print('[gpu] torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())" || {
  echo "[gpu] torch is not installed in the venv. Run: hpc stage --with-torch" >&2
  exit 1
}

python "@CODEDIR@/precision_pc/hpc/remote/gpu_job.py" "@RUNDIR@"
echo "[gpu] complete $(date -u +%FT%TZ)"
