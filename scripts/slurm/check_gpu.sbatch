#!/usr/bin/env bash
#SBATCH --job-name=venus-gpu-checks
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
: "${VENUS_ROOT:?Set VENUS_ROOT to the repository path}"
cd "$VENUS_ROOT"
# A two-GPU MPI build also supplies the single-GPU applications and tests.
export BUILD_JOBS=${SLURM_CPUS_PER_TASK:-4}
bash scripts/build/gpu_mpi.sh
bash scripts/check/gpu.sh build/gpu-mpi
