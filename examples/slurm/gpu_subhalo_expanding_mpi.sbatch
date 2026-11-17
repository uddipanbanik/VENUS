#!/usr/bin/env bash
#SBATCH --job-name=venus-gpu_subhalo_expanding-mpi
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --gpus-per-task=1
set -euo pipefail
: "${VENUS_ROOT:?Set VENUS_ROOT to the absolute repository path}"
cd "$VENUS_ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-python3}" scripts/run_example.py gpu_subhalo_expanding --variant mpi --launcher srun --ranks "${SLURM_NTASKS:-2}" --threads "$OMP_NUM_THREADS" "$@"
