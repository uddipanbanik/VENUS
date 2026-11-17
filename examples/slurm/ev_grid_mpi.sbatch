#!/usr/bin/env bash
#SBATCH --job-name=venus-ev_grid-mpi
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${VENUS_ROOT:?Set VENUS_ROOT to the absolute repository path}"
cd "$VENUS_ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-python3}" scripts/run_example.py ev_grid --variant mpi --launcher "${VENUS_MPI_LAUNCHER:-mpiexec}" --ranks "${SLURM_NTASKS:-2}" --threads "$OMP_NUM_THREADS" "$@"
