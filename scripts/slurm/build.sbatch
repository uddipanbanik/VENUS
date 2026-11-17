#!/usr/bin/env bash
#SBATCH --job-name=venus-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${VENUS_ROOT:?Set VENUS_ROOT to the repository path}"
cd "$VENUS_ROOT"
export BUILD_JOBS=${SLURM_CPUS_PER_TASK:-4}
# Load the site's compiler/CUDA/MPI modules before submitting; Slurm carries the environment.
exec bash scripts/build/build.sh "${VENUS_BUILD:-cpu}"
