#!/bin/bash
# Launch a single-process OpenMP BP calculation through the IAS Slurm environment.
# Input configuration and executable paths may be supplied through exported variables.
# Output/restart directories belong to this job. Scheduler directives below define the allocation.

#
# IAS Apollo/Typhon template for Boltzmann-Poisson collision runs
# (Fokker-Planck/Landau or finite-angle Boltzmann).  BP collisions currently
# run with one MPI task because their velocity-space coefficients and L fluxes
# need cross-rank exchange for distributed L slabs.

#SBATCH --job-name=venus-bp-omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --partition=EDIT_ME
##SBATCH --account=EDIT_ME
##SBATCH --mem=0

# Section: shell safety.
# Stop immediately if a command fails or an expected variable is missing.
set -euo pipefail

# Section: user-overridable paths.
# BP collision jobs should stay on one MPI task, but CONFIG, OUTPUT_DIR,
# EXECUTABLE, and RESTART_DIR can be changed with sbatch --export.  Relative
# CONFIG/OUTPUT_DIR/RESTART_DIR values are resolved from the sbatch submission
# directory; relative EXECUTABLE values are resolved from VENUS_DIR.  Prefer
# SLURM_SUBMIT_DIR when it is the VENUS checkout because Slurm may execute a
# copied script from /var/spool.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve user-relative paths from the submission directory, since Slurm may run a spooled script copy.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -z "${VENUS_DIR:-}" ]; then
  if [ -f "$SUBMIT_DIR/Makefile" ] && [ -d "$SUBMIT_DIR/include/ev" ]; then
    VENUS_DIR="$SUBMIT_DIR"
  else
    VENUS_DIR="$SCRIPT_DIR/../.."
  fi
fi
VENUS_DIR="$(cd "$VENUS_DIR" && pwd)"
VENUS_RUN_ROOT="${VENUS_RUN_ROOT:-$SUBMIT_DIR/venus_runs}"
CONFIG="${CONFIG:-$VENUS_DIR/examples/modular/bp_sidm_corecollapse_yukawa.ini}"
EXECUTABLE="${EXECUTABLE:-$VENUS_DIR/venus_cpu_boltzmann_poisson_fokker_planck_openmp}"
OUTPUT_DIR="${OUTPUT_DIR:-$VENUS_RUN_ROOT/${SLURM_JOB_NAME:-venus}_${SLURM_JOB_ID:-manual}}"
RESTART_DIR="${RESTART_DIR:-$OUTPUT_DIR/restarts}"

case "$CONFIG" in
  /*) ;;
  *) CONFIG="$SUBMIT_DIR/$CONFIG" ;;
esac
# Executable paths use the source directory as their base. Data/output paths use the submission directory.
case "$EXECUTABLE" in
  /*) ;;
  *) EXECUTABLE="$VENUS_DIR/$EXECUTABLE" ;;
esac
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SUBMIT_DIR/$OUTPUT_DIR" ;;
esac
case "$RESTART_DIR" in
  /*) ;;
  *) RESTART_DIR="$SUBMIT_DIR/$RESTART_DIR" ;;
esac

# Section: compiler module.
# BP collision runs use OpenMP only in VENUS CPU, so no MPI module is loaded.
# Set VENUS_SKIP_MODULES=1 when the cluster login environment is already
# configured, or VENUS_PURGE_MODULES=0 to keep preloaded helper modules.
if command -v module >/dev/null 2>&1 && [ "${VENUS_SKIP_MODULES:-0}" != "1" ]; then
  if [ "${VENUS_PURGE_MODULES:-1}" != "0" ]; then
    module purge
  fi
  module load "${IAS_GCC_MODULE:-gcc-toolset/10}"
fi

# Section: build.
# Build the task-specific OpenMP fronts.  The selected executable then rejects
# a configuration belonging to a different collision operator.
cd "$VENUS_DIR"
if [ "${VENUS_BUILD:-1}" != "0" ]; then
  VENUS_MAKE_JOBS="${VENUS_MAKE_JOBS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}}"
  make -j "$VENUS_MAKE_JOBS" cpu-task-solvers-openmp CXX="${CXX:-g++}" OPENMP_FLAGS="${OPENMP_FLAGS:--fopenmp}"
fi
# Require the selected compiled application before creating run output or invoking srun.
if [ ! -x "$EXECUTABLE" ]; then
  echo "Executable not found or not executable: $EXECUTABLE" >&2
  echo "Set EXECUTABLE to a valid binary or leave VENUS_BUILD=1 so the script builds it." >&2
  exit 2
fi

# Section: OpenMP and VENUS runtime environment.
# --cpus-per-task controls the number of worker threads inside the one process.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_DISPLAY_ENV="${OMP_DISPLAY_ENV:-FALSE}"
# Pass the allocation's worker count and resolved data paths to the executable's environment interface.
export EV_THREADS="$OMP_NUM_THREADS"
export EV_CONFIG="$CONFIG"
export EV_OUTPUT_DIR="$OUTPUT_DIR"
export EV_RESTART_DIR="$RESTART_DIR"

mkdir -p "$OUTPUT_DIR" "$RESTART_DIR"

# Section: resolved launch configuration.
echo "VENUS_DIR=$VENUS_DIR"
echo "SUBMIT_DIR=$SUBMIT_DIR"
echo "VENUS_RUN_ROOT=$VENUS_RUN_ROOT"
echo "EXECUTABLE=$EXECUTABLE"
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESTART_DIR=$RESTART_DIR"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "VENUS_BUILD=${VENUS_BUILD:-1}"

# Use a shell array so each option/value remains one argument, including paths containing spaces.
solver_args=(
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --restart-dir "$RESTART_DIR"
  --threads "$OMP_NUM_THREADS"
)

# Append only explicitly supplied run overrides. Remaining physical/numerical controls come from CONFIG.
if [ -n "${T_END:-}" ]; then
  solver_args+=(--t-end "$T_END")
fi
if [ -n "${SAVE_EVERY:-}" ]; then
  solver_args+=(--save-every "$SAVE_EVERY")
fi
if [ -n "${RESTART_EVERY:-}" ]; then
  solver_args+=(--restart-every "$RESTART_EVERY")
fi

# Section: OpenMP-only launch.
# Keep --ntasks=1 for Fokker-Planck/Landau and finite-angle Boltzmann
# collisions; increase --cpus-per-task to use more cores on the node.
# Physics and numerical choices live in CONFIG.  In particular, the Vlasov CFL
# timestep, the interval between collision-operator applications, and any
# internal collision substep are separate controls and should be convergence
# tested independently; allocating more cores changes runtime, not those
# physical timescale comparisons.
srun --ntasks=1 --cpus-per-task="$OMP_NUM_THREADS" --cpu-bind=cores \
  "$EXECUTABLE" \
  "${solver_args[@]}"
