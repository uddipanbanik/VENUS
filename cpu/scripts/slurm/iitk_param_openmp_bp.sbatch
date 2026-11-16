#!/bin/bash
# Launch a single-process OpenMP BP calculation through the PARAM Slurm environment.
# Input configuration and executable paths may be supplied through exported variables.
# Output/restart directories belong to this job. Scheduler directives below define the allocation.

#
# IIT Kanpur PARAM Sanganak template for Boltzmann-Poisson collision runs
# (Fokker-Planck/Landau or finite-angle Boltzmann).  Use one MPI task and fill
# the node with OpenMP threads until BP coefficient exchange and L-boundary
# ghost fluxes are implemented.

#SBATCH --job-name=venus-param-bp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --partition=cpu
##SBATCH --account=EDIT_ME
##SBATCH --mem=0

# Section: shell safety.
# Fail fast when setup commands fail or required variables are unset.
set -euo pipefail

# Section: user-overridable paths.
# Keep the default example conservative; override CONFIG and OUTPUT_DIR for
# production PARAM jobs with sbatch --export.  Relative
# CONFIG/OUTPUT_DIR/RESTART_DIR values are resolved from the sbatch submission
# directory; relative EXECUTABLE values are resolved from VENUS_DIR.  Prefer
# SLURM_SUBMIT_DIR when it is the VENUS checkout because Slurm may execute a
# copied script from its spool directory.
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
CONFIG="${CONFIG:-$VENUS_DIR/examples/modular/bp_expanding_nfw_gamma1p5_weak_collision.ini}"
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
# BP collision kernels are threaded locally; no MPI module is required for this
# VENUS CPU launch path.
# Set VENUS_SKIP_MODULES=1 when the environment is already prepared.
if command -v module >/dev/null 2>&1 && [ "${VENUS_SKIP_MODULES:-0}" != "1" ]; then
  if [ "${VENUS_PURGE_MODULES:-1}" != "0" ]; then
    module purge
  fi
  if [ -n "${PARAM_GCC_MODULE:-}" ]; then
    module load "$PARAM_GCC_MODULE"
  else
    module load gcc || true
  fi
fi

# Section: build.
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
# Keep one MPI task and use --cpus-per-task for threaded BP coefficient,
# Fokker-Planck, and finite-angle collision kernels.
srun --ntasks=1 --cpus-per-task="$OMP_NUM_THREADS" --cpu-bind=cores \
  "$EXECUTABLE" \
  "${solver_args[@]}"
