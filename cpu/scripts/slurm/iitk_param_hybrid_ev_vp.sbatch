#!/bin/bash
# Launch a MPI/OpenMP collisionless EV or VP calculation through the PARAM Slurm environment.
# Input configuration and executable paths may be supplied through exported variables.
# Output/restart directories belong to this job. Scheduler directives below define the allocation.

#
# IIT Kanpur PARAM Sanganak template for MPI+OpenMP Einstein-Vlasov or
# collisionless Vlasov-Poisson runs.  PARAM software module and partition names
# should be checked on login with `module avail` and `sinfo`.

#SBATCH --job-name=venus-param-hybrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --partition=cpu
##SBATCH --account=EDIT_ME
##SBATCH --mem=0

# Section: shell safety.
# Stop on unset variables or failed commands so the job log reveals failures
# before srun rather than launching with partial setup.
set -euo pipefail

# Section: user-overridable paths.
# PARAM site-specific paths and example choices can be overridden by
# sbatch --export without modifying this template.  Relative
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
CONFIG="${CONFIG:-$VENUS_DIR/examples/modular/vp_expanding_nfw_gamma2_collisionless.ini}"
EXECUTABLE="${EXECUTABLE:-$VENUS_DIR/venus_cpu_vlasov_poisson_eulerian_mpi}"
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

# Section: compiler/MPI modules.
# PARAM module names vary by environment.  Explicit PARAM_* variables take
# precedence; otherwise the script tries generic gcc/openmpi names.  Set
# VENUS_SKIP_MODULES=1 when the environment is already prepared.
if command -v module >/dev/null 2>&1 && [ "${VENUS_SKIP_MODULES:-0}" != "1" ]; then
  if [ "${VENUS_PURGE_MODULES:-1}" != "0" ]; then
    module purge
  fi
  if [ -n "${PARAM_GCC_MODULE:-}" ]; then
    module load "$PARAM_GCC_MODULE"
  else
    module load gcc || true
  fi
  if [ -n "${PARAM_MPI_MODULE:-}" ]; then
    module load "$PARAM_MPI_MODULE"
  else
    module load openmpi || true
  fi
fi

# Section: build.
# Use the MPI target for EV and collisionless VP, where L-slab decomposition is
# valid without collision-coefficient exchange.
cd "$VENUS_DIR"
if [ "${VENUS_BUILD:-1}" != "0" ]; then
  VENUS_MAKE_JOBS="${VENUS_MAKE_JOBS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}}"
  make -j "$VENUS_MAKE_JOBS" cpu-task-solvers-mpi CXX="${CXX:-g++}" MPICXX="${MPICXX:-mpicxx}" OPENMP_FLAGS="${OPENMP_FLAGS:--fopenmp}"
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
echo "SLURM_NTASKS=${SLURM_NTASKS:-1}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "VENUS_BUILD=${VENUS_BUILD:-1}"

# Use a shell array so each option/value remains one argument, including paths containing spaces.
solver_args=(
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --threads "$OMP_NUM_THREADS"
)

# Section: executable compatibility.
# The modular MPI executable accepts --restart-dir; the standalone EV executable
# may not, so the template adapts based on the executable basename.
case "$(basename "$EXECUTABLE")" in
  venus_einstein_vlasov*)
    supports_restart_dir=0
    ;;
  *)
    supports_restart_dir=1
    ;;
esac

if [ "${USE_RESTART_DIR:-$supports_restart_dir}" != "0" ]; then
  solver_args+=(--restart-dir "$RESTART_DIR")
fi

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

# Section: MPI launch.
# Do not use this template for active BP collision operators; use the PARAM BP
# OpenMP template until collision MPI support is implemented.
srun --cpu-bind=cores "$EXECUTABLE" \
  "${solver_args[@]}"
