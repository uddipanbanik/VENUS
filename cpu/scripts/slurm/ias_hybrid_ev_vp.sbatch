#!/bin/bash
# Launch a MPI/OpenMP collisionless EV or VP calculation through the IAS Slurm environment.
# Input configuration and executable paths may be supplied through exported variables.
# Output/restart directories belong to this job. Scheduler directives below define the allocation.

#
# IAS Apollo/Typhon template for MPI+OpenMP Einstein-Vlasov or collisionless
# Vlasov-Poisson runs.  Edit the #SBATCH resource lines for queue/account
# policy, and override run choices with sbatch --export.

#SBATCH --job-name=venus-hybrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --partition=EDIT_ME
##SBATCH --account=EDIT_ME
##SBATCH --mem=0

# Section: shell safety.
# Fail on unset variables and failed commands so a bad module load, failed
# compile, or missing config does not silently launch a wrong job.
set -euo pipefail

# Section: user-overridable paths.
# Defaults prefer the sbatch submission directory when it is the VENUS checkout.
# Slurm may execute a copied script from its spool directory, so relying only
# on BASH_SOURCE can otherwise point VENUS_DIR at /var/spool.  Relative
# CONFIG/OUTPUT_DIR/RESTART_DIR values are interpreted relative to the sbatch
# submission directory; relative EXECUTABLE values are interpreted relative to
# VENUS_DIR.
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
CONFIG="${CONFIG:-$VENUS_DIR/examples/einstein_vlasov_high_compactness_horizon_formation.ini}"
EXECUTABLE="${EXECUTABLE:-$VENUS_DIR/venus_cpu_einstein_vlasov_eulerian_mpi}"
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
# Tested on Apollo with gcc-toolset/10 and openmpi/gcc/4.1.0.  Override the
# module names with sbatch --export if IAS updates the compiler stack.  Set
# VENUS_SKIP_MODULES=1 when the scheduler environment is already prepared.
if command -v module >/dev/null 2>&1 && [ "${VENUS_SKIP_MODULES:-0}" != "1" ]; then
  if [ "${VENUS_PURGE_MODULES:-1}" != "0" ]; then
    module purge
  fi
  module load "${IAS_GCC_MODULE:-gcc-toolset/10}"
  module load "${IAS_MPI_MODULE:-openmpi/gcc/4.1.0}"
fi

# Section: build.
# Hybrid EV/collisionless VP runs need the MPI target.  Each MPI rank owns an
# L-slab and uses OpenMP threads inside that slab.
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
# OMP_NUM_THREADS follows --cpus-per-task.  EV_* variables are read by the C++
# parser and also make the resolved configuration visible in scheduler logs.
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
# Print the paths and resource counts before srun so a failed launch can be
# diagnosed from its scheduler log.
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
# The standalone EV executable uses its output directory for restarts; the modular
# executable accepts it.  This keeps one hybrid template usable for both.
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
# Do not use this template for BP collision operators; use the OpenMP BP
# template, because BP collisions are not distributed across MPI L-slabs yet.
srun --cpu-bind=cores "$EXECUTABLE" \
  "${solver_args[@]}"
