#!/bin/bash
# Launch one spherical Einstein--Vlasov WPIC calculation in Schwarzschild or ingoing EF coordinates.
# Export explicit source/output roots and the physical family controls. Relativistic quantities use G=c=1.
# The script builds the CPU executable, records its exact arguments/inputs, runs it through Slurm and independently validates the saved outcome.

#
# IAS Typhon launcher for the VENUS spherical Einstein--Vlasov WPIC
# high-compactness driver.  It supports both Schwarzschild/polar-areal and
# ingoing Eddington--Finkelstein coordinates through the driver's runtime CLI.

#SBATCH --job-name=venus-ev-wpic
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 027

# The caller exports the deployed repository and simulation-output roots.
# Explicit paths remain valid when Slurm runs this script from its spool.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?Set VENUS_SOURCE_ROOT to the deployed repository path}"
VENUS_RUN_ROOT="${VENUS_RUN_ROOT:?Set VENUS_RUN_ROOT to the simulation output root}"

case "$VENUS_SOURCE_ROOT" in
  /*) ;;
  *) VENUS_SOURCE_ROOT="$SUBMIT_DIR/$VENUS_SOURCE_ROOT" ;;
esac
case "$VENUS_RUN_ROOT" in
  /*) ;;
  *) VENUS_RUN_ROOT="$SUBMIT_DIR/$VENUS_RUN_ROOT" ;;
esac

if [ -f "$VENUS_SOURCE_ROOT/cpu/Makefile" ]; then
  VENUS_CPU_DIR="$VENUS_SOURCE_ROOT/cpu"
elif [ -f "$VENUS_SOURCE_ROOT/Makefile" ] \
     && [ -d "$VENUS_SOURCE_ROOT/include/ev" ]; then
  VENUS_CPU_DIR="$VENUS_SOURCE_ROOT"
  VENUS_SOURCE_ROOT="$(cd "$VENUS_CPU_DIR/.." && pwd)"
else
  echo "VENUS_SOURCE_ROOT does not contain a VENUS cpu/Makefile: $VENUS_SOURCE_ROOT" >&2
  exit 1
fi
VENUS_CPU_DIR="$(cd "$VENUS_CPU_DIR" && pwd)"
mkdir -p "$VENUS_RUN_ROOT"
VENUS_RUN_ROOT="$(cd "$VENUS_RUN_ROOT" && pwd)"

# Separate coordinate choice, fresh initialization and native resume before constructing the solver command.
COORDINATES="${COORDINATES:-schwarzschild}"
EXPECTED_OUTCOME="${EXPECTED_OUTCOME:-any}"
INITIALIZE_ONLY="${INITIALIZE_ONLY:-0}"
RESUME="${RESUME:-0}"
THREADS="${SLURM_CPUS_PER_TASK:-88}"
EXECUTABLE="${EXECUTABLE:-$VENUS_CPU_DIR/venus_cpu_einstein_vlasov_wpic_black_hole}"
OUTPUT_DIR="${OUTPUT_DIR:-$VENUS_RUN_ROOT/${SLURM_JOB_NAME:-venus-ev-wpic}_${SLURM_JOB_ID:-manual}}"
PHYSICAL_IC="${PHYSICAL_IC:-$OUTPUT_DIR/physical_ic.bin}"
VENUS_RUN_ARGS_FILE="${VENUS_RUN_ARGS_FILE:-}"
if [ -n "${VENUS_ANALYZER:-}${VENUS_NULL_MAP_PREFLIGHT:-}${VENUS_POSITIVE_BINDING_VALIDATOR:-}" ]; then
  echo "Workflow-tool overrides are forbidden; validators are pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi
ANALYZER="$VENUS_CPU_DIR/tools/analyze_einstein_vlasov_wpic.py"
NULL_MAP_PREFLIGHT="$VENUS_CPU_DIR/tools/validate_wpic_null_map.py"
POSITIVE_BINDING_VALIDATOR="$VENUS_CPU_DIR/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py"
EXPECTED_LAUNCHER_SHA256="${VENUS_LAUNCHER_SHA256:?set VENUS_LAUNCHER_SHA256}"
EXPECTED_ANALYZER_SHA256="${VENUS_ANALYZER_SHA256:?set VENUS_ANALYZER_SHA256}"
EXPECTED_NULL_MAP_PREFLIGHT_SHA256="${VENUS_NULL_MAP_PREFLIGHT_SHA256:?set VENUS_NULL_MAP_PREFLIGHT_SHA256}"
EXPECTED_POSITIVE_BINDING_VALIDATOR_SHA256="${VENUS_POSITIVE_BINDING_VALIDATOR_SHA256:?set VENUS_POSITIVE_BINDING_VALIDATOR_SHA256}"
VENUS_EXPORT_EF_NULL_INITIAL_DATA="${VENUS_EXPORT_EF_NULL_INITIAL_DATA:-}"
VENUS_EF_NULL_INITIAL_DATA="${VENUS_EF_NULL_INITIAL_DATA:-}"
VENUS_EF_NULL_DONOR_DURATION="${VENUS_EF_NULL_DONOR_DURATION:-}"
VENUS_EF_MAP_WORLDTUBE_RADIUS="${VENUS_EF_MAP_WORLDTUBE_RADIUS:-}"
VENUS_HIGH_R_MAX="${VENUS_HIGH_R_MAX:-}"
VENUS_HIGH_INITIAL_CONDITION="${VENUS_HIGH_INITIAL_CONDITION:-}"
VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE="${VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE:-}"
VENUS_HIGH_ENERGY_CUTOFF_ELL0="${VENUS_HIGH_ENERGY_CUTOFF_ELL0:-}"
VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y="${VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y:-}"
VENUS_HIGH_RADIAL_INFALL="${VENUS_HIGH_RADIAL_INFALL:-}"
VENUS_HIGH_INFALL_CORE_RADIUS="${VENUS_HIGH_INFALL_CORE_RADIUS:-}"
VENUS_HIGH_ANGULAR_MOMENTUM_SCALE="${VENUS_HIGH_ANGULAR_MOMENTUM_SCALE:-}"
VENUS_HIGH_SAMPLING_REALIZATION="${VENUS_HIGH_SAMPLING_REALIZATION:-}"
VENUS_HIGH_TARGET_INITIAL_ADM_MASS="${VENUS_HIGH_TARGET_INITIAL_ADM_MASS:-}"
VENUS_CAMPAIGN_CONTROLS_SHA256="${VENUS_CAMPAIGN_CONTROLS_SHA256:-}"
VENUS_PARITY_CERTIFICATE_SHA256="${VENUS_PARITY_CERTIFICATE_SHA256:-}"
VENUS_REQUIRE_POSITIVE_BINDING="${VENUS_REQUIRE_POSITIVE_BINDING:-0}"

case "$EXECUTABLE" in
  /*) ;;
  *) EXECUTABLE="$VENUS_CPU_DIR/$EXECUTABLE" ;;
esac
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SUBMIT_DIR/$OUTPUT_DIR" ;;
esac
case "$PHYSICAL_IC" in
  /*) ;;
  *) PHYSICAL_IC="$SUBMIT_DIR/$PHYSICAL_IC" ;;
esac
if [ -n "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ]; then
  case "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" in
    /*) ;;
    *) VENUS_EXPORT_EF_NULL_INITIAL_DATA="$SUBMIT_DIR/$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ;;
  esac
fi
if [ -n "$VENUS_EF_NULL_INITIAL_DATA" ]; then
  case "$VENUS_EF_NULL_INITIAL_DATA" in
    /*) ;;
    *) VENUS_EF_NULL_INITIAL_DATA="$SUBMIT_DIR/$VENUS_EF_NULL_INITIAL_DATA" ;;
  esac
fi
if [ -n "$VENUS_RUN_ARGS_FILE" ]; then
  case "$VENUS_RUN_ARGS_FILE" in
    /*) ;;
    *) VENUS_RUN_ARGS_FILE="$SUBMIT_DIR/$VENUS_RUN_ARGS_FILE" ;;
  esac
fi

# The submitted tool hashes bind the launch and scientific validators to the files actually executed.
verify_identity() {
  local label="$1"
  local path="$2"
  local expected="$3"
  if ! [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || [ ! -s "$path" ]; then
    echo "$label identity is missing or malformed: $path" >&2
    exit 1
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    echo "$label SHA-256 changed after submission: expected=$expected actual=$actual" >&2
    exit 1
  fi
}
LAUNCHER_IDENTITY="$VENUS_CPU_DIR/scripts/slurm/ias_einstein_vlasov_wpic.sbatch"
verify_identity launcher "$LAUNCHER_IDENTITY" "$EXPECTED_LAUNCHER_SHA256"
verify_identity analyzer "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
verify_identity null-map-preflight "$NULL_MAP_PREFLIGHT" \
  "$EXPECTED_NULL_MAP_PREFLIGHT_SHA256"
verify_identity positive-binding-validator "$POSITIVE_BINDING_VALIDATOR" \
  "$EXPECTED_POSITIVE_BINDING_VALIDATOR_SHA256"

case "$COORDINATES" in
  schwarzschild|eddington_finkelstein) ;;
  *) echo "COORDINATES must be schwarzschild or eddington_finkelstein." >&2; exit 1 ;;
esac
case "$EXPECTED_OUTCOME" in
  any|halo|black_hole) ;;
  *) echo "EXPECTED_OUTCOME must be any, halo, or black_hole." >&2; exit 1 ;;
esac
case "$INITIALIZE_ONLY" in 0|1) ;; *) echo "INITIALIZE_ONLY must be 0 or 1." >&2; exit 1 ;; esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1." >&2; exit 1 ;; esac
case "$VENUS_REQUIRE_POSITIVE_BINDING" in
  0|1) ;;
  *) echo "VENUS_REQUIRE_POSITIVE_BINDING must be 0 or 1." >&2; exit 1 ;;
esac
if [ "$INITIALIZE_ONLY" = 1 ] && [ "$RESUME" = 1 ]; then
  echo "INITIALIZE_ONLY and RESUME cannot both be enabled." >&2
  exit 1
fi
if [ -n "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ] \
   && { [ "$COORDINATES" != schwarzschild ] \
        || [ "$INITIALIZE_ONLY" = 1 ] || [ "$RESUME" = 1 ]; }; then
  echo "VENUS_EXPORT_EF_NULL_INITIAL_DATA requires a fresh evolving Schwarzschild donor." >&2
  exit 1
fi
if [ -n "$VENUS_EF_NULL_INITIAL_DATA" ] \
   && { [ "$COORDINATES" != eddington_finkelstein ] \
        || [ "$INITIALIZE_ONLY" = 1 ]; }; then
  echo "VENUS_EF_NULL_INITIAL_DATA requires an evolving EF consumer." >&2
  exit 1
fi
# An evolving EF run consumes solved null-slice data so its initial events belong to the declared coordinate map.
if [ "$COORDINATES" = eddington_finkelstein ] \
   && [ "$INITIALIZE_ONLY" != 1 ] \
   && [ -z "$VENUS_EF_NULL_INITIAL_DATA" ]; then
  echo "EF production and resume runs require VENUS_EF_NULL_INITIAL_DATA from their solved eikonal donor." >&2
  exit 1
fi
if [ -n "$VENUS_EF_NULL_DONOR_DURATION" ] \
   && [ -z "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ]; then
  echo "VENUS_EF_NULL_DONOR_DURATION is valid only with a null-data export." >&2
  exit 1
fi
if [ -n "$VENUS_EF_MAP_WORLDTUBE_RADIUS" ] \
   && [ -z "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ] \
   && [ -z "$VENUS_EF_NULL_INITIAL_DATA" ]; then
  echo "VENUS_EF_MAP_WORLDTUBE_RADIUS is valid only for a null-data donor or its EF consumer." >&2
  exit 1
fi
if { [ -n "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ] \
     || [ -n "$VENUS_EF_NULL_INITIAL_DATA" ]; } \
   && [ -z "$VENUS_EF_MAP_WORLDTUBE_RADIUS" ]; then
  echo "A solved-null donor or EF consumer requires an explicit VENUS_EF_MAP_WORLDTUBE_RADIUS=R_B; implicit or unrecorded R_B is forbidden." >&2
  exit 1
fi

positive_number_pattern='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
is_positive_number() {
  local value="$1"
  [[ "$value" =~ $positive_number_pattern ]] \
    && awk -v number="$value" 'BEGIN { exit !(number > 0.0) }'
}
is_nonnegative_number() {
  local value="$1"
  [[ "$value" =~ $positive_number_pattern ]] \
    && awk -v number="$value" 'BEGIN { exit !(number >= 0.0) }'
}

if [ -n "$VENUS_HIGH_INITIAL_CONDITION" ]; then
  case "$VENUS_HIGH_INITIAL_CONDITION" in
    shell|plummer|energy_cutoff_shell) ;;
    *) echo "Invalid VENUS_HIGH_INITIAL_CONDITION=$VENUS_HIGH_INITIAL_CONDITION" >&2; exit 1 ;;
  esac
fi
# Require all physical distribution and perturbation controls explicitly for this equilibrium family.
if [ "$VENUS_HIGH_INITIAL_CONDITION" = energy_cutoff_shell ]; then
  for control_name in \
      VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE \
      VENUS_HIGH_ENERGY_CUTOFF_ELL0 \
      VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y \
      VENUS_HIGH_RADIAL_INFALL \
      VENUS_HIGH_INFALL_CORE_RADIUS \
      VENUS_HIGH_ANGULAR_MOMENTUM_SCALE \
      VENUS_HIGH_SAMPLING_REALIZATION \
      VENUS_HIGH_TARGET_INITIAL_ADM_MASS \
      VENUS_HIGH_R_MAX; do
    if [ -z "${!control_name}" ]; then
      echo "energy_cutoff_shell requires explicit $control_name." >&2
      exit 1
    fi
  done
  for control_name in \
      VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE \
      VENUS_HIGH_ENERGY_CUTOFF_ELL0 \
      VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y \
      VENUS_HIGH_TARGET_INITIAL_ADM_MASS \
      VENUS_HIGH_R_MAX; do
    if ! is_positive_number "${!control_name}"; then
      echo "$control_name must be a finite positive decimal." >&2
      exit 1
    fi
  done
  for control_name in VENUS_HIGH_RADIAL_INFALL \
      VENUS_HIGH_INFALL_CORE_RADIUS; do
    if ! is_nonnegative_number "${!control_name}"; then
      echo "$control_name must be a finite non-negative decimal." >&2
      exit 1
    fi
  done
  if ! is_positive_number "$VENUS_HIGH_ANGULAR_MOMENTUM_SCALE"; then
    echo "VENUS_HIGH_ANGULAR_MOMENTUM_SCALE must be a finite strictly positive decimal; s_L=0 is outside the advanced-null production domain." >&2
    exit 1
  fi
  if ! [[ "$VENUS_HIGH_SAMPLING_REALIZATION" =~ ^[0-9]+$ ]]; then
    echo "VENUS_HIGH_SAMPLING_REALIZATION must be an unsigned integer." >&2
    exit 1
  fi
  if [ -n "${VENUS_HIGH_SPECIFIC_ANGULAR_MOMENTUM:-}" ]; then
    echo "energy_cutoff_shell gets ell0 from VENUS_HIGH_ENERGY_CUTOFF_ELL0; VENUS_HIGH_SPECIFIC_ANGULAR_MOMENTUM is unsupported." >&2
    exit 1
  fi
fi
if [ "$VENUS_REQUIRE_POSITIVE_BINDING" = 1 ] \
   && [ "$VENUS_HIGH_INITIAL_CONDITION" != energy_cutoff_shell ]; then
  echo "VENUS_REQUIRE_POSITIVE_BINDING=1 requires energy_cutoff_shell diagnostics." >&2
  exit 1
fi
# The map worldtube lies inside the larger evolution domain, whose exterior is supplied by the analytic vacuum extension.
if [ -n "$VENUS_EF_MAP_WORLDTUBE_RADIUS" ]; then
  if ! is_positive_number "$VENUS_EF_MAP_WORLDTUBE_RADIUS" \
     || [ -z "$VENUS_HIGH_R_MAX" ] \
     || ! is_positive_number "$VENUS_HIGH_R_MAX" \
     || ! awk -v rb="$VENUS_EF_MAP_WORLDTUBE_RADIUS" \
          -v re="$VENUS_HIGH_R_MAX" 'BEGIN { exit !(rb < re) }'; then
    echo "The null map requires finite 0 < R_B < R_E via VENUS_EF_MAP_WORLDTUBE_RADIUS and VENUS_HIGH_R_MAX." >&2
    exit 1
  fi
fi
for digest_name in VENUS_CAMPAIGN_CONTROLS_SHA256 \
    VENUS_PARITY_CERTIFICATE_SHA256; do
  if [ -n "${!digest_name}" ] \
     && ! [[ "${!digest_name}" =~ ^[[:xdigit:]]{64}$ ]]; then
    echo "$digest_name must be a 64-character SHA-256 digest." >&2
    exit 1
  fi
done
if [ -e "$OUTPUT_DIR" ] && [ "$RESUME" != 1 ] \
   && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "Refusing to overwrite non-empty OUTPUT_DIR without RESUME=1: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR" "$(dirname "$PHYSICAL_IC")"
if [ -n "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ]; then
  mkdir -p "$(dirname "$VENUS_EXPORT_EF_NULL_INITIAL_DATA")"
fi

# gcc-toolset/10 is the maintained IAS build environment for this campaign.
if [ "${VENUS_SKIP_MODULES:-0}" != 1 ]; then
  if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  if ! command -v module >/dev/null 2>&1; then
    echo "The IAS module command is unavailable; set VENUS_SKIP_MODULES=1 only after configuring gcc-toolset/10 manually." >&2
    exit 1
  fi
  if [ "${VENUS_PURGE_MODULES:-1}" != 0 ]; then
    module purge
  fi
  module load "${IAS_GCC_MODULE:-gcc-toolset/10}"
fi

# Serialize builds from a shared checkout.  This matters for a coarse scan in
# which many independent Slurm jobs can begin on different Typhon nodes.
if [ "${VENUS_BUILD:-1}" != 0 ]; then
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to protect concurrent builds of the shared WPIC binary." >&2
    exit 1
  fi
  # Use a source-directory-specific lock so concurrent jobs cannot overwrite a shared executable while another build is active.
  build_key="$(printf '%s' "$VENUS_CPU_DIR" | sha256sum | cut -c1-16)"
  mkdir -p "$VENUS_RUN_ROOT/.build-locks"
  exec 9>"$VENUS_RUN_ROOT/.build-locks/$build_key.lock"
  flock 9
  make -C "$VENUS_CPU_DIR" \
    -j "${VENUS_MAKE_JOBS:-$THREADS}" \
    CXX="${CXX:-g++}" \
    venus_cpu_einstein_vlasov_wpic_black_hole
  flock -u 9
fi
if [ ! -x "$EXECUTABLE" ]; then
  echo "WPIC executable not found or not executable: $EXECUTABLE" >&2
  exit 1
fi
if [ ! -f "$ANALYZER" ]; then
  echo "VENUS analysis/validation tool not found: $ANALYZER" >&2
  exit 1
fi
if [ ! -f "$NULL_MAP_PREFLIGHT" ]; then
  echo "Null-map setup validator not found: $NULL_MAP_PREFLIGHT" >&2
  exit 1
fi
if [ ! -f "$POSITIVE_BINDING_VALIDATOR" ]; then
  echo "Positive-binding validator not found: $POSITIVE_BINDING_VALIDATOR" >&2
  exit 1
fi

# The optional argument file uses one literal argv element per non-blank line.
# This avoids shell-eval ambiguity and makes the exact launch reconstructible.
extra_args=()
if [ -n "$VENUS_RUN_ARGS_FILE" ]; then
  if [ ! -f "$VENUS_RUN_ARGS_FILE" ]; then
    echo "VENUS_RUN_ARGS_FILE not found: $VENUS_RUN_ARGS_FILE" >&2
    exit 1
  fi
  while IFS= read -r argument || [ -n "$argument" ]; do
    argument="${argument%$'\r'}"
    case "$argument" in
      ''|'#'*) continue ;;
      --scenario|--coordinates|--spacetime-coordinates|--expected-outcome|--output-dir|--physical-ic|--threads|--initialize-only|--resume|--export-ef-null-initial-data|--ef-null-initial-data|--ef-null-donor-duration|--ef-map-worldtube-radius|--high-r-max|--high-initial-condition|--high-energy-cutoff-amplitude|--high-energy-cutoff-ell0|--high-energy-cutoff-central-y|--high-radial-infall|--high-infall-core-radius|--high-specific-angular-momentum|--high-angular-momentum-scale|--high-sampling-realization|--high-target-initial-adm-mass)
        echo "The launcher manages $argument; remove it from $VENUS_RUN_ARGS_FILE." >&2
        exit 1
        ;;
    esac
    extra_args+=("$argument")
  done < "$VENUS_RUN_ARGS_FILE"
fi

# Store arguments in a Bash array so each physical value remains one literal argv element.
solver_args=(
  --coordinates "$COORDINATES"
  --expected-outcome "$EXPECTED_OUTCOME"
  --output-dir "$OUTPUT_DIR"
  --physical-ic "$PHYSICAL_IC"
  --threads "$THREADS"
)
solver_args+=("${extra_args[@]}")
if [ -n "$VENUS_HIGH_INITIAL_CONDITION" ]; then
  solver_args+=(--high-initial-condition "$VENUS_HIGH_INITIAL_CONDITION")
fi
if [ -n "$VENUS_HIGH_R_MAX" ]; then
  solver_args+=(--high-r-max "$VENUS_HIGH_R_MAX")
fi
if [ -n "$VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE" ]; then
  solver_args+=(
    --high-energy-cutoff-amplitude "$VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE"
  )
fi
if [ -n "$VENUS_HIGH_ENERGY_CUTOFF_ELL0" ]; then
  solver_args+=(--high-energy-cutoff-ell0 "$VENUS_HIGH_ENERGY_CUTOFF_ELL0")
fi
if [ -n "$VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y" ]; then
  solver_args+=(
    --high-energy-cutoff-central-y "$VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y"
  )
fi
if [ -n "$VENUS_HIGH_RADIAL_INFALL" ]; then
  solver_args+=(--high-radial-infall "$VENUS_HIGH_RADIAL_INFALL")
fi
if [ -n "$VENUS_HIGH_INFALL_CORE_RADIUS" ]; then
  solver_args+=(--high-infall-core-radius "$VENUS_HIGH_INFALL_CORE_RADIUS")
fi
if [ -n "${VENUS_HIGH_SPECIFIC_ANGULAR_MOMENTUM:-}" ]; then
  solver_args+=(--high-specific-angular-momentum "$VENUS_HIGH_SPECIFIC_ANGULAR_MOMENTUM")
fi
if [ -n "$VENUS_HIGH_ANGULAR_MOMENTUM_SCALE" ]; then
  solver_args+=(--high-angular-momentum-scale "$VENUS_HIGH_ANGULAR_MOMENTUM_SCALE")
fi
if [ -n "$VENUS_HIGH_SAMPLING_REALIZATION" ]; then
  if ! [[ "$VENUS_HIGH_SAMPLING_REALIZATION" =~ ^[0-9]+$ ]]; then
    echo "VENUS_HIGH_SAMPLING_REALIZATION must be an unsigned integer." >&2
    exit 1
  fi
  solver_args+=(--high-sampling-realization "$VENUS_HIGH_SAMPLING_REALIZATION")
fi
if [ -n "$VENUS_HIGH_TARGET_INITIAL_ADM_MASS" ]; then
  if ! [[ "$VENUS_HIGH_TARGET_INITIAL_ADM_MASS" \
          =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
     || ! awk -v number="$VENUS_HIGH_TARGET_INITIAL_ADM_MASS" \
          'BEGIN { exit !(number > 0.0) }'; then
    echo "VENUS_HIGH_TARGET_INITIAL_ADM_MASS must be positive." >&2
    exit 1
  fi
  solver_args+=(
    --high-target-initial-adm-mass "$VENUS_HIGH_TARGET_INITIAL_ADM_MASS"
  )
fi
if [ -n "${VENUS_EF_EXCISION_A:-}" ]; then
  solver_args+=(--ef-excision-a "$VENUS_EF_EXCISION_A")
fi
if [ -n "${VENUS_EF_EXCISION_BUFFER_CELLS:-}" ]; then
  solver_args+=(--ef-excision-buffer-cells "$VENUS_EF_EXCISION_BUFFER_CELLS")
fi
solver_args+=(--ef-excision-horizon-fraction "${VENUS_EF_EXCISION_HORIZON_FRACTION:-0.1}")
if [ -n "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ]; then
  solver_args+=(
    --export-ef-null-initial-data "$VENUS_EXPORT_EF_NULL_INITIAL_DATA"
  )
  if [ -n "$VENUS_EF_NULL_DONOR_DURATION" ]; then
    solver_args+=(--ef-null-donor-duration "$VENUS_EF_NULL_DONOR_DURATION")
  fi
  solver_args+=(--ef-map-worldtube-radius "$VENUS_EF_MAP_WORLDTUBE_RADIUS")
fi
if [ -n "$VENUS_EF_NULL_INITIAL_DATA" ]; then
  solver_args+=(--ef-null-initial-data "$VENUS_EF_NULL_INITIAL_DATA")
fi
if [ "$INITIALIZE_ONLY" = 1 ]; then
  solver_args+=(--initialize-only)
fi
if [ "$RESUME" = 1 ]; then
  solver_args+=(--resume)
fi

# This launcher-level manifest records the exact family controls supplied to
# the executable.  Values are deliberately retained as strings: the resolved
# command and this manifest together distinguish, for example, an explicitly
# requested zero kick from an omitted control.  The solver summary and null-
# map manifest independently record the realized continuum and map geometry.
# Keep supplied family controls as strings to distinguish an explicitly specified zero from an omitted option.
RUN_CONTROLS_MANIFEST="$OUTPUT_DIR/workflow_controls.manifest.tsv"
{
  printf 'schema\tvenus_v3_wpic_workflow_controls\n'
  printf 'schema_version\t2\n'
  printf 'coordinates\t%s\n' "$COORDINATES"
  printf 'expected_outcome\t%s\n' "$EXPECTED_OUTCOME"
  printf 'initial_condition\t%s\n' "${VENUS_HIGH_INITIAL_CONDITION:--}"
  printf 'energy_cutoff_amplitude_C\t%s\n' "${VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE:--}"
  printf 'energy_cutoff_ell0\t%s\n' "${VENUS_HIGH_ENERGY_CUTOFF_ELL0:--}"
  printf 'energy_cutoff_central_y\t%s\n' "${VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y:--}"
  printf 'inward_radial_kick\t%s\n' "${VENUS_HIGH_RADIAL_INFALL:--}"
  printf 'infall_core_radius\t%s\n' "${VENUS_HIGH_INFALL_CORE_RADIUS:--}"
  printf 'angular_momentum_scale\t%s\n' "${VENUS_HIGH_ANGULAR_MOMENTUM_SCALE:--}"
  printf 'target_initial_adm_mass\t%s\n' "${VENUS_HIGH_TARGET_INITIAL_ADM_MASS:--}"
  printf 'sampling_realization\t%s\n' "${VENUS_HIGH_SAMPLING_REALIZATION:--}"
  printf 'map_worldtube_radius_R_B\t%s\n' "${VENUS_EF_MAP_WORLDTUBE_RADIUS:--}"
  printf 'evolution_outer_radius_R_E\t%s\n' "${VENUS_HIGH_R_MAX:--}"
  printf 'ef_null_donor_duration\t%s\n' "${VENUS_EF_NULL_DONOR_DURATION:--}"
  printf 'campaign_controls_sha256\t%s\n' "${VENUS_CAMPAIGN_CONTROLS_SHA256:--}"
  printf 'parity_certificate_sha256\t%s\n' "${VENUS_PARITY_CERTIFICATE_SHA256:--}"
  printf 'require_positive_binding\t%s\n' "$VENUS_REQUIRE_POSITIVE_BINDING"
  printf 'launcher_sha256\t%s\n' "$EXPECTED_LAUNCHER_SHA256"
  printf 'analyzer_sha256\t%s\n' "$EXPECTED_ANALYZER_SHA256"
  printf 'null_map_preflight_sha256\t%s\n' "$EXPECTED_NULL_MAP_PREFLIGHT_SHA256"
  printf 'positive_binding_validator_sha256\t%s\n' \
    "$EXPECTED_POSITIVE_BINDING_VALIDATOR_SHA256"
  printf 'physical_ic\t%s\n' "$PHYSICAL_IC"
  printf 'null_map_export\t%s\n' "${VENUS_EXPORT_EF_NULL_INITIAL_DATA:--}"
  printf 'null_map_import\t%s\n' "${VENUS_EF_NULL_INITIAL_DATA:--}"
} > "$RUN_CONTROLS_MANIFEST"
sha256sum "$RUN_CONTROLS_MANIFEST" \
  > "$OUTPUT_DIR/workflow_controls.manifest.sha256"

# Capture the precise source and binary used by the scheduled calculation.
(
  cd "$VENUS_CPU_DIR"
  {
    printf '%s\0' Makefile
    find include src -type f -print0
    printf '%s\0' scripts/slurm/ias_einstein_vlasov_wpic.sbatch
    printf '%s\0' scripts/slurm/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch
    printf '%s\0' scripts/slurm/ias_einstein_vlasov_wpic_parity_convergence.sbatch
    printf '%s\0' scripts/slurm/ias_einstein_vlasov_wpic_parity_postprocess.sbatch
    printf '%s\0' scripts/slurm/submit_einstein_vlasov_wpic_parity.sh
    printf '%s\0' scripts/slurm/submit_einstein_vlasov_wpic_angular_scan.sh
    printf '%s\0' scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh
    printf '%s\0' scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gate.sbatch
    printf '%s\0' scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py
    printf '%s\0' scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py
    printf '%s\0' tools/analyze_einstein_vlasov_wpic.py
    printf '%s\0' tools/compare_wpic_dynamic_coordinates.py
    printf '%s\0' tools/build_wpic_dynamic_coordinate_convergence.py
    printf '%s\0' tools/validate_wpic_null_map.py
  } | sort -z | xargs -0 sha256sum
) > "$OUTPUT_DIR/source.sha256"
sha256sum "$EXECUTABLE" > "$OUTPUT_DIR/binary.sha256"
if [ -n "$VENUS_RUN_ARGS_FILE" ]; then
  cp "$VENUS_RUN_ARGS_FILE" "$OUTPUT_DIR/driver_arguments.txt"
  sha256sum "$VENUS_RUN_ARGS_FILE" > "$OUTPUT_DIR/driver_arguments.sha256"
fi
{
  printf '%q ' "$EXECUTABLE"
  printf '%q ' "${solver_args[@]}"
  printf '\n'
} > "$OUTPUT_DIR/resolved_command.sh"

echo "VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT"
echo "VENUS_CPU_DIR=$VENUS_CPU_DIR"
echo "VENUS_RUN_ROOT=$VENUS_RUN_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PHYSICAL_IC=$PHYSICAL_IC"
echo "VENUS_EXPORT_EF_NULL_INITIAL_DATA=$VENUS_EXPORT_EF_NULL_INITIAL_DATA"
echo "VENUS_EF_NULL_INITIAL_DATA=$VENUS_EF_NULL_INITIAL_DATA"
echo "VENUS_EF_MAP_WORLDTUBE_RADIUS=$VENUS_EF_MAP_WORLDTUBE_RADIUS"
echo "VENUS_HIGH_R_MAX=$VENUS_HIGH_R_MAX"
echo "VENUS_HIGH_INITIAL_CONDITION=$VENUS_HIGH_INITIAL_CONDITION"
echo "VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=$VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE"
echo "VENUS_HIGH_ENERGY_CUTOFF_ELL0=$VENUS_HIGH_ENERGY_CUTOFF_ELL0"
echo "VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=$VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y"
echo "VENUS_HIGH_RADIAL_INFALL=$VENUS_HIGH_RADIAL_INFALL"
echo "VENUS_HIGH_INFALL_CORE_RADIUS=$VENUS_HIGH_INFALL_CORE_RADIUS"
echo "VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=$VENUS_HIGH_ANGULAR_MOMENTUM_SCALE"
echo "VENUS_HIGH_SAMPLING_REALIZATION=$VENUS_HIGH_SAMPLING_REALIZATION"
echo "VENUS_HIGH_TARGET_INITIAL_ADM_MASS=$VENUS_HIGH_TARGET_INITIAL_ADM_MASS"
echo "VENUS_REQUIRE_POSITIVE_BINDING=$VENUS_REQUIRE_POSITIVE_BINDING"
echo "COORDINATES=$COORDINATES"
echo "EXPECTED_OUTCOME=$EXPECTED_OUTCOME"
echo "THREADS=$THREADS"
echo "EXECUTABLE=$EXECUTABLE"

# Capture the solver and log-writer statuses separately while preserving the full combined solver output.
set +e
srun --ntasks=1 --cpus-per-task="$THREADS" --cpu-bind=cores \
  "$EXECUTABLE" "${solver_args[@]}" 2>&1 | tee "$OUTPUT_DIR/solver.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
solver_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
printf '%s\n' "$solver_status" > "$OUTPUT_DIR/solver_exit_status.txt"
if [ "$tee_status" -ne 0 ]; then
  echo "tee failed while writing $OUTPUT_DIR/solver.log" >&2
  exit "$tee_status"
fi
if [ -f "$PHYSICAL_IC" ]; then
  sha256sum "$PHYSICAL_IC" > "$OUTPUT_DIR/physical_ic.sha256"
fi
# Positive binding is checked from numerical rest/ADM values when that physical family condition was required.
binding_status=0
if [ "$VENUS_REQUIRE_POSITIVE_BINDING" = 1 ]; then
  set +e
  python3 "$POSITIVE_BINDING_VALIDATOR" \
    --summary "$OUTPUT_DIR/summary.json" \
    --output "$OUTPUT_DIR/positive_binding_validation.json"
  binding_status=$?
  set -e
fi
# Authenticate all mapping outputs used by either a donor or an EF consumer before validating the final run.
coordinate_map_artifacts=()
if [ -n "$VENUS_EXPORT_EF_NULL_INITIAL_DATA" ]; then
  coordinate_map_artifacts+=(
    "$VENUS_EXPORT_EF_NULL_INITIAL_DATA"
    "$VENUS_EXPORT_EF_NULL_INITIAL_DATA.manifest.json"
    "$VENUS_EXPORT_EF_NULL_INITIAL_DATA.crossings.csv"
    "$VENUS_EXPORT_EF_NULL_INITIAL_DATA.slice.csv"
    "$VENUS_EXPORT_EF_NULL_INITIAL_DATA.accepted_steps.csv"
    "$VENUS_EXPORT_EF_NULL_INITIAL_DATA.donor_history.bin"
  )
fi
if [ -n "$VENUS_EF_NULL_INITIAL_DATA" ]; then
  coordinate_map_artifacts+=(
    "$VENUS_EF_NULL_INITIAL_DATA"
    "$VENUS_EF_NULL_INITIAL_DATA.manifest.json"
    "$VENUS_EF_NULL_INITIAL_DATA.crossings.csv"
    "$VENUS_EF_NULL_INITIAL_DATA.slice.csv"
    "$VENUS_EF_NULL_INITIAL_DATA.accepted_steps.csv"
    "$VENUS_EF_NULL_INITIAL_DATA.donor_history.bin"
  )
fi
if [ "${#coordinate_map_artifacts[@]}" -gt 0 ]; then
  for artifact in "${coordinate_map_artifacts[@]}"; do
    if [ ! -s "$artifact" ]; then
      echo "Required solved-eikonal artifact is missing or empty: $artifact" >&2
      exit 2
    fi
  done
  sha256sum "${coordinate_map_artifacts[@]}" \
    > "$OUTPUT_DIR/coordinate_map_artifacts.sha256"
  map_base="$VENUS_EXPORT_EF_NULL_INITIAL_DATA"
  if [ -z "$map_base" ]; then
    map_base="$VENUS_EF_NULL_INITIAL_DATA"
  fi
  python3 "$NULL_MAP_PREFLIGHT" \
    --manifest "$map_base.manifest.json" \
    --summary "$OUTPUT_DIR/summary.json" \
    --output "$OUTPUT_DIR/null_map_preflight.json"
  if [ -n "$VENUS_EF_MAP_WORLDTUBE_RADIUS" ]; then
    python3 - "$map_base.manifest.json" \
      "$VENUS_EF_MAP_WORLDTUBE_RADIUS" "$VENUS_HIGH_R_MAX" <<'PY'
import json
import math
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
requested_rb = float(sys.argv[2])
requested_re = float(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def same(left, right):
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=2.0e-13, abs_tol=2.0e-13 * max(1.0, abs(right)))

actual_rb = float(manifest.get("map_worldtube_radius", math.nan))
actual_re = float(manifest.get("evolution_outer_radius", math.nan))
if not same(actual_rb, requested_rb) or not same(actual_re, requested_re):
    raise SystemExit(
        "null-map manifest does not match requested R_B/R_E: "
        f"manifest=({actual_rb:.17g},{actual_re:.17g}) "
        f"requested=({requested_rb:.17g},{requested_re:.17g})"
    )
PY
  fi
fi

# Validate the machine-readable outcome independently of process status.  The
# driver honors --expected-outcome; an independently validated status 2 is
# accepted by this launcher. Fatal status 1 and malformed or non-conservative
# diagnostics are failed.
# Classify initialization or evolution using saved physical diagnostics as well as the process exit status.
validation_expectation="$EXPECTED_OUTCOME"
if [ "$INITIALIZE_ONLY" = 1 ]; then
  validation_expectation=initialized
fi
set +e
python3 "$ANALYZER" validate-run \
  --run-dir "$OUTPUT_DIR" \
  --expected "$validation_expectation"
validation_status=$?
set -e

# A recognized scientific outcome requires both independent run validation and any required binding audit to succeed.
workflow_status=failed
if [ "$solver_status" -eq 0 ] && [ "$validation_status" -eq 0 ] \
   && [ "$binding_status" -eq 0 ]; then
  workflow_status=success
elif [ "$solver_status" -eq 2 ] && [ "$validation_status" -eq 0 ] \
     && [ "$binding_status" -eq 0 ]; then
  workflow_status=scientific_outcome_normalized
fi
verify_identity launcher "$LAUNCHER_IDENTITY" "$EXPECTED_LAUNCHER_SHA256"
verify_identity analyzer "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
verify_identity null-map-preflight "$NULL_MAP_PREFLIGHT" \
  "$EXPECTED_NULL_MAP_PREFLIGHT_SHA256"
verify_identity positive-binding-validator "$POSITIVE_BINDING_VALIDATOR" \
  "$EXPECTED_POSITIVE_BINDING_VALIDATOR_SHA256"
{
  printf 'workflow_status=%s\n' "$workflow_status"
  printf 'solver_exit_status=%s\n' "$solver_status"
  printf 'validation_exit_status=%s\n' "$validation_status"
  printf 'positive_binding_required=%s\n' "$VENUS_REQUIRE_POSITIVE_BINDING"
  printf 'positive_binding_validation_exit_status=%s\n' "$binding_status"
} > "$OUTPUT_DIR/workflow_status.txt"

if [ "$workflow_status" = failed ]; then
  echo "VENUS run failed validation (solver=$solver_status, validator=$validation_status, positive_binding=$binding_status)." >&2
  if [ "$solver_status" -ne 0 ] && [ "$solver_status" -ne 2 ]; then
    exit "$solver_status"
  fi
  exit 2
fi
echo "VENUS run completed: $workflow_status"
