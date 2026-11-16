#!/bin/bash
# Schedule the certificate-checking entry for one longer Schwarzschild/EF pair.
# Explicit exports specify distribution, physical realization, numerical resolution, map radii and end time in G=c=1 units.
# The initial submission creates a fresh request; its dependent gate verifies the completed certificate before scheduling evolution.

# Queue one authenticated post-certificate Schwarzschild/EF long-run pair.
#
# The helper itself never guesses a physical or numerical control.  It queues
# one small gate job after the supplied strict-certificate job.  Only that gate
# may authenticate the newly written certificate and submit the two evolutions.

set -euo pipefail
umask 027

SUBMIT_DIR="$(pwd)"
if [ -n "${GATE_SCRIPT:-}${SBATCH_SCRIPT:-}${VENUS_ANALYZER:-}${VENUS_THRESHOLD_PROVENANCE_VALIDATOR:-}" ]; then
  echo "Paired-long workflow overrides are forbidden; scripts and tools are pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi

VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?Set VENUS_SOURCE_ROOT to the deployed repository path}"
VENUS_PAIRED_LONG_WORKFLOW_ROOT="${VENUS_PAIRED_LONG_WORKFLOW_ROOT:?set the separate immutable paired-long workflow root}"
VENUS_RUN_ROOT="${VENUS_RUN_ROOT:?Set VENUS_RUN_ROOT to the simulation output root}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?set a new paired-long CAMPAIGN_ROOT}"
PARITY_CERTIFICATE_JOB_ID="${PARITY_CERTIFICATE_JOB_ID:?set the terminal strict-certificate Slurm job id}"
PARITY_CERTIFICATE="${PARITY_CERTIFICATE:?set the future/existing strict certificate path}"
PARITY_FINE_PHYSICAL_IC="${PARITY_FINE_PHYSICAL_IC:?set the certified fine-A physical_ic.bin}"
PARITY_FINE_NULL_INITIAL_DATA="${PARITY_FINE_NULL_INITIAL_DATA:?set the certified fine-A null initial data}"

HIGH_INITIAL_CONDITION="${HIGH_INITIAL_CONDITION:?set HIGH_INITIAL_CONDITION=energy_cutoff_shell}"
ENERGY_CUTOFF_AMPLITUDE="${ENERGY_CUTOFF_AMPLITUDE:?set continuum amplitude C}"
ENERGY_CUTOFF_ELL0="${ENERGY_CUTOFF_ELL0:?set continuum ell0}"
ENERGY_CUTOFF_CENTRAL_Y="${ENERGY_CUTOFF_CENTRAL_Y:?set continuum central y}"
RADIAL_INFALL="${RADIAL_INFALL:?set the fixed inward kick, including explicit 0}"
INFALL_CORE_RADIUS="${INFALL_CORE_RADIUS:?set the fixed kick core radius, including explicit 0}"
ANGULAR_MOMENTUM_SCALE="${ANGULAR_MOMENTUM_SCALE:?set the certified fine-A s_L}"
TARGET_INITIAL_ADM_MASS="${TARGET_INITIAL_ADM_MASS:?set the fixed ADM mass}"
SAMPLING_REALIZATION="${SAMPLING_REALIZATION:?set the certified fine-A realization}"
LONG_MARKERS="${LONG_MARKERS:?set the certified fine-A marker count}"
LONG_RADIAL_CELLS="${LONG_RADIAL_CELLS:?set the certified fine-A radial cells}"
LONG_DT_MAX="${LONG_DT_MAX:?set the certified fine-A dt_max}"
LONG_R_MAX="${LONG_R_MAX:?set the certified evolution radius R_E}"
LONG_T_END="${LONG_T_END:?set the long-run end time}"
LONG_OUTPUT_STEPS="${LONG_OUTPUT_STEPS:?set the saved-output cadence in steps}"
EF_MAP_WORLDTUBE_RADIUS="${EF_MAP_WORLDTUBE_RADIUS:?set the certified R_B}"
EF_NULL_DONOR_DURATION="${EF_NULL_DONOR_DURATION:?set the certified donor duration}"
VENUS_EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set the certified eikonal substep count}"

# Resolve relative input/output paths before recording them so Slurm spool execution has the same interpretation.
for variable in VENUS_SOURCE_ROOT VENUS_PAIRED_LONG_WORKFLOW_ROOT \
                VENUS_RUN_ROOT CAMPAIGN_ROOT \
                PARITY_CERTIFICATE PARITY_FINE_PHYSICAL_IC \
                PARITY_FINE_NULL_INITIAL_DATA; do
  value="${!variable}"
  case "$value" in /*) ;; *) printf -v "$variable" '%s/%s' "$SUBMIT_DIR" "$value" ;; esac
done

GATE_SCRIPT="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gate.sbatch"
SBATCH_SCRIPT="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic.sbatch"
SUBMITTER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh"
GIF_SUBMITTER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh"
GIF_WRAPPER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
GIF_RENDERER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py"
ANALYZER="$VENUS_SOURCE_ROOT/cpu/tools/analyze_einstein_vlasov_wpic.py"
PROVENANCE_VALIDATOR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py"
NULL_MAP_PREFLIGHT="$VENUS_SOURCE_ROOT/cpu/tools/validate_wpic_null_map.py"
POSITIVE_BINDING_VALIDATOR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py"
EXECUTABLE="$VENUS_SOURCE_ROOT/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"

for path in "$GATE_SCRIPT" "$SBATCH_SCRIPT" "$SUBMITTER" \
            "$GIF_SUBMITTER" "$GIF_WRAPPER" "$GIF_RENDERER" "$ANALYZER" \
            "$PROVENANCE_VALIDATOR" "$NULL_MAP_PREFLIGHT" \
            "$POSITIVE_BINDING_VALIDATOR"; do
  if [ ! -s "$path" ]; then
    echo "Required paired-long workflow file is missing or empty: $path" >&2
    exit 1
  fi
done
if [ ! -f "$VENUS_SOURCE_ROOT/cpu/Makefile" ]; then
  echo "VENUS_SOURCE_ROOT is not a deployed VENUS checkout: $VENUS_SOURCE_ROOT" >&2
  exit 1
fi
science_root_real="$(cd "$VENUS_SOURCE_ROOT" && pwd -P)"
workflow_root_real="$(cd "$VENUS_PAIRED_LONG_WORKFLOW_ROOT" && pwd -P)"
case "$science_root_real/" in
  "$workflow_root_real/"*)
    echo "The paired-long workflow root must be separate from the science source root." >&2
    exit 1
    ;;
esac
case "$workflow_root_real/" in
  "$science_root_real/"*)
    echo "The paired-long workflow root must not be inside the science source root." >&2
    exit 1
    ;;
esac
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on an IAS login node." >&2
  exit 1
fi
if [ -e "$CAMPAIGN_ROOT" ]; then
  echo "Refusing to reuse an existing paired-long campaign: $CAMPAIGN_ROOT" >&2
  exit 1
fi
if ! [[ "$PARITY_CERTIFICATE_JOB_ID" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARITY_CERTIFICATE_JOB_ID must be one positive Slurm job id." >&2
  exit 1
fi
if [ "$HIGH_INITIAL_CONDITION" != energy_cutoff_shell ]; then
  echo "Paired-long production requires HIGH_INITIAL_CONDITION=energy_cutoff_shell." >&2
  exit 1
fi

# Validate physical ranges and integer resolutions before creating any campaign output or requesting a job.
positive='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
for assignment in \
    "ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE" \
    "ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0" \
    "ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y" \
    "ANGULAR_MOMENTUM_SCALE=$ANGULAR_MOMENTUM_SCALE" \
    "TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS" \
    "LONG_DT_MAX=$LONG_DT_MAX" "LONG_R_MAX=$LONG_R_MAX" \
    "LONG_T_END=$LONG_T_END" \
    "EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS" \
    "EF_NULL_DONOR_DURATION=$EF_NULL_DONOR_DURATION"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ $positive ]] \
     || ! awk -v number="$value" 'BEGIN { exit !(number > 0.0) }'; then
    echo "$name must be finite and positive: $value" >&2
    exit 1
  fi
done
for assignment in "RADIAL_INFALL=$RADIAL_INFALL" \
                  "INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ $positive ]] \
     || ! awk -v number="$value" 'BEGIN { exit !(number >= 0.0) }'; then
    echo "$name must be finite and non-negative: $value" >&2
    exit 1
  fi
done
for assignment in "SAMPLING_REALIZATION=$SAMPLING_REALIZATION" \
                  "LONG_MARKERS=$LONG_MARKERS" \
                  "LONG_RADIAL_CELLS=$LONG_RADIAL_CELLS" \
                  "LONG_OUTPUT_STEPS=$LONG_OUTPUT_STEPS" \
                  "VENUS_EIKONAL_SUBSTEPS=$VENUS_EIKONAL_SUBSTEPS"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] \
     || { [ "$name" != SAMPLING_REALIZATION ] && [ "$value" -lt 1 ]; }; then
    echo "$name must be an appropriate unsigned integer: $value" >&2
    exit 1
  fi
done
if [ $((LONG_MARKERS % 2)) -ne 0 ]; then
  echo "energy_cutoff_shell requires an even LONG_MARKERS count." >&2
  exit 1
fi
if [ "$VENUS_EIKONAL_SUBSTEPS" -ne 8 ]; then
  echo "This production workflow requires VENUS_EIKONAL_SUBSTEPS=8." >&2
  exit 1
fi
# Require the map worldtube on an actual mesh face and leave complete exterior cells for the analytic vacuum extension.
if ! awk -v rb="$EF_MAP_WORLDTUBE_RADIUS" -v re="$LONG_R_MAX" \
      -v cells="$LONG_RADIAL_CELLS" '
    BEGIN {
      if (!(rb > 0 && rb < re)) exit 1;
      x = rb * cells / re;
      nearest = int(x + 0.5);
      error = x-nearest < 0 ? nearest-x : x-nearest;
      if (error > 2.0e-12 * (x > 1 ? x : 1)) exit 1;
      if ((re-rb) * cells / re < 4.0-2.0e-12) exit 1;
    }'; then
  echo "R_B must be an exact long-run mesh face with R_B<R_E and four vacuum-extension cells." >&2
  exit 1
fi

for value in "$VENUS_SOURCE_ROOT" "$VENUS_PAIRED_LONG_WORKFLOW_ROOT" \
             "$VENUS_RUN_ROOT" "$CAMPAIGN_ROOT" \
             "$PARITY_CERTIFICATE" "$PARITY_FINE_PHYSICAL_IC" \
             "$PARITY_FINE_NULL_INITIAL_DATA"; do
  case "$value" in
    *','*|*$'\n'*) echo "Slurm export values cannot contain commas/newlines: $value" >&2; exit 1 ;;
  esac
done

sha256_of() { sha256sum "$1" | awk '{print $1}'; }
# Hash the scheduling, numerical-validation and rendering dependencies that the queued gate must later recheck.
GATE_SCRIPT_SHA256="$(sha256_of "$GATE_SCRIPT")"
SBATCH_SCRIPT_SHA256="$(sha256_of "$SBATCH_SCRIPT")"
SUBMITTER_SHA256="$(sha256_of "$SUBMITTER")"
GIF_SUBMITTER_SHA256="$(sha256_of "$GIF_SUBMITTER")"
GIF_WRAPPER_SHA256="$(sha256_of "$GIF_WRAPPER")"
GIF_RENDERER_SHA256="$(sha256_of "$GIF_RENDERER")"
ANALYZER_SHA256="$(sha256_of "$ANALYZER")"
PROVENANCE_VALIDATOR_SHA256="$(sha256_of "$PROVENANCE_VALIDATOR")"
NULL_MAP_PREFLIGHT_SHA256="$(sha256_of "$NULL_MAP_PREFLIGHT")"
POSITIVE_BINDING_VALIDATOR_SHA256="$(sha256_of "$POSITIVE_BINDING_VALIDATOR")"

mkdir -p "$CAMPAIGN_ROOT/slurm"
# Separate literal solver argv from the human-readable request. One line contains one argument element.
ARGS_FILE="$CAMPAIGN_ROOT/driver_arguments.txt"
REQUEST_MANIFEST="$CAMPAIGN_ROOT/submission_request.manifest.tsv"
printf '%s\n' \
  --high-t-end "$LONG_T_END" \
  --high-gap-stop 0.001 \
  --high-r-min 0.005 \
  --high-inner-boundary origin_parity \
  --high-radial-grid linear \
  --timestep-omitted-mass-fraction 0.00001 \
  --high-output-steps "$LONG_OUTPUT_STEPS" \
  --high-markers "$LONG_MARKERS" \
  --high-radial-cells "$LONG_RADIAL_CELLS" \
  --high-dt-max "$LONG_DT_MAX" > "$ARGS_FILE"
{
  printf 'schema\tvenus_v3_paired_long_submission_request\n'
  printf 'schema_version\t1\n'
  printf 'certificate_job_id\t%s\n' "$PARITY_CERTIFICATE_JOB_ID"
  printf 'science_source_root\t%s\n' "$VENUS_SOURCE_ROOT"
  printf 'paired_long_workflow_root\t%s\n' "$VENUS_PAIRED_LONG_WORKFLOW_ROOT"
  printf 'certificate_path\t%s\n' "$PARITY_CERTIFICATE"
  printf 'physical_ic\t%s\n' "$PARITY_FINE_PHYSICAL_IC"
  printf 'null_initial_data\t%s\n' "$PARITY_FINE_NULL_INITIAL_DATA"
  printf 'initial_condition\t%s\n' "$HIGH_INITIAL_CONDITION"
  printf 'energy_cutoff_amplitude_C\t%s\n' "$ENERGY_CUTOFF_AMPLITUDE"
  printf 'energy_cutoff_ell0\t%s\n' "$ENERGY_CUTOFF_ELL0"
  printf 'energy_cutoff_central_y\t%s\n' "$ENERGY_CUTOFF_CENTRAL_Y"
  printf 'inward_radial_kick\t%s\n' "$RADIAL_INFALL"
  printf 'infall_core_radius\t%s\n' "$INFALL_CORE_RADIUS"
  printf 'angular_momentum_scale\t%s\n' "$ANGULAR_MOMENTUM_SCALE"
  printf 'target_initial_adm_mass\t%s\n' "$TARGET_INITIAL_ADM_MASS"
  printf 'sampling_realization\t%s\n' "$SAMPLING_REALIZATION"
  printf 'markers\t%s\n' "$LONG_MARKERS"
  printf 'radial_cells\t%s\n' "$LONG_RADIAL_CELLS"
  printf 'dt_max\t%s\n' "$LONG_DT_MAX"
  printf 'end_time\t%s\n' "$LONG_T_END"
  printf 'output_steps\t%s\n' "$LONG_OUTPUT_STEPS"
  printf 'map_worldtube_radius_R_B\t%s\n' "$EF_MAP_WORLDTUBE_RADIUS"
  printf 'evolution_outer_radius_R_E\t%s\n' "$LONG_R_MAX"
  printf 'ef_null_donor_duration\t%s\n' "$EF_NULL_DONOR_DURATION"
  printf 'eikonal_substeps\t%s\n' "$VENUS_EIKONAL_SUBSTEPS"
  printf 'submitter_sha256\t%s\n' "$SUBMITTER_SHA256"
  printf 'gate_script_sha256\t%s\n' "$GATE_SCRIPT_SHA256"
  printf 'gif_submitter_sha256\t%s\n' "$GIF_SUBMITTER_SHA256"
  printf 'gif_wrapper_sha256\t%s\n' "$GIF_WRAPPER_SHA256"
  printf 'gif_renderer_sha256\t%s\n' "$GIF_RENDERER_SHA256"
  printf 'launcher_sha256\t%s\n' "$SBATCH_SCRIPT_SHA256"
  printf 'analyzer_sha256\t%s\n' "$ANALYZER_SHA256"
  printf 'provenance_validator_sha256\t%s\n' "$PROVENANCE_VALIDATOR_SHA256"
  printf 'null_map_preflight_sha256\t%s\n' "$NULL_MAP_PREFLIGHT_SHA256"
  printf 'positive_binding_validator_sha256\t%s\n' "$POSITIVE_BINDING_VALIDATOR_SHA256"
} > "$REQUEST_MANIFEST"
REQUEST_MANIFEST_SHA256="$(sha256_of "$REQUEST_MANIFEST")"
ARGS_FILE_SHA256="$(sha256_of "$ARGS_FILE")"
printf '%s  %s\n' "$REQUEST_MANIFEST_SHA256" \
  "$(basename "$REQUEST_MANIFEST")" > "$REQUEST_MANIFEST.sha256"
printf '%s  %s\n' "$ARGS_FILE_SHA256" "$(basename "$ARGS_FILE")" \
  > "$ARGS_FILE.sha256"

# Prevent --export=ALL from carrying a stale production control into the gate
# or the two children.  The gate independently supplies every accepted value.
# Clear inherited solver controls before exporting this request so previous shell settings cannot alter the pair.
unset EXECUTABLE RESUME VENUS_BUILD VENUS_RUN_ARGS_FILE OUTPUT_DIR PHYSICAL_IC \
  COORDINATES EXPECTED_OUTCOME INITIALIZE_ONLY VENUS_EF_NULL_INITIAL_DATA \
  VENUS_EXPORT_EF_NULL_INITIAL_DATA VENUS_EF_NULL_DONOR_DURATION \
  VENUS_EF_MAP_WORLDTUBE_RADIUS VENUS_EF_EXCISION_A \
  VENUS_EF_EXCISION_BUFFER_CELLS VENUS_HIGH_SPECIFIC_ANGULAR_MOMENTUM \
  VENUS_PARITY_CERTIFICATE_SHA256 VENUS_CAMPAIGN_CONTROLS_SHA256 \
  VENUS_ANALYZER VENUS_NULL_MAP_PREFLIGHT \
  VENUS_POSITIVE_BINDING_VALIDATOR VENUS_THRESHOLD_PROVENANCE_VALIDATOR

gate_export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,VENUS_PAIRED_LONG_WORKFLOW_ROOT=$VENUS_PAIRED_LONG_WORKFLOW_ROOT,VENUS_RUN_ROOT=$VENUS_RUN_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,PARITY_CERTIFICATE_JOB_ID=$PARITY_CERTIFICATE_JOB_ID,PARITY_CERTIFICATE=$PARITY_CERTIFICATE,PARITY_FINE_PHYSICAL_IC=$PARITY_FINE_PHYSICAL_IC,PARITY_FINE_NULL_INITIAL_DATA=$PARITY_FINE_NULL_INITIAL_DATA,HIGH_INITIAL_CONDITION=$HIGH_INITIAL_CONDITION,ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE,ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0,ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y,RADIAL_INFALL=$RADIAL_INFALL,INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS,ANGULAR_MOMENTUM_SCALE=$ANGULAR_MOMENTUM_SCALE,TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS,SAMPLING_REALIZATION=$SAMPLING_REALIZATION,LONG_MARKERS=$LONG_MARKERS,LONG_RADIAL_CELLS=$LONG_RADIAL_CELLS,LONG_DT_MAX=$LONG_DT_MAX,LONG_R_MAX=$LONG_R_MAX,LONG_T_END=$LONG_T_END,LONG_OUTPUT_STEPS=$LONG_OUTPUT_STEPS,EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS,EF_NULL_DONOR_DURATION=$EF_NULL_DONOR_DURATION,VENUS_EIKONAL_SUBSTEPS=$VENUS_EIKONAL_SUBSTEPS,VENUS_PAIRED_LONG_GATE_SHA256=$GATE_SCRIPT_SHA256,VENUS_PAIRED_LONG_SUBMITTER_SHA256=$SUBMITTER_SHA256,VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256=$GIF_SUBMITTER_SHA256,VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256=$GIF_WRAPPER_SHA256,VENUS_PAIRED_LONG_GIF_RENDERER_SHA256=$GIF_RENDERER_SHA256,VENUS_LAUNCHER_SHA256=$SBATCH_SCRIPT_SHA256,VENUS_ANALYZER_SHA256=$ANALYZER_SHA256,VENUS_THRESHOLD_PROVENANCE_VALIDATOR_SHA256=$PROVENANCE_VALIDATOR_SHA256,VENUS_NULL_MAP_PREFLIGHT_SHA256=$NULL_MAP_PREFLIGHT_SHA256,VENUS_POSITIVE_BINDING_VALIDATOR_SHA256=$POSITIVE_BINDING_VALIDATOR_SHA256,VENUS_PAIRED_LONG_REQUEST_SHA256=$REQUEST_MANIFEST_SHA256,VENUS_PAIRED_LONG_ARGS_SHA256=$ARGS_FILE_SHA256"
# Submit only the small prerequisite gate here. Successful certificate authentication controls subsequent evolution submissions.
gate_id="$(sbatch --parsable \
  --dependency="afterok:$PARITY_CERTIFICATE_JOB_ID" \
  --job-name=venus-paired-long-gate \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="$gate_export" "$GATE_SCRIPT")"
gate_id="${gate_id%%;*}"
# Preserve the actual returned scheduler identifier and certificate dependency in the submission receipt.
printf 'gate_job_id\t%s\ncertificate_dependency\tafterok:%s\n' \
  "$gate_id" "$PARITY_CERTIFICATE_JOB_ID" \
  > "$CAMPAIGN_ROOT/gate_submission.tsv"
sha256sum "$CAMPAIGN_ROOT/gate_submission.tsv" \
  > "$CAMPAIGN_ROOT/gate_submission.tsv.sha256"

cat <<EOF
Authenticated paired-long request: $CAMPAIGN_ROOT
Certificate job/file:             $PARITY_CERTIFICATE_JOB_ID $PARITY_CERTIFICATE
Post-certificate gate job:        $gate_id
The gate will submit exactly one SC(any), one EF(black_hole), and one
afterok:SC:EF visualization job.
EOF
