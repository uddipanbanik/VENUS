#!/bin/bash
# Authenticate a completed coordinate-parity certificate before scheduling its longer Schwarzschild and EF calculations.
# The inputs fix one continuum distribution, physical realization, null map and source/executable identity in relativistic units.
# Outputs include validated prerequisite records, literal family controls and exact dependent-job receipts.

# Authenticate a completed strict parity certificate, then submit one fresh
# Schwarzschild/EF long-run pair.  Both children depend on this gate's success.

#SBATCH --job-name=venus-paired-long-gate
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 027

VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?set VENUS_SOURCE_ROOT}"
VENUS_PAIRED_LONG_WORKFLOW_ROOT="${VENUS_PAIRED_LONG_WORKFLOW_ROOT:?set the separate immutable workflow root}"
VENUS_RUN_ROOT="${VENUS_RUN_ROOT:?set VENUS_RUN_ROOT}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?set CAMPAIGN_ROOT}"
PARITY_CERTIFICATE_JOB_ID="${PARITY_CERTIFICATE_JOB_ID:?set certificate job id}"
PARITY_CERTIFICATE="${PARITY_CERTIFICATE:?set strict certificate path}"
PARITY_FINE_PHYSICAL_IC="${PARITY_FINE_PHYSICAL_IC:?set certified fine-A physical IC}"
PARITY_FINE_NULL_INITIAL_DATA="${PARITY_FINE_NULL_INITIAL_DATA:?set certified fine-A null data}"
HIGH_INITIAL_CONDITION="${HIGH_INITIAL_CONDITION:?set initial condition}"
ENERGY_CUTOFF_AMPLITUDE="${ENERGY_CUTOFF_AMPLITUDE:?set C}"
ENERGY_CUTOFF_ELL0="${ENERGY_CUTOFF_ELL0:?set ell0}"
ENERGY_CUTOFF_CENTRAL_Y="${ENERGY_CUTOFF_CENTRAL_Y:?set central y}"
RADIAL_INFALL="${RADIAL_INFALL:?set radial kick}"
INFALL_CORE_RADIUS="${INFALL_CORE_RADIUS:?set kick core}"
ANGULAR_MOMENTUM_SCALE="${ANGULAR_MOMENTUM_SCALE:?set s_L}"
TARGET_INITIAL_ADM_MASS="${TARGET_INITIAL_ADM_MASS:?set ADM mass}"
SAMPLING_REALIZATION="${SAMPLING_REALIZATION:?set realization}"
LONG_MARKERS="${LONG_MARKERS:?set markers}"
LONG_RADIAL_CELLS="${LONG_RADIAL_CELLS:?set radial cells}"
LONG_DT_MAX="${LONG_DT_MAX:?set dt_max}"
LONG_R_MAX="${LONG_R_MAX:?set R_E}"
LONG_T_END="${LONG_T_END:?set end time}"
LONG_OUTPUT_STEPS="${LONG_OUTPUT_STEPS:?set output cadence}"
EF_MAP_WORLDTUBE_RADIUS="${EF_MAP_WORLDTUBE_RADIUS:?set R_B}"
EF_NULL_DONOR_DURATION="${EF_NULL_DONOR_DURATION:?set donor duration}"
VENUS_EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set certified eikonal substeps}"

EXPECTED_GATE_SHA256="${VENUS_PAIRED_LONG_GATE_SHA256:?set gate SHA-256}"
EXPECTED_SUBMITTER_SHA256="${VENUS_PAIRED_LONG_SUBMITTER_SHA256:?set submitter SHA-256}"
EXPECTED_GIF_SUBMITTER_SHA256="${VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256:?set GIF submitter SHA-256}"
EXPECTED_GIF_WRAPPER_SHA256="${VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256:?set GIF wrapper SHA-256}"
EXPECTED_GIF_RENDERER_SHA256="${VENUS_PAIRED_LONG_GIF_RENDERER_SHA256:?set GIF renderer SHA-256}"
EXPECTED_LAUNCHER_SHA256="${VENUS_LAUNCHER_SHA256:?set launcher SHA-256}"
EXPECTED_ANALYZER_SHA256="${VENUS_ANALYZER_SHA256:?set analyzer SHA-256}"
EXPECTED_PROVENANCE_VALIDATOR_SHA256="${VENUS_THRESHOLD_PROVENANCE_VALIDATOR_SHA256:?set provenance validator SHA-256}"
EXPECTED_NULL_MAP_PREFLIGHT_SHA256="${VENUS_NULL_MAP_PREFLIGHT_SHA256:?set null-map validator SHA-256}"
EXPECTED_BINDING_VALIDATOR_SHA256="${VENUS_POSITIVE_BINDING_VALIDATOR_SHA256:?set binding validator SHA-256}"
EXPECTED_REQUEST_SHA256="${VENUS_PAIRED_LONG_REQUEST_SHA256:?set request SHA-256}"
EXPECTED_ARGS_SHA256="${VENUS_PAIRED_LONG_ARGS_SHA256:?set argument-file SHA-256}"

GATE_SCRIPT="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gate.sbatch"
SUBMITTER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh"
GIF_SUBMITTER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh"
GIF_WRAPPER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
GIF_RENDERER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py"
LAUNCHER="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic.sbatch"
ANALYZER="$VENUS_SOURCE_ROOT/cpu/tools/analyze_einstein_vlasov_wpic.py"
PROVENANCE_VALIDATOR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py"
NULL_MAP_PREFLIGHT="$VENUS_SOURCE_ROOT/cpu/tools/validate_wpic_null_map.py"
BINDING_VALIDATOR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py"
EXECUTABLE="$VENUS_SOURCE_ROOT/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
REQUEST_MANIFEST="$CAMPAIGN_ROOT/submission_request.manifest.tsv"
ARGS_FILE="$CAMPAIGN_ROOT/driver_arguments.txt"

verify_identity() {
  local label="$1" path="$2" expected="$3" actual
  if ! [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || [ ! -s "$path" ]; then
    echo "$label identity is missing or malformed: $path" >&2
    exit 1
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    echo "$label SHA-256 changed after submission: expected=$expected actual=$actual" >&2
    exit 1
  fi
}

# Bind every launch, scientific validation and visualization dependency to its submitted SHA before using it.
verify_all_identities() {
  verify_identity gate "$GATE_SCRIPT" "$EXPECTED_GATE_SHA256"
  verify_identity submitter "$SUBMITTER" "$EXPECTED_SUBMITTER_SHA256"
  verify_identity gif-submitter "$GIF_SUBMITTER" "$EXPECTED_GIF_SUBMITTER_SHA256"
  verify_identity gif-wrapper "$GIF_WRAPPER" "$EXPECTED_GIF_WRAPPER_SHA256"
  verify_identity gif-renderer "$GIF_RENDERER" "$EXPECTED_GIF_RENDERER_SHA256"
  verify_identity launcher "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA256"
  verify_identity analyzer "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
  verify_identity provenance-validator "$PROVENANCE_VALIDATOR" \
    "$EXPECTED_PROVENANCE_VALIDATOR_SHA256"
  verify_identity null-map-preflight "$NULL_MAP_PREFLIGHT" \
    "$EXPECTED_NULL_MAP_PREFLIGHT_SHA256"
  verify_identity binding-validator "$BINDING_VALIDATOR" \
    "$EXPECTED_BINDING_VALIDATOR_SHA256"
  verify_identity request "$REQUEST_MANIFEST" "$EXPECTED_REQUEST_SHA256"
  verify_identity arguments "$ARGS_FILE" "$EXPECTED_ARGS_SHA256"
}

verify_all_identities
# Keep scheduling scripts in a separate directory so they cannot shadow the authenticated scientific source imports.
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
if ! [[ "$VENUS_EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]] \
   || [ "$VENUS_EIKONAL_SUBSTEPS" -ne 8 ]; then
  echo "This production workflow requires VENUS_EIKONAL_SUBSTEPS=8." >&2
  exit 1
fi
if [ ! -x "$EXECUTABLE" ]; then
  echo "Certified executable is missing or not executable: $EXECUTABLE" >&2
  exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable inside the paired-long gate job." >&2
  exit 1
fi
for artifact in "$PARITY_CERTIFICATE" "$PARITY_CERTIFICATE.sha256" \
                "$PARITY_FINE_PHYSICAL_IC" "$PARITY_FINE_NULL_INITIAL_DATA" \
                "$PARITY_FINE_NULL_INITIAL_DATA.manifest.json"; do
  if [ ! -s "$artifact" ]; then
    echo "Required certified fine-A artifact is missing or empty: $artifact" >&2
    exit 1
  fi
done
for output in "$CAMPAIGN_ROOT/parity_certificate_gate.json" \
              "$CAMPAIGN_ROOT/parity_provenance_attestation.json" \
              "$CAMPAIGN_ROOT/validated_parity_certificate.json" \
              "$CAMPAIGN_ROOT/physical_ic.bin" \
              "$CAMPAIGN_ROOT/campaign_controls.manifest.tsv" \
              "$CAMPAIGN_ROOT/submission.tsv" \
              "$CAMPAIGN_ROOT/schwarzschild" \
              "$CAMPAIGN_ROOT/eddington_finkelstein"; do
  if [ -e "$output" ]; then
    echo "Refusing to overwrite paired-long output: $output" >&2
    exit 1
  fi
done

# Require the certificate sidecar to identify both the exact bytes and the expected filename.
certificate_sha256="$(sha256sum "$PARITY_CERTIFICATE" | awk '{print $1}')"
read -r sidecar_sha sidecar_name extra \
  < "$PARITY_CERTIFICATE.sha256" || true
sidecar_name="${sidecar_name#\*}"
if [ -n "${extra:-}" ] || [ "$sidecar_sha" != "$certificate_sha256" ] \
   || [ "$sidecar_name" != "$(basename "$PARITY_CERTIFICATE")" ]; then
  echo "Strict certificate SHA-256 sidecar does not identify the completed certificate." >&2
  exit 1
fi

python3 "$ANALYZER" parity-certificate-gate \
  --parity-certificate "$PARITY_CERTIFICATE" \
  --parity-certificate-sha256 "$certificate_sha256" \
  --output "$CAMPAIGN_ROOT/parity_certificate_gate.json"
# The numerical certificate and its physical/source applicability are separate checks. This call verifies the requested long-run family against the certified inputs.
python3 "$PROVENANCE_VALIDATOR" \
  --purpose paired_long \
  --certificate "$PARITY_CERTIFICATE" \
  --source-root "$VENUS_SOURCE_ROOT" \
  --executable "$EXECUTABLE" \
  --coordinates eddington_finkelstein \
  --initial-condition "$HIGH_INITIAL_CONDITION" \
  --amplitude "$ENERGY_CUTOFF_AMPLITUDE" \
  --ell0 "$ENERGY_CUTOFF_ELL0" \
  --central-y "$ENERGY_CUTOFF_CENTRAL_Y" \
  --radial-infall "$RADIAL_INFALL" \
  --infall-core-radius "$INFALL_CORE_RADIUS" \
  --target-initial-adm-mass "$TARGET_INITIAL_ADM_MASS" \
  --angular-momentum-scales "$ANGULAR_MOMENTUM_SCALE" \
  --markers "$LONG_MARKERS" \
  --radial-cells "$LONG_RADIAL_CELLS" \
  --dt-max "$LONG_DT_MAX" \
  --sampling-realization "$SAMPLING_REALIZATION" \
  --physical-ic "$PARITY_FINE_PHYSICAL_IC" \
  --null-initial-data "$PARITY_FINE_NULL_INITIAL_DATA" \
  --map-worldtube-radius "$EF_MAP_WORLDTUBE_RADIUS" \
  --evolution-outer-radius "$LONG_R_MAX" \
  --ef-null-donor-duration "$EF_NULL_DONOR_DURATION" \
  --output "$CAMPAIGN_ROOT/parity_provenance_attestation.json"
python3 - "$CAMPAIGN_ROOT/parity_certificate_gate.json" \
    "$CAMPAIGN_ROOT/parity_provenance_attestation.json" \
    "$VENUS_EIKONAL_SUBSTEPS" <<'PY'
import json
import pathlib
import sys

expected = int(sys.argv[3])
for value in sys.argv[1:3]:
    path = pathlib.Path(value)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("status") != "pass" \
            or audit.get("issues", []) != []:
        raise SystemExit("paired-long prerequisite did not pass: {}".format(path))
    actual = audit.get("eikonal_substeps")
    if isinstance(actual, bool) or not isinstance(actual, int) \
            or actual != expected:
        raise SystemExit(
            "paired-long eikonal substeps mismatch: {} has {}, expected {}".format(
                path, actual, expected))
PY

# Copy only after prerequisite acceptance, then verify the resulting certificate and physical initial-data bytes.
cp "$PARITY_CERTIFICATE" \
  "$CAMPAIGN_ROOT/validated_parity_certificate.json"
cp "$PARITY_FINE_PHYSICAL_IC" "$CAMPAIGN_ROOT/physical_ic.bin"
validated_certificate_sha256="$(sha256sum \
  "$CAMPAIGN_ROOT/validated_parity_certificate.json" | awk '{print $1}')"
copied_physical_sha256="$(sha256sum "$CAMPAIGN_ROOT/physical_ic.bin" | awk '{print $1}')"
original_physical_sha256="$(sha256sum "$PARITY_FINE_PHYSICAL_IC" | awk '{print $1}')"
if [ "$validated_certificate_sha256" != "$certificate_sha256" ] \
   || [ "$copied_physical_sha256" != "$original_physical_sha256" ]; then
  echo "Certificate or physical-IC copy changed during paired-long setup." >&2
  exit 1
fi
printf '%s  %s\n' "$certificate_sha256" validated_parity_certificate.json \
  > "$CAMPAIGN_ROOT/validated_parity_certificate.sha256"
printf '%s  %s\n' "$copied_physical_sha256" physical_ic.bin \
  > "$CAMPAIGN_ROOT/physical_ic.sha256"

attestation_sha256="$(sha256sum \
  "$CAMPAIGN_ROOT/parity_provenance_attestation.json" | awk '{print $1}')"
# Record the complete distribution, perturbation, resolution, map and time controls used by both coordinate evolutions.
controls="$CAMPAIGN_ROOT/campaign_controls.manifest.tsv"
{
  printf 'schema\tvenus_v3_authenticated_paired_long_campaign\n'
  printf 'schema_version\t1\n'
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
  printf 'schwarzschild_expected_outcome\tany\n'
  printf 'eddington_finkelstein_expected_outcome\tblack_hole\n'
  printf 'certificate_job_id\t%s\n' "$PARITY_CERTIFICATE_JOB_ID"
  printf 'certificate_sha256\t%s\n' "$certificate_sha256"
  printf 'parity_provenance_attestation_sha256\t%s\n' "$attestation_sha256"
  printf 'physical_ic_sha256\t%s\n' "$copied_physical_sha256"
  printf 'null_initial_data_sha256\t%s\n' \
    "$(sha256sum "$PARITY_FINE_NULL_INITIAL_DATA" | awk '{print $1}')"
  printf 'launcher_sha256\t%s\n' "$EXPECTED_LAUNCHER_SHA256"
  printf 'analyzer_sha256\t%s\n' "$EXPECTED_ANALYZER_SHA256"
  printf 'provenance_validator_sha256\t%s\n' \
    "$EXPECTED_PROVENANCE_VALIDATOR_SHA256"
  printf 'null_map_preflight_sha256\t%s\n' \
    "$EXPECTED_NULL_MAP_PREFLIGHT_SHA256"
  printf 'positive_binding_validator_sha256\t%s\n' \
    "$EXPECTED_BINDING_VALIDATOR_SHA256"
  printf 'require_positive_binding\t1\n'
  printf 'build_enabled\t0\n'
  printf 'science_source_root\t%s\n' "$VENUS_SOURCE_ROOT"
  printf 'paired_long_workflow_root\t%s\n' "$VENUS_PAIRED_LONG_WORKFLOW_ROOT"
} > "$controls"
controls_sha256="$(sha256sum "$controls" | awk '{print $1}')"
printf '%s  %s\n' "$controls_sha256" "$(basename "$controls")" \
  > "$controls.sha256"

verify_all_identities

# The children inherit no map/excision/resume surprises from the gate shell.
# Clear inherited map/excision/resume options before exporting the explicitly specified child controls.
unset RESUME VENUS_EXPORT_EF_NULL_INITIAL_DATA \
  VENUS_EF_NULL_DONOR_DURATION VENUS_EF_EXCISION_A \
  VENUS_EF_EXCISION_BUFFER_CELLS VENUS_HIGH_SPECIFIC_ANGULAR_MOMENTUM
common_export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,VENUS_RUN_ROOT=$VENUS_RUN_ROOT,VENUS_RUN_ARGS_FILE=$ARGS_FILE,EXECUTABLE=$EXECUTABLE,PHYSICAL_IC=$CAMPAIGN_ROOT/physical_ic.bin,INITIALIZE_ONLY=0,RESUME=0,VENUS_BUILD=0,VENUS_HIGH_INITIAL_CONDITION=$HIGH_INITIAL_CONDITION,VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE,VENUS_HIGH_ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0,VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y,VENUS_HIGH_RADIAL_INFALL=$RADIAL_INFALL,VENUS_HIGH_INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS,VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=$ANGULAR_MOMENTUM_SCALE,VENUS_HIGH_SAMPLING_REALIZATION=$SAMPLING_REALIZATION,VENUS_HIGH_TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS,VENUS_HIGH_R_MAX=$LONG_R_MAX,VENUS_EIKONAL_SUBSTEPS=$VENUS_EIKONAL_SUBSTEPS,VENUS_CAMPAIGN_CONTROLS_SHA256=$controls_sha256,VENUS_PARITY_CERTIFICATE_SHA256=$certificate_sha256,VENUS_REQUIRE_POSITIVE_BINDING=1,VENUS_LAUNCHER_SHA256=$EXPECTED_LAUNCHER_SHA256,VENUS_ANALYZER_SHA256=$EXPECTED_ANALYZER_SHA256,VENUS_NULL_MAP_PREFLIGHT_SHA256=$EXPECTED_NULL_MAP_PREFLIGHT_SHA256,VENUS_POSITIVE_BINDING_VALIDATOR_SHA256=$EXPECTED_BINDING_VALIDATOR_SHA256"

# Both coordinate jobs depend on successful completion of this gate and reuse the accepted physical realization.
sc_id="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID:?}" \
  --job-name=venus-paired-long-sc \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="$common_export,COORDINATES=schwarzschild,EXPECTED_OUTCOME=any,OUTPUT_DIR=$CAMPAIGN_ROOT/schwarzschild" \
  "$LAUNCHER")"
sc_id="${sc_id%%;*}"
ef_id="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID:?}" \
  --job-name=venus-paired-long-ef \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="$common_export,COORDINATES=eddington_finkelstein,EXPECTED_OUTCOME=black_hole,OUTPUT_DIR=$CAMPAIGN_ROOT/eddington_finkelstein,VENUS_EF_NULL_INITIAL_DATA=$PARITY_FINE_NULL_INITIAL_DATA,VENUS_EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS" \
  "$LAUNCHER")"
ef_id="${ef_id%%;*}"

# Write both returned Slurm identifiers and their exact scientific inputs into one completed submission table.
receipt_tmp="$CAMPAIGN_ROOT/submission.tsv.tmp.${SLURM_JOB_ID}"
{
  printf 'role\tjob_id\tdependency\tcoordinates\texpected_outcome\toutput_dir\tphysical_ic\tnull_initial_data\tcertificate_sha256\tcontrols_sha256\n'
  printf 'certificate_gate\t%s\tafterok:%s\t-\t-\t%s\t%s\t%s\t%s\t%s\n' \
    "$SLURM_JOB_ID" "$PARITY_CERTIFICATE_JOB_ID" "$CAMPAIGN_ROOT" \
    "$PARITY_FINE_PHYSICAL_IC" "$PARITY_FINE_NULL_INITIAL_DATA" \
    "$certificate_sha256" "$controls_sha256"
  printf 'long_evolution\t%s\tafterok:%s\tschwarzschild\tany\t%s\t%s\t-\t%s\t%s\n' \
    "$sc_id" "$SLURM_JOB_ID" "$CAMPAIGN_ROOT/schwarzschild" \
    "$CAMPAIGN_ROOT/physical_ic.bin" "$certificate_sha256" "$controls_sha256"
  printf 'long_evolution\t%s\tafterok:%s\teddington_finkelstein\tblack_hole\t%s\t%s\t%s\t%s\t%s\n' \
    "$ef_id" "$SLURM_JOB_ID" "$CAMPAIGN_ROOT/eddington_finkelstein" \
    "$CAMPAIGN_ROOT/physical_ic.bin" "$PARITY_FINE_NULL_INITIAL_DATA" \
    "$certificate_sha256" "$controls_sha256"
} > "$receipt_tmp"

verify_all_identities
mv "$receipt_tmp" "$CAMPAIGN_ROOT/submission.tsv"
sha256sum "$CAMPAIGN_ROOT/submission.tsv" \
  > "$CAMPAIGN_ROOT/submission.tsv.sha256"

gif_submit_log_tmp="$CAMPAIGN_ROOT/gif_submission.out.tmp.${SLURM_JOB_ID}"
# Schedule visualization from an explicit small environment after the two evolution identifiers have been recorded.
env -i PATH="$PATH" USER="${USER:-}" LOGNAME="${LOGNAME:-}" \
  LANG="${LANG:-C}" \
  VENUS_SOURCE_ROOT="$VENUS_SOURCE_ROOT" \
  VENUS_PAIRED_LONG_WORKFLOW_ROOT="$VENUS_PAIRED_LONG_WORKFLOW_ROOT" \
  CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
  VENUS_PAIRED_LONG_SC_JOB_ID="$sc_id" \
  VENUS_PAIRED_LONG_EF_JOB_ID="$ef_id" \
  VENUS_EIKONAL_SUBSTEPS="$VENUS_EIKONAL_SUBSTEPS" \
  bash "$GIF_SUBMITTER" > "$gif_submit_log_tmp"
mv "$gif_submit_log_tmp" "$CAMPAIGN_ROOT/gif_submission.out"

echo "Authenticated paired-long gate passed."
echo "Schwarzschild(any) job: $sc_id"
echo "EF(black_hole) job:     $ef_id"
cat "$CAMPAIGN_ROOT/gif_submission.out"
