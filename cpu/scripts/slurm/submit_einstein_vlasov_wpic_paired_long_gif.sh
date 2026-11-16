#!/bin/bash
# Schedule a paired Schwarzschild/EF movie after both identified evolutions succeed.
# The paired campaign records determine the scientific paths, null map and controls.
# A fresh visualization request and scheduler receipt bind the eventual movie to those exact inputs.

# Queue the authenticated paired-long visualization after both evolutions.

set -euo pipefail
umask 027

if [ -n "${VENUS_PAIRED_LONG_GIF_JOB:-}${VENUS_PAIRED_LONG_GIF_RENDERER:-}" ]; then
  echo "Paired-long GIF workflow overrides are forbidden." >&2
  exit 1
fi

VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?set the frozen scientific source root}"
VENUS_PAIRED_LONG_WORKFLOW_ROOT="${VENUS_PAIRED_LONG_WORKFLOW_ROOT:?set the immutable paired-long workflow root}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?set the paired-long campaign root}"
SC_JOB_ID="${VENUS_PAIRED_LONG_SC_JOB_ID:?set the SC long-run job id}"
EF_JOB_ID="${VENUS_PAIRED_LONG_EF_JOB_ID:?set the EF long-run job id}"
EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set the certified eikonal substeps}"

SUBMITTER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh"
WRAPPER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
RENDERER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py"
MAPPER="$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py"
PROFILE_RENDERER="$VENUS_SOURCE_ROOT/cpu/tools/render_weighted_einstein_vlasov_pic_profile_gifs.py"
ATOMIC_GIF_HELPER="$VENUS_SOURCE_ROOT/cpu/tools/render_weighted_pic_pure_gamma_gifs.py"

SUBMISSION="$CAMPAIGN_ROOT/submission.tsv"
CONTROLS="$CAMPAIGN_ROOT/campaign_controls.manifest.tsv"
REQUEST="$CAMPAIGN_ROOT/visualization_request.manifest.tsv"
RECEIPT="$CAMPAIGN_ROOT/visualization_submission.tsv"
FINAL_DIRECTORY="$CAMPAIGN_ROOT/visualization"

fail() {
  echo "Paired-long GIF submission failed: $*" >&2
  exit 1
}

for value in "$VENUS_SOURCE_ROOT" "$VENUS_PAIRED_LONG_WORKFLOW_ROOT" \
             "$CAMPAIGN_ROOT"; do
  case "$value" in
    /*) ;;
    *) fail "source, workflow, and campaign roots must be absolute" ;;
  esac
  case "$value" in
    *','*|*$'\n'*) fail "Slurm export paths cannot contain commas or newlines" ;;
  esac
done
case "$CAMPAIGN_ROOT" in
  /|/scratch|/scratch/lustre) fail "CAMPAIGN_ROOT is too broad" ;;
esac
for assignment in "SC_JOB_ID=$SC_JOB_ID" "EF_JOB_ID=$EF_JOB_ID"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] \
    || fail "$name must be one positive Slurm job id"
done
[ "$SC_JOB_ID" != "$EF_JOB_ID" ] || fail "SC and EF jobs must be distinct"
[[ "$EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]] \
  || fail "VENUS_EIKONAL_SUBSTEPS must be a positive integer"
command -v sbatch >/dev/null 2>&1 \
  || fail "sbatch is unavailable; run from an IAS login or gate job"

sha256_of() { sha256sum "$1" | awk '{print $1}'; }

# A checksum record must identify the actual artifact bytes and its filename before it supplies scientific paths.
verify_sidecar() {
  local path="$1" actual sidecar_sha sidecar_name extra
  [ -s "$path" ] && [ -s "$path.sha256" ] \
    || fail "artifact or SHA-256 sidecar is missing: $path"
  actual="$(sha256_of "$path")"
  read -r sidecar_sha sidecar_name extra < "$path.sha256" || true
  sidecar_name="${sidecar_name#\*}"
  [ -z "${extra:-}" ] && [ "$sidecar_sha" = "$actual" ] \
    && { [ "$sidecar_name" = "$(basename "$path")" ] \
         || [ "$sidecar_name" = "$path" ]; } \
    || fail "SHA-256 sidecar does not identify $path"
}

for path in "$SUBMITTER" "$WRAPPER" "$RENDERER" "$MAPPER" \
            "$PROFILE_RENDERER" "$ATOMIC_GIF_HELPER"; do
  [ -s "$path" ] || fail "required immutable workflow file is missing: $path"
done
# Keep scientific imports in their authenticated source directory so a scheduling bundle cannot shadow them.
for dependency in compare_wpic_dynamic_coordinates.py \
    render_weighted_einstein_vlasov_pic_profile_gifs.py \
    render_weighted_pic_pure_gamma_gifs.py; do
  [ ! -e "$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/tools/$dependency" ] \
    || fail "workflow root shadows a frozen scientific dependency: $dependency"
done
verify_sidecar "$SUBMISSION"
verify_sidecar "$CONTROLS"
controls_sha256="$(sha256_of "$CONTROLS")"

if [ -e "$REQUEST" ] || [ -e "$REQUEST.sha256" ] \
   || [ -e "$RECEIPT" ] || [ -e "$RECEIPT.sha256" ] \
   || [ -e "$FINAL_DIRECTORY" ]; then
  fail "refusing to reuse a visualization request or output"
fi

SC_OUTPUT="$CAMPAIGN_ROOT/schwarzschild"
EF_OUTPUT="$CAMPAIGN_ROOT/eddington_finkelstein"
# Validate exactly one gate plus the two coordinate-specific evolution rows, including their shared physical realization.
NULL_INITIAL_DATA="$(awk -F '\t' -v sc_id="$SC_JOB_ID" -v ef_id="$EF_JOB_ID" \
    -v root="$CAMPAIGN_ROOT" -v sc_output="$SC_OUTPUT" \
    -v ef_output="$EF_OUTPUT" -v physical_ic="$CAMPAIGN_ROOT/physical_ic.bin" \
    -v controls="$controls_sha256" '
  BEGIN { gate = 0; sc = 0; ef = 0; long_rows = 0; data_rows = 0 }
  NR == 1 {
    expected = "role\tjob_id\tdependency\tcoordinates\texpected_outcome\toutput_dir\tphysical_ic\tnull_initial_data\tcertificate_sha256\tcontrols_sha256"
    if ($0 != expected) exit 1
    next
  }
  {
    data_rows++
    if ($2 !~ /^[1-9][0-9]*$/ || $9 !~ /^[0-9a-f]{64}$/ \
        || $10 != controls) exit 1
    if (certificate == "") certificate = $9
    if ($9 != certificate) exit 1
  }
  $1 == "certificate_gate" {
    gate++
    gate_id = $2
    gate_null_initial_data = $8
    if ($3 !~ /^afterok:[1-9][0-9]*$/ || $4 != "-" || $5 != "-" \
        || $6 != root || $8 == "-") exit 1
    next
  }
  $1 == "long_evolution" {
    long_rows++
    if ($3 != "afterok:" gate_id || $7 != physical_ic) exit 1
    if ($2 == sc_id && $4 == "schwarzschild" && $5 == "any" \
        && $6 == sc_output && $8 == "-") sc++
    if ($2 == ef_id && $4 == "eddington_finkelstein" \
        && $5 == "black_hole" && $6 == ef_output && $8 != "-") {
      ef++
      null_initial_data = $8
    }
    next
  }
  { exit 1 }
  END {
    if (data_rows != 3 || gate != 1 || long_rows != 2 \
        || sc != 1 || ef != 1 || null_initial_data != gate_null_initial_data) exit 1
    print null_initial_data
  }
' "$SUBMISSION")" \
  || fail "submission.tsv does not identify exactly this SC/EF pair"
case "$NULL_INITIAL_DATA" in
  /*) ;;
  *) fail "EF receipt null_initial_data must be absolute" ;;
esac
case "$NULL_INITIAL_DATA" in
  *','*|*$'\n'*) fail "EF receipt null_initial_data cannot contain commas/newlines" ;;
esac
[ -s "$NULL_INITIAL_DATA" ] \
  || fail "certified EF null initial data is missing: $NULL_INITIAL_DATA"
NULL_MANIFEST="$NULL_INITIAL_DATA.manifest.json"
[ -s "$NULL_MANIFEST" ] \
  || fail "certified EF null-map manifest is missing: $NULL_MANIFEST"

submitter_sha256="$(sha256_of "$SUBMITTER")"
wrapper_sha256="$(sha256_of "$WRAPPER")"
renderer_sha256="$(sha256_of "$RENDERER")"
mapper_sha256="$(sha256_of "$MAPPER")"
profile_renderer_sha256="$(sha256_of "$PROFILE_RENDERER")"
atomic_gif_helper_sha256="$(sha256_of "$ATOMIC_GIF_HELPER")"
submission_sha256="$(sha256_of "$SUBMISSION")"
null_manifest_sha256="$(sha256_of "$NULL_MANIFEST")"

# Record frame/mapping choices and every imported tool hash before asking Slurm to render.
request_tmp="$REQUEST.tmp.$$"
{
  printf 'schema\tvenus_v3_paired_long_visualization_request\n'
  printf 'schema_version\t1\n'
  printf 'sc_job_id\t%s\n' "$SC_JOB_ID"
  printf 'ef_job_id\t%s\n' "$EF_JOB_ID"
  printf 'dependency\tafterok:%s:%s\n' "$SC_JOB_ID" "$EF_JOB_ID"
  printf 'schwarzschild_output\t%s\n' "$SC_OUTPUT"
  printf 'eddington_finkelstein_output\t%s\n' "$EF_OUTPUT"
  printf 'null_map_manifest\t%s\n' "$NULL_MANIFEST"
  printf 'null_initial_data\t%s\n' "$NULL_INITIAL_DATA"
  printf 'visualization_output\t%s\n' "$FINAL_DIRECTORY"
  printf 'frames\t72\n'
  printf 'fps\t12\n'
  printf 'maximum_same_event_frames\t12\n'
  printf 'mapped_radial_samples\t16\n'
  printf 'ias_python_runtime\t/opt/sns/anaconda3/2024.06/bin/python3\n'
  printf 'eikonal_substeps\t%s\n' "$EIKONAL_SUBSTEPS"
  printf 'submitter_sha256\t%s\n' "$submitter_sha256"
  printf 'wrapper_sha256\t%s\n' "$wrapper_sha256"
  printf 'renderer_sha256\t%s\n' "$renderer_sha256"
  printf 'production_mapper_sha256\t%s\n' "$mapper_sha256"
  printf 'profile_renderer_sha256\t%s\n' "$profile_renderer_sha256"
  printf 'atomic_gif_helper_sha256\t%s\n' "$atomic_gif_helper_sha256"
  printf 'paired_submission_sha256\t%s\n' "$submission_sha256"
  printf 'paired_controls_sha256\t%s\n' "$controls_sha256"
  printf 'paired_null_manifest_sha256\t%s\n' "$null_manifest_sha256"
} > "$request_tmp"
mv -- "$request_tmp" "$REQUEST"
request_sha256="$(sha256_of "$REQUEST")"
printf '%s  %s\n' "$request_sha256" "$(basename "$REQUEST")" \
  > "$REQUEST.sha256"

render_export="VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,VENUS_PAIRED_LONG_WORKFLOW_ROOT=$VENUS_PAIRED_LONG_WORKFLOW_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,VENUS_PAIRED_LONG_SC_JOB_ID=$SC_JOB_ID,VENUS_PAIRED_LONG_EF_JOB_ID=$EF_JOB_ID,VENUS_EIKONAL_SUBSTEPS=$EIKONAL_SUBSTEPS,VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256=$submitter_sha256,VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256=$wrapper_sha256,VENUS_PAIRED_LONG_GIF_RENDERER_SHA256=$renderer_sha256,VENUS_PARITY_CERTIFIER_SHA256=$mapper_sha256,VENUS_PROFILE_RENDERER_SHA256=$profile_renderer_sha256,VENUS_ATOMIC_GIF_HELPER_SHA256=$atomic_gif_helper_sha256,VENUS_PAIRED_LONG_SUBMISSION_SHA256=$submission_sha256,VENUS_PAIRED_LONG_CONTROLS_SHA256=$controls_sha256,VENUS_PAIRED_LONG_NULL_MANIFEST_SHA256=$null_manifest_sha256,VENUS_PAIRED_LONG_GIF_REQUEST_SHA256=$request_sha256"
# The visualization depends on successful termination of both coordinate jobs, preserving their separate chart histories.
render_id="$(sbatch --parsable \
  --dependency="afterok:$SC_JOB_ID:$EF_JOB_ID" \
  --job-name=venus-paired-long-gif \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="$render_export" "$WRAPPER")"
render_id="${render_id%%;*}"
[[ "$render_id" =~ ^[1-9][0-9]*$ ]] \
  || fail "sbatch returned a malformed visualization job id: $render_id"

# Publish the returned render job identifier with its two dependencies and exact request checksum.
receipt_tmp="$RECEIPT.tmp.$$"
{
  printf 'role\tjob_id\tdependency\tsc_job_id\tef_job_id\toutput_dir\trequest_sha256\n'
  printf 'visualization\t%s\tafterok:%s:%s\t%s\t%s\t%s\t%s\n' \
    "$render_id" "$SC_JOB_ID" "$EF_JOB_ID" "$SC_JOB_ID" "$EF_JOB_ID" \
    "$FINAL_DIRECTORY" "$request_sha256"
} > "$receipt_tmp"
mv -- "$receipt_tmp" "$RECEIPT"
sha256sum "$RECEIPT" > "$RECEIPT.sha256"

echo "Paired-long visualization job: $render_id"
echo "Dependency: afterok:$SC_JOB_ID:$EF_JOB_ID"
echo "Output: $FINAL_DIRECTORY"
