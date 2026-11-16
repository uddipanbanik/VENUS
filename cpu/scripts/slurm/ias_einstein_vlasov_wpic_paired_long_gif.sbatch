#!/bin/bash
# Render an authenticated long Schwarzschild/EF pair after both scheduled evolutions complete.
# Inputs are the paired submission/control records, saved radial fields and the certified null-map geometry.
# The output directory contains the movie, actual frame ledger, representative images and their reproducibility records.

# Render the authenticated SC/EF long pair only after both evolutions succeed.
#
# This job is submitted by the paired-long certificate gate.  It accepts no
# run-directory or output override: every consumed and published path is fixed
# below the one authenticated paired campaign root.

#SBATCH --job-name=venus-paired-long-gif
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 027

VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?set the frozen scientific source root}"
VENUS_PAIRED_LONG_WORKFLOW_ROOT="${VENUS_PAIRED_LONG_WORKFLOW_ROOT:?set the immutable paired-long workflow root}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?set the paired-long campaign root}"
SC_JOB_ID="${VENUS_PAIRED_LONG_SC_JOB_ID:?set the completed SC job id}"
EF_JOB_ID="${VENUS_PAIRED_LONG_EF_JOB_ID:?set the completed EF job id}"
EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set the certified eikonal substeps}"

EXPECTED_WRAPPER_SHA256="${VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256:?set the GIF wrapper SHA-256}"
EXPECTED_SUBMITTER_SHA256="${VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256:?set the GIF submitter SHA-256}"
EXPECTED_RENDERER_SHA256="${VENUS_PAIRED_LONG_GIF_RENDERER_SHA256:?set the GIF renderer SHA-256}"
EXPECTED_MAPPER_SHA256="${VENUS_PARITY_CERTIFIER_SHA256:?set the production mapper SHA-256}"
EXPECTED_PROFILE_RENDERER_SHA256="${VENUS_PROFILE_RENDERER_SHA256:?set the profile renderer SHA-256}"
EXPECTED_ATOMIC_GIF_HELPER_SHA256="${VENUS_ATOMIC_GIF_HELPER_SHA256:?set the GIF helper SHA-256}"
EXPECTED_SUBMISSION_SHA256="${VENUS_PAIRED_LONG_SUBMISSION_SHA256:?set the paired receipt SHA-256}"
EXPECTED_CONTROLS_SHA256="${VENUS_PAIRED_LONG_CONTROLS_SHA256:?set the paired controls SHA-256}"
EXPECTED_NULL_MANIFEST_SHA256="${VENUS_PAIRED_LONG_NULL_MANIFEST_SHA256:?set the paired null-map manifest SHA-256}"
EXPECTED_REQUEST_SHA256="${VENUS_PAIRED_LONG_GIF_REQUEST_SHA256:?set the GIF request SHA-256}"

SUBMITTER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh"
WRAPPER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
RENDERER="$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py"
MAPPER="$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py"
PROFILE_RENDERER="$VENUS_SOURCE_ROOT/cpu/tools/render_weighted_einstein_vlasov_pic_profile_gifs.py"
ATOMIC_GIF_HELPER="$VENUS_SOURCE_ROOT/cpu/tools/render_weighted_pic_pure_gamma_gifs.py"
IAS_ANACONDA_PYTHON=/opt/sns/anaconda3/2024.06/bin/python3
if [ -x "$IAS_ANACONDA_PYTHON" ]; then
  PYTHON_BIN="$IAS_ANACONDA_PYTHON"
else
  # Local contract tests do not have the IAS filesystem; production IAS jobs
  # always take the pinned branch above.
  PYTHON_BIN="$(command -v python3)"
fi

SUBMISSION="$CAMPAIGN_ROOT/submission.tsv"
CONTROLS="$CAMPAIGN_ROOT/campaign_controls.manifest.tsv"
SC_OUTPUT="$CAMPAIGN_ROOT/schwarzschild"
EF_OUTPUT="$CAMPAIGN_ROOT/eddington_finkelstein"
REQUEST="$CAMPAIGN_ROOT/visualization_request.manifest.tsv"
FINAL_DIRECTORY="$CAMPAIGN_ROOT/visualization"

fail() {
  echo "Paired-long GIF gate failed: $*" >&2
  exit 1
}

case "$CAMPAIGN_ROOT" in
  /*) ;;
  *) fail "CAMPAIGN_ROOT must be absolute" ;;
esac
case "$CAMPAIGN_ROOT" in
  /|/scratch|/scratch/lustre) fail "CAMPAIGN_ROOT is too broad" ;;
esac
for value in "$VENUS_SOURCE_ROOT" "$VENUS_PAIRED_LONG_WORKFLOW_ROOT"; do
  case "$value" in
    /*) ;;
    *) fail "source and workflow roots must be absolute" ;;
  esac
done
for value in "$CAMPAIGN_ROOT" "$VENUS_SOURCE_ROOT" \
             "$VENUS_PAIRED_LONG_WORKFLOW_ROOT"; do
  case "$value" in
    *','*|*$'\n'*) fail "paths cannot contain commas or newlines" ;;
  esac
done
for assignment in "SC_JOB_ID=$SC_JOB_ID" "EF_JOB_ID=$EF_JOB_ID"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] \
    || fail "$name must be one positive Slurm job id"
done
if [ "$SC_JOB_ID" = "$EF_JOB_ID" ]; then
  fail "SC and EF must be distinct Slurm jobs"
fi
[[ "$EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]] \
  || fail "VENUS_EIKONAL_SUBSTEPS must be a positive integer"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

verify_identity() {
  local label="$1" path="$2" expected="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
    || fail "$label expected SHA-256 is malformed"
  [ -s "$path" ] || fail "$label is missing or empty: $path"
  actual="$(sha256_of "$path")"
  [ "$actual" = "$expected" ] \
    || fail "$label changed after submission: expected=$expected actual=$actual"
}

verify_sha_sidecar() {
  local label="$1" path="$2" expected="$3" sidecar
  local actual sidecar_sha sidecar_name extra
  sidecar="$path.sha256"
  verify_identity "$label" "$path" "$expected"
  [ -s "$sidecar" ] || fail "$label SHA-256 sidecar is missing: $sidecar"
  read -r sidecar_sha sidecar_name extra < "$sidecar" || true
  sidecar_name="${sidecar_name#\*}"
  actual="$(sha256_of "$path")"
  [ -z "${extra:-}" ] && [ "$sidecar_sha" = "$actual" ] \
    && { [ "$sidecar_name" = "$(basename "$path")" ] \
         || [ "$sidecar_name" = "$path" ]; } \
    || fail "$label SHA-256 sidecar does not identify its artifact"
}

# Authenticate the renderer and scientific imports separately, and reject local files that would shadow those imports.
verify_fixed_inputs() {
  local dependency
  for dependency in compare_wpic_dynamic_coordinates.py \
      render_weighted_einstein_vlasov_pic_profile_gifs.py \
      render_weighted_pic_pure_gamma_gifs.py; do
    [ ! -e "$VENUS_PAIRED_LONG_WORKFLOW_ROOT/cpu/tools/$dependency" ] \
      || fail "workflow root shadows a frozen scientific dependency: $dependency"
  done
  verify_identity submitter "$SUBMITTER" "$EXPECTED_SUBMITTER_SHA256"
  verify_identity wrapper "$WRAPPER" "$EXPECTED_WRAPPER_SHA256"
  verify_identity renderer "$RENDERER" "$EXPECTED_RENDERER_SHA256"
  verify_identity production-mapper "$MAPPER" "$EXPECTED_MAPPER_SHA256"
  verify_identity profile-renderer "$PROFILE_RENDERER" \
    "$EXPECTED_PROFILE_RENDERER_SHA256"
  verify_identity atomic-gif-helper "$ATOMIC_GIF_HELPER" \
    "$EXPECTED_ATOMIC_GIF_HELPER_SHA256"
  verify_sha_sidecar paired-submission "$SUBMISSION" \
    "$EXPECTED_SUBMISSION_SHA256"
  verify_sha_sidecar paired-controls "$CONTROLS" \
    "$EXPECTED_CONTROLS_SHA256"
  verify_sha_sidecar visualization-request "$REQUEST" \
    "$EXPECTED_REQUEST_SHA256"
}

verify_fixed_inputs

# The receipt is authoritative for the two prerequisite jobs and their exact
# paired-campaign paths.  Extra or ambiguous long-evolution rows are rejected.
# Read exactly one gate and two evolution rows, checking both coordinate roles, dependencies and shared physical controls.
NULL_INITIAL_DATA="$(awk -F '\t' -v sc_id="$SC_JOB_ID" -v ef_id="$EF_JOB_ID" \
    -v root="$CAMPAIGN_ROOT" -v sc_output="$SC_OUTPUT" \
    -v ef_output="$EF_OUTPUT" -v physical_ic="$CAMPAIGN_ROOT/physical_ic.bin" \
    -v controls="$EXPECTED_CONTROLS_SHA256" '
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
NULL_MAP_MANIFEST="$NULL_INITIAL_DATA.manifest.json"
verify_identity paired-null-manifest "$NULL_MAP_MANIFEST" \
  "$EXPECTED_NULL_MANIFEST_SHA256"

[ -d "$SC_OUTPUT" ] || fail "SC output directory is missing"
[ -d "$EF_OUTPUT" ] || fail "EF output directory is missing"
for artifact in \
    "$SC_OUTPUT/summary.json" "$SC_OUTPUT/high_compactness_profiles.csv" \
    "$EF_OUTPUT/summary.json" "$EF_OUTPUT/high_compactness_profiles.csv"; do
  [ -s "$artifact" ] || fail "completed-run artifact is missing: $artifact"
done

if [ -e "$FINAL_DIRECTORY" ]; then
  fail "refusing to overwrite an existing visualization: $FINAL_DIRECTORY"
fi
# A job-specific fresh directory isolates incomplete frames from the requested final visualization.
STAGING_DIRECTORY="$CAMPAIGN_ROOT/.visualization.tmp.${SLURM_JOB_ID:?missing SLURM_JOB_ID}"
if [ -e "$STAGING_DIRECTORY" ]; then
  fail "refusing to reuse visualization staging directory: $STAGING_DIRECTORY"
fi
mkdir -m 0750 "$STAGING_DIRECTORY"
OUTPUT_STEM="$STAGING_DIRECTORY/sc_ef_black_hole_profiles"
OUTPUT_GIF="$OUTPUT_STEM.gif"

# Do not inherit a user site or arbitrary module path.  The new renderer lives
# in the immutable workflow bundle; its three imports come only from the
# frozen, certified scientific checkout whose hashes were checked above.
# Use only the authenticated scientific tools for Python imports while keeping plotting cache files in temporary storage.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENUS_SOURCE_ROOT/cpu/tools"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/venus-matplotlib-${SLURM_JOB_ID}"
if [ "$PYTHON_BIN" = "$IAS_ANACONDA_PYTHON" ]; then
  "$PYTHON_BIN" -c 'import numpy, matplotlib, PIL'
fi
PYTHON_BIN_SHA256="$(sha256_of "$PYTHON_BIN")"
PYTHON_VERSION="$($PYTHON_BIN --version 2>&1)"
# The renderer maps the overlap to common events, then identifies EF-only frames beyond the Schwarzschild chart limit.
"$PYTHON_BIN" "$RENDERER" \
  --schwarzschild "$SC_OUTPUT" \
  --ef "$EF_OUTPUT" \
  --manifest "$NULL_MAP_MANIFEST" \
  --eikonal-substeps "$EIKONAL_SUBSTEPS" \
  --frames 72 \
  --fps 12 \
  --maximum-same-event-frames 12 \
  --mapped-radial-samples 16 \
  --output "$OUTPUT_GIF"

# Require the movie, three images, complete frame ledger and input provenance before publication.
expected_outputs=(
  sc_ef_black_hole_profiles.gif
  sc_ef_black_hole_profiles_initial.png
  sc_ef_black_hole_profiles_middle.png
  sc_ef_black_hole_profiles_final.png
  sc_ef_black_hole_profiles_frames.csv
  sc_ef_black_hole_profiles_provenance.json
)
for name in "${expected_outputs[@]}"; do
  [ -s "$STAGING_DIRECTORY/$name" ] \
    || fail "renderer did not create $name"
done
if find "$STAGING_DIRECTORY" -maxdepth 1 -type f -name '*.tmp*' \
    -print -quit | grep -q .; then
  fail "renderer left an incomplete temporary artifact"
fi

# Verify file signatures, frame/provenance semantics, and rewrite only the two
# publication paths that necessarily change when the staging directory is
# atomically renamed.  The numerical/input hashes remain renderer-authored.
"$PYTHON_BIN" - "$STAGING_DIRECTORY" "$FINAL_DIRECTORY" "$SC_OUTPUT" \
    "$EF_OUTPUT" "$NULL_MAP_MANIFEST" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys

stage = pathlib.Path(sys.argv[1])
final = pathlib.Path(sys.argv[2])
sc = pathlib.Path(sys.argv[3])
ef = pathlib.Path(sys.argv[4])
manifest = pathlib.Path(sys.argv[5])
stem = "sc_ef_black_hole_profiles"
gif = stage / f"{stem}.gif"
pngs = [stage / f"{stem}_{name}.png" for name in ("initial", "middle", "final")]
ledger = stage / f"{stem}_frames.csv"
provenance_path = stage / f"{stem}_provenance.json"

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# A completed file must carry its format signature and GIF trailer; named scientific inputs and ledger phases are checked independently below.
data = gif.read_bytes()
if data[:6] not in (b"GIF87a", b"GIF89a") or not data.endswith(b";"):
    raise SystemExit("GIF signature or trailer is incomplete")
for path in pngs:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"PNG signature is invalid: {path}")

with ledger.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 72:
    raise SystemExit(f"expected 72 ledger rows, found {len(rows)}")
phases = {row.get("phase") for row in rows}
if "same_event_overlap" not in phases or "ef_only_after_sc_chart_limit" not in phases:
    raise SystemExit("ledger lacks either same-event overlap or EF-only continuation")

provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
if provenance.get("frames") != 72:
    raise SystemExit("provenance frame count is not 72")
inputs = provenance.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("provenance has no input audit")
if pathlib.Path(inputs.get("schwarzschild_profiles", "")) != \
        sc.resolve() / "high_compactness_profiles.csv":
    raise SystemExit("provenance SC path escaped the paired campaign")
if pathlib.Path(inputs.get("eddington_finkelstein_profiles", "")) != \
        ef.resolve() / "high_compactness_profiles.csv":
    raise SystemExit("provenance EF path escaped the paired campaign")
if pathlib.Path(inputs.get("manifest", "")) != manifest.resolve():
    raise SystemExit("provenance manifest path escaped the paired campaign")
null_root = manifest.parent.resolve()
artifacts = inputs.get("authenticated_artifacts")
if not isinstance(artifacts, dict) or not artifacts:
    raise SystemExit("provenance has no authenticated null-map artifacts")
# Rehash every map artifact after rendering to ensure the displayed coordinate transformation used unchanged inputs.
for name, record in artifacts.items():
    if not isinstance(record, dict) or record.get("verified") is not True:
        raise SystemExit(f"null-map artifact is not verified: {name}")
    path = pathlib.Path(record.get("path", "")).resolve()
    try:
        path.relative_to(null_root)
    except ValueError as error:
        raise SystemExit(
            f"null-map artifact escaped its certified manifest directory: {name}={path}"
        ) from error
    digest = sha256_file(path)
    if digest != record.get("expected_sha256") \
            or digest != record.get("actual_sha256"):
        raise SystemExit(f"null-map artifact changed during rendering: {name}")

# Only the final publication paths change when the staging directory is renamed. Numerical and input identities remain intact.
provenance["gif"] = str(final / f"{stem}.gif")
provenance["frame_ledger"] = str(final / f"{stem}_frames.csv")
temporary = provenance_path.with_name(provenance_path.name + ".publish.tmp")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(provenance, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(provenance_path)
PY

(
  cd "$STAGING_DIRECTORY"
  sha256sum "${expected_outputs[@]}" > artifacts.sha256
)
{
  printf 'schema\tvenus_v3_paired_long_visualization\n'
  printf 'schema_version\t1\n'
  printf 'render_job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'sc_job_id\t%s\n' "$SC_JOB_ID"
  printf 'ef_job_id\t%s\n' "$EF_JOB_ID"
  printf 'dependency\tafterok:%s:%s\n' "$SC_JOB_ID" "$EF_JOB_ID"
  printf 'submitter_sha256\t%s\n' "$EXPECTED_SUBMITTER_SHA256"
  printf 'wrapper_sha256\t%s\n' "$EXPECTED_WRAPPER_SHA256"
  printf 'renderer_sha256\t%s\n' "$EXPECTED_RENDERER_SHA256"
  printf 'production_mapper_sha256\t%s\n' "$EXPECTED_MAPPER_SHA256"
  printf 'profile_renderer_sha256\t%s\n' "$EXPECTED_PROFILE_RENDERER_SHA256"
  printf 'atomic_gif_helper_sha256\t%s\n' "$EXPECTED_ATOMIC_GIF_HELPER_SHA256"
  printf 'paired_submission_sha256\t%s\n' "$EXPECTED_SUBMISSION_SHA256"
  printf 'paired_controls_sha256\t%s\n' "$EXPECTED_CONTROLS_SHA256"
  printf 'paired_null_manifest_sha256\t%s\n' "$EXPECTED_NULL_MANIFEST_SHA256"
  printf 'visualization_request_sha256\t%s\n' "$EXPECTED_REQUEST_SHA256"
  printf 'eikonal_substeps\t%s\n' "$EIKONAL_SUBSTEPS"
  printf 'python_executable\t%s\n' "$PYTHON_BIN"
  printf 'python_sha256\t%s\n' "$PYTHON_BIN_SHA256"
  printf 'python_version\t%s\n' "$PYTHON_VERSION"
} > "$STAGING_DIRECTORY/render_manifest.tsv"
sha256sum "$STAGING_DIRECTORY/render_manifest.tsv" \
  > "$STAGING_DIRECTORY/render_manifest.tsv.sha256"

# Recheck every staged identity after all Python work, then publish the entire
# directory with one same-parent rename.  A failure leaves FINAL_DIRECTORY
# absent, so no partial visualization can be mistaken for a completed result.
verify_fixed_inputs
verify_identity paired-null-manifest "$NULL_MAP_MANIFEST" \
  "$EXPECTED_NULL_MANIFEST_SHA256"
# Publish all validated visualization files with one same-parent directory rename.
mv -- "$STAGING_DIRECTORY" "$FINAL_DIRECTORY"

echo "Authenticated paired-long visualization published: $FINAL_DIRECTORY"
