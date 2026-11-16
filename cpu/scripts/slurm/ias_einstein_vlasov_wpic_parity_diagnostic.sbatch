#!/bin/bash
# Compare one SC/EF pair on common physical events before cross-resolution evidence is supplied.
# Inputs are both run directories, the coordinate-map manifest and marker crossings.
# Output is an authenticated diagnostic whose sole permitted unresolved issue is missing convergence evidence.

# Produce one authenticated single-resolution SC/EF diagnostic.  Exit status 2
# from the comparator is expected here: the only permitted issue is the absent
# cross-resolution evidence that the downstream builder will supply.

#SBATCH --job-name=venus-parity-diag
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 027

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?Set VENUS_SOURCE_ROOT to the deployed repository path}"
SCHWARZSCHILD_OUTPUT="${SCHWARZSCHILD_OUTPUT:?set SCHWARZSCHILD_OUTPUT}"
EF_OUTPUT="${EF_OUTPUT:?set EF_OUTPUT}"
COORDINATE_MAP_MANIFEST="${COORDINATE_MAP_MANIFEST:?set COORDINATE_MAP_MANIFEST}"
MARKER_CROSSINGS="${MARKER_CROSSINGS:?set MARKER_CROSSINGS}"
PARITY_DIAGNOSTIC="${PARITY_DIAGNOSTIC:?set PARITY_DIAGNOSTIC}"
EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set VENUS_EIKONAL_SUBSTEPS}"
EXPECTED_SCRIPT_SHA256="${VENUS_DIAGNOSTIC_SCRIPT_SHA256:?set VENUS_DIAGNOSTIC_SCRIPT_SHA256}"
EXPECTED_CERTIFIER_SHA256="${VENUS_PARITY_CERTIFIER_SHA256:?set VENUS_PARITY_CERTIFIER_SHA256}"

for variable in VENUS_SOURCE_ROOT SCHWARZSCHILD_OUTPUT EF_OUTPUT \
                COORDINATE_MAP_MANIFEST MARKER_CROSSINGS PARITY_DIAGNOSTIC; do
  value="${!variable}"
  case "$value" in /*) ;; *) printf -v "$variable" '%s/%s' "$SUBMIT_DIR" "$value" ;; esac
done
if [ -n "${VENUS_PARITY_CERTIFIER:-}" ]; then
  echo "VENUS_PARITY_CERTIFIER overrides are forbidden; the certifier is pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi
CERTIFIER="$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py"
SCRIPT_IDENTITY="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch"

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
verify_identity diagnostic-script "$SCRIPT_IDENTITY" "$EXPECTED_SCRIPT_SHA256"
verify_identity parity-certifier "$CERTIFIER" "$EXPECTED_CERTIFIER_SHA256"

for directory in "$SCHWARZSCHILD_OUTPUT" "$EF_OUTPUT"; do
  if [ ! -d "$directory" ]; then
    echo "Parity run directory is missing: $directory" >&2
    exit 1
  fi
done
for artifact in "$CERTIFIER" "$COORDINATE_MAP_MANIFEST" "$MARKER_CROSSINGS"; do
  if [ ! -s "$artifact" ]; then
    echo "Diagnostic input is missing or empty: $artifact" >&2
    exit 1
  fi
done
if ! [[ "$EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VENUS_EIKONAL_SUBSTEPS must be a positive integer." >&2
  exit 1
fi
if [ -e "$PARITY_DIAGNOSTIC" ] || [ -e "$PARITY_DIAGNOSTIC.sha256" ]; then
  echo "Refusing to overwrite an existing diagnostic: $PARITY_DIAGNOSTIC" >&2
  exit 1
fi
mkdir -p "$(dirname "$PARITY_DIAGNOSTIC")"

# Use strict empirical-distribution comparison as well as metric, stress and moment tolerances.
command=(
  python3 "$CERTIFIER"
  --schwarzschild "$SCHWARZSCHILD_OUTPUT"
  --ef "$EF_OUTPUT"
  --manifest "$COORDINATE_MAP_MANIFEST"
  --marker-crossings "$MARKER_CROSSINGS"
  --empirical-distribution strict
  --metric-tolerance 0.005
  --stress-tolerance 0.02
  --moment-tolerance 0.02
  --distribution-tolerance 1.0e-10
  --eikonal-substeps "$EIKONAL_SUBSTEPS"
  --output "$PARITY_DIAGNOSTIC"
)
{
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$PARITY_DIAGNOSTIC.resolved_command.sh"

set +e
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  "${command[@]}" 2>&1 | tee "$PARITY_DIAGNOSTIC.log"
# Capture the comparator and tee statuses separately so writing a log cannot hide an analysis failure.
pipeline_status=("${PIPESTATUS[@]}")
set -e
certifier_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
if [ "$tee_status" -ne 0 ]; then
  exit "$tee_status"
fi
printf '%s\n' "$certifier_status" > "$PARITY_DIAGNOSTIC.exit_status.txt"
# This intermediate single-resolution analysis has the declared inconclusive exit code; a different code needs review.
if [ "$certifier_status" -ne 2 ]; then
  echo "Single-resolution diagnostic expected certifier status 2, got $certifier_status." >&2
  exit 2
fi

python3 - "$PARITY_DIAGNOSTIC" "$EIKONAL_SUBSTEPS" \
  "$EXPECTED_CERTIFIER_SHA256" <<'PY'
import json
import pathlib
import sys

# Require the exact missing-convergence explanation, with every other scientific comparison passing.
missing = (
    "single-resolution agreement has no independent quantified "
    "discretization/PIC convergence evidence"
)
path = pathlib.Path(sys.argv[1])
expected_eikonal_substeps = int(sys.argv[2])
expected_certifier_sha256 = sys.argv[3]
value = json.loads(path.read_text())
if value.get("schema") != "venus_wpic_dynamic_coordinate_certificate_v2":
    raise SystemExit("diagnostic has the wrong schema")
if value.get("status") != "inconclusive":
    raise SystemExit("diagnostic must be inconclusive before convergence")
if value.get("eikonal_substeps") != expected_eikonal_substeps:
    raise SystemExit("diagnostic does not record the submitted eikonal substeps")
if value.get("certifier_sha256") != expected_certifier_sha256:
    raise SystemExit("diagnostic records a different certifier SHA-256")
expected_interpolation = {
    "schwarzschild": "production_center_face_lambda_monotone_hermite_mu",
    "eddington_finkelstein": "production_center_face_linear_a_b",
    "donor_history_format_version": 2,
    "production_equivalent": True,
}
if value.get("metric_interpolation") != expected_interpolation:
    raise SystemExit("diagnostic lacks production-equivalent metric reconstruction")
if value.get("issues") != [missing] or value.get("failed_fields") not in ([], None):
    raise SystemExit(
        "diagnostic has a scientific failure beyond missing convergence evidence"
    )
distribution = value.get("distribution", {})
if distribution.get("mode") != "strict" or not distribution.get("evidence") \
        or distribution.get("failed_fields") not in ([], None):
    raise SystemExit("diagnostic lacks strict empirical-distribution parity")
# Every coordinate-map artifact must have been independently verified before this diagnostic is usable.
manifest = value.get("provenance", {}).get("coordinate_map_manifest", {})
if not manifest.get("artifacts") or any(
        record.get("verified") is not True
        for record in manifest.get("artifacts", {}).values()):
    raise SystemExit("diagnostic has unauthenticated coordinate-map artifacts")
PY

if [ ! -s "$PARITY_DIAGNOSTIC.sha256" ]; then
  echo "Certifier did not write the diagnostic SHA-256 sidecar." >&2
  exit 2
fi
sha256sum "$CERTIFIER" "$COORDINATE_MAP_MANIFEST" "$MARKER_CROSSINGS" \
  > "$PARITY_DIAGNOSTIC.inputs.sha256"
sha256sum "$PARITY_DIAGNOSTIC.inputs.sha256" \
  > "$PARITY_DIAGNOSTIC.inputs.sha256.sha256"
verify_identity diagnostic-script "$SCRIPT_IDENTITY" "$EXPECTED_SCRIPT_SHA256"
verify_identity parity-certifier "$CERTIFIER" "$EXPECTED_CERTIFIER_SHA256"
echo "Clean single-resolution diagnostic: $PARITY_DIAGNOSTIC"
