#!/bin/bash
# Build coordinate-comparison uncertainty evidence from two resolutions and two particle realizations.
# Inputs are four diagnostic JSON files and exact comparison/builder identities.
# Outputs include convergence bounds and their command/input records. This job performs analysis only.

# Derive authenticated two-resolution/two-realization uncertainty evidence.

#SBATCH --job-name=venus-parity-conv
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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?Set VENUS_SOURCE_ROOT to the deployed repository path}"
COARSE_A_DIAGNOSTIC="${COARSE_A_DIAGNOSTIC:?set COARSE_A_DIAGNOSTIC}"
COARSE_B_DIAGNOSTIC="${COARSE_B_DIAGNOSTIC:?set COARSE_B_DIAGNOSTIC}"
FINE_A_DIAGNOSTIC="${FINE_A_DIAGNOSTIC:?set FINE_A_DIAGNOSTIC}"
FINE_B_DIAGNOSTIC="${FINE_B_DIAGNOSTIC:?set FINE_B_DIAGNOSTIC}"
CONVERGENCE_EVIDENCE="${CONVERGENCE_EVIDENCE:?set CONVERGENCE_EVIDENCE}"
CONVERGENCE_SAFETY_FACTOR="${CONVERGENCE_SAFETY_FACTOR:-1.5}"
EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set VENUS_EIKONAL_SUBSTEPS}"
EXPECTED_SCRIPT_SHA256="${VENUS_CONVERGENCE_SCRIPT_SHA256:?set VENUS_CONVERGENCE_SCRIPT_SHA256}"
EXPECTED_CERTIFIER_SHA256="${VENUS_PARITY_CERTIFIER_SHA256:?set VENUS_PARITY_CERTIFIER_SHA256}"
EXPECTED_BUILDER_SHA256="${VENUS_CONVERGENCE_BUILDER_SHA256:?set VENUS_CONVERGENCE_BUILDER_SHA256}"

for variable in VENUS_SOURCE_ROOT COARSE_A_DIAGNOSTIC COARSE_B_DIAGNOSTIC \
                FINE_A_DIAGNOSTIC FINE_B_DIAGNOSTIC CONVERGENCE_EVIDENCE; do
  value="${!variable}"
  case "$value" in /*) ;; *) printf -v "$variable" '%s/%s' "$SUBMIT_DIR" "$value" ;; esac
done
if [ -n "${VENUS_CONVERGENCE_BUILDER:-}" ]; then
  echo "VENUS_CONVERGENCE_BUILDER overrides are forbidden; the builder is pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi
BUILDER="$VENUS_SOURCE_ROOT/cpu/tools/build_wpic_dynamic_coordinate_convergence.py"
SCRIPT_IDENTITY="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_convergence.sbatch"

# Bind the analysis entry points to the submitted source before reading their scientific diagnostic outputs.
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
verify_identity convergence-script "$SCRIPT_IDENTITY" "$EXPECTED_SCRIPT_SHA256"
verify_identity convergence-builder "$BUILDER" "$EXPECTED_BUILDER_SHA256"
verify_identity parity-certifier \
  "$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py" \
  "$EXPECTED_CERTIFIER_SHA256"

for artifact in "$BUILDER" "$COARSE_A_DIAGNOSTIC" "$COARSE_B_DIAGNOSTIC" \
                "$FINE_A_DIAGNOSTIC" "$FINE_B_DIAGNOSTIC"; do
  if [ ! -s "$artifact" ]; then
    echo "Convergence input is missing or empty: $artifact" >&2
    exit 1
  fi
done
if ! [[ "$EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VENUS_EIKONAL_SUBSTEPS must be a positive integer." >&2
  exit 1
fi
if [ -e "$CONVERGENCE_EVIDENCE" ] \
   || [ -e "$CONVERGENCE_EVIDENCE.sha256" ]; then
  echo "Refusing to overwrite convergence evidence: $CONVERGENCE_EVIDENCE" >&2
  exit 1
fi
mkdir -p "$(dirname "$CONVERGENCE_EVIDENCE")"

# Keep the measured uncertainty construction and its strict upper ceilings explicit in the executed argument list.
command=(
  python3 "$BUILDER"
  --coarse-a "$COARSE_A_DIAGNOSTIC"
  --coarse-b "$COARSE_B_DIAGNOSTIC"
  --fine-a "$FINE_A_DIAGNOSTIC"
  --fine-b "$FINE_B_DIAGNOSTIC"
  --eikonal-substeps "$EIKONAL_SUBSTEPS"
  --diagnostic-certifier-sha256 "$EXPECTED_CERTIFIER_SHA256"
  --convergence-builder-sha256 "$EXPECTED_BUILDER_SHA256"
  --safety-factor "$CONVERGENCE_SAFETY_FACTOR"
  --metric-ceiling 0.005
  --stress-ceiling 0.02
  --moment-ceiling 0.02
  --distribution-ceiling 1.0e-10
  --output "$CONVERGENCE_EVIDENCE"
)
{
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$CONVERGENCE_EVIDENCE.resolved_command.sh"

srun --ntasks=1 --cpus-per-task=1 "${command[@]}" \
  2>&1 | tee "$CONVERGENCE_EVIDENCE.log"
python3 - "$CONVERGENCE_EVIDENCE" "$EIKONAL_SUBSTEPS" \
  "$EXPECTED_CERTIFIER_SHA256" "$EXPECTED_BUILDER_SHA256" <<'PY'
import json
import pathlib
import sys

# Validate the returned evidence schema and independent-resolution/realization counts after the builder exits.
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected_eikonal_substeps = int(sys.argv[2])
expected_certifier_sha256 = sys.argv[3]
expected_builder_sha256 = sys.argv[4]
# The uncertainty estimate must use the same production metric interpolation as the coordinate comparison.
expected_interpolation = {
    "schwarzschild": "production_center_face_lambda_monotone_hermite_mu",
    "eddington_finkelstein": "production_center_face_linear_a_b",
    "donor_history_format_version": 2,
    "production_equivalent": True,
}
if value.get("schema") != "venus_wpic_dynamic_coordinate_convergence_v1" \
        or value.get("status") != "pass" \
        or value.get("resolution_levels", 0) < 2 \
        or value.get("independent_pic_realizations", 0) < 2 \
        or value.get("monotone_refinement") is not True \
        or value.get("eikonal_substeps") != expected_eikonal_substeps \
        or value.get("diagnostic_certifier_sha256") \
            != expected_certifier_sha256 \
        or value.get("convergence_builder_sha256") \
            != expected_builder_sha256 \
        or value.get("metric_interpolation") != expected_interpolation:
    raise SystemExit("convergence builder did not produce passing evidence")
PY
if [ ! -s "$CONVERGENCE_EVIDENCE.sha256" ]; then
  echo "Convergence builder did not write its SHA-256 sidecar." >&2
  exit 2
fi
# Record complete diagnostic inputs and recheck executable analysis sources after processing.
sha256sum "$BUILDER" "$COARSE_A_DIAGNOSTIC" "$COARSE_B_DIAGNOSTIC" \
  "$FINE_A_DIAGNOSTIC" "$FINE_B_DIAGNOSTIC" \
  > "$CONVERGENCE_EVIDENCE.inputs.sha256"
sha256sum "$CONVERGENCE_EVIDENCE.inputs.sha256" \
  > "$CONVERGENCE_EVIDENCE.inputs.sha256.sha256"
verify_identity convergence-script "$SCRIPT_IDENTITY" "$EXPECTED_SCRIPT_SHA256"
verify_identity convergence-builder "$BUILDER" "$EXPECTED_BUILDER_SHA256"
verify_identity parity-certifier \
  "$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py" \
  "$EXPECTED_CERTIFIER_SHA256"
echo "Strict convergence evidence: $CONVERGENCE_EVIDENCE"
