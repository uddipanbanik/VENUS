#!/bin/bash
# Combine SC/EF event comparisons with independently measured discretization/PIC uncertainty.
# Inputs include the coordinate map, marker crossings and convergence evidence from their preceding jobs.
# Output is a strict coordinate certificate plus portable input/source digests. No simulation is evolved here.

# Strict same-event SC/EF certificate job.  This job is intentionally separate
# from both evolutions: Slurm afterok dependencies make a passing certificate
# the terminal state of the solved-eikonal parity DAG.

#SBATCH --job-name=venus-parity-cert
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
PARITY_CERTIFICATE="${PARITY_CERTIFICATE:?set PARITY_CERTIFICATE}"
CONVERGENCE_EVIDENCE="${CONVERGENCE_EVIDENCE:?set CONVERGENCE_EVIDENCE}"
CONVERGENCE_EVIDENCE_SHA256="${CONVERGENCE_EVIDENCE_SHA256:-}"
EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set VENUS_EIKONAL_SUBSTEPS}"
EXPECTED_SCRIPT_SHA256="${VENUS_POSTPROCESS_SCRIPT_SHA256:?set VENUS_POSTPROCESS_SCRIPT_SHA256}"
EXPECTED_CERTIFIER_SHA256="${VENUS_PARITY_CERTIFIER_SHA256:?set VENUS_PARITY_CERTIFIER_SHA256}"
EXPECTED_BUILDER_SHA256="${VENUS_CONVERGENCE_BUILDER_SHA256:?set VENUS_CONVERGENCE_BUILDER_SHA256}"

case "$VENUS_SOURCE_ROOT" in /*) ;; *) VENUS_SOURCE_ROOT="$SUBMIT_DIR/$VENUS_SOURCE_ROOT" ;; esac
case "$SCHWARZSCHILD_OUTPUT" in /*) ;; *) SCHWARZSCHILD_OUTPUT="$SUBMIT_DIR/$SCHWARZSCHILD_OUTPUT" ;; esac
case "$EF_OUTPUT" in /*) ;; *) EF_OUTPUT="$SUBMIT_DIR/$EF_OUTPUT" ;; esac
case "$COORDINATE_MAP_MANIFEST" in /*) ;; *) COORDINATE_MAP_MANIFEST="$SUBMIT_DIR/$COORDINATE_MAP_MANIFEST" ;; esac
case "$MARKER_CROSSINGS" in /*) ;; *) MARKER_CROSSINGS="$SUBMIT_DIR/$MARKER_CROSSINGS" ;; esac
case "$PARITY_CERTIFICATE" in /*) ;; *) PARITY_CERTIFICATE="$SUBMIT_DIR/$PARITY_CERTIFICATE" ;; esac
case "$CONVERGENCE_EVIDENCE" in /*) ;; *) CONVERGENCE_EVIDENCE="$SUBMIT_DIR/$CONVERGENCE_EVIDENCE" ;; esac
if [ -n "$CONVERGENCE_EVIDENCE_SHA256" ]; then
  case "$CONVERGENCE_EVIDENCE_SHA256" in
    /*) ;;
    *) CONVERGENCE_EVIDENCE_SHA256="$SUBMIT_DIR/$CONVERGENCE_EVIDENCE_SHA256" ;;
  esac
fi

if [ -n "${VENUS_PARITY_CERTIFIER:-}" ]; then
  echo "VENUS_PARITY_CERTIFIER overrides are forbidden; the certifier is pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi
CERTIFIER="$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py"
SCRIPT_IDENTITY="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_postprocess.sbatch"

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
verify_identity postprocess-script "$SCRIPT_IDENTITY" "$EXPECTED_SCRIPT_SHA256"
verify_identity parity-certifier "$CERTIFIER" "$EXPECTED_CERTIFIER_SHA256"
verify_identity convergence-builder \
  "$VENUS_SOURCE_ROOT/cpu/tools/build_wpic_dynamic_coordinate_convergence.py" \
  "$EXPECTED_BUILDER_SHA256"
for directory in "$SCHWARZSCHILD_OUTPUT" "$EF_OUTPUT"; do
  if [ ! -d "$directory" ]; then
    echo "Parity run directory is missing: $directory" >&2
    exit 1
  fi
done
if ! [[ "$EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VENUS_EIKONAL_SUBSTEPS must be a positive integer." >&2
  exit 1
fi
for artifact in "$CERTIFIER" "$COORDINATE_MAP_MANIFEST" "$MARKER_CROSSINGS" \
                "$CONVERGENCE_EVIDENCE"; do
  if [ ! -s "$artifact" ]; then
    echo "Parity input is missing or empty: $artifact" >&2
    exit 1
  fi
done
if [ -e "$PARITY_CERTIFICATE" ]; then
  echo "Refusing to overwrite an existing parity certificate: $PARITY_CERTIFICATE" >&2
  exit 1
fi
mkdir -p "$(dirname "$PARITY_CERTIFICATE")"

# Pass the convergence evidence to the same strict comparator that produced the single-resolution diagnostics.
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
  --convergence-evidence "$CONVERGENCE_EVIDENCE"
  --output "$PARITY_CERTIFICATE"
)
# An explicitly supplied evidence sidecar is an additional checked input, not a replacement for the evidence file.
if [ -n "$CONVERGENCE_EVIDENCE_SHA256" ]; then
  if [ ! -s "$CONVERGENCE_EVIDENCE_SHA256" ]; then
    echo "Convergence-evidence SHA sidecar is missing: $CONVERGENCE_EVIDENCE_SHA256" >&2
    exit 1
  fi
  command+=(--convergence-evidence-sha256 "$CONVERGENCE_EVIDENCE_SHA256")
fi
{
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$PARITY_CERTIFICATE.resolved_command.sh"

set +e
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  "${command[@]}" 2>&1 | tee "$PARITY_CERTIFICATE.log"
# Preserve both scientific-analysis and log-writing failures from the pipeline.
pipeline_status=("${PIPESTATUS[@]}")
set -e
certifier_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
if [ "$tee_status" -ne 0 ]; then
  exit "$tee_status"
fi
printf '%s\n' "$certifier_status" > "$PARITY_CERTIFICATE.exit_status.txt"
if [ "$certifier_status" -ne 0 ]; then
  echo "Strict dynamical-coordinate certification did not pass (status=$certifier_status)." >&2
  exit "$certifier_status"
fi

python3 - "$PARITY_CERTIFICATE" "$EIKONAL_SUBSTEPS" \
  "$EXPECTED_CERTIFIER_SHA256" "$EXPECTED_BUILDER_SHA256" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_eikonal_substeps = int(sys.argv[2])
expected_certifier_sha256 = sys.argv[3]
expected_builder_sha256 = sys.argv[4]
# Require a passing returned record as well as a zero process exit status.
value = json.loads(path.read_text())
if value.get("status") != "pass":
    raise SystemExit("certifier exited zero without a top-level status=pass")
if value.get("eikonal_substeps") != expected_eikonal_substeps:
    raise SystemExit("certificate does not record the submitted eikonal substeps")
if value.get("convergence", {}).get("evidence", {}).get(
        "eikonal_substeps") != expected_eikonal_substeps:
    raise SystemExit("certificate convergence audit uses different eikonal substeps")
if value.get("certifier_sha256") != expected_certifier_sha256:
    raise SystemExit("certificate records a different certifier SHA-256")
if value.get("convergence", {}).get("evidence", {}).get(
        "diagnostic_certifier_sha256") != expected_certifier_sha256:
    raise SystemExit("certificate diagnostics used a different certifier")
if value.get("convergence", {}).get("evidence", {}).get(
        "convergence_builder_sha256") != expected_builder_sha256:
    raise SystemExit("certificate evidence used a different builder")
# Bind the passing result to the production radial metric reconstruction used for the marker comparison.
expected_interpolation = {
    "schwarzschild": "production_center_face_lambda_monotone_hermite_mu",
    "eddington_finkelstein": "production_center_face_linear_a_b",
    "donor_history_format_version": 2,
    "production_equivalent": True,
}
if value.get("metric_interpolation") != expected_interpolation:
    raise SystemExit("passing certificate lacks production-equivalent metric reconstruction")
PY

# Keep the authenticated sidecar portable: its recorded name must be the
# certificate basename, not an absolute cluster path.  The threshold gate
# deliberately rejects sidecars that name a different file.
(
  cd "$(dirname "$PARITY_CERTIFICATE")"
  sha256sum "$(basename "$PARITY_CERTIFICATE")"
) > "$PARITY_CERTIFICATE.sha256"
sha256sum "$CERTIFIER" "$COORDINATE_MAP_MANIFEST" "$MARKER_CROSSINGS" \
  "$CONVERGENCE_EVIDENCE" \
  > "$PARITY_CERTIFICATE.inputs.sha256"
sha256sum "$PARITY_CERTIFICATE.inputs.sha256" \
  > "$PARITY_CERTIFICATE.inputs.sha256.sha256"
verify_identity postprocess-script "$SCRIPT_IDENTITY" "$EXPECTED_SCRIPT_SHA256"
verify_identity parity-certifier "$CERTIFIER" "$EXPECTED_CERTIFIER_SHA256"
verify_identity convergence-builder \
  "$VENUS_SOURCE_ROOT/cpu/tools/build_wpic_dynamic_coordinate_convergence.py" \
  "$EXPECTED_BUILDER_SHA256"
echo "Strict parity certificate: $PARITY_CERTIFICATE"
