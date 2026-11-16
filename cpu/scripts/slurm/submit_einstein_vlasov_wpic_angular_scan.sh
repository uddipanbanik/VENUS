#!/bin/bash
# Schedule a fixed-ADM angular-momentum family from an authenticated coordinate-parity calculation.
# Each member shares the continuum distribution and prescribed radial perturbation, while its angular-momentum scale changes.
# Every EF member receives its own Schwarzschild donor and solved null map because its matter distribution and metric change together.

# Submit a neutral, fixed-ADM static-shell s_L threshold family.  A strict
# solved-eikonal parity
# certificate is mandatory.  Each EF scan point gets its own SC donor and
# mapped null artifact because changing L changes both matter and geometry.

set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="$(pwd)"
if [ -n "${SBATCH_SCRIPT:-}${VENUS_ANALYZER:-}${VENUS_THRESHOLD_PROVENANCE_VALIDATOR:-}" ]; then
  echo "Production threshold workflow overrides are forbidden; scripts and validators are pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi
VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?Set VENUS_SOURCE_ROOT to the deployed repository path}"
VENUS_RUN_ROOT="${VENUS_RUN_ROOT:?Set VENUS_RUN_ROOT to the simulation output root}"
EXECUTABLE="${EXECUTABLE:-$VENUS_SOURCE_ROOT/cpu/venus_cpu_einstein_vlasov_wpic_black_hole}"
PROVENANCE_PYTHON="${VENUS_THRESHOLD_PROVENANCE_PYTHON:-python3}"
SCAN_TAG="${SCAN_TAG:-angular_scan_$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$VENUS_RUN_ROOT/$SCAN_TAG}"
COORDINATES="${COORDINATES:-eddington_finkelstein}"
# Require an explicitly preflighted static continuum, one fixed kick, and an
# explicit scan grid before submitting the threshold family.
HIGH_INITIAL_CONDITION="${HIGH_INITIAL_CONDITION:?set HIGH_INITIAL_CONDITION=energy_cutoff_shell}"
ENERGY_CUTOFF_AMPLITUDE="${ENERGY_CUTOFF_AMPLITUDE:?set continuum amplitude C}"
ENERGY_CUTOFF_ELL0="${ENERGY_CUTOFF_ELL0:?set continuum ell0}"
ENERGY_CUTOFF_CENTRAL_Y="${ENERGY_CUTOFF_CENTRAL_Y:?set continuum central y}"
RADIAL_INFALL="${RADIAL_INFALL:?set the fixed inward kick, including explicit 0}"
INFALL_CORE_RADIUS="${INFALL_CORE_RADIUS:?set the fixed kick core radius, including explicit 0}"
ANGULAR_MOMENTUM_SCALES="${ANGULAR_MOMENTUM_SCALES:?set at least two dimensionless s_L values}"
TARGET_INITIAL_ADM_MASS="${TARGET_INITIAL_ADM_MASS:?set the fixed ADM mass}"
SAMPLING_REALIZATION="${SAMPLING_REALIZATION:?set the quiet-PIC realization}"
SCAN_MARKERS="${SCAN_MARKERS:-512000}"
SCAN_RADIAL_CELLS="${SCAN_RADIAL_CELLS:-8192}"
SCAN_R_MAX="${SCAN_R_MAX:?set the evolution radius R_E}"
SCAN_T_END="${SCAN_T_END:?set the validated scan end time}"
SCAN_DT_MAX="${SCAN_DT_MAX:-0.001}"
SCAN_OUTPUT_STEPS="${SCAN_OUTPUT_STEPS:-50}"
# This member is submitted first. Its successful authenticated null-map
# postflight gates every other donor; it is a scale, not a claimed threshold.
NULL_DOMAIN_GATE_SCALE="${NULL_DOMAIN_GATE_SCALE:?set one member of ANGULAR_MOMENTUM_SCALES}"
PARITY_CERTIFICATE="${PARITY_CERTIFICATE:?set PARITY_CERTIFICATE to a strict status=pass certificate}"
PARITY_CERTIFICATE_SHA256="${PARITY_CERTIFICATE_SHA256:-}"
EF_MAP_WORLDTUBE_RADIUS="${EF_MAP_WORLDTUBE_RADIUS:-}"
EF_NULL_DONOR_DURATION="${EF_NULL_DONOR_DURATION:-}"

case "$VENUS_SOURCE_ROOT" in /*) ;; *) VENUS_SOURCE_ROOT="$SUBMIT_DIR/$VENUS_SOURCE_ROOT" ;; esac
case "$VENUS_RUN_ROOT" in /*) ;; *) VENUS_RUN_ROOT="$SUBMIT_DIR/$VENUS_RUN_ROOT" ;; esac
case "$CAMPAIGN_ROOT" in /*) ;; *) CAMPAIGN_ROOT="$SUBMIT_DIR/$CAMPAIGN_ROOT" ;; esac
SBATCH_SCRIPT="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/ias_einstein_vlasov_wpic.sbatch"
ANALYZER="$VENUS_SOURCE_ROOT/cpu/tools/analyze_einstein_vlasov_wpic.py"
NULL_MAP_PREFLIGHT="$VENUS_SOURCE_ROOT/cpu/tools/validate_wpic_null_map.py"
POSITIVE_BINDING_VALIDATOR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py"
PROVENANCE_VALIDATOR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py"
case "$PARITY_CERTIFICATE" in /*) ;; *) PARITY_CERTIFICATE="$SUBMIT_DIR/$PARITY_CERTIFICATE" ;; esac
case "$EXECUTABLE" in /*) ;; *) EXECUTABLE="$SUBMIT_DIR/$EXECUTABLE" ;; esac
case "$COORDINATES" in schwarzschild|eddington_finkelstein) ;;
  *) echo "Invalid COORDINATES=$COORDINATES" >&2; exit 1 ;;
esac
if [ "$HIGH_INITIAL_CONDITION" != energy_cutoff_shell ]; then
  echo "Threshold production requires HIGH_INITIAL_CONDITION=energy_cutoff_shell; set the static energy-cutoff shell controls explicitly." >&2
  exit 1
fi
if [ "$COORDINATES" = eddington_finkelstein ]; then
  if [ -z "$EF_MAP_WORLDTUBE_RADIUS" ] \
     || [ -z "$EF_NULL_DONOR_DURATION" ]; then
    echo "An EF threshold scan requires explicit EF_MAP_WORLDTUBE_RADIUS=R_B and EF_NULL_DONOR_DURATION." >&2
    exit 1
  fi
elif [ -n "$EF_MAP_WORLDTUBE_RADIUS" ] \
     || [ -n "$EF_NULL_DONOR_DURATION" ]; then
  echo "EF map controls are invalid for a Schwarzschild-only scan." >&2
  exit 1
fi

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "Slurm launcher not found: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$VENUS_SOURCE_ROOT/cpu/Makefile" ] \
   && [ ! -f "$VENUS_SOURCE_ROOT/Makefile" ]; then
  echo "VENUS_SOURCE_ROOT is not a deployed VENUS checkout: $VENUS_SOURCE_ROOT" >&2
  exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on an IAS login node." >&2
  exit 1
fi
if [ -e "$CAMPAIGN_ROOT" ]; then
  echo "Refusing to reuse an existing angular scan: $CAMPAIGN_ROOT" >&2
  exit 1
fi
if [ ! -s "$PARITY_CERTIFICATE" ]; then
  echo "Strict parity certificate is missing or empty: $PARITY_CERTIFICATE" >&2
  exit 1
fi
if [ ! -f "$ANALYZER" ]; then
  echo "Threshold/parity analyzer is missing: $ANALYZER" >&2
  exit 1
fi
for tool in "$NULL_MAP_PREFLIGHT" "$POSITIVE_BINDING_VALIDATOR" \
            "$PROVENANCE_VALIDATOR"; do
  if [ ! -s "$tool" ]; then
    echo "Pinned threshold workflow tool is missing or empty: $tool" >&2
    exit 1
  fi
done

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}
LAUNCHER_SHA256="$(sha256_of "$SBATCH_SCRIPT")"
ANALYZER_SHA256="$(sha256_of "$ANALYZER")"
NULL_MAP_PREFLIGHT_SHA256="$(sha256_of "$NULL_MAP_PREFLIGHT")"
POSITIVE_BINDING_VALIDATOR_SHA256="$(sha256_of "$POSITIVE_BINDING_VALIDATOR")"
PROVENANCE_VALIDATOR_SHA256="$(sha256_of "$PROVENANCE_VALIDATOR")"

verify_identity() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local actual
  actual="$(sha256_of "$path")"
  if [ "$actual" != "$expected" ]; then
    echo "$label SHA-256 changed during threshold setup: expected=$expected actual=$actual" >&2
    exit 1
  fi
}
verify_identity analyzer "$ANALYZER" "$ANALYZER_SHA256"
verify_identity threshold-provenance-validator \
  "$PROVENANCE_VALIDATOR" "$PROVENANCE_VALIDATOR_SHA256"
if [ ! -x "$EXECUTABLE" ]; then
  echo "Certified threshold executable is missing or not executable: $EXECUTABLE" >&2
  exit 1
fi
if ! command -v "$PROVENANCE_PYTHON" >/dev/null 2>&1; then
  echo "Threshold provenance Python is unavailable: $PROVENANCE_PYTHON" >&2
  exit 1
fi
for value in "$VENUS_SOURCE_ROOT" "$VENUS_RUN_ROOT" "$CAMPAIGN_ROOT" \
             "$SBATCH_SCRIPT" "$PARITY_CERTIFICATE" "$EXECUTABLE" \
             "$PROVENANCE_VALIDATOR"; do
  case "$value" in
    *','*|*$'\n'*) echo "Slurm export paths cannot contain commas or newlines: $value" >&2; exit 1 ;;
  esac
done

# Fail before the first sbatch call unless the analyzer authenticates the full
# v2 certificate, its sidecar/explicit digest, manifest, artifacts, strict
# empirical distribution, inverse/composition errors, and authenticated
# two-resolution/two-realization convergence evidence.
python3 - "$PARITY_CERTIFICATE" "$PARITY_CERTIFICATE_SHA256" "$ANALYZER" <<'PY'
import importlib.util
import types
import sys

spec = importlib.util.spec_from_file_location("venus_threshold_analyzer", sys.argv[3])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
gate = module.parity_certificate_gate(types.SimpleNamespace(
    parity_certificate=sys.argv[1],
    parity_certificate_sha256=sys.argv[2] or None,
))
if gate.get("status") != "pass":
    raise SystemExit("threshold submission requires strict parity status=pass")
PY
parity_certificate_sha256="$(sha256sum "$PARITY_CERTIFICATE" | awk '{print $1}')"

# Parse the explicitly requested angular scales only after the strict numerical coordinate certificate passes.
read -r -a scales <<< "$ANGULAR_MOMENTUM_SCALES"
if [ "${#scales[@]}" -lt 2 ]; then
  echo "ANGULAR_MOMENTUM_SCALES must contain at least two non-negative values." >&2
  exit 1
fi
positive_number='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
for assignment in \
    "ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE" \
    "ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0" \
    "ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y" \
    "TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS" \
    "SCAN_R_MAX=$SCAN_R_MAX" \
    "SCAN_T_END=$SCAN_T_END" \
    "SCAN_DT_MAX=$SCAN_DT_MAX"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ $positive_number ]] \
     || ! awk -v number="$value" 'BEGIN { exit !(number > 0.0) }'; then
    echo "$name must be positive: $value" >&2
    exit 1
  fi
done
for assignment in \
    "RADIAL_INFALL=$RADIAL_INFALL" \
    "INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ $positive_number ]] \
     || ! awk -v number="$value" 'BEGIN { exit !(number >= 0.0) }'; then
    echo "$name must be finite and non-negative: $value" >&2
    exit 1
  fi
done
if ! [[ "$NULL_DOMAIN_GATE_SCALE" =~ $positive_number ]] \
   || ! awk -v number="$NULL_DOMAIN_GATE_SCALE" \
        'BEGIN { exit !(number > 0.0) }'; then
  echo "NULL_DOMAIN_GATE_SCALE must be strictly positive for an EF scan." >&2
  exit 1
fi
if [ "$COORDINATES" = eddington_finkelstein ]; then
  for assignment in \
      "EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS" \
      "EF_NULL_DONOR_DURATION=$EF_NULL_DONOR_DURATION"; do
    name="${assignment%%=*}"
    value="${assignment#*=}"
    if ! [[ "$value" =~ $positive_number ]] \
       || ! awk -v number="$value" 'BEGIN { exit !(number > 0.0) }'; then
      echo "$name must be positive: $value" >&2
      exit 1
    fi
  done
fi
if ! [[ "$SAMPLING_REALIZATION" =~ ^[0-9]+$ ]]; then
  echo "SAMPLING_REALIZATION must be an unsigned integer." >&2
  exit 1
fi
for assignment in \
    "SCAN_MARKERS=$SCAN_MARKERS" \
    "SCAN_RADIAL_CELLS=$SCAN_RADIAL_CELLS" \
    "SCAN_OUTPUT_STEPS=$SCAN_OUTPUT_STEPS"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer: $value" >&2
    exit 1
  fi
done
if [ $((SCAN_MARKERS % 2)) -ne 0 ]; then
  echo "energy_cutoff_shell requires an even SCAN_MARKERS count." >&2
  exit 1
fi

# Check numerical uniqueness and safe distinct output tags separately for every angular scale.
configured_ell0_products=()
validated_scales=()
target_tags=()
target_tags_seen="|"
for scale in "${scales[@]}"; do
  if ! [[ "$scale" =~ $positive_number ]] \
     || ! awk -v value="$scale" 'BEGIN { exit !(value > 0.0) }'; then
    echo "Invalid angular-momentum scale $scale: EF production requires s_L>0 because radial center crossing is unsupported." >&2
    exit 1
  fi
  if [ "${#validated_scales[@]}" -gt 0 ]; then
    for prior_scale in "${validated_scales[@]}"; do
      if awk -v left="$scale" -v right="$prior_scale" '
           BEGIN {
             norm = left > right ? left : right;
             if (norm < 1.0) norm = 1.0;
             delta = left-right < 0 ? right-left : left-right;
             exit !(delta <= 1.0e-12 * norm);
           }'; then
        echo "ANGULAR_MOMENTUM_SCALES contains duplicate numerical values: $prior_scale and $scale" >&2
        exit 1
      fi
    done
  fi
  # Record the base angular-momentum scale times the dimensionless multiplier as a physical control.
  configured_ell0_product="$(awk -v base="$ENERGY_CUTOFF_ELL0" \
    -v multiplier="$scale" 'BEGIN { printf "%.12g", base * multiplier }')"
  target_tag="${scale//-/m}"
  target_tag="${target_tag//+/p}"
  target_tag="${target_tag//./p}"
  target_tag="${target_tag//E/e}"
  case "$target_tags_seen" in
    *"|$target_tag|"*)
      echo "ANGULAR_MOMENTUM_SCALES produces a duplicate scale tag for s_L=$scale" >&2
      exit 1
      ;;
  esac
  target_tags_seen="$target_tags_seen$target_tag|"
  validated_scales+=("$scale")
  configured_ell0_products+=("$configured_ell0_product")
  target_tags+=("$target_tag")
done

# Select the declared null-domain test member first so all other donors can depend on its successful mapping.
scan_order=("${!scales[@]}")
null_domain_gate_index=
if [ "$COORDINATES" = eddington_finkelstein ]; then
  for index in "${!scales[@]}"; do
    if awk -v value="${scales[$index]}" \
          -v gate="$NULL_DOMAIN_GATE_SCALE" \
          'BEGIN {
             scale = (value < 0 ? -value : value);
             gate_scale = (gate < 0 ? -gate : gate);
             if (gate_scale > scale) scale = gate_scale;
             if (scale < 1.0) scale = 1.0;
             delta = value - gate;
             if (delta < 0) delta = -delta;
             exit !(delta <= 1.0e-12 * scale);
           }'; then
      null_domain_gate_index="$index"
      break
    fi
  done
  if [ -z "$null_domain_gate_index" ]; then
    echo "The EF scan must contain NULL_DOMAIN_GATE_SCALE=$NULL_DOMAIN_GATE_SCALE." >&2
    exit 1
  fi
  # Submit the gate first even if the caller supplied the scan values in a
  # different order; the remaining donors can then depend on its success.
  scan_order=("$null_domain_gate_index")
  for index in "${!scales[@]}"; do
    if [ "$index" != "$null_domain_gate_index" ]; then
      scan_order+=("$index")
    fi
  done
fi
if [ "$COORDINATES" = eddington_finkelstein ] \
   && ! awk -v rb="$EF_MAP_WORLDTUBE_RADIUS" -v re="$SCAN_R_MAX" \
        -v cells="$SCAN_RADIAL_CELLS" '
      BEGIN {
        if (!(rb > 0 && rb < re)) exit 1;
        x = rb * cells / re;
        nearest = int(x + 0.5);
        error = x-nearest < 0 ? nearest-x : x-nearest;
        if (error > 2.0e-12 * (x > 1 ? x : 1)) exit 1;
        if ((re-rb) * cells / re < 4.0-2.0e-12) exit 1;
      }'; then
  echo "R_B must be an exact scan-mesh face with R_B<R_E and at least four full vacuum-extension cells." >&2
  exit 1
fi

# The numerical certificate gate above proves coordinate equivalence.  This
# second, semantic gate binds that certificate to the exact source manifest,
# executable, static continuum/fixed kick, and R_B/R_E/donor controls about to
# be scanned.  Resolution, realization, cadence, end time, and s_L may differ
# by design.  Its attestation is retained in the immutable campaign root.
# Bind the certificate to this source, executable and continuum family in addition to its numerical comparison criteria.
provenance_attestation="$(mktemp "${TMPDIR:-/tmp}/venus-threshold-provenance.XXXXXX")"
cleanup_provenance_attestation() {
  if [ -n "${provenance_attestation:-}" ] \
     && [ -f "$provenance_attestation" ]; then
    rm -f "$provenance_attestation"
  fi
}
trap cleanup_provenance_attestation EXIT
provenance_command=(
  "$PROVENANCE_PYTHON" "$PROVENANCE_VALIDATOR"
  --certificate "$PARITY_CERTIFICATE"
  --source-root "$VENUS_SOURCE_ROOT"
  --executable "$EXECUTABLE"
  --coordinates "$COORDINATES"
  --initial-condition "$HIGH_INITIAL_CONDITION"
  --amplitude "$ENERGY_CUTOFF_AMPLITUDE"
  --ell0 "$ENERGY_CUTOFF_ELL0"
  --central-y "$ENERGY_CUTOFF_CENTRAL_Y"
  --radial-infall "$RADIAL_INFALL"
  --infall-core-radius "$INFALL_CORE_RADIUS"
  --target-initial-adm-mass "$TARGET_INITIAL_ADM_MASS"
  --angular-momentum-scales "$ANGULAR_MOMENTUM_SCALES"
  --evolution-outer-radius "$SCAN_R_MAX"
  --output "$provenance_attestation"
)
if [ "$COORDINATES" = eddington_finkelstein ]; then
  provenance_command+=(
    --map-worldtube-radius "$EF_MAP_WORLDTUBE_RADIUS"
    --ef-null-donor-duration "$EF_NULL_DONOR_DURATION"
  )
fi
"${provenance_command[@]}"
verify_identity launcher "$SBATCH_SCRIPT" "$LAUNCHER_SHA256"
verify_identity analyzer "$ANALYZER" "$ANALYZER_SHA256"
verify_identity null-map-preflight \
  "$NULL_MAP_PREFLIGHT" "$NULL_MAP_PREFLIGHT_SHA256"
verify_identity positive-binding-validator \
  "$POSITIVE_BINDING_VALIDATOR" "$POSITIVE_BINDING_VALIDATOR_SHA256"
verify_identity threshold-provenance-validator \
  "$PROVENANCE_VALIDATOR" "$PROVENANCE_VALIDATOR_SHA256"

mkdir -p "$CAMPAIGN_ROOT/slurm" "$CAMPAIGN_ROOT/null_maps"
mv "$provenance_attestation" \
  "$CAMPAIGN_ROOT/parity_provenance_attestation.json"
provenance_attestation=
parity_provenance_sha256="$(sha256sum \
  "$CAMPAIGN_ROOT/parity_provenance_attestation.json" | awk '{print $1}')"
printf '%s  %s\n' "$parity_provenance_sha256" \
  parity_provenance_attestation.json \
  > "$CAMPAIGN_ROOT/parity_provenance_attestation.sha256"
# Shared numerical arguments and physical controls are recorded once before defining the individual family members.
COMMON_ARGS_FILE="$CAMPAIGN_ROOT/common_driver_arguments.txt"
PHYSICAL_IC="$CAMPAIGN_ROOT/common_base_physical_ic.bin"
RECEIPT="$CAMPAIGN_ROOT/submission.tsv"
CONTROLS_MANIFEST="$CAMPAIGN_ROOT/campaign_controls.manifest.tsv"
{
  printf 'schema\tvenus_v3_static_angular_momentum_scan\n'
  printf 'schema_version\t2\n'
  printf 'initial_condition\t%s\n' "$HIGH_INITIAL_CONDITION"
  printf 'energy_cutoff_amplitude_C\t%s\n' "$ENERGY_CUTOFF_AMPLITUDE"
  printf 'energy_cutoff_ell0\t%s\n' "$ENERGY_CUTOFF_ELL0"
  printf 'energy_cutoff_central_y\t%s\n' "$ENERGY_CUTOFF_CENTRAL_Y"
  printf 'inward_radial_kick\t%s\n' "$RADIAL_INFALL"
  printf 'infall_core_radius\t%s\n' "$INFALL_CORE_RADIUS"
  printf 'angular_momentum_scales\t%s\n' "$ANGULAR_MOMENTUM_SCALES"
  printf 'target_initial_adm_mass\t%s\n' "$TARGET_INITIAL_ADM_MASS"
  printf 'sampling_realization\t%s\n' "$SAMPLING_REALIZATION"
  printf 'markers\t%s\n' "$SCAN_MARKERS"
  printf 'radial_cells\t%s\n' "$SCAN_RADIAL_CELLS"
  printf 'map_worldtube_radius_R_B\t%s\n' "${EF_MAP_WORLDTUBE_RADIUS:--}"
  printf 'evolution_outer_radius_R_E\t%s\n' "$SCAN_R_MAX"
  printf 'end_time\t%s\n' "$SCAN_T_END"
  printf 'dt_max\t%s\n' "$SCAN_DT_MAX"
  printf 'output_steps\t%s\n' "$SCAN_OUTPUT_STEPS"
  printf 'ef_null_donor_duration\t%s\n' "${EF_NULL_DONOR_DURATION:--}"
  printf 'null_domain_gate_scale\t%s\n' "$NULL_DOMAIN_GATE_SCALE"
  printf 'parity_certificate_sha256\t%s\n' "$parity_certificate_sha256"
  printf 'parity_semantic_provenance_sha256\t%s\n' "$parity_provenance_sha256"
  printf 'certified_binary_sha256\t%s\n' "$(sha256sum "$EXECUTABLE" | awk '{print $1}')"
  printf 'launcher_sha256\t%s\n' "$LAUNCHER_SHA256"
  printf 'analyzer_sha256\t%s\n' "$ANALYZER_SHA256"
  printf 'null_map_preflight_sha256\t%s\n' "$NULL_MAP_PREFLIGHT_SHA256"
  printf 'positive_binding_validator_sha256\t%s\n' \
    "$POSITIVE_BINDING_VALIDATOR_SHA256"
  printf 'threshold_provenance_validator_sha256\t%s\n' \
    "$PROVENANCE_VALIDATOR_SHA256"
  printf 'require_positive_binding\t1\n'
} > "$CONTROLS_MANIFEST"
cp "$CONTROLS_MANIFEST" "$CAMPAIGN_ROOT/campaign_controls.txt"
controls_manifest_sha256="$(sha256sum "$CONTROLS_MANIFEST" | awk '{print $1}')"
printf '%s  %s\n' "$controls_manifest_sha256" \
  "$(basename "$CONTROLS_MANIFEST")" \
  > "$CAMPAIGN_ROOT/campaign_controls.manifest.sha256"
cp "$PARITY_CERTIFICATE" "$CAMPAIGN_ROOT/validated_parity_certificate.json"
(
  cd "$CAMPAIGN_ROOT"
  sha256sum validated_parity_certificate.json
) > "$CAMPAIGN_ROOT/validated_parity_certificate.sha256"

common_args=(
  --high-markers "$SCAN_MARKERS"
  --high-radial-cells "$SCAN_RADIAL_CELLS"
  --high-t-end "$SCAN_T_END"
  --high-dt-max "$SCAN_DT_MAX"
  --high-gap-stop 0.001
  --high-r-min 0.005
  --high-inner-boundary origin_parity
  --high-radial-grid linear
  --timestep-omitted-mass-fraction 0.00001
  --high-output-steps "$SCAN_OUTPUT_STEPS"
)
printf '%s\n' "${common_args[@]}" > "$COMMON_ARGS_FILE"
printf 'role\tjob_id\tdependency\tangular_momentum_scale\tconfigured_energy_cutoff_ell0\tconfigured_ell0_times_scale\tcoordinates\toutput_dir\tphysical_ic\tnull_ic\tnull_manifest\tinitial_condition\tenergy_cutoff_amplitude_C\tenergy_cutoff_central_y\tinward_radial_kick\tinfall_core_radius\ttarget_initial_adm_mass\tsampling_realization\tmap_worldtube_radius_R_B\tevolution_outer_radius_R_E\tparity_certificate_sha256\tcontrols_manifest_sha256\n' > "$RECEIPT"

append_receipt() {
  local first="$1"
  shift
  printf '%s' "$first" >> "$RECEIPT"
  printf '\t%s' "$@" >> "$RECEIPT"
  printf '\n' >> "$RECEIPT"
}

# Build and seed once. Every point starts from this immutable coordinate-neutral
# marker family, then gets its own initialize-only bound/setup gate before any
# long donor/evolution. Each EF angular control also gets an independent map.
# Create one coordinate-neutral particle realization for the family and validate its positive-binding setup.
seed_output="$CAMPAIGN_ROOT/ic_seed"
identity_export="VENUS_LAUNCHER_SHA256=$LAUNCHER_SHA256,VENUS_ANALYZER_SHA256=$ANALYZER_SHA256,VENUS_NULL_MAP_PREFLIGHT_SHA256=$NULL_MAP_PREFLIGHT_SHA256,VENUS_POSITIVE_BINDING_VALIDATOR_SHA256=$POSITIVE_BINDING_VALIDATOR_SHA256"
seed_export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,VENUS_RUN_ROOT=$VENUS_RUN_ROOT,VENUS_RUN_ARGS_FILE=$COMMON_ARGS_FILE,EXECUTABLE=$EXECUTABLE,PHYSICAL_IC=$PHYSICAL_IC,OUTPUT_DIR=$seed_output,COORDINATES=schwarzschild,EXPECTED_OUTCOME=any,INITIALIZE_ONLY=1,VENUS_BUILD=0,VENUS_HIGH_INITIAL_CONDITION=$HIGH_INITIAL_CONDITION,VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE,VENUS_HIGH_ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0,VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y,VENUS_HIGH_RADIAL_INFALL=$RADIAL_INFALL,VENUS_HIGH_INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS,VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=1,VENUS_HIGH_SAMPLING_REALIZATION=$SAMPLING_REALIZATION,VENUS_HIGH_TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS,VENUS_HIGH_R_MAX=$SCAN_R_MAX,VENUS_CAMPAIGN_CONTROLS_SHA256=$controls_manifest_sha256,VENUS_PARITY_CERTIFICATE_SHA256=$parity_certificate_sha256,VENUS_REQUIRE_POSITIVE_BINDING=1,$identity_export"
seed_id="$(sbatch --parsable \
  --job-name=venus-sL-base \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="$seed_export" \
  "$SBATCH_SCRIPT")"
seed_id="${seed_id%%;*}"
append_receipt base_ic "$seed_id" - 1 "$ENERGY_CUTOFF_ELL0" \
  "$ENERGY_CUTOFF_ELL0" schwarzschild "$seed_output" "$PHYSICAL_IC" \
  - - "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
  "$ENERGY_CUTOFF_CENTRAL_Y" "$RADIAL_INFALL" "$INFALL_CORE_RADIUS" \
  "$TARGET_INITIAL_ADM_MASS" "$SAMPLING_REALIZATION" - "$SCAN_R_MAX" \
  "$parity_certificate_sha256" "$controls_manifest_sha256"

null_domain_gate_id=
bound_preflight_ids=()
# Each member first passes its own bound/setup check, then obtains its donor/map and coordinate evolution.
for index in "${scan_order[@]}"; do
  scale="${scales[$index]}"
  configured_ell0_product="${configured_ell0_products[$index]}"
  target_tag="${target_tags[$index]}"
  common_export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,VENUS_RUN_ROOT=$VENUS_RUN_ROOT,VENUS_RUN_ARGS_FILE=$COMMON_ARGS_FILE,EXECUTABLE=$EXECUTABLE,PHYSICAL_IC=$PHYSICAL_IC,EXPECTED_OUTCOME=any,VENUS_BUILD=0,VENUS_HIGH_INITIAL_CONDITION=$HIGH_INITIAL_CONDITION,VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE,VENUS_HIGH_ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0,VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y,VENUS_HIGH_RADIAL_INFALL=$RADIAL_INFALL,VENUS_HIGH_INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS,VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=$scale,VENUS_HIGH_SAMPLING_REALIZATION=$SAMPLING_REALIZATION,VENUS_HIGH_TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS,VENUS_HIGH_R_MAX=$SCAN_R_MAX,VENUS_CAMPAIGN_CONTROLS_SHA256=$controls_manifest_sha256,VENUS_PARITY_CERTIFICATE_SHA256=$parity_certificate_sha256,VENUS_REQUIRE_POSITIVE_BINDING=1,$identity_export"

  point_preflight_output="$CAMPAIGN_ROOT/sL_${target_tag}_bound_preflight"
  point_preflight_id="$(sbatch --parsable \
    --dependency="afterok:$seed_id" \
    --job-name="venus-sL${target_tag}-bound" \
    --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
    --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
    --export="$common_export,OUTPUT_DIR=$point_preflight_output,COORDINATES=schwarzschild,INITIALIZE_ONLY=1" \
    "$SBATCH_SCRIPT")"
  point_preflight_id="${point_preflight_id%%;*}"
  bound_preflight_ids+=("$point_preflight_id")
  append_receipt bound_setup_preflight "$point_preflight_id" \
    "afterok:$seed_id" "$scale" "$ENERGY_CUTOFF_ELL0" \
    "$configured_ell0_product" schwarzschild "$point_preflight_output" \
    "$PHYSICAL_IC" - - "$HIGH_INITIAL_CONDITION" \
    "$ENERGY_CUTOFF_AMPLITUDE" "$ENERGY_CUTOFF_CENTRAL_Y" \
    "$RADIAL_INFALL" "$INFALL_CORE_RADIUS" "$TARGET_INITIAL_ADM_MASS" \
    "$SAMPLING_REALIZATION" - "$SCAN_R_MAX" \
    "$parity_certificate_sha256" "$controls_manifest_sha256"

  if [ "$COORDINATES" = eddington_finkelstein ]; then
    null_ic="$CAMPAIGN_ROOT/null_maps/sL_$target_tag.bin"
    null_manifest="$null_ic.manifest.json"
    donor_output="$CAMPAIGN_ROOT/sL_${target_tag}_schwarzschild_donor"
    donor_export="$common_export,OUTPUT_DIR=$donor_output,COORDINATES=schwarzschild,INITIALIZE_ONLY=0,VENUS_EXPORT_EF_NULL_INITIAL_DATA=$null_ic,VENUS_EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS"
    if [ -n "$EF_NULL_DONOR_DURATION" ]; then
      donor_export="$donor_export,VENUS_EF_NULL_DONOR_DURATION=$EF_NULL_DONOR_DURATION"
    fi
    # The designated donor establishes the common null-domain prerequisite; later donors retain their own matter-dependent maps.
    donor_role=null_donor
    donor_dependency_ids="$point_preflight_id:$null_domain_gate_id"
    if [ "$index" = "$null_domain_gate_index" ]; then
      donor_role=null_domain_gate
      donor_dependency_ids="$point_preflight_id"
    fi
    if [ -z "$donor_dependency_ids" ] \
       || [[ "$donor_dependency_ids" == *: ]]; then
      echo "Internal error: production null-domain gate was not scheduled." >&2
      exit 1
    fi
    donor_id="$(sbatch --parsable \
      --dependency="afterok:$donor_dependency_ids" \
      --job-name="venus-sL${target_tag}-donor" \
      --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
      --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
      --export="$donor_export" \
      "$SBATCH_SCRIPT")"
    donor_id="${donor_id%%;*}"
    if [ "$index" = "$null_domain_gate_index" ]; then
      null_domain_gate_id="$donor_id"
    fi
    append_receipt "$donor_role" "$donor_id" \
      "afterok:$donor_dependency_ids" "$scale" "$ENERGY_CUTOFF_ELL0" \
      "$configured_ell0_product" schwarzschild "$donor_output" \
      "$PHYSICAL_IC" "$null_ic" "$null_manifest" \
      "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
      "$ENERGY_CUTOFF_CENTRAL_Y" "$RADIAL_INFALL" \
      "$INFALL_CORE_RADIUS" "$TARGET_INITIAL_ADM_MASS" \
      "$SAMPLING_REALIZATION" "$EF_MAP_WORLDTUBE_RADIUS" "$SCAN_R_MAX" \
      "$parity_certificate_sha256" "$controls_manifest_sha256"
    dependency_id="$donor_id"
    output_dir="$CAMPAIGN_ROOT/sL_$target_tag"
    run_export="$common_export,OUTPUT_DIR=$output_dir,COORDINATES=eddington_finkelstein,INITIALIZE_ONLY=0,VENUS_EF_NULL_INITIAL_DATA=$null_ic,VENUS_EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS"
  else
    null_ic=-
    null_manifest=-
    dependency_id="$point_preflight_id"
    output_dir="$CAMPAIGN_ROOT/sL_$target_tag"
    run_export="$common_export,OUTPUT_DIR=$output_dir,COORDINATES=schwarzschild,INITIALIZE_ONLY=0"
  fi

  job_id="$(sbatch --parsable \
    --dependency="afterok:$dependency_id" \
    --job-name="venus-sL$target_tag" \
    --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
    --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
    --export="$run_export" \
    "$SBATCH_SCRIPT")"
  job_id="${job_id%%;*}"
  receipt_rb=-
  if [ "$COORDINATES" = eddington_finkelstein ]; then
    receipt_rb="$EF_MAP_WORLDTUBE_RADIUS"
  fi
  # Record the actual job dependency, angular control, input paths and shared certificate for each evolution.
  append_receipt scan "$job_id" "afterok:$dependency_id" "$scale" \
    "$ENERGY_CUTOFF_ELL0" "$configured_ell0_product" "$COORDINATES" \
    "$output_dir" "$PHYSICAL_IC" "$null_ic" "$null_manifest" \
    "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
    "$ENERGY_CUTOFF_CENTRAL_Y" "$RADIAL_INFALL" \
    "$INFALL_CORE_RADIUS" "$TARGET_INITIAL_ADM_MASS" \
    "$SAMPLING_REALIZATION" "$receipt_rb" "$SCAN_R_MAX" \
    "$parity_certificate_sha256" "$controls_manifest_sha256"
  echo "s_L=$scale configured_ell0_times_scale=$configured_ell0_product scan_job=$job_id dependency=$dependency_id output=$output_dir"
done

cat <<EOF
Angular-momentum scan:  $CAMPAIGN_ROOT
Base/build job:         $seed_id
Per-s_L bound/setup gates: ${bound_preflight_ids[*]}
Authenticated s_L=$NULL_DOMAIN_GATE_SCALE null-domain gate: ${null_domain_gate_id:--}
Coordinates:            $COORDINATES
Parity certificate:     $PARITY_CERTIFICATE
Certificate SHA-256:    $parity_certificate_sha256
Target initial ADM mass: $TARGET_INITIAL_ADM_MASS
Static continuum (C,ell0,yc): $ENERGY_CUTOFF_AMPLITUDE $ENERGY_CUTOFF_ELL0 $ENERGY_CUTOFF_CENTRAL_Y
Fixed inward kick/core: $RADIAL_INFALL $INFALL_CORE_RADIUS
Map/evolution radii (R_B,R_E): ${EF_MAP_WORLDTUBE_RADIUS:--} $SCAN_R_MAX
Controls manifest SHA-256: $controls_manifest_sha256
Receipt:                $RECEIPT
EOF
