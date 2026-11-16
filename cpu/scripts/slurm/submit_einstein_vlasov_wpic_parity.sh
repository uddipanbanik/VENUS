#!/bin/bash
# Schedule a Schwarzschild/EF coordinate comparison with two resolutions and two quiet-sampling realizations.
# All four pairs share one explicit continuum family and perturbation in G=c=1 units.
# Independent diagnostics measure agreement first; a later convergence calculation supplies the uncertainty required for the final certificate.

# Submit the strict solved-eikonal SC/EF parity and convergence DAG.
#
# Four coordinate pairs form the fail-closed minimum evidence matrix:
# coarse/fine resolution x deterministic quiet-sampling realizations A/B.
# The fine-A pair is the designated production comparison.  Its strict
# certificate is issued only after all four clean single-resolution audits
# show monotone refinement and the derived uncertainty envelope passes.

set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="$(pwd)"
if [ -n "${SBATCH_SCRIPT:-}${DIAGNOSTIC_SCRIPT:-}${CONVERGENCE_SCRIPT:-}${POSTPROCESS_SCRIPT:-}" ]; then
  echo "Production parity Slurm-script overrides are forbidden; scripts are pinned under VENUS_SOURCE_ROOT." >&2
  exit 1
fi
VENUS_SOURCE_ROOT="${VENUS_SOURCE_ROOT:?Set VENUS_SOURCE_ROOT to the deployed repository path}"
VENUS_RUN_ROOT="${VENUS_RUN_ROOT:?Set VENUS_RUN_ROOT to the simulation output root}"
PAIR_TAG="${PAIR_TAG:-parity_converged_$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$VENUS_RUN_ROOT/$PAIR_TAG}"
# The caller specifies the static continuum, perturbation, mass, and map
# controls after a local bound-family preflight. Empty values fail before
# the first sbatch call.
HIGH_INITIAL_CONDITION="${HIGH_INITIAL_CONDITION:?set HIGH_INITIAL_CONDITION=energy_cutoff_shell}"
ENERGY_CUTOFF_AMPLITUDE="${ENERGY_CUTOFF_AMPLITUDE:?set continuum amplitude C}"
ENERGY_CUTOFF_ELL0="${ENERGY_CUTOFF_ELL0:?set continuum ell0}"
ENERGY_CUTOFF_CENTRAL_Y="${ENERGY_CUTOFF_CENTRAL_Y:?set continuum central y}"
RADIAL_INFALL="${RADIAL_INFALL:?set the fixed inward kick, including explicit 0}"
INFALL_CORE_RADIUS="${INFALL_CORE_RADIUS:?set the fixed kick core radius, including explicit 0}"
ANGULAR_MOMENTUM_SCALE="${ANGULAR_MOMENTUM_SCALE:?set the parity angular-momentum scale}"
TARGET_INITIAL_ADM_MASS="${TARGET_INITIAL_ADM_MASS:?set the fixed ADM mass}"
PARITY_T_END="${PARITY_T_END:?set the validated parity end time}"
PARITY_EXPECTED_EF_OUTCOME="${PARITY_EXPECTED_EF_OUTCOME:?set black_hole, halo, or any after local preflight}"
PARITY_FINE_MARKERS="${PARITY_FINE_MARKERS:-65536}"
PARITY_FINE_CELLS="${PARITY_FINE_CELLS:-2048}"
PARITY_FINE_DT_MAX="${PARITY_FINE_DT_MAX:-0.004}"
PARITY_COARSE_MARKERS="${PARITY_COARSE_MARKERS:-16384}"
PARITY_COARSE_CELLS="${PARITY_COARSE_CELLS:-1024}"
PARITY_COARSE_DT_MAX="${PARITY_COARSE_DT_MAX:-0.008}"
PARITY_R_MAX="${PARITY_R_MAX:?set the evolution radius R_E}"
EF_MAP_WORLDTUBE_RADIUS="${EF_MAP_WORLDTUBE_RADIUS:?set the solved-map worldtube radius R_B}"
PARITY_REALIZATION_A="${PARITY_REALIZATION_A:?set quiet-PIC realization A}"
PARITY_REALIZATION_B="${PARITY_REALIZATION_B:?set quiet-PIC realization B}"
CONVERGENCE_SAFETY_FACTOR="${CONVERGENCE_SAFETY_FACTOR:-1.5}"
EF_NULL_DONOR_DURATION="${EF_NULL_DONOR_DURATION:?set the preflighted reverse-donor duration}"
VENUS_EIKONAL_SUBSTEPS="${VENUS_EIKONAL_SUBSTEPS:?set the explicit parity eikonal substep count}"

# These names used to select arbitrary tools inside the jobs.  Remove inherited
# values before every --export=ALL call; the job scripts independently reject
# direct overrides and invoke only the tools pinned under VENUS_SOURCE_ROOT.
unset VENUS_PARITY_CERTIFIER VENUS_CONVERGENCE_BUILDER

case "$VENUS_SOURCE_ROOT" in /*) ;; *) VENUS_SOURCE_ROOT="$SUBMIT_DIR/$VENUS_SOURCE_ROOT" ;; esac
case "$VENUS_RUN_ROOT" in /*) ;; *) VENUS_RUN_ROOT="$SUBMIT_DIR/$VENUS_RUN_ROOT" ;; esac
case "$CAMPAIGN_ROOT" in /*) ;; *) CAMPAIGN_ROOT="$SUBMIT_DIR/$CAMPAIGN_ROOT" ;; esac
PINNED_SLURM_DIR="$VENUS_SOURCE_ROOT/cpu/scripts/slurm"
SBATCH_SCRIPT="$PINNED_SLURM_DIR/ias_einstein_vlasov_wpic.sbatch"
DIAGNOSTIC_SCRIPT="$PINNED_SLURM_DIR/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch"
CONVERGENCE_SCRIPT="$PINNED_SLURM_DIR/ias_einstein_vlasov_wpic_parity_convergence.sbatch"
POSTPROCESS_SCRIPT="$PINNED_SLURM_DIR/ias_einstein_vlasov_wpic_parity_postprocess.sbatch"
PARITY_CERTIFIER="$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py"
CONVERGENCE_BUILDER="$VENUS_SOURCE_ROOT/cpu/tools/build_wpic_dynamic_coordinate_convergence.py"
ANALYZER="$VENUS_SOURCE_ROOT/cpu/tools/analyze_einstein_vlasov_wpic.py"
NULL_MAP_PREFLIGHT="$VENUS_SOURCE_ROOT/cpu/tools/validate_wpic_null_map.py"
POSITIVE_BINDING_VALIDATOR="$PINNED_SLURM_DIR/validate_einstein_vlasov_wpic_positive_binding.py"

for script in "$SBATCH_SCRIPT" "$DIAGNOSTIC_SCRIPT" "$CONVERGENCE_SCRIPT" \
              "$POSTPROCESS_SCRIPT"; do
  if [ ! -f "$script" ]; then
    echo "Required Slurm script not found: $script" >&2
    exit 1
  fi
done
for tool in "$PARITY_CERTIFIER" "$CONVERGENCE_BUILDER" "$ANALYZER" \
            "$NULL_MAP_PREFLIGHT" "$POSITIVE_BINDING_VALIDATOR"; do
  if [ ! -s "$tool" ]; then
    echo "Required pinned workflow tool is missing or empty: $tool" >&2
    exit 1
  fi
done
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
  echo "Refusing to reuse an existing parity campaign: $CAMPAIGN_ROOT" >&2
  exit 1
fi
for value in "$VENUS_SOURCE_ROOT" "$VENUS_RUN_ROOT" "$CAMPAIGN_ROOT" \
             "$SBATCH_SCRIPT" "$DIAGNOSTIC_SCRIPT" "$CONVERGENCE_SCRIPT" \
             "$POSTPROCESS_SCRIPT"; do
  case "$value" in
    *','*|*$'\n'*) echo "Slurm export paths cannot contain commas or newlines: $value" >&2; exit 1 ;;
  esac
done

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}
SUBMITTER_SHA256="$(sha256_of "${BASH_SOURCE[0]}")"
LAUNCHER_SHA256="$(sha256_of "$SBATCH_SCRIPT")"
DIAGNOSTIC_SCRIPT_SHA256="$(sha256_of "$DIAGNOSTIC_SCRIPT")"
CONVERGENCE_SCRIPT_SHA256="$(sha256_of "$CONVERGENCE_SCRIPT")"
POSTPROCESS_SCRIPT_SHA256="$(sha256_of "$POSTPROCESS_SCRIPT")"
PARITY_CERTIFIER_SHA256="$(sha256_of "$PARITY_CERTIFIER")"
CONVERGENCE_BUILDER_SHA256="$(sha256_of "$CONVERGENCE_BUILDER")"
ANALYZER_SHA256="$(sha256_of "$ANALYZER")"
NULL_MAP_PREFLIGHT_SHA256="$(sha256_of "$NULL_MAP_PREFLIGHT")"
POSITIVE_BINDING_VALIDATOR_SHA256="$(sha256_of "$POSITIVE_BINDING_VALIDATOR")"

positive_number='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
if [ "$HIGH_INITIAL_CONDITION" != energy_cutoff_shell ]; then
  echo "Parity production requires HIGH_INITIAL_CONDITION=energy_cutoff_shell; set the static energy-cutoff shell controls explicitly." >&2
  exit 1
fi
case "$PARITY_EXPECTED_EF_OUTCOME" in
  black_hole|halo|any) ;;
  *) echo "PARITY_EXPECTED_EF_OUTCOME must be black_hole, halo, or any." >&2; exit 1 ;;
esac
for assignment in \
    "ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE" \
    "ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0" \
    "ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y" \
    "TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS" \
    "PARITY_T_END=$PARITY_T_END" \
    "PARITY_R_MAX=$PARITY_R_MAX" \
    "EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS" \
    "EF_NULL_DONOR_DURATION=$EF_NULL_DONOR_DURATION" \
    "PARITY_FINE_DT_MAX=$PARITY_FINE_DT_MAX" \
    "PARITY_COARSE_DT_MAX=$PARITY_COARSE_DT_MAX" \
    "CONVERGENCE_SAFETY_FACTOR=$CONVERGENCE_SAFETY_FACTOR"; do
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
if ! [[ "$ANGULAR_MOMENTUM_SCALE" =~ $positive_number ]] \
   || ! awk -v number="$ANGULAR_MOMENTUM_SCALE" \
        'BEGIN { exit !(number > 0.0) }'; then
  echo "ANGULAR_MOMENTUM_SCALE must be strictly positive for an EF parity campaign; s_L=0 has unsupported radial center-crossing characteristics." >&2
  exit 1
fi
for assignment in \
    "PARITY_FINE_MARKERS=$PARITY_FINE_MARKERS" \
    "PARITY_FINE_CELLS=$PARITY_FINE_CELLS" \
    "PARITY_COARSE_MARKERS=$PARITY_COARSE_MARKERS" \
    "PARITY_COARSE_CELLS=$PARITY_COARSE_CELLS" \
    "PARITY_REALIZATION_A=$PARITY_REALIZATION_A" \
    "PARITY_REALIZATION_B=$PARITY_REALIZATION_B"; do
  name="${assignment%%=*}"
  value="${assignment#*=}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be an unsigned integer: $value" >&2
    exit 1
  fi
done
if ! [[ "$VENUS_EIKONAL_SUBSTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VENUS_EIKONAL_SUBSTEPS must be a positive integer: $VENUS_EIKONAL_SUBSTEPS" >&2
  exit 1
fi
if [ "$PARITY_REALIZATION_A" = "$PARITY_REALIZATION_B" ]; then
  echo "PARITY_REALIZATION_A and PARITY_REALIZATION_B must differ." >&2
  exit 1
fi
if [ $((PARITY_FINE_MARKERS % 2)) -ne 0 ] \
   || [ $((PARITY_COARSE_MARKERS % 2)) -ne 0 ]; then
  echo "energy_cutoff_shell requires even coarse and fine marker counts." >&2
  exit 1
fi
# Fine resolution must increase both particle and radial counts while reducing the timestep cap.
if [ "$PARITY_COARSE_MARKERS" -ge "$PARITY_FINE_MARKERS" ] \
   || [ "$PARITY_COARSE_CELLS" -ge "$PARITY_FINE_CELLS" ] \
   || ! awk -v coarse="$PARITY_COARSE_DT_MAX" -v fine="$PARITY_FINE_DT_MAX" \
        'BEGIN { exit !(coarse > fine) }'; then
  echo "Fine must strictly refine marker count, radial cells, and dt_max." >&2
  exit 1
fi
if ! awk -v factor="$CONVERGENCE_SAFETY_FACTOR" \
      'BEGIN { exit !(factor >= 1.5) }'; then
  echo "CONVERGENCE_SAFETY_FACTOR must be at least 1.5." >&2
  exit 1
fi
# Place the same physical worldtube on both meshes and retain complete exterior vacuum-extension cells.
if ! awk -v rb="$EF_MAP_WORLDTUBE_RADIUS" -v re="$PARITY_R_MAX" \
      -v coarse="$PARITY_COARSE_CELLS" -v fine="$PARITY_FINE_CELLS" '
    function on_face(cells, x, nearest) {
      x = rb * cells / re;
      nearest = int(x + 0.5);
      return (x-nearest < 0 ? nearest-x : x-nearest) <= 2.0e-12 * (x > 1 ? x : 1);
    }
    BEGIN {
      if (!(rb > 0 && rb < re)) exit 1;
      if (!on_face(coarse) || !on_face(fine)) exit 1;
      if ((re-rb) * coarse / re < 4.0-2.0e-12) exit 1;
      if ((re-rb) * fine / re < 4.0-2.0e-12) exit 1;
    }'; then
  echo "R_B must be an exact coarse/fine mesh face with R_B<R_E and at least four full vacuum-extension cells." >&2
  exit 1
fi

mkdir -p "$CAMPAIGN_ROOT/slurm" "$CAMPAIGN_ROOT/config" \
  "$CAMPAIGN_ROOT/pairs" "$CAMPAIGN_ROOT/diagnostics"
RECEIPT="$CAMPAIGN_ROOT/submission.tsv"
CONVERGENCE_EVIDENCE="$CAMPAIGN_ROOT/dynamic_coordinate_convergence.json"
PARITY_CERTIFICATE="$CAMPAIGN_ROOT/dynamic_coordinate_certificate.json"
# Record every physical family parameter, numerical level, realization and comparison tolerance before scheduling children.
CONTROLS_MANIFEST="$CAMPAIGN_ROOT/campaign_controls.manifest.tsv"
{
  printf 'schema\tvenus_v3_static_parity_campaign\n'
  printf 'schema_version\t1\n'
  printf 'initial_condition\t%s\n' "$HIGH_INITIAL_CONDITION"
  printf 'energy_cutoff_amplitude_C\t%s\n' "$ENERGY_CUTOFF_AMPLITUDE"
  printf 'energy_cutoff_ell0\t%s\n' "$ENERGY_CUTOFF_ELL0"
  printf 'energy_cutoff_central_y\t%s\n' "$ENERGY_CUTOFF_CENTRAL_Y"
  printf 'inward_radial_kick\t%s\n' "$RADIAL_INFALL"
  printf 'infall_core_radius\t%s\n' "$INFALL_CORE_RADIUS"
  printf 'angular_momentum_scale\t%s\n' "$ANGULAR_MOMENTUM_SCALE"
  printf 'target_initial_adm_mass\t%s\n' "$TARGET_INITIAL_ADM_MASS"
  printf 'parity_end_time\t%s\n' "$PARITY_T_END"
  printf 'expected_ef_outcome\t%s\n' "$PARITY_EXPECTED_EF_OUTCOME"
  printf 'map_worldtube_radius_R_B\t%s\n' "$EF_MAP_WORLDTUBE_RADIUS"
  printf 'evolution_outer_radius_R_E\t%s\n' "$PARITY_R_MAX"
  printf 'ef_null_donor_duration\t%s\n' "$EF_NULL_DONOR_DURATION"
  printf 'parity_eikonal_substeps\t%s\n' "$VENUS_EIKONAL_SUBSTEPS"
  printf 'submitter_sha256\t%s\n' "$SUBMITTER_SHA256"
  printf 'launcher_sha256\t%s\n' "$LAUNCHER_SHA256"
  printf 'diagnostic_script_sha256\t%s\n' "$DIAGNOSTIC_SCRIPT_SHA256"
  printf 'convergence_script_sha256\t%s\n' "$CONVERGENCE_SCRIPT_SHA256"
  printf 'postprocess_script_sha256\t%s\n' "$POSTPROCESS_SCRIPT_SHA256"
  printf 'parity_certifier_sha256\t%s\n' "$PARITY_CERTIFIER_SHA256"
  printf 'convergence_builder_sha256\t%s\n' "$CONVERGENCE_BUILDER_SHA256"
  printf 'analyzer_sha256\t%s\n' "$ANALYZER_SHA256"
  printf 'null_map_preflight_sha256\t%s\n' "$NULL_MAP_PREFLIGHT_SHA256"
  printf 'positive_binding_validator_sha256\t%s\n' "$POSITIVE_BINDING_VALIDATOR_SHA256"
  printf 'realization_a\t%s\n' "$PARITY_REALIZATION_A"
  printf 'realization_b\t%s\n' "$PARITY_REALIZATION_B"
  printf 'fine_markers\t%s\n' "$PARITY_FINE_MARKERS"
  printf 'fine_radial_cells\t%s\n' "$PARITY_FINE_CELLS"
  printf 'fine_dt_max\t%s\n' "$PARITY_FINE_DT_MAX"
  printf 'coarse_markers\t%s\n' "$PARITY_COARSE_MARKERS"
  printf 'coarse_radial_cells\t%s\n' "$PARITY_COARSE_CELLS"
  printf 'coarse_dt_max\t%s\n' "$PARITY_COARSE_DT_MAX"
  printf 'convergence_safety_factor\t%s\n' "$CONVERGENCE_SAFETY_FACTOR"
  printf 'metric_relative_error_ceiling\t0.005\n'
  printf 'stress_relative_error_ceiling\t0.02\n'
  printf 'moment_relative_error_ceiling\t0.02\n'
  printf 'distribution_relative_error_ceiling\t1.0e-10\n'
  printf 'require_positive_binding\t1\n'
} > "$CONTROLS_MANIFEST"
cp "$CONTROLS_MANIFEST" "$CAMPAIGN_ROOT/campaign_controls.txt"
controls_manifest_sha256="$(sha256sum "$CONTROLS_MANIFEST" | awk '{print $1}')"
printf '%s  %s\n' "$controls_manifest_sha256" \
  "$(basename "$CONTROLS_MANIFEST")" \
  > "$CAMPAIGN_ROOT/campaign_controls.manifest.sha256"
sha256sum "$SBATCH_SCRIPT" "$DIAGNOSTIC_SCRIPT" "$CONVERGENCE_SCRIPT" \
  "$POSTPROCESS_SCRIPT" "$PARITY_CERTIFIER" "$CONVERGENCE_BUILDER" \
  "$ANALYZER" "$NULL_MAP_PREFLIGHT" "$POSITIVE_BINDING_VALIDATOR" \
  > "$CAMPAIGN_ROOT/workflow_source.sha256"
workflow_source_sha256="$(sha256_of "$CAMPAIGN_ROOT/workflow_source.sha256")"
printf '%s  %s\n' "$workflow_source_sha256" workflow_source.sha256 \
  > "$CAMPAIGN_ROOT/workflow_source.sha256.sha256"
printf 'role\tjob_id\tdependency\tpair\tresolution\trealization\tmarkers\tradial_cells\tdt_max\tcoordinates\toutput_dir\tphysical_ic\tnull_ic\tnull_manifest\taudit\tinitial_condition\tenergy_cutoff_amplitude_C\tenergy_cutoff_ell0\tenergy_cutoff_central_y\tinward_radial_kick\tinfall_core_radius\tangular_momentum_scale\ttarget_initial_adm_mass\tmap_worldtube_radius_R_B\tevolution_outer_radius_R_E\teikonal_substeps\tcontrols_manifest_sha256\n' > "$RECEIPT"

common_args=(
  --high-t-end "$PARITY_T_END"
  --high-gap-stop 0.001
  --high-r-min 0.005
  --high-inner-boundary origin_parity
  --high-radial-grid linear
  --timestep-omitted-mass-fraction 0.00001
  --high-output-steps 1
)

append_receipt() {
  local first="$1"
  shift
  printf '%s' "$first" >> "$RECEIPT"
  printf '\t%s' "$@" >> "$RECEIPT"
  printf '\n' >> "$RECEIPT"
}

# Write one literal argv element per line, combining common physics-independent settings with each resolution.
write_args_file() {
  local path="$1"
  local markers="$2"
  local cells="$3"
  local dt_max="$4"
  {
    printf '%s\n' "${common_args[@]}"
    printf '%s\n' --high-markers "$markers"
    printf '%s\n' --high-radial-cells "$cells"
    printf '%s\n' --high-dt-max "$dt_max"
  } > "$path"
}

coarse_args="$CAMPAIGN_ROOT/config/coarse_driver_arguments.txt"
fine_args="$CAMPAIGN_ROOT/config/fine_driver_arguments.txt"
write_args_file "$coarse_args" "$PARITY_COARSE_MARKERS" \
  "$PARITY_COARSE_CELLS" "$PARITY_COARSE_DT_MAX"
write_args_file "$fine_args" "$PARITY_FINE_MARKERS" \
  "$PARITY_FINE_CELLS" "$PARITY_FINE_DT_MAX"

diagnostic_ids=()
build_seed_id=
production_map_gate_id=

# Fine-A is submitted first and is the sole build job and designated
# production/reference pair.  Its exact production donor is also the mapping
# setup gate: no auxiliary seed/donor fan-out starts until its authenticated
# manifest, artifacts, vacuum worldtube, unique crossings, and eikonal error
# estimates pass in the launcher post-step.
# Run fine-A first as the shared build and production-map prerequisite, then release the auxiliary resolution/realization pairs.
pair_order=(fine_a coarse_a coarse_b fine_b)
# Select the numerical level and independent quiet realization before forming that pair's input/output paths.
for pair in "${pair_order[@]}"; do
  level="${pair%%_*}"
  realization_label="${pair##*_}"
  if [ "$level" = fine ]; then
    markers="$PARITY_FINE_MARKERS"
    cells="$PARITY_FINE_CELLS"
    dt_max="$PARITY_FINE_DT_MAX"
    args_file="$fine_args"
  else
    markers="$PARITY_COARSE_MARKERS"
    cells="$PARITY_COARSE_CELLS"
    dt_max="$PARITY_COARSE_DT_MAX"
    args_file="$coarse_args"
  fi
  if [ "$realization_label" = a ]; then
    realization="$PARITY_REALIZATION_A"
  else
    realization="$PARITY_REALIZATION_B"
  fi

  pair_root="$CAMPAIGN_ROOT/pairs/$pair"
  physical_ic="$pair_root/physical_ic.bin"
  seed_output="$pair_root/ic_seed"
  sc_output="$pair_root/schwarzschild"
  ef_output="$pair_root/eddington_finkelstein"
  null_ic="$pair_root/null_map/ef_null_initial_data.bin"
  manifest="$null_ic.manifest.json"
  crossings="$null_ic.crossings.csv"
  audit="$CAMPAIGN_ROOT/diagnostics/$pair.json"
  mkdir -p "$pair_root/null_map"

  build=0
  if [ "$pair" = fine_a ]; then
    build=1
  fi
  common_export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,VENUS_RUN_ROOT=$VENUS_RUN_ROOT,VENUS_RUN_ARGS_FILE=$args_file,PHYSICAL_IC=$physical_ic,VENUS_HIGH_INITIAL_CONDITION=$HIGH_INITIAL_CONDITION,VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=$ENERGY_CUTOFF_AMPLITUDE,VENUS_HIGH_ENERGY_CUTOFF_ELL0=$ENERGY_CUTOFF_ELL0,VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=$ENERGY_CUTOFF_CENTRAL_Y,VENUS_HIGH_RADIAL_INFALL=$RADIAL_INFALL,VENUS_HIGH_INFALL_CORE_RADIUS=$INFALL_CORE_RADIUS,VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=$ANGULAR_MOMENTUM_SCALE,VENUS_HIGH_SAMPLING_REALIZATION=$realization,VENUS_HIGH_TARGET_INITIAL_ADM_MASS=$TARGET_INITIAL_ADM_MASS,VENUS_HIGH_R_MAX=$PARITY_R_MAX,VENUS_CAMPAIGN_CONTROLS_SHA256=$controls_manifest_sha256,VENUS_REQUIRE_POSITIVE_BINDING=1,VENUS_LAUNCHER_SHA256=$LAUNCHER_SHA256,VENUS_ANALYZER_SHA256=$ANALYZER_SHA256,VENUS_NULL_MAP_PREFLIGHT_SHA256=$NULL_MAP_PREFLIGHT_SHA256,VENUS_POSITIVE_BINDING_VALIDATOR_SHA256=$POSITIVE_BINDING_VALIDATOR_SHA256"
  if [ "$pair" = fine_a ]; then
    seed_id="$(sbatch --parsable \
      --job-name="venus-$pair-seed" \
      --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
      --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
      --export="$common_export,EXPECTED_OUTCOME=any,COORDINATES=schwarzschild,INITIALIZE_ONLY=1,OUTPUT_DIR=$seed_output,VENUS_BUILD=$build" \
      "$SBATCH_SCRIPT")"
  else
    seed_id="$(sbatch --parsable \
      --dependency="afterok:$production_map_gate_id" \
      --job-name="venus-$pair-seed" \
      --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
      --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
      --export="$common_export,EXPECTED_OUTCOME=any,COORDINATES=schwarzschild,INITIALIZE_ONLY=1,OUTPUT_DIR=$seed_output,VENUS_BUILD=$build" \
      "$SBATCH_SCRIPT")"
  fi
  seed_id="${seed_id%%;*}"
  if [ "$pair" = fine_a ]; then
    build_seed_id="$seed_id"
    seed_dependency_text=-
  else
    seed_dependency_text="afterok:$production_map_gate_id"
  fi
  append_receipt base_ic "$seed_id" "$seed_dependency_text" "$pair" \
    "$level" "$realization" "$markers" "$cells" "$dt_max" \
    schwarzschild "$seed_output" "$physical_ic" - - - \
    "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
    "$ENERGY_CUTOFF_ELL0" "$ENERGY_CUTOFF_CENTRAL_Y" \
    "$RADIAL_INFALL" "$INFALL_CORE_RADIUS" "$ANGULAR_MOMENTUM_SCALE" \
    "$TARGET_INITIAL_ADM_MASS" - "$PARITY_R_MAX" \
    "$VENUS_EIKONAL_SUBSTEPS" "$controls_manifest_sha256"

  # Evolve the Schwarzschild donor from its own physical realization and solve the initial-null mapping for the EF consumer.
  donor_export="$common_export,EXPECTED_OUTCOME=any,COORDINATES=schwarzschild,INITIALIZE_ONLY=0,OUTPUT_DIR=$sc_output,VENUS_BUILD=0,VENUS_EXPORT_EF_NULL_INITIAL_DATA=$null_ic,VENUS_EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS"
  if [ -n "$EF_NULL_DONOR_DURATION" ]; then
    donor_export="$donor_export,VENUS_EF_NULL_DONOR_DURATION=$EF_NULL_DONOR_DURATION"
  fi
  donor_id="$(sbatch --parsable \
    --dependency="afterok:$seed_id" \
    --job-name="venus-$pair-sc" \
    --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
    --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
    --export="$donor_export" \
    "$SBATCH_SCRIPT")"
  donor_id="${donor_id%%;*}"
  donor_role=sc_null_donor
  if [ "$pair" = fine_a ]; then
    donor_role=production_map_gate
    production_map_gate_id="$donor_id"
  fi
  append_receipt "$donor_role" "$donor_id" "afterok:$seed_id" "$pair" \
    "$level" "$realization" "$markers" "$cells" "$dt_max" \
    schwarzschild "$sc_output" "$physical_ic" "$null_ic" "$manifest" - \
    "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
    "$ENERGY_CUTOFF_ELL0" "$ENERGY_CUTOFF_CENTRAL_Y" \
    "$RADIAL_INFALL" "$INFALL_CORE_RADIUS" "$ANGULAR_MOMENTUM_SCALE" \
    "$TARGET_INITIAL_ADM_MASS" "$EF_MAP_WORLDTUBE_RADIUS" \
    "$PARITY_R_MAX" "$VENUS_EIKONAL_SUBSTEPS" \
    "$controls_manifest_sha256"

  ef_id="$(sbatch --parsable \
    --dependency="afterok:$donor_id" \
    --job-name="venus-$pair-ef" \
    --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
    --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
    --export="$common_export,EXPECTED_OUTCOME=$PARITY_EXPECTED_EF_OUTCOME,COORDINATES=eddington_finkelstein,INITIALIZE_ONLY=0,OUTPUT_DIR=$ef_output,VENUS_BUILD=0,VENUS_EF_NULL_INITIAL_DATA=$null_ic,VENUS_EF_MAP_WORLDTUBE_RADIUS=$EF_MAP_WORLDTUBE_RADIUS" \
    "$SBATCH_SCRIPT")"
  ef_id="${ef_id%%;*}"
  append_receipt ef_mapped "$ef_id" "afterok:$donor_id" "$pair" \
    "$level" "$realization" "$markers" "$cells" "$dt_max" \
    eddington_finkelstein "$ef_output" "$physical_ic" "$null_ic" \
    "$manifest" - "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
    "$ENERGY_CUTOFF_ELL0" "$ENERGY_CUTOFF_CENTRAL_Y" \
    "$RADIAL_INFALL" "$INFALL_CORE_RADIUS" "$ANGULAR_MOMENTUM_SCALE" \
    "$TARGET_INITIAL_ADM_MASS" "$EF_MAP_WORLDTUBE_RADIUS" \
    "$PARITY_R_MAX" "$VENUS_EIKONAL_SUBSTEPS" \
    "$controls_manifest_sha256"

  # Compare each coordinate pair only after its EF evolution succeeds, preserving the single-resolution diagnostic separately.
  diagnostic_id="$(sbatch --parsable \
    --dependency="afterok:$ef_id" \
    --job-name="venus-$pair-diag" \
    --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
    --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
    --export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,SCHWARZSCHILD_OUTPUT=$sc_output,EF_OUTPUT=$ef_output,COORDINATE_MAP_MANIFEST=$manifest,MARKER_CROSSINGS=$crossings,PARITY_DIAGNOSTIC=$audit,VENUS_EIKONAL_SUBSTEPS=$VENUS_EIKONAL_SUBSTEPS,VENUS_DIAGNOSTIC_SCRIPT_SHA256=$DIAGNOSTIC_SCRIPT_SHA256,VENUS_PARITY_CERTIFIER_SHA256=$PARITY_CERTIFIER_SHA256" \
    "$DIAGNOSTIC_SCRIPT")"
  diagnostic_id="${diagnostic_id%%;*}"
  append_receipt single_resolution_diagnostic "$diagnostic_id" \
    "afterok:$ef_id" "$pair" "$level" "$realization" "$markers" \
    "$cells" "$dt_max" coordinate_map "$pair_root" "$physical_ic" \
    "$null_ic" "$manifest" "$audit" "$HIGH_INITIAL_CONDITION" \
    "$ENERGY_CUTOFF_AMPLITUDE" "$ENERGY_CUTOFF_ELL0" \
    "$ENERGY_CUTOFF_CENTRAL_Y" "$RADIAL_INFALL" \
    "$INFALL_CORE_RADIUS" "$ANGULAR_MOMENTUM_SCALE" \
    "$TARGET_INITIAL_ADM_MASS" "$EF_MAP_WORLDTUBE_RADIUS" \
    "$PARITY_R_MAX" "$VENUS_EIKONAL_SUBSTEPS" \
    "$controls_manifest_sha256"

  diagnostic_ids+=("$diagnostic_id")
done

coarse_a_audit="$CAMPAIGN_ROOT/diagnostics/coarse_a.json"
coarse_b_audit="$CAMPAIGN_ROOT/diagnostics/coarse_b.json"
fine_a_audit="$CAMPAIGN_ROOT/diagnostics/fine_a.json"
fine_b_audit="$CAMPAIGN_ROOT/diagnostics/fine_b.json"
production_root="$CAMPAIGN_ROOT/pairs/fine_a"
production_sc="$production_root/schwarzschild"
production_ef="$production_root/eddington_finkelstein"
production_physical_ic="$production_root/physical_ic.bin"
production_null_ic="$production_root/null_map/ef_null_initial_data.bin"
production_manifest="$production_null_ic.manifest.json"
production_crossings="$production_null_ic.crossings.csv"

# The convergence calculation waits for all four diagnostics so resolution and realization errors can be assessed together.
diagnostic_dependency="afterok:$(IFS=:; echo "${diagnostic_ids[*]}")"
convergence_id="$(sbatch --parsable \
  --dependency="$diagnostic_dependency" \
  --job-name=venus-parity-convergence \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,COARSE_A_DIAGNOSTIC=$coarse_a_audit,COARSE_B_DIAGNOSTIC=$coarse_b_audit,FINE_A_DIAGNOSTIC=$fine_a_audit,FINE_B_DIAGNOSTIC=$fine_b_audit,CONVERGENCE_EVIDENCE=$CONVERGENCE_EVIDENCE,CONVERGENCE_SAFETY_FACTOR=$CONVERGENCE_SAFETY_FACTOR,VENUS_EIKONAL_SUBSTEPS=$VENUS_EIKONAL_SUBSTEPS,VENUS_CONVERGENCE_SCRIPT_SHA256=$CONVERGENCE_SCRIPT_SHA256,VENUS_PARITY_CERTIFIER_SHA256=$PARITY_CERTIFIER_SHA256,VENUS_CONVERGENCE_BUILDER_SHA256=$CONVERGENCE_BUILDER_SHA256" \
  "$CONVERGENCE_SCRIPT")"
convergence_id="${convergence_id%%;*}"
append_receipt convergence_evidence "$convergence_id" \
  "$diagnostic_dependency" matrix coarse+fine A+B - - - coordinate_map \
  "$CAMPAIGN_ROOT" - - - "$CONVERGENCE_EVIDENCE" \
  "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
  "$ENERGY_CUTOFF_ELL0" "$ENERGY_CUTOFF_CENTRAL_Y" "$RADIAL_INFALL" \
  "$INFALL_CORE_RADIUS" "$ANGULAR_MOMENTUM_SCALE" \
  "$TARGET_INITIAL_ADM_MASS" "$EF_MAP_WORLDTUBE_RADIUS" \
  "$PARITY_R_MAX" "$VENUS_EIKONAL_SUBSTEPS" \
  "$controls_manifest_sha256"

# Only this terminal job can issue the certificate used by threshold scans.
# Issue the production fine-A certificate only after the quantified convergence evidence has passed.
certificate_id="$(sbatch --parsable \
  --dependency="afterok:$convergence_id" \
  --job-name=venus-parity-certificate \
  --output="$CAMPAIGN_ROOT/slurm/%x-%j.out" \
  --error="$CAMPAIGN_ROOT/slurm/%x-%j.err" \
  --export="ALL,VENUS_SOURCE_ROOT=$VENUS_SOURCE_ROOT,SCHWARZSCHILD_OUTPUT=$production_sc,EF_OUTPUT=$production_ef,COORDINATE_MAP_MANIFEST=$production_manifest,MARKER_CROSSINGS=$production_crossings,CONVERGENCE_EVIDENCE=$CONVERGENCE_EVIDENCE,PARITY_CERTIFICATE=$PARITY_CERTIFICATE,VENUS_EIKONAL_SUBSTEPS=$VENUS_EIKONAL_SUBSTEPS,VENUS_POSTPROCESS_SCRIPT_SHA256=$POSTPROCESS_SCRIPT_SHA256,VENUS_PARITY_CERTIFIER_SHA256=$PARITY_CERTIFIER_SHA256,VENUS_CONVERGENCE_BUILDER_SHA256=$CONVERGENCE_BUILDER_SHA256" \
  "$POSTPROCESS_SCRIPT")"
certificate_id="${certificate_id%%;*}"
append_receipt strict_certificate "$certificate_id" \
  "afterok:$convergence_id" fine_a fine "$PARITY_REALIZATION_A" \
  "$PARITY_FINE_MARKERS" "$PARITY_FINE_CELLS" "$PARITY_FINE_DT_MAX" \
  coordinate_map "$production_sc" "$production_physical_ic" \
  "$production_null_ic" "$production_manifest" "$PARITY_CERTIFICATE" \
  "$HIGH_INITIAL_CONDITION" "$ENERGY_CUTOFF_AMPLITUDE" \
  "$ENERGY_CUTOFF_ELL0" "$ENERGY_CUTOFF_CENTRAL_Y" "$RADIAL_INFALL" \
  "$INFALL_CORE_RADIUS" "$ANGULAR_MOMENTUM_SCALE" \
  "$TARGET_INITIAL_ADM_MASS" "$EF_MAP_WORLDTUBE_RADIUS" \
  "$PARITY_R_MAX" "$VENUS_EIKONAL_SUBSTEPS" \
  "$controls_manifest_sha256"

# Close the full dependency and physical-input table with its own checksum.
receipt_sha256="$(sha256sum "$RECEIPT" | awk '{print $1}')"
printf '%s  %s\n' "$receipt_sha256" "$(basename "$RECEIPT")" \
  > "$RECEIPT.sha256"

cat <<EOF
Solved-eikonal parity/convergence campaign: $CAMPAIGN_ROOT
Build + production fine-A seed:          $build_seed_id
Authenticated production map gate:       $production_map_gate_id
Single-resolution diagnostic jobs:       ${diagnostic_ids[*]}
Convergence-evidence job:                 $convergence_id
Strict fine-A certificate job:            $certificate_id
Fine-A SC output:                         $production_sc
Fine-A EF output:                         $production_ef
Convergence evidence:                     $CONVERGENCE_EVIDENCE
Certificate:                              $PARITY_CERTIFICATE
Target initial ADM mass:                  $TARGET_INITIAL_ADM_MASS
Static continuum (C,ell0,yc):            $ENERGY_CUTOFF_AMPLITUDE $ENERGY_CUTOFF_ELL0 $ENERGY_CUTOFF_CENTRAL_Y
Fixed inward kick/core:                   $RADIAL_INFALL $INFALL_CORE_RADIUS
Map/evolution radii (R_B,R_E):            $EF_MAP_WORLDTUBE_RADIUS $PARITY_R_MAX
Controls manifest SHA-256:                $controls_manifest_sha256
Eikonal RK4 substeps:                     $VENUS_EIKONAL_SUBSTEPS
Receipt:                                  $RECEIPT
EOF
