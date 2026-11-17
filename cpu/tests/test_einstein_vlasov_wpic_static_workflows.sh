#!/usr/bin/env bash
# Shell-only contract tests for the static-shell IAS launch plumbing.  Fake
# scheduler/process commands exercise argument and receipt construction without
# contacting Slurm or starting a solver.

set -euo pipefail

real_python3=$(command -v python3)
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
slurm_dir="$source_root/cpu/scripts/slurm"
launcher="$slurm_dir/ias_einstein_vlasov_wpic.sbatch"
parity_submit="$slurm_dir/submit_einstein_vlasov_wpic_parity.sh"
scan_submit="$slurm_dir/submit_einstein_vlasov_wpic_angular_scan.sh"
paired_submit="$slurm_dir/submit_einstein_vlasov_wpic_paired_long.sh"
paired_gate="$slurm_dir/ias_einstein_vlasov_wpic_paired_long_gate.sbatch"
paired_gif_submit="$slurm_dir/submit_einstein_vlasov_wpic_paired_long_gif.sh"
paired_gif_wrapper="$slurm_dir/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/venus-static-workflows.XXXXXX")

cleanup() {
  case "$scratch" in
    "${TMPDIR:-/tmp}"/venus-static-workflows.*) rm -rf "$scratch" ;;
    *) echo "Refusing unsafe test cleanup path: $scratch" >&2 ;;
  esac
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

mkdir -p "$scratch/bin" "$scratch/run"
deployed_root="$scratch/deployed/VENUS"
mkdir -p "$deployed_root/cpu/include" "$deployed_root/cpu/src" \
  "$deployed_root/cpu/tools" "$deployed_root/cpu/scripts/slurm"
printf 'all:\n\t@true\n' > "$deployed_root/cpu/Makefile"
for script in ias_einstein_vlasov_wpic.sbatch \
    ias_einstein_vlasov_wpic_parity_diagnostic.sbatch \
    ias_einstein_vlasov_wpic_parity_convergence.sbatch \
    ias_einstein_vlasov_wpic_parity_postprocess.sbatch \
    ias_einstein_vlasov_wpic_paired_long_gate.sbatch \
    submit_einstein_vlasov_wpic_parity.sh \
    submit_einstein_vlasov_wpic_angular_scan.sh \
    submit_einstein_vlasov_wpic_paired_long.sh; do
  cp "$slurm_dir/$script" "$deployed_root/cpu/scripts/slurm/$script"
done
paired_workflow_root="$scratch/deployed/paired-long-workflow"
mkdir -p "$paired_workflow_root/cpu/scripts/slurm" \
  "$paired_workflow_root/cpu/tools"
cp "$paired_gate" "$paired_workflow_root/cpu/scripts/slurm/"
cp "$paired_submit" "$paired_workflow_root/cpu/scripts/slurm/"
cp "$paired_gif_submit" "$paired_workflow_root/cpu/scripts/slurm/"
cp "$paired_gif_wrapper" "$paired_workflow_root/cpu/scripts/slurm/"
cp "$source_root/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py" \
  "$paired_workflow_root/cpu/tools/"
cp "$slurm_dir/validate_einstein_vlasov_wpic_positive_binding.py" \
  "$deployed_root/cpu/scripts/slurm/"
cp "$slurm_dir/validate_einstein_vlasov_wpic_threshold_provenance.py" \
  "$deployed_root/cpu/scripts/slurm/"
for tool in analyze_einstein_vlasov_wpic.py \
    compare_wpic_dynamic_coordinates.py \
    build_wpic_dynamic_coordinate_convergence.py \
    validate_wpic_null_map.py \
    render_weighted_einstein_vlasov_pic_profile_gifs.py \
    render_weighted_pic_pure_gamma_gifs.py; do
  printf '# dry-run fixture\n' > "$deployed_root/cpu/tools/$tool"
done
printf '1000\n' > "$scratch/sbatch-counter"
printf '#!/usr/bin/env bash\nset -euo pipefail\nfake_root=$(cd "$(dirname "$0")/.." && pwd)\n: "${SBATCH_LOG:=$fake_root/sbatch.log}"\n: "${SBATCH_COUNTER:=$fake_root/sbatch-counter}"\nprintf "%%q " "$@" >> "$SBATCH_LOG"\nprintf "\\n" >> "$SBATCH_LOG"\nvalue=$(awk "{print \\$1 + 1}" "$SBATCH_COUNTER")\nprintf "%%s\\n" "$value" > "$SBATCH_COUNTER"\nprintf "%%s\\n" "$value"\n' \
  > "$scratch/bin/sbatch"
printf '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%%q " "$@" >> "$SRUN_LOG"\nprintf "\\n" >> "$SRUN_LOG"\n' \
  > "$scratch/bin/srun"
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
  'previous=' \
  'for argument in "$@"; do' \
  '  if [ "$previous" = --output ]; then' \
  '    printf "{\"status\":\"pass\",\"issues\":[],\"eikonal_substeps\":8}\\n" > "$argument"' \
  '  fi' \
  '  previous="$argument"' \
  'done' \
  'exit 0' > "$scratch/bin/python3"
printf '#!/usr/bin/env bash\nexit 0\n' > "$scratch/fake-solver"
cp "$scratch/fake-solver" \
  "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
chmod +x "$scratch/bin/sbatch" "$scratch/bin/srun" \
  "$scratch/bin/python3" "$scratch/fake-solver" \
  "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
export PATH="$scratch/bin:$PATH"
export SBATCH_LOG="$scratch/sbatch.log"
export SBATCH_COUNTER="$scratch/sbatch-counter"
export SRUN_LOG="$scratch/srun.log"
: > "$SBATCH_LOG"
: > "$SRUN_LOG"

# A production helper must reject the former implicit Gaussian/default setup.
if env -u HIGH_INITIAL_CONDITION -u ENERGY_CUTOFF_AMPLITUDE \
    -u ENERGY_CUTOFF_ELL0 -u ENERGY_CUTOFF_CENTRAL_Y \
    bash "$parity_submit" > "$scratch/missing.out" 2>&1; then
  fail "parity helper accepted omitted static-family controls"
fi

# An exactly radial EF family is outside the advanced-null pusher's supported
# domain because its characteristics can cross r=0.  Both production helpers
# must reject s_L=0 before the first scheduler mutation.
before_zero=$(wc -l < "$SBATCH_LOG")
if env \
    VENUS_SOURCE_ROOT="$deployed_root" \
    VENUS_RUN_ROOT="$scratch/run" \
    CAMPAIGN_ROOT="$scratch/zero-parity" \
    HIGH_INITIAL_CONDITION=energy_cutoff_shell \
    ENERGY_CUTOFF_AMPLITUDE=0.5 \
    ENERGY_CUTOFF_ELL0=1.0 \
    ENERGY_CUTOFF_CENTRAL_Y=0.13 \
    RADIAL_INFALL=0.05 \
    INFALL_CORE_RADIUS=0.25 \
    ANGULAR_MOMENTUM_SCALE=0 \
    TARGET_INITIAL_ADM_MASS=0.85 \
    PARITY_T_END=1 \
    PARITY_EXPECTED_EF_OUTCOME=black_hole \
    PARITY_R_MAX=8 \
    EF_MAP_WORLDTUBE_RADIUS=6 \
    PARITY_REALIZATION_A=7 \
    PARITY_REALIZATION_B=8 \
    EF_NULL_DONOR_DURATION=2 \
    VENUS_EIKONAL_SUBSTEPS=8 \
    bash "$parity_submit" > "$scratch/zero-parity.out" 2>&1; then
  fail "parity helper accepted s_L=0"
fi
after_zero=$(wc -l < "$SBATCH_LOG")
[[ "$before_zero" = "$after_zero" ]] \
  || fail "zero-L parity input reached sbatch before failing"

parity_contract_env=(
  VENUS_SOURCE_ROOT="$deployed_root"
  VENUS_RUN_ROOT="$scratch/run"
  HIGH_INITIAL_CONDITION=energy_cutoff_shell
  ENERGY_CUTOFF_AMPLITUDE=0.5
  ENERGY_CUTOFF_ELL0=1.0
  ENERGY_CUTOFF_CENTRAL_Y=0.13
  RADIAL_INFALL=0.05
  INFALL_CORE_RADIUS=0.25
  ANGULAR_MOMENTUM_SCALE=1
  TARGET_INITIAL_ADM_MASS=0.85
  PARITY_T_END=1
  PARITY_EXPECTED_EF_OUTCOME=black_hole
  PARITY_R_MAX=8
  EF_MAP_WORLDTUBE_RADIUS=6
  PARITY_REALIZATION_A=7
  PARITY_REALIZATION_B=8
  EF_NULL_DONOR_DURATION=2
)
before_contract=$(wc -l < "$SBATCH_LOG")
if env -u VENUS_EIKONAL_SUBSTEPS "${parity_contract_env[@]}" \
    CAMPAIGN_ROOT="$scratch/missing-substeps" \
    bash "$parity_submit" > "$scratch/missing-substeps.out" 2>&1; then
  fail "parity helper accepted omitted VENUS_EIKONAL_SUBSTEPS"
fi
if env "${parity_contract_env[@]}" VENUS_EIKONAL_SUBSTEPS=0 \
    CAMPAIGN_ROOT="$scratch/zero-substeps" \
    bash "$parity_submit" > "$scratch/zero-substeps.out" 2>&1; then
  fail "parity helper accepted zero VENUS_EIKONAL_SUBSTEPS"
fi
if env "${parity_contract_env[@]}" VENUS_EIKONAL_SUBSTEPS=11 \
    SBATCH_SCRIPT="$scratch/forbidden-launcher" \
    CAMPAIGN_ROOT="$scratch/script-override" \
    bash "$parity_submit" > "$scratch/script-override.out" 2>&1; then
  fail "parity helper accepted a production Slurm-script override"
fi
after_contract=$(wc -l < "$SBATCH_LOG")
[[ "$before_contract" = "$after_contract" ]] \
  || fail "invalid substep/script contracts reached sbatch before failing"

parity_root="$scratch/parity"
env \
  VENUS_SOURCE_ROOT="$deployed_root" \
  VENUS_RUN_ROOT="$scratch/run" \
  CAMPAIGN_ROOT="$parity_root" \
  HIGH_INITIAL_CONDITION=energy_cutoff_shell \
  ENERGY_CUTOFF_AMPLITUDE=0.5 \
  ENERGY_CUTOFF_ELL0=1.0 \
  ENERGY_CUTOFF_CENTRAL_Y=0.13 \
  RADIAL_INFALL=0.05 \
  INFALL_CORE_RADIUS=0.25 \
  ANGULAR_MOMENTUM_SCALE=1 \
  TARGET_INITIAL_ADM_MASS=0.85 \
  PARITY_T_END=1 \
  PARITY_EXPECTED_EF_OUTCOME=black_hole \
  PARITY_FINE_MARKERS=200 \
  PARITY_FINE_CELLS=64 \
  PARITY_FINE_DT_MAX=0.01 \
  PARITY_COARSE_MARKERS=100 \
  PARITY_COARSE_CELLS=32 \
  PARITY_COARSE_DT_MAX=0.02 \
  PARITY_R_MAX=8 \
  EF_MAP_WORLDTUBE_RADIUS=6 \
  PARITY_REALIZATION_A=7 \
  PARITY_REALIZATION_B=8 \
  EF_NULL_DONOR_DURATION=2 \
  VENUS_EIKONAL_SUBSTEPS=11 \
  VENUS_PARITY_CERTIFIER="$scratch/forbidden-certifier" \
  VENUS_CONVERGENCE_BUILDER="$scratch/forbidden-builder" \
  bash "$parity_submit" > "$scratch/parity.out"

manifest="$parity_root/campaign_controls.manifest.tsv"
[[ -s "$manifest" ]] || fail "parity controls manifest is missing"
grep -Eq $'initial_condition\tenergy_cutoff_shell' "$manifest" \
  || fail "parity manifest omitted the static IC type"
grep -Eq $'map_worldtube_radius_R_B\t6' "$manifest" \
  || fail "parity manifest omitted R_B"
grep -Eq $'evolution_outer_radius_R_E\t8' "$manifest" \
  || fail "parity manifest omitted R_E"
grep -Eq $'parity_eikonal_substeps\t11' "$manifest" \
  || fail "parity manifest omitted the explicit eikonal substeps"
awk -F '\t' 'NF != 27 || (NR > 1 && $26 != 11) { exit 1 }' \
  "$parity_root/submission.tsv" \
  || fail "parity receipt rows do not bind their 27-column substep schema"
[[ -s "$parity_root/submission.tsv.sha256" ]] \
  || fail "parity receipt has no immutable SHA-256 sidecar"
if grep -Eq -- '--high-(shell|total-mass|specific-angular-momentum)' \
    "$parity_root/config/coarse_driver_arguments.txt" \
    "$parity_root/config/fine_driver_arguments.txt"; then
  fail "parity argument files retained a Gaussian/specific-L flag"
fi
grep -Eq 'VENUS_HIGH_INITIAL_CONDITION=energy_cutoff_shell' "$SBATCH_LOG" \
  || fail "parity exports omitted the static IC"
grep -Eq 'VENUS_EF_MAP_WORLDTUBE_RADIUS=6' "$SBATCH_LOG" \
  || fail "parity donor/consumer exports omitted R_B"
grep -Eq 'EXPECTED_OUTCOME=black_hole.*COORDINATES=eddington_finkelstein' \
  "$SBATCH_LOG" \
  || fail "parity EF jobs do not enforce the preflighted physical outcome"
grep -Eq 'VENUS_REQUIRE_POSITIVE_BINDING=1' "$SBATCH_LOG" \
  || fail "parity jobs do not require launcher-level positive binding"
substep_exports=$(grep -Eo 'VENUS_EIKONAL_SUBSTEPS=11' "$SBATCH_LOG" | wc -l | tr -d ' ')
[[ "$substep_exports" = 6 ]] \
  || fail "four diagnostics, convergence, and certificate do not share one substep export"
if grep -Eq 'VENUS_(PARITY_CERTIFIER|CONVERGENCE_BUILDER)=' "$SBATCH_LOG"; then
  fail "--export=ALL leaked an inherited production-tool override"
fi
grep -Eq 'CERTIFIER="\$VENUS_SOURCE_ROOT/cpu/tools/compare_wpic_dynamic_coordinates.py"' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_postprocess.sbatch" \
  || fail "parity certifier is not pinned under VENUS_SOURCE_ROOT"
grep -Eq 'BUILDER="\$VENUS_SOURCE_ROOT/cpu/tools/build_wpic_dynamic_coordinate_convergence.py"' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_convergence.sbatch" \
  || fail "convergence builder is not pinned under VENUS_SOURCE_ROOT"
grep -Eq -- '--metric-tolerance 0.005' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_postprocess.sbatch" \
  || fail "parity comparators do not use the 0.005 metric guardrail"
grep -Eq -- '--stress-tolerance 0.02' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_postprocess.sbatch" \
  || fail "parity comparators do not use the 0.02 stress guardrail"
grep -Eq -- '--moment-tolerance 0.02' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_postprocess.sbatch" \
  || fail "parity comparators do not use the 0.02 moment guardrail"
grep -Eq -- '--distribution-tolerance 1.0e-10' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_postprocess.sbatch" \
  || fail "parity comparators do not use strict distribution parity"
grep -Eq -- '--metric-ceiling 0.005' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_convergence.sbatch" \
  || fail "convergence evidence does not use the 0.005 metric ceiling"
grep -Eq -- '--stress-ceiling 0.02' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_convergence.sbatch" \
  || fail "convergence evidence does not use the 0.02 stress ceiling"
grep -Eq -- '--moment-ceiling 0.02' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_convergence.sbatch" \
  || fail "convergence evidence does not use the 0.02 moment ceiling"
grep -Eq -- '--eikonal-substeps "\$EIKONAL_SUBSTEPS"' \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_convergence.sbatch" \
  "$slurm_dir/ias_einstein_vlasov_wpic_parity_postprocess.sbatch" \
  || fail "the complete parity evidence chain does not pass explicit substeps"

# Direct jobs must reject both the retired override and a staged script that
# no longer matches its submit-time identity before reading scientific data.
diag_script="$deployed_root/cpu/scripts/slurm/ias_einstein_vlasov_wpic_parity_diagnostic.sbatch"
diag_certifier="$deployed_root/cpu/tools/compare_wpic_dynamic_coordinates.py"
diag_script_sha256=$(sha256sum "$diag_script" | awk '{print $1}')
diag_certifier_sha256=$(sha256sum "$diag_certifier" | awk '{print $1}')
diag_env=(
  VENUS_SOURCE_ROOT="$deployed_root"
  SCHWARZSCHILD_OUTPUT="$scratch/not-run-sc"
  EF_OUTPUT="$scratch/not-run-ef"
  COORDINATE_MAP_MANIFEST="$scratch/not-run-map"
  MARKER_CROSSINGS="$scratch/not-run-crossings"
  PARITY_DIAGNOSTIC="$scratch/not-run-diagnostic.json"
  VENUS_EIKONAL_SUBSTEPS=11
  VENUS_DIAGNOSTIC_SCRIPT_SHA256="$diag_script_sha256"
  VENUS_PARITY_CERTIFIER_SHA256="$diag_certifier_sha256"
)
if env "${diag_env[@]}" VENUS_PARITY_CERTIFIER="$scratch/forbidden" \
    bash "$diag_script" > "$scratch/diag-override.out" 2>&1; then
  fail "diagnostic accepted VENUS_PARITY_CERTIFIER override"
fi
if env "${diag_env[@]}" \
    VENUS_DIAGNOSTIC_SCRIPT_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
    bash "$diag_script" > "$scratch/diag-mutation.out" 2>&1; then
  fail "diagnostic accepted a changed staged script identity"
fi
grep -Eq 'changed after submission' "$scratch/diag-mutation.out" \
  || fail "diagnostic mutation rejection was not explicit"

# Build a minimal but internally authenticated provenance graph for the
# semantic threshold gate.  The heavyweight numerical certificate gate still
# uses the fake python command in this shell-only scheduler test.
provenance_validator="$deployed_root/cpu/scripts/slurm/validate_einstein_vlasov_wpic_threshold_provenance.py"
binding_validator="$deployed_root/cpu/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py"
"$real_python3" - "$deployed_root" "$scratch" "$provenance_validator" <<'PY'
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

source_root = pathlib.Path(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
validator_path = sys.argv[3]
spec = importlib.util.spec_from_file_location("threshold_provenance", validator_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

parity = scratch / "certified-parity"
sc_dir = parity / "schwarzschild"
ef_dir = parity / "eddington_finkelstein"
sc_dir.mkdir(parents=True)
ef_dir.mkdir(parents=True)
high_base = {
    "initial_condition": "energy_cutoff_shell",
    "inner_boundary": "origin_parity",
    "radial_grid": "linear",
    "r_min": 0.0,
    "r_max": 8.0,
    "target_initial_adm_mass": 0.85,
    "initial_radial_infall": 0.05,
    "infall_core_radius": 0.25,
    "angular_momentum_scale": 1.0,
    "markers": 100,
    "radial_cells": 32,
    "dt_max": 0.02,
    "sampling_realization": 7,
    "eikonal_boundary_radius": 6.0,
    "eikonal_evolution_outer_radius": 8.0,
    "eikonal_mapped_initial_data": True,
    "fixed_family_binding_diagnostics_available": True,
    "fixed_family_positive_binding_energy": True,
    "fixed_family_rest_weight": 0.9,
    "fixed_family_adm_mass": 0.85,
    "fixed_family_binding_energy": 0.05,
    "fixed_family_binding_fraction": 0.05 / 0.9,
    "energy_cutoff_shell": {
        "configured_amplitude_C": 0.5,
        "configured_ell0": 1.0,
        "configured_central_y": 0.13,
        "configured_cutoff_power_k": 1.0,
        "realized_cutoff_power_k": 1.0,
    },
}
summary_paths = {}
for coordinates, directory in (("schwarzschild", sc_dir),
                               ("eddington_finkelstein", ef_dir)):
    high = dict(high_base)
    high["coordinates"] = coordinates
    summary = {"coordinates": coordinates, "high_compactness": high}
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    summary_paths[coordinates] = path

map_path = parity / "ef_null_initial_data.bin.manifest.json"
null_path = parity / "ef_null_initial_data.bin"
physical_ic_path = parity / "physical_ic.bin"
null_path.write_bytes(b"synthetic mapped null initial data\n")
physical_ic_path.write_bytes(b"synthetic fine-A physical IC\n")
donor_path = parity / "ef_null_initial_data.bin.donor_history.bin"
donor_path.write_bytes(b"synthetic face-aware donor v2\n")
map_path.write_text(json.dumps({
    "schema": "venus_v3_sc_to_ef_null_map",
    "schema_version": 2,
    "donor_history_format_version": 2,
    "saved_metric_interpolation": (
        "SC center-face lambda plus monotone-Hermite mu; EF center-face a,b"
    ),
    "map_worldtube_radius": 6.0,
    "evolution_outer_radius": 8.0,
    "eikonal_outer_reverse_time": 2.0,
}, sort_keys=True) + "\n", encoding="utf-8")

inventory = module.source_inventory(str(source_root / "cpu"))
source_text = "".join("{}  {}\n".format(value, name)
                      for name, value in sorted(inventory.items()))
executable = source_root / "cpu" / "venus_cpu_einstein_vlasov_wpic_black_hole"
binary_digest = digest(executable)
records = []
for coordinates, directory in (("schwarzschild", sc_dir),
                               ("eddington_finkelstein", ef_dir)):
    source_record = directory / "source.sha256"
    source_record.write_text(source_text, encoding="utf-8")
    binary_record = directory / "binary.sha256"
    binary_record.write_text("{}  {}\n".format(binary_digest, executable),
                             encoding="utf-8")
    records.extend((
        {
            "label": "fine_a",
            "kind": "source_checksum_manifest",
            "coordinates": coordinates,
            "path": str(source_record),
            "sha256": digest(source_record),
        },
        {
            "label": "fine_a",
            "kind": "binary_checksum_record",
            "coordinates": coordinates,
            "path": str(binary_record),
            "sha256": digest(binary_record),
        },
    ))
for coordinates, kind in (("schwarzschild", "schwarzschild_summary"),
                          ("eddington_finkelstein",
                           "eddington_finkelstein_summary")):
    path = summary_paths[coordinates]
    records.append({
        "label": "fine_a", "kind": kind, "path": str(path),
        "sha256": digest(path),
    })

source_digest = digest(sc_dir / "source.sha256")
evidence_path = parity / "dynamic_coordinate_convergence.json"
certifier_digest = digest(
    source_root / "cpu" / "tools" / "compare_wpic_dynamic_coordinates.py")
builder_digest = digest(
    source_root / "cpu" / "tools" /
    "build_wpic_dynamic_coordinate_convergence.py")
evidence_path.write_text(json.dumps({
    "schema": "venus_wpic_dynamic_coordinate_convergence_v1",
    "status": "pass",
    "eikonal_substeps": 8,
    "diagnostic_certifier_sha256": certifier_digest,
    "convergence_builder_sha256": builder_digest,
    "metric_interpolation": module.PRODUCTION_METRIC_INTERPOLATION,
    "production_pair": "fine_a",
    "source_manifest_sha256": source_digest,
    "binary_sha256": binary_digest,
    "physics_configuration": {
        "initial_condition": "energy_cutoff_shell",
        "inner_boundary": "origin_parity",
        "radial_grid": "linear",
        "r_min": 0.0,
        "r_max": 8.0,
        "target_initial_adm_mass": 0.85,
        "initial_radial_infall": 0.05,
        "infall_core_radius": 0.25,
        "energy_cutoff_shell": {
            "configured_amplitude_C": 0.5,
            "configured_ell0": 1.0,
            "configured_central_y": 0.13,
        "configured_cutoff_power_k": 1.0,
        "realized_cutoff_power_k": 1.0,
            "continuum_rescaled_to_target_adm_mass": True,
            "continuum_binding_energy": 0.05,
            "continuum_binding_fraction": 0.05 / 0.9,
            "marker_weight_measure": "invariant_schwarzschild_wpic_rest_weight",
            "post_sampling_canonicalization_passes": 0,
            "configured_inward_pair_mean_kick": 0.05,
            "configured_kick_core_radius": 0.25,
        },
    },
    "artifacts": records,
}, sort_keys=True) + "\n", encoding="utf-8")
certificate_path = scratch / "certificate.json"
certificate_path.write_text(json.dumps({
    "schema": "venus_wpic_dynamic_coordinate_certificate_v2",
    "status": "pass",
    "eikonal_substeps": 8,
    "certifier_sha256": certifier_digest,
    "metric_interpolation": module.PRODUCTION_METRIC_INTERPOLATION,
    "provenance": {
        "schwarzschild_summary": str(summary_paths["schwarzschild"]),
        "eddington_finkelstein_summary": str(
            summary_paths["eddington_finkelstein"]),
        "coordinate_map_manifest": {
            "path": str(map_path),
            "actual_sha256": digest(map_path),
            "schema": "venus_v3_sc_to_ef_null_map",
            "schema_version": 2,
            "donor_history_format_version": 2,
            "saved_metric_interpolation": module.SAVED_METRIC_INTERPOLATION,
        "artifacts": {"donor_history": {
                "path": str(donor_path),
                "verified": True,
                "expected_sha256": digest(donor_path),
                "actual_sha256": digest(donor_path),
            }, "null_initial_data": {
                "path": str(null_path),
                "verified": True,
                "expected_sha256": digest(null_path),
                "actual_sha256": digest(null_path),
            }},
        },
        "dense_time_reversed_donor_history": {
            "path": str(donor_path),
            "format_version": 2,
            "production_metric_reconstruction": True,
        },
    },
    "convergence": {"evidence": {
        "path": str(evidence_path),
        "actual_sha256": digest(evidence_path),
        "eikonal_substeps": 8,
        "diagnostic_certifier_sha256": certifier_digest,
        "convergence_builder_sha256": builder_digest,
        "metric_interpolation": module.PRODUCTION_METRIC_INTERPOLATION,
    }},
}, sort_keys=True) + "\n", encoding="utf-8")
(certificate_path.parent / (certificate_path.name + ".sha256")).write_text(
    "{}  {}\n".format(digest(certificate_path), certificate_path.name),
    encoding="utf-8",
)
legacy = json.loads(certificate_path.read_text(encoding="utf-8"))
legacy["metric_interpolation"] = {
    "schwarzschild": "legacy_center_center_linear",
    "eddington_finkelstein": "legacy_center_center_linear",
    "donor_history_format_version": 1,
    "production_equivalent": False,
}
(scratch / "certificate-legacy.json").write_text(
    json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
substep_mismatch = json.loads(certificate_path.read_text(encoding="utf-8"))
substep_mismatch["eikonal_substeps"] = 4
(scratch / "certificate-substep-mismatch.json").write_text(
    json.dumps(substep_mismatch, sort_keys=True) + "\n", encoding="utf-8")
PY

provenance_args=(
  --certificate "$scratch/certificate.json"
  --source-root "$deployed_root"
  --executable "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
  --coordinates eddington_finkelstein
  --initial-condition energy_cutoff_shell
  --amplitude 0.5
  --ell0 1.0
  --central-y 0.13
  --radial-infall 0.05
  --infall-core-radius 0.25
  --target-initial-adm-mass 0.85
  --angular-momentum-scales '0.5 1'
  --map-worldtube-radius 6
  --evolution-outer-radius 8
  --ef-null-donor-duration 2
)
"$real_python3" "$provenance_validator" "${provenance_args[@]}" \
  --output "$scratch/provenance-pass.json" \
  || fail "valid parity/source/binary/family/map provenance was rejected"
paired_provenance_args=(
  --purpose paired_long
  --certificate "$scratch/certificate.json"
  --source-root "$deployed_root"
  --executable "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
  --coordinates eddington_finkelstein
  --initial-condition energy_cutoff_shell
  --amplitude 0.5
  --ell0 1.0
  --central-y 0.13
  --radial-infall 0.05
  --infall-core-radius 0.25
  --target-initial-adm-mass 0.85
  --angular-momentum-scales 1
  --markers 100
  --radial-cells 32
  --dt-max 0.02
  --sampling-realization 7
  --physical-ic "$scratch/certified-parity/physical_ic.bin"
  --null-initial-data "$scratch/certified-parity/ef_null_initial_data.bin"
  --map-worldtube-radius 6
  --evolution-outer-radius 8
  --ef-null-donor-duration 2
)
"$real_python3" "$provenance_validator" "${paired_provenance_args[@]}" \
  --output "$scratch/paired-provenance-pass.json" \
  || fail "valid exact-reference paired-long provenance was rejected"
if "$real_python3" "$provenance_validator" \
    "${paired_provenance_args[@]}" --angular-momentum-scales 0.9 \
    --output "$scratch/paired-provenance-scale-fail.json" \
    > "$scratch/paired-provenance-scale-fail.out" 2>&1; then
  fail "paired-long provenance accepted a non-reference s_L"
fi
if "$real_python3" "$provenance_validator" \
    "${paired_provenance_args[@]}" --radial-cells 64 \
    --output "$scratch/paired-provenance-grid-fail.json" \
    > "$scratch/paired-provenance-grid-fail.out" 2>&1; then
  fail "paired-long provenance accepted a non-reference grid"
fi
legacy_provenance_args=("${provenance_args[@]}")
legacy_provenance_args[1]="$scratch/certificate-legacy.json"
if "$real_python3" "$provenance_validator" "${legacy_provenance_args[@]}" \
    --output "$scratch/provenance-legacy-fail.json" \
    > "$scratch/provenance-legacy-fail.out" 2>&1; then
  fail "threshold provenance gate accepted legacy center-only reconstruction"
fi
substep_provenance_args=("${provenance_args[@]}")
substep_provenance_args[1]="$scratch/certificate-substep-mismatch.json"
if "$real_python3" "$provenance_validator" "${substep_provenance_args[@]}" \
    --output "$scratch/provenance-substep-fail.json" \
    > "$scratch/provenance-substep-fail.out" 2>&1; then
  fail "threshold provenance gate accepted mismatched eikonal substeps"
fi
if "$real_python3" "$provenance_validator" "${provenance_args[@]}" \
    --central-y 0.14 --output "$scratch/provenance-family-fail.json" \
    > "$scratch/provenance-family-fail.out" 2>&1; then
  fail "threshold provenance gate accepted a different static continuum"
fi
printf '# changed after parity\n' \
  >> "$deployed_root/cpu/tools/validate_wpic_null_map.py"
if "$real_python3" "$provenance_validator" "${provenance_args[@]}" \
    --output "$scratch/provenance-source-fail.json" \
    > "$scratch/provenance-source-fail.out" 2>&1; then
  fail "threshold provenance gate accepted changed source"
fi
printf '# dry-run fixture\n' \
  > "$deployed_root/cpu/tools/validate_wpic_null_map.py"
printf '# changed binary\n' \
  >> "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
if "$real_python3" "$provenance_validator" "${provenance_args[@]}" \
    --output "$scratch/provenance-binary-fail.json" \
    > "$scratch/provenance-binary-fail.out" 2>&1; then
  fail "threshold provenance gate accepted a changed executable"
fi
cp "$scratch/fake-solver" \
  "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"
chmod +x "$deployed_root/cpu/venus_cpu_einstein_vlasov_wpic_black_hole"

# The launcher-level binding validator checks both Boolean provenance and the
# algebra B=M0-M_ADM, b=B/M0.
"$real_python3" "$binding_validator" \
  --summary "$scratch/certified-parity/schwarzschild/summary.json" \
  --output "$scratch/binding-pass.json" \
  || fail "positive-binding validator rejected a consistent bound family"
"$real_python3" - "$scratch/certified-parity/schwarzschild/summary.json" \
  "$scratch/unbound-summary.json" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
value["high_compactness"]["fixed_family_positive_binding_energy"] = False
pathlib.Path(sys.argv[2]).write_text(json.dumps(value) + "\n")
PY
if "$real_python3" "$binding_validator" \
    --summary "$scratch/unbound-summary.json" \
    --output "$scratch/binding-fail.json" \
    > "$scratch/binding-fail.out" 2>&1; then
  fail "positive-binding validator accepted a false binding flag"
fi

# A paired-long request queues only a certificate-dependent gate.  The gate
# authenticates the now-existing certificate and then submits exactly one SC
# and one EF child, both dependent on the gate's success.
paired_root="$scratch/paired-long"
before_paired=$(wc -l < "$SBATCH_LOG")
env \
  VENUS_SOURCE_ROOT="$deployed_root" \
  VENUS_PAIRED_LONG_WORKFLOW_ROOT="$paired_workflow_root" \
  VENUS_RUN_ROOT="$scratch/run" \
  CAMPAIGN_ROOT="$paired_root" \
  PARITY_CERTIFICATE_JOB_ID=4321 \
  PARITY_CERTIFICATE="$scratch/certificate.json" \
  PARITY_FINE_PHYSICAL_IC="$scratch/certified-parity/physical_ic.bin" \
  PARITY_FINE_NULL_INITIAL_DATA="$scratch/certified-parity/ef_null_initial_data.bin" \
  HIGH_INITIAL_CONDITION=energy_cutoff_shell \
  ENERGY_CUTOFF_AMPLITUDE=0.5 \
  ENERGY_CUTOFF_ELL0=1.0 \
  ENERGY_CUTOFF_CENTRAL_Y=0.13 \
  RADIAL_INFALL=0.05 \
  INFALL_CORE_RADIUS=0.25 \
  ANGULAR_MOMENTUM_SCALE=1 \
  TARGET_INITIAL_ADM_MASS=0.85 \
  SAMPLING_REALIZATION=7 \
  LONG_MARKERS=100 \
  LONG_RADIAL_CELLS=32 \
  LONG_DT_MAX=0.02 \
  LONG_R_MAX=8 \
  LONG_T_END=80 \
  LONG_OUTPUT_STEPS=2 \
  EF_MAP_WORLDTUBE_RADIUS=6 \
  EF_NULL_DONOR_DURATION=2 \
  VENUS_EIKONAL_SUBSTEPS=8 \
  bash "$paired_submit" > "$scratch/paired-submit.out"
after_paired=$(wc -l < "$SBATCH_LOG")
[[ $((after_paired-before_paired)) = 1 ]] \
  || fail "paired-long submitter did not queue exactly one gate job"
tail -n 1 "$SBATCH_LOG" | grep -Eq -- '--dependency=afterok:4321' \
  || fail "paired-long gate does not depend on the strict certificate job"
tail -n 1 "$SBATCH_LOG" | grep -Eq 'ias_einstein_vlasov_wpic_paired_long_gate.sbatch' \
  || fail "paired-long submitter did not use its pinned gate script"
if env \
    VENUS_SOURCE_ROOT="$deployed_root" VENUS_RUN_ROOT="$scratch/run" \
    VENUS_PAIRED_LONG_WORKFLOW_ROOT="$paired_workflow_root" \
    CAMPAIGN_ROOT="$paired_root" PARITY_CERTIFICATE_JOB_ID=4321 \
    PARITY_CERTIFICATE="$scratch/certificate.json" \
    PARITY_FINE_PHYSICAL_IC="$scratch/certified-parity/physical_ic.bin" \
    PARITY_FINE_NULL_INITIAL_DATA="$scratch/certified-parity/ef_null_initial_data.bin" \
    HIGH_INITIAL_CONDITION=energy_cutoff_shell ENERGY_CUTOFF_AMPLITUDE=0.5 \
    ENERGY_CUTOFF_ELL0=1 ENERGY_CUTOFF_CENTRAL_Y=0.13 \
    RADIAL_INFALL=0.05 INFALL_CORE_RADIUS=0.25 \
    ANGULAR_MOMENTUM_SCALE=1 TARGET_INITIAL_ADM_MASS=0.85 \
    SAMPLING_REALIZATION=7 LONG_MARKERS=100 LONG_RADIAL_CELLS=32 \
    LONG_DT_MAX=0.02 LONG_R_MAX=8 LONG_T_END=80 LONG_OUTPUT_STEPS=2 \
    EF_MAP_WORLDTUBE_RADIUS=6 EF_NULL_DONOR_DURATION=2 \
    VENUS_EIKONAL_SUBSTEPS=8 \
    bash "$paired_submit" > "$scratch/paired-reuse.out" 2>&1; then
  fail "paired-long submitter reused an existing campaign"
fi
[[ "$(wc -l < "$SBATCH_LOG")" = "$after_paired" ]] \
  || fail "paired-long no-overwrite rejection reached sbatch"

deployed_gate="$paired_workflow_root/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gate.sbatch"
deployed_paired_submit="$paired_workflow_root/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long.sh"
deployed_gif_submit="$paired_workflow_root/cpu/scripts/slurm/submit_einstein_vlasov_wpic_paired_long_gif.sh"
deployed_gif_wrapper="$paired_workflow_root/cpu/scripts/slurm/ias_einstein_vlasov_wpic_paired_long_gif.sbatch"
deployed_gif_renderer="$paired_workflow_root/cpu/tools/render_weighted_einstein_vlasov_coordinate_pair_gif.py"
request_sha=$(sha256sum "$paired_root/submission_request.manifest.tsv" | awk '{print $1}')
args_sha=$(sha256sum "$paired_root/driver_arguments.txt" | awk '{print $1}')
before_gate=$(wc -l < "$SBATCH_LOG")
env \
  SLURM_JOB_ID=7654 \
  VENUS_SOURCE_ROOT="$deployed_root" \
  VENUS_PAIRED_LONG_WORKFLOW_ROOT="$paired_workflow_root" \
  VENUS_RUN_ROOT="$scratch/run" \
  CAMPAIGN_ROOT="$paired_root" \
  PARITY_CERTIFICATE_JOB_ID=4321 \
  PARITY_CERTIFICATE="$scratch/certificate.json" \
  PARITY_FINE_PHYSICAL_IC="$scratch/certified-parity/physical_ic.bin" \
  PARITY_FINE_NULL_INITIAL_DATA="$scratch/certified-parity/ef_null_initial_data.bin" \
  HIGH_INITIAL_CONDITION=energy_cutoff_shell \
  ENERGY_CUTOFF_AMPLITUDE=0.5 ENERGY_CUTOFF_ELL0=1 \
  ENERGY_CUTOFF_CENTRAL_Y=0.13 RADIAL_INFALL=0.05 \
  INFALL_CORE_RADIUS=0.25 ANGULAR_MOMENTUM_SCALE=1 \
  TARGET_INITIAL_ADM_MASS=0.85 SAMPLING_REALIZATION=7 \
  LONG_MARKERS=100 LONG_RADIAL_CELLS=32 LONG_DT_MAX=0.02 \
  LONG_R_MAX=8 LONG_T_END=80 LONG_OUTPUT_STEPS=2 \
  EF_MAP_WORLDTUBE_RADIUS=6 EF_NULL_DONOR_DURATION=2 \
  VENUS_EIKONAL_SUBSTEPS=8 \
  VENUS_PAIRED_LONG_GATE_SHA256="$(sha256sum "$deployed_gate" | awk '{print $1}')" \
  VENUS_PAIRED_LONG_SUBMITTER_SHA256="$(sha256sum "$deployed_paired_submit" | awk '{print $1}')" \
  VENUS_PAIRED_LONG_GIF_SUBMITTER_SHA256="$(sha256sum "$deployed_gif_submit" | awk '{print $1}')" \
  VENUS_PAIRED_LONG_GIF_WRAPPER_SHA256="$(sha256sum "$deployed_gif_wrapper" | awk '{print $1}')" \
  VENUS_PAIRED_LONG_GIF_RENDERER_SHA256="$(sha256sum "$deployed_gif_renderer" | awk '{print $1}')" \
  VENUS_LAUNCHER_SHA256="$(sha256sum "$deployed_root/cpu/scripts/slurm/ias_einstein_vlasov_wpic.sbatch" | awk '{print $1}')" \
  VENUS_ANALYZER_SHA256="$(sha256sum "$deployed_root/cpu/tools/analyze_einstein_vlasov_wpic.py" | awk '{print $1}')" \
  VENUS_THRESHOLD_PROVENANCE_VALIDATOR_SHA256="$(sha256sum "$provenance_validator" | awk '{print $1}')" \
  VENUS_NULL_MAP_PREFLIGHT_SHA256="$(sha256sum "$deployed_root/cpu/tools/validate_wpic_null_map.py" | awk '{print $1}')" \
  VENUS_POSITIVE_BINDING_VALIDATOR_SHA256="$(sha256sum "$binding_validator" | awk '{print $1}')" \
  VENUS_PAIRED_LONG_REQUEST_SHA256="$request_sha" \
  VENUS_PAIRED_LONG_ARGS_SHA256="$args_sha" \
  bash "$deployed_gate" > "$scratch/paired-gate.out"
after_gate=$(wc -l < "$SBATCH_LOG")
[[ $((after_gate-before_gate)) = 3 ]] \
  || fail "paired-long gate did not submit two evolutions plus one GIF job"
tail -n 3 "$SBATCH_LOG" | grep -Fq 'COORDINATES=schwarzschild\,EXPECTED_OUTCOME=any' \
  || { tail -n 2 "$SBATCH_LOG" >&2; fail "paired-long gate did not submit SC with expected=any"; }
tail -n 3 "$SBATCH_LOG" | grep -Fq 'COORDINATES=eddington_finkelstein\,EXPECTED_OUTCOME=black_hole' \
  || fail "paired-long gate did not submit EF with expected=black_hole"
[[ "$(tail -n 3 "$SBATCH_LOG" | grep -Eo -- '--dependency=afterok:7654' | wc -l | tr -d ' ')" = 2 ]] \
  || fail "paired-long children are not both held behind the gate success"
tail -n 3 "$SBATCH_LOG" | grep -Eq 'VENUS_BUILD=0' \
  || fail "paired-long child could rebuild the certified executable"
sc_child_id=$(awk -F '\t' '$1 == "long_evolution" && $4 == "schwarzschild" {print $2}' \
  "$paired_root/submission.tsv")
ef_child_id=$(awk -F '\t' '$1 == "long_evolution" && $4 == "eddington_finkelstein" {print $2}' \
  "$paired_root/submission.tsv")
tail -n 1 "$SBATCH_LOG" | grep -Eq -- "--dependency=afterok:${sc_child_id}:${ef_child_id}" \
  || fail "paired-long GIF is not held behind both evolutions"
tail -n 1 "$SBATCH_LOG" | grep -Eq 'ias_einstein_vlasov_wpic_paired_long_gif.sbatch' \
  || fail "paired-long gate did not use the pinned GIF wrapper"
awk -F '\t' 'NF != 10 { exit 1 } END { if (NR != 4) exit 1 }' \
  "$paired_root/submission.tsv" \
  || fail "paired-long receipt does not contain gate plus exactly two runs"
[[ -s "$paired_root/parity_certificate_gate.json" \
   && -s "$paired_root/parity_provenance_attestation.json" ]] \
  || fail "paired-long gate did not retain its certificate/provenance audits"

export VENUS_THRESHOLD_PROVENANCE_PYTHON="$real_python3"
before_zero=$(wc -l < "$SBATCH_LOG")
if env \
    VENUS_SOURCE_ROOT="$deployed_root" \
    VENUS_RUN_ROOT="$scratch/run" \
    CAMPAIGN_ROOT="$scratch/zero-scan" \
    HIGH_INITIAL_CONDITION=energy_cutoff_shell \
    ENERGY_CUTOFF_AMPLITUDE=0.5 \
    ENERGY_CUTOFF_ELL0=1.0 \
    ENERGY_CUTOFF_CENTRAL_Y=0.13 \
    RADIAL_INFALL=0.05 \
    INFALL_CORE_RADIUS=0.25 \
    ANGULAR_MOMENTUM_SCALES='0 1' \
    TARGET_INITIAL_ADM_MASS=0.85 \
    SAMPLING_REALIZATION=7 \
    SCAN_R_MAX=8 \
    SCAN_T_END=1 \
    NULL_DOMAIN_GATE_SCALE=1 \
    EF_MAP_WORLDTUBE_RADIUS=6 \
    EF_NULL_DONOR_DURATION=2 \
    PARITY_CERTIFICATE="$scratch/certificate.json" \
    bash "$scan_submit" > "$scratch/zero-scan.out" 2>&1; then
  fail "scan helper accepted s_L=0"
fi
after_zero=$(wc -l < "$SBATCH_LOG")
[[ "$before_zero" = "$after_zero" ]] \
  || fail "zero-L scan input reached sbatch before failing"

before_invalid=$(wc -l < "$SBATCH_LOG")
if env \
    VENUS_SOURCE_ROOT="$deployed_root" \
    VENUS_RUN_ROOT="$scratch/run" \
    CAMPAIGN_ROOT="$scratch/invalid-scan" \
    HIGH_INITIAL_CONDITION=energy_cutoff_shell \
    ENERGY_CUTOFF_AMPLITUDE=0.5 \
    ENERGY_CUTOFF_ELL0=1.0 \
    ENERGY_CUTOFF_CENTRAL_Y=0.13 \
    RADIAL_INFALL=0.05 \
    INFALL_CORE_RADIUS=0.25 \
    ANGULAR_MOMENTUM_SCALES='0.5 1' \
    TARGET_INITIAL_ADM_MASS=0.85 \
    SAMPLING_REALIZATION=7 \
    SCAN_MARKERS=100 \
    SCAN_RADIAL_CELLS=32 \
    SCAN_R_MAX=8 \
    SCAN_T_END=1 \
    NULL_DOMAIN_GATE_SCALE=0.5 \
    EF_MAP_WORLDTUBE_RADIUS=6.1 \
    EF_NULL_DONOR_DURATION=2 \
    PARITY_CERTIFICATE="$scratch/certificate.json" \
    bash "$scan_submit" > "$scratch/invalid-scan.out" 2>&1; then
  fail "scan helper accepted an off-face R_B"
fi
after_invalid=$(wc -l < "$SBATCH_LOG")
[[ "$before_invalid" = "$after_invalid" ]] \
  || fail "invalid scan reached sbatch before failing"

before_mismatch=$(wc -l < "$SBATCH_LOG")
if env \
    VENUS_SOURCE_ROOT="$deployed_root" \
    VENUS_RUN_ROOT="$scratch/run" \
    CAMPAIGN_ROOT="$scratch/mismatched-family-scan" \
    HIGH_INITIAL_CONDITION=energy_cutoff_shell \
    ENERGY_CUTOFF_AMPLITUDE=0.5 \
    ENERGY_CUTOFF_ELL0=1.0 \
    ENERGY_CUTOFF_CENTRAL_Y=0.14 \
    RADIAL_INFALL=0.05 \
    INFALL_CORE_RADIUS=0.25 \
    ANGULAR_MOMENTUM_SCALES='0.5 1' \
    TARGET_INITIAL_ADM_MASS=0.85 \
    SAMPLING_REALIZATION=7 \
    SCAN_MARKERS=100 \
    SCAN_RADIAL_CELLS=32 \
    SCAN_R_MAX=8 \
    SCAN_T_END=1 \
    NULL_DOMAIN_GATE_SCALE=0.5 \
    EF_MAP_WORLDTUBE_RADIUS=6 \
    EF_NULL_DONOR_DURATION=2 \
    PARITY_CERTIFICATE="$scratch/certificate.json" \
    bash "$scan_submit" > "$scratch/mismatched-family-scan.out" 2>&1; then
  fail "scan helper accepted a certificate for a different static family"
fi
after_mismatch=$(wc -l < "$SBATCH_LOG")
[[ "$before_mismatch" = "$after_mismatch" ]] \
  || fail "semantic family mismatch reached sbatch before failing"

scan_root="$scratch/scan"
env \
  VENUS_SOURCE_ROOT="$deployed_root" \
  VENUS_RUN_ROOT="$scratch/run" \
  CAMPAIGN_ROOT="$scan_root" \
  HIGH_INITIAL_CONDITION=energy_cutoff_shell \
  ENERGY_CUTOFF_AMPLITUDE=0.5 \
  ENERGY_CUTOFF_ELL0=1.0 \
  ENERGY_CUTOFF_CENTRAL_Y=0.13 \
  RADIAL_INFALL=0.05 \
  INFALL_CORE_RADIUS=0.25 \
  ANGULAR_MOMENTUM_SCALES='0.5 1' \
  TARGET_INITIAL_ADM_MASS=0.85 \
  SAMPLING_REALIZATION=7 \
  SCAN_MARKERS=100 \
  SCAN_RADIAL_CELLS=32 \
  SCAN_R_MAX=8 \
  SCAN_T_END=1 \
  SCAN_DT_MAX=0.02 \
  SCAN_OUTPUT_STEPS=2 \
  NULL_DOMAIN_GATE_SCALE=0.5 \
  EF_MAP_WORLDTUBE_RADIUS=6 \
  EF_NULL_DONOR_DURATION=2 \
  PARITY_CERTIFICATE="$scratch/certificate.json" \
  bash "$scan_submit" > "$scratch/scan.out"

[[ -s "$scan_root/campaign_controls.manifest.tsv" ]] \
  || fail "scan controls manifest is missing"
[[ -s "$scan_root/parity_provenance_attestation.json" ]] \
  || fail "scan semantic parity-provenance attestation is missing"
"$real_python3" - "$scan_root/parity_provenance_attestation.json" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("schema") != "venus_v3_threshold_parity_provenance_v1" \
        or value.get("status") != "pass":
    raise SystemExit("scan parity-provenance attestation is not a pass")
PY
grep -Eq $'angular_momentum_scales\t0.5 1' \
  "$scan_root/campaign_controls.manifest.tsv" \
  || fail "scan manifest omitted the scale grid"
awk -F '\t' 'NF != 22 { exit 1 }' "$scan_root/submission.tsv" \
  || fail "scan receipt rows do not match their 22-column schema"
if grep -Eq -- '--high-(shell|total-mass|specific-angular-momentum)' \
    "$scan_root/common_driver_arguments.txt"; then
  fail "scan argument file retained a Gaussian/specific-L flag"
fi
grep -Eq 'configured_energy_cutoff_ell0' "$scan_root/submission.tsv" \
  || fail "scan receipt does not distinguish configured ell0 from s_L"
grep -Eq 'VENUS_REQUIRE_POSITIVE_BINDING=1' "$SBATCH_LOG" \
  || fail "threshold jobs do not require launcher-level positive binding"
grep -Eq 'VENUS_BUILD=0.*VENUS_REQUIRE_POSITIVE_BINDING=1' "$SBATCH_LOG" \
  || fail "threshold seed could rebuild the certified immutable binary"

# Finally run the generic launcher in initialize-only mode with fake srun.  It
# must reconstruct all static controls and emit its per-run manifest.
printf '%s\n' \
  --high-markers 100 \
  --high-radial-cells 32 \
  --high-t-end 1 \
  --high-dt-max 0.02 \
  --high-gap-stop 0.001 \
  --high-r-min 0.005 \
  --high-inner-boundary origin_parity \
  --high-radial-grid linear \
  --high-output-steps 2 \
  > "$scratch/launcher.args"
deployed_launcher_sha256=$(sha256sum \
  "$deployed_root/cpu/scripts/slurm/ias_einstein_vlasov_wpic.sbatch" \
  | awk '{print $1}')
deployed_analyzer_sha256=$(sha256sum \
  "$deployed_root/cpu/tools/analyze_einstein_vlasov_wpic.py" \
  | awk '{print $1}')
deployed_null_preflight_sha256=$(sha256sum \
  "$deployed_root/cpu/tools/validate_wpic_null_map.py" | awk '{print $1}')
deployed_binding_sha256=$(sha256sum \
  "$deployed_root/cpu/scripts/slurm/validate_einstein_vlasov_wpic_positive_binding.py" \
  | awk '{print $1}')
before_radial=$(wc -l < "$SRUN_LOG")
if env \
    SLURM_SUBMIT_DIR="$source_root" \
    SLURM_CPUS_PER_TASK=2 \
    VENUS_SOURCE_ROOT="$deployed_root" \
    VENUS_RUN_ROOT="$scratch/run" \
    VENUS_RUN_ARGS_FILE="$scratch/launcher.args" \
    EXECUTABLE="$scratch/fake-solver" \
    OUTPUT_DIR="$scratch/launcher-zero-L" \
    PHYSICAL_IC="$scratch/launcher-zero-L.physical_ic.bin" \
    COORDINATES=schwarzschild \
    EXPECTED_OUTCOME=any \
    INITIALIZE_ONLY=1 \
    VENUS_BUILD=0 \
    VENUS_SKIP_MODULES=1 \
    VENUS_HIGH_INITIAL_CONDITION=energy_cutoff_shell \
    VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=0.5 \
    VENUS_HIGH_ENERGY_CUTOFF_ELL0=1.0 \
    VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=0.13 \
    VENUS_HIGH_RADIAL_INFALL=0.05 \
    VENUS_HIGH_INFALL_CORE_RADIUS=0.25 \
    VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=0 \
    VENUS_HIGH_SAMPLING_REALIZATION=7 \
    VENUS_HIGH_TARGET_INITIAL_ADM_MASS=0.85 \
    VENUS_HIGH_R_MAX=8 \
    VENUS_LAUNCHER_SHA256="$deployed_launcher_sha256" \
    VENUS_ANALYZER_SHA256="$deployed_analyzer_sha256" \
    VENUS_NULL_MAP_PREFLIGHT_SHA256="$deployed_null_preflight_sha256" \
    VENUS_POSITIVE_BINDING_VALIDATOR_SHA256="$deployed_binding_sha256" \
    bash "$launcher" > "$scratch/launcher-zero-L.out" 2>&1; then
  fail "generic launcher accepted energy_cutoff_shell with s_L=0"
fi
after_radial=$(wc -l < "$SRUN_LOG")
[[ "$before_radial" = "$after_radial" ]] \
  || fail "generic launcher reached srun with s_L=0"
launcher_output="$scratch/launcher-output"
env \
  SLURM_SUBMIT_DIR="$source_root" \
  SLURM_CPUS_PER_TASK=2 \
  VENUS_SOURCE_ROOT="$deployed_root" \
  VENUS_RUN_ROOT="$scratch/run" \
  VENUS_RUN_ARGS_FILE="$scratch/launcher.args" \
  EXECUTABLE="$scratch/fake-solver" \
  OUTPUT_DIR="$launcher_output" \
  PHYSICAL_IC="$scratch/launcher.physical_ic.bin" \
  COORDINATES=schwarzschild \
  EXPECTED_OUTCOME=any \
  INITIALIZE_ONLY=1 \
  VENUS_BUILD=0 \
  VENUS_SKIP_MODULES=1 \
  VENUS_HIGH_INITIAL_CONDITION=energy_cutoff_shell \
  VENUS_HIGH_ENERGY_CUTOFF_AMPLITUDE=0.5 \
  VENUS_HIGH_ENERGY_CUTOFF_ELL0=1.0 \
  VENUS_HIGH_ENERGY_CUTOFF_CENTRAL_Y=0.13 \
  VENUS_HIGH_RADIAL_INFALL=0.05 \
  VENUS_HIGH_INFALL_CORE_RADIUS=0.25 \
  VENUS_HIGH_ANGULAR_MOMENTUM_SCALE=1 \
  VENUS_HIGH_SAMPLING_REALIZATION=7 \
  VENUS_HIGH_TARGET_INITIAL_ADM_MASS=0.85 \
  VENUS_HIGH_R_MAX=8 \
  VENUS_REQUIRE_POSITIVE_BINDING=1 \
  VENUS_CAMPAIGN_CONTROLS_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  VENUS_LAUNCHER_SHA256="$deployed_launcher_sha256" \
  VENUS_ANALYZER_SHA256="$deployed_analyzer_sha256" \
  VENUS_NULL_MAP_PREFLIGHT_SHA256="$deployed_null_preflight_sha256" \
  VENUS_POSITIVE_BINDING_VALIDATOR_SHA256="$deployed_binding_sha256" \
  bash "$launcher" > "$scratch/launcher.out"

grep -Eq $'energy_cutoff_amplitude_C\t0.5' \
  "$launcher_output/workflow_controls.manifest.tsv" \
  || fail "launcher manifest omitted C"
grep -Eq $'campaign_controls_sha256\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  "$launcher_output/workflow_controls.manifest.tsv" \
  || fail "launcher manifest omitted the campaign provenance link"
grep -Eq $'require_positive_binding\t1' \
  "$launcher_output/workflow_controls.manifest.tsv" \
  || fail "launcher manifest omitted the positive-binding gate"
grep -Eq '^positive_binding_required=1$' \
  "$launcher_output/workflow_status.txt" \
  || fail "launcher workflow status omitted the binding-gate state"
grep -Eq -- '--high-energy-cutoff-amplitude 0.5' \
  "$launcher_output/resolved_command.sh" \
  || fail "launcher did not resolve the static continuum flags"
grep -Eq -- '--high-radial-infall 0.05' "$SRUN_LOG" \
  || fail "launcher did not pass the inward kick"

echo "PASS: static-shell IAS workflow plumbing fails closed and preserves exact controls."
