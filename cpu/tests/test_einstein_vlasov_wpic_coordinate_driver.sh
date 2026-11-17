#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: test_einstein_vlasov_wpic_coordinate_driver.sh BINARY PYTHON}
python=${2:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/venus-ev-coordinates.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

sha256_file() {
  "$python" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

mkdir -p "$scratch/sc" "$scratch/ef"
common=(
  --expected-outcome any
  --physical-ic "$scratch/common.phys"
  --threads 2
  --high-initial-condition shell
  --high-markers 256
  --high-radial-cells 64
  --high-r-min 0.01
  --high-r-max 6
  --high-inner-boundary origin_parity
  --high-radial-grid linear
  --high-total-mass 0.08
  --high-shell-center 2
  --high-shell-width 0.20
  --high-shell-radial-momentum -0.30
  --high-shell-radial-momentum-dispersion 0.01
  --high-shell-tangential-velocity-dispersion 0
  --high-specific-angular-momentum 0.08
  --high-sampling-realization 7
  --high-target-initial-adm-mass 0.07
  --high-t-end 0.08
  --high-dt-max 0.04
  --high-output-steps 1
)

"$binary" --coordinates schwarzschild \
  --export-ef-null-initial-data "$scratch/common.nullmap" \
  --ef-null-donor-duration 6 \
  --ef-map-worldtube-radius 5.25 \
  --output-dir "$scratch/sc" "${common[@]}" >/dev/null
before=$(sha256_file "$scratch/common.phys")
"$binary" --coordinates eddington_finkelstein \
  --ef-null-initial-data "$scratch/common.nullmap" \
  --output-dir "$scratch/ef" "${common[@]}" >/dev/null
after=$(sha256_file "$scratch/common.phys")
test "$before" = "$after"

mkdir -p "$scratch/realization8"
"$binary" --coordinates schwarzschild --initialize-only \
  --output-dir "$scratch/realization8" "${common[@]}" \
  --physical-ic "$scratch/realization8.phys" \
  --high-sampling-realization 8 >/dev/null
realization8=$(sha256_file "$scratch/realization8.phys")
test "$before" != "$realization8"

# The same coordinate-neutral marker family at a different L must reach the
# same prescribed ADM mass by changing only the common invariant-rest-weight
# scale.  This is an initialization-only check, not an outcome claim.
mkdir -p "$scratch/fixed_adm_l2"
"$binary" --coordinates schwarzschild --initialize-only \
  --output-dir "$scratch/fixed_adm_l2" "${common[@]}" \
  --high-angular-momentum-scale 2 >/dev/null

"$python" - "$scratch" <<'PY'
import csv
import json
import math
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
sc = json.loads((root / "sc" / "summary.json").read_text())["high_compactness"]
ef = json.loads((root / "ef" / "summary.json").read_text())["high_compactness"]
l2 = json.loads((root / "fixed_adm_l2" / "summary.json").read_text())["high_compactness"]
assert sc["coordinates"] == "schwarzschild"
assert ef["coordinates"] == "eddington_finkelstein"
for high in (sc, ef, l2):
    assert high["target_initial_adm_mass"] == 0.07
    assert high["initial_rest_weight_scale"] > 0.0
    assert high["realized_initial_rest_weight"] > 0.0
    assert math.isclose(high["fixed_family_adm_mass"], 0.07, rel_tol=5.0e-11)
    assert high["fixed_family_binding_diagnostics_available"] is True
    assert high["fixed_family_rest_weight"] > 0.0
    assert math.isclose(
        high["fixed_family_binding_energy"],
        high["fixed_family_rest_weight"] - high["fixed_family_adm_mass"],
        rel_tol=0.0,
        abs_tol=2.0e-15,
    )
    assert math.isclose(
        high["fixed_family_binding_fraction"],
        high["fixed_family_binding_energy"]
            / high["fixed_family_rest_weight"],
        rel_tol=0.0,
        abs_tol=2.0e-15,
    )
    assert high["fixed_family_positive_binding_energy"] is (
        high["fixed_family_binding_fraction"] > 0.0
    )
    assert high["initial_energy_at_infinity_proxy_minimum"] > 0.0
    assert high["initial_energy_at_infinity_proxy_maximum"] >= (
        high["initial_energy_at_infinity_proxy_minimum"]
    )
    assert high["initial_energy_at_infinity_proxy_minimum"] <= (
        high["initial_energy_at_infinity_proxy_rest_weighted_mean"]
    ) <= high["initial_energy_at_infinity_proxy_maximum"]
    assert 0.0 <= (
        high[
            "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one"
        ]
    ) <= 1.0
    assert "not conserved" in high[
        "initial_energy_at_infinity_proxy_definition"
    ]
for key in (
    "fixed_family_rest_weight", "fixed_family_adm_mass",
    "fixed_family_binding_energy", "fixed_family_binding_fraction",
    "fixed_family_positive_binding_energy",
    "initial_energy_at_infinity_proxy_minimum",
    "initial_energy_at_infinity_proxy_maximum",
    "initial_energy_at_infinity_proxy_rest_weighted_mean",
    "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one",
):
    assert sc[key] == ef[key]
assert (
    (root / "sc" / "fixed_family_initial_bound_diagnostics.csv").read_text()
    == (root / "ef" / "fixed_family_initial_bound_diagnostics.csv").read_text()
)
assert math.isclose(sc["initial_adm_mass"], 0.07, rel_tol=5.0e-11)
assert math.isclose(l2["initial_adm_mass"], 0.07, rel_tol=5.0e-11)
assert sc["initial_deposited_adm_mass"] == sc["initial_adm_mass"]
assert sc["null_slice_conserved_adm_mass"] > 0.0
assert ef["initial_deposited_adm_mass"] == ef["initial_adm_mass"]
assert ef["null_slice_conserved_adm_mass"] > 0.0
assert math.isfinite(sc["null_slice_adm_relative_drift_from_family"])
assert math.isfinite(ef["null_slice_adm_relative_drift_from_family"])
assert math.isfinite(
    ef["initial_deposition_adm_relative_error_from_null_slice"]
)
assert not math.isclose(
    sc["realized_initial_rest_weight"],
    l2["realized_initial_rest_weight"],
    rel_tol=1.0e-10,
    abs_tol=1.0e-12,
)
assert sc["initial_rest_weight_scale"] != l2["initial_rest_weight_scale"]
assert sc["base_ic_fingerprint"] == ef["base_ic_fingerprint"]
assert sc["realized_ic_fingerprint"] != ef["realized_ic_fingerprint"]
assert sc["rest_mass_relative_error"] < 1.0e-13
assert ef["rest_mass_relative_error"] < 1.0e-13
assert sc["eikonal_mapped_initial_data"] is True
assert ef["eikonal_mapped_initial_data"] is True
assert sc["eikonal_boundary_radius"] == 5.25
assert ef["eikonal_boundary_radius"] == 5.25
assert sc["eikonal_evolution_outer_radius"] == 6.0
assert ef["eikonal_evolution_outer_radius"] == 6.0
assert sc["eikonal_analytic_vacuum_extension"] is True
assert ef["eikonal_analytic_vacuum_extension"] is True
assert sc["coordinate_map_manifest_sha256"] == ef["coordinate_map_manifest_sha256"]
assert len(sc["coordinate_map_manifest_sha256"]) == 64
assert ef["initial_schwarzschild_map_a_linf"] < 0.35
assert ef["initial_schwarzschild_map_b_linf"] < 0.35
assert ef["initial_schwarzschild_map_T_rr_relative_l2"] < 0.75
assert ef["initial_schwarzschild_map_S_relative_l2"] < 0.75
assert ef["initial_min_b"] > 0.0
audit = ef["ef_post_horizon_evidence"]
assert audit["schema"] == "venus_wpic_ef_post_horizon_evidence_v1"
assert audit["available"] is True
assert audit["accepted_step_audit_complete"] is True
assert audit["accepted_step_audit_samples"] == ef["steps"]
assert audit["minimum_p_r_over_evolution"] > 0.0
assert audit["minimum_b_over_evolution"] > 0.0
assert audit["finite_fields_throughout"] is True
assert audit["fixed_adm_bookkeeping_throughout"] is True
assert audit["causal_excision_valid_throughout"] is True
assert audit["configured_excision_a_threshold"] < 0.0
assert audit["configured_excision_buffer_cells"] >= 1
assert audit["configured_excision_horizon_fraction"] == 0.1
assert (root / "ef" / "ef_post_horizon_audit.state").is_file()
with (root / "sc" / "high_compactness_profiles.csv").open(newline="") as stream:
    reader = csv.DictReader(stream)
    assert {
        "m", "a", "lambda", "mu", "lambda_t", "rho", "p_radial",
        "p_tangential_sum", "j", "metric_sample_kind", "metric_cell",
        "face_left_r", "face_right_r", "face_left_lambda",
        "face_right_lambda", "center_mu_r", "face_left_mu",
        "face_right_mu", "face_left_mu_r", "face_right_mu_r",
    }.issubset(set(reader.fieldnames or []))
with (root / "ef" / "high_compactness_history.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) >= 3
assert all(float(row["min_b"]) > 0.0 for row in rows)
assert all(int(row["active_marker_count"]) == 256 for row in rows)
assert all(float(row["active_trapped_rest_weight"]) == 0.0 for row in rows)
assert all(float(row["active_min_radius"]) > 0.0 for row in rows)
with (root / "ef" / "high_compactness_profiles.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert rows
assert {"T_vv", "T_vr", "T_theta_theta"}.issubset(rows[0])
assert {
    "metric_sample_kind", "metric_cell", "face_left_r", "face_right_r",
    "face_left_a", "face_right_a", "face_left_b", "face_right_b",
}.issubset(rows[0])
assert all(float(row["T_vv"]) >= 0.0 for row in rows)
assert all(float(row["T_vr"]) <= 0.0 for row in rows)
assert all(float(row["T_theta_theta"]) >= 0.0 for row in rows)
assert all(math.isfinite(float(row["rho_infall"]))
           and float(row["rho_infall"]) >= 0.0 for row in rows)
assert any(float(row["rho_infall"]) > 0.0 for row in rows)
assert all(int(row["excision_mask"]) == 0 for row in rows)
manifest = json.loads((root / "common.nullmap.manifest.json").read_text())
assert manifest["schema_version"] == 2
assert manifest["donor_history_format_version"] == 2
assert manifest["saved_metric_interpolation"] == (
    "SC center-face lambda plus monotone-Hermite mu; EF center-face a,b"
)
donor_header = (root / "common.nullmap.donor_history.bin").read_bytes()[:20]
assert donor_header[:12] == b"VENUSSCHIST1"
assert struct.unpack_from("<I", donor_header, 16)[0] == 2
assert manifest["complete"] is True
assert manifest["all_markers_crossed_once"] is True
assert manifest["outer_worldtube_verified_vacuum"] is True
assert manifest["crossing_count"] == manifest["marker_count"]
assert manifest["unique_crossing_count"] == manifest["marker_count"]
assert manifest["map_worldtube_radius"] == 5.25
assert manifest["evolution_outer_radius"] == 6.0
assert manifest["map_radial_cell_count"] == 56
assert manifest["evolution_radial_cell_count"] == 64
assert manifest["analytic_vacuum_extension"] is True
assert manifest["analytic_vacuum_extension_start_index"] == 56
assert manifest["minimum_worldtube_matter_clearance"] >= (
    manifest["required_worldtube_clearance"]
)
assert manifest["minimum_crossing_radius_clearance"] >= (
    manifest["required_worldtube_clearance"]
)
assert manifest["maximum_analytic_vacuum_field_residual"] < 1.0e-13
assert manifest["maximum_analytic_vacuum_source"] < 1.0e-13
assert manifest["maximum_analytic_inverse_map_residual"] < 1.0e-13
assert manifest["minimum_analytic_vacuum_a"] > 0.0
assert manifest["maximum_outer_b_normalization_error"] <= (
    manifest["outer_b_normalization_tolerance"]
)
assert manifest["map_grid_fingerprint"] != manifest["grid_fingerprint"]
assert manifest["time_reversal_momentum"] == "w_reverse=-w_schwarzschild"
assert manifest["sampling_realization"] == 7
assert manifest["target_initial_adm_mass"] == 0.07
assert manifest["donor_initial_adm_mass"] == sc["fixed_family_adm_mass"]
assert manifest["null_slice_adm_mass"] == sc["null_slice_conserved_adm_mass"]
assert manifest["null_slice_adm_mass"] == ef["null_slice_conserved_adm_mass"]
assert manifest["eikonal_integration_method"] == (
    "per_cell_RK4_step_doubling_n_vs_2n_accept_2n"
)
assert manifest["eikonal_richardson_error_normalization"] == (
    "abs(Y_2n-Y_n)/(15*max(1,abs(Y_in),abs(Y_2n)));Y={s_sigma,ln_A}"
)
reverse_error = manifest["maximum_eikonal_reverse_time_richardson_error"]
log_a_error = manifest["maximum_eikonal_log_A_richardson_error"]
assert reverse_error >= 0.0 and log_a_error >= 0.0
assert manifest["maximum_eikonal_ode_residual"] == max(
    reverse_error, log_a_error
)
assert manifest["realized_initial_rest_weight"] > 0.0
assert manifest["fixed_family_rest_weight"] == sc["fixed_family_rest_weight"]
assert manifest["fixed_family_binding_energy"] == sc[
    "fixed_family_binding_energy"
]
assert manifest["fixed_family_binding_fraction"] == sc[
    "fixed_family_binding_fraction"
]
assert manifest["fixed_family_positive_binding_energy"] is sc[
    "fixed_family_positive_binding_energy"
]
for key in (
    "initial_energy_at_infinity_proxy_minimum",
    "initial_energy_at_infinity_proxy_maximum",
    "initial_energy_at_infinity_proxy_rest_weighted_mean",
    "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one",
):
    assert manifest[key] == sc[key]
assert set(manifest["artifacts"]) == {
    "null_initial_data", "crossings", "slice", "accepted_steps",
    "donor_history",
}
with (root / "common.nullmap.crossings.csv").open(newline="") as stream:
    crossings = list(csv.DictReader(stream))
assert len(crossings) == manifest["marker_count"]
assert len({row["id"] for row in crossings}) == len(crossings)
assert all(float(row["p_r_ef"]) > 0.0 for row in crossings)
assert max(abs(float(row["residual"])) for row in crossings) < 1.0e-10
with (root / "ef" / "initial_coordinate_map_residuals.csv").open(newline="") as stream:
    residuals = list(csv.DictReader(stream))
assert residuals
with (root / "common.nullmap.slice.csv").open(newline="") as stream:
    mapped_slice = list(csv.DictReader(stream))
vacuum = [row for row in mapped_slice if float(row["r"]) > 5.25]
assert vacuum
mass = manifest["null_slice_adm_mass"]
boundary_tau = -manifest["eikonal_outer_reverse_time"]
for row in vacuum:
    radius = float(row["r"])
    exact_a = 1.0 - 2.0 * mass / radius
    tortoise = (radius - 5.25) + 2.0 * mass * math.log(
        (radius - 2.0 * mass) / (5.25 - 2.0 * mass)
    )
    assert float(row["A"]) == 1.0
    assert float(row["F"]) == 1.0
    assert float(row["b"]) == 1.0
    assert math.isclose(float(row["a"]), exact_a, abs_tol=2.0e-15)
    assert math.isclose(float(row["m"]), mass, abs_tol=2.0e-15)
    assert math.isclose(
        float(row["schwarzschild_time"]), boundary_tau - tortoise,
        abs_tol=2.0e-14,
    )
    for key in (
        "rho", "p_radial", "p_tangential_sum", "j", "T_vv", "T_vr",
        "T_rr", "T_theta_theta", "S",
    ):
        assert float(row[key]) == 0.0
PY

"$python" "$script_dir/../tools/validate_wpic_null_map.py" \
  --manifest "$scratch/common.nullmap.manifest.json" \
  --summary "$scratch/ef/summary.json" \
  --output "$scratch/null_map_preflight.json" >/dev/null

resume=("${common[@]}")
for ((i=0; i<${#resume[@]}; ++i)); do
  if [[ ${resume[i]} == --high-t-end ]]; then
    resume[i+1]=0.12
  fi
done

# A changed angular-momentum control must not relabel an existing SC state.
# Check every original output byte after rejection, then verify a consistent
# continuation still succeeds.
cp -R "$scratch/sc" "$scratch/sc-wrong-ic"
before_wrong=$("$python" - "$scratch/sc-wrong-ic" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
print([(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
       for p in sorted(root.rglob('*')) if p.is_file()])
PY
)
if "$binary" --coordinates schwarzschild --resume \
    --output-dir "$scratch/sc-wrong-ic" "${resume[@]}" \
    --high-angular-momentum-scale 2 >"$scratch/wrong-ic.log" 2>&1; then
  echo "SC resume accepted different immutable angular momenta" >&2
  exit 1
fi
grep -q "Restart immutable marker attributes" "$scratch/wrong-ic.log"
after_wrong=$("$python" - "$scratch/sc-wrong-ic" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
print([(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
       for p in sorted(root.rglob('*')) if p.is_file()])
PY
)
test "$before_wrong" = "$after_wrong"
"$binary" --coordinates schwarzschild --resume \
  --output-dir "$scratch/sc" "${resume[@]}" >/dev/null

"$binary" --coordinates eddington_finkelstein --resume \
  --ef-null-initial-data "$scratch/common.nullmap" \
  --output-dir "$scratch/ef" "${resume[@]}" >/dev/null
"$python" - "$scratch/ef/summary.json" \
  "$scratch/common.nullmap.manifest.json" <<'PY'
import json
import math
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())["high_compactness"]
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert summary["steps"] >= 3
assert summary["base_ic_fingerprint"] != "0x0000000000000000"
assert summary["realized_ic_fingerprint"] != "0x0000000000000000"
# ADM normalization is solved numerically; resume preserves the realized
# donor-family value exactly and the original target to its stated tolerance.
assert math.isclose(summary["fixed_family_adm_mass"], 0.07, rel_tol=5.0e-11)
assert summary["fixed_family_adm_mass"] == manifest["donor_initial_adm_mass"]
assert summary["null_slice_conserved_adm_mass"] == manifest["null_slice_adm_mass"]
assert summary["initial_deposited_adm_mass"] > 0.0
assert summary["coordinate_map_manifest_sha256"]
assert summary["fixed_family_rest_weight"] == manifest[
    "fixed_family_rest_weight"
]
assert summary["fixed_family_binding_fraction"] == manifest[
    "fixed_family_binding_fraction"
]
assert summary["fixed_family_positive_binding_energy"] is manifest[
    "fixed_family_positive_binding_energy"
]
assert summary[
    "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one"
] == manifest[
    "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one"
]
audit = summary["ef_post_horizon_evidence"]
assert audit["accepted_step_audit_complete"] is True
assert audit["accepted_step_audit_samples"] == summary["steps"]
assert audit["minimum_p_r_over_evolution"] > 0.0
assert audit["minimum_b_over_evolution"] > 0.0
assert audit["finite_fields_throughout"] is True
assert audit["fixed_adm_bookkeeping_throughout"] is True
assert audit["causal_excision_valid_throughout"] is True
PY

# The same authenticated null slice supports a fresh interior run. Its
# restart preserves the mode and every active marker; toggling it is rejected.
"$python" - "$scratch/ef/ef_interior_controls.txt" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
header, payload = path.read_text().splitlines()
assert header == "VENUS_EF_INTERIOR_CONTROLS 2"
parts = payload.split()
assert parts.pop() == "1"
path.write_text("VENUS_EF_INTERIOR_CONTROLS 1\n" + " ".join(parts) + "\n")
PY
"$binary" --coordinates eddington_finkelstein --resume \
  --ef-null-initial-data "$scratch/common.nullmap" \
  --output-dir "$scratch/ef" "${resume[@]}" >/dev/null
mkdir -p "$scratch/ef-interior"
"$binary" --coordinates eddington_finkelstein --ef-evolve-interior \
  --ef-null-initial-data "$scratch/common.nullmap" \
  --output-dir "$scratch/ef-interior" "${common[@]}" >/dev/null
"$binary" --coordinates eddington_finkelstein --ef-evolve-interior --resume \
  --ef-null-initial-data "$scratch/common.nullmap" \
  --output-dir "$scratch/ef-interior" "${resume[@]}" >/dev/null
if "$binary" --coordinates eddington_finkelstein --resume \
    --ef-null-initial-data "$scratch/common.nullmap" \
    --output-dir "$scratch/ef-interior" "${resume[@]}" \
    >"$scratch/rejected-mode.log" 2>&1; then
  echo "EF restart accepted an excision-mode change" >&2
  exit 1
fi
grep -q "matching interior/excision controls" "$scratch/rejected-mode.log"
if "$binary" --coordinates eddington_finkelstein --ef-evolve-interior --resume \
    --ef-null-initial-data "$scratch/common.nullmap" \
    --output-dir "$scratch/ef" "${resume[@]}" \
    >"$scratch/rejected-mode-disabled.log" 2>&1; then
  echo "EF enabled restart accepted interior mode" >&2
  exit 1
fi
grep -q "matching interior/excision controls" "$scratch/rejected-mode-disabled.log"
"$python" - "$scratch/ef-interior" <<'PY'
import csv
import json
import pathlib
import struct
import sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())["high_compactness"]
assert summary["ef_excision_enabled"] is False
status = summary["ef_evolution_status"]
assert status["requested_end_time_reached"] is True
assert status["termination_reason"] == "requested_end_time"
assert status["minimum_active_radius"] > 0
assert status["spatial_resolution_assessment"] == "requires_convergence"
assert status["origin_mass_residual_available"] is True
assert 0 <= status["origin_mass_residual_over_adm"] < 1
assert status["density_peak_half_maximum_cells"] > 0
assert status["density_peak_radius"] > 0
assert status["minimum_active_radius_over_first_cell_width"] > 0
assert summary["excised_rest_mass"] == 0.0
data = (root / "high_eddington_finkelstein.restart").read_bytes()
assert struct.unpack_from("<I", data, 16)[0] == 2
assert data[44] == 0
with (root / "high_compactness_history.csv").open() as stream:
    rows = list(csv.DictReader(stream))
assert all(float(row["excision_radius"]) == 0.0 for row in rows)
assert all(float(row["excised_rest_weight"]) == 0.0 for row in rows)
assert all(int(row["active_marker_count"]) == 256 for row in rows)
with (root / "high_compactness_profiles.csv").open() as stream:
    profiles = list(csv.DictReader(stream))
assert all(int(row["excision_mask"]) == 0 for row in profiles)
PY

# Continue this deliberately coarse shell until its accepted numerical limit.
# The independent central constraint and the resolved length scales must expose
# spatial failure while preserving the actual timestep termination and restart.
"$binary" --coordinates eddington_finkelstein --ef-evolve-interior --resume \
  --ef-null-initial-data "$scratch/common.nullmap" \
  --output-dir "$scratch/ef-interior" "${resume[@]}" \
  --high-t-end 20 --high-output-steps 25 >/dev/null
"$python" - "$scratch/ef-interior" <<'PY'
import csv
import json
import math
import pathlib
import struct
import sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())["high_compactness"]
status = summary["ef_evolution_status"]
assert summary["outcome"] == "interior_evolution_limit"
assert status["termination_reason"] == "interior_time_resolution_limit"
assert status["requested_end_time_reached"] is False
assert status["spatial_resolution_assessment"] == "spatial_resolution_lost"
assert status["origin_mass_residual_available"] is True
data = (root / "high_eddington_finkelstein.restart").read_bytes()
version, time, step, adm, mode, count, fingerprint = struct.unpack_from(
    "<IdqdBQQ", data, 16)
assert version == 2 and mode == 0 and count == 256
assert time == status["final_advanced_time"] and step == summary["steps"]
records = [struct.unpack_from("<ddddQQB", data, 61 + 49 * i)
           for i in range(count)]
assert all(record[-1] == 0 for record in records)
assert summary["excised_rest_mass"] == summary["escaped_rest_mass"] == 0
assert math.isclose(status["origin_mass_residual_over_adm"],
                    abs(status["inner_face_mass"]) / adm, rel_tol=2e-15)
assert status["origin_mass_residual_over_adm"] >= 1
assert status["minimum_active_radius_over_first_cell_width"] < 1e-3
assert math.isclose(status["minimum_active_radius_over_first_cell_width"],
    min(record[1] for record in records) / status["first_cell_outer_radius"],
    rel_tol=2e-15)
with (root / "high_compactness_profiles.csv").open() as stream:
    final = [row for row in csv.DictReader(stream) if int(row["step"]) == step
             and row["metric_sample_kind"] == "cell_center"]
density = [float(row["rho_infall"]) for row in final]
peak = max(range(len(final)), key=density.__getitem__)
left = right = peak
while left > 0 and density[left - 1] >= .5 * density[peak]:
    left -= 1
while right + 1 < len(final) and density[right + 1] >= .5 * density[peak]:
    right += 1
assert status["density_peak_radius"] == float(final[peak]["r"])
assert status["density_peak_half_maximum_cells"] == right - left + 1
assert 1 <= status["density_peak_half_maximum_cells"] <= 3
print("EF accepted numerical limit: spatial failure, raw residual, density width, "
      "radius/cell ratio and retained state verified")
PY

# Focused neutral static energy-cutoff-shell integration.  The dedicated
# shifted sampler honors any even marker count, retains unique source IDs,
# and leaves its already-invariant weights untouched.
static_zero_common=(
  --coordinates schwarzschild
  --initialize-only
  --expected-outcome any
  --threads 2
  --high-initial-condition energy_cutoff_shell
  --high-markers 250
  --high-radial-cells 64
  --high-r-min 0.01
  --high-r-max 3
  --high-inner-boundary origin_parity
  --high-radial-grid linear
  --high-target-initial-adm-mass 0.15
  --high-t-end 0.01
  --high-dt-max 0.005
  --high-output-steps 1
)
mkdir -p "$scratch/static-r21" "$scratch/static-r22" "$scratch/static-sl08"
"$binary" --output-dir "$scratch/static-r21" \
  --physical-ic "$scratch/static-r21.phys" \
  --high-sampling-realization 21 "${static_zero_common[@]}" >/dev/null
"$binary" --output-dir "$scratch/static-r22" \
  --physical-ic "$scratch/static-r22.phys" \
  --high-sampling-realization 22 "${static_zero_common[@]}" >/dev/null
"$binary" --output-dir "$scratch/static-sl08" \
  --physical-ic "$scratch/static-r21.phys" \
  --high-sampling-realization 21 --high-angular-momentum-scale 0.8 \
  "${static_zero_common[@]}" >/dev/null

# A stale realization, odd count, and insufficient four-cell TSC vacuum
# buffer must each fail closed.
if "$binary" --output-dir "$scratch/static-stale" \
    --physical-ic "$scratch/static-r21.phys" \
    --high-sampling-realization 22 "${static_zero_common[@]}" \
    >"$scratch/static-stale.log" 2>&1; then
  echo "energy_cutoff_shell accepted a stale physical IC" >&2
  exit 1
fi
grep -q "does not match the configured continuum" "$scratch/static-stale.log"
if "$binary" --coordinates schwarzschild --initialize-only \
    --expected-outcome any --threads 2 \
    --output-dir "$scratch/static-odd" \
    --physical-ic "$scratch/static-odd.phys" \
    --high-initial-condition energy_cutoff_shell \
    --high-markers 251 --high-radial-cells 64 \
    --high-r-min 0.01 --high-r-max 3 \
    --high-inner-boundary origin_parity --high-radial-grid linear \
    --high-target-initial-adm-mass 0.15 \
    >"$scratch/static-odd.log" 2>&1; then
  echo "energy_cutoff_shell accepted an odd marker count" >&2
  exit 1
fi
grep -q "must be even" "$scratch/static-odd.log"
if "$binary" --coordinates schwarzschild --initialize-only \
    --expected-outcome any --threads 2 \
    --output-dir "$scratch/static-near-boundary" \
    --physical-ic "$scratch/static-near-boundary.phys" \
    --high-initial-condition energy_cutoff_shell \
    --high-markers 250 --high-radial-cells 64 \
    --high-r-min 0.01 --high-r-max 0.95 \
    --high-shell-center 0.5 \
    --high-inner-boundary origin_parity --high-radial-grid linear \
    --high-target-initial-adm-mass 0.15 \
    >"$scratch/static-near-boundary.log" 2>&1; then
  echo "energy_cutoff_shell accepted an unsafe TSC boundary buffer" >&2
  exit 1
fi
grep -q "at least four vacuum cells" "$scratch/static-near-boundary.log"

# Positive static-family SC->EF map and EF resume.  The explicit kick is a
# perturbation only; zero remains the exact current-free default.
static_map_common=(
  --expected-outcome any
  --physical-ic "$scratch/static-map.phys"
  --threads 2
  --high-initial-condition energy_cutoff_shell
  --high-markers 128
  --high-radial-cells 192
  --high-r-min 0.01
  --high-r-max 6
  --high-inner-boundary origin_parity
  --high-radial-grid linear
  --high-target-initial-adm-mass 0.15
  --high-radial-infall 0.1
  --high-sampling-realization 9
  --high-t-end 0.01
  --high-dt-max 0.005
  --high-output-steps 1
)
mkdir -p "$scratch/static-map-sc" "$scratch/static-map-ef"
"$binary" --coordinates schwarzschild \
  --output-dir "$scratch/static-map-sc" \
  --export-ef-null-initial-data "$scratch/static-map.null" \
  --ef-null-donor-duration 7 --ef-map-worldtube-radius 5.5 \
  "${static_map_common[@]}" >/dev/null
"$binary" --coordinates eddington_finkelstein \
  --output-dir "$scratch/static-map-ef" \
  --ef-null-initial-data "$scratch/static-map.null" \
  "${static_map_common[@]}" >/dev/null
static_map_resume=("${static_map_common[@]}")
for ((i=0; i<${#static_map_resume[@]}; ++i)); do
  if [[ ${static_map_resume[i]} == --high-t-end ]]; then
    static_map_resume[i+1]=0.015
  fi
done
"$binary" --coordinates eddington_finkelstein --resume \
  --output-dir "$scratch/static-map-ef" \
  --ef-null-initial-data "$scratch/static-map.null" \
  "${static_map_resume[@]}" >/dev/null

"$python" - "$scratch" <<'PY'
import csv
import json
import math
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])

def read_physical(path):
    data = path.read_bytes()
    assert data[:14] == b"VENUSEVPHYSIC1"
    version, = struct.unpack_from("=I", data, 16)
    count, = struct.unpack_from("=Q", data, 20)
    assert version == 1
    record = struct.Struct("=ddddQQ")
    assert len(data) == 36 + count * record.size
    return [record.unpack_from(data, 36 + q * record.size)
            for q in range(count)]

r21 = read_physical(root / "static-r21.phys")
r22 = read_physical(root / "static-r22.phys")
assert len(r21) == len(r22) == 250
assert (root / "static-r21.phys").read_bytes() != (
    root / "static-r22.phys").read_bytes()
for markers in (r21, r22):
    assert len({row[4] for row in markers}) == len(markers)
    assert len({row[5] for row in markers}) == len(markers)
    assert all(row[4] == row[5] for row in markers)
    for first, second in zip(markers[::2], markers[1::2]):
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert first[2] == -second[2]
        assert first[3] == second[3]

s21 = json.loads((root / "static-r21" / "summary.json").read_text())[
    "high_compactness"
]
s22 = json.loads((root / "static-r22" / "summary.json").read_text())[
    "high_compactness"
]
sl08 = json.loads((root / "static-sl08" / "summary.json").read_text())[
    "high_compactness"
]
for summary in (s21, s22, sl08):
    continuum = summary["energy_cutoff_shell"]
    assert continuum["sampling_rule"] == "shifted_stratified"
    assert continuum["sampling_pair_count"] == 125
    assert continuum["post_sampling_canonicalization_passes"] == 0
    assert continuum["marker_weight_measure"] == (
        "invariant_schwarzschild_wpic_rest_weight"
    )
    assert 0.0 < continuum["cutoff_energy_E0"] < 1.0
    assert continuum["continuum_binding_energy"] > 0.0
    assert continuum["continuum_binding_fraction"] > 0.0
    assert continuum["inner_matter_radius"] > 4 * (3.0 / 64.0)
    assert continuum["outer_matter_radius"] < 3.0 - 4 * (3.0 / 64.0)
    assert math.isclose(
        continuum["continuum_adm_mass"], 0.15,
        rel_tol=0.0, abs_tol=2.0e-16,
    )
    assert math.isclose(summary["fixed_family_adm_mass"], 0.15,
                        rel_tol=5.0e-11)
    assert summary["fixed_family_positive_binding_energy"] is True
assert math.isclose(
    sum(row[0] for row in r21),
    s21["energy_cutoff_shell"]["sampled_rest_weight"],
    rel_tol=2.0e-15,
)
assert s21["base_ic_fingerprint"] == sl08["base_ic_fingerprint"]
assert s21["realized_ic_fingerprint"] != sl08["realized_ic_fingerprint"]
assert math.isclose(
    sl08["mass_weighted_mean_specific_angular_momentum"],
    0.8 * sl08["energy_cutoff_shell"]["realized_ell0"],
    rel_tol=2.0e-15,
)

mapped_physical = read_physical(root / "static-map.phys")
for first, second in zip(mapped_physical[::2], mapped_physical[1::2]):
    assert math.isclose(0.5 * (first[2] + second[2]), -0.1,
                        rel_tol=0.0, abs_tol=8.0e-17)
source_by_id = {int(row[4]): int(row[5]) for row in mapped_physical}
assert len(source_by_id) == len(mapped_physical)
with (root / "static-map.null.crossings.csv").open(newline="") as stream:
    crossings = list(csv.DictReader(stream))
assert len(crossings) == len(mapped_physical)
assert len({row["source_id"] for row in crossings}) == len(crossings)
assert all(
    int(row["source_id"]) == source_by_id[int(row["id"])] == int(row["id"])
    for row in crossings
)
manifest = json.loads((root / "static-map.null.manifest.json").read_text())
ef = json.loads((root / "static-map-ef" / "summary.json").read_text())[
    "high_compactness"
]
assert manifest["base_ic_fingerprint"] == ef["base_ic_fingerprint"]
assert ef["energy_cutoff_shell"]["sampling_pair_count"] == 64
assert ef["energy_cutoff_shell"]["configured_inward_pair_mean_kick"] == 0.1
assert ef["steps"] >= 3
assert ef["fixed_family_positive_binding_energy"] is True
PY

echo "SC/EF WPIC coordinate and static-shell driver integration test passed"
