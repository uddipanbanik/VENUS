#!/usr/bin/env python3

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_wpic_dynamic_coordinate_convergence.py"
CERTIFIER = ROOT / "tools" / "compare_wpic_dynamic_coordinates.py"
MISSING = (
    "single-resolution agreement has no independent quantified "
    "discretization/PIC convergence evidence"
)

STATIC_SHELL_PHYSICS = {
    "ansatz": "f=C*(1-E/E0)_+^k*delta(ell-ell0)",
    "configured_amplitude_C": 0.5,
    "configured_ell0": 1.0,
    "configured_central_y": 0.13,
    "configured_cutoff_power_k": 1.0,
    "unscaled_adm_mass": 0.5491208983910536,
    "continuum_rescaled_to_target_adm_mass": True,
    "realized_amplitude_C": 0.323012293171208,
    "realized_cutoff_power_k": 1.0,
    "realized_ell0": 1.5479287029332416,
    "cutoff_energy_E0": 0.855490141036525,
    "inner_matter_radius": 2.8406900548847442,
    "outer_matter_radius": 5.181241546096228,
    "continuum_adm_mass": 0.85,
    "continuum_rest_weight": 0.8853796966557554,
    "continuum_binding_energy": 0.03537969665575551,
    "continuum_binding_fraction": 0.03995991413558638,
    "central_lapse": 0.7512019840417592,
    "sampling_rule": "shifted_stratified",
    "marker_weight_measure": "invariant_schwarzschild_wpic_rest_weight",
    "post_sampling_canonicalization_passes": 0,
    "configured_inward_pair_mean_kick": 0.01,
    "configured_kick_core_radius": 0.0,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sidecar(path):
    pathlib.Path(str(path) + ".sha256").write_text(
        "{}  {}\n".format(sha256(path), path.name)
    )


class DynamicCoordinateConvergenceBuilderTest(unittest.TestCase):
    def summary(self, coordinates, cells, markers, dt_max, realization,
                fingerprint, manifest_digest):
        high = {
            "initial_condition": "shell",
            "coordinates": coordinates,
            "inner_boundary": "origin_parity",
            "radial_grid": "linear",
            "r_min": 0.0,
            "r_max": 5.0,
            "end_time": 8.0,
            "gap_stop": 0.001,
            "total_marker_mass": 0.48,
            "scale_radius": 1.0,
            "initial_velocity_factor": 0.2,
            "initial_radial_infall": 0.0,
            "infall_core_radius": 0.0,
            "radial_importance_exponent": 1.0,
            "shell_center": 2.0,
            "shell_width": 0.3,
            "shell_radial_importance_factor": 1.7,
            "shell_radial_momentum": -0.65,
            "shell_radial_momentum_dispersion": 0.03,
            "shell_tangential_velocity_dispersion": 0.0,
            "fixed_specific_angular_momentum": 0.1,
            "angular_momentum_scale": 1.0,
            "target_initial_adm_mass": 0.48,
            # The SC normalization realizes the target exactly.  The mapped
            # EF deposit has a small resolution error which is certified by
            # initial_misner_sharp_mass and the convergence envelope.
            "initial_adm_mass": (
                0.48 if coordinates == "schwarzschild"
                else 0.48 * (1.0 - 1.0e-5)
            ),
            "timestep_omitted_mass_fraction": 1.0e-5,
            "markers": markers,
            "radial_cells": cells,
            "dt_max": dt_max,
            "output_steps": 1,
            "sampling_realization": realization,
            "mass_weighted_mean_specific_angular_momentum": 0.1,
            "mass_weighted_rms_specific_angular_momentum": 0.1,
            "maximum_specific_angular_momentum": 0.1,
            "base_ic_fingerprint": fingerprint,
            "eikonal_mapped_initial_data": True,
            "coordinate_map_manifest_sha256": manifest_digest,
        }
        return {"coordinates": coordinates, "high_compactness": high}

    def write_pair(self, root, name, cells, markers, dt_max, realization,
                   fingerprint, error):
        pair = root / name
        sc_dir = pair / "sc"
        ef_dir = pair / "ef"
        sc_dir.mkdir(parents=True)
        ef_dir.mkdir(parents=True)
        for run_dir in (sc_dir, ef_dir):
            (run_dir / "source.sha256").write_text(
                "{}  cpu/Makefile\n{}  cpu/src/driver.cpp\n".format(
                    "1" * 64, "2" * 64))
            (run_dir / "binary.sha256").write_text(
                "{}  /cluster/stage/venus_wpic\n".format("3" * 64))
        manifest_digest = hashlib.sha256(name.encode()).hexdigest()
        donor_path = pair / "donor_history.bin"
        donor_path.write_bytes(b"synthetic face-aware donor\n")
        sc_summary = sc_dir / "summary.json"
        ef_summary = ef_dir / "summary.json"
        sc_summary.write_text(json.dumps(self.summary(
            "schwarzschild", cells, markers, dt_max, realization,
            fingerprint, manifest_digest,
        )))
        ef_summary.write_text(json.dumps(self.summary(
            "eddington_finkelstein", cells, markers, dt_max, realization,
            fingerprint, manifest_digest,
        )))
        record = {
            "samples": 100,
            "relative_l2": error,
            "relative_linf": 1.1 * error,
            "maximum_predicted_magnitude": 1.0,
            "maximum_reference_magnitude": 1.0,
        }
        distribution_record = dict(record)
        distribution_record["relative_l2"] = error * 1.0e-9
        distribution_record["relative_linf"] = 1.1 * error * 1.0e-9
        initial_metric_record = dict(record)
        initial_metric_record["relative_l2"] = 1.25 * error
        initial_metric_record["relative_linf"] = 1.3 * error
        initial_stress_record = dict(record)
        initial_stress_record["relative_l2"] = 1.15 * error
        initial_stress_record["relative_linf"] = 1.2 * error
        audit = {
            "schema": "venus_wpic_dynamic_coordinate_certificate_v2",
            "status": "inconclusive",
            "eikonal_substeps": 8,
            "certifier_sha256": sha256(CERTIFIER),
            "inverse_equation": "tau_r+(a*b)^(-1)*tau_v=0",
            "metric_interpolation": dict(
                self.module_metric_interpolation()
            ),
            "issues": [MISSING],
            "failed_fields": [],
            "provenance": {
                "schwarzschild_summary": str(sc_summary),
                "eddington_finkelstein_summary": str(ef_summary),
                "coordinate_map_manifest": {
                    "schema": "venus_v3_sc_to_ef_null_map",
                    "schema_version": 2,
                    "donor_history_format_version": 2,
                    "saved_metric_interpolation": (
                        "SC center-face lambda plus monotone-Hermite mu; "
                        "EF center-face a,b"
                    ),
                    "actual_sha256": manifest_digest,
                    "schwarzschild_summary_sha256": manifest_digest,
                    "eddington_finkelstein_summary_sha256": manifest_digest,
                    "artifacts": {
                        "crossings": {
                            "verified": True,
                            "expected_sha256": "a" * 64,
                            "actual_sha256": "a" * 64,
                        },
                        "donor_history": {
                            "path": str(donor_path),
                            "verified": True,
                            "expected_sha256": sha256(donor_path),
                            "actual_sha256": sha256(donor_path),
                        },
                    },
                },
                "dense_time_reversed_donor_history": {
                    "path": str(donor_path),
                    "format_version": 2,
                    "production_metric_reconstruction": True,
                },
            },
            "mapping": {
                "inverse_maximum_step_refinement_error": 1.0e-7,
                "maximum_composition_time_error": 1.0e-7,
                "maximum_composition_jacobian_error": 1.0e-7,
            },
            "tolerances": {
                "eikonal_step_refinement": 2.0e-5,
                "forward_inverse_composition": 5.0e-4,
            },
            "errors": {
                "metric_a": record,
                "stress_T_vv": record,
                "moment_rho": record,
            },
            "initial_mapped_slice": {
                "errors": {
                    "initial_metric_a": initial_metric_record,
                    "initial_stress_T_vv": initial_stress_record,
                },
            },
            "distribution": {
                "mode": "strict",
                "failed_fields": [],
                "evidence": [{
                    "failed_fields": [],
                    "errors": {"distribution_eta": distribution_record},
                }],
            },
        }
        audit_path = pair / "audit.json"
        audit_path.write_text(json.dumps(audit, sort_keys=True))
        write_sidecar(audit_path)
        return audit_path

    @staticmethod
    def module_metric_interpolation():
        return {
            "schwarzschild": (
                "production_center_face_lambda_monotone_hermite_mu"
            ),
            "eddington_finkelstein": "production_center_face_linear_a_b",
            "donor_history_format_version": 2,
            "production_equivalent": True,
        }

    def matrix(self, root, duplicate_fingerprint=False):
        return {
            "coarse_a": self.write_pair(
                root, "coarse_a", 128, 4096, 0.01, 0, "0xaaa1", 4.0e-3),
            "coarse_b": self.write_pair(
                root, "coarse_b", 128, 4096, 0.01, 1,
                "0xaaa1" if duplicate_fingerprint else "0xbbb1", 5.0e-3),
            "fine_a": self.write_pair(
                root, "fine_a", 256, 16384, 0.005, 0, "0xaaa2", 1.0e-3),
            "fine_b": self.write_pair(
                root, "fine_b", 256, 16384, 0.005, 1, "0xbbb2", 1.2e-3),
        }

    def convert_matrix_to_static_shell(self, matrix):
        for audit_path in matrix.values():
            audit = json.loads(audit_path.read_text())
            provenance = audit["provenance"]
            for key in ("schwarzschild_summary", "eddington_finkelstein_summary"):
                summary_path = pathlib.Path(provenance[key])
                summary = json.loads(summary_path.read_text())
                high = summary["high_compactness"]
                high["initial_condition"] = "energy_cutoff_shell"
                high["fixed_specific_angular_momentum"] = -1.0
                high["initial_radial_infall"] = 0.01
                high["target_initial_adm_mass"] = 0.85
                high["initial_adm_mass"] = (
                    0.85 if high["coordinates"] == "schwarzschild"
                    else 0.85 * (1.0 - 1.0e-5)
                )
                high["energy_cutoff_shell"] = dict(STATIC_SHELL_PHYSICS)
                summary_path.write_text(json.dumps(summary))
        return matrix

    def run_builder(self, matrix, output, builder_sha256=None):
        if builder_sha256 is None:
            builder_sha256 = sha256(TOOL)
        return subprocess.run([
            sys.executable,
            str(TOOL),
            "--coarse-a", str(matrix["coarse_a"]),
            "--coarse-b", str(matrix["coarse_b"]),
            "--fine-a", str(matrix["fine_a"]),
            "--fine-b", str(matrix["fine_b"]),
            "--eikonal-substeps", "8",
            "--diagnostic-certifier-sha256", sha256(CERTIFIER),
            "--convergence-builder-sha256", builder_sha256,
            "--output", str(output),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
           check=False)

    def test_builds_authenticated_derived_bounds(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            output = root / "convergence.json"
            completed = self.run_builder(self.matrix(root), output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["resolution_levels"], 2)
            self.assertEqual(result["independent_pic_realizations"], 2)
            self.assertTrue(result["monotone_refinement"])
            self.assertEqual(result["eikonal_substeps"], 8)
            self.assertEqual(
                result["diagnostic_certifier_sha256"], sha256(CERTIFIER)
            )
            self.assertEqual(result["convergence_builder_sha256"], sha256(TOOL))
            self.assertEqual(
                result["metric_interpolation"],
                self.module_metric_interpolation(),
            )
            self.assertEqual(len(result["artifacts"]), 28)
            self.assertTrue(pathlib.Path(str(output) + ".sha256").is_file())
            self.assertGreater(
                result["uncertainty_bounds"]["metric"],
                result["pair_matrix"]["fine_a"]["errors"]["metric"],
            )
            self.assertAlmostEqual(
                result["pair_matrix"]["fine_a"]["errors"]["metric"],
                1.3e-3,
            )
            self.assertAlmostEqual(
                result["pair_matrix"]["fine_a"]["errors"]["stress"],
                1.2e-3,
            )
            self.assertEqual(result["binary_sha256"], "3" * 64)
            self.assertEqual(
                result["pair_matrix"]["coarse_b"]["source_manifest_sha256"],
                result["source_manifest_sha256"],
            )

    def test_rejects_duplicate_realization_ic(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            completed = self.run_builder(
                self.matrix(root, duplicate_fingerprint=True),
                root / "convergence.json",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("fingerprints are not independent", completed.stderr)

    def test_rejects_source_capture_drift_between_jobs(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.matrix(root)
            (matrix["coarse_b"].parent / "sc" / "source.sha256").write_text(
                "{}  cpu/Makefile\n{}  cpu/src/driver.cpp\n".format(
                    "4" * 64, "2" * 64))
            (matrix["coarse_b"].parent / "ef" / "source.sha256").write_text(
                "{}  cpu/Makefile\n{}  cpu/src/driver.cpp\n".format(
                    "4" * 64, "2" * 64))
            completed = self.run_builder(
                matrix, root / "convergence.json")
            self.assertEqual(completed.returncode, 2)
            self.assertIn("production source capture", completed.stderr)

    def test_rejects_legacy_center_only_pair_audit(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.matrix(root)
            path = matrix["coarse_a"]
            audit = json.loads(path.read_text())
            audit["metric_interpolation"] = {
                "schwarzschild": "legacy_center_center_linear",
                "eddington_finkelstein": "legacy_center_center_linear",
                "donor_history_format_version": 1,
                "production_equivalent": False,
            }
            path.write_text(json.dumps(audit, sort_keys=True))
            write_sidecar(path)
            completed = self.run_builder(
                matrix, root / "convergence.json"
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "production-equivalent center/face metric reconstruction",
                completed.stderr,
            )

    def test_rejects_legacy_coordinate_map_manifest_provenance(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.matrix(root)
            path = matrix["coarse_a"]
            audit = json.loads(path.read_text())
            audit["provenance"]["coordinate_map_manifest"][
                "schema_version"
            ] = 1
            path.write_text(json.dumps(audit, sort_keys=True))
            write_sidecar(path)
            completed = self.run_builder(
                matrix, root / "convergence.json"
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("face-aware v2 contract", completed.stderr)

    def test_rejects_mixed_eikonal_substep_contract(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.matrix(root)
            path = matrix["fine_b"]
            audit = json.loads(path.read_text())
            audit["eikonal_substeps"] = 4
            path.write_text(json.dumps(audit, sort_keys=True))
            write_sidecar(path)
            completed = self.run_builder(
                matrix, root / "convergence.json"
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "eikonal substeps do not match the campaign contract 8",
                completed.stderr,
            )

    def test_rejects_diagnostic_from_different_certifier(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.matrix(root)
            path = matrix["coarse_b"]
            audit = json.loads(path.read_text())
            audit["certifier_sha256"] = "f" * 64
            path.write_text(json.dumps(audit, sort_keys=True))
            write_sidecar(path)
            completed = self.run_builder(
                matrix, root / "convergence.json"
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("different parity certifier", completed.stderr)

    def test_rejects_builder_changed_after_submission(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            completed = self.run_builder(
                self.matrix(root), root / "convergence.json", "f" * 64
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("changed after campaign submission", completed.stderr)

    def test_authenticates_static_shell_continuum_physics(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.convert_matrix_to_static_shell(self.matrix(root))
            output = root / "convergence.json"
            completed = self.run_builder(matrix, output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(
                result["physics_configuration"]["energy_cutoff_shell"],
                STATIC_SHELL_PHYSICS,
            )

    def test_rejects_mixed_static_shell_continuum_physics(self):
        with tempfile.TemporaryDirectory(prefix="venus-convergence-") as tmp:
            root = pathlib.Path(tmp)
            matrix = self.convert_matrix_to_static_shell(self.matrix(root))
            audit = json.loads(matrix["fine_b"].read_text())
            summary_path = pathlib.Path(
                audit["provenance"]["eddington_finkelstein_summary"])
            summary = json.loads(summary_path.read_text())
            summary["high_compactness"]["energy_cutoff_shell"][
                "configured_central_y"] = 0.14
            summary_path.write_text(json.dumps(summary))
            completed = self.run_builder(
                matrix, root / "convergence.json")
            self.assertEqual(completed.returncode, 2)
            self.assertIn("differs in static-shell physics", completed.stderr)


if __name__ == "__main__":
    unittest.main()
