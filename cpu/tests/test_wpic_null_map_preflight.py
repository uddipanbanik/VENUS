#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest


TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" \
    / "validate_wpic_null_map.py"
SPEC = importlib.util.spec_from_file_location("venus_null_preflight", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_donor_history(path, version=2, snapshots=101, cells=32):
    payload = bytearray(MODULE.DONOR_HISTORY_MAGIC)
    payload.extend(struct.pack("<IQQ", version, snapshots, cells))
    if version == 2:
        doubles_per_snapshot = 5 + 7 * cells + 4 * (cells + 1) + cells
    else:
        doubles_per_snapshot = 5 + 7 * cells
    payload.extend(b"\0" * (8 * snapshots * doubles_per_snapshot))
    path.write_bytes(payload)


class NullMapPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        artifacts = {}
        for name in MODULE.REQUIRED_ARTIFACTS:
            path = self.root / (name + ".dat")
            if name == "donor_history":
                write_donor_history(path)
            else:
                path.write_text(name + "\n")
            artifacts[name] = {"path": str(path), "sha256": digest(path)}
        self.manifest = {
            "schema": "venus_v3_sc_to_ef_null_map",
            "schema_version": 2,
            "complete": True,
            "all_markers_crossed_once": True,
            "outer_worldtube_verified_vacuum": True,
            "mapping_scope": "initial_null_hypersurface",
            "donor_direction": "East_time_reversed_outgoing_s=-tau",
            "time_reversal_momentum": "w_reverse=-w_schwarzschild",
            "eikonal_integration_method": MODULE.EIKONAL_METHOD,
            "eikonal_richardson_error_normalization": (
                MODULE.EIKONAL_NORMALIZATION
            ),
            "donor_history_format_version": (
                MODULE.DONOR_HISTORY_FORMAT_VERSION
            ),
            "saved_metric_interpolation": MODULE.SAVED_METRIC_INTERPOLATION,
            "marker_count": 64,
            "crossing_count": 64,
            "unique_crossing_count": 64,
            "radial_cell_count": 32,
            "evolution_radial_cell_count": 32,
            "map_radial_cell_count": 20,
            "accepted_step_count": 100,
            "history_snapshot_count": 101,
            "maximum_eikonal_reverse_time_richardson_error": 1.0e-7,
            "maximum_eikonal_log_A_richardson_error": 2.0e-7,
            "maximum_eikonal_ode_residual": 2.0e-7,
            "maximum_marker_crossing_residual": 1.0e-14,
            "maximum_replay_time_error": 0.0,
            "maximum_outer_b_normalization_error": 0.0,
            "outer_b_normalization_tolerance": 1.0e-8,
            "maximum_outer_source": 0.0,
            "target_initial_adm_mass": 0.58,
            "donor_initial_adm_mass": 0.58,
            "null_slice_adm_mass": 0.5800056,
            "donor_adm_mass_relative_drift": 0.0000056 / 0.58,
            "outer_boundary_radius": 12.0,
            "map_worldtube_radius": 12.0,
            "evolution_outer_radius": 20.0,
            "analytic_vacuum_extension": True,
            "analytic_vacuum_extension_start_index": 20,
            "grid_fingerprint": "0x0123456789abcdef",
            "map_grid_fingerprint": "0xfedcba9876543210",
            "required_worldtube_clearance": 0.5,
            "maximum_donor_marker_radius": 11.0,
            "maximum_crossing_radius": 10.5,
            "minimum_worldtube_matter_clearance": 1.0,
            "minimum_crossing_radius_clearance": 1.5,
            "maximum_analytic_vacuum_field_residual": 0.0,
            "maximum_analytic_vacuum_source": 0.0,
            "maximum_analytic_inverse_map_residual": 0.0,
            "minimum_analytic_vacuum_a": 0.9,
            "ef_v_offset": 24.0,
            "artifacts": artifacts,
        }
        self.manifest_path = self.root / "map.manifest.json"
        self.output = self.root / "preflight.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        manifest_sha = digest(self.manifest_path)
        summary = {
            "module": "weighted_einstein_vlasov_pic",
            "coordinates": "schwarzschild",
            "high_compactness": {
                "eikonal_mapped_initial_data": True,
                "coordinate_map_manifest_sha256": manifest_sha,
                "markers": 64,
                "radial_cells": 32,
                "eikonal_boundary_radius": 12.0,
                "eikonal_evolution_outer_radius": 20.0,
                "eikonal_analytic_vacuum_extension": True,
                "eikonal_v_offset": 24.0,
                "fixed_family_adm_mass": 0.58,
                "null_slice_conserved_adm_mass": 0.5800056,
                "initial_deposited_adm_mass": 0.58,
            },
        }
        summary_path = self.root / "summary.json"
        summary_path.write_text(json.dumps(summary))
        return summary_path

    def invoke(self):
        summary = self.write_inputs()
        status = MODULE.main([
            "--manifest", str(self.manifest_path),
            "--summary", str(summary),
            "--output", str(self.output),
        ])
        return status, json.loads(self.output.read_text())

    def test_pass_authenticates_all_artifacts_and_adm_channels(self):
        status, result = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(pathlib.Path(str(self.output) + ".sha256").is_file())
        self.assertAlmostEqual(
            result["manifest"]["fixed_family_adm_mass"], 0.58
        )
        self.assertAlmostEqual(
            result["manifest"]["null_slice_conserved_adm_mass"], 0.5800056
        )
        self.assertTrue(all(
            record["verified"]
            for record in result["manifest"]["artifacts"].values()
        ))
        donor = result["manifest"]["donor_history"]
        self.assertEqual(donor["format_version"], 2)
        self.assertTrue(donor["face_reconstruction"])

    def test_legacy_center_only_donor_fails_closed_even_when_rehashed(self):
        record = self.manifest["artifacts"]["donor_history"]
        path = pathlib.Path(record["path"])
        write_donor_history(path, version=1)
        record["sha256"] = digest(path)
        status, result = self.invoke()
        self.assertEqual(status, MODULE.INCONCLUSIVE)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any(
            "format version 1 is legacy" in issue
            for issue in result["issues"]
        ))

    def test_truncated_face_payload_fails_closed_even_when_rehashed(self):
        record = self.manifest["artifacts"]["donor_history"]
        path = pathlib.Path(record["path"])
        path.write_bytes(path.read_bytes()[:-8])
        record["sha256"] = digest(path)
        status, result = self.invoke()
        self.assertEqual(status, MODULE.INCONCLUSIVE)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any(
            "payload" in issue for issue in result["issues"]
        ))

    def test_component_error_and_legacy_mismatch_fail_closed(self):
        self.manifest[
            "maximum_eikonal_reverse_time_richardson_error"
        ] = 3.0e-5
        self.manifest["maximum_eikonal_ode_residual"] = 2.0e-7
        status, result = self.invoke()
        self.assertEqual(status, MODULE.INCONCLUSIVE)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("reverse_time" in issue for issue in result["issues"]))
        self.assertTrue(any("legacy" in issue for issue in result["issues"]))

    def test_artifact_tampering_and_wrong_family_mass_fail_closed(self):
        path = pathlib.Path(
            self.manifest["artifacts"]["crossings"]["path"]
        )
        path.write_text("tampered\n")
        self.manifest["donor_initial_adm_mass"] = 0.57
        status, result = self.invoke()
        self.assertEqual(status, MODULE.INCONCLUSIVE)
        self.assertTrue(any("crossings" in issue for issue in result["issues"]))
        self.assertTrue(any("fixed-family" in issue for issue in result["issues"]))

    def test_large_ef_deposition_mismatch_fails_closed(self):
        summary_path = self.write_inputs()
        summary = json.loads(summary_path.read_text())
        summary["high_compactness"]["initial_deposited_adm_mass"] = 0.60
        summary_path.write_text(json.dumps(summary))
        status = MODULE.main([
            "--manifest", str(self.manifest_path),
            "--summary", str(summary_path),
            "--output", str(self.output),
        ])
        result = json.loads(self.output.read_text())
        self.assertEqual(status, MODULE.INCONCLUSIVE)
        self.assertTrue(any(
            "deposition ADM mismatch" in issue for issue in result["issues"]
        ))

    def test_split_worldtube_clearance_and_inverse_map_fail_closed(self):
        self.manifest["minimum_worldtube_matter_clearance"] = 0.25
        self.manifest["maximum_donor_marker_radius"] = 11.75
        self.manifest["maximum_analytic_inverse_map_residual"] = 1.0e-4
        status, result = self.invoke()
        self.assertEqual(status, MODULE.INCONCLUSIVE)
        self.assertTrue(any(
            "matter support" in issue for issue in result["issues"]
        ))
        self.assertTrue(any(
            "inverse_map" in issue for issue in result["issues"]
        ))


if __name__ == "__main__":
    unittest.main()
