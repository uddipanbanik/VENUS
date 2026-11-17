#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import pathlib
import tempfile
import types
import unittest


ANALYZER = pathlib.Path(__file__).resolve().parents[1] / "tools" \
    / "analyze_einstein_vlasov_wpic.py"
SPEC = importlib.util.spec_from_file_location("venus_wpic_adm_analyzer", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_preflight(run, high):
    digest = "e" * 64
    audit = {
        "schema": "venus_wpic_null_map_preflight_v2",
        "status": "pass",
        "issues": [],
        "manifest": {
            "actual_sha256": high["coordinate_map_manifest_sha256"],
            "fixed_family_adm_mass": high.get("fixed_family_adm_mass"),
            "null_slice_conserved_adm_mass": high.get(
                "null_slice_conserved_adm_mass"
            ),
            "donor_to_null_slice_relative_drift": (
                abs(high["null_slice_conserved_adm_mass"]
                    - high["fixed_family_adm_mass"])
                / high["fixed_family_adm_mass"]
                if high.get("fixed_family_adm_mass") else None
            ),
            "eikonal_errors": {
                "reverse_time": 1.0e-7,
                "log_A": 2.0e-7,
                "legacy_maximum": 2.0e-7,
            },
            "residuals": {
                "maximum_marker_crossing_residual": 1.0e-12,
                "maximum_replay_time_error": 1.0e-6,
                "maximum_outer_b_normalization_error": 1.0e-6,
                "maximum_outer_source": 0.0,
            },
            "analytic_vacuum_residuals": {
                "maximum_analytic_vacuum_field_residual": 0.0,
                "maximum_analytic_vacuum_source": 0.0,
                "maximum_analytic_inverse_map_residual": 0.0,
            },
            "map_worldtube_radius": 14.0,
            "evolution_outer_radius": 20.0,
            "map_radial_cell_count": 1400,
            "evolution_radial_cell_count": 2000,
            "analytic_vacuum_extension": True,
            "donor_history": {
                "path": "donor_history",
                "format_version": 2,
                "face_reconstruction": True,
            },
            "artifacts": {
                name: {
                    "path": name,
                    "expected_sha256": digest,
                    "actual_sha256": digest,
                    "verified": True,
                }
                for name in (
                    "null_initial_data", "crossings", "slice",
                    "accepted_steps", "donor_history",
                )
            },
        },
        "summary": {
            "manifest_sha256": high["coordinate_map_manifest_sha256"],
            "fixed_family_adm_mass": high.get("fixed_family_adm_mass"),
            "null_slice_conserved_adm_mass": high.get(
                "null_slice_conserved_adm_mass"
            ),
            "initial_deposited_adm_mass": high.get(
                "initial_deposited_adm_mass"
            ),
            "initial_deposition_adm_relative_error_from_null_slice": (
                abs(high["initial_deposited_adm_mass"]
                    - high["null_slice_conserved_adm_mass"])
                / high["null_slice_conserved_adm_mass"]
            ),
        },
        "tolerances": {
            "eikonal_convergence": 2.0e-5,
            "crossing_residual": 1.0e-8,
            "composition": 5.0e-4,
            "outer_vacuum": 1.0e-12,
            "analytic_vacuum": 1.0e-12,
            "fixed_adm_relative": 1.0e-8,
            "donor_adm_drift": 1.0e-3,
            "initial_deposition_adm": 1.0e-3,
        },
    }
    rendered = json.dumps(audit, sort_keys=True) + "\n"
    (run / "null_map_preflight.json").write_text(rendered)
    (run / "null_map_preflight.json.sha256").write_text(
        "{}  null_map_preflight.json\n".format(
            hashlib.sha256(rendered.encode()).hexdigest()
        )
    )


class ThresholdAdmRatioTest(unittest.TestCase):
    def test_record_distinguishes_target_and_measured_adm(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-ratio-") as tmp:
            run = pathlib.Path(tmp)
            measured_adm = 0.5800056
            summary = {
                "module": "weighted_einstein_vlasov_pic",
                "coordinates": "eddington_finkelstein",
                "high_compactness": {
                    "outcome": "black_hole",
                    "horizon_formed": True,
                    "first_trapped_time": 15.0,
                    "end_time": 20.0,
                    "mass_weighted_rms_specific_angular_momentum": 1.0,
                    "total_marker_mass": 0.48,
                    "initial_adm_mass": measured_adm,
                    "initial_deposited_adm_mass": measured_adm,
                    "null_slice_conserved_adm_mass": 0.5800055,
                    "fixed_family_adm_mass": 0.58,
                    "target_initial_adm_mass": 0.58,
                    "eikonal_mapped_initial_data": True,
                    "coordinate_map_manifest_sha256": "b" * 64,
                    "adm_mass_relative_drift": 1.0e-5,
                    "rest_mass_relative_error": 1.0e-14,
                },
            }
            (run / "summary.json").write_text(json.dumps(summary))
            write_preflight(run, summary["high_compactness"])
            (run / "high_compactness_history.csv").write_text(
                "step,advanced_time,total_rest_weight,adm_mass,max_compactness,"
                "r10,r50,r90,virial_ratio\n"
                "0,0,0.48,0.580000001,0.2,1,2,3,0.5\n"
                "1,15,0.48,0.580000001,0.999,0.5,1,2,1\n"
            )
            args = types.SimpleNamespace(
                coordinates="eddington_finkelstein",
                angular_statistic="mass_weighted_rms_specific_angular_momentum",
                max_adm_drift=0.01,
                max_rest_mass_error=1.0e-12,
                allow_unfixed_adm_family=False,
                fixed_adm_relative_tolerance=1.0e-8,
                end_time_tolerance=1.0e-6,
                minimum_late_samples=5,
                late_window_fraction=0.30,
                black_hole_compactness_min=0.99,
            )
            record = MODULE.classify_threshold_run(
                str(run / "summary.json"), args)

            # This fixture omits the accepted-step P/b/excision audit.
            # Its ADM ratios remain reportable, but the raw driver outcome
            # must not be promoted to a certified black-hole endpoint.
            self.assertEqual(record["classification"], "undetermined")
            self.assertEqual(record["post_horizon_intrinsic_status"], "fail")
            self.assertAlmostEqual(
                record["angular_momentum_over_target_initial_adm_mass"],
                1.0 / 0.58,
            )
            self.assertAlmostEqual(
                record["angular_momentum_over_measured_initial_adm_mass"],
                1.0 / measured_adm,
            )
            self.assertEqual(
                record["angular_momentum_over_initial_adm_mass"],
                record["angular_momentum_over_measured_initial_adm_mass"],
            )
            self.assertAlmostEqual(
                record["angular_momentum_over_fixed_family_adm_mass"],
                1.0 / 0.58,
            )
            self.assertEqual(record["fixed_adm_relative_error"], 0.0)
            self.assertGreater(
                record["coordinate_slice_adm_relative_error_from_target"],
                1.0e-8,
            )

    def test_mapped_ef_missing_or_wrong_fixed_family_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-fixed-") as tmp:
            run = pathlib.Path(tmp)
            high = {
                "outcome": "black_hole",
                "horizon_formed": True,
                "first_trapped_time": 1.0,
                "end_time": 2.0,
                "mass_weighted_rms_specific_angular_momentum": 1.0,
                "total_marker_mass": 0.48,
                "initial_adm_mass": 0.58,
                "initial_deposited_adm_mass": 0.58,
                "null_slice_conserved_adm_mass": 0.58,
                "target_initial_adm_mass": 0.58,
                "eikonal_mapped_initial_data": True,
                "coordinate_map_manifest_sha256": "c" * 64,
                "adm_mass_relative_drift": 0.0,
                "rest_mass_relative_error": 0.0,
            }
            (run / "high_compactness_history.csv").write_text(
                "step,advanced_time,total_rest_weight,adm_mass,max_compactness,"
                "r10,r50,r90,virial_ratio\n"
                "0,0,0.48,0.58,0.2,1,2,3,0.5\n"
                "1,1,0.48,0.58,0.999,0.5,1,2,1\n"
            )
            args = types.SimpleNamespace(
                coordinates="eddington_finkelstein",
                angular_statistic="mass_weighted_rms_specific_angular_momentum",
                max_adm_drift=0.01,
                max_rest_mass_error=1.0e-12,
                allow_unfixed_adm_family=False,
                fixed_adm_relative_tolerance=1.0e-8,
                end_time_tolerance=1.0e-6,
                minimum_late_samples=5,
                late_window_fraction=0.30,
                black_hole_compactness_min=0.99,
            )
            for value in (None, 0.57):
                candidate = dict(high)
                if value is not None:
                    candidate["fixed_family_adm_mass"] = value
                (run / "summary.json").write_text(json.dumps({
                    "module": "weighted_einstein_vlasov_pic",
                    "coordinates": "eddington_finkelstein",
                    "high_compactness": candidate,
                }))
                write_preflight(run, candidate)
                record = MODULE.classify_threshold_run(
                    str(run / "summary.json"), args
                )
                self.assertEqual(record["classification"], "undetermined")
                self.assertTrue(record["reasons"])


if __name__ == "__main__":
    unittest.main()
