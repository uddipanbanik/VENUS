#!/usr/bin/env python3
"""Focused tests for the conservative fixed-ADM threshold classifier."""

import csv
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "tools" / "analyze_einstein_vlasov_wpic.py"
SPEC = importlib.util.spec_from_file_location("venus_wpic_analyzer", ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def write_null_map_preflight(root, summary):
    high = summary["high_compactness"]
    artifact_digest = "d" * 64
    artifacts = {
        name: {
            "path": str(root / (name + ".fixture")),
            "expected_sha256": artifact_digest,
            "actual_sha256": artifact_digest,
            "verified": True,
        }
        for name in (
            "null_initial_data", "crossings", "slice", "accepted_steps",
            "donor_history",
        )
    }
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
                "path": artifacts["donor_history"]["path"],
                "format_version": 2,
                "face_reconstruction": True,
            },
            "artifacts": artifacts,
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
    path = root / "null_map_preflight.json"
    rendered = json.dumps(audit, sort_keys=True) + "\n"
    path.write_text(rendered)
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    (root / "null_map_preflight.json.sha256").write_text(
        "{}  null_map_preflight.json\n".format(digest)
    )


def classifier_options():
    return types.SimpleNamespace(
        coordinates="any",
        angular_statistic="mass_weighted_rms_specific_angular_momentum",
        max_adm_drift=0.01,
        max_rest_mass_error=1.0e-12,
        end_time_tolerance=1.0e-6,
        fixed_adm_relative_tolerance=1.0e-8,
        allow_unfixed_adm_family=False,
        black_hole_compactness_min=0.99,
        late_window_fraction=0.30,
        minimum_late_samples=5,
        late_virial_min=0.50,
        late_virial_max=1.50,
        minimum_late_virial_fraction=0.75,
        halo_compression_fraction=0.95,
        halo_max_late_r50_over_initial=1.50,
        halo_max_r50_relative_spread=0.75,
        halo_max_dimensionless_r50_trend=0.25,
        halo_max_dimensionless_r90_trend=0.25,
        halo_max_compactness_absolute_spread=0.05,
        halo_max_compactness_absolute_trend=0.05,
        halo_max_anisotropic_jeans_relative_l1=0.25,
        halo_max_anisotropic_jeans_late_increase=0.05,
        minimum_late_jeans_fraction=0.75,
        halo_max_escaped_fraction=0.05,
        dispersal_min_escaped_fraction=0.20,
        dispersal_min_late_r50_over_initial=1.75,
        dispersal_min_dimensionless_r50_trend=0.20,
    )


def passing_post_horizon_gate_fixture():
    """Isolate aggregation tests from the separately tested EF gate."""
    return {
        "schema": "venus_wpic_post_horizon_intrinsic_gate_v1",
        "status": "pass",
        "classification_if_passed": "black_hole_candidate",
        "checks": {},
        "failed_checks": [],
        "measurements": {
            "post_horizon_duration_over_fixed_family_adm_mass": 6.0,
            "strictly_post_horizon_saved_samples": 6,
            "minimum_accepted_step_p_r": 0.125,
            "minimum_accepted_step_b": 0.5,
        },
    }


def passing_excision_sensitivity_fixture():
    """Represent the certified result exercised in the dedicated gate tests."""
    return {
        "schema": "venus_wpic_excision_sensitivity_gate_v1",
        "status": "pass",
        "certifies_black_hole": True,
        "checks": {},
        "failed_checks": [],
    }


def base_summary(outcome="undetermined", horizon=False, halo=False):
    return {
        "module": "weighted_einstein_vlasov_pic",
        "coordinates": "eddington_finkelstein",
        "high_compactness": {
            "coordinates": "eddington_finkelstein",
            "outcome": outcome,
            "total_marker_mass": 0.48,
            "target_initial_adm_mass": 0.58,
            "fixed_family_adm_mass": 0.58,
            "fixed_family_rest_weight": 0.60,
            "fixed_family_binding_energy": 0.02,
            "fixed_family_binding_fraction": 1.0 / 30.0,
            "fixed_family_binding_diagnostics_available": True,
            "fixed_family_positive_binding_energy": True,
            "initial_energy_at_infinity_proxy_definition": (
                "E_proxy=exp(mu(t=0,r))*sqrt(1+w^2+2*eta/r^2); "
                "instantaneous frozen polar-areal metric; not conserved "
                "for dynamical data"
            ),
            "initial_energy_at_infinity_proxy_minimum": 0.85,
            "initial_energy_at_infinity_proxy_maximum": 1.10,
            "initial_energy_at_infinity_proxy_rest_weighted_mean": 0.95,
            "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one": 0.10,
            "null_slice_conserved_adm_mass": 0.5800056,
            "initial_deposited_adm_mass": 0.5800057,
            "initial_adm_mass": 0.5800057,
            "eikonal_mapped_initial_data": True,
            "coordinate_map_manifest_sha256": "a" * 64,
            "initial_rest_weight_scale": 0.91,
            "realized_initial_rest_weight": 0.60,
            "adm_mass_relative_drift": 1.0e-4,
            "rest_mass_relative_error": 1.0e-14,
            "end_time": 10.0,
            "horizon_formed": horizon,
            "first_trapped_time": 8.0 if horizon else None,
            "virialized_halo": halo,
            "initial_r50": 2.0,
            "minimum_r50": 1.2,
            "final_r50": 1.35,
            "final_virial_ratio": 1.0,
            "final_anisotropic_jeans_relative_l1": 0.10,
            "escaped_rest_mass": 0.0,
            "mass_weighted_rms_specific_angular_momentum": 1.0,
            "angular_momentum_scale": 1.0,
            "fixed_specific_angular_momentum": 1.0,
            "base_ic_fingerprint": "0x0123456789abcdef",
            "realized_ic_fingerprint": "0xfedcba9876543210",
        },
    }


def write_run(root, summary, r50, r90, virial, compactness, jeans=None):
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(summary))
    if summary.get("coordinates") == "eddington_finkelstein":
        write_null_map_preflight(root, summary)
    columns = [
        "step", "advanced_time", "dt", "total_rest_weight",
        "active_rest_weight", "excised_rest_weight", "escaped_rest_weight",
        "adm_mass", "max_compactness", "min_a", "min_b",
        "apparent_horizon_radius", "r10", "r50", "r90", "virial_ratio",
        "excision_radius", "anisotropic_jeans_relative_l1",
    ]
    with (root / "high_compactness_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for step in range(21):
            writer.writerow({
                "step": step,
                "advanced_time": 0.5 * step,
                "dt": 0.5 if step else 0.0,
                "total_rest_weight": 0.4368,
                "active_rest_weight": 0.4368,
                "excised_rest_weight": 0.0,
                "escaped_rest_weight": 0.0,
                "adm_mass": 0.58,
                "max_compactness": compactness(step),
                "min_a": 0.4,
                "min_b": 0.8,
                "apparent_horizon_radius": 0.0,
                "r10": 0.5 * r50(step),
                "r50": r50(step),
                "r90": r90(step),
                "virial_ratio": virial(step),
                "excision_radius": 0.0,
                "anisotropic_jeans_relative_l1": (
                    0.10 if jeans is None else jeans(step)
                ),
            })
    return root / "summary.json"


class ThresholdTimeSeriesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.args = classifier_options()

    def tearDown(self):
        self.temporary.cleanup()

    def classify(
            self, name, summary, r50, r90, virial, compactness, jeans=None):
        path = write_run(
            self.root / name, summary, r50, r90, virial, compactness, jeans
        )
        return ANALYZER.classify_threshold_run(str(path), self.args)

    def test_raw_horizon_is_not_a_certified_black_hole(self):
        summary = base_summary("black_hole", horizon=True)
        record = self.classify(
            "black_hole", summary,
            lambda step: 2.0 - 0.04 * step,
            lambda step: 3.5 - 0.05 * step,
            lambda step: 0.5,
            lambda step: 0.40 + 0.031 * step,
        )
        self.assertEqual(record["classification"], "undetermined")
        self.assertTrue(record["evidence"]["black_hole_series_ok"])
        self.assertEqual(record["post_horizon_intrinsic_status"], "fail")
        self.assertIn(
            "accepted_step_audit_schema",
            record["evidence"]["post_horizon_evidence"]["failed_checks"],
        )

        summary = base_summary("black_hole", horizon=True)
        record = self.classify(
            "unsupported_black_hole", summary,
            lambda step: 2.0 - 0.02 * step,
            lambda step: 3.5 - 0.02 * step,
            lambda step: 0.5,
            lambda step: 0.70,
        )
        self.assertEqual(record["classification"], "undetermined")

    def test_sustained_bound_late_time_state_is_halo(self):
        summary = base_summary("virialized_halo", halo=True)
        record = self.classify(
            "halo", summary,
            lambda step: 2.0 - 0.08 * min(step, 10)
                + (0.02 if step % 2 else -0.02),
            lambda step: 3.6 - 0.08 * min(step, 10)
                + (0.03 if step % 2 else -0.03),
            lambda step: 1.0 + (0.08 if step % 2 else -0.08),
            lambda step: 0.50,
        )
        self.assertEqual(record["classification"], "halo")
        self.assertTrue(all(record["evidence"]["halo_checks"].values()))

    def test_relativistic_halo_does_not_require_newtonian_q_near_one(self):
        summary = base_summary("virialized_halo", halo=True)
        summary["high_compactness"]["final_virial_ratio"] = 3.9
        record = self.classify(
            "halo_q_outside_newtonian_band", summary,
            lambda step: 2.0 - 0.08 * min(step, 10)
                + (0.01 if step % 2 else -0.01),
            lambda step: 3.6 - 0.08 * min(step, 10)
                + (0.01 if step % 2 else -0.01),
            lambda step: 3.9,
            lambda step: 0.50,
            lambda step: 0.08,
        )
        self.assertEqual(record["classification"], "halo")
        self.assertFalse(record["evidence"]["late_virial_used_as_halo_gate"])
        self.assertEqual(
            record["evidence"]["late_virial_in_band_fraction"], 0.0
        )

    def test_missing_or_large_gr_balance_residual_fails_halo_closed(self):
        for name, jeans in (
                ("missing_jeans", lambda step: -1.0),
                ("large_jeans", lambda step: 0.60),
                ("growing_jeans", lambda step: 0.01 * step)):
            summary = base_summary("virialized_halo", halo=True)
            record = self.classify(
                name, summary,
                lambda step: 2.0 - 0.08 * min(step, 10),
                lambda step: 3.6 - 0.08 * min(step, 10),
                lambda step: 1.0,
                lambda step: 0.50,
                jeans,
            )
            self.assertEqual(record["classification"], "undetermined")
            jeans_checks = record["evidence"]["halo_checks"]
            self.assertFalse(
                jeans_checks["late_anisotropic_jeans_balance"]
                and jeans_checks["late_anisotropic_jeans_nonsecular"]
            )

    def test_near_trapped_trace_is_not_a_halo(self):
        # This state combines a compact shell with a positive energy drift:
        # it is bound, compressed, and radius-bounded, but C~0.996 and the GR
        # stress-balance residual is O(0.8).  Neither a missing horizon flag
        # nor an erroneous driver flag may certify it as a halo.
        summary = base_summary("virialized_halo", halo=True)
        record = self.classify(
            "near_trapped_false_halo", summary,
            lambda step: 2.0 - 0.08 * min(step, 10),
            lambda step: 3.6 - 0.08 * min(step, 10),
            lambda step: 1.0,
            lambda step: 0.996,
            lambda step: 0.80,
        )
        self.assertEqual(record["classification"], "undetermined")
        checks = record["evidence"]["halo_checks"]
        self.assertFalse(checks["not_near_horizon_collapse"])
        self.assertFalse(checks["late_anisotropic_jeans_balance"])

    def test_escape_or_sustained_expansion_is_dispersal(self):
        summary = base_summary()
        summary["high_compactness"]["escaped_rest_mass"] = 0.12
        record = self.classify(
            "dispersal", summary,
            lambda step: 2.0 + 0.12 * step,
            lambda step: 3.6 + 0.18 * step,
            lambda step: 2.2,
            lambda step: 0.30,
        )
        self.assertEqual(record["classification"], "dispersal")
        self.assertTrue(record["evidence"]["dispersal_checks"]["large_escape"])

    def test_ambiguous_and_non_fixed_adm_states_remain_undetermined(self):
        summary = base_summary()
        record = self.classify(
            "ambiguous", summary,
            lambda step: 2.0 - 0.02 * min(step, 6) + 0.02 * max(step - 6, 0),
            lambda step: 3.6 + 0.01 * step,
            lambda step: 2.0,
            lambda step: 0.45,
        )
        self.assertEqual(record["classification"], "undetermined")

        summary = base_summary("virialized_halo", halo=True)
        summary["high_compactness"]["fixed_family_adm_mass"] = 0.57
        record = self.classify(
            "wrong_adm", summary,
            lambda step: 2.0 - 0.08 * min(step, 10),
            lambda step: 3.6 - 0.08 * min(step, 10),
            lambda step: 1.0,
            lambda step: 0.50,
        )
        self.assertEqual(record["classification"], "undetermined")
        self.assertTrue(any("fixed-family ADM" in reason for reason in record["reasons"]))

    def test_non_increasing_saved_times_fail_closed(self):
        summary = base_summary()
        path = write_run(
            self.root / "bad_times", summary,
            lambda step: 2.0,
            lambda step: 3.6,
            lambda step: 1.0,
            lambda step: 0.4,
        )
        history = path.parent / "high_compactness_history.csv"
        with history.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]["advanced_time"] = rows[-2]["advanced_time"]
        with history.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "non-increasing output times"):
            ANALYZER.classify_threshold_run(str(path), self.args)

    def test_missing_or_tampered_null_map_preflight_fails_closed(self):
        summary = base_summary("black_hole", horizon=True)
        path = write_run(
            self.root / "tampered_map_audit", summary,
            lambda step: 2.0 - 0.04 * step,
            lambda step: 3.5 - 0.05 * step,
            lambda step: 0.5,
            lambda step: 0.40 + 0.031 * step,
        )
        (path.parent / "null_map_preflight.json.sha256").write_text(
            "{}  null_map_preflight.json\n".format("0" * 64)
        )
        record = ANALYZER.classify_threshold_run(str(path), self.args)
        self.assertEqual(record["classification"], "undetermined")
        self.assertTrue(any(
            "preflight authentication failed" in reason
            for reason in record["reasons"]
        ))

    def test_legacy_center_only_null_map_preflight_fails_closed(self):
        summary = base_summary("black_hole", horizon=True)
        path = write_run(
            self.root / "legacy_map_audit", summary,
            lambda step: 2.0 - 0.04 * step,
            lambda step: 3.5 - 0.05 * step,
            lambda step: 0.5,
            lambda step: 0.40 + 0.031 * step,
        )
        preflight = path.parent / "null_map_preflight.json"
        audit = json.loads(preflight.read_text())
        audit["manifest"]["donor_history"]["format_version"] = 1
        rendered = json.dumps(audit, sort_keys=True) + "\n"
        preflight.write_text(rendered)
        (path.parent / "null_map_preflight.json.sha256").write_text(
            "{}  null_map_preflight.json\n".format(
                hashlib.sha256(rendered.encode()).hexdigest()
            )
        )
        record = ANALYZER.classify_threshold_run(str(path), self.args)
        self.assertEqual(record["classification"], "undetermined")
        self.assertTrue(any(
            "face-aware v2 donor" in reason for reason in record["reasons"]
        ))

    def test_both_mapped_adm_mismatch_channels_fail_closed(self):
        for name, update in (
                ("donor_drift", {"null_slice_conserved_adm_mass": 0.582}),
                ("deposition_error", {"initial_deposited_adm_mass": 0.582})):
            summary = base_summary("black_hole", horizon=True)
            summary["high_compactness"].update(update)
            path = write_run(
                self.root / name, summary,
                lambda step: 2.0 - 0.04 * step,
                lambda step: 3.5 - 0.05 * step,
                lambda step: 0.5,
                lambda step: 0.40 + 0.031 * step,
            )
            record = ANALYZER.classify_threshold_run(str(path), self.args)
            self.assertEqual(record["classification"], "undetermined")
            self.assertTrue(any(
                "preflight authentication failed" in reason
                for reason in record["reasons"]
            ))

    def test_family_bracket_requires_realized_fixed_adm_accuracy(self):
        campaign = self.root / "campaign"
        campaign.mkdir()
        black_hole = base_summary("black_hole", horizon=True)
        black_hole["high_compactness"][
            "mass_weighted_rms_specific_angular_momentum"
        ] = 1.0
        halo = base_summary("virialized_halo", halo=True)
        halo["high_compactness"][
            "mass_weighted_rms_specific_angular_momentum"
        ] = 2.0
        write_run(
            campaign / "black_hole", black_hole,
            lambda step: 2.0 - 0.04 * step,
            lambda step: 3.5 - 0.05 * step,
            lambda step: 0.5,
            lambda step: 0.40 + 0.031 * step,
        )
        write_run(
            campaign / "halo", halo,
            lambda step: 2.0 - 0.08 * min(step, 10)
                + (0.02 if step % 2 else -0.02),
            lambda step: 3.6 - 0.08 * min(step, 10)
                + (0.03 if step % 2 else -0.03),
            lambda step: 1.0,
            lambda step: 0.50,
        )
        args = classifier_options()
        args.campaign_root = str(campaign)
        args.run_dir = None
        args.output = str(self.root / "threshold.json")
        args.csv_output = None
        args.allow_missing_provenance = True
        with mock.patch.object(
                ANALYZER, "parity_certificate_gate",
                return_value={"status": "pass", "test_fixture": True}), \
                mock.patch.object(
                    ANALYZER, "intrinsic_post_horizon_evidence",
                    return_value=passing_post_horizon_gate_fixture()), \
                mock.patch.object(
                    ANALYZER, "excision_sensitivity_gate",
                    return_value=passing_excision_sensitivity_fixture()):
            self.assertEqual(ANALYZER.classify_threshold(args), 0)
        result = json.loads(pathlib.Path(args.output).read_text())
        self.assertEqual(result["status"], "bracketed")
        self.assertEqual(result["fixed_initial_adm_family"]["status"], "pass")
        self.assertEqual(
            result["fixed_initial_adm_family"][
                "maximum_realized_relative_error"
            ],
            0.0,
        )

        preflight_sidecar = (
            campaign / "black_hole" / "null_map_preflight.json.sha256"
        )
        preflight_sidecar.write_text(
            "{}  null_map_preflight.json\n".format("0" * 64)
        )
        with mock.patch.object(
                ANALYZER, "parity_certificate_gate",
                return_value={"status": "pass", "test_fixture": True}), \
                mock.patch.object(
                    ANALYZER, "intrinsic_post_horizon_evidence",
                    return_value=passing_post_horizon_gate_fixture()), \
                mock.patch.object(
                    ANALYZER, "excision_sensitivity_gate",
                    return_value=passing_excision_sensitivity_fixture()):
            self.assertEqual(
                ANALYZER.classify_threshold(args),
                ANALYZER.SCIENTIFIC_INCONCLUSIVE,
            )
        result = json.loads(pathlib.Path(args.output).read_text())
        self.assertEqual(result["status"], "inconclusive_due_to_map_provenance")
        self.assertEqual(
            result["null_map_preflight_authentication"]["status"], "fail"
        )
        write_null_map_preflight(campaign / "black_hole", black_hole)

        dispersal = base_summary()
        dispersal["high_compactness"][
            "mass_weighted_rms_specific_angular_momentum"
        ] = 1.5
        dispersal["high_compactness"]["escaped_rest_mass"] = 0.12
        write_run(
            campaign / "dispersal", dispersal,
            lambda step: 2.0 + 0.12 * step,
            lambda step: 3.6 + 0.18 * step,
            lambda step: 2.2,
            lambda step: 0.30,
        )
        with mock.patch.object(
                ANALYZER, "parity_certificate_gate",
                return_value={"status": "pass", "test_fixture": True}), \
                mock.patch.object(
                    ANALYZER, "intrinsic_post_horizon_evidence",
                    return_value=passing_post_horizon_gate_fixture()), \
                mock.patch.object(
                    ANALYZER, "excision_sensitivity_gate",
                    return_value=passing_excision_sensitivity_fixture()):
            self.assertEqual(
                ANALYZER.classify_threshold(args),
                ANALYZER.SCIENTIFIC_INCONCLUSIVE,
            )
        result = json.loads(pathlib.Path(args.output).read_text())
        self.assertEqual(result["status"], "inconclusive_due_to_dispersal")
        self.assertIsNotNone(result["bracket"])

        halo["high_compactness"]["fixed_family_adm_mass"] = 0.57
        (campaign / "halo" / "summary.json").write_text(json.dumps(halo))
        with mock.patch.object(
                ANALYZER, "parity_certificate_gate",
                return_value={"status": "pass", "test_fixture": True}), \
                mock.patch.object(
                    ANALYZER, "intrinsic_post_horizon_evidence",
                    return_value=passing_post_horizon_gate_fixture()), \
                mock.patch.object(
                    ANALYZER, "excision_sensitivity_gate",
                    return_value=passing_excision_sensitivity_fixture()):
            self.assertEqual(
                ANALYZER.classify_threshold(args),
                ANALYZER.SCIENTIFIC_INCONCLUSIVE,
            )
        result = json.loads(pathlib.Path(args.output).read_text())
        self.assertEqual(result["fixed_initial_adm_family"]["status"], "fail")
        self.assertGreater(
            result["fixed_initial_adm_family"][
                "maximum_realized_relative_error"
            ],
            args.fixed_adm_relative_tolerance,
        )


if __name__ == "__main__":
    unittest.main()
