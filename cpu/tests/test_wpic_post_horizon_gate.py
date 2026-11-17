#!/usr/bin/env python3
"""Fail-closed post-horizon and excision-sensitivity gate regressions."""

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
SPEC = importlib.util.spec_from_file_location(
    "venus_wpic_post_horizon_analyzer", ANALYZER_PATH
)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)

BINARY_DIGEST = "b" * 64
SOURCE_ENTRY_DIGEST = "d" * 64
SOURCE_MANIFEST_TEXT = "{}  src/solver.cpp\n".format(SOURCE_ENTRY_DIGEST)


def options():
    return types.SimpleNamespace(
        minimum_post_horizon_mass_times=5.0,
        minimum_post_horizon_saved_samples=3,
        fixed_adm_relative_tolerance=1.0e-8,
        max_adm_drift=0.01,
        max_rest_mass_error=1.0e-12,
        end_time_tolerance=1.0e-6,
        excision_sensitivity_certificate=None,
        excision_sensitivity_certificate_sha256=None,
        excision_sensitivity_max_horizon_time_spread_over_adm=1.0,
        excision_sensitivity_max_horizon_radius_relative_spread=0.05,
    )


def post_horizon_summary():
    return {
        "coordinates": "eddington_finkelstein",
        "high_compactness": {
            "outcome": "black_hole",
            "horizon_formed": True,
            "evolved_past_horizon": True,
            "first_trapped_time": 4.0,
            "horizon_time": 4.0,
            "post_horizon_advanced_time": 6.0,
            "target_initial_adm_mass": 1.0,
            "fixed_family_adm_mass": 1.0,
            "null_slice_conserved_adm_mass": 1.0,
            "initial_deposited_adm_mass": 1.0,
            "ef_post_horizon_evidence": {
                "schema": "venus_wpic_ef_post_horizon_evidence_v1",
                "available": True,
                "accepted_step_audit_complete": True,
                "accepted_step_audit_samples": 100,
                "minimum_p_r_over_evolution": 0.125,
                "minimum_b_over_evolution": 0.5,
                "finite_fields_throughout": True,
                "fixed_adm_bookkeeping_throughout": True,
                "configured_excision_a_threshold": -0.01,
                "configured_excision_buffer_cells": 4,
                "maximum_excision_radius": 0.5,
                "maximum_excised_rest_weight": 0.1,
                "causal_excision_inside_horizon_seen": True,
                "causal_excision_valid_throughout": True,
                "excised_interior_rest_weight_seen": True,
            },
        },
    }


def post_horizon_history():
    rows = []
    for step in range(11):
        after = step > 4
        horizon = step >= 4
        rows.append({
            "step": float(step),
            "time": float(step),
            "rest_mass": 1.2,
            "active_rest_mass": 1.1 if after else 1.2,
            "excised_rest_mass": 0.1 if after else 0.0,
            "escaped_rest_mass": 0.0,
            "adm_mass": 1.0,
            "max_compactness": 1.01 if horizon else 0.8,
            "min_a": -0.01 if horizon else 0.2,
            "min_b": 0.5,
            "apparent_horizon_radius": 1.0 if horizon else -1.0,
            "excision_radius": 0.5 if after else 0.0,
            "r10": 0.5,
            "r50": 1.0,
            "r90": 2.0,
            "virial_ratio": 1.0,
            "anisotropic_jeans_relative_l1": -1.0,
        })
    return rows


class IntrinsicPostHorizonGateTests(unittest.TestCase):
    def audit(self, summary=None, history=None):
        summary = post_horizon_summary() if summary is None else summary
        history = post_horizon_history() if history is None else history
        return ANALYZER.intrinsic_post_horizon_evidence(
            summary, summary["high_compactness"], history, options()
        )

    def test_complete_intrinsic_evidence_is_candidate_only(self):
        audit = self.audit()
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(
            audit["classification_if_passed"], "black_hole_candidate"
        )
        self.assertEqual(audit["failed_checks"], [])
        self.assertEqual(
            audit["measurements"]["strictly_post_horizon_saved_samples"], 6
        )
        self.assertEqual(
            audit["measurements"][
                "post_horizon_duration_over_fixed_family_adm_mass"
            ],
            6.0,
        )

    def test_each_missing_physical_evidence_class_fails_closed(self):
        mutations = {
            "short_interval": lambda high, rows: high.update({
                "first_trapped_time": 8.0,
                "horizon_time": 8.0,
                "post_horizon_advanced_time": 2.0,
            }),
            "missing_p": lambda high, rows: high[
                "ef_post_horizon_evidence"
            ].pop("minimum_p_r_over_evolution"),
            "nonpositive_b": lambda high, rows: high[
                "ef_post_horizon_evidence"
            ].update({"minimum_b_over_evolution": 0.0}),
            "nonfinite_fields": lambda high, rows: high[
                "ef_post_horizon_evidence"
            ].update({"finite_fields_throughout": False}),
            "lost_fixed_adm": lambda high, rows: high[
                "ef_post_horizon_evidence"
            ].update({"fixed_adm_bookkeeping_throughout": False}),
            "no_causal_excision": lambda high, rows: [
                row.update({"excision_radius": 0.0}) for row in rows
            ],
            "no_excised_weight": lambda high, rows: [
                row.update({
                    "active_rest_mass": row["rest_mass"],
                    "excised_rest_mass": 0.0,
                }) for row in rows
            ],
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                summary = post_horizon_summary()
                history = post_horizon_history()
                mutate(summary["high_compactness"], history)
                audit = self.audit(summary, history)
                self.assertEqual(audit["status"], "fail")
                self.assertTrue(audit["failed_checks"])

    def test_three_strictly_post_horizon_samples_are_required(self):
        summary = post_horizon_summary()
        history = post_horizon_history()[:7]
        summary["high_compactness"]["post_horizon_advanced_time"] = 2.0
        audit = self.audit(summary, history)
        self.assertIn(
            "minimum_strictly_post_horizon_saved_samples",
            audit["failed_checks"],
        )

    def test_deep_radius_cap_is_checked_against_saved_history(self):
        summary = post_horizon_summary()
        summary["high_compactness"]["ef_post_horizon_evidence"]["configured_excision_horizon_fraction"] = 0.1
        history = post_horizon_history()
        self.assertIn("excision_radius_cap_respected", self.audit(summary, history)["failed_checks"])
        for row in history:
            if row["excision_radius"] > 0:
                row["excision_radius"] = 0.09
        self.assertEqual(self.audit(summary, history)["status"], "pass")

    def test_local_excision_check_is_required_throughout(self):
        for evidence in (None, False):
            with self.subTest(evidence=evidence):
                summary = post_horizon_summary()
                audit = summary["high_compactness"]["ef_post_horizon_evidence"]
                if evidence is None:
                    audit.pop("causal_excision_valid_throughout")
                else:
                    audit["causal_excision_valid_throughout"] = evidence
                result = self.audit(summary, post_horizon_history())
                self.assertIn("causal_excision_inside_horizon",
                              result["failed_checks"])


class ExcisionSensitivityGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="venus-post-horizon-sensitivity-"
        )
        self.root = pathlib.Path(self.temporary.name)
        self.args = options()

    def tearDown(self):
        self.temporary.cleanup()

    def write_summary_artifact(self, name):
        path = self.root / name / "summary.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"name": name}, sort_keys=True))
        self.write_build_provenance(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    @staticmethod
    def write_build_provenance(
            summary_path, binary_digest=BINARY_DIGEST,
            source_manifest_text=SOURCE_MANIFEST_TEXT):
        run_dir = summary_path.parent
        (run_dir / "binary.sha256").write_text(
            "{}  /opt/venus/solver\n".format(binary_digest)
        )
        (run_dir / "source.sha256").write_text(source_manifest_text)

    @staticmethod
    def record(path, threshold, buffer, horizon_time, horizon_radius):
        intrinsic = {
            "measurements": {"first_trapped_time": horizon_time},
        }
        return {
            "run_dir": str(path.parent),
            "classification": "black_hole_candidate",
            "angular_momentum": 0.8,
            "fixed_family_adm_mass": 1.0,
            "base_ic_fingerprint": "0x1234",
            "realized_ic_fingerprint": "0x5678",
            "coordinate_map_manifest_sha256": "c" * 64,
            "physical_ic_sha256": "a" * 64,
            "apparent_horizon_radius": horizon_radius,
            "ef_post_horizon_evidence": {
                "configured_excision_a_threshold": threshold,
                "configured_excision_buffer_cells": buffer,
            },
            "evidence": {"post_horizon_evidence": intrinsic},
        }

    def make_certificate(self, primary, repeats, record_updates=None):
        certificate = self.root / "sensitivity.json"
        manifest_record = {
            "primary_summary_sha256": hashlib.sha256(
                primary.read_bytes()
            ).hexdigest(),
            "physical_ic_sha256": "a" * 64,
            "binary_sha256": BINARY_DIGEST,
            "source_manifest_sha256": hashlib.sha256(
                (primary.parent / "source.sha256").read_bytes()
            ).hexdigest(),
            "repeats": [{
                "summary": str(path.relative_to(self.root)),
                "summary_sha256": digest,
            } for path, digest in repeats],
        }
        manifest_record.update(record_updates or {})
        payload = {
            "schema": "venus_wpic_excision_sensitivity_certificate_v1",
            "status": "pass",
            "records": [manifest_record],
        }
        certificate.write_text(json.dumps(payload, sort_keys=True))
        digest = hashlib.sha256(certificate.read_bytes()).hexdigest()
        (self.root / "sensitivity.json.sha256").write_text(
            "{}  sensitivity.json\n".format(digest)
        )
        self.args.excision_sensitivity_certificate = str(certificate)
        return certificate

    def test_authenticated_independent_variations_certify_candidate(self):
        primary, _ = self.write_summary_artifact("primary")
        threshold_repeat, threshold_digest = self.write_summary_artifact(
            "threshold_repeat"
        )
        buffer_repeat, buffer_digest = self.write_summary_artifact(
            "buffer_repeat"
        )
        primary_record = self.record(primary, -0.01, 4, 4.0, 1.0)
        repeat_records = {
            str(threshold_repeat): self.record(
                threshold_repeat, -0.02, 4, 4.2, 1.01
            ),
            str(buffer_repeat): self.record(
                buffer_repeat, -0.01, 6, 4.1, 0.99
            ),
        }
        self.make_certificate(primary, [
            (threshold_repeat, threshold_digest),
            (buffer_repeat, buffer_digest),
        ])
        with mock.patch.object(
                ANALYZER, "classify_threshold_run",
                side_effect=lambda path, _args, check_excision_sensitivity=False:
                    repeat_records[path]):
            gate = ANALYZER.excision_sensitivity_gate(
                str(primary), primary_record, self.args
            )
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["certifies_black_hole"])
        self.assertTrue(
            gate["checks"]["independent_excision_a_threshold_variation"]
        )
        self.assertTrue(
            gate["checks"]["independent_excision_buffer_variation"]
        )
        self.assertEqual(
            gate["build_provenance"]["binary_sha256"], BINARY_DIGEST
        )
        self.assertEqual(
            gate["build_provenance"]["source_manifest_sha256"],
            hashlib.sha256(SOURCE_MANIFEST_TEXT.encode()).hexdigest(),
        )

    def test_missing_certificate_leaves_candidate_uncertified(self):
        primary, _ = self.write_summary_artifact("primary")
        gate = ANALYZER.excision_sensitivity_gate(
            str(primary), self.record(primary, -0.01, 4, 4.0, 1.0),
            self.args,
        )
        self.assertEqual(gate["status"], "not_supplied")
        self.assertFalse(gate["certifies_black_hole"])

    def test_claimed_pass_without_buffer_variation_fails_closed(self):
        primary, _ = self.write_summary_artifact("primary")
        first, first_digest = self.write_summary_artifact("first")
        second, second_digest = self.write_summary_artifact("second")
        primary_record = self.record(primary, -0.01, 4, 4.0, 1.0)
        repeat_records = {
            str(first): self.record(first, -0.02, 4, 4.1, 1.0),
            str(second): self.record(second, -0.03, 4, 4.2, 1.0),
        }
        self.make_certificate(primary, [
            (first, first_digest), (second, second_digest),
        ])
        with mock.patch.object(
                ANALYZER, "classify_threshold_run",
                side_effect=lambda path, _args, check_excision_sensitivity=False:
                    repeat_records[path]):
            gate = ANALYZER.excision_sensitivity_gate(
                str(primary), primary_record, self.args
            )
        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["certifies_black_hole"])
        self.assertIn(
            "independent_excision_buffer_variation", gate["failed_checks"]
        )

    def test_missing_primary_or_repeat_build_provenance_fails_closed(self):
        for target in ("primary", "threshold_repeat"):
            for artifact in ("binary.sha256", "source.sha256"):
                label = "{}_{}".format(target, artifact.replace(".", "_"))
                with self.subTest(target=target, artifact=artifact):
                    primary, _ = self.write_summary_artifact(
                        label + "_primary"
                    )
                    threshold_repeat, threshold_digest = (
                        self.write_summary_artifact(
                            label + "_threshold_repeat"
                        )
                    )
                    buffer_repeat, buffer_digest = self.write_summary_artifact(
                        label + "_buffer_repeat"
                    )
                    primary_record = self.record(
                        primary, -0.01, 4, 4.0, 1.0
                    )
                    repeat_records = {
                        str(threshold_repeat): self.record(
                            threshold_repeat, -0.02, 4, 4.2, 1.01
                        ),
                        str(buffer_repeat): self.record(
                            buffer_repeat, -0.01, 6, 4.1, 0.99
                        ),
                    }
                    self.make_certificate(primary, [
                        (threshold_repeat, threshold_digest),
                        (buffer_repeat, buffer_digest),
                    ])
                    target_path = (
                        primary if target == "primary" else threshold_repeat
                    )
                    (target_path.parent / artifact).unlink()
                    with mock.patch.object(
                            ANALYZER, "classify_threshold_run",
                            side_effect=lambda path, _args,
                            check_excision_sensitivity=False:
                                repeat_records[path]):
                        gate = ANALYZER.excision_sensitivity_gate(
                            str(primary), primary_record, self.args
                        )
                    self.assertEqual(gate["status"], "fail")
                    self.assertFalse(gate["certifies_black_hole"])
                    self.assertIn(
                        "run {} provenance is missing".format(
                            "binary" if artifact == "binary.sha256"
                            else "source"
                        ),
                        gate["error"],
                    )

    def test_repeat_binary_or_source_mismatch_fails_closed(self):
        for artifact in ("binary.sha256", "source.sha256"):
            label = "mismatch_{}".format(artifact.replace(".", "_"))
            with self.subTest(artifact=artifact):
                primary, _ = self.write_summary_artifact(label + "_primary")
                threshold_repeat, threshold_digest = self.write_summary_artifact(
                    label + "_threshold_repeat"
                )
                buffer_repeat, buffer_digest = self.write_summary_artifact(
                    label + "_buffer_repeat"
                )
                primary_record = self.record(primary, -0.01, 4, 4.0, 1.0)
                repeat_records = {
                    str(threshold_repeat): self.record(
                        threshold_repeat, -0.02, 4, 4.2, 1.01
                    ),
                    str(buffer_repeat): self.record(
                        buffer_repeat, -0.01, 6, 4.1, 0.99
                    ),
                }
                self.make_certificate(primary, [
                    (threshold_repeat, threshold_digest),
                    (buffer_repeat, buffer_digest),
                ])
                if artifact == "binary.sha256":
                    self.write_build_provenance(
                        threshold_repeat, binary_digest="e" * 64
                    )
                    expected_error = "changes executable provenance"
                else:
                    self.write_build_provenance(
                        threshold_repeat,
                        source_manifest_text="{}  src/solver.cpp\n".format(
                            "e" * 64
                        ),
                    )
                    expected_error = "changes source provenance"
                with mock.patch.object(
                        ANALYZER, "classify_threshold_run",
                        side_effect=lambda path, _args,
                        check_excision_sensitivity=False:
                            repeat_records[path]):
                    gate = ANALYZER.excision_sensitivity_gate(
                        str(primary), primary_record, self.args
                    )
                self.assertEqual(gate["status"], "fail")
                self.assertFalse(gate["certifies_black_hole"])
                self.assertIn(expected_error, gate["error"])

    def test_certificate_must_bind_primary_binary_and_source(self):
        cases = {
            "missing_binary": {"binary_sha256": None},
            "mismatched_binary": {"binary_sha256": "e" * 64},
            "missing_source": {"source_manifest_sha256": None},
            "mismatched_source": {"source_manifest_sha256": "e" * 64},
        }
        for label, update in cases.items():
            with self.subTest(label=label):
                primary, _ = self.write_summary_artifact(label + "_primary")
                threshold_repeat, threshold_digest = self.write_summary_artifact(
                    label + "_threshold_repeat"
                )
                buffer_repeat, buffer_digest = self.write_summary_artifact(
                    label + "_buffer_repeat"
                )
                primary_record = self.record(primary, -0.01, 4, 4.0, 1.0)
                self.make_certificate(
                    primary,
                    [(threshold_repeat, threshold_digest),
                     (buffer_repeat, buffer_digest)],
                    record_updates=update,
                )
                gate = ANALYZER.excision_sensitivity_gate(
                    str(primary), primary_record, self.args
                )
                self.assertEqual(gate["status"], "fail")
                self.assertFalse(gate["certifies_black_hole"])
                self.assertIn("certificate is not bound", gate["error"])


class ThresholdAggregationTests(unittest.TestCase):
    @staticmethod
    def aggregate_record(classification, angular_momentum, run_dir):
        return {
            "classification": classification,
            "angular_momentum": angular_momentum,
            "run_dir": run_dir,
            "coordinates": "eddington_finkelstein",
            "base_ic_fingerprint": "0x1234",
            "physical_ic_sha256": "a" * 64,
            "null_map_preflight_path": run_dir + "/null_map_preflight.json",
            "null_map_preflight_sha256": "b" * 64,
            "target_initial_adm_mass": 1.0,
            "fixed_adm_relative_error": 0.0,
            "coordinate_slice_adm_relative_error_from_target": 0.0,
            "null_slice_adm_relative_drift_from_family": 0.0,
            "initial_deposition_adm_relative_error_from_null_slice": 0.0,
            "angular_momentum_over_input_marker_mass": angular_momentum,
            "angular_momentum_over_initial_adm_mass": angular_momentum,
            "angular_momentum_over_measured_initial_adm_mass": angular_momentum,
            "angular_momentum_over_fixed_family_adm_mass": angular_momentum,
            "angular_momentum_over_target_initial_adm_mass": angular_momentum,
        }

    def test_candidate_never_closes_final_bracket(self):
        with tempfile.TemporaryDirectory(
                prefix="venus-candidate-bracket-") as temporary:
            output = pathlib.Path(temporary) / "threshold.json"
            args = types.SimpleNamespace(
                parity_certificate="unused",
                parity_certificate_sha256=None,
                campaign_root=None,
                run_dir=["unused"],
                coordinates="eddington_finkelstein",
                angular_statistic=(
                    "mass_weighted_rms_specific_angular_momentum"
                ),
                allow_missing_provenance=False,
                allow_unfixed_adm_family=False,
                fixed_adm_relative_tolerance=1.0e-8,
                output=str(output),
                csv_output=None,
            )
            records = [
                self.aggregate_record(
                    "black_hole_candidate", 0.8, "/candidate"
                ),
                self.aggregate_record("halo", 1.0, "/halo"),
            ]
            with mock.patch.object(
                    ANALYZER, "parity_certificate_gate",
                    return_value={"status": "pass"}), mock.patch.object(
                    ANALYZER, "discover_summary_paths",
                    return_value=["candidate", "halo"]), mock.patch.object(
                    ANALYZER, "classify_threshold_run",
                    side_effect=records):
                status = ANALYZER.classify_threshold(args)
            result = json.loads(output.read_text())
            self.assertEqual(status, ANALYZER.SCIENTIFIC_INCONCLUSIVE)
            self.assertEqual(
                result["status"],
                "inconclusive_due_to_uncertified_black_hole_candidate",
            )
            self.assertFalse(result["bracket_is_certified"])
            self.assertIsNone(result["bracket"])
            self.assertEqual(result["counts"]["black_hole_candidate"], 1)
            self.assertEqual(result["counts"]["black_hole_certified"], 0)


if __name__ == "__main__":
    unittest.main()
