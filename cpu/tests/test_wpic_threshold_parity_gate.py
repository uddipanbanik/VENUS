#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest


ANALYZER = pathlib.Path(__file__).resolve().parents[1] / "tools" \
    / "analyze_einstein_vlasov_wpic.py"
SPEC = importlib.util.spec_from_file_location("venus_wpic_analyzer", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ThresholdParityGateTest(unittest.TestCase):
    def fixture(self, root):
        manifest_digest = "a" * 64
        donor_path = root / "donor_history.bin"
        donor_path.write_bytes(b"face-aware donor v2\n")
        audit_paths = [root / "coarse.json", root / "fine.json"]
        for index, audit_path in enumerate(audit_paths):
            audit_path.write_text("audit {}\n".format(index))
        convergence_artifacts = [{
            "path": str(audit_path),
            "sha256": digest(audit_path),
        } for audit_path in audit_paths]
        uncertainty_bounds = {
            "metric": 0.01,
            "stress": 0.02,
            "moment": 0.02,
            "distribution": 1.0e-10,
        }
        evidence_path = root / "convergence.json"
        evidence_path.write_text(json.dumps({
            "schema": "venus_wpic_dynamic_coordinate_convergence_v1",
            "status": "pass",
            "coordinate_map_manifest_sha256": manifest_digest,
            "resolution_levels": 2,
            "independent_pic_realizations": 2,
            "monotone_refinement": True,
            "eikonal_substeps": 8,
            "diagnostic_certifier_sha256": "c" * 64,
            "convergence_builder_sha256": "d" * 64,
            "metric_interpolation": dict(
                MODULE.PRODUCTION_METRIC_INTERPOLATION
            ),
            "uncertainty_bounds": uncertainty_bounds,
            "artifacts": convergence_artifacts,
        }))
        evidence_digest = digest(evidence_path)
        artifact = {
            "verified": True,
            "expected_sha256": "b" * 64,
            "actual_sha256": "b" * 64,
        }
        donor_artifact = {
            "path": str(donor_path),
            "verified": True,
            "expected_sha256": digest(donor_path),
            "actual_sha256": digest(donor_path),
        }
        certificate = {
            "schema": "venus_wpic_dynamic_coordinate_certificate_v2",
            "status": "pass",
            "issues": [],
            "failed_fields": [],
            "inverse_equation": "tau_r+(a*b)^(-1)*tau_v=0",
            "eikonal_substeps": 8,
            "certifier_sha256": "c" * 64,
            "metric_interpolation": dict(
                MODULE.PRODUCTION_METRIC_INTERPOLATION
            ),
            "provenance": {"coordinate_map_manifest": {
                "schema": "venus_v3_sc_to_ef_null_map",
                "schema_version": 2,
                "donor_history_format_version": 2,
                "saved_metric_interpolation": MODULE.SAVED_METRIC_INTERPOLATION,
                "actual_sha256": manifest_digest,
                "schwarzschild_summary_sha256": manifest_digest,
                "eddington_finkelstein_summary_sha256": manifest_digest,
                "artifacts": {
                    "crossings": artifact,
                    "donor_history": donor_artifact,
                },
            }, "dense_time_reversed_donor_history": {
                "path": str(donor_path),
                "format_version": 2,
                "production_metric_reconstruction": True,
            }},
            "distribution": {
                "mode": "strict", "evidence": [{"samples": 10}],
                "failed_fields": [],
            },
            "mapping": {
                "inverse_maximum_step_refinement_error": 1.0e-6,
                "maximum_composition_time_error": 1.0e-6,
                "maximum_composition_jacobian_error": 1.0e-6,
            },
            "tolerances": {
                "eikonal_step_refinement": 2.0e-5,
                "forward_inverse_composition": 5.0e-4,
            },
            "convergence": {
                "evidence": {
                    "path": str(evidence_path),
                    "actual_sha256": evidence_digest,
                    "sidecar_sha256": evidence_digest,
                    "sidecar_named_file": evidence_path.name,
                    "resolution_levels": 2,
                    "independent_pic_realizations": 2,
                    "monotone_refinement": True,
                    "eikonal_substeps": 8,
                    "diagnostic_certifier_sha256": "c" * 64,
                    "convergence_builder_sha256": "d" * 64,
                    "metric_interpolation": dict(
                        MODULE.PRODUCTION_METRIC_INTERPOLATION
                    ),
                    "uncertainty_bounds": uncertainty_bounds,
                    "artifacts": [{
                        "path": record["path"],
                        "verified": True,
                        "expected_sha256": record["sha256"],
                        "actual_sha256": record["sha256"],
                    } for record in convergence_artifacts],
                },
                "observed_maximum_relative_error": {
                    "metric": 0.005,
                    "stress": 0.01,
                    "moment": 0.01,
                    "distribution": 1.0e-12,
                },
            },
        }
        certificate_path = root / "certificate.json"
        certificate_path.write_text(json.dumps(certificate, sort_keys=True))
        certificate_digest = digest(certificate_path)
        pathlib.Path(str(certificate_path) + ".sha256").write_text(
            "{}  {}\n".format(certificate_digest, certificate_path.name)
        )
        return certificate_path, certificate

    def test_accepts_fully_authenticated_convergence(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-gate-") as tmp:
            certificate_path, _certificate = self.fixture(pathlib.Path(tmp))
            result = MODULE.parity_certificate_gate(types.SimpleNamespace(
                parity_certificate=str(certificate_path),
                parity_certificate_sha256=None,
            ))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["convergence_resolution_levels"], 2)
            self.assertEqual(
                result["convergence_independent_pic_realizations"], 2)
            self.assertEqual(result["eikonal_substeps"], 8)

    def test_rejects_convergence_with_different_eikonal_substeps(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-gate-") as tmp:
            root = pathlib.Path(tmp)
            certificate_path, certificate = self.fixture(root)
            certificate["convergence"]["evidence"]["eikonal_substeps"] = 4
            certificate_path.write_text(json.dumps(certificate, sort_keys=True))
            pathlib.Path(str(certificate_path) + ".sha256").write_text(
                "{}  {}\n".format(digest(certificate_path), certificate_path.name)
            )
            with self.assertRaisesRegex(ValueError, "different eikonal substeps"):
                MODULE.parity_certificate_gate(types.SimpleNamespace(
                    parity_certificate=str(certificate_path),
                    parity_certificate_sha256=None,
                ))

    def test_rejects_diagnostics_from_different_certifier(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-gate-") as tmp:
            root = pathlib.Path(tmp)
            certificate_path, certificate = self.fixture(root)
            certificate["convergence"]["evidence"][
                "diagnostic_certifier_sha256"
            ] = "f" * 64
            certificate_path.write_text(json.dumps(certificate, sort_keys=True))
            pathlib.Path(str(certificate_path) + ".sha256").write_text(
                "{}  {}\n".format(digest(certificate_path), certificate_path.name)
            )
            with self.assertRaisesRegex(ValueError, "different certifiers"):
                MODULE.parity_certificate_gate(types.SimpleNamespace(
                    parity_certificate=str(certificate_path),
                    parity_certificate_sha256=None,
                ))

    def test_rejects_missing_convergence(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-gate-") as tmp:
            root = pathlib.Path(tmp)
            certificate_path, certificate = self.fixture(root)
            del certificate["convergence"]
            certificate_path.write_text(json.dumps(certificate, sort_keys=True))
            pathlib.Path(str(certificate_path) + ".sha256").write_text(
                "{}  {}\n".format(digest(certificate_path), certificate_path.name)
            )
            with self.assertRaisesRegex(ValueError, "convergence audit"):
                MODULE.parity_certificate_gate(types.SimpleNamespace(
                    parity_certificate=str(certificate_path),
                    parity_certificate_sha256=None,
                ))

    def test_rejects_legacy_center_only_certificate(self):
        with tempfile.TemporaryDirectory(prefix="venus-threshold-gate-") as tmp:
            root = pathlib.Path(tmp)
            certificate_path, certificate = self.fixture(root)
            certificate["metric_interpolation"] = {
                "schwarzschild": "legacy_center_center_linear",
                "eddington_finkelstein": "legacy_center_center_linear",
                "donor_history_format_version": 1,
                "production_equivalent": False,
            }
            certificate_path.write_text(json.dumps(certificate, sort_keys=True))
            pathlib.Path(str(certificate_path) + ".sha256").write_text(
                "{}  {}\n".format(digest(certificate_path), certificate_path.name)
            )
            with self.assertRaisesRegex(ValueError, "production-equivalent"):
                MODULE.parity_certificate_gate(types.SimpleNamespace(
                    parity_certificate=str(certificate_path),
                    parity_certificate_sha256=None,
                ))


if __name__ == "__main__":
    unittest.main()
