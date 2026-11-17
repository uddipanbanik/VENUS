#!/usr/bin/env python3
"""Focused initially-bound-family gates for the VENUS threshold tool."""

import csv
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
    "venus_wpic_bound_family_analyzer", ANALYZER_PATH
)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def classifier_options():
    return types.SimpleNamespace(
        coordinates="eddington_finkelstein",
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


def base_summary(outcome="virialized_halo", horizon=False, halo=True):
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
            "initial_energy_at_infinity_proxy_minimum": 0.84,
            "initial_energy_at_infinity_proxy_maximum": 1.12,
            "initial_energy_at_infinity_proxy_rest_weighted_mean": 0.96,
            "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one": 0.14,
            "null_slice_conserved_adm_mass": 0.5800001,
            "initial_deposited_adm_mass": 0.5800002,
            "initial_adm_mass": 0.5800002,
            "eikonal_mapped_initial_data": True,
            "coordinate_map_manifest_sha256": "a" * 64,
            "initial_rest_weight_scale": 0.90,
            "realized_initial_rest_weight": 0.60,
            "adm_mass_relative_drift": 1.0e-5,
            "rest_mass_relative_error": 1.0e-14,
            "end_time": 10.0,
            "horizon_formed": horizon,
            "first_trapped_time": 6.5 if horizon else -1.0,
            "horizon_time": 6.5 if horizon else -1.0,
            "post_horizon_advanced_time": 3.5 if horizon else 0.0,
            "evolved_past_horizon": horizon,
            "apparent_horizon_radius": 1.0 if horizon else -1.0,
            "ef_post_horizon_evidence": {
                "schema": "venus_wpic_ef_post_horizon_evidence_v1",
                "available": True,
                "accepted_step_audit_complete": True,
                "accepted_step_audit_samples": 20,
                "minimum_p_r_over_evolution": 0.25,
                "minimum_b_over_evolution": 0.75,
                "finite_fields_throughout": True,
                "fixed_adm_bookkeeping_throughout": True,
                "configured_excision_a_threshold": -0.01,
                "configured_excision_buffer_cells": 4,
                "maximum_excision_radius": 0.5 if horizon else 0.0,
                "maximum_excised_rest_weight": 0.1 if horizon else 0.0,
                "causal_excision_inside_horizon_seen": horizon,
                "causal_excision_valid_throughout": True,
                "excised_interior_rest_weight_seen": horizon,
            },
            "virialized_halo": halo,
            "initial_r50": 2.0,
            "minimum_r50": 1.2,
            "final_r50": 1.35,
            "final_virial_ratio": 1.0,
            "final_anisotropic_jeans_relative_l1": 0.10,
            "escaped_rest_mass": 0.0,
            "mass_weighted_rms_specific_angular_momentum": 1.0,
        },
    }


def write_history(root, black_hole=False):
    columns = [
        "step", "advanced_time", "dt", "total_rest_weight",
        "active_rest_weight", "excised_rest_weight", "escaped_rest_weight",
        "adm_mass", "max_compactness", "min_a", "min_b",
        "apparent_horizon_radius", "r10", "r50", "r90", "virial_ratio",
        "excision_radius", "anisotropic_jeans_relative_l1",
    ]
    with (root / "high_compactness_history.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for step in range(21):
            late_oscillation = 0.015 if step % 2 else -0.015
            post_horizon = black_hole and step > 13
            horizon_present = black_hole and step >= 13
            writer.writerow({
                "step": step,
                "advanced_time": 0.5 * step,
                "dt": 0.5 if step else 0.0,
                "total_rest_weight": 0.60,
                "active_rest_weight": 0.50 if post_horizon else 0.60,
                "excised_rest_weight": 0.10 if post_horizon else 0.0,
                "escaped_rest_weight": 0.0,
                "adm_mass": 0.58,
                "max_compactness": (
                    0.40 + 0.031 * step if black_hole else 0.50
                ),
                "min_a": 0.40,
                "min_b": 0.80,
                "apparent_horizon_radius": 1.0 if horizon_present else -1.0,
                "r10": 0.60,
                "r50": 2.0 - 0.08 * min(step, 10) + late_oscillation,
                "r90": 3.6 - 0.08 * min(step, 10) + late_oscillation,
                "virial_ratio": 1.0 + (0.05 if step % 2 else -0.05),
                "excision_radius": 0.5 if post_horizon else 0.0,
                "anisotropic_jeans_relative_l1": 0.10,
            })


class BoundFamilyThresholdTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="venus-bound-threshold-"
        )
        self.root = pathlib.Path(self.temporary.name)
        self.args = classifier_options()

    def tearDown(self):
        self.temporary.cleanup()

    def classify(self, summary, black_hole=False):
        run = self.root / ("run_{}".format(len(list(self.root.iterdir()))))
        run.mkdir()
        (run / "summary.json").write_text(json.dumps(summary))
        write_history(run, black_hole=black_hole)
        authenticated = {
            "path": str(run / "null_map_preflight.json"),
            "sha256": "b" * 64,
        }
        with mock.patch.object(
                ANALYZER, "authenticate_null_map_preflight",
                return_value=authenticated):
            return ANALYZER.classify_threshold_run(
                str(run / "summary.json"), self.args
            )

    def test_positive_global_binding_allows_but_does_not_replace_halo_gates(self):
        record = self.classify(base_summary())
        self.assertEqual(record["classification"], "halo")
        self.assertTrue(record["fixed_family_positive_binding_energy"])
        self.assertTrue(
            record["evidence"]["halo_checks"][
                "fixed_family_positive_binding_energy"
            ]
        )
        proxy = record["initial_energy_at_infinity_proxy"]
        self.assertTrue(proxy["available"])
        self.assertIn("not a halo gate", proxy["interpretation"])
        self.assertAlmostEqual(
            record[
                "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one"
            ],
            0.14,
        )

    def test_missing_zero_negative_or_inconsistent_binding_fails_halo_closed(self):
        candidates = []
        missing = base_summary()
        for key in (
            "fixed_family_rest_weight", "fixed_family_binding_energy",
            "fixed_family_binding_fraction",
            "fixed_family_binding_diagnostics_available",
            "fixed_family_positive_binding_energy",
        ):
            missing["high_compactness"].pop(key)
        candidates.append(missing)

        zero = base_summary()
        zero["high_compactness"].update({
            "fixed_family_rest_weight": 0.58,
            "fixed_family_binding_energy": 0.0,
            "fixed_family_binding_fraction": 0.0,
            "fixed_family_positive_binding_energy": False,
        })
        candidates.append(zero)

        negative = base_summary()
        negative["high_compactness"].update({
            "fixed_family_rest_weight": 0.55,
            "fixed_family_binding_energy": -0.03,
            "fixed_family_binding_fraction": -0.03 / 0.55,
            "fixed_family_positive_binding_energy": False,
        })
        candidates.append(negative)

        inconsistent = base_summary()
        inconsistent["high_compactness"]["fixed_family_binding_fraction"] = 0.2
        candidates.append(inconsistent)

        for summary in candidates:
            record = self.classify(summary)
            self.assertEqual(record["classification"], "undetermined")
            self.assertFalse(
                record["evidence"]["halo_checks"][
                    "fixed_family_positive_binding_energy"
                ]
            )
            self.assertTrue(any(
                "positive fixed_family_binding_fraction" in reason
                for reason in record["reasons"]
            ))

    def test_black_hole_candidate_does_not_require_binding(self):
        summary = base_summary("black_hole", horizon=True, halo=False)
        for key in (
            "fixed_family_rest_weight", "fixed_family_binding_energy",
            "fixed_family_binding_fraction",
            "fixed_family_binding_diagnostics_available",
            "fixed_family_positive_binding_energy",
        ):
            summary["high_compactness"].pop(key)
        record = self.classify(summary, black_hole=True)
        self.assertEqual(record["classification"], "black_hole_candidate")
        self.assertEqual(
            record["post_horizon_intrinsic_status"], "pass"
        )
        self.assertEqual(
            record["excision_sensitivity_status"], "not_supplied"
        )
        self.assertFalse(record["fixed_family_positive_binding_energy"])


if __name__ == "__main__":
    unittest.main()
