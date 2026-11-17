#!/usr/bin/env python3
"""Targeted tests for Einstein--Vlasov WPIC profile rendering diagnostics."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "render_weighted_einstein_vlasov_pic_profile_gifs.py"
)
SPEC = importlib.util.spec_from_file_location("ev_wpic_profile_renderer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load renderer from {MODULE_PATH}")
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


class StableLogSlopeTests(unittest.TestCase):
    def test_origin_parity_retains_and_recovers_even_core_slope(self) -> None:
        spacing = 0.01
        faces = np.arange(129, dtype=float) * spacing
        inner = faces[:-1]
        outer = faces[1:]
        # Spherical finite-volume cell centroids are not uniformly spaced in
        # the first few cells even when their faces are linear in radius.
        radius = 0.75 * (
            (outer**4 - inner**4) / (outer**3 - inner**3)
        )
        coefficient = 1.7
        density = np.exp(-coefficient * radius**2)

        slope_radius, smooth_log_density, slope = (
            RENDERER.boundary_aware_log_profile(
                radius,
                density,
                floor_fraction=0.0,
                window_points=21,
                origin_parity=True,
            )
        )
        wrapper_radius, wrapper_slope = RENDERER.stable_log_slope(
            radius,
            density,
            floor_fraction=0.0,
            window_points=21,
            origin_parity=True,
        )
        expected = -2.0 * coefficient * slope_radius**2
        finite = np.isfinite(slope)

        self.assertTrue(finite[0])
        self.assertTrue(finite[-1])
        self.assertAlmostEqual(slope_radius[0], radius[0])
        np.testing.assert_allclose(wrapper_radius, slope_radius)
        np.testing.assert_allclose(wrapper_slope, slope)
        np.testing.assert_allclose(
            smooth_log_density,
            np.log10(density),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertLess(np.max(np.abs(slope[finite] - expected[finite])), 1.0e-11)

    def test_finite_inner_boundary_keeps_exact_power_law_slope(self) -> None:
        radius = np.geomspace(0.015, 20.0, 384)
        density = radius ** -3.0

        slope_radius, slope = RENDERER.stable_log_slope(
            radius,
            density,
            floor_fraction=0.0,
            window_points=41,
            origin_parity=False,
        )
        finite = np.isfinite(slope)

        self.assertTrue(finite[0])
        self.assertTrue(finite[-1])
        self.assertAlmostEqual(slope_radius[0], radius[0])
        self.assertLess(np.max(np.abs(slope[finite] + 3.0)), 1.0e-11)

    def test_axes_use_the_full_radial_mesh(self) -> None:
        radius = (np.arange(64, dtype=float) + 0.5) * 0.05
        data = np.zeros(
            radius.size,
            dtype=[
                ("time", float),
                ("r", float),
                ("rho", float),
                ("compactness", float),
            ],
        )
        data["r"] = radius
        data["rho"] = np.exp(-radius**2)
        data["compactness"] = radius / radius[-1]

        limits = RENDERER.collect_limits(
            data,
            np.array([0.0]),
            [("rho", "compactness")],
            floor_fraction=1.0e-6,
            density_window=11,
            slope_window=11,
            origin_parity=True,
        )

        self.assertAlmostEqual(limits[0][0], np.log10(radius[0]))
        self.assertAlmostEqual(limits[0][1], np.log10(radius[-1]))


if __name__ == "__main__":
    unittest.main()
