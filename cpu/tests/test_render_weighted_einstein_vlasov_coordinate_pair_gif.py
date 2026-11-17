#!/usr/bin/env python3
"""Focused tests for the same-event SC/EF three-panel GIF renderer."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS / "render_weighted_einstein_vlasov_coordinate_pair_gif.py"
SPEC = importlib.util.spec_from_file_location("ev_coordinate_pair_renderer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
RENDERER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDERER
SPEC.loader.exec_module(RENDERER)


SC_HEADER = (
    "step,time,r,rho,m,a,lambda,mu,lambda_t,p_radial,"
    "p_tangential_sum,j,metric_sample_kind,metric_cell,face_left_r,"
    "face_right_r,face_left_lambda,face_right_lambda,center_mu_r,"
    "face_left_mu,face_right_mu,face_left_mu_r,face_right_mu_r"
)
EF_HEADER = (
    "step,advanced_time,r,a,b,m,T_rr,S,T_vv,T_vr,T_theta_theta,"
    "metric_sample_kind,metric_cell,face_left_r,face_right_r,"
    "face_left_a,face_right_a,face_left_b,face_right_b"
)


def write_flat_profiles(root: Path, cells: int = 4) -> tuple[object, object]:
    """Write an exactly mappable flat pair with production face columns."""
    faces = np.linspace(0.0, 2.0, cells + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    sc_path = root / "sc.csv"
    ef_path = root / "ef.csv"
    with sc_path.open("w") as stream:
        stream.write(SC_HEADER + "\n")
        for step, tau in enumerate(np.arange(0.0, 6.0)):
            for cell, radius in enumerate(centers):
                rho = 10.0 + tau + radius
                stream.write(
                    f"{step},{tau},{radius},{rho},0,1,0,0,0,0,0,0,"
                    f"cell_center,{cell},{faces[cell]},{faces[cell + 1]},"
                    "0,0,0,0,0,0,0\n"
                )
            stream.write(
                f"{step},{tau},2,0,0,1,0,0,0,0,0,0,outer_face,{cells},"
                "2,2,0,0,0,0,0,0,0\n"
            )
    with ef_path.open("w") as stream:
        stream.write(EF_HEADER + "\n")
        for step, advanced_time in enumerate(np.arange(0.0, 8.0)):
            # The flat eikonal map with R=2 and v_offset=2 is v=tau+r.
            # Thus T_vv=10+v reproduces SC rho=10+tau+r at the same event.
            for cell, radius in enumerate(centers):
                t_vv = 10.0 + advanced_time
                stream.write(
                    f"{step},{advanced_time},{radius},1,1,0,0,0,{t_vv},0,0,"
                    f"cell_center,{cell},{faces[cell]},{faces[cell + 1]},"
                    "1,1,1,1\n"
                )
            stream.write(
                f"{step},{advanced_time},2,1,1,0,0,0,0,0,0,outer_face,{cells},"
                "2,2,1,1,1,1\n"
            )
    sc = RENDERER.parity.RectilinearProfile(
        str(sc_path),
        "time",
        {
            "m", "a", "lambda", "mu", "lambda_t", "rho",
            "p_radial", "p_tangential_sum", "j",
        },
    )
    ef = RENDERER.parity.RectilinearProfile(
        str(ef_path),
        "advanced_time",
        {"a", "b", "m", "T_rr", "S", "T_vv", "T_vr", "T_theta_theta"},
    )
    return sc, ef


class CoordinatePairRendererTests(unittest.TestCase):
    def test_zero_requested_post_frames_selects_nothing(self) -> None:
        self.assertEqual(RENDERER.evenly_spaced_values([1.0, 2.0], 0), [])

    def test_compact_native_profiles_keep_every_cell_and_exclude_outer_face(self) -> None:
        with tempfile.TemporaryDirectory(prefix="venus-pair-full-grid-test-") as tmp:
            sc, ef = write_flat_profiles(Path(tmp), cells=16)
            expected = (np.arange(16) + 0.5) / 8.0
            for profile, coordinates in ((sc, "schwarzschild"),
                                         (ef, "eddington_finkelstein")):
                self.assertIsInstance(profile.metric_reconstruction[0], dict)
                np.testing.assert_array_equal(RENDERER.cell_radii(profile), expected)
                curve = RENDERER.direct_curve(profile, 2.0, coordinates, 1.0e-6)
                np.testing.assert_array_equal(curve.radius, expected)
                self.assertEqual(curve.density.size, 16)
                self.assertEqual(curve.compactness.size, 16)
                self.assertTrue(np.all(np.isfinite(curve.density)))

    def test_legacy_donor_records_keep_every_cell_and_exclude_outer_face(self) -> None:
        with tempfile.TemporaryDirectory(prefix="venus-pair-legacy-grid-test-") as tmp:
            sc, ef = write_flat_profiles(Path(tmp), cells=16)
            expected = (np.arange(16) + 0.5) / 8.0
            for profile in (sc, ef):
                # The reverse-donor history reader retains one dictionary
                # per cell, whereas native CSV slices use arrays per field.
                first = profile.metric_reconstruction[0]
                profile.metric_reconstruction[0] = [
                    {name: values[cell] for name, values in first.items()}
                    for cell in range(16)
                ]
                np.testing.assert_array_equal(RENDERER.cell_radii(profile), expected)
                self.assertLess(RENDERER.cell_radii(profile)[-1], profile.radii[-1])

    def test_flat_same_event_map_overlays_density_and_compactness(self) -> None:
        with tempfile.TemporaryDirectory(prefix="venus-pair-render-test-") as tmp:
            sc, ef = write_flat_profiles(Path(tmp))
            frame = RENDERER.mapped_overlap_frame(
                sc,
                ef,
                tau=2.0,
                boundary_radius=2.0,
                v_offset=2.0,
                eikonal_substeps=2,
                eikonal_tolerance=1.0e-12,
                composition_tolerance=1.0e-12,
                minimum_a=1.0e-6,
                r_min=0.0,
                r_max=2.0,
                mapped_radial_samples=5,
                density_floor_fraction=1.0e-9,
            )

        self.assertEqual(frame.phase, "same_event_overlap")
        self.assertAlmostEqual(frame.ef_v_min, 2.25)
        # Display the four cell centers.  The exact r=2 outer face remains
        # available to the ray integrator as the normalization boundary.
        self.assertAlmostEqual(frame.ef_v_max, 3.75)
        self.assertEqual(frame.sc.radius.size, 4)
        self.assertEqual(frame.ef.radius.size, 4)
        np.testing.assert_allclose(frame.sc.density, frame.ef.density, atol=1e-13)
        np.testing.assert_allclose(frame.sc.compactness, 0.0, atol=0.0)
        np.testing.assert_allclose(frame.ef.compactness, 0.0, atol=0.0)
        self.assertLessEqual(frame.maximum_forward_refinement_error, 1.0e-14)
        self.assertLessEqual(frame.maximum_inverse_refinement_error, 1.0e-14)
        self.assertLessEqual(frame.maximum_composition_error, 1.0e-14)

    def test_ef_density_is_masked_when_static_observer_ceases_to_exist(self) -> None:
        self.assertAlmostEqual(
            RENDERER.reconstruct_ef_density(0.5, 2.0, 6.0, 1.0e-3),
            3.0,
        )
        self.assertTrue(np.isnan(
            RENDERER.reconstruct_ef_density(-0.1, 1.0, 1.0, 1.0e-3)
        ))
        self.assertTrue(np.isnan(
            RENDERER.reconstruct_ef_density(1.0e-4, 1.0, 1.0, 1.0e-3)
        ))

    def test_density_is_unmodified_and_power_law_slope_is_recovered(self) -> None:
        radius = np.geomspace(0.1, 10.0, 65)
        density = radius ** -2.0
        curve = RENDERER.ProfileCurve(
            radius=radius,
            density=density,
            compactness=np.zeros_like(radius),
            origin_parity=False,
        )
        density_x, density_y, _slope_x, slope = RENDERER.derived_curve(
            curve, density_floor_fraction=1.0e-12, slope_window=7
        )
        np.testing.assert_allclose(density_x, np.log10(radius))
        np.testing.assert_allclose(density_y, np.log10(density))
        np.testing.assert_allclose(slope[np.isfinite(slope)], -2.0, atol=1.0e-11)


if __name__ == "__main__":
    unittest.main()
