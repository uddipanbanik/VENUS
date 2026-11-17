#!/usr/bin/env python3
"""Check the scientific conventions of the WPIC analysis tools."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import plot_subhalo_virial_1 as virial
import render_weighted_pic_gamma_fits_gifs as density
import plot_wpic_scaled_profiles as scaled
import render_density_slope_sigma2_gif as dispersion
import spherical_profile_rendering as profiles


class WPICAnalysisConventions(unittest.TestCase):
    """Compare inferred geometry, integrals and slopes to independent controls."""
    def test_generic_reader_excludes_a_partial_physical_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'profiles.csv'
            radii = np.array([1., 2., 3., 4.])
            rows = [(time, time, radius, radius**-1.5)
                    for time, count in [(0, 4), (1, 4), (2, 2)]
                    for radius in radii[:count]]
            np.savetxt(path, rows, delimiter=',',
                       header='time,step,r,rho', comments='')
            loaded = profiles.load_profiles(path)
            np.testing.assert_array_equal(np.unique(loaded['time']), [0., 1.])
            self.assertEqual(len(loaded), 8)

    def test_early_frame_allocation_retains_the_final_physical_state(self):
        times = np.arange(11, dtype=float)
        for early_frames in [4, 20]:
            selected = profiles.build_target_times(
                times, 4, 3., early_frames, 'time')
            self.assertEqual(len(selected), 4)
            self.assertEqual(selected[0], 0.)
            self.assertEqual(selected[-1], 10.)
        np.testing.assert_array_equal(
            profiles.build_target_times(times, 1, 3., 1, 'time'), [10.])
        np.testing.assert_array_equal(profiles.unique_index_sample(times, 1), [10.])

    def test_centered_pressure_preserves_variance_under_bulk_velocity(self):
        rows = np.zeros(2, dtype=[('rho', float), ('p', float), ('j', float)])
        rows['rho'] = 2
        rows['p'] = 8  # Velocities about either mean differ by +/-2.
        rows['j'] = 2 * np.array([3., 103.])
        np.testing.assert_allclose(dispersion.radial_dispersion_squared(rows), 4)
        rows['p'] += rows['j'] ** 2 / rows['rho']
        np.testing.assert_allclose(
            dispersion.radial_dispersion_squared(rows, 'raw'), 4)

    def test_rebinned_dispersion_retains_between_cell_bulk_motion(self):
        rows = np.zeros(2, dtype=[(name, float) for name in
                        ('time', 'r', 'rho', 'p', 'pt', 'j')])
        rows['r'] = [1, 3]
        volumes = 4 * np.pi / 3 * np.diff(np.array([0., 2., 3.]) ** 3)
        rows['rho'] = 1 / volumes  # Equal unit masses in unequal shells.
        rows['p'] = rows['rho']    # Unit variance about each cell's mean.
        rows['j'] = rows['rho'] * np.array([3., -3.])
        merged = dispersion.conservative_adaptive_rebin(rows, 1., 2, 4.)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(float(merged['j'][0]), 0., places=14)
        self.assertAlmostEqual(
            float(dispersion.radial_dispersion_squared(merged)[0]), 10., places=13)

    def test_radial_moment_convention_follows_solver_metadata(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps({'bp_collision_operator': 'spherical_dsmc'}))
            self.assertEqual(dispersion.resolve_radial_moment_convention(
                [root], 'metadata.json', 'auto'), 'central')
            metadata.write_text(json.dumps({'bp_collision_operator': 'finite_angle'}))
            self.assertEqual(dispersion.resolve_radial_moment_convention(
                [root], 'metadata.json', 'auto'), 'raw')
            self.assertEqual(dispersion.resolve_radial_moment_convention(
                [root], 'metadata.json', 'central'), 'central')

    def test_arithmetic_log_centers_recover_mesh_volume(self):
        faces = np.geomspace(0.1, 100, 33)
        centers = (faces[:-1] + faces[1:]) / 2
        np.testing.assert_allclose(virial.shell_volumes(centers),
            4 * np.pi / 3 * np.diff(faces ** 3), rtol=5e-14)

    def test_uniform_density_mass_radius_uses_shell_faces(self):
        faces = np.geomspace(0.1, 10, 65)
        centers = (faces[:-1] + faces[1:]) / 2
        radius, mass = scaled.size_from_mass_fraction(centers, np.ones(64), 0.1)
        self.assertAlmostEqual(radius, (faces[0]**3 + 0.1 * (faces[-1]**3 - faces[0]**3))**(1/3), places=12)
        self.assertAlmostEqual(mass, 4 * np.pi / 3 * (10**3 - 0.1**3), places=10)

    def test_uniform_shell_energy_and_full_velocity_second_moment(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            faces = np.geomspace(0.1, 10, 257)
            r = (faces[:-1] + faces[1:]) / 2
            rows = np.column_stack((np.zeros(256), r, np.ones(256),
                                    np.full(256, 2.), np.full(256, 3.), np.full(256, 5.)))
            path = root/'profile_00000.csv'
            np.savetxt(path, rows, delimiter=',',
                       header='time,radius,density,radial_mean,radial_variance,tangential_second',comments='')
            time, ratio, kinetic, potential, mass = virial.virial_quantities_for_profile(path)
            expected_mass = 4*np.pi/3*(10**3-0.1**3)
            expected_potential = -16*np.pi**2/3*((10**5-0.1**5)/5 - 0.1**3*(10**2-0.1**2)/2)
            self.assertAlmostEqual(mass/expected_mass, 1., places=13)
            self.assertAlmostEqual(kinetic/mass, 6., places=13)
            self.assertLess(abs(potential/expected_potential-1), 3e-4)
            self.assertAlmostEqual(ratio, 2*kinetic/abs(potential), places=14)

    def test_powerlaw_slope_sign(self):
        r = np.geomspace(0.1, 100, 128)
        _, slope = density.local_poly_log_slope(r, r ** -1.5)
        np.testing.assert_allclose(slope[np.isfinite(slope)], -1.5, atol=2e-13)
        self.assertAlmostEqual(density.fit_power_law(r, r**-1.5, -0.75, 0.25)[0], -1.5, places=12)

    def test_outer_fit_requires_sampling(self):
        r = np.geomspace(0.1, 100, 128)
        self.assertIsNone(density.detect_shoulder_fit(r, r**-1.5, np.ones(128)))

    def test_complete_endpoints_survive_partial_latest_profile(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'metadata.json').write_text(json.dumps({'radial_cells': 8}))
            r = np.geomspace(.1, 10, 8)
            for i, count in enumerate((8, 8, 8, 4)):
                np.savetxt(root/f'profile_{i:05d}.csv',
                    np.column_stack((np.full(count, i), r[:count], r[:count]**-1.5, np.full(count,4096))),
                    delimiter=',',header='time,radius,density,neff',comments='')
            self.assertEqual([t for t,_ in density.load_profiles(root,stride=10)],[0,2])


if __name__ == '__main__':
    unittest.main()
