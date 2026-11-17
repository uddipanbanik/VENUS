"""Check saved phase-cell completeness and Python boundary declarations."""
from pathlib import Path
import csv
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from ev_solver.output_reader import load_phase_snapshots
from ev_solver.params import EVParams
from ev_solver.solver_setup import prepare_simulation


class PythonReferenceContract(unittest.TestCase):
    def write_frames(self, path, final_pairs):
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['time', 'step', 'r_index', 'w_index', 'r', 'w', 'g'])
            for time, pairs in ((0, [(i, j) for i in range(2) for j in range(3)]),
                                (1, final_pairs)):
                for i, j in pairs:
                    writer.writerow([time, time, i, j, i + 1, j - 1 + time / 2, 10 * i + j])

    def test_partial_final_radial_rows_are_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'phase.csv'
            self.write_frames(path, [(0, j) for j in range(3)])
            times, _, _, frames = load_phase_snapshots(path)
            np.testing.assert_array_equal(times, [0])
            self.assertEqual(frames[0].shape, (2, 3))

    def test_duplicate_pair_cannot_replace_missing_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'phase.csv'
            self.write_frames(path, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 1)])
            times, _, _, _ = load_phase_snapshots(path)
            np.testing.assert_array_equal(times, [0])

    def test_complete_remapped_frame_keeps_its_own_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'phase.csv'
            self.write_frames(path, list(reversed([(i, j) for i in range(2) for j in range(3)])))
            times, _, momentum, frames = load_phase_snapshots(path)
            np.testing.assert_array_equal(times, [0, 1])
            np.testing.assert_array_equal(momentum[1], momentum[0] + .5)
            np.testing.assert_array_equal(frames[1], frames[0])

    def test_binary_ic_rejects_unsupported_boundaries_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initial = root / 'initial.bin'
            with initial.open('wb') as handle:
                handle.write(b'EVICRWL1' + b'\0' * 8)
                handle.write(struct.pack('<iiii', 1, 3, 3, 2))
                for values in ([1, 2, 3], [-1, 0, 1], [0, 1], np.ones(18) * 1e-6):
                    handle.write(np.asarray(values, dtype='<f8').tobytes())
            for name, value in [('inner_boundary', 'origin_parity'),
                                ('outer_boundary', 'reflecting'),
                                ('momentum_boundary', 'reflecting'),
                                ('time_integrator_order', 4),
                                ('time_integrator_substeps', 2)]:
                with self.subTest(name=name):
                    output = root / ('unsupported_' + name)
                    par = EVParams(initial_data='binary_rwl', initial_data_file=str(initial),
                                   output_dir=str(output))
                    setattr(par, name, value)
                    with self.assertRaisesRegex(ValueError, 'Python reference solver'):
                        prepare_simulation(par)
                    self.assertFalse(output.exists())
            # The supported binary path still reconstructs its supplied grid.
            par = EVParams(initial_data='binary_rwl', initial_data_file=str(initial),
                           output_dir=str(root / 'supported'))
            state = prepare_simulation(par)
            self.assertEqual(state.f.shape, (3, 3, 2))


if __name__ == '__main__':
    unittest.main()
