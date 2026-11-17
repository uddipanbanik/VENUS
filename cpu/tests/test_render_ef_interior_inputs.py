"""A live EF movie consumes complete native accepted profiles."""
from pathlib import Path
import csv
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from render_ef_interior_gif import load_complete_profile_selection


class EFInteriorInputTest(unittest.TestCase):
    def write(self, path, missing_cell=False, duplicate_cell=False):
        fields = ['step', 'advanced_time', 'r', 'a', 'compactness', 'rho_infall',
                  'excision_mask', 'metric_sample_kind', 'metric_cell']
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for step in range(3):
                for cell in range(4):
                    # The live final block stops after two valid radial rows.
                    if step == 2 and cell == 2:
                        return
                    if step == 1 and cell == 2 and missing_cell:
                        continue
                    identifier = 1 if step == 1 and cell == 2 and duplicate_cell else cell
                    writer.writerow([step, step / 10, cell + .5, .8, .2, 1,
                                     0, 'cell_center', identifier])
                writer.writerow([step, step / 10, 4, .8, .2, 0, 0, 'outer_face', 4])

    def test_final_partial_block_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profiles.csv'
            self.write(path)
            selected, lower, upper = load_complete_profile_selection(path, 20)
            self.assertEqual(list(selected.step.unique()), [0, 1])
            self.assertEqual(len(selected), 8)
            self.assertEqual((lower, upper), (0, .1))

    def test_outer_face_requires_every_unique_center(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profiles.csv'
            for kwargs in ({'missing_cell': True}, {'duplicate_cell': True}):
                with self.subTest(kwargs=kwargs):
                    self.write(path, **kwargs)
                    with self.assertRaisesRegex(RuntimeError, 'radial block'):
                        load_complete_profile_selection(path, 20)


if __name__ == '__main__':
    unittest.main()
