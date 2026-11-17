#!/usr/bin/env python3
"""Physics-clock tests for collision-age-aligned WPIC animations."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
MODULE_PATH = TOOLS / "render_weighted_pic_subhalo_collision_age_gif.py"
SPEC = importlib.util.spec_from_file_location("collision_age_renderer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load renderer from {MODULE_PATH}")
RENDERER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDERER
SPEC.loader.exec_module(RENDERER)


class CollisionClockTests(unittest.TestCase):
    """Check the numerical meaning of integral dt/t_coll(t)."""

    @staticmethod
    def write_history(directory: Path, rows: list[tuple[float, float]]) -> None:
        lines = ["time,min_collision_time"]
        lines.extend(f"{time},{collision_time}" for time, collision_time in rows)
        (directory / "collision_history.csv").write_text("\n".join(lines) + "\n")

    def test_constant_collision_time_counts_elapsed_collision_times(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # The zero at launch is the solver's unevaluated placeholder.
            self.write_history(directory, [(0.0, 0.0), (10.0, 5.0), (20.0, 5.0)])
            clock = RENDERER.load_collision_clock(directory, 20.0)

        np.testing.assert_allclose(clock.time, [0.0, 10.0, 20.0])
        np.testing.assert_allclose(clock.cumulative_age, [0.0, 2.0, 4.0])
        np.testing.assert_allclose(clock.instantaneous_ratio, [0.0, 2.0, 4.0])

    def test_evolving_rate_uses_trapezoidal_hazard_integral(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_history(directory, [(10.0, 10.0), (20.0, 5.0)])
            clock = RENDERER.load_collision_clock(directory, 20.0)

        # Integral from 0..10 is 10/10=1.  From 10..20 the trapezoidal
        # average rate is (1/10+1/5)/2, contributing 1.5 more.
        np.testing.assert_allclose(clock.cumulative_age, [0.0, 1.0, 2.5])

    def test_instantaneous_ratio_can_reverse_while_collision_age_cannot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_history(directory, [(10.0, 1.0), (20.0, 100.0)])
            clock = RENDERER.load_collision_clock(directory, 20.0)

        self.assertLess(clock.instantaneous_ratio[-1], clock.instantaneous_ratio[-2])
        self.assertTrue(np.all(np.diff(clock.cumulative_age) > 0.0))


if __name__ == "__main__":
    unittest.main()
