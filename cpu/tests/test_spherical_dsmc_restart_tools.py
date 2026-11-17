"""Check diagnostic/import readers against every SphDSMC checkpoint layout."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load(name):
    specification = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


AUDIT = load("audit_frank_collision_density")
EXPORT = load("export_venus_restart_to_frank_ic")
SHELLS = load("diagnose_spherical_dsmc_restart_shells")


def checkpoint(version, clock=0.125):
    count, mass = 32, 0.01
    header = struct.pack("<16sIQddqdQqq", b"VENUSSPHDSMC1", version,
                         count, mass, 1.25, 12, -0.7, 17, 0, 0)
    if version >= 6:
        header += struct.pack("<dQ", clock, 1234)
    record = struct.Struct("<dddQd" if version >= 4 else "<dddQ")
    records = []
    for index in range(count):
        fields = (0.5 + 0.05 * index, 0.1 * math.sin(index), 0.2, index)
        records.append(record.pack(*(fields + (mass,) if version >= 4 else fields)))
    remainder = (struct.pack("<Q", count) + struct.pack("<32d", *([0.05] * count))
                 if version in (3, 5, 7) else b"")
    return header + b"".join(records) + remainder


class SphericalDSMCRestartReaders(unittest.TestCase):
    def test_all_layouts_preserve_particle_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "restart.bin"
            reference = None
            for version in (2, 3, 4, 5, 6, 7):
                with self.subTest(version=version):
                    path.write_bytes(checkpoint(version))
                    metadata, particles = AUDIT.read_restart(path)
                    time, exported = EXPORT.read_restart(path)
                    self.assertEqual(metadata["version"], version)
                    self.assertEqual(time, 1.25)
                    current = [(p.radius, p.radial_velocity, p.angular_momentum, p.mass)
                               for p in exported]
                    self.assertEqual(particles, [p[:3] for p in current])
                    if reference is None:
                        reference = current
                    self.assertEqual(current, reference)
                    text = io.StringIO()
                    with patch.object(sys, "argv", ["shells", str(path), "--sigma", "0.1"]):
                        with contextlib.redirect_stdout(text):
                            SHELLS.main()
                    result = json.loads(text.getvalue())
                    self.assertEqual(result["particle_count"], 32)

    def test_invalid_or_truncated_clock_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "restart.bin"
            for version in (6, 7):
                for clock in (0.0, -1.0, math.nan):
                    with self.subTest(version=version, clock=clock):
                        path.write_bytes(checkpoint(version, clock))
                        with self.assertRaises(RuntimeError):
                            AUDIT.read_restart(path)
                        with self.assertRaises(RuntimeError):
                            EXPORT.read_restart(path)
                        with patch.object(sys, "argv", ["shells", str(path), "--sigma", "0.1"]):
                            with self.assertRaises(RuntimeError):
                                SHELLS.main()
                path.write_bytes(checkpoint(version)[:88])
                with self.assertRaises(RuntimeError):
                    AUDIT.read_restart(path)
                with self.assertRaises(RuntimeError):
                    EXPORT.read_restart(path)

    def test_unbounded_collision_clock_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "restart.bin"
            for version in (6, 7):
                path.write_bytes(checkpoint(version, math.inf))
                self.assertEqual(AUDIT.read_restart(path)[0]["count"], 32)
                self.assertEqual(len(EXPORT.read_restart(path)[1]), 32)


if __name__ == "__main__":
    unittest.main()
