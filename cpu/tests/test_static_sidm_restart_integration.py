#!/usr/bin/env python3
"""Compare actual static SIDM driver continuation with uninterrupted evolution.

The split occurs at a natural collision endpoint while its gravitational
field is still frozen. The test compares complete final checkpoint bytes,
then verifies that changed physical controls fail before touching output.
"""
import argparse
import csv
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def write_initial(path):
    count = 1536
    header = b"VENUSWVPIC1\0".ljust(16, b"\0")
    header += struct.pack("<IIQdqQQQQdddd", 6, 0x01020304, count,
                          0.0, 0, 1234, 0, 5678, 0,
                          0.0, 0.0, float("inf"), float("inf"))
    with path.open("wb") as stream:
        stream.write(header)
        for i in range(count):
            radius = 0.6 + 1.2 * (i + 0.5) / count
            velocity = ((i * 7) % 19 - 9) * 0.15
            eta = 0.01 + 0.003 * (i % 23)
            stream.write(struct.pack("<6dQQB2d", 0.001, radius, velocity,
                                     eta, 0.0, 0.0, i, 0, 0, 0.0, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    with tempfile.TemporaryDirectory(prefix="venus-static-restart-") as temp:
        root = Path(temp)
        initial = root / "initial.bin"
        write_initial(initial)
        common = ["--maximum-timestep", "0.001", "--dynamical-eta", "0.01",
                  "--potential-event-eta", "0.1", "--collision-event-eta", "0.1",
                  "--radial-cells", "16", "--radial-minimum", "0.1",
                  "--radial-maximum", "4", "--radial-velocity-cells", "24",
                  "--eta-cells", "32", "--radial-velocity-maximum", "6",
                  "--angular-momentum-maximum", "8", "--threads", "2",
                  "--sidm-sigma0", "350", "--collision-target-shell-markers", "256",
                  "--collision-minimum-shell-markers", "16",
                  "--collision-cfl", "0.2", "--collision-maximum-substeps", "65536",
                  "--collision-adaptive-shell-timestep",
                  "--collision-adaptive-maximum-interval", "0.001",
                  "--allow-nonlocal-collision-shell", "--output-every", "1",
                  "--phase-space-projection-at-output-only",
                  "--skip-final-phase-space-projection"]

        def run(source, output, end, extra=()):
            result = subprocess.run(
                [binary, "--pic-restart", str(source), "--output-directory", str(output),
                 "--end-time", str(end)] + common + list(extra),
                capture_output=True, text=True)
            return result

        continuous, first, resumed = (root / name for name in ["continuous", "first", "resumed"])
        for source, output, end in [(initial, continuous, 0.013),
                                    (initial, first, 0.001),
                                    (first / "restart_state.bin", resumed, 0.013)]:
            result = run(source, output, end)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        a = (continuous / "restart_state.bin").read_bytes()
        b = (resumed / "restart_state.bin").read_bytes()
        assert a == b, "Static SIDM split continuation differs from uninterrupted evolution"
        assert struct.unpack_from("<I", a, 16)[0] == 7
        bad_output = root / "bad-controls"
        result = run(first / "restart_state.bin", bad_output, 0.013,
                     ["--sidm-sigma0", "351"])
        assert result.returncode != 0
        assert "controls differ" in result.stderr
        assert not bad_output.exists(), "Failed restart changed its output directory"
        result = run(first / "restart_state.bin", bad_output, 0.013,
                     ["--sidm-sigma0", "351", "--refresh-restart-field"])
        assert result.returncode != 0
        assert "saved physical gravity and collision laws" in result.stderr
        assert not bad_output.exists()
        refreshed = root / "refreshed-mesh"
        result = run(first / "restart_state.bin", refreshed, 0.002,
                     ["--radial-cells", "32", "--refresh-restart-field"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "explicit_field_refresh=1" in (refreshed / "restart_import.txt").read_text()

        def snapshot(directory):
            return {p.relative_to(directory): p.read_bytes()
                    for p in directory.rglob("*") if p.is_file()}

        # A normal continuation in the same directory preserves all earlier
        # products and reaches the same final state as a separate directory.
        same_directory = root / "same-directory"
        shutil.copytree(first, same_directory)
        prior_history = (same_directory / "history.csv").read_bytes()
        prior_profiles = {p.name: p.read_bytes()
                          for p in same_directory.glob("profile_*.csv")}
        result = run(same_directory / "restart_state.bin", same_directory, 0.013)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (same_directory / "restart_state.bin").read_bytes() == a
        assert (same_directory / "history.csv").read_bytes().startswith(prior_history)
        assert all((same_directory / name).read_bytes() == raw
                   for name, raw in prior_profiles.items())

        # Each independently written product must precede the checkpoint.
        # Use a later profile row to ensure the entire profile is inspected.
        targets = ["history.csv", "operator_mass_budget.csv",
                   next(first.glob("profile_*.csv")).name]
        for index, target in enumerate(targets):
            stale = root / ("stale-" + str(index))
            shutil.copytree(first, stale)
            path = stale / target
            rows = list(csv.reader(path.read_text().splitlines()))
            if len(rows) == 1:
                rows.append(["0"] * len(rows[0]))
            column = rows[0].index("time")
            row = -1 if target.startswith("profile_") else 1
            rows[row][column] = "0.002"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            before = snapshot(stale)
            result = run(first / "restart_state.bin", stale, 0.013)
            assert result.returncode != 0, target
            assert "later than restart time" in result.stderr, result.stderr
            assert snapshot(stale) == before, "Rejected restart changed " + target

        malformed = root / "nonfinite-time"
        shutil.copytree(first, malformed)
        path = malformed / "history.csv"
        rows = list(csv.reader(path.read_text().splitlines()))
        rows[1][rows[0].index("time")] = "nan"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        before = snapshot(malformed)
        result = run(first / "restart_state.bin", malformed, 0.013)
        assert result.returncode != 0
        assert "Invalid diagnostic time" in result.stderr
        assert snapshot(malformed) == before
    print("Static SIDM driver restart is byte-identical after 12 continued collision intervals")
    print("Same-directory continuation preserves prior output and rejects stale diagnostics")


if __name__ == "__main__":
    main()
