#!/usr/bin/env python3
"""Exercise custom/static WPIC backgrounds and checkpoint compatibility.

The finite integration compares an uninterrupted run with a resumed run at
identical step boundaries, then checks rejection of changed expansion laws.
Pass --binary to execute it; unittest discovery leaves this CLI fixture idle.
"""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import tempfile


def run_checks(binary: Path) -> None:
    """Require deterministic restart and validated metadata for each background."""
    binary = binary.resolve()
    with tempfile.TemporaryDirectory(prefix="venus-background-") as name:
        root = Path(name)
        common = [str(binary), "--gamma", "1.5", "--radial-cells", "16",
                  "--radial-minimum", "0.1", "--radial-maximum", "10",
                  "--particles-per-radial-cell", "128", "--maximum-timestep", "0.001",
                  "--output-interval", "0.001", "--history-every", "1",
                  "--restart-every", "1", "--threads", "2"]
        initial_physical_energy = {}
        for mode, alpha, factor in (("static", "0.7", "1"),
                                    ("static", "0.7", "0.25"),
                                    ("expanding", "0.5", "1"),
                                    ("expanding", "0.5", "0.25")):
            settings = ["--vp-universe", mode, "--vp-alpha", alpha,
                        "--vp-time-offset", "8", "--initial-velocity-factor", factor]
            full, half, resumed = [root / (mode + factor + tag) for tag in ("full", "half", "resumed")]
            def execute(end, output, extra=(), expected=0):
                result = subprocess.run(common + settings + ["--end-time", end,
                    "--output-directory", str(output)] + list(extra),
                    capture_output=True, text=True, timeout=60)
                assert (result.returncode != 0 if expected is None else result.returncode == expected), result.stdout + result.stderr
                return result
            execute("0.002", full)
            execute("0.001", half)
            restart = ["--restart", str(half / "restart_state.bin")]
            execute("0.002", resumed, restart)
            assert (full / "restart_state.bin").read_bytes() == (resumed / "restart_state.bin").read_bytes()
            meta = json.loads((full / "metadata.json").read_text())
            assert meta["vp_universe"] == mode
            assert meta["vp_alpha"] == (0.0 if mode == "static" else 0.5)
            assert meta["vp_time_offset"] == 8.0
            assert meta["initial_velocity_factor"] == float(factor)
            with (full / "virial_budget.csv").open() as handle:
                moments = [{k: float(v) for k, v in row.items()}
                           for row in csv.DictReader(handle)]
            initial_physical_energy[mode, factor] = moments[0]["K_physical"]
            for row in moments:
                assert abs(row["mobile_mass"] - row["resolved_mass"]
                           - row["hidden_inner_mass"]) < 1e-14
                assert abs(row["K_physical"] - row["K_pec"]
                           - row["H"] * row["C"]
                           - 0.5 * row["H"] ** 2 * row["I"]) < 1e-14
            # Compare independent finite differences of particle moments with
            # the orbit equations, including expansion and hidden particles.
            first, middle, last = moments
            span = last["time"] - first["time"]
            measured_C_dot = (last["C"] - first["C"]) / span
            force_scale = 2 * middle["K_pec"] + abs(middle["W_force"])
            assert abs(measured_C_dot - middle["C_dot_orbit"]) < 1e-4 * force_scale
            measured_I_dot = (last["I"] - first["I"]) / span
            expected_I_dot = 2 * middle["H"] * middle["I"] + 2 * middle["C"]
            assert abs(measured_I_dot - expected_I_dot) < 1e-7 * max(1, middle["I"])
            failed = execute("0.002", root / "bad", restart + ["--vp-time-offset", "9"], None)
            assert "background" in failed.stderr.lower()
            failed = execute("0.002", root / "bad_ic", restart + ["--initial-velocity-factor", "0.5"], None)
            assert "initial-condition contract" in failed.stderr.lower()
        for mode in ("static", "expanding"):
            ratio = (initial_physical_energy[mode, "0.25"]
                     / initial_physical_energy[mode, "1"])
            assert abs(ratio - 0.25 ** 2) < 1e-14
        for extra in (["--vp-universe", "typo"], ["--vp-alpha", "nan"],
                      ["--vp-alpha", "1.1"], ["--vp-time-offset", "0"],
                      ["--initial-velocity-factor", "0"],
                      ["--initial-velocity-factor", "nan"]):
            result = subprocess.run(common + ["--end-time", "0.001",
                "--output-directory", str(root / "invalid")] + extra,
                capture_output=True, text=True, timeout=30)
            assert result.returncode != 0
            assert not (root / "invalid").exists()
    print("WPIC backgrounds, cold initial data, force virial budgets, and exact restarts passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    run_checks(parser.parse_args().binary)
