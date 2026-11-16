#!/usr/bin/env python3
# Summarize SIDM profile evolution and the elastic-collision invariant ledger.
# Input is one output directory plus reference central density, relaxation time and core radius.
# Output JSON reports core trends and collision-only mass/momentum/kinetic residuals.

"""Summarize SIDM benchmark evolution and operator-invariant ledgers.

The profile summary tracks central density and the radial second velocity moment,
while the operator ledger checks that elastic SIDM collisions conserve
represented mass, radial momentum, and kinetic energy to numerical accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one solver diagnostic table without changing its stored precision."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_max(values: list[float]) -> float:
    """Largest finite absolute residual, or zero when no value is available."""
    array = np.asarray(values, dtype=float)
    array = np.abs(array[np.isfinite(array)])
    return float(array.max()) if array.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rho-c", type=float, required=True)
    parser.add_argument("--trc", type=float, required=True)
    parser.add_argument("--core-radius", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    history = read_csv(args.run_dir / "history.csv")
    profiles = read_csv(args.run_dir / "profiles.csv")
    ledger = read_csv(args.run_dir / "operator_mass_budget.csv")
    if not history or not profiles:
        raise SystemExit("history.csv and profiles.csv must be non-empty")

    # Compare the earliest and latest complete radial states.  Here
    # p/rho=<v_r^2>. For these equilibrium benchmarks the mean radial
    # velocity is negligible, so this moment also approximates the variance.
    times = sorted({float(row["time"]) for row in profiles})
    first = [row for row in profiles if float(row["time"]) == times[0]]
    final = [row for row in profiles if float(row["time"]) == times[-1]]
    r = np.asarray([float(row["r"]) for row in first])
    rho0 = np.asarray([float(row["rho"]) for row in first])
    rho1 = np.asarray([float(row["rho"]) for row in final])
    p0 = np.asarray([float(row["p"]) for row in first])
    p1 = np.asarray([float(row["p"]) for row in final])
    # These profile ratios use the radial second moment p/rho; their variance interpretation requires negligible radial mean.
    sigma20 = np.divide(p0, rho0, out=np.zeros_like(p0), where=rho0 > 0.0)
    sigma21 = np.divide(p1, rho1, out=np.zeros_like(p1), where=rho1 > 0.0)

    # A cell-center r^2 weight approximates the spherical volume measure.
    # This diagnostic is a resolved-core trend indicator, not an exact
    # finite-volume integral (the profile table does not carry cell edges).
    core = (r <= args.core_radius) & (rho0 > 0.0)
    if not np.any(core):
        core[np.argmin(r)] = True
    # Use the stated cell-center r^2 core trend weighting because this table supplies no cell-edge volumes.
    volume_weight = r[core] ** 2
    if not np.any(volume_weight > 0.0):
        volume_weight = np.ones_like(volume_weight)

    h0 = history[0]
    h1 = history[-1]
    # Restrict invariant extrema to collision rows.  Advection and field
    # updates have separate budgets and should not be attributed to SIDM.
    collision = [row for row in ledger if row["operator"].startswith("collision")]
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "initial_time": float(h0["time"]),
        "final_time": float(h1["time"]),
        "final_time_over_trc": float(h1["time"]) / args.trc,
        "steps": int(h1["step"]),
        "initial_mass": float(h0["matter_mass"]),
        "final_mass": float(h1["matter_mass"]),
        "relative_mass_change": float(h1["matter_mass"]) / float(h0["matter_mass"]) - 1.0,
        "max_rho_over_input_rho_c_initial": float(h0["max_rho"]) / args.rho_c,
        "max_rho_over_input_rho_c_final": float(h1["max_rho"]) / args.rho_c,
        "resolved_core_rho_ratio": float(
            np.average(rho1[core], weights=volume_weight)
            / np.average(rho0[core], weights=volume_weight)
        ),
        "resolved_core_sigma2_ratio": float(
            np.average(sigma21[core], weights=volume_weight)
            / np.average(sigma20[core], weights=volume_weight)
        ),
        "collision_ledger_rows": len(collision),
        "collision_max_abs_relative_mass_delta": finite_max(
            [float(row["relative_mass_delta"]) for row in collision]
        ),
    }
    # Report additional momentum and kinetic invariants only when the ledger schema carries them.
    if collision and "radial_momentum_delta" in collision[0]:
        summary.update(
            {
                "collision_max_abs_radial_momentum_delta": finite_max(
                    [float(row["radial_momentum_delta"]) for row in collision]
                ),
                "collision_max_abs_relative_kinetic_energy_delta": finite_max(
                    [float(row["relative_kinetic_energy_delta"]) for row in collision]
                ),
            }
        )

    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
