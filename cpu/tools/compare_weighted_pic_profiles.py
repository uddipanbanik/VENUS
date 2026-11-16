#!/usr/bin/env python3
# Compare two named-column WPIC snapshots on their resolved radial support.
# Inputs are lower/higher marker-count profiles in the same physical units.
# Output JSON records each actual epoch and shared density/slope convergence measures.

"""Compare two weighted-PIC profile snapshots with the convergence metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_weighted_pic_ppc_convergence import compare_profiles


def load_profile(path: Path) -> np.ndarray:
    """Load one WPIC profile and require the fields used by convergence tests."""

    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    # Require the time, radial geometry, mass density and sampling fields used by the shared comparison.
    required = {"time", "radius", "density", "neff"}
    if not required.issubset(rows.dtype.names or ()):
        raise RuntimeError(f"{path} is missing {sorted(required)}")
    return rows


def main() -> None:
    """Compare a lower- and higher-marker profile on their resolved support."""

    # The density floor excludes effectively empty cells, while the fitting
    # window fixes the physical scale used for logarithmic density slopes.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--density-floor-relative", type=float, default=1.0e-10)
    parser.add_argument("--slope-window-points", type=int, default=21)
    arguments = parser.parse_args()

    # Use the shared convergence kernel so command-line comparisons and the
    # marker-per-cell comparisons apply identical density and slope definitions.
    low = load_profile(arguments.low)
    high = load_profile(arguments.high)
    # Keep both actual times in the report so comparing the profiles does not imply identical epochs.
    result = {
        "low": str(arguments.low),
        "high": str(arguments.high),
        "time_low": float(low["time"][0]),
        "time_high": float(high["time"][0]),
        "zones": compare_profiles(
            low,
            high,
            arguments.density_floor_relative,
            arguments.slope_window_points,
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
