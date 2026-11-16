#!/usr/bin/env python3
# Compare density and logarithmic slope at two marker-count resolutions and one physical epoch.
# Inputs are named profile CSVs in common radial code units; the shared renderer supplies the radial/display mask and local polynomial derivative.
# The output shows each supplied profile at its own radial samples.

"""Plot a two-resolution weighted-PIC density/slope convergence overlay.

Both profiles must represent the same physical time.  Their deposited density
and locally fitted logarithmic slope are drawn with identical radial/noise
cuts, making marker-number dependence visible without modifying either run.
Each named-column CSV must contain ``time``, ``radius``, and ``density``;
additional deposited moments are preserved but not used.  The saved figure is
a static diagnostic, so the source profiles are neither resampled nor
interpolated.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_weighted_pic_pure_gamma_gifs import (
    display_mask,
    local_poly_log_slope,
    style_axes,
)


def load_profile(path: Path) -> np.ndarray:
    """Read one WPIC profile and require time, radius, and density columns."""
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = {"time", "radius", "density"}
    if not required.issubset(rows.dtype.names or ()):
        raise RuntimeError(f"{path} is missing {sorted(required)}")
    return rows


def main() -> None:
    """Validate a same-time pair, derive slopes, and save the overlay figure."""
    # The lower-PPC (particles-per-cell) curve is the candidate resolution and
    # the higher-PPC curve is the visual reference.  This is a diagnostic, not
    # an assertion that the highest supplied resolution is mathematically exact.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--low-label", required=True)
    parser.add_argument("--high-label", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=900.0)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    arguments = parser.parse_args()
    if (
        not math.isfinite(arguments.r_min)
        or not math.isfinite(arguments.r_max)
        or arguments.r_min <= 0.0
        or arguments.r_max <= arguments.r_min
    ):
        raise ValueError("--r-min and --r-max must define a positive interval")
    if (
        not math.isfinite(arguments.density_floor_rel)
        or arguments.density_floor_rel < 0.0
    ):
        raise ValueError("--density-floor-rel must be finite and non-negative")

    low = load_profile(arguments.low)
    high = load_profile(arguments.high)
    time_low = float(low["time"][0])
    time_high = float(high["time"][0])
    # A resolution comparison requires the two measured source clocks to coincide within the stated floating-point comparison.
    if not np.isclose(time_low, time_high):
        raise RuntimeError(f"Profile times differ: {time_low} versus {time_high}")

    figure, (density_axis, slope_axis) = plt.subplots(1, 2, figsize=(13.0, 5.25), dpi=160)
    figure.subplots_adjust(left=0.10, right=0.975, bottom=0.18, top=0.84, wspace=0.33)
    styles = (
        (low, arguments.low_label, "0.45", "--", 2.0),
        (high, arguments.high_label, "b", "-", 2.5),
    )
    for rows, label, color, linestyle, linewidth in styles:
        # Apply the same radial and relative-density criteria to each input before computing its local logarithmic derivative.
        mask = display_mask(
            rows,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
        )
        radius = np.asarray(rows["radius"][mask], dtype=float)
        density = np.asarray(rows["density"][mask], dtype=float)
        # The fitted derivative is d log rho / d log r, so it has the same value for natural or base-ten logarithms.
        slope_radius, slope = local_poly_log_slope(radius, density)
        density_axis.plot(
            np.log10(radius),
            np.log10(density),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        # Draw only derivative centers with finite fits while retaining the separately selected density samples.
        finite = np.isfinite(slope)
        slope_axis.plot(
            np.log10(slope_radius[finite]),
            slope[finite],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    x_limits = (math.log10(arguments.r_min), math.log10(arguments.r_max))
    for axis in (density_axis, slope_axis):
        axis.set_xlim(*x_limits)
        style_axes(axis, 17)
        axis.legend(frameon=False, fontsize=15, loc="best")
    density_axis.set_xlabel(r"$\log_{10} r$", fontsize=23, labelpad=9)
    density_axis.set_ylabel(r"$\log_{10}\rho$", fontsize=23, labelpad=11)
    slope_axis.set_xlabel(r"$\log_{10} r$", fontsize=23, labelpad=9)
    slope_axis.set_ylabel(r"$\mathrm{d}\log\rho/\mathrm{d}\log r$", fontsize=23, labelpad=11)
    figure.suptitle(rf"$\gamma_i={arguments.gamma:g},\qquad t={time_high:g}$", fontsize=23, y=0.965)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    plt.close(figure)
    print(arguments.output)


if __name__ == "__main__":
    main()
