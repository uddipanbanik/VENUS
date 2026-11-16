#!/usr/bin/env python3
# Animate benchmark-normalized WPIC density and its local logarithmic slope.
# The shared loader conserves shell mass and raw velocity moments when grouping sparsely sampled cells.
# Each frame is one distinct saved epoch, with code time converted to Gyr and the earliest profile shown as reference.

"""Render Frank-initialized WPIC density and logarithmic-slope evolution.

Each ``profile_*.csv`` input uses the weighted-PIC radial-profile schema:
``time``, ``radius``, ``density``, ``radial_mean``, ``radial_variance``, and
``neff``.  :func:`render_frank_wpic_gif.load_profile` conservatively groups
noisy cells and converts the deposited moments to the Frank benchmark units.
This renderer then derives ``d log(rho) / d log(r)`` with a matched-bandwidth
local fit and compares every distinct saved time with the earliest profile.

Profiles may be collected from several checkpoint directories.  Duplicate
times are collapsed, code time is converted to Gyr, and full RGB frames are
streamed to a temporary GIF.  The requested output path is replaced only
after the trailer is written; a failed render removes the temporary file.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "venus-matplotlib-cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
for font_cache in (Path.home() / ".matplotlib").glob("fontlist*.json"):
    target = matplotlib_cache / font_cache.name
    if not target.exists():
        shutil.copy2(font_cache, target)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spherical_profile_rendering import (
    add_slope_reference_lines,
    fig_to_rgb,
    style_axes,
)
from render_density_slope_sigma2_gif import AtomicStreamingGif
from render_frank_venus_sphdsmc_gif import (
    matched_bandwidth_log_slope,
    text_layout_qa,
)
from render_frank_wpic_gif import TIME_GYR, load_profile


def frame(
    curve: tuple[np.ndarray, np.ndarray, np.ndarray],
    initial: tuple[np.ndarray, np.ndarray, np.ndarray],
    time_gyr: float,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw density and derived logarithmic slope for one physical time.

    ``curve`` and ``initial`` are ``(radius, density, sigma_r_squared)``
    tuples returned by :func:`load_profile`; the dispersion component is not
    used in this two-panel view.  Both slopes are evaluated on the live
    profile's radial sampling so their visual bandwidths remain comparable.
    """
    r, rho, _ = curve
    initial_r, initial_rho, _ = initial
    fig, axes_array = plt.subplots(1, 2, figsize=(11.3, 4.35), dpi=135)
    axes = list(axes_array)
    fig.subplots_adjust(left=0.105, right=0.978, bottom=0.17, top=0.84, wspace=0.37)
    fig._ic_annotation = fig.text(
        0.106,
        0.972,
        r"IC: $(\alpha,\beta,\gamma)=(4,4,0.1),\ c=20,\ \Delta_{\rm vir}=200$"
        "\n"
        r"$M_{\rm host}=10^{12}\,M_\odot/h,\ N_{\rm p}=4\times10^6,\ "
        r"\sigma_m=25\ {\rm cm^2\,g^{-1}}$",
        ha="left",
        va="top",
        fontsize=10.8,
        linespacing=1.20,
    )

    minimum_radius = 3.0e-3
    maximum_radius = 1.0e3
    x_limits = (math.log10(minimum_radius), math.log10(maximum_radius))

    # Select positive finite density on the stated physical radial interval independently for initial and current states.
    initial_mask = (
        np.isfinite(initial_rho)
        & (initial_rho > 0.0)
        & (initial_r >= minimum_radius)
        & (initial_r <= maximum_radius)
    )
    live_mask = (
        np.isfinite(rho)
        & (rho > 0.0)
        & (r >= minimum_radius)
        & (r <= maximum_radius)
    )
    axes[0].plot(
        np.log10(initial_r[initial_mask]),
        np.log10(initial_rho[initial_mask]),
        "k:",
        lw=1.8,
        label="Initial",
        zorder=1,
    )
    axes[0].plot(
        np.log10(r[live_mask]),
        np.log10(rho[live_mask]),
        color="b",
        lw=2.0,
        label="WPIC",
        zorder=2,
    )

    for source_r, source_rho, color, line_style in (
        (initial_r, initial_rho, "k", ":"),
        (r, rho, "b", "-"),
    ):
        # Fit both densities on a bandwidth tied to the current radial sampling so their derivatives have comparable smoothing.
        slope_r, slope = matched_bandwidth_log_slope(
            source_r, source_rho, r, window_points=9, poly_order=2
        )
        # Reserve outer neighbors for a centered local derivative instead of plotting an incomplete boundary estimate.
        maximum_slope_radius = r[-5] if r.size >= 5 else r[-1]
        mask = (
            np.isfinite(slope)
            & (slope_r >= minimum_radius)
            & (slope_r <= min(maximum_radius, maximum_slope_radius))
        )
        axes[1].plot(
            np.log10(slope_r[mask]),
            slope[mask],
            color=color,
            ls=line_style,
            lw=2.0,
        )

    axes[0].set(
        xlim=x_limits,
        ylim=(-10.0, 6.0),
        xlabel=r"$\log_{10}(r/r_s)$",
        ylabel=r"$\log_{10}(\rho/\rho_s)$",
    )
    axes[1].set(
        xlim=x_limits,
        ylim=(-6.0, 2.0),
        xlabel=r"$\log_{10}(r/r_s)$",
        ylabel=r"$\mathrm{d}\log\rho/\mathrm{d}\log r$",
    )
    axes[1].axhline(0.0, color="0.6", lw=1.0, ls=":")
    add_slope_reference_lines(axes[1], *x_limits, -6.0, 2.0)

    for axis in axes:
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(16)
        axis.xaxis.labelpad = 7
        axis.yaxis.labelpad = 8
        style_axes(axis)
    axes[0].legend(
        loc="lower left", fontsize=12.5, frameon=False, handlelength=3.0
    )
    fig._time_title = fig.suptitle(
        rf"$t={time_gyr:.3f}\,\mathrm{{Gyr}}$", fontsize=16, x=0.77, y=0.97
    )
    return fig, axes


def main() -> None:
    """Load, time-order, render, and atomically publish the profile sequence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    arguments = parser.parse_args()

    paths = sorted(
        {path for directory in arguments.profile_dir for path in directory.glob("profile_*.csv")}
    )
    if not paths:
        raise RuntimeError("No profile_*.csv files were found.")
    profiles = [load_profile(path) for path in paths]
    # Collapse duplicate continuation clocks before chronological rendering. Existing source profiles supply all frame data.
    profiles = sorted(
        {round(time, 12): (time, curve) for time, curve in profiles}.values()
    )
    initial = profiles[0][1]

    # Stream RGB frames to a temporary GIF, then publish only after successful finalization; errors remove the unfinished file.
    writer = AtomicStreamingGif(arguments.output, arguments.fps)
    try:
        for index, (time_code, curve) in enumerate(profiles):
            figure, axes = frame(curve, initial, time_code * TIME_GYR)
            if index == 0:
                text_layout_qa(figure, axes)
            writer.append(fig_to_rgb(figure))
            plt.close(figure)
        writer.close()
    except Exception:
        writer.abort()
        raise

    print(
        f"wrote {arguments.output} "
        f"({len(profiles)} frames, t={profiles[-1][0] * TIME_GYR:.6f} Gyr)"
    )


if __name__ == "__main__":
    main()
