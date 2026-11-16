#!/usr/bin/env python3
# Compare a collisionless WPIC control with subhalo scattering/capture at common saved physical times.
# Input CSVs supply deposited radial density and sampling information in the same code units.
# Output density/slope frames retain the initial reference and use one shared set of radial and derivative settings.

"""Overlay a collisional subhalo WPIC run on its collisionless control.

The two profile sequences must have the same physical output times.  The
upper panel compares density, while the lower panel compares the local
logarithmic density slope computed with the same smoothing rule for both
runs.  The GIF is published atomically only after every frame is complete.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-matplotlib-cache")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_weighted_pic_pure_gamma_gifs import (
    AtomicGif,
    density_limits,
    display_mask,
    figure_to_rgb,
    load_profiles,
    local_poly_log_slope,
    style_axes,
)


# ---- Input-time matching and common profile diagnostics -------------------
def exact_profile_pairs(
    collisionless: list[tuple[float, np.ndarray]],
    collisional: list[tuple[float, np.ndarray]],
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Pair control and collisional profiles at identical physical times.

    No interpolation is used: a displayed difference must compare states with
    the same elapsed Vlasov evolution, not neighboring output dumps.
    """
    scale = max(
        1.0,
        abs(collisionless[-1][0]),
        abs(collisional[-1][0]),
    )
    # Allow the specified small clock-comparison tolerance, then require one counterpart for every saved profile.
    tolerance = 1.0e-10 * scale
    if len(collisionless) != len(collisional):
        raise RuntimeError(
            "Collisionless and collisional profile counts differ: "
            f"{len(collisionless)} != {len(collisional)}"
        )
    paired: list[tuple[float, np.ndarray, np.ndarray]] = []
    for (time_blue, rows_blue), (time_red, rows_red) in zip(
        collisionless, collisional
    ):
        if abs(time_blue - time_red) > tolerance:
            raise RuntimeError(
                f"Profile times differ: {time_blue} versus {time_red}"
            )
        paired.append((time_blue, rows_blue, rows_red))
    return paired


def masked_density(
    rows: np.ndarray,
    r_min: float,
    r_max: float,
    floor_rel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive, resolved ``(r, rho)`` samples in the display window."""
    mask = display_mask(rows, r_min, r_max, floor_rel)
    return (
        np.asarray(rows["radius"][mask], dtype=float),
        np.asarray(rows["density"][mask], dtype=float),
    )


def number_tag(value: float) -> str:
    """Encode a floating-point physics parameter in a filesystem-safe name."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


# ---- Matched density/slope frame construction -----------------------------
def render(arguments: argparse.Namespace) -> Path:
    """Build a matched collisionless-versus-collisional density/slope GIF.

    Solid blue follows Vlasov--Poisson evolution alone.  Dashed red includes
    subhalo scattering and capture.  The dotted initial profile exposes both
    secular numerical drift in the control and genuine collision-driven
    evolution in the collisional run.
    """
    collisionless = load_profiles(arguments.collisionless)
    collisional = load_profiles(arguments.collisional)
    # Establish actual common epochs before deriving sequence-wide display limits.
    paired = exact_profile_pairs(collisionless, collisional)
    all_profiles = collisionless + collisional
    rho_y_min, rho_y_max = density_limits(
        all_profiles,
        arguments.r_min,
        arguments.r_max,
        arguments.density_floor_rel,
    )
    initial_radius, initial_density = masked_density(
        collisionless[0][1],
        arguments.r_min,
        arguments.r_max,
        arguments.density_floor_rel,
    )
    initial_slope_radius, initial_slope = local_poly_log_slope(
        initial_radius, initial_density
    )
    initial_slope_mask = np.isfinite(initial_slope)
    radius_all = np.asarray(collisionless[0][1]["radius"], dtype=float)
    x_min = math.floor(math.log10(float(np.min(radius_all))))
    x_max = math.ceil(math.log10(float(np.max(radius_all))))

    time_min, time_max = paired[0][0], paired[-1][0]
    tag = f"_{arguments.output_tag}" if arguments.output_tag else ""
    output = arguments.output_directory / (
        f"wpic_gamma{number_tag(arguments.gamma)}_"
        "subhalo_collisional_vs_collisionless_"
        f"logrho_slope{tag}_t{time_min:g}_to_t{time_max:g}.gif"
    )
    writer = AtomicGif(output, arguments.fps)
    for time, blue_rows, red_rows in paired:
        blue_radius, blue_density = masked_density(
            blue_rows,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
        )
        red_radius, red_density = masked_density(
            red_rows,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
        )
        # Differentiate each physical profile on its own radial samples using the same local-fit algorithm.
        blue_slope_radius, blue_slope = local_poly_log_slope(
            blue_radius, blue_density
        )
        red_slope_radius, red_slope = local_poly_log_slope(
            red_radius, red_density
        )
        blue_slope_mask = np.isfinite(blue_slope)
        # A missing derivative remains a gap while its density samples keep their separately defined display selection.
        red_slope_mask = np.isfinite(red_slope)

        figure, (rho_axis, slope_axis) = plt.subplots(
            1, 2, figsize=(13.0, 5.6), dpi=140
        )
        figure.subplots_adjust(
            left=0.10,
            right=0.975,
            bottom=0.17,
            top=0.77,
            wspace=0.33,
        )
        blue_line = rho_axis.plot(
            np.log10(blue_radius),
            np.log10(blue_density),
            color="b",
            ls="-",
            lw=2.5,
            zorder=2,
            label="Collisionless",
        )[0]
        red_line = rho_axis.plot(
            np.log10(red_radius),
            np.log10(red_density),
            color="r",
            ls="--",
            lw=2.5,
            zorder=3,
            label="Scattering + capture",
        )[0]
        initial_line = rho_axis.plot(
            np.log10(initial_radius),
            np.log10(initial_density),
            color="k",
            ls=":",
            lw=2.3,
            zorder=4,
            label="Initial",
        )[0]
        rho_axis.set_xlim(x_min, x_max)
        rho_axis.set_ylim(rho_y_min, rho_y_max)
        rho_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        rho_axis.set_ylabel(r"$\log_{10}\rho$", fontsize=24, labelpad=11)
        style_axes(rho_axis, 19)

        slope_axis.plot(
            np.log10(blue_slope_radius[blue_slope_mask]),
            blue_slope[blue_slope_mask],
            color="b",
            ls="-",
            lw=2.5,
            zorder=2,
        )
        slope_axis.plot(
            np.log10(red_slope_radius[red_slope_mask]),
            red_slope[red_slope_mask],
            color="r",
            ls="--",
            lw=2.5,
            zorder=3,
        )
        slope_axis.plot(
            np.log10(initial_slope_radius[initial_slope_mask]),
            initial_slope[initial_slope_mask],
            color="k",
            ls=":",
            lw=2.3,
            zorder=4,
        )
        slope_axis.axhline(
            -3.0, color="0.55", ls="--", lw=1.7, zorder=1
        )
        slope_axis.text(
            0.08,
            -3.0,
            r"$-3$",
            transform=slope_axis.get_yaxis_transform(),
            color="0.35",
            fontsize=18,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 1.0,
            },
            zorder=5,
        )
        slope_axis.set_xlim(x_min, x_max)
        slope_axis.set_ylim(arguments.slope_y_min, arguments.slope_y_max)
        slope_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        slope_axis.set_ylabel(
            r"$\mathrm{d}\log\rho/\mathrm{d}\log r$",
            fontsize=24,
            labelpad=11,
        )
        style_axes(slope_axis, 19)

        if arguments.collision_strength is None:
            title = rf"$\gamma_i={arguments.gamma:g},\qquad t={time:g}$"
        else:
            title = (
                rf"$\gamma_i={arguments.gamma:g},\qquad "
                rf"\mathrm{{strength}}={arguments.collision_strength:g},"
                rf"\qquad t={time:g}$"
            )
        figure.suptitle(title, fontsize=24, y=0.97)
        figure.legend(
            handles=[initial_line, blue_line, red_line],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.895),
            ncol=3,
            fontsize=16,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="0.75",
            borderpad=0.35,
            handlelength=2.8,
            columnspacing=1.8,
        )
        writer.append(figure_to_rgb(figure))
        plt.close(figure)
    writer.close()

    # Save the physical timeline behind the matched frames beside the streamed movie.
    times_path = output.with_name(f"{output.stem}_frame_times.txt")
    times_path.write_text(
        "\n".join(
            f"{index},{time:g}" for index, (time, _, _) in enumerate(paired)
        )
        + "\n"
    )
    return output


def parse_arguments() -> argparse.Namespace:
    """Parse input runs, physical labels, and non-invasive display cuts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collisionless", type=Path, required=True)
    parser.add_argument("--collisional", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--collision-strength", type=float)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=900.0)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    parser.add_argument("--slope-y-min", type=float, default=-6.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(render(parse_arguments()))
