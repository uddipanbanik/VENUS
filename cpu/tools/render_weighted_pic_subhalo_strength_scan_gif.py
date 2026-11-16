#!/usr/bin/env python3
# Overlay the collisionless control and specified subhalo-collision strengths in density and logarithmic slope.
# Exact mode compares common saved epochs. Held-state mode uses the latest preceding actual collisional state on the control timeline.
# The sidecars record frame clocks and, for held states, each curve's final physical epoch.

"""Compare one collisionless halo with four collisional subhalo strengths.

Every curve is read from the weighted-PIC radial-profile schema.  Exact mode
requires identical physical output times.  Frozen mode uses the collisionless
timeline and holds each collisional sequence at its latest complete profile;
it never extrapolates beyond a saved physical state.
"""

from __future__ import annotations

import argparse
import bisect
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


# ---- Published strength/color convention ----------------------------------
STRENGTHS = (0.1, 0.3, 1.0, 3.0)
# Smaller collision strength is deliberately the deeper red, as specified for
# the comparison figures.  Keeping this mapping explicit prevents
# a colormap reversal from silently changing the physical legend.
RED_LEVEL_BY_STRENGTH = dict(zip(STRENGTHS, (0.88, 0.74, 0.61, 0.48)))


def number_tag(value: float) -> str:
    """Encode a numeric collision strength in a filesystem-safe name."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


# ---- Exact and frozen physical-time alignment ------------------------------
def exact_profile_sets(
    collisionless: list[tuple[float, np.ndarray]],
    collisional_sets: list[list[tuple[float, np.ndarray]]],
    strengths: list[float],
) -> list[tuple[float, np.ndarray, list[np.ndarray]]]:
    """Require all strengths to describe the same physical output times.

    This is the publication mode: every blue/red comparison has undergone the
    same amount of collisionless transport and elapsed physical time.
    """
    scale = max(
        1.0,
        abs(collisionless[-1][0]),
        *(abs(profiles[-1][0]) for profiles in collisional_sets),
    )
    # Exact comparisons require both equal profile counts and equal measured clocks within the stated tolerance.
    tolerance = 1.0e-10 * scale
    expected_count = len(collisionless)
    for strength, profiles in zip(strengths, collisional_sets):
        if len(profiles) != expected_count:
            raise RuntimeError(
                f"Strength {strength:g} has {len(profiles)} profiles; "
                f"collisionless has {expected_count}"
            )

    paired: list[tuple[float, np.ndarray, list[np.ndarray]]] = []
    for index, (time_blue, rows_blue) in enumerate(collisionless):
        red_rows: list[np.ndarray] = []
        for strength, profiles in zip(strengths, collisional_sets):
            time_red, rows = profiles[index]
            if abs(time_blue - time_red) > tolerance:
                raise RuntimeError(
                    f"Profile {index} time mismatch for strength "
                    f"{strength:g}: {time_blue} versus {time_red}"
                )
            red_rows.append(rows)
        paired.append((time_blue, rows_blue, red_rows))
    return paired


def frozen_profile_sets(
    collisionless: list[tuple[float, np.ndarray]],
    collisional_sets: list[list[tuple[float, np.ndarray]]],
    strengths: list[float],
) -> list[tuple[float, np.ndarray, list[np.ndarray]]]:
    """Use the full control timeline, freezing each partial collisional curve.

    This mode never extrapolates a run.  Once a collisional sequence has no
    newer complete profile, its last measured state is held visibly fixed
    while the collisionless control supplies later frame times.
    """
    scale = max(
        1.0,
        abs(collisionless[-1][0]),
        *(abs(profiles[-1][0]) for profiles in collisional_sets),
    )
    tolerance = 1.0e-10 * scale
    timeline = collisionless
    if not timeline:
        raise RuntimeError("Collisionless control has no profiles")

    collisional_times: list[list[float]] = []
    for strength, profiles in zip(strengths, collisional_sets):
        times = [time for time, _ in profiles]
        if times[0] > timeline[0][0] + tolerance:
            raise RuntimeError(
                f"Strength {strength:g} begins at t={times[0]:g}, "
                f"after the control begins at t={timeline[0][0]:g}"
            )
        if any(right <= left for left, right in zip(times, times[1:])):
            raise RuntimeError(
                f"Strength {strength:g} profile times are not strictly increasing"
            )
        collisional_times.append(times)

    paired: list[tuple[float, np.ndarray, list[np.ndarray]]] = []
    for time_blue, rows_blue in timeline:
        red_rows: list[np.ndarray] = []
        for strength, profiles, times in zip(
            strengths, collisional_sets, collisional_times
        ):
            # Held-state playback chooses the latest source at or before the control epoch, preserving a real previously saved state.
            index = bisect.bisect_right(times, time_blue + tolerance) - 1
            if index < 0:
                raise RuntimeError(
                    f"Strength {strength:g} has no profile at or before t={time_blue:g}"
                )
            red_rows.append(profiles[index][1])
        paired.append((time_blue, rows_blue, red_rows))
    return paired


def masked_density(
    rows: np.ndarray,
    r_min: float,
    r_max: float,
    floor_rel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return resolved positive density samples within the plotted radius."""
    mask = display_mask(rows, r_min, r_max, floor_rel)
    return (
        np.asarray(rows["radius"][mask], dtype=float),
        np.asarray(rows["density"][mask], dtype=float),
    )


# ---- Density/slope frame construction and atomic output -------------------
def render(arguments: argparse.Namespace) -> Path:
    """Render the collision-strength response at one initial cusp slope.

    Solid blue is the collisionless Vlasov--Poisson control.  Dashed red
    curves include scattering and capture; the explicitly fixed red shades
    preserve the requested strength ordering in every regenerated GIF.
    """
    collisionless = load_profiles(arguments.collisionless)
    selected = [
        (0.1, arguments.collisional_0p1),
        (0.3, arguments.collisional_0p3),
        (1.0, arguments.collisional_1),
        (3.0, arguments.collisional_3),
    ]
    selected = [(strength, path) for strength, path in selected if path is not None]
    strengths = [strength for strength, _ in selected]
    collisional_sets = [load_profiles(path) for _, path in selected]
    # Apply a common physical upper-time bound before pairing or holding the sequences.
    if arguments.maximum_time is not None:
        tolerance = 1.0e-10 * max(1.0, abs(arguments.maximum_time))
        collisionless = [
            profile
            for profile in collisionless
            if profile[0] <= arguments.maximum_time + tolerance
        ]
        collisional_sets = [
            [
                profile
                for profile in profiles
                if profile[0] <= arguments.maximum_time + tolerance
            ]
            for profiles in collisional_sets
        ]
        if not collisionless or any(not profiles for profiles in collisional_sets):
            raise RuntimeError(
                "No complete profiles exist at or below the requested maximum time"
            )
    # Choose the explicit time-alignment policy before calculating shared density limits.
    paired = (
        frozen_profile_sets(collisionless, collisional_sets, strengths)
        if arguments.freeze_after_last
        else exact_profile_sets(collisionless, collisional_sets, strengths)
    )
    all_profiles = [(time, rows) for time, rows, _ in paired] + [
        profile for profiles in collisional_sets for profile in profiles
    ]
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
    # The fixed strength-to-shade mapping makes the collision parameter identifiable in every frame.
    red_colors = [
        plt.get_cmap("Reds")(RED_LEVEL_BY_STRENGTH[strength])
        for strength in strengths
    ]

    time_min, time_max = paired[0][0], paired[-1][0]
    tag = f"_{arguments.output_tag}" if arguments.output_tag else ""
    output = arguments.output or arguments.output_directory / (
        f"wpic_gamma{number_tag(arguments.gamma)}_"
        "subhalo_strength_scan_logrho_slope"
        f"{tag}_t{time_min:g}_to_t{time_max:g}.gif"
    )
    writer = AtomicGif(output, arguments.fps)
    for time, blue_rows, red_rows_set in paired:
        blue_radius, blue_density = masked_density(
            blue_rows,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
        )
        blue_slope_radius, blue_slope = local_poly_log_slope(
            blue_radius, blue_density
        )
        blue_slope_mask = np.isfinite(blue_slope)

        red_density_sets: list[tuple[np.ndarray, np.ndarray]] = []
        red_slope_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # Compute the derivative for each strength from its actual density with the same local fit and mask.
        for rows in red_rows_set:
            radius, density = masked_density(
                rows,
                arguments.r_min,
                arguments.r_max,
                arguments.density_floor_rel,
            )
            slope_radius, slope = local_poly_log_slope(radius, density)
            red_density_sets.append((radius, density))
            red_slope_sets.append(
                (slope_radius, slope, np.isfinite(slope))
            )

        figure, (rho_axis, slope_axis) = plt.subplots(
            1, 2, figsize=(13.3, 5.7), dpi=140
        )
        figure.subplots_adjust(
            left=0.12,
            right=0.985,
            bottom=0.16,
            # Reserve an explicit header band.  The title occupies its top,
            # the one-row legend occupies the middle, and a clear gap remains
            # above the axes so labels never collide with plotted content.
            top=0.80,
            wspace=0.27,
        )
        initial_line = rho_axis.plot(
            np.log10(initial_radius),
            np.log10(initial_density),
            color="k",
            ls=":",
            lw=2.2,
            zorder=5,
            label="Initial",
        )[0]
        blue_line = rho_axis.plot(
            np.log10(blue_radius),
            np.log10(blue_density),
            color="b",
            ls="-",
            lw=2.6,
            zorder=2,
            label="Collisionless",
        )[0]
        red_lines = []
        for strength, color, (radius, density) in zip(
            strengths, red_colors, red_density_sets
        ):
            red_lines.append(
                rho_axis.plot(
                    np.log10(radius),
                    np.log10(density),
                    color=color,
                    ls="--",
                    lw=2.4,
                    zorder=3,
                    label=rf"Strength ${strength:g}$",
                )[0]
            )
        rho_axis.set_xlim(x_min, x_max)
        rho_axis.set_ylim(rho_y_min, rho_y_max)
        rho_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        rho_axis.set_ylabel(r"$\log_{10}\rho$", fontsize=24, labelpad=11)
        style_axes(rho_axis, 19)

        slope_axis.plot(
            np.log10(initial_slope_radius[initial_slope_mask]),
            initial_slope[initial_slope_mask],
            color="k",
            ls=":",
            lw=2.2,
            zorder=5,
        )
        slope_axis.plot(
            np.log10(blue_slope_radius[blue_slope_mask]),
            blue_slope[blue_slope_mask],
            color="b",
            ls="-",
            lw=2.6,
            zorder=2,
        )
        for color, (slope_radius, slope, slope_mask) in zip(
            red_colors, red_slope_sets
        ):
            slope_axis.plot(
                np.log10(slope_radius[slope_mask]),
                slope[slope_mask],
                color=color,
                ls="--",
                lw=2.4,
                zorder=3,
            )
        for guide in (-1.0, -1.5, -3.0):
            slope_axis.axhline(
                guide, color="0.55", ls="--", lw=1.7, zorder=1
            )
            slope_axis.text(
                0.08,
                guide,
                rf"${guide:g}$",
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
                zorder=6,
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

        figure.suptitle(
            rf"$\gamma_i={arguments.gamma:g},\qquad t={time:g}$",
            fontsize=24,
            y=0.992,
        )
        legend_handles = [initial_line, blue_line, *red_lines]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            # All six entries fit across the 13.3-inch canvas.  A single row
            # prevents a two-row legend from descending into the axes or
            # rising into the gamma/time title.
            ncol=len(legend_handles),
            fontsize=13,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="0.75",
            borderpad=0.45,
            handlelength=2.5,
            columnspacing=1.25,
        )
        writer.append(figure_to_rgb(figure))
        plt.close(figure)
    writer.close()

    times_path = output.with_name(f"{output.stem}_frame_times.txt")
    times_path.write_text(
        "\n".join(
            f"{index},{time:g}" for index, (time, _, _) in enumerate(paired)
        )
        + "\n"
    )
    # Retain the actual last source time for each held curve separately from the control frame timeline.
    if arguments.freeze_after_last:
        freeze_path = output.with_name(f"{output.stem}_curve_last_times.txt")
        freeze_path.write_text(
            "curve,last_profile_time\n"
            f"collisionless,{paired[-1][0]:g}\n"
            + "\n".join(
                f"strength_{strength:g},{profiles[-1][0]:g}"
                for strength, profiles in zip(strengths, collisional_sets)
            )
            + "\n"
        )
    return output


def parse_arguments() -> argparse.Namespace:
    """Parse run directories, cumulative-time cutoff, and visual controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collisionless", type=Path, required=True)
    parser.add_argument("--collisional-0p1", type=Path)
    parser.add_argument("--collisional-0p3", type=Path)
    parser.add_argument("--collisional-1", type=Path)
    parser.add_argument("--collisional-3", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact atomic GIF path; repeated calls overwrite this file",
    )
    parser.add_argument("--output-tag", default="")
    parser.add_argument(
        "--maximum-time",
        type=float,
        help="Use only profiles at or before this cumulative time horizon",
    )
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=900.0)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    parser.add_argument("--slope-y-min", type=float, default=-6.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    parser.add_argument(
        "--freeze-after-last",
        action="store_true",
        help=(
            "Render the full collisionless timeline, holding each collisional "
            "curve at its last saved profile"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(render(parse_arguments()))
