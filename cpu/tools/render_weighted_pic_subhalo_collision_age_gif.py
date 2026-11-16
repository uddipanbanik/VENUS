#!/usr/bin/env python3
# Compare collisional subhalo profiles at equal integrated minimum-collision-time diagnostic.
# The clock integrates dt/t_coll from the saved rate history. Its global minimum-timescale definition is distinct from a particle's accumulated encounter count.
# This renderer explicitly interpolates density between saved profiles for display and records that policy and each curve's physical time.

"""Render WPIC strength scans at equal accumulated collision age.

For a time-dependent collision rate ``Gamma_coll(t)=1/t_coll(t)``, the
dimensionless number of collision times that have elapsed is

    tau_coll(t) = integral_0^t dt' / t_coll(t').

VENUS writes ``min_collision_time`` in ``collision_history.csv`` from the
evolving distribution function, including the velocity-dependent cross
section.  This renderer integrates that diagnostic for every collisional run
and compares all strengths at the same ``tau_coll``.  The alternative
``t/t_coll(t)`` is recorded for audit purposes, but is not used as the clock:
it is not an accumulated interaction count and need not be monotonic when the
core density or velocity dispersion evolves.

The collisionless control has no collision age.  At each frame it is sampled
at the physical time of the strength-one trajectory at the displayed
``tau_coll``.  A sidecar CSV records the actual physical time selected for
every curve, so the normalized comparison remains fully auditable.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import tempfile
from dataclasses import dataclass
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
from render_weighted_pic_subhalo_strength_scan_gif import (
    RED_LEVEL_BY_STRENGTH,
)


STRENGTHS = (0.1, 0.3, 1.0, 3.0)


@dataclass(frozen=True)
class CollisionClock:
    """Evolving collision time and its two dimensionless time coordinates."""

    time: np.ndarray
    collision_time: np.ndarray
    cumulative_age: np.ndarray
    instantaneous_ratio: np.ndarray

    def cumulative_at(self, physical_time: np.ndarray | float) -> np.ndarray:
        """Interpolate the monotonic accumulated collision age."""
        values = np.asarray(physical_time, dtype=float)
        return np.interp(values, self.time, self.cumulative_age)


@dataclass(frozen=True)
class CollisionCurve:
    """Profiles from one collision strength with their collision-age clock."""

    strength: float
    directory: Path
    profiles: list[tuple[float, np.ndarray]]
    profile_age: np.ndarray
    clock: CollisionClock


def number_tag(value: float) -> str:
    """Encode a floating-point parameter in a filesystem-safe filename."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def truncated_profiles(
    directory: Path, maximum_time: float
) -> list[tuple[float, np.ndarray]]:
    """Load complete profiles no later than the requested physical horizon."""
    tolerance = 1.0e-10 * max(1.0, abs(maximum_time))
    profiles = [
        profile
        for profile in load_profiles(directory)
        if profile[0] <= maximum_time + tolerance
    ]
    if not profiles:
        raise RuntimeError(f"No complete profiles at t <= {maximum_time:g}: {directory}")
    return profiles


def load_collision_clock(directory: Path, maximum_time: float) -> CollisionClock:
    """Integrate the solver's evolving minimum collision-time diagnostic.

    The first production row is at ``t=0`` and carries a zero placeholder
    because no collision-cell diagnostic has yet been evaluated.  The first
    positive solver estimate is therefore extended back over the short first
    output interval.  Subsequent intervals use trapezoidal quadrature of the
    collision rate ``1/t_coll``.
    """
    path = directory / "collision_history.csv"
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = {"time", "min_collision_time"}
    if not required.issubset(rows.dtype.names or ()):
        raise RuntimeError(f"{path} is missing {sorted(required)}")

    time = np.asarray(rows["time"], dtype=float)
    collision_time = np.asarray(rows["min_collision_time"], dtype=float)
    valid = (
        np.isfinite(time)
        & np.isfinite(collision_time)
        & (time >= 0.0)
        & (time <= maximum_time + 1.0e-10 * max(1.0, maximum_time))
        & (collision_time > 0.0)
    )
    time = time[valid]
    collision_time = collision_time[valid]
    if time.size == 0:
        raise RuntimeError(f"No positive min_collision_time values in {path}")

    order = np.argsort(time, kind="stable")
    time = time[order]
    collision_time = collision_time[order]
    # A checkpoint continuation may repeat its launch time.  The later row is
    # authoritative, so collapse duplicates by keeping the last occurrence.
    # Collapse repeated continuation clocks by the last positive diagnostic before integrating the rate.
    last_collision_time = {
        float(sample_time): float(sample_collision_time)
        for sample_time, sample_collision_time in zip(time, collision_time)
    }
    time = np.asarray(sorted(last_collision_time), dtype=float)
    collision_time = np.asarray(
        [last_collision_time[float(sample_time)] for sample_time in time],
        dtype=float,
    )
    # Extend the first positive collision-time estimate back to zero over the first unsampled interval, as declared by this estimator.
    if time[0] > 0.0:
        time = np.insert(time, 0, 0.0)
        collision_time = np.insert(collision_time, 0, collision_time[0])

    inverse_time = 1.0 / collision_time
    cumulative_age = np.zeros_like(time)
    if time.size > 1:
        # Trapezoidal integration of inverse collision time gives a monotone diagnostic age even when t/t_coll decreases.
        cumulative_age[1:] = np.cumsum(
            0.5 * (inverse_time[1:] + inverse_time[:-1]) * np.diff(time)
        )
    instantaneous_ratio = time / collision_time
    return CollisionClock(
        time=time,
        collision_time=collision_time,
        cumulative_age=cumulative_age,
        instantaneous_ratio=instantaneous_ratio,
    )


def build_curve(
    strength: float, directory: Path, maximum_time: float
) -> CollisionCurve:
    """Load one collisional trajectory and map each profile to collision age."""
    profiles = truncated_profiles(directory, maximum_time)
    clock = load_collision_clock(directory, maximum_time)
    profile_time = np.asarray([time for time, _ in profiles], dtype=float)
    tolerance = 1.0e-8 * max(1.0, maximum_time)
    # Require collision-rate history through every profile epoch before assigning a collision age.
    if profile_time[-1] > clock.time[-1] + tolerance:
        raise RuntimeError(
            f"Profiles extend to t={profile_time[-1]:g}, but collision diagnostics "
            f"end at t={clock.time[-1]:g}: {directory}"
        )
    profile_age = clock.cumulative_at(profile_time)
    if np.any(np.diff(profile_age) <= 0.0):
        raise RuntimeError(f"Collision age is not increasing for {directory}")
    return CollisionCurve(
        strength=strength,
        directory=directory,
        profiles=profiles,
        profile_age=profile_age,
        clock=clock,
    )


def interpolated_profile(
    coordinate: np.ndarray | list[float],
    profiles: list[tuple[float, np.ndarray]],
    target: float,
) -> tuple[float, np.ndarray]:
    """Interpolate density between the complete profiles bracketing ``target``.

    The radial mesh is fixed in these runs.  Density is interpolated in its
    logarithm because the secular SIDM evolution is multiplicative and spans
    many orders of magnitude.  The interpolation is only a visualization
    operation: it never changes a solver output or a restart file.
    """
    values = np.asarray(coordinate, dtype=float)
    if values.size != len(profiles) or values.size == 0:
        raise RuntimeError("Profile coordinate and profile list are inconsistent")
    upper = bisect.bisect_left(values.tolist(), target)
    if upper <= 0:
        return float(profiles[0][0]), profiles[0][1]
    if upper >= values.size:
        return float(profiles[-1][0]), profiles[-1][1]
    if math.isclose(values[upper], target, rel_tol=1.0e-12, abs_tol=1.0e-14):
        return float(profiles[upper][0]), profiles[upper][1]

    lower = upper - 1
    denominator = values[upper] - values[lower]
    if denominator <= 0.0:
        raise RuntimeError("Profile interpolation coordinate is not increasing")
    # The display interpolation fraction comes from the declared profile-time or collision-age coordinate.
    weight = float((target - values[lower]) / denominator)
    lower_time, lower_rows = profiles[lower]
    upper_time, upper_rows = profiles[upper]
    if lower_rows.shape != upper_rows.shape:
        raise RuntimeError("Neighboring profiles use different radial meshes")
    lower_radius = np.asarray(lower_rows["radius"], dtype=float)
    upper_radius = np.asarray(upper_rows["radius"], dtype=float)
    if not np.allclose(lower_radius, upper_radius, rtol=1.0e-11, atol=1.0e-13):
        raise RuntimeError("Neighboring profiles do not share a fixed radial mesh")

    rows = lower_rows.copy()
    lower_density = np.asarray(lower_rows["density"], dtype=float)
    upper_density = np.asarray(upper_rows["density"], dtype=float)
    # Use log-density interpolation where both neighboring values are positive and linear interpolation at support boundaries.
    positive = (lower_density > 0.0) & (upper_density > 0.0)
    interpolated_density = (
        (1.0 - weight) * lower_density + weight * upper_density
    )
    interpolated_density[positive] = np.exp(
        (1.0 - weight) * np.log(lower_density[positive])
        + weight * np.log(upper_density[positive])
    )
    rows["density"] = interpolated_density
    physical_time = (1.0 - weight) * lower_time + weight * upper_time
    return float(physical_time), rows


def masked_density(
    rows: np.ndarray, r_min: float, r_max: float, floor_rel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return resolved positive density samples in the plotted radial range."""
    mask = display_mask(rows, r_min, r_max, floor_rel)
    return (
        np.asarray(rows["radius"][mask], dtype=float),
        np.asarray(rows["density"][mask], dtype=float),
    )


def render(arguments: argparse.Namespace) -> Path:
    """Render all strengths at equal accumulated collision age."""
    control = truncated_profiles(arguments.collisionless, arguments.maximum_time)
    requested = [
        (0.1, arguments.collisional_0p1),
        (0.3, arguments.collisional_0p3),
        (1.0, arguments.collisional_1),
        (3.0, arguments.collisional_3),
    ]
    if any(path is None for _, path in requested):
        raise RuntimeError("Collision-age comparison requires all four strengths")
    curves = [
        build_curve(strength, path, arguments.maximum_time)
        for strength, path in requested
        if path is not None
    ]
    by_strength = {curve.strength: curve for curve in curves}
    if arguments.reference_strength not in by_strength:
        raise RuntimeError(
            f"Reference strength {arguments.reference_strength:g} is unavailable"
        )
    reference = by_strength[arguments.reference_strength]

    # Restrict the comparison to the collision-age interval reached by every requested strength.
    common_age_max = min(float(curve.profile_age[-1]) for curve in curves)
    if not math.isfinite(common_age_max) or common_age_max <= 0.0:
        raise RuntimeError("The common accumulated collision-age interval is empty")
    frame_count = min(
        arguments.maximum_frames,
        min(len(curve.profiles) for curve in curves),
    )
    frame_count = max(2, frame_count)
    target_ages = np.linspace(0.0, common_age_max, frame_count)

    control_times = np.asarray([time for time, _ in control], dtype=float)
    all_profiles = control + [
        profile for curve in curves for profile in curve.profiles
    ]
    density_y_min, density_y_max = density_limits(
        all_profiles,
        arguments.r_min,
        arguments.r_max,
        arguments.density_floor_rel,
    )
    initial_radius, initial_density = masked_density(
        control[0][1],
        arguments.r_min,
        arguments.r_max,
        arguments.density_floor_rel,
    )
    initial_slope_radius, initial_slope = local_poly_log_slope(
        initial_radius, initial_density
    )
    initial_slope_mask = np.isfinite(initial_slope)
    radius_all = np.asarray(control[0][1]["radius"], dtype=float)
    x_min = math.floor(math.log10(float(np.min(radius_all))))
    x_max = math.ceil(math.log10(float(np.max(radius_all))))
    colors = [
        plt.get_cmap("Reds")(RED_LEVEL_BY_STRENGTH[curve.strength])
        for curve in curves
    ]

    output = arguments.output or arguments.output_directory / (
        f"wpic_gamma{number_tag(arguments.gamma)}_"
        "subhalo_collision_age_logrho_slope.gif"
    )
    writer = AtomicGif(output, arguments.fps)
    mapping_rows: list[list[float]] = []

    for target_age in target_ages:
        selected: list[tuple[CollisionCurve, float, np.ndarray]] = []
        for curve in curves:
            selected.append(
                (
                    curve,
                    *interpolated_profile(
                        curve.profile_age, curve.profiles, target_age
                    ),
                )
            )

        # The collisionless curve uses the physical time of the nominated collisional reference at the displayed age.
        reference_time, _ = interpolated_profile(
            reference.profile_age, reference.profiles, target_age
        )
        control_time, control_rows = interpolated_profile(
            control_times, control, reference_time
        )

        control_radius, control_density = masked_density(
            control_rows,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
        )
        control_slope_radius, control_slope = local_poly_log_slope(
            control_radius, control_density
        )
        control_slope_mask = np.isfinite(control_slope)

        collisional_density: list[tuple[np.ndarray, np.ndarray]] = []
        collisional_slope: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        selected_times: list[float] = []
        selected_ages: list[float] = []
        for curve, physical_time, rows in selected:
            radius, density = masked_density(
                rows,
                arguments.r_min,
                arguments.r_max,
                arguments.density_floor_rel,
            )
            slope_radius, slope = local_poly_log_slope(radius, density)
            collisional_density.append((radius, density))
            collisional_slope.append(
                (slope_radius, slope, np.isfinite(slope))
            )
            selected_times.append(float(physical_time))
            selected_ages.append(float(target_age))

        figure, (density_axis, slope_axis) = plt.subplots(
            1, 2, figsize=(13.6, 5.9), dpi=140
        )
        figure.subplots_adjust(
            # Keep the y label fully visible while reclaiming excess whitespace.
            left=0.105,
            right=0.985,
            bottom=0.16,
            # The axes, legend, and title occupy separate vertical bands.
            # This leaves clearance both above the legend and below it for the
            # upper tick labels.
            top=0.755,
            wspace=0.27,
        )

        initial_line = density_axis.plot(
            np.log10(initial_radius),
            np.log10(initial_density),
            color="k",
            ls=":",
            lw=2.2,
            zorder=5,
            label="Initial",
        )[0]
        control_line = density_axis.plot(
            np.log10(control_radius),
            np.log10(control_density),
            color="b",
            ls="-",
            lw=2.6,
            zorder=2,
            label="Collisionless",
        )[0]
        collisional_lines = []
        for curve, color, (radius, density) in zip(
            curves, colors, collisional_density
        ):
            collisional_lines.append(
                density_axis.plot(
                    np.log10(radius),
                    np.log10(density),
                    color=color,
                    ls="--",
                    lw=2.4,
                    zorder=3,
                    label=rf"Strength ${curve.strength:g}$",
                )[0]
            )
        density_axis.set_xlim(x_min, x_max)
        density_axis.set_ylim(
            math.floor(density_y_min), math.ceil(density_y_max)
        )
        density_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        density_axis.set_ylabel(r"$\log_{10}\rho$", fontsize=24, labelpad=11)
        style_axes(density_axis, 19)

        slope_axis.plot(
            np.log10(initial_slope_radius[initial_slope_mask]),
            initial_slope[initial_slope_mask],
            color="k",
            ls=":",
            lw=2.2,
            zorder=5,
        )
        slope_axis.plot(
            np.log10(control_slope_radius[control_slope_mask]),
            control_slope[control_slope_mask],
            color="b",
            ls="-",
            lw=2.6,
            zorder=2,
        )
        for color, (radius, slope, finite) in zip(colors, collisional_slope):
            slope_axis.plot(
                np.log10(radius[finite]),
                slope[finite],
                color=color,
                ls="--",
                lw=2.4,
                zorder=3,
            )
        for guide in (-1.0, -1.5, -3.0):
            slope_axis.axhline(guide, color="0.55", ls="--", lw=1.7, zorder=1)
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
            rf"$\gamma_i={arguments.gamma:g},\quad "
            rf"\tau_{{\rm coll}}=\int_0^t \mathrm{{d}}t'/t_{{\rm coll}}(t')"
            rf"={target_age:.3g}$",
            fontsize=20,
            y=0.992,
        )
        legend_handles = [initial_line, control_line, *collisional_lines]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.900),
            ncol=len(legend_handles),
            fontsize=12.5,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="0.75",
            borderpad=0.4,
            handlelength=2.4,
            columnspacing=1.15,
        )
        writer.append(figure_to_rgb(figure))
        plt.close(figure)

        # Preserve the display time and target age so the normalized comparison remains explicit.
        mapping_rows.append(
            [
                float(target_age),
                float(control_time),
                *selected_times,
                *selected_ages,
            ]
        )
    writer.close()

    stem = output.with_suffix("")
    mapping_path = stem.with_name(f"{stem.name}_frame_mapping.csv")
    mapping_header = [
        "target_cumulative_collision_age",
        "collisionless_reference_time",
        *[f"strength_{curve.strength:g}_physical_time" for curve in curves],
        *[f"strength_{curve.strength:g}_actual_collision_age" for curve in curves],
    ]
    np.savetxt(
        mapping_path,
        np.asarray(mapping_rows, dtype=float),
        delimiter=",",
        header=",".join(mapping_header),
        comments="",
        fmt="%.17g",
    )

    clock_path = stem.with_name(f"{stem.name}_collision_clocks.csv")
    with clock_path.open("w") as handle:
        handle.write(
            "strength,physical_time,collision_time,"
            "cumulative_collision_age,instantaneous_t_over_collision_time\n"
        )
        for curve in curves:
            for values in zip(
                curve.clock.time,
                curve.clock.collision_time,
                curve.clock.cumulative_age,
                curve.clock.instantaneous_ratio,
            ):
                handle.write(
                    f"{curve.strength:.17g},"
                    + ",".join(f"{float(value):.17g}" for value in values)
                    + "\n"
                )

    summary = {
        "gamma_i": arguments.gamma,
        "physical_time_ceiling": arguments.maximum_time,
        "clock": "cumulative_collision_age",
        "clock_definition": "integral_0^t dt_prime / t_coll(t_prime)",
        "collision_time_source": (
            "min_collision_time in each run's collision_history.csv; "
            "computed by VENUS from the evolving distribution function"
        ),
        "collisionless_sampling": (
            f"physical time of the strength-{arguments.reference_strength:g} "
            "trajectory at the displayed collision age"
        ),
        "profile_sampling": (
            "log-linear density interpolation between the two neighboring "
            "complete solver profiles"
        ),
        "common_collision_age_max": common_age_max,
        "frame_count": frame_count,
        "instantaneous_t_over_tcoll_not_used": (
            "It is not an accumulated collision count and can be nonmonotonic "
            "when t_coll evolves."
        ),
        "curves": {},
    }
    for curve in curves:
        # Record nonmonotonic instantaneous t/t_coll separately from the integrated age that drives this movie.
        decreases = int(np.count_nonzero(np.diff(curve.clock.instantaneous_ratio) < 0.0))
        profiles_in_common_interval = curve.profile_age[
            curve.profile_age <= common_age_max
        ]
        spacing_coordinates = profiles_in_common_interval
        if (
            spacing_coordinates.size == 0
            or spacing_coordinates[-1] < common_age_max
        ):
            spacing_coordinates = np.append(
                spacing_coordinates, common_age_max
            )
        collision_age_spacings = np.diff(spacing_coordinates)
        summary["curves"][f"strength_{curve.strength:g}"] = {
            "directory": str(curve.directory),
            "first_positive_collision_time": float(curve.clock.collision_time[0]),
            "last_collision_time": float(curve.clock.collision_time[-1]),
            "last_physical_profile_time": float(curve.profiles[-1][0]),
            "last_cumulative_collision_age": float(curve.profile_age[-1]),
            "instantaneous_clock_decrease_count": decreases,
            "independent_profiles_in_common_collision_age_interval": int(
                profiles_in_common_interval.size
            ),
            "median_collision_age_between_saved_profiles": float(
                np.median(collision_age_spacings)
            ),
            "maximum_collision_age_between_saved_profiles": float(
                np.max(collision_age_spacings)
            ),
        }
    summary_path = stem.with_name(f"{stem.name}_collision_age_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output


def parse_arguments() -> argparse.Namespace:
    """Define the command-line interface for the collision-age renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collisionless", type=Path, required=True)
    parser.add_argument("--collisional-0p1", type=Path, required=True)
    parser.add_argument("--collisional-0p3", type=Path, required=True)
    parser.add_argument("--collisional-1", type=Path, required=True)
    parser.add_argument("--collisional-3", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--maximum-time", type=float, default=10000.0)
    parser.add_argument("--maximum-frames", type=int, default=501)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=900.0)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    parser.add_argument("--slope-y-min", type=float, default=-6.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_arguments())
