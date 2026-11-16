#!/usr/bin/env python3
# Render saved Einstein--Vlasov radial density, slope and compactness sequences.
# Low-compactness data can compare EV and VP on common saved times. The high-compactness table describes one relativistic sequence.
# Smoothing follows the declared inner-boundary geometry and all selected frame clocks remain actual source epochs.

"""Render profile GIFs for weighted Einstein--Vlasov particle calculations.

The low-compactness named-column CSV contains ``time``, ``r``, ``ev_rho``,
``ev_compactness``, ``vp_rho``, and ``vp_compactness`` and compares the
Einstein--Vlasov and Vlasov--Poisson fields at the same saved times.  The
high-compactness CSV contains ``time``, ``r``, ``rho``, and ``compactness``
and displays a single relativistic sequence.  Required columns are validated
before rendering, and rows at each time are sorted onto increasing radius.

Density curves and ``d log(rho) / d log(r)`` are derived with local polynomial
fits after a relative-density floor is applied.  Origin-reaching data use the
regular spherical form ``log(rho) = F(r**2)``; finite inner boundaries use
log-radius fits.  A common set of panel limits is computed across all selected
times so apparent evolution is not produced by autoscaling.

Saved times are retained or reduced by evenly spaced indices, with no temporal
interpolation.  Each GIF is streamed one full frame at a time with fixed
initial-profile references, accompanied by up to three initial/middle/final
PNG previews and a text sidecar containing the exact frame times.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "venus-matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    for font_cache in (Path.home() / ".matplotlib").glob("fontlist*.json"):
        target = mpl_cache / font_cache.name
        if not target.exists():
            shutil.copy2(font_cache, target)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---- Command line and input-schema validation ------------------------------
def parse_args() -> argparse.Namespace:
    """Parse input tables, rendering limits, smoothing, cadence, and outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-profiles", type=Path, required=True)
    parser.add_argument("--high-profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--slope-window", type=int, default=11)
    parser.add_argument("--density-window", type=int)
    parser.add_argument("--high-slope-window", type=int)
    parser.add_argument("--high-density-window", type=int)
    parser.add_argument("--density-floor-fraction", type=float, default=1.0e-7)
    parser.add_argument("--minimum-log-radius", type=float)
    return parser.parse_args()


def load_csv(path: Path, required: set[str]) -> np.ndarray:
    """Load a named-column profile table and validate its required schema."""
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if data.size == 0:
        raise RuntimeError(f"No profile rows found in {path}")
    names = set(data.dtype.names or ())
    if not required.issubset(names):
        missing = ", ".join(sorted(required - names))
        raise RuntimeError(f"{path} is missing columns: {missing}")
    return data


def select_times(data: np.ndarray, frame_count: int) -> np.ndarray:
    """Return all saved times or an endpoint-preserving index-uniform subset."""
    times = np.unique(np.asarray(data["time"], dtype=float))
    if times.size <= frame_count:
        return times
    indices = np.unique(
        np.rint(np.linspace(0, times.size - 1, frame_count)).astype(int)
    )
    return times[indices]


def rows_at_time(data: np.ndarray, time: float) -> np.ndarray:
    """Select one saved time within roundoff tolerance and sort it by radius."""
    rows = data[np.isclose(data["time"], time, rtol=0.0, atol=1.0e-12)]
    return rows[np.argsort(rows["r"])]


# ---- Stable logarithmic density and slope diagnostics ----------------------
def finite_density_profile(
    radius: np.ndarray,
    density: np.ndarray,
    floor_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite positive samples above a relative density floor.

    The floor is a display and differentiation threshold relative to the
    largest positive density in the supplied profile; it does not modify the
    input table.
    """
    radius = np.asarray(radius, dtype=float)
    density = np.asarray(density, dtype=float)
    positive = density[np.isfinite(density) & (density > 0.0)]
    if positive.size == 0:
        return np.empty(0), np.empty(0)
    # The relative positive-density floor defines the display and differentiation support on a copied input profile.
    threshold = max(
        float(np.max(positive)) * floor_fraction,
        np.finfo(float).tiny,
    )
    mask = (
        np.isfinite(radius)
        & np.isfinite(density)
        & (radius > 0.0)
        & (density > threshold)
    )
    return radius[mask], density[mask]


def local_log_profile(
    radius: np.ndarray,
    density: np.ndarray,
    floor_fraction: float,
    window_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit smoothed log density and slope on a finite log-radius profile.

    A centered quadratic fit is evaluated at each retained sample.  Slope
    estimates needing one-sided endpoint windows are masked because they do
    not have the same radial support as the interior estimates.
    """
    radius, density = finite_density_profile(radius, density, floor_fraction)
    if radius.size < 3:
        log_density = np.log10(density)
        return radius, log_density, np.full(radius.shape, np.nan)
    x = np.log10(radius)
    y = np.log10(density)
    window = max(3, int(window_points))
    if window % 2 == 0:
        window += 1
    window = min(window, radius.size if radius.size % 2 == 1 else radius.size - 1)
    if window < 3:
        return radius, y, np.gradient(y, x)
    half = window // 2
    smooth_density = np.full(radius.shape, np.nan)
    slope = np.full(radius.shape, np.nan)
    for index in range(radius.size):
        begin = max(0, index - half)
        end = min(radius.size, index + half + 1)
        if end - begin < window:
            if begin == 0:
                end = min(radius.size, window)
            else:
                begin = max(0, radius.size - window)
        degree = min(2, end - begin - 1)
        # A centered quadratic in log radius gives fitted log density and its local first derivative.
        coefficients = np.polyfit(
            x[begin:end] - x[index],
            y[begin:end],
            degree,
        )
        smooth_density[index] = coefficients[-1]
        slope[index] = coefficients[-2]
    # Do not present one-sided endpoint fits as resolved logarithmic slopes.
    slope[:half] = np.nan
    slope[-half:] = np.nan
    return radius, smooth_density, slope


def boundary_aware_log_profile(
    radius: np.ndarray,
    density: np.ndarray,
    floor_fraction: float,
    window_points: int,
    origin_parity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smoothed log10(rho) and d ln(rho)/d ln(r).

    A fixed-width local polynomial in log(r) is appropriate away from an
    origin.  At an origin-parity boundary, however, dropping half a window
    removes the resolved central profile.  If the density support reaches the
    first mesh cell, fit the regular spherical form ln(rho)=F(r^2) and evaluate
    d ln(rho)/d ln(r)=2 r^2 F'(r^2).  At finite or density-threshold endpoints,
    retain the local polynomial's one-sided estimate instead of silently
    deleting an interval from the plotted radial domain.
    """

    raw_radius = np.asarray(radius, dtype=float)
    raw_density = np.asarray(density, dtype=float)
    raw_positive_radius = raw_radius[
        np.isfinite(raw_radius)
        & np.isfinite(raw_density)
        & (raw_radius > 0.0)
    ]
    mesh_inner_radius = (
        float(np.min(raw_positive_radius))
        if raw_positive_radius.size
        else np.nan
    )
    radius, density = finite_density_profile(
        raw_radius,
        raw_density,
        floor_fraction,
    )
    if radius.size < 3:
        return (
            radius,
            np.log10(density),
            np.full(radius.shape, np.nan),
        )

    requested_window = max(3, int(window_points))
    if requested_window % 2 == 0:
        requested_window += 1
    regular_window = min(
        requested_window,
        radius.size if radius.size % 2 == 1 else radius.size - 1,
    )
    if regular_window < 3:
        log_density = np.log10(density)
        return (
            radius,
            log_density,
            np.gradient(log_density, np.log10(radius)),
        )
    regular_half = regular_window // 2

    # Use the regular spherical fit only when the declared origin boundary and the actual positive-density support both reach the innermost mesh cell.
    support_reaches_origin = (
        origin_parity
        and np.isfinite(mesh_inner_radius)
        and np.isclose(
            radius[0],
            mesh_inner_radius,
            rtol=0.0,
            atol=32.0 * np.finfo(float).eps
                * max(1.0, abs(mesh_inner_radius)),
        )
    )
    # For a finite inner support boundary, retain the declared local log-radius endpoint estimates.
    if not support_reaches_origin:
        x = np.log10(radius)
        y = np.log10(density)
        smooth_density = np.full(radius.shape, np.nan)
        slope = np.full(radius.shape, np.nan)
        for index in range(radius.size):
            begin = max(0, index - regular_half)
            end = min(radius.size, index + regular_half + 1)
            if end - begin < regular_window:
                if begin == 0:
                    end = min(radius.size, regular_window)
                else:
                    begin = max(0, radius.size - regular_window)
            degree = min(2, end - begin - 1)
            coefficients = np.polyfit(
                x[begin:end] - x[index],
                y[begin:end],
                degree,
            )
            smooth_density[index] = coefficients[-1]
            slope[index] = coefficients[-2]
        return radius, smooth_density, slope

    parity_window = regular_window
    parity_half = regular_half
    log_density = np.log(density)
    # Spherical regularity makes log density a smooth function of r squared near the center.
    radius_squared = radius**2
    smooth_density = np.full(radius.shape, np.nan)
    slope = np.full(radius.shape, np.nan)
    for index, center_radius in enumerate(radius):
        begin = max(0, index - parity_half)
        end = min(radius.size, index + parity_half + 1)
        if end - begin < parity_window:
            if begin == 0:
                end = min(radius.size, parity_window)
            else:
                begin = max(0, radius.size - parity_window)
        degree = min(2, end - begin - 1)
        coefficients = np.polyfit(
            radius_squared[begin:end] - center_radius**2,
            log_density[begin:end],
            degree,
        )
        smooth_density[index] = coefficients[-1] / np.log(10.0)
        # The chain rule converts the derivative with respect to r squared into d log rho/d log r.
        slope[index] = 2.0 * center_radius**2 * coefficients[-2]
    return radius, smooth_density, slope


def stable_log_slope(
    radius: np.ndarray,
    density: np.ndarray,
    floor_fraction: float,
    window_points: int,
    origin_parity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radius and the boundary-aware logarithmic density slope."""
    slope_radius, _, slope = boundary_aware_log_profile(
        radius,
        density,
        floor_fraction,
        window_points,
        origin_parity,
    )
    return slope_radius, slope


def padded_limits(
    values: list[np.ndarray],
    lower_quantile: float,
    upper_quantile: float,
    minimum_span: float,
) -> tuple[float, float]:
    """Choose quantile-based fixed limits with padding and a minimum span."""
    finite_parts = [
        np.asarray(part)[np.isfinite(part)]
        for part in values
        if np.asarray(part).size
    ]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return -1.0, 1.0
    joined = np.concatenate(finite_parts)
    lower, upper = np.quantile(joined, [lower_quantile, upper_quantile])
    if upper - lower < minimum_span:
        center = 0.5 * (lower + upper)
        lower = center - 0.5 * minimum_span
        upper = center + 0.5 * minimum_span
    padding = 0.08 * (upper - lower)
    return float(lower - padding), float(upper + padding)


def style_axis(axis: plt.Axes) -> None:
    """Apply the common inward-tick and spine styling to one panel."""
    axis.minorticks_on()
    axis.tick_params(
        which="major",
        direction="in",
        length=9,
        width=1.8,
        labelsize=17,
        top=True,
        right=True,
        pad=5,
    )
    axis.tick_params(
        which="minor",
        direction="in",
        length=5,
        width=1.3,
        top=True,
        right=True,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.6)


def figure_to_rgb(figure: plt.Figure) -> np.ndarray:
    """Rasterize a Matplotlib figure to an RGB-compatible image array."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor="white")
    buffer.seek(0)
    return imageio.imread(buffer)


def profile_arrays(
    rows: np.ndarray,
    density_name: str,
    compactness_name: str,
    floor_fraction: float,
    density_window: int,
    slope_window: int,
    origin_parity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the five arrays needed by one rendered profile series.

    The result contains log radius and smoothed log density, log radius and
    logarithmic slope, plus the compactness column on its original radial
    mesh.  Density smoothing and slope smoothing may use different windows.
    """
    radius = np.asarray(rows["r"], dtype=float)
    density = np.asarray(rows[density_name], dtype=float)
    compactness = np.asarray(rows[compactness_name], dtype=float)
    # Density and slope use their own requested fitting windows while compactness remains on the original saved radial mesh.
    profile_radius, smooth_log_density, _ = boundary_aware_log_profile(
        radius,
        density,
        floor_fraction,
        density_window,
        origin_parity,
    )
    slope_radius, slope = stable_log_slope(
        radius,
        density,
        floor_fraction,
        slope_window,
        origin_parity,
    )
    return (
        np.log10(profile_radius),
        smooth_log_density,
        np.log10(slope_radius),
        slope,
        compactness,
    )


def collect_limits(
    data: np.ndarray,
    times: np.ndarray,
    series: list[tuple[str, str]],
    floor_fraction: float,
    density_window: int,
    slope_window: int,
    origin_parity: bool,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute fixed radius, density, slope, and compactness limits.

    Every selected time and every requested physical series contributes, so
    frame-to-frame changes cannot arise from axis rescaling.
    """
    log_radius_values: list[np.ndarray] = []
    log_density_values: list[np.ndarray] = []
    slope_values: list[np.ndarray] = []
    compactness_values: list[np.ndarray] = []
    # Collect limits across every selected epoch and physical series before creating any animated artist.
    for time in times:
        rows = rows_at_time(data, float(time))
        full_radius = np.asarray(rows["r"], dtype=float)
        full_radius = full_radius[np.isfinite(full_radius) & (full_radius > 0.0)]
        log_radius_values.append(np.log10(full_radius))
        for density_name, compactness_name in series:
            density_x, density_y, _, slope, compactness = profile_arrays(
                rows,
                density_name,
                compactness_name,
                floor_fraction,
                density_window,
                slope_window,
                origin_parity,
            )
            log_density_values.append(density_y)
            slope_values.append(slope)
            compactness_values.append(compactness)
    joined_log_radius = np.concatenate(log_radius_values)
    x_limits = (
        float(np.min(joined_log_radius)),
        float(np.max(joined_log_radius)),
    )
    density_limits = padded_limits(log_density_values, 0.002, 0.998, 1.0)
    slope_limits = padded_limits(slope_values, 0.01, 0.99, 2.0)
    compactness = np.concatenate(compactness_values)
    compactness = compactness[np.isfinite(compactness)]
    compactness_upper = max(float(np.max(compactness)) * 1.06, 1.0e-8)
    compactness_limits = (0.0, compactness_upper)
    return x_limits, density_limits, slope_limits, compactness_limits


# ---- Fixed-layout frame construction and streamed GIF output --------------
def render_case(
    data: np.ndarray,
    times: np.ndarray,
    series: list[tuple[str, str, str, str, str]],
    output: Path,
    fps: float,
    floor_fraction: float,
    density_window: int,
    slope_window: int,
    minimum_log_radius: float | None,
    slope_limits_override: tuple[float, float] | None = None,
    origin_parity: bool = False,
) -> None:
    """Render one same-time profile sequence and its frame-time sidecars.

    Lines are allocated once, updated in place for each selected saved time,
    and rasterized into a streaming GIF writer.  The earliest available
    profile remains visible as a dotted reference.  Up to three representative
    PNG frames and ``<output>.times.txt`` preserve the exact visual and
    temporal sampling used by the animation.
    """
    limits = collect_limits(
        data,
        times,
        [(item[0], item[1]) for item in series],
        floor_fraction,
        density_window,
        slope_window,
        origin_parity,
    )
    if minimum_log_radius is not None:
        limits = (
            (max(limits[0][0], minimum_log_radius), limits[0][1]),
            limits[1],
            limits[2],
            limits[3],
        )
    figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.45))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.85, wspace=0.28)
    labels = (
        r"$\log_{10}\rho$",
        r"$d\log\rho/d\log r$",
        r"$2m/r$",
    )
    for axis, ylabel in zip(axes, labels):
        style_axis(axis)
        axis.set_xlabel(r"$\log_{10} r$", fontsize=21)
        axis.set_ylabel(ylabel, fontsize=21)
        axis.set_xlim(*limits[0])
    axes[0].set_ylim(*limits[1])
    axes[1].set_ylim(
        *(slope_limits_override if slope_limits_override is not None else limits[2])
    )
    axes[2].set_ylim(*limits[3])

    line_sets: list[tuple[plt.Line2D, plt.Line2D, plt.Line2D]] = []
    for _, _, color, linestyle, label in series:
        density_line, = axes[0].plot(
            [], [], color=color, linestyle=linestyle, lw=2.8, label=label
        )
        slope_line, = axes[1].plot(
            [], [], color=color, linestyle=linestyle, lw=2.8
        )
        compactness_line, = axes[2].plot(
            [], [], color=color, linestyle=linestyle, lw=2.8
        )
        line_sets.append((density_line, slope_line, compactness_line))

    # The earliest physical source profile remains a fixed reference throughout the movie.
    initial_rows = rows_at_time(data, float(np.min(data["time"])))
    initial_density_name, initial_compactness_name = series[0][:2]
    (
        initial_density_x,
        initial_density_y,
        initial_slope_x,
        initial_slope,
        initial_compactness,
    ) = profile_arrays(
        initial_rows,
        initial_density_name,
        initial_compactness_name,
        floor_fraction,
        density_window,
        slope_window,
        origin_parity,
    )
    initial_log_radius = np.log10(np.asarray(initial_rows["r"], dtype=float))
    axes[0].plot(
        initial_density_x,
        initial_density_y,
        color="k",
        linestyle=":",
        lw=2.8,
        label="Initial",
        zorder=4,
    )
    axes[1].plot(
        initial_slope_x,
        initial_slope,
        color="k",
        linestyle=":",
        lw=2.8,
        zorder=4,
    )
    axes[2].plot(
        initial_log_radius,
        initial_compactness,
        color="k",
        linestyle=":",
        lw=2.8,
        zorder=4,
    )
    axes[0].legend(
        loc="best",
        fontsize=15,
        frameon=False,
        handlelength=3.0,
    )
    title = figure.suptitle("", fontsize=23)

    output.parent.mkdir(parents=True, exist_ok=True)
    previews = {
        0: output.with_name(output.stem + "_initial.png"),
        len(times) // 2: output.with_name(output.stem + "_middle.png"),
        len(times) - 1: output.with_name(output.stem + "_final.png"),
    }
    with imageio.get_writer(
        output,
        mode="I",
        duration=1.0 / fps,
        loop=0,
        subrectangles=True,
    ) as writer:
        # Update each field from its same-time radial rows and keep representative images tied to exact frame indices.
        for frame_index, time in enumerate(times):
            rows = rows_at_time(data, float(time))
            log_radius_all = np.log10(np.asarray(rows["r"], dtype=float))
            for item, lines in zip(series, line_sets):
                density_name, compactness_name, _, _, _ = item
                density_x, density_y, slope_x, slope, compactness = profile_arrays(
                    rows,
                    density_name,
                    compactness_name,
                    floor_fraction,
                    density_window,
                    slope_window,
                    origin_parity,
                )
                lines[0].set_data(density_x, density_y)
                lines[1].set_data(slope_x, slope)
                lines[2].set_data(log_radius_all, compactness)
            title.set_text(f"t = {float(time):.3f}")
            frame = figure_to_rgb(figure)
            writer.append_data(frame)
            if frame_index in previews:
                imageio.imwrite(previews[frame_index], frame)
    plt.close(figure)
    # Save full-precision actual frame clocks beside the encoded movie.
    output.with_suffix(".times.txt").write_text(
        "\n".join(f"{float(time):.17g}" for time in times) + "\n"
    )


# ---- Low/high scenario orchestration ---------------------------------------
def main() -> None:
    """Validate both profile schemas and render low/high-compactness GIFs."""
    args = parse_args()
    if args.frames < 2 or args.fps <= 0.0:
        raise RuntimeError("--frames must be at least 2 and --fps must be positive")
    low = load_csv(
        args.low_profiles,
        {
            "time",
            "r",
            "ev_rho",
            "ev_compactness",
            "vp_rho",
            "vp_compactness",
        },
    )
    high = load_csv(
        args.high_profiles,
        {"time", "r", "rho", "compactness"},
    )
    low_times = select_times(low, args.frames)
    high_times = select_times(high, args.frames)
    density_window = args.density_window or args.slope_window
    high_density_window = args.high_density_window or density_window
    high_slope_window = args.high_slope_window or args.slope_window
    for name, value in (
        ("--density-window", density_window),
        ("--slope-window", args.slope_window),
        ("--high-density-window", high_density_window),
        ("--high-slope-window", high_slope_window),
    ):
        if value < 3:
            raise RuntimeError(f"{name} must be at least 3")
    render_case(
        low,
        low_times,
        [
            ("ev_rho", "ev_compactness", "b", "-", "Einstein--Vlasov"),
            ("vp_rho", "vp_compactness", "r", "--", "Vlasov--Poisson"),
        ],
        args.output_dir / "low_compactness_profiles.gif",
        args.fps,
        args.density_floor_fraction,
        density_window,
        args.slope_window,
        args.minimum_log_radius,
        origin_parity=False,
    )
    render_case(
        high,
        high_times,
        [("rho", "compactness", "b", "-", "Einstein--Vlasov")],
        args.output_dir / "high_compactness_profiles.gif",
        args.fps,
        args.density_floor_fraction,
        high_density_window,
        high_slope_window,
        args.minimum_log_radius,
        (-6.0, 2.0),
        origin_parity=True,
    )
    print(args.output_dir / "low_compactness_profiles.gif")
    print(args.output_dir / "high_compactness_profiles.gif")


if __name__ == "__main__":
    main()
