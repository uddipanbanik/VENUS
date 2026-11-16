#!/usr/bin/env python3
# Animate radial density and signed logarithmic slopes, with fixed inner and moving outer local power-law fits.
# Profile CSVs and metadata supply actual time, density, radial support and effective marker counts.
# The outer fit tracks a defined profile feature; output tables preserve its estimator settings separately from a physical asymptotic interpretation.

"""Render density and logarithmic-slope GIFs for pure-power-law WPIC runs.

Each complete ``profile_*.csv`` file supplies one physical time.  Incomplete
live-output frames are ignored, density slopes use local polynomial fits in
log radius, and the resulting animation is atomically renamed into place.

Adapted from Sachiet Kapur's analysis tools. The derivative panel gives the
signed slope s=dlog(rho)/dlog(r); the evolution plots give gamma_in=-s_in
and gamma_out=-s_out, with rho proportional to r**(-gamma). The moving
outer fit is an empirical profile-feature estimate; inspect its window and
marker sampling before assigning it a physical interpretation.

The initial cusp slope gamma_i is read from each run's ``metadata.json`` (the
``gamma_i`` field written by the expanding WPIC driver), so any gamma the
binary can produce is supported without editing this script.
"""

from __future__ import annotations

import argparse
import json
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
from PIL import GifImagePlugin, Image


# ---- Run identity read from the driver's own metadata ---------------------
def read_gamma(directory: Path) -> float:
    """Return gamma_i for a run by reading its ``metadata.json``.

    The expanding WPIC driver writes ``"gamma_i": <value>`` as a top-level
    key.  Reading it here removes any hardcoded slope list, so a run at any
    gamma the binary accepts is plotted with the correct label automatically.
    """
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"No metadata.json in {directory}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if "gamma_i" not in metadata:
        raise RuntimeError(f"{metadata_path} has no 'gamma_i' field")
    return float(metadata["gamma_i"])


def gamma_label(gamma: float) -> str:
    """Build a filename-safe label like ``gamma1p0`` from a gamma value."""
    return "gamma" + f"{gamma:g}".replace(".", "p").replace("-", "m")


def read_ppc(directory: Path) -> int | None:
    """Return particles_per_radial_cell from metadata.json, or None if absent."""
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    value = metadata.get("particles_per_radial_cell")
    return int(value) if value is not None else None


# ---- Fixed two-power-law fit configuration (program constants) -------------
# Each power law is fit over a window centred on a fixed log10(r) location and
# drawn over a slightly narrower window so the dashed guide sits inside the
# fitted data. The constants below define the radial centers and widths
# used by each fit throughout the animation.
INNER_FIT_CENTER_LOG10R = -0.75
OUTER_FIT_CENTER_LOG10R = 1.25
FIT_HALF_WIDTH_LOG10R = 0.25   # used for the actual least-squares fit
DRAW_HALF_WIDTH_LOG10R = 0.20  # used only for the drawn dashed segment


def read_radial_bounds(directory: Path) -> tuple[float, float] | tuple[None, None]:
    """Return (radial_minimum, radial_maximum) from metadata.json if present.

    The expanding/collisional WPIC drivers record the mesh limits the run was
    launched with.  Plotting exactly within those limits avoids assuming a
    fixed window that may not match the run's true radial domain.  Missing keys
    return (None, None) so the caller can fall back to the data extent.
    """
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return (None, None)
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    lo = metadata.get("radial_minimum")
    hi = metadata.get("radial_maximum")
    if lo is None or hi is None:
        return (None, None)
    return (float(lo), float(hi))


def fit_power_law(
    radius: np.ndarray,
    density: np.ndarray,
    center_log10r: float,
    half_width_log10r: float,
) -> tuple[float, float] | None:
    """Least-squares fit of log10(rho) = slope*log10(r) + intercept.

    The fit uses only points whose log10(r) lies within ``half_width_log10r``
    of ``center_log10r`` and whose density is finite and positive.  Returns
    ``(slope, intercept)`` in log10-log10 space, or ``None`` when fewer than
    two usable points fall in the window (so the caller can skip drawing).
    """
    log_r = np.log10(radius)
    finite = np.isfinite(density) & (density > 0.0) & np.isfinite(log_r)
    # Fit log density over the explicit fixed log-radius interval, using only finite positive data.
    window = finite & (np.abs(log_r - center_log10r) <= half_width_log10r)
    if np.count_nonzero(window) < 2:
        return None
    x = log_r[window]
    y = np.log10(density[window])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def draw_power_law(
    axis,
    fit: tuple[float, float] | None,
    center_log10r: float,
    draw_half_width_log10r: float,
    color: str,
    lw: float = 1.8,
) -> None:
    """Draw a dashed power-law guide over a narrow window around its centre."""
    if fit is None:
        return
    slope, intercept = fit
    x = np.array(
        [center_log10r - draw_half_width_log10r,
         center_log10r + draw_half_width_log10r]
    )
    axis.plot(x, slope * x + intercept, color=color, ls="--", lw=lw, zorder=4)


# ---- Shoulder (ejected-material) outer-slope detection --------------------
# alpha_out is a single-centre power-law fit, like the inner one, but its
# centre is detected per frame and therefore MOVES as the profile evolves.
# Walking inward on the RAW (unsmoothed) log-log profile from the outer edge,
# the detector stops at the FIRST point that is either flat
# (|d2 log rho / d(log r)^2| < SHOULDER_D2_FLAT) or sharp (|d2| >
# SHOULDER_KINK_D2).  The fit centre is the midpoint in log r between the
# outer edge and that first-encountered feature; alpha_out is then fit over a
# fixed half-width window around that centre.  Early on this sits on the steep
# truncation (large negative alpha_out, e.g. ~-27); as the shoulder develops
# the centre moves outward and the slope shallows.
SHOULDER_NEFF_MIN = 200.0        # outer edge = outermost cell with >= this neff
SHOULDER_D2_FLAT = 2.0           # |d2| below this = flat feature
SHOULDER_KINK_D2 = 6.0           # |d2| above this = sharp feature
SHOULDER_FIT_HALF_WIDTH_LOG10R = 0.25   # least-squares window half-width
SHOULDER_DRAW_HALF_WIDTH_LOG10R = 0.20  # drawn dashed-guide half-width


def _raw_derivatives(
    log_r: np.ndarray, log_rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (unsmoothed) first and second derivatives via np.gradient."""
    d1 = np.gradient(log_rho, log_r)
    d2 = np.gradient(d1, log_r)
    return d1, d2


def detect_shoulder_fit(
    radius: np.ndarray,
    density: np.ndarray,
    neff: np.ndarray | None,
) -> tuple[float, float, float] | None:
    """Detect the moving alpha_out centre and fit the slope there (raw data).

    Returns ``(slope, intercept, center_log10r)`` for a single-centre power-law
    fit, or ``None`` when no centre can be located.  The centre is the midpoint
    between the outer edge and the first flat-or-sharp feature found walking
    inward, and is recomputed for every frame so the drawn guide tracks the
    evolving profile.
    """
    log_r = np.log10(radius)
    finite = np.isfinite(density) & (density > 0.0) & np.isfinite(log_r)
    if np.count_nonzero(finite) < 4:
        return None
    lr = log_r[finite]
    lrho = np.log10(density[finite])
    ne = neff[finite] if neff is not None else np.full(lr.size, np.inf)
    order = np.argsort(lr)
    lr, lrho, ne = lr[order], lrho[order], ne[order]

    d1, d2 = _raw_derivatives(lr, lrho)

    # outer edge: outermost cell that still has markers
    # The outer feature search starts at the last radius satisfying the specified effective-count threshold.
    have_markers = np.where(ne >= SHOULDER_NEFF_MIN)[0]
    if not have_markers.size:
        return None
    edge = int(have_markers[-1])

    # walk inward: first flat OR sharp feature
    feature = None
    # Scan inward for the first finite second-derivative feature, then define a fit center between that feature and the sampled outer edge.
    for i in range(edge - 1, -1, -1):
        if not np.isfinite(d2[i]):
            continue
        if abs(d2[i]) < SHOULDER_D2_FLAT or abs(d2[i]) > SHOULDER_KINK_D2:
            feature = i
            break
    if feature is None:
        return None

    center = 0.5 * (lr[edge] + lr[feature])

    # fit over a fixed window around the (moving) centre
    # The moving outer fit uses its fixed width and count condition, preserving the measured location in every frame.
    window = ((np.abs(lr - center) <= SHOULDER_FIT_HALF_WIDTH_LOG10R)
              & (ne >= SHOULDER_NEFF_MIN) & (lr <= lr[edge]))
    if np.count_nonzero(window) < 2:
        return None
    slope, intercept = np.polyfit(lr[window], lrho[window], 1)
    return float(slope), float(intercept), float(center)


def draw_shoulder(axis, shoulder, color: str, lw: float = 2.0) -> None:
    """Draw the dashed alpha_out guide at the detected (moving) centre."""
    if shoulder is None:
        return
    slope, intercept, center = shoulder
    x = np.array(
        [center - SHOULDER_DRAW_HALF_WIDTH_LOG10R,
         center + SHOULDER_DRAW_HALF_WIDTH_LOG10R]
    )
    axis.plot(x, slope * x + intercept, color=color, ls="--", lw=lw, zorder=4)


# Key log-slope reference values drawn on every derivative panel.
SLOPE_REFERENCE_LEVELS = (-1.0, -1.5, -3.0)


def draw_slope_references(axis) -> None:
    """Draw dashed log-slope reference lines, labelled in the clear right margin.

    Each line stops short of the frame so its white-boxed value sits beside the
    panel rather than on top of the curve.
    """
    for level in SLOPE_REFERENCE_LEVELS:
        axis.axhline(level, xmax=0.88, color="0.55", ls="--", lw=1.7, zorder=1)
        axis.text(0.92, level, rf"${level:g}$",
                  transform=axis.get_yaxis_transform(), color="0.35",
                  fontsize=13.5, ha="center", va="center",
                  bbox={"boxstyle": "square,pad=0.12", "facecolor": "white",
                        "edgecolor": "none", "alpha": 1.0}, zorder=5)


class AtomicGif:
    """Stream frames into a full-frame GIF and atomically publish it.

    Frames are never accumulated in memory.  The temporary file is renamed
    only after its GIF trailer has been written, so readers cannot mistake a
    half-written animation for a physical result.
    """

    def __init__(self, output: Path, fps: float) -> None:
        """Open a sibling temporary GIF with the requested fixed frame rate."""
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.temporary = output.with_name(f"{output.stem}.tmp.gif")
        self.duration_ms = int(round(1000.0 / max(fps, 1.0e-6)))
        self.handle = self.temporary.open("wb")
        self.frame_count = 0

    def append(self, frame: np.ndarray) -> None:
        """Quantize and append one complete RGB frame to the temporary GIF."""
        image = GifImagePlugin._normalize_mode(Image.fromarray(frame))
        encoder = {"duration": self.duration_ms, "disposal": 2}
        image = GifImagePlugin._normalize_palette(image, None, encoder)
        if self.frame_count == 0:
            encoder["loop"] = 0
            for block in GifImagePlugin._get_global_header(image, encoder):
                self.handle.write(block)
        else:
            encoder["include_color_table"] = True
        GifImagePlugin._write_frame_data(self.handle, image, (0, 0), encoder)
        self.frame_count += 1

    def close(self) -> None:
        """Write the trailer, synchronize the file, and replace the output."""
        if self.frame_count == 0:
            self.handle.close()
            self.temporary.unlink(missing_ok=True)
            raise RuntimeError("Cannot write a GIF without frames")
        self.handle.write(b";")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.temporary.replace(self.output)


# ---- Weighted-PIC profile ingestion and physics diagnostics ---------------
def load_profiles(directory: Path, stride: int = 1) -> list[tuple[float, np.ndarray]]:
    """Read complete radial profiles and return one snapshot per physical time.

    A profile supplies the spherical density ``rho(r)`` and the effective
    marker count ``neff`` deposited on the field mesh.  Repeated times are
    replaced by the last complete file, which is the desired behavior after a
    checkpoint continuation rewrites its launch snapshot.
    """
    def step_of(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return 0

    all_paths = sorted(directory.glob("profile_*.csv"), key=step_of)
    if not all_paths:
        raise RuntimeError(f"No profile_*.csv files in {directory}")
    expected_cells = None
    metadata = directory / "metadata.json"
    if metadata.is_file():
        expected_cells = json.loads(metadata.read_text()).get("radial_cells")
    loaded: dict[float, np.ndarray] = {}
    for path in all_paths:
        try:
            rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        except ValueError:
            continue  # A live writer may still be completing the last row.
        required = {"time", "radius", "density", "neff"}
        if not required.issubset(rows.dtype.names or ()):
            raise RuntimeError(f"{path} requires columns {sorted(required)}")
        if rows.size < 3 or (expected_cells is not None and rows.size != int(expected_cells)):
            continue
        time = float(rows["time"][0])
        if (not np.isfinite(time) or np.any(rows["time"] != time)
                or np.any(~np.isfinite(rows["radius"]))
                or np.any(rows["radius"] <= 0) or np.any(np.diff(rows["radius"]) <= 0)
                or np.any(~np.isfinite(rows["density"])) or np.any(rows["density"] < 0)
                or np.any(~np.isfinite(rows["neff"])) or np.any(rows["neff"] < 0)):
            continue
        loaded[time] = rows
    if not loaded:
        raise RuntimeError(f"No complete profile_*.csv files in {directory}")
    # Validate all candidate profiles before applying the requested stride, retaining the earliest and latest complete states.
    complete = sorted(loaded.items())
    keep = set(range(0, len(complete), max(1, int(stride))))
    keep.update((0, len(complete) - 1))
    return [complete[i] for i in sorted(keep)]


def display_mask(rows: np.ndarray, r_min: float, r_max: float, floor_rel: float) -> np.ndarray:
    """Select the radial interval where the deposited density is meaningful.

    The relative floor removes empty-marker tails before taking logarithms; it
    is a plotting/noise cut and does not modify the simulated mass profile.
    """
    radius = np.asarray(rows["radius"], dtype=float)
    density = np.asarray(rows["density"], dtype=float)
    finite_positive = np.isfinite(density) & (density > 0.0)
    peak = float(np.max(density[finite_positive])) if np.any(finite_positive) else 0.0
    floor = max(peak * floor_rel, np.finfo(float).tiny)
    return (
        np.isfinite(radius)
        & (radius >= r_min)
        & (radius <= r_max)
        & np.isfinite(density)
        & (density >= floor)
    )


def local_poly_log_slope(
    radius: np.ndarray,
    density: np.ndarray,
    window_points: int = 21,
    poly_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``d log(rho) / d log(r)`` with a local polynomial.

    The derivative is taken in logarithmic coordinates, so the reported value
    is the physical power-law index: ``rho proportional to r**slope`` locally.
    Edge estimates are hidden because their one-sided fit is especially
    sensitive to the inner and outer numerical boundaries.
    """
    order = np.argsort(radius)
    radius = np.asarray(radius[order], dtype=float)
    density = np.asarray(density[order], dtype=float)
    radius, unique = np.unique(radius, return_index=True)
    density = density[unique]
    x = np.log10(radius)
    y = np.log10(density)
    count = x.size
    window = min(max(int(window_points) | 1, 5), count if count % 2 else count - 1)
    slopes = np.full(count, np.nan, dtype=float)
    if window < 3:
        return radius, slopes
    half = window // 2
    for index in range(count):
        begin = max(0, index - half)
        end = min(count, index + half + 1)
        if end - begin < window:
            if begin == 0:
                end = min(count, window)
            else:
                begin = max(0, count - window)
        degree = min(poly_order, end - begin - 1)
        # The centered local polynomial supplies the signed derivative at each log-radius sample.
        coefficients = np.polyfit(x[begin:end] - x[index], y[begin:end], degree)
        slopes[index] = float(coefficients[-2])
    # The first and last half-window are one-sided and most sensitive to the
    # marker cutoff and radial boundary; omit them from the displayed slope.
    slopes[:half] = np.nan
    slopes[-half:] = np.nan
    return radius, slopes


def style_axes(axis: plt.Axes, tick_label_size: float) -> None:
    """Apply the common publication style without altering plotted data."""
    axis.minorticks_on()
    axis.tick_params(
        which="major",
        direction="in",
        length=11,
        width=2.0,
        labelsize=tick_label_size,
        top=True,
        right=True,
        pad=6,
    )
    axis.tick_params(
        which="minor",
        direction="in",
        length=6,
        width=1.5,
        top=True,
        right=True,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)


def figure_to_rgb(figure: plt.Figure) -> np.ndarray:
    """Rasterize one completed Matplotlib figure for the GIF encoder."""
    figure.canvas.draw()
    # ``FigureCanvasAgg.tostring_rgb`` was removed in Matplotlib 3.10.
    # ``buffer_rgba`` is public across every Matplotlib version supported by
    # VENUS; copy the RGB channels because the canvas reuses its buffer.
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def density_limits(
    profiles: list[tuple[float, np.ndarray]],
    r_min: float,
    r_max: float,
    floor_rel: float,
) -> tuple[float, float]:
    """Choose shared logarithmic-density limits over the full time sequence."""
    values: list[np.ndarray] = []
    for _, rows in profiles:
        mask = display_mask(rows, r_min, r_max, floor_rel)
        values.append(np.log10(np.asarray(rows["density"][mask], dtype=float)))
    joined = np.concatenate(values)
    if not joined.size:
        raise RuntimeError("No positive density in the selected radial interval")
    lower = math.floor((float(np.min(joined)) - 0.15) * 2.0) / 2.0
    upper = math.ceil((float(np.max(joined)) + 0.15) * 2.0) / 2.0
    return lower, upper


# ---- Per-slope density/slope animation ------------------------------------
def render_case(
    input_directory: Path,
    output_directory: Path,
    label: str,
    gamma: float,
    fps: float,
    r_min: float,
    r_max: float,
    density_floor_rel: float,
    slope_y_min: float,
    slope_y_max: float,
    stride: int = 1,
    ppc: int | None = None,
) -> Path:
    """Render one initial-cusp-slope sequence.

    Blue is the evolving numerical density/slope and dotted black is the
    initial condition.  Holding both panels' limits fixed makes changes in the
    central cusp visible as evolution rather than as camera rescaling.
    """
    profiles = load_profiles(input_directory, stride)
    rho_y_min, rho_y_max = density_limits(profiles, r_min, r_max, density_floor_rel)
    time_min, time_max = profiles[0][0], profiles[-1][0]
    ppc_tag = f"_{ppc}ppc" if ppc is not None else ""
    output = output_directory / (
        f"wpic_{label}{ppc_tag}_logrho_slope_t{time_min:g}_to_t{time_max:g}.gif"
    )
    writer = AtomicGif(output, fps)
    # Prefer the run's true mesh limits (metadata.json); fall back to the data
    # extent only when those keys are absent.
    bound_lo, bound_hi = read_radial_bounds(input_directory)
    if bound_lo is not None and bound_hi is not None and bound_lo > 0.0:
        axis_log_r_min = math.log10(bound_lo)
        axis_log_r_max = math.log10(bound_hi)
    else:
        axis_log_r_min = math.floor(math.log10(float(np.min(profiles[0][1]["radius"]))))
        axis_log_r_max = math.ceil(math.log10(float(np.max(profiles[0][1]["radius"]))))
    initial_rows = profiles[0][1]
    initial_mask = display_mask(initial_rows, r_min, r_max, density_floor_rel)
    initial_radius = np.asarray(initial_rows["radius"][initial_mask], dtype=float)
    initial_density = np.asarray(initial_rows["density"][initial_mask], dtype=float)
    initial_neff = np.asarray(initial_rows["neff"][initial_mask], dtype=float)
    initial_slope_radius, initial_slope = local_poly_log_slope(
        initial_radius,
        initial_density,
    )
    initial_slope_mask = np.isfinite(initial_slope)
    # Evaluate the initial reference with the same fixed-inner and feature-tracking outer estimators used during evolution.
    initial_inner_fit = fit_power_law(
        initial_radius, initial_density,
        INNER_FIT_CENTER_LOG10R, FIT_HALF_WIDTH_LOG10R)
    initial_outer_fit = detect_shoulder_fit(
        initial_radius, initial_density, initial_neff)
    final_inner_fit = None
    final_outer_fit = None
    final_radius = initial_radius
    final_density = initial_density
    final_slope_radius = initial_slope_radius
    final_slope = initial_slope

    # Keep each actual clock and both signed fit values so the index curves can be reconstructed from numerical data.
    alpha_time = []
    alpha_in_series = []
    alpha_out_series = []
    for time, rows in profiles:
        mask = display_mask(rows, r_min, r_max, density_floor_rel)
        radius = np.asarray(rows["radius"][mask], dtype=float)
        density = np.asarray(rows["density"][mask], dtype=float)
        neff = np.asarray(rows["neff"][mask], dtype=float)
        slope_radius, slope = local_poly_log_slope(radius, density)
        inner_fit = fit_power_law(
            radius, density, INNER_FIT_CENTER_LOG10R, FIT_HALF_WIDTH_LOG10R)
        outer_fit = detect_shoulder_fit(radius, density, neff)
        alpha_time.append(time)
        alpha_in_series.append(inner_fit[0] if inner_fit is not None else np.nan)
        alpha_out_series.append(outer_fit[0] if outer_fit is not None else np.nan)
        # Remember the most recent (final) frame for the superimposed PNG.
        final_inner_fit = inner_fit
        final_outer_fit = outer_fit
        final_radius = radius
        final_density = density
        final_slope_radius = slope_radius
        final_slope = slope

        figure, (rho_axis, slope_axis) = plt.subplots(1, 2, figsize=(13.0, 5.25), dpi=140)
        figure.subplots_adjust(left=0.10, right=0.975, bottom=0.18, top=0.84, wspace=0.33)

        rho_axis.plot(
            np.log10(radius),
            np.log10(density),
            color="b",
            ls="-",
            lw=2.5,
            zorder=2, label="evolving profile",
        )
        rho_axis.plot(
            np.log10(initial_radius),
            np.log10(initial_density),
            color="k",
            ls=":",
            lw=2.3,
            zorder=3, label="initial profile",
        )
        rho_axis.set_xlim(axis_log_r_min, axis_log_r_max)
        rho_axis.set_ylim(rho_y_min, rho_y_max)
        rho_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        rho_axis.set_ylabel(r"$\log_{10}\rho$", fontsize=24, labelpad=11)
        draw_power_law(rho_axis, inner_fit, INNER_FIT_CENTER_LOG10R,
                       DRAW_HALF_WIDTH_LOG10R, "k")
        draw_shoulder(rho_axis, outer_fit, "k")
        annotation_lines = []
        if inner_fit is not None:
            annotation_lines.append(rf"$s_{{\rm in}}={inner_fit[0]:.2f}$")
        if outer_fit is not None:
            annotation_lines.append(rf"$s_{{\rm out}}={outer_fit[0]:.2f}$")
        if annotation_lines:
            rho_axis.text(
                0.04, 0.06, "\n".join(annotation_lines),
                transform=rho_axis.transAxes, fontsize=16, ha="left", va="bottom",
                zorder=6)
        rho_axis.plot([], [], color="k", ls="--", lw=1.8, label="local power-law fits")
        rho_axis.legend(loc="upper right", fontsize=11, frameon=True, facecolor="white")
        style_axes(rho_axis, 19)

        slope_mask = np.isfinite(slope)
        slope_axis.plot(
            np.log10(slope_radius[slope_mask]),
            slope[slope_mask],
            color="b",
            ls="-",
            lw=2.5,
            zorder=2, label="evolving profile",
        )
        slope_axis.plot(
            np.log10(initial_slope_radius[initial_slope_mask]),
            initial_slope[initial_slope_mask],
            color="k",
            ls=":",
            lw=2.3,
            zorder=3, label="initial profile",
        )
        draw_slope_references(slope_axis)
        slope_axis.set_xlim(axis_log_r_min, axis_log_r_max)
        slope_axis.set_ylim(slope_y_min, slope_y_max)
        slope_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        slope_axis.set_ylabel(r"$\mathrm{d}\log\rho/\mathrm{d}\log r$", fontsize=24, labelpad=11)
        slope_axis.legend(loc="upper right", fontsize=11, frameon=True, facecolor="white")
        style_axes(slope_axis, 19)

        figure.suptitle(rf"$\gamma_i={gamma:g},\qquad t={time:g}$", fontsize=24, y=0.965)
        writer.append(figure_to_rgb(figure))
        plt.close(figure)
    writer.close()

    # ---- Initial-vs-final density PNG (log rho, with fitted slopes) --------
    density_png = output.with_name(f"{output.stem}_first_last_density.png")
    fig_d, ax_d = plt.subplots(figsize=(7.2, 5.6), dpi=140)
    fig_d.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.9)
    ax_d.plot(np.log10(initial_radius), np.log10(initial_density),
              color="k", ls=":", lw=2.3, label=f"initial (t={time_min:g})", zorder=3)
    ax_d.plot(np.log10(final_radius), np.log10(final_density),
              color="b", ls="-", lw=2.5, label=f"final (t={time_max:g})", zorder=2)
    draw_power_law(ax_d, initial_inner_fit, INNER_FIT_CENTER_LOG10R,
                   DRAW_HALF_WIDTH_LOG10R, "0.4")
    draw_shoulder(ax_d, initial_outer_fit, "0.4")
    draw_power_law(ax_d, final_inner_fit, INNER_FIT_CENTER_LOG10R,
                   DRAW_HALF_WIDTH_LOG10R, "r")
    draw_shoulder(ax_d, final_outer_fit, "r")
    numbers = []
    if initial_inner_fit is not None:
        numbers.append(rf"init $s_{{\rm in}}={initial_inner_fit[0]:.2f}$")
    if initial_outer_fit is not None:
        numbers.append(rf"init $s_{{\rm out}}={initial_outer_fit[0]:.2f}$")
    if final_inner_fit is not None:
        numbers.append(rf"final $s_{{\rm in}}={final_inner_fit[0]:.2f}$")
    if final_outer_fit is not None:
        numbers.append(rf"final $s_{{\rm out}}={final_outer_fit[0]:.2f}$")
    if numbers:
        ax_d.text(0.04, 0.06, "\n".join(numbers), transform=ax_d.transAxes,
                  fontsize=13, ha="left", va="bottom", zorder=6)
    ax_d.set_xlim(axis_log_r_min, axis_log_r_max)
    ax_d.set_ylim(rho_y_min, rho_y_max)
    ax_d.set_xlabel(r"$\log_{10} r$", fontsize=22, labelpad=9)
    ax_d.set_ylabel(r"$\log_{10}\rho$", fontsize=22, labelpad=11)
    ax_d.plot([], [], color="0.4", ls="--", lw=1.8, label="initial fits")
    ax_d.plot([], [], color="r", ls="--", lw=1.8, label="final fits")
    ax_d.legend(loc="upper right", fontsize=12, frameon=True, facecolor="white")
    style_axes(ax_d, 17)
    fig_d.suptitle(rf"$\gamma_i={gamma:g}$: initial vs final", fontsize=18, y=0.97)
    fig_d.savefig(density_png)
    plt.close(fig_d)

    # ---- Initial-vs-final log-slope PNG -----------------------------------
    slope_png = output.with_name(f"{output.stem}_first_last_slope.png")
    fig_s, ax_s = plt.subplots(figsize=(7.2, 5.6), dpi=140)
    fig_s.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.9)
    ini_sm = np.isfinite(initial_slope)
    fin_sm = np.isfinite(final_slope)
    ax_s.plot(np.log10(initial_slope_radius[ini_sm]), initial_slope[ini_sm],
              color="k", ls=":", lw=2.3, label=f"initial (t={time_min:g})", zorder=3)
    ax_s.plot(np.log10(final_slope_radius[fin_sm]), final_slope[fin_sm],
              color="b", ls="-", lw=2.5, label=f"final (t={time_max:g})", zorder=2)
    draw_slope_references(ax_s)
    ax_s.set_xlim(axis_log_r_min, axis_log_r_max)
    ax_s.set_ylim(slope_y_min, slope_y_max)
    ax_s.set_xlabel(r"$\log_{10} r$", fontsize=22, labelpad=9)
    ax_s.set_ylabel(r"$\mathrm{d}\log\rho/\mathrm{d}\log r$", fontsize=22, labelpad=11)
    ax_s.legend(loc="upper left", fontsize=12, frameon=True, facecolor="white")
    style_axes(ax_s, 17)
    fig_s.suptitle(rf"$\gamma_i={gamma:g}$: initial vs final", fontsize=18, y=0.97)
    fig_s.savefig(slope_png)
    plt.close(fig_s)

    # ---- gamma(t) and gamma_out(t) evolution PNGs (one panel each) ---------
    at = np.asarray(alpha_time, dtype=float)
    ain = np.asarray(alpha_in_series, dtype=float)
    aout = np.asarray(alpha_out_series, dtype=float)

    gamma_png = output.with_name(f"{output.stem}_gamma_evolution.png")
    fig_g, ax_g = plt.subplots(figsize=(7.5, 4.8), dpi=140)
    fig_g.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.9)
    in_ok = np.isfinite(ain)
    # Convert signed slope s into the density exponent gamma=-s for the time-series panel.
    ax_g.plot(at[in_ok], -ain[in_ok], color="k", lw=1.8)
    # The -1.5 reference is labelled via the legend, not a floating tag, because
    # gamma(t) often sits right on -1.5 where an in-plot label is unreadable.
    ax_g.axhline(1.5, color="0.55", ls="--", lw=1.7, zorder=1, label=r"$\gamma_{\rm in}=1.5$")
    ax_g.set_xlabel("time", fontsize=18, labelpad=7)
    ax_g.set_ylabel(r"$\gamma_{\rm in}$", fontsize=20, labelpad=8)
    ax_g.legend(loc="upper right", fontsize=13, frameon=True, facecolor="white")
    style_axes(ax_g, 16)
    fig_g.suptitle(rf"$\gamma_i={gamma:g}$: inner index evolution", fontsize=18, y=0.97)
    fig_g.savefig(gamma_png)
    plt.close(fig_g)

    gammaout_png = output.with_name(f"{output.stem}_gammaout_evolution.png")
    fig_go, ax_go = plt.subplots(figsize=(7.5, 4.8), dpi=140)
    fig_go.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.9)
    out_ok = np.isfinite(aout)
    ax_go.plot(at[out_ok], -aout[out_ok], color="b", lw=1.8)
    ax_go.set_xlabel("time", fontsize=18, labelpad=7)
    ax_go.set_ylabel(r"$\gamma_{\rm out}$", fontsize=20, labelpad=8)
    style_axes(ax_go, 16)
    fig_go.suptitle(rf"$\gamma_i={gamma:g}$: $\gamma_{{\rm out}}$ evolution",
                    fontsize=18, y=0.97)
    fig_go.savefig(gammaout_png)
    plt.close(fig_go)

    # Export both sign conventions and every fit-control constant beside the movie.
    fit_path = output.with_name(f"{output.stem}_slopes.csv")
    np.savetxt(fit_path, np.column_stack((at, ain, aout, -ain, -aout)), delimiter=",",
               header="time,slope_in,slope_out,gamma_in,gamma_out", comments="")
    output.with_name(f"{output.stem}_fit_settings.json").write_text(json.dumps({
        "gamma_definition": "gamma=-dlog(rho)/dlog(r)",
        "inner_center_log10r": INNER_FIT_CENTER_LOG10R,
        "inner_half_width_log10r": FIT_HALF_WIDTH_LOG10R,
        "shoulder_neff_min": SHOULDER_NEFF_MIN,
        "shoulder_flat_curvature": SHOULDER_D2_FLAT,
        "shoulder_kink_curvature": SHOULDER_KINK_D2,
        "shoulder_half_width_log10r": SHOULDER_FIT_HALF_WIDTH_LOG10R,
        "outer_fit_interpretation": "empirical profile-feature fit; verify window and resolution",
    }, indent=2) + "\n")
    times_path = output.with_name(f"{output.stem}_frame_times.txt")
    times_path.write_text("\n".join(f"{index},{time:g}" for index, (time, _) in enumerate(profiles)) + "\n")
    for produced in (density_png, slope_png, gamma_png, gammaout_png):
        print(f"  wrote {produced}")
    return output


def parse_arguments() -> argparse.Namespace:
    """Define run paths and purely visual radial/slope controls.

    ``--run-dir`` may be given more than once; each is an output directory
    that contains ``profile_*.csv`` and ``metadata.json``.  gamma_i is read
    from that metadata, so no slope has to be registered ahead of time.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run output directory (repeatable). Must contain metadata.json.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=14.0)
    parser.add_argument("--stride", type=int, default=10,
                        help="Render every Nth profile (first and last always kept).")
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=900.0)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    parser.add_argument("--slope-y-min", type=float, default=-6.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    parser.add_argument("--inner-fit-center-log10r", type=float, default=INNER_FIT_CENTER_LOG10R)
    parser.add_argument("--inner-fit-half-width-log10r", type=float, default=FIT_HALF_WIDTH_LOG10R)
    parser.add_argument("--shoulder-neff-min", type=float, default=SHOULDER_NEFF_MIN)
    result = parser.parse_args()
    if (not np.isfinite(result.inner_fit_center_log10r)
            or not np.isfinite(result.inner_fit_half_width_log10r)
            or result.inner_fit_half_width_log10r <= 0
            or not np.isfinite(result.shoulder_neff_min) or result.shoulder_neff_min < 0
            or not np.isfinite(result.fps) or not 0 < result.fps <= 100
            or not 0 < result.r_min < result.r_max or result.stride < 1
            or not 0 <= result.density_floor_rel < 1
            or not result.slope_y_min < result.slope_y_max):
        parser.error("Require finite fit settings, positive radius range/stride/fps, and ordered slope limits")
    return result


# ---- Selected-run command-line dispatch -----------------------------------
def main() -> None:
    """Render each requested run, reading its slope from metadata.json."""
    global INNER_FIT_CENTER_LOG10R, FIT_HALF_WIDTH_LOG10R, SHOULDER_NEFF_MIN
    arguments = parse_arguments()
    INNER_FIT_CENTER_LOG10R = arguments.inner_fit_center_log10r
    FIT_HALF_WIDTH_LOG10R = arguments.inner_fit_half_width_log10r
    SHOULDER_NEFF_MIN = arguments.shoulder_neff_min
    for run_dir in arguments.run_dir:
        gamma = read_gamma(run_dir)
        ppc = read_ppc(run_dir)
        output = render_case(
            run_dir,
            arguments.output_directory,
            gamma_label(gamma),
            gamma,
            arguments.fps,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
            arguments.slope_y_min,
            arguments.slope_y_max,
            arguments.stride,
            ppc,
        )
        print(output)


if __name__ == "__main__":
    main()
