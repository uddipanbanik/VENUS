#!/usr/bin/env python3
# Compare VENUS and reference SphDSMC density, log slope and radial random variance in common benchmark units.
# Native reference profiles contain cumulative inner moments followed by independent shells. Recover nonoverlapping extensive moments before comparing radial resolution.
# Output movie and sidecars retain both actual clocks, conservative regrouping errors and any held endpoint states.

"""Render a mutually normalized Frank/VENUS SphDSMC comparison GIF.

The animation compares density, local logarithmic density slope, and radial
velocity dispersion at matched physical times.  Frank's cumulative inner
profiles are first converted back to non-overlapping spherical cells, then
both codes are conservatively grouped at the same effective radial resolution
so that different output binning is not mistaken for different physics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

# The laptop's older Matplotlib can crash while rebuilding its font cache.
# Reuse the established isolated cache used by the VENUS renderers.
if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(tempfile.gettempdir()) / "venus-matplotlib-cache"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    for _font_cache in (Path.home() / ".matplotlib").glob("fontlist*.json"):
        _target = _mpl_cache / _font_cache.name
        if not _target.exists():
            shutil.copy2(_font_cache, _target)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

# Keep the local plotting process single-threaded before NumPy/Matplotlib load.
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
from PIL import Image

from spherical_profile_rendering import (
    add_slope_reference_lines,
    fig_to_rgb,
    load_profiles,
    style_axes,
    unique_r_rows,
)
from render_density_slope_sigma2_gif import (
    AtomicStreamingGif,
    conservative_adaptive_rebin,
)


# Shared conversion from the simulation units to the benchmark's paper units.
# PARTICLES is the equal-mass Frank realization used to interpret enclosed
# particle counts as enclosed mass.
MU = 1.1252580933331495
TIME_GYR = 0.021151235023496799
PARTICLES = 1_000_000


def venus_snapshots(paths: list[Path]) -> dict[int, np.ndarray]:
    """Merge restart continuations into one complete VENUS step sequence."""
    by_step: dict[int, np.ndarray] = {}
    for path in paths:
        profiles = load_profiles(path)
        for step_float in np.unique(profiles["step"]):
            step = int(step_float)
            rows = unique_r_rows(profiles[profiles["step"] == step_float])
            if rows.size:
                # Later continuation files replace a duplicated launch
                # checkpoint while preserving all distinct solver steps.
                by_step[step] = rows
    return dict(sorted(by_step.items()))


def venus_target_outer_edges(rows: np.ndarray, particle_mass: float) -> np.ndarray:
    """Return the exact conservative adaptive-bin outer edges used by VENUS.

    Inner groups contain at least 256 represented particles whenever possible;
    outer cells remain unmerged.  The returned edges define the common radial
    resolution later imposed on Frank's independent profile format.
    """
    rows = unique_r_rows(rows)
    r = np.asarray(rows["r"], dtype=float)
    edges = np.empty(r.size + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = r[-1]
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    volumes = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    masses = np.maximum(np.asarray(rows["rho"], dtype=float), 0.0) * volumes
    # Use the same represented-particle target and inner radial cutoff that define the VENUS adaptive display bins.
    target_mass = 256.0 * particle_mass
    cutoff = int(np.searchsorted(r, 0.3, side="right"))
    groups: list[tuple[int, int]] = []
    begin = 0
    accumulated = 0.0
    for end in range(cutoff):
        accumulated += masses[end]
        if accumulated >= target_mass:
            groups.append((begin, end + 1))
            begin = end + 1
            accumulated = 0.0
    if begin < cutoff:
        if groups:
            groups[-1] = (groups[-1][0], cutoff)
        else:
            groups.append((begin, cutoff))
    groups.extend((index, index + 1) for index in range(cutoff, r.size))
    return np.asarray([edges[end] for _, end in groups], dtype=float)


def venus_curve(
    rows: np.ndarray,
    particle_mass: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Convert one VENUS profile to paper-normalized ``r, rho, sigma_r^2``."""
    merged = conservative_adaptive_rebin(rows, particle_mass, 256, 0.3)
    r = np.asarray(merged["r"], dtype=float)
    rho_code = np.asarray(merged["rho"], dtype=float)
    sigma2 = np.divide(
        merged["p"], rho_code, out=np.zeros_like(rho_code), where=rho_code > 0.0
    )
    return (
        (r, 4.0 * math.pi * rho_code, np.maximum(sigma2, 0.0)),
        venus_target_outer_edges(rows, particle_mass),
    )


def frank_snapshot(path: Path) -> tuple[int, float, np.ndarray]:
    """Read one complete NSphere PROFILE snapshot and its native time."""
    with path.open(encoding="ascii") as handle:
        header = handle.readline().split()
        if len(header) < 3:
            raise RuntimeError(f"Incomplete Frank profile header: {path}")
        step = int(float(header[0]))
        time_frank = float(header[1])
        rows: list[list[float]] = []
        for line in handle:
            fields = line.split()
            if len(fields) < 17:
                continue
            try:
                rows.append([float(field) for field in fields[:17]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No complete Frank profile rows: {path}")
    return step, time_frank, np.asarray(rows, dtype=float)


def frank_nonoverlapping_cells(rows: np.ndarray) -> dict[str, np.ndarray]:
    """Recover non-overlapping shell integrals from NSphere PROFILE output.

    NSphere's first 40 rows are cumulative 25-particle profiles through 1000
    particles.  Subsequent rows are independent shells.  Differencing the
    cumulative integrated moments recovers the inner non-overlapping cells
    without treating cumulative rows as independent mass.
    """
    cumulative_rows = 0
    for index in range(min(rows.shape[0], 40)):
        if abs(rows[index, 13] - 25.0 * (index + 1)) <= 0.5:
            cumulative_rows += 1
        else:
            break
    if cumulative_rows != 40:
        raise RuntimeError(
            f"Expected 40 cumulative inner NSphere rows, found {cumulative_rows}"
        )

    sigma2 = np.maximum(rows[:, 6], 0.0) ** 2
    mean_vr = rows[:, 12]
    beta = rows[:, 8]
    tangential2 = np.maximum(2.0 * sigma2 * (1.0 - beta), 0.0)

    cumulative_mass = rows[:cumulative_rows, 4]
    cumulative_radial_momentum = cumulative_mass * mean_vr[:cumulative_rows]
    # Restore mean-flow contributions before differencing cumulative second moments into disjoint inner shells.
    cumulative_radial_raw2 = cumulative_mass * (
        sigma2[:cumulative_rows] + mean_vr[:cumulative_rows] ** 2
    )
    cumulative_tangential2 = cumulative_mass * tangential2[:cumulative_rows]
    cumulative_radius1 = cumulative_mass * rows[:cumulative_rows, 3]
    zeros = np.zeros(1, dtype=float)

    # Difference integrated cumulative quantities so each inner particle mass and velocity moment contributes exactly once.
    inner_mass = np.diff(np.concatenate((zeros, cumulative_mass)))
    inner_momentum = np.diff(np.concatenate((zeros, cumulative_radial_momentum)))
    inner_radial_raw2 = np.diff(np.concatenate((zeros, cumulative_radial_raw2)))
    inner_tangential2 = np.diff(np.concatenate((zeros, cumulative_tangential2)))
    inner_radius1 = np.diff(np.concatenate((zeros, cumulative_radius1)))
    inner_outer = np.asarray(rows[:cumulative_rows, 2], dtype=float)
    inner_inner = np.concatenate((zeros, inner_outer[:-1]))
    inner_particles = np.full(cumulative_rows, 25.0)

    outer = rows[cumulative_rows:]
    outer_mass = np.asarray(outer[:, 4], dtype=float)
    outer_mean_vr = np.asarray(outer[:, 12], dtype=float)
    outer_sigma2 = np.maximum(np.asarray(outer[:, 6], dtype=float), 0.0) ** 2
    outer_beta = np.asarray(outer[:, 8], dtype=float)
    outer_tangential2_per_mass = np.maximum(
        2.0 * outer_sigma2 * (1.0 - outer_beta), 0.0
    )

    return {
        "inner": np.concatenate((inner_inner, np.asarray(outer[:, 1], dtype=float))),
        "outer": np.concatenate((inner_outer, np.asarray(outer[:, 2], dtype=float))),
        "mass": np.concatenate((inner_mass, outer_mass)),
        "radial_momentum": np.concatenate((inner_momentum, outer_mass * outer_mean_vr)),
        "radial_raw2": np.concatenate(
            (inner_radial_raw2, outer_mass * (outer_sigma2 + outer_mean_vr**2))
        ),
        "tangential2": np.concatenate(
            (inner_tangential2, outer_mass * outer_tangential2_per_mass)
        ),
        "radius1": np.concatenate((inner_radius1, outer_mass * outer[:, 3])),
        "particles": np.concatenate(
            (inner_particles, np.asarray(outer[:, 13], dtype=float))
        ),
    }


def _relative_partition_error(source: np.ndarray, grouped: np.ndarray) -> float:
    """Measure whether conservative regrouping preserves an integrated moment."""
    scale = max(float(np.sum(np.abs(source))), np.finfo(float).tiny)
    return abs(float(np.sum(grouped)) - float(np.sum(source))) / scale


def frank_curve(
    rows: np.ndarray,
    target_outer_edges: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Conservatively merge SphDSMC cells at VENUS's radial resolution.

    Shell mass, radial momentum, raw radial second moment, and tangential
    second moment are summed before density and velocity dispersion are
    reconstructed.  The returned audit demonstrates that comparison
    preprocessing has not changed any of those integrated physical moments.
    """
    cells = frank_nonoverlapping_cells(rows)
    representative = np.divide(
        cells["radius1"],
        cells["mass"],
        out=0.5 * (cells["inner"] + cells["outer"]),
        where=cells["mass"] > 0.0,
    )
    usable = (
        np.isfinite(representative)
        & np.isfinite(cells["mass"])
        & (cells["mass"] > 0.0)
        & (representative <= float(target_outer_edges[-1]))
    )
    cells = {name: values[usable] for name, values in cells.items()}
    representative = representative[usable]

    groups: list[tuple[int, int]] = []
    cursor = 0
    # Group reference shells toward the VENUS radial resolution using whole source cells and preserving their integrated moments.
    for target_outer in target_outer_edges:
        if cursor >= representative.size:
            break
        begin = cursor
        while cursor < representative.size and cells["outer"][cursor] <= target_outer * (1.0 + 1.0e-12):
            cursor += 1
        if cursor == begin and representative[cursor] <= target_outer:
            cursor += 1
        if cursor > begin:
            groups.append((begin, cursor))
    if cursor < representative.size:
        groups.append((cursor, representative.size))

    grouped: dict[str, list[float]] = {
        "mass": [],
        "radial_momentum": [],
        "radial_raw2": [],
        "tangential2": [],
        "radius1": [],
        "volume": [],
        "particles": [],
    }
    for begin, end in groups:
        grouped["mass"].append(float(np.sum(cells["mass"][begin:end])))
        grouped["radial_momentum"].append(
            float(np.sum(cells["radial_momentum"][begin:end]))
        )
        grouped["radial_raw2"].append(float(np.sum(cells["radial_raw2"][begin:end])))
        grouped["tangential2"].append(float(np.sum(cells["tangential2"][begin:end])))
        grouped["radius1"].append(float(np.sum(cells["radius1"][begin:end])))
        grouped["particles"].append(float(np.sum(cells["particles"][begin:end])))
        grouped["volume"].append(
            float(
                np.sum(
                    (4.0 * math.pi / 3.0)
                    * (cells["outer"][begin:end] ** 3 - cells["inner"][begin:end] ** 3)
                )
            )
        )
    arrays = {name: np.asarray(values, dtype=float) for name, values in grouped.items()}
    mean_vr = np.divide(
        arrays["radial_momentum"], arrays["mass"], out=np.zeros_like(arrays["mass"]), where=arrays["mass"] > 0.0
    )
    # Recover random variance only after summing raw moments and subtracting the merged mean velocity squared.
    sigma2 = np.divide(
        arrays["radial_raw2"], arrays["mass"], out=np.zeros_like(arrays["mass"]), where=arrays["mass"] > 0.0
    ) - mean_vr**2
    r = np.divide(
        arrays["radius1"], arrays["mass"], out=np.zeros_like(arrays["mass"]), where=arrays["mass"] > 0.0
    )
    rho = np.divide(
        arrays["mass"], arrays["volume"], out=np.zeros_like(arrays["mass"]), where=arrays["volume"] > 0.0
    )
    audit = {
        "mass_partition_relative_error": _relative_partition_error(cells["mass"], arrays["mass"]),
        "radial_momentum_partition_relative_error": _relative_partition_error(
            cells["radial_momentum"], arrays["radial_momentum"]
        ),
        "radial_raw2_partition_relative_error": _relative_partition_error(
            cells["radial_raw2"], arrays["radial_raw2"]
        ),
        "tangential2_partition_relative_error": _relative_partition_error(
            cells["tangential2"], arrays["tangential2"]
        ),
        "minimum_group_particles": float(np.min(arrays["particles"])),
        "median_group_particles": float(np.median(arrays["particles"])),
        "maximum_group_particles": float(np.max(arrays["particles"])),
        "source_cells": int(representative.size),
        "display_groups": int(r.size),
    }
    return (
        (r, 4.0 * math.pi * MU * rho, MU * np.maximum(sigma2, 0.0)),
        audit,
    )


def matched_bandwidth_log_slope(
    r: np.ndarray,
    rho: np.ndarray,
    reference_r: np.ndarray,
    window_points: int = 9,
    poly_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a derivative with the same physical radial span for both curves."""
    mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > 0.0)
    r_fit = np.asarray(r[mask], dtype=float)
    rho_fit = np.asarray(rho[mask], dtype=float)
    order = np.argsort(r_fit)
    r_fit, rho_fit = r_fit[order], rho_fit[order]
    x, y = np.log10(r_fit), np.log10(rho_fit)
    reference_x = np.sort(np.log10(reference_r[np.isfinite(reference_r) & (reference_r > 0.0)]))
    slopes = np.full(x.size, np.nan, dtype=float)
    count = min(max(int(window_points), 5), reference_x.size)
    for index, center in enumerate(x):
        # Use the reference mesh to define a common physical log-radius fit span for both independently sampled curves.
        nearest = int(np.argmin(np.abs(reference_x - center)))
        begin = max(0, nearest - count // 2)
        end = min(reference_x.size, begin + count)
        begin = max(0, end - count)
        lo, hi = reference_x[begin], reference_x[end - 1]
        local = np.flatnonzero((x >= lo) & (x <= hi))
        if local.size < 5:
            local = np.argsort(np.abs(x - center))[: min(5, x.size)]
            local = np.sort(local)
        if local.size < 3:
            continue
        degree = min(poly_order, local.size - 1)
        coefficients = np.polyfit(x[local] - center, y[local], degree)
        slopes[index] = float(coefficients[-2])
    return r_fit, slopes


def curve_rms(
    reference_r: np.ndarray,
    reference_y: np.ndarray,
    comparison_r: np.ndarray,
    comparison_y: np.ndarray,
    rlo: float,
    rhi: float,
) -> tuple[float, float, int]:
    """Compare two positive profiles in logarithmic amplitude over one radius."""
    mask = (
        np.isfinite(reference_r)
        & np.isfinite(reference_y)
        & (reference_r >= rlo)
        & (reference_r <= rhi)
        & (reference_y > 0.0)
    )
    rr = reference_r[mask]
    yy = reference_y[mask]
    valid_comparison = (
        np.isfinite(comparison_r) & np.isfinite(comparison_y) & (comparison_y > 0.0)
    )
    cr = comparison_r[valid_comparison]
    cy = comparison_y[valid_comparison]
    inside = (rr >= np.min(cr)) & (rr <= np.max(cr))
    rr, yy = rr[inside], yy[inside]
    # Measure logarithmic amplitude differences only inside the shared positive radial support.
    interp = np.interp(np.log10(rr), np.log10(cr), np.log10(cy))
    delta = np.log10(yy) - interp
    return float(np.sqrt(np.mean(delta**2))), float(np.mean(delta)), int(delta.size)


def text_layout_qa(fig: plt.Figure, axes: list[plt.Axes]) -> dict[str, object]:
    """Reject clipped labels or overlapping annotations before publication."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    text_boxes = []
    for artist in fig.findobj(match=lambda item: hasattr(item, "get_text")):
        try:
            if not artist.get_visible() or not artist.get_text():
                continue
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        if np.all(np.isfinite(box.extents)):
            text_boxes.append((artist.get_text(), box))
    clipped = [
        {"text": text, "bbox_pixels": [float(v) for v in box.extents]}
        for text, box in text_boxes
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height + 0.5
    ]
    tight = [axis.get_tightbbox(renderer) for axis in axes]
    horizontal_gaps = [float(tight[i + 1].x0 - tight[i].x1) for i in range(len(tight) - 1)]
    ic_box = fig._ic_annotation.get_window_extent(renderer)
    title_box = fig._time_title.get_window_extent(renderer)
    ic_title_overlap = bool(ic_box.overlaps(title_box))
    ic_axes_overlap = bool(any(ic_box.overlaps(box) for box in tight))
    result = {
        "canvas_pixels": [int(width), int(height)],
        "clipped_text_boxes": clipped,
        "tight_bbox_horizontal_gaps_pixels": horizontal_gaps,
        "axes_plot_heights_pixels": [float(axis.bbox.height) for axis in axes],
        "axes_plot_widths_pixels": [float(axis.bbox.width) for axis in axes],
        "ic_annotation_bbox_pixels": [float(v) for v in ic_box.extents],
        "time_title_bbox_pixels": [float(v) for v in title_box.extents],
        "ic_annotation_overlaps_time_title": ic_title_overlap,
        "ic_annotation_overlaps_axes": ic_axes_overlap,
    }
    if clipped or min(horizontal_gaps) < 5.0 or ic_title_overlap or ic_axes_overlap:
        raise RuntimeError(f"Layout QA failed: {result}")
    return result


def plot_frame(
    venus: tuple[np.ndarray, np.ndarray, np.ndarray],
    frank: tuple[np.ndarray, np.ndarray, np.ndarray],
    initial: tuple[np.ndarray, np.ndarray, np.ndarray],
    time_gyr: float,
    sigma_limits: tuple[float, float],
    setup_annotation: str,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw one matched-time density, slope, and dispersion comparison."""
    vr, vrho, vs2 = venus
    fr, frho, fs2 = frank
    initial_r, initial_rho, initial_sigma2 = initial
    fig, axes_array = plt.subplots(1, 3, figsize=(15.2, 4.35), dpi=135)
    axes = list(axes_array)
    fig.subplots_adjust(left=0.086, right=0.985, bottom=0.17, top=0.84, wspace=0.39)
    fig._ic_annotation = fig.text(
        0.087,
        0.972,
        setup_annotation,
        ha="left",
        va="top",
        fontsize=11.5,
        linespacing=1.22,
    )
    xlimits = (math.log10(3.0e-3), math.log10(1.0e3))
    slope_reference_r = np.sort(
        vr[np.isfinite(vr) & (vr > 0.0)]
    )
    # A local nine-point derivative is not well conditioned in the outermost
    # half-window, where the fit becomes one-sided at VENUS's profile boundary.
    # Do not display those edge-conditioned derivative estimates as physics.
    slope_outer_limit = float(slope_reference_r[-5])

    initial_density_mask = (
        np.isfinite(initial_r)
        & np.isfinite(initial_rho)
        & (initial_r >= 3.0e-3)
        & (initial_r <= 1.0e3)
        & (initial_rho > 0.0)
    )
    axes[0].plot(
        np.log10(initial_r[initial_density_mask]),
        np.log10(initial_rho[initial_density_mask]),
        color="black",
        ls=":",
        lw=1.8,
        label="Initial",
        zorder=1,
    )
    initial_slope_r, initial_slope = matched_bandwidth_log_slope(
        initial_r, initial_rho, initial_r, 9, 2
    )
    initial_slope_mask = (
        np.isfinite(initial_slope_r)
        & np.isfinite(initial_slope)
        & (initial_slope_r >= 3.0e-3)
        & (initial_slope_r <= slope_outer_limit)
    )
    axes[1].plot(
        np.log10(initial_slope_r[initial_slope_mask]),
        initial_slope[initial_slope_mask],
        color="black",
        ls=":",
        lw=1.8,
        zorder=1,
    )
    initial_sigma_mask = (
        np.isfinite(initial_r)
        & np.isfinite(initial_sigma2)
        & (initial_r >= 3.0e-3)
        & (initial_r <= 1.0e3)
        & (initial_sigma2 > 0.0)
    )
    axes[2].plot(
        np.log10(initial_r[initial_sigma_mask]),
        np.log10(initial_sigma2[initial_sigma_mask]),
        color="black",
        ls=":",
        lw=1.8,
        zorder=1,
    )

    for r, rho, color, style, label in (
        (vr, vrho, "#1555d6", "-", "VENUS"),
        (fr, frho, "#d62728", "--", "SphDSMC"),
    ):
        mask = np.isfinite(r) & np.isfinite(rho) & (r >= 3.0e-3) & (r <= 1.0e3) & (rho > 0.0)
        axes[0].plot(
            np.log10(r[mask]),
            np.log10(rho[mask]),
            color=color,
            ls=style,
            lw=2.0,
            label=label,
            zorder=2,
        )
        slope_r, slope = matched_bandwidth_log_slope(r, rho, vr, 9, 2)
        slope_mask = (
            np.isfinite(slope_r)
            & np.isfinite(slope)
            & (slope_r >= 3.0e-3)
            & (slope_r <= slope_outer_limit)
        )
        axes[1].plot(
            np.log10(slope_r[slope_mask]),
            slope[slope_mask],
            color=color,
            ls=style,
            lw=2.0,
            zorder=2,
        )

    for r, sigma2, color, style in (
        (vr, vs2, "#1555d6", "-"),
        (fr, fs2, "#d62728", "--"),
    ):
        mask = (
            np.isfinite(r) & np.isfinite(sigma2) & (r >= 3.0e-3) & (r <= 1.0e3) & (sigma2 > 0.0)
        )
        axes[2].plot(
            np.log10(r[mask]),
            np.log10(sigma2[mask]),
            color=color,
            ls=style,
            lw=2.0,
            zorder=2,
        )

    axes[0].set(xlim=xlimits, ylim=(-10.0, 6.0), xlabel=r"$\log_{10}(r/r_s)$", ylabel=r"$\log_{10}(\rho/\rho_s)$")
    axes[1].set(xlim=xlimits, ylim=(-6.0, 2.0), xlabel=r"$\log_{10}(r/r_s)$", ylabel=r"$\mathrm{d}\log\rho/\mathrm{d}\log r$")
    axes[1].axhline(0.0, color="0.6", lw=1.0, ls=":")
    add_slope_reference_lines(axes[1], *xlimits, -6.0, 2.0)
    axes[2].set(
        xlim=xlimits,
        ylim=sigma_limits,
        xlabel=r"$\log_{10}(r/r_s)$",
        ylabel=r"$\log_{10}(\sigma_r^2/v_s^2)$",
    )
    for axis in axes:
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(16)
        axis.xaxis.labelpad = 7
        axis.yaxis.labelpad = 8
        style_axes(axis)
        axis.get_xaxis().get_offset_text().set_visible(False)
        axis.get_yaxis().get_offset_text().set_visible(False)
    axes[0].legend(loc="lower left", fontsize=12.5, frameon=False, handlelength=3.0)
    fig._time_title = fig.suptitle(
        rf"$t={time_gyr:.3f}\,\mathrm{{Gyr}}$", fontsize=16, y=0.97
    )
    return fig, axes


def main() -> None:
    """Align Frank/VENUS snapshots, render matched frames, and write QA data."""
    # Section 1: validate inputs and build the complete continuation timelines.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venus-profile", type=Path, action="append", required=True)
    parser.add_argument("--venus-metadata", type=Path, required=True)
    parser.add_argument("--frank-profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument(
        "--freeze-after-end",
        action="store_true",
        help=(
            "continue to the later run's final snapshot while holding the "
            "earlier run at its final complete profile"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument(
        "--max-canonical-step-gap",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frame-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--frame-stop", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frank-time-to-gyr", type=float, default=TIME_GYR)
    parser.add_argument(
        "--venus-time-to-gyr", type=float, default=math.sqrt(MU) * TIME_GYR
    )
    parser.add_argument(
        "--setup-annotation",
        default=(
            r"IC: $(\alpha,\beta,\gamma)=(4,4,0.1),\ c=20,\ \Delta_{\rm vir}=200$"
            "\n"
            r"$M_{\rm host}=10^{12}\,M_\odot/h,\ N=10^6,\ \sigma_m=25\ {\rm cm^2\,g^{-1}}$"
        ),
    )
    parser.add_argument("--qa-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.setup_annotation = args.setup_annotation.replace(r"\n", "\n")

    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")

    def evenly_sample(sequence: list, limit: int) -> list:
        """Keep all entries or an endpoint-preserving uniform-index subset."""
        if len(sequence) <= limit:
            return sequence
        indices = np.linspace(0, len(sequence) - 1, limit, dtype=int)
        indices[-1] = len(sequence) - 1
        return [sequence[int(index)] for index in indices]

    with args.venus_metadata.open(encoding="ascii") as handle:
        metadata = json.load(handle)
    particle_mass = float(metadata["bp_dsmc_total_mass"]) / int(metadata["bp_dsmc_particles"])
    venus_by_step = venus_snapshots(args.venus_profile)
    venus_times = np.array(
        [float(rows["time"][0]) * args.venus_time_to_gyr for rows in venus_by_step.values()]
    )
    venus_steps = np.array(list(venus_by_step), dtype=int)
    venus_step_gaps = np.diff(venus_steps)
    # Check the expected VENUS saved-step coverage before selecting a shorter display sequence.
    missing_profile_ranges = [
        (int(venus_steps[index]), int(venus_steps[index + 1]))
        for index in np.flatnonzero(
            venus_step_gaps > args.max_canonical_step_gap
        )
    ]
    if missing_profile_ranges:
        preview = ", ".join(
            f"{lower}->{upper}" for lower, upper in missing_profile_ranges[:8]
        )
        raise RuntimeError(
            "Canonical VENUS profile sequence has missing save-every-100 "
            f"snapshots: {preview}"
        )

    # Section 2: pair Frank output with the nearest VENUS physical time.  If
    # requested, extend the movie by visibly freezing whichever run ended first.
    all_frank_entries = [
        frank_snapshot(path) for path in sorted(args.frank_profile_dir.glob("prof.*"))
    ]
    venus_endpoint = float(venus_times[-1])
    frank_endpoint = float(all_frank_entries[-1][1] * args.frank_time_to_gyr)
    common_endpoint = min(venus_endpoint, frank_endpoint)
    common_frank_entries = [
        entry
        for entry in all_frank_entries
        if entry[1] * args.frank_time_to_gyr <= common_endpoint + 1.0e-12
    ]
    if not common_frank_entries:
        raise RuntimeError("No common-time Frank/VENUS profile snapshots")

    has_extension = args.freeze_after_end and not math.isclose(
        venus_endpoint, frank_endpoint, rel_tol=0.0, abs_tol=1.0e-12
    )
    common_limit = args.max_frames - 1 if has_extension else args.max_frames
    frank_entries = evenly_sample(common_frank_entries, common_limit)

    frame_data = []
    mismatches = []
    venus_snapshot_times = []
    frank_snapshot_times = []
    venus_frozen_flags = []
    frank_frozen_flags = []
    for frank_step, frank_time, frank_rows in frank_entries:
        physical_time = frank_time * args.frank_time_to_gyr
        # Pair a reference snapshot with the nearest actual VENUS clock and save their signed mismatch.
        venus_index = int(np.argmin(np.abs(venus_times - physical_time)))
        venus_step = int(venus_steps[venus_index])
        mismatch = float(venus_times[venus_index] - physical_time)
        mismatches.append(mismatch)
        venus_snapshot_times.append(float(venus_times[venus_index]))
        frank_snapshot_times.append(float(physical_time))
        venus_frozen_flags.append(False)
        frank_frozen_flags.append(False)
        venus, target_outer_edges = venus_curve(
            venus_by_step[venus_step], particle_mass
        )
        sphdsmc, conservation_audit = frank_curve(
            frank_rows, target_outer_edges
        )
        frame_data.append(
            (
                physical_time,
                venus_step,
                frank_step,
                venus,
                sphdsmc,
                conservation_audit,
            )
        )

    frozen_series = None
    frozen_after_gyr = None
    # When explicitly requested, extend the display by holding the completed reference state while the later run continues.
    if args.freeze_after_end and venus_endpoint > frank_endpoint + 1.0e-12:
        frozen_series = "Frank"
        frozen_after_gyr = frank_endpoint
        extension_indices = [
            index
            for index, time_gyr in enumerate(venus_times)
            if time_gyr > venus_snapshot_times[-1] + 1.0e-12
        ]
        available = args.max_frames - len(frame_data)
        extension_indices = evenly_sample(extension_indices, available)
        frozen_frank = frame_data[-1][4]
        frozen_audit = frame_data[-1][5]
        frozen_frank_step = all_frank_entries[-1][0]
        for venus_index in extension_indices:
            physical_time = float(venus_times[venus_index])
            venus_step = int(venus_steps[venus_index])
            venus, _ = venus_curve(venus_by_step[venus_step], particle_mass)
            mismatch = physical_time - frank_endpoint
            mismatches.append(mismatch)
            venus_snapshot_times.append(physical_time)
            frank_snapshot_times.append(frank_endpoint)
            venus_frozen_flags.append(False)
            frank_frozen_flags.append(True)
            frame_data.append(
                (
                    physical_time,
                    venus_step,
                    frozen_frank_step,
                    venus,
                    frozen_frank,
                    frozen_audit,
                )
            )
    # The reciprocal extension holds the completed VENUS state and keeps its true endpoint time in the sidecar.
    elif args.freeze_after_end and frank_endpoint > venus_endpoint + 1.0e-12:
        frozen_series = "VENUS"
        frozen_after_gyr = venus_endpoint
        frozen_venus_step = int(venus_steps[-1])
        frozen_venus, frozen_target_outer_edges = venus_curve(
            venus_by_step[frozen_venus_step], particle_mass
        )
        extension_entries = [
            entry
            for entry in all_frank_entries
            if entry[1] * args.frank_time_to_gyr > venus_endpoint + 1.0e-12
        ]
        available = args.max_frames - len(frame_data)
        extension_entries = evenly_sample(extension_entries, available)
        for frank_step, frank_time, frank_rows in extension_entries:
            physical_time = float(frank_time * args.frank_time_to_gyr)
            sphdsmc, conservation_audit = frank_curve(
                frank_rows, frozen_target_outer_edges
            )
            mismatch = venus_endpoint - physical_time
            mismatches.append(mismatch)
            venus_snapshot_times.append(venus_endpoint)
            frank_snapshot_times.append(physical_time)
            venus_frozen_flags.append(True)
            frank_frozen_flags.append(False)
            frame_data.append(
                (
                    physical_time,
                    frozen_venus_step,
                    frank_step,
                    frozen_venus,
                    sphdsmc,
                    conservation_audit,
                )
            )

    v0, f0 = frame_data[0][3], frame_data[0][4]
    density_rms, density_bias, density_points = curve_rms(*v0[:2], *f0[:2], 0.03, 30.0)
    sigma_rms, sigma_bias, sigma_points = curve_rms(v0[0], v0[2], f0[0], f0[2], 0.03, 30.0)
    # Check the initial density and variance normalization before interpreting subsequent cross-code evolution.
    if density_rms > 0.12 or sigma_rms > 0.12:
        raise RuntimeError(
            f"t=0 unit/parser preflight failed: density RMS={density_rms}, sigma2 RMS={sigma_rms}"
        )

    sigma_logs = []
    for _, _, _, venus, frank, _ in frame_data:
        for r, sigma2 in ((venus[0], venus[2]), (frank[0], frank[2])):
            mask = np.isfinite(r) & np.isfinite(sigma2) & (r >= 3.0e-3) & (r <= 1.0e3) & (sigma2 > 0.0)
            sigma_logs.append(np.log10(sigma2[mask]))
    sigma_all = np.concatenate(sigma_logs)
    sigma_lower = float(np.nanpercentile(sigma_all, 0.5) - 0.15)
    sigma_upper = float(np.nanpercentile(sigma_all, 99.5) + 0.15)
    # Snap outward to half-dex limits.  This keeps every automatically chosen
    # major tick inside the axes; a raw percentile limit just below an integer
    # can otherwise leave that integer tick label outside the canvas.
    sigma_limits = (
        math.floor(2.0 * sigma_lower) / 2.0,
        math.ceil(2.0 * sigma_upper) / 2.0,
    )

    full_frame_count = len(frame_data)
    initial_curve = frame_data[0][3]
    frame_stop = full_frame_count if args.frame_stop is None else args.frame_stop
    if not (0 <= args.frame_start < frame_stop <= full_frame_count):
        raise ValueError(
            f"Invalid frame slice [{args.frame_start}:{frame_stop}] for "
            f"{full_frame_count} frames"
        )
    frame_data = frame_data[args.frame_start:frame_stop]

    args.qa_dir.mkdir(parents=True, exist_ok=True)
    selected_qa = {0: "first", len(frame_data) // 2: "middle", len(frame_data) - 1: "final"}
    layout_results: dict[str, object] = {}
    if args.qa_only:
        for index, label in selected_qa.items():
            time_gyr, _, _, venus, frank, _ = frame_data[index]
            fig, axes = plot_frame(
                venus, frank, initial_curve, time_gyr, sigma_limits, args.setup_annotation
            )
            layout_results[label] = text_layout_qa(fig, axes)
            Image.fromarray(fig_to_rgb(fig)).save(args.qa_dir / f"{label}_frame.png")
            plt.close(fig)
        print(
            json.dumps(
                {
                    "full_timeline_frame_count": full_frame_count,
                    "frame_slice": [args.frame_start, frame_stop],
                    "layout_qa": layout_results,
                },
                indent=2,
            )
        )
        return

    writer = AtomicStreamingGif(args.output, args.fps)
    try:
        for index, (time_gyr, venus_step, frank_step, venus, frank, _) in enumerate(frame_data):
            fig, axes = plot_frame(
                venus, frank, initial_curve, time_gyr, sigma_limits, args.setup_annotation
            )
            if index in selected_qa:
                layout_results[selected_qa[index]] = text_layout_qa(fig, axes)
            frame = fig_to_rgb(fig)
            writer.append(frame)
            if index in selected_qa:
                Image.fromarray(frame).save(args.qa_dir / f"{selected_qa[index]}_frame.png")
            plt.close(fig)
            del axes, fig, frame
            if index % 5 == 4:
                gc.collect()
    except BaseException:
        writer.abort()
        raise
    writer.close()

    # Keep timing mismatches, held-state flags, unit conversions and moment-conservation errors alongside the rendered comparison.
    sidecar = {
        "output": str(args.output),
        "frame_count": len(frame_data),
        "full_timeline_frame_count": full_frame_count,
        "frame_slice": [args.frame_start, frame_stop],
        "duration_ms": int(round(1000.0 / args.fps)),
        "fps": args.fps,
        "loop": 0,
        "disposal": 2,
        "first_time_gyr": frame_data[0][0],
        "final_time_gyr": frame_data[-1][0],
        "timeline_mode": (
            "freeze_earlier_run_after_final_snapshot"
            if args.freeze_after_end
            else "latest_common_time"
        ),
        "venus_final_snapshot_time_gyr": venus_endpoint,
        "frank_final_snapshot_time_gyr": frank_endpoint,
        "frozen_series": frozen_series,
        "frozen_after_time_gyr": frozen_after_gyr,
        "venus_snapshot_times_gyr": venus_snapshot_times,
        "frank_snapshot_times_gyr": frank_snapshot_times,
        "venus_frozen_frames": int(sum(venus_frozen_flags)),
        "frank_frozen_frames": int(sum(frank_frozen_flags)),
        "maximum_canonical_venus_step_gap": int(np.max(venus_step_gaps)),
        "maximum_adjacent_rendered_venus_time_gap_gyr": float(
            np.max(np.diff(venus_snapshot_times))
        ),
        "maximum_absolute_time_mismatch_gyr": max(abs(value) for value in mismatches),
        "time_mismatches_gyr": mismatches,
        "t0_density_log10_rms_dex": density_rms,
        "t0_density_log10_bias_venus_minus_frank_dex": density_bias,
        "t0_density_points": density_points,
        "t0_sigma2_log10_rms_dex": sigma_rms,
        "t0_sigma2_log10_bias_venus_minus_frank_dex": sigma_bias,
        "t0_sigma2_points": sigma_points,
        "sigma_log10_limits": sigma_limits,
        "sphdsmc_conservative_rebin": {
            "method": (
                "difference cumulative inner integrated moments; preserve independent "
                "outer shell moments; merge at VENUS adaptive radial resolution"
            ),
            "maximum_mass_partition_relative_error": max(
                frame[5]["mass_partition_relative_error"] for frame in frame_data
            ),
            "maximum_radial_momentum_partition_relative_error": max(
                frame[5]["radial_momentum_partition_relative_error"]
                for frame in frame_data
            ),
            "maximum_radial_raw2_partition_relative_error": max(
                frame[5]["radial_raw2_partition_relative_error"]
                for frame in frame_data
            ),
            "maximum_tangential2_partition_relative_error": max(
                frame[5]["tangential2_partition_relative_error"]
                for frame in frame_data
            ),
            "display_groups_first_middle_final": [
                frame_data[index][5]["display_groups"]
                for index in (0, len(frame_data) // 2, len(frame_data) - 1)
            ],
            "median_group_particles_first_middle_final": [
                frame_data[index][5]["median_group_particles"]
                for index in (0, len(frame_data) // 2, len(frame_data) - 1)
            ],
        },
        "slope_estimator": (
            "quadratic local derivative using the same VENUS nine-point physical "
            "log-radius span for both curves; outermost half-window omitted because "
            "it lacks two-sided support at the VENUS r_max profile boundary"
        ),
        "initial_reference": (
            "t=0 VENUS profile in common paper units; black dotted in all panels"
        ),
        "layout_qa": layout_results,
        "unit_conversions": {
            "mu": MU,
            "venus_density_multiplier": 4.0 * math.pi,
            "frank_density_multiplier": 4.0 * math.pi * MU,
            "venus_sigma2_multiplier": 1.0,
            "frank_sigma2_multiplier": MU,
            "venus_time_to_gyr": args.venus_time_to_gyr,
            "frank_time_to_gyr": args.frank_time_to_gyr,
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
