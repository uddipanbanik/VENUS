#!/usr/bin/env python3
# Plot radial fields, conserved-mass histories and projected phase-space output from the general initial-value solvers.
# Named CSV inputs retain their solver code units. An adaptive momentum grid belongs to its own saved time slice.
# Outputs include profile snapshots, compactness, fitted density slopes, inward/outward fractions and optional actual-frame GIFs.

"""Plot data written by the general Einstein--Vlasov IVP solvers.

``profiles.csv`` is a named table grouped by ``time`` and supplies ``r``,
``rho``, ``p``, ``m``, metric fields ``mu``/``lambda``, and compactness gap
``gap``.  ``history.csv`` supplies time-series fields including
``matter_mass`` and ``min_gap``.  Phase-space files contain radius ``r``,
radial momentum ``w``, and projected distribution ``g``; appended snapshots
may use a different momentum grid at every saved time after adaptive remaps.

The script derives compactness ``2m/r``, smoothed logarithmic density slopes,
radial power-law fits, and weighted reflection diagnostics.  Static figures
sample representative saved times, while GIFs preserve sequence endpoints and
use limits computed across all displayed frames.  Floors and smoothing are
visualization operations only and do not modify solver output.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(tempfile.gettempdir()) / "venus-matplotlib-cache"
    try:
        _mpl_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        _mpl_cache = Path.home() / ".cache" / "venus-matplotlib"
        _mpl_cache.mkdir(parents=True, exist_ok=True)
    for _font_cache in (Path.home() / ".matplotlib").glob("fontlist*.json"):
        _target = _mpl_cache / _font_cache.name
        if not _target.exists():
            shutil.copy2(_font_cache, _target)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# Section: argparse compatibility.
# Keep --flag/--no-flag behavior on older Python versions used on clusters.
if hasattr(argparse, "BooleanOptionalAction"):
    BooleanOptionalAction = argparse.BooleanOptionalAction
else:
    class BooleanOptionalAction(argparse.Action):
        """Backport ``--flag``/``--no-flag`` parsing for older Python."""

        def __init__(self, option_strings, dest, default=None, **kwargs):
            """Register positive and automatically generated negative flags."""
            strings = []
            for option in option_strings:
                strings.append(option)
                if option.startswith("--"):
                    strings.append("--no-" + option[2:])
            super().__init__(strings, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            """Store whether the positive rather than negative flag was used."""
            setattr(namespace, self.dest, not (option_string or "").startswith("--no-"))


TIME_LABEL = r"$t$"
PHASE_COLOR_LABEL = r"$\log_{10}\!\left(\int dL\,L f\right)$"


# Section: CSV readers.
# These functions read solver output files without assuming that adaptive-w
# snapshots share one global momentum grid.

def load_named_csv(path: Path) -> np.ndarray:
    """Load a named CSV while preserving one-row files as one-dimensional arrays."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def select_times(times: np.ndarray, n_snapshots: int) -> np.ndarray:
    """Return all saved times or an endpoint-preserving uniform-index subset."""
    unique = np.unique(times)
    if unique.size <= n_snapshots:
        return unique
    indices = np.unique(np.linspace(0, unique.size - 1, n_snapshots, dtype=int))
    return unique[indices]


def load_phase(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a complete named ``r,w,g`` table onto its Cartesian phase grid."""
    data = load_named_csv(path)
    r = np.unique(data["r"])
    w = np.unique(data["w"])
    g = data["g"].reshape(r.size, w.size)
    return r, w, g


def load_phase_snapshots(path: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load complete phase snapshots, allowing r/w grids to vary with time.

    Adaptive momentum remaps keep the same number of saved w indices but change
    their coordinates.  A global ``np.unique(w)`` would therefore create a fake
    oversized grid and reshape data incorrectly, so each time slice carries its
    own r and w coordinate arrays.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, invalid_raise=False)
    raw = np.atleast_2d(raw)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if raw.size == 0:
        return np.array([]), [], [], []

    times: list[float] = []
    r_frames: list[np.ndarray] = []
    w_frames: list[np.ndarray] = []
    snapshots: list[np.ndarray] = []
    for t in np.unique(raw[:, 0]):
        rows = raw[raw[:, 0] == t]
        r_indices = np.unique(rows[:, 2].astype(int))
        w_indices = np.unique(rows[:, 3].astype(int))
        # A phase slice must contain the full radial-by-momentum index product before it can be reshaped.
        n_cells = r_indices.size * w_indices.size
        if rows.shape[0] != n_cells:
            # The solver may be appending this frame right now; skip partial rows.
            continue
        # Recover storage order from explicit indices, then construct physical coordinate arrays separately for each adaptive slice.
        order = np.lexsort((rows[:, 3], rows[:, 2]))
        rows = rows[order]
        r_by_index = {int(i): float(rows[rows[:, 2].astype(int) == int(i)][0, 4]) for i in r_indices}
        w_by_index = {int(j): float(rows[rows[:, 3].astype(int) == int(j)][0, 5]) for j in w_indices}
        r = np.array([r_by_index[int(i)] for i in r_indices], dtype=float)
        w = np.array([w_by_index[int(j)] for j in w_indices], dtype=float)
        snapshots.append(rows[:, 6].reshape(r.size, w.size))
        times.append(float(t))
        r_frames.append(r)
        w_frames.append(w)
    return np.asarray(times, dtype=float), r_frames, w_frames, snapshots


# Section: plotting utilities.
# Shared grid-edge, style, limits, parser, and slope-fit helpers used by the
# concrete plot and animation functions below.

def trap_weights(x: np.ndarray) -> np.ndarray:
    """Return one-dimensional trapezoidal quadrature weights at grid centers."""
    x = np.ravel(np.asarray(x, dtype=float))
    if x.size < 2:
        return np.ones_like(x)
    weights = np.empty_like(x)
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    return weights


def cell_edges_from_centers(x: np.ndarray) -> np.ndarray:
    """Return cell edges for a monotone, possibly nonuniform center grid."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Need at least two grid centers for cell edges.")
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    if np.any(x > 0.0):
        edges[0] = max(edges[0], np.nextafter(0.0, 1.0))
    return edges


def active_phase_limits(
    r: np.ndarray,
    w: np.ndarray,
    snapshots: list[np.ndarray],
    floor_fraction: float = 1.0e-8,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find common momentum/radius limits from fixed-grid phase support."""
    if not snapshots:
        return (float(w[0]), float(w[-1])), (float(r[0]), float(r[-1]))
    vmax = max(float(np.nanmax(g)) for g in snapshots)
    threshold = max(vmax * floor_fraction, np.finfo(float).tiny)
    r_active = np.zeros(r.size, dtype=bool)
    w_active = np.zeros(w.size, dtype=bool)
    for g in snapshots:
        active = np.isfinite(g) & (g > threshold)
        r_active |= np.any(active, axis=1)
        w_active |= np.any(active, axis=0)
    if not np.any(r_active):
        r_limits = (float(r[0]), float(r[-1]))
    else:
        active_r = r[r_active]
        lo, hi = float(active_r[0]), float(active_r[-1])
        pad = 0.12 * max(hi - lo, np.min(np.diff(r)))
        r_limits = (max(float(r[0]), lo - pad), min(float(r[-1]), hi + pad))
    if not np.any(w_active):
        w_limits = (float(w[0]), float(w[-1]))
    else:
        active_w = w[w_active]
        lo, hi = float(active_w[0]), float(active_w[-1])
        pad = 0.10 * max(hi - lo, np.min(np.diff(w)))
        w_limits = (max(float(w[0]), lo - pad), min(float(w[-1]), hi + pad))
    return w_limits, r_limits


def active_phase_limits_variable(
    r_frames: list[np.ndarray],
    w_frames: list[np.ndarray],
    snapshots: list[np.ndarray],
    floor_fraction: float = 1.0e-8,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find common limits from active support on per-frame phase grids."""
    if not snapshots:
        return (0.0, 1.0), (1.0, 10.0)
    vmax = max(float(np.nanmax(g)) for g in snapshots)
    threshold = max(vmax * floor_fraction, np.finfo(float).tiny)
    active_rs: list[np.ndarray] = []
    active_ws: list[np.ndarray] = []
    all_r = np.concatenate(r_frames)
    all_w = np.concatenate(w_frames)
    # Measure active support on each actual adaptive grid before choosing common visual limits.
    for r, w, g in zip(r_frames, w_frames, snapshots):
        active = np.isfinite(g) & (g > threshold)
        if np.any(active):
            active_rs.append(r[np.any(active, axis=1)])
            active_ws.append(w[np.any(active, axis=0)])
    if not active_rs:
        return (float(np.min(all_w)), float(np.max(all_w))), (float(np.min(all_r)), float(np.max(all_r)))
    r_active = np.concatenate(active_rs)
    w_active = np.concatenate(active_ws)
    r_lo, r_hi = float(np.min(r_active)), float(np.max(r_active))
    w_lo, w_hi = float(np.min(w_active)), float(np.max(w_active))
    r_pad = 0.12 * max(r_hi - r_lo, float(np.min(np.diff(np.unique(all_r)))))
    w_pad = 0.10 * max(w_hi - w_lo, float(np.min(np.diff(np.unique(all_w)))))
    return (
        (max(float(np.min(all_w)), w_lo - w_pad), min(float(np.max(all_w)), w_hi + w_pad)),
        (max(float(np.min(all_r)), r_lo - r_pad), min(float(np.max(all_r)), r_hi + r_pad)),
    )


def style_axes(ax: plt.Axes) -> None:
    """Apply the shared inward-tick and spine styling to one axis."""
    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        length=9,
        width=1.7,
        labelsize=12,
        top=True,
        right=True,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=5,
        width=1.3,
        top=True,
        right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def positive_limits(values: np.ndarray, floor_fraction: float = 1.0e-10) -> tuple[float, float]:
    """Return finite positive limits, flooring the lower log-plot bound."""
    values = np.asarray(values)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("No positive finite values found for logarithmic plot.")
    vmax = float(np.max(positive))
    vmin = max(float(np.min(positive)), floor_fraction * vmax)
    return vmin, vmax


def parse_range(text: str) -> tuple[float, float]:
    """Parse ``lo,hi`` and require a strictly positive increasing interval."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be formatted as lo,hi")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 < lo < hi):
        raise argparse.ArgumentTypeError("range must satisfy 0 < lo < hi")
    return lo, hi


def parse_float_range(text: str) -> tuple[float, float]:
    """Parse ``lo,hi`` and require an increasing floating-point interval."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be formatted as lo,hi")
    lo, hi = float(parts[0]), float(parts[1])
    if not (lo < hi):
        raise argparse.ArgumentTypeError("range must satisfy lo < hi")
    return lo, hi


def normalize_config_key(key: str) -> str:
    """Canonicalize a plot-config key across CLI and file spellings."""
    return key.strip().lstrip("-").replace("_", "-").lower()


def strip_config_comment(line: str) -> str:
    """Remove ``#`` or ``;`` comments that occur outside quoted text."""
    in_single = False
    in_double = False
    for i, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double and char in "#;":
            return line[:i]
    return line


def parse_bool_text(value: str) -> bool:
    """Parse common true/false spellings used by plotting config files."""
    token = normalize_config_key(value)
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def load_plot_config(path: Path, defaults: dict[str, object]) -> dict[str, object]:
    """Overlay typed key-value plot settings on a defaults dictionary."""
    # Configuration values inherit their types from known defaults, so misspelled or malformed controls produce actionable errors.
    key_to_name = {normalize_config_key(name): name for name in defaults}
    out = dict(defaults)
    with path.open() as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = strip_config_comment(raw).strip()
            if not line or (line.startswith("[") and line.endswith("]")):
                continue
            sep = "=" if "=" in line else ":"
            if sep not in line:
                raise ValueError(f"Expected 'key = value' in {path}:{line_no}")
            key, value = (part.strip() for part in line.split(sep, 1))
            normalized = normalize_config_key(key)
            if normalized not in key_to_name:
                raise ValueError(f"Unknown plot config key {key!r} in {path}:{line_no}")
            name = key_to_name[normalized]
            value = value.strip().strip('"').strip("'")
            current = out[name]
            if isinstance(current, bool):
                out[name] = parse_bool_text(value)
            elif isinstance(current, int):
                out[name] = int(value)
            elif isinstance(current, float) or current is None:
                out[name] = None if value.lower() in {"none", "null"} else float(value)
            elif isinstance(current, tuple):
                out[name] = parse_float_range(value) if name == "slope_y_range" else parse_range(value)
            else:
                out[name] = value
    return out


def smooth_log_slope(
    r: np.ndarray,
    rho: np.ndarray,
    floor_fraction: float = 1.0e-8,
    smooth_points: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``d log(rho) / d log(r)`` above a relative density floor."""
    threshold = max(float(np.nanmax(rho)) * floor_fraction, np.finfo(float).tiny)
    mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > threshold)
    r_fit = r[mask]
    rho_fit = rho[mask]
    if r_fit.size < 3:
        return r_fit, np.full_like(r_fit, np.nan)
    # Differentiate log density on positive radial support, then smooth the plotted derivative with the specified boxcar.
    slope = np.gradient(np.log(rho_fit), np.log(r_fit))
    window = min(smooth_points, slope.size if slope.size % 2 == 1 else slope.size - 1)
    if window >= 3:
        kernel = np.ones(window) / float(window)
        slope = np.convolve(slope, kernel, mode="same")
    return r_fit, slope


def power_law_fit(
    r: np.ndarray,
    rho: np.ndarray,
    fit_range: tuple[float, float],
    floor_fraction: float = 1.0e-8,
    min_points: int = 3,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """Fit a radial power law and return its slope plus a visual guide curve."""
    threshold = max(float(np.nanmax(rho)) * floor_fraction, np.finfo(float).tiny)
    lo, hi = fit_range
    mask = (
        np.isfinite(r)
        & np.isfinite(rho)
        & (r >= lo)
        & (r <= hi)
        & (r > 0.0)
        & (rho > threshold)
    )
    if np.count_nonzero(mask) < min_points:
        return None
    # Fit the physical exponent on the declared radial interval. Only the guide normalization below is shifted for visibility.
    slope, intercept = np.polyfit(np.log(r[mask]), np.log(rho[mask]), 1)
    rr = np.geomspace(float(np.min(r[mask])), float(np.max(r[mask])), 80)
    yy = np.exp(intercept) * rr**slope
    measured = np.exp(np.interp(np.log(rr), np.log(r[mask]), np.log(rho[mask])))
    # Keep the guide a little above the measured curve so both remain visible.
    yy *= max(1.25, 1.25 * float(np.max(measured / yy)))
    return float(slope), float(intercept), rr, yy


# Section: static radial-profile figures.

def plot_profiles(profiles: np.ndarray, outdir: Path, n_snapshots: int, r_max_plot: float | None) -> None:
    """Save density, metric, and compactness-gap snapshots at selected times."""
    times = select_times(profiles["time"], n_snapshots)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(times.min()), vmax=float(times.max()))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for t in times:
        rows = profiles[profiles["time"] == t]
        ax.plot(rows["r"], rows["rho"], color=cmap(norm(float(t))), lw=1.8)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho$")
    ax.set_yscale("log")
    if r_max_plot is not None:
        ax.set_xlim(profiles["r"].min(), r_max_plot)
    style_axes(ax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(TIME_LABEL)
    fig.savefig(outdir / "density_snapshots.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for t in times:
        rows = profiles[profiles["time"] == t]
        color = cmap(norm(float(t)))
        ax.plot(rows["r"], rows["mu"], color=color, lw=1.7, ls="-")
        ax.plot(rows["r"], rows["lambda"], color=color, lw=1.7, ls="--")
    ax.plot([], [], color="k", lw=1.7, ls="-", label=r"$\mu$")
    ax.plot([], [], color="k", lw=1.7, ls="--", label=r"$\lambda$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\mu,\lambda$")
    if r_max_plot is not None:
        ax.set_xlim(profiles["r"].min(), r_max_plot)
    ax.legend(frameon=False)
    style_axes(ax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(TIME_LABEL)
    fig.savefig(outdir / "metric_snapshots.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for t in times:
        rows = profiles[profiles["time"] == t]
        ax.plot(rows["r"], rows["gap"], color=cmap(norm(float(t))), lw=1.8)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$1-2m(r)/r$")
    if r_max_plot is not None:
        ax.set_xlim(profiles["r"].min(), r_max_plot)
    style_axes(ax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(TIME_LABEL)
    fig.savefig(outdir / "compactness_gap_snapshots.png", dpi=220)
    plt.close(fig)


def plot_compactness_ratio_loglog(
    profiles: np.ndarray,
    outdir: Path,
    n_snapshots: int,
    r_max_plot: float | None,
) -> None:
    """Plot the derived compactness ``2m/r`` at representative saved times."""
    r_all = profiles["r"]
    # Use the saved enclosed mass and areal radius to reconstruct the relativistic compactness diagnostic.
    compactness_all = 2.0 * profiles["m"] / r_all
    finite = np.isfinite(compactness_all) & np.isfinite(r_all) & (r_all > 0.0)
    if not np.any(finite):
        raise ValueError("No finite positive-radius compactness data found.")

    max_index = int(np.nanargmax(np.where(finite, compactness_all, np.nan)))
    max_compactness = float(compactness_all[max_index])
    if max_compactness >= 1.0:
        print(
            "WARNING: max 2m/r >= 1: "
            f"{max_compactness:.6e} at t={profiles['time'][max_index]:.6e}, "
            f"r={profiles['r'][max_index]:.6e}"
        )

    times = select_times(profiles["time"], n_snapshots)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(times.min()), vmax=float(times.max()))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    min_positive = np.inf
    max_positive = 0.0
    for t in times:
        rows = profiles[profiles["time"] == t]
        ratio = 2.0 * rows["m"] / rows["r"]
        mask = np.isfinite(rows["r"]) & np.isfinite(ratio) & (rows["r"] > 0.0) & (ratio > 0.0)
        if not np.any(mask):
            continue
        min_positive = min(min_positive, float(np.min(ratio[mask])))
        max_positive = max(max_positive, float(np.max(ratio[mask])))
        ax.plot(rows["r"][mask], ratio[mask], color=cmap(norm(float(t))), lw=1.8)

    ax.axhline(1.0, color="k", lw=1.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$2m(r)/r$")
    if r_max_plot is not None:
        ax.set_xlim(profiles["r"].min(), r_max_plot)
    if np.isfinite(min_positive) and max_positive > 0.0:
        ax.set_ylim(0.7 * min_positive, max(1.2, 1.2 * max_positive))
    style_axes(ax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(TIME_LABEL)
    fig.savefig(outdir / "compactness_ratio_loglog_snapshots.png", dpi=220)
    plt.close(fig)


def plot_density_loglog_snapshots(
    profiles: np.ndarray,
    outdir: Path,
    n_snapshots: int,
    r_max_plot: float | None,
) -> None:
    """Save positive density profiles on fixed logarithmic axes."""
    times = select_times(profiles["time"], n_snapshots)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(times.min()), vmax=float(times.max()))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    rho_min, rho_max = positive_limits(profiles["rho"], floor_fraction=1.0e-8)
    r_min, r_max = positive_limits(profiles["r"])
    if r_max_plot is not None:
        r_max = r_max_plot

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for t in times:
        rows = profiles[profiles["time"] == t]
        mask = np.isfinite(rows["r"]) & np.isfinite(rows["rho"]) & (rows["r"] > 0.0) & (rows["rho"] > 0.0)
        if np.any(mask):
            ax.plot(rows["r"][mask], rows["rho"][mask], color=cmap(norm(float(t))), lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho$")
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0.7 * rho_min, 1.3 * rho_max)
    style_axes(ax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(TIME_LABEL)
    fig.savefig(outdir / "density_loglog_snapshots.png", dpi=220)
    plt.close(fig)


def plot_pressure_vs_density_loglog(
    profiles: np.ndarray,
    outdir: Path,
    n_snapshots: int,
) -> None:
    """Plot the positive ``p(rho)`` relation at representative saved times."""
    times = select_times(profiles["time"], n_snapshots)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(times.min()), vmax=float(times.max()))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    rho_min, rho_max = positive_limits(profiles["rho"], floor_fraction=1.0e-8)
    p_min, p_max = positive_limits(profiles["p"], floor_fraction=1.0e-8)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for t in times:
        rows = profiles[profiles["time"] == t]
        mask = (
            np.isfinite(rows["rho"])
            & np.isfinite(rows["p"])
            & (rows["rho"] > 0.0)
            & (rows["p"] > 0.0)
        )
        if not np.any(mask):
            continue
        # Order pressure/density pairs by density for an instantaneous p(rho) curve.
        order = np.argsort(rows["rho"][mask])
        ax.plot(rows["rho"][mask][order], rows["p"][mask][order], color=cmap(norm(float(t))), lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$p$")
    ax.set_xlim(0.7 * rho_min, 1.3 * rho_max)
    ax.set_ylim(0.7 * p_min, 1.3 * p_max)
    style_axes(ax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(TIME_LABEL)
    fig.savefig(outdir / "pressure_vs_density_loglog_snapshots.png", dpi=220)
    plt.close(fig)


# Section: GIF frame selection and phase-space animations.

def selected_frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    """Return all indices or an endpoint-preserving evenly spaced subset."""
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, max_frames, dtype=int))


def plot_phase_space_gif(input_dir: Path, outdir: Path, max_frames: int, fps: int) -> Path | None:
    """Render complete adaptive-grid phase snapshots as a fixed-scale GIF."""
    path = input_dir / "phase_space_snapshots.csv"
    if not path.exists():
        print(f"Skipping phase-space GIF: missing {path}")
        return None

    times, r_frames, w_frames, snapshots = load_phase_snapshots(path)
    if times.size == 0:
        print(f"Skipping phase-space GIF: no complete snapshots in {path}")
        return None
    frame_indices = selected_frame_indices(len(snapshots), max_frames)
    # Select real saved frames before setting their common phase-space color and support scales.
    selected_snapshots = [snapshots[int(i)] for i in frame_indices]
    selected_r = [r_frames[int(i)] for i in frame_indices]
    selected_w = [w_frames[int(i)] for i in frame_indices]
    w_limits, r_limits = active_phase_limits_variable(selected_r, selected_w, selected_snapshots)
    global_max = max(float(np.max(snapshots[i])) for i in frame_indices)
    global_max = max(global_max, 1.0)
    floor = max(global_max * 1.0e-10, np.finfo(float).tiny)
    vmin = np.log10(floor)
    vmax = np.log10(max(global_max, floor))

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.12, top=0.91)
    first_index = int(frame_indices[0])
    first = np.log10(np.maximum(snapshots[first_index], floor))
    w_edges = cell_edges_from_centers(w_frames[first_index])
    r_edges = cell_edges_from_centers(r_frames[first_index])
    im = ax.pcolormesh(
        w_edges,
        r_edges,
        first,
        shading="auto",
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
    )
    title = ax.set_title(rf"$t={times[int(frame_indices[0])]:.2f}$")
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$r$")
    ax.set_xlim(*w_limits)
    ax.set_ylim(*r_limits)
    ax.set_yscale("log")
    style_axes(ax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(PHASE_COLOR_LABEL)

    def update(frame: int):
        """Replace the adaptive-grid mesh and time label for one GIF frame."""
        nonlocal im
        snap_index = int(frame_indices[frame])
        # A changing grid requires new mesh edges as well as new color values.
        im.remove()
        im = ax.pcolormesh(
            cell_edges_from_centers(w_frames[snap_index]),
            cell_edges_from_centers(r_frames[snap_index]),
            np.log10(np.maximum(snapshots[snap_index], floor)),
            shading="auto",
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        title.set_text(rf"$t={times[snap_index]:.2f}$")
        return (im, title)

    outpath = outdir / "phase_space_rw_logdf.gif"
    animation = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    animation.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return outpath


def plot_density_loglog_gif(
    profiles: np.ndarray,
    outdir: Path,
    max_frames: int,
    fps: int,
    r_max_plot: float | None,
) -> Path:
    """Render selected ``time,r,rho`` profile blocks as a fixed-scale GIF."""
    times = np.unique(profiles["time"])
    frame_indices = selected_frame_indices(len(times), max_frames)
    selected = profiles[np.isin(profiles["time"], times[frame_indices])]

    rho_min, rho_max = positive_limits(selected["rho"], floor_fraction=1.0e-8)
    r_min, r_max = positive_limits(selected["r"])
    if r_max_plot is not None:
        r_max = r_max_plot

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.12, top=0.91)
    (line,) = ax.plot([], [], color="tab:blue", lw=2.0)
    title = ax.set_title("")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$", fontsize=15)
    ax.set_ylabel(r"$\rho$", fontsize=15)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0.7 * rho_min, 1.3 * rho_max)
    style_axes(ax)

    def update(frame: int):
        """Update the positive density curve for one selected saved time."""
        t = times[int(frame_indices[frame])]
        rows = profiles[profiles["time"] == t]
        mask = np.isfinite(rows["r"]) & np.isfinite(rows["rho"]) & (rows["r"] > 0.0) & (rows["rho"] > 0.0)
        line.set_data(rows["r"][mask], rows["rho"][mask])
        title.set_text(rf"$t={t:.2f}$")
        return (line, title)

    outpath = outdir / "density_loglog.gif"
    animation = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    animation.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return outpath


def plot_density_slope_powerlaw_gif(
    profiles: np.ndarray,
    outdir: Path,
    max_frames: int,
    fps: int,
    r_max_plot: float | None,
    inner_fit_range: tuple[float, float],
    outer_fit_range: tuple[float, float],
    slope_ylim: tuple[float, float],
) -> tuple[Path, tuple[float, float]]:
    """Animate density, smoothed slope, and two radial power-law fits."""
    times = np.unique(profiles["time"])
    frame_indices = selected_frame_indices(len(times), max_frames)
    selected = profiles[np.isin(profiles["time"], times[frame_indices])]

    rho_min, rho_max = positive_limits(selected["rho"], floor_fraction=1.0e-8)
    r_min, r_max = positive_limits(selected["r"])
    if r_max_plot is not None:
        r_max = r_max_plot
    log_r_limits = (np.log10(r_min), np.log10(r_max))
    log_rho_limits = (np.log10(0.7 * rho_min), np.log10(2.4 * rho_max))

    fig, (ax_rho, ax_slope) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.10, top=0.92)

    (rho_line,) = ax_rho.plot([], [], color="b", lw=3.0)
    (inner_line,) = ax_rho.plot([], [], color="k", lw=2.0, ls="--")
    (outer_line,) = ax_rho.plot([], [], color="k", lw=2.0, ls="--")
    (slope_line,) = ax_slope.plot([], [], color="b", lw=3.0)
    inner_text = ax_rho.text(0.04, 0.15, "", transform=ax_rho.transAxes, fontsize=13)
    outer_text = ax_rho.text(0.04, 0.08, "", transform=ax_rho.transAxes, fontsize=13)
    title = ax_rho.set_title("", fontsize=16)

    ax_rho.set_ylabel(r"$\log_{10}\rho$", fontsize=16)
    ax_rho.set_xlim(*log_r_limits)
    ax_rho.set_ylim(*log_rho_limits)
    style_axes(ax_rho)

    ax_slope.axhline(0.0, color="0.55", lw=1.0, ls=":")
    ax_slope.set_xlabel(r"$\log_{10} r$", fontsize=16)
    ax_slope.set_ylabel(r"$d\log\rho/d\log r$", fontsize=16)
    ax_slope.set_ylim(*slope_ylim)
    style_axes(ax_slope)

    # Measure the final inner/outer power-law values directly from the latest saved profile.
    final_rows = profiles[profiles["time"] == times[-1]]
    final_inner = power_law_fit(final_rows["r"], final_rows["rho"], inner_fit_range, min_points=2)
    final_outer = power_law_fit(final_rows["r"], final_rows["rho"], outer_fit_range)
    final_slopes = (
        final_inner[0] if final_inner is not None else float("nan"),
        final_outer[0] if final_outer is not None else float("nan"),
    )

    def update(frame: int):
        """Update density, slope, fit guides, and annotations for one frame."""
        t = times[int(frame_indices[frame])]
        rows = profiles[profiles["time"] == t]
        r = rows["r"]
        rho = rows["rho"]
        mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > 0.0)
        rho_line.set_data(np.log10(r[mask]), np.log10(rho[mask]))

        slope_r, slope = smooth_log_slope(r, rho)
        slope_mask = np.isfinite(slope_r) & np.isfinite(slope) & (slope_r > 0.0)
        slope_line.set_data(np.log10(slope_r[slope_mask]), slope[slope_mask])

        inner = power_law_fit(r, rho, inner_fit_range, min_points=2)
        outer = power_law_fit(r, rho, outer_fit_range)
        if inner is None:
            inner_line.set_data([], [])
            inner_text.set_text(r"$\alpha_{\rm in}:$ unavailable")
        else:
            inner_slope, _, rr, yy = inner
            inner_line.set_data(np.log10(rr), np.log10(yy))
            inner_text.set_text(rf"$\alpha_{{\rm in}}={inner_slope:.2f}$")
        if outer is None:
            outer_line.set_data([], [])
            outer_text.set_text(r"$\alpha_{\rm out}:$ unavailable")
        else:
            outer_slope, _, rr, yy = outer
            outer_line.set_data(np.log10(rr), np.log10(yy))
            outer_text.set_text(rf"$\alpha_{{\rm out}}={outer_slope:.2f}$")
        title.set_text(rf"$t={t:.2f}$")
        return (rho_line, inner_line, outer_line, slope_line, inner_text, outer_text, title)

    outpath = outdir / "density_slope_powerlaw_fits.gif"
    animation = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    animation.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return outpath, final_slopes


# Section: history and comparison diagnostics.

def plot_history(history: np.ndarray, outdir: Path) -> None:
    """Plot total matter mass and minimum compactness gap against time."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(history["time"], history["matter_mass"], color="tab:blue", lw=1.8)
    axes[0].set_ylabel(r"$M_{\rm matter}$")
    style_axes(axes[0])

    axes[1].plot(history["time"], history["min_gap"], color="tab:red", lw=1.8, label=r"$\min_r(1-2m/r)$")
    axes[1].set_xlabel(TIME_LABEL)
    axes[1].set_ylabel(r"$\min_r(1-2m/r)$")
    style_axes(axes[1])
    fig.savefig(outdir / "history_mass_gap.png", dpi=220)
    plt.close(fig)


def plot_initial_final_density_slope_comparison(
    profiles: np.ndarray,
    outdir: Path,
    r_max_plot: float | None,
    slope_ylim: tuple[float, float],
) -> Path:
    """Compare the first and last density profiles and their derived slopes."""
    times = np.unique(profiles["time"])
    selected_times = [times[0], times[-1]]
    selected = profiles[np.isin(profiles["time"], selected_times)]
    rho_min, rho_max = positive_limits(selected["rho"], floor_fraction=1.0e-8)
    r_min, r_max = positive_limits(selected["r"])
    if r_max_plot is not None:
        r_max = r_max_plot

    fig, (ax_rho, ax_slope) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.10, top=0.92)

    styles = [
        (times[0], "0.55", 2.0, r"$\mathrm{initial}$"),
        (times[-1], "b", 3.0, r"$\mathrm{final}$"),
    ]
    for t, color, lw, label in styles:
        rows = profiles[profiles["time"] == t]
        r = rows["r"]
        rho = rows["rho"]
        mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > 0.0)
        ax_rho.plot(np.log10(r[mask]), np.log10(rho[mask]), color=color, lw=lw, label=label)
        slope_r, slope = smooth_log_slope(r, rho)
        slope_mask = np.isfinite(slope_r) & np.isfinite(slope) & (slope_r > 0.0)
        ax_slope.plot(np.log10(slope_r[slope_mask]), slope[slope_mask], color=color, lw=lw)

    ax_rho.set_ylabel(r"$\log_{10}\rho$", fontsize=16)
    ax_rho.set_xlim(np.log10(r_min), np.log10(r_max))
    ax_rho.set_ylim(np.log10(0.7 * rho_min), np.log10(2.0 * rho_max))
    ax_rho.legend(frameon=False, fontsize=13)
    style_axes(ax_rho)

    ax_slope.axhline(-2.0, color="k", lw=1.8, ls="--", label=r"$-2$")
    ax_slope.set_xlabel(r"$\log_{10} r$", fontsize=16)
    ax_slope.set_ylabel(r"$d\log\rho/d\log r$", fontsize=16)
    ax_slope.set_ylim(*slope_ylim)
    style_axes(ax_slope)

    outpath = outdir / "initial_vs_final_density_slope_comparison.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_inward_reflection_diagnostics(input_dir: Path, outdir: Path) -> Path | None:
    """Measure weighted phase-space transport near the reflecting boundary."""
    path = input_dir / "phase_space_snapshots.csv"
    if not path.exists():
        print(f"Skipping inward/reflection diagnostics: missing {path}")
        return None

    times, r_frames, w_frames, snapshots = load_phase_snapshots(path)
    if len(snapshots) == 0:
        return None

    cuts = (0.01, 0.02, 0.05, 0.1, 0.2)
    records = []
    for t, r, w, g in zip(times, r_frames, w_frames, snapshots):
        r = np.ravel(np.asarray(r, dtype=float))
        w = np.ravel(np.asarray(w, dtype=float))
        if r.size < 1 or w.size < 1 or g.shape != (r.size, w.size):
            continue
        r_weights = trap_weights(r)
        w_weights = trap_weights(w)
        # Trapezoidal radial/momentum weights turn the projected distribution into the diagnostic population measure.
        weight = np.maximum(g, 0.0) * r_weights[:, None] * w_weights[None, :]
        total = float(np.sum(weight))
        if total <= 0.0:
            continue
        radial_weight = np.sum(weight, axis=1)
        r_peak = float(r[int(np.argmax(radial_weight))])
        r_mean = float(np.sum(r[:, None] * weight) / total)
        w_mean = float(np.sum(w[None, :] * weight) / total)
        neg_frac = float(np.sum(weight[:, w < 0.0]) / total)
        pos_frac = float(np.sum(weight[:, w > 0.0]) / total)
        cut_values = []
        for cut in cuts:
            near = r <= cut
            # Inner outward fractions use each radial cut's own mass denominator, while inner mass fraction uses the global projected mass.
            near_mass = float(np.sum(weight[near, :])) if np.any(near) else 0.0
            if near_mass > 0.0:
                out_frac = float(np.sum(weight[np.ix_(near, w > 0.0)]) / near_mass)
                near_w_mean = float(np.sum(w[None, :] * weight[near, :]) / near_mass)
            else:
                out_frac = np.nan
                near_w_mean = np.nan
            cut_values.extend([near_mass / total, out_frac, near_w_mean])
        records.append([float(t), r_peak, r_mean, w_mean, neg_frac, pos_frac, *cut_values])

    if not records:
        return None

    diagnostics = np.array(records, dtype=float)
    columns = ["time", "r_peak", "r_mean", "w_mean", "neg_frac", "pos_frac"]
    for cut in cuts:
        tag = f"{cut:g}".replace(".", "p")
        columns.extend([f"mass_r_lt_{tag}", f"out_frac_r_lt_{tag}", f"w_mean_r_lt_{tag}"])
    np.savetxt(
        outdir / "inward_reflection_diagnostics.csv",
        diagnostics,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 10.0), sharex=True, constrained_layout=True)
    axes[0].plot(diagnostics[:, 0], diagnostics[:, 1], color="b", lw=2.8, label=r"$r_{\rm peak}$")
    axes[0].plot(diagnostics[:, 0], diagnostics[:, 2], color="0.35", lw=1.8, label=r"$\langle r\rangle$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$r$", fontsize=15)
    axes[0].legend(frameon=False, fontsize=12)

    axes[1].plot(diagnostics[:, 0], diagnostics[:, 3], color="b", lw=2.8)
    axes[1].axhline(0.0, color="k", lw=1.0)
    axes[1].set_ylabel(r"$\langle w\rangle$", fontsize=15)

    axes[2].plot(diagnostics[:, 0], diagnostics[:, 4], color="b", lw=2.8, label=r"$w<0$")
    axes[2].plot(diagnostics[:, 0], diagnostics[:, 5], color="crimson", lw=2.0, label=r"$w>0$")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel(r"global fraction", fontsize=15)
    axes[2].legend(frameon=False, ncol=2, fontsize=12)

    mass_005 = columns.index("mass_r_lt_0p05")
    out_005 = columns.index("out_frac_r_lt_0p05")
    out_01 = columns.index("out_frac_r_lt_0p1")
    axes[3].plot(diagnostics[:, 0], diagnostics[:, out_005], color="b", lw=2.8, label=r"$r<0.05,\ w>0$")
    axes[3].plot(diagnostics[:, 0], diagnostics[:, out_01], color="crimson", lw=2.0, label=r"$r<0.1,\ w>0$")
    axes[3].plot(
        diagnostics[:, 0],
        diagnostics[:, mass_005],
        color="0.35",
        lw=1.6,
        ls="--",
        label=r"$r<0.05$ mass",
    )
    axes[3].set_ylim(-0.03, 1.03)
    axes[3].set_xlabel(r"$t$", fontsize=15)
    axes[3].set_ylabel(r"inner fraction", fontsize=15)
    axes[3].legend(frameon=False, ncol=2, fontsize=10)

    for ax in axes:
        style_axes(ax)

    outpath = outdir / "inward_reflection_diagnostics.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_phase(input_dir: Path, outdir: Path) -> None:
    """Plot initial/final ``(r,w,g)`` projections with one color floor."""
    r0, w0, g0 = load_phase(input_dir / "phase_space_initial.csv")
    r1, w1, g1 = load_phase(input_dir / "phase_space_final.csv")
    floor = max(float(np.max(g0)), float(np.max(g1)), 1.0) * 1.0e-14
    w_limits, r_limits = active_phase_limits(r0, w0, [g0, g1])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharex=True, sharey=True, constrained_layout=True)
    for ax, r, w, g, title in [
        (axes[0], r0, w0, g0, r"$\mathrm{initial}$"),
        (axes[1], r1, w1, g1, r"$\mathrm{final}$"),
    ]:
        im = ax.imshow(
            np.log10(np.maximum(g, floor)),
            origin="lower",
            aspect="auto",
            extent=[w[0], w[-1], r[0], r[-1]],
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel(r"$w$")
        ax.set_xlim(*w_limits)
        ax.set_ylim(*r_limits)
        style_axes(ax)
    axes[0].set_ylabel(r"$r$")
    cbar = fig.colorbar(im, ax=axes)
    cbar.set_label(PHASE_COLOR_LABEL)
    fig.savefig(outdir / "phase_space_initial_final.png", dpi=220)
    plt.close(fig)


def plot_phase_montage(
    input_dir: Path,
    outdir: Path,
    n_snapshots: int,
    montage_name: str,
) -> Path | None:
    """Save an endpoint-spanning montage of complete phase-space snapshots."""
    path = input_dir / "phase_space_snapshots.csv"
    if not path.exists():
        return None

    times, r_frames, w_frames, snapshots = load_phase_snapshots(path)
    if times.size == 0:
        return None
    # The montage spans actual endpoints and retains every selected frame's adaptive coordinates.
    indices = np.unique(np.linspace(0, times.size - 1, n_snapshots, dtype=int))
    n = indices.size
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    if n == 16:
        nrows, ncols = 4, 4

    selected_snapshots = [snapshots[int(i)] for i in indices]
    selected_r = [r_frames[int(i)] for i in indices]
    selected_w = [w_frames[int(i)] for i in indices]
    w_limits, r_limits = active_phase_limits_variable(selected_r, selected_w, selected_snapshots)
    global_max = max(float(np.max(g)) for g in selected_snapshots)
    global_max = max(global_max, 1.0)
    floor = max(global_max * 1.0e-10, np.finfo(float).tiny)
    vmin = np.log10(floor)
    vmax = np.log10(max(global_max, floor))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for ax, snap_index in zip(axes.ravel(), indices):
        r = r_frames[int(snap_index)]
        w = w_frames[int(snap_index)]
        g = snapshots[int(snap_index)]
        image = ax.pcolormesh(
            cell_edges_from_centers(w),
            cell_edges_from_centers(r),
            np.log10(np.maximum(g, floor)),
            shading="auto",
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(rf"$t={times[int(snap_index)]:.2f}$", fontsize=16)
        ax.set_xlabel(r"$w$", fontsize=15)
        ax.set_ylabel(r"$r$", fontsize=15)
        ax.set_xlim(*w_limits)
        ax.set_ylim(*r_limits)
        ax.tick_params(axis="both", which="major", labelsize=11, length=5, width=1.1, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel()[:n].tolist(), shrink=0.86)
        cbar.set_label(PHASE_COLOR_LABEL, fontsize=15)

    outpath = outdir / montage_name
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


# Section: command-line interface and orchestration.

def parse_args() -> argparse.Namespace:
    """Load optional typed defaults and parse all rendering controls."""
    defaults: dict[str, object] = {
        "input_dir": "data/einstein_vlasov_cpp",
        "output_dir": "plots/einstein_vlasov_cpp",
        "n_snapshots": 25,
        "montage_snapshots": 16,
        "montage_name": "Einstein_Vlasov_DF_montage_reproduction.png",
        "make_gifs": False,
        "gif_max_frames": 120,
        "gif_fps": 12,
        "inner_fit_range": (0.1, 0.3),
        "outer_fit_range": (2.0, 30.0),
        "slope_y_range": (-6.0, 0.0),
        "r_max_plot": None,
    }
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    # Read the optional configuration first, then let explicit command-line values override its typed defaults.
    known, remaining = pre.parse_known_args()
    if known.config is not None:
        defaults = load_plot_config(known.config, defaults)

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--input-dir", default=defaults["input_dir"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--n-snapshots", type=int, default=defaults["n_snapshots"])
    parser.add_argument("--montage-snapshots", type=int, default=defaults["montage_snapshots"])
    parser.add_argument("--montage-name", default=defaults["montage_name"])
    parser.add_argument("--make-gifs", action=BooleanOptionalAction, default=defaults["make_gifs"])
    parser.add_argument("--gif-max-frames", type=int, default=defaults["gif_max_frames"])
    parser.add_argument("--gif-fps", type=int, default=defaults["gif_fps"])
    parser.add_argument("--inner-fit-range", type=parse_range, default=defaults["inner_fit_range"])
    parser.add_argument("--outer-fit-range", type=parse_range, default=defaults["outer_fit_range"])
    parser.add_argument("--slope-y-range", type=parse_float_range, default=defaults["slope_y_range"])
    parser.add_argument("--r-max-plot", type=float, default=defaults["r_max_plot"])
    return parser.parse_args(remaining)


def main() -> None:
    """Load one solver output directory and write the requested diagnostics."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    profiles = load_named_csv(input_dir / "profiles.csv")
    history = load_named_csv(input_dir / "history.csv")
    plot_profiles(profiles, outdir, args.n_snapshots, args.r_max_plot)
    plot_compactness_ratio_loglog(profiles, outdir, args.n_snapshots, args.r_max_plot)
    plot_density_loglog_snapshots(profiles, outdir, args.n_snapshots, args.r_max_plot)
    plot_pressure_vs_density_loglog(profiles, outdir, args.n_snapshots)
    plot_history(history, outdir)
    comparison = plot_initial_final_density_slope_comparison(
        profiles, outdir, args.r_max_plot, args.slope_y_range
    )
    motion_diagnostics = plot_inward_reflection_diagnostics(input_dir, outdir)
    plot_phase(input_dir, outdir)
    montage = plot_phase_montage(input_dir, outdir, args.montage_snapshots, args.montage_name)
    gif_paths = []
    # Generate time-dependent products only when requested, reusing the same loaded physical outputs and fit settings.
    if args.make_gifs:
        phase_gif = plot_phase_space_gif(input_dir, outdir, args.gif_max_frames, args.gif_fps)
        if phase_gif is not None:
            gif_paths.append(phase_gif)
        gif_paths.append(
            plot_density_loglog_gif(profiles, outdir, args.gif_max_frames, args.gif_fps, args.r_max_plot)
        )
        combined_gif, final_slopes = plot_density_slope_powerlaw_gif(
            profiles,
            outdir,
            args.gif_max_frames,
            args.gif_fps,
            args.r_max_plot,
            args.inner_fit_range,
            args.outer_fit_range,
            args.slope_y_range,
        )
        gif_paths.append(combined_gif)
        print(
            "Final-time density slopes: "
            f"inner={final_slopes[0]:.6g}, outer={final_slopes[1]:.6g}"
        )
    print(f"Wrote plots to {outdir.resolve()}")
    print(f"Wrote initial/final density comparison to {comparison.resolve()}")
    if motion_diagnostics is not None:
        print(f"Wrote inward/reflection diagnostics to {motion_diagnostics.resolve()}")
    if montage is not None:
        print(f"Wrote phase-space montage to {montage.resolve()}")
    for gif_path in gif_paths:
        print(f"Wrote GIF to {gif_path.resolve()}")


if __name__ == "__main__":
    main()
