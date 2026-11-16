#!/usr/bin/env python3
# Load complete spherical density profiles and share the radial slope, frame selection and image-output routines.
# Named CSV columns carry physical time, step, radius and density in each run's units.
# Logarithmic fits and display floors operate on copied plotting arrays; the solver files remain the physical source.

"""Reusable spherical-profile loading, diagnostics, and GIF rendering.

The public helpers read complete named-column profile tables with ``time``,
``step``, ``r``, and ``rho``; select endpoint-preserving frame times; collapse
duplicate radii; derive ``d log(rho) / d log(r)`` by a local polynomial or raw
gradient; and apply the common VENUS plot style.  ``fig_to_rgb``,
``save_gif_atomic``, and ``write_frame_times`` provide the shared raster and
publication pipeline used by the maintained Frank and SphDSMC renderers.

The module also remains directly executable as a generic two-panel
density/slope renderer.  It ignores incomplete trailing CSV rows, fixes panel
limits across selected saved times, writes the GIF through a sibling temporary
file, and publishes the exact frame times in an atomic sidecar.  All floors and
slope fits are visualization operations and never modify solver output.
"""

from __future__ import annotations

import argparse
import io
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
# Section: conservative plotting-thread defaults.
# Plotting usually runs best as a one-thread post-processing task, but respect
# any values a cluster wrapper or user shell already supplied.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None
    from PIL import Image
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Section: robust profile loading.
# The renderer can be used while a solver is still appending profiles.csv, so
# incomplete trailing rows are ignored before NumPy parses the snapshot.
def read_complete_csv_snapshot(path: Path) -> str:
    """Return the header and only newline-terminated, width-complete CSV rows."""
    complete: list[str] = []
    with path.open() as handle:
        header = handle.readline().rstrip("\n")
        if not header:
            raise RuntimeError(f"{path} is empty")
        expected_commas = header.count(",")
        complete.append(header)
        for line in handle:
            if not line.endswith("\n"):
                continue
            row = line.rstrip("\n")
            if row.count(",") == expected_commas:
                complete.append(row)
    if len(complete) == 1:
        raise RuntimeError(f"{path} has no complete CSV rows")
    return "\n".join(complete) + "\n"


def load_profiles(path: Path) -> np.ndarray:
    """Load complete snapshots on the largest stored radial mesh.

    A solver may have written several complete rows of its next snapshot
    before this reader opens the file. Newline termination validates a row;
    agreement with the full radial mesh validates the physical snapshot.
    """
    raw = np.genfromtxt(io.StringIO(read_complete_csv_snapshot(path)), delimiter=",", names=True)
    data = np.atleast_1d(raw)
    if data.size == 0:
        raise RuntimeError(f"No profile rows found in {path}")
    names = data.dtype.names or ()
    required = {"time", "step", "r", "rho"}
    if not required.issubset(names):
        raise RuntimeError(f"{path} is missing required profile columns {sorted(required)}")
    finite = (
        np.isfinite(data["time"])
        & np.isfinite(data["step"])
        & np.isfinite(data["r"])
        & np.isfinite(data["rho"])
    )
    data = data[finite]
    if data.size == 0:
        raise RuntimeError(f"No finite profile rows found in {path}")
    times = np.unique(data["time"])
    meshes = [np.unique(data["r"][data["time"] == time]) for time in times]
    # Use the largest stored radial mesh to identify complete physical slices after filtering incomplete individual CSV rows.
    reference = max(meshes, key=lambda mesh: mesh.size)
    complete_times = times[
        [np.array_equal(mesh, reference) for mesh in meshes]
    ]
    return data[np.isin(data["time"], complete_times)]


# Section: Matplotlib style helpers.
# These routines keep animated frames visually consistent and add common
# reference slopes used when comparing cusp evolution.
def style_axes(ax: plt.Axes) -> None:
    """Apply the shared inward-tick and spine styling to one profile panel."""
    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        length=9,
        width=1.7,
        labelsize=14,
        top=True,
        right=True,
        pad=4,
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


def add_slope_reference_lines(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Add labeled horizontal guides for common density power-law slopes."""
    refs = [(-1.0, r"$-1$"), (-1.5, r"$-3/2$"), (-2.0, r"$-2$"), (-3.0, r"$-3$")]
    text_x = x_min + 0.78 * (x_max - x_min)
    for y, label in refs:
        if y_min <= y <= y_max:
            ax.axhline(y, color="0.55", lw=0.95, ls="--", zorder=0)
            ax.text(
                text_x,
                y,
                label,
                color="0.25",
                fontsize=11,
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.1},
            )


def positive_limits(values: np.ndarray, floor_fraction: float = 1.0e-8) -> tuple[float, float]:
    """Return finite positive limits with a relative lower display floor."""
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0e-12, 1.0
    vmax = float(np.max(positive))
    return max(float(np.min(positive)), floor_fraction * vmax), vmax


# Section: frame selection.
# Supports uniformly sampled output indices or approximately uniform physical
# times, with optional extra resolution at early times.
def unique_index_sample(unique: np.ndarray, frames: int) -> np.ndarray:
    """Select both endpoints, or the latest state for a single frame."""
    count = min(max(int(frames), 1), unique.size)
    if count >= unique.size:
        return unique
    if count == 1:
        return unique[-1:]
    indices = np.linspace(0, unique.size - 1, count, dtype=int)
    return unique[np.unique(indices)]


def build_target_times(
    times: np.ndarray,
    frames: int,
    early_time_max: float | None,
    early_frames: int,
    frame_sampling: str = "index",
) -> np.ndarray:
    """Choose saved frame times by index or nearest uniform physical time.

    Optional early-time targets allocate extra frames before
    ``early_time_max``.  If nearest-time collisions underfill the request, the
    deterministic index sampler supplies the requested number instead.
    """
    unique = np.unique(times)
    frames = max(int(frames), 1)
    if unique.size <= 1:
        return unique
    if frames >= unique.size:
        return unique
    if frames == 1:
        return unique[-1:]
    # Index sampling and nearest-physical-time sampling answer different display requests. Each returns actual saved times.
    if frame_sampling == "index":
        return unique_index_sample(unique, frames)
    t_min = float(unique[0])
    t_max = float(unique[-1])
    if early_time_max is None or early_frames <= 0 or early_time_max <= t_min:
        targets = np.linspace(t_min, t_max, frames)
    else:
        early_t_max = min(float(early_time_max), t_max)
        # Keep one late frame whenever the early window ends before the
        # simulation. Even an all-early requested allocation must retain the
        # final physical state in this endpoint-preserving renderer.
        early_limit = frames - 1 if early_t_max < t_max else frames
        early_count = min(max(int(early_frames), 1), early_limit)
        late_count = max(frames - early_count, 0)
        early_targets = np.linspace(t_min, early_t_max, early_count)
        if late_count == 0 or early_t_max >= t_max:
            targets = early_targets
        else:
            late_targets = np.linspace(early_t_max, t_max, late_count + 1)[1:]
            targets = np.concatenate([early_targets, late_targets])
    # Resolve requested time targets to existing states, then remove duplicate selections.
    selected = np.array([unique[int(np.argmin(np.abs(unique - t)))] for t in targets], dtype=float)
    selected = np.unique(selected)
    if selected.size < min(frames, unique.size):
        return unique_index_sample(unique, frames)
    return selected


def unique_r_rows(rows: np.ndarray) -> np.ndarray:
    """Sort a profile by radius and retain one row per distinct radius."""
    order = np.argsort(rows["r"])
    rows = rows[order]
    _, indices = np.unique(rows["r"], return_index=True)
    return rows[np.sort(indices)]


# Section: log-density slope estimators.
# Both methods work in log(r)-log(rho) space; local-polynomial fitting is the
# default because it is less noisy than a raw gradient for sparse snapshots.
def local_poly_log_slope(
    r: np.ndarray,
    rho: np.ndarray,
    floor_fraction: float,
    window_points: int,
    poly_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``d log(rho) / d log(r)`` with a moving polynomial fit."""
    threshold = 0.0 if floor_fraction <= 0.0 else max(float(np.nanmax(rho)) * floor_fraction, np.finfo(float).tiny)
    mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > threshold)
    r_fit = np.asarray(r[mask], dtype=float)
    rho_fit = np.asarray(rho[mask], dtype=float)
    if r_fit.size < 3:
        return r_fit, np.full_like(r_fit, np.nan)
    order = np.argsort(r_fit)
    r_fit = r_fit[order]
    rho_fit = rho_fit[order]
    r_fit, unique_indices = np.unique(r_fit, return_index=True)
    rho_fit = rho_fit[unique_indices]
    if r_fit.size < 3:
        return r_fit, np.full_like(r_fit, np.nan)
    x = np.log10(r_fit)
    y = np.log10(rho_fit)
    n = x.size
    # Choose an odd neighborhood large enough for a local derivative and small enough for the available positive data.
    window = max(int(window_points), 3)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return r_fit, np.gradient(y, x)
    half = window // 2
    degree = min(max(int(poly_order), 1), window - 1)
    slope = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < window:
            if lo == 0:
                hi = min(n, window)
            elif hi == n:
                lo = max(0, n - window)
        if hi - lo < 3:
            continue
        # Center log radius on the evaluation point. The linear polynomial coefficient is the logarithmic density slope.
        coeff = np.polyfit(x[lo:hi] - x[i], y[lo:hi], min(degree, hi - lo - 1))
        slope[i] = float(coeff[-2])
    return r_fit, slope


def gradient_log_slope(r: np.ndarray, rho: np.ndarray, floor_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the logarithmic density slope by a direct finite gradient."""
    threshold = 0.0 if floor_fraction <= 0.0 else max(float(np.nanmax(rho)) * floor_fraction, np.finfo(float).tiny)
    mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > threshold)
    r_fit = np.asarray(r[mask], dtype=float)
    rho_fit = np.asarray(rho[mask], dtype=float)
    if r_fit.size < 3:
        return r_fit, np.full_like(r_fit, np.nan)
    order = np.argsort(r_fit)
    r_fit = r_fit[order]
    rho_fit = rho_fit[order]
    return r_fit, np.gradient(np.log10(rho_fit), np.log10(r_fit))


# Section: GIF output helpers.
# Frames are converted to RGB arrays and written atomically, along with a
# sidecar listing the physical time shown in each frame.
def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Rasterize a Matplotlib figure into an owned RGB array."""
    fig.canvas.draw()
    # Matplotlib may reuse the canvas buffer for later figures.  Retain an
    # owned RGB copy for each rendered frame.
    # ``buffer_rgba`` replaces ``tostring_rgb``, removed in Matplotlib 3.10.
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif_atomic(outpath: Path, frames: list[np.ndarray], fps: float) -> None:
    """Encode full RGB frames and atomically replace the requested GIF."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # Encode to a sibling temporary file before replacing the requested GIF, keeping incomplete output separate.
    tmp = outpath.with_name(outpath.stem + ".tmp.gif")
    if imageio is not None:
        imageio.mimsave(tmp, frames, duration=1.0 / max(float(fps), 1.0e-6), loop=0)
    else:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            tmp,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000.0 / max(float(fps), 1.0e-6)),
            loop=0,
        )
    tmp.replace(outpath)


def write_frame_times(outpath: Path, target_times: np.ndarray) -> None:
    """Atomically write frame indices and exact physical times beside a GIF."""
    times_path = outpath.with_name(outpath.stem + "_frame_times.txt")
    tmp = times_path.with_name(times_path.name + ".tmp")
    tmp.write_text(
        "\n".join(f"{index},{float(time):.17g}" for index, time in enumerate(target_times)) + "\n"
    )
    tmp.replace(times_path)


# Section: GIF rendering workflow.
# This ties together CSV loading, frame selection, slope measurement, and
# plotting into one command-line renderer.
def render(args: argparse.Namespace) -> Path:
    """Render fixed-limit density and slope panels for selected saved times."""
    profiles = np.concatenate([load_profiles(input_dir / args.profiles_file) for input_dir in args.input_dir])
    times = np.unique(profiles["time"])
    if times.size == 0:
        raise RuntimeError("No profile times available.")
    # Select the actual frame clocks before deriving shared density and radial limits.
    target_times = build_target_times(
        times,
        min(args.frames, times.size),
        args.early_time_max,
        args.early_frames,
        args.frame_sampling,
    )
    selected = profiles[np.isin(profiles["time"], target_times)]
    r_vmin, r_vmax = positive_limits(selected["r"])
    r_max = min(float(args.r_max_plot), r_vmax) if args.r_max_plot is not None else r_vmax
    r_min = max(float(args.r_min_plot), r_vmin) if args.r_min_plot is not None else r_vmin
    log_r_min = float(np.log10(r_min))
    log_r_max = float(np.log10(r_max))

    rho_vmin, rho_vmax = positive_limits(selected["rho"], floor_fraction=args.rho_floor_rel)
    rho_log_y_min = float(np.log10(rho_vmin)) if args.rho_log_y_min is None else args.rho_log_y_min
    rho_log_y_max = float(np.log10(rho_vmax * 1.2)) if args.rho_log_y_max is None else args.rho_log_y_max

    frames: list[np.ndarray] = []
    for t in target_times:
        # Order one physical slice by radius and retain one record per radius before forming logarithms.
        rows = unique_r_rows(profiles[profiles["time"] == t])
        mask = (
            np.isfinite(rows["r"])
            & np.isfinite(rows["rho"])
            & (rows["r"] > 0.0)
            & (rows["rho"] > 0.0)
        )
        r = rows["r"][mask]
        rho = rows["rho"][mask]

        fig, (ax_rho, ax_slope) = plt.subplots(1, 2, figsize=(10.5, 4.35), dpi=135)
        fig.subplots_adjust(left=0.085, right=0.965, bottom=0.145, top=0.895, wspace=0.31)
        ax_rho.plot(np.log10(r), np.log10(rho), color=args.rho_color, lw=args.line_width)
        ax_rho.set_xlim(log_r_min, log_r_max)
        ax_rho.set_ylim(rho_log_y_min, rho_log_y_max)
        ax_rho.set_xlabel(r"$\log_{10} r$", fontsize=args.label_size, labelpad=7)
        ax_rho.set_ylabel(r"$\log_{10}\rho$", fontsize=args.label_size, labelpad=8)
        style_axes(ax_rho)

        if r.size >= 8:
            if args.slope_method == "gradient":
                sr, slope = gradient_log_slope(r, rho, args.slope_floor_rel)
            else:
                sr, slope = local_poly_log_slope(
                    r,
                    rho,
                    args.slope_floor_rel,
                    args.slope_window_points,
                    args.slope_poly_order,
                )
            smask = np.isfinite(sr) & np.isfinite(slope) & (sr > 0.0)
            ax_slope.plot(np.log10(sr[smask]), slope[smask], color=args.slope_color, lw=args.line_width)
        ax_slope.axhline(0.0, color="0.6", lw=1.0, ls=":")
        add_slope_reference_lines(
            ax_slope,
            log_r_min,
            log_r_max,
            args.slope_y_min,
            args.slope_y_max,
        )
        ax_slope.set_xlim(log_r_min, log_r_max)
        ax_slope.set_ylim(args.slope_y_min, args.slope_y_max)
        ax_slope.set_xlabel(r"$\log_{10} r$", fontsize=args.label_size, labelpad=7)
        ax_slope.set_ylabel(r"$\mathrm{d}\log\rho/\mathrm{d}\log r$", fontsize=args.label_size, labelpad=8)
        style_axes(ax_slope)
        fig.suptitle(rf"$t={float(t):.3g}$", fontsize=args.title_size, y=0.975)
        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    save_gif_atomic(args.output, frames, args.fps)
    write_frame_times(args.output, target_times)
    return args.output


# Section: command-line interface.
def parse_args() -> argparse.Namespace:
    """Parse profile sources, frame sampling, diagnostics, and GIF controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--profiles-file", default="profiles.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-sampling", choices=("index", "time"), default="index")
    parser.add_argument("--early-time-max", type=float, default=None)
    parser.add_argument("--early-frames", type=int, default=0)
    parser.add_argument("--r-min-plot", type=float, default=0.08)
    parser.add_argument("--r-max-plot", type=float, default=6.0)
    parser.add_argument("--rho-floor-rel", type=float, default=1.0e-8)
    parser.add_argument("--rho-log-y-min", type=float, default=None)
    parser.add_argument("--rho-log-y-max", type=float, default=None)
    parser.add_argument("--slope-y-min", type=float, default=-6.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    parser.add_argument("--label-size", type=float, default=16.0)
    parser.add_argument("--tick-label-size", type=float, default=14.0)
    parser.add_argument("--title-size", type=float, default=16.0)
    parser.add_argument("--rho-color", default="b")
    parser.add_argument("--slope-color", default="r")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--slope-method", choices=("spline", "pchip", "local-poly", "gradient"), default="local-poly")
    parser.add_argument("--slope-floor-rel", type=float, default=0.0)
    parser.add_argument("--slope-window-points", type=int, default=9)
    parser.add_argument("--slope-poly-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Render the requested profile sequence and report the output path."""
    outpath = render(parse_args())
    print(f"Wrote {outpath.resolve()}")


if __name__ == "__main__":
    main()
