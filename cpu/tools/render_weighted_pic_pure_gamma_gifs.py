#!/usr/bin/env python3
# Animate deposited WPIC density and locally fitted logarithmic slope for the named initial power-law cases.
# Input profile CSVs retain physical time, radius, density and effective marker count in code units.
# The same radial/floor criteria and centered local-polynomial derivative apply to initial and evolving profiles.

"""Render density and logarithmic-slope GIFs for pure-power-law WPIC runs.

Each complete ``profile_*.csv`` file supplies one physical time.  Incomplete
live-output frames are ignored, density slopes use local polynomial fits in
log radius, and the resulting animation is atomically renamed into place.
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
from PIL import GifImagePlugin, Image


# ---- Named slope cases and crash-safe streamed GIF writer -----------------
CASES = {
    "gamma0p1": 0.1,
    "gamma0p5": 0.5,
    "gamma1p5": 1.5,
    "gamma2p0": 2.0,
    "gamma2p5": 2.5,
}


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
def load_profiles(directory: Path) -> list[tuple[float, np.ndarray]]:
    """Read complete radial profiles and return one snapshot per physical time.

    A profile supplies the spherical density ``rho(r)`` and the effective
    marker count ``neff`` deposited on the field mesh.  Repeated times are
    replaced by the last complete file, which is the desired behavior after a
    checkpoint continuation rewrites its launch snapshot.
    """
    loaded: dict[float, np.ndarray] = {}
    for path in sorted(directory.glob("profile_*.csv")):
        rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        required = {"time", "radius", "density", "neff"}
        if not required.issubset(rows.dtype.names or ()):
            raise RuntimeError(f"{path} is missing {sorted(required)}")
        time = float(rows["time"][0])
        if np.all(np.isfinite(rows["radius"])) and np.all(np.isfinite(rows["density"])):
            # A repeated continuation time contributes the last finite profile encountered; sorted measured clocks define the sequence.
            loaded[time] = rows
    if not loaded:
        raise RuntimeError(f"No complete profile_*.csv files in {directory}")
    return sorted(loaded.items())


def display_mask(rows: np.ndarray, r_min: float, r_max: float, floor_rel: float) -> np.ndarray:
    """Select the radial interval where the deposited density is meaningful.

    The relative floor removes empty-marker tails before taking logarithms; it
    is a plotting/noise cut and does not modify the simulated mass profile.
    """
    radius = np.asarray(rows["radius"], dtype=float)
    density = np.asarray(rows["density"], dtype=float)
    finite_positive = np.isfinite(density) & (density > 0.0)
    peak = float(np.max(density[finite_positive])) if np.any(finite_positive) else 0.0
    # The relative plotting floor follows the current density peak while radial limits are explicit user controls.
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
    # An odd neighborhood permits a centered local polynomial fit wherever enough positive samples exist.
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
        # Center log radius at the evaluation point so the linear coefficient directly estimates d log rho/d log r.
        coefficients = np.polyfit(x[begin:end] - x[index], y[begin:end], degree)
        slopes[index] = float(coefficients[-2])
    # The first and last half-window are one-sided and most sensitive to the
    # marker cutoff and radial boundary; omit them from the displayed slope.
    # Hide boundary derivatives that lack a full centered neighborhood.
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
) -> Path:
    """Render one initial-cusp-slope sequence.

    Blue is the evolving numerical density/slope and dotted black is the
    initial condition.  Holding both panels' limits fixed makes changes in the
    central cusp visible as evolution rather than as camera rescaling.
    """
    profiles = load_profiles(input_directory)
    rho_y_min, rho_y_max = density_limits(profiles, r_min, r_max, density_floor_rel)
    time_min, time_max = profiles[0][0], profiles[-1][0]
    output = output_directory / (
        f"wpic_{label}_logrho_slope_t{time_min:g}_to_t{time_max:g}.gif"
    )
    writer = AtomicGif(output, fps)
    axis_log_r_min = math.floor(math.log10(float(np.min(profiles[0][1]["radius"]))))
    axis_log_r_max = math.ceil(math.log10(float(np.max(profiles[0][1]["radius"]))))
    # Measure the initial reference once using exactly the same density selection and derivative routine as later frames.
    initial_rows = profiles[0][1]
    initial_mask = display_mask(initial_rows, r_min, r_max, density_floor_rel)
    initial_radius = np.asarray(initial_rows["radius"][initial_mask], dtype=float)
    initial_density = np.asarray(initial_rows["density"][initial_mask], dtype=float)
    initial_slope_radius, initial_slope = local_poly_log_slope(
        initial_radius,
        initial_density,
    )
    initial_slope_mask = np.isfinite(initial_slope)

    for time, rows in profiles:
        mask = display_mask(rows, r_min, r_max, density_floor_rel)
        radius = np.asarray(rows["radius"][mask], dtype=float)
        density = np.asarray(rows["density"][mask], dtype=float)
        slope_radius, slope = local_poly_log_slope(radius, density)

        figure, (rho_axis, slope_axis) = plt.subplots(1, 2, figsize=(13.0, 5.25), dpi=140)
        figure.subplots_adjust(left=0.10, right=0.975, bottom=0.18, top=0.84, wspace=0.33)

        rho_axis.plot(
            np.log10(radius),
            np.log10(density),
            color="b",
            ls="-",
            lw=2.5,
            zorder=2,
        )
        rho_axis.plot(
            np.log10(initial_radius),
            np.log10(initial_density),
            color="k",
            ls=":",
            lw=2.3,
            zorder=3,
        )
        rho_axis.set_xlim(axis_log_r_min, axis_log_r_max)
        rho_axis.set_ylim(rho_y_min, rho_y_max)
        rho_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        rho_axis.set_ylabel(r"$\log_{10}\rho$", fontsize=24, labelpad=11)
        style_axes(rho_axis, 19)

        slope_mask = np.isfinite(slope)
        slope_axis.plot(
            np.log10(slope_radius[slope_mask]),
            slope[slope_mask],
            color="b",
            ls="-",
            lw=2.5,
            zorder=2,
        )
        slope_axis.plot(
            np.log10(initial_slope_radius[initial_slope_mask]),
            initial_slope[initial_slope_mask],
            color="k",
            ls=":",
            lw=2.3,
            zorder=3,
        )
        slope_axis.axhline(
            -3.0,
            color="0.55",
            ls="--",
            lw=1.7,
            zorder=1,
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
        slope_axis.set_xlim(axis_log_r_min, axis_log_r_max)
        slope_axis.set_ylim(slope_y_min, slope_y_max)
        slope_axis.set_xlabel(r"$\log_{10} r$", fontsize=24, labelpad=9)
        slope_axis.set_ylabel(r"$\mathrm{d}\log\rho/\mathrm{d}\log r$", fontsize=24, labelpad=11)
        style_axes(slope_axis, 19)

        figure.suptitle(rf"$\gamma_i={gamma:g},\qquad t={time:g}$", fontsize=24, y=0.965)
        writer.append(figure_to_rgb(figure))
        plt.close(figure)
    writer.close()

    # Record the actual saved clock shown in each fixed-rate encoded frame.
    times_path = output.with_name(f"{output.stem}_frame_times.txt")
    times_path.write_text("\n".join(f"{index},{time:g}" for index, (time, _) in enumerate(profiles)) + "\n")
    return output


def parse_arguments() -> argparse.Namespace:
    """Define case paths and purely visual radial/slope controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=tuple(CASES), default=[])
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=900.0)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    parser.add_argument("--slope-y-min", type=float, default=-6.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    return parser.parse_args()


# ---- Selected-case command-line dispatch ----------------------------------
def main() -> None:
    """Render each requested initial-slope case and report its GIF path."""
    arguments = parse_arguments()
    labels = arguments.case or list(CASES)
    for label in labels:
        output = render_case(
            arguments.input_root / label,
            arguments.output_directory,
            label,
            CASES[label],
            arguments.fps,
            arguments.r_min,
            arguments.r_max,
            arguments.density_floor_rel,
            arguments.slope_y_min,
            arguments.slope_y_max,
        )
        print(output)


if __name__ == "__main__":
    main()
