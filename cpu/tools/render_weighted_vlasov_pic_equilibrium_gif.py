#!/usr/bin/env python3
# Compare deposited Plummer density, logarithmic slope and radial variance to the analytic equilibrium.
# Named snapshot CSVs use G=M=a=1 and one common ordered radial mesh.
# The streamed GIF uses fixed sequence-wide scales and preserves gaps at poorly sampled derivative points.

"""Render weighted-PIC Plummer radial snapshots as a three-panel GIF.

Each frame compares the deposited numerical density, logarithmic density
slope, and radial velocity dispersion with the analytic Plummer equilibrium.
The comparison is therefore both a physics test of stationary
Vlasov--Poisson evolution and a visual check of marker noise.

Each ``weighted_pic_plummer_*.csv`` file contains one time on a common radial
grid and must provide ``time``, ``radius``, ``density``, ``radial_variance``,
and effective marker count ``neff``.  The slope is derived only where sampling
is adequate, while analytic density, slope, and dispersion are evaluated in
``G=M=a=1`` units.  Fixed limits are computed over the complete sequence.

Pillow draws and quantizes one full frame at a time.  Frames are streamed to a
sibling temporary GIF, which is synchronized and atomically renamed only after
its trailer is written; failures close and remove the temporary file.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image, ImageDraw, ImageFont


REGULAR_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
BOLD_FONT_CANDIDATES = (
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def load_font(size: int, bold: bool = False):
    """Load a portable TrueType font, with Pillow's bitmap font as fallback."""

    environment_name = "VENUS_BOLD_FONT" if bold else "VENUS_FONT"
    candidates = ((os.environ[environment_name],) if environment_name in os.environ else ())
    candidates += BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


class AtomicStreamingGif:
    """Stream complete frames and publish the GIF only after finalization."""

    def __init__(self, output, fps):
        """Open a sibling temporary GIF and validate the requested frame rate."""
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError("--fps must be finite and positive")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.output.with_name(self.output.stem + ".tmp.gif")
        self.duration_ms = int(round(1000.0 / max(float(fps), 1.0e-6)))
        self.handle = self.temporary.open("wb")
        self.frame_count = 0

    def append(self, frame):
        """Quantize and append one complete Pillow image to the GIF stream."""
        image = frame.quantize(colors=128, dither=Image.Dither.NONE)
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

    def close(self):
        """Finish a nonempty GIF, synchronize it, and replace the output."""
        if self.frame_count == 0:
            raise RuntimeError("Cannot write a GIF without frames")
        self.handle.write(b";")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.temporary.replace(self.output)

    def abort(self):
        """Close and remove a partial GIF after a rendering failure."""
        if not self.handle.closed:
            self.handle.close()
        if self.temporary.exists():
            self.temporary.unlink()


def parse_arguments():
    """Parse the snapshot tree, animation path, and optional marker label."""
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_directory")
    parser.add_argument("output_gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--marker-count", type=int, default=0)
    return parser.parse_args()


def load_snapshots(directory):
    """Load all complete Plummer profile snapshots in step-name order."""
    paths = sorted(glob.glob(os.path.join(directory, "weighted_pic_plummer_*.csv")))
    if not paths:
        raise RuntimeError("No weighted PIC Plummer snapshots found")
    snapshots = []
    reference_radius = None
    required = {
        "time", "radius", "density", "radial_variance", "neff"
    }
    for path in paths:
        snapshot = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        if not required.issubset(snapshot.dtype.names or ()):
            raise RuntimeError(f"{path} is missing required columns {sorted(required)}")
        radius = np.asarray(snapshot["radius"], dtype=float)
        if (
            radius.size < 3
            or not np.all(np.isfinite(radius))
            or not np.all(radius > 0.0)
            or not np.all(np.diff(radius) > 0.0)
        ):
            raise RuntimeError(f"{path} has an invalid radial grid")
        # Require all saved states to share the first ordered radial grid before comparing a common analytic profile.
        if reference_radius is None:
            reference_radius = radius
        elif radius.shape != reference_radius.shape or not np.allclose(
            radius, reference_radius, rtol=1.0e-12, atol=0.0
        ):
            raise RuntimeError(f"{path} does not use the first snapshot's radial grid")
        if not np.allclose(snapshot["time"], snapshot["time"][0], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"{path} mixes multiple snapshot times")
        snapshots.append(snapshot)
    return snapshots


def logarithmic_slope(radius, density, neff):
    """Evaluate the local density power-law index where sampling is adequate.

    ``neff`` is the effective deposited marker count.  Cells with fewer than
    four effective markers are hidden because their finite-difference slope
    is dominated by shot noise rather than resolved Vlasov evolution.
    """
    positive = np.maximum(density, np.finfo(float).tiny)
    slope = np.gradient(np.log10(positive), np.log10(radius))
    # The effective-count threshold marks insufficient local slope sampling while positive density remains a separate observable.
    slope[(density <= 0.0) | (neff < 4.0)] = np.nan
    return slope


def padded_limits(series, lower_quantile=0.01, upper_quantile=0.99, padding=0.08):
    """Choose fixed panel limits from all finite numerical/analytic values."""
    # Choose shared panel limits from all finite numerical and analytic values, using the declared quantile range.
    finite = np.concatenate([values[np.isfinite(values)] for values in series])
    lower, upper = np.quantile(finite, [lower_quantile, upper_quantile])
    span = max(upper - lower, 1.0e-12)
    return float(lower - padding * span), float(upper + padding * span)


def text_size(draw, text, font):
    """Return the rendered text box used for deterministic label placement."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered(draw, position, text, font, fill):
    """Draw text centered on a requested pixel position."""
    width, height = text_size(draw, text, font)
    draw.text(
        (position[0] - width / 2.0, position[1] - height / 2.0),
        text,
        font=font,
        fill=fill,
    )


def draw_rotated_label(image, center, text, font, fill):
    """Draw one vertical panel label without relying on a GUI backend."""
    scratch = Image.new("RGBA", (360, 55), (255, 255, 255, 0))
    draw = ImageDraw.Draw(scratch)
    draw_centered(draw, (180, 27), text, font, fill)
    rotated = scratch.rotate(90, expand=True)
    image.alpha_composite(
        rotated,
        (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)),
    )


def transform(value, lower, upper, pixel_lower, pixel_upper):
    """Map one data coordinate linearly into the panel's pixel coordinate."""
    return pixel_lower + (value - lower) / (upper - lower) * (pixel_upper - pixel_lower)


def line_segments(x, y, x_limits, y_limits, bounds):
    """Split a curve at non-finite samples and convert it to pixel segments."""
    left, top, right, bottom = bounds
    points = []
    segments = []
    for x_value, y_value in zip(x, y):
        # Split plotted lines at missing samples so a gap in physical support is not bridged by an artificial segment.
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            if len(points) >= 2:
                segments.append(points)
            points = []
            continue
        px = transform(x_value, x_limits[0], x_limits[1], left, right)
        py = transform(y_value, y_limits[0], y_limits[1], bottom, top)
        points.append((px, py))
    if len(points) >= 2:
        segments.append(points)
    return segments


def draw_dashed(draw, points, fill, width=3, dash=9):
    """Draw the analytic reference with a backend-independent dashed stroke."""
    for first, second in zip(points[:-1], points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0e-12)
        position = 0.0
        while position < length:
            end = min(position + dash, length)
            if int(position / dash) % 2 == 0:
                draw.line(
                    (
                        first[0] + dx * position / length,
                        first[1] + dy * position / length,
                        first[0] + dx * end / length,
                        first[1] + dy * end / length,
                    ),
                    fill=fill,
                    width=width,
                )
            position = end


def draw_panel(
    image,
    bounds,
    x,
    simulation,
    analytic,
    x_limits,
    y_limits,
    y_label,
    fonts,
):
    """Draw one numerical-versus-analytic radial diagnostic panel."""
    draw = ImageDraw.Draw(image)
    foreground = (31, 37, 43, 255)
    grid = (218, 222, 226, 255)
    simulation_color = (0, 114, 178, 255)
    analytic_color = (213, 94, 0, 255)
    left, top, right, bottom = bounds

    x_ticks = np.linspace(np.ceil(x_limits[0] * 2.0) / 2.0,
                          np.floor(x_limits[1] * 2.0) / 2.0,
                          int(round((np.floor(x_limits[1] * 2.0)
                                     - np.ceil(x_limits[0] * 2.0)) / 1.0)) + 1)
    y_ticks = np.linspace(y_limits[0], y_limits[1], 5)
    for tick in x_ticks:
        px = transform(tick, x_limits[0], x_limits[1], left, right)
        draw.line((px, top, px, bottom), fill=grid, width=1)
        draw_centered(draw, (px, bottom + 20), f"{tick:.1f}", fonts["tick"], foreground)
    for tick in y_ticks:
        py = transform(tick, y_limits[0], y_limits[1], bottom, top)
        draw.line((left, py, right, py), fill=grid, width=1)
        label = f"{tick:.1f}"
        width, height = text_size(draw, label, fonts["tick"])
        draw.text((left - width - 9, py - height / 2), label, font=fonts["tick"], fill=foreground)

    draw.line((left, top, left, bottom), fill=foreground, width=2)
    draw.line((left, bottom, right, bottom), fill=foreground, width=2)
    analytic_segments = line_segments(x, analytic, x_limits, y_limits, bounds)
    for segment in analytic_segments:
        draw_dashed(draw, segment, analytic_color, width=3)
    simulation_segments = line_segments(x, simulation, x_limits, y_limits, bounds)
    for segment in simulation_segments:
        draw.line(segment, fill=simulation_color, width=3, joint="curve")
        for point in segment:
            draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                fill=simulation_color,
            )
    draw_centered(
        draw,
        ((left + right) / 2.0, bottom + 49),
        "log10 r",
        fonts["label"],
        foreground,
    )
    draw_rotated_label(
        image,
        (left - 55, (top + bottom) / 2.0),
        y_label,
        fonts["label"],
        foreground,
    )


def main():
    """Load snapshots, derive Plummer references, and publish the animation."""
    # Analytic Plummer references in G=M=a=1 units:
    # rho=3(1+r^2)^(-5/2)/(4 pi), dlnrho/dlnr=-5r^2/(1+r^2),
    # and sigma_r^2=1/[6 sqrt(1+r^2)] for the isotropic equilibrium.
    arguments = parse_arguments()
    if arguments.fps <= 0:
        raise ValueError("--fps must be positive")
    if arguments.marker_count < 0:
        raise ValueError("--marker-count must be non-negative")
    snapshots = load_snapshots(arguments.snapshot_directory)
    radius = snapshots[0]["radius"]
    x = np.log10(radius)
    # Evaluate the exact Plummer density, its analytic log derivative and isotropic Jeans variance on the saved radii.
    analytic_density = 3.0 / (4.0 * np.pi) * (1.0 + radius * radius) ** -2.5
    analytic_slope = -5.0 * radius * radius / (1.0 + radius * radius)
    analytic_variance = 1.0 / (6.0 * np.sqrt(1.0 + radius * radius))

    density_series = []
    slope_series = []
    variance_series = []
    for snapshot in snapshots:
        density = snapshot["density"]
        variance = snapshot["radial_variance"]
        log_density = np.full_like(density, np.nan)
        np.log10(density, out=log_density, where=density > 0.0)
        density_series.append(log_density)
        slope_series.append(logarithmic_slope(radius, density, snapshot["neff"]))
        log_variance = np.full_like(variance, np.nan)
        np.log10(variance, out=log_variance, where=variance > 0.0)
        variance_series.append(log_variance)

    analytic = [
        np.log10(analytic_density),
        analytic_slope,
        np.log10(analytic_variance),
    ]
    # Use the complete time sequence to keep visual scaling fixed as equilibrium drift and marker noise evolve.
    y_limits = [
        padded_limits(density_series + [analytic[0]], 0.0, 1.0, 0.04),
        padded_limits(slope_series + [analytic[1]], 0.01, 0.99, 0.08),
        padded_limits(variance_series + [analytic[2]], 0.01, 0.99, 0.08),
    ]
    series = [density_series, slope_series, variance_series]
    labels = ["log10 density", "d log density / d log r", "log10 sigma_r^2"]

    fonts = {
        "title": load_font(25, bold=True),
        "label": load_font(18),
        "tick": load_font(14),
        "small": load_font(14),
    }
    width = 1500
    height = 540
    panel_width = 405
    panel_gap = 84
    panel_left = 90
    bounds = [
        (
            panel_left + panel * (panel_width + panel_gap),
            78,
            panel_left + panel * (panel_width + panel_gap) + panel_width,
            443,
        )
        for panel in range(3)
    ]
    x_limits = (float(x[0]), float(x[-1]))
    writer = AtomicStreamingGif(arguments.output_gif, arguments.fps)

    # Render every saved physical time with limits fixed across the movie, so
    # apparent profile evolution cannot be produced by per-frame autoscaling.
    try:
        for frame, snapshot in enumerate(snapshots):
            image = Image.new("RGBA", (width, height), (250, 251, 252, 255))
            draw = ImageDraw.Draw(image)
            marker_label = (
                f"     {arguments.marker_count:,} markers"
                if arguments.marker_count > 0
                else ""
            )
            draw_centered(
                draw,
                (width / 2.0, 27),
                "Weighted PIC Plummer equilibrium"
                + marker_label
                + f"     t = {float(snapshot['time'][0]):.2f}",
                fonts["title"],
                (25, 31, 37, 255),
            )
            for panel in range(3):
                draw_panel(
                    image,
                    bounds[panel],
                    x,
                    series[panel][frame],
                    analytic[panel],
                    x_limits,
                    y_limits[panel],
                    labels[panel],
                    fonts,
                )

            legend_x = bounds[0][0] + 17
            legend_y = bounds[0][1] + 17
            draw.line(
                (legend_x, legend_y, legend_x + 34, legend_y),
                fill=(0, 114, 178, 255),
                width=3,
            )
            draw.text(
                (legend_x + 42, legend_y - 8),
                "weighted PIC",
                font=fonts["small"],
                fill=(31, 37, 43, 255),
            )
            draw_dashed(
                draw,
                [(legend_x, legend_y + 23), (legend_x + 34, legend_y + 23)],
                (213, 94, 0, 255),
                width=3,
            )
            draw.text(
                (legend_x + 42, legend_y + 15),
                "analytic Plummer",
                font=fonts["small"],
                fill=(31, 37, 43, 255),
            )
            footer = "Raw centered slope shown where N_eff >= 4"
            footer_width, footer_height = text_size(draw, footer, fonts["small"])
            draw.text(
                ((width - footer_width) / 2.0, height - footer_height - 5),
                footer,
                font=fonts["small"],
                fill=(80, 87, 94, 255),
            )
            # Encode one completed RGB image at a time, keeping movie memory independent of the number of snapshots.
            writer.append(image.convert("RGB"))
        writer.close()
    except BaseException:
        writer.abort()
        raise


if __name__ == "__main__":
    main()
