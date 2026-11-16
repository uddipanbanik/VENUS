#!/usr/bin/env python3
# Render the benchmark WPIC profiles with playback durations derived from their physical timestamps.
# Density/slope/dispersion arrays and unit conversions come from the shared conservative loader.
# Output is a streamed GIF whose frame holds represent the saved states' midpoint time intervals.

"""Render WPIC profiles at a constant amount of simulation time per second.

Input ``profile_*.csv`` files follow the Frank weighted-PIC schema consumed by
:func:`render_frank_wpic_gif.load_profile`: ``time``, ``radius``, ``density``,
``radial_mean``, ``radial_variance``, and ``neff``.  Profiles from multiple
directories are combined by distinct code time, conservatively grouped and
converted to the benchmark normalization, then drawn by the common
density/slope/dispersion frame routine.

Unlike a fixed-FPS movie, each saved profile owns the interval halfway to its
neighbors in simulation time.  Those Voronoi spans are mapped onto the
requested total playback duration, so equal wall-clock playback represents
equal elapsed simulation time even when output cadence is irregular.  GIF
centisecond rounding is accumulated rather than applied independently, and a
temporary file is atomically renamed only after all frames are complete.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image

from render_frank_wpic_gif import TIME_GYR, frame, load_profile
from spherical_profile_rendering import fig_to_rgb
from render_frank_venus_sphdsmc_gif import text_layout_qa
import matplotlib.pyplot as plt


class SimulationTimeStreamingGif:
    """Stream a variable-duration GIF and publish it after finalization.

    Pillow's low-level writer keeps memory bounded to one RGB frame.  Frame
    delays are quantized cumulatively to the GIF centisecond clock, and the
    public output remains untouched until :meth:`close` writes the trailer,
    flushes the temporary file, and renames it into place.
    """

    def __init__(self, output: Path) -> None:
        """Open a sibling temporary GIF for the requested output path."""
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = output.with_name(output.stem + ".tmp.gif")
        self.handle = self.temporary.open("wb")
        self.frame_count = 0
        self.ideal_elapsed_centiseconds = 0.0
        self.written_elapsed_centiseconds = 0

    def append(self, pixels: np.ndarray, duration_seconds: float) -> None:
        """Append one RGB frame with a cumulatively rounded hold duration."""
        image = GifImagePlugin._normalize_mode(Image.fromarray(pixels))
        # Accumulate ideal total duration before rounding to the GIF clock, limiting cumulative timing drift.
        self.ideal_elapsed_centiseconds += 100.0 * max(duration_seconds, 0.01)
        elapsed_after = max(
            self.written_elapsed_centiseconds + 1,
            int(round(self.ideal_elapsed_centiseconds)),
        )
        frame_duration_ms = 10 * (
            elapsed_after - self.written_elapsed_centiseconds
        )
        self.written_elapsed_centiseconds = elapsed_after
        encoder = {"duration": frame_duration_ms, "disposal": 2}
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
        """Finish a nonempty GIF, synchronize it, and replace the output."""
        if self.frame_count == 0:
            raise RuntimeError("Cannot write a GIF without frames.")
        self.handle.write(b";")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.temporary.replace(self.output)

    def abort(self) -> None:
        """Close and remove the temporary GIF after a rendering failure."""
        if not self.handle.closed:
            self.handle.close()
        if self.temporary.exists():
            self.temporary.unlink()


def frame_durations(
    simulation_times: np.ndarray, playback_seconds: float
) -> tuple[np.ndarray, float]:
    """Map profile-centered simulation-time spans to playback durations.

    Interior boundaries are midpoints between adjacent saved times; the first
    and last boundaries coincide with the sequence endpoints.  The returned
    rate is simulation-time units per playback second and is zero for a
    single-frame sequence.
    """
    if simulation_times.size == 1:
        return np.asarray([playback_seconds]), 0.0
    boundaries = np.empty(simulation_times.size + 1, dtype=float)
    boundaries[0] = simulation_times[0]
    boundaries[-1] = simulation_times[-1]
    # Adjacent-time midpoints divide the represented physical interval among actual saved profiles.
    boundaries[1:-1] = 0.5 * (simulation_times[:-1] + simulation_times[1:])
    # The full physical span divided by requested playback duration sets one common time-to-playback conversion.
    simulation_rate = (
        simulation_times[-1] - simulation_times[0]
    ) / playback_seconds
    return np.diff(boundaries) / simulation_rate, simulation_rate


def main() -> None:
    """Load distinct profiles and render them on a uniform-time movie clock."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--playback-seconds", type=float, default=60.0)
    parser.add_argument("--particles", type=int, default=4_000_000)
    arguments = parser.parse_args()
    if not (arguments.playback_seconds > 0.0):
        raise ValueError("--playback-seconds must be positive")

    paths = sorted(
        {path for directory in arguments.profile_dir for path in directory.glob("profile_*.csv")}
    )
    if not paths:
        raise RuntimeError("No profile_*.csv files were found.")
    profiles = [load_profile(path) for path in paths]
    # Merge continuation directories by distinct actual clock before assigning durations.
    profiles = sorted(
        {round(time, 12): (time, curve) for time, curve in profiles}.values()
    )
    times = np.asarray([profile[0] for profile in profiles], dtype=float)
    durations, simulation_rate = frame_durations(times, arguments.playback_seconds)
    initial = profiles[0][1]

    writer = SimulationTimeStreamingGif(arguments.output)
    try:
        for index, ((time_code, curve), duration) in enumerate(
            zip(profiles, durations)
        ):
            figure, axes = frame(
                curve,
                initial,
                time_code * TIME_GYR,
                arguments.particles,
            )
            if index == 0:
                text_layout_qa(figure, axes)
            # Hold each unchanged source profile for its assigned interval. The data themselves are never interpolated.
            writer.append(fig_to_rgb(figure), float(duration))
            plt.close(figure)
        writer.close()
    # Remove the temporary output after a failed frame or finalization so a partial movie is not published.
    except Exception:
        writer.abort()
        raise

    print(
        f"wrote {arguments.output} ({len(profiles)} frames, "
        f"t={times[-1] * TIME_GYR:.6f} Gyr, "
        f"rate={simulation_rate * TIME_GYR:.8f} Gyr/s)"
    )


if __name__ == "__main__":
    main()
