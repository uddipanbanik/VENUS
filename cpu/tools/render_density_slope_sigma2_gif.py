#!/usr/bin/env python3
# Animate spherical density, logarithmic slope and random radial velocity variance from saved moment tables.
# Metadata or an explicit option distinguishes centered pressure from the raw radial second moment.
# Optional neighboring-shell grouping conserves extensive mass, radial momentum and raw second moments before recovering the merged dispersion.

"""Render the standard spherical density/slope/radial-dispersion GIF.

The three panels show ``rho(r)``, its local logarithmic power-law index, and
the centered radial velocity variance ``sigma_r^2``.  Optional adaptive rebinning
combines noisy inner cells by represented particle number while conserving
mass, radial momentum, and second velocity moments.

Input profile tables are grouped by ``time`` and use ``step``, ``r``, ``rho``,
radial momentum density ``j``, radial moment ``p``, and tangential moment
``pt``. Spherical DSMC stores ``p = rho sigma_r^2``; Newtonian grid output
stores ``p = rho <v_r^2>``. Metadata selects this convention automatically;
``--radial-moment-convention`` supplies it explicitly for external tables.
Incomplete radial blocks are excluded before frame selection. RGB
frames are streamed into a sibling temporary GIF; the output is replaced only
after finalization, and failures remove the partial file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from spherical_profile_rendering import (
    add_slope_reference_lines,
    build_target_times,
    fig_to_rgb,
    load_profiles,
    local_poly_log_slope,
    style_axes,
    unique_r_rows,
    write_frame_times,
)
import matplotlib.pyplot as plt
import numpy as np
from PIL import GifImagePlugin, Image


class AtomicStreamingGif:
    """Write full-frame GIF images atomically with bounded memory.

    This is safe for live solver output: the public path is replaced only
    after every selected snapshot has become a complete GIF frame.
    """

    def __init__(self, output: Path, fps: float) -> None:
        """Open a sibling temporary GIF at the requested fixed frame rate."""
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = output.with_name(output.stem + ".tmp.gif")
        self.duration_ms = int(round(1000.0 / max(float(fps), 1.0e-6)))
        self.handle = self.temporary.open("wb")
        self.frame_count = 0

    def append(self, frame: np.ndarray) -> None:
        """Quantize and append one full RGB frame with bounded memory."""
        image = GifImagePlugin._normalize_mode(Image.fromarray(frame))
        # Round cumulative frame duration to the GIF centisecond clock so per-frame rounding does not accumulate an avoidable drift.
        ideal_centiseconds = self.duration_ms / 10.0
        elapsed_before = math.floor(self.frame_count * ideal_centiseconds + 1.0e-12)
        elapsed_after = math.floor(
            (self.frame_count + 1) * ideal_centiseconds + 1.0e-12
        )
        frame_duration_ms = max(10, 10 * (elapsed_after - elapsed_before))
        encoder = {"duration": frame_duration_ms, "disposal": 2}
        image = GifImagePlugin._normalize_palette(image, None, encoder)
        if self.frame_count == 0:
            encoder["loop"] = 0
            for block in GifImagePlugin._get_global_header(image, encoder):
                self.handle.write(block)
        else:
            encoder["include_color_table"] = True
        GifImagePlugin._write_frame_data(
            self.handle,
            image,
            (0, 0),
            encoder,
        )
        self.frame_count += 1

    def close(self) -> None:
        """Finish a nonempty GIF, synchronize it, and replace the output."""
        if self.handle.closed:
            return
        if self.frame_count == 0:
            self.handle.close()
            if self.temporary.exists():
                self.temporary.unlink()
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


def radial_dispersion_squared(
    rows: np.ndarray, moment_convention: str = "central"
) -> np.ndarray:
    """Return the central radial second moment ``<v_r^2>-<v_r>^2``.

    Here ``rho`` is mass density, ``j`` is radial momentum density, and ``p``
    stores either central pressure or a raw second moment. The central
    pressure already includes the subtraction of the local mean flow.
    """
    rho = np.asarray(rows["rho"], dtype=float)
    sigma2 = np.divide(
        rows["p"],
        rho,
        out=np.zeros_like(rho),
        where=rho > 0.0,
    )
    # Subtract mean radial velocity squared only when p/rho is the raw second moment.
    if moment_convention == "raw":
        mean_vr = np.divide(
            rows["j"], rho, out=np.zeros_like(rho), where=rho > 0.0
        )
        sigma2 -= mean_vr * mean_vr
    elif moment_convention != "central":
        raise ValueError("Radial moment convention must be central or raw.")
    return np.maximum(sigma2, 0.0)


def resolve_radial_moment_convention(
    input_dirs: list[Path], metadata_file: str, requested: str
) -> str:
    """Read the solver's pressure convention before combining continuations.

    Spherical DSMC writes centered pressure. Other Newtonian grid methods
    write the raw radial second moment. Tables without metadata retain the
    grid convention; their author can choose central pressure explicitly.
    """
    if requested != "auto":
        return requested
    conventions = set()
    for directory in input_dirs:
        path = directory / metadata_file
        if path.is_file():
            metadata = json.loads(path.read_text())
            conventions.add(
                "central" if metadata.get("bp_collision_operator") == "spherical_dsmc"
                else "raw"
            )
    if len(conventions) > 1:
        raise ValueError("Profile continuations have different radial moment conventions.")
    return next(iter(conventions), "raw")


def conservative_adaptive_rebin(
    rows: np.ndarray,
    particle_mass: float,
    minimum_particles: int,
    maximum_radius: float,
) -> np.ndarray:
    """Merge adjacent spherical cells until they represent enough particles.

    The input radial moment p is centered. Restore its raw second moment
    with p+j^2/rho, then sum the volume-integrated mass, radial momentum
    and second moments. Subtract the merged mean radial velocity squared
    to recover its variance.
    """
    if minimum_particles <= 0:
        return rows
    if not (particle_mass > 0.0):
        raise RuntimeError("Adaptive profile bins require a positive particle mass.")

    rows = unique_r_rows(rows)
    r = np.asarray(rows["r"], dtype=float)
    edges = np.empty(r.size + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = r[-1]
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    masses = np.maximum(np.asarray(rows["rho"], dtype=float), 0.0) * volumes
    # For equal particle masses the requested count corresponds to a definite represented mass in each merged inner bin.
    target_mass = float(minimum_particles) * particle_mass

    cutoff = int(np.searchsorted(r, maximum_radius, side="right"))
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
    # Leave cells beyond the configured adaptive radius separate so only the intended noisy inner region is rebinned.
    groups.extend((index, index + 1) for index in range(cutoff, r.size))

    merged = np.empty(len(groups), dtype=rows.dtype)
    for output_index, (begin, end) in enumerate(groups):
        merged[output_index] = rows[end - 1]
        inner = edges[begin]
        outer = edges[end]
        volume = np.sum(volumes[begin:end])
        mass = np.sum(masses[begin:end])
        rho = mass / volume if volume > 0.0 else 0.0

        source_rho = np.asarray(rows["rho"][begin:end], dtype=float)
        source_j = np.asarray(rows["j"][begin:end], dtype=float)
        source_p = np.asarray(rows["p"][begin:end], dtype=float)
        source_pt = np.asarray(rows["pt"][begin:end], dtype=float)
        source_volume = volumes[begin:end]
        radial_momentum = np.sum(source_j * source_volume)
        # Restore the extensive raw second moment before summing cells with different radial bulk velocities.
        radial_second_moment = np.sum(
            (
                source_p
                + np.divide(
                    source_j * source_j,
                    source_rho,
                    out=np.zeros_like(source_rho),
                    where=source_rho > 0.0,
                )
            )
            * source_volume
        )
        tangential_second_moment = np.sum(source_pt * source_volume)
        mean_vr = radial_momentum / mass if mass > 0.0 else 0.0
        sigma_r2 = (
            max(radial_second_moment / mass - mean_vr * mean_vr, 0.0)
            if mass > 0.0
            else 0.0
        )
        sigma_t2 = (
            max(tangential_second_moment / mass, 0.0)
            if mass > 0.0
            else 0.0
        )
        # The displayed group radius is the volume centroid of its spherical shell.
        representative_r = (
            0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)
            if outer > inner
            else r[end - 1]
        )
        merged["r"][output_index] = representative_r
        merged["rho"][output_index] = rho
        merged["p"][output_index] = rho * sigma_r2
        merged["pt"][output_index] = rho * sigma_t2
        merged["j"][output_index] = rho * mean_vr
    return merged


def canonicalize_restart_snapshots(profiles: np.ndarray) -> np.ndarray:
    """Assign duplicate continuation rows the earliest time for their step."""
    profiles = profiles.copy()
    for step in np.unique(profiles["step"]):
        step_mask = profiles["step"] == step
        # Repeated continuation rows share a canonical time for their step before complete physical slices are selected.
        profiles["time"][step_mask] = float(np.min(profiles["time"][step_mask]))
    return profiles


def complete_times(profiles: np.ndarray) -> np.ndarray:
    """Keep only times containing the full expected set of radial cells."""
    times = np.unique(profiles["time"])
    if times.size == 0:
        return times
    unique_r_counts = np.array(
        [np.unique(profiles["r"][profiles["time"] == time]).size for time in times],
        dtype=int,
    )
    expected = int(np.max(unique_r_counts))
    return times[unique_r_counts == expected]


def render(args: argparse.Namespace) -> Path:
    """Load, conservatively condition, and render the physical profile series."""
    profiles = canonicalize_restart_snapshots(
        np.concatenate(
            [load_profiles(input_dir / args.profiles_file) for input_dir in args.input_dir]
        )
    )
    convention = resolve_radial_moment_convention(
        args.input_dir, args.metadata_file, args.radial_moment_convention
    )
    # Rebinning combines central pressure with the squared mean flow to
    # recover the extensive raw moment. Normalize the input once here.
    if convention == "raw":
        profiles["p"] = (
            profiles["rho"]
            * radial_dispersion_squared(profiles, "raw")
        )
    times = complete_times(profiles)
    if times.size == 0:
        raise RuntimeError("No complete profile snapshots are available.")
    # Obtain the equal-particle mass from explicit input or solver metadata before using a count-based rebinning threshold.
    particle_mass = args.particle_mass
    if args.adaptive_min_particles > 0 and particle_mass is None:
        metadata_path = args.input_dir[0] / args.metadata_file
        with metadata_path.open(encoding="ascii") as handle:
            metadata = json.load(handle)
        particle_mass = (
            float(metadata["bp_dsmc_total_mass"])
            / int(metadata["bp_dsmc_particles"])
        )
    target_times = build_target_times(
        times,
        min(args.frames, times.size),
        args.early_time_max,
        args.early_frames,
        args.frame_sampling,
    )

    # Determine a common dispersion scale from the same selected and rebinned source profiles that will be rendered.
    sigma_log_values: list[np.ndarray] = []
    for time in target_times:
        rows = unique_r_rows(profiles[profiles["time"] == time])
        if args.adaptive_min_particles > 0:
            rows = conservative_adaptive_rebin(
                rows,
                float(particle_mass),
                args.adaptive_min_particles,
                args.adaptive_max_radius,
            )
        sigma2 = radial_dispersion_squared(rows)
        mask = (
            np.isfinite(rows["r"])
            & np.isfinite(sigma2)
            & (rows["r"] >= args.r_min_plot)
            & (rows["r"] <= args.r_max_plot)
            & (sigma2 > 0.0)
        )
        if np.any(mask):
            sigma_log_values.append(np.log10(sigma2[mask]))
    if sigma_log_values:
        all_sigma_log = np.concatenate(sigma_log_values)
        sigma_y_min = (
            float(np.nanpercentile(all_sigma_log, 0.5)) - 0.15
            if args.sigma_log_y_min is None
            else args.sigma_log_y_min
        )
        sigma_y_max = (
            float(np.nanpercentile(all_sigma_log, 99.5)) + 0.15
            if args.sigma_log_y_max is None
            else args.sigma_log_y_max
        )
    else:
        sigma_y_min = -3.0 if args.sigma_log_y_min is None else args.sigma_log_y_min
        sigma_y_max = 1.0 if args.sigma_log_y_max is None else args.sigma_log_y_max

    log_r_min = float(np.log10(args.r_min_plot))
    log_r_max = float(np.log10(args.r_max_plot))
    writer = AtomicStreamingGif(args.output, args.fps)
    last_frame: np.ndarray | None = None
    try:
        for time in target_times:
            rows = unique_r_rows(profiles[profiles["time"] == time])
            if args.adaptive_min_particles > 0:
                rows = conservative_adaptive_rebin(
                    rows,
                    float(particle_mass),
                    args.adaptive_min_particles,
                    args.adaptive_max_radius,
                )
            r = np.asarray(rows["r"], dtype=float)
            rho = np.asarray(rows["rho"], dtype=float)
            sigma2 = radial_dispersion_squared(rows)
            radial_mask = (
                np.isfinite(r)
                & (r >= args.r_min_plot)
                & (r <= args.r_max_plot)
            )
            # Density and dispersion have separate finite-positive masks because one observable can vanish while the other remains sampled.
            density_mask = radial_mask & np.isfinite(rho) & (rho > 0.0)
            sigma_mask = radial_mask & np.isfinite(sigma2) & (sigma2 > 0.0)

            fig, (ax_rho, ax_slope, ax_sigma) = plt.subplots(
                1,
                3,
                figsize=(15.2, 4.35),
                dpi=135,
            )
            fig.subplots_adjust(
                left=0.075,
                right=0.985,
                bottom=0.145,
                top=0.895,
                wspace=0.31,
            )

            ax_rho.plot(
                np.log10(r[density_mask]),
                np.log10(rho[density_mask]),
                color=args.rho_color,
                lw=args.line_width,
            )
            ax_rho.set_xlim(log_r_min, log_r_max)
            ax_rho.set_ylim(args.rho_log_y_min, args.rho_log_y_max)
            ax_rho.set_xlabel(r"$\log_{10} r$", fontsize=args.label_size, labelpad=7)
            ax_rho.set_ylabel(r"$\log_{10}\rho$", fontsize=args.label_size, labelpad=8)
            style_axes(ax_rho)

            # Derive the log slope from the selected radial density using the shared local-polynomial method.
            slope_r, slope = local_poly_log_slope(
                r[radial_mask],
                rho[radial_mask],
                args.slope_floor_rel,
                args.slope_window_points,
                args.slope_poly_order,
            )
            slope_mask = (
                np.isfinite(slope_r)
                & np.isfinite(slope)
                & (slope_r >= args.r_min_plot)
                & (slope_r <= args.r_max_plot)
            )
            ax_slope.plot(
                np.log10(slope_r[slope_mask]),
                slope[slope_mask],
                color=args.slope_color,
                lw=args.line_width,
            )
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
            ax_slope.set_ylabel(
                r"$\mathrm{d}\log\rho/\mathrm{d}\log r$",
                fontsize=args.label_size,
                labelpad=8,
            )
            style_axes(ax_slope)

            ax_sigma.plot(
                np.log10(r[sigma_mask]),
                np.log10(sigma2[sigma_mask]),
                color=args.sigma_color,
                lw=args.line_width,
            )
            ax_sigma.set_xlim(log_r_min, log_r_max)
            ax_sigma.set_ylim(sigma_y_min, sigma_y_max)
            ax_sigma.set_xlabel(r"$\log_{10} r$", fontsize=args.label_size, labelpad=7)
            ax_sigma.set_ylabel(
                r"$\log_{10}\sigma_r^2$",
                fontsize=args.label_size,
                labelpad=8,
            )
            style_axes(ax_sigma)

            fig.suptitle(rf"$t={float(time):.3g}$", fontsize=args.title_size, y=0.975)
            last_frame = fig_to_rgb(fig)
            writer.append(last_frame)
            plt.close(fig)
    # Remove an unfinished GIF on any rendering interruption, preserving the existing completed output.
    except BaseException:
        writer.abort()
        raise
    writer.close()

    # Publish the actual selected clocks and radial-moment convention beside the movie for physical interpretation.
    write_frame_times(args.output, target_times)
    args.output.with_suffix(".moment_convention.json").write_text(
        json.dumps({"input_radial_moment_convention": convention,
                    "rendered_quantity": "centered radial velocity variance",
                    "input_directories": [str(path) for path in args.input_dir]},
                   indent=2) + "\n"
    )
    if args.final_frame_output is not None:
        if last_frame is None:
            raise RuntimeError("No final GIF frame was rendered.")
        Image.fromarray(last_frame).save(args.final_frame_output)
    return args.output


def parse_args() -> argparse.Namespace:
    """Parse input continuations, physical display range, and noise controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--profiles-file", default="profiles.csv")
    parser.add_argument("--metadata-file", default="metadata.json")
    parser.add_argument(
        "--radial-moment-convention", choices=("auto", "central", "raw"),
        default="auto",
        help="Pressure convention: metadata detects spherical DSMC (central); "
             "other Newtonian tables use raw second moments.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-frame-output", type=Path)
    parser.add_argument("--frames", type=int, default=1000000)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-sampling", choices=("index", "time"), default="index")
    parser.add_argument("--early-time-max", type=float)
    parser.add_argument("--early-frames", type=int, default=0)
    parser.add_argument("--r-min-plot", type=float, default=3.0e-3)
    parser.add_argument("--r-max-plot", type=float, default=1.0e3)
    parser.add_argument("--adaptive-min-particles", type=int, default=0)
    parser.add_argument("--adaptive-max-radius", type=float, default=0.3)
    parser.add_argument("--particle-mass", type=float)
    parser.add_argument("--rho-log-y-min", type=float, default=-10.0)
    parser.add_argument("--rho-log-y-max", type=float, default=10.0)
    parser.add_argument("--slope-y-min", type=float, default=-10.0)
    parser.add_argument("--slope-y-max", type=float, default=2.0)
    parser.add_argument("--sigma-log-y-min", type=float)
    parser.add_argument("--sigma-log-y-max", type=float)
    parser.add_argument("--label-size", type=float, default=16.0)
    parser.add_argument("--title-size", type=float, default=16.0)
    parser.add_argument("--rho-color", default="b")
    parser.add_argument("--slope-color", default="r")
    parser.add_argument("--sigma-color", default="#6a2cff")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--slope-floor-rel", type=float, default=0.0)
    parser.add_argument("--slope-window-points", type=int, default=9)
    parser.add_argument("--slope-poly-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Parse rendering controls, generate the animation, and report its path."""
    output = render(parse_args())
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
