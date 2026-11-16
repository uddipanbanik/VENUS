#!/usr/bin/env python3
# Overlay saved radial density profiles with color proportional to their actual physical times.
# Input CSVs supply time, radius and density in the run's code units; metadata supplies the mesh limits when available.
# The output image uses positive density above the requested relative display floor.

"""Time-stacked density waterfall: log rho(r) for every profile on one plot.

This is the single-figure equivalent of the density panel of the WPIC gamma
GIF.  Every saved profile's log10(rho) versus log10(r) curve is drawn on one
set of axes, coloured by its physical time, with a colourbar mapping colour to
time.  The colour scale is linear in time (t_min to t_max), so a curve's colour
reflects its true time even when the output cadence is irregular.

Reads the weighted-PIC radial-profile schema (time, radius, density, ..., neff)
and, like the gamma renderer, takes gamma_i, radial_minimum/maximum and
particles_per_radial_cell from metadata.json when present.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np


def read_metadata(directory: Path) -> dict:
    """Return metadata.json as a dict, or empty when absent/unreadable."""
    path = directory / "metadata.json"
    if not path.is_file():
        return {}
    try:
        with path.open() as handle:
            return json.load(handle)
    except Exception:
        return {}


def load_density(path: Path, floor_rel: float) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Return (time, log10 r, log10 rho) for one profile, floored and sorted."""
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not {"time", "radius", "density"}.issubset(rows.dtype.names or ()):
        return None
    time = float(rows["time"][0])
    r = np.asarray(rows["radius"], dtype=float)
    rho = np.asarray(rows["density"], dtype=float)
    # Positive finite radius and density admit logarithms. The relative floor is a display selection tied to this profile peak.
    finite = np.isfinite(r) & (r > 0.0) & np.isfinite(rho) & (rho > 0.0)
    if np.count_nonzero(finite) < 2:
        return None
    r, rho = r[finite], rho[finite]
    peak = float(np.max(rho))
    floor = max(peak * floor_rel, np.finfo(float).tiny)
    keep = rho >= floor
    if np.count_nonzero(keep) < 2:
        return None
    r, rho = r[keep], rho[keep]
    order = np.argsort(r)
    return time, np.log10(r[order]), np.log10(rho[order])


def main() -> None:
    """Draw every profile's density curve on one axis, coloured by time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run output directory with profile_*.csv + metadata.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG (default: <run>_density_waterfall.png in --output-directory)")
    parser.add_argument("--output-directory", type=Path, default=None,
                        help="Directory for the auto-named PNG when --output is omitted")
    parser.add_argument("--number", type=int, default=20,
                        help="Number of profiles to plot, spread evenly over the "
                             "time interval (first and last always included)")
    parser.add_argument("--colorscheme", type=str, default="plasma",
                        choices=("plasma", "turbo", "coolwarm", "plasma", "jet"),
                        help="Time->colour map (default: viridis, the astro/cosmo "
                             "perceptually-uniform standard)")
    parser.add_argument("--line-width", type=float, default=1.1)
    parser.add_argument("--density-floor-rel", type=float, default=1.0e-10)
    arguments = parser.parse_args()

    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return 0

    paths = sorted(arguments.run_dir.glob("profile_*.csv"), key=step_of)
    if not paths:
        raise RuntimeError(f"No profile_*.csv files in {arguments.run_dir}")

    # Load every usable profile first, then select by UNIFORM TIME below.
    all_curves = [load_density(p, arguments.density_floor_rel) for p in paths]
    all_curves = [c for c in all_curves if c is not None]
    if not all_curves:
        raise RuntimeError("No profile yielded a usable density curve")
    all_curves.sort(key=lambda c: c[0])
    all_times = np.asarray([c[0] for c in all_curves], dtype=float)

    # Collapse duplicate times (keep the last, as after a restart continuation).
    # A repeated saved epoch contributes its last encountered profile, then the time order is restored.
    unique = {}
    for curve in all_curves:
        unique[round(curve[0], 12)] = curve
    all_curves = [unique[k] for k in sorted(unique)]
    all_times = np.asarray([c[0] for c in all_curves], dtype=float)

    number = max(2, int(arguments.number))
    if number >= all_times.size:
        # Asking for as many (or more) than exist: plot them all.
        curves = all_curves
    else:
        # Choose ``number`` targets spaced evenly in time, each represented by
        # its nearest available profile; endpoints are always retained.
        t_lo, t_hi = float(all_times[0]), float(all_times[-1])
        # Uniform target times select nearest existing profiles. Their measured clocks determine colors and the first/last states remain visible.
        chosen = {0, all_times.size - 1}
        for index in range(number):
            target = t_lo + (t_hi - t_lo) * index / (number - 1)
            chosen.add(int(np.argmin(np.abs(all_times - target))))
        curves = [all_curves[i] for i in sorted(chosen)]

    times = np.asarray([c[0] for c in curves], dtype=float)
    t_min, t_max = float(times[0]), float(times[-1])

    metadata = read_metadata(arguments.run_dir)
    gamma = metadata.get("gamma_i")
    ppc = metadata.get("particles_per_radial_cell")
    r_lo = metadata.get("radial_minimum")
    r_hi = metadata.get("radial_maximum")

    # Colour normalisation: linear in time.
    # Linear normalization maps physical elapsed time to color even when the selected output intervals differ.
    norm = Normalize(vmin=t_min, vmax=t_max if t_max > t_min else t_min + 1.0)
    cmap = plt.get_cmap(arguments.colorscheme)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=140)
    for time, log_r, log_rho in curves:
        ax.plot(log_r, log_rho, color=cmap(norm(time)),
                lw=arguments.line_width, alpha=0.9, zorder=2)

    # x-limits from the run's true mesh when available, else from the data.
    # Use the physical mesh limits when metadata provides them so changing particle support does not change the plotted domain.
    if r_lo is not None and r_hi is not None and float(r_lo) > 0.0:
        ax.set_xlim(math.log10(float(r_lo)), math.log10(float(r_hi)))
    ax.set_xlabel(r"$\log_{10} r$", fontsize=18, labelpad=8)
    ax.set_ylabel(r"$\log_{10}\rho$", fontsize=18, labelpad=9)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=13)
    ax.grid(True, which="major", alpha=0.3, lw=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label("time", fontsize=15, labelpad=8)
    cbar.ax.tick_params(labelsize=12)

    heading = (rf"$\gamma={float(gamma):g}$: Density Evolution"
               if gamma is not None else "Density Evolution")
    fig.suptitle(heading, fontsize=18, y=0.97)

    fig.subplots_adjust(left=0.11, right=0.99, top=0.92, bottom=0.11)

    if arguments.output is not None:
        output = arguments.output
    else:
        out_dir = arguments.output_directory or arguments.run_dir
        label = ("gamma" + f"{float(gamma):g}".replace(".", "p").replace("-", "m")
                 if gamma is not None else arguments.run_dir.name)
        ppc_tag = f"_{int(ppc)}ppc" if ppc is not None else ""
        output = out_dir / (
            f"wpic_{label}{ppc_tag}_density_waterfall_t{t_min:g}_to_t{t_max:g}.png"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output} ({len(curves)} profiles, t={t_min:g} to {t_max:g})")


if __name__ == "__main__":
    main()
