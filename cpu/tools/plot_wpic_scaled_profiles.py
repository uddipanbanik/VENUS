#!/usr/bin/env python3
# Compare radial density shapes using a resolved enclosed-mass radius measured independently at every saved epoch.
# Inputs are WPIC profile CSVs on a complete logarithmic mesh. Output is a raw/rescaled overlay and optional fixed-band log slope.
# The inner carrier lies outside this resolved-shell mass normalization.

"""
Compare WPIC profiles after scaling radius by a resolved enclosed-mass radius.

For each selected snapshot it measures a size R(t) from the state itself
(the radius enclosing a fixed fraction of the mass -- no growth law assumed),
then plots density against the rescaled radius r / R(t). The optional density scaling compares shapes in units M_resolved/R(t)^3.
A self-similar interpretation also requires stable scaled profiles and
resolution, time, and boundary checks. The inner carrier has a separate mass budget.

Each input file is one snapshot profile with (at least) a header row naming columns:
    time, radius, density, neff
matching the WPIC profile_XXXXX.csv format. Column names are overridable via flags.

Dependencies: numpy and matplotlib only (no pandas).

Typical use:
    python overlay.py "profile_*.csv" -n 6 -f 0.1 -o overlay.png

Everything the script needs is taken from the terminal as flags; run with -h to see them.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-matplotlib-cache"))
from plot_subhalo_virial_1 import shell_volumes

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay size-normalized density profiles to test self-similar collapse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("profiles", nargs="+",
                   help="Profile CSV files or glob patterns, e.g. 'profile_*.csv'. "
                        "Quote the pattern so the script (not the shell) expands it.")
    p.add_argument("-n", "--num", type=int, default=0,
                   help="Number of snapshots to overlay. They are picked evenly spaced "
                        "in TIME across all the files found. 0 means use every file.")
    p.add_argument("-f", "--frac", type=float, default=0.1,
                   help="Mass fraction that defines the size R(t): R is the radius "
                        "enclosing this fraction of the total mass. Must be in (0,1).")
    p.add_argument("--neff-min", type=float, default=20.0,
                   help="Minimum effective marker count for a bin to be drawn/trusted. "
                        "Bins below this are dropped from the plot.")
    p.add_argument("--slope-window", type=str, default="",
                   help="Optional 'lo,hi' in RESCALED radius (r/R) units; if given, a "
                        "log-log slope is fit over that window per snapshot and printed.")
    p.add_argument("--panels", choices=["both", "rescaled", "raw"], default="both",
                   help="Which panels to draw: raw (density vs r), rescaled (density vs r/R), or both.")
    p.add_argument("--normalize-density", action="store_true",
                   help="Plot density in units of resolved mass / R(t)^3.")
    p.add_argument("--time-col", default="time")
    p.add_argument("--radius-col", default="radius")
    p.add_argument("--density-col", default="density")
    p.add_argument("--neff-col", default="neff")
    p.add_argument("--xlim", default="", help="Optional 'lo,hi' x-limits for the RESCALED panel.")
    p.add_argument("--ylim", default="", help="Optional 'lo,hi' y-limits (density) for both panels.")
    p.add_argument("-o", "--output", default="overlay.png", help="Output image path.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def expand_files(patterns):
    files = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else ([pat] if os.path.exists(pat) else []))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    if not out:
        sys.exit("No matching files found. Check the path/pattern (quote globs).")
    return out


def read_profile(path, cols):
    """Read a CSV with a header row into a numpy structured array (no pandas)."""
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    except Exception as e:
        sys.exit(f"{path}: could not read as CSV ({e}).")
    data = np.atleast_1d(data)               # guard single-row files
    names = data.dtype.names or ()
    for c in cols:
        if c not in names:
            sys.exit(f"{path}: missing expected column '{c}'. "
                     f"Found {list(names)}. Override with the *-col flags.")
    return data


def col(data, name):
    return np.asarray(data[name], dtype=float)


def size_from_mass_fraction(r, rho, frac):
    """R = radius enclosing `frac` of the mass, from M(<r) = int 4 pi r^2 rho dr."""
    order = np.argsort(r)
    r, rho = r[order], rho[order]
    if np.any(~np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError("Require finite nonnegative density on the complete log mesh")
    # Integrate the complete nonnegative density array with spherical volumes before applying a plotting sampling cut.
    mass = rho * shell_volumes(r)
    cumulative = np.r_[0.0, np.cumsum(mass)]
    total = cumulative[-1]
    if not total > 0:
        raise ValueError("A resolved mass radius requires positive resolved mass")
    q = r[1] / r[0]
    edges = np.r_[2 * r / (1 + q), 2 * r[-1] * q / (1 + q)]
    # Find the shell enclosing the requested fraction of resolved mass, then use constant density inside that shell.
    target = frac * total
    i = min(int(np.searchsorted(cumulative, target, side="right") - 1), r.size - 1)
    # Uniform cell density makes enclosed mass linear in r^3 inside a shell.
    fraction = (target - cumulative[i]) / mass[i]
    # Spherical volume is linear in r cubed. Interpolation in that variable gives the enclosed-mass radius.
    radius = (edges[i]**3 + fraction * (edges[i + 1]**3 - edges[i]**3))**(1.0 / 3.0)
    return float(radius), float(total)


def pick_snapshots(times, n):
    if n <= 0 or n >= len(times):
        return list(range(len(times)))
    # Uniform requested clocks select existing nearest snapshots, preserving their measured times.
    targets = np.linspace(times.min(), times.max(), n)
    return sorted({int(np.argmin(np.abs(times - t))) for t in targets})


def parse_pair(s):
    lo, hi = (float(x) for x in s.split(","))
    return lo, hi


def main():
    a = parse_args()
    if not (0.0 < a.frac < 1.0):
        sys.exit("--frac must be between 0 and 1.")

    cols = [a.time_col, a.radius_col, a.density_col, a.neff_col]
    files = expand_files(a.profiles)

    # Read every file's time stamp so snapshots can be picked evenly in time.
    recs = []
    for f in files:
        data = read_profile(f, cols)
        recs.append((f, float(col(data, a.time_col)[0]), data))
    recs.sort(key=lambda x: x[1])
    all_times = np.array([t for _, t, _ in recs])
    sel = pick_snapshots(all_times, a.num)
    chosen = [recs[i] for i in sel]

    print(f"Found {len(files)} files; overlaying {len(chosen)} snapshots.")
    print(f"R(t) = radius enclosing {a.frac*100:.0f}% of mass.\n")
    header = f"{'time':>12} {'R(t)':>12} {'Mtot':>10}"
    if a.slope_window:
        header += f" {'slope':>8}"
    print(header)

    slope_win = parse_pair(a.slope_window) if a.slope_window else None

    panels = ["raw", "rescaled"] if a.panels == "both" else [a.panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.2), squeeze=False)
    axes = axes[0]
    norm = Normalize(vmin=all_times.min(), vmax=all_times.max())
    cmap = cm.plasma

    for f, t, data in chosen:
        r = col(data, a.radius_col)
        rho = col(data, a.density_col)
        neff = col(data, a.neff_col)

        R, Mtot = size_from_mass_fraction(r, rho, a.frac)
        # The effective-count cut controls plotted density samples after the full resolved mass radius has been measured.
        m = (neff >= a.neff_min) & (rho > 0) & (r > 0)
        rr, dd = r[m], rho[m]
        if a.normalize_density:
            # Express density in units M_resolved/R cubed so curves compare dimensionless shapes.
            dd = dd * R**3 / Mtot
        color = cmap(norm(t))

        line = f"{t:12.4g} {R:12.4g} {Mtot:10.4f}"
        if slope_win and np.isfinite(R):
            x = rr / R
            w = (x >= slope_win[0]) & (x <= slope_win[1])
            if w.sum() >= 2:
                # A straight-line fit of log density against log rescaled radius estimates the slope over the explicitly requested interval.
                s = np.polyfit(np.log10(x[w]), np.log10(dd[w]), 1)[0]
                line += f" {s:8.3f}"
            else:
                line += f" {'n/a':>8}"
        print(line)

        for ax, kind in zip(axes, panels):
            xvals = rr / R if (kind == "rescaled" and np.isfinite(R)) else rr
            ax.plot(xvals, dd, lw=1.3, color=color, label=f"t={t:.4g}")

    for ax, kind in zip(axes, panels):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$r / R(t)$" if kind == "rescaled" else r"$r$")
        ax.set_ylabel(r"$\rho R^3/M_{\rm resolved}$" if a.normalize_density else r"$\rho$")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.set_title("size-normalized" if kind == "rescaled" else "raw")
        if a.ylim:
            ax.set_ylim(*parse_pair(a.ylim))
        if kind == "rescaled" and a.xlim:
            ax.set_xlim(*parse_pair(a.xlim))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=list(axes), label="time", fraction=0.046, pad=0.02)
    fig.suptitle(f"Scaled profile comparison  (R = {a.frac*100:.0f}% mass radius, "
                 f"neff >= {a.neff_min:g})")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=a.dpi, bbox_inches="tight")
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
