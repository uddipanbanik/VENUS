#!/usr/bin/env python3
# Plot the earliest state written by a VENUS setup calculation.
# Inputs are metadata, named radial moments and the projected (r,w) distribution in the solver's own code units.
# Output figures show density, slope, enclosed mass, velocity structure and the recorded initial controls; relativistic compactness appears for Einstein--Vlasov data.

"""Plot the *initial* distribution VENUS actually built from a config.

The figures use the initial state written by VENUS. First do a zero-evolution setup run so VENUS writes its own
genuine t=0 outputs, e.g.::

    ./venus_cpu_einstein_vlasov_eulerian     --config my.ini --t-end 0 --output-dir setup_run
    ./venus_cpu_vlasov_poisson_eulerian       --config my.ini --t-end 0 --output-dir setup_run

That produces ``metadata.json``, ``profiles.csv`` (with a t=0 row) and
``phase_space_initial.csv`` in ``setup_run/``.  This script then plots exactly
those files, so what you see is precisely the state the solver would evolve --
with the solver's normalization and initial-data choices.

It auto-detects the physics model from ``metadata.json``: Einstein-Vlasov gets
the GR panels (compactness 2m/r, since that is meaningful there); Vlasov-Poisson
and Boltzmann-Poisson omit them, because 2m/r has no metric meaning in Newtonian
gravity and would be misleading.

Usage::

    python3 plot_initial.py --input-dir setup_run --output-dir setup_run/ic_figures
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "venus-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Self-contained loaders / helpers (mirrored from the main plotting script so
# this file has no import dependency on it).
# --------------------------------------------------------------------------
def load_named_csv(path: Path) -> np.ndarray:
    """Load a named CSV, preserving a one-row file as a 1-D record array."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data)


def load_phase(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a complete named ``r,w,g`` table onto its Cartesian phase grid."""
    data = load_named_csv(path)
    r = np.unique(data["r"])
    w = np.unique(data["w"])
    # The phase CSV storage order maps radius-major rows onto their complete Cartesian r,w grid.
    g = data["g"].reshape(r.size, w.size)
    return r, w, g


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


def smooth_log_slope(
    r: np.ndarray,
    rho: np.ndarray,
    floor_fraction: float = 1.0e-8,
    smooth_points: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``d log(rho)/d log(r)`` above a relative density floor."""
    threshold = max(float(np.nanmax(rho)) * floor_fraction, np.finfo(float).tiny)
    mask = np.isfinite(r) & np.isfinite(rho) & (r > 0.0) & (rho > threshold)
    r_fit, rho_fit = r[mask], rho[mask]
    if r_fit.size < 3:
        return r_fit, np.full_like(r_fit, np.nan)
    slope = np.gradient(np.log(rho_fit), np.log(r_fit))
    window = min(smooth_points, slope.size if slope.size % 2 == 1 else slope.size - 1)
    if window >= 3:
        kernel = np.ones(window) / float(window)
        slope = np.convolve(slope, kernel, mode="same")
    return r_fit, slope


def style_axes(ax: plt.Axes) -> None:
    """Apply shared inward-tick and spine styling to one axis."""
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", length=9, width=1.7,
                   labelsize=12, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=5, width=1.3,
                   top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# --------------------------------------------------------------------------
# Data selection: the t=0 slice of profiles.csv.
# --------------------------------------------------------------------------
def initial_profile_slice(profiles: np.ndarray) -> np.ndarray:
    """Return the rows of profiles.csv at the earliest recorded time, r-sorted."""
    if "time" in profiles.dtype.names:
        # Select the earliest measured profile clock, then sort its radial rows before calculating diagnostics.
        t0 = float(np.min(profiles["time"]))
        rows = profiles[profiles["time"] == t0]
    else:
        rows = profiles
    order = np.argsort(np.asarray(rows["r"], dtype=float))
    return rows[order]


def read_metadata(input_dir: Path) -> dict:
    """Load metadata.json if present; return {} otherwise."""
    path = input_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def is_einstein_vlasov(meta: dict) -> bool:
    """True only when the run is Einstein-Vlasov (so GR panels are meaningful)."""
    return str(meta.get("physics_model", "")).lower() == "einstein_vlasov"


# --------------------------------------------------------------------------
# Individual initial-condition figures.
# --------------------------------------------------------------------------
def plot_initial_phase(input_dir: Path, outdir: Path) -> Path | None:
    """Standalone initial phase-space projection g(r,w) = log10( int L f dL )."""
    path = input_dir / "phase_space_initial.csv"
    if not path.exists():
        print(f"[skip] initial phase: {path.name} not found.")
        return None
    r, w, g = load_phase(path)
    floor = max(float(np.max(g)), 1.0) * 1.0e-14

    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    # Use the saved grid geometry and a positive display floor for the angular-integrated phase projection.
    image = ax.pcolormesh(
        cell_edges_from_centers(w),
        cell_edges_from_centers(r),
        np.log10(np.maximum(g, floor)),
        shading="auto",
        cmap="magma",
    )
    ax.set_xlabel(r"$w$", fontsize=14)
    ax.set_ylabel(r"$r$", fontsize=14)
    ax.set_title(r"Initial phase space  $\log_{10}\!\int L\,f\,dL$", fontsize=13)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$\log_{10}\left(\int dL\,L\,f\right)$", fontsize=12)
    style_axes(ax)
    outpath = outdir / "ic_phase_space.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_initial_density(rows: np.ndarray, outdir: Path) -> Path | None:
    """Initial density profile, linear-r and log-log side by side."""
    if "rho" not in rows.dtype.names:
        return None
    r = np.asarray(rows["r"], dtype=float)
    rho = np.asarray(rows["rho"], dtype=float)
    good = np.isfinite(r) & np.isfinite(rho) & (rho > 0)
    if np.count_nonzero(good) < 2:
        return None
    r, rho = r[good], rho[good]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    axes[0].semilogy(r, rho, color="navy", lw=2.0)
    axes[0].set_xlabel(r"$r$", fontsize=14)
    axes[0].set_ylabel(r"$\rho$", fontsize=14)
    axes[0].set_title("Initial density (linear $r$)", fontsize=12)

    axes[1].loglog(r, rho, color="navy", lw=2.0)
    axes[1].set_xlabel(r"$r$", fontsize=14)
    axes[1].set_ylabel(r"$\rho$", fontsize=14)
    axes[1].set_title("Initial density (log-log)", fontsize=12)

    for ax in axes:
        style_axes(ax)
    outpath = outdir / "ic_density.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_initial_slope(rows: np.ndarray, outdir: Path) -> Path | None:
    """Initial d log rho / d log r vs log r, to confirm the intended profile."""
    if "rho" not in rows.dtype.names:
        return None
    r = np.asarray(rows["r"], dtype=float)
    rho = np.asarray(rows["rho"], dtype=float)
    r_fit, slope = smooth_log_slope(r, rho)
    if r_fit.size < 3:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(np.log10(r_fit), slope, color="darkgreen", lw=2.0)
    for ref in (-1.0, -2.0, -3.0):
        ax.axhline(ref, ls="--", color="0.6", lw=1.0)
    ax.set_xlabel(r"$\log_{10} r$", fontsize=14)
    ax.set_ylabel(r"$d\log\rho/d\log r$", fontsize=14)
    ax.set_title("Initial density slope", fontsize=12)
    style_axes(ax)
    outpath = outdir / "ic_density_slope.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_initial_mass(
    rows: np.ndarray,
    outdir: Path,
    einstein_vlasov: bool,
    r_min_plot: float | None = 3.0e-2,
) -> Path | None:
    """Initial enclosed mass m(r); for EV also 2m/r (GR compactness).

    Both axes are clamped to where the data actually lives.  The previous
    version floored zero/near-zero values at ``np.finfo(float).tiny`` (~1e-308),
    which let matplotlib autoscale the y-axis down to ~1e-281 and squashed the
    real curve into a flat line at the top.  Here the floor and the y-limits are
    instead taken from the smallest *positive* value of each quantity, and the
    x-axis lower bound is ``r_min_plot`` (default 3e-2) so the crowded innermost
    log cells near r_min do not dominate the view.  Pass ``r_min_plot=None`` to
    let the x-axis autoscale to the full radial range instead.
    """
    if "m" not in rows.dtype.names:
        return None
    r = np.asarray(rows["r"], dtype=float)
    m = np.asarray(rows["m"], dtype=float)
    good = np.isfinite(r) & np.isfinite(m) & (r > 0)
    if np.count_nonzero(good) < 2:
        return None
    r, m = r[good], m[good]

    x_lo = r_min_plot if r_min_plot is not None else float(r.min())
    x_hi = float(r.max())

    n_panels = 2 if einstein_vlasov else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.2),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]

    # --- enclosed mass panel ---
    # Set logarithmic mass limits from actual positive values so empty central samples do not dominate the display scale.
    m_pos = m[m > 0]
    m_floor = float(m_pos.min()) if m_pos.size else 1.0e-12
    axes[0].loglog(r, np.maximum(m, m_floor), color="purple", lw=2.0)
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_ylim(m_floor, float(m.max()) * 2.0)
    axes[0].set_xlabel(r"$r$", fontsize=14)
    axes[0].set_ylabel(r"$m(r)$", fontsize=14)
    axes[0].set_title("Initial enclosed mass", fontsize=12)

    # --- compactness panel (EV only) ---
    if einstein_vlasov:
        # For relativistic data the same areal radius and mass determine compactness; unity is the geometric trapping reference.
        compactness = 2.0 * m / r
        c_pos = compactness[compactness > 0]
        c_floor = float(c_pos.min()) if c_pos.size else 1.0e-12
        axes[1].loglog(r, np.maximum(compactness, c_floor), color="crimson", lw=2.0)
        axes[1].axhline(1.0, ls="--", color="k", lw=1.2)
        axes[1].set_xlim(x_lo, x_hi)
        # Always keep the 2m/r = 1 reference line in view, even if the data
        # sits well below it, so the horizon threshold stays visible.
        axes[1].set_ylim(c_floor, max(float(compactness.max()) * 2.0, 2.0))
        axes[1].set_xlabel(r"$r$", fontsize=14)
        axes[1].set_ylabel(r"$2m(r)/r$", fontsize=14)
        axes[1].set_title("Initial compactness (GR)", fontsize=12)

    for ax in axes:
        style_axes(ax)
    outpath = outdir / "ic_enclosed_mass.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_initial_w_structure(input_dir: Path, outdir: Path) -> Path | None:
    """Mean radial velocity and w-dispersion vs r from the initial phase file.

    Uses g(r,w) = int L f dL as an (angular-momentum-integrated) weight in w to
    show the bulk infall velocity (w_in) and how cold/warm the shell is at each
    radius.  This is a projection, not the full f, but it is exactly what the
    phase-space file stores and is the right thing for reading off infall.
    """
    path = input_dir / "phase_space_initial.csv"
    if not path.exists():
        return None
    r, w, g = load_phase(path)
    weight = np.maximum(g, 0.0)
    # These projected velocity moments use the stored point weights on the w grid. Their interpretation follows that discrete sampling convention.
    norm = weight.sum(axis=1)
    good = norm > 0
    if np.count_nonzero(good) < 2:
        return None

    mean_w = np.full(r.size, np.nan)
    disp_w = np.full(r.size, np.nan)
    mean_w[good] = (weight[good] * w[None, :]).sum(axis=1) / norm[good]
    # Compute the radial spread around the local mean, preserving the distinction between coherent infall and dispersion.
    var = (weight[good] * (w[None, :] - mean_w[good, None]) ** 2).sum(axis=1) / norm[good]
    disp_w[good] = np.sqrt(np.maximum(var, 0.0))

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(r[good], mean_w[good], color="tab:blue", lw=2.0, label=r"$\langle w\rangle(r)$")
    ax.fill_between(
        r[good],
        mean_w[good] - disp_w[good],
        mean_w[good] + disp_w[good],
        color="tab:blue",
        alpha=0.2,
        label=r"$\pm\,\sigma_w(r)$",
    )
    ax.axhline(0.0, ls="--", color="0.5", lw=1.0)
    ax.set_xlabel(r"$r$", fontsize=14)
    ax.set_ylabel(r"$w$", fontsize=14)
    ax.set_title("Initial radial-velocity structure", fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    style_axes(ax)
    outpath = outdir / "ic_w_structure.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


def plot_ic_summary_card(meta: dict, outdir: Path) -> Path | None:
    """A text-only figure echoing the key IC/grid parameters for provenance."""
    if not meta:
        return None
    keys_of_interest = [
        "physics_model", "vp_universe", "vp_alpha", "vp_time_offset",
        "initial_data", "shell_initial_df", "gaussian_density_shell",
        "R0", "dR", "beta", "amp", "w_in",
        "Nr", "Nw", "NL", "r_min", "r_max", "w_max", "L_max",
        "radial_grid", "inner_boundary", "outer_boundary",
        "vp_interpolation", "kinetic_grid",
    ]
    lines = []
    for key in keys_of_interest:
        if key in meta:
            lines.append(f"{key} = {meta[key]}")
    if not lines:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(lines) + 1.0),
                           constrained_layout=True)
    ax.axis("off")
    ax.text(0.02, 0.98, "Initial-condition parameters", fontsize=13,
            fontweight="bold", va="top", ha="left")
    ax.text(0.02, 0.90, "\n".join(lines), fontsize=10, va="top", ha="left",
            family="monospace")
    outpath = outdir / "ic_summary_card.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    return outpath


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------
def safe_call(label, func, *args, **kwargs):
    """Run one figure step; report missing optional files and propagate data errors."""
    try:
        return func(*args, **kwargs)
    except FileNotFoundError as exc:
        print(f"[skip] {label}: missing input file ({exc}).")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing metadata.json, profiles.csv, and "
        "phase_space_initial.csv from a `--t-end 0` setup run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the initial-condition figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(input_dir)
    ev = is_einstein_vlasov(meta)
    model = meta.get("physics_model", "unknown")
    print(f"Detected physics_model = {model}"
          + ("  (GR compactness panel enabled)" if ev
             else "  (Newtonian: skipping GR compactness)"))

    try:
        profiles = load_named_csv(input_dir / "profiles.csv")
        rows = initial_profile_slice(profiles)
    except FileNotFoundError:
        print(f"[warn] profiles.csv not found in {input_dir}; "
              "radial IC panels skipped.")
        rows = None

    written = []
    if rows is not None:
        # Generate radial diagnostics from the selected initial slice, with optional phase/metadata panels handled independently below.
        for label, fn, a in [
            ("ic_density", plot_initial_density, (rows, outdir)),
            ("ic_density_slope", plot_initial_slope, (rows, outdir)),
            ("ic_enclosed_mass", plot_initial_mass, (rows, outdir, ev)),
        ]:
            p = safe_call(label, fn, *a)
            if p is not None:
                written.append(p)

    for label, fn, a in [
        ("ic_phase_space", plot_initial_phase, (input_dir, outdir)),
        ("ic_w_structure", plot_initial_w_structure, (input_dir, outdir)),
        ("ic_summary_card", plot_ic_summary_card, (meta, outdir)),
    ]:
        p = safe_call(label, fn, *a)
        if p is not None:
            written.append(p)

    if written:
        print(f"Wrote {len(written)} initial-condition figures to {outdir.resolve()}:")
        for p in written:
            print(f"  {p.name}")
    else:
        print("No figures written -- check that the setup run produced "
              "profiles.csv / phase_space_initial.csv.")


if __name__ == "__main__":
    main()
