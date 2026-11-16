#!/usr/bin/env python3
# Animate WPIC density, slope and random radial dispersion in the cored benchmark units.
# Named profile CSVs carry density, radial mean/variance and effective marker count on the saved log mesh.
# The shell reduction preserves deposited mass and raw velocity moments while converting the output to rho_s, v_s and Gyr.

"""Render weighted-PIC profiles in the fixed Frank benchmark unit convention.

Raw code-unit profiles are converted to Frank's paper normalization before
plotting density, logarithmic slope, and radial velocity dispersion.  Inner
radial cells are merged until they carry enough effective markers, reducing
shot noise while conserving their deposited shell mass and velocity moments.
The input convention is G=M_total=r_s=1 with M_total/M_s=MU below; TIME_GYR
is specific to that benchmark. Native G=M_s=r_s=1 profiles require a separate
unit conversion.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "venus-matplotlib-cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
for font_cache in (Path.home() / ".matplotlib").glob("fontlist*.json"):
    target = matplotlib_cache / font_cache.name
    if not target.exists():
        shutil.copy2(font_cache, target)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spherical_profile_rendering import add_slope_reference_lines, fig_to_rgb, style_axes
from render_density_slope_sigma2_gif import AtomicStreamingGif
from render_frank_venus_sphdsmc_gif import matched_bandwidth_log_slope, text_layout_qa
from plot_subhalo_virial_1 import shell_volumes

# Unit conversion shared with the Frank SphDSMC comparison renderer.
# The plotted transformations are rho/rho_s = 4*pi*MU*rho_code and
# sigma_r^2/v_s^2 = MU*sigma_r,code^2.  TIME_GYR is the physical duration of
# one Frank N-body time unit for this benchmark.
MU = 1.1252580933331495
TIME_GYR = 0.021151235023496799


# ---- Conservative radial reduction of deposited WPIC moments -------------
def load_profile(path: Path) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load and conservatively group one deposited WPIC radial profile.

    A group closes after accumulating 128 effective markers.  Its density is
    total group mass divided by total shell volume; its radial dispersion is
    reconstructed from the mass-weighted raw second moment after subtracting
    the group's bulk radial motion.
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    r = np.asarray(data["radius"], float)
    rho = np.asarray(data["density"], float)
    mean = np.asarray(data["radial_mean"], float)
    var = np.maximum(np.asarray(data["radial_variance"], float), 0.0)
    neff = np.asarray(data["neff"], float)
    # Native WPIC radii are arithmetic centers even on a logarithmic mesh.
    volume = shell_volumes(r)
    # Spherical shell mass supplies the weights for every grouped physical moment.
    mass = rho * volume
    # ``neff`` is the effective deposited marker count, not represented mass.
    # It therefore controls sampling noise while the sums below remain
    # explicitly mass weighted for unequal-weight marker populations.
    groups, begin, count = [], 0, 0.0
    # Accumulate the specified effective-count target from inner to outer cells and merge an incomplete tail with its last neighbor.
    for end in range(r.size):
        count += max(neff[end], 0.0)
        if count >= 128.0:
            groups.append((begin, end + 1)); begin, count = end + 1, 0.0
    if begin < r.size:
        if groups: groups[-1] = (groups[-1][0], r.size)
        else: groups.append((0, r.size))
    rr, dd, ss = [], [], []
    for lo, hi in groups:
        m = np.sum(mass[lo:hi]); v = np.sum(volume[lo:hi])
        if m <= 0 or v <= 0: continue
        # Compute group mean and raw second moment before subtracting squared mean, preserving the effect of coherent radial motion.
        u = np.sum(mass[lo:hi] * mean[lo:hi]) / m
        raw2 = np.sum(mass[lo:hi] * (var[lo:hi] + mean[lo:hi]**2)) / m
        # Use mass-weighted group radius, mass/volume density and nonnegative random variance in the reference units.
        rr.append(np.sum(mass[lo:hi] * r[lo:hi]) / m)
        dd.append(4.0 * math.pi * MU * m / v)
        ss.append(MU * max(raw2 - u*u, 0.0))
    return float(data["time"][0]), (np.asarray(rr), np.asarray(dd), np.asarray(ss))


# ---- Matched density, slope, and random radial support frame ---------------
def frame(curve, initial, time_gyr, particle_count=1_000_000):
    """Construct density, slope, and radial-dispersion panels for one time."""
    r, rho, sigma2 = curve
    ir, irho, isigma2 = initial
    fig, aa = plt.subplots(1, 3, figsize=(15.2, 4.35), dpi=135)
    axes = list(aa)
    fig.subplots_adjust(left=0.086, right=0.985, bottom=0.17, top=0.84, wspace=0.39)
    if particle_count % 1_000_000 == 0:
        multiplier = particle_count // 1_000_000
        particle_tex = r"10^6" if multiplier == 1 else rf"{multiplier}\times10^6"
    else:
        particle_tex = f"{particle_count:,}"
    fig._ic_annotation = fig.text(0.087, 0.972,
        rf"Weighted particle-in-cell, $N_{{\rm p}}={particle_tex}$",
        ha="left", va="top", fontsize=11.5, linespacing=1.22)
    xlim = (math.log10(3e-3), math.log10(1e3))
    for ax, y, iy in ((axes[0], rho, irho), (axes[2], sigma2, isigma2)):
        mask = np.isfinite(iy) & (iy > 0) & (ir >= 3e-3) & (ir <= 1e3)
        ax.plot(np.log10(ir[mask]), np.log10(iy[mask]), "k:", lw=1.8,
                label="Initial" if ax is axes[0] else None, zorder=1)
        mask = np.isfinite(y) & (y > 0) & (r >= 3e-3) & (r <= 1e3)
        ax.plot(np.log10(r[mask]), np.log10(y[mask]), color="b", ls="-", lw=2.0,
                label="WPIC" if ax is axes[0] else None, zorder=2)
    for source_r, source_rho, color, ls in ((ir, irho, "k", ":"), (r, rho, "b", "-")):
        # Evaluate initial and current density slopes with the same current-profile radial bandwidth.
        sr, slope = matched_bandwidth_log_slope(source_r, source_rho, r, 9, 2)
        mask = np.isfinite(slope) & (sr >= 3e-3) & (sr <= r[-5])
        axes[1].plot(np.log10(sr[mask]), slope[mask], color=color, ls=ls, lw=2.0)
    axes[0].set(xlim=xlim, ylim=(-10, 6), xlabel=r"$\log_{10}(r/r_s)$", ylabel=r"$\log_{10}(\rho/\rho_s)$")
    axes[1].set(xlim=xlim, ylim=(-6, 2), xlabel=r"$\log_{10}(r/r_s)$", ylabel=r"$\mathrm{d}\log\rho/\mathrm{d}\log r$")
    axes[1].axhline(0, color="0.6", lw=1, ls=":")
    add_slope_reference_lines(axes[1], *xlim, -6, 2)
    axes[2].set(xlim=xlim, ylim=(-3.8, -0.25), xlabel=r"$\log_{10}(r/r_s)$", ylabel=r"$\log_{10}(\sigma_r^2/v_s^2)$")
    for ax in axes:
        ax.xaxis.label.set_size(16); ax.yaxis.label.set_size(16)
        ax.xaxis.labelpad = 7; ax.yaxis.labelpad = 8; style_axes(ax)
    axes[0].legend(loc="lower left", fontsize=12.5, frameon=False, handlelength=3.0)
    fig._time_title = fig.suptitle(rf"$t={time_gyr:.3f}\,\mathrm{{Gyr}}$", fontsize=16, y=0.97)
    return fig, axes


# ---- Restart-continuation discovery and streamed GIF publication -----------
def main():
    """Combine continuation profiles and stream a benchmark-normalized GIF."""
    # Multiple --profile-dir arguments support checkpoint continuations.
    # Duplicate physical times are collapsed before frames are produced.
    p = argparse.ArgumentParser()
    p.add_argument("--profile-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--particles", type=int, default=1_000_000)
    a = p.parse_args()
    paths = sorted({q for d in a.profile_dir for q in d.glob("profile_*.csv")})
    loaded = [load_profile(q) for q in paths]
    # Distinct measured times order continuation profiles; the first physical profile remains the comparison baseline.
    loaded = sorted({round(t, 12): (t, c) for t, c in loaded}.values())
    initial = loaded[0][1]
    writer = AtomicStreamingGif(a.output, a.fps)
    for index, (time_code, curve) in enumerate(loaded):
        fig, axes = frame(curve, initial, time_code * TIME_GYR, a.particles)
        if index == 0: text_layout_qa(fig, axes)
        writer.append(fig_to_rgb(fig)); plt.close(fig)
    writer.close()
    print(f"wrote {a.output} ({len(loaded)} frames, t={loaded[-1][0]*TIME_GYR:.6f} Gyr)")


if __name__ == "__main__": main()
