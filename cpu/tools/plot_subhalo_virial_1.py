#!/usr/bin/env python3
# Integrate a resolved-shell Newtonian virial diagnostic from complete WPIC moment profiles.
# The CSV supplies saved radial mean/variance and the sum of tangential second moments in code units with G=1.
# Output PNG and CSV contain the same resolved-mass kinetic energy, half-shell potential estimate and their ratio.

"""Plot the virial ratio 2T/|W| versus time for a WPIC run.

Reads the weighted-PIC radial-profile schema

    time, radius, density, radial_mean, radial_variance,
    tangential_second, neff

and, for every saved profile, forms two global integrals over the resolved
shells (radial_minimum to radial_maximum, as written in the profiles):

  Kinetic energy
      T = 1/2 * sum_shell  m_shell * <v^2>,
      <v^2> = <v_r^2> + <v_t^2>,
      <v_r^2> = radial_variance + radial_mean^2,   (var + mean^2)
      <v_t^2> = tangential_second.                 (tangential 2nd moment)

  Potential energy (Newtonian, spherical, G = 1)
      W = - sum_shell  M(<r) / r * m_shell,
      M(<r) = resolved mass through the shell midpoint (half-shell convention),
      m_shell = density * shell_volume.

The virial ratio 2T/|W| is then plotted against time in a single panel.

This reports a resolved-shell Newtonian diagnostic with G=1 and the saved
radial-velocity convention. The boundary pressure, background expansion,
unresolved inner carrier, and subhalo internal binding/heat have separate
budgets. Establish equilibrium with those budgets and resolution checks.
The derivative and metric diagnostics in the EF manual apply to relativity.
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
import numpy as np


def read_gamma(directory: Path):
    """Return gamma_i from <directory>/metadata.json, or None if unavailable."""
    path = directory / "metadata.json"
    if not path.is_file():
        return None
    try:
        with path.open() as handle:
            return json.load(handle).get("gamma_i")
    except Exception:
        return None


def shell_volumes(radius: np.ndarray) -> np.ndarray:
    """Recover log-mesh faces from their arithmetic dr-centers.

    VENUS stores r_i=(r_left+r_right)/2 even on a log mesh. For face ratio
    q, r_left=2*r_i/(1+q). Geometric midpoints of centers would systematically
    overestimate every shell volume. Keep empty-density cells in this mesh.
    """
    r = np.asarray(radius, dtype=float)
    if r.size < 2 or np.any(~np.isfinite(r)) or np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("Profile radii must be finite, positive and increasing")
    # The writer stores arithmetic centers of logarithmically spaced faces. Recover those faces before evaluating spherical shell volumes.
    q = r[1] / r[0]
    if not np.allclose(r[1:] / r[:-1], q, rtol=1e-9, atol=0):
        raise ValueError("Newtonian WPIC virial tool requires a complete logarithmic mesh")
    edges = np.r_[2.0 * r / (1.0 + q), 2.0 * r[-1] * q / (1.0 + q)]
    return 4.0 * math.pi / 3.0 * np.diff(edges ** 3)


def virial_quantities_for_profile(path: Path) -> tuple[float, float, float, float, float] | None:
    """Return time, ratio, T, W and resolved mass on one complete saved mesh."""
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    needed = {
        "time",
        "radius",
        "density",
        "radial_mean",
        "radial_variance",
        "tangential_second",
    }
    if not needed.issubset(rows.dtype.names or ()):
        return None
    metadata = path.parent / "metadata.json"
    if metadata.is_file():
        expected = json.loads(metadata.read_text()).get("radial_cells")
        if expected is not None and rows.size != int(expected):
            return None
    time = float(rows["time"][0])
    if not np.isfinite(time) or np.any(rows["time"] != time):
        return None
    r = np.asarray(rows["radius"], dtype=float)
    rho = np.asarray(rows["density"], dtype=float)
    vr_mean = np.asarray(rows["radial_mean"], dtype=float)
    vr_var = np.asarray(rows["radial_variance"], dtype=float)
    vt2 = np.asarray(rows["tangential_second"], dtype=float)

    # Validate the complete mesh before forming shell masses. Dropping a bad
    # row would merge neighboring shells and change the represented volume.
    if r.size < 3 or any(np.any(~np.isfinite(x)) for x in (r, rho, vr_mean, vr_var, vt2)):
        return None
    if np.any(r <= 0) or np.any(rho < 0) or np.any(vr_var < 0) or np.any(vt2 < 0):
        return None
    order = np.argsort(r)
    r, rho, vr_mean, vr_var, vt2 = (x[order] for x in (r, rho, vr_mean, vr_var, vt2))

    volume = shell_volumes(r)
    # Retain empty shells in the geometry and multiply each density by its own exact spherical volume.
    m_shell = rho * volume

    # Kinetic energy: 1/2 * sum m <v^2>, with <v^2> = <v_r^2> + <v_t^2>.
    # The radial second moment equals variance plus squared mean. Adding tangential second moment gives the full saved speed squared.
    v_r2 = np.maximum(vr_var, 0.0) + vr_mean * vr_mean
    v_t2 = np.maximum(vt2, 0.0)
    kinetic = 0.5 * float(np.sum(m_shell * (v_r2 + v_t2)))

    # Potential energy: Newtonian, G=1, resolved shells only.
    # Enclosed mass interior to each shell centre (half-shell convention).
    # Place half of a shell mass inside its center for this resolved-shell potential estimate. Inner-carrier and background terms have separate budgets.
    enclosed = np.cumsum(m_shell) - 0.5 * m_shell
    with np.errstate(divide="ignore", invalid="ignore"):
        potential = -float(np.sum(enclosed * m_shell / r))

    if not (potential < 0.0) or not np.isfinite(kinetic):
        return None
    ratio = 2.0 * kinetic / abs(potential)
    if not np.isfinite(ratio):
        return None
    return time, ratio, kinetic, potential, float(np.sum(m_shell))


def virial_ratio_for_profile(path: Path) -> tuple[float, float] | None:
    """Return the retained (time, 2T/|W|) interface for existing callers."""
    result = virial_quantities_for_profile(path)
    return None if result is None else result[:2]


def main() -> None:
    """Load a run's profiles, compute 2T/|W| per profile, and plot vs time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run output directory containing profile_*.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output PNG path")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth profile (first and last always kept)")
    parser.add_argument("--title", type=str, default=None)
    arguments = parser.parse_args()

    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return 0

    paths = sorted(arguments.run_dir.glob("profile_*.csv"), key=step_of)
    if not paths:
        raise RuntimeError(f"No profile_*.csv files in {arguments.run_dir}")
    # Validate all available snapshots before selecting a stride. A partial
    # newest file must leave the preceding complete endpoint in the report.
    complete = [virial_quantities_for_profile(p) for p in paths]
    complete = {x[0]: x for x in complete if x is not None}
    results = [complete[t] for t in sorted(complete)]
    if not results:
        raise RuntimeError("No complete profile yielded a finite virial ratio")
    # Select a plotting stride only after validating every available profile, preserving the first and newest complete epochs.
    keep = set(range(0, len(results), max(1, int(arguments.stride))))
    keep.update((0, len(results) - 1))
    table = np.asarray([results[i] for i in sorted(keep)], dtype=float)
    times, ratios = table[:, 0], table[:, 1]
    metadata_path = arguments.run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    represented_mass = metadata.get("mobile_mass")
    # Compare resolved profile mass with the represented mobile mass when that normalization is recorded.
    mass_fraction = np.full(times.size, np.nan)
    if represented_mass is not None and np.isfinite(float(represented_mass)) and float(represented_mass) > 0:
        mass_fraction = table[:, 4] / float(represented_mass)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=140)
    ax.plot(times, ratios, color="b", lw=1.8, label="resolved Newtonian ratio")
    ax.axhline(1.0, color="0.55", ls="--", lw=1.5)
    ax.text(0.98, 1.0, r"$2T/|W|=1$", transform=ax.get_yaxis_transform(),
            color="0.35", fontsize=11, ha="right", va="center",
            bbox={"boxstyle": "square,pad=0.15", "facecolor": "white",
                  "edgecolor": "none"})
    ax.legend(loc="best", frameon=True, facecolor="white")
    ax.set_xlabel("time", fontsize=15, labelpad=7)
    ax.set_ylabel(r"$2T/|W|$", fontsize=16, labelpad=8)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    gamma = read_gamma(arguments.run_dir)
    if arguments.title:
        heading = arguments.title
    elif gamma is not None:
        heading = rf"$\gamma={float(gamma):g}$"
    else:
        heading = arguments.run_dir.name
    fig.suptitle(f"{heading}: resolved Newtonian virial ratio", fontsize=14, y=0.97)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.14)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the quantities behind the curve so late-time settling and changing
    # resolved mass can be assessed together without reading image pixels.
    np.savetxt(arguments.output.with_suffix(".csv"),
               np.column_stack((table, mass_fraction)), delimiter=",",
               header="time,virial_ratio,kinetic_energy,potential_energy,resolved_mass,resolved_mass_fraction",
               comments="")
    if np.any(np.isfinite(mass_fraction)):
        fig.text(0.5, 0.02,
                 f"Resolved mass / represented mass: {mass_fraction[0]:.4f} to {mass_fraction[-1]:.4f}",
                 ha="center", fontsize=9)
        fig.subplots_adjust(bottom=0.19)
    fig.savefig(arguments.output)
    plt.close(fig)
    print(f"wrote {arguments.output} ({times.size} profiles, "
          f"final 2T/|W|={ratios[-1]:.4f})")


if __name__ == "__main__":
    main()
