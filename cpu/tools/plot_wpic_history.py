#!/usr/bin/env python3
# Display density, mass, energy, timestep and effective marker count from a WPIC history CSV.
# Named columns preserve the solver's saved code units and distinguish mobile mass from the field matter-mass diagnostic.
# The figure and console annotations summarize actual recorded series and their first-to-last changes.

"""Plot density, mass, energy, timestep, and marker sampling from WPIC history.

Adapted from Sachiet Kapur's diagnostic. max_density is the largest resolved
cell density. mobile_mass includes the unresolved inner carrier; total_mass
is the field solver's matter-mass diagnostic. Their definitions and boundary
budgets determine the interpretation of a change. The energy column includes
orbital kinetic energy, half the self-potential contribution, and internal
energy. Expansion and finite boundaries enter its physical budget. A density
peak or a stationary Newtonian virial ratio requires supporting evidence
before a physical collapse or equilibrium interpretation is established.
"""

from __future__ import annotations

import argparse
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


EXPECTED_COLUMNS = (
    "step",
    "time",
    "dt",
    "mobile_mass",
    "total_mass",
    "total_energy",
    "max_density",
    "minimum_neff",
    "timestep_limit",
    "wall_seconds",
)


def load_history(path: Path) -> np.ndarray:
    """Load history.csv as a named structured array, skipping partial rows.

    A live run may be mid-write, so ``genfromtxt`` with ``invalid_raise=False``
    drops any trailing truncated line instead of aborting.
    """
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    )
    present = rows.dtype.names or ()
    missing = [name for name in ("time", "max_density") if name not in present]
    if missing:
        raise RuntimeError(f"{path} is missing required columns {missing}")
    # Drop any row with a non-finite time (an artifact of a partial final line).
    # A finite recorded clock identifies usable parsed rows in this live-history reader. Other observables retain their own finite masks.
    rows = rows[np.isfinite(rows["time"])]
    if rows.size == 0:
        raise RuntimeError(f"{path} has no complete finite-time data")
    return rows


def relative_drift(time: np.ndarray, values: np.ndarray) -> str:
    """Return a short 'first -> last (Delta%)' string for finite series."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return "no finite data"
    # Measure changes from the first and last finite values of each observable, allowing different missing-data patterns.
    first = float(values[finite][0])
    last = float(values[finite][-1])
    if first != 0.0:
        pct = 100.0 * (last - first) / abs(first)
        return f"{first:.4g} -> {last:.4g}  ({pct:+.1f}%)"
    return f"{first:.4g} -> {last:.4g}"


def add_panel(
    axis: plt.Axes,
    time: np.ndarray,
    rows: np.ndarray,
    column: str,
    label: str,
    color: str,
    logy: bool = False,
) -> None:
    """Draw one series against time if the column exists, else annotate absence."""
    names = rows.dtype.names or ()
    if column not in names:
        axis.text(
            0.5,
            0.5,
            f"'{column}' not in history.csv",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="0.4",
        )
        axis.set_ylabel(label)
        return
    values = np.asarray(rows[column], dtype=float)
    finite = np.isfinite(values)
    if logy:
        # A logarithmic panel requires positive values. Preserve unavailable/nonpositive entries as gaps.
        values = np.where(finite & (values > 0), values, np.nan)
    axis.plot(time, values, color=color, lw=1.8)
    if logy:
        positive = finite & (values > 0.0)
        if np.any(positive):
            axis.set_yscale("log")
    axis.set_ylabel(label)
    axis.grid(False)


def main() -> None:
    """Read one history.csv and write a stacked multi-panel time-series figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True, help="Path to history.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--title", type=str, default=None, help="Optional figure title (e.g. run name)"
    )
    arguments = parser.parse_args()

    rows = load_history(arguments.history)
    time = np.asarray(rows["time"], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(9.0, 13.0), dpi=130, sharex=True)

    # Largest resolved cell density; the peak can lie away from the center.
    add_panel(axes[0], time, rows, "max_density", r"$\rho_{\max}$", "C3", logy=True)

    # Mass budget -- two lines on one panel.
    # Plot the two mass definitions together so their separate physical budgets remain visible.
    names = rows.dtype.names or ()
    if "mobile_mass" in names:
        axes[1].plot(
            time,
            np.asarray(rows["mobile_mass"], float),
            color="C0",
            lw=1.8,
            label="mobile mass (including inner carrier)",
        )
    if "total_mass" in names:
        axes[1].plot(
            time,
            np.asarray(rows["total_mass"], float),
            color="C1",
            lw=1.2,
            ls="--",
            label="field matter mass",
        )
    axes[1].set_ylabel("mass")
    axes[1].grid(False)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=9, frameon=True, facecolor="white")

    # Energy history; assess its change using the background and boundary budget.
    add_panel(axes[2], time, rows, "total_energy", "total energy", "C2")

    # Timestep and its limit -- resolution / slowdown diagnostic.
    # Show accepted timestep and reported limit as separate positive series when their columns exist.
    if "dt" in names:
        axes[3].plot(time, np.where(np.asarray(rows["dt"], float) > 0, rows["dt"], np.nan), color="C4", lw=1.8, label="dt")
    if "timestep_limit" in names:
        axes[3].plot(
            time,
            np.where(np.asarray(rows["timestep_limit"], float) > 0, rows["timestep_limit"], np.nan),
            color="0.5",
            lw=1.2,
            ls="--",
            label="timestep_limit",
        )
    axes[3].set_ylabel("timestep")
    axes[3].set_yscale("log")
    axes[3].grid(False)
    if axes[3].get_legend_handles_labels()[0]:
        axes[3].legend(fontsize=9, frameon=True, facecolor="white")

    # Marker sampling floor -- noise flag.
    add_panel(axes[4], time, rows, "minimum_neff", "min neff", "C5")

    for axis in axes:
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
    axes[-1].set_xlabel("time")

    # Drift annotations for the two conservation-relevant series.
    # The annotations report changes in the saved observables. Their physical interpretation follows the background and boundary budgets.
    lines = []
    if "max_density" in names:
        lines.append("max_density change: " + relative_drift(time, np.asarray(rows["max_density"], float)))
    if "total_mass" in names:
        lines.append("total_mass:  " + relative_drift(time, np.asarray(rows["total_mass"], float)))
    if "total_energy" in names:
        lines.append("energy change: " + relative_drift(time, np.asarray(rows["total_energy"], float)))
    annotation = "\n".join(lines)

    header = arguments.title or arguments.history.parent.name
    fig.suptitle(header, fontsize=13, y=0.995)
    if annotation:
        fig.text(
            0.5,
            0.965,
            annotation,
            ha="center",
            va="top",
            fontsize=9,
            family="monospace",
            color="0.25",
        )

    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.06, hspace=0.12)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(arguments.output)
    plt.close(fig)
    print(f"wrote {arguments.output}")
    if annotation:
        print(annotation)


if __name__ == "__main__":
    main()
