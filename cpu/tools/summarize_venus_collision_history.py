#!/usr/bin/env python3
# Summarize collision clocks, locality measures and event columns from WPIC history CSVs.
# Input directories contain collision_history.csv in solver code units. Output is one pipe-delimited row per run.

"""Summarize the physical clocks and event ledger of WPIC collision runs.

Each positional argument is an output directory containing
``collision_history.csv``. One pipe-delimited record is emitted per run. The
reported minimum collision time is the most restrictive sampled cell clock,
not a halo-averaged cumulative collision age.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable


def values(rows: Iterable[dict[str, str]], name: str) -> list[float]:
    """Return numeric values from one optional CSV column."""
    return [
        float(row[name])
        for row in rows
        if row.get(name, "") not in ("", None)
    ]


def require_nonempty(sequence: list[float], label: str, source: Path) -> None:
    """Reject a required diagnostic column that has no positive-time data."""
    if not sequence:
        raise RuntimeError(f"{source}: no positive-time values for {label}")


def summarize(directory: Path) -> str:
    """Build the stable pipe-delimited summary for one output directory."""
    source = directory / "collision_history.csv"
    with source.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if float(row.get("time", "0")) > 0.0
        ]
    if not rows:
        raise RuntimeError(f"{source}: no positive-time collision records")

    # Clock summaries use positive sampled timescales. Empty required columns trigger a descriptive error.
    tcoll = [value for value in values(rows, "min_collision_time") if value > 0.0]
    tdyn = [value for value in values(rows, "min_dynamical_time") if value > 0.0]
    tphi = [
        value
        for value in values(rows, "min_potential_change_time")
        if value > 0.0
    ]
    markers = values(rows, "marker_count")
    dr_mfp = values(rows, "max_dr_over_mfp")
    substeps = values(rows, "substeps")
    accepted = values(rows, "accepted_pairs")
    captures = values(rows, "capture_events")
    scatters = values(rows, "scattering_events")

    for sequence, label in (
        (tcoll, "min_collision_time"),
        (tdyn, "min_dynamical_time"),
        (tphi, "min_potential_change_time"),
        (markers, "marker_count"),
        (dr_mfp, "max_dr_over_mfp"),
        (substeps, "substeps"),
    ):
        require_nonempty(sequence, label, source)

    # Report first/last/median clock values, extrema and sums while keeping the final physical epoch explicit.
    return "|".join(
        (
            directory.name,
            f"t={float(rows[-1]['time']):.9g}",
            f"tcoll_first={tcoll[0]:.6g}",
            f"tcoll_last={tcoll[-1]:.6g}",
            f"tcoll_median={statistics.median(tcoll):.6g}",
            f"tdyn_last={tdyn[-1]:.6g}",
            f"tphi_last={tphi[-1]:.6g}",
            f"tcoll_over_tdyn_last={tcoll[-1] / tdyn[-1]:.6g}",
            f"tcoll_over_tphi_last={tcoll[-1] / tphi[-1]:.6g}",
            f"max_dr_mfp={max(dr_mfp):.6g}",
            f"markers_first={markers[0]:.0f}",
            f"markers_last={markers[-1]:.0f}",
            f"markers_min={min(markers):.0f}",
            f"substeps_mean={statistics.mean(substeps):.6g}",
            f"substeps_max={max(substeps):.0f}",
            f"accepted_sum={sum(accepted):.0f}",
            f"captures_sum={sum(captures):.0f}",
            f"scatters_sum={sum(scatters):.0f}",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="output directories containing collision_history.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in args.directories:
        print(summarize(directory))


if __name__ == "__main__":
    main()
