#!/usr/bin/env python3
# Copy actual WPIC profiles nearest a uniform physical-time selection grid.
# Input is a profile directory with history.csv. A fresh destination receives whole CSVs and selection metadata.
# Every selected state keeps its actual source time and both endpoint states are retained.

"""Stage endpoint-preserving WPIC profiles on a uniform simulation-time grid.

The source directory contains ``profile_<step>.csv`` snapshots and a
``history.csv`` table mapping ``step`` to simulation ``time``.  If a profile's
step is absent from the history table, the script reads ``time`` from the
profile's first named row.  Duplicate physical times are collapsed before
sampling.

Requested grid points are uniformly spaced between the earliest and latest
available times and are represented by their nearest saved profiles.  Both
endpoints are always retained; if nearest-neighbor collisions leave too few
files, remaining snapshots are added in chronological order.  The new,
fresh destination receives copied profiles, ``history.csv``,
and ``selection.json`` describing the exact staged subset.
"""

import argparse
import bisect
import csv
import json
import shutil
from pathlib import Path


def main():
    """Select, copy, and describe a uniformly sampled profile subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least two")

    history_path = args.source / "history.csv"
    with history_path.open(newline="") as handle:
        history = {
            int(row["step"]): float(row["time"])
            for row in csv.DictReader(handle)
        }

    # Collapse duplicate physical epochs before selecting representative saved profiles.
    by_time = {}
    for path in args.source.glob("profile_*.csv"):
        step = int(path.stem.split("_")[-1])
        time = history.get(step)
        if time is None:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                time = float(next(reader)["time"])
        by_time[round(time, 14)] = (time, step, path)
    entries = sorted(by_time.values())
    if len(entries) < 2:
        raise RuntimeError("Fewer than two distinct WPIC profile times found")

    count = min(args.frames, len(entries))
    times = [entry[0] for entry in entries]
    chosen = {0, len(entries) - 1}
    for index in range(count):
        # Space requested targets uniformly in simulation time, then choose the nearest existing snapshot.
        target = times[0] + (times[-1] - times[0]) * index / (count - 1)
        insertion = bisect.bisect_left(times, target)
        candidates = [
            candidate
            for candidate in (insertion - 1, insertion)
            if 0 <= candidate < len(entries)
        ]
        chosen.add(min(candidates, key=lambda candidate: abs(times[candidate] - target)))
    # Distinct targets may select the same file. Fill remaining slots with actual chronological source states.
    if len(chosen) < count:
        for index in range(len(entries)):
            chosen.add(index)
            if len(chosen) == count:
                break

    # Require a new destination so existing render inputs are preserved.
    args.destination.mkdir(parents=True, exist_ok=False)
    selected = [entries[index] for index in sorted(chosen)]
    for _, _, source in selected:
        shutil.copy2(source, args.destination / source.name)
    shutil.copy2(history_path, args.destination / "history.csv")
    metadata = {
        "source": str(args.source),
        "available_distinct_profile_times": len(entries),
        "selected_profiles": len(selected),
        "first_step": selected[0][1],
        "first_time_code": selected[0][0],
        "last_step": selected[-1][1],
        "last_time_code": selected[-1][0],
        "selection": "nearest available profiles on a uniform simulation-time grid",
    }
    (args.destination / "selection.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
