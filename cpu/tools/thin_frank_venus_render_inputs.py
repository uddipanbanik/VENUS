#!/usr/bin/env python3
# Copy complete comparison profiles at a reduced physical-time cadence.
# VENUS input consists of appended CSV radial blocks. Frank input consists of separate prof.* files.
# Each format uses its own conversion to Gyr, and both preserve actual endpoint snapshots.

"""Create cadence-thinned, endpoint-preserving inputs for comparison renders."""

from __future__ import annotations

import argparse
import bisect
import math
import shutil
from pathlib import Path


MU = 1.1252580933331495
TIME_GYR = 0.021151235023496799


def thin_venus(
    sources: list[Path],
    destination: Path,
    interval_gyr: float,
    final_step: int,
) -> tuple[int, int]:
    """Thin VENUS profile snapshots at nearly uniform physical-time cadence.

    Code time is converted to gigayears before assigning cadence buckets.
    Complete radial snapshots remain indivisible, and the initial and final
    physical states are retained even when they share a bucket.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    seen_buckets: set[int] = set()
    written_steps: set[int] = set()
    header: str | None = None
    groups_seen = 0
    # Convert VENUS code time through the model mass normalization and the physical reference time.
    factor = math.sqrt(MU) * TIME_GYR

    with destination.open("w", encoding="ascii", newline="") as output:
        for source in sources:
            with source.open(encoding="ascii", newline="") as handle:
                source_header = handle.readline()
                if header is None:
                    header = source_header
                    output.write(header)
                elif source_header != header:
                    raise RuntimeError(f"Incompatible VENUS profile header: {source}")

                current_step: int | None = None
                current_rows: list[str] = []
                current_time = 0.0

                def flush() -> None:
                    """Commit one complete radial snapshot if its bucket is new."""

                    nonlocal groups_seen
                    if current_step is None:
                        return
                    groups_seen += 1
                    # Assign a whole radial block to its nearest requested time interval. The stored rows remain unchanged.
                    bucket = int(round((current_time * factor) / interval_gyr))
                    keep = (
                        current_step == 0
                        or current_step == final_step
                        or bucket not in seen_buckets
                    )
                    if keep and current_step not in written_steps:
                        output.writelines(current_rows)
                        written_steps.add(current_step)
                        seen_buckets.add(bucket)

                for line in handle:
                    fields = line.split(",", 3)
                    if len(fields) < 3:
                        continue
                    try:
                        time_code = float(fields[0])
                        step = int(float(fields[1]))
                    except ValueError:
                        continue
                    if current_step is None:
                        current_step = step
                        current_time = time_code
                    # A new step closes the preceding radial block before the next one is accumulated.
                    elif step != current_step:
                        flush()
                        current_step = step
                        current_time = time_code
                        current_rows = []
                    current_rows.append(line)
                flush()

    # Require the requested final physical state to have survived cadence thinning.
    if final_step not in written_steps:
        raise RuntimeError(f"Final VENUS step {final_step} was not found")
    return groups_seen, len(written_steps)


def thin_frank(source_dir: Path, destination_dir: Path, count: int) -> int:
    """Select Frank snapshots nearest uniformly spaced physical target times."""

    # Read the physical time stored in the first record of every Frank profile.
    entries: list[tuple[float, Path]] = []
    for path in sorted(source_dir.glob("prof.*")):
        with path.open(encoding="ascii") as handle:
            fields = handle.readline().split()
        if len(fields) >= 2:
            entries.append((float(fields[1]) * TIME_GYR, path))
    if not entries:
        raise RuntimeError(f"No Frank profiles in {source_dir}")

    count = min(count, len(entries))
    times = [entry[0] for entry in entries]
    chosen: set[int] = {0, len(entries) - 1}
    if count > 2:
        # Nearest-time selection keeps the two solvers' rendered cadence
        # comparable without interpolating either physical density profile.
        start, stop = times[0], times[-1]
        for index in range(count):
            target = start + (stop - start) * index / (count - 1)
            insertion = bisect.bisect_left(times, target)
            candidates = [
                candidate
                for candidate in (insertion - 1, insertion)
                if 0 <= candidate < len(times)
            ]
            chosen.add(min(candidates, key=lambda candidate: abs(times[candidate] - target)))

    if len(chosen) < count:
        for index in range(len(entries)):
            chosen.add(index)
            if len(chosen) == count:
                break

    # Copy full source snapshots, retaining their original names and metadata.
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index in sorted(chosen):
        source = entries[index][1]
        shutil.copy2(source, destination_dir / source.name)
    return len(chosen)


def main() -> None:
    """Create matched, endpoint-preserving VENUS and Frank render inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venus-profile", type=Path, action="append", required=True)
    parser.add_argument("--venus-output", type=Path, required=True)
    parser.add_argument("--venus-interval-gyr", type=float, required=True)
    parser.add_argument("--venus-final-step", type=int, required=True)
    parser.add_argument("--frank-profile-dir", type=Path, required=True)
    parser.add_argument("--frank-output-dir", type=Path, required=True)
    parser.add_argument("--frank-count", type=int, required=True)
    args = parser.parse_args()

    # Thin each format with its native time metadata, then report the retained
    # sample counts for reproducible rendering.
    groups_seen, venus_written = thin_venus(
        args.venus_profile,
        args.venus_output,
        args.venus_interval_gyr,
        args.venus_final_step,
    )
    frank_written = thin_frank(
        args.frank_profile_dir,
        args.frank_output_dir,
        args.frank_count,
    )
    print(
        f"VENUS: retained {venus_written} of {groups_seen} snapshot groups; "
        f"Frank: retained {frank_written} profiles"
    )


if __name__ == "__main__":
    main()
