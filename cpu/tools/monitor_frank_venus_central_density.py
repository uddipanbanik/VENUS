#!/usr/bin/env python3
# Compare evolving central densities using a common enclosed 250-particle mass fraction.
# Inputs are VENUS history CSVs and Frank native profiles/logs. Recorded units are converted to Gyr and the reference density normalization.
# Output appends measured epochs, interpolation brackets and recent density growth rates to a monitor CSV.

"""Append an apples-to-apples Frank/VENUS central-density monitor row.

The preferred comparison encloses the same 250-particle mass fraction in both
codes.  VENUS is interpolated in log density to the exact Frank profile time,
and both densities are converted to the paper's ``rho_s`` normalization before
rates or differences are reported.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MU = 1.1252580933331495
FRANK_ANALYTIC_MU = 1.1252588927745819
FRANK_TIME_GYR = 0.021151235023496799
PARTICLE_COUNT = 1_000_000


def complete_csv(path: Path) -> np.ndarray:
    """Read only newline-terminated CSV records with the expected column count.

    Solver output may be inspected while a row is still being written; dropping
    that partial tail prevents a live monitor from reporting a nonphysical
    transient parse value.
    """
    with path.open(encoding="ascii") as handle:
        header = handle.readline().rstrip("\n")
        expected = header.count(",")
        rows = [header]
        for line in handle:
            if line.endswith("\n") and line.rstrip("\n").count(",") == expected:
                rows.append(line.rstrip("\n"))
    return np.atleast_1d(
        np.genfromtxt(io.StringIO("\n".join(rows)), delimiter=",", names=True)
    )


def venus_history(
    paths: list[Path], frank_time_gyr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge VENUS continuations into step, physical-time, and log-density arrays."""
    by_step: dict[int, tuple[float, float]] = {}
    for path in paths:
        for row in complete_csv(path):
            step = int(row["step"])
            time = float(row["time"])
            rho = float(row["central_density"])
            if math.isfinite(time) and math.isfinite(rho) and rho > 0.0:
                # Later continuation files replace repeated steps, then the physical time and density conversions use the shared mass normalization.
                by_step[step] = (time, rho)
    ordered = sorted((step, *values) for step, values in by_step.items())
    steps = np.array([item[0] for item in ordered], dtype=int)
    time_gyr = np.array(
        [item[1] * math.sqrt(MU) * frank_time_gyr for item in ordered], dtype=float
    )
    log_rho = np.log10(4.0 * math.pi * np.array([item[2] for item in ordered]))
    return steps, time_gyr, log_rho


def frank_native_inner200(
    path: Path, frank_time_gyr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Parse Frank's native inner-200 central-density time series.

    This diagnostic is retained as a secondary native-code comparison.  The
    common inner-250 estimator below is the like-for-like physical measure.
    """
    values: list[list[float]] = []
    with path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 20:
                continue
            try:
                row = [float(field) for field in fields]
            except ValueError:
                continue
            if row[0] >= 0.0 and row[15] > 0.0:
                values.append(row)
    if not values:
        raise RuntimeError(f"No Frank logfile diagnostic rows in {path}")
    data = np.asarray(values, dtype=float)
    time_gyr = data[:, 0] * frank_time_gyr
    # CENTRALDENSITY() is already multiplied by Frank's analytic DENS_FAC.
    # The tiny correction below uses the shared finite-N realization's mu.
    log_rho = np.log10(data[:, 15] * MU / FRANK_ANALYTIC_MU)
    return time_gyr, log_rho


def frank_common_inner250(
    profile_dir: Path, frank_time_gyr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover Frank density inside the radius enclosing 250 equal particles."""
    records: list[tuple[int, float, float]] = []
    for path in sorted(profile_dir.glob("prof.*")):
        with path.open(encoding="ascii") as handle:
            header = handle.readline().split()
            if len(header) < 3:
                continue
            step = int(float(header[0]))
            time_frank = float(header[1])
            selected: list[float] | None = None
            for line in handle:
                fields = line.split()
                if len(fields) < 14:
                    continue
                try:
                    row = [float(field) for field in fields]
                except ValueError:
                    continue
                if int(round(row[13])) == 250:
                    selected = row
                    break
            if selected is None:
                continue
            outer_radius = selected[2]
            if not (outer_radius > 0.0):
                continue
            # Frank has Mtot=1 and equal particle masses. Convert the enclosed
            # density to the paper's rho_s using 4*pi*mu before taking log10.
            # Mean density inside the equal-particle sphere is enclosed mass divided by 4*pi*r cubed/3. Restore the published density unit.
            rho_paper = 3.0 * (250.0 / PARTICLE_COUNT) * MU / outer_radius**3
            records.append((step, time_frank * frank_time_gyr, math.log10(rho_paper)))
    if not records:
        raise RuntimeError(f"No complete Frank inner-250 profile rows in {profile_dir}")
    records.sort()
    return (
        np.array([record[0] for record in records], dtype=int),
        np.array([record[1] for record in records], dtype=float),
        np.array([record[2] for record in records], dtype=float),
    )


def interpolate_log(time: np.ndarray, log_value: np.ndarray, target: float) -> tuple[float, float, float]:
    """Linearly interpolate log density and return its bracketing times."""
    if target < time[0] or target > time[-1]:
        raise RuntimeError(f"Target {target} lies outside [{time[0]}, {time[-1]}]")
    upper = int(np.searchsorted(time, target, side="left"))
    if upper == 0:
        return float(log_value[0]), float(time[0]), float(time[0])
    if upper == time.size:
        return float(log_value[-1]), float(time[-1]), float(time[-1])
    if time[upper] == target:
        return float(log_value[upper]), float(time[upper]), float(time[upper])
    lower = upper - 1
    # Interpolate log density only between available bracketing times and return those bounds with the comparison.
    fraction = (target - time[lower]) / (time[upper] - time[lower])
    value = log_value[lower] + fraction * (log_value[upper] - log_value[lower])
    return float(value), float(time[lower]), float(time[upper])


def recent_rate(time: np.ndarray, log_value: np.ndarray, end: float, width: float = 0.20) -> float:
    """Fit the recent central-density amplification rate in dex per Gyr."""
    # A local straight-line fit of log density versus physical time gives an amplification rate in dex per Gyr.
    mask = (time <= end + 1.0e-12) & (time >= end - width)
    indices = np.flatnonzero(mask)
    if indices.size < 6:
        indices = np.flatnonzero(time <= end + 1.0e-12)[-min(8, time.size) :]
    if indices.size < 3:
        return float("nan")
    return float(np.polyfit(time[indices], log_value[indices], 1)[0])


def append_row(path: Path, row: dict[str, object]) -> None:
    """Append one timestamped comparison without replacing prior monitoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    # Build the common-time comparison first.  The newest Frank profile at or
    # before both runs' endpoints defines the physical matching time.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venus-history", type=Path, action="append", required=True)
    parser.add_argument("--frank-logfile", type=Path, required=True)
    parser.add_argument("--frank-profile-dir", type=Path, required=True)
    parser.add_argument(
        "--frank-time-gyr",
        type=float,
        default=FRANK_TIME_GYR,
        help="Physical Gyr per Frank N-body time unit",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, venus_time, venus_log250 = venus_history(
        args.venus_history, args.frank_time_gyr
    )
    frank_native_time, frank_native_log200 = frank_native_inner200(
        args.frank_logfile, args.frank_time_gyr
    )
    frank_steps, frank_profile_time, frank_log250 = frank_common_inner250(
        args.frank_profile_dir, args.frank_time_gyr
    )

    # Choose an actual Frank profile at or before both endpoints, then evaluate the VENUS history at that physical epoch.
    common_time = min(float(venus_time[-1]), float(frank_profile_time[-1]))
    profile_index = int(np.flatnonzero(frank_profile_time <= common_time + 1.0e-12)[-1])
    matched_time = float(frank_profile_time[profile_index])
    venus_log_at_match, bracket_lo, bracket_hi = interpolate_log(
        venus_time, venus_log250, matched_time
    )
    frank_log_at_match = float(frank_log250[profile_index])

    venus_rate = recent_rate(venus_time, venus_log250, matched_time)
    frank_rate = recent_rate(
        frank_profile_time[: profile_index + 1],
        frank_log250[: profile_index + 1],
        matched_time,
    )
    rate_ratio = venus_rate / frank_rate if frank_rate != 0.0 else float("nan")

    frank_native_end = min(float(frank_native_time[-1]), float(venus_time[-1]))
    frank_native_index = int(
        np.flatnonzero(frank_native_time <= frank_native_end + 1.0e-12)[-1]
    )
    # Keep the inner-200 native-log diagnostic separate from the common inner-250 estimator.
    native_time = float(frank_native_time[frank_native_index])
    venus_native_match, _, _ = interpolate_log(venus_time, venus_log250, native_time)

    row: dict[str, object] = {
        "check_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latest_frank_native_time_gyr": f"{frank_native_time[-1]:.12g}",
        "latest_frank_profile_time_gyr": f"{frank_profile_time[-1]:.12g}",
        "latest_venus_time_gyr": f"{venus_time[-1]:.12g}",
        "latest_common_profile_time_gyr": f"{matched_time:.12g}",
        "frank_profile_step": int(frank_steps[profile_index]),
        "venus_bracket_lower_time_gyr": f"{bracket_lo:.12g}",
        "venus_bracket_upper_time_gyr": f"{bracket_hi:.12g}",
        "maximum_time_mismatch_gyr": f"{max(matched_time-bracket_lo, bracket_hi-matched_time):.6g}",
        "comparison_method": "Frank exact inner250; VENUS local log-rho interpolation",
        "log10_rhoc_venus_inner250": f"{venus_log_at_match:.9g}",
        "log10_rhoc_frank_inner250": f"{frank_log_at_match:.9g}",
        "delta_log10_rhoc_common250_venus_minus_frank": f"{venus_log_at_match-frank_log_at_match:.9g}",
        "recent_rate_venus_dex_per_gyr": f"{venus_rate:.9g}",
        "recent_rate_frank_dex_per_gyr": f"{frank_rate:.9g}",
        "recent_rate_ratio_venus_over_frank": f"{rate_ratio:.9g}",
        "latest_native_match_time_gyr": f"{native_time:.12g}",
        "log10_rhoc_venus_inner250_at_native_time": f"{venus_native_match:.9g}",
        "log10_rhoc_frank_native_inner200": f"{frank_native_log200[frank_native_index]:.9g}",
        "delta_native_250_minus_200_dex": f"{venus_native_match-frank_native_log200[frank_native_index]:.9g}",
    }
    append_row(args.output, row)
    for key, value in row.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
