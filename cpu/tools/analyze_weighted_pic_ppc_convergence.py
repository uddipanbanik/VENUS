#!/usr/bin/env python3
# Compare WPIC density and logarithmic slopes across adjacent marker-count resolutions.
# Inputs are gamma/PPC-named run directories with actual profile CSVs and optional runtime logs.
# Outputs retain common actual epochs, fixed radial bands, volume-weighted errors and effective-count statistics.

"""Measure radial-profile convergence across weighted-PIC marker counts.

Adjacent particles-per-cell (PPC) levels are compared at common physical
times.  The report separates density error from logarithmic-slope error and
records the effective marker number, so apparent disagreement can be related
to sampling noise rather than treated as an unexplained profile difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


CASE_PATTERN = re.compile(r"gamma(?P<gamma>[0-9]+p[0-9]+)_ppc(?P<ppc>[0-9]+)$")
ELAPSED_PATTERN = re.compile(r"\belapsed=(?P<elapsed>[0-9.]+)")
ZONES = {
    "inner": (-1.0, 0.0),
    "middle": (0.0, 1.0),
    "outer": (1.0, 3.0),
    "all": (-1.0, 3.0),
}


def parse_gamma(token: str) -> float:
    """Decode a directory-safe initial cusp slope such as ``0p5``."""
    return float(token.replace("p", "."))


def load_case(directory: Path) -> tuple[float, int, dict[float, np.ndarray], float | None]:
    """Load one named gamma/PPC run and its optional final wall-clock cost."""
    match = CASE_PATTERN.fullmatch(directory.name)
    if match is None:
        raise RuntimeError(f"Unrecognized case directory {directory}")
    gamma = parse_gamma(match.group("gamma"))
    ppc = int(match.group("ppc"))
    profiles: dict[float, np.ndarray] = {}
    for path in sorted(directory.glob("profile_*.csv")):
        rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        required = {"time", "radius", "density", "neff"}
        if not required.issubset(rows.dtype.names or ()):
            raise RuntimeError(f"{path} is missing {sorted(required)}")
        profiles[float(rows["time"][0])] = rows
    elapsed = None
    log_path = directory / "run.log"
    if log_path.exists():
        matches = ELAPSED_PATTERN.findall(log_path.read_text(errors="replace"))
        if matches:
            elapsed = float(matches[-1])
    return gamma, ppc, profiles, elapsed


def cell_volumes(radius: np.ndarray) -> np.ndarray:
    """Infer logarithmic shell edges and return spherical cell volumes."""
    edges = np.empty(radius.size + 1, dtype=float)
    # Infer shell faces as geometric means of neighboring logarithmic radial centers.
    edges[1:-1] = np.sqrt(radius[:-1] * radius[1:])
    edges[0] = radius[0] * math.sqrt(radius[0] / radius[1])
    edges[-1] = radius[-1] * math.sqrt(radius[-1] / radius[-2])
    return 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)


def valid_density_mask(rows: np.ndarray, floor_relative: float) -> np.ndarray:
    """Select positive, resolved density cells in the comparison domain."""
    radius = np.asarray(rows["radius"], dtype=float)
    density = np.asarray(rows["density"], dtype=float)
    positive = np.isfinite(density) & (density > 0.0)
    peak = float(np.max(density[positive])) if np.any(positive) else 0.0
    return (
        np.isfinite(radius)
        & (radius >= 0.1)
        & (radius <= 900.0)
        & positive
        & (density >= max(peak * floor_relative, np.finfo(float).tiny))
    )


def local_poly_slope(
    radius: np.ndarray,
    density: np.ndarray,
    window_points: int,
    poly_order: int = 2,
) -> np.ndarray:
    """Estimate the local power-law index ``d log(rho)/d log(r)``."""
    x = np.log10(radius)
    y = np.log10(density)
    count = x.size
    # Use an odd fitting window that fits inside the available positive-density array.
    window = min(max(int(window_points) | 1, 5), count if count % 2 else count - 1)
    slope = np.full(count, np.nan, dtype=float)
    if window < 3:
        return slope
    half = window // 2
    for index in range(count):
        begin = max(0, index - half)
        end = min(count, index + half + 1)
        if end - begin < window:
            if begin == 0:
                end = min(count, window)
            else:
                begin = max(0, count - window)
        coefficients = np.polyfit(
            x[begin:end] - x[index],
            y[begin:end],
            min(poly_order, end - begin - 1),
        )
        slope[index] = float(coefficients[-2])
    # Exclude boundary centers lacking a complete centered window from subsequent slope statistics.
    slope[:half] = np.nan
    slope[-half:] = np.nan
    return slope


def compare_profiles(
    low: np.ndarray,
    high: np.ndarray,
    floor_relative: float,
    slope_window: int,
) -> list[dict[str, float | str]]:
    """Measure density and slope differences in fixed logarithmic-radius zones.

    ``density_mass_l1`` volume-weights the absolute density error and
    normalizes it by reference mass in the zone.  The dex RMS measures
    fractional profile shape, while ``slope_rms`` isolates cusp-gradient
    convergence.
    """
    low_mask = valid_density_mask(low, floor_relative)
    high_mask = valid_density_mask(high, floor_relative)
    common = low_mask & high_mask
    radius = np.asarray(high["radius"], dtype=float)
    low_density = np.asarray(low["density"], dtype=float)
    high_density = np.asarray(high["density"], dtype=float)
    volumes = cell_volumes(radius)

    slope_low = np.full(radius.size, np.nan, dtype=float)
    slope_high = np.full(radius.size, np.nan, dtype=float)
    slope_low[low_mask] = local_poly_slope(radius[low_mask], low_density[low_mask], slope_window)
    slope_high[high_mask] = local_poly_slope(radius[high_mask], high_density[high_mask], slope_window)

    log_radius = np.log10(radius)
    rows: list[dict[str, float | str]] = []
    for zone, (minimum, maximum) in ZONES.items():
        zone_mask = common & (log_radius >= minimum) & (log_radius <= maximum)
        if not np.any(zone_mask):
            continue
        log_delta = np.log10(low_density[zone_mask]) - np.log10(high_density[zone_mask])
        # Normalize the integrated absolute density difference by reference mass in that same radial band.
        mass_scale = float(np.sum(volumes[zone_mask] * high_density[zone_mask]))
        density_l1 = (
            float(np.sum(volumes[zone_mask] * np.abs(low_density[zone_mask] - high_density[zone_mask])))
            / mass_scale
            if mass_scale > 0.0
            else float("nan")
        )
        slope_mask = zone_mask & np.isfinite(slope_low) & np.isfinite(slope_high)
        slope_rms = (
            float(np.sqrt(np.mean((slope_low[slope_mask] - slope_high[slope_mask]) ** 2)))
            if np.any(slope_mask)
            else float("nan")
        )
        neff = np.asarray(high["neff"][zone_mask], dtype=float)
        rows.append(
            {
                "zone": zone,
                "radial_bins": int(np.count_nonzero(zone_mask)),
                "density_log_rms_dex": float(np.sqrt(np.mean(log_delta**2))),
                "density_mass_l1": density_l1,
                "slope_rms": slope_rms,
                "reference_neff_p10": float(np.quantile(neff, 0.10)),
                "reference_neff_median": float(np.median(neff)),
            }
        )
    return rows


def main() -> None:
    # Compare only adjacent available PPC levels; this reveals whether errors
    # decrease systematically rather than merely matching one chosen reference.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--density-floor-relative", type=float, default=1.0e-10)
    parser.add_argument("--slope-window-points", type=int, default=21)
    arguments = parser.parse_args()

    cases: dict[float, dict[int, tuple[dict[float, np.ndarray], float | None]]] = {}
    for directory in sorted(path for path in arguments.input_root.iterdir() if path.is_dir()):
        match = CASE_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        gamma, ppc, profiles, elapsed = load_case(directory)
        cases.setdefault(gamma, {})[ppc] = (profiles, elapsed)

    detailed: list[dict[str, float | int | str | None]] = []
    runtime_rows: list[dict[str, float | int | None]] = []
    for gamma, levels in sorted(cases.items()):
        for ppc, (_, elapsed) in sorted(levels.items()):
            runtime_rows.append({"gamma": gamma, "ppc": ppc, "elapsed_seconds": elapsed})
        ppcs = sorted(levels)
        for low_ppc, high_ppc in zip(ppcs[:-1], ppcs[1:]):
            low_profiles = levels[low_ppc][0]
            high_profiles = levels[high_ppc][0]
            # Compare only actual epochs present at both resolutions.
            for time in sorted(set(low_profiles) & set(high_profiles)):
                for metrics in compare_profiles(
                    low_profiles[time],
                    high_profiles[time],
                    arguments.density_floor_relative,
                    arguments.slope_window_points,
                ):
                    detailed.append(
                        {
                            "gamma": gamma,
                            "ppc_low": low_ppc,
                            "ppc_high": high_ppc,
                            "time": time,
                            **metrics,
                        }
                    )

    arguments.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = arguments.output_prefix.with_name(arguments.output_prefix.name + "_detail.csv")
    runtime_path = arguments.output_prefix.with_name(arguments.output_prefix.name + "_runtime.csv")
    summary_path = arguments.output_prefix.with_name(arguments.output_prefix.name + "_summary.json")

    if not detailed:
        raise RuntimeError("No adjacent-resolution comparisons were available")
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    with runtime_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runtime_rows[0]))
        writer.writeheader()
        writer.writerows(runtime_rows)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for gamma, levels in sorted(cases.items()):
        gamma_key = f"{gamma:g}"
        summary[gamma_key] = {}
        ppcs = sorted(levels)
        for low_ppc, high_ppc in zip(ppcs[:-1], ppcs[1:]):
            pair_key = f"{low_ppc}_to_{high_ppc}"
            selected = [
                row
                for row in detailed
                if row["gamma"] == gamma
                and row["ppc_low"] == low_ppc
                and row["ppc_high"] == high_ppc
                and row["time"] > 0.0
            ]
            summary[gamma_key][pair_key] = {}
            for zone in ZONES:
                zone_rows = [row for row in selected if row["zone"] == zone]
                if not zone_rows:
                    continue
                # Keep worst-time and final-time errors separate when assessing a resolution trend.
                summary[gamma_key][pair_key][zone] = {
                    "max_density_log_rms_dex": max(float(row["density_log_rms_dex"]) for row in zone_rows),
                    "max_density_mass_l1": max(float(row["density_mass_l1"]) for row in zone_rows),
                    "max_slope_rms": max(float(row["slope_rms"]) for row in zone_rows),
                    "final_density_log_rms_dex": float(zone_rows[-1]["density_log_rms_dex"]),
                    "final_density_mass_l1": float(zone_rows[-1]["density_mass_l1"]),
                    "final_slope_rms": float(zone_rows[-1]["slope_rms"]),
                }

    summary_path.write_text(json.dumps({"convergence": summary, "runtime": runtime_rows}, indent=2) + "\n")
    print(detail_path)
    print(runtime_path)
    print(summary_path)


if __name__ == "__main__":
    main()
