#!/usr/bin/env python3
# Measure inner density and logarithmic-slope changes relative to a collisionless WPIC control.
# Inputs are labeled profile directories and requested common epochs in radial code units.
# Output CSV/text reports fixed-band mass/shape errors, slope changes and sampling counts.

"""Measure density and inner-slope response in a subhalo-strength scan."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


RADIAL_BANDS = {
    "inner_0p1_0p3": (0.1, 0.3),
    "inner_0p3_1": (0.3, 1.0),
    "inner_0p1_1": (0.1, 1.0),
    "middle_1_3": (1.0, 3.0),
}
# Every reported band is a closed interval in the simulation's radial code
# coordinate.  The configured report domain is 0.1 <= r <= 3.0; the wider
# 0.1 <= r <= 900.0 interval below supplies context for centered slope fits.


def load_profiles(directory: Path) -> list[tuple[float, np.ndarray]]:
    loaded: dict[float, np.ndarray] = {}
    for path in sorted(directory.glob("profile_*.csv")):
        rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        required = {"time", "radius", "density", "neff"}
        if not required.issubset(rows.dtype.names or ()):
            raise RuntimeError(f"{path} is missing {sorted(required)}")
        loaded[float(rows["time"][0])] = rows
    if not loaded:
        raise RuntimeError(f"No profiles found in {directory}")
    return sorted(loaded.items())


def display_mask(
    rows: np.ndarray,
    minimum_radius: float,
    maximum_radius: float,
    floor_relative: float,
) -> np.ndarray:
    radius = np.asarray(rows["radius"], dtype=float)
    density = np.asarray(rows["density"], dtype=float)
    positive = np.isfinite(density) & (density > 0.0)
    peak = float(np.max(density[positive])) if np.any(positive) else 0.0
    return (
        np.isfinite(radius)
        & (radius >= minimum_radius)
        & (radius <= maximum_radius)
        & positive
        & (density >= max(peak * floor_relative, np.finfo(float).tiny))
    )


def local_poly_log_slope(
    radius: np.ndarray,
    density: np.ndarray,
    window_points: int = 21,
    poly_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Estimate the local logarithmic density slope on the supplied domain.

    After sorting and removing duplicate radii, the routine works in
    ``x=log10(r)`` and ``y=log10(rho)``.  At each center ``x_i`` it fits
    ``y = a_0 + a_1 (x-x_i) + ...`` by unweighted least squares, so the
    returned slope is ``a_1 = d log10(rho)/d log10(r) = d ln(rho)/d ln(r)``.
    Inputs must already be finite and positive; callers define the radial
    domain explicitly.  Half a window is returned as NaN at each domain edge,
    preventing one-sided boundary fits from entering band statistics.
    """
    order = np.argsort(radius)
    radius = np.asarray(radius[order], dtype=float)
    density = np.asarray(density[order], dtype=float)
    radius, unique = np.unique(radius, return_index=True)
    density = density[unique]
    x = np.log10(radius)
    y = np.log10(density)
    count = x.size
    window = min(
        max(int(window_points) | 1, 5),
        count if count % 2 else count - 1,
    )
    slope = np.full(count, np.nan, dtype=float)
    if window < 3:
        return radius, slope
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
    slope[:half] = np.nan
    slope[-half:] = np.nan
    return radius, slope


def parse_case(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("cases must be LABEL=PATH")
    return label, Path(raw_path)


def cell_volumes(radius: np.ndarray) -> np.ndarray:
    r"""Return spherical shell volumes associated with positive radial centers.

    Interior faces are geometric means ``sqrt(r_i r_{i+1})``.  The two outer
    faces continue the same logarithmic spacing by one half-step, and
    ``Delta V_i = 4 pi (r_{i+1/2}^3-r_{i-1/2}^3)/3``.  Thus subsequent density
    differences are weighted over the full three-dimensional shell volume,
    even though the profile is tabulated on a one-dimensional radial grid.
    """
    edges = np.empty(radius.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(radius[:-1] * radius[1:])
    edges[0] = radius[0] * math.sqrt(radius[0] / radius[1])
    edges[-1] = radius[-1] * math.sqrt(radius[-1] / radius[-2])
    return 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)


def profile_map(directory: Path) -> dict[float, np.ndarray]:
    return {time: rows for time, rows in load_profiles(directory)}


def paired_profile(
    reference: dict[float, np.ndarray],
    case: dict[float, np.ndarray],
    requested_time: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    if not reference or not case:
        raise RuntimeError("Reference and case must both contain profiles")
    reference_time = min(reference, key=lambda candidate: abs(candidate - requested_time))
    case_time = min(case, key=lambda candidate: abs(candidate - requested_time))
    # Require both actual profile clocks to represent the requested common epoch before measuring a response.
    tolerance = 1.0e-9 * max(1.0, abs(requested_time))
    if (
        abs(reference_time - requested_time) > tolerance
        or abs(case_time - requested_time) > tolerance
        or abs(reference_time - case_time) > tolerance
    ):
        raise RuntimeError(
            f"No paired profile at t={requested_time:g}; nearest times are "
            f"{reference_time:g} and {case_time:g}"
        )
    return reference_time, reference[reference_time], case[case_time]


def band_metrics(
    reference_rows: np.ndarray,
    case_rows: np.ndarray,
    minimum_radius: float,
    maximum_radius: float,
) -> dict[str, float | int]:
    r"""Compare two profiles on the closed band ``minimum_radius <= r <= maximum_radius``.

    Density residuals use bins common to both profiles in that band.
    ``density_mass_l1`` is
    ``sum(Delta V |rho-rho_ref|) / sum(Delta V rho_ref)``, with the spherical
    volumes defined by :func:`cell_volumes`; logarithmic RMS/median residuals
    are per-bin statistics.  Slopes are fitted on the wider, explicitly fixed
    context domain ``0.1 <= r <= 900.0`` so a centered 21-point polynomial
    window can support the requested report bands, then only slope centers
    inside the closed report band enter the medians and quantiles.
    """
    reference_valid = display_mask(reference_rows, minimum_radius, maximum_radius, 1.0e-10)
    case_valid = display_mask(case_rows, minimum_radius, maximum_radius, 1.0e-10)
    common = reference_valid & case_valid
    radius = np.asarray(reference_rows["radius"], dtype=float)
    reference_density = np.asarray(reference_rows["density"], dtype=float)
    case_density = np.asarray(case_rows["density"], dtype=float)
    if np.count_nonzero(common) < 3:
        raise RuntimeError(
            f"Too few common bins in r=[{minimum_radius:g},{maximum_radius:g}]"
        )

    # Fit slopes on a wider fixed domain so the requested report bands have neighboring support.
    all_reference_valid = display_mask(reference_rows, 0.1, 900.0, 1.0e-10)
    all_case_valid = display_mask(case_rows, 0.1, 900.0, 1.0e-10)
    reference_slope_radius, reference_slope = local_poly_log_slope(
        radius[all_reference_valid],
        reference_density[all_reference_valid],
    )
    case_slope_radius, case_slope = local_poly_log_slope(
        radius[all_case_valid],
        case_density[all_case_valid],
    )
    case_slope_finite = np.isfinite(case_slope)
    # Sample the case slope on the control's fitted radial coordinates within their common finite support.
    case_slope_on_reference = np.interp(
        reference_slope_radius,
        case_slope_radius[case_slope_finite],
        case_slope[case_slope_finite],
        left=np.nan,
        right=np.nan,
    )
    slope_band = (
        np.isfinite(reference_slope)
        & np.isfinite(case_slope_on_reference)
        & (reference_slope_radius >= minimum_radius)
        & (reference_slope_radius <= maximum_radius)
    )

    log_delta = np.log10(case_density[common]) - np.log10(reference_density[common])
    volumes = cell_volumes(radius)
    # Weight density differences by three-dimensional shell volume and normalize by control mass.
    reference_mass = float(np.sum(volumes[common] * reference_density[common]))
    density_mass_l1 = float(
        np.sum(volumes[common] * np.abs(case_density[common] - reference_density[common]))
        / reference_mass
    )
    return {
        "radial_bins": int(np.count_nonzero(common)),
        "density_log_rms_dex": float(np.sqrt(np.mean(log_delta**2))),
        "density_log_median_dex": float(np.median(log_delta)),
        "density_mass_l1": density_mass_l1,
        "reference_slope_median": float(np.median(reference_slope[slope_band])),
        "case_slope_median": float(np.median(case_slope_on_reference[slope_band])),
        "slope_change_median": float(
            np.median(
                case_slope_on_reference[slope_band] - reference_slope[slope_band]
            )
        ),
        "case_gamma_median": float(
            -np.median(case_slope_on_reference[slope_band])
        ),
        "case_slope_p10": float(
            np.quantile(case_slope_on_reference[slope_band], 0.10)
        ),
        "case_slope_p90": float(
            np.quantile(case_slope_on_reference[slope_band], 0.90)
        ),
        "case_neff_p10": float(
            np.quantile(np.asarray(case_rows["neff"][common], dtype=float), 0.10)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collisionless", type=Path, required=True)
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--time", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    reference = profile_map(arguments.collisionless)
    rows: list[dict[str, float | int | str]] = []
    for label, directory in arguments.case:
        case = profile_map(directory)
        for requested_time in arguments.time:
            time, reference_rows, case_rows = paired_profile(
                reference, case, requested_time
            )
            # Apply the same closed radial bands to every strength and requested actual time.
            for band, (minimum_radius, maximum_radius) in RADIAL_BANDS.items():
                rows.append(
                    {
                        "case": label,
                        "time": time,
                        "band": band,
                        **band_metrics(
                            reference_rows,
                            case_rows,
                            minimum_radius,
                            maximum_radius,
                        ),
                    }
                )

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            f"{row['case']} t={row['time']:g} {row['band']}: "
            f"gamma={row['case_gamma_median']:.4f}, "
            f"delta_slope={row['slope_change_median']:+.4f}, "
            f"density_rms={row['density_log_rms_dex']:.4f} dex, "
            f"mass_L1={row['density_mass_l1']:.4f}, "
            f"Neff_p10={row['case_neff_p10']:.1f}"
        )


if __name__ == "__main__":
    main()
