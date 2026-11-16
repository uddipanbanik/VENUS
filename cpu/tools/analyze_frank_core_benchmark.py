#!/usr/bin/env python3
# Compare cored-halo density, random velocity dispersion and anisotropy with digitized reference curves.
# Inputs are VENUS moment-profile CSVs and a named reference table. Output JSON and an optional plot use identical rows.
# An explicit time-unit conversion and tolerance determine which actual solver epochs are comparable.

"""Compare VENUS core-halo profiles with Frank Figure 3 particle curves.

VENUS stores spherical mass moments, whereas the digitized reference file
stores already reduced density, one-dimensional dispersion, and anisotropy
curves.  This script first reconstructs those same physical observables from
``rho``, ``j``, ``p``, and ``pt`` and only then measures profile differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RHO_S_CODE = 1.0 / (4.0 * np.pi)
# Published comparison times, expressed in Frank's core-relaxation time.
REFERENCE_TIMES = np.array([0.0, 115.96119929453263, 220.8994708994709, 250.0])
COLORS = {"nbody": "#3298c7", "sphdsmc": "#ff982f", "venus": "#161616"}
LABELS = {
    "density": r"$\log_{10}(\rho/\rho_s)$",
    "sigma": r"$\sigma_{\rm 1D}/v_s$",
    "beta": r"$\beta_{\rm ani}$",
}
YLIMS = {"density": (-11.0, 5.0), "sigma": (0.0, 0.51), "beta": (-1.0, 1.0)}


def load_csv(path: Path) -> np.ndarray:
    """Read a named-column CSV, including the reference's string columns."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="ascii")


def venus_profile(data: np.ndarray, time: float) -> dict[str, np.ndarray]:
    """Reduce one VENUS snapshot to the three observables in Figure 3.

    The radial bulk speed is ``u_r=j/rho`` and the random radial variance is
    ``sigma_r^2=p/rho-u_r^2``.  VENUS ``pt/rho`` is the sum of the two
    tangential variances, so

    ``sigma_1D^2=(sigma_r^2+sigma_t^2)/3`` and
    ``beta_ani=1-sigma_t^2/(2 sigma_r^2)``.

    These definitions separate coherent inflow/outflow from random kinetic
    support before the benchmark comparison is made.
    """
    rows = data[np.isclose(data["time"], time, rtol=0.0, atol=1.0e-12)]
    rho = rows["rho"]
    # Remove coherent radial motion before interpreting the second velocity moment as random dispersion.
    radial_mean = np.divide(rows["j"], rho, out=np.zeros_like(rho), where=rho > 0.0)
    sigma_r2 = np.maximum(
        np.divide(rows["p"], rho, out=np.zeros_like(rho), where=rho > 0.0)
        - radial_mean * radial_mean,
        0.0,
    )
    sigma_t2 = np.maximum(
        np.divide(rows["pt"], rho, out=np.zeros_like(rho), where=rho > 0.0),
        0.0,
    )
    # Average all three random velocity components to form the reference one-dimensional dispersion.
    sigma_1d = np.sqrt(np.maximum((sigma_r2 + sigma_t2) / 3.0, 0.0))
    beta = np.divide(
        sigma_t2,
        2.0 * sigma_r2,
        out=np.zeros_like(sigma_t2),
        where=sigma_r2 > 0.0,
    )
    beta = 1.0 - beta
    return {
        "x": np.log10(rows["r"]),
        "density": np.log10(np.maximum(rho / RHO_S_CODE, 1.0e-300)),
        "sigma": sigma_1d,
        "beta": beta,
    }


def reference_curve(
    reference: np.ndarray,
    source: str,
    quantity: str,
    time_tcore: float,
    bin_width: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and lightly consolidate one digitized reference curve.

    Several digitized points may occupy the same narrow log-radius interval.
    Taking their median suppresses digitizer oversampling without mixing
    distinct physical radii or smoothing the VENUS prediction.
    """
    mask = (
        (reference["source"] == source)
        & (reference["quantity"] == quantity)
        & np.isclose(reference["time_tcore"], time_tcore, rtol=0.0, atol=1.0e-8)
    )
    x = np.asarray(reference["log10_r_over_rs"][mask], dtype=float)
    y = np.asarray(reference["value"][mask], dtype=float)
    if x.size == 0:
        return x, y
    # Consolidate digitizer oversampling within narrow reference log-radius intervals.
    bins = np.floor((x - x.min()) / bin_width).astype(int)
    unique_bins = np.unique(bins)
    xb = np.array([np.median(x[bins == value]) for value in unique_bins])
    yb = np.array([np.median(y[bins == value]) for value in unique_bins])
    order = np.argsort(xb)
    return xb[order], yb[order]


def curve_metrics(
    venus_x: np.ndarray,
    venus_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> dict[str, float]:
    """Measure VENUS-minus-reference residuals on reference radii.

    Density values are already logarithmic and therefore yield dex residuals;
    dispersion and anisotropy retain the linear units stored in the reference.
    Interpolation is restricted to the mutually represented radial interval.
    """
    use = (
        (ref_x >= max(-2.3, venus_x.min()))
        & (ref_x <= min(2.5, venus_x.max()))
        & np.isfinite(ref_y)
    )
    if np.count_nonzero(use) < 3:
        return {"points": 0, "bias": np.nan, "mae": np.nan, "rms": np.nan}
    # Compare on reference radii inside the shared represented interval. Density residuals are measured in dex.
    residual = np.interp(ref_x[use], venus_x, venus_y) - ref_y[use]
    return {
        "points": int(residual.size),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rms": float(np.sqrt(np.mean(residual * residual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument(
        "--profile-time-to-tcore",
        type=float,
        default=1.0,
        help=(
            "Multiply the profiles.csv time column by this factor before "
            "matching the published t/t_core values (default: 1)."
        ),
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.05,
        help="Maximum |t-t_reference| in t_core units for including a VENUS snapshot.",
    )
    args = parser.parse_args()

    # Match each published epoch to the nearest available solver snapshot.
    # The explicit tolerance prevents a distant output from being mislabeled
    # as a benchmark time merely because it is the nearest one in the file.
    profiles = load_csv(args.profiles)
    reference = load_csv(args.reference)
    available_native = np.unique(profiles["time"])
    available = available_native * args.profile_time_to_tcore
    matches: list[tuple[float, float, float]] = []
    for target in REFERENCE_TIMES:
        index = int(np.argmin(np.abs(available - target)))
        # Keep the actual converted source epoch distinct from the requested published epoch in the report.
        actual_tcore = float(available[index])
        if abs(actual_tcore - target) <= args.time_tolerance:
            matches.append((target, float(available_native[index]), actual_tcore))
    if not matches:
        raise RuntimeError("No VENUS snapshots match the published benchmark times.")

    # The JSON report is the quantitative product.  The optional figure uses
    # the same selected rows and reductions, so visual and numerical
    # comparisons cannot silently diverge.
    metrics: dict[str, object] = {
        "profiles": str(args.profiles),
        "reference": str(args.reference),
        "rho_s_code": RHO_S_CODE,
        "profile_time_to_tcore": args.profile_time_to_tcore,
        "matches": [],
    }
    figure = None
    axes = None
    if args.output_plot is not None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            3,
            len(matches),
            figsize=(3.35 * len(matches), 8.2),
            sharex=True,
            squeeze=False,
        )
    for column, (target, actual_native, actual_tcore) in enumerate(matches):
        venus = venus_profile(profiles, actual_native)
        match_metrics: dict[str, object] = {
            "target_time_tcore": target,
            "venus_time_native": actual_native,
            "venus_time_tcore": actual_tcore,
            "curves": {},
        }
        for row, quantity in enumerate(("density", "sigma", "beta")):
            axis = axes[row, column] if axes is not None else None
            for source in ("nbody", "sphdsmc"):
                ref_x, ref_y = reference_curve(reference, source, quantity, target)
                if axis is not None and ref_x.size:
                    axis.plot(
                        ref_x,
                        ref_y,
                        color=COLORS[source],
                        linewidth=1.7,
                        label=source if row == 0 and column == 0 else None,
                    )
                match_metrics["curves"][f"{source}_{quantity}"] = curve_metrics(
                    venus["x"], venus[quantity], ref_x, ref_y
                )
            if axis is not None:
                axis.plot(
                    venus["x"],
                    venus[quantity],
                    color=COLORS["venus"],
                    linewidth=1.4,
                    label="VENUS" if row == 0 and column == 0 else None,
                )
                axis.set_xlim(-2.5, 2.8)
                axis.set_ylim(*YLIMS[quantity])
                axis.tick_params(direction="in", top=True, right=True)
                if column == 0:
                    axis.set_ylabel(LABELS[quantity])
                if row == 0:
                    axis.set_title(rf"$t/t_{{\rm core}}={target:.3g}$")
                if row == 2:
                    axis.set_xlabel(r"$\log_{10}(r/r_s)$")
        metrics["matches"].append(match_metrics)

    if figure is not None and axes is not None:
        axes[0, 0].legend(frameon=False, fontsize=9)
        figure.tight_layout()
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output_plot, dpi=180)
        import matplotlib.pyplot as plt

        plt.close(figure)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="ascii") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output_json}")
    if args.output_plot is not None:
        print(f"Wrote {args.output_plot}")


if __name__ == "__main__":
    main()
