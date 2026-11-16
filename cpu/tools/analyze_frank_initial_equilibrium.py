#!/usr/bin/env python3
# Assess the earliest cored-halo profile against its analytic isotropic equilibrium.
# Construct target density, potential and Jeans radial dispersion in the same halo units.
# Output mass-weighted profile errors separates the resolved core and finite-wall potential domain.

"""Compare a VENUS t=0 profile with Frank's analytic isotropic equilibrium.

The target density is converted into its self-consistent potential and
isotropic Jeans velocity dispersion.  Density, potential, and random radial
support are then compared with the corresponding VENUS moments over the
radial region where the analytic model is represented above numerical noise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


PI = np.pi
# Frank's code normalization: rho_s = M_s/(4 pi r_s^3), with M_s=r_s=1.
RHO_S = 1.0 / (4.0 * PI)


def load_eddington_helpers(repo: Path):
    """Load the shared analytic density, mass, and potential constructions."""
    path = repo / "tools" / "sidm_nfw_eddington.py"
    spec = importlib.util.spec_from_file_location("venus_eddington", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_relative_l1(
    numerical: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return ``sum(weight*|numerical-target|)/sum(weight*|target|)``.

    The caller supplies spherical shell-mass weights, so well represented
    radial mass contributes more than nearly empty outer grid cells.
    """
    return float(
        np.sum(weight * np.abs(numerical - target))
        / np.sum(weight * np.maximum(np.abs(target), 1.0e-300))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-r-max", type=float, default=1.0e6)
    parser.add_argument("--model-points", type=int, default=250000)
    args = parser.parse_args()

    # Always test the earliest complete profile.  This keeps the diagnostic an
    # initial-condition test rather than mixing in subsequent Vlasov transport
    # or collisional evolution.
    table = np.genfromtxt(args.profiles, delimiter=",", names=True)
    initial_time = float(np.min(table["time"]))
    rows = table[np.isclose(table["time"], initial_time, rtol=0.0, atol=1.0e-14)]
    rows = rows[np.argsort(rows["r"])]
    radius = np.asarray(rows["r"], dtype=float)

    # Build the analytic equilibrium on a much finer logarithmic mesh.  The
    # enclosed mass gives dPhi/dr=GM(<r)/r^2; the helper applies the finite
    # outer-wall convention used by the benchmark initial condition.
    repo = Path(__file__).resolve().parents[1]
    helper = load_eddington_helpers(repo)
    model_radius = np.geomspace(1.0e-8, args.model_r_max, args.model_points)
    model_density = helper.frank_core_density(model_radius, RHO_S)
    model_mass = helper.cumulative_trapezoid(
        4.0 * PI * model_radius**2 * model_density,
        model_radius,
    )
    model_psi = helper.wall_relative_potential(model_radius, model_mass, 1.0)
    # Isotropic spherical Jeans equilibrium obeys
    #   d(rho sigma_r^2)/dr = -rho G M(<r)/r^2 .
    # With G=1 and vanishing pressure at the outer boundary, integrating
    # inward gives rho*sigma_r^2 = integral_r^R rho M/r'^2 dr'.
    jeans_integrand = model_density * model_mass / model_radius**2
    jeans_segment = 0.5 * (jeans_integrand[:-1] + jeans_integrand[1:]) * np.diff(model_radius)
    jeans_tail = np.zeros_like(model_radius)
    # Integrate the pressure contribution inward from vanishing outer pressure.
    jeans_tail[:-1] = np.cumsum(jeans_segment[::-1])[::-1]
    model_sigma_r2 = jeans_tail / model_density

    target_density = helper.frank_core_density(radius, RHO_S)
    target_potential = -np.interp(np.log(radius), np.log(model_radius), model_psi)
    target_sigma_r2 = np.interp(
        np.log(radius), np.log(model_radius), model_sigma_r2
    )
    # VENUS stores j=rho<u_r> and p=rho<u_r^2>.  Subtracting the squared bulk
    # speed is essential: coherent radial flow is not velocity dispersion.
    radial_mean = np.divide(
        rows["j"], rows["rho"], out=np.zeros_like(rows["rho"]), where=rows["rho"] > 0.0
    )
    # Subtract the squared radial bulk velocity from p/rho to obtain the random variance.
    numerical_sigma_r2 = np.divide(
        rows["p"], rows["rho"], out=np.zeros_like(rows["rho"]), where=rows["rho"] > 0.0
    ) - radial_mean**2
    # Weight errors by target mass in each spherical radial interval.  Separate
    # masks distinguish the resolved model, the central core, and the region
    # where the finite-wall potential comparison remains informative.
    shell_weight = (
        4.0 * PI * radius**2 * target_density * np.gradient(radius)
    )
    represented = target_density > target_density.max() * 1.0e-10
    core = represented & (radius < 1.0)
    potential_region = represented & (radius < 100.0)
    # For an isotropic distribution the two tangential second moments sum to twice the radial one.
    isotropy_residual = np.divide(
        rows["pt"],
        2.0 * rows["p"],
        out=np.ones_like(rows["p"]),
        where=rows["p"] > 0.0,
    ) - 1.0

    # Apply each observable's stated radial mask and weight to the same initial physical state.
    report = {
        "profiles": str(args.profiles),
        "time": initial_time,
        "density_weighted_relative_l1": weighted_relative_l1(
            rows["rho"][represented], target_density[represented], shell_weight[represented]
        ),
        "core_density_weighted_relative_l1_r_lt_1": weighted_relative_l1(
            rows["rho"][core], target_density[core], shell_weight[core]
        ),
        "sigma_r2_weighted_relative_l1": weighted_relative_l1(
            numerical_sigma_r2[represented], target_sigma_r2[represented], shell_weight[represented]
        ),
        "core_sigma_r2_weighted_relative_l1_r_lt_1": weighted_relative_l1(
            numerical_sigma_r2[core], target_sigma_r2[core], shell_weight[core]
        ),
        "potential_weighted_relative_l1_r_lt_100": weighted_relative_l1(
            rows["mu"][potential_region], target_potential[potential_region], shell_weight[potential_region]
        ),
        "maximum_abs_isotropy_residual": float(np.max(np.abs(isotropy_residual[represented]))),
        "inner_density_relative_error": float(rows["rho"][0] / target_density[0] - 1.0),
        "inner_sigma_r2_relative_error": float(numerical_sigma_r2[0] / target_sigma_r2[0] - 1.0),
        "inner_potential_relative_error": float(rows["mu"][0] / target_potential[0] - 1.0),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
