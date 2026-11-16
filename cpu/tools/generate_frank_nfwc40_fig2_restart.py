#!/usr/bin/env python3
# Construct and sample an isotropic energy-truncated NFW equilibrium.
# Use G=M_s=r_s=1, with M_s=4*pi*rho_s*r_s^3. Input controls set the lowered DF and quadrature.
# Output is an equal-mass SphDSMC restart plus the self-consistent model and sampling metadata.

"""Generate a self-consistent energy-truncated isotropic NFW IC.

The model follows Drakos, Taylor & Benson (2017): lower the isotropic NFW
distribution function at a dimensionless truncation energy ``Z_t`` and solve
Poisson's equation self-consistently.  ``Z_t`` is chosen so that

    M_total / M_vir,untruncated = 1.27

The defaults reproduce NFWc40* in Frank's Table 1; command-line parameters
select the other tabulated halos without changing the construction.  Units
are G=M_s=r_s=1, where M_s=4*pi*rho_s*r_s^3.  Consequently the code density
is (rho/rho_s)/(4*pi), while radii and velocities are in r_s and
sqrt(G*M_s/r_s), respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.integrate as scipy_integrate
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


if not hasattr(scipy_integrate, "cumulative_trapezoid"):
    scipy_integrate.cumulative_trapezoid = scipy_integrate.cumtrapz
cumulative_trapezoid = scipy_integrate.cumulative_trapezoid
solve_ivp = scipy_integrate.solve_ivp


PI = math.pi
PIANISTPY_TREE_SHA = "304d217b5e50aa029f64af0ead9ad7c3601e4fc3"


def nfw_mass_untruncated(radius: float | np.ndarray) -> float | np.ndarray:
    radius = np.asarray(radius)
    return np.log1p(radius) - radius / (1.0 + radius)


def nfw_df_lane2022(energy: np.ndarray) -> np.ndarray:
    """PIANISTpy's fitted isotropic NFW DF in rho_s/v_s^3 units."""
    energy = np.asarray(energy, dtype=float)
    output = np.zeros_like(energy)
    # Evaluate the fitted NFW distribution only on its physical relative-energy interval.
    valid = (energy > 0.0) & (energy < 1.0)
    e = energy[valid]
    one_minus_e = 1.0 - e
    polynomial = (
        0.0926081086527954
        + 0.0232272797489981 * e
        + 0.258346829924101 * e**2
        - 2.97848277491979 * e**3
        + 14.9776586391246 * e**4
        - 46.6587221550258 * e**5
        + 92.6397009471828 * e**6
        - 117.647787290798 * e**7
        + 92.5144063082258 * e**8
        - 41.0268009529576 * e**9
        + 7.84806318891231 * e**10
    )
    output[valid] = (
        e**1.5
        / one_minus_e**2.5
        * (-np.log(e) / one_minus_e) ** (-2.75)
        * polynomial
    )
    return output


# The retained energy is relative to the finite tidal boundary, where psi=0.
# Subtracting f(Z_t) makes the lowered DF vanish at that energy boundary.
def lowered_df(energy: np.ndarray, z_trunc: float) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    f_cut = float(nfw_df_lane2022(np.array([z_trunc]))[0])
    output = nfw_df_lane2022(energy + z_trunc) - f_cut
    return np.maximum(output, 0.0)


# Integrate the isotropic velocity measure 4*pi*v^2*dv at fixed psi.
# This table supplies the density for the self-consistent Poisson solve.
def make_density_of_potential(
    z_trunc: float,
    potential_points: int,
    quadrature_points: int,
) -> PchipInterpolator:
    psi_center = 1.0 - z_trunc
    parameter = np.linspace(0.0, 1.0, potential_points)
    # Cluster points at both the tidal boundary and the divergent NFW cusp.
    psi = psi_center * np.sin(0.5 * PI * parameter) ** 2
    nodes, weights = leggauss(quadrature_points)
    unit_energy = 0.5 * (nodes + 1.0)
    unit_weight = 0.5 * weights
    # Map the integration variable E/Psi to [0,1] at each potential value.
    energy = psi[:, None] * unit_energy[None, :]
    integrand = lowered_df(energy, z_trunc) * np.sqrt(
        1.0 - unit_energy[None, :]
    )
    # Changing variables from speed to relative energy gives the factor 4*pi*sqrt(2)*Psi^(3/2).
    density = (
        4.0
        * PI
        * math.sqrt(2.0)
        * psi**1.5
        * np.sum(unit_weight[None, :] * integrand, axis=1)
    )
    density[0] = 0.0
    return PchipInterpolator(psi, density, extrapolate=False)


def solve_lowered_nfw(
    z_trunc: float,
    potential_points: int,
    quadrature_points: int,
) -> tuple[float, float, PchipInterpolator, object]:
    psi_center = 1.0 - z_trunc
    rho_of_psi = make_density_of_potential(
        z_trunc, potential_points, quadrature_points
    )
    radius_start = 2.0e-7
    # The lowered model retains the NFW rho~1/r cusp, for which
    # psi_center-psi~r/2 and m~r^2/2.
    initial_state = np.array([
        psi_center - 0.5 * radius_start,
        0.5 * radius_start**2,
    ])

    def right_hand_side(radius: float, state: np.ndarray) -> np.ndarray:
        psi, mass = state
        delta = psi_center - psi
        if psi <= 0.0:
            density_tilde = 0.0
        elif delta < 2.0e-6:
            density_tilde = 1.0 / max(2.0 * delta, 1.0e-300)
        else:
            density_tilde = float(rho_of_psi(min(psi, psi_center)))
        # The equations advance dPsi/dr=-m/r^2 and dm/dr=r^2*(rho/rho_s) in these halo units.
        return np.array([
            -mass / radius**2,
            radius**2 * density_tilde,
        ])

    def tidal_boundary(_radius: float, state: np.ndarray) -> float:
        return float(state[0])

    # Stop the self-consistent model at the first outward crossing of Psi=0.
    tidal_boundary.terminal = True
    tidal_boundary.direction = -1
    solution = solve_ivp(
        right_hand_side,
        (radius_start, 2.0e4),
        initial_state,
        events=tidal_boundary,
        rtol=2.0e-10,
        atol=(2.0e-12, 2.0e-13),
        max_step=0.1,
        dense_output=True,
    )
    if not solution.t_events[0].size:
        raise RuntimeError(f"No tidal boundary found for Z_t={z_trunc}")
    tidal_radius = float(solution.t_events[0][0])
    total_mass = float(solution.sol(tidal_radius)[1])
    return tidal_radius, total_mass, rho_of_psi, solution


def tune_truncation(
    concentration: float,
    total_to_virial_mass: float,
    potential_points: int,
    quadrature_points: int,
) -> tuple[float, float, float, PchipInterpolator, object]:
    untruncated_virial_mass = float(nfw_mass_untruncated(concentration))
    # Use the untruncated NFW mass at the requested concentration to define the target total mass.
    target_total_mass = total_to_virial_mass * untruncated_virial_mass

    def residual(z_trunc: float) -> float:
        return solve_lowered_nfw(
            z_trunc, potential_points, quadrature_points
        )[1] - target_total_mass

    # Adjust the lowering energy until the self-consistent model has that target mass.
    z_trunc = brentq(residual, 0.005, 0.20, xtol=2.0e-11, rtol=2.0e-11)
    tidal_radius, total_mass, rho_of_psi, solution = solve_lowered_nfw(
        z_trunc, potential_points, quadrature_points
    )
    return z_trunc, tidal_radius, total_mass, rho_of_psi, solution


# Write equal-mass SphDSMC records in increasing radius. Stored L is specific
# angular momentum. Each record keeps its radius, radial velocity and L together.
def write_restart(
    path: Path,
    radius: np.ndarray,
    radial_velocity: np.ndarray,
    angular_momentum: np.ndarray,
    particle_mass: float,
) -> dict[str, float]:
    order = np.argsort(radius, kind="stable")
    radius = np.asarray(radius[order], dtype="<f8")
    radial_velocity = np.asarray(radial_velocity[order], dtype="<f8")
    angular_momentum = np.asarray(angular_momentum[order], dtype="<f8")
    particle_ids = np.arange(radius.size, dtype="<u8")
    kinetic = particle_mass * np.sum(
        0.5
        * (
            radial_velocity**2
            + (angular_momentum / radius) ** 2
        )
    )
    ranks = np.arange(radius.size, dtype=float)
    potential = -particle_mass * np.sum(
        (ranks + 0.5) * particle_mass / radius
    )
    initial_energy = float(kinetic + potential)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        magic = b"VENUSSPHDSMC1"
        handle.write(magic + b"\0" * (16 - len(magic)))
        handle.write(struct.pack(
            "<IQddqdQqq",
            2,
            radius.size,
            particle_mass,
            0.0,
            0,
            initial_energy,
            0,
            0,
            0,
        ))
        records = np.empty(
            radius.size,
            dtype=np.dtype([
                ("r", "<f8"),
                ("vr", "<f8"),
                ("L", "<f8"),
                ("id", "<u8"),
            ]),
        )
        records["r"] = radius
        records["vr"] = radial_velocity
        records["L"] = angular_momentum
        records["id"] = particle_ids
        handle.write(records.tobytes())
        handle.flush()
    temporary.replace(path)
    return {
        "kinetic_energy": float(kinetic),
        "potential_energy": float(potential),
        "total_energy": initial_energy,
        "virial_ratio": float(2.0 * kinetic / abs(potential)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument(
        "--paper-target",
        default="Frank Figure 2 NFWc40*",
        help="Descriptive provenance label written to the metadata",
    )
    parser.add_argument("--concentration", type=float, default=40.0)
    parser.add_argument("--total-to-virial-mass", type=float, default=1.27)
    parser.add_argument("--potential-points", type=int, default=6000)
    parser.add_argument("--quadrature-points", type=int, default=160)
    parser.add_argument("--radial-points", type=int, default=200_000)
    parser.add_argument("--speed-cdf-points", type=int, default=512)
    parser.add_argument("--sample-batch", type=int, default=4096)
    args = parser.parse_args()
    if args.particles < 2:
        raise ValueError("--particles must be at least two")

    z_trunc, tidal_radius, total_mass, rho_of_psi, solution = tune_truncation(
        args.concentration,
        args.total_to_virial_mass,
        args.potential_points,
        args.quadrature_points,
    )
    radial_grid = np.geomspace(2.0e-7, tidal_radius, args.radial_points)
    psi_grid, mass_grid = solution.sol(radial_grid)
    mass_grid = np.maximum.accumulate(mass_grid)
    rng = np.random.default_rng(args.seed)
    # Invert the solved enclosed mass for particle radii, preserving the model's spherical mass measure.
    mass_quantiles = np.sort(rng.random(args.particles)) * total_mass
    radius = np.interp(mass_quantiles, mass_grid, radial_grid)
    psi = np.clip(
        np.interp(radius, radial_grid, psi_grid),
        0.0,
        1.0 - z_trunc,
    )

    x_speed = np.linspace(0.0, 1.0, args.speed_cdf_points) ** 2
    x_speed2 = x_speed**2
    speed = np.empty(args.particles, dtype=float)
    asymptotic_outer_samples = 0
    for begin in range(0, args.particles, args.sample_batch):
        end = min(args.particles, begin + args.sample_batch)
        psi_batch = psi[begin:end]
        relative_energy = psi_batch[:, None] * (
            1.0 - x_speed2[None, :]
        )
        # The isotropic conditional speed law contains the v^2 velocity-space Jacobian.
        speed_pdf = x_speed2[None, :] * lowered_df(
            relative_energy, z_trunc
        )
        speed_cdf = cumulative_trapezoid(
            speed_pdf, x_speed, axis=1, initial=0.0
        )
        normalization = speed_cdf[:, -1]
        valid = np.isfinite(normalization) & (normalization > 0.0)
        x_sample = np.empty(end - begin, dtype=float)
        if np.any(valid):
            valid_cdf = speed_cdf[valid] / normalization[valid, None]
            uniform = rng.random(np.count_nonzero(valid))
            upper = np.sum(valid_cdf < uniform[:, None], axis=1)
            upper = np.clip(upper, 1, args.speed_cdf_points - 1)
            row = np.arange(upper.size)
            cdf0 = valid_cdf[row, upper - 1]
            cdf1 = valid_cdf[row, upper]
            fraction = np.divide(
                uniform - cdf0,
                cdf1 - cdf0,
                out=np.zeros_like(uniform),
                where=cdf1 > cdf0,
            )
            x_sample[valid] = (
                x_speed[upper - 1]
                + fraction * (x_speed[upper] - x_speed[upper - 1])
            )
        if np.any(~valid):
            # At the tidal edge f_T(E)~E, so x^2 follows Beta(3/2, 2).
            x_sample[~valid] = np.sqrt(
                rng.beta(1.5, 2.0, size=np.count_nonzero(~valid))
            )
            asymptotic_outer_samples += int(np.count_nonzero(~valid))
        speed[begin:end] = np.sqrt(2.0 * psi_batch) * x_sample

    # Sample a uniform direction cosine, then retain radial velocity and the magnitude of specific angular momentum.
    mu = 2.0 * rng.random(args.particles) - 1.0
    radial_velocity = speed * mu
    angular_momentum = radius * speed * np.sqrt(
        np.maximum(1.0 - mu**2, 0.0)
    )
    particle_mass = total_mass / args.particles
    energy_diagnostics = write_restart(
        args.output,
        radius,
        radial_velocity,
        angular_momentum,
        particle_mass,
    )
    untruncated_virial_mass = float(
        nfw_mass_untruncated(args.concentration)
    )
    truncated_mass_at_virial = float(solution.sol(args.concentration)[1])
    metadata = {
        "profile": "Drakos2017 energy-truncated isotropic NFW",
        "paper_target": args.paper_target,
        "pianistpy_tree_sha": PIANISTPY_TREE_SHA,
        "particles": args.particles,
        "seed": args.seed,
        "units": "G=M_s=r_s=1",
        "concentration": args.concentration,
        "target_total_to_untruncated_virial_mass": args.total_to_virial_mass,
        "z_trunc": z_trunc,
        "tidal_radius_over_r_s": tidal_radius,
        "total_mass_over_M_s": total_mass,
        "untruncated_virial_mass_over_M_s": untruncated_virial_mass,
        "truncated_mass_at_rvir_over_M_s": truncated_mass_at_virial,
        "truncated_to_untruncated_mass_at_rvir": (
            truncated_mass_at_virial / untruncated_virial_mass
        ),
        "total_to_truncated_mass_at_rvir": (
            total_mass / truncated_mass_at_virial
        ),
        "particle_mass": particle_mass,
        "asymptotic_outer_velocity_samples": asymptotic_outer_samples,
        **energy_diagnostics,
    }
    # Write the actual truncation, mass ratios, sampling controls and initial energies beside the restart.
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
