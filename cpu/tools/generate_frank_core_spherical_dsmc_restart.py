#!/usr/bin/env python3
# Sample an isotropic cored Eddington distribution into equal-mass SphDSMC particles.
# Inputs choose the model/CDF quadratures, particle count and RNG seed in G=M_s=r_s=1 units.
# Output is a version-2 binary restart containing sorted (r,v_r,L,id) records and initial energy.

"""Sample Frank's analytic isotropic Core DF directly into a SphDSMC restart.

This generator samples spherical particle coordinates directly.
Radii are sampled from the analytic enclosed-mass CDF and speeds are sampled
from the Eddington distribution
``p(v|r) proportional to v^2 f(Psi-v^2/2)``.  It is intended to distinguish
particle-orbit errors from errors introduced by sampling a coarse DF grid.
"""

from __future__ import annotations

import argparse
import importlib.util
import struct
from pathlib import Path

import numpy as np
import scipy.integrate as scipy_integrate


PI = np.pi
RHO_S_CODE = 1.0 / (4.0 * PI)
if not hasattr(scipy_integrate, "cumulative_trapezoid"):
    scipy_integrate.cumulative_trapezoid = scipy_integrate.cumtrapz
cumulative_trapezoid = scipy_integrate.cumulative_trapezoid


def load_eddington_helpers(repo: Path):
    """Load the maintained spherical density and Eddington-inversion API."""
    path = repo / "tools" / "sidm_nfw_eddington.py"
    spec = importlib.util.spec_from_file_location("venus_eddington_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The version-2 payload stores (r,vr,L,id) with one common particle mass.
# The initial potential energy uses the thin-shell self-energy convention.
def write_restart(
    path: Path,
    radius: np.ndarray,
    radial_velocity: np.ndarray,
    angular_momentum: np.ndarray,
    particle_mass: float,
) -> None:
    # Sort particle coordinates together so velocity and angular momentum stay attached to their radius.
    order = np.argsort(radius, kind="stable")
    radius = np.asarray(radius[order], dtype="<f8")
    radial_velocity = np.asarray(radial_velocity[order], dtype="<f8")
    angular_momentum = np.asarray(angular_momentum[order], dtype="<f8")
    particle_ids = np.arange(radius.size, dtype="<u8")

    kinetic = particle_mass * np.sum(
        0.5 * (
            radial_velocity * radial_velocity
            + (angular_momentum / radius) ** 2
        )
    )
    ranks = np.arange(radius.size, dtype=float)
    # For equal-mass shells, rank+1/2 includes the half self-shell energy in the spherical potential sum.
    potential = -particle_mass * np.sum(
        (ranks + 0.5) * particle_mass / radius
    )
    initial_energy = float(kinetic + potential)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Complete the binary payload before replacing the destination, so a reader sees a closed state.
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
    print(
        f"Wrote {radius.size} direct-DF particles to {path}; "
        f"M={particle_mass * radius.size:.17g}, "
        f"K0={kinetic:.17g}, U0={potential:.17g}, "
        f"2K/|U|={2.0 * kinetic / abs(potential):.12g}, "
        f"E0={initial_energy:.17g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--model-r-max", type=float, default=1.0e6)
    parser.add_argument("--model-points", type=int, default=250_000)
    parser.add_argument("--energy-points", type=int, default=32768)
    parser.add_argument("--eddington-quadrature", type=int, default=256)
    parser.add_argument("--speed-cdf-points", type=int, default=512)
    parser.add_argument("--sample-batch", type=int, default=4096)
    parser.add_argument(
        "--radial-sampling",
        choices=("iid", "stratified"),
        default="iid",
        help=(
            "Draw independent enclosed-mass quantiles as in a particle IC "
            "generator, or use one randomly shifted systematic quantile per "
            "particle. Frank's PIANISTpy IC description corresponds to iid."
        ),
    )
    args = parser.parse_args()
    if args.particles < 2:
        raise ValueError("--particles must be at least two")

    repo = Path(__file__).resolve().parents[1]
    eddington = load_eddington_helpers(repo)
    r_model = np.geomspace(1.0e-8, args.model_r_max, args.model_points)
    rho_model = eddington.frank_core_density(r_model, RHO_S_CODE)
    rho_model[-1] = 0.0
    # Integrate 4*pi*r^2*rho to obtain the radial mass CDF used for particle positions.
    mass_model = cumulative_trapezoid(
        4.0 * PI * r_model * r_model * rho_model,
        r_model,
        initial=0.0,
    )
    # Binding energy uses psi=Phi(model_r_max)-Phi(r), with psi=0 at the
    # finite model boundary. The Eddington speed CDF uses the same offset.
    psi_model = eddington.wall_relative_potential(r_model, mass_model, 1.0)
    energy, f_energy, diagnostics = eddington.eddington_table(
        psi_model,
        rho_model,
        args.energy_points,
        args.eddington_quadrature,
        True,
    )
    rng = np.random.default_rng(args.seed)
    # Independent quantiles reproduce ordinary random particle sampling. The alternative spreads quantiles systematically.
    if args.radial_sampling == "iid":
        quantiles = np.sort(rng.random(args.particles))
    else:
        offset = rng.random()
        quantiles = (
            np.arange(args.particles, dtype=float) + offset
        ) / args.particles
    target_mass = quantiles * mass_model[-1]
    radius = np.interp(target_mass, mass_model, r_model)
    psi = np.clip(
        np.interp(np.log(radius), np.log(r_model), psi_model),
        energy[0],
        energy[-1],
    )
    if args.speed_cdf_points < 32:
        raise ValueError("--speed-cdf-points must be at least 32")
    if args.sample_batch < 1:
        raise ValueError("--sample-batch must be positive")
    # Sample the exact conditional speed law
    #     p(x | Psi) dx proportional to x^2 f(Psi [1-x^2]) dx,
    # where x=v/v_esc.  Direct inverse CDF sampling is robust even for the
    # innermost particles, whose f(E) is sharply concentrated near E=Psi and
    # makes simple rejection sampling pathologically inefficient.  A
    # quadratic x grid resolves that low-speed peak.
    x_speed = np.linspace(0.0, 1.0, args.speed_cdf_points) ** 2
    x_speed2 = x_speed * x_speed
    speed = np.empty(args.particles, dtype=float)
    asymptotic_outer_samples = 0
    for begin in range(0, args.particles, args.sample_batch):
        end = min(args.particles, begin + args.sample_batch)
        psi_batch = psi[begin:end]
        # At fixed radius, E=Psi-v^2/2=Psi*(1-x^2) sets the conditional speed distribution.
        relative_energy = np.clip(
            psi_batch[:, None] * (1.0 - x_speed2[None, :]),
            energy[0],
            energy[-1],
        )
        speed_pdf = x_speed2[None, :] * np.interp(
            relative_energy.ravel(),
            energy,
            f_energy,
        ).reshape(relative_energy.shape)
        speed_cdf = cumulative_trapezoid(
            speed_pdf,
            x_speed,
            axis=1,
            initial=0.0,
        )
        # Normalize each radius's speed CDF independently before inverting its random quantile.
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
            # Interpolate inside the bracketing speed-CDF interval, retaining a finite value on flat segments.
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
            # At the extreme outer tail the tabulated f(E) can round to zero.
            # There rho~r^-4 in a Kepler potential, hence
            # f(E)~E^(5/2) and x^2=v^2/v_esc^2 follows
            # Beta(3/2, 7/2).  This affects only a handful of the outermost
            # million markers and avoids extrapolating a numerically zero DF.
            x_sample[~valid] = np.sqrt(
                rng.beta(1.5, 3.5, size=np.count_nonzero(~valid))
            )
            asymptotic_outer_samples += int(np.count_nonzero(~valid))
        speed[begin:end] = np.sqrt(2.0 * psi_batch) * x_sample
    # Uniform cos(theta) gives isotropic directions. Split the speed into radial motion and L=r*v_t.
    mu = 2.0 * rng.random(args.particles) - 1.0
    radial_velocity = speed * mu
    angular_momentum = radius * speed * np.sqrt(
        np.maximum(1.0 - mu * mu, 0.0)
    )
    print(
        f"Eddington table: f_min={diagnostics['f_raw_min']:.6g}, "
        f"negative={diagnostics['negative_f_count']}; "
        f"radial sampling={args.radial_sampling}; "
        f"speed CDF points={args.speed_cdf_points}; "
        f"outer asymptotic samples={asymptotic_outer_samples}"
    )
    particle_mass = float(mass_model[-1]) / args.particles
    write_restart(
        args.output,
        radius,
        radial_velocity,
        angular_momentum,
        particle_mass,
    )


if __name__ == "__main__":
    main()
