#!/usr/bin/env python3
# Generate particle-only WPIC-v6 Gaussian shell inputs for static and expanding VP.
# Use G=1 code units. Both files share physical r,u,eta,mass and IDs; stored radial velocity is u-Hr.
# The manifest records the specified finite radial/velocity support and initialization diagnostics.

"""Generate the bound Gaussian shell used by the collisionless paper examples.

The physical density is Gaussian in radius, centered at 10 with width 1 and
truncated to 7 < r < 13. Sampling its mass distribution includes the spherical
Jacobian r**2. At fixed radius, physical radial velocity is Gaussian and the
transverse speed is Rayleigh distributed, as required by two Gaussian
transverse components. Both speeds are truncated at three Gaussian scales.
Opposite radial samples give zero mean physical radial velocity.

The static and expanding files contain the same physical positions, velocities,
specific angular momenta, masses and IDs. Expansion stores the peculiar radial
velocity w = u - H*r, with H = 1/18 initially. Each output is a particle-only
WPIC-v6 initialization. The solver constructs its initial gravitational field
and event clocks when it imports that file.

Example::

    python3 generate_bound_gaussian_shell_restart.py \
        --output-directory gaussian_shell --particles 1048576

NumPy supplies quadrature, inverse-CDF sampling and packed binary records.
The output directory must be new. The default count and distributions specify
the paper initial states. Smaller even counts are sampling studies.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import struct

import numpy as np


# WPIC-v6 uses little-endian, packed 81-byte particle records. The internal
# energy and dispersion are spectator fields for the collisionless solver.
MARKER_DTYPE = np.dtype([
    ("mass", "<f8"), ("r", "<f8"), ("w", "<f8"), ("eta", "<f8"),
    ("internal_energy", "<f8"), ("sigma", "<f8"), ("id", "<u8"),
    ("source_cell", "<u8"), ("state", "u1"), ("inner_speed", "<f8"),
    ("remaining_inner_time", "<f8"),
])
INITIAL_SEED = 0x6A09E667F3BCC909
COLLISION_SEED = 0x73756268616C6F31


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    """Hash an output without loading the whole particle file into memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def radical_inverse(count, base, shift):
    """Build a shifted low-discrepancy coordinate strictly inside (0, 1)."""
    integer = np.arange(1, count + 1, dtype=np.uint64)
    q = np.zeros(count)
    factor = 1. / base
    while np.any(integer):
        # Accumulate successive base digits to spread each sampling coordinate through the unit interval.
        q += (integer % base) * factor
        integer //= base
        factor /= base
    return np.clip((q + shift) % 1., np.finfo(float).eps,
                   1. - np.finfo(float).eps)


def generate(output, count):
    """Sample one physical shell and write its two background representations."""
    require(count >= 1024 and count % 2 == 0,
            "Use an even particle count of at least 1024.")
    require(MARKER_DTYPE.itemsize == 81, "Packed WPIC record size differs.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    total_mass = 3.5968447592701334
    radius, width = 10., 1.
    left, right = 7., 13.
    escape_speed = math.sqrt(2 * total_mass / radius)
    mean_velocity = 0.
    dispersion = .1 * escape_speed
    radial_faces = np.exp(np.linspace(math.log(.01), math.log(10000.), 513))
    radial_faces[0], radial_faces[-1] = .01, 10000.

    # Integrate r**2*rho(r), then invert the normalized mass CDF. A second
    # quadrature with twice the interval width checks the normalization.
    x, weights = np.polynomial.legendre.leggauss(16)
    faces = np.linspace(left, right, 65537)
    half = .5 * np.diff(faces)
    center = .5 * (faces[1:] + faces[:-1])
    node = center[:, None] + half[:, None] * x
    # Integrate r^2*rho on fine radial intervals. The factor 4*pi enters the density normalization later.
    increment = half * np.sum(
        weights * node**2 * np.exp(-.5 * ((node - radius) / width)**2), axis=1)
    cdf = np.r_[0., np.cumsum(increment)]
    integral = float(cdf[-1])
    cdf /= cdf[-1]
    coarse_faces = faces[::2]
    coarse_half = .5 * np.diff(coarse_faces)
    coarse_center = .5 * (coarse_faces[1:] + coarse_faces[:-1])
    coarse_node = coarse_center[:, None] + coarse_half[:, None] * x
    coarse = float(np.sum(coarse_half * np.sum(
        weights * coarse_node**2 *
        np.exp(-.5 * ((coarse_node - radius) / width)**2), axis=1)))
    require(abs(coarse / integral - 1) < 2e-13,
            "Radial mass quadrature has not converged.")

    # One radius serves an opposite radial-velocity pair. Independent bases
    # sample radius, radial velocity and transverse speed without a random
    # generator or a change to the solver's stored random counters.
    pairs = count // 2
    # Invert the normalized enclosed-mass CDF, so equal-weight markers represent spherical mass.
    r = np.interp(radical_inverse(pairs, 2, .3819660112501051), cdf, faces)
    zgrid = np.linspace(-3., 3., 131073)
    zpdf = np.exp(-.5 * zgrid**2)
    zcdf = np.r_[0., np.cumsum(.5 * (zpdf[:-1] + zpdf[1:]) * np.diff(zgrid))]
    zcdf /= zcdf[-1]
    z = np.interp(radical_inverse(pairs, 3, .4142135623730951), zcdf, zgrid)
    qt = radical_inverse(pairs, 5, .7320508075688773)
    # Invert the truncated Rayleigh CDF for the magnitude of two independent Gaussian transverse components.
    vt2 = -2 * dispersion**2 * np.log1p(-qt * (-math.expm1(-4.5)))
    # Pair opposite radial velocities at a common radius while preserving the same transverse speed.
    sampled = np.column_stack((mean_velocity + dispersion * z,
                               mean_velocity - dispersion * z)).ravel()
    r = np.repeat(r, 2)
    vt2 = np.repeat(vt2, 2)

    # Form the expanding peculiar velocity first. Recovering its physical
    # velocity defines the identical binary64 physical velocity in both files.
    H = 1. / 18.
    peculiar = sampled - H * r
    physical = peculiar + H * r
    marker = np.zeros(count, dtype=MARKER_DTYPE)
    marker["mass"] = total_mass / count
    marker["r"] = r
    marker["w"] = physical
    # Store eta=L^2/2=r^2*v_t^2/2, the centrifugal invariant used by the Newtonian transport.
    marker["eta"] = .5 * r * r * vt2
    marker["sigma"] = .1
    marker["internal_energy"] = 1.5 * marker["mass"] * .1**2
    marker["id"] = np.arange(count, dtype=np.uint64)
    marker["source_cell"] = np.searchsorted(radial_faces, r, side="right") - 1
    require(np.all((r > left) & (r < right)) and
            abs(float(np.mean(physical))) < 1e-16,
            "Initial radius support or mean physical velocity differs.")

    # Check binding using the actual spherical marker weights. At a marker,
    # inner mass contributes -M_inner/r and each outer shell contributes -m/r_j.
    # Its own contribution is excluded from this initialization diagnostic.
    # Order complete particle records by radius before accumulating their spherical self-potential.
    order = np.argsort(r, kind="stable")
    rs = r[order]
    mass = marker["mass"][order]
    enclosed = np.cumsum(mass) - mass
    outer = np.cumsum((mass / rs)[::-1])[::-1] - mass / rs
    phi = -(enclosed / rs + outer)
    kinetic = .5 * float(np.sum(mass * (physical[order]**2 + vt2[order])))
    # The factor one half counts each gravitational pair once in the total potential energy.
    potential = .5 * float(np.sum(mass * phi))
    energy = .5 * (physical[order]**2 + vt2[order]) + phi
    require(np.max(energy) < 0 and kinetic > 0 and potential < 0,
            "The initial shell must be self-bound.")

    # The particle-only restart has fresh event ages and infinite initial
    # timescale estimates. Initial field construction supplies those estimates.
    header = struct.pack("<16sIIQdqQQQQdddd", b"VENUSWVPIC1", 6, 0x01020304,
                         count, 0., 0, INITIAL_SEED, count, COLLISION_SEED, 0,
                         0., 0., math.inf, math.inf)
    require(len(header) == 112, "WPIC-v6 header size differs.")
    reports = []
    # Write the same physical realization in each background convention, with its own explicit sidecars.
    for background in ["static", "expanding"]:
        marker["w"] = physical if background == "static" else peculiar
        directory = output / (background + "_vp")
        directory.mkdir()
        path = directory / "restart_state.bin"
        with path.open("xb") as stream:
            stream.write(header)
            stream.write(marker.tobytes())
        (directory / "radial_mesh_contract.txt").write_text(
            "VENUS_WPIC_RADIAL_MESH 1\nradial_cells 512\n"
            "radial_minimum 0.01\nradial_maximum 10000\n")
        (directory / "background_contract.txt").write_text(
            "VENUS_WPIC_BACKGROUND 1\n" +
            ("static 0 12\n" if background == "static" else
             "expanding 0.66666666666666663 12\n"))
        (directory / "initial_condition_contract.txt").write_text(
            "VENUS_WPIC_INITIAL_CONDITION 1\n0.5 1\n")
        reports.append(dict(case=background + "_vp", file=str(path),
                            sha256=sha256(path), bytes=path.stat().st_size))

    report = dict(
        density="rho=A*exp[-(r-10)^2/2] for 7<r<13, zero elsewhere",
        velocity="exp[-(u^2+v_t^2)/(2*sigma^2)], |u|<3sigma, 0<=v_t<3sigma",
        density_normalization=total_mass / (4 * math.pi * integral),
        mass=total_mass, N=count, marker_mass=total_mass / count,
        mean_physical_radial_velocity=mean_velocity,
        velocity_Gaussian_scale=dispersion, reference_escape_speed=escape_speed,
        physical_speed_max=float(np.max(np.sqrt(physical**2 + vt2))),
        radial_velocity_range=[float(np.min(physical)), float(np.max(physical))],
        K_physical=kinetic, U_self=potential, Q_self=2 * kinetic / abs(potential),
        maximum_particle_self_energy=float(np.max(energy)), self_bound_fraction=1.,
        radial_mass_quadrature_relative_change=coarse / integral - 1.,
        pairing="Identical physical positions, velocities, eta, masses and IDs. Stored w differs by H*r.",
        background="Static total density; expanding positive density contrast with alpha=2/3 and time offset=12.",
        radial_grid=dict(cells=512, minimum=.01, maximum=10000.),
        cases=reports, numpy_version=np.__version__,
        generator_sha256=sha256(Path(__file__)))
    (output / "initial_state_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=1048576)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_directory, args.particles), indent=2))


if __name__ == "__main__":
    main()
