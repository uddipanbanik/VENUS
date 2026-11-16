#!/usr/bin/env python3
# Measure collision-shell locality and core kinematics from a complete SphDSMC restart.
# Decode individual weights and persisted clocks according to the native version.
# Output JSON reports actual weighted central moments, half-density core radius and unresolved shell ranges.

"""Measure mean-free-path shell resolution in a spherical DSMC restart.

Both legacy common-mass restarts (versions 2/3) and current per-marker-mass
restarts (versions 4–7) are decoded, including persisted collision clocks in
versions 6/7. Density and velocity diagnostics use
the actual stored marker masses, so unequal-mass checkpoints are not silently
treated as equal-mass states.
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import List, Tuple


HEADER = struct.Struct("<16sIQddqdQqq")
LEGACY_PARTICLE = struct.Struct("<dddQ")
CURRENT_PARTICLE = struct.Struct("<dddQd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--target-particles", type=int, default=4096)
    parser.add_argument("--minimum-particles", type=int, default=32)
    args = parser.parse_args()
    if not math.isfinite(args.sigma) or args.sigma < 0.0:
        raise ValueError("--sigma must be finite and non-negative")
    if args.minimum_particles < 2:
        raise ValueError("--minimum-particles must be at least two")
    if args.target_particles < args.minimum_particles:
        raise ValueError(
            "--target-particles must be at least --minimum-particles"
        )

    raw = args.restart.read_bytes()
    if len(raw) < HEADER.size:
        raise RuntimeError("Truncated spherical DSMC restart header.")
    header = HEADER.unpack_from(raw)
    magic, version, count, particle_mass, time, step = header[:6]
    if (
        not magic.startswith(b"VENUSSPHDSMC1")
        or version not in (2, 3, 4, 5, 6, 7)
        or count < 2
    ):
        raise RuntimeError("Unsupported spherical DSMC restart header.")
    payload_start = HEADER.size
    if version >= 6:
        if len(raw) < payload_start + 16:
            raise RuntimeError("Truncated spherical DSMC collision clock.")
        next_collision_dt = struct.unpack_from("<d", raw, payload_start)[0]
        if not next_collision_dt > 0.0 or math.isnan(next_collision_dt):
            raise RuntimeError("Invalid spherical DSMC collision clock.")
        payload_start += 16  # Limit plus collision-control fingerprint.
    # Select per-particle masses for current files and the common header mass for legacy files.
    record = CURRENT_PARTICLE if version >= 4 else LEGACY_PARTICLE
    particle_bytes = count * record.size
    records = raw[payload_start : payload_start + particle_bytes]
    if len(records) != particle_bytes:
        raise RuntimeError("Restart particle payload has the wrong size.")
    trailer = raw[payload_start + particle_bytes :]
    if version in (2, 4, 6) and trailer:
        raise RuntimeError(f"Version-{version} restart has an unexpected trailer.")
    if version in (3, 5, 7):
        if len(trailer) < 8:
            raise RuntimeError(
                f"Version-{version} restart is missing remainder state."
            )
        remainder_count = struct.unpack_from("<Q", trailer)[0]
        expected = 8 + 8 * remainder_count
        if remainder_count != count or len(trailer) != expected:
            raise RuntimeError(
                f"Version-{version} remainder payload has the wrong size."
            )
        remainders = struct.unpack_from(f"<{remainder_count}d", trailer, 8)
        if not all(math.isfinite(value) for value in remainders):
            raise RuntimeError(
                f"Version-{version} remainder payload is non-finite."
            )

    particles = []
    for fields in record.iter_unpack(records):
        radius, radial_velocity, angular_momentum, identifier = fields[:4]
        mass = fields[4] if version >= 4 else particle_mass
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not math.isfinite(radial_velocity)
            or not math.isfinite(angular_momentum)
            or angular_momentum < 0.0
            or not math.isfinite(mass)
            or mass <= 0.0
        ):
            raise RuntimeError("Restart contains an invalid particle record.")
        particles.append((radius, radial_velocity, angular_momentum, identifier, mass))
    if {particle[3] for particle in particles} != set(range(count)):
        raise RuntimeError("Restart particle IDs are not a permutation.")
    particles.sort(key=lambda particle: particle[0])
    radii = [particle[0] for particle in particles]
    masses = [particle[4] for particle in particles]

    central_kinematic_count = min(250, count)
    central_velocities = [
        particle[1] for particle in particles[:central_kinematic_count]
    ]
    central_masses = masses[:central_kinematic_count]
    # Use actual inner particle weights for the radial mean and variance.
    central_kinematic_mass = math.fsum(central_masses)
    central_radial_velocity_mean = (
        math.fsum(
            mass * velocity
            for mass, velocity in zip(central_masses, central_velocities)
        )
        / central_kinematic_mass
        if central_kinematic_mass > 0.0
        else math.nan
    )
    central_radial_velocity_variance = (
        max(
            0.0,
            math.fsum(
                mass * velocity * velocity
                for mass, velocity in zip(central_masses, central_velocities)
            )
            / central_kinematic_mass
            - central_radial_velocity_mean**2,
        )
        if central_kinematic_mass > 0.0
        else math.nan
    )

    # Frank & Tokayer define rho_0 as the mean density of the sphere
    # enclosing the innermost 250 particles.  Estimate the local profile for
    # rho(r_c)=rho_0/2 with adjacent 256-particle finite-volume cells.  Cell
    # faces lie midway between neighboring particle radii, so every cell uses
    # its exact integrated particle mass and no fitted profile.
    central_count = min(250, count)
    central_radius = radii[central_count - 1]
    central_density = math.fsum(masses[:central_count]) / (
        (4.0 * math.pi / 3.0) * central_radius**3
    )
    half_central_density = 0.5 * central_density
    core_profile_particles = min(256, count)
    core_profile = []
    begin = 0
    inner_face = 0.0
    while begin < count:
        end = min(count, begin + core_profile_particles)
        if end < count:
            # The descriptive core profile uses midpoint shell faces, distinct from the collision-shell extrema below.
            outer_face = 0.5 * (radii[end - 1] + radii[end])
        else:
            outer_face = radii[-1]
        volume = (4.0 * math.pi / 3.0) * max(
            outer_face**3 - inner_face**3,
            1.0e-300,
        )
        density = math.fsum(masses[begin:end]) / volume
        radius = (0.5 * (inner_face**3 + outer_face**3)) ** (1.0 / 3.0)
        core_profile.append((radius, density))
        inner_face = outer_face
        begin = end

    core_radius = math.nan
    for index in range(1, len(core_profile)):
        r0, rho0 = core_profile[index - 1]
        r1, rho1 = core_profile[index]
        if rho0 > half_central_density >= rho1:
            if rho0 > 0.0 and rho1 > 0.0 and rho0 != rho1:
                # Locate the half-central-density crossing by interpolation in log density and log radius.
                fraction = (
                    math.log(half_central_density) - math.log(rho0)
                ) / (math.log(rho1) - math.log(rho0))
                core_radius = math.exp(
                    math.log(r0) + fraction * (math.log(r1) - math.log(r0))
                )
            else:
                core_radius = r1
            break

    shells = []  # type: List[Tuple[int, int, float, float, float]]

    def describe(begin: int, end: int) -> Tuple[int, int, float, float, float]:
        # Frank & Tokayer equation (6): use the extrema of the particles in
        # the shell, not midpoint control-volume faces.
        inner = radii[begin]
        outer = radii[end - 1]
        volume = (4.0 * math.pi / 3.0) * max(outer**3 - inner**3, 1.0e-300)
        density = math.fsum(masses[begin:end]) / volume
        # Zero cross section gives an infinite mean free path. Positive cross section uses the local shell mass density.
        mean_free_path = (
            1.0 / (density * args.sigma)
            if density > 0.0 and args.sigma > 0.0
            else math.inf
        )
        ratio = (outer - inner) / mean_free_path
        return begin, end, inner, outer, ratio

    begin = 0
    while begin < count:
        end = min(count, begin + args.target_particles)
        shell = describe(begin, end)
        while (
            shell[4] > 1.0
            and end - begin >= 2 * args.minimum_particles
        ):
            end = begin + (end - begin) // 2
            shell = describe(begin, end)
        shells.append(shell)
        # Frank resets the next candidate to 4096 particles at the newly
        # accepted cursor.  There is no recursive traversal of the unused
        # half of a fixed parent shell.
        begin = end

    # Collect all neighborhoods that remain wider than a mean free path after the finite splitting rule.
    unresolved = [shell for shell in shells if shell[4] > 1.0]
    r_fail = max((shell[3] for shell in unresolved), default=0.0)
    result = {
        "restart": str(args.restart),
        "time": time,
        "step": step,
        "particle_count": count,
        "particle_mass": particle_mass,
        "total_particle_mass": math.fsum(masses),
        "minimum_particle_mass": min(masses),
        "maximum_particle_mass": max(masses),
        "shell_count": len(shells),
        "minimum_shell_particles": min(end - begin for begin, end, *_ in shells),
        "maximum_shell_width_over_mean_free_path": max(
            shell[4] for shell in shells
        ),
        "nonlocal_shells": len(unresolved),
        "central_density_250": central_density,
        "central_radial_velocity_mean_250": central_radial_velocity_mean,
        "central_radial_velocity_variance_250": central_radial_velocity_variance,
        "central_radial_velocity_dispersion_250": math.sqrt(
            central_radial_velocity_variance
        ),
        "half_central_density": half_central_density,
        "core_radius_half_central_density": core_radius,
        "core_radius_profile_particles": core_profile_particles,
        "r_fail": r_fail,
        "r_fail_over_core_radius": (
            r_fail / core_radius
            if math.isfinite(core_radius) and core_radius > 0.0
            else math.nan
        ),
        "nonlocal_shells_detail": [
            {
                "begin": begin,
                "end": end,
                "particles": end - begin,
                "inner_edge": inner,
                "outer_edge": outer,
                "width_over_mean_free_path": ratio,
            }
            for begin, end, inner, outer, ratio in unresolved
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
