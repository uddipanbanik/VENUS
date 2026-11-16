#!/usr/bin/env python3
# Convert spherical SphDSMC restart particles to NSphere text initial data.
# Inputs retain each particle's mass,r,v_r,L and validate the versioned native layout.
# Output rescales mass, velocity and time from M_s=1 to M_total=1 while retaining radii.

"""Export a VENUS spherical-DSMC restart to NSphere's Cartesian IC format.

Restart versions 2/3 store one common marker mass, while current versions
4–7 store a mass in every particle record; versions 6/7 also retain the next
collision timestep. The exporter accepts these layouts
and preserves the represented mass of every marker when changing from the
VENUS ``M_s=1`` normalization to NSphere's ``M_tot=1`` normalization.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import NamedTuple


HEADER = struct.Struct("<16sIQddqdQqq")
LEGACY_PARTICLE = struct.Struct("<dddQ")
CURRENT_PARTICLE = struct.Struct("<dddQd")
REMAINDER_COUNT = struct.Struct("<Q")


class Particle(NamedTuple):
    """One validated spherical marker in restart storage order."""

    radius: float
    radial_velocity: float
    angular_momentum: float
    identifier: int
    mass: float


def read_restart(path: Path) -> tuple[float, list[Particle]]:
    """Decode and validate every supported SphDSMC restart generation."""

    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise RuntimeError("Truncated VENUS spherical-DSMC restart header")
    magic, version, count, common_mass, time, *_ = HEADER.unpack_from(raw)
    if not magic.startswith(b"VENUSSPHDSMC1") or version not in (2, 3, 4, 5, 6, 7):
        raise RuntimeError("Unsupported VENUS spherical-DSMC restart")
    if count < 2:
        raise RuntimeError("Spherical-DSMC restart must contain at least two particles")

    payload_start = HEADER.size
    if version >= 6:
        if len(raw) < payload_start + 16:
            raise RuntimeError("Truncated spherical-DSMC collision clock")
        next_collision_dt = struct.unpack_from("<d", raw, payload_start)[0]
        if not next_collision_dt > 0.0 or math.isnan(next_collision_dt):
            raise RuntimeError("Invalid spherical-DSMC collision clock")
        payload_start += 16  # Limit plus collision-control fingerprint.
    # Versions with individual weights append mass to each record. Earlier records use the common header mass.
    record = CURRENT_PARTICLE if version >= 4 else LEGACY_PARTICLE
    payload_end = payload_start + count * record.size
    if payload_end > len(raw):
        raise RuntimeError("Truncated restart particle payload")
    payload = raw[payload_start:payload_end]
    particles: list[Particle] = []
    for fields in record.iter_unpack(payload):
        radius, radial_velocity, angular_momentum, identifier = fields[:4]
        mass = fields[4] if version >= 4 else common_mass
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not math.isfinite(radial_velocity)
            or not math.isfinite(angular_momentum)
            or angular_momentum < 0.0
            or not math.isfinite(mass)
            or mass <= 0.0
        ):
            raise RuntimeError("Invalid particle record in spherical-DSMC restart")
        particles.append(
            Particle(radius, radial_velocity, angular_momentum, identifier, mass)
        )
    # A complete particle-ID permutation detects missing or duplicated records before exporting coordinates.
    identifiers = {particle.identifier for particle in particles}
    if identifiers != set(range(count)):
        raise RuntimeError("Spherical-DSMC restart particle IDs are not a permutation")

    trailer = raw[payload_end:]
    # Validate the optional paper-exact remainder array even though the Cartesian exporter uses only particle data.
    if version in (3, 5, 7):
        if len(trailer) < REMAINDER_COUNT.size:
            raise RuntimeError("Restart is missing paper-exact remainder state")
        remainder_count = REMAINDER_COUNT.unpack_from(trailer)[0]
        expected = REMAINDER_COUNT.size + remainder_count * 8
        if remainder_count != count or len(trailer) != expected:
            raise RuntimeError("Restart paper-exact remainder payload has the wrong size")
        remainders = struct.unpack_from(f"<{remainder_count}d", trailer, 8)
        if not all(math.isfinite(value) for value in remainders):
            raise RuntimeError("Restart paper-exact remainder payload is non-finite")
    elif trailer:
        raise RuntimeError("Restart has an unexpected trailing payload")
    return time, particles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    time, particles = read_restart(args.restart)
    count = len(particles)

    # VENUS uses G=M_s=r_s=1.  NSphere's Core IC uses G=M_tot=r_s=1,
    # where M_tot/M_s equals the total mass in the VENUS restart.
    total_mass_venus = math.fsum(particle.mass for particle in particles)
    # Changing the mass unit changes v_s by sqrt(M_total). The corresponding time unit changes inversely.
    velocity_scale = math.sqrt(total_mass_venus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", buffering=1024 * 1024) as output:
        output.write(f"{count} {time * velocity_scale:.17g}\n")
        for particle in particles:
            # Convert conserved specific angular momentum back to tangential speed at the particle radius.
            tangential_velocity = particle.angular_momentum / particle.radius
            # A common Cartesian orientation is sufficient: NSphere keeps only
            # r, v_r and |v_t|, then draws its own permanent azimuth per marker.
            output.write(
                f"{particle.mass / total_mass_venus:.17g} "
                f"{particle.radius:.17g} 0 0 "
                f"{particle.radial_velocity / velocity_scale:.17g} "
                f"{tangential_velocity / velocity_scale:.17g} 0 0 0\n"
            )


if __name__ == "__main__":
    main()
