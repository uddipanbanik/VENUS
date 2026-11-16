#!/usr/bin/env python3
# Evaluate two collision-density estimators on one unchanged equal-mass SphDSMC state.
# Input is a versioned native restart and physical cross section per mass in its code units.
# Output compares shell density, width/mean-free-path and collision-time estimates without evolving particles.

"""Compare the Frank--Tokayer shell-density formula with their supplied code.

The paper's equation (6) uses the mass and volume of each collision shell.
The supplied Fortran instead calls DENSPROF/SG_SMOOTH once per global step
and interpolates that smoothed profile at the median radius of every shell.
This diagnostic reproduces both choices on one VENUS restart without changing
the particle state.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import struct
from pathlib import Path
from typing import Callable


HEADER = struct.Struct("<16sIQddqdQqq")
LEGACY_PARTICLE = struct.Struct("<dddQ")
CURRENT_PARTICLE = struct.Struct("<dddQd")
FOUR_PI_OVER_THREE = 4.0 * math.pi / 3.0


# ---- Decode and radius-sort one supported equal-mass Frank checkpoint -------
def read_restart(path: Path) -> tuple[dict[str, float | int], list[tuple[float, float, float]]]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise RuntimeError("Truncated spherical DSMC restart header")
    values = HEADER.unpack_from(raw)
    magic, version, count, particle_mass, time, step = values[:6]
    if (
        not magic.startswith(b"VENUSSPHDSMC1")
        or version not in (2, 3, 4, 5, 6, 7)
        or count < 25
    ):
        raise RuntimeError("Unsupported spherical DSMC restart header")
    payload_start = HEADER.size
    if version >= 6:
        if len(raw) < payload_start + 16:
            raise RuntimeError("Truncated spherical DSMC collision clock")
        next_collision_dt = struct.unpack_from("<d", raw, payload_start)[0]
        if not next_collision_dt > 0.0 or math.isnan(next_collision_dt):
            raise RuntimeError("Invalid spherical DSMC collision clock")
        payload_start += 16  # Limit plus collision-control fingerprint.
    # Decode the version's actual record layout before checking whether all masses satisfy the common-mass estimator.
    record = CURRENT_PARTICLE if version >= 4 else LEGACY_PARTICLE
    payload_size = count * record.size
    payload = raw[payload_start : payload_start + payload_size]
    if len(payload) != payload_size:
        raise RuntimeError("Truncated spherical DSMC particle payload")
    trailer = raw[payload_start + payload_size :]
    if version in (3, 5, 7):
        if len(trailer) < 8:
            raise RuntimeError("Spherical DSMC restart is missing remainder state")
        remainder_count = struct.unpack_from("<Q", trailer)[0]
        expected = 8 + 8 * remainder_count
        if remainder_count != count or len(trailer) != expected:
            raise RuntimeError("Spherical DSMC remainder payload has the wrong size")
        remainders = struct.unpack_from(f"<{remainder_count}d", trailer, 8)
        if not all(math.isfinite(value) for value in remainders):
            raise RuntimeError("Spherical DSMC remainder payload is non-finite")
    elif trailer:
        raise RuntimeError("Spherical DSMC restart has an unexpected trailer")

    decoded = []
    identifiers = set()
    for fields in record.iter_unpack(payload):
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
            or identifier in identifiers
            or identifier >= count
        ):
            raise RuntimeError("Invalid spherical DSMC particle record")
        identifiers.add(identifier)
        decoded.append((radius, radial_velocity, angular_momentum, mass))
    if identifiers != set(range(count)):
        raise RuntimeError("Spherical DSMC particle IDs are not a permutation")

    # NSphere's DENSPROF bins number counts times one common mass.  Applying
    # that algorithm to unequal-mass markers would be a different estimator,
    # so reject such input rather than returning a physically misleading audit.
    particle_mass = decoded[0][3]
    if any(
        not math.isclose(fields[3], particle_mass, rel_tol=1.0e-13, abs_tol=0.0)
        for fields in decoded[1:]
    ):
        raise RuntimeError(
            "Frank collision-density audit requires an equal-mass restart"
        )
    particles = [(r, vr, angular_momentum) for r, vr, angular_momentum, _ in decoded]
    if any(particles[i][0] > particles[i + 1][0] for i in range(len(particles) - 1)):
        particles.sort(key=lambda value: value[0])
    return (
        {
            "version": version,
            "count": count,
            "particle_mass": particle_mass,
            "time": time,
            "step": step,
        },
        particles,
    )


# ---- Reproduce Frank's smoothed density estimator exactly -------------------
# DENSPROF forms adaptive radial bins; SG_SMOOTH fits a local quadratic in
# log(r).  The no-pivot solve is retained because this script is an audit of
# the supplied algorithm, not a replacement density estimator.
def solve_three_by_three_no_pivot(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Match SG_SMOOTH's no-pivot Gaussian elimination."""
    a = [row[:] for row in matrix]
    b = rhs[:]
    # Reproduce the supplied no-pivot elimination order for the local quadratic smoothing normal equations.
    for k in range(3):
        pivot = a[k][k]
        if abs(pivot) < 1.0e-30:
            return None
        for j in range(k + 1, 3):
            factor = a[j][k] / pivot
            a[j][k] = 0.0
            for column in range(k + 1, 3):
                a[j][column] -= factor * a[k][column]
            b[j] -= factor * b[k]
    c2 = b[2] / a[2][2]
    c1 = (b[1] - a[1][2] * c2) / a[1][1]
    c0 = (b[0] - a[0][1] * c1 - a[0][2] * c2) / a[0][0]
    return [c0, c1, c2]


def frank_density_profile(
    particles: list[tuple[float, float, float]], particle_mass: float, window: int = 13
) -> tuple[list[float], list[float], list[float]]:
    """Reproduce NSphere.f DENSPROF and the density part of SG_SMOOTH."""
    radii = [particle[0] for particle in particles]
    count = len(radii)
    index = 0
    cumulative_count = 0
    profile_count = 0
    bin_mass = 0.0
    bin_radii: list[float] = []
    inner = radii[-1]
    outer = 0.0
    log_radius: list[float] = []
    log_density: list[float] = []

    while index < count:
        profile_count += 1
        particles_in_current_piece = 0
        completed = False
        while index < count:
            radius = radii[index]
            inner = min(inner, radius)
            outer = max(outer, radius)
            particles_in_current_piece += 1
            cumulative_count += 1
            bin_mass += particle_mass
            bin_radii.append(radius)
            index += 1
            log_width = math.log10(outer) - math.log10(inner)
            # Adaptive density bins follow the supplied particle-count and radial-width rules.
            completed = (
                cumulative_count <= 1000 and particles_in_current_piece == 25
            ) or (
                cumulative_count > 1000
                and (
                    particles_in_current_piece >= 2500
                    or (log_width > 0.1 and particles_in_current_piece >= 25)
                )
            )
            if completed:
                mean_radius = sum(bin_radii) / len(bin_radii)
                density = 3.0 * bin_mass / (
                    4.0 * math.pi * max(outer**3 - inner**3, 1.0e-300)
                )
                log_radius.append(math.log10(mean_radius))
                log_density.append(math.log10(density))
                if cumulative_count < 1000:
                    outer = 0.0
                else:
                    inner = outer
                    outer = 0.0
                    bin_mass = 0.0
                    bin_radii = []
                break
        if not completed:
            # The Fortran enters one final incomplete bin, then sets
            # NDTOT=IBIN-1 and discards it.
            profile_count -= 1
            break

    half_window = window // 2
    smoothed: list[float] = []
    for i, centre in enumerate(log_radius):
        first = max(0, i - half_window)
        last = min(len(log_radius), i + half_window + 1)
        matrix = [[0.0] * 3 for _ in range(3)]
        rhs = [0.0] * 3
        for j in range(first, last):
            dx = log_radius[j] - centre
            powers = [1.0, dx, dx * dx, dx * dx * dx, dx**4]
            # Accumulate polynomial normal equations in centered log radius before evaluating the fitted density at its center.
            matrix[0][0] += powers[0]
            matrix[0][1] += powers[1]
            matrix[0][2] += powers[2]
            matrix[1][0] += powers[1]
            matrix[1][1] += powers[2]
            matrix[1][2] += powers[3]
            matrix[2][0] += powers[2]
            matrix[2][1] += powers[3]
            matrix[2][2] += powers[4]
            rhs[0] += log_density[j]
            rhs[1] += log_density[j] * dx
            rhs[2] += log_density[j] * dx * dx
        coefficients = solve_three_by_three_no_pivot(matrix, rhs)
        smoothed.append(log_density[i] if coefficients is None else coefficients[0])
    return log_radius, log_density, smoothed


# ---- Interpolate the smoothed profile and define the paper shell density ----
def interpolator(x_values: list[float], y_values: list[float]) -> Callable[[float], float]:
    def evaluate(radius: float) -> float:
        x = math.log10(radius)
        if x <= x_values[0]:
            return 10.0 ** y_values[0]
        if x >= x_values[-1]:
            return 10.0 ** y_values[-1]
        right = bisect.bisect_right(x_values, x)
        left = right - 1
        fraction = (x - x_values[left]) / (x_values[right] - x_values[left])
        return 10.0 ** (y_values[left] + fraction * (y_values[right] - y_values[left]))

    return evaluate


def raw_shell_density(
    radii: list[float], particle_mass: float, begin: int, end: int
) -> float:
    return (end - begin) * particle_mass / (
        FOUR_PI_OVER_THREE
        * max(radii[end - 1] ** 3 - radii[begin] ** 3, 1.0e-300)
    )


# ---- Build collision shells and evaluate locality and collision clocks ------
# For each estimator, Delta r/lambda_mfp = Delta r rho (sigma/m).  A value
# above unity means that one Monte Carlo neighborhood spans more than a mean
# free path and therefore no longer represents a local Boltzmann collision.
def build_shells(
    particles: list[tuple[float, float, float]],
    particle_mass: float,
    sigma: float,
    density_at_shell: Callable[[int, int], float],
    target: int,
    minimum: int,
    skip_short_tail: bool,
) -> list[dict[str, float | int]]:
    radii = [particle[0] for particle in particles]
    speeds = [math.hypot(vr, angular_momentum / radius) for radius, vr, angular_momentum in particles]
    shells: list[dict[str, float | int]] = []
    begin = 0
    while begin < len(particles):
        end = min(len(particles), begin + target)
        if skip_short_tail and end - begin < minimum:
            break
        while True:
            density = density_at_shell(begin, end)
            width = radii[end - 1] - radii[begin]
            # For cross section per mass, the local mean-free-path inverse is rho*(sigma/m).
            width_over_mfp = width * density * sigma
            if width_over_mfp < 1.0 or end - begin < 2 * minimum:
                break
            # Halve the candidate neighborhood when its physical width exceeds a mean free path and enough particles remain.
            end = begin + (end - begin) // 2
        maximum_speed = max(speeds[begin:end])
        raw_density = raw_shell_density(radii, particle_mass, begin, end)
        shells.append(
            {
                "begin": begin,
                "end": end,
                "particles": end - begin,
                "inner": radii[begin],
                "outer": radii[end - 1],
                "density": density,
                "raw_density_same_range": raw_density,
                "smooth_over_raw": density / raw_density,
                "width_over_mfp": width_over_mfp,
                "maximum_speed": maximum_speed,
                "trial_rate_majorant": 0.5
                * (end - begin - 1)
                * density
                * sigma
                * (4.0 * maximum_speed),
                "frank_collision_dt": 1.0
                / max(density * sigma * 4.0 * maximum_speed, 1.0e-300),
                "paper_collision_dt": 1.0
                / max(density * sigma * maximum_speed, 1.0e-300),
            }
        )
        begin = end
    return shells


# ---- Reduce shell-by-shell discrepancies to presentation diagnostics --------
def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def summarize(shells: list[dict[str, float | int]]) -> dict[str, object]:
    ratios = [float(shell["smooth_over_raw"]) for shell in shells]
    unresolved = [shell for shell in shells if float(shell["width_over_mfp"]) > 1.0]
    return {
        "shell_count": len(shells),
        "covered_particles": sum(int(shell["particles"]) for shell in shells),
        "minimum_shell_particles": min(int(shell["particles"]) for shell in shells),
        "maximum_width_over_mfp": max(float(shell["width_over_mfp"]) for shell in shells),
        "unresolved_shells": len(unresolved),
        "r_fail": max((float(shell["outer"]) for shell in unresolved), default=0.0),
        "minimum_collision_dt_frank_code": min(float(shell["frank_collision_dt"]) for shell in shells),
        "minimum_collision_dt_paper_eq12": min(float(shell["paper_collision_dt"]) for shell in shells),
        "total_majorant_trial_rate": sum(float(shell["trial_rate_majorant"]) for shell in shells),
        "density_estimator_over_raw": {
            "minimum": min(ratios),
            "p10": percentile(ratios, 0.10),
            "median": percentile(ratios, 0.50),
            "p90": percentile(ratios, 0.90),
            "maximum": max(ratios),
        },
        "first_eight_shells": shells[:8],
    }


# ---- Run both estimators on the same immutable particle state ----------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--target-particles", type=int, default=4096)
    parser.add_argument("--minimum-particles", type=int, default=32)
    args = parser.parse_args()
    if not math.isfinite(args.sigma) or args.sigma <= 0.0:
        raise ValueError("--sigma must be finite and positive")
    if args.minimum_particles < 2:
        raise ValueError("--minimum-particles must be at least two")
    if args.target_particles < args.minimum_particles:
        raise ValueError(
            "--target-particles must be at least --minimum-particles"
        )

    header, particles = read_restart(args.restart)
    particle_mass = float(header["particle_mass"])
    radii = [particle[0] for particle in particles]
    profile_radius, profile_density, profile_smoothed = frank_density_profile(
        particles, particle_mass
    )
    smoothed_density = interpolator(profile_radius, profile_smoothed)

    def paper_density(begin: int, end: int) -> float:
        return raw_shell_density(radii, particle_mass, begin, end)

    def frank_code_density(begin: int, end: int) -> float:
        return smoothed_density(radii[begin + (end - begin) // 2])

    # Evaluate each density prescription on the same ordered particle state with its specified shell/tail rules.
    paper_shells = build_shells(
        particles,
        particle_mass,
        args.sigma,
        paper_density,
        args.target_particles,
        args.minimum_particles,
        False,
    )
    frank_shells = build_shells(
        particles,
        particle_mass,
        args.sigma,
        frank_code_density,
        args.target_particles,
        args.minimum_particles,
        True,
    )
    result = {
        "restart": str(args.restart),
        "time": header["time"],
        "step": header["step"],
        "particle_count": header["count"],
        "particle_mass": particle_mass,
        "sigma": args.sigma,
        "frank_density_profile_points": len(profile_radius),
        "frank_density_profile_log10_radius_range": [profile_radius[0], profile_radius[-1]],
        "frank_density_profile_unsmoothed_log10_density_range": [
            min(profile_density),
            max(profile_density),
        ],
        "paper_eq6_venus": summarize(paper_shells),
        "supplied_frank_code": summarize(frank_shells),
        "frank_over_paper_total_majorant_trial_rate": sum(
            float(shell["trial_rate_majorant"]) for shell in frank_shells
        )
        / sum(float(shell["trial_rate_majorant"]) for shell in paper_shells),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
