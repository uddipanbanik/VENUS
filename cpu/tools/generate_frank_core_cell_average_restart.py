#!/usr/bin/env python3
# Construct the cored benchmark distribution as phase-space cell averages and an optional stationary inner carrier.
# Canonical eta=L squared/2 gives the spherical measure 8*pi squared*dr*dw*deta in G=1 code units.
# Output EVLOGRST1 version 9 includes grid controls, distribution cells and each delayed-return packet; JSON reports support, mass and inner optical depth.

"""Generate Frank's Core-halo equilibrium on the VENUS cell-average grid."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from sidm_nfw_eddington import (  # noqa: E402
    cumulative_trapezoid,
    eddington_table,
    frank_core_density,
    wall_relative_potential,
)


PI = np.pi
FRANK_RHO_S = 1.0 / (4.0 * PI)
FRANK_TOTAL_MASS = 1.1252581096075398
FRANK_SIGMA0 = 32.354198286197665

# A carrier stores reduced mass, eta-cell index, specific eta, inward
# boundary speed and the remaining time before return to the resolved grid.
CarrierPacket = Tuple[float, int, float, float, float]


def cell_edges(lower: float, upper: float, cells: int, logarithmic: bool) -> np.ndarray:
    if logarithmic:
        edge = np.geomspace(lower, upper, cells + 1)
    else:
        edge = np.linspace(lower, upper, cells + 1)
    edge[0] = lower
    edge[-1] = upper
    return edge


def eta_edges(L_max: float, cells: int, power: float) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, cells + 1)
    return 0.5 * L_max * L_max * fraction**power


def composite_eta_edges(
    L_max: float,
    cells: int,
    power: float,
    eta_join: float,
    inner_cells: int,
) -> np.ndarray:
    eta_max = 0.5 * L_max * L_max
    if not (0 < inner_cells < cells and 0.0 < eta_join < eta_max):
        raise ValueError(
            "Composite eta grid requires 0 < inner_cells < NL and "
            "0 < eta_join < L_max^2/2."
        )
    edge = np.empty(cells + 1, dtype=float)
    inner_fraction = np.linspace(0.0, 1.0, inner_cells + 1)
    edge[: inner_cells + 1] = eta_join * inner_fraction**power
    outer_fraction = np.linspace(0.0, 1.0, cells - inner_cells + 1)
    edge[inner_cells:] = eta_join * (eta_max / eta_join) ** outer_fraction
    edge[0] = 0.0
    edge[inner_cells] = eta_join
    edge[-1] = eta_max
    return edge


def stretched_w_edges(w_max: float, cells: int, stretch: float) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, cells + 1)
    if stretch == 0.0:
        return w_max * coordinate
    edge = w_max * np.sinh(stretch * coordinate) / np.sinh(stretch)
    edge[0] = -w_max
    edge[-1] = w_max
    return edge


def map_rule_to_cells(
    edges: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint = 0.5 * (edges[:-1] + edges[1:])
    half_width = 0.5 * (edges[1:] - edges[:-1])
    points = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    # Divide mapped Gauss weights by each full cell width to produce cell averages rather than cell integrals.
    normalized_weights = 0.5 * np.broadcast_to(weights[None, :], points.shape)
    return points, normalized_weights


def frank_eddington_table(
    r_min: float,
    model_r_max: float,
    n_model: int,
    n_energy: int,
    n_eddington_quad: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PchipInterpolator,
    dict[str, float],
]:
    """Build the isotropic Eddington DF used by the cell-average quadrature.

    The model domain is ``r_model[0] <= r <= model_r_max`` with a zero-density
    outer wall at ``model_r_max``.  ``wall_relative_potential`` supplies the
    relative potential psi(r), and ``eddington_table`` inverts rho(psi) to
    f(E) for relative binding energy E >= 0.  PCHIP interpolation preserves the
    tabulated nonnegative shape without extrapolating beyond its energy domain.
    """
    r_model = np.geomspace(min(1.0e-8, r_min * 1.0e-4), model_r_max, n_model)
    r_model[-1] = model_r_max
    rho_model = frank_core_density(r_model, FRANK_RHO_S)
    rho_model[-1] = 0.0
    mass_model = cumulative_trapezoid(4.0 * PI * r_model**2 * rho_model, r_model)
    # psi is measured from the finite model wall. The velocity support uses
    # E=psi-w^2/2-eta/r^2 with this same zero throughout the quadrature.
    psi_model = wall_relative_potential(r_model, mass_model, 1.0)
    energy, f_energy, diagnostics = eddington_table(
        psi_model,
        rho_model,
        n_energy,
        n_eddington_quad,
        True,
    )
    f_of_energy = PchipInterpolator(energy, f_energy, extrapolate=False)
    return r_model, mass_model, psi_model, f_of_energy, {
        **{f"eddington_{key}": value for key, value in diagnostics.items()},
        "model_mass": float(mass_model[-1]),
    }


def integrate_cell_averages(
    r_edges: np.ndarray,
    w_edges: np.ndarray,
    eta_edge: np.ndarray,
    r_model: np.ndarray,
    psi_model: np.ndarray,
    f_of_energy: PchipInterpolator,
    quadrature_points: int,
) -> np.ndarray:
    r"""Integrate the Eddington DF over every ``(r, w, eta)`` cell.

    The returned quantity is

    ``fbar_ijk = 1/(dr_i dw_j deta_k) integral_cell f(E) dr dw deta``,

    where ``E = psi(r) - w**2/2 - eta/r**2`` and ``eta = L**2/2``.
    This is the natural VENUS cell average because the spherical phase-space
    measure is the constant ``8 pi**2 dr dw deta``.  The explicit rectangular
    domain is ``r_edges[0] <= r <= r_edges[-1]``,
    ``w_edges[0] <= w <= w_edges[-1]``, and
    ``eta_edge[0] <= eta <= eta_edge[-1]``.  Within it the physical Eddington
    support is further restricted to ``E > 0``, equivalently
    ``eta < r**2 * (psi(r) - w**2/2)``.

    Tensor-product Gauss--Legendre rules average over r and w.  The eta rule is
    remapped to the occupied fraction of each eta cell, then divided by the
    full eta-cell width; this integrates a sharp E=0 support crossing instead
    of evaluating the DF at a potentially unbound cell center.
    """
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    r_quad, r_weight = map_rule_to_cells(r_edges, nodes, weights)
    w_quad, w_weight = map_rule_to_cells(w_edges, nodes, weights)
    nr = r_edges.size - 1
    nw = w_edges.size - 1
    nl = eta_edge.size - 1
    cell_average = np.zeros((nr, nw, nl), dtype=np.float64)
    # Only interpolate on the closed energy range produced by the Eddington
    # inversion.  Values outside physical support remain exactly zero.
    energy_min = float(f_of_energy.x[0])
    energy_max = float(f_of_energy.x[-1])

    for radial_cell in range(nr):
        shell_average = np.zeros((nw, nl), dtype=np.float64)
        for qr in range(quadrature_points):
            radius = r_quad[radial_cell, qr]
            psi = float(np.interp(np.log(radius), np.log(r_model), psi_model))
            for qw in range(quadrature_points):
                radial_speed2 = w_quad[:, qw] ** 2
                # At fixed (r,w), E>0 truncates the nominal eta interval at
                # eta_support=r^2[psi(r)-w^2/2].
                eta_support = radius * radius * (
                    psi - 0.5 * radial_speed2
                )
                eta_lower = eta_edge[:-1][None, :]
                # At fixed radius and radial velocity only the interval below the positive-binding support contributes to an eta cell.
                eta_upper = np.minimum(
                    eta_edge[1:][None, :], eta_support[:, None]
                )
                occupied_width = np.maximum(eta_upper - eta_lower, 0.0)
                eta_midpoint = eta_lower + 0.5 * occupied_width
                eta_half_width = 0.5 * occupied_width
                for qe in range(quadrature_points):
                    eta_point = eta_midpoint + eta_half_width * nodes[qe]
                    # leggauss weights integrate on [-1,1].  Multiplication
                    # by the occupied half-width maps that integral to eta;
                    # division by the full cell width converts it to the
                    # requested rectangular-cell average.
                    eta_average_weight = (
                        weights[qe]
                        * eta_half_width
                        / np.diff(eta_edge)[None, :]
                    )
                    relative_energy = (
                        psi
                        - 0.5 * radial_speed2[:, None]
                        - eta_point / (radius * radius)
                    )
                    values = np.zeros_like(relative_energy)
                    bound = (occupied_width > 0.0) & (relative_energy > 0.0)
                    if np.any(bound):
                        evaluation_energy = np.clip(
                            relative_energy[bound], energy_min, energy_max
                        )
                        values[bound] = np.maximum(
                            f_of_energy(evaluation_energy), 0.0
                        )
                    # Accumulate radial, velocity and occupied-eta weights in the declared order, preserving the full rectangular-cell normalization.
                    shell_average += (
                        r_weight[radial_cell, qr]
                        * w_weight[:, qw, None]
                        * eta_average_weight
                        * values
                    )
        cell_average[radial_cell] = shell_average
    return cell_average


def delayed_inner_round_trip_time(
    boundary_speed: float,
    angular_momentum: float,
    hidden_mass: float,
    r_min: float,
) -> float:
    """Match the solver's uniform-central-cell flight-time formula."""
    radius = np.longdouble(r_min)
    speed = np.longdouble(boundary_speed)
    angular_momentum_ld = np.longdouble(angular_momentum)
    mass = np.longdouble(hidden_mass)
    angular_momentum2 = angular_momentum_ld * angular_momentum_ld
    central_frequency2 = mass / (radius * radius * radius)
    coefficient = (
        speed * speed
        + angular_momentum2 / (radius * radius)
        + central_frequency2 * radius * radius
    )
    if not coefficient > 0.0:
        return 0.0

    epsilon = np.finfo(np.longdouble).eps
    # Switch to the straight-flight limit when the central harmonic frequency is too small for stable turning-point evaluation.
    small_frequency2 = 64.0 * epsilon * coefficient / (radius * radius)
    if not central_frequency2 > small_frequency2:
        pericentre2 = angular_momentum2 / coefficient
        radial_span = max(np.longdouble(0.0), radius * radius - pericentre2)
        return float(2.0 * np.sqrt(radial_span / coefficient))

    discriminant = (
        coefficient * coefficient
        - 4.0 * central_frequency2 * angular_momentum2
    )
    discriminant_scale = max(coefficient * coefficient, np.longdouble(1.0))
    if discriminant < 0.0 and abs(discriminant) <= (
        256.0 * epsilon * discriminant_scale
    ):
        discriminant = np.longdouble(0.0)
    if discriminant < 0.0:
        raise ValueError("The delayed-return central orbit has no real pericentre")
    root = np.sqrt(discriminant)
    outer_turning_point2 = (
        coefficient + root
    ) / (2.0 * central_frequency2)
    # Use the rationalized inner turning point to avoid cancellation when angular momentum is small.
    pericentre2 = (
        2.0 * angular_momentum2 / (coefficient + root)
        if angular_momentum2 > 0.0
        else np.longdouble(0.0)
    )
    half_span = 0.5 * (outer_turning_point2 - pericentre2)
    if not half_span > 0.0:
        return 0.0
    midpoint = 0.5 * (outer_turning_point2 + pericentre2)
    cosine = np.clip(
        (midpoint - radius * radius) / half_span,
        np.longdouble(-1.0),
        np.longdouble(1.0),
    )
    return float(np.arccos(cosine) / np.sqrt(central_frequency2))


def initialize_stationary_delayed_carrier(
    *,
    r_min: float,
    eta_edge: np.ndarray,
    r_model: np.ndarray,
    psi_model: np.ndarray,
    f_of_energy: PchipInterpolator,
    normalization: float,
    hidden_mass_guess: float,
    eta_points: int,
    speed_points: int,
    phase_points: int,
) -> tuple[list[CarrierPacket], dict[str, float | int]]:
    """Populate the omitted sphere from its stationary inward boundary flux."""
    if eta_points < 1 or speed_points < 2 or phase_points < 2:
        raise ValueError(
            "Carrier initialization needs at least one eta point and two "
            "speed and phase points"
        )
    nodes, weights = np.polynomial.legendre.leggauss(speed_points)
    eta_nodes, eta_weights = np.polynomial.legendre.leggauss(eta_points)
    eta_width = np.diff(eta_edge)
    psi_boundary = float(np.interp(np.log(r_min), np.log(r_model), psi_model))
    eta_support_boundary = psi_boundary * r_min * r_min
    energy_min = float(f_of_energy.x[0])
    energy_max = float(f_of_energy.x[-1])

    def active_eta_rule(k: int) -> tuple[np.ndarray, np.ndarray]:
        lower = eta_edge[k]
        upper = min(eta_edge[k + 1], eta_support_boundary)
        if not upper > lower:
            return np.empty(0), np.empty(0)
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        values = midpoint + half_width * eta_nodes
        normalized_weights = half_width * eta_weights / eta_width[k]
        return values, normalized_weights

    # Stationary interior mass equals inward boundary flux times its orbit residence time, integrated over the supported velocities.
    def population_mass(hidden_mass: float) -> float:
        physical_mass = 0.0
        for k in range(eta_width.size):
            eta_value, normalized_eta_weight = active_eta_rule(k)
            for eta_q, eta_weight_q in zip(
                eta_value, normalized_eta_weight
            ):
                angular_momentum_k = np.sqrt(2.0 * eta_q)
                available_speed2 = (
                    2.0 * psi_boundary
                    - angular_momentum_k * angular_momentum_k
                    / (r_min * r_min)
                )
                if not available_speed2 > 0.0:
                    continue
                maximum_speed = np.sqrt(available_speed2)
                speed = 0.5 * maximum_speed * (nodes + 1.0)
                speed_weight = 0.5 * maximum_speed * weights
                relative_energy = psi_boundary - 0.5 * (
                    speed * speed
                    + angular_momentum_k * angular_momentum_k
                    / (r_min * r_min)
                )
                distribution = normalization * np.maximum(
                    f_of_energy(np.clip(relative_energy, energy_min, energy_max)),
                    0.0,
                )
                duration = np.array([
                    delayed_inner_round_trip_time(
                        float(value),
                        float(angular_momentum_k),
                        hidden_mass,
                        r_min,
                    )
                    for value in speed
                ])
                reduced_mass = np.sum(
                    speed_weight * speed * distribution * duration
                )
                physical_mass += (
                    8.0 * PI**2 * eta_width[k] * eta_weight_q * reduced_mass
                )
        return float(physical_mass)

    hidden_mass = max(0.0, float(hidden_mass_guess))
    iterations = 0
    # The hidden mass affects the central orbital period. Solve that mass/residence-time relation to a finite fixed-point tolerance.
    for iterations in range(1, 101):
        updated_mass = population_mass(hidden_mass)
        if abs(updated_mass - hidden_mass) <= (
            1.0e-12 * max(updated_mass, hidden_mass, 1.0e-300)
        ):
            hidden_mass = updated_mass
            break
        hidden_mass = updated_mass
    else:
        raise RuntimeError("Stationary delayed-carrier mass did not converge")

    packets: list[CarrierPacket] = []
    represented_mass = 0.0
    maximum_round_trip_time = 0.0
    for k in range(eta_width.size):
        eta_value, normalized_eta_weight = active_eta_rule(k)
        for eta_q, eta_weight_q in zip(
            eta_value, normalized_eta_weight
        ):
            angular_momentum_k = np.sqrt(2.0 * eta_q)
            available_speed2 = (
                2.0 * psi_boundary
                - angular_momentum_k * angular_momentum_k / (r_min * r_min)
            )
            if not available_speed2 > 0.0:
                continue
            maximum_speed = np.sqrt(available_speed2)
            speed = 0.5 * maximum_speed * (nodes + 1.0)
            speed_weight = 0.5 * maximum_speed * weights
            relative_energy = psi_boundary - 0.5 * (
                speed * speed
                + angular_momentum_k * angular_momentum_k / (r_min * r_min)
            )
            distribution = normalization * np.maximum(
                f_of_energy(np.clip(relative_energy, energy_min, energy_max)),
                0.0,
            )
            for boundary_speed, quadrature_weight, distribution_value in zip(
                speed, speed_weight, distribution
            ):
                duration = delayed_inner_round_trip_time(
                    float(boundary_speed),
                    float(angular_momentum_k),
                    hidden_mass,
                    r_min,
                )
                reduced_mass = (
                    eta_weight_q
                    * quadrature_weight
                    * boundary_speed
                    * distribution_value
                    * duration
                )
                if not reduced_mass > 0.0 or not duration > 0.0:
                    continue
                packet_mass = float(reduced_mass / phase_points)
                for phase_index in range(phase_points):
                    # Distribute each orbit population uniformly across return phases so a stationary boundary flux populates the omitted sphere.
                    remaining_time = duration * (
                        phase_index + 0.5
                    ) / phase_points
                    packets.append((
                        packet_mass,
                        k,
                        float(eta_q),
                        float(boundary_speed),
                        float(remaining_time),
                    ))
                represented_mass += (
                    8.0 * PI**2 * eta_width[k] * reduced_mass
                )
                maximum_round_trip_time = max(
                    maximum_round_trip_time, duration
                )

    return packets, {
        "carrier_initialization_speed_points": speed_points,
        "carrier_initialization_eta_points": eta_points,
        "carrier_initialization_phase_points": phase_points,
        "carrier_initialization_iterations": iterations,
        "carrier_packet_count": len(packets),
        "carrier_hidden_mass": float(represented_mass),
        "carrier_fixed_point_residual": float(represented_mass - hidden_mass),
        "carrier_maximum_round_trip_time": float(maximum_round_trip_time),
        "carrier_boundary_relative_potential": psi_boundary,
        "carrier_boundary_eta_support": eta_support_boundary,
    }


# Serialize cell averages and delayed-return carriers in the solver layout.
# Physical cell mass is 8*pi^2*fbar*dr*dw*deta, with carrier mass added separately.
def write_restart(
    path: Path,
    f: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    w_max: float,
    w_stretch: float,
    L_max: float,
    eta_grid: str,
    eta_power: float,
    eta_join: float,
    eta_inner_cells: int,
    inner_boundary: str,
    carrier_packets: list[CarrierPacket] | None = None,
) -> None:
    nr, nw, nl = f.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        magic = b"EVLOGRST1"
        handle.write(magic + b"\0" * (16 - len(magic)))
        # Write the exact solver layout, including optional composite eta controls and delayed-return records after the distribution payload.
        version = 9
        handle.write(struct.pack("<iiii", version, nr, nw, nl))
        eta_grid_code = 2 if eta_grid == "composite_power" else 1
        handle.write(struct.pack("<iii", 0, 1, eta_grid_code))
        handle.write(struct.pack("<d", eta_power))
        if eta_grid == "composite_power":
            handle.write(struct.pack("<di", eta_join, eta_inner_cells))
        handle.write(struct.pack("<dd", r_min, r_max))
        handle.write(struct.pack("<di", 0.1, 96))
        handle.write(struct.pack("<dddd", w_max, L_max, w_stretch, 0.0))
        handle.write(struct.pack("<ddq", 0.0, 0.0, 0))
        handle.write(struct.pack("<i", 0))
        handle.write(struct.pack(
            "<iii",
            1 if inner_boundary == "delayed_return" else 0,
            0,
            0,
        ))
        handle.write(struct.pack("<Q", f.size))
        handle.write(np.asarray(f, dtype="<f8").tobytes(order="C"))
        if inner_boundary == "delayed_return":
            packets = carrier_packets if carrier_packets is not None else []
            handle.write(struct.pack("<Q", len(packets)))
            for (
                reduced_mass,
                eta_index,
                eta,
                boundary_speed,
                remaining_time,
            ) in packets:
                handle.write(struct.pack(
                    "<diddd",
                    reduced_mass,
                    eta_index,
                    eta,
                    boundary_speed,
                    remaining_time,
                ))


# Integrate the complete rectangular velocity cells, then divide each shell
# mass by its spherical physical volume to obtain the shell-average density.
def shell_density(
    f: np.ndarray,
    r_edges: np.ndarray,
    w_edges: np.ndarray,
    eta_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shell_mass = (
        8.0
        * PI**2
        * np.sum(f * np.diff(w_edges)[None, :, None] * np.diff(eta_edge)[None, None, :], axis=(1, 2))
        * np.diff(r_edges)
    )
    shell_volume = 4.0 * PI * (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
    return shell_mass, shell_mass / shell_volume


def eta_center_support_diagnostics(
    f: np.ndarray,
    r_edges: np.ndarray,
    w_edges: np.ndarray,
    eta_edge: np.ndarray,
    r_model: np.ndarray,
    psi_model: np.ndarray,
) -> dict[str, float | int]:
    """Measure how strongly occupied cells cross the bound-support surface.

    An analytic bound equilibrium has support only
    where psi(r) - w^2/2 - eta/r^2 > 0. A positive cell average can straddle
    that boundary even when its center lies outside it. The returned mass
    fractions expose this resolution error before a run starts.
    """
    r_center = 0.5 * (r_edges[:-1] + r_edges[1:])
    w_center = 0.5 * (w_edges[:-1] + w_edges[1:])
    eta_center = 0.5 * (eta_edge[:-1] + eta_edge[1:])
    psi_center = np.interp(np.log(r_center), np.log(r_model), psi_model)

    relative_energy_at_center = (
        psi_center[:, None, None]
        - 0.5 * w_center[None, :, None] ** 2
        - eta_center[None, None, :] / r_center[:, None, None] ** 2
    )
    eta_support = r_center[:, None] ** 2 * (
        psi_center[:, None] - 0.5 * w_center[None, :] ** 2
    )
    cell_mass = (
        8.0
        * PI**2
        * f
        * np.diff(r_edges)[:, None, None]
        * np.diff(w_edges)[None, :, None]
        * np.diff(eta_edge)[None, None, :]
    )
    shell_mass = np.sum(cell_mass, axis=(1, 2))
    positive_mass = cell_mass > 0.0
    # A positive cell average can straddle the E=0 surface while its center is unbound. Measure that geometric sampling error explicitly.
    center_outside = positive_mass & (relative_energy_at_center <= 0.0)
    straddles_support = positive_mass & (
        (eta_edge[:-1][None, None, :] < eta_support[:, :, None])
        & (eta_support[:, :, None] < eta_edge[1:][None, None, :])
    )

    center_outside_shell_mass = np.sum(
        np.where(center_outside, cell_mass, 0.0), axis=(1, 2)
    )
    straddling_shell_mass = np.sum(
        np.where(straddles_support, cell_mass, 0.0), axis=(1, 2)
    )
    center_outside_shell_fraction = np.divide(
        center_outside_shell_mass,
        shell_mass,
        out=np.zeros_like(shell_mass),
        where=shell_mass > 0.0,
    )
    straddling_shell_fraction = np.divide(
        straddling_shell_mass,
        shell_mass,
        out=np.zeros_like(shell_mass),
        where=shell_mass > 0.0,
    )
    cell_shell_fraction = np.divide(
        cell_mass,
        shell_mass[:, None, None],
        out=np.zeros_like(cell_mass),
        where=shell_mass[:, None, None] > 0.0,
    )
    cell_shell_fraction = np.where(center_outside, cell_shell_fraction, 0.0)
    largest_flat = int(np.argmax(cell_shell_fraction))
    largest_index = np.unravel_index(largest_flat, cell_shell_fraction.shape)
    largest_i, largest_j, largest_k = (int(value) for value in largest_index)
    inner_flat = int(np.argmax(cell_shell_fraction[0]))
    inner_j, inner_k = (
        int(value)
        for value in np.unravel_index(inner_flat, cell_shell_fraction[0].shape)
    )
    total_mass = float(np.sum(cell_mass))
    outside_mass = float(np.sum(center_outside_shell_mass))

    return {
        "eta_center_outside_support_mass": outside_mass,
        "eta_center_outside_support_mass_fraction": (
            outside_mass / total_mass if total_mass > 0.0 else 0.0
        ),
        "inner_shell_eta_center_outside_support_mass_fraction": float(
            center_outside_shell_fraction[0]
        ),
        "maximum_shell_eta_center_outside_support_mass_fraction": float(
            np.max(center_outside_shell_fraction)
        ),
        "inner_shell_support_straddling_mass_fraction": float(
            straddling_shell_fraction[0]
        ),
        "maximum_shell_support_straddling_mass_fraction": float(
            np.max(straddling_shell_fraction)
        ),
        "largest_center_outside_cell_shell_mass_fraction": float(
            cell_shell_fraction[largest_index]
        ),
        "largest_center_outside_cell_radial_index": largest_i,
        "largest_center_outside_cell_w_index": largest_j,
        "largest_center_outside_cell_eta_index": largest_k,
        "largest_center_outside_cell_r_center": float(r_center[largest_i]),
        "largest_center_outside_cell_w_center": float(w_center[largest_j]),
        "largest_center_outside_cell_eta_lower": float(eta_edge[largest_k]),
        "largest_center_outside_cell_eta_center": float(eta_center[largest_k]),
        "largest_center_outside_cell_eta_upper": float(eta_edge[largest_k + 1]),
        "largest_center_outside_cell_eta_support": float(
            eta_support[largest_i, largest_j]
        ),
        "inner_largest_center_outside_cell_shell_mass_fraction": float(
            cell_shell_fraction[0, inner_j, inner_k]
        ),
        "inner_largest_center_outside_cell_w_index": inner_j,
        "inner_largest_center_outside_cell_eta_index": inner_k,
        "inner_largest_center_outside_cell_w_center": float(w_center[inner_j]),
        "inner_largest_center_outside_cell_eta_lower": float(eta_edge[inner_k]),
        "inner_largest_center_outside_cell_eta_center": float(eta_center[inner_k]),
        "inner_largest_center_outside_cell_eta_upper": float(eta_edge[inner_k + 1]),
        "inner_largest_center_outside_cell_eta_support": float(
            eta_support[0, inner_j]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--nr", type=int, default=96)
    parser.add_argument("--nw", type=int, default=128)
    parser.add_argument("--nl", type=int, default=64)
    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--r-max", type=float, default=1.0e3)
    parser.add_argument("--w-max", type=float, default=3.0)
    parser.add_argument(
        "--w-stretch",
        type=float,
        default=0.0,
        help="Symmetric sinh grading strength; zero gives uniform w cells.",
    )
    parser.add_argument("--L-max", type=float, default=100.0)
    parser.add_argument("--eta-power", type=float, default=10.0)
    parser.add_argument(
        "--eta-grid",
        choices=("power", "composite_power"),
        default="power",
    )
    parser.add_argument("--eta-join", type=float, default=5000.0)
    parser.add_argument("--eta-inner-cells", type=int, default=32)
    parser.add_argument(
        "--quadrature-points",
        type=int,
        choices=range(1, 13),
        default=12,
        help=(
            "Gauss-Legendre points per phase-space axis. Frank's weak "
            "central cusp requires 12 points for a converged innermost "
            "cell average on the benchmark grid."
        ),
    )
    parser.add_argument("--model-r-max", type=float, default=1.0e6)
    parser.add_argument("--n-model", type=int, default=50000)
    parser.add_argument("--n-energy", type=int, default=900)
    parser.add_argument("--n-eddington-quad", type=int, default=96)
    parser.add_argument(
        "--normalization-mode",
        choices=("exact_profile", "match_total_mass"),
        default="exact_profile",
        help=(
            "Keep Frank's analytic rho_s normalization, or reproduce the "
            "legacy global rescaling of the finite phase-space domain."
        ),
    )
    parser.add_argument(
        "--inner-boundary",
        choices=("reflecting", "delayed_return"),
        default="delayed_return",
    )
    parser.add_argument("--carrier-speed-points", type=int, default=128)
    parser.add_argument("--carrier-phase-points", type=int, default=64)
    parser.add_argument(
        "--carrier-eta-points",
        type=int,
        choices=range(1, 13),
        default=2,
        help="Gauss-Legendre eta nodes used to populate each carrier cell.",
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=FRANK_SIGMA0,
        help="Dimensionless cross section used only for the central optical-depth report.",
    )
    args = parser.parse_args()

    if args.nr < 3 or args.nw < 3 or args.nl < 2:
        raise ValueError("Require Nr >= 3, Nw >= 3, and NL >= 2.")
    if not (0.0 < args.r_min < args.r_max <= args.model_r_max):
        raise ValueError("Require 0 < r_min < r_max <= model_r_max.")
    if (
        args.w_max <= 0.0
        or args.L_max <= 0.0
        or args.eta_power < 1.0
        or args.w_stretch < 0.0
    ):
        raise ValueError("Velocity limits must be positive and eta_power must be >= 1.")
    if args.carrier_speed_points < 2 or args.carrier_phase_points < 2:
        raise ValueError("Carrier speed and phase quadrature counts must be at least two.")

    r_edge = cell_edges(args.r_min, args.r_max, args.nr, True)
    w_edge = stretched_w_edges(args.w_max, args.nw, args.w_stretch)
    eta_edge = (
        eta_edges(args.L_max, args.nl, args.eta_power)
        if args.eta_grid == "power"
        else composite_eta_edges(
            args.L_max,
            args.nl,
            args.eta_power,
            args.eta_join,
            args.eta_inner_cells,
        )
    )
    if args.sigma0 < 0.0:
        raise ValueError("--sigma0 must be nonnegative")

    (
        r_model,
        mass_model,
        psi_model,
        f_of_energy,
        eddington_diagnostics,
    ) = frank_eddington_table(
        args.r_min,
        args.model_r_max,
        args.n_model,
        args.n_energy,
        args.n_eddington_quad,
    )
    f = integrate_cell_averages(
        r_edge,
        w_edge,
        eta_edge,
        r_model,
        psi_model,
        f_of_energy,
        args.quadrature_points,
    )

    cell_measure = (
        np.diff(r_edge)[:, None, None]
        * np.diff(w_edge)[None, :, None]
        * np.diff(eta_edge)[None, None, :]
    )
    mass_before = float(8.0 * PI**2 * np.sum(f * cell_measure))
    # The exact-profile mode preserves the analytic density amplitude. The alternative rescales the finite represented domain to a requested total mass.
    normalization = (
        FRANK_TOTAL_MASS / mass_before
        if args.normalization_mode == "match_total_mass"
        else 1.0
    )
    f *= normalization
    represented_mass = float(8.0 * PI**2 * np.sum(f * cell_measure))
    shell_mass, represented_density = shell_density(f, r_edge, w_edge, eta_edge)
    eta_support_diagnostics = eta_center_support_diagnostics(
        f,
        r_edge,
        w_edge,
        eta_edge,
        r_model,
        psi_model,
    )
    shell_center = 0.5 * (r_edge[:-1] + r_edge[1:])
    target_density = frank_core_density(shell_center, FRANK_RHO_S)
    useful = target_density > target_density.max() * 1.0e-10
    relative_density_error = (
        represented_density[useful] - target_density[useful]
    ) / target_density[useful]

    model_inner_radius = float(r_model[0])
    # Add the analytic small-radius cusp integral to the tabulated enclosed mass when estimating the omitted sphere.
    unresolved_mass_below_model = model_inner_radius**2.9 / 2.9
    unresolved_mass = unresolved_mass_below_model + float(
        np.interp(args.r_min, r_model, mass_model)
    )
    carrier_packets: list[CarrierPacket] = []
    carrier_diagnostics: dict[str, float | int] = {}
    if args.inner_boundary == "delayed_return":
        carrier_packets, carrier_diagnostics = initialize_stationary_delayed_carrier(
            r_min=args.r_min,
            eta_edge=eta_edge,
            r_model=r_model,
            psi_model=psi_model,
            f_of_energy=f_of_energy,
            normalization=normalization,
            hidden_mass_guess=unresolved_mass,
            eta_points=args.carrier_eta_points,
            speed_points=args.carrier_speed_points,
            phase_points=args.carrier_phase_points,
        )

    write_restart(
        args.output,
        f,
        r_min=args.r_min,
        r_max=args.r_max,
        w_max=args.w_max,
        w_stretch=args.w_stretch,
        L_max=args.L_max,
        eta_grid=args.eta_grid,
        eta_power=args.eta_power,
        eta_join=args.eta_join,
        eta_inner_cells=args.eta_inner_cells,
        inner_boundary=args.inner_boundary,
        carrier_packets=carrier_packets,
    )

    density_column_below_model = (
        FRANK_RHO_S * model_inner_radius**0.9 / 0.9
    )
    inside = r_model <= args.r_min
    column_radius = np.concatenate(
        (r_model[inside], np.array([args.r_min], dtype=float))
    )
    column_radius = np.unique(column_radius)
    column_density = frank_core_density(column_radius, FRANK_RHO_S)
    density_column = density_column_below_model + float(
        np.trapz(column_density, column_radius)
    )
    # Cross section per mass times the radial density column gives the central optical depth used in the diagnostic report.
    central_optical_depth = args.sigma0 * density_column
    density_at_r_min = float(frank_core_density(
        np.array([args.r_min]), FRANK_RHO_S
    )[0])
    local_mean_free_path = (
        1.0 / (args.sigma0 * density_at_r_min)
        if args.sigma0 > 0.0
        else float("inf")
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "output": str(args.output),
        "restart_format": "EVLOGRST1 version 9",
        "inner_boundary": args.inner_boundary,
        "kinetic_grid": "cell_average_eta",
        "radial_grid": "log",
        "kinetic_eta_grid": args.eta_grid,
        "Nr": args.nr,
        "Nw": args.nw,
        "NL": args.nl,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "w_max": args.w_max,
        "w_stretch": args.w_stretch,
        "minimum_dw": float(np.min(np.diff(w_edge))),
        "maximum_dw": float(np.max(np.diff(w_edge))),
        "L_max": args.L_max,
        "kinetic_eta_power": args.eta_power,
        "kinetic_eta_join": args.eta_join,
        "kinetic_eta_inner_cells": args.eta_inner_cells,
        "cell_quadrature_points_per_axis": args.quadrature_points,
        "rho_s": FRANK_RHO_S,
        "target_total_mass": FRANK_TOTAL_MASS,
        "normalization_mode": args.normalization_mode,
        "represented_mass_before_normalization": mass_before,
        "mass_normalization_factor": normalization,
        "represented_mass": represented_mass,
        "represented_total_mass_with_carrier": (
            represented_mass
            + float(carrier_diagnostics.get("carrier_hidden_mass", 0.0))
        ),
        "finite_domain_missing_mass": float(
            eddington_diagnostics["model_mass"] - represented_mass
        ),
        "sum_shell_mass": float(np.sum(shell_mass)),
        "minimum_f": float(np.min(f)),
        "maximum_f": float(np.max(f)),
        "nonfinite_f_count": int(np.count_nonzero(~np.isfinite(f))),
        "negative_f_count": int(np.count_nonzero(f < 0.0)),
        "inner_shell_center": float(shell_center[0]),
        "inner_target_density": float(target_density[0]),
        "inner_represented_shell_density": float(represented_density[0]),
        "median_abs_relative_density_error": float(
            np.median(np.abs(relative_density_error))
        ),
        "max_abs_relative_density_error": float(
            np.max(np.abs(relative_density_error))
        ),
        "first_eta_edge": float(eta_edge[1]),
        "first_L_edge": float(np.sqrt(2.0 * eta_edge[1])),
        "model_r_max": args.model_r_max,
        "analytic_mass_inside_r_min": unresolved_mass,
        "analytic_mass_inside_r_min_fraction": (
            unresolved_mass / eddington_diagnostics["model_mass"]
        ),
        "carrier_to_analytic_inner_mass_ratio": (
            float(carrier_diagnostics.get("carrier_hidden_mass", 0.0))
            / unresolved_mass
            if unresolved_mass > 0.0 and carrier_diagnostics
            else 0.0
        ),
        "sigma0_for_optical_depth": args.sigma0,
        "central_optical_depth_0_to_r_min": central_optical_depth,
        "density_at_r_min": density_at_r_min,
        "local_mean_free_path_at_r_min": local_mean_free_path,
        "r_min_over_local_mean_free_path": (
            args.r_min / local_mean_free_path
            if np.isfinite(local_mean_free_path)
            else 0.0
        ),
        **eta_support_diagnostics,
        **carrier_diagnostics,
        **eddington_diagnostics,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
