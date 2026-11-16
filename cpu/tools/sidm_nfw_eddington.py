#!/usr/bin/env python3
# Construct isotropic spherical equilibria by solving for their potential and inverting density as a function of relative potential.
# The carrier grid uses radius r, radial specific velocity w and specific angular momentum L. Physical density integrates with 2*pi*L/r squared in velocity space.
# Output EVICRWL1 stores little-endian axes and row-major distribution values; the optional report measures their represented mass and density accuracy.

"""Construct spherical SIDM/NFW equilibria by isotropic Eddington inversion.

The reusable API supplies analytic NFW, cored, beta-core, Plummer, and Frank
Core density laws; smooth finite-radius truncation; enclosed-mass and
wall-relative-potential construction; and the numerical ``f(E)`` inversion.
Maintained Frank initial-condition tools import these functions so every
particle and cell-average generator uses the same equilibrium definition.

As a command-line generator, the module samples the resulting distribution on
an ``(r,w,L)`` carrier grid and writes the versioned ``EVICRWL1`` binary format
consumed by the CPU solvers, together with machine-readable diagnostics.
Quadrature weights and arrays follow solver storage order; input profiles are
never inferred from machine-specific paths.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import erf


PI = np.pi


# Section: quadrature and EVICRWL1 output.
# The generated f(r,w,L) is written in the same binary format consumed by the
# C++ and Python solvers.

def weights(x: np.ndarray) -> np.ndarray:
    """Return trapezoidal quadrature weights at one-dimensional grid nodes."""
    out = np.empty_like(x)
    out[0] = 0.5 * (x[1] - x[0])
    out[-1] = 0.5 * (x[-1] - x[-2])
    out[1:-1] = 0.5 * (x[2:] - x[:-2])
    return out


def write_ev_ic(path: Path, r: np.ndarray, w: np.ndarray, L: np.ndarray, f: np.ndarray) -> None:
    """Write an ``EVICRWL1`` distribution in solver row-major order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        magic = b"EVICRWL1"
        handle.write(magic + b"\0" * (16 - len(magic)))
        handle.write(struct.pack("<iiii", 1, r.size, w.size, L.size))
        handle.write(np.asarray(r, dtype="<f8").tobytes())
        handle.write(np.asarray(w, dtype="<f8").tobytes())
        handle.write(np.asarray(L, dtype="<f8").tobytes())
        handle.write(np.asarray(f, dtype="<f8").tobytes(order="C"))


# Section: density models and smooth truncation.
# These functions define the target spherical density profile before
# Eddington inversion.

def nfw_density(r: np.ndarray, rho0: float, eps: float) -> np.ndarray:
    """Evaluate a softened NFW density with scale radius equal to one."""
    return rho0 / ((r + eps) * (1.0 + r) ** 2)


def cored_nfw_density(r: np.ndarray, rho0: float) -> np.ndarray:
    """Evaluate the finite-central-density ``(1+r)^-3`` halo model."""
    return rho0 / (1.0 + r) ** 3


def cored_r2_nfw_density(r: np.ndarray, rho0: float) -> np.ndarray:
    """Evaluate the cored ``(1+r^2)^-3/2`` halo model."""
    return rho0 / (1.0 + r * r) ** 1.5


def beta_core_density(r: np.ndarray, rho0: float, core_radius: float, alpha: float) -> np.ndarray:
    """Evaluate a beta-core density with independent radius and exponent."""
    x = r / core_radius
    return rho0 / (1.0 + x * x) ** alpha


def plummer_density(r: np.ndarray, rho0: float, scale_radius: float) -> np.ndarray:
    """Evaluate the spherical Plummer density law."""
    x = r / scale_radius
    return rho0 / (1.0 + x * x) ** 2.5


def gravothermal_core_density(
    r: np.ndarray,
    rho0: float,
    core_radius: float,
    halo_slope: float,
    transition_radius: float,
    outer_slope: float,
    transition_order: float,
) -> np.ndarray:
    """Cored profile with independent self-similar and finite-mass slopes."""
    core = (1.0 + (r / core_radius) ** 2) ** (-0.5 * halo_slope)
    transition = (1.0 + (r / transition_radius) ** transition_order) ** (
        -(outer_slope - halo_slope) / transition_order
    )
    return rho0 * core * transition


def frank_core_density(r: np.ndarray, rho0: float) -> np.ndarray:
    """Finite-mass Core halo from van den Bosch & Tokayer (2026), equation (2)."""
    return rho0 / (r**0.1 * (1.0 + r**4) ** 0.975)


def base_density(
    r: np.ndarray,
    rho0: float,
    eps: float,
    profile: str,
    core_radius: float,
    beta_alpha: float,
    halo_slope: float,
    transition_radius: float,
    outer_slope: float,
    transition_order: float,
) -> np.ndarray:
    """Dispatch a named analytic density model on the supplied radius grid."""
    if profile == "nfw":
        return nfw_density(r, rho0, eps)
    if profile == "cored_nfw":
        return cored_nfw_density(r, rho0)
    if profile == "cored_r2_nfw":
        return cored_r2_nfw_density(r, rho0)
    if profile == "beta_core":
        return beta_core_density(r, rho0, core_radius, beta_alpha)
    if profile == "plummer":
        return plummer_density(r, rho0, core_radius)
    if profile == "gravothermal_core":
        return gravothermal_core_density(
            r,
            rho0,
            core_radius,
            halo_slope,
            transition_radius,
            outer_slope,
            transition_order,
        )
    if profile == "frank_core":
        return frank_core_density(r, rho0)
    raise ValueError(f"Unknown density profile: {profile}")


def truncation(r: np.ndarray, r_edge: float, power: float, order: float) -> np.ndarray:
    """Return the compact-support taper applied at the outer model edge."""
    x = np.clip(r / r_edge, 0.0, 1.0) ** power
    return np.maximum(1.0 - x, 0.0) ** order


def target_density(
    r: np.ndarray,
    rho0: float,
    eps: float,
    r_edge: float,
    power: float,
    order: float,
    profile: str,
    core_radius: float,
    beta_alpha: float,
    halo_slope: float,
    transition_radius: float,
    outer_slope: float,
    transition_order: float,
    apply_truncation: bool = True,
) -> np.ndarray:
    """Evaluate the chosen base density with an optional smooth truncation."""
    density = base_density(
        r,
        rho0,
        eps,
        profile,
        core_radius,
        beta_alpha,
        halo_slope,
        transition_radius,
        outer_slope,
        transition_order,
    )
    if apply_truncation:
        density *= truncation(r, r_edge, power, order)
    return density


# Section: potential and solver-grid mass construction.

def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Integrate sampled values cumulatively with the trapezoidal rule."""
    out = np.zeros_like(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1]))
    return out


def wall_relative_potential(r: np.ndarray, mass: np.ndarray, gravity: float) -> np.ndarray:
    """Integrate ``G M(<r)/r^2`` inward from a zero-potential outer wall."""
    # Integrate the positive inward potential-depth contribution from the outer wall, which defines relative potential zero.
    force = gravity * mass / np.maximum(r * r, 1.0e-300)
    psi = np.zeros_like(r)
    increments = 0.5 * (force[:-1] + force[1:]) * (r[1:] - r[:-1])
    psi[:-1] = np.cumsum(increments[::-1])[::-1]
    return psi


def solver_grid_mass_from_density(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Construct enclosed spherical mass on the solver's radial nodes."""
    mass = np.zeros_like(r)
    # The innermost unresolved sphere uses a uniform-density central estimate before integrating the sampled outer density.
    mass[0] = 4.0 * PI * rho[0] * r[0] ** 3 / 3.0
    if r.size > 1:
        y = r * r * rho
        mass[1:] = mass[0] + np.cumsum(4.0 * PI * 0.5 * (y[:-1] + y[1:]) * (r[1:] - r[:-1]))
    return mass


def solver_grid_relative_potential(
    r: np.ndarray,
    rho: np.ndarray,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wall-relative potential and enclosed mass on a solver grid."""
    mass = solver_grid_mass_from_density(r, rho)
    return wall_relative_potential(r, mass, gravity), mass


def stretched_w_grid(w_max: float, n: int, stretch: float) -> np.ndarray:
    """Build a symmetric linear or sinh-stretched radial-momentum grid."""
    x = np.linspace(-1.0, 1.0, n)
    if stretch <= 0.0:
        return w_max * x
    return w_max * np.sinh(stretch * x) / np.sinh(stretch)


def power_l_grid(L_max: float, n: int, power: float) -> np.ndarray:
    """Build a non-negative angular-momentum grid with power-law spacing."""
    x = np.linspace(0.0, 1.0, n)
    return L_max * x**power


def guarded_w_grid(
    w_max: float,
    n: int,
    stretch: float,
    core_max: float | None,
    guard_cells_per_side: int,
) -> np.ndarray:
    """Build a stretched momentum core with linear guard cells at both ends."""
    if guard_cells_per_side <= 0:
        return stretched_w_grid(w_max, n, stretch)
    if core_max is None or not (0.0 < core_max < w_max):
        raise ValueError("Guarded w grids require 0 < --w-core-max < --w-max.")
    core_n = n - 2 * guard_cells_per_side
    if core_n < 3:
        raise ValueError("Guarded w grids require at least three non-guard cells.")
    core = stretched_w_grid(core_max, core_n, stretch)
    # Extra high-speed cells extend the velocity domain beyond the finely sampled core while preserving symmetric radial velocities.
    positive_guard = np.linspace(core_max, w_max, guard_cells_per_side + 1)[1:]
    return np.concatenate((-positive_guard[::-1], core, positive_guard))


def guarded_l_grid(
    L_max: float,
    n: int,
    power: float,
    core_max: float | None,
    guard_cells: int,
) -> np.ndarray:
    """Build a power-spaced angular-momentum core plus outer guard cells."""
    if guard_cells <= 0:
        return power_l_grid(L_max, n, power)
    if core_max is None or not (0.0 < core_max < L_max):
        raise ValueError("Guarded L grids require 0 < --L-core-max < --L-max.")
    core_n = n - guard_cells
    if core_n < 2:
        raise ValueError("Guarded L grids require at least two non-guard cells.")
    core = power_l_grid(core_max, core_n, power)
    guard = np.linspace(core_max, L_max, guard_cells + 1)[1:]
    return np.concatenate((core, guard))


def origin_centered_log_grid(r_min: float, r_max: float, n: int) -> np.ndarray:
    """Log-like centers whose innermost control-volume centroid is r_min."""
    if n < 3:
        raise ValueError("origin_centered_log_grid requires at least three points.")
    if 3.0 * r_min >= r_max:
        raise ValueError("origin_centered_log_grid requires 3*r_min < r_max.")
    r = np.empty(n)
    r[0] = r_min
    r[1:] = np.geomspace(3.0 * r_min, r_max, n - 1)
    return r


def origin_linear_log_grid(
    r_min: float,
    r_max: float,
    n: int,
    inner_linear_cells: int,
) -> np.ndarray:
    """Cell-centered uniform core followed by a logarithmic outer grid."""
    if not (2 <= inner_linear_cells <= n - 2):
        raise ValueError("origin_linear_log_grid requires 2 <= inner_linear_cells <= n-2.")
    r = np.empty(n)
    indices = np.arange(inner_linear_cells, dtype=float)
    r[:inner_linear_cells] = (2.0 * indices + 1.0) * r_min
    first_outer = (2.0 * inner_linear_cells + 1.0) * r_min
    if first_outer >= r_max:
        raise ValueError("origin_linear_log_grid inner uniform core reaches r_max.")
    r[inner_linear_cells:] = np.geomspace(first_outer, r_max, n - inner_linear_cells)
    return r


def king_density_function(W: np.ndarray | float) -> np.ndarray:
    """Dimensionless density of the isotropic lowered-isothermal King DF."""
    values = np.maximum(np.asarray(W, dtype=float), 0.0)
    out = np.zeros_like(values)
    regular = values >= 1.0e-5
    if np.any(regular):
        wr = values[regular]
        root = np.sqrt(wr)
        out[regular] = np.exp(wr) * erf(root) - (2.0 / np.sqrt(PI)) * root * (1.0 + 2.0 * wr / 3.0)
    if np.any(~regular):
        ws = values[~regular]
        # The small-W King density uses its series limit to avoid subtracting nearly equal terms at the tidal boundary.
        out[~regular] = (8.0 / (15.0 * np.sqrt(PI))) * ws**2.5
    return out


def solve_dimensionless_king_model(
    W0: float,
) -> tuple[object, float, float]:
    """Solve the King Poisson equation through its finite tidal radius."""
    central_density_function = float(king_density_function(W0))
    if central_density_function <= 0.0:
        raise ValueError("--king-w0 must be positive.")

    x0 = 1.0e-6

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        """Evaluate the dimensionless King potential/mass ODE system."""
        density_ratio = float(king_density_function(y[0])) / central_density_function
        return np.array([y[1], -2.0 * y[1] / x - 9.0 * density_ratio, x * x * density_ratio])

    def tidal_event(_x: float, y: np.ndarray) -> float:
        """Stop integration where the relative King potential reaches zero."""
        return float(y[0])

    tidal_event.terminal = True
    tidal_event.direction = -1.0
    # Start the King equation with the regular central series so the spherical 2/x derivative term remains finite.
    initial = np.array([W0 - 1.5 * x0 * x0, -3.0 * x0, x0**3 / 3.0])
    solution = solve_ivp(
        rhs,
        (x0, 1.0e5),
        initial,
        events=tidal_event,
        dense_output=True,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step=0.05,
    )
    if not solution.success or solution.t_events[0].size != 1:
        raise RuntimeError("King-model integration did not reach a unique tidal radius.")
    tidal_x = float(solution.t_events[0][0])
    dimensionless_mass = float(solution.y_events[0][0][2])
    return solution, tidal_x, dimensionless_mass


def generate_king_initial_data(args: argparse.Namespace) -> None:
    """Generate an exact isotropic King f(E) equilibrium on the solver grid."""
    if args.target_mass is None or args.target_mass <= 0.0:
        raise ValueError("--profile king requires a positive --target-mass.")
    if args.king_r0 <= 0.0:
        raise ValueError("--king-r0 must be positive.")

    solution, tidal_x, dimensionless_mass = solve_dimensionless_king_model(args.king_w0)
    tidal_radius = args.king_r0 * tidal_x
    if tidal_radius >= args.r_max:
        raise ValueError(
            f"King tidal radius {tidal_radius:.8g} must lie inside r_max={args.r_max:.8g}."
        )

    # Scale the dimensionless King solution to its requested physical mass and core radius before sampling its distribution.
    central_density = args.target_mass / (
        4.0 * PI * args.king_r0**3 * dimensionless_mass
    )
    sigma2 = 4.0 * PI * args.gravity * central_density * args.king_r0**2 / 9.0
    density_function_0 = float(king_density_function(args.king_w0))
    density_scale = central_density / density_function_0

    if args.radial_grid == "origin_centered_log":
        r = origin_centered_log_grid(args.r_min, args.r_max, args.nr)
    elif args.radial_grid == "origin_linear_log":
        r = origin_linear_log_grid(args.r_min, args.r_max, args.nr, args.inner_linear_cells)
    else:
        r = np.geomspace(args.r_min, args.r_max, args.nr)
    w = guarded_w_grid(
        args.w_max, args.nw, args.w_stretch, args.w_core_max, args.w_guard_cells_per_side
    )
    L = guarded_l_grid(args.L_max, args.nl, args.L_power, args.L_core_max, args.L_guard_cells)

    x = r / args.king_r0
    W = np.zeros_like(r)
    inside = x < tidal_x
    W[inside] = np.maximum(solution.sol(x[inside])[0], 0.0)
    relative_potential = sigma2 * W
    rr = r[:, None, None]
    speed2 = w[None, :, None] ** 2 + (L[None, None, :] / rr) ** 2
    # At fixed radius the bound velocity support has relative energy Psi minus one-half speed squared.
    relative_energy = relative_potential[:, None, None] - 0.5 * speed2
    f = np.zeros_like(relative_energy)
    bound = relative_energy > 0.0
    f[bound] = density_scale / (2.0 * PI * sigma2) ** 1.5 * np.expm1(
        relative_energy[bound] / sigma2
    )
    f[~np.isfinite(f)] = 0.0
    write_ev_ic(args.output, r, w, L, f)

    if args.report is None:
        return

    rho_grid = density_from_f(f, r, w, L)
    rho_target = density_scale * king_density_function(W)
    rq = weights(r)
    represented_mass = float(np.sum(4.0 * PI * r * r * rho_grid * rq))
    target_grid_mass = float(np.sum(4.0 * PI * r * r * rho_target * rq))
    rel_err = np.divide(
        rho_grid - rho_target,
        np.maximum(rho_target, 1.0e-300),
        out=np.zeros_like(rho_grid),
    )
    use = rho_target > max(np.max(rho_target) * 1.0e-10, 1.0e-30)
    x_samples = np.linspace(0.0, tidal_x, 20000)
    mass_samples = solution.sol(np.maximum(x_samples, 1.0e-6))[2]
    half_mass_x = float(np.interp(0.5 * dimensionless_mass, mass_samples, x_samples))
    vesc = np.sqrt(np.maximum(2.0 * relative_potential, 0.0))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "output": str(args.output),
        "profile": "king",
        "Nr": args.nr,
        "Nw": args.nw,
        "NL": args.nl,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "radial_grid": args.radial_grid,
        "inner_linear_cells": args.inner_linear_cells,
        "w_max": args.w_max,
        "w_stretch": args.w_stretch,
        "w_core_max": args.w_core_max,
        "w_guard_cells_per_side": args.w_guard_cells_per_side,
        "L_max": args.L_max,
        "L_power": args.L_power,
        "L_core_max": args.L_core_max,
        "L_guard_cells": args.L_guard_cells,
        "target_mass_input": args.target_mass,
        "king_W0": args.king_w0,
        "king_r0": args.king_r0,
        "king_tidal_x": tidal_x,
        "king_tidal_radius": tidal_radius,
        "king_dimensionless_mass": dimensionless_mass,
        "king_half_mass_radius": args.king_r0 * half_mass_x,
        "king_r0_over_half_mass_radius": 1.0 / half_mass_x,
        "central_density": central_density,
        "king_sigma2_parameter": sigma2,
        "target_grid_mass": target_grid_mass,
        "represented_grid_mass": represented_mass,
        "rho_inner_target": float(rho_target[0]),
        "rho_inner_grid": float(rho_grid[0]),
        "v_escape_max_on_grid": float(np.max(vesc)),
        "max_r_times_vesc": float(np.max(r * vesc)),
        "L_first_nonzero": float(L[1]),
        "w_first_positive": float(w[np.searchsorted(w, 0.0)]),
        "max_abs_rel_density_error_used": float(np.max(np.abs(rel_err[use]))) if np.any(use) else 0.0,
        "median_abs_rel_density_error_used": float(np.median(np.abs(rel_err[use]))) if np.any(use) else 0.0,
        "negative_f_count": int(np.count_nonzero(f < 0.0)),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


# Section: Eddington inversion.
# The table maps relative energy to an isotropic f(E); distribution_from_potential
# samples that table on the solver's (r,w,L) grid.

def eddington_table(
    psi: np.ndarray,
    rho: np.ndarray,
    n_energy: int,
    n_quad: int,
    clip_negative: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Invert ``rho(Psi)`` into an isotropic distribution table ``f(E)``.

    The second derivative of a monotone PCHIP density interpolant is integrated
    with Gauss--Legendre quadrature on a cosine-spaced energy grid.  Returned
    diagnostics expose negative raw samples and the central endpoint rule.
    """
    # Invert to increasing relative potential, retaining distinct abscissae for the shape-preserving density interpolant.
    psi_asc = psi[::-1].copy()
    rho_asc = rho[::-1].copy()
    psi_asc[0] = 0.0
    rho_asc[0] = 0.0
    keep = np.concatenate(([True], np.diff(psi_asc) > 0.0))
    psi_asc = psi_asc[keep]
    rho_asc = rho_asc[keep]

    rho_of_psi = PchipInterpolator(psi_asc, rho_asc, extrapolate=False)
    rho2 = rho_of_psi.derivative(2)

    x = np.linspace(0.0, 1.0, n_energy)
    energy = psi_asc[-1] * 0.5 * (1.0 - np.cos(PI * x))
    energy[0] = 0.0
    nodes, qweights = np.polynomial.legendre.leggauss(n_quad)
    # The substitution Psi=E*sin(theta) squared removes the square-root endpoint singularity in Eddington inversion.
    theta = 0.25 * PI * (nodes + 1.0)
    theta_w = 0.25 * PI * qweights
    sin_theta = np.sin(theta)
    sin2 = sin_theta * sin_theta

    f_raw = np.zeros_like(energy)
    prefactor = 1.0 / (np.sqrt(8.0) * PI * PI)
    # Do not evaluate rho'' exactly at the maximum relative potential.  For a
    # flat-density core, the one-sided PCHIP endpoint derivative is not a
    # convergent representation of the central limit and can create one huge,
    # sign-indefinite value in an otherwise positive f(E).  The cosine energy
    # grid places the penultimate point O(n_energy^-2) from the endpoint, so
    # its value is the resolved central limit to much better accuracy than the
    # velocity-grid sampling.
    for i in range(1, energy.size - 1):
        E = energy[i]
        values = rho2(E * sin2)
        integral = 2.0 * np.sqrt(E) * np.sum(theta_w * values * sin_theta)
        f_raw[i] = prefactor * integral
    f_raw[-1] = f_raw[-2]

    f_clip = np.maximum(f_raw, 0.0) if clip_negative else f_raw
    diagnostics = {
        "psi_max": float(psi_asc[-1]),
        "f_raw_min": float(np.min(f_raw)),
        "f_raw_max": float(np.max(f_raw)),
        "negative_f_count": int(np.count_nonzero(f_raw < 0.0)),
        "negative_f_min": float(np.min(f_raw[f_raw < 0.0])) if np.any(f_raw < 0.0) else 0.0,
        "central_endpoint_treatment": "penultimate_energy_limit",
    }
    return energy, f_clip, diagnostics


def density_from_f(f: np.ndarray, r: np.ndarray, w: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Integrate ``f(r,w,L)`` over momentum to recover spherical density."""
    wq = weights(w)
    Lq = weights(L)
    # The tangential velocity Jacobian gives L*dL/r squared. Azimuth integration supplies 2*pi.
    moment = np.einsum("ijk,j,k,k->i", f, wq, Lq, L, optimize=True)
    return 2.0 * PI * moment / (r * r)


def distribution_from_potential(
    psi: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    f_of_energy: PchipInterpolator,
) -> np.ndarray:
    """Sample an isotropic ``f(E)`` interpolant on an ``(r,w,L)`` grid."""
    rr = r[:, None, None]
    speed2 = w[None, :, None] * w[None, :, None] + (L[None, None, :] / rr) ** 2
    eps = psi[:, None, None] - 0.5 * speed2
    f = np.zeros_like(eps)
    mask = eps > 0.0
    energy_grid = getattr(f_of_energy, "x", None)
    if energy_grid is not None and len(energy_grid) >= 2:
        eval_eps = np.clip(eps[mask], energy_grid[0], energy_grid[-1])
    else:
        eval_eps = eps[mask]
    f[mask] = np.maximum(f_of_energy(eval_eps), 0.0)
    f[~np.isfinite(f)] = 0.0
    return f


# Section: command-line interface and generation workflow.

def main() -> None:
    """Generate a selected equilibrium grid, binary IC, and diagnostic report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--nr", type=int, default=128)
    parser.add_argument("--nw", type=int, default=128)
    parser.add_argument("--nl", type=int, default=80)
    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--r-max", type=float, default=100.0)
    parser.add_argument(
        "--radial-grid",
        choices=("log", "origin_centered_log", "origin_linear_log"),
        default="log",
        help="Solver radial centers; origin_* grids center the innermost r=0 control volume.",
    )
    parser.add_argument(
        "--inner-linear-cells",
        type=int,
        default=24,
        help="Number of uniform cell-centered core points for --radial-grid origin_linear_log.",
    )
    parser.add_argument("--w-max", type=float, default=1.0)
    parser.add_argument("--w-stretch", type=float, default=2.5)
    parser.add_argument(
        "--w-core-max",
        type=float,
        default=None,
        help="Interior stretched-grid endpoint when empty high-|w| guard cells are requested.",
    )
    parser.add_argument(
        "--w-guard-cells-per-side",
        type=int,
        default=0,
        help="Linear guard cells appended beyond each side of --w-core-max.",
    )
    parser.add_argument("--L-max", type=float, default=15.0)
    parser.add_argument("--L-power", type=float, default=4.0)
    parser.add_argument(
        "--L-core-max",
        type=float,
        default=None,
        help="Interior power-grid endpoint when empty high-L guard cells are requested.",
    )
    parser.add_argument(
        "--L-guard-cells",
        type=int,
        default=0,
        help="Linear guard cells appended beyond --L-core-max.",
    )
    parser.add_argument("--rho-norm", type=float, default=0.020)
    parser.add_argument("--target-mass", type=float, default=None)
    parser.add_argument(
        "--profile",
        choices=(
            "nfw",
            "cored_nfw",
            "cored_r2_nfw",
            "beta_core",
            "plummer",
            "gravothermal_core",
            "frank_core",
            "king",
        ),
        default="nfw",
    )
    parser.add_argument(
        "--core-radius",
        "--scale-radius",
        dest="core_radius",
        type=float,
        default=1.0,
        help="Core radius, or Plummer scale radius for --profile plummer.",
    )
    parser.add_argument("--beta-alpha", type=float, default=1.5)
    parser.add_argument(
        "--halo-slope",
        type=float,
        default=2.19,
        help="Intermediate density slope for --profile gravothermal_core.",
    )
    parser.add_argument(
        "--transition-radius",
        type=float,
        default=10.0,
        help="Radius where gravothermal_core steepens to its finite-mass outer slope.",
    )
    parser.add_argument(
        "--outer-slope",
        type=float,
        default=5.0,
        help="Asymptotic density slope beyond --transition-radius.",
    )
    parser.add_argument(
        "--transition-order",
        type=float,
        default=4.0,
        help="Sharpness of the gravothermal_core outer-slope transition.",
    )
    parser.add_argument("--king-w0", type=float, default=7.0)
    parser.add_argument(
        "--king-r0",
        type=float,
        default=0.02,
        help="King radius r0 satisfying r0^2=9 sigma_K^2/(4 pi G rho_0).",
    )
    parser.add_argument("--cusp-softening", type=float, default=1.0e-5)
    parser.add_argument("--trunc-power", type=float, default=8.0)
    parser.add_argument("--trunc-order", type=float, default=4.0)
    parser.add_argument(
        "--no-truncation",
        action="store_true",
        help=(
            "Do not taper the density at r_max. Intended for intrinsically "
            "finite-mass profiles such as --profile frank_core."
        ),
    )
    parser.add_argument(
        "--model-r-max",
        type=float,
        default=None,
        help=(
            "Outer radius used for the density, potential, and Eddington tables. "
            "May exceed the solver r_max; useful for untruncated finite-mass profiles."
        ),
    )
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--n-model", type=int, default=50000)
    parser.add_argument("--n-energy", type=int, default=900)
    parser.add_argument("--n-quad", type=int, default=96)
    parser.add_argument("--grid-self-consistent-iters", type=int, default=0)
    parser.add_argument("--grid-self-consistent-damping", type=float, default=0.7)
    parser.add_argument(
        "--clip-negative",
        dest="clip_negative",
        action="store_true",
        default=True,
        help="Clip small negative Eddington-inversion values to zero (default).",
    )
    parser.add_argument(
        "--no-clip-negative",
        dest="clip_negative",
        action="store_false",
        help="Keep raw negative Eddington-inversion values for diagnostics.",
    )
    args = parser.parse_args()

    if args.nr < 4 or args.nw < 8 or args.nl < 8:
        raise ValueError("Need at least nr>=4, nw>=8, nl>=8.")
    if args.w_guard_cells_per_side < 0 or args.L_guard_cells < 0:
        raise ValueError("Velocity guard-cell counts must be non-negative.")
    if args.core_radius <= 0.0:
        raise ValueError("--core-radius/--scale-radius must be positive.")
    if args.profile == "king":
        generate_king_initial_data(args)
        return
    if args.profile == "gravothermal_core":
        if args.halo_slope <= 0.0:
            raise ValueError("--halo-slope must be positive.")
        if args.transition_radius <= args.core_radius:
            raise ValueError("--transition-radius must exceed --core-radius.")
        if args.outer_slope <= max(args.halo_slope, 3.0):
            raise ValueError("--outer-slope must exceed both --halo-slope and 3.")
        if args.transition_order <= 0.0:
            raise ValueError("--transition-order must be positive.")

    apply_truncation = not args.no_truncation
    model_r_max = args.r_max if args.model_r_max is None else args.model_r_max
    if model_r_max < args.r_max:
        raise ValueError("--model-r-max must be at least --r-max.")
    if args.no_truncation and model_r_max <= args.r_max:
        raise ValueError(
            "--no-truncation requires --model-r-max greater than --r-max so "
            "the Eddington table includes the exterior potential."
        )
    if args.no_truncation and args.profile not in ("plummer", "gravothermal_core", "frank_core"):
        raise ValueError(
            "--no-truncation is only supported for profiles with a guaranteed finite mass."
        )
    if args.no_truncation and args.grid_self_consistent_iters > 0:
        raise ValueError(
            "--grid-self-consistent-iters is incompatible with --no-truncation: "
            "the solver-grid wall potential omits the exterior potential used by "
            "the untruncated Eddington model."
        )

    r_model = np.geomspace(min(args.r_min * 1.0e-4, 1.0e-8), model_r_max, args.n_model)
    r_model[-1] = model_r_max
    if args.target_mass is not None:
        unit_density = target_density(
            r_model,
            1.0,
            args.cusp_softening,
            args.r_max,
            args.trunc_power,
            args.trunc_order,
            args.profile,
            args.core_radius,
            args.beta_alpha,
            args.halo_slope,
            args.transition_radius,
            args.outer_slope,
            args.transition_order,
            apply_truncation,
        )
        unit_density[-1] = 0.0
        # Normalize the analytic density on the model domain before computing the self-consistent potential and distribution table.
        unit_mass = cumulative_trapezoid(4.0 * PI * r_model * r_model * unit_density, r_model)[-1]
        if unit_mass <= 0.0:
            raise ValueError("Unit-mass normalization failed; density integral is non-positive.")
        args.rho_norm = float(args.target_mass / unit_mass)

    rho_model = target_density(
        r_model,
        args.rho_norm,
        args.cusp_softening,
        args.r_max,
        args.trunc_power,
        args.trunc_order,
        args.profile,
        args.core_radius,
        args.beta_alpha,
        args.halo_slope,
        args.transition_radius,
        args.outer_slope,
        args.transition_order,
        apply_truncation,
    )
    rho_model[-1] = 0.0
    mass_model = cumulative_trapezoid(4.0 * PI * r_model * r_model * rho_model, r_model)
    psi_model = wall_relative_potential(r_model, mass_model, args.gravity)
    energy, f_energy, ediag = eddington_table(
        psi_model, rho_model, args.n_energy, args.n_quad, args.clip_negative
    )
    f_of_energy = PchipInterpolator(energy, f_energy, extrapolate=False)

    if args.radial_grid == "origin_centered_log":
        r = origin_centered_log_grid(args.r_min, args.r_max, args.nr)
    elif args.radial_grid == "origin_linear_log":
        r = origin_linear_log_grid(args.r_min, args.r_max, args.nr, args.inner_linear_cells)
    else:
        r = np.geomspace(args.r_min, args.r_max, args.nr)
    w = guarded_w_grid(
        args.w_max, args.nw, args.w_stretch, args.w_core_max, args.w_guard_cells_per_side
    )
    L = guarded_l_grid(args.L_max, args.nl, args.L_power, args.L_core_max, args.L_guard_cells)
    psi = np.interp(np.log(r), np.log(r_model), psi_model)
    rho_target = target_density(
        r,
        args.rho_norm,
        args.cusp_softening,
        args.r_max,
        args.trunc_power,
        args.trunc_order,
        args.profile,
        args.core_radius,
        args.beta_alpha,
        args.halo_slope,
        args.transition_radius,
        args.outer_slope,
        args.transition_order,
        apply_truncation,
    )
    rq_grid = weights(r)
    # Keep the requested finite solver-grid mass distinct from the larger model-domain mass used for the potential.
    target_grid_mass_for_relax = float(np.sum(4.0 * PI * r * r * rho_target * rq_grid))

    if args.grid_self_consistent_iters < 0:
        raise ValueError("--grid-self-consistent-iters must be non-negative.")
    if not (0.0 < args.grid_self_consistent_damping <= 1.0):
        raise ValueError("--grid-self-consistent-damping must lie in (0, 1].")

    f = distribution_from_potential(psi, r, w, L, f_of_energy)
    rho_grid = density_from_f(f, r, w, L)
    grid_iter_history = []
    solver_grid_mass = None
    if args.grid_self_consistent_iters > 0:
        def normalize_to_target_grid_mass() -> float:
            """Rescale the sampled distribution to its target grid mass."""
            grid_mass_now = float(np.sum(4.0 * PI * r * r * rho_grid * rq_grid))
            if grid_mass_now > 0.0 and target_grid_mass_for_relax > 0.0:
                scale = target_grid_mass_for_relax / grid_mass_now
                f[:] *= scale
                rho_grid[:] *= scale
                return scale
            return 1.0

        normalize_to_target_grid_mass()
        # Recompute the potential from the sampled grid density, then damp the update before resampling the same energy-distribution table.
        for iteration in range(args.grid_self_consistent_iters):
            psi_new, solver_grid_mass = solver_grid_relative_potential(r, rho_grid, args.gravity)
            rel_delta = np.max(
                np.abs(psi_new - psi) / np.maximum(np.maximum(np.abs(psi), np.abs(psi_new)), 1.0e-300)
            )
            psi = (
                args.grid_self_consistent_damping * psi_new
                + (1.0 - args.grid_self_consistent_damping) * psi
            )
            f = distribution_from_potential(psi, r, w, L, f_of_energy)
            rho_grid = density_from_f(f, r, w, L)
            # Restore the finite-grid target mass after each optional potential/distribution iteration and record the required scale.
            scale = normalize_to_target_grid_mass()
            grid_iter_history.append(
                {
                    "iteration": iteration + 1,
                    "max_relative_psi_update": float(rel_delta),
                    "mass_normalization_scale": float(scale),
                    "represented_grid_mass": float(np.sum(4.0 * PI * r * r * rho_grid * weights(r))),
                    "rho_inner_grid": float(rho_grid[0]),
                }
            )
        psi_new, solver_grid_mass = solver_grid_relative_potential(r, rho_grid, args.gravity)
        psi = psi_new
        f = distribution_from_potential(psi, r, w, L, f_of_energy)
        rho_grid = density_from_f(f, r, w, L)
        normalize_to_target_grid_mass()
    write_ev_ic(args.output, r, w, L, f)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        rq = weights(r)
        model_mass = float(mass_model[-1])
        grid_mass = float(np.sum(4.0 * PI * r * r * rho_grid * rq))
        target_mass = float(np.sum(4.0 * PI * r * r * rho_target * rq))
        rel_err = np.divide(
            rho_grid - rho_target,
            np.maximum(rho_target, 1.0e-300),
            out=np.zeros_like(rho_grid),
        )
        use = rho_target > max(np.max(rho_target) * 1.0e-10, 1.0e-30)
        vesc = np.sqrt(np.maximum(2.0 * psi, 0.0))
        report = {
            "output": str(args.output),
            "Nr": args.nr,
            "Nw": args.nw,
            "NL": args.nl,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "radial_grid": args.radial_grid,
            "inner_linear_cells": args.inner_linear_cells,
            "w_max": args.w_max,
            "w_stretch": args.w_stretch,
            "w_core_max": args.w_core_max,
            "w_guard_cells_per_side": args.w_guard_cells_per_side,
            "L_max": args.L_max,
            "L_power": args.L_power,
            "L_core_max": args.L_core_max,
            "L_guard_cells": args.L_guard_cells,
            "rho_norm": args.rho_norm,
            "target_mass_input": args.target_mass,
            "profile": args.profile,
            "core_radius": args.core_radius,
            "beta_alpha": args.beta_alpha,
            "halo_slope": args.halo_slope,
            "transition_radius": args.transition_radius,
            "outer_slope": args.outer_slope,
            "transition_order": args.transition_order,
            "cusp_softening": args.cusp_softening,
            "trunc_power": args.trunc_power,
            "trunc_order": args.trunc_order,
            "apply_truncation": apply_truncation,
            "model_r_max": model_r_max,
            "grid_self_consistent_iters": args.grid_self_consistent_iters,
            "grid_self_consistent_damping": args.grid_self_consistent_damping,
            "grid_self_consistent_history": grid_iter_history,
            "taper_at_half_box": (
                float(
                    truncation(
                        np.array([0.5 * args.r_max]),
                        args.r_max,
                        args.trunc_power,
                        args.trunc_order,
                    )[0]
                )
                if apply_truncation
                else 1.0
            ),
            "model_mass": model_mass,
            "target_grid_mass": target_mass,
            "represented_grid_mass": grid_mass,
            "solver_grid_mass_outer": float(solver_grid_mass[-1]) if solver_grid_mass is not None else None,
            "rho_inner_target": float(rho_target[0]),
            "rho_inner_grid": float(rho_grid[0]),
            "rho_outer_target": float(rho_target[-1]),
            "rho_outer_grid": float(rho_grid[-1]),
            "psi_inner": float(psi[0]),
            "psi_max_model": ediag["psi_max"],
            "v_escape_max_on_grid": float(np.max(vesc)),
            "max_r_times_vesc": float(np.max(r * vesc)),
            "L_first_nonzero": float(L[1]),
            "w_first_positive": float(w[np.searchsorted(w, 0.0)]),
            "max_abs_rel_density_error_used": float(np.max(np.abs(rel_err[use]))) if np.any(use) else 0.0,
            "median_abs_rel_density_error_used": float(np.median(np.abs(rel_err[use]))) if np.any(use) else 0.0,
            **ediag,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
