# Reconstruct the polar-areal Einstein metric from energy density, radial pressure and current.
# Use G=c=1. Return enclosed mass, lapse, radial metric factor and their characteristic derivatives.

"""Metric constraint reconstruction for the Python EV reference solver.

Sections implement cumulative radial integration and the polar-areal Einstein
constraints for ``m``, ``lambda``, ``mu``, and their characteristic
derivatives.  The formulas mirror ``include/ev/metric.hpp``.
"""

from __future__ import annotations

import math

import numpy as np

from .params import EVParams, PI


# Section: radial integration helper.

# Forms a cumulative trapezoidal integral on a one-dimensional grid.
def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


# Section: Einstein constraint solve.

# Solves the spherical Einstein constraint equations from the radial matter profiles.
def solve_metric(
    rho: np.ndarray,
    p: np.ndarray,
    j: np.ndarray,
    r: np.ndarray,
    par: EVParams,
) -> dict[str, np.ndarray | float]:
    """Integrate the spherical Einstein constraints from rho, p and j.

    rho is energy density, p radial pressure, and j radial current, in
    G=c=1 units. Accumulate enclosed mass with trapezoidal quadrature,
    adding the specified central mass or a uniform innermost matter ball.
    Compute lambda from 1-2m/r, integrate mu inward from its exterior
    Schwarzschild value, then form the radial derivatives and lambda_t.
    The returned raw compactness gap controls stopping; metric_floor
    bounds denominators during the constraint calculation.
    """
    # Integrating 4*pi*r^2*rho gives the gravitational matter mass inside each radial node.
    m_shell = 4.0 * PI * cumulative_trapezoid(r * r * rho, r)
    inner_matter_mass = 0.0
    # For the regular-center model, approximate unresolved inner matter by a uniform ball.
    if par.central_mass <= 0.0:
        inner_matter_mass = (4.0 * PI / 3.0) * rho[0] * r[0] ** 3
    m = par.central_mass + inner_matter_mass + m_shell

    # Keep the physical gap 1-2m/r for stopping. A separate floor protects metric denominators.
    gap_raw = 1.0 - 2.0 * m / r
    gap = np.maximum(gap_raw, par.metric_floor)
    lam = -0.5 * np.log(gap)

    lam_r = (4.0 * PI * r * rho - m / (r * r)) / gap
    mu_r = (m / (r * r) + 4.0 * PI * r * p) / gap

    mu = np.zeros_like(r)
    outer_gap = max(1.0 - 2.0 * m[-1] / r[-1], par.metric_floor)
    # Match the outer lapse to the vacuum Schwarzschild field, then integrate its radial gradient inward.
    mu[-1] = 0.5 * math.log(outer_gap)
    for i in range(len(r) - 2, -1, -1):
        dr = r[i + 1] - r[i]
        mu[i] = mu[i + 1] - 0.5 * (mu_r[i + 1] + mu_r[i]) * dr

    # The radial energy current determines the time derivative of lambda through the Einstein constraint.
    lam_t = -4.0 * PI * r * np.exp(lam + mu) * j
    return {
        "m": m,
        "lambda": lam,
        "mu": mu,
        "lambda_r": lam_r,
        "mu_r": mu_r,
        "lambda_t": lam_t,
        "gap": gap_raw,
        "min_gap": float(np.min(gap_raw)),
    }
