# Trace Einstein--Vlasov characteristics and interpolate the reference distribution.
# Inputs are nodal f(r,w,L), a solved metric and code-unit spacings. Output is a new f and diagnostics.
# The metric is held fixed within each kick-drift-kick pullback.

"""Numerical kernels for the Python Einstein--Vlasov reference solver.

The routines here implement timestep selection, characteristic pullback,
interpolation, boundary handling, phase projection, and escape diagnostics.
File I/O lives in ev_solver.io.
"""

from __future__ import annotations

import numpy as np

from .grids import radial_coordinate
from .params import EVParams, PI


# Section: timestep selection.

# Chooses a timestep from active characteristic speeds and CFL limits.
def choose_dt(
    f: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    mu_r: np.ndarray,
    lam_t: np.ndarray,
    radial_step: float,
    min_dw: float,
    par: EVParams,
) -> float:
    max_f = float(np.max(f))
    # Only occupied nodes above the configured activity floor enter the characteristic CFL bound.
    threshold = max(par.active_floor_abs, par.active_floor_rel * max_f)
    active = np.nonzero(f > threshold)
    if active[0].size == 0:
        return par.dt_max

    ii, jj, kk = active
    rr = r[ii]
    ww = w[jj]
    LL = L[kk]
    eps = np.sqrt(1.0 + ww * ww + (LL / rr) ** 2)
    fac = np.exp(mu[ii] - lam[ii])

    v_r = fac * ww / eps
    v_w = -lam_t[ii] * ww - fac * mu_r[ii] * eps + fac * LL * LL / (rr**3 * eps)

    # For a logarithmic radial grid, dr/dt becomes d(log r)/dt=(dr/dt)/r.
    v_r_coord = v_r / rr if par.radial_grid == "log" else v_r
    max_vr_coord = max(float(np.max(np.abs(v_r_coord))), 1.0e-14)
    max_vw = max(float(np.max(np.abs(v_w))), 1.0e-14)
    return min(par.dt_max, par.cfl * radial_step / max_vr_coord, par.cfl * min_dw / max_vw)


# Section: radial boundary handling.

# Reflects characteristic footpoints that cross the finite inner wall.
def reflect_inner(r_dep: np.ndarray, w_dep: np.ndarray, r_min: float) -> None:
    mask = r_dep < r_min
    if not np.any(mask):
        return
    # Mirror radius and reverse radial momentum together when tracing through the reflecting inner wall.
    r_dep[mask] = r_min + (r_min - r_dep[mask])
    w_dep[mask] = -w_dep[mask]


# Enforce f(r_min,w,L)=f(r_min,-w,L) after interpolation.  This keeps the small
# finite inner wall acting like a regular spherical center.
# Symmetrizes the r_min slice so the wall mimics regular-center reflection.
def enforce_inner_wall_symmetry(f: np.ndarray, w: np.ndarray) -> None:
    boundary = np.array(f[0, :, :], copy=True)
    for k in range(f.shape[2]):
        mirrored = np.interp(-w, w, boundary[:, k], left=0.0, right=0.0)
        f[0, :, k] = 0.5 * (boundary[:, k] + mirrored)


# Section: interpolation and metric sampling.

# Bilinearly interpolates one fixed-L f slice at off-grid radius and momentum points.
def bilinear_radial_grid(
    values: np.ndarray,
    r_dep: np.ndarray,
    w_dep: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    par: EVParams,
) -> np.ndarray:
    """Interpolate one fixed-L distribution slice at departure points.

    Find the two neighboring radial-coordinate and w nodes, then combine
    their four values with bilinear weights. The radial coordinate is
    log(r) or r according to par. Points outside either supplied grid
    receive the vacuum value zero; inner reflection is applied before
    this routine is called.
    """
    Nr, Nw = values.shape
    out = np.zeros_like(r_dep)

    mask = (r_dep >= r[0]) & (r_dep <= r[-1]) & (w_dep >= w[0]) & (w_dep <= w[-1])
    if not np.any(mask):
        return out

    x_grid = np.asarray(radial_coordinate(r, par))
    xr = np.asarray(radial_coordinate(r_dep[mask], par))
    xw = w_dep[mask]

    ir = np.searchsorted(x_grid, xr, side="right") - 1
    iw = np.searchsorted(w, xw, side="right") - 1
    ir = np.clip(ir, 0, Nr - 2)
    iw = np.clip(iw, 0, Nw - 2)

    # Measure each departure point within its radial/momentum cell to form the four interpolation weights.
    tr = (xr - x_grid[ir]) / (x_grid[ir + 1] - x_grid[ir])
    tw = (xw - w[iw]) / (w[iw + 1] - w[iw])

    f00 = values[ir, iw]
    f10 = values[ir + 1, iw]
    f01 = values[ir, iw + 1]
    f11 = values[ir + 1, iw + 1]
    out[mask] = (
        (1.0 - tr) * (1.0 - tw) * f00
        + tr * (1.0 - tw) * f10
        + (1.0 - tr) * tw * f01
        + tr * tw * f11
    )
    return out


# Interpolates a radial profile using the selected radial coordinate.
def interp_profile_radial(
    values: np.ndarray,
    r_pos: np.ndarray,
    r_grid: np.ndarray,
    par: EVParams,
) -> np.ndarray:
    x_grid = np.asarray(radial_coordinate(r_grid, par))
    x = np.asarray(radial_coordinate(np.clip(r_pos, r_grid[0], r_grid[-1]), par))
    return np.interp(x.ravel(), x_grid, values).reshape(r_pos.shape)


# Samples all metric fields needed for characteristic tracing at requested radii.
def sample_metric(
    r_pos: np.ndarray,
    r_grid: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    mu_r: np.ndarray,
    lam_t: np.ndarray,
    par: EVParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr = np.clip(r_pos, r_grid[0], r_grid[-1])
    return (
        interp_profile_radial(mu, rr, r_grid, par),
        interp_profile_radial(lam, rr, r_grid, par),
        interp_profile_radial(mu_r, rr, r_grid, par),
        interp_profile_radial(lam_t, rr, r_grid, par),
    )


# Computes dr/dt for characteristics in the frozen metric.
def radial_velocity(
    r_pos: np.ndarray,
    w_pos: np.ndarray,
    Lk: float,
    r_grid: np.ndarray,
    mu_i: np.ndarray,
    lam_i: np.ndarray,
) -> np.ndarray:
    rr = np.clip(r_pos, r_grid[0], r_grid[-1])
    eps = np.sqrt(1.0 + w_pos * w_pos + (Lk / rr) ** 2)
    fac = np.exp(mu_i - lam_i)
    return fac * w_pos / eps


# Computes dw/dt for characteristics in the frozen metric.
def radial_acceleration(
    r_pos: np.ndarray,
    w_pos: np.ndarray,
    Lk: float,
    r_grid: np.ndarray,
    mu_i: np.ndarray,
    lam_i: np.ndarray,
    mu_r_i: np.ndarray,
    lam_t_i: np.ndarray,
) -> np.ndarray:
    rr = np.clip(r_pos, r_grid[0], r_grid[-1])
    eps = np.sqrt(1.0 + w_pos * w_pos + (Lk / rr) ** 2)
    fac = np.exp(mu_i - lam_i)
    # The three terms are time-dependent metric drag, gravitational acceleration and the centrifugal term.
    return -lam_t_i * w_pos - fac * mu_r_i * eps + fac * Lk * Lk / (rr**3 * eps)


# Section: semi-Lagrangian transport.

# Advances the distribution function by one semi-Lagrangian kick-drift-kick step.
def advect_f(
    f: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    mu_r: np.ndarray,
    lam_t: np.ndarray,
    dt: float,
    par: EVParams,
) -> np.ndarray:
    # Semi-Lagrangian kick-drift-kick with the metric frozen over one timestep.
    """Trace each arrival node backward through the frozen metric.

    At fixed L, undo a half momentum kick, a radial drift, and a second
    half kick. Reflect inner-wall crossings together with w, sample the
    metric at the departure radius, and interpolate the old distribution.
    Finally symmetrize the inner nodal slice and apply the configured
    distribution floor. This nodal pullback returns a new f array.
    """
    Nr, Nw, NL = f.shape
    f_new = np.zeros_like(f)
    RR = r[:, None]
    WW = w[None, :]

    for k, Lk in enumerate(L):
        mu_t, lam_tgt, mu_r_t, lam_tt = sample_metric(RR, r, mu, lam, mu_r, lam_t, par)
        a_target = radial_acceleration(RR, WW, float(Lk), r, mu_t, lam_tgt, mu_r_t, lam_tt)
        # Undo the last half kick before the drift because the distribution is traced backward from arrival nodes.
        w_half = WW - 0.5 * dt * a_target
        v_half = radial_velocity(RR, w_half, float(Lk), r, mu_t, lam_tgt)

        r_dep = RR - dt * v_half
        w_mid = np.array(w_half, copy=True)
        r_dep = np.array(r_dep, copy=True)
        reflect_inner(r_dep, w_mid, par.r_min)
        # Sample the metric at the reflected departure radius for the remaining half kick.
        mu_d, lam_d, mu_r_d, lam_t_d = sample_metric(r_dep, r, mu, lam, mu_r, lam_t, par)
        w_dep = w_mid - 0.5 * dt * radial_acceleration(
            r_dep, w_mid, float(Lk), r, mu_d, lam_d, mu_r_d, lam_t_d
        )
        r_dep = np.array(r_dep, copy=True)
        w_dep = np.array(w_dep, copy=True)
        reflect_inner(r_dep, w_dep, par.r_min)
        f_new[:, :, k] = bilinear_radial_grid(f[:, :, k], r_dep, w_dep, r, w, par)

    # Restore the even-in-w inner boundary after interpolation has filled the new array.
    enforce_inner_wall_symmetry(f_new, w)

    if par.f_floor is not None:
        np.maximum(f_new, par.f_floor, out=f_new)
    return f_new


# Section: diagnostics.

# Integrates L f over L to form the two-dimensional g(r,w) projection.
def phase_space_projection(f: np.ndarray, L: np.ndarray, L_weights: np.ndarray) -> np.ndarray:
    return np.einsum("ijk,k,k->ij", f, L, L_weights, optimize=True)


# Estimates mass that will cross outer r or w boundaries during the next step.
def estimate_escapes(
    f: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    r_weights: np.ndarray,
    w_weights: np.ndarray,
    L_weights: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    mu_r: np.ndarray,
    lam_t: np.ndarray,
    dt: float,
    par: EVParams,
) -> dict[str, float | int]:
    # These are forward characteristic estimates, not exact conservative fluxes.
    """Estimate energy-weighted matter crossing finite grid boundaries.

    Select occupied nodes using the configured floor, take one forward
    characteristic estimate, and reflect inner-wall crossings. Integrate
    8*pi^2*epsilon*L*f*dr*dw*dL over nodes predicted to leave the outer
    radius or either w boundary. The combined total counts their union
    once; individual boundary estimates can overlap. These are diagnostic
    forward estimates rather than fluxes of a conservative transport scheme.
    """
    max_f = float(np.max(f))
    threshold = max(par.active_floor_abs, par.active_floor_rel * max_f)
    active = np.nonzero(f > threshold)
    if active[0].size == 0:
        return {
            "outer_mass": 0.0,
            "w_low_mass": 0.0,
            "w_high_mass": 0.0,
            "total_mass": 0.0,
            "outer_cells": 0,
            "w_low_cells": 0,
            "w_high_cells": 0,
        }

    ii, jj, kk = active
    rr = r[ii]
    ww = w[jj]
    LL = L[kk]
    eps = np.sqrt(1.0 + ww * ww + (LL / rr) ** 2)
    fac = np.exp(mu[ii] - lam[ii])
    drdt = fac * ww / eps
    dwdt = -lam_t[ii] * ww - fac * mu_r[ii] * eps + fac * LL * LL / (rr**3 * eps)

    r_forw = rr + dt * drdt
    w_forw = ww + dt * dwdt
    reflect_inner(r_forw, w_forw, par.r_min)

    out_outer = r_forw > r[-1]
    out_w_low = w_forw < w[0]
    out_w_high = w_forw > w[-1]
    # Use the union of boundary-crossing masks for the total so one node contributes once.
    out_any = out_outer | out_w_low | out_w_high

    # Integrate the diagnostic with 4*pi*r^2 times the spherical momentum measure; the r^2 factors cancel.
    cell_mass = (
        8.0
        * PI
        * PI
        * eps
        * LL
        * f[active]
        * r_weights[ii]
        * w_weights[jj]
        * L_weights[kk]
    )
    return {
        "outer_mass": float(np.sum(cell_mass[out_outer])),
        "w_low_mass": float(np.sum(cell_mass[out_w_low])),
        "w_high_mass": float(np.sum(cell_mass[out_w_high])),
        "total_mass": float(np.sum(cell_mass[out_any])),
        "outer_cells": int(np.count_nonzero(out_outer)),
        "w_low_cells": int(np.count_nonzero(out_w_low)),
        "w_high_cells": int(np.count_nonzero(out_w_high)),
    }
