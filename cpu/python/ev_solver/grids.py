# Construct the reference solver's r,w,L nodes and integration weights.
# r is areal radius, w radial momentum per unit rest mass, and L specific angular momentum.
# Return nodal arrays and minimum coordinate spacings for interpolation and CFL selection.

"""Grid construction and one-dimensional quadrature helpers.

The Python solver mirrors the C++ grid conventions closely enough that its
restart files, binary initial data, and plotting outputs remain compatible with
the VENUS CPU examples.
"""

from __future__ import annotations

import math

import numpy as np

from .params import EVParams


# Section: quadrature and one-dimensional grids.

# Computes trapezoidal weights for a possibly nonuniform one-dimensional grid.
def quadrature_weights(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        raise ValueError("Need at least two grid points for trapezoid quadrature")
    weights = np.zeros_like(x, dtype=float)
    # Each endpoint receives half its adjacent interval. Interior nodes receive halves from both sides.
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


# Builds a sinh-stretched momentum grid concentrated near the chosen center.
def stretched_w_grid(w_max: float, stretch: float, center: float, n: int) -> np.ndarray:
    if stretch <= 0.0:
        return np.linspace(-w_max, w_max, n)
    # Map a uniform auxiliary coordinate through sinh to concentrate momentum nodes near center.
    u = np.linspace(-1.0, 1.0, n)
    denom = math.sinh(stretch)
    left = center - (center + w_max) * np.sinh(stretch * (-u)) / denom
    right = center + (w_max - center) * np.sinh(stretch * u) / denom
    return np.where(u < 0.0, left, right)


# Section: restart metadata helpers.

# Encodes a radial-grid name for restart-file metadata.
def radial_grid_code(name: str) -> int:
    if name == "log":
        return 0
    if name == "linear":
        return 1
    if name == "piecewise-linear":
        return 2
    if name == "custom":
        return 3
    raise ValueError("radial_grid must be 'log', 'linear', 'piecewise-linear', or 'custom'")


# Decodes a restart-file radial-grid integer into a name.
def radial_grid_name(code: int) -> str:
    if code == 0:
        return "log"
    if code == 1:
        return "linear"
    if code == 2:
        return "piecewise-linear"
    if code == 3:
        return "custom"
    raise ValueError("Restart file has unknown radial grid code")


# Returns whether the params request binary EVICRWL1 initial data.
def using_binary_initial_data(par: EVParams) -> bool:
    return par.initial_data in {"binary_rwl", "binary", "file"}


# Section: radial coordinate and grid construction.

# Converts physical radius to the coordinate used for interpolation and CFL estimates.
def radial_coordinate(r: np.ndarray | float, par: EVParams) -> np.ndarray | float:
    if par.radial_grid == "log" or (par.radial_grid == "custom" and par.custom_radial_coordinate == "log"):
        return np.log(r)
    return r


# Grid construction.  The piecewise-linear option is useful for long runs
# because it refines the inner region without making r_min set the whole CFL.
# Builds a radial grid with explicit inner refinement and a coarser outer segment.
def piecewise_linear_grid(par: EVParams) -> np.ndarray:
    split = par.inner_refine_radius
    inner_cells = min(max(int(par.inner_refine_cells), 1), par.Nr - 2)
    # Include the split in the fine segment, then omit its duplicate from the coarse segment.
    inner = np.linspace(par.r_min, split, inner_cells + 1)
    outer_cells = par.Nr - inner.size
    if outer_cells < 1:
        raise ValueError("piecewise-linear grid needs at least one outer cell")
    outer = np.linspace(split, par.r_max, outer_cells + 1)[1:]
    return np.concatenate([inner, outer])


# Validates analytic-grid parameters and returns r, w, L plus minimum coordinate spacings.
def make_grids(par: EVParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Validate the analytic-grid setup and construct its three coordinates.

    Radius uses the selected logarithmic, linear, or two-segment grid;
    radial momentum w can use sinh stretching; L is specific angular
    momentum. Return r, w, L and the smallest interpolation-coordinate
    and momentum spacings used by the characteristic timestep bound.
    Binary initial data supply their own coordinates through initial_data.
    """
    if par.r_min <= 0.0:
        raise ValueError("r_min must be positive")
    if par.r_max <= par.r_min:
        raise ValueError("r_max must be larger than r_min")
    radial_grid_code(par.radial_grid)
    if par.radial_grid == "custom":
        raise ValueError("A custom radial grid must be loaded from initial_data_file")
    if par.Nr < 3 or par.Nw < 3 or par.NL < 2:
        raise ValueError("Need Nr >= 3, Nw >= 3, NL >= 2")
    if par.radial_grid == "piecewise-linear":
        if not (par.r_min < par.inner_refine_radius < par.r_max):
            raise ValueError("Require r_min < inner_refine_radius < r_max for piecewise-linear grid")
        if not (1 <= par.inner_refine_cells <= par.Nr - 2):
            raise ValueError("Require 1 <= inner_refine_cells <= Nr - 2 for piecewise-linear grid")
    if par.w_max <= 0.0 or par.L_max <= 0.0:
        raise ValueError("w_max and L_max must be positive")
    if par.w_stretch < 0.0:
        raise ValueError("w_stretch must be non-negative")
    if not (-par.w_max < par.w_center < par.w_max):
        raise ValueError("Require -w_max < w_center < w_max")
    if abs(par.w_center) > 1.0e-12:
        print(
            "Warning: the inner reflecting wall mirrors w about 0, but "
            "--w-center is nonzero. This can under-resolve the reflected "
            "outgoing branch. Prefer --w-center 0 for reflecting-center runs."
        )
    # A prescribed central mass requires the innermost node to lie outside its Schwarzschild radius.
    if par.central_mass > 0.0 and par.r_min <= 2.05 * par.central_mass:
        raise ValueError(
            "For a nonzero central mass use r_min safely outside 2M, or leave "
            "central_mass=0 for the reflecting regular-center run."
        )
    if par.shell_initial_df and par.R0 <= 0.0:
        raise ValueError("Require R0 > 0 for shell initial data")
    if par.shell_initial_df and par.dR <= 0.0:
        raise ValueError("Require dR > 0 for shell initial data")
    if par.shell_initial_df and not (par.r_min <= par.R0 <= par.r_max):
        raise ValueError("Shell center R0 must lie inside the radial grid")
    # The shell flags prescribe alternative radial envelopes and therefore are mutually exclusive.
    shell_mode_count = sum(
        bool(x) for x in (par.top_hat_shell, par.top_hat_density_shell, par.gaussian_density_shell)
    )
    if shell_mode_count > 1:
        raise ValueError(
            "Choose only one shell radial mode: --top-hat-shell, "
            "--top-hat-density-shell, or --gaussian-density-shell"
        )
    if (par.top_hat_shell or par.top_hat_density_shell) and (
        par.R0 - 0.5 * par.dR < par.r_min or par.R0 + 0.5 * par.dR > par.r_max
    ):
        print(
            "Warning: top-hat shell support extends outside the radial grid; "
            "the initial shell will be truncated."
        )
    if par.beta <= 0.0:
        raise ValueError("Require beta > 0 for the boundary-seed initial DF")
    if par.amp < 0.0:
        raise ValueError("Require amp >= 0 for the initial DF")
    if not (-par.w_max < par.w_in < par.w_max):
        raise ValueError("The initial DF velocity center w_in must lie inside the w grid")
    if par.t_end < 0.0 or par.dt_max <= 0.0 or par.cfl <= 0.0:
        raise ValueError("Require t_end >= 0, dt_max > 0, and cfl > 0")
    if par.integrator != "semi_lagrangian_kick_drift_kick":
        raise ValueError("Only integrator=semi_lagrangian_kick_drift_kick is implemented")
    if par.inner_boundary != "reflecting":
        raise ValueError("Only inner_boundary=reflecting is implemented")
    if par.outer_boundary != "outflow_vacuum":
        raise ValueError("Only outer_boundary=outflow_vacuum is implemented")
    if par.momentum_boundary != "outflow_vacuum":
        raise ValueError("Only momentum_boundary=outflow_vacuum is implemented")
    if par.max_steps < 0:
        raise ValueError("Require max_steps >= 0")
    if par.metric_floor <= 0.0 or par.gap_stop <= 0.0:
        raise ValueError("Require metric_floor > 0 and gap_stop > 0")
    if par.active_floor_rel < 0.0 or par.active_floor_abs < 0.0:
        raise ValueError("Active-cell floors must be non-negative")

    # Construct physical nodes first. The radial interpolation coordinate sets the reported CFL spacing.
    if par.radial_grid == "log":
        r = np.geomspace(par.r_min, par.r_max, par.Nr)
    elif par.radial_grid == "linear":
        r = np.linspace(par.r_min, par.r_max, par.Nr)
    else:
        r = piecewise_linear_grid(par)
    w = stretched_w_grid(par.w_max, par.w_stretch, par.w_center, par.Nw)
    L = np.linspace(0.0, par.L_max, par.NL)
    x_r = np.asarray(radial_coordinate(r, par))
    return r, w, L, float(np.min(np.diff(x_r))), float(np.min(np.diff(w)))
