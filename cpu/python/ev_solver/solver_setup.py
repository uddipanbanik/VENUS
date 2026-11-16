# Validate a reference configuration and construct its initial SimulationState.
# Inputs are analytic parameters, EVICRWL1 data or a compatible EVLOGRST1 restart.
# Return one distribution with matching grids, integration weights, clock and output streams.

"""Simulation setup for the Python Einstein--Vlasov reference solver.

Sections construct grids and quadrature, load/generate the distribution,
restore a compatible restart when requested, create output directories and
CSV headers, and return one fully owned ``SimulationState``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .grids import make_grids, quadrature_weights, using_binary_initial_data
from .initial_data import grids_from_binary_initial_data, initial_distribution
from .io import read_restart_state, write_restart_state
from .metric import solve_metric
from .moments import compute_moments
from .numerics import phase_space_projection
from .params import EVParams
from .simulation_state import SimulationState


# Section: option validation.

# Rejects parameter choices that the Python reference solver does not implement.
def validate_python_solver_options(par: EVParams) -> None:
    if par.physics_model != "einstein_vlasov":
        raise ValueError("The Python reference solver only implements physics_model='einstein_vlasov'; use the C++ solver for vlasov_poisson.")
    if par.initial_data != "analytic" and not using_binary_initial_data(par):
        raise ValueError("initial_data must be 'analytic' or 'binary_rwl'")
    # The Python advection operates on a fixed momentum grid, so adaptive remapping belongs to the C++ path.
    if par.adaptive_w:
        raise ValueError("adaptive_w is available only in the C++ solver; set adaptive_w=false for Python")
    if par.integrator != "semi_lagrangian_kick_drift_kick":
        raise ValueError(
            "The Python reference solver only implements integrator='semi_lagrangian_kick_drift_kick'; "
            "use the C++ modular solver for RK or Adams-Bashforth characteristics."
        )
    if par.time_integrator_order != 2 or par.time_integrator_substeps != 1:
        raise ValueError(
            "The Python reference solver implements time_integrator_order=2 "
            "and time_integrator_substeps=1."
        )
    # Binary ICs supply their own grids and bypass make_grids(). Apply the
    # implemented boundary contract here for every initial-data path.
    for name, supported in (
        ("inner_boundary", "reflecting"),
        ("outer_boundary", "outflow_vacuum"),
        ("momentum_boundary", "outflow_vacuum"),
    ):
        if getattr(par, name) != supported:
            raise ValueError(f"The Python reference solver requires {name}='{supported}'.")
    if par.double_gaussian_density_shell:
        raise ValueError("double_gaussian_density_shell is available only in the C++ solver")
    if par.custom_ic_normalize_mass and par.custom_ic_target_mass <= 0.0:
        raise ValueError("custom_ic_target_mass must be positive when normalization is enabled")


# Section: initial distribution construction.

# Builds or restores the initial f array before the Python evolution loop starts.
def build_initial_distribution(
    par: EVParams,
    r: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    w_weights: np.ndarray,
    L_weights: np.ndarray,
    custom_f: np.ndarray | None,
) -> tuple[float, int, np.ndarray]:
    """Choose a saved state, a supplied grid distribution, or analytic data.

    A restart restores its accepted time, step and f. A supplied f is
    copied and optionally normalized by its integrated gravitational
    matter mass. Analytic data alternate distribution construction and
    the Einstein constraint solve for initial_metric_iters iterations.
    Return time, step and the initialized distribution.
    """
    if par.restart_from:
        t, step, f = read_restart_state(Path(par.restart_from), par)
        print(f"Restarted from {par.restart_from} at step={step} t={t:.17e}")
        return t, step, f

    if custom_f is not None:
        f = np.array(custom_f, copy=True)
        if par.custom_ic_normalize_mass:
            rho0, p0, j0 = compute_moments(f, r, w, L, w_weights, L_weights)
            metric0 = solve_metric(rho0, p0, j0, r, par)
            matter_mass = float(np.asarray(metric0["m"])[-1] - par.central_mass)
            if not (matter_mass > 0.0 and math.isfinite(matter_mass)):
                raise RuntimeError("Cannot normalize custom IC: computed matter mass is not positive")
            # Scale the supplied distribution by the ratio of target to integrated gravitational matter mass.
            f *= par.custom_ic_target_mass / matter_mass
            print(
                f"Normalized custom initial data mass from {matter_mass:.12e} "
                f"to {par.custom_ic_target_mass:.12e}"
            )
        return 0.0, 0, f

    mu_seed = np.zeros_like(r)
    f = initial_distribution(r, w, L, w_weights, L_weights, mu_seed, par)
    # Re-evaluate the lapse-dependent seed after each constraint solve to couple the initial matter and metric.
    for _ in range(par.initial_metric_iters):
        rho0, p0, j0 = compute_moments(f, r, w, L, w_weights, L_weights)
        metric0 = solve_metric(rho0, p0, j0, r, par)
        f = initial_distribution(r, w, L, w_weights, L_weights, np.asarray(metric0["mu"]), par)
    return 0.0, 0, f


# Section: full simulation state preparation.

# Constructs grids, weights, initial f, output directory, restart, and snapshot writer.
def prepare_simulation(par: EVParams) -> SimulationState:
    """Construct the state consumed by the Python evolution loop.

    Validate the supported equation and boundaries, read or build grids,
    form quadrature weights, initialize f, and write the initial restart.
    Store these arrays and output settings in SimulationState. Open the
    optional phase table with its explicit column header before evolution.
    """
    validate_python_solver_options(par)
    custom_f: np.ndarray | None = None
    if using_binary_initial_data(par):
        r, w, L, custom_f, radial_step, min_dw = grids_from_binary_initial_data(par)
    else:
        r, w, L, radial_step, min_dw = make_grids(par)

    # Build integration weights from the actual imported or constructed coordinates.
    r_weights = quadrature_weights(r)
    w_weights = quadrature_weights(w)
    L_weights = quadrature_weights(L)
    outdir = Path(par.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t, step, f = build_initial_distribution(par, r, w, L, w_weights, L_weights, custom_f)
    # Save the initialized distribution and its clock before starting the evolution.
    write_restart_state(outdir / "restart_state.bin", par, t, step, f)

    state = SimulationState(
        par=par,
        r=r,
        w=w,
        L=L,
        f=f,
        r_weights=r_weights,
        w_weights=w_weights,
        L_weights=L_weights,
        radial_step=radial_step,
        min_dw=min_dw,
        outdir=outdir,
        t=t,
        step=step,
        f_initial_projection=phase_space_projection(f, L, L_weights),
    )

    # Open the optional projected-phase table once. The evolution appends complete accepted frames.
    if par.phase_every > 0:
        state.phase_snapshot_handle = (outdir / "phase_space_snapshots.csv").open("w", newline="")
        state.phase_snapshot_writer = csv.writer(state.phase_snapshot_handle)
        state.phase_snapshot_writer.writerow(["time", "step", "r_index", "w_index", "r", "w", "g"])
    return state
