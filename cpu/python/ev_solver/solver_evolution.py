# Advance the reference distribution and record its matter/metric diagnostics.
# Each accepted clock owns one reconstructed field. Output/restart cadences are step counts in this solver.

"""Timestep loop for the Python Einstein--Vlasov reference solver.

Each loop reconstructs moments/metric, checks output and compactness stopping,
selects the characteristic timestep, estimates boundary loss, performs nodal
semi-Lagrangian pullback on the fixed velocity grid, and writes cadenced
profiles/restarts.
"""

from __future__ import annotations

import math

import numpy as np

from .io import write_phase_snapshot_rows, write_restart_state
from .metric import solve_metric
from .moments import compute_moments
from .numerics import advect_f, choose_dt, estimate_escapes, phase_space_projection
from .simulation_state import SimulationState


# Section: saved diagnostics.

# Stores profiles, time-series rows, and optional phase snapshots for one saved time.
def save_current_diagnostics(
    state: SimulationState,
    rho: np.ndarray,
    p: np.ndarray,
    j: np.ndarray,
    metric: dict[str, np.ndarray | float],
) -> None:
    """Record moments and metric at the current accepted time.

    Copy radial profiles at save_every steps and write the angular-momentum
    projection at phase_every steps. Append one history row on every call,
    including the most recent timestep and accumulated escape estimates.
    The final accepted time is included in each enabled output family.
    """
    par = state.par
    # Subtract the prescribed central mass when reporting the gravitational matter contribution.
    mass = float(metric["m"][-1] - par.central_mass)  # type: ignore[index]
    max_rho = float(np.max(rho))
    min_gap = float(metric["min_gap"])

    if state.step % max(1, par.save_every) == 0 or state.t >= par.t_end:
        state.snapshots.append(
            {
                "step": state.step,
                "time": state.t,
                "rho": rho.copy(),
                "p": p.copy(),
                "j": j.copy(),
                "m": np.array(metric["m"], copy=True),
                "mu": np.array(metric["mu"], copy=True),
                "lambda": np.array(metric["lambda"], copy=True),
                "gap": np.array(metric["gap"], copy=True),
            }
        )
    if state.phase_snapshot_writer is not None and (state.step % par.phase_every == 0 or state.t >= par.t_end):
        write_phase_snapshot_rows(
            state.phase_snapshot_writer,
            state.t,
            state.step,
            state.r,
            state.w,
            phase_space_projection(state.f, state.L, state.L_weights),
            par,
        )

    state.history.append(
        {
            "step": state.step,
            "time": state.t,
            "matter_mass": mass,
            "max_rho": max_rho,
            "min_gap": min_gap,
            "dt": state.last_dt,
            "last_outer_escape_est": float(state.last_escape["outer_mass"]),
            "last_w_low_escape_est": float(state.last_escape["w_low_mass"]),
            "last_w_high_escape_est": float(state.last_escape["w_high_mass"]),
            "cum_outer_escape_est": state.cumulative_escape["outer_mass"],
            "cum_w_low_escape_est": state.cumulative_escape["w_low_mass"],
            "cum_w_high_escape_est": state.cumulative_escape["w_high_mass"],
            "last_outer_cells": int(state.last_escape["outer_cells"]),
            "last_w_low_cells": int(state.last_escape["w_low_cells"]),
            "last_w_high_cells": int(state.last_escape["w_high_cells"]),
        }
    )


# Section: stopping and advancement.

# Applies time, compactness-gap, and max-step stopping criteria.
def stop_requested(state: SimulationState, min_gap: float) -> bool:
    par = state.par
    if state.t >= par.t_end:
        return True
    if min_gap <= par.gap_stop:
        print(f"Stopping: compactness gap reached {min_gap:.3e} at t={state.t:.6e}.")
        return True
    if state.step >= par.max_steps:
        print(f"Stopping: reached max_steps={par.max_steps}.")
        return True
    return False


# Performs CFL selection, escape estimates, semi-Lagrangian advection, and restart cadence.
def advance_one_step(state: SimulationState, metric: dict[str, np.ndarray | float]) -> float:
    """Advance f and its accepted clock using the supplied metric.

    Select the occupied-characteristic CFL step and shorten it at t_end.
    Estimate boundary losses, apply the frozen-metric nodal pullback,
    increment time and step, and save a periodic native restart. Moments
    and metric are reconstructed by run_evolution before the next step.
    """
    par = state.par
    dt = choose_dt(
        state.f,
        state.r,
        state.w,
        state.L,
        np.asarray(metric["mu"]),
        np.asarray(metric["lambda"]),
        np.asarray(metric["mu_r"]),
        np.asarray(metric["lambda_t"]),
        state.radial_step,
        state.min_dw,
        par,
    )
    if state.t + dt > par.t_end:
        dt = par.t_end - state.t
    if not (dt > 0.0 and math.isfinite(dt)):
        raise RuntimeError("Non-positive or non-finite timestep.")

    # Estimate which occupied characteristics leave the finite domain during this chosen step.
    state.last_escape = estimate_escapes(
        state.f,
        state.r,
        state.w,
        state.L,
        state.r_weights,
        state.w_weights,
        state.L_weights,
        np.asarray(metric["mu"]),
        np.asarray(metric["lambda"]),
        np.asarray(metric["mu_r"]),
        np.asarray(metric["lambda_t"]),
        dt,
        par,
    )
    state.cumulative_escape["outer_mass"] += float(state.last_escape["outer_mass"])
    state.cumulative_escape["w_low_mass"] += float(state.last_escape["w_low_mass"])
    state.cumulative_escape["w_high_mass"] += float(state.last_escape["w_high_mass"])

    # Pull the old distribution back along the frozen-metric characteristics to obtain the new nodal values.
    state.f = advect_f(
        state.f,
        state.r,
        state.w,
        state.L,
        np.asarray(metric["mu"]),
        np.asarray(metric["lambda"]),
        np.asarray(metric["mu_r"]),
        np.asarray(metric["lambda_t"]),
        dt,
        par,
    )
    # Advance the accepted clock only after advection has produced the new distribution.
    state.t += dt
    state.last_dt = dt
    state.step += 1
    if par.restart_every > 0 and state.step % par.restart_every == 0:
        write_restart_state(state.outdir / "restart_state.bin", par, state.t, state.step, state.f)
    return dt


# Section: top-level evolution loop.

# Runs the Python timestep loop until a stopping condition is reached.
def run_evolution(state: SimulationState) -> None:
    par = state.par
    while state.step <= par.max_steps:
        # Recompute moments and constraints from the current f before saving or selecting the next step.
        rho, p, j = compute_moments(state.f, state.r, state.w, state.L, state.w_weights, state.L_weights)
        metric = solve_metric(rho, p, j, state.r, par)
        mass = float(metric["m"][-1] - par.central_mass)  # type: ignore[index]
        max_rho = float(np.max(rho))
        min_gap = float(metric["min_gap"])

        # Save the same matter and metric state whose stopping conditions are checked next.
        save_current_diagnostics(state, rho, p, j, metric)
        if stop_requested(state, min_gap):
            break

        dt = advance_one_step(state, metric)
        if state.step % max(1, 5 * par.save_every) == 0:
            print(
                f"step={state.step:6d} t={state.t:10.5f} dt={dt:9.3e} "
                f"mass={mass:.6e} max_rho={max_rho:.3e} min_gap={min_gap:.3e}"
            )
    else:
        print(f"Stopping: reached max_steps={par.max_steps}.")
