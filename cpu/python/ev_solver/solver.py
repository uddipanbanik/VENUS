# Run a small Python Einstein--Vlasov reference calculation from EVParams.
# Return arrays and diagnostics for write_outputs. This entry point keeps setup, evolution and final projection in order.

"""Public Python API for running a small Einstein--Vlasov reference solve.

``run_simulation`` separates setup, the timestep loop, and final phase output.
It is intended for transparent small comparisons; production scaling belongs
to the modular C++ executable.
"""

from __future__ import annotations

from .numerics import phase_space_projection
from .solver_evolution import run_evolution
from .solver_setup import prepare_simulation
from .params import EVParams


# Section: public entry point.

# Public Python entry point that prepares state, runs evolution, and returns output-ready data.
def run_simulation(par: EVParams) -> dict[str, object]:
    state = prepare_simulation(par)
    # Always close the optional phase stream, including when an evolution or I/O error propagates.
    try:
        run_evolution(state)
    finally:
        state.close_phase_snapshots()

    # Project the accepted final distribution over L before packaging the output dictionary.
    return state.to_result(phase_space_projection(state.f, state.L, state.L_weights))
