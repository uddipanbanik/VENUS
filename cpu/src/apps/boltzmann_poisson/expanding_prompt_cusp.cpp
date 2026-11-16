
// Evolve the expanding, self-gravitating subhalo Boltzmann--Poisson model.
//
// This task combines a density-contrast Poisson field with direct finite-angle
// scattering and the softened-gravity subhalo cross section. The shared solver
// reads the phase-space initial distribution and background parameters and
// writes its physical moments and restart state.

#include "ev/solver.hpp"

// Select the expanding subhalo physical tuple before constructing its state.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv,
        ev::SolverTaskProfile::boltzmann_poisson_expanding_prompt_cusp);
}
