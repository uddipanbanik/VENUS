
// Evolve Boltzmann--Poisson with a streamed reversible collision graph.
//
// The backend visits its reaction graph in bounded working storage. The
// initial distribution, gravitational parameters and phase-space grid come
// from the input. The shared driver owns temporal ordering, conservation
// diagnostics and restart output.

#include "ev/solver.hpp"

// Select the streamed reaction backend and its common evolution loop.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv,
        ev::SolverTaskProfile::boltzmann_poisson_reversible_streamed);
}
