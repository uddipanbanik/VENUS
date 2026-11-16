
// Evolve Boltzmann--Poisson with the deterministic reversible collision
// discretization.
//
// The positive reaction rates connect discrete incoming and outgoing states
// while retaining the operator conservation constraints. The common solver
// supplies spatial transport, Poisson gravity, state construction and physical
// diagnostics from the requested initial data.

#include "ev/solver.hpp"

// Choose the deterministic reversible backend in the shared task profile.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv, ev::SolverTaskProfile::boltzmann_poisson_reversible);
}
