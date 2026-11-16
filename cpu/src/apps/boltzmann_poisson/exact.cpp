
// Evolve the Eulerian Boltzmann--Poisson distribution with direct finite-angle
// collision quadrature.
//
// The parameter file supplies the phase-space mesh, self-gravity and material
// or subhalo law. The exact task selects boltzmann_exact, while the shared
// solver computes transport, collision gain/loss and the Poisson field.

#include "ev/solver.hpp"

// Pass the exact finite-angle task to the shared parser and evolution loop.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv, ev::SolverTaskProfile::boltzmann_poisson_exact);
}
