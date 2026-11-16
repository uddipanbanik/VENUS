
// Evolve a collisionless Newtonian distribution with the Vlasov--Poisson
// equations.
//
// The parameter file supplies a static or expanding background, spherical
// phase-space grid and initial distribution. The common solver advances the
// distribution under its Poisson field and writes physical moments, invariant
// budgets and restart state.

#include "ev/solver.hpp"

// Select the collisionless Newtonian task before the shared setup and loop.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv, ev::SolverTaskProfile::vlasov_poisson_eulerian);
}
