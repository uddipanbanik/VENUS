
// Evolve Boltzmann--Poisson with sampled reversible collision reactions.
//
// The input defines the distribution, grid and coupling. The task fixes the
// sampled reversible collision backend while reusing the same transport,
// gravity and output modules as the deterministic formulation. Random reaction
// selection belongs to that backend.

#include "ev/solver.hpp"

// Select sampled reversible reactions before entering the common solver.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv,
        ev::SolverTaskProfile::boltzmann_poisson_reversible_sampled);
}
