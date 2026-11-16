
// Evolve a spherical self-interacting dark matter system with a constant
// isotropic material cross section.
//
// Persistent DSMC particles carry the distribution between force and collision
// steps. The input supplies the physical cross section, material
// normalization, initial state and radial grid. The common solver applies
// self-gravity, no-time-counter encounters and conservation diagnostics.

#include "ev/solver.hpp"

// Select the constant-area spherical DSMC task in the common solver.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv, ev::SolverTaskProfile::sidm_sphdsmc_constant);
}
