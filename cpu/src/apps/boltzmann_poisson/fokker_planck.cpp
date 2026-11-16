
// Evolve Boltzmann--Poisson with the configured Fokker--Planck collision
// operator.
//
// The shared solver builds the velocity-space drift and diffusion
// coefficients, applies transport and reconstructs the gravitational field.
// The task fixes the collision-operator name while the input supplies its
// physical parameters, mesh and initial distribution.

#include "ev/solver.hpp"

// Select this Fokker--Planck operator before allocating the solver state.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv,
        ev::SolverTaskProfile::boltzmann_poisson_fokker_planck);
}
