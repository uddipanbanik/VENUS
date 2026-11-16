
// Select a finite-angle Eulerian Boltzmann--Poisson calculation.
//
// The input chooses direct quadrature or a supported deterministic, sampled or
// streamed reversible collision operator. All choices share phase-space
// transport and Poisson gravity. Their distinct angular and collision
// discretizations remain selected by the parameter file.

#include "ev/solver.hpp"

// Admit the finite-angle family and let the shared solver select its backend.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv,
        ev::SolverTaskProfile::boltzmann_poisson_finite_angle);
}
