
// Evolve the Landau--Fokker--Planck form of Boltzmann--Poisson.
//
// The landau_fp task uses the shared velocity-space collision coefficients,
// gravitational field solve and transport loop. The parameter file supplies
// the phase-space grid, initial distribution and physical coupling. Output and
// restart ordering remain in the common solver.

#include "ev/solver.hpp"

// Select the Landau operator and delegate its complete evolution.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv, ev::SolverTaskProfile::boltzmann_poisson_landau);
}
