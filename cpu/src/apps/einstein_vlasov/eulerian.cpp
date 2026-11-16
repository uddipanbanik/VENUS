
// Evolve the collisionless Einstein--Vlasov distribution on an Eulerian phase-
// space grid.
//
// The input specifies the spherical initial distribution, mesh and
// relativistic field controls. The shared solver computes its stress moments,
// radial metric constraints and characteristic transport and records the
// resulting physical fields and invariants.

#include "ev/solver.hpp"

// Select collisionless relativistic gravity before the shared setup step.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv, ev::SolverTaskProfile::einstein_vlasov_eulerian);
}
