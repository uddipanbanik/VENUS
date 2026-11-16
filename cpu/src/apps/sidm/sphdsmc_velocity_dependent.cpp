
// Evolve spherical DSMC particles with a velocity-dependent material collision
// law.
//
// The task admits the supported power-law, softened-gravity or Yukawa/Born
// law. Physical parameters, initial data and grid come from the input; the
// common solver supplies orbital transport, pair sampling, self-gravity and
// output.

#include "ev/solver.hpp"

// Select the velocity-dependent spherical DSMC family before setup.
int main(int argc, char** argv) {
    return ev::run_solver_for_task(
        argc, argv,
        ev::SolverTaskProfile::sidm_sphdsmc_velocity_dependent);
}
