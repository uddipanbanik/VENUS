
// Evolve collisionless weighted particles in spherical Newtonian gravity.
//
// The shared driver transports the canonical particle variables, deposits the
// density source and solves the Poisson field in the declared background. This
// task sets subhalo encounter strength to zero while retaining physical
// profile and restart diagnostics.

#include "ev/apps/wpic_driver_entry.hpp"

// Select the collisionless task in the shared weighted-particle driver.
int main(int argc, char** argv) {
    return ev::apps::run_expanding_weighted_vlasov_pic(argc, argv);
}
