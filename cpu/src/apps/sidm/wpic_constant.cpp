
// Run weighted particles with Newtonian self-gravity and constant-area elastic
// dark matter scattering.
//
// The static marker driver combines spherical Poisson gravity with event-timed
// collisions. Inputs determine the physical coupling and particle state; the
// driver records represented-mass conservation, collision clocks, profiles and
// complete restart data.

#include "ev/apps/wpic_driver_entry.hpp"

// Use the constant-cross-section task profile in the static SIDM driver.
int main(int argc, char** argv) {
    return ev::apps::run_static_sidm_weighted_vlasov_pic(argc, argv);
}
