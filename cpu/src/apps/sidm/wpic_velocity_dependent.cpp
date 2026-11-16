
// Run weighted particles with Newtonian self-gravity and velocity-dependent
// elastic scattering.
//
// The selected power-law or Yukawa/Born cross section weights relative-speed-
// dependent pair encounters. The static SIDM driver handles force transport,
// collision scheduling, conservation diagnostics and restart state using the
// supplied particle data.

#include "ev/apps/wpic_driver_entry.hpp"

// Use the velocity-dependent task profile in the static SIDM driver.
int main(int argc, char** argv) {
    return ev::apps::run_static_sidm_weighted_vlasov_pic(argc, argv);
}
