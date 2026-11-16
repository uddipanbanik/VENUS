
// Evolve weighted subhalos with spherical Poisson gravity, Plummer scattering
// and capture.
//
// The shared driver reads the background and canonical particle state and
// applies encounters at collision deadlines. Positive subhalo collision
// strength selects this Boltzmann path. Background forces and the applied halo
// field are handled by the same transport implementation.

#include "ev/apps/wpic_driver_entry.hpp"

// Pass the subhalo task to the shared weighted-particle driver.
int main(int argc, char** argv) {
    return ev::apps::run_expanding_weighted_vlasov_pic(argc, argv);
}
