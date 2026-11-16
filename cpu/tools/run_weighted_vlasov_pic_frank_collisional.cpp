
// Provide the combined static self-interacting dark matter marker command.
// The supplied cross-section parameters choose the constant or supported
// velocity-dependent material law. The shared driver owns spherical Poisson
// gravity, orbital transport, collision deadlines, conservation diagnostics
// and complete restart state.

#include "ev/apps/wpic_driver_entry.hpp"

// The static SIDM parser receives the original arguments and returns the
// physical driver status to the calling process.
int main(int argc, char** argv) {
    return ev::apps::run_static_sidm_weighted_vlasov_pic(argc, argv);
}
