
// Run the low-compactness spherical Einstein--Vlasov weighted-marker task.
//
// The shared driver reads the halo initial or restart state, deposits its
// relativistic stress and advances particles with the reconstructed metric.
// The task profile fixes the halo scenario while preserving the common rest-
// mass, profile and restart bookkeeping.

#include "ev/apps/wpic_driver_entry.hpp"

// Use the halo task selected at build time in the shared relativistic driver.
int main(int argc, char** argv) {
    return ev::apps::run_weighted_einstein_vlasov_pic(argc, argv);
}
