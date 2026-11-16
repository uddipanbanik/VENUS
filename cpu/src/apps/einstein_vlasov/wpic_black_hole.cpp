
// Run the high-compactness spherical Einstein--Vlasov weighted-marker task.
//
// The driver constructs or restores weighted particles, deposits their stress
// and solves the metric constraints during characteristic transport. Its high-
// compactness profile records the metric gap and compactness and applies the
// declared stopping condition when the chart reaches its limit.

#include "ev/apps/wpic_driver_entry.hpp"

// Pass the actual arguments to the shared relativistic marker driver.
int main(int argc, char** argv) {
    return ev::apps::run_weighted_einstein_vlasov_pic(argc, argv);
}
