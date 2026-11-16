
// Provide the combined Newtonian weighted-particle command interface.
// The input background and encounter strength select the collisionless
// Vlasov--Poisson or collisional subhalo path in the same driver. Canonical
// particle transport, density deposition, Poisson gravity and encounter clocks
// therefore use the same implementation as the task-specific executables.

#include "ev/apps/wpic_driver_entry.hpp"

// Forward the supplied physical arguments unchanged; the shared parser
// constructs the state and selects its permitted physical task.
int main(int argc, char** argv) {
    return ev::apps::run_expanding_weighted_vlasov_pic(argc, argv);
}
