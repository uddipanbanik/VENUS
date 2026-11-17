// Evolve isolated collisionless matter with the shared spherical WPIC kernel.
// The build profile fixes H=0 and zero encounter strength before initialization.
#include "ev/apps/wpic_driver_entry.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        return ev::apps::run_expanding_weighted_vlasov_pic(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Static Vlasov-Poisson error: " << error.what() << '\n';
        return 1;
    }
}
