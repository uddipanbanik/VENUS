// Run collisionless spherical Vlasov-Poisson evolution on one GPU.
// Command-line inputs supply a restart, static or EdS background, mesh, time
// controls, and output directory. The shared driver advances gravity and
// orbits and writes profiles, histories, and atomic checkpoints. This fixed
// task selects the collisionless parser and evolution path.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc, argv, venus_gpu::driver::ProductionTask::vlasov_poisson);
}
