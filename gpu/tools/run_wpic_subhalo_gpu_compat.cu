// Run the compatibility GPU interface for spherical VP or subhalo BP.
// Command-line arguments provide a particle restart, mesh, physical options,
// and output controls. Zero subhalo strength selects collisionless motion;
// positive strength selects scattering/capture. The shared driver publishes
// profiles, histories, checkpoints, and the process completion status.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc,
        argv,
        venus_gpu::driver::ProductionTask::legacy_subhalo_auto);
}
