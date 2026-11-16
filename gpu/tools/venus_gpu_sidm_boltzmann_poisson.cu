// Run material dark matter Boltzmann-Poisson evolution in a static halo.
// The command line supplies a particle restart, an explicit cross-section
// law and strength, mesh/time controls, and output paths. The shared driver
// combines gravity with elastic material encounters and publishes density
// profiles, collision diagnostics, histories, and atomic checkpoints.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc,
        argv,
        venus_gpu::driver::ProductionTask::sidm_boltzmann_poisson);
}
