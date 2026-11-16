// Run spherical subhalo Boltzmann-Poisson evolution on one GPU.
// Inputs are a restart with particle masses/internal dispersions, an explicit
// positive encounter strength, a static or EdS background, and mesh/time
// controls. The shared driver combines orbital motion with calibrated local
// scattering/capture and writes profiles, event budgets, and checkpoints.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc,
        argv,
        venus_gpu::driver::ProductionTask::subhalo_boltzmann_poisson);
}
