// Run static material dark matter evolution with a softened power-law
// cross section. Inputs include a particle restart, positive strength, sigma0,
// and a softening speed, followed by the common mesh/time/output controls.
// The shared driver applies this fixed law and writes profiles, encounter
// budgets, histories, and atomic checkpoints.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc, argv, venus_gpu::driver::ProductionTask::sidm_power_law);
}
