// Run static material dark matter evolution with a constant cross section.
// The command line supplies a particle restart, positive collision strength,
// sigma0 normalization, and mesh/time controls. The fixed task below selects
// this law in the shared driver, which produces profiles, collision budgets,
// histories, checkpoints, and a process completion status.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc, argv, venus_gpu::driver::ProductionTask::sidm_constant);
}
