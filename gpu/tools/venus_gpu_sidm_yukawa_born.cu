// Run static material dark matter evolution with the Yukawa-Born law.
// Inputs include a particle restart, coupling, dark matter mass, mediator
// mass, and mesh/time/output controls. The fixed task chooses both the
// velocity-dependent total rate and angular scattering law; the shared driver
// returns profiles, encounter budgets, histories, and checkpoints.

#include "venus_gpu/production_driver.hpp"

int main(int argc, char** argv) {
    return venus_gpu::driver::run_production_task(
        argc, argv, venus_gpu::driver::ProductionTask::sidm_yukawa_born);
}
