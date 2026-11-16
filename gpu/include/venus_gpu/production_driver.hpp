#pragma once

// Connect task-specific GPU executables to the shared restart-based driver.
// Inputs are command-line arguments and a fixed ProductionTask. The task
// selects the allowed background and collision family before particle loading;
// the driver then coordinates gravity, encounters, output, and checkpoints.
// Single-GPU VP and subhalo tasks support static or EdS backgrounds. MPI VP
// and subhalo tasks use EdS; material dark matter tasks use static backgrounds.
// The return value records completion, configuration failure, a physics safety
// stop, or a requested checkpoint stop.

namespace venus_gpu {
namespace driver {

enum class ProductionTask {
    // Compatibility mode used only by run_wpic_subhalo_gpu*.  A
    // zero subhalo strength selects collisionless VP; a positive strength
    // selects the calibrated subhalo operator.
    legacy_subhalo_auto,

    // Explicit task names keep the physical collision model fixed while
    // the shared driver parses mesh, clock, and output controls.
    vlasov_poisson,
    subhalo_boltzmann_poisson,
    sidm_boltzmann_poisson,
    sidm_constant,
    sidm_power_law,
    sidm_yukawa_born,
};

/// Run one restart-based WPIC task under its task-locked universe contract.
///
/// The function catches user/configuration/runtime exceptions and returns the
/// same process status contract as the original production executable:
/// 0=complete, 2=error, 97/98=physics safety stop, 99=walltime checkpoint.
int run_production_task(
    int argc,
    char** argv,
    ProductionTask task);

}  // namespace driver
}  // namespace venus_gpu
