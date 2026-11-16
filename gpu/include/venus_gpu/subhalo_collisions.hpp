#pragma once

// Evolve local encounters between unequal-mass Plummer subhalos on the GPU.
// Inputs are masses, velocities, internal dispersions, a radial mesh, elapsed
// time, collision controls, and sampler state. apply() updates the particle
// population and internal energies and returns event and conservation budgets.
// Calibrated cross sections separate scattering from capture. Scattering
// changes two orbits; capture combines their mass and removes one record.
// Both maps account for orbital energy transferred into internal energy.

#include "venus_gpu/collision_common.hpp"
#include "venus_gpu/radial_mesh.hpp"
#include "venus_gpu/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace venus_gpu {

struct SubhaloCollisionOptions {
    // velocity_scale retains the compatibility input; pair speed scales use
    // the actual internal dispersions. plummer_area_per_mass sets pair area,
    // and low_speed_regularization sets the finite table argument near rest.
    double velocity_scale = 1.0;
    double plummer_area_per_mass = 1.0;
    double low_speed_regularization = 0.05;
    double strength = 0.0;
    // Bound the sampled encounter probability in each substep. A capture
    // changes masses and dispersions, so the local rate must then be rebuilt.
    double collision_cfl = 0.2;
    int maximum_substeps = 256;
    // Advance disjoint radial cells with independent local CFL clocks inside
    // one globally scheduled operator batch.  Several local substeps are
    // batched per CUDA launch; a cell stops early after capture so its changed
    // mass/dispersion proposal can be rebuilt before it advances again.
    bool adaptive_cell_timestep = false;
    int adaptive_batch_substeps = 256;
    // The unresolved inner region may be split into collision cells; lanes
    // partition candidate work within a cell while preserving local pairing.
    int inner_collision_cells = 1;
    int collision_lanes = 1;
    std::size_t minimum_cell_population = 2;
    bool enable_scattering = true;
    bool enable_capture = true;
    // MPI ownership uses an offset into the common radial mesh and a
    // collective maximum so all ranks use a consistent global batch interval.
    std::size_t resolved_cell_offset = 0;
    double (*maximum_frequency_reducer)(double, void*) = nullptr;
    void* maximum_frequency_context = nullptr;
};

// Pair outputs separate physical scatter/capture areas from the upper
// bound on speed times area used to propose and accept encounters.
struct SubhaloCrossSections {
    double pair_velocity_scale = 0.0;
    double pair_area = 0.0;
    double scattering = 0.0;
    double capture = 0.0;
    double u_sigma_majorant = 0.0;
};

SubhaloCrossSections subhalo_cross_sections(
    double relative_speed,
    double first_mass,
    double second_mass,
    double first_internal_dispersion,
    double second_internal_dispersion,
    const SubhaloCollisionOptions& options);

// Event mass and orbital/internal energy changes close the capture and
// heating budgets. Width over mean free path tests the assumed spatial
// locality separately from the probability bound on a collision substep.
struct SubhaloCollisionDiagnostics {
    std::uint64_t substeps = 0;
    std::uint64_t maximum_cell_substeps = 0;
    std::uint64_t cell_visits = 0;
    std::uint64_t kernel_batches = 0;
    std::uint64_t proposal_rebuilds = 0;
    std::uint64_t trial_pairs = 0;
    std::uint64_t skipped_already_collided_trials = 0;
    std::uint64_t accepted_pairs = 0;
    std::uint64_t scattering_events = 0;
    std::uint64_t capture_events = 0;
    std::uint64_t inner_scattering_events = 0;
    std::uint64_t inner_capture_events = 0;
    std::size_t active_markers_before = 0;
    std::size_t active_markers_after = 0;
    std::size_t removed_markers = 0;
    std::size_t nonlocal_cells = 0;
    double represented_scattering_mass = 0.0;
    double represented_capture_mass = 0.0;
    double orbital_kinetic_energy_change = 0.0;
    double internal_energy_change = 0.0;
    double maximum_pair_momentum_error = 0.0;
    double maximum_pair_total_energy_error = 0.0;
    double minimum_mean_free_path = 0.0;
    double minimum_collision_timescale = 0.0;
    double maximum_cell_width_over_mean_free_path = 0.0;
    double minimum_pair_velocity_scale = 0.0;
    double maximum_pair_velocity_scale = 0.0;
};

class SubhaloCollisionModule {
public:
    explicit SubhaloCollisionModule(const RadialMesh& mesh);
    ~SubhaloCollisionModule();
    SubhaloCollisionModule(SubhaloCollisionModule&&) noexcept;
    SubhaloCollisionModule& operator=(SubhaloCollisionModule&&) noexcept;
    SubhaloCollisionModule(const SubhaloCollisionModule&) = delete;
    SubhaloCollisionModule& operator=(const SubhaloCollisionModule&) = delete;

    // Evaluate encounters at fixed current radii over dt and advance the
    // sampler with them. Gravity is rebuilt by the caller after the batch.
    SubhaloCollisionDiagnostics apply(
        WpicState& state,
        double dt,
        const SubhaloCollisionOptions& options,
        CollisionSamplerState& sampler);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace venus_gpu
