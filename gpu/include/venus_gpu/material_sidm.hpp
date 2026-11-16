#pragma once

// Apply elastic material dark matter encounters to a GPU particle population.
// Inputs are particle state, elapsed collision time, a cross-section law, and
// sampler state. apply() updates velocities and the sampler, then returns
// encounter counts, conservation errors, and shell-locality diagnostics.
// Radius-local shells approximate the spatially local Boltzmann operator.
// Disjoint pairs preserve pair momentum and kinetic energy. The independent
// Vlasov-Poisson module supplies the gravitational orbit update.

#include "venus_gpu/collision_common.hpp"
#include "venus_gpu/radial_mesh.hpp"
#include "venus_gpu/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace venus_gpu {

// All laws return cross section per unit mass; the selected law also sets
// the outgoing-angle distribution used by an accepted encounter.
enum class CrossSectionModel { constant, power_law, yukawa_born };

struct CollisionOptions {
    CrossSectionModel cross_section = CrossSectionModel::constant;
    // The strength multiplies the physical rate. sigma0 and the velocity
    // scales parameterize the selected cross-section law in the run units.
    double collision_strength = 1.0;
    double sigma0 = 0.0;
    double sigma_power = 4.0;
    double sigma_u0 = 0.0;
    double yukawa_alpha = 1.0e-2;
    double yukawa_mchi = 1.0;
    double yukawa_mphi = 1.0;
    // Radius sorting forms approximately fixed-occupancy shells. The
    // minimum population limits undersampled shells. Shell width relative to
    // the physical mean free path is always reported; fail_on_nonlocal_shell
    // enables rejection when that locality criterion is exceeded.
    std::size_t target_shell_markers = 4096;
    std::size_t minimum_shell_markers = 32;
    // Limit the largest pair acceptance probability per local substep.
    // maximum_substeps bounds the work required to cover the supplied time.
    double collision_cfl = 0.2;
    int maximum_substeps = 200;
    // Advance each fixed-occupancy radial shell with an independent local CFL
    // clock.  Shell geometry is unchanged by elastic scattering, while the
    // velocity-dependent majorant is refreshed after every local substep.
    bool adaptive_shell_timestep = false;
    bool fail_on_nonlocal_shell = false;
};

// Evaluate the same rate and angular laws on the host for configuration
// checks and comparisons with the device sampler.
double material_cross_section_per_mass(
    double relative_speed,
    const CollisionOptions& options);
double material_scatter_cosine(
    double relative_speed,
    double unit,
    const CollisionOptions& options);

// Counts describe sampled work. Momentum and energy errors measure the
// effect of the complete batch; pair errors expose individual encounter maps.
struct CollisionDiagnostics {
    std::uint64_t substeps = 0;
    std::uint64_t shell_visits = 0;
    std::uint64_t trial_pairs = 0;
    std::uint64_t accepted_pairs = 0;
    std::size_t active_markers = 0;
    std::size_t shells = 0;
    std::size_t nonlocal_shells = 0;
    double maximum_shell_width_over_mean_free_path = 0.0;
    double radial_momentum_error = 0.0;
    double kinetic_energy_error = 0.0;
    double max_pair_momentum_error = 0.0;
    double max_pair_kinetic_energy_error = 0.0;
};

class CollisionModule {
public:
    explicit CollisionModule(const RadialMesh& mesh);
    ~CollisionModule();
    CollisionModule(CollisionModule&&) noexcept;
    CollisionModule& operator=(CollisionModule&&) noexcept;
    CollisionModule(const CollisionModule&) = delete;
    CollisionModule& operator=(const CollisionModule&) = delete;

    // Advance collisions at the current radii over dt. Adaptive shells
    // cover that interval with independent local clocks and refreshed rates.
    CollisionDiagnostics apply(
        WpicState& state,
        double dt,
        const CollisionOptions& options,
        CollisionSamplerState& sampler);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace venus_gpu
