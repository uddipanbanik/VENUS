#pragma once

#include <chrono>

// Event-driven semi-Lagrangian carrier evolution for static spherical
// Boltzmann-Poisson.  Continuous weighted markers carry the distribution
// through a frozen potential.  Independent potential and collision clocks
// choose whether endpoint reconstruction occurs before or after local NTC
// scattering.
//
// Evolution order:
//   1. Initialize or restore weighted markers and the current frozen Poisson
//      field.  The collision core accepts arbitrary positive marker masses.
//   2. Query the two-clock scheduler for the next orbit interval.
//   3. Advance continuous marker orbits through that frozen field.
//   4. At the endpoint, apply the requested collision/reconstruction order;
//      reconstruction groups exact masses and represents an odd singleton by
//      two half-weight projection packets without changing the marker.
//   5. Refresh the field only from a conservative marker-to-grid remap and
//      explicitly budgeted carriers, then accumulate transport, collision,
//      remap, and clock diagnostics.
//
// The current coupling is static, cell-average spherical VP.  Expanding
// subhalo capture uses its dedicated runner and collision module instead.

#include "ev/hamiltonian_sldg_transport.hpp"
#include "ev/weighted_vlasov_pic_collisions.hpp"
#include "ev/weighted_vlasov_pic_event_scheduler.hpp"
#include "ev/weighted_vlasov_pic_phase_remap.hpp"

namespace ev {

// Evolve static spherical Boltzmann--Poisson with continuous weighted particles.
// The stored Poisson field advances their orbits between potential events; local
// scattering uses a separately accumulated collision interval. Inputs include the
// field/phase meshes, physical parameters and two clock fractions. Each accepted
// event returns the updated particles, clocks, field and conservation diagnostics.
struct WeightedPICBoltzmannEventOptions {
    double potential_fraction = 0.1;
    double collision_fraction = 0.1;
    double maximum_orbit_interval = 0.02;
    // In adaptive-shell mode, local collision CFL substeps are performed
    // inside a less frequent global operator batch.  The batch interval is
    // never longer than this cap or the current potential-refresh deadline.
    // This keeps gravity/collision splitting controlled while preventing the
    // fastest central shell from synchronizing the complete halo.
    bool adaptive_collision_shell_timestep = false;
    double adaptive_collision_maximum_interval = 0.02;
    unsigned int threads = 1;
    WeightedPICParticleShape field_shape = WeightedPICParticleShape::tsc;
    // The full (r,w,eta) projection is a diagnostic representation for the
    // persistent-marker solver; marker transport, gravity, and collisions use
    // the continuous markers and radial moment deposit directly.  Production
    // can therefore defer this projection while retaining the same trajectory.
    bool phase_space_projection_at_events = true;
    WeightedPICPhaseRemapOptions phase_remap;
};

struct WeightedPICBoltzmannEventState {
    std::vector<WeightedPICMarker> markers;
    std::vector<double> reconstructed_f;
    std::vector<std::size_t> reconstruction_carrier_marker_indices;
    double reconstruction_carrier_mass = 0.0;
    bool reconstructed_f_current = false;
    WeightedPICDepositedMoments deposited;
    VlasovPoissonField field;
    WeightedPICEventClockState clock;
    double potential_timescale = std::numeric_limits<double>::infinity();
    double collision_timescale = std::numeric_limits<double>::infinity();
    WeightedPICCollisionSamplerState collision_sampler;
    WeightedPICCollisionWorkspace collision_workspace;
    double time = 0.0;
    std::int64_t step = 0;
};

struct WeightedPICBoltzmannEventDiagnostics {
    WeightedPICEventPlan plan;
    WeightedPICCollisionDiagnostics collision;
    WeightedPICPhaseRemapDiagnostics reconstruction;
    std::size_t reconstruction_count = 0;
    bool reconstructed_before_collision = false;
    bool reconstructed_after_collision = false;
    bool clock_probe = false;
    double marker_mass_before = 0.0;
    double marker_mass_after = 0.0;
    double orbit_seconds = 0.0;
    double deposition_and_field_seconds = 0.0;
    double collision_seconds = 0.0;
    double clock_refresh_seconds = 0.0;
};

inline void validate_weighted_pic_boltzmann_event_options(
    const WeightedPICBoltzmannEventOptions& options) {
    if (!(options.potential_fraction > 0.0)
        || !(options.potential_fraction <= 1.0)
        || !std::isfinite(options.potential_fraction)
        || !(options.collision_fraction > 0.0)
        || !(options.collision_fraction <= 1.0)
        || !std::isfinite(options.collision_fraction)
        || !(options.maximum_orbit_interval > 0.0)
        || !std::isfinite(options.maximum_orbit_interval)
        || !(options.adaptive_collision_maximum_interval > 0.0)
        || !std::isfinite(options.adaptive_collision_maximum_interval)
        || options.threads == 0) {
        throw std::runtime_error(
            "Invalid weighted PIC Boltzmann event options.");
    }
}

// The radial field deposit and the conservative (r,w,eta) reconstruction share
// the same physical cell centres. This static event operator assumes that geometry.
inline void validate_weighted_pic_boltzmann_geometry(
    const WeightedPICFieldGrid& grid,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    if (p.physics_model != "vlasov_poisson"
        || p.vp_universe != "static"
        || !using_cell_average_eta_grid(p)
        || !using_conservative_phase_space_cells(p)) {
        throw std::runtime_error(
            "Weighted PIC Boltzmann events require static Vlasov-Poisson "
            "with conservative cell-average (r,w,eta) geometry.");
    }
    validate_cell_average_eta_geometry(
        geometry, p, "weighted PIC Boltzmann event geometry");
    if (grid.mesh.venus_center.size() != geometry.radial.center.size()
        || grid.radius.size() != geometry.radial.center.size()) {
        throw std::runtime_error(
            "Weighted PIC Boltzmann field and phase grids have different radial sizes.");
    }
    for (std::size_t i = 0; i < geometry.radial.center.size(); ++i) {
        const double scale = std::max({
            1.0,
            std::abs(grid.mesh.venus_center[i]),
            std::abs(geometry.radial.center[i]),
        });
        if (std::abs(grid.mesh.venus_center[i] - geometry.radial.center[i])
            > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
            throw std::runtime_error(
                "Weighted PIC Boltzmann field and phase grids do not share radial centres.");
        }
    }
}

inline double weighted_pic_boltzmann_potential_timescale(
    const WeightedPICBoltzmannEventState& state,
    const WeightedPICFieldGrid& grid,
    const Params& p) {
    const Moments moments = weighted_pic_as_venus_moments(
        state.deposited, grid.mesh);
    if (using_expanding_vp(p)) {
        return std::numeric_limits<double>::infinity();
    }
    const VlasovPoissonField derivative = vp_potential_time_derivative(
        moments, grid.radius, state.time, p);
    if (state.deposited.shell_mass.size() != grid.radius.size()
        || state.field.phi.size() != grid.radius.size()
        || derivative.phi.size() != grid.radius.size()) {
        throw std::runtime_error(
            "Weighted PIC potential clock size mismatch.");
    }

    // Shot noise in an almost empty radial cell can make a pointwise
    // |Phi/Phi_t| or |g/g_t| minimum arbitrarily small even though that cell
    // contributes negligibly to the self-consistent field.  Control instead
    // the represented-mass-weighted RMS change of Phi and g.  This is the
    // kinetic analogue of measuring the Hamiltonian error over the particles
    // whose orbits the field advances.
    // Normalize the field-change norm with the represented shell masses. Squared
    // potential and force rates then define their separate RMS variation times.
    long double total_mass = 0.0L;
    for (double shell_mass : state.deposited.shell_mass) {
        total_mass += std::max(0.0, shell_mass);
    }
    if (!(total_mass > 0.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    long double phi2 = 0.0L;
    long double phi_rate2 = 0.0L;
    long double force2 = 0.0L;
    long double force_rate2 = 0.0L;
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        const long double weight =
            std::max(0.0, state.deposited.shell_mass[i]) / total_mass;
        phi2 += weight * state.field.phi[i] * state.field.phi[i];
        phi_rate2 += weight * derivative.phi[i] * derivative.phi[i];
        force2 += weight * state.field.phi_r[i] * state.field.phi_r[i];
        force_rate2 +=
            weight * derivative.phi_r[i] * derivative.phi_r[i];
    }
    const long double floor =
        4096.0L * std::numeric_limits<double>::epsilon();
    double potential_timescale = std::numeric_limits<double>::infinity();
    double force_timescale = std::numeric_limits<double>::infinity();
    if (phi2 > floor * floor && phi_rate2 > floor * floor) {
        potential_timescale = static_cast<double>(
            std::sqrt(phi2 / phi_rate2));
    }
    if (force2 > floor * floor && force_rate2 > floor * floor) {
        force_timescale = static_cast<double>(
            std::sqrt(force2 / force_rate2));
    }
    return std::min(potential_timescale, force_timescale);
}

inline double weighted_pic_boltzmann_deposited_marker_mass(
    const WeightedPICDepositedMoments& deposited) {
    return deposited.active_marker_mass
        + deposited.hidden_inner_mass
        + deposited.escaped_mass;
}

inline double weighted_pic_boltzmann_collision_timescale(
    WeightedPICBoltzmannEventState& state,
    const WeightedPICFieldGrid& grid,
    const Params& p,
    const WeightedPICCollisionOptions& collision_options,
    unsigned int requested_threads = 1,
    bool reuse_collision_endpoint_layout = false) {
    if (reuse_collision_endpoint_layout
        && weighted_pic_collision_shell_layout_is_reusable(
            p, collision_options, state.collision_workspace)) {
        refresh_weighted_pic_collision_shell_kinematics(
            state.markers,
            grid.mesh,
            p,
            collision_options,
            state.collision_workspace,
            requested_threads);
    } else {
        build_weighted_pic_collision_shells(
            state.markers,
            grid.mesh,
            p,
            collision_options,
            state.collision_workspace,
            nullptr,
            nullptr,
            requested_threads);
    }
    return weighted_pic_sampled_collision_timescale(
        state.markers,
        p,
        collision_options,
        state.collision_workspace);
}

inline WeightedPICPhaseRemapDiagnostics
// Reconstruct a diagnostic phase distribution from the persistent particles.
// Carriers retain any explicitly unresolved mass so the representation has a
// complete mass budget; future trajectories continue from the particles.
synchronize_weighted_pic_boltzmann_distribution(
    WeightedPICBoltzmannEventState& state,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const WeightedPICPhaseRemapOptions& options = {}) {
    const WeightedPICPhaseRemapResult remap =
        remap_weighted_vlasov_pic_to_phase_space(
            state.markers, geometry, p, options);
    state.reconstructed_f = remap.f;
    state.reconstruction_carrier_marker_indices =
        remap.carrier_marker_indices;
    state.reconstruction_carrier_mass = remap.diagnostics.carrier_mass;
    state.reconstructed_f_current = true;
    return remap.diagnostics;
}

inline WeightedPICBoltzmannEventState
initialize_weighted_pic_boltzmann_event_state(
    std::vector<WeightedPICMarker> markers,
    double time,
    std::int64_t step,
    const WeightedPICEventClockState& clock,
    const WeightedPICCollisionSamplerState& collision_sampler,
    const WeightedPICFieldGrid& grid,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const WeightedPICBoltzmannEventOptions& options,
    const WeightedPICCollisionOptions& collision_options) {
    validate_weighted_pic_boltzmann_event_options(options);
    validate_weighted_pic_boltzmann_geometry(grid, geometry, p);
    if (!(time >= 0.0) || !std::isfinite(time) || step < 0
        || markers.empty()) {
        throw std::runtime_error(
            "Invalid weighted PIC Boltzmann initial event state.");
    }

    WeightedPICBoltzmannEventState state;
    state.markers = std::move(markers);
    state.time = time;
    state.step = step;
    // Carry both field/collision ages and the collision random stream from the
    // input state. These quantities determine the next event and stochastic sample.
    state.clock = clock;
    state.collision_sampler = collision_sampler;
    state.deposited = deposit_weighted_vlasov_pic_parallel(
        state.markers,
        grid.mesh,
        options.field_shape,
        options.threads);
    state.field = solve_weighted_pic_poisson_field(
        state.deposited, grid.mesh, state.time, p);
    if (options.phase_space_projection_at_events) {
        synchronize_weighted_pic_boltzmann_distribution(
            state, geometry, p, options.phase_remap);
    }
    state.potential_timescale = weighted_pic_boltzmann_potential_timescale(
        state, grid, p);
    state.collision_timescale = weighted_pic_boltzmann_collision_timescale(
        state,
        grid,
        p,
        collision_options,
        options.threads);
    return state;
}

/// Restore the field held between potential events and the already sampled
/// clock estimates. Rebuilding either from the current particles inserts an
/// extra potential event and can change the next collision/transport order.
/// Versions 1--6 supply particle initial data; their callers retain the
/// explicitly documented field reconstruction at import.
inline bool restore_weighted_pic_boltzmann_frozen_field(
    WeightedPICBoltzmannEventState& state,
    const WeightedPICRestartState& checkpoint,
    const std::string& expected_contract) {
    if (!checkpoint.has_frozen_field) return false;
    if (checkpoint.frozen_field_contract != expected_contract
        || checkpoint.frozen_field.radial_edge != state.field.radial_edge) {
        throw std::runtime_error(
            "Static SIDM restart field or evolution controls differ. "
            "Repeat the saved physical, mesh, orbit, collision and thread settings.");
    }
    state.field = checkpoint.frozen_field;
    state.potential_timescale = checkpoint.potential_timescale_estimate;
    state.collision_timescale = checkpoint.collision_timescale_estimate;
    return true;
}

/// Advances one event selected by the independent field and collision clocks.
///
/// A collision-first endpoint executes transport, NTC scattering, and one
/// reconstruction.  A potential-first endpoint executes transport,
/// reconstruction, and the field solve; its collision age is retained.  If
/// both deadlines coincide with potential-first ordering, collision follows
/// the field refresh and marks the reconstructed distribution stale.  A
/// caller that needs phase-space output at that point calls the synchronization
/// function above; marker transport itself needs no extra remap.
inline WeightedPICBoltzmannEventDiagnostics
advance_weighted_pic_boltzmann_event(
    WeightedPICBoltzmannEventState& state,
    const WeightedPICFieldGrid& grid,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const CharacteristicIntegratorPlan& orbit_plan,
    const WeightedPICBoltzmannEventOptions& options,
    const WeightedPICCollisionOptions& collision_options,
    double remaining_run_time = std::numeric_limits<double>::infinity()) {
    validate_weighted_pic_boltzmann_event_options(options);
    validate_weighted_pic_boltzmann_geometry(grid, geometry, p);
    if (!(remaining_run_time > 0.0) || std::isnan(remaining_run_time)) {
        throw std::runtime_error(
            "Weighted PIC Boltzmann event needs positive remaining run time.");
    }

    WeightedPICBoltzmannEventDiagnostics diagnostics;
    diagnostics.marker_mass_before =
        weighted_pic_boltzmann_deposited_marker_mass(state.deposited);
    const double orbit_cap = std::min(
        options.maximum_orbit_interval, remaining_run_time);
    // Adaptive shell collisions substep locally inside a global operator batch.
    // The potential deadline also limits that batch, controlling the force split.
    double planning_collision_timescale = state.collision_timescale;
    if (options.adaptive_collision_shell_timestep) {
        double operator_interval = options.adaptive_collision_maximum_interval;
        if (std::isfinite(state.potential_timescale)) {
            operator_interval = std::min(
                operator_interval,
                options.potential_fraction * state.potential_timescale);
        }
        planning_collision_timescale =
            operator_interval / options.collision_fraction;
    }
    diagnostics.plan = plan_weighted_pic_event(
        state.clock,
        state.potential_timescale,
        planning_collision_timescale,
        options.potential_fraction,
        options.collision_fraction,
        orbit_cap);
    const double endpoint_time = state.time + diagnostics.plan.orbit_interval;
    const double accumulated_collision_interval =
        state.clock.collision_age + diagnostics.plan.orbit_interval;
    bool collision_has_run = false;

    // Deposit the endpoint particles after the requested event ordering. The
    // Poisson field changes only when its own clock is due; output remaps may be stale.
    auto reconstruct = [&] {
        const auto operation_start = std::chrono::steady_clock::now();
        if (options.phase_space_projection_at_events) {
            diagnostics.reconstruction =
                synchronize_weighted_pic_boltzmann_distribution(
                    state, geometry, p, options.phase_remap);
            ++diagnostics.reconstruction_count;
            if (collision_has_run) {
                diagnostics.reconstructed_after_collision = true;
            } else {
                diagnostics.reconstructed_before_collision = true;
            }
        } else {
            state.reconstructed_f_current = false;
        }
        state.deposited = deposit_weighted_vlasov_pic_parallel(
            state.markers,
            grid.mesh,
            options.field_shape,
            options.threads);
        if (diagnostics.plan.potential_due) {
            state.field = solve_weighted_pic_poisson_field(
                state.deposited, grid.mesh, endpoint_time, p);
        }
        diagnostics.deposition_and_field_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - operation_start).count();
    };
    // Apply scattering over the full carried collision age plus this orbit
    // interval. The scattering module updates the random stream and unequal weights.
    auto collide = [&] {
        const auto operation_start = std::chrono::steady_clock::now();
        diagnostics.collision = apply_weighted_pic_collision_interval(
            state.markers,
            grid.mesh,
            accumulated_collision_interval,
            p,
            collision_options,
            state.collision_sampler,
            state.collision_workspace,
            options.threads);
        collision_has_run = true;
        state.reconstructed_f_current = false;
        diagnostics.collision_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - operation_start).count();
    };

    execute_weighted_pic_event(
        diagnostics.plan,
        [&] {
            const auto operation_start = std::chrono::steady_clock::now();
            advance_weighted_pic_markers_with_venus_parallel(
                state.markers,
                diagnostics.plan.orbit_interval,
                grid.radial_coordinate,
                grid.radius,
                state.field,
                p,
                orbit_plan,
                options.threads);
            state.reconstructed_f_current = false;
            diagnostics.orbit_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - operation_start).count();
        },
        reconstruct,
        collide);

    if (diagnostics.plan.order == WeightedPICEventOrder::none) {
        diagnostics.clock_probe = true;
        const auto operation_start = std::chrono::steady_clock::now();
        state.deposited = deposit_weighted_vlasov_pic_parallel(
            state.markers,
            grid.mesh,
            options.field_shape,
            options.threads);
        diagnostics.deposition_and_field_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - operation_start).count();
    } else if (diagnostics.plan.order
               == WeightedPICEventOrder::reconstruction_then_collision) {
        // Local collisions preserve shell mass and momentum, while this
        // redeposit refreshes pressure diagnostics and clock inputs.
        const auto operation_start = std::chrono::steady_clock::now();
        state.deposited = deposit_weighted_vlasov_pic_parallel(
            state.markers,
            grid.mesh,
            options.field_shape,
            options.threads);
        diagnostics.deposition_and_field_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - operation_start).count();
    }

    // Commit the accepted event once, then estimate fresh clocks from its final
    // particles and field. Clock ages reset according to the actual operators applied.
    state.time = endpoint_time;
    ++state.step;
    commit_weighted_pic_event(state.clock, diagnostics.plan);
    const auto clock_refresh_start = std::chrono::steady_clock::now();
    state.potential_timescale = weighted_pic_boltzmann_potential_timescale(
        state, grid, p);
    state.collision_timescale = weighted_pic_boltzmann_collision_timescale(
        state,
        grid,
        p,
        collision_options,
        options.threads,
        collision_has_run);
    diagnostics.clock_refresh_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_refresh_start).count();

    diagnostics.marker_mass_after =
        weighted_pic_boltzmann_deposited_marker_mass(state.deposited);
    // Close the represented mass budget across active, inner and escaped particles
    // after transport, local scattering and any diagnostic reconstruction.
    const double mass_scale = std::max(1.0, diagnostics.marker_mass_before);
    if (std::abs(
            diagnostics.marker_mass_after - diagnostics.marker_mass_before)
        > 128.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        throw std::runtime_error(
            "Weighted PIC Boltzmann event changed represented marker mass.");
    }
    return diagnostics;
}

}  // namespace ev
