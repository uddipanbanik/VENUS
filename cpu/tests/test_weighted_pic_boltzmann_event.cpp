// End-to-end tests for frozen-field weighted-PIC transport, elastic NTC
// collision ordering, conservative phase remap, field refresh, and mass closure.

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "ev/weighted_pic_boltzmann_event.hpp"

namespace {

ev::Params event_params() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_orbit_regularization = "auto";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = 3;
    p.Nw = 24;
    p.NL = 64;
    p.r_min = 0.5;
    p.r_max = 2.0;
    p.radial_grid = "linear";
    p.w_max = 6.0;
    p.L_max = 8.0;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "reflecting";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 3.5;
    p.bp_reversible_candidate_radius = 4;
    p.bp_reversible_projection_tolerance = 2.0e-10;
    return p;
}

std::vector<ev::WeightedPICMarker> event_markers(
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    constexpr std::size_t per_radius = 512;
    std::vector<ev::WeightedPICMarker> markers;
    markers.reserve(per_radius * static_cast<std::size_t>(p.Nr));
    for (int i = 0; i < p.Nr; ++i) {
        for (std::size_t q = 0; q < per_radius; ++q) {
            ev::WeightedPICMarker marker;
            marker.mass = 1.0e-3;
            marker.r = geometry.radial.center[static_cast<std::size_t>(i)]
                + 0.1 * geometry.radial.width[static_cast<std::size_t>(i)]
                    * (static_cast<double>(q) + 0.5
                       - 0.5 * static_cast<double>(per_radius))
                    / static_cast<double>(per_radius);
            marker.w = geometry.radial_velocity.center[
                static_cast<std::size_t>(7 + (5 * q + i) % 10)];
            marker.eta = geometry.eta.center[
                static_cast<std::size_t>(2 + (7 * q + i) % 7)];
            marker.id = markers.size();
            markers.push_back(marker);
        }
    }
    return markers;
}

bool close(double first, double second, double tolerance = 3.0e-12) {
    return std::abs(first - second)
        <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

void assert_same_marker_state(
    const ev::WeightedPICBoltzmannEventState& first,
    const ev::WeightedPICBoltzmannEventState& second) {
    assert(first.markers.size() == second.markers.size());
    for (std::size_t i = 0; i < first.markers.size(); ++i) {
        const ev::WeightedPICMarker& a = first.markers[i];
        const ev::WeightedPICMarker& b = second.markers[i];
        assert(a.mass == b.mass);
        assert(a.r == b.r);
        assert(a.w == b.w);
        assert(a.eta == b.eta);
        assert(a.id == b.id);
        assert(a.source_cell == b.source_cell);
        assert(a.state == b.state);
        assert(a.inner_boundary_speed == b.inner_boundary_speed);
        assert(a.remaining_inner_time == b.remaining_inner_time);
        assert(a.internal_energy == b.internal_energy);
        assert(a.internal_velocity_dispersion
               == b.internal_velocity_dispersion);
    }
    assert(first.deposited.shell_mass == second.deposited.shell_mass);
    assert(first.deposited.shell_radial_momentum
           == second.deposited.shell_radial_momentum);
    assert(first.deposited.shell_radial_second_moment
           == second.deposited.shell_radial_second_moment);
    assert(first.deposited.shell_tangential_second_moment
           == second.deposited.shell_tangential_second_moment);
    assert(first.deposited.density == second.deposited.density);
    assert(first.field.phi == second.field.phi);
    assert(first.field.phi_r == second.field.phi_r);
    assert(first.clock.potential_age == second.clock.potential_age);
    assert(first.clock.collision_age == second.clock.collision_age);
    assert(first.collision_sampler.seed == second.collision_sampler.seed);
    assert(first.collision_sampler.next_counter
           == second.collision_sampler.next_counter);
    assert(first.potential_timescale == second.potential_timescale);
    assert(first.collision_timescale == second.collision_timescale);
    assert(first.time == second.time);
    assert(first.step == second.step);
}

ev::WeightedPICBoltzmannEventState make_state(
    const ev::Params& p,
    const ev::WeightedPICFieldGrid& grid,
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::WeightedPICBoltzmannEventOptions& event_options,
    const ev::WeightedPICCollisionOptions& collision_options) {
    return ev::initialize_weighted_pic_boltzmann_event_state(
        event_markers(geometry, p),
        0.0,
        0,
        {},
        {UINT64_C(0x12345678), 0},
        grid,
        geometry,
        p,
        event_options,
        collision_options);
}

void test_collision_first_then_reconstruction() {
    const ev::Params p = event_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    ev::WeightedPICBoltzmannEventOptions event_options;
    event_options.potential_fraction = 1.0;
    event_options.collision_fraction = 1.0;
    event_options.maximum_orbit_interval = 0.01;
    event_options.threads = 2;
    ev::WeightedPICCollisionOptions collision_options;
    collision_options.target_shell_markers = 256;
    collision_options.minimum_shell_markers = 16;
    collision_options.fail_on_nonlocal_shell = false;
    ev::WeightedPICBoltzmannEventState state = make_state(
        p, grid, geometry, event_options, collision_options);
    state.potential_timescale = 1.0;
    state.collision_timescale = 0.01;
    const ev::CharacteristicIntegratorPlan orbit_plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::WeightedPICBoltzmannEventDiagnostics diagnostic =
        ev::advance_weighted_pic_boltzmann_event(
            state,
            grid,
            geometry,
            p,
            orbit_plan,
            event_options,
            collision_options,
            0.02);
    assert(diagnostic.plan.order
           == ev::WeightedPICEventOrder::collision_then_reconstruction);
    assert(diagnostic.reconstruction_count == 1);
    assert(diagnostic.reconstructed_after_collision);
    assert(!diagnostic.reconstructed_before_collision);
    assert(diagnostic.collision.substeps > 0);
    assert(diagnostic.collision.accepted_pairs > 0);
    assert(state.reconstructed_f_current);
    assert(close(diagnostic.marker_mass_before, diagnostic.marker_mass_after));
    const ev::KineticInvariantLedger ledger =
        ev::evaluate_kinetic_invariant_ledger(
            state.reconstructed_f, geometry, p);
    assert(close(ledger.mass, diagnostic.marker_mass_after));
}

void test_potential_first_retains_collision_age() {
    const ev::Params p = event_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    ev::WeightedPICBoltzmannEventOptions event_options;
    event_options.potential_fraction = 1.0;
    event_options.collision_fraction = 1.0;
    event_options.maximum_orbit_interval = 0.01;
    event_options.threads = 2;
    ev::WeightedPICCollisionOptions collision_options;
    collision_options.target_shell_markers = 256;
    collision_options.minimum_shell_markers = 16;
    collision_options.fail_on_nonlocal_shell = false;
    ev::WeightedPICBoltzmannEventState state = make_state(
        p, grid, geometry, event_options, collision_options);
    state.potential_timescale = 0.005;
    state.collision_timescale = 1.0;
    const ev::CharacteristicIntegratorPlan orbit_plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::WeightedPICBoltzmannEventDiagnostics diagnostic =
        ev::advance_weighted_pic_boltzmann_event(
            state,
            grid,
            geometry,
            p,
            orbit_plan,
            event_options,
            collision_options,
            0.02);
    assert(diagnostic.plan.order
           == ev::WeightedPICEventOrder::reconstruction_only);
    assert(diagnostic.reconstruction_count == 1);
    assert(diagnostic.reconstructed_before_collision);
    assert(state.reconstructed_f_current);
    assert(close(state.clock.potential_age, 0.0));
    assert(close(state.clock.collision_age, 0.005));
}

void test_coincident_potential_first_marks_cache_stale() {
    const ev::Params p = event_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    ev::WeightedPICBoltzmannEventOptions event_options;
    event_options.potential_fraction = 1.0;
    event_options.collision_fraction = 1.0;
    event_options.maximum_orbit_interval = 0.005;
    event_options.threads = 2;
    ev::WeightedPICCollisionOptions collision_options;
    collision_options.target_shell_markers = 256;
    collision_options.minimum_shell_markers = 16;
    collision_options.fail_on_nonlocal_shell = false;
    ev::WeightedPICBoltzmannEventState state = make_state(
        p, grid, geometry, event_options, collision_options);
    state.potential_timescale = 0.005;
    state.collision_timescale = 0.005;
    const ev::CharacteristicIntegratorPlan orbit_plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::WeightedPICBoltzmannEventDiagnostics diagnostic =
        ev::advance_weighted_pic_boltzmann_event(
            state,
            grid,
            geometry,
            p,
            orbit_plan,
            event_options,
            collision_options,
            0.02);
    assert(diagnostic.plan.order
           == ev::WeightedPICEventOrder::reconstruction_then_collision);
    assert(diagnostic.reconstruction_count == 1);
    assert(diagnostic.reconstructed_before_collision);
    assert(!state.reconstructed_f_current);
    const ev::WeightedPICPhaseRemapDiagnostics synchronized =
        ev::synchronize_weighted_pic_boltzmann_distribution(
            state, geometry, p);
    assert(state.reconstructed_f_current);
    assert(close(
        synchronized.active_mass, diagnostic.marker_mass_after));
}

void test_unequal_mass_event_initialization_collision_and_remap() {
    const ev::Params p = event_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    std::vector<ev::WeightedPICMarker> markers =
        event_markers(geometry, p);
    std::vector<double> mass_by_id(markers.size());
    long double expected_mass = 0.0L;
    for (ev::WeightedPICMarker& marker : markers) {
        marker.mass *= 0.5
            + 0.5 * static_cast<double>(1 + marker.id % 5);
        mass_by_id[static_cast<std::size_t>(marker.id)] = marker.mass;
        expected_mass += static_cast<long double>(marker.mass);
    }

    ev::WeightedPICBoltzmannEventOptions event_options;
    event_options.potential_fraction = 1.0;
    event_options.collision_fraction = 1.0;
    event_options.maximum_orbit_interval = 0.01;
    event_options.threads = 2;
    ev::WeightedPICCollisionOptions collision_options;
    collision_options.target_shell_markers = 256;
    collision_options.minimum_shell_markers = 16;
    collision_options.fail_on_nonlocal_shell = false;
    ev::WeightedPICBoltzmannEventState state =
        ev::initialize_weighted_pic_boltzmann_event_state(
            markers,
            0.0,
            0,
            {},
            {UINT64_C(0xabcddcba), 0},
            grid,
            geometry,
            p,
            event_options,
            collision_options);
    assert(close(
        ev::weighted_pic_total_marker_mass(state.markers),
        static_cast<double>(expected_mass)));

    state.potential_timescale = 1.0;
    state.collision_timescale = 0.01;
    const ev::CharacteristicIntegratorPlan orbit_plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::WeightedPICBoltzmannEventDiagnostics diagnostic =
        ev::advance_weighted_pic_boltzmann_event(
            state,
            grid,
            geometry,
            p,
            orbit_plan,
            event_options,
            collision_options,
            0.02);
    assert(diagnostic.plan.order
           == ev::WeightedPICEventOrder::collision_then_reconstruction);
    assert(diagnostic.collision.accepted_pairs > 0);
    assert(state.reconstructed_f_current);
    for (const ev::WeightedPICMarker& marker : state.markers) {
        assert(marker.mass
               == mass_by_id[static_cast<std::size_t>(marker.id)]);
    }
    const ev::KineticInvariantLedger ledger =
        ev::evaluate_kinetic_invariant_ledger(
            state.reconstructed_f, geometry, p);
    assert(close(ledger.mass, static_cast<double>(expected_mass)));
    assert(close(diagnostic.marker_mass_before, diagnostic.marker_mass_after));
}

void test_deferred_phase_projection_leaves_physical_trajectory_unchanged() {
    const ev::Params p = event_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    ev::WeightedPICBoltzmannEventOptions eager_options;
    eager_options.potential_fraction = 1.0;
    eager_options.collision_fraction = 1.0;
    eager_options.maximum_orbit_interval = 0.005;
    eager_options.threads = 2;
    ev::WeightedPICBoltzmannEventOptions deferred_options = eager_options;
    deferred_options.phase_space_projection_at_events = false;
    ev::WeightedPICCollisionOptions collision_options;
    collision_options.target_shell_markers = 256;
    collision_options.minimum_shell_markers = 16;
    collision_options.fail_on_nonlocal_shell = false;

    ev::WeightedPICBoltzmannEventState eager = make_state(
        p, grid, geometry, eager_options, collision_options);
    ev::WeightedPICBoltzmannEventState deferred = make_state(
        p, grid, geometry, deferred_options, collision_options);
    eager.potential_timescale = deferred.potential_timescale = 0.005;
    eager.collision_timescale = deferred.collision_timescale = 0.005;
    const ev::CharacteristicIntegratorPlan orbit_plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::WeightedPICBoltzmannEventDiagnostics eager_diagnostic =
        ev::advance_weighted_pic_boltzmann_event(
            eager,
            grid,
            geometry,
            p,
            orbit_plan,
            eager_options,
            collision_options,
            0.02);
    const ev::WeightedPICBoltzmannEventDiagnostics deferred_diagnostic =
        ev::advance_weighted_pic_boltzmann_event(
            deferred,
            grid,
            geometry,
            p,
            orbit_plan,
            deferred_options,
            collision_options,
            0.02);
    assert(eager_diagnostic.plan.order
           == ev::WeightedPICEventOrder::reconstruction_then_collision);
    assert(eager_diagnostic.reconstruction_count == 1);
    assert(deferred_diagnostic.reconstruction_count == 0);
    assert(eager.reconstructed_f_current == false);
    assert(deferred.reconstructed_f_current == false);
    assert_same_marker_state(eager, deferred);
}

void test_restart_retains_frozen_field_and_event_schedule() {
    auto p = event_params();
    p.bp_sigma0 = 350.0;
    const auto geometry = ev::make_cell_average_eta_geometry(
        ev::make_positive_radial_cell_axis(p), p);
    const auto grid = ev::make_weighted_pic_field_grid(p, p.Nr);
    ev::WeightedPICBoltzmannEventOptions options;
    options.phase_space_projection_at_events = false;
    options.maximum_orbit_interval = 0.001;
    options.potential_fraction = options.collision_fraction = 1.0;
    options.adaptive_collision_shell_timestep = true;
    options.adaptive_collision_maximum_interval = 0.001;
    options.threads = 2;
    ev::WeightedPICCollisionOptions collisions;
    collisions.target_shell_markers = 256;
    collisions.minimum_shell_markers = 16;
    collisions.adaptive_shell_timestep = true;
    auto state = make_state(p, grid, geometry, options, collisions);
    state.potential_timescale = 1.0;
    const auto plan = ev::make_characteristic_integrator_plan(p);
    const auto first = ev::advance_weighted_pic_boltzmann_event(
        state, grid, geometry, p, plan, options, collisions);
    assert(!first.plan.potential_due);
    assert(first.collision.accepted_pairs > 0);

    ev::WeightedPICRestartState saved;
    saved.markers = state.markers;
    saved.collision_sampler = state.collision_sampler;
    saved.time = state.time;
    saved.step = state.step;
    saved.potential_event_age = state.clock.potential_age;
    saved.collision_event_age = state.clock.collision_age;
    saved.potential_timescale_estimate = state.potential_timescale;
    saved.collision_timescale_estimate = state.collision_timescale;
    saved.has_frozen_field = true;
    saved.frozen_field = state.field;
    saved.frozen_field_contract = "frozen-field-regression-v1";
    const auto path = std::filesystem::temp_directory_path()
        / "venus_boltzmann_frozen_restart_test.bin";
    ev::write_weighted_pic_restart(path, saved);
    const auto loaded = ev::read_weighted_pic_restart(path);
    std::filesystem::remove(path);
    assert(loaded.has_frozen_field);
    auto resumed = ev::initialize_weighted_pic_boltzmann_event_state(
        loaded.markers, loaded.time, loaded.step,
        {loaded.potential_event_age, loaded.collision_event_age},
        loaded.collision_sampler, grid, geometry, p, options, collisions);
    // Reinitialization alone inserts an extra potential solve. This assertion
    // makes the test sensitive to an omitted frozen-field state.
    assert(resumed.field.phi_r != state.field.phi_r);
    bool rejected = false;
    try {
        ev::restore_weighted_pic_boltzmann_frozen_field(resumed, loaded, "changed");
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(ev::restore_weighted_pic_boltzmann_frozen_field(
        resumed, loaded, saved.frozen_field_contract));
    assert_same_marker_state(state, resumed);
    for (int i = 0; i < 12; ++i) {
        const auto a = ev::advance_weighted_pic_boltzmann_event(
            state, grid, geometry, p, plan, options, collisions);
        const auto b = ev::advance_weighted_pic_boltzmann_event(
            resumed, grid, geometry, p, plan, options, collisions);
        assert(a.collision.accepted_pairs == b.collision.accepted_pairs);
        assert(a.plan.order == b.plan.order);
        assert_same_marker_state(state, resumed);
    }

    // A truncated extension is a malformed v7 checkpoint, and cannot silently
    // fall back to the particle-only import.
    ev::write_weighted_pic_restart(path, saved);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 8);
    rejected = false;
    try { (void) ev::read_weighted_pic_restart(path); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::filesystem::remove(path);
    saved.has_frozen_field = false;
    ev::write_weighted_pic_restart(path, saved);
    const auto legacy = ev::read_weighted_pic_restart(path);
    assert(!legacy.has_frozen_field);
    assert(!ev::restore_weighted_pic_boltzmann_frozen_field(
        resumed, legacy, saved.frozen_field_contract));
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_collision_first_then_reconstruction();
    test_potential_first_retains_collision_age();
    test_coincident_potential_first_marks_cache_stale();
    test_unequal_mass_event_initialization_collision_and_remap();
    test_deferred_phase_projection_leaves_physical_trajectory_unchanged();
    test_restart_retains_frozen_field_and_event_schedule();
    std::cout << "weighted PIC Boltzmann event integration tests passed\n";
    return 0;
}
