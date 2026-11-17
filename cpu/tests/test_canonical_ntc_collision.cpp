// Tests equal-parcel NTC sampling and conservative source ownership on the
// canonical orbit-phase representation, including restart continuity.

#include "ev/canonical_boltzmann_poisson_restart.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <set>

int main() {
    // Roundoff scales with represented mass and is independent of the chosen
    // mass unit. Some supported platforms store long double as binary64.
    const auto mass_roundoff_for = [](const auto& d) {
        return 16.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(d.physical_mass_before),
                        std::abs(d.physical_mass_after)});
    };
    assert(!ev::canonical_ntc_eta_nearly_equal(
        4.2321610030942754e-9, 4.2320423760083068e-9));
    assert(ev::canonical_ntc_eta_nearly_equal(
        4.2320423760083068e-9,
        std::nextafter(
            4.2320423760083068e-9,
            std::numeric_limits<double>::infinity())));
    for (int points = 1; points <= 12; ++points) {
        const auto check_radau = [&](const auto& rule) {
            for (int power = 0; power <= 2 * points - 2; ++power) {
                long double represented = 0.0L;
                for (std::size_t q = 0; q < rule.first.size(); ++q) {
                    represented += static_cast<long double>(rule.second[q])
                        * std::pow(
                            static_cast<long double>(rule.first[q]), power);
                }
                const long double exact = power % 2 == 0
                    ? 2.0L / static_cast<long double>(power + 1)
                    : 0.0L;
                assert(std::abs(represented - exact) < 2.0e-12L);
            }
        };
        check_radau(ev::phase_space_gauss_radau_left_rule(points));
        check_radau(ev::phase_space_gauss_radau_right_rule(points));
    }

    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.vp_universe = "static";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "power";
    p.kinetic_eta_power = 2.0;
    p.radial_grid = "log";
    p.inner_boundary = "delayed_return";
    p.r_min = 0.01;
    p.r_max = 4.0;
    p.w_max = 4.0;
    p.L_max = 6.0;
    p.Nr = 16;
    p.Nw = 40;
    p.NL = 24;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.2;
    p.bp_reversible_candidate_radius = 8;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);

    ev::CanonicalAtlasBuildOptions atlas_options;
    atlas_options.phase_cells = 8;
    atlas_options.mapped_boundary_segments = 16;
    atlas_options.maximum_patch_depth = 6;
    atlas_options.mapped_volume_tolerance = 2.0e-2;
    atlas_options.maximum_energy = 5.0;
    atlas_options.eta_quadrature_points = 2;
    auto construction = ev::build_canonical_bound_orbit_atlas(
        geometry,
        p,
        8,
        [](double radius) { return 0.5 * radius * radius; },
        0.0,
        atlas_options);
    assert(construction.diagnostics.eta_rows_built > 0);

    std::vector<double> equilibrium(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        const double radius = geometry.radial.center[static_cast<std::size_t>(i)];
        for (int j = 0; j < p.Nw; ++j) {
            const double velocity =
                geometry.radial_velocity.center[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const double eta = geometry.eta.center[static_cast<std::size_t>(k)];
                const double h = 0.5 * radius * radius
                    + eta / (radius * radius) + 0.5 * velocity * velocity;
                equilibrium[ev::idx(i, j, k, p)] = h < 2.5
                    ? std::exp(-h) : 0.0;
            }
        }
    }
    ev::CanonicalOrbitAtlas atlas = std::move(construction.atlas);
    const auto initial_projection = ev::canonical_orbit_project_from_eulerian(
        atlas, equilibrium);
    assert(initial_projection.destination_mass > 0.0);
    ev::CanonicalOrbitAtlas exact_lift_control = atlas;
    ev::CanonicalOrbitAtlas detached_control = atlas;
    ev::CanonicalOrbitAtlas signed_control = atlas;
    ev::CanonicalOrbitAtlas reconstruction_control = atlas;
    std::vector<double> control_grid;
    std::vector<ev::CanonicalHiddenCellMoments> control_hidden;
    const auto control_to_grid = ev::canonical_orbit_project_to_eulerian(
        reconstruction_control, geometry, p, control_grid, &control_hidden);
    const auto control_to_atlas =
        ev::canonical_orbit_project_from_eulerian_conservative(
            reconstruction_control,
            geometry,
            p,
            control_grid,
            nullptr,
            &control_hidden);
    assert(std::abs(control_to_atlas.residual) < 2.0e-13);
    std::vector<double> control_distribution;
    const auto control_final_projection = ev::canonical_orbit_project_to_eulerian(
        reconstruction_control, geometry, p, control_distribution);
    std::cerr << "canonical round-trip ledger: source="
              << control_final_projection.source_mass
              << " destination=" << control_final_projection.destination_mass
              << " visible=" << control_final_projection.visible_mass
              << " hidden=" << control_final_projection.hidden_mass
              << " escaped=" << control_final_projection.escaped_mass
              << " residual=" << control_final_projection.residual
              << '\n';
    assert(std::abs(
        control_final_projection.destination_mass
        + control_final_projection.hidden_mass
        - control_to_grid.source_mass) < 2.0e-11 * control_to_grid.source_mass);

    ev::CanonicalNTCCollisionOptions collision_options;
    collision_options.marker_budget = 32768;
    collision_options.markers_per_occupied_cell = 32;
    collision_options.threads = 4;
    collision_options.virtual_collision_batches = 4;
    collision_options.maximum_unrepresented_mass_fraction = 1.0e-10;
    collision_options.collision.target_shell_markers = 512;
    collision_options.collision.minimum_shell_markers = 16;
    collision_options.collision.collision_cfl = 1.0;
    collision_options.collision.maximum_substeps = 1;
    collision_options.collision.fail_on_nonlocal_shell = false;
    collision_options.remap.maximum_candidate_radius = 40;
    collision_options.remap.maximum_carrier_mass_fraction = 0.0;
    collision_options.retain_collision_products_as_detached_carriers = false;
    ev::CanonicalNTCCollisionState collision_state;
    const auto diagnostics = ev::apply_canonical_ntc_collision(
        atlas,
        geometry,
        0.01,
        p,
        collision_options,
        collision_state,
        [](double radius) { return 0.5 * radius * radius; });
    std::cerr << "direct_lift markers=" << diagnostics.initialization.marker_count
              << " trials=" << diagnostics.collision.trial_pairs
              << " accepted=" << diagnostics.collision.accepted_pairs
              << " shells=" << diagnostics.collision.shell_visits
              << " active=" << diagnostics.collision.active_markers
              << " min_shell=" << diagnostics.collision.minimum_shell_population
              << " limited=" << diagnostics.positivity_limited_pair_fraction
              << " open=" << diagnostics.physical_open_mass
              << " high_energy_open="
              << diagnostics.physical_high_energy_open_mass
              << " eta_unrepresented="
              << diagnostics.unrepresented_eta_mass
              << " missing_eta_row="
              << diagnostics.unrepresented_missing_eta_row_mass
              << " energy_unrepresented="
              << diagnostics.unrepresented_energy_mass
              << " orbit_failure="
              << diagnostics.unrepresented_orbit_geometry_mass
              << " source_failure=" << diagnostics.source_lift_failure_mass
              << " energy_quantization="
              << diagnostics.atlas_energy_quantization_error << '\n';
    assert(diagnostics.collision.accepted_pairs > 0);
    assert(diagnostics.virtual_collision_batches == 4);
    assert(diagnostics.collision.substeps == 4);
    assert(std::abs(diagnostics.physical_mass_residual)
        < mass_roundoff_for(diagnostics));
    assert(std::abs(diagnostics.physical_radial_momentum_residual) < 1.0e-10);
    assert(std::abs(diagnostics.physical_kinetic_energy_residual) < 1.0e-10);
    assert(std::abs(diagnostics.atlas_energy_quantization_error) < 1.0e-10);
    assert(diagnostics.unrepresented_eta_mass == 0.0);
    assert(diagnostics.unrepresented_missing_eta_row_mass == 0.0);
    assert(diagnostics.unrepresented_energy_mass == 0.0);
    assert(diagnostics.unrepresented_orbit_geometry_mass == 0.0);
    assert(diagnostics.positivity_roundoff_mass < 1.0e-13);
    assert(diagnostics.positivity_limited_pair_fraction < 1.0e-2);

    ev::CanonicalNTCCollisionOptions signed_options = collision_options;
    signed_options.virtual_collision_batches = 16;
    signed_options.retain_collision_products_as_detached_carriers = false;
    signed_options.retain_collision_increment_as_signed_carriers = true;
    ev::CanonicalNTCCollisionState signed_state;
    const double signed_atlas_mass_before =
        ev::canonical_orbit_atlas_mass(signed_control);
    const auto signed_diagnostics = ev::apply_canonical_ntc_collision(
        signed_control,
        geometry,
        0.001,
        p,
        signed_options,
        signed_state,
        [](double radius) { return 0.5 * radius * radius; });
    assert(signed_diagnostics.collision.accepted_pairs > 0);
    assert(signed_state.signed_increment_positive.size()
        == 2 * signed_diagnostics.collision.accepted_pairs);
    assert(signed_state.signed_increment_negative.empty());
    assert(std::abs(signed_diagnostics.physical_mass_residual)
        < mass_roundoff_for(signed_diagnostics));
    assert(std::abs(signed_diagnostics.physical_radial_momentum_residual)
        < 1.0e-10);
    assert(std::abs(signed_diagnostics.physical_kinetic_energy_residual)
        < 1.0e-10);
    long double signed_positive_mass = 0.0L;
    for (const auto& marker : signed_state.signed_increment_positive) {
        signed_positive_mass += marker.mass;
    }
    constexpr long double signed_phase_factor =
        8.0L * static_cast<long double>(ev::PI)
        * static_cast<long double>(ev::PI);
    const long double signed_withdrawn_mass = signed_phase_factor * (
        signed_atlas_mass_before
        - ev::canonical_orbit_atlas_mass(signed_control));
    const double signed_mass_partition_error = static_cast<double>(
        signed_withdrawn_mass - signed_positive_mass);
    std::cerr << "Canonical deferred-outgoing partition: dM="
              << signed_mass_partition_error
              << " dPr="
              << signed_diagnostics.physical_radial_momentum_residual
              << " dK="
              << signed_diagnostics.physical_kinetic_energy_residual << '\n';
    assert(std::abs(signed_mass_partition_error) < 1.0e-13);
    {
        ev::CanonicalOrbitAtlas transaction_atlas = signed_control;
        ev::CanonicalNTCCollisionState transaction_state;
        transaction_state.signed_increment_positive.push_back(
            signed_state.signed_increment_positive.front());
        transaction_state.signed_increment_negative.push_back(
            signed_state.signed_increment_positive.front());
        transaction_state.signed_increment_negative.front().mass = 10.0;
        transaction_state.signed_increment_negative.front().source_cell =
            std::numeric_limits<std::uint32_t>::max();
        const double transaction_mass_before =
            ev::canonical_orbit_atlas_mass(transaction_atlas);
        const auto positive_before =
            transaction_state.signed_increment_positive;
        const auto negative_before =
            transaction_state.signed_increment_negative;
        bool rejected = false;
        try {
            (void)ev::canonical_materialize_signed_collision_increment(
                transaction_atlas,
                geometry,
                p,
                0.0,
                signed_options.maximum_lift_phase_interpolation_cell_error,
                signed_options.threads,
                signed_options.open_compression,
                transaction_state,
                [](double radius) { return 0.5 * radius * radius; });
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected);
        assert(ev::canonical_orbit_atlas_mass(transaction_atlas)
            == transaction_mass_before);
        assert(transaction_state.signed_increment_positive.size()
            == positive_before.size());
        assert(transaction_state.signed_increment_negative.size()
            == negative_before.size());
        assert(transaction_state.signed_increment_positive.front().mass
            == positive_before.front().mass);
        assert(transaction_state.signed_increment_negative.front().mass
            == negative_before.front().mass);
    }
    const auto signed_materialization =
        ev::canonical_materialize_signed_collision_increment(
            signed_control,
            geometry,
            p,
            0.0,
            signed_options.maximum_lift_phase_interpolation_cell_error,
            signed_options.threads,
            signed_options.open_compression,
            signed_state,
            [](double radius) { return 0.5 * radius * radius; });
    assert(signed_state.signed_increment_positive.empty());
    assert(signed_state.signed_increment_negative.empty());
    std::cerr << "Canonical signed materialization: dM="
              << signed_materialization.mass_residual
              << " dH=" << signed_materialization.hamiltonian_residual
              << " fallback_mass="
              << signed_materialization.positivity_fallback_mass
              << " fallback_markers="
              << signed_materialization.positivity_fallback_markers << '\n';
    assert(std::abs(signed_materialization.mass_residual) < 1.0e-12);
    assert(std::abs(signed_materialization.hamiltonian_residual) < 1.0e-6);
    std::cerr << "Canonical signed NTC: events="
              << signed_diagnostics.collision.accepted_pairs
              << " partition_dM=" << signed_mass_partition_error
              << " pair_dPr="
              << signed_diagnostics.physical_radial_momentum_residual
              << " pair_dK="
              << signed_diagnostics.physical_kinetic_energy_residual
              << " materialized_dM="
              << signed_materialization.mass_residual
              << " materialized_dH="
              << signed_materialization.hamiltonian_residual << '\n';

    std::vector<double> final_distribution;
    const auto final_projection = ev::canonical_orbit_project_to_eulerian(
        atlas, geometry, p, final_distribution);
    long double open_mass = 0.0L;
    for (const ev::WeightedPICMarker& marker : collision_state.open_marker) {
        open_mass += marker.mass;
    }
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(ev::PI)
        * static_cast<long double>(ev::PI);
    assert(std::abs(
        final_projection.destination_mass
        + final_projection.hidden_mass
        + static_cast<double>(open_mass / phase_factor)
        - initial_projection.destination_mass)
        < 2.0e-11 * initial_projection.destination_mass);
    long double l1_change = 0.0L;
    long double collision_l1_change = 0.0L;
    long double l1_reference = 0.0L;
    for (std::size_t q = 0; q < equilibrium.size(); ++q) {
        l1_change += std::abs(final_distribution[q] - equilibrium[q]);
        collision_l1_change += std::abs(
            final_distribution[q] - control_distribution[q]);
        l1_reference += std::abs(equilibrium[q]);
    }
    std::cout << "Canonical NTC collision: markers="
              << diagnostics.initialization.marker_count
              << " trials=" << diagnostics.collision.trial_pairs
              << " accepted=" << diagnostics.collision.accepted_pairs
              << " mass_error=" << diagnostics.physical_mass_residual
              << " momentum_error="
              << diagnostics.physical_radial_momentum_residual
              << " energy_error="
              << diagnostics.physical_kinetic_energy_residual
              << " relative_l1_change="
              << static_cast<double>(l1_change / l1_reference)
              << " collision_relative_l1_change="
              << static_cast<double>(collision_l1_change / l1_reference)
              << " limited_pair_fraction="
              << diagnostics.positivity_limited_pair_fraction
              << " density_noise="
              << diagnostics.maximum_relative_density_noise
              << " heat_flux_noise="
              << diagnostics.maximum_relative_energy_flux_noise
              << " radial_energy_batch_error="
              << diagnostics.maximum_relative_collision_radial_energy_batch_error
              << " collision_heat_flux_batch_error="
              << diagnostics.maximum_relative_collision_heat_flux_batch_error
              << '\n';
    assert(diagnostics.outgoing_cached_phase_lifts > 0);

    ev::CanonicalNTCCollisionOptions detached_options = collision_options;
    detached_options.retain_collision_products_as_detached_carriers = true;
    ev::CanonicalNTCCollisionState detached_state;
    const double detached_initial_mass = 8.0 * ev::PI * ev::PI
        * ev::canonical_orbit_atlas_mass(detached_control);
    const auto detached_first = ev::apply_canonical_ntc_collision(
        detached_control,
        geometry,
        0.01,
        p,
        detached_options,
        detached_state,
        [](double radius) { return 0.5 * radius * radius; });
    assert(detached_first.collision.accepted_pairs > 0);
    assert(detached_first.outgoing_cached_phase_lifts == 0);
    assert(detached_first.outgoing_exact_phase_lifts == 0);
    assert(detached_first.outgoing_lift_seconds < 0.1);
    assert(!detached_state.detached_marker.empty());
    assert(std::abs(detached_first.physical_mass_residual)
        < mass_roundoff_for(detached_first));
    const double detached_mass_after_first = 8.0 * ev::PI * ev::PI
        * ev::canonical_orbit_atlas_mass(detached_control)
        + ev::canonical_detached_marker_mass(detached_state);
    assert(std::abs(detached_mass_after_first - detached_initial_mass)
        < 2.0e-13 * detached_initial_mass);
    const auto detached_second = ev::apply_canonical_ntc_collision(
        detached_control,
        geometry,
        0.001,
        p,
        detached_options,
        detached_state,
        [](double radius) { return 0.5 * radius * radius; });
    assert(detached_second.active_detached_collision_sources > 0);
    assert(std::abs(detached_second.physical_mass_residual)
        < mass_roundoff_for(detached_second));
    const double detached_mass_after_second = 8.0 * ev::PI * ev::PI
        * ev::canonical_orbit_atlas_mass(detached_control)
        + ev::canonical_detached_marker_mass(detached_state);
    assert(std::abs(detached_mass_after_second - detached_initial_mass)
        < 2.0e-13 * detached_initial_mass);
    std::cout << "Canonical detached NTC: first_events="
              << detached_first.collision.accepted_pairs
              << " detached_sources_second="
              << detached_second.active_detached_collision_sources
              << " detached_markers="
              << detached_state.detached_marker.size()
              << " mass_error="
              << detached_second.physical_mass_residual << '\n';

    ev::CanonicalNTCCollisionOptions exact_lift_options = collision_options;
    exact_lift_options.maximum_lift_phase_interpolation_cell_error = 1.0e-14;
    ev::CanonicalNTCCollisionState exact_lift_state;
    const auto exact_lift_diagnostics = ev::apply_canonical_ntc_collision(
        exact_lift_control,
        geometry,
        0.01,
        p,
        exact_lift_options,
        exact_lift_state,
        [](double radius) { return 0.5 * radius * radius; });
    std::vector<double> exact_lift_distribution;
    ev::canonical_orbit_project_to_eulerian(
        exact_lift_control, geometry, p, exact_lift_distribution);
    long double lift_difference = 0.0L;
    long double lift_reference = 0.0L;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = ev::idx(i, j, k, p);
                const double measure = ev::canonical_cell_measure(
                    geometry, i, j, k);
                lift_difference += measure * std::abs(
                    final_distribution[cell] - exact_lift_distribution[cell]);
                lift_reference += measure * std::abs(
                    exact_lift_distribution[cell]);
            }
        }
    }
    const double relative_lift_difference = static_cast<double>(
        lift_difference / lift_reference);
    std::cout << "Canonical cached lift: cached="
              << diagnostics.outgoing_cached_phase_lifts
              << " exact_fallback=" << diagnostics.outgoing_exact_phase_lifts
              << " relative_to_exact=" << relative_lift_difference
              << " control_cached="
              << exact_lift_diagnostics.outgoing_cached_phase_lifts
              << '\n';
    assert(relative_lift_difference < 1.0e-3);

    // A local elastic Boltzmann operator has an exact Maxwellian null space.
    // Initialize that state directly in canonical H cells so this test does
    // not measure an Eulerian initialization error.
    constexpr double maxwell_temperature = 0.2;
    ev::CanonicalAtlasBuildOptions maxwell_atlas_options = atlas_options;
    maxwell_atlas_options.stationary_distribution = [=](double, double h) {
        return 100.0 * std::exp(-h / maxwell_temperature);
    };
    maxwell_atlas_options.energy_partition_weight = [=](double, double h) {
        return std::exp(-0.5 * h / maxwell_temperature);
    };
    auto maxwell_construction = ev::build_canonical_bound_orbit_atlas(
        geometry,
        p,
        8,
        [](double radius) { return 0.5 * radius * radius; },
        0.0,
        maxwell_atlas_options);
    std::vector<double> maxwell_before;
    ev::canonical_orbit_project_to_eulerian(
        maxwell_construction.atlas, geometry, p, maxwell_before);
    ev::CanonicalNTCCollisionOptions maxwell_collision_options =
        collision_options;
    maxwell_collision_options.marker_budget = 32768;
    maxwell_collision_options.markers_per_occupied_cell = 32;
    maxwell_collision_options.maximum_unrepresented_mass_fraction = 1.0e-6;
    ev::Params maxwell_p = p;
    maxwell_p.bp_sigma0 = 2.0;
    struct MaxwellNullResult {
        double relative_l1 = 0.0;
        double event_mass_fraction = 0.0;
        double limited_fraction = 0.0;
        double open_fraction = 0.0;
        std::uint64_t trials = 0;
        std::uint64_t accepted = 0;
        std::size_t markers = 0;
    };
    auto maxwell_null = [&](int batches) {
        ev::CanonicalOrbitAtlas null_atlas = maxwell_construction.atlas;
        ev::CanonicalNTCCollisionOptions null_options =
            maxwell_collision_options;
        null_options.virtual_collision_batches = batches;
        ev::CanonicalNTCCollisionState null_state;
        const auto step = ev::apply_canonical_ntc_collision(
            null_atlas,
            geometry,
            5.0e-6,
            maxwell_p,
            null_options,
            null_state,
            [](double radius) { return 0.5 * radius * radius; });
        std::vector<double> after;
        ev::canonical_orbit_project_to_eulerian(
            null_atlas, geometry, p, after);
        long double difference = 0.0L;
        long double reference = 0.0L;
        for (int i = 0; i < p.Nr; ++i) {
            for (int j = 0; j < p.Nw; ++j) {
                for (int k = 0; k < p.NL; ++k) {
                    const std::size_t cell = ev::idx(i, j, k, p);
                    const double measure = ev::canonical_cell_measure(
                        geometry, i, j, k);
                    difference += measure * std::abs(
                        after[cell] - maxwell_before[cell]);
                    reference += measure * std::abs(maxwell_before[cell]);
                }
            }
        }
        MaxwellNullResult result;
        result.relative_l1 = static_cast<double>(difference / reference);
        result.event_mass_fraction = step.collision.represented_event_mass
            / step.physical_mass_before;
        result.limited_fraction = step.collision.represented_event_mass > 0.0
            ? step.positivity_limited_pair_mass
                / step.collision.represented_event_mass
            : 0.0;
        result.open_fraction = ev::weighted_pic_total_marker_mass(
            null_state.open_marker) / step.physical_mass_before;
        result.trials = step.collision.trial_pairs;
        result.accepted = step.collision.accepted_pairs;
        result.markers = step.initialization.marker_count;
        return result;
    };
    const MaxwellNullResult maxwell_k128 = maxwell_null(128);
    const MaxwellNullResult maxwell_k512 = maxwell_null(512);
    std::cout << "Canonical Maxwellian null: K128_l1="
              << maxwell_k128.relative_l1
              << " K512_l1=" << maxwell_k512.relative_l1
              << " trials=" << maxwell_k512.trials
              << " accepted=" << maxwell_k512.accepted
              << " markers=" << maxwell_k512.markers
              << " event_mass_fraction="
              << maxwell_k512.event_mass_fraction
              << " positivity_limited_fraction="
              << maxwell_k512.limited_fraction
              << " open_fraction=" << maxwell_k512.open_fraction
              << '\n';
    assert(maxwell_k512.accepted > 100);
    assert(maxwell_k512.limited_fraction < 1.0e-8);
    assert(maxwell_k512.relative_l1 < 7.0e-3);

    // Unequal-weight open carriers participate in the next collision call
    // together with the canonical orbit populations.
    const double canonical_marker_mass =
        diagnostics.initialization.represented_mass
        / static_cast<double>(diagnostics.initialization.marker_count);
    assert(canonical_marker_mass > 0.0);
    constexpr std::size_t injected_open_markers = 1024;
    for (std::size_t q = 0; q < injected_open_markers; ++q) {
        ev::WeightedPICMarker marker;
        marker.mass = 0.25 * canonical_marker_mass;
        marker.r = geometry.radial.center[6];
        marker.w = q % 2 == 0 ? 0.15 : -0.15;
        marker.eta = geometry.eta.center[2];
        marker.id = collision_state.next_open_marker_id++;
        marker.state = ev::WeightedPICMarkerState::active;
        collision_state.open_marker.push_back(marker);
    }
    const double injected_open_mass = ev::weighted_pic_total_marker_mass(
        collision_state.open_marker);
    const std::size_t expected_active_open_sources = static_cast<std::size_t>(
        std::count_if(
            collision_state.open_marker.begin(),
            collision_state.open_marker.end(),
            [](const ev::WeightedPICMarker& marker) {
                return marker.state == ev::WeightedPICMarkerState::active
                    && marker.mass > 0.0;
            }));
    const auto open_diagnostics = ev::apply_canonical_ntc_collision(
        atlas,
        geometry,
        0.001,
        p,
        collision_options,
        collision_state,
        [](double radius) { return 0.5 * radius * radius; });
    assert(open_diagnostics.active_open_collision_sources
        == expected_active_open_sources);
    assert(open_diagnostics.canonical_open_events > 0);
    assert(std::abs(open_diagnostics.physical_mass_residual)
        < mass_roundoff_for(open_diagnostics));
    assert(std::abs(open_diagnostics.physical_radial_momentum_residual)
        < 1.0e-10);
    assert(std::abs(open_diagnostics.physical_kinetic_energy_residual)
        < 1.0e-10);
    const double retained_open_mass = ev::weighted_pic_total_marker_mass(
        collision_state.open_marker);
    assert(retained_open_mass
        < injected_open_mass + open_diagnostics.physical_open_mass);
    std::set<std::uint64_t> open_ids;
    for (const ev::WeightedPICMarker& marker : collision_state.open_marker) {
        assert(open_ids.insert(marker.id).second);
    }
    assert(std::isfinite(
        open_diagnostics.maximum_relative_collision_radial_energy_batch_error));
    assert(std::isfinite(
        open_diagnostics.maximum_relative_collision_heat_flux_batch_error));
    std::cout << "Canonical weighted-open NTC: active_open="
              << open_diagnostics.active_open_collision_sources
              << " canonical_open_events="
              << open_diagnostics.canonical_open_events
              << " open_open_events=" << open_diagnostics.open_open_events
              << " generated_open_mass="
              << open_diagnostics.physical_open_mass
              << " retained_open_mass=" << retained_open_mass
              << " mass_error=" << open_diagnostics.physical_mass_residual
              << '\n';

    ev::CanonicalBoltzmannPoissonEventState restart_state;
    restart_state.atlas = atlas;
    restart_state.collision = collision_state;
    restart_state.collision.detached_marker = {
        detached_state.detached_marker.front()};
    restart_state.collision.signed_increment_positive = {
        detached_state.detached_marker.front()};
    restart_state.collision.signed_increment_negative = {
        detached_state.detached_marker.front()};
    restart_state.field.phi = {1.0, 2.0};
    restart_state.field.phi_r = {3.0, 4.0};
    restart_state.field.source_line_density = {5.0, 6.0};
    restart_state.field.source_line_density_linear = {0.5, 0.6};
    restart_state.field.source_line_density_quadratic = {0.05, 0.06};
    restart_state.poisson_reference = restart_state.field;
    restart_state.eulerian_f = final_distribution;
    restart_state.time = 0.125;
    restart_state.step = 17;
    restart_state.potential_age = 0.025;
    restart_state.collision_age = 0.0125;
    restart_state.potential_timescale = 4.0;
    restart_state.potential_deadline_age = 0.2;
    restart_state.collision_timescale = 2.0;
    restart_state.collision_majorant_timescale = 1.0;
    restart_state.initial_physical_mass =
        8.0 * ev::PI * ev::PI * ev::canonical_orbit_atlas_mass(atlas)
        + ev::canonical_total_carrier_mass(restart_state.collision);
    restart_state.hidden_physical_mass = 1.0e-8;
    restart_state.startup_field_iterations = 1;
    restart_state.startup_field_mismatch = 1.0e-4;
    const std::filesystem::path restart_path =
        std::filesystem::temp_directory_path()
        / "venus_canonical_ntc_roundtrip.bin";
    ev::write_canonical_boltzmann_poisson_restart(
        restart_path, restart_state);
    const ev::CanonicalBoltzmannPoissonEventState restored =
        ev::read_canonical_boltzmann_poisson_restart(restart_path);
    std::filesystem::remove(restart_path);
    assert(restored.time == restart_state.time);
    assert(restored.step == restart_state.step);
    assert(restored.potential_age == restart_state.potential_age);
    assert(restored.collision_age == restart_state.collision_age);
    assert(restored.atlas.row.size() == restart_state.atlas.row.size());
    assert(restored.atlas.column.size() == restart_state.atlas.column.size());
    assert(restored.atlas.overlap.size() == restart_state.atlas.overlap.size());
    assert(restored.collision.open_marker.size()
        == restart_state.collision.open_marker.size());
    assert(restored.collision.detached_marker.size()
        == restart_state.collision.detached_marker.size());
    assert(restored.collision.detached_marker.front().mass
        == restart_state.collision.detached_marker.front().mass);
    assert(restored.collision.signed_increment_positive.size() == 1);
    assert(restored.collision.signed_increment_negative.size() == 1);
    assert(restored.collision.signed_increment_positive.front().mass
        == restart_state.collision.signed_increment_positive.front().mass);
    assert(restored.collision.signed_increment_negative.front().mass
        == restart_state.collision.signed_increment_negative.front().mass);
    assert(restored.collision.next_open_marker_id
        == restart_state.collision.next_open_marker_id);
    assert(restored.field.phi == restart_state.field.phi);
    assert(restored.field.source_line_density_linear
        == restart_state.field.source_line_density_linear);
    assert(restored.field.source_line_density_quadratic
        == restart_state.field.source_line_density_quadratic);
    assert(restored.eulerian_f == restart_state.eulerian_f);
    assert(ev::canonical_orbit_atlas_mass(restored.atlas)
        == ev::canonical_orbit_atlas_mass(restart_state.atlas));
    return 0;
}
