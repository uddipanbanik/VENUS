// Tests turning points, periods, directed phase inversion, and hidden-return
// geometry used by the canonical spherical orbit atlas.

#include "ev/canonical_orbit_phase_transport.hpp"

#include <cassert>
#include <iostream>

int main() {
    ev::HamiltonianOrbitPhaseRow row;
    row.eta = 0.2;
    row.energy = 1.5;
    row.radial_period = ev::PI;
    row.cell_average = {0.3, 1.0, 2.0, 0.7, 0.2, 0.5, 1.4, 0.9};
    const double initial_mean = ev::hamiltonian_orbit_phase_row_mean(row);
    ev::HamiltonianOrbitPhaseRow accumulated = row;
    for (int step = 0; step < 10000; ++step) {
        ev::advance_hamiltonian_orbit_phase_row(
            accumulated, 0.001 * row.radial_period);
    }
    ev::HamiltonianOrbitPhaseRow single = row;
    ev::advance_hamiltonian_orbit_phase_row(
        single, 10.0 * row.radial_period);
    assert(std::abs(accumulated.phase_offset - single.phase_offset)
        < 2.0e-12);
    assert(ev::hamiltonian_orbit_phase_row_mean(accumulated) == initial_mean);
    assert(std::abs(
        ev::hamiltonian_orbit_phase_integral(accumulated, 0.0, 1.0)
        - initial_mean) < 2.0e-15);

    constexpr double eta = 0.2;
    constexpr double energy = 1.5;
    auto effective_potential = [](double radius) {
        return 0.5 * radius * radius + eta / (radius * radius);
    };
    std::vector<double> scan_radius(513, 0.0);
    const double log_min = std::log(1.0e-2);
    const double log_max = std::log(10.0);
    for (std::size_t q = 0; q < scan_radius.size(); ++q) {
        scan_radius[q] = std::exp(
            log_min + (log_max - log_min) * static_cast<double>(q)
                / static_cast<double>(scan_radius.size() - 1));
    }
    const ev::HamiltonianOrbitPhaseGeometry geometry =
        ev::build_hamiltonian_orbit_phase_geometry(
            eta, energy, scan_radius, 128, effective_potential);
    const double root_discriminant = std::sqrt(energy * energy - 2.0 * eta);
    const double exact_pericenter = std::sqrt(energy - root_discriminant);
    const double exact_apocenter = std::sqrt(energy + root_discriminant);
    assert(std::abs(geometry.pericenter - exact_pericenter) < 2.0e-12);
    assert(std::abs(geometry.apocenter - exact_apocenter) < 2.0e-12);
    assert(std::abs(geometry.radial_period - ev::PI) < 2.0e-11);
    double maximum_energy_error = 0.0;
    double maximum_reversal_error = 0.0;
    double maximum_phase_error = 0.0;
    double maximum_cached_phase_error = 0.0;
    double maximum_table_phase_error = 0.0;
    double maximum_single_state_phase_error = 0.0;
    for (std::size_t q = 0; q < geometry.phase_center.size(); ++q) {
        const ev::CharacteristicState& state = geometry.phase_center[q].state;
        assert(geometry.phase_center[q].visible);
        maximum_energy_error = std::max(
            maximum_energy_error,
            std::abs(
                0.5 * state.w * state.w
                + effective_potential(state.r) - energy));
        const ev::CharacteristicState& reverse =
            geometry.phase_center[geometry.phase_center.size() - 1 - q].state;
        maximum_reversal_error = std::max(
            maximum_reversal_error,
            std::max(
                std::abs(state.r - reverse.r),
                std::abs(state.w + reverse.w)));
        const double represented_phase = ev::hamiltonian_orbit_phase_of_state(
            geometry, state, effective_potential);
        const double exact_phase = (static_cast<double>(q) + 0.5)
            / static_cast<double>(geometry.phase_center.size());
        maximum_phase_error = std::max(
            maximum_phase_error,
            std::abs(represented_phase - exact_phase));
        const auto cached = ev::hamiltonian_orbit_location_at_phase_cached(
            geometry, exact_phase, effective_potential);
        maximum_cached_phase_error = std::max(
            maximum_cached_phase_error,
            std::abs(ev::hamiltonian_orbit_phase_of_state(
                geometry, cached.state, effective_potential) - exact_phase));
        const auto table = ev::hamiltonian_orbit_location_at_phase_table(
            geometry, exact_phase, effective_potential);
        maximum_table_phase_error = std::max(
            maximum_table_phase_error,
            std::abs(ev::hamiltonian_orbit_phase_of_state(
                geometry, table.state, effective_potential) - exact_phase));
        const auto single_state = ev::hamiltonian_orbit_exact_phase_of_state(
            energy, state, scan_radius, effective_potential);
        maximum_single_state_phase_error = std::max(
            maximum_single_state_phase_error,
            std::abs(single_state.phase - exact_phase));
    }
    assert(maximum_energy_error < 3.0e-13);
    assert(maximum_reversal_error < 3.0e-13);
    assert(maximum_phase_error < 3.0e-13);
    assert(maximum_cached_phase_error < 3.0e-13);
    assert(maximum_table_phase_error < 2.0e-5);
    assert(maximum_single_state_phase_error < 3.0e-11);

    // The harmonic effective potential has radial period pi for every bound
    // orbit, including the nearly circular limit where the turning points
    // almost coincide.  This is the most cancellation-sensitive atlas row.
    constexpr double circular_eta = 7.515625;
    const double circular_minimum_energy = std::sqrt(2.0 * circular_eta);
    const double circular_energy = circular_minimum_energy
        + 1.890645182294204e-9;
    const double circular_minimum_radius =
        std::pow(2.0 * circular_eta, 0.25);
    auto circular_effective_potential = [](double radius) {
        return 0.5 * radius * radius
            + circular_eta / (radius * radius);
    };
    std::vector<double> circular_scan = scan_radius;
    circular_scan.push_back(circular_minimum_radius);
    std::sort(circular_scan.begin(), circular_scan.end());
    const auto circular_geometry =
        ev::build_hamiltonian_orbit_phase_geometry(
            circular_eta,
            circular_energy,
            circular_scan,
            64,
            circular_effective_potential);
    const double circular_period_error =
        std::abs(circular_geometry.radial_period - ev::PI);
    assert(circular_period_error < 2.0e-6);
    const double period_only = ev::hamiltonian_orbit_radial_period(
        energy, scan_radius, effective_potential);
    const double period_only_error = std::abs(
        period_only - geometry.radial_period);
    assert(period_only_error < 5.0e-10);

    auto radial_effective_potential = [](double radius) {
        return 0.5 * radius * radius;
    };
    std::vector<double> radial_scan(401, 0.0);
    for (std::size_t q = 0; q < radial_scan.size(); ++q) {
        radial_scan[q] = 0.1
            + 1.9 * static_cast<double>(q)
                / static_cast<double>(radial_scan.size() - 1);
    }
    constexpr double hidden_time = 0.2;
    const ev::HamiltonianOrbitPhaseGeometry hidden_geometry =
        ev::build_hamiltonian_orbit_phase_geometry(
            0.0,
            0.5,
            radial_scan,
            160,
            radial_effective_potential,
            hidden_time);
    assert(!hidden_geometry.inner_is_turning_point);
    assert(std::abs(hidden_geometry.hidden_return_time - hidden_time) < 1.0e-15);
    const double exact_visible_half = std::acos(0.1);
    assert(std::abs(
        hidden_geometry.radial_period
        - (2.0 * exact_visible_half + hidden_time)) < 3.0e-11);
    std::size_t hidden_centers = 0;
    for (const auto& location : hidden_geometry.phase_center) {
        hidden_centers += location.visible ? 0 : 1;
    }
    assert(hidden_centers > 0);
    const double hidden_mid_phase =
        (2.0 * hidden_geometry.visible_outgoing_time + 0.5 * hidden_time)
        / hidden_geometry.radial_period;
    const auto hidden_mid = ev::hamiltonian_orbit_location_at_phase(
        hidden_geometry, hidden_mid_phase, radial_effective_potential);
    assert(!hidden_mid.visible);
    assert(std::abs(hidden_mid.hidden_remaining_time - 0.5 * hidden_time)
        < 2.0e-15);
    const auto hidden_outward_boundary =
        ev::hamiltonian_orbit_location_at_phase_cached(
            hidden_geometry, 0.0, radial_effective_potential);
    const double hidden_inward_phase =
        2.0 * hidden_geometry.visible_outgoing_time
        / hidden_geometry.radial_period;
    const auto hidden_inward_boundary =
        ev::hamiltonian_orbit_location_at_phase_cached(
            hidden_geometry,
            hidden_inward_phase,
            radial_effective_potential);
    const double exact_boundary_speed = std::sqrt(1.0 - 0.1 * 0.1);
    assert(std::abs(hidden_outward_boundary.state.w - exact_boundary_speed)
        < 3.0e-13);
    assert(std::abs(hidden_inward_boundary.state.w + exact_boundary_speed)
        < 3.0e-13);
    for (const auto& location : hidden_geometry.phase_center) {
        if (!location.visible) {
            continue;
        }
        const double table_phase = ev::hamiltonian_orbit_phase_of_state(
            hidden_geometry, location.state, radial_effective_potential);
        const auto single_state = ev::hamiltonian_orbit_exact_phase_of_state(
            0.5,
            location.state,
            radial_scan,
            radial_effective_potential,
            hidden_time);
        assert(std::abs(single_state.phase - table_phase) < 3.0e-11);
    }

    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.kinetic_grid = "cell_average_eta";
    p.radial_grid = "log";
    p.r_min = 0.1;
    p.r_max = 2.0;
    p.w_max = 2.0;
    p.L_max = 2.0;
    p.Nr = 24;
    p.Nw = 32;
    p.NL = 2;
    const ev::PhaseSpaceCellGeometry collision_geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const auto lower_orbit = ev::build_hamiltonian_orbit_phase_geometry(
        eta, 1.45, scan_radius, 64, effective_potential);
    const auto upper_orbit = ev::build_hamiltonian_orbit_phase_geometry(
        eta, 1.55, scan_radius, 64, effective_potential);
    const auto lower_left = ev::hamiltonian_orbit_location_at_phase(
        lower_orbit, 0.10, effective_potential);
    const auto lower_right = ev::hamiltonian_orbit_location_at_phase(
        lower_orbit, 0.12, effective_potential);
    const auto upper_left = ev::hamiltonian_orbit_location_at_phase(
        upper_orbit, 0.10, effective_potential);
    const auto upper_right = ev::hamiltonian_orbit_location_at_phase(
        upper_orbit, 0.12, effective_potential);
    assert(lower_left.visible && lower_right.visible
        && upper_left.visible && upper_right.visible);
    ev::SLDGPolygon patch{
        {lower_left.state.r, lower_left.state.w},
        {upper_left.state.r, upper_left.state.w},
        {upper_right.state.r, upper_right.state.w},
        {lower_right.state.r, lower_right.state.w},
    };
    if (ev::sldg_polygon_signed_area(patch) < 0.0) {
        std::reverse(patch.begin(), patch.end());
    }
    std::vector<ev::CanonicalOverlap> overlap;
    const ev::CanonicalColumnLedger overlap_ledger =
        ev::canonical_orbit_append_visible_patch_overlaps(
            patch, 0, 0, collision_geometry, p, overlap);
    assert(!overlap.empty());
    assert(std::abs(overlap_ledger.residual)
        < 2.0e-12 * std::max(1.0, overlap_ledger.column_volume));
    for (const ev::CanonicalOverlap& entry : overlap) {
        assert(entry.volume > 0.0);
        for (double moment : entry.eulerian_p2_moment) {
            assert(std::isfinite(moment));
        }
    }

    ev::CanonicalOrbitAtlas atlas;
    ev::CanonicalAtlasBuildOptions atlas_options;
    atlas_options.phase_cells = 16;
    atlas_options.maximum_patch_depth = 0;
    atlas_options.mapped_boundary_segments = 16;
    atlas_options.mapped_volume_tolerance = 2.0e-3;
    const double atlas_eta = collision_geometry.eta.center.front();
    auto atlas_effective_potential = [atlas_eta](double radius) {
        return 0.5 * radius * radius + atlas_eta / (radius * radius);
    };
    const std::size_t atlas_row = ev::canonical_orbit_append_bound_row(
        atlas,
        0,
        1.45,
        1.55,
        scan_radius,
        collision_geometry,
        p,
        atlas_effective_potential,
        atlas_options);
    assert(atlas_row == 0);
    assert(atlas.row.size() == 1);
    assert(atlas.column.size()
        == static_cast<std::size_t>(atlas_options.phase_cells));
    double maximum_geometric_scale_error = 0.0;
    double maximum_patch_scale_error = 0.0;
    for (const auto& column : atlas.column) {
        maximum_geometric_scale_error = std::max(
            maximum_geometric_scale_error,
            std::abs(column.geometric_scale - 1.0));
        maximum_patch_scale_error = std::max(
            maximum_patch_scale_error,
            column.maximum_patch_scale_error);
        assert(std::abs(column.residual) < 1.0e-13);
    }
    const std::size_t eulerian_cells = static_cast<std::size_t>(p.Nr)
        * p.Nw * p.NL;
    constexpr double constant_f = 2.3;
    std::vector<double> constant_distribution(eulerian_cells, constant_f);
    const auto to_atlas = ev::canonical_orbit_project_from_eulerian(
        atlas, constant_distribution);
    const double expected_atlas_mass =
        constant_f * atlas.row.front().phase_volume;
    assert(std::abs(to_atlas.destination_mass - expected_atlas_mass)
        < 3.0e-12 * expected_atlas_mass);
    ev::canonical_orbit_advance_frozen(atlas, 0.371 * ev::PI);
    std::vector<double> projected_distribution;
    const auto from_atlas = ev::canonical_orbit_project_to_eulerian(
        atlas, collision_geometry, p, projected_distribution);
    assert(std::abs(from_atlas.residual)
        < 3.0e-12 * expected_atlas_mass);
    assert(std::abs(from_atlas.destination_mass - expected_atlas_mass)
        < 3.0e-12 * expected_atlas_mass);

    std::vector<double> equilibrium_distribution(eulerian_cells, 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            const double h = atlas_effective_potential(
                collision_geometry.radial.center[static_cast<std::size_t>(i)])
                + 0.5 * std::pow(
                    collision_geometry.radial_velocity.center[
                        static_cast<std::size_t>(j)],
                    2);
            equilibrium_distribution[ev::idx(i, j, 0, p)] = std::exp(-h);
        }
    }
    auto phase_density_for_segments = [&](int segments) {
        ev::CanonicalOrbitAtlas candidate;
        ev::CanonicalAtlasBuildOptions build = atlas_options;
        build.mapped_boundary_segments = segments;
        build.mapped_volume_tolerance = 2.0e-2;
        ev::canonical_orbit_append_bound_row(
            candidate,
            0,
            1.45,
            1.55,
            scan_radius,
            collision_geometry,
            p,
            atlas_effective_potential,
            build);
        ev::canonical_orbit_project_from_eulerian(
            candidate, equilibrium_distribution);
        std::vector<double> density(
            candidate.row.front().phase_mass.size(), 0.0);
        const double volume = candidate.row.front().phase_volume
            / static_cast<double>(density.size());
        for (std::size_t q = 0; q < density.size(); ++q) {
            density[q] = candidate.row.front().phase_mass[q] / volume;
        }
        return density;
    };
    const auto phase_density_8 = phase_density_for_segments(8);
    const auto phase_density_12 = phase_density_for_segments(12);
    const auto phase_density_16 = phase_density_for_segments(16);
    auto relative_l2 = [](const auto& first, const auto& second) {
        long double difference = 0.0L;
        long double reference = 0.0L;
        for (std::size_t q = 0; q < first.size(); ++q) {
            difference += std::pow(first[q] - second[q], 2);
            reference += std::pow(second[q], 2);
        }
        return std::sqrt(static_cast<double>(difference / reference));
    };
    const double boundary_difference_8_12 = relative_l2(
        phase_density_8, phase_density_12);
    const double boundary_difference_12_16 = relative_l2(
        phase_density_12, phase_density_16);
    assert(boundary_difference_12_16 < boundary_difference_8_12);

    ev::PhaseSpaceCellGeometry delayed_geometry = collision_geometry;
    delayed_geometry.eta.center.front() = 0.0;
    auto delayed_effective_potential = [](double radius) {
        return 0.5 * radius * radius;
    };
    ev::CanonicalOrbitAtlas delayed_atlas;
    ev::CanonicalAtlasBuildOptions delayed_options = atlas_options;
    delayed_options.maximum_patch_depth = 8;
    const std::size_t delayed_row = ev::canonical_orbit_append_bound_row(
        delayed_atlas,
        0,
        0.45,
        0.55,
        radial_scan,
        delayed_geometry,
        p,
        delayed_effective_potential,
        delayed_options,
        [](double) { return hidden_time; });
    assert(delayed_row == 0);
    ev::CanonicalOrbitAtlas completed_delayed_atlas = delayed_atlas;
    const auto completed_visible = ev::canonical_orbit_project_from_eulerian(
        completed_delayed_atlas, constant_distribution);
    const auto delayed_completion =
        ev::canonical_orbit_complete_stationary_hidden_phase(
            completed_delayed_atlas);
    const double completed_delayed_mass = completed_visible.destination_mass
        + delayed_completion.hidden_mass_added;
    std::vector<ev::CanonicalHiddenCellMoments> delayed_hidden_mass(
        delayed_atlas.column.size());
    double delayed_hidden_volume = 0.0;
    for (std::size_t q = 0; q < delayed_atlas.column.size(); ++q) {
        delayed_hidden_volume += delayed_atlas.column[q].hidden_volume;
        delayed_hidden_mass[q].mass = constant_f
            * delayed_atlas.column[q].hidden_volume;
        delayed_hidden_mass[q].energy_mass = delayed_atlas.row.front().energy
            * delayed_hidden_mass[q].mass;
        delayed_hidden_mass[q].energy2_mass = std::pow(
            delayed_atlas.row.front().energy, 2)
            * delayed_hidden_mass[q].mass;
    }
    assert(delayed_hidden_volume > 0.0);
    const auto delayed_to_atlas = ev::canonical_orbit_project_from_eulerian(
        delayed_atlas,
        constant_distribution,
        nullptr,
        &delayed_hidden_mass);
    const double expected_delayed_mass = constant_f
        * delayed_atlas.row.front().phase_volume;
    assert(delayed_completion.completed_rows == 1);
    assert(std::abs(completed_delayed_mass - expected_delayed_mass)
        < 5.0e-11 * expected_delayed_mass);
    assert(std::abs(delayed_to_atlas.destination_mass - expected_delayed_mass)
        < 5.0e-11 * expected_delayed_mass);
    std::vector<double> delayed_projected;
    const auto delayed_from_atlas = ev::canonical_orbit_project_to_eulerian(
        delayed_atlas, delayed_geometry, p, delayed_projected);
    assert(std::abs(delayed_from_atlas.residual)
        < 5.0e-11 * expected_delayed_mass);
    assert(std::abs(
        delayed_from_atlas.hidden_mass
        - constant_f * delayed_hidden_volume)
        < 5.0e-11 * expected_delayed_mass);

    ev::Params auto_p = p;
    auto_p.Nr = 12;
    auto_p.Nw = 16;
    auto_p.NL = 1;
    auto_p.L_max = std::sqrt(0.2);
    auto_p.kinetic_eta_grid = "uniform";
    const ev::PhaseSpaceCellGeometry auto_geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(auto_p), auto_p);
    ev::CanonicalAtlasBuildOptions auto_options;
    auto_options.phase_cells = 8;
    auto_options.mapped_boundary_segments = 8;
    auto_options.maximum_patch_depth = 2;
    auto_options.mapped_volume_tolerance = 2.0e-2;
    auto_options.eta_quadrature_points = 2;
    const auto auto_atlas_result = ev::build_canonical_bound_orbit_atlas(
        auto_geometry,
        auto_p,
        4,
        [](double radius) { return 0.5 * radius * radius; },
        0.0,
        auto_options);
    assert(auto_atlas_result.diagnostics.eta_rows_built == 2);
    assert(auto_atlas_result.diagnostics.energy_rows_built == 8);
    assert(auto_atlas_result.atlas.row.size() == 8);
    assert(auto_atlas_result.atlas.column.size() == 64);
    const std::size_t auto_cells = static_cast<std::size_t>(auto_p.Nr)
        * auto_p.Nw * auto_p.NL;
    std::vector<double> auto_f(auto_cells, 0.0);
    const double auto_eta = auto_geometry.eta.center.front();
    for (int i = 0; i < auto_p.Nr; ++i) {
        for (int j = 0; j < auto_p.Nw; ++j) {
            const double radius = auto_geometry.radial.center[
                static_cast<std::size_t>(i)];
            const double velocity = auto_geometry.radial_velocity.center[
                static_cast<std::size_t>(j)];
            const double h = 0.5 * radius * radius
                + auto_eta / (radius * radius) + 0.5 * velocity * velocity;
            auto_f[ev::idx(i, j, 0, auto_p)] = std::exp(-h);
        }
    }
    ev::CanonicalOrbitAtlas automatic_atlas = auto_atlas_result.atlas;
    const auto auto_to_atlas = ev::canonical_orbit_project_from_eulerian(
        automatic_atlas, auto_f);
    ev::canonical_orbit_advance_frozen(automatic_atlas, 0.37);
    std::vector<double> auto_projected;
    const auto auto_from_atlas = ev::canonical_orbit_project_to_eulerian(
        automatic_atlas, auto_geometry, auto_p, auto_projected);
    assert(std::abs(
        auto_from_atlas.destination_mass - auto_to_atlas.destination_mass)
        < 5.0e-12 * auto_to_atlas.destination_mass);

    struct DirectMapResult {
        std::vector<double> projected;
        double mass_residual = 0.0;
        double maximum_scale_error = 0.0;
        std::size_t overlaps = 0;
        std::size_t limited_cells = 0;
        std::vector<std::vector<double>> action_fraction_edge;
    };
    auto direct_map_result = [&](int energy_points) {
        ev::CanonicalAtlasBuildOptions direct_options = auto_options;
        direct_options.direct_tensor_quadrature = true;
        direct_options.mapping_energy_points = energy_points;
        direct_options.mapping_phase_points = 4;
        const auto built = ev::build_canonical_bound_orbit_atlas(
            auto_geometry,
            auto_p,
            4,
            [](double radius) { return 0.5 * radius * radius; },
            0.0,
            direct_options);
        ev::CanonicalOrbitAtlas direct_atlas = built.atlas;
        ev::canonical_orbit_project_from_eulerian(direct_atlas, auto_f);
        ev::canonical_orbit_initialize_stationary_phase(direct_atlas);
        DirectMapResult result;
        const auto projection = ev::canonical_orbit_project_to_eulerian(
            direct_atlas, auto_geometry, auto_p, result.projected);
        result.mass_residual = projection.residual;
        result.overlaps = direct_atlas.overlap.size();
        result.action_fraction_edge =
            ev::canonical_orbit_action_fraction_edges(
                direct_atlas, auto_p.NL);
        for (const auto& column : direct_atlas.column) {
            assert(column.partition_energy_moments_resolved);
            result.maximum_scale_error = std::max(
                result.maximum_scale_error,
                column.maximum_patch_scale_error);
        }
        for (const auto& direct_row : direct_atlas.row) {
            for (std::size_t q = 0; q < direct_row.phase_mass.size(); ++q) {
                const std::size_t cell = direct_row.first_cell + q;
                const auto mode = ev::canonical_energy_p2_reconstruction(
                    direct_row,
                    direct_row.phase_mass[q],
                    direct_row.phase_energy_mass[q],
                    direct_row.phase_energy2_mass[q],
                    direct_atlas.column[cell]);
                result.limited_cells += mode.positivity_theta < 1.0 ? 1 : 0;
            }
        }
        assert(std::abs(result.mass_residual)
            < 5.0e-12 * projection.source_mass);
        assert(result.overlaps < 10000);
        return result;
    };
    const DirectMapResult direct_q2 = direct_map_result(2);
    const DirectMapResult direct_q4 = direct_map_result(4);
    const DirectMapResult direct_q8 = direct_map_result(8);
    auto phase_space_relative_l1 = [&](const std::vector<double>& first,
                                       const std::vector<double>& second) {
        long double difference = 0.0L;
        long double reference = 0.0L;
        for (int i = 0; i < auto_p.Nr; ++i) {
            for (int j = 0; j < auto_p.Nw; ++j) {
                for (int k = 0; k < auto_p.NL; ++k) {
                    const std::size_t cell = ev::idx(i, j, k, auto_p);
                    const double volume = ev::canonical_cell_measure(
                        auto_geometry, i, j, k);
                    difference += volume
                        * std::abs(first[cell] - second[cell]);
                    reference += volume * std::abs(second[cell]);
                }
            }
        }
        return static_cast<double>(difference / reference);
    };
    const double direct_q2_q8 = phase_space_relative_l1(
        direct_q2.projected, direct_q8.projected);
    const double direct_q4_q8 = phase_space_relative_l1(
        direct_q4.projected, direct_q8.projected);
    assert(direct_q4_q8 < direct_q2_q8);
    ev::CanonicalAtlasBuildOptions preserved_options = auto_options;
    preserved_options.direct_tensor_quadrature = true;
    preserved_options.mapping_energy_points = 8;
    preserved_options.mapping_phase_points = 4;
    preserved_options.action_fraction_edge_by_eta =
        direct_q8.action_fraction_edge;
    const auto preserved_build = ev::build_canonical_bound_orbit_atlas(
        auto_geometry,
        auto_p,
        1,
        [](double radius) { return 0.5 * radius * radius; },
        0.0,
        preserved_options);
    const auto preserved_fraction = ev::canonical_orbit_action_fraction_edges(
        preserved_build.atlas, auto_p.NL);
    assert(preserved_fraction.size() == direct_q8.action_fraction_edge.size());
    for (std::size_t k = 0; k < preserved_fraction.size(); ++k) {
        assert(preserved_fraction[k].size()
            == direct_q8.action_fraction_edge[k].size());
        for (std::size_t q = 0; q < preserved_fraction[k].size(); ++q) {
            assert(std::abs(
                preserved_fraction[k][q]
                - direct_q8.action_fraction_edge[k][q]) < 2.0e-5);
        }
    }
    ev::CanonicalAtlasBuildOptions transfer_source_options = auto_options;
    transfer_source_options.direct_tensor_quadrature = true;
    transfer_source_options.mapping_energy_points = 8;
    transfer_source_options.mapping_phase_points = 4;
    auto transfer_source_build = ev::build_canonical_bound_orbit_atlas(
        auto_geometry,
        auto_p,
        4,
        [](double radius) { return 0.5 * radius * radius; },
        0.0,
        transfer_source_options);
    ev::canonical_orbit_project_from_eulerian(
        transfer_source_build.atlas, auto_f);
    ev::canonical_orbit_initialize_stationary_phase(
        transfer_source_build.atlas);
    std::vector<double> transfer_before;
    ev::canonical_orbit_project_to_eulerian(
        transfer_source_build.atlas,
        auto_geometry,
        auto_p,
        transfer_before);
    ev::CanonicalOrbitAtlas transfer_destination = preserved_build.atlas;
    const auto topology_transfer = ev::canonical_orbit_transfer_preserved_topology(
        transfer_source_build.atlas, transfer_destination);
    std::vector<double> transfer_after;
    ev::canonical_orbit_project_to_eulerian(
        transfer_destination, auto_geometry, auto_p, transfer_after);
    const double transfer_identity_l1 = phase_space_relative_l1(
        transfer_before, transfer_after);
    assert(std::abs(topology_transfer.residual) < 2.0e-14);
    assert(transfer_identity_l1 < 2.0e-6);
    std::cout << "Canonical orbit phase: period_error="
              << std::abs(geometry.radial_period - ev::PI)
              << " energy_error=" << maximum_energy_error
              << " reversal_error=" << maximum_reversal_error
              << " phase_error=" << maximum_phase_error
              << " cached_phase_error=" << maximum_cached_phase_error
              << " table_phase_error=" << maximum_table_phase_error
              << " circular_period_error=" << circular_period_error
              << " period_only_error=" << period_only_error
              << " hidden_centers=" << hidden_centers
              << " overlap_count=" << overlap.size()
              << " overlap_residual=" << overlap_ledger.residual
              << " atlas_overlaps=" << atlas.overlap.size()
              << " atlas_scale_error=" << maximum_geometric_scale_error
              << " atlas_patch_scale_error=" << maximum_patch_scale_error
              << " atlas_mass_residual=" << from_atlas.residual
              << " boundary_8_12=" << boundary_difference_8_12
              << " boundary_12_16=" << boundary_difference_12_16
              << " delayed_hidden_fraction="
              << delayed_hidden_volume / delayed_atlas.row.front().phase_volume
              << " delayed_mass_residual=" << delayed_from_atlas.residual
              << " automatic_rows=" << auto_atlas_result.atlas.row.size()
              << " automatic_overlaps=" << auto_atlas_result.atlas.overlap.size()
              << " automatic_mass_residual=" << auto_from_atlas.residual
              << " direct_q2_q8=" << direct_q2_q8
              << " direct_q4_q8=" << direct_q4_q8
              << " direct_q8_overlaps=" << direct_q8.overlaps
              << " direct_q8_scale=" << direct_q8.maximum_scale_error
              << " direct_q8_limited=" << direct_q8.limited_cells
              << " topology_transfer_l1=" << transfer_identity_l1
              << " accumulated_offset=" << accumulated.phase_offset
              << '\n';
    return 0;
}
