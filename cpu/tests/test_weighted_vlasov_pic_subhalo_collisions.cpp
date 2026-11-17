// Unequal-mass Plummer subhalo tests: pair cross sections, whole-object
// scattering/capture, conservation, determinism, and restart state.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

#include "ev/weighted_vlasov_pic_population_control.hpp"
#include "ev/weighted_vlasov_pic_subhalo_collisions.hpp"

namespace {

ev::WeightedPICRadialMesh thin_shell_mesh() {
    ev::WeightedPICRadialMesh mesh;
    mesh.edge = {0.99, 1.01};
    const double q0 = std::pow(mesh.edge[0], 3) / 3.0;
    const double q1 = std::pow(mesh.edge[1], 3) / 3.0;
    mesh.shell_volume = {4.0 * ev::PI * (q1 - q0)};
    mesh.volume_coordinate_center = {0.5 * (q0 + q1)};
    mesh.venus_center = {
        std::cbrt(3.0 * mesh.volume_coordinate_center[0])};
    return mesh;
}

ev::WeightedPICMarker marker(
    double mass,
    double radial_velocity,
    std::uint64_t id,
    double internal_energy = 0.0) {
    ev::WeightedPICMarker result;
    result.mass = mass;
    result.r = 1.0;
    result.w = radial_velocity;
    result.eta = 0.0;
    result.id = id;
    result.internal_energy = internal_energy;
    result.internal_velocity_dispersion = std::sqrt(
        2.0 * internal_energy / (3.0 * mass));
    return result;
}

ev::WeightedPICSubhaloCollisionOptions capture_only() {
    ev::WeightedPICSubhaloCollisionOptions options;
    options.velocity_scale = 1.0;
    options.plummer_area_per_mass = 1.0;
    options.enable_scattering = false;
    options.enable_capture = true;
    options.low_speed_regularization = 0.05;
    options.strength = 1.0;
    options.collision_cfl = 1.0e9;
    options.maximum_substeps = 256;
    return options;
}

void test_kernel_asymptotes() {
    ev::WeightedPICSubhaloCollisionOptions options = capture_only();
    options.enable_scattering = true;
    const auto log_slope = [](double y1, double y2, double x1, double x2) {
        return std::log(y2 / y1) / std::log(x2 / x1);
    };
    const double scatter_slope = log_slope(
        ev::weighted_pic_subhalo_scattering_cross_section_per_mass(
            10.0, options),
        ev::weighted_pic_subhalo_scattering_cross_section_per_mass(
            20.0, options),
        10.0,
        20.0);
    const double capture_slope = log_slope(
        ev::weighted_pic_subhalo_capture_cross_section_per_mass(
            0.06, options),
        ev::weighted_pic_subhalo_capture_cross_section_per_mass(
            0.10, options),
        0.06,
        0.10);
    assert(std::abs(scatter_slope + 4.0) < 0.08);
    // The exact focused direct area contains both the dominant x^-2 term and
    // a small geometric constant, so the finite-interval slope approaches
    // -2 rather than being forced to equal it algebraically.
    assert(std::abs(capture_slope + 2.0) < 0.01);
    assert(
        ev::weighted_pic_subhalo_scattering_cross_section_per_mass(
            0.18, options) == 0.0);
    assert(
        ev::weighted_pic_subhalo_capture_cross_section_per_mass(
            1.0, options) == 0.0);
    const double crossover_capture =
        ev::weighted_pic_subhalo_capture_cross_section_per_mass(
            0.54101942984306961, options);
    const double crossover_scatter =
        ev::weighted_pic_subhalo_scattering_cross_section_per_mass(
            0.54101942984306961, options);
    assert(crossover_capture > 0.0);
    assert(crossover_scatter > 0.0);
    assert(crossover_capture / crossover_scatter > 0.8);
    assert(crossover_capture / crossover_scatter < 1.2);
}

void test_joint_kernel_majorant_bounds_interpolation() {
    ev::WeightedPICSubhaloCollisionOptions options = capture_only();
    options.enable_scattering = true;
    const double majorant =
        ev::weighted_pic_subhalo_u_sigma_majorant(options);
    for (int index = 0; index <= 200000; ++index) {
        const double log_speed =
            -8.0 + 16.0 * static_cast<double>(index) / 200000.0;
        const double speed = std::exp(log_speed);
        const double joint_kernel = speed
            * (ev::weighted_pic_subhalo_scattering_cross_section_per_mass(
                   speed, options)
               + ev::weighted_pic_subhalo_capture_cross_section_per_mass(
                   speed, options));
        assert(
            joint_kernel
            <= majorant
                * (1.0 + 32.0 * std::numeric_limits<double>::epsilon()));
    }
}

void test_pair_specific_majorant_bounds_unequal_mass_kernel() {
    ev::WeightedPICSubhaloCollisionOptions options = capture_only();
    options.enable_scattering = true;
    const std::array<double, 5> masses{
        1.0e-8, 3.0e-5, 0.2, 7.0, 1.0e3};
    const std::array<double, 5> dispersions{
        0.01, 0.04, 0.2, 0.7, 3.0};
    for (std::size_t first = 0; first < masses.size(); ++first) {
        for (std::size_t second = 0; second < masses.size(); ++second) {
            const double pair_sigma_squared =
                (std::sqrt(masses[first])
                     * dispersions[first] * dispersions[first]
                 + std::sqrt(masses[second])
                     * dispersions[second] * dispersions[second])
                / (std::sqrt(masses[first]) + std::sqrt(masses[second]));
            const double pair_sigma = std::sqrt(pair_sigma_squared);
            const double majorant =
                ev::weighted_pic_subhalo_pair_u_sigma_majorant(
                    masses[first],
                    masses[second],
                    dispersions[first],
                    dispersions[second],
                    options);
            for (int index = 0; index <= 4000; ++index) {
                const double x = std::exp(
                    -10.0 + 20.0 * static_cast<double>(index) / 4000.0);
                const double speed = x * pair_sigma;
                const double kernel = speed
                    * (ev::weighted_pic_subhalo_scattering_cross_section(
                           speed,
                           masses[first],
                           masses[second],
                           pair_sigma,
                           options)
                       + ev::weighted_pic_subhalo_capture_cross_section(
                           speed,
                           masses[first],
                           masses[second],
                           pair_sigma,
                           options));
                assert(
                    kernel
                    <= majorant
                        * (1.0
                           + 32.0
                               * std::numeric_limits<double>::epsilon()));
            }
        }
    }
}

void test_equal_weight_capture_merges_and_heats() {
    std::vector<ev::WeightedPICMarker> markers{
        marker(1.0, -0.2, 1, 0.3),
        marker(1.0, 0.2, 2, 0.5),
    };
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x11223344), 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        markers,
        thin_shell_mesh(),
        10.0,
        capture_only(),
        sampler,
        workspace,
        2);
    assert(stats.capture_events == 1);
    assert(stats.scattering_events == 0);
    assert(markers.size() == 1);
    assert(markers[0].mass == 2.0);
    assert(std::abs(markers[0].w) < 1.0e-14);
    assert(markers[0].eta < 1.0e-28);
    assert(std::abs(markers[0].internal_energy - 0.84) < 1.0e-13);
    assert(
        std::abs(
            markers[0].internal_velocity_dispersion
            - std::sqrt(0.28))
        < 1.0e-13);
    assert(std::abs(stats.orbital_kinetic_energy_change + 0.04) < 1.0e-13);
    assert(std::abs(stats.internal_energy_change - 0.04) < 1.0e-13);
    assert(stats.maximum_pair_momentum_error < 1.0e-13);
    assert(stats.maximum_pair_total_energy_error < 1.0e-13);
}

void test_unequal_weight_capture_conserves_mass_momentum_energy() {
    std::vector<ev::WeightedPICMarker> markers{
        marker(1.0, -0.1, 1, 0.1),
        marker(3.0, 0.1, 2, 0.6),
    };
    const double mass_before = ev::weighted_pic_total_marker_mass(markers);
    const double momentum_before =
        markers[0].mass * markers[0].w + markers[1].mass * markers[1].w;
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x55667788), 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        markers,
        thin_shell_mesh(),
        10.0,
        capture_only(),
        sampler,
        workspace,
        1);
    assert(stats.capture_events == 1);
    assert(markers.size() == 1);
    assert(ev::weighted_pic_total_marker_mass(markers) == mass_before);
    double momentum_after = 0.0;
    double total_internal = 0.0;
    for (const auto& value : markers) {
        momentum_after += value.mass * value.w;
        total_internal += value.internal_energy;
    }
    assert(std::abs(momentum_after - momentum_before) < 1.0e-13);
    assert(markers[0].mass == 4.0);
    assert(std::abs(markers[0].w - 0.05) < 1.0e-13);
    assert(std::abs(total_internal - 0.715) < 1.0e-13);
    const double expected_dispersion = std::sqrt(
        2.0 * 0.715 / (3.0 * 4.0));
    assert(std::abs(
        markers[0].internal_velocity_dispersion - expected_dispersion)
        < 1.0e-13);
    assert(stats.removed_markers == 1);
    assert(stats.spawned_markers == 0);
    assert(stats.maximum_pair_momentum_error < 1.0e-13);
    assert(stats.maximum_pair_total_energy_error < 1.0e-13);
}

void test_unequal_mass_scattering_keeps_whole_objects() {
    ev::WeightedPICSubhaloCollisionOptions options = capture_only();
    options.enable_capture = false;
    options.enable_scattering = true;
    std::vector<ev::WeightedPICMarker> markers{
        marker(1.0, 0.0, 1, 0.2),
        marker(2.0, 1.0, 2, 0.4),
    };
    const double mass_before = ev::weighted_pic_total_marker_mass(markers);
    const double kinetic_before = 1.0;
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x99aabbcc), 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        markers,
        thin_shell_mesh(),
        1.0,
        options,
        sampler,
        workspace,
        2);
    assert(stats.scattering_events == 1);
    assert(stats.capture_events == 0);
    assert(markers.size() == 2);
    assert(ev::weighted_pic_total_marker_mass(markers) == mass_before);
    double kinetic_after = 0.0;
    double internal_after = 0.0;
    for (const auto& value : markers) {
        kinetic_after += 0.5 * value.mass
            * (value.w * value.w + 2.0 * value.eta / (value.r * value.r));
        internal_after += value.internal_energy;
    }
    assert(std::abs(kinetic_after - kinetic_before) < 1.0e-13);
    assert(std::abs(internal_after - 0.6) < 1.0e-13);
    assert(stats.spawned_markers == 0);
    assert(markers[0].mass == 1.0);
    assert(markers[1].mass == 2.0);
    assert(stats.maximum_pair_momentum_error < 1.0e-13);
    assert(stats.maximum_pair_total_energy_error < 1.0e-13);
}

void test_restart_preserves_internal_energy() {
    ev::WeightedPICRestartState state;
    state.markers = {marker(1.25, 0.1, 7, 0.375)};
    state.time = 2.0;
    state.step = 4;
    state.collision_sampler = {
        UINT64_C(0x73756268616c6f31), UINT64_C(918273645)};
    state.collision_event_age = 1.875;
    state.collision_timescale_estimate = 0.75;
    state.potential_timescale_estimate = 1.25;
    const auto path = std::filesystem::temp_directory_path()
        / "venus_subhalo_collision_restart.bin";
    std::filesystem::remove(path);
    ev::write_weighted_pic_restart(path, state);
    const auto loaded = ev::read_weighted_pic_restart(path);
    std::filesystem::remove(path);
    assert(loaded.markers.size() == 1);
    assert(loaded.markers[0].internal_energy == 0.375);
    assert(
        loaded.markers[0].internal_velocity_dispersion
        == state.markers[0].internal_velocity_dispersion);
    assert(loaded.collision_sampler.seed == state.collision_sampler.seed);
    assert(
        loaded.collision_sampler.next_counter
        == state.collision_sampler.next_counter);
    assert(loaded.collision_event_age == state.collision_event_age);
    assert(
        loaded.collision_timescale_estimate
        == state.collision_timescale_estimate);
    assert(
        loaded.potential_timescale_estimate
        == state.potential_timescale_estimate);
}

void test_heated_dispersion_changes_the_next_kernel() {
    std::vector<ev::WeightedPICMarker> markers{
        marker(1.0, -0.2, 1, 0.3),
        marker(1.0, 0.2, 2, 0.5),
    };
    const double pair_sigma_before =
        ev::weighted_pic_subhalo_collision_detail::pair_velocity_scale(
            markers[0], markers[1]);
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0xa17e5eed), 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        markers,
        thin_shell_mesh(),
        10.0,
        capture_only(),
        sampler,
        workspace,
        1);
    assert(stats.capture_events == 1);
    const ev::WeightedPICMarker partner = marker(1.0, 0.0, 3, 0.3);
    const double pair_sigma_after =
        ev::weighted_pic_subhalo_collision_detail::pair_velocity_scale(
            markers[0], partner);
    assert(pair_sigma_after != pair_sigma_before);
    const double speed = 0.25;
    const double old_capture =
        ev::weighted_pic_subhalo_capture_cross_section_per_mass(
            speed, pair_sigma_before, capture_only());
    const double next_capture =
        ev::weighted_pic_subhalo_capture_cross_section_per_mass(
            speed, pair_sigma_after, capture_only());
    assert(next_capture != old_capture);
}

void test_unresolved_origin_core_is_collisional() {
    std::vector<ev::WeightedPICMarker> markers{
        marker(1.0, -0.2, 1, 0.3),
        marker(1.0, 0.2, 2, 0.5),
    };
    markers[0].r = 0.2;
    markers[1].r = 0.2;
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x1aa3c0de), 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        markers,
        thin_shell_mesh(),
        10.0,
        capture_only(),
        sampler,
        workspace,
        2);
    assert(stats.capture_events == 1);
    assert(stats.inner_capture_events == 1);
    assert(stats.inner_cell_visits > 0);
    assert(markers.size() == 1);
    assert(markers[0].r < thin_shell_mesh().edge.front());
}

void test_unequal_mass_pair_cross_section_is_symmetric() {
    auto options = capture_only();
    options.enable_scattering = true;
    const double expected_area =
        std::pow(1.0 + std::sqrt(3.0), 2) / 4.0;
    const double area_13 =
        ev::weighted_pic_subhalo_pair_area(1.0, 3.0, options);
    const double area_31 =
        ev::weighted_pic_subhalo_pair_area(3.0, 1.0, options);
    assert(std::abs(area_13 - expected_area) < 1.0e-14);
    assert(area_13 == area_31);
    for (double speed : {0.08, 0.3, 2.0, 20.0}) {
        const double scatter_13 =
            ev::weighted_pic_subhalo_scattering_cross_section(
                speed, 1.0, 3.0, 1.0, options);
        const double scatter_31 =
            ev::weighted_pic_subhalo_scattering_cross_section(
                speed, 3.0, 1.0, 1.0, options);
        const double capture_13 =
            ev::weighted_pic_subhalo_capture_cross_section(
                speed, 1.0, 3.0, 1.0, options);
        const double capture_31 =
            ev::weighted_pic_subhalo_capture_cross_section(
                speed, 3.0, 1.0, 1.0, options);
        assert(scatter_13 == scatter_31);
        assert(capture_13 == capture_31);
    }
}

void test_mass_ratio_changes_the_calibrated_outcome_shapes() {
    auto options = capture_only();
    options.enable_scattering = true;
    const double speed = 0.4;
    const double equal_area =
        ev::weighted_pic_subhalo_pair_area(1.0, 1.0, options);
    const double unequal_area =
        ev::weighted_pic_subhalo_pair_area(1.0, 0.1, options);
    const double equal_capture =
        ev::weighted_pic_subhalo_capture_cross_section(
            speed, 1.0, 1.0, 1.0, options) / equal_area;
    const double unequal_capture =
        ev::weighted_pic_subhalo_capture_cross_section(
            speed, 1.0, 0.1, 1.0, options) / unequal_area;
    const double equal_scatter =
        ev::weighted_pic_subhalo_scattering_cross_section(
            speed, 1.0, 1.0, 1.0, options) / equal_area;
    const double unequal_scatter =
        ev::weighted_pic_subhalo_scattering_cross_section(
            speed, 1.0, 0.1, 1.0, options) / unequal_area;

    // A universal equal-mass shape would make both pairs identical after
    // division by their pair area.  The independently evolved cores do not:
    // q remains a physical parameter of both capture and transport.
    assert(std::abs(unequal_capture / equal_capture - 1.0) > 0.1);
    assert(std::abs(unequal_scatter / equal_scatter - 1.0) > 0.1);
}

void test_accepted_table_resolves_the_extreme_mass_ratio_limit() {
    using namespace ev::weighted_pic_subhalo_plummer_table;
    assert(mass_ratio_count == 11);
    assert(speed_count == 72);
    assert(std::abs(mass_ratio.front() - 1.0e-6) < 1.0e-21);
    assert(mass_ratio.back() == 1.0);

    auto options = capture_only();
    options.enable_scattering = true;
    const auto normalized_capture = [&options](
            double second_mass,
            double speed_value) {
        const double area = ev::weighted_pic_subhalo_pair_area(
            1.0, second_mass, options);
        return ev::weighted_pic_subhalo_capture_cross_section(
            speed_value, 1.0, second_mass, 1.0, options) / area;
    };
    const auto normalized_scatter = [&options](
            double second_mass,
            double speed_value) {
        const double area = ev::weighted_pic_subhalo_pair_area(
            1.0, second_mass, options);
        return ev::weighted_pic_subhalo_scattering_cross_section(
            speed_value, 1.0, second_mass, 1.0, options) / area;
    };

    // The accepted calculation resolves the closing capture channel rather
    // than extending the equal-mass curve to arbitrarily small partners.
    assert(normalized_capture(1.0, 0.05) > 0.0);
    assert(normalized_capture(1.0e-4, 0.05) > 0.0);
    assert(normalized_capture(1.0e-5, 0.05) == 0.0);
    assert(normalized_capture(1.0e-6, 0.05) == 0.0);

    // Ratios below the calibrated domain use the accepted q=1e-6 limiting
    // shape.  Only the separately derived geometric pair area continues to
    // change with the two physical masses.
    for (double speed_value : {0.05, 0.2, 2.0, 40.0}) {
        const double at_floor = normalized_scatter(1.0e-6, speed_value);
        const double below_floor = normalized_scatter(1.0e-9, speed_value);
        assert(std::abs(at_floor - below_floor)
            <= 4.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(at_floor)));
    }

    // The high-speed continuation is matched continuously to every q row
    // and has the analytically required x^-4 transport asymptote.
    const double scatter_40 = normalized_scatter(1.0e-6, 40.0);
    const double scatter_80 = normalized_scatter(1.0e-6, 80.0);
    assert(std::abs(scatter_40 / scatter_80 - 16.0) < 1.0e-12);
}

void test_velocity_dependent_mean_free_path_recovers_frank_limit() {
    const double partner_number_density = 7.0;
    const double constant_pair_area = 3.0;
    const long double relative_speed_sum = 12.5L;
    const long double kernel_sum =
        relative_speed_sum * constant_pair_area;
    const double mean_free_path =
        ev::weighted_pic_subhalo_mean_free_path_from_pair_samples(
            partner_number_density,
            relative_speed_sum,
            kernel_sum);
    assert(std::abs(
        mean_free_path
            - 1.0 / (partner_number_density * constant_pair_area))
        < 1.0e-15);
}

void test_mean_free_path_uses_evolving_pair_velocity_and_cross_section() {
    const auto mesh = thin_shell_mesh();
    auto options = capture_only();
    options.enable_scattering = true;
    std::vector<ev::WeightedPICMarker> markers{
        marker(1.0, -0.2, 1, 0.015),
        marker(1.0, 0.2, 2, 0.015),
    };
    const double relative_speed = 0.4;
    const double pair_velocity_scale =
        ev::weighted_pic_subhalo_collision_detail::pair_velocity_scale(
            markers[0], markers[1]);
    const double pair_cross_section =
        ev::weighted_pic_subhalo_scattering_cross_section(
            relative_speed,
            1.0,
            1.0,
            pair_velocity_scale,
            options)
        + ev::weighted_pic_subhalo_capture_cross_section(
            relative_speed,
            1.0,
            1.0,
            pair_velocity_scale,
            options);
    // Each of the two objects has one possible partner, so the finite-cell
    // partner density is (N-1)/V rather than the continuum estimate N/V.
    const double partner_number_density =
        1.0 / mesh.shell_volume.front();
    const double expected_mean_free_path =
        1.0 / (partner_number_density * pair_cross_section);

    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x6d667074657374), 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        markers,
        mesh,
        1.0e-12,
        options,
        sampler,
        workspace,
        1);
    assert(stats.mean_free_path_pair_samples == 1);
    assert(std::abs(
        stats.minimum_mean_free_path / expected_mean_free_path - 1.0)
        < 1.0e-13);
    const double expected_collision_time =
        expected_mean_free_path / relative_speed;
    assert(std::abs(
        stats.minimum_collision_timescale / expected_collision_time - 1.0)
        < 1.0e-13);
    const double expected_orbital_dispersion = std::sqrt(0.04 / 3.0);
    assert(std::abs(
        stats.minimum_orbital_velocity_dispersion
            - expected_orbital_dispersion)
        < 1.0e-14);
    assert(
        stats.maximum_orbital_velocity_dispersion
        == stats.minimum_orbital_velocity_dispersion);
    assert(stats.nonlocal_cells
           == static_cast<std::size_t>(
               (mesh.edge.back() - mesh.edge.front())
                   > expected_mean_free_path));
}

void test_uniform_core_equal_volume_shell_geometry() {
    // The initial origin core is uniform.  Dividing its volume into I equal
    // pieces places the first physical face at r_min I^{-1/3}; this central
    // piece is the widest.  This test is deliberately geometric: for a
    // velocity-dependent capture kernel, lambda_mfp may also decrease toward
    // the centre, so more shells need not improve Delta r/lambda_mfp.
    const auto mesh = thin_shell_mesh();
    auto options = capture_only();
    options.collision_lanes = 8;
    for (const int inner_cells : {1, 8, 64}) {
        options.inner_collision_cells = inner_cells;
        const double expected_width =
            mesh.edge.front() / std::cbrt(static_cast<double>(inner_cells));
        const double measured_width =
            ev::weighted_pic_subhalo_collision_detail::collision_cell_width(
                0, mesh, options);
        assert(std::abs(measured_width / expected_width - 1.0) < 1.0e-15);

        long double core_volume = 0.0L;
        const std::size_t inner_lane_count =
            static_cast<std::size_t>(inner_cells * options.collision_lanes);
        for (std::size_t cell = 0; cell < inner_lane_count; ++cell) {
            core_volume +=
                ev::weighted_pic_subhalo_collision_detail::collision_cell_volume(
                    cell, mesh, options);
        }
        const long double expected_volume =
            4.0L * ev::PI * std::pow(mesh.edge.front(), 3) / 3.0L;
        assert(std::abs(
            static_cast<double>(core_volume / expected_volume - 1.0L))
            < 64.0 * std::numeric_limits<double>::epsilon());
    }
}

void test_collision_lane_rate_estimator_is_unbiased() {
    const auto mesh = thin_shell_mesh();
    auto options = capture_only();
    options.collision_lanes = 8;
    std::vector<ev::WeightedPICMarker> markers;
    markers.reserve(32);
    for (std::uint64_t index = 0; index < 32; ++index) {
        auto value = marker(
            1.0 + static_cast<double>((7 * index) % 19),
            0.0,
            index + 1,
            0.3);
        value.r = 1.0;
        markers.push_back(value);
    }
    long double exact_pair_weight = 0.0L;
    for (std::size_t first = 0; first < markers.size(); ++first) {
        for (std::size_t second = first + 1;
             second < markers.size();
             ++second) {
            exact_pair_weight += ev::weighted_pic_subhalo_pair_area(
                markers[first].mass,
                markers[second].mass,
                options);
        }
    }
    constexpr std::uint64_t epochs = 20000;
    long double estimated_sum = 0.0L;
    for (std::uint64_t epoch = 0; epoch < epochs; ++epoch) {
        std::array<std::vector<std::size_t>, 8> lane_members;
        for (std::size_t index = 0; index < markers.size(); ++index) {
            const std::size_t cell =
                ev::weighted_pic_subhalo_collision_detail::collision_cell(
                markers[index].r,
                markers[index].id,
                epoch,
                mesh,
                options);
            const std::size_t lane =
                cell % static_cast<std::size_t>(options.collision_lanes);
            lane_members[lane].push_back(index);
        }
        long double within_lane_weight = 0.0L;
        for (const auto& lane : lane_members) {
            for (std::size_t first = 0; first < lane.size(); ++first) {
                for (std::size_t second = first + 1;
                     second < lane.size();
                     ++second) {
                    within_lane_weight += ev::weighted_pic_subhalo_pair_area(
                        markers[lane[first]].mass,
                        markers[lane[second]].mass,
                        options);
                }
            }
        }
        estimated_sum += options.collision_lanes * within_lane_weight;
    }
    const long double estimated_mean = estimated_sum / epochs;
    const double relative_error = std::abs(
        static_cast<double>(
            estimated_mean / exact_pair_weight - 1.0L));
    assert(relative_error < 0.004);

    const std::size_t resolved_cell =
        static_cast<std::size_t>(options.inner_collision_cells)
        * static_cast<std::size_t>(options.collision_lanes);
    double lane_volume_sum = 0.0;
    for (int lane = 0; lane < options.collision_lanes; ++lane) {
        lane_volume_sum +=
            ev::weighted_pic_subhalo_collision_detail::collision_cell_volume(
            resolved_cell + static_cast<std::size_t>(lane),
            mesh,
            options);
    }
    assert(std::abs(
        lane_volume_sum / mesh.shell_volume.front() - 1.0)
        < 32.0 * std::numeric_limits<double>::epsilon());
}

void test_inner_collision_lanes_are_parallel_deterministic() {
    std::vector<ev::WeightedPICMarker> serial_markers;
    serial_markers.reserve(1024);
    for (std::uint64_t index = 0; index < 1024; ++index) {
        auto value = marker(
            1.0e-4,
            index % 2 == 0 ? -0.05 : 0.05,
            index + 1,
            6.0e-6);
        value.r = 0.2 + 1.0e-5 * static_cast<double>(index);
        serial_markers.push_back(value);
    }
    std::vector<ev::WeightedPICMarker> parallel_markers = serial_markers;
    auto options = capture_only();
    options.collision_cfl = 0.2;
    options.collision_lanes = 16;
    options.pair_sampling =
        ev::WeightedPICSubhaloPairSampling::maximum_weight_importance;
    ev::WeightedPICCollisionSamplerState serial_sampler{
        UINT64_C(0xc0111510a9e5), 0};
    ev::WeightedPICCollisionSamplerState parallel_sampler = serial_sampler;
    ev::WeightedPICSubhaloCollisionWorkspace serial_workspace;
    ev::WeightedPICSubhaloCollisionWorkspace parallel_workspace;
    const auto serial_stats =
        ev::apply_weighted_pic_subhalo_collision_interval(
            serial_markers,
            thin_shell_mesh(),
            1.0,
            options,
            serial_sampler,
            serial_workspace,
            1);
    const auto parallel_stats =
        ev::apply_weighted_pic_subhalo_collision_interval(
            parallel_markers,
            thin_shell_mesh(),
            1.0,
            options,
            parallel_sampler,
            parallel_workspace,
            8);
    assert(serial_stats.capture_events > 0);
    assert(serial_stats.inner_capture_events == serial_stats.capture_events);
    assert(serial_stats.inner_cell_visits > 1);
    assert(
        serial_stats.capture_events
        == parallel_stats.capture_events);
    assert(
        serial_stats.represented_capture_mass
        == parallel_stats.represented_capture_mass);
    assert(serial_markers.size() == parallel_markers.size());
    for (std::size_t index = 0; index < serial_markers.size(); ++index) {
        const auto& first = serial_markers[index];
        const auto& second = parallel_markers[index];
        assert(first.mass == second.mass);
        assert(first.r == second.r);
        assert(first.w == second.w);
        assert(first.eta == second.eta);
        assert(first.internal_energy == second.internal_energy);
        assert(
            first.internal_velocity_dispersion
            == second.internal_velocity_dispersion);
        assert(first.id == second.id);
        assert(first.state == second.state);
    }
}

void test_collision_restart_trajectory_is_deterministic() {
    std::vector<ev::WeightedPICMarker> uninterrupted;
    uninterrupted.reserve(512);
    for (std::uint64_t index = 0; index < 512; ++index) {
        auto value = marker(
            2.0e-4 * (1.0 + static_cast<double>(index % 5)),
            index % 2 == 0 ? -0.08 : 0.08,
            index + 1,
            1.2e-5 * (1.0 + static_cast<double>(index % 5)));
        value.r = index % 3 == 0 ? 0.2 : 1.0;
        uninterrupted.push_back(value);
    }
    std::vector<ev::WeightedPICMarker> restarted = uninterrupted;
    auto options = capture_only();
    options.enable_scattering = true;
    options.strength = 0.2;
    options.collision_cfl = 0.2;
    options.collision_lanes = 8;
    options.pair_sampling =
        ev::WeightedPICSubhaloPairSampling::maximum_weight_importance;
    ev::WeightedPICCollisionSamplerState uninterrupted_sampler{
        UINT64_C(0x5e57a47), 0};
    ev::WeightedPICCollisionSamplerState restarted_sampler =
        uninterrupted_sampler;
    ev::WeightedPICSubhaloCollisionWorkspace uninterrupted_workspace;
    ev::WeightedPICSubhaloCollisionWorkspace restarted_workspace;
    const auto first_uninterrupted =
        ev::apply_weighted_pic_subhalo_collision_interval(
            uninterrupted,
            thin_shell_mesh(),
            0.01,
            options,
            uninterrupted_sampler,
            uninterrupted_workspace,
            4);
    const auto first_restarted =
        ev::apply_weighted_pic_subhalo_collision_interval(
            restarted,
            thin_shell_mesh(),
            0.01,
            options,
            restarted_sampler,
            restarted_workspace,
            4);
    assert(
        first_uninterrupted.accepted_pairs
        == first_restarted.accepted_pairs);

    ev::WeightedPICRestartState state;
    state.markers = restarted;
    state.time = 0.01;
    state.step = 1;
    state.collision_sampler = restarted_sampler;
    state.collision_event_age = 0.0;
    const auto path = std::filesystem::temp_directory_path()
        / "venus_subhalo_collision_trajectory_restart.bin";
    std::filesystem::remove(path);
    ev::write_weighted_pic_restart(path, state);
    auto loaded = ev::read_weighted_pic_restart(path);
    std::filesystem::remove(path);

    const auto second_uninterrupted =
        ev::apply_weighted_pic_subhalo_collision_interval(
            uninterrupted,
            thin_shell_mesh(),
            0.01,
            options,
            uninterrupted_sampler,
            uninterrupted_workspace,
            4);
    ev::WeightedPICSubhaloCollisionWorkspace loaded_workspace;
    const auto second_restarted =
        ev::apply_weighted_pic_subhalo_collision_interval(
            loaded.markers,
            thin_shell_mesh(),
            0.01,
            options,
            loaded.collision_sampler,
            loaded_workspace,
            4);
    assert(
        second_uninterrupted.accepted_pairs
        == second_restarted.accepted_pairs);
    assert(
        uninterrupted_sampler.next_counter
        == loaded.collision_sampler.next_counter);
    assert(uninterrupted.size() == loaded.markers.size());
    for (std::size_t index = 0; index < uninterrupted.size(); ++index) {
        const auto& first = uninterrupted[index];
        const auto& second = loaded.markers[index];
        assert(first.mass == second.mass);
        assert(first.r == second.r);
        assert(first.w == second.w);
        assert(first.eta == second.eta);
        assert(first.internal_energy == second.internal_energy);
        assert(
            first.internal_velocity_dispersion
            == second.internal_velocity_dispersion);
        assert(first.id == second.id);
        assert(first.state == second.state);
    }
}

void assert_same_adaptive_markers(
    const std::vector<ev::WeightedPICMarker>& first,
    const std::vector<ev::WeightedPICMarker>& second) {
    assert(first.size() == second.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        assert(first[index].r == second[index].r);
        assert(first[index].w == second[index].w);
        assert(first[index].eta == second[index].eta);
        assert(first[index].mass == second[index].mass);
        assert(first[index].id == second[index].id);
        assert(first[index].state == second[index].state);
        assert(first[index].internal_energy == second[index].internal_energy);
        assert(first[index].internal_velocity_dispersion
            == second[index].internal_velocity_dispersion);
        assert(first[index].source_cell == second[index].source_cell);
        assert(first[index].inner_boundary_speed
            == second[index].inner_boundary_speed);
        assert(first[index].remaining_inner_time
            == second[index].remaining_inner_time);
    }
}

void assert_same_adaptive_diagnostics(
    const ev::WeightedPICSubhaloCollisionDiagnostics& first,
    const ev::WeightedPICSubhaloCollisionDiagnostics& second) {
    assert(first.substeps == second.substeps);
    assert(first.cell_visits == second.cell_visits);
    assert(first.trial_pairs == second.trial_pairs);
    assert(first.accepted_pairs == second.accepted_pairs);
    assert(first.scattering_events == second.scattering_events);
    assert(first.capture_events == second.capture_events);
    assert(first.inner_cell_visits == second.inner_cell_visits);
    assert(first.inner_scattering_events == second.inner_scattering_events);
    assert(first.inner_capture_events == second.inner_capture_events);
    assert(first.skipped_already_collided_trials
        == second.skipped_already_collided_trials);
    assert(first.active_markers_before == second.active_markers_before);
    assert(first.active_markers_after == second.active_markers_after);
    assert(first.spawned_markers == second.spawned_markers);
    assert(first.removed_markers == second.removed_markers);
    assert(first.nonlocal_cells == second.nonlocal_cells);
    assert(first.represented_scattering_mass
        == second.represented_scattering_mass);
    assert(first.represented_capture_mass == second.represented_capture_mass);
    assert(first.orbital_kinetic_energy_change
        == second.orbital_kinetic_energy_change);
    assert(first.internal_energy_change == second.internal_energy_change);
    assert(first.maximum_pair_momentum_error
        == second.maximum_pair_momentum_error);
    assert(first.maximum_pair_total_energy_error
        == second.maximum_pair_total_energy_error);
    assert(first.minimum_mean_free_path == second.minimum_mean_free_path);
    assert(first.minimum_collision_timescale
        == second.minimum_collision_timescale);
    assert(first.maximum_cell_width_over_mean_free_path
        == second.maximum_cell_width_over_mean_free_path);
    assert(first.mean_free_path_pair_samples
        == second.mean_free_path_pair_samples);
    assert(first.minimum_orbital_velocity_dispersion
        == second.minimum_orbital_velocity_dispersion);
    assert(first.maximum_orbital_velocity_dispersion
        == second.maximum_orbital_velocity_dispersion);
    assert(first.maximum_weight_ratio == second.maximum_weight_ratio);
    assert(first.minimum_internal_velocity_dispersion
        == second.minimum_internal_velocity_dispersion);
    assert(first.maximum_internal_velocity_dispersion
        == second.maximum_internal_velocity_dispersion);
    assert(first.minimum_pair_velocity_scale
        == second.minimum_pair_velocity_scale);
    assert(first.maximum_pair_velocity_scale
        == second.maximum_pair_velocity_scale);
}

ev::WeightedPICRadialMesh adaptive_test_mesh() {
    ev::WeightedPICRadialMesh mesh;
    mesh.edge = {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4};
    for (std::size_t index = 1; index < mesh.edge.size(); ++index) {
        const double q0 = std::pow(mesh.edge[index - 1], 3) / 3.0;
        const double q1 = std::pow(mesh.edge[index], 3) / 3.0;
        mesh.shell_volume.push_back(4.0 * ev::PI * (q1 - q0));
        mesh.volume_coordinate_center.push_back(0.5 * (q0 + q1));
        mesh.venus_center.push_back(std::cbrt(1.5 * (q0 + q1)));
    }
    return mesh;
}

std::vector<ev::WeightedPICMarker> adaptive_test_markers(int scenario) {
    std::vector<ev::WeightedPICMarker> markers;
    const auto mesh = adaptive_test_mesh();
    std::uint64_t id = 1;
    // The central and resolved shells have different populations and volumes.
    // Include an empty shell, a single-marker shell, escaped objects and
    // active objects outside the mesh to exercise every membership branch.
    for (int cell = 0; cell < 8; ++cell) {
        const int population = cell == 3 ? 0
            : (cell == 4 ? 1 : (cell == 0 ? 24 : 12));
        for (int index = 0; index < population; ++index) {
            ev::WeightedPICMarker value;
            value.id = id++;
            value.mass = 1.0e-5 * (1 + index % 4);
            value.r = cell == 0 ? 0.075
                : (cell == 7 ? 8.0
                   : 0.5 * (mesh.edge[cell - 1] + mesh.edge[cell]));
            value.r *= 1.0 + 0.015 * (index % 5 - 2);
            value.w = (scenario == 0 ? 0.025 : 0.03) * (index % 7 - 3);
            value.eta = 0.5 * value.r * value.r
                * std::pow(0.013 * (index % 3), 2);
            value.internal_velocity_dispersion = 0.1 * (1 + 0.07 * (index % 3));
            value.internal_energy = 1.5 * value.mass
                * std::pow(value.internal_velocity_dispersion, 2);
            if (cell == 6 && index % 3 == 0) {
                value.state = ev::WeightedPICMarkerState::escaped;
            }
            markers.push_back(value);
        }
    }
    return markers;
}

void test_adaptive_cells_preserve_parallel_event_sequences() {
    std::uint64_t captures = 0;
    std::uint64_t scatterings = 0;
    bool different_local_costs = false;
    // Vary the channel and lane count while retaining identical initial
    // particles, epoch and seed for the one- and four-thread calculations.
    // Every workspace is reused across four complete collision intervals.
    for (int scenario : {0, 1, 2}) {
        for (int lanes : {1, 3}) {
            auto serial = adaptive_test_markers(scenario);
            auto parallel = serial;
            ev::WeightedPICSubhaloCollisionOptions options;
            options.strength = 1.0;
            options.collision_cfl = 0.05;
            options.maximum_substeps = 20000;
            options.enable_scattering = scenario != 0;
            options.enable_capture = scenario != 1;
            options.adaptive_cell_timestep = true;
            options.collision_lanes = lanes;
            options.inner_collision_cells = 2;
            ev::WeightedPICCollisionSamplerState serial_sampler{0x13579, 17};
            auto parallel_sampler = serial_sampler;
            ev::WeightedPICSubhaloCollisionWorkspace serial_workspace;
            ev::WeightedPICSubhaloCollisionWorkspace parallel_workspace;
            for (int interval = 0; interval < 4; ++interval) {
                const auto serial_stats =
                    ev::apply_weighted_pic_subhalo_collision_interval(
                        serial, adaptive_test_mesh(), 0.5, options,
                        serial_sampler, serial_workspace, 1);
                const auto parallel_stats =
                    ev::apply_weighted_pic_subhalo_collision_interval(
                        parallel, adaptive_test_mesh(), 0.5, options,
                        parallel_sampler, parallel_workspace, 4);
                assert_same_adaptive_markers(serial, parallel);
                assert_same_adaptive_diagnostics(serial_stats, parallel_stats);
                assert(serial_sampler.seed == parallel_sampler.seed);
                assert(serial_sampler.next_counter
                    == static_cast<std::uint64_t>(18 + interval));
                assert(serial_sampler.next_counter
                    == parallel_sampler.next_counter);
                captures += serial_stats.capture_events;
                scatterings += serial_stats.scattering_events;
                std::uint64_t minimum_visits = UINT64_MAX;
                std::uint64_t maximum_visits = 0;
                for (const auto& cell : serial_workspace.cell_diagnostics) {
                    if (cell.cell_visits != 0) {
                        minimum_visits = std::min(minimum_visits, cell.cell_visits);
                        maximum_visits = std::max(maximum_visits, cell.cell_visits);
                    }
                }
                different_local_costs = different_local_costs
                    || maximum_visits > minimum_visits;
            }
        }
    }
    assert(captures > 0);
    assert(scatterings > 0);
    assert(different_local_costs);

    // A failed interval has the same first failing cell and partial marker
    // state across thread counts.  The driver discards that failed state.
    for (int limit : {1, 3}) {
        auto serial = adaptive_test_markers(2);
        auto parallel = serial;
        ev::WeightedPICSubhaloCollisionOptions options;
        options.strength = 2.0;
        options.collision_cfl = 0.02;
        options.maximum_substeps = limit;
        options.adaptive_cell_timestep = true;
        ev::WeightedPICCollisionSamplerState serial_sampler{0x987654, 2};
        auto parallel_sampler = serial_sampler;
        ev::WeightedPICSubhaloCollisionWorkspace serial_workspace;
        ev::WeightedPICSubhaloCollisionWorkspace parallel_workspace;
        std::string serial_error, parallel_error;
        try {
            ev::apply_weighted_pic_subhalo_collision_interval(
                serial, adaptive_test_mesh(), 1.0, options,
                serial_sampler, serial_workspace, 1);
        } catch (const std::runtime_error& error) {
            serial_error = error.what();
        }
        try {
            ev::apply_weighted_pic_subhalo_collision_interval(
                parallel, adaptive_test_mesh(), 1.0, options,
                parallel_sampler, parallel_workspace, 4);
        } catch (const std::runtime_error& error) {
            parallel_error = error.what();
        }
        assert(!serial_error.empty());
        assert(serial_error == parallel_error);
        assert_same_adaptive_markers(serial, parallel);
        assert(serial_sampler.next_counter
                    == parallel_sampler.next_counter);
    }
}

void test_origin_collision_preserves_stored_energy() {
    // A Cartesian velocity perpendicular to the chosen pair axis still
    // leaves the origin with its complete speed and zero angular momentum.
    auto at_origin = marker(1.0, 1.0, 1, 0.015);
    at_origin.r = 0.0;
    ev::weighted_pic_subhalo_collision_detail::install_velocity(
        at_origin, {0.0, 1.0, 0.0});
    assert(at_origin.w == 1.0 && at_origin.eta == 0.0);
    const auto energy = [](const std::vector<ev::WeightedPICMarker>& values) {
        long double result = 0.0L;
        for (const auto& value : values) {
            const double tangential_squared = value.r == 0.0 ? 0.0
                : 2.0 * value.eta / (value.r * value.r);
            result += 0.5L * value.mass
                * (value.w * value.w + tangential_squared)
                + value.internal_energy;
        }
        return result;
    };
    for (int origin_count : {1, 2}) {
        for (bool capture : {false, true}) {
            std::vector<ev::WeightedPICMarker> values{
                marker(0.1, capture ? 0.01 : 0.08, 1, 0.0015),
                marker(0.17, capture ? 0.02 : 0.11, 2, 0.00255)};
            values[0].r = 0.0;
            values[1].r = origin_count == 2 ? 0.0 : 0.2;
            const long double initial_energy = energy(values);
            auto options = capture_only();
            options.enable_capture = capture;
            options.enable_scattering = !capture;
            options.strength = 100.0;
            options.collision_cfl = 0.2;
            options.maximum_substeps = 100000;
            options.adaptive_cell_timestep = true;
            ev::WeightedPICCollisionSamplerState sampler{0x54321, 0};
            ev::WeightedPICSubhaloCollisionWorkspace workspace;
            std::uint64_t events = 0;
            for (int interval = 0; interval < 16 && events == 0; ++interval) {
                const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
                    values, thin_shell_mesh(), 0.5, options, sampler, workspace, 4);
                events += stats.accepted_pairs;
                assert(std::isfinite(stats.maximum_orbital_velocity_dispersion));
                assert(std::isfinite(stats.minimum_collision_timescale));
                assert(stats.maximum_pair_momentum_error < 1.0e-14);
                assert(stats.maximum_pair_total_energy_error < 1.0e-14);
                assert(std::abs(energy(values) - initial_energy) < 1.0e-14L);
                for (const auto& value : values) {
                    if (value.r == 0.0) {
                        assert(value.w >= 0.0);
                        assert(value.eta == 0.0);
                    }
                }
            }
            assert(events > 0);
            assert(values.size() == (capture ? 1u : 2u));
            // Feed the accepted collision state to the ordinary origin-parity
            // KDK and deposition paths. A zero force isolates conservation of
            // the stored kinetic energy during the subsequent departure.
            ev::Params params;
            params.physics_model = "vlasov_poisson";
            params.vp_universe = "static";
            params.vp_interpolation = "hamiltonian_sldg";
            params.kinetic_grid = "cell_average_eta";
            params.kinetic_eta_grid = "uniform";
            params.Nr = 3; params.Nw = 4; params.NL = 2;
            params.r_min = 1.0; params.r_max = 4.0;
            params.radial_grid = "linear";
            params.w_max = 2.0; params.L_max = 2.0;
            params.inner_boundary = "origin_parity";
            params.outer_boundary = "reflecting";
            params.vp_orbit_regularization = "off";
            params.integrator = "semi_lagrangian_kick_drift_kick";
            const auto geometry = ev::make_cell_average_eta_geometry(
                ev::make_positive_radial_cell_axis(params), params);
            const auto mesh = ev::make_weighted_pic_radial_mesh(geometry);
            const auto coordinate = ev::radial_coordinate_grid(
                geometry.radial.center, params);
            ev::Moments zero;
            zero.rho.assign(params.Nr, 0.0); zero.p.assign(params.Nr, 0.0);
            zero.pt.assign(params.Nr, 0.0); zero.j.assign(params.Nr, 0.0);
            const auto field = ev::solve_vlasov_poisson_field(
                zero, geometry.radial.center, 0.0, params);
            ev::begin_weighted_pic_self_consistent_kdk_parallel(
                values, 0.0, 0.01, coordinate, geometry.radial.center,
                field, params, 1);
            ev::finish_weighted_pic_self_consistent_kdk_parallel(
                values, 0.01, 0.01, coordinate, geometry.radial.center,
                field, params, 1);
            assert(std::abs(energy(values) - initial_energy) < 1.0e-14L);
            const auto deposit = ev::deposit_weighted_vlasov_pic_parallel(
                values, mesh, ev::WeightedPICParticleShape::tsc, 1, true);
            const auto deposited_field = ev::solve_weighted_pic_poisson_field(
                deposit, mesh, 0.01, params);
            for (double value : deposit.density) { assert(std::isfinite(value)); }
            for (double value : deposited_field.phi_r) { assert(std::isfinite(value)); }
            for (double value : deposited_field.phi) { assert(std::isfinite(value)); }
            assert(std::abs(deposit.deposited_mesh_mass
                + deposit.hidden_inner_mass - 0.27) < 1.0e-14);
        }
    }

    // Equal outgoing speeds at the coordinate origin have isotropic incoming
    // directions.  Their relative speed and collision clock are nonzero.
    std::vector<ev::WeightedPICMarker> isotropic{
        marker(1.0, 0.1, 1, 0.015), marker(1.0, 0.1, 2, 0.015)};
    for (auto& value : isotropic) { value.r = 0.0; }
    auto options = capture_only();
    options.enable_scattering = true;
    ev::WeightedPICCollisionSamplerState sampler{0x112233, 0};
    ev::WeightedPICSubhaloCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_subhalo_collision_interval(
        isotropic, thin_shell_mesh(), std::numeric_limits<double>::min(),
        options, sampler, workspace, 1);
    assert(std::isfinite(stats.minimum_collision_timescale));
    assert(std::abs(stats.maximum_orbital_velocity_dispersion
        - 0.1 / std::sqrt(3.0)) < 1.0e-15);

    auto tiny_radius = isotropic;
    tiny_radius[0].r = tiny_radius[1].r = 1.0e-200;
    tiny_radius[1].w = 0.2;
    const auto tiny_stats = ev::apply_weighted_pic_subhalo_collision_interval(
        tiny_radius, thin_shell_mesh(), std::numeric_limits<double>::min(),
        options, sampler, workspace, 1);
    assert(std::abs(tiny_stats.maximum_orbital_velocity_dispersion
        - 0.05 / std::sqrt(3.0)) < 1.0e-15);

    isotropic[0].eta = 0.01;
    const auto saved_counter = sampler.next_counter;
    bool rejected = false;
    try {
        ev::apply_weighted_pic_subhalo_collision_interval(
            isotropic, thin_shell_mesh(), 0.01, options, sampler, workspace, 1);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("zero angular momentum")
            != std::string::npos;
    }
    assert(rejected && sampler.next_counter == saved_counter);
}

void test_global_population_refinement_preserves_the_instantaneous_df() {
    std::vector<ev::WeightedPICMarker> markers{
        marker(2.0, -0.25, 7, 0.06),
        marker(8.0, 0.5, 11, 0.96),
        marker(5.0, 0.1, 19, 0.15),
    };
    markers[0].eta = 0.125;
    markers[1].eta = 0.5;
    markers[2].state = ev::WeightedPICMarkerState::escaped;
    const std::vector<ev::WeightedPICMarker> original = markers;

    ev::WeightedPICPopulationControlOptions options;
    options.trigger_live_markers = 3;
    options.refinement_factor = 4;
    const auto diagnostics =
        ev::refine_weighted_pic_population_if_needed(markers, options);

    assert(diagnostics.applied);
    assert(diagnostics.live_markers_before == 2);
    assert(diagnostics.live_markers_after == 8);
    assert(diagnostics.escaped_markers == 1);
    assert(diagnostics.injected_markers == 6);
    assert(markers.size() == 9);
    assert(diagnostics.maximum_relative_invariant_residual
        < 256.0 * std::numeric_limits<double>::epsilon());

    std::size_t first_daughters = 0;
    std::size_t second_daughters = 0;
    std::size_t escaped = 0;
    std::set<std::uint64_t> id;
    for (const ev::WeightedPICMarker& value : markers) {
        assert(id.insert(value.id).second);
        if (value.state == ev::WeightedPICMarkerState::escaped) {
            ++escaped;
            assert(value.mass == original[2].mass);
            continue;
        }
        if (value.r == original[0].r && value.w == original[0].w
            && value.eta == original[0].eta) {
            ++first_daughters;
            assert(value.mass == original[0].mass / 4.0);
            assert(
                value.internal_energy
                == original[0].internal_energy / 4.0);
            assert(
                value.internal_velocity_dispersion
                == original[0].internal_velocity_dispersion);
        } else {
            ++second_daughters;
            assert(value.mass == original[1].mass / 4.0);
            assert(
                value.internal_energy
                == original[1].internal_energy / 4.0);
            assert(
                value.internal_velocity_dispersion
                == original[1].internal_velocity_dispersion);
        }
    }
    assert(first_daughters == 4);
    assert(second_daughters == 4);
    assert(escaped == 1);
}

void test_common_refinement_preserves_unequal_pair_continuum_rate() {
    ev::WeightedPICSubhaloCollisionOptions options = capture_only();
    options.enable_scattering = true;
    const double first_mass = 2.0;
    const double second_mass = 8.0;
    const double speed = 0.7;
    const double pair_sigma = 0.2;
    const double before =
        ev::weighted_pic_subhalo_scattering_cross_section(
            speed, first_mass, second_mass, pair_sigma, options)
        + ev::weighted_pic_subhalo_capture_cross_section(
            speed, first_mass, second_mass, pair_sigma, options);
    const int factor = 4;
    const double after =
        ev::weighted_pic_subhalo_scattering_cross_section(
            speed,
            first_mass / factor,
            second_mass / factor,
            pair_sigma,
            options)
        + ev::weighted_pic_subhalo_capture_cross_section(
            speed,
            first_mass / factor,
            second_mass / factor,
            pair_sigma,
            options);
    assert(std::abs(factor * after - before)
        < 64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, before));
    const double majorant_before =
        ev::weighted_pic_subhalo_pair_u_sigma_majorant(
            first_mass, second_mass, 0.1, 0.2, options);
    const double majorant_after =
        ev::weighted_pic_subhalo_pair_u_sigma_majorant(
            first_mass / factor,
            second_mass / factor,
            0.1,
            0.2,
            options);
    assert(std::abs(factor * majorant_after - majorant_before)
        < 64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, majorant_before));
}

void test_refinement_preserves_measured_collision_mass_rate() {
    std::vector<ev::WeightedPICMarker> base;
    for (std::uint64_t index = 0; index < 256; ++index) {
        const double mass =
            1.0e-2 * (1.0 + static_cast<double>(index % 5));
        const double velocity =
            0.24 * (
                static_cast<double>(index % 17) / 16.0 - 0.5);
        base.push_back(marker(
            mass,
            velocity,
            index + 1,
            1.5 * mass * 0.1 * 0.1));
        base.back().eta =
            0.5 * std::pow(
                0.04 * static_cast<double>(index % 7), 2);
    }
    std::vector<ev::WeightedPICMarker> refined = base;
    ev::WeightedPICPopulationControlOptions refinement;
    refinement.trigger_live_markers = base.size() + 1;
    refinement.refinement_factor = 2;
    const auto refinement_diagnostics =
        ev::refine_weighted_pic_population_if_needed(
            refined, refinement);
    assert(refinement_diagnostics.applied);

    auto options = capture_only();
    options.enable_scattering = true;
    options.strength = 0.2;
    options.collision_cfl = 0.1;
    options.maximum_substeps = 4096;
    long double base_scattering_mass = 0.0L;
    long double base_capture_mass = 0.0L;
    long double refined_scattering_mass = 0.0L;
    long double refined_capture_mass = 0.0L;
    std::uint64_t base_events = 0;
    std::uint64_t refined_events = 0;
    for (std::uint64_t realization = 0; realization < 48; ++realization) {
        std::vector<ev::WeightedPICMarker> base_sample = base;
        std::vector<ev::WeightedPICMarker> refined_sample = refined;
        const std::uint64_t seed =
            UINT64_C(0x726174655f746573) + realization;
        ev::WeightedPICCollisionSamplerState base_sampler{seed, 0};
        ev::WeightedPICCollisionSamplerState refined_sampler{seed, 0};
        ev::WeightedPICSubhaloCollisionWorkspace base_workspace;
        ev::WeightedPICSubhaloCollisionWorkspace refined_workspace;
        const auto base_stats =
            ev::apply_weighted_pic_subhalo_collision_interval(
                base_sample,
                thin_shell_mesh(),
                0.002,
                options,
                base_sampler,
                base_workspace,
                1);
        const auto refined_stats =
            ev::apply_weighted_pic_subhalo_collision_interval(
                refined_sample,
                thin_shell_mesh(),
                0.002,
                options,
                refined_sampler,
                refined_workspace,
                1);
        base_scattering_mass += base_stats.represented_scattering_mass;
        base_capture_mass += base_stats.represented_capture_mass;
        refined_scattering_mass +=
            refined_stats.represented_scattering_mass;
        refined_capture_mass += refined_stats.represented_capture_mass;
        base_events += base_stats.accepted_pairs;
        refined_events += refined_stats.accepted_pairs;
    }
    assert(base_events > 100);
    assert(refined_events > base_events);
    const auto relative_difference = [](long double first, long double second) {
        return static_cast<double>(
            std::abs(first - second)
            / std::max({1.0e-300L, std::abs(first), std::abs(second)}));
    };
    // Event counts increase with resolution, but the represented mass
    // processed per unit time is the continuum quantity and must converge to
    // the same value.  This deterministic ensemble allows ordinary sampling
    // scatter while rejecting a systematic factor-of-k rate error.
    assert(relative_difference(
               base_scattering_mass, refined_scattering_mass) < 0.2);
    assert(relative_difference(
               base_capture_mass, refined_capture_mass) < 0.2);
}

}  // namespace

int main() {
    test_kernel_asymptotes();
    test_joint_kernel_majorant_bounds_interpolation();
    test_pair_specific_majorant_bounds_unequal_mass_kernel();
    test_equal_weight_capture_merges_and_heats();
    test_unequal_weight_capture_conserves_mass_momentum_energy();
    test_unequal_mass_scattering_keeps_whole_objects();
    test_restart_preserves_internal_energy();
    test_heated_dispersion_changes_the_next_kernel();
    test_unresolved_origin_core_is_collisional();
    test_unequal_mass_pair_cross_section_is_symmetric();
    test_mass_ratio_changes_the_calibrated_outcome_shapes();
    test_accepted_table_resolves_the_extreme_mass_ratio_limit();
    test_velocity_dependent_mean_free_path_recovers_frank_limit();
    test_mean_free_path_uses_evolving_pair_velocity_and_cross_section();
    test_uniform_core_equal_volume_shell_geometry();
    test_collision_lane_rate_estimator_is_unbiased();
    test_inner_collision_lanes_are_parallel_deterministic();
    test_collision_restart_trajectory_is_deterministic();
    test_adaptive_cells_preserve_parallel_event_sequences();
    test_origin_collision_preserves_stored_energy();
    test_global_population_refinement_preserves_the_instantaneous_df();
    test_common_refinement_preserves_unequal_pair_continuum_rate();
    test_refinement_preserves_measured_collision_mass_rate();
    std::cout << "weighted PIC subhalo collision tests passed\n";
    return 0;
}
