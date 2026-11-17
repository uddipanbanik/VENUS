// Regression coverage for the weighted Einstein--Vlasov PIC operators.
// The tests check deposition, field reconstruction, characteristic motion,
// physical boundaries, and conservation diagnostics.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "ev/weighted_einstein_vlasov_pic.hpp"

namespace {

bool nearly_equal(
    double first,
    double second,
    double relative_tolerance = 2.0e-13,
    double absolute_tolerance = 2.0e-15) {
    return std::abs(first - second)
        <= absolute_tolerance
            + relative_tolerance
                * std::max(std::abs(first), std::abs(second));
}

ev::Params ev_pic_params(int radial_cells = 64) {
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.vp_universe = "static";
    p.bp_collision_operator = "none";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.vp_interpolation = "hamiltonian_sldg";
    p.Nr = radial_cells;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = 0.1;
    p.r_max = 4.0;
    p.radial_grid = "linear";
    p.w_max = 1.0;
    p.L_max = 1.0;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "outflow_vacuum";
    p.metric_floor = 1.0e-12;
    p.gap_stop = 1.0e-3;
    return p;
}

ev::WeightedPICMarker marker(
    double mass,
    double radius,
    double radial_momentum,
    double eta,
    std::uint64_t id) {
    ev::WeightedPICMarker result;
    result.mass = mass;
    result.r = radius;
    result.w = radial_momentum;
    result.eta = eta;
    result.id = id;
    result.source_cell = id;
    return result;
}

ev::WeightedEinsteinVlasovPICField empty_field(
    const ev::WeightedPICFieldGrid& grid) {
    return ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
        {},
        grid,
        ev::WeightedPICParticleShape::tsc,
        1);
}

void test_origin_tsc_constant_density() {
    ev::Params origin = ev_pic_params(128);
    origin.r_min = 0.0;
    origin.r_max = 1.0;
    origin.inner_boundary = "origin_parity";
    origin.radial_grid = "linear";
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            origin, origin.Nr);

    constexpr int samples_per_shell = 32;
    std::vector<ev::WeightedPICMarker> markers;
    markers.reserve(
        grid.mesh.shell_volume.size()
        * static_cast<std::size_t>(samples_per_shell));
    std::uint64_t id = 1;
    for (std::size_t i = 0; i < grid.mesh.shell_volume.size(); ++i) {
        const double q_left =
            std::pow(grid.mesh.edge[i], 3) / 3.0;
        const double q_right =
            std::pow(grid.mesh.edge[i + 1], 3) / 3.0;
        for (int sample = 0; sample < samples_per_shell; ++sample) {
            const double fraction =
                (static_cast<double>(sample) + 0.5)
                / static_cast<double>(samples_per_shell);
            const double q = q_left + fraction * (q_right - q_left);
            markers.push_back(marker(
                grid.mesh.shell_volume[i]
                    / static_cast<double>(samples_per_shell),
                std::cbrt(3.0 * q),
                0.0,
                0.0,
                id++));
        }
    }
    const auto deposited = ev::deposit_weighted_vlasov_pic(
        markers,
        grid.mesh,
        ev::WeightedPICParticleShape::tsc);
    // A constant finite-volume density must remain regular at the true origin.
    // The uncorrected logical-coordinate TSC cloud overestimated cell zero by
    // roughly a factor of 2.7 on this linear-r/nonuniform-q mesh.
    for (std::size_t i = 0; i < 8; ++i) {
        assert(std::abs(deposited.density[i] - 1.0) < 0.12);
    }
}

void test_exact_origin_wall_and_outflow_orbits() {
    ev::Params origin = ev_pic_params(256);
    origin.r_min = 0.0;
    origin.r_max = 4.0;
    origin.inner_boundary = "origin_parity";
    origin.radial_grid = "linear";
    origin.vp_orbit_regularization = "always";
    origin.vp_orbit_lc_relative_tolerance = 1.0e-11;
    origin.vp_orbit_lc_absolute_tolerance = 1.0e-13;
    const ev::WeightedPICFieldGrid origin_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            origin, origin.Nr);
    const auto minkowski = empty_field(origin_grid);

    {
        constexpr double r0 = 0.01;
        constexpr double w0 = -0.5;
        constexpr double dt = 0.1;
        const double epsilon = std::sqrt(1.0 + w0 * w0);
        const double signed_radius = r0 + dt * w0 / epsilon;
        std::vector<ev::WeightedPICMarker> orbit{
            marker(1.0e-12, r0, w0, 0.0, 30001),
        };
        ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
            orbit, dt, origin_grid, minkowski.metric);
        assert(nearly_equal(
            orbit.front().r, std::abs(signed_radius), 2.0e-12));
        assert(nearly_equal(orbit.front().w, -w0, 2.0e-12));
    }

    {
        constexpr double r0 = 0.05;
        constexpr double w0 = -0.8;
        constexpr double angular_momentum = 0.015;
        constexpr double dt = 0.1;
        const double tangential_momentum = angular_momentum / r0;
        const double epsilon = std::sqrt(
            1.0 + w0 * w0
            + tangential_momentum * tangential_momentum);
        const double x = r0 + dt * w0 / epsilon;
        const double y = dt * tangential_momentum / epsilon;
        const double exact_radius = std::hypot(x, y);
        const double exact_radial_momentum =
            (x * w0 + y * tangential_momentum) / exact_radius;
        std::vector<ev::WeightedPICMarker> orbit{
            marker(
                1.0e-12,
                r0,
                w0,
                0.5 * angular_momentum * angular_momentum,
                30002),
        };
        const double eta_before = orbit.front().eta;
        ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
            orbit, dt, origin_grid, minkowski.metric);
        assert(nearly_equal(
            orbit.front().r, exact_radius, 2.0e-9, 2.0e-12));
        assert(nearly_equal(
            orbit.front().w,
            exact_radial_momentum,
            2.0e-9,
            2.0e-12));
        assert(orbit.front().eta == eta_before);
    }

    {
        std::vector<ev::WeightedPICMarker> turning_point{
            marker(
                1.0e-12,
                0.05,
                0.0,
                0.5 * 0.015 * 0.015,
                30006),
        };
        const double regularized_dt =
            ev::choose_weighted_einstein_vlasov_pic_timestep(
                turning_point,
                origin_grid,
                minkowski.metric,
                0.3,
                0.1);
        ev::Params unregularized = origin;
        unregularized.vp_orbit_regularization = "off";
        const auto unregularized_grid =
            ev::make_weighted_einstein_vlasov_pic_field_grid(
                unregularized, unregularized.Nr);
        const auto unregularized_minkowski =
            empty_field(unregularized_grid);
        const double unregularized_dt =
            ev::choose_weighted_einstein_vlasov_pic_timestep(
                turning_point,
                unregularized_grid,
                unregularized_minkowski.metric,
                0.3,
                0.1);
        assert(nearly_equal(regularized_dt, 0.1));
        assert(unregularized_dt < regularized_dt);
    }

    {
        ev::Params no_omission = origin;
        no_omission.vp_timestep_omitted_mass_fraction = 0.0;
        const auto no_omission_grid =
            ev::make_weighted_einstein_vlasov_pic_field_grid(
                no_omission, no_omission.Nr);
        const auto no_omission_minkowski =
            empty_field(no_omission_grid);
        std::vector<ev::WeightedPICMarker> tracers{
            marker(1.0e-6, 0.05, 1.0, 0.0, 30007),
            marker(1.0, 2.0, 0.0, 0.0, 30008),
        };
        const double strict_dt =
            ev::choose_weighted_einstein_vlasov_pic_timestep(
                tracers,
                no_omission_grid,
                no_omission_minkowski.metric,
                0.3,
                0.1);
        ev::Params with_omission = no_omission;
        with_omission.vp_timestep_omitted_mass_fraction = 1.0e-5;
        const auto with_omission_grid =
            ev::make_weighted_einstein_vlasov_pic_field_grid(
                with_omission, with_omission.Nr);
        const auto with_omission_minkowski =
            empty_field(with_omission_grid);
        const double mass_weighted_dt =
            ev::choose_weighted_einstein_vlasov_pic_timestep(
                tracers,
                with_omission_grid,
                with_omission_minkowski.metric,
                0.3,
                0.1);
        assert(strict_dt < 0.1);
        assert(nearly_equal(mass_weighted_dt, 0.1));
    }

    {
        // The endpoint has returned to roughly its initial radius, so an
        // endpoint-only stiffness detector would miss this very radial
        // pericentre passage.  The flat closest-approach time must trigger
        // the Levi-Civita retry.
        ev::Params automatic = origin;
        automatic.vp_orbit_regularization = "auto";
        const auto automatic_grid =
            ev::make_weighted_einstein_vlasov_pic_field_grid(
                automatic, automatic.Nr);
        const auto automatic_minkowski = empty_field(automatic_grid);
        constexpr double r0 = 1.0;
        constexpr double w0 = -5.0;
        constexpr double angular_momentum = 0.01;
        constexpr double dt = 2.05;
        const double tangential_momentum = angular_momentum / r0;
        const double epsilon = std::sqrt(
            1.0 + w0 * w0
            + tangential_momentum * tangential_momentum);
        const double x = r0 + dt * w0 / epsilon;
        const double y = dt * tangential_momentum / epsilon;
        const double exact_radius = std::hypot(x, y);
        const double exact_radial_momentum =
            (x * w0 + y * tangential_momentum) / exact_radius;
        assert(
            ev::weighted_einstein_vlasov_pic_detail::
                should_use_levi_civita(
                    {r0, w0},
                    {exact_radius, exact_radial_momentum},
                    angular_momentum,
                    dt,
                    automatic));
        std::vector<ev::WeightedPICMarker> orbit{
            marker(
                1.0e-12,
                r0,
                w0,
                0.5 * angular_momentum * angular_momentum,
                30005),
        };
        ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
            orbit,
            dt,
            automatic_grid,
            automatic_minkowski.metric);
        assert(nearly_equal(
            orbit.front().r, exact_radius, 2.0e-8, 2.0e-11));
        assert(nearly_equal(
            orbit.front().w,
            exact_radial_momentum,
            2.0e-8,
            2.0e-11));
    }

    ev::Params wall = ev_pic_params(128);
    wall.r_min = 0.1;
    wall.r_max = 4.0;
    wall.inner_boundary = "reflecting";
    wall.outer_boundary = "outflow_vacuum";
    wall.vp_orbit_lc_relative_tolerance = 1.0e-11;
    wall.vp_orbit_lc_absolute_tolerance = 1.0e-13;
    const ev::WeightedPICFieldGrid wall_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            wall, wall.Nr);
    const auto wall_minkowski = empty_field(wall_grid);
    {
        constexpr double r0 = 0.12;
        constexpr double w0 = -0.5;
        constexpr double dt = 0.1;
        const double displacement =
            dt * std::abs(w0) / std::sqrt(1.0 + w0 * w0);
        const double exact_radius =
            wall.r_min + std::abs((r0 - wall.r_min) - displacement);
        std::vector<ev::WeightedPICMarker> orbit{
            marker(1.0e-12, r0, w0, 0.0, 30003),
        };
        ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
            orbit, dt, wall_grid, wall_minkowski.metric);
        assert(nearly_equal(
            orbit.front().r, exact_radius, 2.0e-10, 2.0e-12));
        assert(nearly_equal(orbit.front().w, -w0, 2.0e-10));
    }
    {
        std::vector<ev::WeightedPICMarker> orbit{
            marker(1.0e-12, 3.99, 0.5, 0.0, 30004),
        };
        ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
            orbit, 0.1, wall_grid, wall_minkowski.metric);
        assert(
            orbit.front().state
            == ev::WeightedPICMarkerState::escaped);
        const auto deposited = ev::deposit_weighted_einstein_vlasov_pic(
            orbit, wall_grid.mesh);
        assert(nearly_equal(deposited.escaped_rest_mass, 1.0e-12));
        assert(deposited.deposited_rest_mass == 0.0);
    }

    bool rejected_reflecting_outer = false;
    try {
        ev::Params invalid = wall;
        invalid.outer_boundary = "reflecting";
        (void)ev::make_weighted_einstein_vlasov_pic_field_grid(
            invalid, invalid.Nr);
    } catch (const std::runtime_error&) {
        rejected_reflecting_outer = true;
    }
    assert(rejected_reflecting_outer);
}

void test_constraint_lapse_interpolation_and_time_convergence() {
    auto midpoint_integral = [](int steps) {
        const long double r_left = 0.4L;
        const long double r_right = 2.0L;
        const long double q_left = r_left * r_left * r_left / 3.0L;
        const long double q_right =
            r_right * r_right * r_right / 3.0L;
        constexpr long double canonical_energy_per_q = 0.04L;
        long double mass = 0.02L;
        for (int n = 0; n < steps; ++n) {
            const long double qa = q_left
                + (q_right - q_left)
                    * static_cast<long double>(n)
                    / static_cast<long double>(steps);
            const long double qb = q_left
                + (q_right - q_left)
                    * static_cast<long double>(n + 1)
                    / static_cast<long double>(steps);
            const auto step =
                ev::weighted_einstein_vlasov_pic_detail::
                    radial_mass_midpoint_step(
                        mass,
                        qa,
                        qb,
                        canonical_energy_per_q);
            assert(step.subcritical);
            mass = step.mass;
        }
        return mass;
    };
    const long double reference = midpoint_integral(131072);
    const long double spatial_error_64 =
        std::abs(midpoint_integral(64) - reference);
    const long double spatial_error_128 =
        std::abs(midpoint_integral(128) - reference);
    const long double spatial_error_256 =
        std::abs(midpoint_integral(256) - reference);
    assert(spatial_error_128 < spatial_error_64);
    assert(spatial_error_256 < spatial_error_128);
    assert(spatial_error_64 / spatial_error_128 > 3.3L);
    assert(spatial_error_128 / spatial_error_256 > 3.5L);

    auto vacuum_lapse_error = [](int radial_cells) {
        ev::Params p = ev_pic_params(radial_cells);
        p.r_min = 0.4;
        p.r_max = 4.0;
        p.central_mass = 0.05;
        p.radial_grid = "linear";
        p.inner_boundary = "reflecting";
        p.outer_boundary = "outflow_vacuum";
        const auto grid =
            ev::make_weighted_einstein_vlasov_pic_field_grid(
                p, p.Nr);
        const auto field = empty_field(grid);
        double maximum_error = 0.0;
        for (std::size_t face = 0;
             face < field.metric.face_radius.size();
             ++face) {
            const double radius = field.metric.face_radius[face];
            const double exact =
                0.5 * std::log(1.0 - 2.0 * p.central_mass / radius);
            maximum_error = std::max(
                maximum_error,
                std::abs(field.metric.face_mu[face] - exact));
        }
        return maximum_error;
    };
    const double lapse_error_64 = vacuum_lapse_error(64);
    const double lapse_error_128 = vacuum_lapse_error(128);
    const double lapse_error_256 = vacuum_lapse_error(256);
    assert(lapse_error_128 < lapse_error_64);
    assert(lapse_error_256 < lapse_error_128);
    assert(lapse_error_64 / lapse_error_128 > 3.0);
    assert(lapse_error_128 / lapse_error_256 > 3.0);

    {
        ev::Params p = ev_pic_params(256);
        p.r_min = 0.4;
        p.r_max = 4.0;
        p.central_mass = 0.05;
        p.radial_grid = "linear";
        p.inner_boundary = "reflecting";
        p.outer_boundary = "outflow_vacuum";
        p.vp_orbit_lc_relative_tolerance = 1.0e-11;
        p.vp_orbit_lc_absolute_tolerance = 1.0e-13;
        const auto grid =
            ev::make_weighted_einstein_vlasov_pic_field_grid(
                p, p.Nr);
        const auto field = empty_field(grid);
        constexpr double angular_momentum = 0.2;
        std::vector<ev::WeightedPICMarker> orbit{
            marker(
                1.0e-12,
                1.2,
                -0.2,
                0.5 * angular_momentum * angular_momentum,
                31001),
        };
        auto killing_energy = [&](const ev::WeightedPICMarker& value) {
            const auto sample =
                ev::weighted_einstein_vlasov_pic_detail::
                    sample_metric_on_grid(
                        value.r, grid, field.metric);
            const double epsilon = std::sqrt(
                1.0 + value.w * value.w
                + 2.0 * value.eta / (value.r * value.r));
            return std::exp(sample.mu) * epsilon;
        };
        const double energy_before = killing_energy(orbit.front());
        ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
            orbit, 0.5, grid, field.metric, 16);
        const double energy_after = killing_energy(orbit.front());
        assert(
            std::abs(energy_after - energy_before) / energy_before
            < 2.0e-9);
    }

    ev::Params dynamic = ev_pic_params(96);
    dynamic.r_min = 0.0;
    dynamic.r_max = 4.0;
    dynamic.radial_grid = "linear";
    dynamic.inner_boundary = "origin_parity";
    dynamic.outer_boundary = "outflow_vacuum";
    dynamic.vp_orbit_regularization = "auto";
    const auto dynamic_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            dynamic, dynamic.Nr);
    std::vector<ev::WeightedPICMarker> initial;
    constexpr std::size_t marker_count = 192;
    initial.reserve(marker_count);
    for (std::size_t q = 0; q < marker_count; ++q) {
        const double u =
            (static_cast<double>(q) + 0.5)
            / static_cast<double>(marker_count);
        initial.push_back(marker(
            0.08 / static_cast<double>(marker_count),
            0.65 + 0.35 * u,
            -0.28 + 0.04 * std::sin(13.0 * u),
            2.0e-4 * (0.5 + u),
            static_cast<std::uint64_t>(32000 + q)));
    }
    auto evolve = [&](double dt) {
        std::vector<ev::WeightedPICMarker> particles = initial;
        ev::WeightedEinsteinVlasovPICOptions options;
        options.threads = 1;
        options.characteristic_substeps = 2;
        constexpr double final_time = 0.08;
        const int steps =
            static_cast<int>(std::llround(final_time / dt));
        for (int n = 0; n < steps; ++n) {
            const auto result =
                ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                    particles, dt, dynamic_grid, options);
            assert(nearly_equal(result.accepted_dt, dt));
        }
        return particles;
    };
    const auto coarse = evolve(0.04);
    const auto medium = evolve(0.02);
    const auto fine = evolve(0.01);
    auto difference = [](const auto& first, const auto& second) {
        long double sum = 0.0L;
        for (std::size_t q = 0; q < first.size(); ++q) {
            const long double dr = first[q].r - second[q].r;
            const long double dw = first[q].w - second[q].w;
            sum += dr * dr + dw * dw;
        }
        return std::sqrt(
            static_cast<double>(
                sum / static_cast<long double>(first.size())));
    };
    const double coarse_difference = difference(coarse, medium);
    const double fine_difference = difference(medium, fine);
    assert(fine_difference < coarse_difference);
    assert(coarse_difference / fine_difference > 2.8);

    const auto old_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            initial,
            dynamic_grid,
            ev::WeightedPICParticleShape::tsc,
            1);
    std::vector<ev::WeightedPICMarker> residual_markers = initial;
    ev::WeightedEinsteinVlasovPICOptions residual_options;
    residual_options.threads = 1;
    residual_options.characteristic_substeps = 2;
    constexpr double residual_dt = 2.0e-4;
    const auto residual_step =
        ev::advance_weighted_einstein_vlasov_pic_self_consistent(
            residual_markers,
            residual_dt,
            dynamic_grid,
            residual_options);
    long double lambda_t_residual_square = 0.0L;
    long double lambda_t_scale_square = 0.0L;
    for (std::size_t i = 0; i < dynamic_grid.radius.size(); ++i) {
        const long double finite_difference =
            (residual_step.endpoint_field.metric.lambda[i]
             - old_field.metric.lambda[i]) / residual_dt;
        const long double evolution_value =
            residual_step.midpoint_field.metric.lambda_t[i];
        const long double residual =
            finite_difference - evolution_value;
        lambda_t_residual_square += residual * residual;
        lambda_t_scale_square +=
            evolution_value * evolution_value;
    }
    const double relative_lambda_t_residual = std::sqrt(
        static_cast<double>(
            lambda_t_residual_square
            / std::max(
                lambda_t_scale_square,
                static_cast<long double>(1.0e-300))));
    assert(relative_lambda_t_residual < 0.06);

    std::cout
        << "EV convergence: radial_ratios="
        << static_cast<double>(spatial_error_64 / spatial_error_128)
        << ',' << static_cast<double>(spatial_error_128 / spatial_error_256)
        << " lapse_ratios="
        << lapse_error_64 / lapse_error_128
        << ',' << lapse_error_128 / lapse_error_256
        << " temporal_ratio="
        << coarse_difference / fine_difference
        << " lambda_t_residual="
        << relative_lambda_t_residual << '\n';
}

void test_relativistic_deposition_and_constraints() {
    const ev::Params p = ev_pic_params(48);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    std::vector<ev::WeightedPICMarker> markers{
        marker(0.2, 0.65, 0.3, 0.04, 1),
        marker(0.1, 1.35, -0.2, 0.02, 2),
        marker(0.05, 2.2, 0.1, 0.01, 3),
    };
    const auto serial = ev::deposit_weighted_einstein_vlasov_pic(
        markers, grid.mesh, ev::WeightedPICParticleShape::tsc);
    const auto parallel =
        ev::deposit_weighted_einstein_vlasov_pic_parallel(
            markers,
            grid.mesh,
            ev::WeightedPICParticleShape::tsc,
            3);

    long double expected_rest_mass = 0.0L;
    long double expected_energy = 0.0L;
    long double expected_radial_pressure = 0.0L;
    long double expected_tangential_pressure = 0.0L;
    long double expected_current = 0.0L;
    for (const auto& value : markers) {
        const long double vt2 =
            2.0L * value.eta / (value.r * value.r);
        const long double epsilon =
            std::sqrt(1.0L + value.w * value.w + vt2);
        expected_rest_mass += value.mass;
        expected_energy += value.mass * epsilon;
        expected_radial_pressure +=
            value.mass * value.w * value.w / epsilon;
        expected_tangential_pressure += value.mass * vt2 / epsilon;
        expected_current += value.mass * value.w;
    }
    auto sum = [](const std::vector<double>& value) {
        return std::accumulate(value.begin(), value.end(), 0.0);
    };
    assert(nearly_equal(
        sum(serial.shell_rest_mass),
        static_cast<double>(expected_rest_mass)));
    assert(nearly_equal(
        sum(serial.shell_energy),
        static_cast<double>(expected_energy)));
    assert(nearly_equal(
        sum(serial.shell_radial_pressure),
        static_cast<double>(expected_radial_pressure)));
    assert(nearly_equal(
        sum(serial.shell_tangential_pressure),
        static_cast<double>(expected_tangential_pressure)));
    assert(nearly_equal(
        sum(serial.shell_current),
        static_cast<double>(expected_current)));
    for (std::size_t i = 0; i < serial.shell_energy.size(); ++i) {
        assert(serial.shell_rest_mass[i] == parallel.shell_rest_mass[i]);
        assert(serial.shell_energy[i] == parallel.shell_energy[i]);
        assert(
            serial.shell_radial_pressure[i]
            == parallel.shell_radial_pressure[i]);
        assert(
            serial.shell_tangential_pressure[i]
            == parallel.shell_tangential_pressure[i]);
        assert(serial.shell_current[i] == parallel.shell_current[i]);
    }

    const auto field =
        ev::solve_weighted_einstein_vlasov_pic_field(serial, grid);
    assert(field.adm_mass > 0.0);
    assert(field.adm_mass < static_cast<double>(expected_energy));
    assert(nearly_equal(field.rest_mass, 0.35));
    assert(field.metric.min_gap < 1.0);
    assert(field.metric.min_gap > 0.0);
    assert(field.maximum_compactness > 0.0);
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        assert(std::isfinite(field.metric.mu[i]));
        assert(std::isfinite(field.metric.lambda[i]));
        assert(std::isfinite(field.metric.lambda_t[i]));
    }

    std::cout << "EV deposition: rest_mass=" << field.rest_mass
              << " ADM_mass=" << field.adm_mass
              << " max_2m_over_r=" << field.maximum_compactness << '\n';
}

void test_initialization_bridge_and_minkowski_orbit() {
    ev::Params p = ev_pic_params(3);
    p.r_min = 1.0;
    p.r_max = 4.0;
    p.Nw = 4;
    p.NL = 2;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    std::vector<double> f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                f[ev::idx(i, j, k, p)] =
                    i == 0 ? 1.0e-5 : 0.0;
            }
        }
    }
    ev::WeightedPICInitializationOptions options;
    options.global_marker_budget = 64;
    options.markers_per_occupied_cell = 16;
    options.velocity_pairing =
        ev::WeightedPICVelocityPairing::required;
    options.use_piecewise_constant_in_cell = true;
    ev::WeightedPICSamplerState sampler{UINT64_C(99173), 0};
    const auto initialized =
        ev::initialize_weighted_einstein_vlasov_pic(
            f, geometry, p, options, sampler);
    assert(initialized.markers.size() == 64);
    assert(initialized.diagnostics.velocity_paired);
    assert(initialized.diagnostics.represented_mass
        >= initialized.diagnostics.source_mass);

    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    ev::Metric minkowski;
    const std::size_t cells = grid.radius.size();
    minkowski.m.assign(cells, 0.0);
    minkowski.lambda.assign(cells, 0.0);
    minkowski.mu.assign(cells, 0.0);
    minkowski.lambda_r.assign(cells, 0.0);
    minkowski.mu_r.assign(cells, 0.0);
    minkowski.lambda_t.assign(cells, 0.0);
    minkowski.gap.assign(cells, 1.0);
    minkowski.min_gap = 1.0;
    std::vector<ev::WeightedPICMarker> orbit{
        marker(1.0e-8, 2.0, 0.1, 0.0, 1001),
    };
    constexpr double dt = 0.2;
    ev::advance_weighted_einstein_vlasov_pic_frozen_metric(
        orbit, dt, grid, minkowski);
    const double expected_radius =
        2.0 + dt * 0.1 / std::sqrt(1.0 + 0.1 * 0.1);
    assert(nearly_equal(orbit.front().r, expected_radius, 2.0e-12));
    assert(nearly_equal(orbit.front().w, 0.1));

    std::cout << "EV initialization/orbit: markers="
              << initialized.markers.size()
              << " minkowski_r=" << orbit.front().r << '\n';
}

void test_newtonian_limit_agrees_with_vlasov_poisson() {
    ev::Params p = ev_pic_params(256);
    p.r_min = 0.02;
    p.r_max = 4.0;
    const ev::WeightedPICFieldGrid ev_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    std::vector<ev::WeightedPICMarker> markers;
    constexpr std::size_t count = 4096;
    constexpr double total_mass = 1.0e-6;
    markers.reserve(count);
    for (std::size_t q = 0; q < count; ++q) {
        const double u =
            (static_cast<double>(q) + 0.5) / static_cast<double>(count);
        markers.push_back(marker(
            total_mass / static_cast<double>(count),
            0.45 + 0.35 * u,
            (q & 1U) == 0 ? 1.0e-4 : -1.0e-4,
            0.0,
            static_cast<std::uint64_t>(q + 1)));
    }
    const auto ev_deposit =
        ev::deposit_weighted_einstein_vlasov_pic(
            markers, ev_grid.mesh, ev::WeightedPICParticleShape::tsc);
    const auto ev_field =
        ev::solve_weighted_einstein_vlasov_pic_field(
            ev_deposit, ev_grid);

    ev::Params vp = ev_grid.parameters;
    vp.physics_model = "vlasov_poisson";
    // The VP KDK regression path requires a reflecting numerical
    // outer wall.  No marker approaches that wall in this local weak-field
    // comparison; EV itself permits only isolated outflow at the outer face.
    vp.outer_boundary = "reflecting";
    vp.vp_poisson_reconstruction = "p2_line_mass";
    const auto vp_deposit = ev::deposit_weighted_vlasov_pic(
        markers, ev_grid.mesh, ev::WeightedPICParticleShape::tsc);
    const ev::VlasovPoissonField vp_field =
        ev::solve_weighted_pic_poisson_field(
            vp_deposit, ev_grid.mesh, 0.0, vp);
    constexpr double sample_radius = 2.5;
    const ev::MetricSample relativistic =
        ev::sample_metric(
            sample_radius,
            ev_grid.radial_coordinate,
            ev_grid.radius,
            ev_field.metric,
            ev_grid.parameters);
    const ev::VPSample newtonian = ev::sample_vp_field(
        sample_radius,
        ev_grid.radial_coordinate,
        ev_grid.radius,
        vp_field,
        vp);
    const double ev_acceleration = ev::radial_acceleration(
        sample_radius, 0.0, 0.0, relativistic, ev_grid.radius);
    const double vp_acceleration = -newtonian.phi_r;
    const double relative_error =
        std::abs(ev_acceleration - vp_acceleration)
        / std::abs(vp_acceleration);
    assert(relative_error < 2.0e-4);
    assert(ev_field.maximum_compactness < 5.0e-6);

    std::vector<ev::WeightedPICMarker> ev_markers = markers;
    std::vector<ev::WeightedPICMarker> vp_markers = markers;
    ev::WeightedEinsteinVlasovPICOptions ev_options;
    ev_options.shape = ev::WeightedPICParticleShape::tsc;
    ev_options.threads = 2;
    constexpr double dt = 0.01;
    const auto step =
        ev::advance_weighted_einstein_vlasov_pic_self_consistent(
            ev_markers, dt, ev_grid, ev_options);
    ev::begin_weighted_pic_self_consistent_kdk_parallel(
        vp_markers,
        0.0,
        dt,
        ev_grid.radial_coordinate,
        ev_grid.radius,
        vp_field,
        vp,
        2);
    const auto vp_endpoint_deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            vp_markers,
            ev_grid.mesh,
            ev::WeightedPICParticleShape::tsc,
            2);
    const ev::VlasovPoissonField vp_endpoint =
        ev::solve_weighted_pic_poisson_field(
            vp_endpoint_deposit, ev_grid.mesh, dt, vp);
    ev::finish_weighted_pic_self_consistent_kdk_parallel(
        vp_markers,
        dt,
        dt,
        ev_grid.radial_coordinate,
        ev_grid.radius,
        vp_endpoint,
        vp,
        2);
    double rms_radius_difference = 0.0;
    double rms_velocity_difference = 0.0;
    for (std::size_t q = 0; q < count; ++q) {
        rms_radius_difference +=
            std::pow(ev_markers[q].r - vp_markers[q].r, 2);
        rms_velocity_difference +=
            std::pow(ev_markers[q].w - vp_markers[q].w, 2);
    }
    rms_radius_difference =
        std::sqrt(rms_radius_difference / static_cast<double>(count));
    rms_velocity_difference =
        std::sqrt(rms_velocity_difference / static_cast<double>(count));
    assert(rms_radius_difference < 2.0e-9);
    assert(rms_velocity_difference < 2.0e-9);
    assert(nearly_equal(
        step.rest_mass_before, step.rest_mass_after, 2.0e-13));

    std::cout << "Newtonian limit: acceleration_relative_error="
              << relative_error
              << " rms_dr=" << rms_radius_difference
              << " rms_dw=" << rms_velocity_difference << '\n';
}

void test_self_consistent_mass_and_horizon_diagnostic() {
    ev::Params p = ev_pic_params(192);
    p.r_min = 0.02;
    p.r_max = 4.0;
    p.gap_stop = 0.03;
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    std::vector<ev::WeightedPICMarker> markers;
    constexpr std::size_t count = 2048;
    markers.reserve(count);
    for (std::size_t q = 0; q < count; ++q) {
        const double u =
            (static_cast<double>(q) + 0.5) / static_cast<double>(count);
        markers.push_back(marker(
            0.15 / static_cast<double>(count),
            1.4 + 0.2 * u,
            -0.15,
            1.0e-4,
            static_cast<std::uint64_t>(q + 1)));
    }
    const double initial_rest_mass =
        ev::weighted_pic_total_marker_mass(markers);
    ev::WeightedEinsteinVlasovPICOptions options;
    options.threads = 3;
    options.characteristic_substeps = 2;
    const auto step =
        ev::advance_weighted_einstein_vlasov_pic_self_consistent(
            markers, 0.01, grid, options);
    assert(step.accepted_dt > 0.0);
    assert(step.accepted_dt <= 0.01);
    assert(step.endpoint_field.maximum_compactness < 1.0);
    assert(nearly_equal(initial_rest_mass, step.rest_mass_after, 3.0e-13));
    for (const auto& value : markers) {
        assert(value.mass == 0.15 / static_cast<double>(count));
        assert(value.eta == 1.0e-4);
    }

    std::vector<ev::WeightedPICMarker> supercritical;
    supercritical.reserve(count);
    for (std::size_t q = 0; q < count; ++q) {
        const double u =
            (static_cast<double>(q) + 0.5) / static_cast<double>(count);
        supercritical.push_back(marker(
            5.0 / static_cast<double>(count),
            0.72 + 0.04 * u,
            0.0,
            0.0,
            static_cast<std::uint64_t>(10000 + q)));
    }
    const auto concentrated_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            supercritical,
            grid,
            ev::WeightedPICParticleShape::tsc,
            2);
    assert(concentrated_field.maximum_compactness < 1.0);
    assert(concentrated_field.metric.min_gap > 0.0);
    assert(
        concentrated_field.metric.face_gap.size()
        == grid.radius.size() + 1);
    const double minimum_face_gap = *std::min_element(
        concentrated_field.metric.face_gap.begin(),
        concentrated_field.metric.face_gap.end());
    const double minimum_face_lapse = std::exp(*std::min_element(
        concentrated_field.metric.face_mu.begin(),
        concentrated_field.metric.face_mu.end()));
    assert(minimum_face_gap > 0.0);
    assert(nearly_equal(
        minimum_face_gap,
        concentrated_field.metric.min_gap,
        2.0e-12,
        2.0e-14));

    ev::Params origin = p;
    origin.inner_boundary = "origin_parity";
    origin.radial_grid = "linear";
    const ev::WeightedPICFieldGrid origin_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            origin, origin.Nr);
    assert(origin_grid.mesh.edge.front() == 0.0);
    assert(nearly_equal(
        origin_grid.mesh.edge[1],
        origin.r_max / static_cast<double>(origin.Nr)));
    std::vector<ev::WeightedPICMarker> central_marker{
        marker(1.0e-6, 0.25 * p.r_min, -0.02, 0.0, 20001)};
    const auto central_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            central_marker,
            origin_grid,
            ev::WeightedPICParticleShape::tsc,
            2);
    assert(nearly_equal(central_field.rest_mass, 1.0e-6, 1.0e-13));
    assert(central_field.maximum_compactness < 1.0);
    assert(central_field.metric.face_lambda.front() == 0.0);
    assert(central_field.metric.face_mu_r.front() == 0.0);
    assert(central_field.metric.face_lambda_t.front() == 0.0);
    const ev::MetricSample central_metric =
        ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
            0.0,
            origin_grid,
            central_field.metric);
    assert(central_metric.lambda == 0.0);
    assert(central_metric.mu_r == 0.0);
    assert(central_metric.lambda_t == 0.0);

    const auto empty_origin_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            {},
            origin_grid,
            ev::WeightedPICParticleShape::tsc,
            1);
    const double orbit_radius = 0.25 * origin_grid.mesh.edge[1];
    constexpr double orbit_angular_momentum = 2.0e-4;
    const auto orbit_rhs =
        ev::weighted_einstein_vlasov_pic_detail::characteristic_rhs(
            ev::CharacteristicState{orbit_radius, 0.0},
            orbit_angular_momentum,
            origin_grid,
            empty_origin_field.metric);
    const double orbit_epsilon = std::sqrt(
        1.0
        + orbit_angular_momentum * orbit_angular_momentum
            / (orbit_radius * orbit_radius));
    const double expected_angular_acceleration =
        orbit_angular_momentum * orbit_angular_momentum
        / (orbit_radius * orbit_radius * orbit_radius * orbit_epsilon);
    assert(nearly_equal(
        orbit_rhs.dw,
        expected_angular_acceleration,
        2.0e-13,
        2.0e-15));

    bool rejected_log_origin = false;
    try {
        ev::Params log_origin = origin;
        log_origin.radial_grid = "log";
        (void)ev::make_weighted_einstein_vlasov_pic_field_grid(
            log_origin, log_origin.Nr);
    } catch (const std::runtime_error&) {
        rejected_log_origin = true;
    }
    assert(rejected_log_origin);

    bool rejected_expansion = false;
    try {
        ev::Params expanding = p;
        expanding.vp_universe = "expanding";
        (void)ev::make_weighted_einstein_vlasov_pic_field_grid(
            expanding, expanding.Nr);
    } catch (const std::runtime_error&) {
        rejected_expansion = true;
    }
    assert(rejected_expansion);

    std::cout << "EV self-consistency: rest_mass=" << step.rest_mass_after
              << " concentrated_Cmax="
              << concentrated_field.maximum_compactness
              << " concentrated_min_gap="
              << concentrated_field.metric.min_gap
              << " concentrated_min_lapse="
              << minimum_face_lapse << '\n';
}

void test_isolated_vacuum_outer_guard_is_transactional() {
    auto p = ev_pic_params(64);
    p.inner_boundary = "origin_parity";
    p.r_max = 8.0;
    const auto grid = ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    ev::WeightedEinsteinVlasovPICOptions open;
    open.threads = 1;
    auto isolated = open;
    isolated.require_vacuum_outer_boundary = true;
    for (double radius : {7.99, 7.95}) {
        const std::vector<ev::WeightedPICMarker> initial{
            marker(1.0e-3, radius, 1.0, 0.0, 0)};
        auto outflow = initial;
        const auto legacy = ev::advance_weighted_einstein_vlasov_pic_self_consistent(
            outflow, 0.1, grid, open);
        assert(outflow[0].state == ev::WeightedPICMarkerState::escaped);
        assert(legacy.escaped_rest_mass == initial[0].mass);
        auto guarded = initial;
        bool rejected = false;
        try {
            (void)ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                guarded, 0.1, grid, isolated);
        } catch (const ev::WeightedEinsteinVlasovPICVacuumOuterBoundary&) {
            rejected = true;
        }
        assert(rejected);
        assert(guarded[0].state == initial[0].state);
        assert(guarded[0].r == initial[0].r && guarded[0].w == initial[0].w);
        assert(guarded[0].mass == initial[0].mass && guarded[0].eta == initial[0].eta);
        // Previously escaped positive mass is also incompatible with this
        // isolated exterior, including at zero requested duration.
        rejected = false;
        try {
            (void)ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                outflow, 0.0, grid, isolated);
        } catch (const ev::WeightedEinsteinVlasovPICVacuumOuterBoundary&) {
            rejected = true;
        }
        assert(rejected);
    }
    auto ordinary = std::vector<ev::WeightedPICMarker>{marker(0.01, 2.0, -0.2, 0.01, 0)};
    auto guarded = ordinary;
    const auto before = ev::advance_weighted_einstein_vlasov_pic_self_consistent(
        ordinary, 0.01, grid, open);
    const auto after = ev::advance_weighted_einstein_vlasov_pic_self_consistent(
        guarded, 0.01, grid, isolated);
    assert(ordinary[0].r == guarded[0].r && ordinary[0].w == guarded[0].w);
    assert(ordinary[0].mass == guarded[0].mass && ordinary[0].eta == guarded[0].eta);
    assert(before.accepted_dt == after.accepted_dt);
    assert(before.endpoint_field.adm_mass == after.endpoint_field.adm_mass);
    assert(before.endpoint_field.metric.m == after.endpoint_field.metric.m);
    assert(before.endpoint_field.metric.mu == after.endpoint_field.metric.mu);
    assert(before.endpoint_field.metric.lambda == after.endpoint_field.metric.lambda);
    auto negligible = ordinary;
    negligible[0].state = ev::WeightedPICMarkerState::escaped;
    negligible[0].mass = std::numeric_limits<double>::denorm_min();
    bool rejected = false;
    try { ev::check_weighted_einstein_vlasov_pic_vacuum_outer_boundary(negligible); }
    catch (const ev::WeightedEinsteinVlasovPICVacuumOuterBoundary&) { rejected = true; }
    assert(rejected);
    negligible[0].mass = 0.0;
    ev::check_weighted_einstein_vlasov_pic_vacuum_outer_boundary(negligible);
    std::cout << "SC vacuum guard: midpoint/endpoint rejection, unchanged accepted "
                 "state, zero-escape identity, and library outflow passed\n";
}

}  // namespace

int main() {
    test_origin_tsc_constant_density();
    test_exact_origin_wall_and_outflow_orbits();
    test_constraint_lapse_interpolation_and_time_convergence();
    test_relativistic_deposition_and_constraints();
    test_initialization_bridge_and_minkowski_orbit();
    test_newtonian_limit_agrees_with_vlasov_poisson();
    test_self_consistent_mass_and_horizon_diagnostic();
    test_isolated_vacuum_outer_guard_is_transactional();
    std::cout << "weighted Einstein-Vlasov PIC tests passed\n";
    return 0;
}
