// Focused regression tests for the standalone uncharged ingoing
// Eddington--Finkelstein weighted Einstein--Vlasov PIC kernel.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ev/weighted_einstein_vlasov_pic_schwarzschild.hpp"
#include "ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp"

namespace ef = ev::eddington_finkelstein;

namespace {

bool close(
    double first,
    double second,
    double relative_tolerance = 2.0e-12,
    double absolute_tolerance = 2.0e-14) {
    return std::abs(first - second)
        <= absolute_tolerance
            + relative_tolerance
                * std::max(std::abs(first), std::abs(second));
}

ev::Params parameters(int cells = 256) {
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.vp_universe = "static";
    p.bp_collision_operator = "none";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = cells;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = 0.25;
    p.r_max = 8.0;
    p.radial_grid = "linear";
    p.w_max = 2.0;
    p.L_max = 2.0;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "outflow_vacuum";
    return p;
}

ef::Marker marker(
    double rest_weight,
    double radius,
    double p_r,
    double eta,
    std::uint64_t id) {
    ef::Marker result;
    result.rest_weight = rest_weight;
    result.r = radius;
    result.p_r = p_r;
    result.eta = eta;
    result.id = id;
    result.source_id = 1000 + id;
    return result;
}

ef::Field minkowski_field(const ev::WeightedPICFieldGrid& grid) {
    return ef::deposit_and_solve_field(
        {}, grid, ev::WeightedPICParticleShape::tsc, 1);
}

void test_origin_parity_grid_contract() {
    ev::Params p = parameters(64);
    // The production convention deliberately retains a positive configured
    // r_min; origin_parity, not the scalar r_min value, selects the r=0 mesh.
    p.r_min = 0.25;
    p.inner_boundary = "origin_parity";
    p.radial_grid = "linear";
    const ev::WeightedPICFieldGrid grid = ef::make_field_grid(p, p.Nr);
    assert(grid.mesh.edge.front() == 0.0);
    assert(grid.mesh.edge.back() == p.r_max);
    assert(grid.radius.front() > 0.0);
    assert(grid.radius.front() < grid.mesh.edge[1]);
    assert(close(
        grid.radius.front(),
        std::cbrt(0.5 * std::pow(grid.mesh.edge[1], 3))));
    const ef::Field flat = minkowski_field(grid);
    assert(flat.metric.face_a.front() == 1.0);
    assert(flat.metric.face_b.back() == 1.0);

    bool rejected_singular_angular_marker = false;
    try {
        static_cast<void>(ef::deposit_sources(
            {marker(1.0e-6, 0.0, 1.0, 0.1, 99)},
            grid.mesh,
            ev::WeightedPICParticleShape::cic));
    } catch (const std::runtime_error&) {
        rejected_singular_angular_marker = true;
    }
    assert(rejected_singular_angular_marker);
}

void test_minkowski_deposition_and_geodesics() {
    const ev::WeightedPICFieldGrid grid = ef::make_field_grid(parameters(), 256);
    const std::size_t cell = 80;
    const double radius = grid.radius[cell];
    const double dm = 0.037;
    const double p_r = 1.7;
    const double eta = 0.23;
    const std::vector<ef::Marker> markers{
        marker(dm, radius, p_r, eta, 1),
    };
    const ef::DepositedSources source = ef::deposit_sources(
        markers, grid.mesh, ev::WeightedPICParticleShape::cic);
    const double chi = 1.0 + 2.0 * eta / (radius * radius);
    assert(close(source.deposited_rest_weight, dm));
    assert(close(source.shell_rest_weight[cell], dm));
    assert(close(source.shell_t_rr_numerator[cell], dm * p_r));
    assert(close(source.shell_s_numerator[cell], dm * chi / p_r));
    assert(close(source.t_rr[cell], dm * p_r / grid.mesh.shell_volume[cell]));
    assert(close(source.s[cell], dm * chi / (p_r * grid.mesh.shell_volume[cell])));

    const ef::Field flat = minkowski_field(grid);
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        assert(flat.metric.a[i] == 1.0);
        assert(flat.metric.b[i] == 1.0);
        assert(flat.metric.m[i] == 0.0);
    }

    {
        std::vector<ef::Marker> radial{
            marker(1.0e-12, 3.0, 0.5, 0.0, 2),
            marker(1.0e-12, 3.5, 2.0, 0.0, 3),
        };
        const std::vector<ef::Marker> before = radial;
        constexpr double dt = 0.08;
        ef::advance_frozen_metric(radial, dt, grid, flat.metric, 1);
        for (std::size_t i = 0; i < radial.size(); ++i) {
            const double velocity =
                0.5 * (1.0 - 1.0 / (before[i].p_r * before[i].p_r));
            assert(close(radial[i].r, before[i].r + dt * velocity));
            assert(radial[i].p_r == before[i].p_r);
            assert(radial[i].rest_weight == before[i].rest_weight);
            assert(radial[i].eta == before[i].eta);
            assert(radial[i].id == before[i].id);
            assert(radial[i].source_id == before[i].source_id);
        }
    }

    {
        std::vector<ef::Marker> angular{
            marker(1.0e-12, 3.0, 1.4, 0.7, 4),
        };
        auto energy = [](const ef::Marker& value) {
            const double chi_value =
                1.0 + 2.0 * value.eta / (value.r * value.r);
            return 0.5 * (value.p_r + chi_value / value.p_r);
        };
        const double before = energy(angular.front());
        ef::advance_frozen_metric(angular, 0.1, grid, flat.metric, 32);
        assert(close(energy(angular.front()), before, 2.0e-11, 2.0e-13));
    }
}

void test_exactly_radial_tiny_radius_limits() {
    auto p = parameters(8);
    p.inner_boundary = "origin_parity";
    const auto grid = ef::make_field_grid(p, 8);
    const auto flat = minkowski_field(grid);
    for (double radius : {1.0e-100, 1.0e-120, 1.0e-160, 1.0e-200}) {
        for (double momentum : {0.5, 1.0, 2.0}) {
            const double velocity = 0.5 * (1.0 - 1.0 / (momentum * momentum));
            const auto derivative = ef::characteristic_rhs(
                {radius, momentum}, 0.0, ef::MetricSample{});
            assert(derivative.dr == velocity && derivative.dp_r == 0.0);
            assert(ef::infall_observer_energy(radius, momentum, 0.0)
                == 0.5 * (momentum + 1.0 / momentum));
            const double dt = 0.01 * radius;
            const auto advanced = ef::rk4_advance(
                {radius, momentum}, 0.0, dt, 1, grid, flat.metric);
            assert(std::abs((advanced.r - (radius + dt * velocity)) / radius)
                < 1.0e-14);
            assert(advanced.p_r == momentum);
        }
        for (double w : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
            const double expected_energy = std::hypot(1.0, w);
            const double canonical = ef::canonical_p_r_from_schwarzschild_w(
                w, 0.0, radius, 1.0);
            assert(close(canonical, expected_energy + w));
            assert(close(ef::schwarzschild_w_from_canonical_p_r(
                canonical, 0.0, radius, 1.0), w));
            assert(close(ef::outgoing_canonical_p_r_from_schwarzschild_w(
                w, 0.0, radius, 1.0), w - expected_energy));
        }
    }
    std::cout << "EF exactly radial positive-radius limits: flat characteristics, "
        "log-momentum RK4, observer energy and coordinate maps passed\n";
}

void test_metric_sampler_entry_validation() {
    auto p = parameters(8);
    p.inner_boundary = "origin_parity";
    const auto grid = ef::make_field_grid(p, 8);
    const auto flat = minkowski_field(grid);
    const ef::MetricSampler sampled(grid, flat.metric);
    for (double radius : {0.0, 0.1, 1.0, 8.0, 9.0}) {
        const auto value = sampled(radius);
        const auto standalone = ef::sample_metric(radius, grid, flat.metric);
        assert(value.a == standalone.a && value.b == standalone.b);
        assert(value.a_r == standalone.a_r && value.b_r == standalone.b_r);
        assert(value.m == standalone.m);
    }
    for (int defect = 0; defect < 8; ++defect) {
        auto invalid_grid = grid;
        auto invalid_metric = flat.metric;
        if (defect == 0) invalid_grid.mesh.edge[3] = invalid_grid.mesh.edge[2];
        if (defect == 1) invalid_metric.face_radius.clear();
        if (defect == 2) invalid_metric.face_m.clear();
        if (defect == 3) invalid_metric.radius.clear();
        if (defect == 4) invalid_metric.face_radius[3] =
            std::numeric_limits<double>::quiet_NaN();
        if (defect == 5) invalid_metric.face_radius[3] =
            invalid_metric.face_radius[2];
        if (defect == 6) invalid_metric.face_radius[3] += 0.01;
        if (defect == 7) invalid_metric.radius[3] += 0.01;
        bool single_rejected = false;
        try { (void)ef::sample_metric(1.0, invalid_grid, invalid_metric); }
        catch (const std::runtime_error&) { single_rejected = true; }
        assert(single_rejected);
        std::vector<ef::Marker> values{marker(1e-12, 1.0, 0.5, 0.0, 7)};
        bool batch_rejected = false;
        try {
            ef::advance_frozen_metric_parallel(
                values, 0.1, invalid_grid, invalid_metric, 2);
        } catch (const std::runtime_error&) { batch_rejected = true; }
        assert(batch_rejected);
        assert(values[0].r == 1.0 && values[0].p_r == 0.5);
    }
}

void test_conversion_round_trip() {
    constexpr double radius = 3.7;
    constexpr double eta = 0.41;
    constexpr double a = 0.37;
    for (double w : {-1.0e6, -7.0, -0.2, 0.0, 0.3, 9.0, 1.0e6}) {
        const double p_r = ef::canonical_p_r_from_schwarzschild_w(
            w, eta, radius, a);
        assert(p_r > 0.0);
        const double recovered = ef::schwarzschild_w_from_canonical_p_r(
            p_r, eta, radius, a);
        assert(close(recovered, w, 3.0e-11, 3.0e-11));
    }

    // Audit the local dynamical coordinate transformation as a covector map.
    // F=(partial_tau v)^-1 is a free positive null-time normalization and is
    // encoded in b=F exp(lambda+mu).  It must appear in -p_v even though it
    // cancels out of the radial covector P.
    constexpr double mu = -0.23;
    constexpr double integrating_factor = 1.7;
    const double lambda = -0.5 * std::log(a);
    const double b = integrating_factor * std::exp(lambda + mu);
    for (double w : {-2.1, -0.3, 0.0, 0.8, 3.4}) {
        const double chi = 1.0 + 2.0 * eta / (radius * radius);
        const double epsilon = std::sqrt(chi + w * w);
        const double polar_p_tau = -std::exp(mu) * epsilon;
        const double polar_p_r = std::exp(lambda) * w;
        const double partial_r_tau_at_fixed_v = -std::exp(lambda - mu);
        const double transformed_p_r = polar_p_r
            + polar_p_tau * partial_r_tau_at_fixed_v;
        const double transformed_p_v =
            integrating_factor * polar_p_tau;
        const double direct_p_r = ef::canonical_p_r_from_schwarzschild_w(
            w, eta, radius, a);
        const double ef_hamiltonian = 0.5 * b
            * (a * direct_p_r + chi / direct_p_r);
        assert(close(transformed_p_r, direct_p_r));
        assert(close(
            ef_hamiltonian,
            integrating_factor * std::exp(mu) * epsilon));
        assert(close(ef_hamiltonian, -transformed_p_v));
    }

    // East's construction can instead start with a future-directed outgoing
    // (retarded-chart) particle, p_r<0, and apply t->-t.  Reversing both its
    // physical radial motion and signed canonical momentum must give exactly
    // the direct ingoing map used by the VENUS shell sampler.
    for (double outgoing_w : {0.0, 0.2, 2.0, 1.0e6}) {
        const double outgoing_p_r =
            ef::outgoing_canonical_p_r_from_schwarzschild_w(
                outgoing_w, eta, radius, a);
        assert(outgoing_p_r < 0.0);
        const double reversed =
            ef::time_reverse_outgoing_canonical_p_r(outgoing_p_r);
        const double direct_ingoing =
            ef::canonical_p_r_from_schwarzschild_w(
                -outgoing_w, eta, radius, a);
        assert(close(reversed, direct_ingoing, 3.0e-12, 3.0e-12));
    }

    ev::WeightedPICMarker schwarzschild;
    schwarzschild.mass = 0.3;
    schwarzschild.r = radius;
    schwarzschild.w = -0.8;
    schwarzschild.eta = eta;
    schwarzschild.id = 55;
    schwarzschild.source_cell = 77;
    const ef::Marker converted = ef::from_schwarzschild_marker(schwarzschild, a);
    const ev::WeightedPICMarker recovered =
        ef::to_schwarzschild_marker(converted, a);
    assert(recovered.mass == schwarzschild.mass);
    assert(recovered.r == schwarzschild.r);
    assert(close(recovered.w, schwarzschild.w));
    assert(recovered.eta == schwarzschild.eta);
    assert(recovered.id == schwarzschild.id);
    assert(recovered.source_cell == schwarzschild.source_cell);

    schwarzschild.internal_energy = 1.0e-4;
    bool rejected_internal_energy = false;
    try {
        static_cast<void>(ef::from_schwarzschild_marker(schwarzschild, a));
    } catch (const std::runtime_error&) {
        rejected_internal_energy = true;
    }
    assert(rejected_internal_energy);
}

void test_dynamic_eikonal_tensor_pullback_identities() {
    // This is a local, time-dependent coordinate-map test.  It deliberately
    // uses A=partial_tau v != 1, so a static tortoise substitution would fail
    // even though the exact eikonal and tensor identities remain valid.
    constexpr double lambda = 0.31;
    constexpr double mu = -0.17;
    constexpr double a_factor = 1.63;
    const double inverse_factor = 1.0 / a_factor;
    const double h = std::exp(lambda - mu);
    const double v_r = a_factor * h;
    const double a = std::exp(-2.0 * lambda);
    const double b = inverse_factor * std::exp(lambda + mu);

    // Both forms of the ingoing eikonal equation vanish.
    const double eikonal = -std::exp(-2.0 * mu)
            * a_factor * a_factor
        + std::exp(-2.0 * lambda) * v_r * v_r;
    const double inverse_eikonal = -h
        + inverse_factor / (a * b);
    assert(close(eikonal, 0.0, 0.0, 3.0e-15));
    assert(close(inverse_eikonal, 0.0, 0.0, 3.0e-15));
    assert(close(v_r, 1.0 / (a * b)));

    // Pull the EF metric back with J=d(v,r)/d(tau,r).
    const double pulled_g_tt = -a * b * b * a_factor * a_factor;
    const double pulled_g_tr = -a * b * b * a_factor * v_r
        + b * a_factor;
    const double pulled_g_rr = -a * b * b * v_r * v_r
        + 2.0 * b * v_r;
    assert(close(pulled_g_tt, -std::exp(2.0 * mu)));
    assert(close(pulled_g_tr, 0.0, 0.0, 3.0e-15));
    assert(close(pulled_g_rr, std::exp(2.0 * lambda)));

    // Start with an arbitrary symmetric radial stress tensor in SC
    // components, transform it with the inverse Jacobian, then pull it back.
    // This tests the signs of both off-diagonal transformations.
    constexpr double sc_tt = 0.73;
    constexpr double sc_tr = -0.19;
    constexpr double sc_rr = 0.41;
    const double ef_vv = inverse_factor * inverse_factor * sc_tt;
    const double ef_vr = -inverse_factor * h * sc_tt
        + inverse_factor * sc_tr;
    const double ef_rr = h * h * sc_tt - 2.0 * h * sc_tr + sc_rr;
    const double recovered_tt = a_factor * a_factor * ef_vv;
    const double recovered_tr = a_factor
        * (v_r * ef_vv + ef_vr);
    const double recovered_rr = v_r * v_r * ef_vv
        + 2.0 * v_r * ef_vr + ef_rr;
    assert(close(recovered_tt, sc_tt));
    assert(close(recovered_tr, sc_tr));
    assert(close(recovered_rr, sc_rr));
}

void test_characteristics_are_canonical_hamilton_equations() {
    constexpr double radius = 3.2;
    constexpr double p_r = 0.73;
    constexpr double eta = 0.42;
    const ef::MetricSample metric{
        0.38, 0.81, 0.17, 0.05, 0.992,
    };
    const ef::CharacteristicDerivative rhs = ef::characteristic_rhs(
        {radius, p_r}, eta, metric);
    const double chi = 1.0 + 2.0 * eta / (radius * radius);
    const double d_chi_dr = -4.0 * eta
        / (radius * radius * radius);
    const double d_hamiltonian_dp = 0.5 * metric.b
        * (metric.a - chi / (p_r * p_r));
    const double d_hamiltonian_dr = 0.5 * metric.b_r
            * (metric.a * p_r + chi / p_r)
        + 0.5 * metric.b
            * (metric.a_r * p_r + d_chi_dr / p_r);
    assert(close(rhs.dr, d_hamiltonian_dp));
    assert(close(rhs.dp_r, -d_hamiltonian_dr));
    // Consequently a static metric conserves H=-p_v along a characteristic.
    assert(close(
        d_hamiltonian_dr * rhs.dr
            + d_hamiltonian_dp * rhs.dp_r,
        0.0));
}

void test_current_free_schwarzschild_to_ef_constraint_map() {
    ev::Params p = parameters(384);
    const ev::WeightedPICFieldGrid schwarzschild_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    const ev::WeightedPICFieldGrid ef_grid = ef::make_field_grid(p, p.Nr);
    assert(schwarzschild_grid.radius == ef_grid.radius);
    assert(schwarzschild_grid.mesh.edge == ef_grid.mesh.edge);

    // w=0 makes the instantaneous radial current vanish.  On such a slice the
    // continuum coordinate relations a=e^{-2 lambda}, b=e^{lambda+mu} and the
    // canonical momentum map must give the same radial constraints.  This is
    // a direct normalization regression for the marker T_rr and S deposits.
    std::vector<ev::WeightedPICMarker> schwarzschild_markers;
    for (std::size_t i = 55; i < 315; i += 11) {
        ev::WeightedPICMarker value;
        value.mass = 3.0e-6
            * (1.0 + static_cast<double>(i % 4));
        value.r = schwarzschild_grid.radius[i];
        value.w = 0.0;
        value.eta = 0.08 + 0.0005 * static_cast<double>(i);
        value.id = 10000 + i;
        value.source_cell = i;
        schwarzschild_markers.push_back(value);
    }
    const ev::WeightedEinsteinVlasovPICField schwarzschild_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            schwarzschild_markers,
            schwarzschild_grid,
            ev::WeightedPICParticleShape::tsc,
            1);
    std::vector<ef::Marker> ef_markers;
    ef_markers.reserve(schwarzschild_markers.size());
    for (const ev::WeightedPICMarker& source : schwarzschild_markers) {
        const ev::MetricSample sampled =
            ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
                source.r, schwarzschild_grid, schwarzschild_field.metric);
        ef_markers.push_back(ef::from_schwarzschild_marker(
            source, std::exp(-2.0 * sampled.lambda)));
        assert(ef_markers.back().rest_weight == source.mass);
        assert(ef_markers.back().eta == source.eta);
    }
    const ef::Field ef_field = ef::deposit_and_solve_field(
        ef_markers,
        ef_grid,
        ev::WeightedPICParticleShape::tsc,
        1);
    assert(close(
        ef_field.metric.adm_mass,
        schwarzschild_field.adm_mass,
        2.0e-8,
        2.0e-12));
    for (std::size_t i = 0; i < ef_grid.radius.size(); ++i) {
        const double expected_a = std::exp(
            -2.0 * schwarzschild_field.metric.lambda[i]);
        const double expected_b = std::exp(
            schwarzschild_field.metric.lambda[i]
                + schwarzschild_field.metric.mu[i]);
        assert(close(ef_field.metric.a[i], expected_a, 3.0e-8, 3.0e-11));
        assert(close(ef_field.metric.b[i], expected_b, 3.0e-8, 3.0e-11));
        assert(close(
            ef_field.metric.m[i],
            schwarzschild_field.metric.m[i],
            3.0e-8,
            3.0e-11));
    }

    // With radial current the two constant-time hypersurfaces are not the
    // same physical slice, so b need not agree under a pointwise marker map.
    // The algebraic mass-source identity does still hold marker by marker:
    // (a P + chi/P)/2 = sqrt(a) epsilon.  Exercise it at nonzero w without
    // making the scientifically incorrect claim that the lapse/shift agrees.
    std::vector<ev::WeightedPICMarker> moving = schwarzschild_markers;
    for (std::size_t q = 0; q < moving.size(); ++q) {
        moving[q].w = -0.9 + 1.8 * static_cast<double>(q)
            / static_cast<double>(moving.size() - 1);
    }
    const ev::WeightedEinsteinVlasovPICField moving_schwarzschild =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            moving,
            schwarzschild_grid,
            ev::WeightedPICParticleShape::tsc,
            1);
    std::vector<ef::Marker> moving_ef;
    moving_ef.reserve(moving.size());
    for (const ev::WeightedPICMarker& source : moving) {
        const ev::MetricSample sampled =
            ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
                source.r, schwarzschild_grid, moving_schwarzschild.metric);
        moving_ef.push_back(ef::from_schwarzschild_marker(
            source, std::exp(-2.0 * sampled.lambda)));
    }
    const ef::Field moving_ef_field = ef::deposit_and_solve_field(
        moving_ef,
        ef_grid,
        ev::WeightedPICParticleShape::tsc,
        1);
    assert(close(
        moving_ef_field.metric.adm_mass,
        moving_schwarzschild.adm_mass,
        3.0e-8,
        3.0e-11));
    for (std::size_t i = 0; i < ef_grid.radius.size(); ++i) {
        assert(close(
            moving_ef_field.metric.a[i],
            moving_schwarzschild.metric.gap[i],
            5.0e-8,
            5.0e-11));
        assert(close(
            moving_ef_field.metric.m[i],
            moving_schwarzschild.metric.m[i],
            5.0e-8,
            5.0e-11));
    }
}

void test_vacuum_schwarzschild_and_horizon_finite_rhs() {
    ev::Params p = parameters(512);
    const ev::WeightedPICFieldGrid grid = ef::make_field_grid(p, p.Nr);
    ef::ConstraintOptions fixed;
    fixed.fixed_adm_mass = 1.0;
    const ef::Field vacuum = ef::deposit_and_solve_field(
        {}, grid, ev::WeightedPICParticleShape::tsc, 1, fixed);
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        const double exact_a = 1.0 - 2.0 / grid.radius[i];
        assert(close(vacuum.metric.a[i], exact_a, 1.0e-13));
        assert(vacuum.metric.b[i] == 1.0);
        assert(vacuum.metric.m[i] == 1.0);
        assert(close(vacuum.metric.a_r[i], 2.0 / std::pow(grid.radius[i], 2)));
        assert(vacuum.metric.b_r[i] == 0.0);
    }
    assert(vacuum.metric.apparent_horizon_found);
    assert(std::abs(vacuum.metric.apparent_horizon_radius - 2.0) < 2.0e-3);
    assert(vacuum.metric.minimum_a < 0.0);

    const ef::MetricSample horizon{0.0, 1.0, 0.5, 0.0, 1.0};
    const ef::CharacteristicDerivative rhs = ef::characteristic_rhs(
        {2.0, 0.75}, 0.2, horizon);
    const double chi = 1.0 + 0.4 / 4.0;
    assert(close(rhs.dr, -0.5 * chi / (0.75 * 0.75)));
    assert(close(rhs.dp_r, -0.5 * 0.75 * 0.5 + 0.4 / (0.75 * 8.0)));
    assert(std::isfinite(rhs.dr));
    assert(std::isfinite(rhs.dp_r));

    std::vector<ef::Marker> near_horizon{
        marker(1.0e-12, 2.01, 0.75, 0.2, 56),
    };
    const double dt = ef::choose_timestep(
        near_horizon, grid, vacuum.metric, 0.3, 0.05);
    assert(dt > 0.0 && dt <= 0.05 && std::isfinite(dt));
    ef::advance_frozen_metric(
        near_horizon, std::min(dt, 1.0e-3), grid, vacuum.metric, 4);
    assert(std::isfinite(near_horizon.front().r));
    assert(near_horizon.front().p_r > 0.0);
}

void test_log_momentum_pusher_preserves_future_branch() {
    const ev::WeightedPICFieldGrid grid = ef::make_field_grid(parameters(), 256);
    ef::Metric metric = minkowski_field(grid).metric;
    // A deliberately stiff but spatially constant synthetic sample has
    // d(log p_r)/dv=-50.  A direct explicit half-stage in p_r would be
    // negative for dt=0.1; the production log-p_r stages remain positive.
    std::fill(metric.a.begin(), metric.a.end(), 1.0);
    std::fill(metric.face_a.begin(), metric.face_a.end(), 1.0);
    std::fill(metric.b.begin(), metric.b.end(), 1.0e-8);
    std::fill(metric.face_b.begin(), metric.face_b.end(), 1.0e-8);
    std::fill(metric.a_r.begin(), metric.a_r.end(), 1.0e10);
    std::fill(metric.face_a_r.begin(), metric.face_a_r.end(), 1.0e10);
    std::fill(metric.b_r.begin(), metric.b_r.end(), 0.0);
    std::fill(metric.face_b_r.begin(), metric.face_b_r.end(), 0.0);
    const ef::CharacteristicDerivative initial = ef::characteristic_rhs(
        {3.0, 1.0}, 0.0, ef::sample_metric(3.0, grid, metric));
    assert(close(initial.dp_r, -50.0));
    assert(1.0 + 0.05 * initial.dp_r < 0.0);

    const ef::CharacteristicState advanced = ef::rk4_advance(
        {3.0, 1.0}, 0.0, 0.1, 1, grid, metric);
    assert(advanced.p_r > 0.0);
    assert(close(advanced.p_r, std::exp(-5.0), 2.0e-12));
    assert(advanced.r > 0.0);
}

void test_constraint_identities_and_parallel_determinism() {
    ev::Params p = parameters(192);
    const ev::WeightedPICFieldGrid grid = ef::make_field_grid(p, p.Nr);
    std::vector<ef::Marker> markers;
    for (std::size_t i = 12; i < 170; i += 3) {
        markers.push_back(marker(
            1.0e-5 * (1.0 + static_cast<double>(i % 5)),
            grid.radius[i],
            0.6 + 0.01 * static_cast<double>(i),
            0.002 * static_cast<double>(i),
            100 + i));
    }
    const ef::DepositedSources serial = ef::deposit_sources_parallel(
        markers, grid.mesh, ev::WeightedPICParticleShape::tsc, 1);
    const ef::DepositedSources parallel = ef::deposit_sources_parallel(
        markers, grid.mesh, ev::WeightedPICParticleShape::tsc, 7);
    assert(serial.shell_rest_weight == parallel.shell_rest_weight);
    assert(serial.shell_t_rr_numerator == parallel.shell_t_rr_numerator);
    assert(serial.shell_s_numerator == parallel.shell_s_numerator);
    assert(serial.t_rr == parallel.t_rr);
    assert(serial.s == parallel.s);
    assert(serial.effective_marker_number == parallel.effective_marker_number);
    assert(serial.active_rest_weight == parallel.active_rest_weight);

    const ef::Field field = ef::solve_field(serial, grid);
    assert(close(field.metric.face_mb.front(), 0.0, 0.0, 2.0e-18));
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        const double b_constraint = 4.0 * ev::PI * grid.radius[i]
            * field.metric.b[i] * serial.t_rr[i];
        const double a_constraint =
            (1.0 - field.metric.a[i]) / grid.radius[i]
            - 4.0 * ev::PI * grid.radius[i]
                * (field.metric.a[i] * serial.t_rr[i] + serial.s[i]);
        assert(close(field.metric.b_r[i], b_constraint));
        assert(close(field.metric.a_r[i], a_constraint));
        assert(field.metric.b[i] > 0.0);
        assert(field.metric.face_b[i] <= field.metric.face_b[i + 1]);
        assert(field.metric.face_mb[i] <= field.metric.face_mb[i + 1]);
    }
    assert(field.metric.adm_mass > 0.0);
    assert(close(
        field.metric.face_mb.back(),
        field.metric.adm_mass,
        1.0e-13));

    // The first regular slice is solved outward to infer M.  Re-solving the
    // identical source inward with that fixed ADM value must reproduce the
    // same geometry; otherwise the orientation or normalization of (mb)_r is
    // inconsistent.
    ef::ConstraintOptions fixed;
    fixed.fixed_adm_mass = field.metric.adm_mass;
    const ef::Field inward = ef::solve_field(serial, grid, fixed);
    assert(inward.metric.fixed_adm_mass_used);
    assert(close(inward.metric.face_mb.front(), 0.0, 0.0, 2.0e-17));
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        assert(close(inward.metric.a[i], field.metric.a[i], 2.0e-12));
        assert(close(inward.metric.b[i], field.metric.b[i], 2.0e-12));
        assert(close(inward.metric.m[i], field.metric.m[i], 2.0e-12));
        assert(close(inward.metric.mb[i], field.metric.mb[i], 2.0e-12));
    }

    const ef::Field flat = minkowski_field(grid);
    std::vector<ef::Marker> serial_push = markers;
    std::vector<ef::Marker> parallel_push = markers;
    ef::advance_frozen_metric_parallel(
        serial_push, 0.01, grid, flat.metric, 1, 4);
    ef::advance_frozen_metric_parallel(
        parallel_push, 0.01, grid, flat.metric, 8, 4);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        assert(serial_push[i].r == parallel_push[i].r);
        assert(serial_push[i].p_r == parallel_push[i].p_r);
        assert(serial_push[i].state == parallel_push[i].state);
    }
}

void test_self_consistent_horizon_excision_and_mass_retention() {
    ev::Params p = parameters(512);
    const ev::WeightedPICFieldGrid grid = ef::make_field_grid(p, p.Nr);
    std::vector<ef::Marker> markers{
        marker(2.0e-6, 0.8, 0.9, 0.0, 9001),
        marker(3.0e-6, 3.0, 1.2, 0.03, 9002),
    };
    ef::EvolutionOptions options;
    options.shape = ev::WeightedPICParticleShape::tsc;
    options.threads = 3;
    options.fixed_adm_mass = 1.0;
    options.enable_excision = true;
    options.excision_a_threshold = -0.01;
    options.excision_buffer_cells = 4;
    options.excision_horizon_fraction = 1.0; // cap at the horizon radius
    const ef::StepResult step = ef::advance_self_consistent(
        markers, 1.0e-4, grid, options);
    assert(step.apparent_horizon_found);
    assert(step.endpoint_field.metric.excision_radius > 0.8);
    assert(markers[0].state == ef::MarkerState::excised);
    assert(markers[1].state == ef::MarkerState::active);
    assert(step.newly_excised_markers == 1);
    assert(close(step.newly_excised_rest_weight, 2.0e-6));
    assert(close(step.endpoint_field.sources.excised_rest_weight, 2.0e-6));
    assert(close(step.rest_weight_before, step.rest_weight_after, 1.0e-14));
    assert(step.endpoint_field.metric.fixed_adm_mass_used);
    assert(step.endpoint_field.metric.adm_mass == 1.0);

    // The accepted marker array and returned endpoint field must describe the
    // same constrained slice, including after a newly triggered excision.
    ef::ConstraintOptions constraints;
    constraints.fixed_adm_mass = 1.0;
    constraints.excision_a_threshold = options.excision_a_threshold;
    constraints.excision_buffer_cells = options.excision_buffer_cells;
    constraints.excision_horizon_fraction = options.excision_horizon_fraction;
    const ef::Field independently_reconstructed = ef::deposit_and_solve_field(
        markers,
        grid,
        options.shape,
        options.threads,
        constraints);
    assert(independently_reconstructed.sources.shell_rest_weight
        == step.endpoint_field.sources.shell_rest_weight);
    assert(independently_reconstructed.sources.t_rr
        == step.endpoint_field.sources.t_rr);
    assert(independently_reconstructed.sources.s
        == step.endpoint_field.sources.s);
    assert(independently_reconstructed.metric.a
        == step.endpoint_field.metric.a);
    assert(independently_reconstructed.metric.b
        == step.endpoint_field.metric.b);
    assert(independently_reconstructed.metric.m
        == step.endpoint_field.metric.m);
}

void test_deep_excision_keeps_horizon_crossing_particles() {
    ev::Params p = parameters(1024);
    p.inner_boundary = "origin_parity";
    const auto grid = ef::make_field_grid(p, p.Nr);
    ef::ConstraintOptions options;
    options.fixed_adm_mass = 1.0;
    const auto vacuum = ef::deposit_and_solve_field(
        {}, grid, ev::WeightedPICParticleShape::tsc, 1, options);
    const double rex = vacuum.metric.excision_radius;
    assert(rex > 0.18 && rex <= 0.1 * vacuum.metric.apparent_horizon_radius);
    assert(ef::sample_metric(rex, grid, vacuum.metric).a < -0.01);
    std::vector<ef::Marker> particles{
        marker(1e-12, 2.1, 1.0 / (1.0 + std::sqrt(2.0 / 2.1)), 0.0, 9100)};
    bool crossed_horizon = false, reached_deep_interior = false;
    for (int n = 0; n < 20000 && particles[0].state == ef::MarkerState::active; ++n) {
        const double dt = ef::choose_timestep(particles, grid, vacuum.metric, 0.2, 0.002);
        ef::advance_frozen_metric(particles, dt, grid, vacuum.metric, 2);
        if (particles[0].state == ef::MarkerState::active) {
            crossed_horizon = crossed_horizon || particles[0].r < 2.0;
            reached_deep_interior = reached_deep_interior || particles[0].r < 0.3;
            assert(particles[0].r > rex);
            assert(ef::infall_observer_energy(particles[0].r, particles[0].p_r, 0.0) > 0.0);
        }
    }
    assert(crossed_horizon && reached_deep_interior);
    assert(particles[0].state == ef::MarkerState::excised);
    assert(particles[0].r <= rex && particles[0].r > 0.0);
    assert(particles[0].rest_weight == 1e-12);

    // An outer trapped shell cannot justify excising an untrapped cavity.
    std::vector<ef::Marker> shell{marker(2.0, 1.0, 1.0, 0.0, 9200)};
    ef::ConstraintOptions regular;
    const auto shell_field = ef::deposit_and_solve_field(
        shell, grid, ev::WeightedPICParticleShape::tsc, 1, regular);
    assert(shell_field.metric.apparent_horizon_found);
    assert(shell_field.metric.a.front() > 0.0);
    assert(shell_field.metric.excision_radius == 0.0);

    // Invalid fractions fail rather than silently restoring shallow excision.
    for (double invalid : {0.0, -0.1, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        options.excision_horizon_fraction = invalid;
        bool rejected = false;
        try { (void)ef::deposit_and_solve_field({}, grid, ev::WeightedPICParticleShape::tsc, 1, options); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
}

void test_interior_horizon_crossing_and_retained_sources() {
    auto p = parameters(1024);
    p.inner_boundary = "origin_parity";
    const auto grid = ef::make_field_grid(p, p.Nr);
    // The added control is last, preserving the older four-field aggregates.
    const ef::ConstraintOptions legacy{1.0, -0.01, 4, 0.1};
    assert(legacy.enable_excision);
    auto controls = legacy;
    controls.enable_excision = false;
    const auto vacuum = ef::deposit_and_solve_field(
        {}, grid, ev::WeightedPICParticleShape::tsc, 1, controls);
    assert(vacuum.metric.apparent_horizon_found);
    assert(vacuum.metric.excision_radius == 0.0);
    const auto flat = ef::deposit_and_solve_field(
        {}, grid, ev::WeightedPICParticleShape::tsc, 1, ef::ConstraintOptions{});
    const std::vector<ef::Marker> near_origin{
        marker(1.0e-12, 1.0e-4, 0.5, 0.0, 9299)};
    const double approach_dt = ef::choose_timestep(
        near_origin, grid, flat.metric, 0.2, 1.0);
    const double exact_domain_bound = 0.2 * 1.0e-4 / 1.5;
    assert(close(approach_dt, exact_domain_bound));
    std::vector<ef::Marker> particles{
        marker(1.0e-12, 2.1, 1.0 / (1.0 + std::sqrt(2.0 / 2.1)), 0.0, 9300)};
    bool crossed = false;
    for (int n = 0; n < 20000 && particles[0].r > 0.12; ++n) {
        const double dt = ef::choose_timestep(particles, grid, vacuum.metric, 0.2, 0.002);
        ef::advance_frozen_metric(particles, dt, grid, vacuum.metric, 2, false);
        crossed = crossed || particles[0].r < 2.0;
        assert(particles[0].state == ef::MarkerState::active);
        assert(particles[0].r > 0.0 && particles[0].p_r > 0.0);
    }
    assert(crossed && particles[0].r <= 0.12);
    assert(particles[0].r < 0.1 * vacuum.metric.apparent_horizon_radius);
    ef::EvolutionOptions evolution;
    evolution.fixed_adm_mass = 1.0;
    evolution.enable_excision = false;
    evolution.threads = 1;
    const auto step = ef::advance_self_consistent(particles, 1.0e-7, grid, evolution);
    assert(step.endpoint_field.metric.apparent_horizon_found);
    assert(step.endpoint_field.metric.excision_radius == 0.0);
    assert(step.newly_excised_markers == 0);
    assert(step.endpoint_field.sources.excised_rest_weight == 0.0);
    assert(step.endpoint_field.sources.active_rest_weight == 1.0e-12);
    assert(step.rest_weight_after == step.rest_weight_before);
    assert(std::all_of(step.endpoint_field.sources.t_rr.begin(),
        step.endpoint_field.sources.t_rr.end(), [](double x) { return std::isfinite(x); }));
    assert(std::all_of(step.endpoint_field.metric.a.begin(),
        step.endpoint_field.metric.a.end(), [](double x) { return std::isfinite(x); }));
    auto unavailable = particles;
    unavailable[0].state = ef::MarkerState::excised;
    bool rejected = false;
    try { (void)ef::advance_self_consistent(unavailable, 1.0e-7, grid, evolution); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::cout << "interior horizon crossing: retained r=" << particles[0].r
              << " P=" << particles[0].p_r << " active_weight="
              << step.endpoint_field.sources.active_rest_weight << '\n';
}

void test_regular_observer_density_identity() {
    const double radius = 0.4, pr = 0.7, eta = 0.13;
    const double chi = 1.0 + 2.0 * eta / (radius * radius);
    const double energy = ef::infall_observer_energy(radius, pr, eta);
    for (double a : {1.0, 0.001, 0.0, -1.0, -9.0, -100.0}) {
        const double b = 0.6, uv = 1.0 / b, ur = 0.5 * (a - 1.0);
        assert(close(-a * b * b * uv * uv + 2.0 * b * uv * ur, -1.0));
        const double pv = -0.5 * b * (a * pr + chi / pr);
        assert(close(-pv * uv - pr * ur, energy));
        const double tvv = pv * pv / pr, tvr = pv, trr = pr;
        const double contraction = tvv * uv * uv + 2.0 * tvr * uv * ur + trr * ur * ur;
        assert(close(contraction, energy * energy / pr, 1e-11));
    }
}

void test_isolated_outer_boundary_rejects_gravitating_escape() {
    auto p = parameters(64);
    p.inner_boundary = "origin_parity";
    const auto grid = ef::make_field_grid(p, p.Nr);
    const auto outgoing = marker(1.0e-3, 7.99, 2.0, 0.0, 700);
    std::vector<ef::Marker> particles{outgoing};
    const auto initial = ef::deposit_and_solve_field(particles, grid);
    ef::EvolutionOptions options;
    options.threads = 1;
    options.enable_excision = false;
    options.require_vacuum_outer_boundary = true;
    bool rejected = false;
    try { (void)ef::advance_self_consistent(particles, 0.2, grid, options); }
    catch (const ef::VacuumOuterBoundaryViolation&) { rejected = true; }
    assert(rejected);
    assert(particles.front().r == outgoing.r && particles.front().p_r == outgoing.p_r);
    assert(particles.front().state == ef::MarkerState::active);
    const auto unchanged = ef::deposit_and_solve_field(particles, grid);
    assert(unchanged.metric.adm_mass == initial.metric.adm_mass);
    assert(unchanged.metric.m.front() == 0.0);
    // Every positive weight matters, including the smallest normal binary64.
    auto escaped = outgoing;
    escaped.state = ef::MarkerState::escaped;
    escaped.rest_weight = std::numeric_limits<double>::min();
    rejected = false;
    try { ef::check_vacuum_outer_boundary({escaped}); }
    catch (const ef::VacuumOuterBoundaryViolation&) { rejected = true; }
    assert(rejected);
    escaped.rest_weight = 0.0;
    ef::check_vacuum_outer_boundary({escaped});
    // Fixed-field/library outflow bookkeeping remains explicitly available.
    options.require_vacuum_outer_boundary = false;
    const auto unguarded = ef::advance_self_consistent(particles, 0.2, grid, options);
    assert(unguarded.endpoint_field.sources.escaped_rest_weight == outgoing.rest_weight);
    assert(unguarded.endpoint_field.metric.m.front() == initial.metric.adm_mass);
    std::cout << "isolated EF outer boundary: positive-weight escape rejected transactionally\n";
}

}  // namespace

int main() {
    test_origin_parity_grid_contract();
    test_minkowski_deposition_and_geodesics();
    test_exactly_radial_tiny_radius_limits();
    test_metric_sampler_entry_validation();
    test_conversion_round_trip();
    test_dynamic_eikonal_tensor_pullback_identities();
    test_characteristics_are_canonical_hamilton_equations();
    test_current_free_schwarzschild_to_ef_constraint_map();
    test_vacuum_schwarzschild_and_horizon_finite_rhs();
    test_log_momentum_pusher_preserves_future_branch();
    test_constraint_identities_and_parallel_determinism();
    test_self_consistent_horizon_excision_and_mass_retention();
    test_deep_excision_keeps_horizon_crossing_particles();
    test_interior_horizon_crossing_and_retained_sources();
    test_regular_observer_density_identity();
    test_isolated_outer_boundary_rejects_gravitating_escape();
    std::cout << "weighted EF Einstein--Vlasov PIC tests passed\n";
    return 0;
}
