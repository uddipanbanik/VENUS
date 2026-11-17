#include <cassert>
#include <cmath>
#include <iostream>

#include "ev/sha256.hpp"
#include "ev/weighted_einstein_vlasov_coordinate_map.hpp"
#include "ev/weighted_einstein_vlasov_null_vacuum_extension.hpp"

namespace {

ev::coordinate_map::SchwarzschildSnapshot minkowski_snapshot(
    double reverse_time,
    std::size_t cells) {
    ev::coordinate_map::SchwarzschildSnapshot result;
    result.reverse_time = reverse_time;
    result.field.metric.lambda.assign(cells, 0.0);
    result.field.metric.mu.assign(cells, 0.0);
    result.field.metric.m.assign(cells, 0.0);
    result.field.metric.lambda_r.assign(cells, 0.0);
    result.field.metric.mu_r.assign(cells, 0.0);
    result.field.metric.lambda_t.assign(cells, 0.0);
    result.field.metric.gap.assign(cells, 1.0);
    result.field.metric.min_gap = 1.0;
    result.field.moments.rho.assign(cells, 0.0);
    result.field.moments.p.assign(cells, 0.0);
    result.field.moments.pt.assign(cells, 0.0);
    result.field.moments.j.assign(cells, 0.0);
    return result;
}

double exponential_profile_integral_to_outer(
    double radius,
    const ev::WeightedPICFieldGrid& grid,
    double exponent) {
    const double first_center = grid.radius.front();
    const double last_center = grid.radius.back();
    const double outer_edge = grid.mesh.edge.back();
    double integral = 0.0;
    double left = radius;
    if (left < first_center) {
        integral += (first_center - left)
            * std::exp(exponent * first_center);
        left = first_center;
    }
    if (left < last_center) {
        integral += (std::exp(exponent * last_center)
            - std::exp(exponent * left)) / exponent;
        left = last_center;
    }
    if (left < outer_edge) {
        integral += (outer_edge - left)
            * std::exp(exponent * last_center);
    }
    return integral;
}

}  // namespace

int main() {
    assert(ev::integrity::sha256_string("abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    ev::Params parameters;
    parameters.physics_model = "einstein_vlasov";
    parameters.vp_universe = "static";
    parameters.bp_collision_operator = "none";
    parameters.kinetic_grid = "cell_average_eta";
    parameters.kinetic_eta_grid = "uniform";
    parameters.inner_boundary = "origin_parity";
    parameters.outer_boundary = "outflow_vacuum";
    parameters.radial_grid = "linear";
    parameters.r_min = 0.01;
    parameters.r_max = 6.0;
    parameters.Nr = 24;
    parameters.metric_floor = 1.0e-12;
    parameters.gap_stop = 0.03;
    const auto grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            parameters, parameters.Nr);

    ev::coordinate_map::SchwarzschildHistory history;
    history.snapshot.push_back(minkowski_snapshot(0.0, grid.radius.size()));
    history.snapshot.push_back(minkowski_snapshot(6.0, grid.radius.size()));
    const auto surface =
        ev::coordinate_map::solve_ingress_eikonal_surface(
            history, grid, 6.0, 4);
    assert(std::abs(surface.point.front().reverse_time) < 2.0e-13);
    assert(std::abs(surface.point.back().reverse_time - 6.0) < 2.0e-13);
    for (const auto& point : surface.point) {
        assert(std::abs(point.reverse_time - point.r) < 2.0e-12);
        assert(std::abs(point.log_A) < 2.0e-13);
    }
    assert(surface.maximum_reverse_time_richardson_error < 2.0e-14);
    assert(surface.maximum_log_A_richardson_error < 2.0e-14);
    assert(surface.maximum_ode_residual < 2.0e-14);

    // The map worldtube can be an exact mesh edge strictly inside the field
    // history/evolution grid.  Only the solved prefix is returned for tensor
    // mapping; the exterior is supplied by the analytic-vacuum continuation.
    const auto interior_surface =
        ev::coordinate_map::solve_ingress_eikonal_surface(
            history, grid, 4.0, 4, 4.0);
    assert(interior_surface.outer_radius == 4.0);
    assert(std::abs(interior_surface.point.front().reverse_time)
        < 2.0e-13);
    const auto mapped_interior =
        ev::coordinate_map::map_schwarzschild_slice_tensors(
            history, interior_surface, grid);
    assert(mapped_interior.size() == 16);
    assert(mapped_interior.back().r < interior_surface.outer_radius);
    bool rejected_non_edge_worldtube = false;
    try {
        (void)ev::coordinate_map::solve_ingress_eikonal_surface(
            history, grid, 4.0, 4, 4.01);
    } catch (const std::runtime_error&) {
        rejected_non_edge_worldtube = true;
    }
    assert(rejected_non_edge_worldtube);

    // Exact genuinely dynamical test: h(s)=1+k*s, so
    // s+1/k=(T+1/k)exp(k(r-r_out)) and ln A=k(r_out-r).
    constexpr double k = 0.01;
    constexpr double dynamic_T = 7.0;
    ev::coordinate_map::SchwarzschildHistory dynamic_history;
    dynamic_history.snapshot.push_back(
        minkowski_snapshot(0.0, grid.radius.size()));
    auto dynamic_end =
        minkowski_snapshot(dynamic_T, grid.radius.size());
    dynamic_end.field.metric.lambda.assign(
        grid.radius.size(), std::log(1.0 + k * dynamic_T));
    dynamic_history.snapshot.push_back(std::move(dynamic_end));
    const auto dynamic_surface =
        ev::coordinate_map::solve_ingress_eikonal_surface(
            dynamic_history, grid, dynamic_T, 16);
    for (const auto& point : dynamic_surface.point) {
        const double exact_s = (dynamic_T + 1.0 / k)
            * std::exp(k * (point.r - grid.mesh.edge.back())) - 1.0 / k;
        const double exact_log_A =
            k * (grid.mesh.edge.back() - point.r);
        assert(std::abs(point.reverse_time - exact_s) < 2.0e-10);
        assert(std::abs(point.log_A - exact_log_A) < 2.0e-12);
    }

    // RK4 step-doubling regression with both equations nontrivial.  The
    // centered spatial interpolant gives
    //   h(s,r)=exp(alpha*r_clamped)*(1+k*s),
    // so partial_s h=k*exp(alpha*r_clamped).  This has a closed-form
    // characteristic and a nonconstant ln(A) right-hand side.
    ev::Params richardson_parameters = parameters;
    richardson_parameters.Nr = 4;
    const auto richardson_grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            richardson_parameters, richardson_parameters.Nr);
    constexpr double richardson_T = 60.0;
    constexpr double richardson_k = 0.005;
    constexpr double profile_exponent = 0.5;
    ev::coordinate_map::SchwarzschildHistory richardson_history;
    for (double reverse_time : {0.0, richardson_T}) {
        auto snapshot = minkowski_snapshot(
            reverse_time, richardson_grid.radius.size());
        for (std::size_t cell = 0;
             cell < richardson_grid.radius.size(); ++cell) {
            snapshot.field.metric.lambda[cell] =
                profile_exponent * richardson_grid.radius[cell]
                + std::log(1.0 + richardson_k * reverse_time);
        }
        richardson_history.snapshot.push_back(std::move(snapshot));
    }
    const auto richardson_n1 =
        ev::coordinate_map::solve_ingress_eikonal_surface(
            richardson_history, richardson_grid, richardson_T, 1);
    const auto richardson_n2 =
        ev::coordinate_map::solve_ingress_eikonal_surface(
            richardson_history, richardson_grid, richardson_T, 2);
    const auto richardson_n4 =
        ev::coordinate_map::solve_ingress_eikonal_surface(
            richardson_history, richardson_grid, richardson_T, 4);

    auto maximum_state_error = [&](const auto& candidate) {
        double maximum = 0.0;
        for (const auto& point : candidate.point) {
            const double integral = exponential_profile_integral_to_outer(
                point.r, richardson_grid, profile_exponent);
            const double exact_log_A = richardson_k * integral;
            const double exact_s = ((1.0 + richardson_k * richardson_T)
                * std::exp(-richardson_k * integral) - 1.0)
                / richardson_k;
            maximum = std::max({
                maximum,
                std::abs(point.reverse_time - exact_s),
                std::abs(point.log_A - exact_log_A),
            });
        }
        return maximum;
    };
    const double state_error_n1 = maximum_state_error(richardson_n1);
    const double state_error_n2 = maximum_state_error(richardson_n2);
    const double state_error_n4 = maximum_state_error(richardson_n4);

    for (const auto* candidate : {
             &richardson_n1, &richardson_n2, &richardson_n4}) {
        assert(candidate->maximum_reverse_time_richardson_error > 0.0);
        assert(candidate->maximum_log_A_richardson_error > 0.0);
        assert(candidate->maximum_ode_residual == std::max(
            candidate->maximum_reverse_time_richardson_error,
            candidate->maximum_log_A_richardson_error));
    }
    // Halving the RK4 substep should reduce both local Richardson estimates
    // and the accepted-surface error by approximately 2^4.  Bounds of 8--24
    // detect loss of fourth-order behavior while remaining roundoff robust.
    auto assert_fourth_order = [](double coarse, double fine) {
        const double ratio = coarse / fine;
        assert(ratio > 8.0);
        assert(ratio < 24.0);
    };
    assert_fourth_order(
        richardson_n1.maximum_reverse_time_richardson_error,
        richardson_n2.maximum_reverse_time_richardson_error);
    assert_fourth_order(
        richardson_n2.maximum_reverse_time_richardson_error,
        richardson_n4.maximum_reverse_time_richardson_error);
    assert_fourth_order(
        richardson_n1.maximum_log_A_richardson_error,
        richardson_n2.maximum_log_A_richardson_error);
    assert_fourth_order(
        richardson_n2.maximum_log_A_richardson_error,
        richardson_n4.maximum_log_A_richardson_error);
    assert_fourth_order(state_error_n1, state_error_n2);
    assert_fourth_order(state_error_n2, state_error_n4);
    // This is the unchanged strict certifier threshold used by production.
    assert(richardson_n4.maximum_ode_residual < 2.0e-5);

    ev::WeightedPICMarker before;
    before.mass = 0.25;
    before.r = 0.8;
    before.w = 0.4;
    before.eta = 0.0;
    before.id = 17;
    before.source_cell = 3;
    ev::WeightedPICMarker after = before;
    after.r = 1.2;
    const auto crossing = ev::coordinate_map::interpolate_marker_crossing(
        before, after, 0.5, 1.5, surface, history, grid, 1);
    assert(crossing.ef_marker.id == before.id);
    assert(crossing.ef_marker.source_id == before.source_cell);
    assert(crossing.ef_marker.rest_weight == before.mass);
    assert(std::abs(crossing.reverse_time - 1.0) < 2.0e-14);
    assert(std::abs(crossing.ef_marker.r - 1.0) < 2.0e-14);
    assert(std::abs(crossing.w_reverse - 0.4) < 2.0e-14);
    assert(std::abs(crossing.w_schwarzschild + 0.4) < 2.0e-14);
    assert(std::abs(crossing.ef_marker.p_r
        - (std::sqrt(1.0 + 0.4 * 0.4) - 0.4)) < 2.0e-14);
    assert(std::abs(crossing.residual) < 2.0e-14);
    {
        // Reverse-donor inward motion maps to outgoing physical motion.
        // At w=1e8, sqrt(1+w^2)-w rounds to zero, while the future EF
        // momentum epsilon+w remains finite and must survive extraction.
        auto outgoing_before = before;
        outgoing_before.r = 1.1;
        outgoing_before.w = -1.0e8;
        auto outgoing_after = outgoing_before;
        outgoing_after.r = 0.9;
        const auto outgoing = ev::coordinate_map::interpolate_marker_crossing(
            outgoing_before, outgoing_after, 0.9, 1.1,
            surface, history, grid, 1);
        assert(std::abs(outgoing.reverse_time - 1.0) < 2.0e-14);
        assert(std::abs(outgoing.ef_marker.r - 1.0) < 2.0e-14);
        assert(outgoing.w_schwarzschild == 1.0e8);
        assert(outgoing.ef_marker.p_r == 2.0e8);
        assert(outgoing.ef_marker.rest_weight == outgoing_before.mass);
        assert(outgoing.ef_marker.eta == outgoing_before.eta);
        assert(outgoing.ef_marker.id == outgoing_before.id);
        assert(std::abs(outgoing.residual) < 2.0e-14);
    }

    const auto mapped =
        ev::coordinate_map::map_schwarzschild_slice_tensors(
            history, surface, grid);
    assert(mapped.size() == grid.radius.size());
    for (const auto& point : mapped) {
        assert(std::abs(point.a - 1.0) < 2.0e-14);
        assert(std::abs(point.b - 1.0) < 2.0e-14);
        assert(std::abs(point.m) < 2.0e-14);
        assert(std::abs(point.t_rr) < 2.0e-14);
    }

    // A solved map may terminate at an authenticated vacuum worldtube and
    // continue analytically to a larger EF evolution boundary.  In normalized
    // static vacuum A=F=b=1.  The tortoise derivative, not F, supplies the
    // nontrivial inverse-map coefficient d tau/dr|_v=-1/a.
    constexpr double vacuum_mass = 0.8;
    constexpr double map_boundary = 4.0;
    constexpr double map_boundary_reverse_time = 7.0;
    const std::vector<double> extended_radius{1.0, 3.0, 5.0, 8.0};
    std::vector<ev::coordinate_map::MappedSlicePoint> mapped_prefix(2);
    mapped_prefix[0].r = extended_radius[0];
    mapped_prefix[1].r = extended_radius[1];
    const auto extended =
        ev::coordinate_map::extend_mapped_slice_with_schwarzschild_vacuum(
            mapped_prefix, extended_radius, map_boundary,
            map_boundary_reverse_time, vacuum_mass);
    assert(extended.size() == extended_radius.size());
    for (std::size_t i = mapped_prefix.size(); i < extended.size(); ++i) {
        const double exact_a = 1.0 - 2.0 * vacuum_mass / extended[i].r;
        assert(extended[i].A == 1.0);
        assert(extended[i].F == 1.0);
        assert(extended[i].b == 1.0);
        assert(std::abs(extended[i].a - exact_a) < 2.0e-15);
        assert(extended[i].m == vacuum_mass);
        assert(extended[i].rho == 0.0);
        assert(extended[i].p == 0.0);
        assert(extended[i].pt == 0.0);
        assert(extended[i].j == 0.0);
        assert(extended[i].t_vv == 0.0);
        assert(extended[i].t_vr == 0.0);
        assert(extended[i].t_rr == 0.0);
        assert(extended[i].t_theta_theta == 0.0);
        assert(extended[i].s == 0.0);
        assert(std::abs(
            ev::coordinate_map::schwarzschild_time_radial_derivative_at_fixed_v(
                extended[i].r, vacuum_mass)
            + 1.0 / exact_a) < 2.0e-15);
    }
    const double exact_delta = (extended_radius.back() - map_boundary)
        + 2.0 * vacuum_mass * std::log(
            (extended_radius.back() - 2.0 * vacuum_mass)
            / (map_boundary - 2.0 * vacuum_mass));
    assert(std::abs(
        extended.back().reverse_time
        - (map_boundary_reverse_time + exact_delta)) < 2.0e-15);
    const auto vacuum_audit =
        ev::coordinate_map::audit_schwarzschild_vacuum_extension(
            extended, mapped_prefix.size(), map_boundary,
            map_boundary_reverse_time, vacuum_mass);
    assert(vacuum_audit.maximum_field_residual == 0.0);
    assert(vacuum_audit.maximum_source == 0.0);
    assert(vacuum_audit.maximum_inverse_map_residual == 0.0);
    assert(vacuum_audit.minimum_a > 0.0);
    std::cout << "RK4 eikonal Richardson residuals (n=1,2,4): "
              << richardson_n1.maximum_ode_residual << ", "
              << richardson_n2.maximum_ode_residual << ", "
              << richardson_n4.maximum_ode_residual << "\n";
    std::cout << "Dynamic eikonal coordinate-map tests passed\n";
    return 0;
}
