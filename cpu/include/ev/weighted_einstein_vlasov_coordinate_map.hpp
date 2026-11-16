#pragma once

// Map a time dependent polar-areal Einstein--Vlasov solution onto an ingoing
// null hypersurface. The donor history supplies the metric and matter at each
// spacetime event; the eikonal equation supplies the surface through that history.
// The inputs are dense Schwarzschild snapshots and accepted marker endpoints.
// The outputs are EF metric/stress samples and the same particles at their null
// crossings, with their rest weights and angular momenta preserved. Geometrized
// units use G=c=1. The polar time and advanced null time label different slices.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ev/weighted_einstein_vlasov_pic_schwarzschild.hpp"
#include "ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp"

namespace ev::coordinate_map {

/// One dense donor snapshot.  `reverse_time` is s=-tau for the East-style
/// time-reversed outgoing replay of incoming Schwarzschild data.
struct SchwarzschildSnapshot {
    double reverse_time = 0.0;
    WeightedEinsteinVlasovPICField field;
};

struct SchwarzschildHistory {
    std::vector<SchwarzschildSnapshot> snapshot;
};

struct MetricHistorySample {
    double lambda = 0.0;
    double mu = 0.0;
    double m = 0.0;
    double rho = 0.0;
    double p = 0.0;
    double pt = 0.0;
    double j_reverse = 0.0;
};

/// One point of v=const.  A=v_tau>0, F=tau_v=1/A.  The stored time is the
/// reverse-donor time s=-tau; hence ds_sigma/dr=exp(lambda-mu).
struct EikonalPoint {
    double r = 0.0;
    double reverse_time = 0.0;
    double log_A = 0.0;
};

struct EikonalSurface {
    std::vector<EikonalPoint> point;  // strictly increasing radius
    double outer_radius = 0.0;
    double outer_reverse_time = 0.0;
    double outer_advanced_time = 0.0;
    // Per-cell step-doubling estimates for the accepted (2n-substep) RK4
    // state.  Each is |Y^(2n)-Y^(n)|/[15 max(1,|Y_in|,|Y^(2n)|)].
    double maximum_reverse_time_richardson_error = 0.0;
    double maximum_log_A_richardson_error = 0.0;
    // Retained as the serialized compatibility field: the maximum of the
    // two normalized Richardson errors above, not a midpoint slope defect.
    double maximum_ode_residual = 0.0;
};

struct SurfaceSample {
    double reverse_time = 0.0;
    double log_A = 0.0;
};

struct MarkerCrossing {
    eddington_finkelstein::Marker ef_marker;
    double reverse_time = 0.0;
    double schwarzschild_time = 0.0;
    double w_reverse = 0.0;
    double w_schwarzschild = 0.0;
    double lambda = 0.0;
    double mu = 0.0;
    double surface_reverse_time = 0.0;
    double residual = 0.0;
    std::size_t replay_step = 0;
    double step_fraction = 0.0;
};

struct MappedSlicePoint {
    double r = 0.0;
    double reverse_time = 0.0;
    double schwarzschild_time = 0.0;
    double A = 1.0;
    double F = 1.0;
    double a = 1.0;
    double b = 1.0;
    double m = 0.0;
    double rho = 0.0;
    double p = 0.0;
    double pt = 0.0;
    double j = 0.0;
    double t_vv = 0.0;
    double t_vr = 0.0;
    double t_rr = 0.0;
    double t_theta_theta = 0.0;
    double s = 0.0;
};

// The coordinate construction uses the exterior polar chart. Ordered donor times
// and positive metric gap provide the time interpolation domain for every ray.
inline void validate_history(
    const SchwarzschildHistory& history,
    const WeightedPICFieldGrid& grid) {
    if (history.snapshot.size() < 2 || grid.radius.empty()) {
        throw std::runtime_error(
            "Dynamic eikonal mapping requires at least two dense donor "
            "snapshots and a non-empty radial grid.");
    }
    double previous = -std::numeric_limits<double>::infinity();
    for (const auto& snapshot : history.snapshot) {
        const auto& field = snapshot.field;
        const std::size_t cells = grid.radius.size();
        if (!(snapshot.reverse_time > previous)
            || !std::isfinite(snapshot.reverse_time)
            || field.metric.lambda.size() != cells
            || field.metric.mu.size() != cells
            || field.metric.m.size() != cells
            || field.moments.rho.size() != cells
            || field.moments.p.size() != cells
            || field.moments.pt.size() != cells
            || field.moments.j.size() != cells
            || !(field.metric.min_gap > 0.0)) {
            throw std::runtime_error(
                "Dynamic eikonal donor history is incomplete, unordered, "
                "or has left the Schwarzschild exterior chart.");
        }
        previous = snapshot.reverse_time;
    }
}

inline double interpolate_centered_array(
    double radius,
    const std::vector<double>& grid_radius,
    const std::vector<double>& value) {
    if (grid_radius.empty() || value.size() != grid_radius.size()) {
        throw std::runtime_error(
            "Dynamic map radial interpolation received inconsistent data.");
    }
    if (radius <= grid_radius.front()) {
        return value.front();
    }
    if (radius >= grid_radius.back()) {
        return value.back();
    }
    const auto upper = std::upper_bound(
        grid_radius.begin(), grid_radius.end(), radius);
    const std::size_t right = static_cast<std::size_t>(
        upper - grid_radius.begin());
    const std::size_t left = right - 1;
    const double fraction = (radius - grid_radius[left])
        / (grid_radius[right] - grid_radius[left]);
    return (1.0 - fraction) * value[left] + fraction * value[right];
}

inline MetricHistorySample sample_snapshot(
    const SchwarzschildSnapshot& snapshot,
    double radius,
    const WeightedPICFieldGrid& grid) {
    const MetricSample metric =
        weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
            radius, grid, snapshot.field.metric);
    MetricHistorySample result;
    result.lambda = metric.lambda;
    result.mu = metric.mu;
    result.m = interpolate_centered_array(
        radius, grid.radius, snapshot.field.metric.m);
    result.rho = interpolate_centered_array(
        radius, grid.radius, snapshot.field.moments.rho);
    result.p = interpolate_centered_array(
        radius, grid.radius, snapshot.field.moments.p);
    result.pt = interpolate_centered_array(
        radius, grid.radius, snapshot.field.moments.pt);
    result.j_reverse = interpolate_centered_array(
        radius, grid.radius, snapshot.field.moments.j);
    return result;
}

// Permit endpoint roundoff within the recorded donor interval. Extrapolating
// beyond that interval would introduce a metric absent from the supplied solution.
inline std::pair<std::size_t, double> history_time_bracket(
    const SchwarzschildHistory& history,
    double reverse_time) {
    const double first = history.snapshot.front().reverse_time;
    const double last = history.snapshot.back().reverse_time;
    const double scale = std::max({1.0, std::abs(first), std::abs(last)});
    const double tolerance =
        128.0 * std::numeric_limits<double>::epsilon() * scale;
    if (reverse_time < first - tolerance || reverse_time > last + tolerance) {
        throw std::runtime_error(
            "Eikonal surface requested a time outside the dense donor "
            "history; increase --ef-null-donor-duration.");
    }
    const double clamped = std::clamp(reverse_time, first, last);
    const auto upper = std::upper_bound(
        history.snapshot.begin(),
        history.snapshot.end(),
        clamped,
        [](double value, const SchwarzschildSnapshot& snapshot) {
            return value < snapshot.reverse_time;
        });
    if (upper == history.snapshot.begin()) {
        return {0, 0.0};
    }
    if (upper == history.snapshot.end()) {
        return {history.snapshot.size() - 2, 1.0};
    }
    const std::size_t left = static_cast<std::size_t>(
        upper - history.snapshot.begin() - 1);
    const double t0 = history.snapshot[left].reverse_time;
    const double t1 = history.snapshot[left + 1].reverse_time;
    return {left, (clamped - t0) / (t1 - t0)};
}

inline MetricHistorySample sample_history(
    const SchwarzschildHistory& history,
    double reverse_time,
    double radius,
    const WeightedPICFieldGrid& grid) {
    const auto bracket = history_time_bracket(history, reverse_time);
    const MetricHistorySample left = sample_snapshot(
        history.snapshot[bracket.first], radius, grid);
    const MetricHistorySample right = sample_snapshot(
        history.snapshot[bracket.first + 1], radius, grid);
    auto blend = [&](double x0, double x1) {
        return (1.0 - bracket.second) * x0 + bracket.second * x1;
    };
    MetricHistorySample result;
    result.lambda = blend(left.lambda, right.lambda);
    result.mu = blend(left.mu, right.mu);
    result.m = blend(left.m, right.m);
    result.rho = blend(left.rho, right.rho);
    result.p = blend(left.p, right.p);
    result.pt = blend(left.pt, right.pt);
    result.j_reverse = blend(left.j_reverse, right.j_reverse);
    return result;
}

/// Returns h=exp(lambda-mu) and partial_s h from the same piecewise-linear
/// spacetime interpolant.  Since tau=-s, partial_tau h=-partial_s h.
// Differentiate the same time interpolant used for h. This keeps the surface
// position and its time Jacobian consistent at donor interval boundaries.
inline std::pair<double, double> sample_h_and_time_derivative(
    const SchwarzschildHistory& history,
    double reverse_time,
    double radius,
    const WeightedPICFieldGrid& grid) {
    const auto bracket = history_time_bracket(history, reverse_time);
    const auto left = sample_snapshot(
        history.snapshot[bracket.first], radius, grid);
    const auto right = sample_snapshot(
        history.snapshot[bracket.first + 1], radius, grid);
    const double h0 = std::exp(left.lambda - left.mu);
    const double h1 = std::exp(right.lambda - right.mu);
    const double dt = history.snapshot[bracket.first + 1].reverse_time
        - history.snapshot[bracket.first].reverse_time;
    return {
        (1.0 - bracket.second) * h0 + bracket.second * h1,
        (h1 - h0) / dt,
    };
}

/// Solves the ingoing eikonal characteristic in the reversed donor:
///
///   ds_sigma/dr = h,       d ln(A)/dr = -partial_s h,
///   s_sigma(r_out)=T,      A(r_out)=1.
///
/// This is exactly tau_sigma'=-h and d ln(A)/dr=partial_tau h in the
/// physical Schwarzschild time tau=-s.  The advanced-time label at the outer
/// boundary is v=tau=-T; the EF evolution may subsequently shift it to zero.
/// Every radial cell is integrated from the same incoming state with n and
/// 2n RK4 substeps.  The 2n state is retained, and its normalized local error
/// estimate is |Y^(2n)-Y^(n)|/[15 max(1,|Y_in|,|Y^(2n)|)] for both
/// Y=s_sigma and Y=ln(A).  The factor 15=2^4-1 is the RK4 step-doubling
/// Richardson divisor.
inline EikonalSurface solve_ingress_eikonal_surface(
    const SchwarzschildHistory& history,
    const WeightedPICFieldGrid& grid,
    double outer_reverse_time,
    int radial_substeps = 4,
    double map_outer_radius = -1.0) {
    validate_history(history, grid);
    const double requested_outer_radius = map_outer_radius > 0.0
        ? map_outer_radius : grid.mesh.edge.back();
    const double radial_tolerance = 128.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(grid.mesh.edge.back()));
    const auto boundary = std::lower_bound(
        grid.mesh.edge.begin(), grid.mesh.edge.end(),
        requested_outer_radius - radial_tolerance);
    const bool boundary_is_mesh_edge = boundary != grid.mesh.edge.end()
        && std::abs(*boundary - requested_outer_radius) <= radial_tolerance;
    const std::size_t map_cells = boundary_is_mesh_edge
        ? static_cast<std::size_t>(boundary - grid.mesh.edge.begin()) : 0;
    if (!(outer_reverse_time >= history.snapshot.front().reverse_time)
        || !(outer_reverse_time <= history.snapshot.back().reverse_time)
        || radial_substeps < 1
        || radial_substeps > std::numeric_limits<int>::max() / 2
        || grid.mesh.edge.size() != grid.radius.size() + 1
        || !boundary_is_mesh_edge || map_cells == 0
        || map_cells > grid.radius.size()) {
        throw std::runtime_error(
            "Invalid controls or non-edge worldtube radius for the dynamic "
            "eikonal surface solve.");
    }
    std::vector<double> radius;
    radius.reserve(map_cells + 2);
    radius.push_back(grid.mesh.edge.front());
    radius.insert(
        radius.end(), grid.radius.begin(), grid.radius.begin() + map_cells);
    radius.push_back(*boundary);

    std::vector<EikonalPoint> descending;
    descending.reserve(radius.size());
    double r = radius.back();
    double s = outer_reverse_time;
    double log_A = 0.0;
    descending.push_back({r, s, log_A});
    double max_reverse_time_error = 0.0;
    double max_log_A_error = 0.0;

    struct RKState {
        double reverse_time;
        double log_A;
    };
    // Integrate inward in radius. The first equation locates the null surface; the
    // second transports its time Jacobian, stored logarithmically to keep A positive.
    auto integrate_segment = [&](double start_radius,
                                 double target_radius,
                                 const RKState& initial,
                                 int substeps) {
        const double dr = (target_radius - start_radius)
            / static_cast<double>(substeps);
        RKState state = initial;
        auto rhs = [&](double radius_value, double reverse_time_value) {
            const auto h = sample_h_and_time_derivative(
                history, reverse_time_value, radius_value, grid);
            return std::pair<double, double>{h.first, -h.second};
        };
        for (int substep = 0; substep < substeps; ++substep) {
            const double substep_radius = start_radius
                + static_cast<double>(substep) * dr;
            const auto k1 = rhs(substep_radius, state.reverse_time);
            const auto k2 = rhs(
                substep_radius + 0.5 * dr,
                state.reverse_time + 0.5 * dr * k1.first);
            const auto k3 = rhs(
                substep_radius + 0.5 * dr,
                state.reverse_time + 0.5 * dr * k2.first);
            const auto k4 = rhs(
                substep_radius + dr,
                state.reverse_time + dr * k3.first);
            state.reverse_time += dr * (k1.first + 2.0 * k2.first
                + 2.0 * k3.first + k4.first) / 6.0;
            state.log_A += dr * (k1.second + 2.0 * k2.second
                + 2.0 * k3.second + k4.second) / 6.0;
        }
        return state;
    };

    for (std::size_t reverse_index = radius.size() - 1;
         reverse_index > 0;
         --reverse_index) {
        const double target = radius[reverse_index - 1];
        const RKState initial{s, log_A};
        // Compare n and 2n integrations of this radial segment. The fourth-order
        // Richardson divisor estimates local integration error; the refined state is kept.
        const RKState coarse = integrate_segment(
            r, target, initial, radial_substeps);
        const RKState refined = integrate_segment(
            r, target, initial, 2 * radial_substeps);
        if (!std::isfinite(coarse.reverse_time)
            || !std::isfinite(coarse.log_A)
            || !std::isfinite(refined.reverse_time)
            || !std::isfinite(refined.log_A)) {
            throw std::runtime_error(
                "Dynamic eikonal integration produced a non-finite "
                "step-doubling state.");
        }

        constexpr double richardson_divisor = 15.0;
        const double reverse_time_scale = std::max({
            1.0,
            std::abs(initial.reverse_time),
            std::abs(refined.reverse_time),
        });
        const double log_A_scale = std::max({
            1.0,
            std::abs(initial.log_A),
            std::abs(refined.log_A),
        });
        const double reverse_time_error = std::abs(
            refined.reverse_time - coarse.reverse_time)
            / (richardson_divisor * reverse_time_scale);
        const double log_A_error = std::abs(
            refined.log_A - coarse.log_A)
            / (richardson_divisor * log_A_scale);
        max_reverse_time_error = std::max(
            max_reverse_time_error, reverse_time_error);
        max_log_A_error = std::max(max_log_A_error, log_A_error);

        // The refined 2n solution, rather than either the coarse solution or
        // a Richardson extrapolation, is the accepted characteristic state.
        s = refined.reverse_time;
        log_A = refined.log_A;
        r = target;
        descending.push_back({r, s, log_A});
    }
    std::reverse(descending.begin(), descending.end());
    EikonalSurface result;
    result.point = std::move(descending);
    result.outer_radius = *boundary;
    result.outer_reverse_time = outer_reverse_time;
    result.outer_advanced_time = -outer_reverse_time;
    result.maximum_reverse_time_richardson_error =
        max_reverse_time_error;
    result.maximum_log_A_richardson_error = max_log_A_error;
    result.maximum_ode_residual = std::max(
        max_reverse_time_error, max_log_A_error);
    const double coverage_scale = std::max(
        1.0, std::abs(outer_reverse_time));
    const double coverage_tolerance = 256.0
        * std::numeric_limits<double>::epsilon() * coverage_scale;
    if (result.point.front().reverse_time
            < history.snapshot.front().reverse_time - coverage_tolerance
        || !std::isfinite(result.point.front().log_A)) {
        throw std::runtime_error(
            "The ingoing eikonal slice extends before the donor history; "
            "increase --ef-null-donor-duration or reduce the map radius.");
    }
    for (const auto& point : result.point) {
        if (!std::isfinite(point.reverse_time)
            || !std::isfinite(point.log_A)
            || !(std::exp(point.log_A) > 0.0)) {
            throw std::runtime_error(
                "Dynamic eikonal integration produced a non-finite or "
                "non-oriented coordinate map.");
        }
    }
    return result;
}

inline SurfaceSample sample_surface(
    const EikonalSurface& surface,
    double radius) {
    if (surface.point.size() < 2) {
        throw std::runtime_error("Cannot sample an empty eikonal surface.");
    }
    if (radius < surface.point.front().r
        || radius > surface.point.back().r) {
        throw std::runtime_error(
            "Marker lies outside the solved eikonal surface domain.");
    }
    const auto upper = std::upper_bound(
        surface.point.begin(),
        surface.point.end(),
        radius,
        [](double value, const EikonalPoint& point) {
            return value < point.r;
        });
    if (upper == surface.point.begin()) {
        return {upper->reverse_time, upper->log_A};
    }
    if (upper == surface.point.end()) {
        const auto& point = surface.point.back();
        return {point.reverse_time, point.log_A};
    }
    const auto& right = *upper;
    const auto& left = *(upper - 1);
    const double fraction = (radius - left.r) / (right.r - left.r);
    return {
        (1.0 - fraction) * left.reverse_time
            + fraction * right.reverse_time,
        (1.0 - fraction) * left.log_A + fraction * right.log_A,
    };
}

inline double crossing_function(
    double reverse_time,
    double radius,
    const EikonalSurface& surface) {
    return reverse_time - sample_surface(surface, radius).reverse_time;
}

/// Locates one same-event crossing using the dense endpoints of an accepted
/// deterministic replay step.  Linear worldline interpolation is consistent
/// with the second-order self-consistent time update; the nonlinear root in
/// theta is solved by bisection against the independently solved surface.
// A particle keeps its identity, rest mass and eta across a coordinate change.
// Bracket its crossing along one accepted donor step before changing its momentum.
inline MarkerCrossing interpolate_marker_crossing(
    const WeightedPICMarker& before,
    const WeightedPICMarker& after,
    double reverse_time_before,
    double reverse_time_after,
    const EikonalSurface& surface,
    const SchwarzschildHistory& history,
    const WeightedPICFieldGrid& grid,
    std::size_t replay_step) {
    if (before.id != after.id || before.source_cell != after.source_cell
        || before.mass != after.mass || before.eta != after.eta
        || before.state != WeightedPICMarkerState::active
        || after.state != WeightedPICMarkerState::active
        || !(reverse_time_after > reverse_time_before)) {
        throw std::runtime_error(
            "Invalid marker endpoints for a same-event null crossing.");
    }
    auto state_at = [&](double theta) {
        WeightedPICMarker marker = before;
        marker.r = before.r + theta * (after.r - before.r);
        marker.w = before.w + theta * (after.w - before.w);
        return marker;
    };
    auto g = [&](double theta) {
        const auto marker = state_at(theta);
        const double time = reverse_time_before
            + theta * (reverse_time_after - reverse_time_before);
        return crossing_function(time, marker.r, surface);
    };
    double left = 0.0;
    double right = 1.0;
    double g_left = g(left);
    double g_right = g(right);
    const double scale = std::max({
        1.0,
        std::abs(reverse_time_before),
        std::abs(reverse_time_after),
    });
    const double tolerance =
        256.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(g_left) <= tolerance) {
        right = left;
    } else if (std::abs(g_right) <= tolerance) {
        left = right;
    } else if (!(g_left < 0.0 && g_right > 0.0)) {
        throw std::runtime_error(
            "Marker endpoints do not bracket a unique future-directed "
            "crossing of the ingoing null surface.");
    } else {
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double middle = 0.5 * (left + right);
            const double g_middle = g(middle);
            if (g_middle <= 0.0) {
                left = middle;
            } else {
                right = middle;
            }
        }
    }
    const double theta = 0.5 * (left + right);
    const WeightedPICMarker reverse_marker = state_at(theta);
    const double reverse_time = reverse_time_before
        + theta * (reverse_time_after - reverse_time_before);
    const SurfaceSample surface_sample = sample_surface(
        surface, reverse_marker.r);
    const MetricHistorySample metric = sample_history(
        history, reverse_time, reverse_marker.r, grid);
    // Reverse the donor momentum back to the physical polar solution. The future
    // mass shell then gives positive EF canonical p_r=exp(lambda)*(epsilon+w).
    const double w_schwarzschild = -reverse_marker.w;
    const double chi = reverse_marker.eta == 0.0 ? 1.0
        : 1.0 + 2.0 * reverse_marker.eta
            / (reverse_marker.r * reverse_marker.r);
    const double epsilon = std::sqrt(
        chi + w_schwarzschild * w_schwarzschild);
    // Use epsilon+w for outgoing motion and its rationalized expression for
    // ingoing motion. Each branch avoids subtracting nearly equal energies.
    // For unit rest mass, chi>0 ensures positive canonical momentum.
    const double p_r = w_schwarzschild < 0.0
        ? std::exp(metric.lambda) * chi / (epsilon - w_schwarzschild)
        : std::exp(metric.lambda) * (epsilon + w_schwarzschild);
    if (!(p_r > 0.0) || !std::isfinite(p_r)) {
        throw std::runtime_error(
            "SC-to-EF crossing map violated future orientation p_r>0.");
    }
    MarkerCrossing result;
    result.ef_marker.rest_weight = reverse_marker.mass;
    result.ef_marker.r = reverse_marker.r;
    result.ef_marker.p_r = p_r;
    result.ef_marker.eta = reverse_marker.eta;
    result.ef_marker.id = reverse_marker.id;
    result.ef_marker.source_id = reverse_marker.source_cell;
    result.ef_marker.state =
        eddington_finkelstein::MarkerState::active;
    result.reverse_time = reverse_time;
    result.schwarzschild_time = -reverse_time;
    result.w_reverse = reverse_marker.w;
    result.w_schwarzschild = w_schwarzschild;
    result.lambda = metric.lambda;
    result.mu = metric.mu;
    result.surface_reverse_time = surface_sample.reverse_time;
    result.residual = reverse_time - surface_sample.reverse_time;
    result.replay_step = replay_step;
    result.step_fraction = theta;
    return result;
}

// Transform both covariant tensor indices at the same event. In polar variables
// pt is the sum of the two tangential stresses, so T_theta_theta=r^2*pt/2.
inline std::vector<MappedSlicePoint> map_schwarzschild_slice_tensors(
    const SchwarzschildHistory& history,
    const EikonalSurface& surface,
    const WeightedPICFieldGrid& grid) {
    std::vector<MappedSlicePoint> result;
    result.reserve(grid.radius.size());
    for (double radius : grid.radius) {
        if (!(radius < surface.outer_radius)) {
            break;
        }
        const SurfaceSample surface_value = sample_surface(surface, radius);
        const MetricHistorySample source = sample_history(
            history, surface_value.reverse_time, radius, grid);
        const double A = std::exp(surface_value.log_A);
        const double F = 1.0 / A;
        const double a = std::exp(-2.0 * source.lambda);
        const double b = F * std::exp(source.lambda + source.mu);
        const double h = std::exp(source.lambda - source.mu);
        // Time reversal changes the polar-areal current sign.  Every other
        // deposited moment and both metric potentials are even.
        const double j = -source.j_reverse;
        const double t_tau_tau = std::exp(2.0 * source.mu) * source.rho;
        const double t_tau_r =
            -std::exp(source.lambda + source.mu) * j;
        const double t_rr_polar = std::exp(2.0 * source.lambda) * source.p;
        MappedSlicePoint point;
        point.r = radius;
        point.reverse_time = surface_value.reverse_time;
        point.schwarzschild_time = -surface_value.reverse_time;
        point.A = A;
        point.F = F;
        point.a = a;
        point.b = b;
        point.m = 0.5 * radius * (1.0 - a);
        point.rho = source.rho;
        point.p = source.p;
        point.pt = source.pt;
        point.j = j;
        point.t_vv = F * F * t_tau_tau;
        point.t_vr = F * (t_tau_r - h * t_tau_tau);
        point.t_rr = t_rr_polar - 2.0 * h * t_tau_r
            + h * h * t_tau_tau;
        point.t_theta_theta = 0.5 * radius * radius * source.pt;
        // This combination is the EF source entering the radial mass constraint. It is
        // formed from the transformed tensor and the transformed metric on the same slice.
        point.s = -2.0 * point.t_vr / b - a * point.t_rr;
        if (!(point.a > 0.0) || !(point.b > 0.0)
            || !std::isfinite(point.s)) {
            throw std::runtime_error(
                "Mapped metric/stress tensor is non-finite or outside the "
                "Schwarzschild chart.");
        }
        result.push_back(point);
    }
    return result;
}

}  // namespace ev::coordinate_map
