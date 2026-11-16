#pragma once

// Extend an ingoing-null coordinate map into a known Schwarzschild vacuum.
// The caller establishes that all matter lies inside a boundary worldtube R_B.
// For r>=R_B>2M, the exterior has constant mass M, EF b=1 and a=1-2M/r.
// Inputs are the mapped interior, the exterior radius grid and the boundary
// reverse time. Outputs append analytic vacuum map points and an audit of
// their fields, zero source and inverse coordinate relation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ev/weighted_einstein_vlasov_coordinate_map.hpp"

namespace ev::coordinate_map {

struct SchwarzschildVacuumExtensionAudit {
    double maximum_field_residual = 0.0;
    double maximum_source = 0.0;
    double maximum_inverse_map_residual = 0.0;
    double minimum_a = 1.0;
};

// The analytic exterior used here is outside the Schwarzschild radius.
// A domain reaching r=2M requires a different admitted map/continuation.
inline double schwarzschild_vacuum_a(double radius, double mass) {
    if (!(radius > 0.0) || !(mass >= 0.0)
        || !std::isfinite(radius) || !std::isfinite(mass)
        || !(radius > 2.0 * mass)) {
        throw std::runtime_error(
            "Schwarzschild vacuum extension requires r>2M>=0.");
    }
    return 1.0 - 2.0 * mass / radius;
}

// r_*(r)-r_*(R_B), written as a ratio so no dimensionful logarithm appears.
inline double schwarzschild_tortoise_difference(
    double radius,
    double boundary_radius,
    double mass) {
    if (!(radius >= boundary_radius) || !(boundary_radius > 2.0 * mass)
        || !(mass >= 0.0) || !std::isfinite(radius)
        || !std::isfinite(boundary_radius) || !std::isfinite(mass)) {
        throw std::runtime_error(
            "Invalid Schwarzschild tortoise-extension domain.");
    }
    if (mass == 0.0) {
        return radius - boundary_radius;
    }
    // At fixed advanced time, t changes by minus the tortoise-radius
    // difference. The logarithm uses a ratio of two lengths, so its
    // argument is dimensionless.
    return (radius - boundary_radius)
        + 2.0 * mass * std::log(
            (radius - 2.0 * mass)
            / (boundary_radius - 2.0 * mass));
}

inline double schwarzschild_time_radial_derivative_at_fixed_v(
    double radius,
    double mass) {
    return -1.0 / schwarzschild_vacuum_a(radius, mass);
}

inline MappedSlicePoint make_schwarzschild_vacuum_extension_point(
    double radius,
    double boundary_radius,
    double boundary_reverse_time,
    double mass) {
    if (!(boundary_reverse_time >= 0.0)
        || !std::isfinite(boundary_reverse_time)) {
        throw std::runtime_error(
            "Vacuum null extension requires finite nonnegative boundary s.");
    }
    const double a = schwarzschild_vacuum_a(radius, mass);
    const double reverse_time = boundary_reverse_time
        + schwarzschild_tortoise_difference(
            radius, boundary_radius, mass);
    MappedSlicePoint point;
    point.r = radius;
    point.reverse_time = reverse_time;
    point.schwarzschild_time = -reverse_time;
    // A and F are coordinate-map derivatives in MappedSlicePoint, not the
    // Schwarzschild radial metric coefficient used by a collision tetrad.
    // Their unit values implement the standard vacuum time normalization.
    point.A = 1.0;
    point.F = 1.0;
    point.a = a;
    point.b = 1.0;
    point.m = mass;
    // rho,p,pt,j,T_ab and the scalar source s remain exactly zero.
    return point;
}

inline std::vector<MappedSlicePoint>
extend_mapped_slice_with_schwarzschild_vacuum(
    const std::vector<MappedSlicePoint>& mapped_interior,
    const std::vector<double>& evolution_radius,
    double boundary_radius,
    double boundary_reverse_time,
    double mass) {
    if (mapped_interior.empty()
        || mapped_interior.size() >= evolution_radius.size()) {
        throw std::runtime_error(
            "Vacuum extension requires a nonempty proper radial prefix.");
    }
    // The mapped data must be the actual prefix of the requested evolution
    // grid below R_B. Appending vacuum cannot replace an unaccounted
    // matter-containing interval between different radial grids.
    const double tolerance = 128.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(boundary_radius));
    for (std::size_t i = 0; i < mapped_interior.size(); ++i) {
        if (std::abs(mapped_interior[i].r - evolution_radius[i])
                > tolerance
            || !(mapped_interior[i].r < boundary_radius)) {
            throw std::runtime_error(
                "Mapped interior is not the evolution-grid prefix below R_B.");
        }
    }
    if (!(evolution_radius[mapped_interior.size()] > boundary_radius)) {
        throw std::runtime_error(
            "The first analytic-vacuum cell must lie outside R_B.");
    }
    std::vector<MappedSlicePoint> result = mapped_interior;
    result.reserve(evolution_radius.size());
    for (std::size_t i = result.size(); i < evolution_radius.size(); ++i) {
        result.push_back(make_schwarzschild_vacuum_extension_point(
            evolution_radius[i], boundary_radius,
            boundary_reverse_time, mass));
    }
    return result;
}

inline SchwarzschildVacuumExtensionAudit
audit_schwarzschild_vacuum_extension(
    const std::vector<MappedSlicePoint>& mapped_slice,
    std::size_t extension_start_index,
    double boundary_radius,
    double boundary_reverse_time,
    double mass) {
    if (extension_start_index == 0
        || extension_start_index >= mapped_slice.size()) {
        throw std::runtime_error(
            "Vacuum extension audit received an invalid start index.");
    }
    SchwarzschildVacuumExtensionAudit audit;
    for (std::size_t i = extension_start_index;
         i < mapped_slice.size(); ++i) {
        const auto& point = mapped_slice[i];
        const double exact_a = schwarzschild_vacuum_a(point.r, mass);
        const double exact_reverse_time = boundary_reverse_time
            + schwarzschild_tortoise_difference(
                point.r, boundary_radius, mass);
        audit.maximum_field_residual = std::max({
            audit.maximum_field_residual,
            std::abs(point.A - 1.0),
            std::abs(point.F - 1.0),
            std::abs(point.A * point.F - 1.0),
            std::abs(point.a - exact_a),
            std::abs(point.b - 1.0),
            std::abs(point.m - mass),
        });
        // Report every stored vacuum source component separately from metric
        // residuals. Physical absence of exterior matter is the caller
        // worldtube condition stated above, not a result inferred from zero
        // default values in newly constructed map points.
        audit.maximum_source = std::max({
            audit.maximum_source,
            std::abs(point.rho), std::abs(point.p),
            std::abs(point.pt), std::abs(point.j),
            std::abs(point.t_vv), std::abs(point.t_vr),
            std::abs(point.t_rr), std::abs(point.t_theta_theta),
            std::abs(point.s),
        });
        audit.maximum_inverse_map_residual = std::max({
            audit.maximum_inverse_map_residual,
            std::abs(point.reverse_time - exact_reverse_time),
            std::abs(point.schwarzschild_time + exact_reverse_time),
        });
        audit.minimum_a = std::min(audit.minimum_a, point.a);
    }
    return audit;
}

}  // namespace ev::coordinate_map
