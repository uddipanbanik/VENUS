#pragma once

// Transport spherical Vlasov-Poisson cell averages by moving source-cell mass
// through a frozen characteristic map.  Inputs are f(r,w,eta), exact cell faces,
// a gravitational field, a time interval and optional polynomial/inner-carrier
// state.  Outputs are destination cell averages, optional P2 modes and complete
// visible, hidden, escaped and geometric-fallback mass diagnostics.
//
// At fixed eta, the canonical (r,w) map preserves area.  Reconstruct f inside
// each source cell, trace a 3x3 lattice and approximate an admissible image by
// eight affine triangles.  Integrating the reconstructed source over each
// destination overlap transfers mass locally.  Refine distorted/folded images
// until their geometry passes or the bounded Gauss fallback is explicitly used.
// Persistent delayed-return packets carry mass while its orbit is inside r_min.
// The final budget joins deposited mass, carrier mass and escaped mass; its
// acceptance test exposes errors in tracing, geometry and boundary partition.

#include "ev/delayed_inner_orbit_carrier.hpp"
#include "ev/kinetic_invariant_ledger.hpp"
#include "ev/vlasov_poisson.hpp"

#include <array>

namespace ev {

// A traced point also records whether it remains inside the unresolved inner
// domain and the speed/time needed for its later outward return.
struct SLDGPoint {
    double r = 0.0;
    double w = 0.0;
    bool hidden = false;
    bool touched_inner = false;
    double boundary_speed = 0.0;
    double remaining_hidden_time = 0.0;
};

using SLDGPolygon = std::vector<SLDGPoint>;

// Physical mass diagnostics include the 8*pi^2*deta measure.  Geometry errors
// and work counters distinguish accepted affine transport from bounded fallback.
struct SLDGStepDiagnostics {
    double source_mass = 0.0;
    double deposited_mass = 0.0;
    double escaped_mass = 0.0;
    double carrier_mass_before = 0.0;
    double carrier_mass_after = 0.0;
    double carrier_captured_mass = 0.0;
    double carrier_released_mass = 0.0;
    double carrier_release_deposited_mass = 0.0;
    double carrier_mass_residual = 0.0;
    double quadrature_fallback_mass = 0.0;
    double maximum_area_defect = 0.0;
    double maximum_map_defect = 0.0;
    double maximum_rejected_area_defect = 0.0;
    double maximum_rejected_map_defect = 0.0;
    double largest_fallback_mass = 0.0;
    double largest_fallback_r = 0.0;
    double largest_fallback_w = 0.0;
    double largest_fallback_L = 0.0;
    int largest_fallback_refinement = 0;
    long long largest_fallback_work = 0;
    long long transported_subcells = 0;
    long long traced_rectangles = 0;
    long long rejected_rectangles = 0;
    long long quadrature_fallback_subcells = 0;
    long long maximum_source_work = 0;
    long long characteristic_trace_requests = 0;
    long long characteristic_traces = 0;
    long long reused_characteristic_points = 0;
    long long carrier_packets_before = 0;
    long long carrier_packets_after = 0;
    long long carrier_packets_captured = 0;
    long long carrier_packets_released = 0;
    long long carrier_compressions = 0;
    long long carrier_compression_packets_before = 0;
    long long carrier_compression_packets_after = 0;
    double carrier_compression_mass_delta = 0.0;
    double carrier_compression_momentum_delta = 0.0;
    double carrier_compression_energy_delta = 0.0;
    double carrier_compression_return_time_moment_delta = 0.0;
    double midpoint_predictor_mass_before = 0.0;
    double midpoint_predictor_mass_after = 0.0;
    double midpoint_predictor_escaped_mass = 0.0;
    double midpoint_predictor_hidden_mass_before = 0.0;
    double midpoint_predictor_hidden_mass_after = 0.0;
    double midpoint_predictor_hidden_mass_change = 0.0;
    long long midpoint_predictor_limited_faces = 0;
    long long hamiltonian_reconstructed_cells = 0;
    long long hamiltonian_limited_cells = 0;
    long long hamiltonian_projected_limited_cells = 0;
    long long hamiltonian_rank_deficient_cells = 0;
    long long hamiltonian_phase_degenerate_cells = 0;
    long long hamiltonian_eta_limited_cells = 0;
    double hamiltonian_maximum_phase_gradient_fraction = 0.0;
    double hamiltonian_reference_mass_fraction = 0.0;
    double hamiltonian_residual_l1_fraction = 1.0;
    long long hamiltonian_reference_active_planes = 0;
    double hamiltonian_minimum_positive_reference_scale = 1.0;
    long long hamiltonian_positive_reference_scaled_planes = 0;
    double hamiltonian_positivity_redistributed_mass = 0.0;
    double hamiltonian_positivity_redistributed_fraction = 0.0;
};

struct SLDGMappedRectangle {
    SLDGPolygon polygon;
};

struct SLDGStepResult {
    std::vector<double> f;
    std::vector<std::array<double, 5>> modal_coefficients;
    bool has_modal_coefficients = false;
    VlasovPoissonField midpoint_field;
    SLDGStepDiagnostics diagnostics;
};

/// Chooses the next static-potential transport cap from represented fallback mass.
///
/// The controller is active only when vp_sldg_initial_transport_dt is
/// positive.  It aims below the hard fallback tolerance so map complexity can
/// grow between steps without immediately reaching the rejection boundary.
inline double sldg_next_transport_dt_cap(
    double accepted_dt,
    const SLDGStepDiagnostics& diagnostics,
    const Params& p) {
    if (!(p.vp_sldg_initial_transport_dt > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    if (!(accepted_dt > 0.0) || !std::isfinite(accepted_dt)) {
        throw std::runtime_error(
            "SLDG cadence controller received a non-positive timestep.");
    }
    const double fallback_fraction =
        (diagnostics.source_mass > 0.0)
        ? diagnostics.quadrature_fallback_mass / diagnostics.source_mass
        : 0.0;
    double factor = p.vp_sldg_timestep_growth;
    if (fallback_fraction > 0.0) {
        if (p.vp_sldg_fallback_mass_tolerance == 0.0) {
            factor = p.vp_sldg_timestep_shrink;
        } else {
            const double target =
                p.vp_sldg_target_fallback_fraction
                * p.vp_sldg_fallback_mass_tolerance;
            factor = std::sqrt(target / fallback_fraction);
            factor = std::clamp(
                factor,
                p.vp_sldg_timestep_shrink,
                p.vp_sldg_timestep_growth);
        }
    }
    return std::min(p.dt_max, accepted_dt * factor);
}

// Retain a common signed slope only when neighboring estimates agree, limiting
// new extrema while preserving the source cell average.
inline double sldg_minmod(double a, double b, double c) {
    if ((a > 0.0 && b > 0.0 && c > 0.0)
        || (a < 0.0 && b < 0.0 && c < 0.0)) {
        const double sign = (a > 0.0) ? 1.0 : -1.0;
        return sign * std::min({std::abs(a), std::abs(b), std::abs(c)});
    }
    return 0.0;
}

// Translate vertices to a nearby origin before the shoelace sum, retaining
// small patch areas when absolute r/w coordinates are much larger.
inline long double sldg_polygon_signed_area_extended(
    const SLDGPolygon& polygon) {
    if (polygon.size() < 3) {
        return 0.0L;
    }
    const long double origin_r = polygon.front().r;
    const long double origin_w = polygon.front().w;
    long double twice_area = 0.0;
    for (std::size_t q = 0; q < polygon.size(); ++q) {
        const SLDGPoint& a = polygon[q];
        const SLDGPoint& b = polygon[(q + 1) % polygon.size()];
        const long double ar = static_cast<long double>(a.r) - origin_r;
        const long double aw = static_cast<long double>(a.w) - origin_w;
        const long double br = static_cast<long double>(b.r) - origin_r;
        const long double bw = static_cast<long double>(b.w) - origin_w;
        twice_area += ar * bw - br * aw;
    }
    return 0.5L * twice_area;
}

inline double sldg_polygon_signed_area(const SLDGPolygon& polygon) {
    return static_cast<double>(sldg_polygon_signed_area_extended(polygon));
}

inline double sldg_polygon_area(const SLDGPolygon& polygon) {
    return std::abs(sldg_polygon_signed_area(polygon));
}

inline double sldg_oriented_cross(
    const SLDGPoint& a,
    const SLDGPoint& b,
    const SLDGPoint& c) {
    return (b.r - a.r) * (c.w - b.w) - (b.w - a.w) * (c.r - b.r);
}

// A usable patch has finite vertices, positive orientation and convex turns.
// Folded images require further characteristic refinement.
inline bool sldg_is_finite_convex_ccw_quadrilateral(
    const SLDGPolygon& polygon,
    double area_tolerance) {
    if (polygon.size() != 4) {
        return false;
    }
    for (const SLDGPoint& point : polygon) {
        if (!std::isfinite(point.r) || !std::isfinite(point.w)) {
            return false;
        }
    }
    const double signed_area = sldg_polygon_signed_area(polygon);
    if (!(signed_area > area_tolerance)) {
        return false;
    }
    for (std::size_t q = 0; q < polygon.size(); ++q) {
        const double turn = sldg_oriented_cross(
            polygon[q],
            polygon[(q + 1) % polygon.size()],
            polygon[(q + 2) % polygon.size()]);
        if (turn < -area_tolerance) {
            return false;
        }
    }
    return true;
}

// Clip each directed polygon edge against one destination half-plane, inserting
// an intersection only when that edge crosses the face.
template <class Inside, class Intersect>
inline SLDGPolygon sldg_clip_half_plane(
    const SLDGPolygon& input,
    Inside&& inside,
    Intersect&& intersect) {
    SLDGPolygon output;
    if (input.empty()) {
        return output;
    }
    SLDGPoint previous = input.back();
    bool previous_inside = inside(previous);
    for (const SLDGPoint& current : input) {
        const bool current_inside = inside(current);
        if (current_inside != previous_inside) {
            output.push_back(intersect(previous, current));
        }
        if (current_inside) {
            output.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    return output;
}

// Four half-plane clips give the portion inside one (r,w) destination cell.
inline SLDGPolygon sldg_clip_rectangle(
    SLDGPolygon polygon,
    double r0,
    double r1,
    double w0,
    double w1) {
    auto vertical_intersection = [](const SLDGPoint& a, const SLDGPoint& b, double edge) {
        const double denominator = b.r - a.r;
        const double t = (std::abs(denominator) > 0.0)
            ? std::clamp((edge - a.r) / denominator, 0.0, 1.0)
            : 0.0;
        return SLDGPoint{edge, a.w + t * (b.w - a.w)};
    };
    auto horizontal_intersection = [](const SLDGPoint& a, const SLDGPoint& b, double edge) {
        const double denominator = b.w - a.w;
        const double t = (std::abs(denominator) > 0.0)
            ? std::clamp((edge - a.w) / denominator, 0.0, 1.0)
            : 0.0;
        return SLDGPoint{a.r + t * (b.r - a.r), edge};
    };
    polygon = sldg_clip_half_plane(
        polygon,
        [&](const SLDGPoint& point) { return point.r >= r0; },
        [&](const SLDGPoint& a, const SLDGPoint& b) {
            return vertical_intersection(a, b, r0);
        });
    polygon = sldg_clip_half_plane(
        polygon,
        [&](const SLDGPoint& point) { return point.r <= r1; },
        [&](const SLDGPoint& a, const SLDGPoint& b) {
            return vertical_intersection(a, b, r1);
        });
    polygon = sldg_clip_half_plane(
        polygon,
        [&](const SLDGPoint& point) { return point.w >= w0; },
        [&](const SLDGPoint& a, const SLDGPoint& b) {
            return horizontal_intersection(a, b, w0);
        });
    return sldg_clip_half_plane(
        polygon,
        [&](const SLDGPoint& point) { return point.w <= w1; },
        [&](const SLDGPoint& a, const SLDGPoint& b) {
            return horizontal_intersection(a, b, w1);
        });
}

// Extended precision and bounded polygon storage support tiny refined overlap
// integrals without a heap allocation for every destination intersection.
struct SLDGExtendedPoint {
    long double r = 0.0L;
    long double w = 0.0L;
};

struct SLDGExtendedPolygon {
    // Canonical orbit patches trace curved cell edges with several vertices.
    // Four rectangle clips can add at most four more vertices.
    static constexpr std::size_t capacity = 80;

    std::array<SLDGExtendedPoint, capacity> point{};
    std::size_t count = 0;

    bool empty() const {
        return count == 0;
    }

    std::size_t size() const {
        return count;
    }

    void reserve(std::size_t requested) const {
        if (requested > capacity) {
            throw std::runtime_error(
                "SLDG fixed polygon capacity is too small for clipping.");
        }
    }

    void push_back(const SLDGExtendedPoint& value) {
        if (count >= capacity) {
            throw std::runtime_error(
                "SLDG clipping produced too many polygon vertices.");
        }
        point[count++] = value;
    }

    SLDGExtendedPoint& operator[](std::size_t q) {
        return point[q];
    }

    const SLDGExtendedPoint& operator[](std::size_t q) const {
        return point[q];
    }

    SLDGExtendedPoint& front() {
        return point[0];
    }

    const SLDGExtendedPoint& front() const {
        return point[0];
    }

    SLDGExtendedPoint& back() {
        return point[count - 1];
    }

    const SLDGExtendedPoint& back() const {
        return point[count - 1];
    }

    auto begin() {
        return point.begin();
    }

    auto end() {
        return point.begin() + static_cast<std::ptrdiff_t>(count);
    }

    auto begin() const {
        return point.begin();
    }

    auto end() const {
        return point.begin() + static_cast<std::ptrdiff_t>(count);
    }
};

// The translated shoelace formula evaluates the positive overlap area.
inline long double sldg_extended_polygon_area(
    const SLDGExtendedPolygon& polygon) {
    if (polygon.size() < 3) {
        return 0.0L;
    }
    const long double origin_r = polygon.front().r;
    const long double origin_w = polygon.front().w;
    long double twice_area = 0.0L;
    for (std::size_t q = 0; q < polygon.size(); ++q) {
        const SLDGExtendedPoint& a = polygon[q];
        const SLDGExtendedPoint& b =
            polygon[(q + 1) % polygon.size()];
        const long double ar = a.r - origin_r;
        const long double aw = a.w - origin_w;
        const long double br = b.r - origin_r;
        const long double bw = b.w - origin_w;
        twice_area += ar * bw - br * aw;
    }
    return std::abs(0.5L * twice_area);
}

// Area and first/second moments suffice for a quadratic integral.  Offset
// moments retain precision when the overlap is small relative to its location.
struct SLDGExtendedPolygonMoments {
    long double area = 0.0L;
    long double centroid_r = 0.0L;
    long double centroid_w = 0.0L;
    long double second_rr = 0.0L;
    long double second_rw = 0.0L;
    long double second_ww = 0.0L;
    long double reference_r = 0.0L;
    long double reference_w = 0.0L;
    long double first_offset_r = 0.0L;
    long double first_offset_w = 0.0L;
    long double second_offset_rr = 0.0L;
    long double second_offset_rw = 0.0L;
    long double second_offset_ww = 0.0L;
};

/// Mean-preserving quadratic reconstruction on one source cell.
///
/// The normalized coordinates x=(r-r_c)/dr and y=(w-w_c)/dw range over
/// [-1/2,1/2] on the parent cell.  The quadratic basis subtracts 1/12 from
/// x^2 and y^2, so every nonconstant mode has exactly zero parent-cell mean.
struct SLDGSourcePolynomial {
    double cell_average = 0.0;
    double center_r = 0.0;
    double center_w = 0.0;
    double scale_r = 1.0;
    double scale_w = 1.0;
    // x, y, x^2-1/12, xy, y^2-1/12
    std::array<double, 5> mode{};

    double value(double source_r, double source_w) const {
        const double x = (source_r - center_r) / scale_r;
        const double y = (source_w - center_w) / scale_w;
        return cell_average
            + mode[0] * x
            + mode[1] * y
            + mode[2] * (x * x - 1.0 / 12.0)
            + mode[3] * x * y
            + mode[4] * (y * y - 1.0 / 12.0);
    }

    double rectangle_integral(
        double r0,
        double r1,
        double w0,
        double w1) const {
        const double mean_x =
            (0.5 * (r0 + r1) - center_r) / scale_r;
        const double mean_y =
            (0.5 * (w0 + w1) - center_w) / scale_w;
        const double width_x = (r1 - r0) / scale_r;
        const double width_y = (w1 - w0) / scale_w;
        const double mean_x2 = mean_x * mean_x + width_x * width_x / 12.0;
        const double mean_y2 = mean_y * mean_y + width_y * width_y / 12.0;
        const double mean_value = cell_average
            + mode[0] * mean_x
            + mode[1] * mean_y
            + mode[2] * (mean_x2 - 1.0 / 12.0)
            + mode[3] * mean_x * mean_y
            + mode[4] * (mean_y2 - 1.0 / 12.0);
        return (r1 - r0) * (w1 - w0) * mean_value;
    }

    // A quadratic minimum can occur at a corner, edge stationary point or
    // interior stationary point.  Evaluate all candidates inside the cell.
    double minimum_on_parent_cell() const {
        const double constant =
            cell_average - (mode[2] + mode[4]) / 12.0;
        auto evaluate = [&](double x, double y) {
            return constant
                + mode[0] * x
                + mode[1] * y
                + mode[2] * x * x
                + mode[3] * x * y
                + mode[4] * y * y;
        };
        double minimum = std::numeric_limits<double>::infinity();
        auto include = [&](double x, double y) {
            if (x >= -0.5 && x <= 0.5 && y >= -0.5 && y <= 0.5) {
                minimum = std::min(minimum, evaluate(x, y));
            }
        };
        for (double x : {-0.5, 0.5}) {
            for (double y : {-0.5, 0.5}) {
                include(x, y);
            }
            if (mode[4] > 0.0) {
                include(
                    x,
                    std::clamp(
                        -(mode[1] + mode[3] * x) / (2.0 * mode[4]),
                        -0.5,
                        0.5));
            }
        }
        for (double y : {-0.5, 0.5}) {
            if (mode[2] > 0.0) {
                include(
                    std::clamp(
                        -(mode[0] + mode[3] * y) / (2.0 * mode[2]),
                        -0.5,
                        0.5),
                    y);
            }
        }
        const double determinant =
            4.0 * mode[2] * mode[4] - mode[3] * mode[3];
        const double determinant_scale = std::max({
            std::numeric_limits<double>::min(),
            std::abs(4.0 * mode[2] * mode[4]),
            mode[3] * mode[3],
        });
        if (std::abs(determinant)
            > 64.0 * std::numeric_limits<double>::epsilon()
                * determinant_scale) {
            include(
                (-2.0 * mode[4] * mode[0] + mode[3] * mode[1])
                    / determinant,
                (mode[3] * mode[0] - 2.0 * mode[2] * mode[1])
                    / determinant);
        }
        return minimum;
    }

    // Scale every mean-free mode by one factor toward the positive average.
    // Analytic extrema and a final bracket protect positivity at roundoff.
    double enforce_positivity() {
        if (!(cell_average > 0.0)) {
            mode.fill(0.0);
            return 0.0;
        }
        const double minimum = minimum_on_parent_cell();
        if (minimum >= 0.0) {
            return 1.0;
        }
        const std::array<double, 5> original_mode = mode;
        double theta = std::clamp(
            cell_average / (cell_average - minimum), 0.0, 1.0);
        if (theta > 0.0 && theta < 1.0) {
            theta = std::nextafter(theta, 0.0);
        }
        auto apply_scale = [&](double scale) {
            for (std::size_t q = 0; q < mode.size(); ++q) {
                mode[q] = scale * original_mode[q];
            }
        };
        apply_scale(theta);
        if (minimum_on_parent_cell() < 0.0) {
            theta *= 1.0
                - 128.0 * std::numeric_limits<double>::epsilon();
            apply_scale(theta);
        }
        if (minimum_on_parent_cell() < 0.0) {
            double lower = 0.0;
            double upper = theta;
            for (int iteration = 0; iteration < 64; ++iteration) {
                const double middle = 0.5 * (lower + upper);
                apply_scale(middle);
                if (minimum_on_parent_cell() >= 0.0) {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            theta = lower > 0.0 ? std::nextafter(lower, 0.0) : 0.0;
            apply_scale(theta);
        }
        if (!(minimum_on_parent_cell() >= 0.0)) {
            throw std::runtime_error(
                "P2 positivity scaling failed to produce a nonnegative polynomial.");
        }
        return theta;
    }
};

// Polygon boundary sums integrate 1,r,w,r^2,r*w,w^2.  Carry both local-offset
// and physical-coordinate moments for subsequent affine polynomial integration.
inline SLDGExtendedPolygonMoments sldg_extended_polygon_moments(
    const SLDGExtendedPolygon& polygon) {
    SLDGExtendedPolygonMoments moments;
    if (polygon.size() < 3) {
        return moments;
    }
    const long double origin_r = polygon.front().r;
    const long double origin_w = polygon.front().w;
    long double twice_signed_area = 0.0L;
    long double first_r = 0.0L;
    long double first_w = 0.0L;
    long double second_rr = 0.0L;
    long double second_rw = 0.0L;
    long double second_ww = 0.0L;
    for (std::size_t q = 0; q < polygon.size(); ++q) {
        const SLDGExtendedPoint& a = polygon[q];
        const SLDGExtendedPoint& b =
            polygon[(q + 1) % polygon.size()];
        const long double ar = a.r - origin_r;
        const long double aw = a.w - origin_w;
        const long double br = b.r - origin_r;
        const long double bw = b.w - origin_w;
        const long double cross = ar * bw - br * aw;
        twice_signed_area += cross;
        first_r += (ar + br) * cross;
        first_w += (aw + bw) * cross;
        second_rr += (ar * ar + ar * br + br * br) * cross;
        second_rw += (
            2.0L * ar * aw + ar * bw + br * aw + 2.0L * br * bw)
            * cross;
        second_ww += (aw * aw + aw * bw + bw * bw) * cross;
    }
    if (twice_signed_area == 0.0L) {
        return moments;
    }
    const long double orientation = twice_signed_area > 0.0L ? 1.0L : -1.0L;
    moments.area = std::abs(0.5L * twice_signed_area);
    const long double local_first_r = orientation * first_r / 6.0L;
    const long double local_first_w = orientation * first_w / 6.0L;
    const long double local_second_rr = orientation * second_rr / 12.0L;
    const long double local_second_rw = orientation * second_rw / 24.0L;
    const long double local_second_ww = orientation * second_ww / 12.0L;
    moments.centroid_r = origin_r + local_first_r / moments.area;
    moments.centroid_w = origin_w + local_first_w / moments.area;
    moments.reference_r = origin_r;
    moments.reference_w = origin_w;
    moments.first_offset_r = local_first_r;
    moments.first_offset_w = local_first_w;
    moments.second_offset_rr = local_second_rr;
    moments.second_offset_rw = local_second_rw;
    moments.second_offset_ww = local_second_ww;
    moments.second_rr =
        moments.area * origin_r * origin_r
        + 2.0L * origin_r * local_first_r
        + local_second_rr;
    moments.second_rw =
        moments.area * origin_r * origin_w
        + origin_r * local_first_w
        + origin_w * local_first_r
        + local_second_rw;
    moments.second_ww =
        moments.area * origin_w * origin_w
        + 2.0L * origin_w * local_first_w
        + local_second_ww;
    return moments;
}

// Apply the same conservative clipping in normalized extended coordinates.
template <class Inside, class Intersect>
inline SLDGExtendedPolygon sldg_clip_half_plane_extended(
    const SLDGExtendedPolygon& input,
    Inside&& inside,
    Intersect&& intersect) {
    SLDGExtendedPolygon output;
    if (input.empty()) {
        return output;
    }
    SLDGExtendedPoint previous = input.back();
    bool previous_inside = inside(previous);
    for (const SLDGExtendedPoint& current : input) {
        const bool current_inside = inside(current);
        if (current_inside != previous_inside) {
            output.push_back(intersect(previous, current));
        }
        if (current_inside) {
            output.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    return output;
}

inline SLDGExtendedPolygon sldg_clip_normalized_rectangle(
    SLDGExtendedPolygon clipped,
    long double nr0,
    long double nr1,
    long double nw0,
    long double nw1) {
    auto vertical_intersection = [](
        const SLDGExtendedPoint& a,
        const SLDGExtendedPoint& b,
        long double edge) {
        const long double denominator = b.r - a.r;
        const long double t = denominator != 0.0L
            ? std::clamp((edge - a.r) / denominator, 0.0L, 1.0L)
            : 0.0L;
        return SLDGExtendedPoint{
            edge,
            a.w + t * (b.w - a.w),
        };
    };
    auto horizontal_intersection = [](
        const SLDGExtendedPoint& a,
        const SLDGExtendedPoint& b,
        long double edge) {
        const long double denominator = b.w - a.w;
        const long double t = denominator != 0.0L
            ? std::clamp((edge - a.w) / denominator, 0.0L, 1.0L)
            : 0.0L;
        return SLDGExtendedPoint{
            a.r + t * (b.r - a.r),
            edge,
        };
    };
    clipped = sldg_clip_half_plane_extended(
        clipped,
        [&](const SLDGExtendedPoint& point) { return point.r >= nr0; },
        [&](const SLDGExtendedPoint& a, const SLDGExtendedPoint& b) {
            return vertical_intersection(a, b, nr0);
        });
    clipped = sldg_clip_half_plane_extended(
        clipped,
        [&](const SLDGExtendedPoint& point) { return point.r <= nr1; },
        [&](const SLDGExtendedPoint& a, const SLDGExtendedPoint& b) {
            return vertical_intersection(a, b, nr1);
        });
    clipped = sldg_clip_half_plane_extended(
        clipped,
        [&](const SLDGExtendedPoint& point) { return point.w >= nw0; },
        [&](const SLDGExtendedPoint& a, const SLDGExtendedPoint& b) {
            return horizontal_intersection(a, b, nw0);
        });
    clipped = sldg_clip_half_plane_extended(
        clipped,
        [&](const SLDGExtendedPoint& point) { return point.w <= nw1; },
        [&](const SLDGExtendedPoint& a, const SLDGExtendedPoint& b) {
            return horizontal_intersection(a, b, nw1);
        });
    return clipped;
}

inline long double sldg_clip_normalized_rectangle_area(
    SLDGExtendedPolygon clipped,
    long double nr0,
    long double nr1,
    long double nw0,
    long double nw1) {
    return sldg_extended_polygon_area(
        sldg_clip_normalized_rectangle(
            clipped, nr0, nr1, nw0, nw1));
}

/// A mapped polygon normalized once for all destination-cell intersections.
///
/// Deep refinement produces polygons that are tiny compared with their
/// absolute coordinates.  The context translates and scales such a polygon to
/// an O(1) box in long double.  Every destination-cell clip then reuses those
/// normalized vertices and the same coordinate transform.
struct SLDGExtendedClipContext {
    long double origin_r = 0.0L;
    long double origin_w = 0.0L;
    long double scale_r = 0.0L;
    long double scale_w = 0.0L;
    SLDGExtendedPolygon normalized;

    explicit SLDGExtendedClipContext(const SLDGPolygon& polygon) {
        if (polygon.size() < 3) {
            return;
        }
        double min_r = polygon.front().r;
        double max_r = min_r;
        double min_w = polygon.front().w;
        double max_w = min_w;
        for (const SLDGPoint& value : polygon) {
            min_r = std::min(min_r, value.r);
            max_r = std::max(max_r, value.r);
            min_w = std::min(min_w, value.w);
            max_w = std::max(max_w, value.w);
        }
        origin_r = static_cast<long double>(min_r);
        origin_w = static_cast<long double>(min_w);
        scale_r = static_cast<long double>(max_r) - origin_r;
        scale_w = static_cast<long double>(max_w) - origin_w;
        if (!(scale_r > 0.0L) || !(scale_w > 0.0L)) {
            scale_r = 0.0L;
            scale_w = 0.0L;
            return;
        }
        normalized.reserve(polygon.size());
        for (const SLDGPoint& value : polygon) {
            normalized.push_back({
                normalize_r(value.r),
                normalize_w(value.w),
            });
        }
    }

    bool valid() const {
        return normalized.size() >= 3 && scale_r > 0.0L && scale_w > 0.0L;
    }

    long double normalize_r(double value) const {
        return (static_cast<long double>(value) - origin_r) / scale_r;
    }

    long double normalize_w(double value) const {
        return (static_cast<long double>(value) - origin_w) / scale_w;
    }

    long double physical_area() const {
        return valid()
            ? sldg_extended_polygon_area(normalized) * scale_r * scale_w
            : 0.0L;
    }

    long double rectangle_area(
        double r0,
        double r1,
        double w0,
        double w1) const {
        if (!valid()) {
            return 0.0L;
        }
        return sldg_clip_normalized_rectangle_area(
            normalized,
            normalize_r(r0),
            normalize_r(r1),
            normalize_w(w0),
            normalize_w(w1))
            * scale_r * scale_w;
    }

    SLDGExtendedPolygon rectangle_normalized_polygon(
        double r0,
        double r1,
        double w0,
        double w1) const {
        if (!valid()) {
            return {};
        }
        return sldg_clip_normalized_rectangle(
            normalized,
            normalize_r(r0),
            normalize_r(r1),
            normalize_w(w0),
            normalize_w(w1));
    }

    SLDGExtendedPolygonMoments rectangle_moments(
        double r0,
        double r1,
        double w0,
        double w1) const {
        if (!valid()) {
            return {};
        }
        SLDGExtendedPolygonMoments moments =
            sldg_extended_polygon_moments(
                sldg_clip_normalized_rectangle(
                    normalized,
                    normalize_r(r0),
                    normalize_r(r1),
                    normalize_w(w0),
                    normalize_w(w1)));
        const long double normalized_area = moments.area;
        const long double normalized_first_r =
            normalized_area * moments.centroid_r;
        const long double normalized_first_w =
            normalized_area * moments.centroid_w;
        const long double normalized_second_rr = moments.second_rr;
        const long double normalized_second_rw = moments.second_rw;
        const long double normalized_second_ww = moments.second_ww;
        const long double jacobian = scale_r * scale_w;
        moments.area = jacobian * normalized_area;
        moments.second_rr = jacobian * (
            origin_r * origin_r * normalized_area
            + 2.0L * origin_r * scale_r * normalized_first_r
            + scale_r * scale_r * normalized_second_rr);
        moments.second_rw = jacobian * (
            origin_r * origin_w * normalized_area
            + origin_r * scale_w * normalized_first_w
            + origin_w * scale_r * normalized_first_r
            + scale_r * scale_w * normalized_second_rw);
        moments.second_ww = jacobian * (
            origin_w * origin_w * normalized_area
            + 2.0L * origin_w * scale_w * normalized_first_w
            + scale_w * scale_w * normalized_second_ww);
        moments.centroid_r = origin_r
            + moments.centroid_r * scale_r;
        moments.centroid_w = origin_w
            + moments.centroid_w * scale_w;
        moments.reference_r = origin_r;
        moments.reference_w = origin_w;
        moments.first_offset_r =
            jacobian * scale_r * normalized_first_r;
        moments.first_offset_w =
            jacobian * scale_w * normalized_first_w;
        moments.second_offset_rr =
            jacobian * scale_r * scale_r * normalized_second_rr;
        moments.second_offset_rw =
            jacobian * scale_r * scale_w * normalized_second_rw;
        moments.second_offset_ww =
            jacobian * scale_w * scale_w * normalized_second_ww;
        return moments;
    }
};

/// Returns the polygon/rectangle overlap in physical (r,w) area.
inline long double sldg_clip_rectangle_area_extended(
    const SLDGPolygon& polygon,
    double r0,
    double r1,
    double w0,
    double w1) {
    return SLDGExtendedClipContext(polygon).rectangle_area(
        r0, r1, w0, w1);
}

// Domain faces belong to their adjacent cell; an outside coordinate reports
// escape through the caller's mass ledger.
inline int sldg_cell_for_coordinate(const std::vector<double>& edges, double value) {
    if (value < edges.front() || value > edges.back()) {
        return -1;
    }
    if (value == edges.back()) {
        return static_cast<int>(edges.size()) - 2;
    }
    const auto it = std::upper_bound(edges.begin(), edges.end(), value);
    return std::clamp(
        static_cast<int>(std::distance(edges.begin(), it)) - 1,
        0,
        static_cast<int>(edges.size()) - 2);
}

// Construct limited r/w gradients.  Reflect physical radial velocity at a wall:
// peculiar w maps to -w-2*H*r_wall.  A common final slope factor makes all
// corners nonnegative while preserving the mean.
inline std::array<double, 2> sldg_limited_slopes(
    const std::vector<double>& f,
    int i,
    int j,
    int k,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p, double hubble = 0.0) {
    const double center_value = f[idx(i, j, k, p)];
    if (center_value < std::numeric_limits<double>::min()) {
        // A subnormal average remains represented, while its slope cannot be
        // evaluated reliably enough to guarantee positive corner values.
        return {0.0, 0.0};
    }
    auto reflected_value = [&](double wall) {
        const auto& velocity = geometry.radial_velocity.center;
        const double reflected_w = -velocity[j] - 2.0 * hubble * wall;
        if (reflected_w < velocity.front() || reflected_w > velocity.back()) {
            return 0.0;
        }
        const int jm = lower_cell(velocity, reflected_w);
        const double fraction = (reflected_w - velocity[jm])
            / (velocity[jm + 1] - velocity[jm]);
        return (1.0 - fraction) * f[idx(i, jm, k, p)]
            + fraction * f[idx(i, jm + 1, k, p)];
    };
    double slope_r = 0.0;
    if (i > 0 && i + 1 < p.Nr) {
        const double left = f[idx(i - 1, j, k, p)];
        const double right = f[idx(i + 1, j, k, p)];
        const double backward = (center_value - left)
            / (geometry.radial.center[static_cast<std::size_t>(i)]
               - geometry.radial.center[static_cast<std::size_t>(i - 1)]);
        const double forward = (right - center_value)
            / (geometry.radial.center[static_cast<std::size_t>(i + 1)]
               - geometry.radial.center[static_cast<std::size_t>(i)]);
        const double centered = (right - left)
            / (geometry.radial.center[static_cast<std::size_t>(i + 1)]
               - geometry.radial.center[static_cast<std::size_t>(i - 1)]);
        slope_r = sldg_minmod(2.0 * backward, centered, 2.0 * forward);
    } else if (i == 0 && p.inner_boundary == "reflecting") {
        const double ghost = reflected_value(geometry.radial.edge.front());
        const double ghost_center =
            2.0 * geometry.radial.edge.front() - geometry.radial.center.front();
        const double backward = (center_value - ghost)
            / (geometry.radial.center.front() - ghost_center);
        if (p.Nr > 1) {
            const double forward = (f[idx(1, j, k, p)] - center_value)
                / (geometry.radial.center[1] - geometry.radial.center[0]);
            const double centered = (f[idx(1, j, k, p)] - ghost)
                / (geometry.radial.center[1] - ghost_center);
            slope_r =
                sldg_minmod(2.0 * backward, centered, 2.0 * forward);
        }
    } else if (i == p.Nr - 1 && p.outer_boundary == "reflecting") {
        const double ghost = reflected_value(geometry.radial.edge.back());
        const double ghost_center =
            2.0 * geometry.radial.edge.back() - geometry.radial.center.back();
        const double forward = (ghost - center_value)
            / (ghost_center - geometry.radial.center.back());
        if (p.Nr > 1) {
            const double backward =
                (center_value - f[idx(p.Nr - 2, j, k, p)])
                / (geometry.radial.center[static_cast<std::size_t>(p.Nr - 1)]
                   - geometry.radial.center[static_cast<std::size_t>(p.Nr - 2)]);
            const double centered =
                (ghost - f[idx(p.Nr - 2, j, k, p)])
                / (ghost_center
                   - geometry.radial.center[static_cast<std::size_t>(p.Nr - 2)]);
            slope_r =
                sldg_minmod(2.0 * backward, centered, 2.0 * forward);
        }
    }

    double slope_w = 0.0;
    if (j > 0 && j + 1 < p.Nw) {
        const double lower = f[idx(i, j - 1, k, p)];
        const double upper = f[idx(i, j + 1, k, p)];
        const double backward = (center_value - lower)
            / (geometry.radial_velocity.center[static_cast<std::size_t>(j)]
               - geometry.radial_velocity.center[static_cast<std::size_t>(j - 1)]);
        const double forward = (upper - center_value)
            / (geometry.radial_velocity.center[static_cast<std::size_t>(j + 1)]
               - geometry.radial_velocity.center[static_cast<std::size_t>(j)]);
        const double centered = (upper - lower)
            / (geometry.radial_velocity.center[static_cast<std::size_t>(j + 1)]
               - geometry.radial_velocity.center[static_cast<std::size_t>(j - 1)]);
        slope_w = sldg_minmod(2.0 * backward, centered, 2.0 * forward);
    }

    const double half_dr = 0.5 * geometry.radial.width[static_cast<std::size_t>(i)];
    const double half_dw = 0.5 * geometry.radial_velocity.width[static_cast<std::size_t>(j)];
    double minimum_corner = center_value;
    for (double sr : {-1.0, 1.0}) {
        for (double sw : {-1.0, 1.0}) {
            minimum_corner = std::min(
                minimum_corner,
                center_value + sr * half_dr * slope_r + sw * half_dw * slope_w);
        }
    }
    if (minimum_corner < 0.0) {
        const double denominator = center_value - minimum_corner;
        const double theta = (denominator > 0.0)
            ? std::clamp(center_value / denominator, 0.0, 1.0)
            : 0.0;
        const double interior_theta = theta > 0.0
            ? std::nextafter(theta, 0.0)
            : 0.0;
        slope_r *= interior_theta;
        slope_w *= interior_theta;
    }
    return {slope_r, slope_w};
}

// Alternate resolved exterior orbit segments with analytic inner-cell round
// trips.  Return the remaining hidden time if the requested endpoint lies
// inside the inner domain, or continue outward after the physical return.
inline SLDGPoint sldg_trace_delayed_inner_arrival(
    double source_r,
    double source_w,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p) {
    if (!(dt >= 0.0) || !std::isfinite(dt)) {
        throw std::runtime_error(
            "Delayed-return carrier tracing requires a finite nonnegative interval.");
    }
    CharacteristicState state{source_r, source_w};
    double remaining = dt;
    bool touched_inner = false;
    const double time_tolerance = std::max(
        64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, dt),
        0.1 * p.vp_orbit_lc_absolute_tolerance);

    auto rhs = [&](const CharacteristicState& y) {
        if (!std::isfinite(y.r) || !std::isfinite(y.w)
            || y.r < p.r_min) {
            const double invalid =
                std::numeric_limits<double>::quiet_NaN();
            return CharacteristicDerivative{invalid, invalid};
        }
        const VPSample sample = sample_vp_field(
            std::max(y.r, p.r_min), x_r, r, field, p);
        return CharacteristicDerivative{
            vp_radial_velocity(y.r, y.w, field),
            vp_radial_acceleration(
                y.r, y.w, Lk, sample, r, field),
        };
    };
    auto apply_outer_boundary = [&](CharacteristicState& value) {
        const double tolerance =
            128.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, p.r_min);
        if (value.r < p.r_min - tolerance) {
            throw std::runtime_error(
                "Delayed-return orbit crossed r_min without a recorded carrier event.");
        }
        value.r = std::max(value.r, p.r_min);
        if (using_reflecting_outer(p)) {
            for (int reflection = 0; reflection < 8; ++reflection) {
                if (value.r <= vp_radial_outer_boundary(r, p)) {
                    break;
                }
                reflect_outer(
                    value.r,
                    value.w,
                    vp_radial_outer_boundary(r, p), field.background.H);
            }
        }
    };

    for (int segment = 0;
         segment < p.vp_inner_carrier_max_orbit_segments;
         ++segment) {
        if (remaining <= time_tolerance) {
            apply_outer_boundary(state);
            return {
                state.r, state.w, false, touched_inner, 0.0, 0.0};
        }
        const RegularizedOrbitResult outside =
            advance_radial_characteristic_reflecting_wall(
                state,
                remaining,
                p.r_min,
                rhs,
                p.vp_orbit_lc_relative_tolerance,
                p.vp_orbit_lc_absolute_tolerance,
                p.vp_orbit_lc_max_steps,
                true);
        if (!outside.converged) {
            throw std::runtime_error(
                "Delayed-return carrier could not resolve an exterior orbit segment.");
        }
        if (outside.wall_crossings == 0
            || !std::isfinite(outside.first_wall_crossing_time)) {
            CharacteristicState arrival = outside.state;
            apply_outer_boundary(arrival);
            return {
                arrival.r,
                arrival.w,
                false,
                touched_inner,
                0.0,
                0.0,
            };
        }

        const double hit_time = std::clamp(
            outside.first_wall_crossing_time, 0.0, remaining);
        const double boundary_speed =
            std::abs(outside.first_wall_crossing_velocity);
        const double hidden_time =
            delayed_inner_uniform_cell_round_trip_time(
                boundary_speed,
                Lk,
                field.hidden_carrier_mass,
                p);
        touched_inner = true;
        const double time_after_hit = remaining - hit_time;
        if (time_after_hit + time_tolerance < hidden_time) {
            return {
                p.r_min,
                boundary_speed,
                true,
                true,
                boundary_speed,
                hidden_time - time_after_hit,
            };
        }

        remaining = std::max(0.0, time_after_hit - hidden_time);
        state = {p.r_min, boundary_speed};
    }
    throw std::runtime_error(
        "Delayed-return carrier exceeded vp_inner_carrier_max_orbit_segments; "
        "reduce the Vlasov timestep or increase the configured segment limit.");
}

// Select the configured inner-boundary characteristic treatment and retain
// its boundary metadata alongside the final canonical coordinates.
inline SLDGPoint sldg_trace_arrival(
    double source_r,
    double source_w,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    bool regularization_eligible) {
    const double boundary_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, p.r_min);
    if (source_r < p.r_min - boundary_tolerance) {
        throw std::runtime_error(
            "SLDG attempted to trace a source point below its positive r_min face.");
    }
    if (using_delayed_inner_return(p)) {
        return sldg_trace_delayed_inner_arrival(
            std::max(source_r, p.r_min),
            source_w,
            Lk,
            x_r,
            r,
            field,
            dt,
            p);
    }
    const CharacteristicState arrival = trace_vp_characteristic_arrival(
        std::max(source_r, p.r_min),
        source_w,
        Lk,
        x_r,
        r,
        field,
        dt,
        p,
        plan,
        regularization_eligible);
    return SLDGPoint{arrival.r, arrival.w};
}

// Corners, edge midpoints and center share one trace lattice.  These nine
// points measure both geometric validity and directional map curvature.
struct SLDGTracePatch {
    std::array<SLDGPoint, 9> point{};

    SLDGPoint& at(int ir, int iw) {
        return point[static_cast<std::size_t>(3 * ir + iw)];
    }

    const SLDGPoint& at(int ir, int iw) const {
        return point[static_cast<std::size_t>(3 * ir + iw)];
    }

    SLDGPolygon corner_polygon() const {
        return {
            at(0, 0),
            at(2, 0),
            at(2, 2),
            at(0, 2),
        };
    }
};

// A source triangle and its three arrivals define one local affine map.
struct SLDGAffineTriangle {
    std::array<SLDGPoint, 3> source{};
    std::array<SLDGPoint, 3> arrival{};

    SLDGPolygon arrival_polygon() const {
        return {arrival[0], arrival[1], arrival[2]};
    }
};

// A positive determinant gives a locally orientation-preserving triangle.
inline long double sldg_triangle_twice_signed_area(
    const std::array<SLDGPoint, 3>& triangle) {
    const long double x10 =
        static_cast<long double>(triangle[1].r) - triangle[0].r;
    const long double y10 =
        static_cast<long double>(triangle[1].w) - triangle[0].w;
    const long double x20 =
        static_cast<long double>(triangle[2].r) - triangle[0].r;
    const long double y20 =
        static_cast<long double>(triangle[2].w) - triangle[0].w;
    return x10 * y20 - x20 * y10;
}

// Barycentric arrival coordinates recover their source point in the affine
// approximation used by the conservative overlap integral.
inline SLDGPoint sldg_affine_triangle_source_point(
    const SLDGAffineTriangle& triangle,
    long double arrival_r,
    long double arrival_w) {
    const long double ax = triangle.arrival[0].r;
    const long double ay = triangle.arrival[0].w;
    const long double v0x =
        static_cast<long double>(triangle.arrival[1].r) - ax;
    const long double v0y =
        static_cast<long double>(triangle.arrival[1].w) - ay;
    const long double v1x =
        static_cast<long double>(triangle.arrival[2].r) - ax;
    const long double v1y =
        static_cast<long double>(triangle.arrival[2].w) - ay;
    const long double v2x = arrival_r - ax;
    const long double v2y = arrival_w - ay;
    const long double determinant = v0x * v1y - v1x * v0y;
    if (!(determinant > 0.0L)) {
        throw std::runtime_error(
            "SLDG affine triangle has a nonpositive mapped Jacobian.");
    }
    const long double beta =
        (v2x * v1y - v1x * v2y) / determinant;
    const long double gamma =
        (v0x * v2y - v2x * v0y) / determinant;
    return {
        static_cast<double>(
            static_cast<long double>(triangle.source[0].r)
            + beta * (
                static_cast<long double>(triangle.source[1].r)
                - triangle.source[0].r)
            + gamma * (
                static_cast<long double>(triangle.source[2].r)
                - triangle.source[0].r)),
        static_cast<double>(
            static_cast<long double>(triangle.source[0].w)
            + beta * (
                static_cast<long double>(triangle.source[1].w)
                - triangle.source[0].w)
            + gamma * (
                static_cast<long double>(triangle.source[2].w)
                - triangle.source[0].w)),
    };
}

// Express one source coordinate as a constant plus linear r/w arrival terms,
// allowing source polynomials to be integrated with polygon moments.
inline std::array<long double, 3>
sldg_affine_triangle_source_coordinate(
    const SLDGAffineTriangle& triangle,
    bool radial_coordinate) {
    const long double ar = triangle.arrival[0].r;
    const long double aw = triangle.arrival[0].w;
    const long double v0r =
        static_cast<long double>(triangle.arrival[1].r) - ar;
    const long double v0w =
        static_cast<long double>(triangle.arrival[1].w) - aw;
    const long double v1r =
        static_cast<long double>(triangle.arrival[2].r) - ar;
    const long double v1w =
        static_cast<long double>(triangle.arrival[2].w) - aw;
    const long double determinant = v0r * v1w - v1r * v0w;
    if (!(determinant > 0.0L)) {
        throw std::runtime_error(
            "SLDG affine triangle has a nonpositive mapped Jacobian.");
    }
    const long double beta_r = v1w / determinant;
    const long double beta_w = -v1r / determinant;
    const long double beta_0 = (-ar * v1w + aw * v1r) / determinant;
    const long double gamma_r = -v0w / determinant;
    const long double gamma_w = v0r / determinant;
    const long double gamma_0 = (ar * v0w - aw * v0r) / determinant;
    const int component = radial_coordinate ? 0 : 1;
    auto source_component = [&](int vertex) -> long double {
        return component == 0
            ? triangle.source[static_cast<std::size_t>(vertex)].r
            : triangle.source[static_cast<std::size_t>(vertex)].w;
    };
    const long double source0 = source_component(0);
    const long double source_beta = source_component(1) - source0;
    const long double source_gamma = source_component(2) - source0;
    return {
        source0 + source_beta * beta_0 + source_gamma * gamma_0,
        source_beta * beta_r + source_gamma * gamma_r,
        source_beta * beta_w + source_gamma * gamma_w,
    };
}

/// Integrates a quadratic source polynomial over an arrival-space overlap.
/// The inverse affine triangle map makes normalized source x and y affine in
/// arrival (r,w), so area, first moments, and second moments are sufficient.
inline long double sldg_integrate_source_polynomial_over_arrival_overlap(
    const SLDGSourcePolynomial& polynomial,
    const SLDGAffineTriangle& triangle,
    const SLDGExtendedPolygonMoments& overlap) {
    if (!(overlap.area > 0.0L)) {
        return 0.0L;
    }
    const std::array<long double, 3> source_r_map =
        sldg_affine_triangle_source_coordinate(triangle, true);
    const std::array<long double, 3> source_w_map =
        sldg_affine_triangle_source_coordinate(triangle, false);
    const SLDGPoint source_reference =
        sldg_affine_triangle_source_point(
            triangle,
            overlap.reference_r,
            overlap.reference_w);
    const std::array<long double, 3> x{{
        (static_cast<long double>(source_reference.r) - polynomial.center_r)
            / polynomial.scale_r,
        source_r_map[1] / polynomial.scale_r,
        source_r_map[2] / polynomial.scale_r,
    }};
    const std::array<long double, 3> y{{
        (static_cast<long double>(source_reference.w) - polynomial.center_w)
            / polynomial.scale_w,
        source_w_map[1] / polynomial.scale_w,
        source_w_map[2] / polynomial.scale_w,
    }};

    const long double area = overlap.area;
    const long double first_r = overlap.first_offset_r;
    const long double first_w = overlap.first_offset_w;
    auto integrate_affine = [&](const std::array<long double, 3>& a) {
        return a[0] * area + a[1] * first_r + a[2] * first_w;
    };
    auto integrate_product = [&](
        const std::array<long double, 3>& a,
        const std::array<long double, 3>& b) {
        return a[0] * b[0] * area
            + (a[0] * b[1] + a[1] * b[0]) * first_r
            + (a[0] * b[2] + a[2] * b[0]) * first_w
            + a[1] * b[1] * overlap.second_offset_rr
            + (a[1] * b[2] + a[2] * b[1]) * overlap.second_offset_rw
            + a[2] * b[2] * overlap.second_offset_ww;
    };
    const long double integral_x = integrate_affine(x);
    const long double integral_y = integrate_affine(y);
    const long double integral_xx = integrate_product(x, x);
    const long double integral_xy = integrate_product(x, y);
    const long double integral_yy = integrate_product(y, y);
    return static_cast<long double>(polynomial.cell_average) * area
        + static_cast<long double>(polynomial.mode[0]) * integral_x
        + static_cast<long double>(polynomial.mode[1]) * integral_y
        + static_cast<long double>(polynomial.mode[2])
            * (integral_xx - area / 12.0L)
        + static_cast<long double>(polynomial.mode[3]) * integral_xy
        + static_cast<long double>(polynomial.mode[4])
            * (integral_yy - area / 12.0L);
}

/// Integrates the five destination-cell P2 test functions over one overlap.
///
/// The source reconstruction and every destination basis function are
/// quadratic in the arrival coordinates.  Their product has degree four.
/// The six-point Dunavant rule is exact through degree four on each triangle,
/// so triangulating the convex clipped polygon gives the modal moments to
/// roundoff without fitting point samples.
inline std::array<long double, 5>
sldg_integrate_destination_modal_moments(
    const SLDGSourcePolynomial& polynomial,
    const SLDGAffineTriangle& triangle,
    const SLDGExtendedClipContext& context,
    const SLDGExtendedPolygon& normalized_overlap,
    double destination_center_r,
    double destination_center_w,
    double destination_scale_r,
    double destination_scale_w) {
    std::array<long double, 5> integral{};
    if (normalized_overlap.size() < 3) {
        return integral;
    }

    const std::array<long double, 3> source_r_map =
        sldg_affine_triangle_source_coordinate(triangle, true);
    const std::array<long double, 3> source_w_map =
        sldg_affine_triangle_source_coordinate(triangle, false);
    const long double reference_r = context.origin_r
        + context.scale_r * normalized_overlap.front().r;
    const long double reference_w = context.origin_w
        + context.scale_w * normalized_overlap.front().w;
    const long double reference_source_r =
        source_r_map[0] + source_r_map[1] * reference_r
        + source_r_map[2] * reference_w;
    const long double reference_source_w =
        source_w_map[0] + source_w_map[1] * reference_r
        + source_w_map[2] * reference_w;

    constexpr std::array<std::array<long double, 4>, 6> rule{{
        {{0.445948490915965L, 0.445948490915965L,
          0.108103018168070L, 0.111690794839005L}},
        {{0.445948490915965L, 0.108103018168070L,
          0.445948490915965L, 0.111690794839005L}},
        {{0.108103018168070L, 0.445948490915965L,
          0.445948490915965L, 0.111690794839005L}},
        {{0.091576213509771L, 0.091576213509771L,
          0.816847572980459L, 0.054975871827661L}},
        {{0.091576213509771L, 0.816847572980459L,
          0.091576213509771L, 0.054975871827661L}},
        {{0.816847572980459L, 0.091576213509771L,
          0.091576213509771L, 0.054975871827661L}},
    }};

    const SLDGExtendedPoint& a = normalized_overlap.front();
    for (std::size_t q = 1; q + 1 < normalized_overlap.size(); ++q) {
        const SLDGExtendedPoint& b = normalized_overlap[q];
        const SLDGExtendedPoint& c = normalized_overlap[q + 1];
        const long double normalized_jacobian = std::abs(
            (b.r - a.r) * (c.w - a.w)
            - (c.r - a.r) * (b.w - a.w));
        const long double physical_jacobian = normalized_jacobian
            * context.scale_r * context.scale_w;
        if (!(physical_jacobian > 0.0L)) {
            continue;
        }
        for (const auto& quadrature : rule) {
            const long double normalized_r =
                quadrature[0] * a.r
                + quadrature[1] * b.r
                + quadrature[2] * c.r;
            const long double normalized_w =
                quadrature[0] * a.w
                + quadrature[1] * b.w
                + quadrature[2] * c.w;
            const long double arrival_r =
                context.origin_r + context.scale_r * normalized_r;
            const long double arrival_w =
                context.origin_w + context.scale_w * normalized_w;
            const long double offset_r = arrival_r - reference_r;
            const long double offset_w = arrival_w - reference_w;
            const long double source_r = reference_source_r
                + source_r_map[1] * offset_r
                + source_r_map[2] * offset_w;
            const long double source_w = reference_source_w
                + source_w_map[1] * offset_r
                + source_w_map[2] * offset_w;
            const long double x =
                (source_r - polynomial.center_r) / polynomial.scale_r;
            const long double y =
                (source_w - polynomial.center_w) / polynomial.scale_w;
            const long double source_value =
                static_cast<long double>(polynomial.cell_average)
                + static_cast<long double>(polynomial.mode[0]) * x
                + static_cast<long double>(polynomial.mode[1]) * y
                + static_cast<long double>(polynomial.mode[2])
                    * (x * x - 1.0L / 12.0L)
                + static_cast<long double>(polynomial.mode[3]) * x * y
                + static_cast<long double>(polynomial.mode[4])
                    * (y * y - 1.0L / 12.0L);
            const long double xd =
                (arrival_r - destination_center_r) / destination_scale_r;
            const long double yd =
                (arrival_w - destination_center_w) / destination_scale_w;
            const std::array<long double, 5> basis{{
                xd,
                yd,
                xd * xd - 1.0L / 12.0L,
                xd * yd,
                yd * yd - 1.0L / 12.0L,
            }};
            const long double weight =
                physical_jacobian * quadrature[3] * source_value;
            for (std::size_t mode = 0; mode < integral.size(); ++mode) {
                integral[mode] += weight * basis[mode];
            }
        }
    }
    return integral;
}

// Split the four lattice rectangles along fixed diagonals, retaining the same
// source/arrival vertex correspondence on shared edges.
inline std::array<SLDGAffineTriangle, 8> sldg_trace_patch_triangles(
    const SLDGTracePatch& patch,
    double r0,
    double r1,
    double w0,
    double w1) {
    std::array<SLDGAffineTriangle, 8> triangles;
    const std::array<double, 3> source_r{
        r0, 0.5 * (r0 + r1), r1};
    const std::array<double, 3> source_w{
        w0, 0.5 * (w0 + w1), w1};
    int triangle_index = 0;
    for (int ir = 0; ir < 2; ++ir) {
        for (int iw = 0; iw < 2; ++iw) {
            const std::array<std::array<int, 2>, 4> corner{{
                {ir, iw},
                {ir + 1, iw},
                {ir + 1, iw + 1},
                {ir, iw + 1},
            }};
            for (const std::array<int, 3>& local :
                 std::array<std::array<int, 3>, 2>{{
                     {0, 1, 2},
                     {0, 2, 3},
                 }}) {
                SLDGAffineTriangle& triangle =
                    triangles[static_cast<std::size_t>(triangle_index++)];
                for (int q = 0; q < 3; ++q) {
                    const int point =
                        local[static_cast<std::size_t>(q)];
                    const int pr =
                        corner[static_cast<std::size_t>(point)][0];
                    const int pw =
                        corner[static_cast<std::size_t>(point)][1];
                    triangle.source[static_cast<std::size_t>(q)] = {
                        source_r[static_cast<std::size_t>(pr)],
                        source_w[static_cast<std::size_t>(pw)],
                    };
                    triangle.arrival[static_cast<std::size_t>(q)] =
                        patch.at(pr, pw);
                }
            }
        }
    }
    return triangles;
}

// Count actual characteristic solves separately from requests satisfied by
// previously traced shared lattice points.
inline SLDGPoint sldg_trace_arrival_counted(
    double source_r,
    double source_w,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    bool regularization_eligible,
    SLDGStepDiagnostics& diagnostics) {
    ++diagnostics.characteristic_trace_requests;
    ++diagnostics.characteristic_traces;
    return sldg_trace_arrival(
        source_r,
        source_w,
        Lk,
        x_r,
        r,
        field,
        dt,
        p,
        plan,
        regularization_eligible);
}

// Trace one lattice for this source rectangle; dt=0 uses the identity map.
inline SLDGTracePatch sldg_trace_patch(
    double r0,
    double r1,
    double w0,
    double w1,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    SLDGStepDiagnostics& diagnostics) {
    SLDGTracePatch patch;
    if (dt == 0.0) {
        for (int ir = 0; ir < 3; ++ir) {
            for (int iw = 0; iw < 3; ++iw) {
                patch.at(ir, iw) = {
                    r0 + 0.5 * static_cast<double>(ir) * (r1 - r0),
                    w0 + 0.5 * static_cast<double>(iw) * (w1 - w0),
                };
            }
        }
        return patch;
    }
    for (int ir = 0; ir < 3; ++ir) {
        const double source_r =
            r0 + 0.5 * static_cast<double>(ir) * (r1 - r0);
        for (int iw = 0; iw < 3; ++iw) {
            const double source_w =
                w0 + 0.5 * static_cast<double>(iw) * (w1 - w0);
            patch.at(ir, iw) = sldg_trace_arrival_counted(
                source_r,
                source_w,
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
        }
    }
    return patch;
}

// Compare the traced center/edge midpoints with the corner bilinear fit.
// Normalize by source/mapped spans so the defect measures non-affine bending.
inline double sldg_trace_patch_map_defect(
    const SLDGTracePatch& patch,
    double source_r_span,
    double source_w_span) {
    const SLDGPolygon polygon = patch.corner_polygon();
    double mapped_r_min = polygon.front().r;
    double mapped_r_max = mapped_r_min;
    double mapped_w_min = polygon.front().w;
    double mapped_w_max = mapped_w_min;
    for (const SLDGPoint& value : polygon) {
        mapped_r_min = std::min(mapped_r_min, value.r);
        mapped_r_max = std::max(mapped_r_max, value.r);
        mapped_w_min = std::min(mapped_w_min, value.w);
        mapped_w_max = std::max(mapped_w_max, value.w);
    }
    const double r_scale =
        std::max(source_r_span, mapped_r_max - mapped_r_min);
    const double w_scale =
        std::max(source_w_span, mapped_w_max - mapped_w_min);
    auto fitted_point = [&](double xi, double zeta) {
        const std::array<double, 4> shape{
            0.25 * (1.0 - xi) * (1.0 - zeta),
            0.25 * (1.0 + xi) * (1.0 - zeta),
            0.25 * (1.0 + xi) * (1.0 + zeta),
            0.25 * (1.0 - xi) * (1.0 + zeta),
        };
        SLDGPoint fitted;
        for (std::size_t q = 0; q < shape.size(); ++q) {
            fitted.r += shape[q] * polygon[q].r;
            fitted.w += shape[q] * polygon[q].w;
        }
        return fitted;
    };
    double defect = 0.0;
    for (const std::array<int, 2>& index :
         std::array<std::array<int, 2>, 5>{{
             {1, 1},
             {0, 1},
             {2, 1},
             {1, 0},
             {1, 2},
         }}) {
        const int ir = index[0];
        const int iw = index[1];
        const double xi = static_cast<double>(ir - 1);
        const double zeta = static_cast<double>(iw - 1);
        const SLDGPoint fitted = fitted_point(xi, zeta);
        const SLDGPoint& traced = patch.at(ir, iw);
        defect = std::max(
            defect,
            std::hypot(
                (traced.r - fitted.r) / r_scale,
                (traced.w - fitted.w) / w_scale));
    }
    return defect;
}

/// Chooses the source coordinate with the larger measured map curvature.
///
/// The nine traced points already available in a patch contain a directional
/// error estimate: along either source coordinate, a locally affine map puts
/// the traced midpoint at the average of the two traced endpoints.  Refining
/// the coordinate with the larger normalized midpoint residual resolves a
/// strongly anisotropic pericentre map without building an unnecessary full
/// binary tree in the other coordinate.  Nearly equal residuals retain the
/// caller's alternating choice.
inline bool sldg_trace_patch_prefers_radial_split(
    const SLDGTracePatch& patch,
    double source_r_span,
    double source_w_span,
    bool alternating_choice) {
    const SLDGPolygon polygon = patch.corner_polygon();
    double mapped_r_min = polygon.front().r;
    double mapped_r_max = mapped_r_min;
    double mapped_w_min = polygon.front().w;
    double mapped_w_max = mapped_w_min;
    for (const SLDGPoint& value : polygon) {
        mapped_r_min = std::min(mapped_r_min, value.r);
        mapped_r_max = std::max(mapped_r_max, value.r);
        mapped_w_min = std::min(mapped_w_min, value.w);
        mapped_w_max = std::max(mapped_w_max, value.w);
    }
    const double r_scale = std::max({
        source_r_span,
        mapped_r_max - mapped_r_min,
        std::numeric_limits<double>::min(),
    });
    const double w_scale = std::max({
        source_w_span,
        mapped_w_max - mapped_w_min,
        std::numeric_limits<double>::min(),
    });
    auto midpoint_residual = [&](
        const SLDGPoint& lower,
        const SLDGPoint& midpoint,
        const SLDGPoint& upper) {
        return std::hypot(
            (midpoint.r - 0.5 * (lower.r + upper.r)) / r_scale,
            (midpoint.w - 0.5 * (lower.w + upper.w)) / w_scale);
    };

    double radial_curvature = 0.0;
    for (int iw = 0; iw < 3; ++iw) {
        radial_curvature = std::max(
            radial_curvature,
            midpoint_residual(
                patch.at(0, iw), patch.at(1, iw), patch.at(2, iw)));
    }
    double velocity_curvature = 0.0;
    for (int ir = 0; ir < 3; ++ir) {
        velocity_curvature = std::max(
            velocity_curvature,
            midpoint_residual(
                patch.at(ir, 0), patch.at(ir, 1), patch.at(ir, 2)));
    }

    const double larger = std::max(radial_curvature, velocity_curvature);
    const double smaller = std::min(radial_curvature, velocity_curvature);
    if (!(larger > 64.0 * std::numeric_limits<double>::epsilon())
        || larger <= 1.05 * smaller) {
        return alternating_choice;
    }
    return radial_curvature > velocity_curvature;
}

// Child patches reuse shared parent endpoints and midpoints, tracing only
// the additional lattice points needed by the finer rectangles.
inline std::array<SLDGTracePatch, 2> sldg_split_trace_patch(
    const SLDGTracePatch& parent,
    double r0,
    double r1,
    double w0,
    double w1,
    double Lk,
    bool split_radially,
    double split_fraction,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    SLDGStepDiagnostics& diagnostics) {
    std::array<SLDGTracePatch, 2> child;
    if (split_radially) {
        const double split_r = r0 + split_fraction * (r1 - r0);
        const bool reuses_parent_midpoint =
            std::abs(split_fraction - 0.5)
            <= 8.0 * std::numeric_limits<double>::epsilon();
        for (int iw = 0; iw < 3; ++iw) {
            const double source_w =
                w0 + 0.5 * static_cast<double>(iw) * (w1 - w0);
            const SLDGPoint split_point = reuses_parent_midpoint
                ? parent.at(1, iw)
                : sldg_trace_arrival_counted(
                    split_r,
                    source_w,
                    Lk,
                    x_r,
                    r,
                    field,
                    dt,
                    p,
                    plan,
                    true,
                    diagnostics);
            child[0].at(0, iw) = parent.at(0, iw);
            child[0].at(2, iw) = split_point;
            child[1].at(0, iw) = split_point;
            child[1].at(2, iw) = parent.at(2, iw);
            child[0].at(1, iw) = sldg_trace_arrival_counted(
                0.5 * (r0 + split_r),
                source_w,
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
            child[1].at(1, iw) = sldg_trace_arrival_counted(
                0.5 * (split_r + r1),
                source_w,
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
        }
        const long long reused = reuses_parent_midpoint ? 12 : 9;
        diagnostics.characteristic_trace_requests += reused;
        diagnostics.reused_characteristic_points += reused;
    } else {
        const double split_w = w0 + split_fraction * (w1 - w0);
        const bool reuses_parent_midpoint =
            std::abs(split_fraction - 0.5)
            <= 8.0 * std::numeric_limits<double>::epsilon();
        for (int ir = 0; ir < 3; ++ir) {
            const double source_r =
                r0 + 0.5 * static_cast<double>(ir) * (r1 - r0);
            const SLDGPoint split_point = reuses_parent_midpoint
                ? parent.at(ir, 1)
                : sldg_trace_arrival_counted(
                    source_r,
                    split_w,
                    Lk,
                    x_r,
                    r,
                    field,
                    dt,
                    p,
                    plan,
                    true,
                    diagnostics);
            child[0].at(ir, 0) = parent.at(ir, 0);
            child[0].at(ir, 2) = split_point;
            child[1].at(ir, 0) = split_point;
            child[1].at(ir, 2) = parent.at(ir, 2);
            child[0].at(ir, 1) = sldg_trace_arrival_counted(
                source_r,
                0.5 * (w0 + split_w),
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
            child[1].at(ir, 1) = sldg_trace_arrival_counted(
                source_r,
                0.5 * (split_w + w1),
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
        }
        const long long reused = reuses_parent_midpoint ? 12 : 9;
        diagnostics.characteristic_trace_requests += reused;
        diagnostics.reused_characteristic_points += reused;
    }
    return child;
}

struct SLDGInnerReflectionSplit {
    bool straddles = false;
    bool split_radially = true;
};

/// Detects a source rectangle cut by the finite-wall impact separatrix.
///
/// conservative_sldg requires static kick-drift-kick characteristics. The
/// minimum radius along the free angular drift determines whether the orbit
/// reaches the inner wall, including a crossing followed by an outward return.
/// Comparing its distance from the wall along the two source
/// axes gives an anisotropic split that resolves a thin impact sliver without
/// spending half of the refinement depth in the wrong coordinate.
inline SLDGInnerReflectionSplit sldg_inner_reflection_split(
    double r0,
    double r1,
    double w0,
    double w1,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p) {
    SLDGInnerReflectionSplit result;
    if (dt == 0.0 || p.inner_boundary != "reflecting") {
        return result;
    }
    auto unfolded_distance = [&](double source_r, double source_w) {
        const VPSample sample =
            sample_vp_field(source_r, x_r, r, field, p);
        const double w_half = source_w - 0.5 * dt * sample.phi_r;
        const double tangential = Lk / source_r;
        const double speed2 = w_half * w_half + tangential * tangential;
        const double closest_time = speed2 > 0.0
            ? std::max(std::min(0.0, dt),
                std::min(std::max(0.0, dt), -source_r * w_half / speed2))
            : 0.0;
        return std::hypot(source_r + closest_time * w_half,
            closest_time * tangential) - p.r_min;
    };
    const std::array<double, 4> distance{
        unfolded_distance(r0, w0),
        unfolded_distance(r1, w0),
        unfolded_distance(r0, w1),
        unfolded_distance(r1, w1),
    };
    for (double value : distance) {
        if (!std::isfinite(value)) {
            return result;
        }
    }
    const auto bounds = std::minmax_element(distance.begin(), distance.end());
    const double tolerance =
        128.0 * std::numeric_limits<double>::epsilon()
        * std::max({
            1.0,
            std::abs(*bounds.first),
            std::abs(*bounds.second),
            std::abs(r0),
            std::abs(r1),
        });
    result.straddles =
        *bounds.first <= tolerance && *bounds.second >= -tolerance;
    if (!result.straddles) {
        return result;
    }
    const double radial_variation = std::max(
        std::abs(distance[1] - distance[0]),
        std::abs(distance[3] - distance[2]));
    const double velocity_variation = std::max(
        std::abs(distance[2] - distance[0]),
        std::abs(distance[3] - distance[1]));
    result.split_radially = radial_variation >= velocity_variation;
    return result;
}

// The two-point tensor Gauss rule exactly integrates a quadratic source
// reconstruction before its finite masses are carried along characteristics.
inline std::array<SLDGPoint, 4> sldg_trace_subcell_gauss_points(
    double r0,
    double r1,
    double w0,
    double w1,
    double Lk,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    bool regularization_eligible,
    SLDGStepDiagnostics& diagnostics) {
    constexpr double gauss = 0.577350269189625764509148780501957456;
    const double rc = 0.5 * (r0 + r1);
    const double wc = 0.5 * (w0 + w1);
    const double hr = 0.5 * (r1 - r0);
    const double hw = 0.5 * (w1 - w0);
    std::array<SLDGPoint, 4> mapped;
    int q = 0;
    for (double xi : {-gauss, gauss}) {
        for (double zeta : {-gauss, gauss}) {
            mapped[static_cast<std::size_t>(q++)] = sldg_trace_arrival_counted(
                rc + hr * xi,
                wc + hw * zeta,
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                regularization_eligible,
                diagnostics);
        }
    }
    return mapped;
}

// Integrate each positive source triangle over destination overlaps.  The
// source/mapped area ratio is the inverse affine Jacobian, ensuring the
// original source mass is transferred despite a small admitted map-area defect.
inline double sldg_deposit_affine_patch_with_quadratic_reconstruction(
    const std::array<SLDGAffineTriangle, 8>& triangles,
    double source_mass,
    const SLDGSourcePolynomial& polynomial,
    int k,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::vector<double>& deposited_mass,
    std::vector<std::array<long double, 5>>* deposited_modal_moments,
    bool allow_signed_source = false) {
    long double total_deposited = 0.0L;
    double source_r_min = triangles.front().source.front().r;
    double source_r_max = source_r_min;
    double source_w_min = triangles.front().source.front().w;
    double source_w_max = source_w_min;
    for (const SLDGAffineTriangle& triangle : triangles) {
        for (const SLDGPoint& point : triangle.source) {
            source_r_min = std::min(source_r_min, point.r);
            source_r_max = std::max(source_r_max, point.r);
            source_w_min = std::min(source_w_min, point.w);
            source_w_max = std::max(source_w_max, point.w);
        }
    }
    const long double positivity_scale = std::max<long double>({
        1.0L,
        std::abs(static_cast<long double>(polynomial.cell_average)),
        std::abs(static_cast<long double>(polynomial.mode[0])),
        std::abs(static_cast<long double>(polynomial.mode[1])),
        std::abs(static_cast<long double>(polynomial.mode[2])),
        std::abs(static_cast<long double>(polynomial.mode[3])),
        std::abs(static_cast<long double>(polynomial.mode[4])),
    });
    const long double positivity_tolerance =
        4096.0L * std::numeric_limits<long double>::epsilon()
        * positivity_scale;

    for (const SLDGAffineTriangle& triangle : triangles) {
        const long double source_twice_area =
            sldg_triangle_twice_signed_area(triangle.source);
        const long double mapped_twice_area =
            sldg_triangle_twice_signed_area(triangle.arrival);
        if (!(source_twice_area > 0.0L)
            || !(mapped_twice_area > 0.0L)) {
            throw std::runtime_error(
                "SLDG affine-triangle deposition received a folded patch.");
        }
        const long double source_triangle_area =
            0.5L * source_twice_area;
        const SLDGExtendedClipContext clip_context(
            triangle.arrival_polygon());
        const long double mapped_triangle_area =
            clip_context.physical_area();
        if (!(mapped_triangle_area > 0.0L)) {
            throw std::runtime_error(
                "SLDG affine-triangle deposition received zero mapped area.");
        }
        // Integrating in arrival area needs this source-area factor.
        // It is local to the affine patch and preserves its source mass.
        const long double inverse_jacobian =
            source_triangle_area / mapped_triangle_area;

        double min_r = triangle.arrival[0].r;
        double max_r = min_r;
        double min_w = triangle.arrival[0].w;
        double max_w = min_w;
        for (const SLDGPoint& point : triangle.arrival) {
            min_r = std::min(min_r, point.r);
            max_r = std::max(max_r, point.r);
            min_w = std::min(min_w, point.w);
            max_w = std::max(max_w, point.w);
        }
        const int ir0 = sldg_cell_for_coordinate(
            geometry.radial.edge,
            std::max(min_r, geometry.radial.edge.front()));
        const int ir1 = sldg_cell_for_coordinate(
            geometry.radial.edge,
            std::min(max_r, geometry.radial.edge.back()));
        const int jw0 = sldg_cell_for_coordinate(
            geometry.radial_velocity.edge,
            std::max(
                min_w,
                geometry.radial_velocity.edge.front()));
        const int jw1 = sldg_cell_for_coordinate(
            geometry.radial_velocity.edge,
            std::min(
                max_w,
                geometry.radial_velocity.edge.back()));

        long double overlap_sum = 0.0L;
        if (ir0 >= 0 && ir1 >= 0 && jw0 >= 0 && jw1 >= 0) {
            for (int id = ir0; id <= ir1; ++id) {
                for (int jd = jw0; jd <= jw1; ++jd) {
                    const SLDGExtendedPolygonMoments overlap =
                        clip_context.rectangle_moments(
                            geometry.radial.edge[
                                static_cast<std::size_t>(id)],
                            geometry.radial.edge[
                                static_cast<std::size_t>(id + 1)],
                            geometry.radial_velocity.edge[
                                static_cast<std::size_t>(jd)],
                            geometry.radial_velocity.edge[
                                static_cast<std::size_t>(jd + 1)]);
                    if (!(overlap.area > 0.0L)) {
                        continue;
                    }
                    overlap_sum += overlap.area;
                    long double source_integral =
                        sldg_integrate_source_polynomial_over_arrival_overlap(
                            polynomial, triangle, overlap);
                    if (!std::isfinite(source_integral)
                        || (!allow_signed_source
                            && source_integral
                                < -positivity_tolerance * overlap.area)) {
                        throw std::runtime_error(
                            "SLDG affine-triangle remap produced a negative "
                            "quadratic source integral.");
                    }
                    if (!allow_signed_source) {
                        source_integral = std::max(0.0L, source_integral);
                    }
                    const long double mass =
                        inverse_jacobian * source_integral;
                    const std::size_t destination = idx(id, jd, k, p);
                    deposited_mass[destination] += static_cast<double>(mass);
                    if (deposited_modal_moments != nullptr) {
                        const SLDGExtendedPolygon normalized_overlap =
                            clip_context.rectangle_normalized_polygon(
                                geometry.radial.edge[
                                    static_cast<std::size_t>(id)],
                                geometry.radial.edge[
                                    static_cast<std::size_t>(id + 1)],
                                geometry.radial_velocity.edge[
                                    static_cast<std::size_t>(jd)],
                                geometry.radial_velocity.edge[
                                    static_cast<std::size_t>(jd + 1)]);
                        const std::array<long double, 5> modal_integral =
                            sldg_integrate_destination_modal_moments(
                                polynomial,
                                triangle,
                                clip_context,
                                normalized_overlap,
                                geometry.radial.center[
                                    static_cast<std::size_t>(id)],
                                geometry.radial_velocity.center[
                                    static_cast<std::size_t>(jd)],
                                geometry.radial.width[
                                    static_cast<std::size_t>(id)],
                                geometry.radial_velocity.width[
                                    static_cast<std::size_t>(jd)]);
                        for (std::size_t mode = 0;
                             mode < modal_integral.size();
                             ++mode) {
                            (*deposited_modal_moments)[destination][mode] +=
                                inverse_jacobian * modal_integral[mode];
                        }
                    }
                    total_deposited += mass;
                }
            }
        }

        // Destination cells must tile the in-domain part of this triangle.
        // The remaining source mass is recorded as domain escape by the caller.
        const SLDGExtendedPolygonMoments domain_overlap =
            clip_context.rectangle_moments(
                geometry.radial.edge.front(),
                geometry.radial.edge.back(),
                geometry.radial_velocity.edge.front(),
                geometry.radial_velocity.edge.back());
        const long double overlap_tolerance =
            std::max<long double>(
                p.vp_sldg_geometry_tolerance,
                4096.0L
                    * std::numeric_limits<long double>::epsilon())
            * std::max({
                std::numeric_limits<long double>::min(),
                mapped_triangle_area,
                domain_overlap.area,
                overlap_sum,
            });
        if (domain_overlap.area - mapped_triangle_area
            > overlap_tolerance) {
            throw std::runtime_error(
                "SLDG affine-triangle domain overlap exceeds mapped area.");
        }
        if (std::abs(overlap_sum - domain_overlap.area)
            > overlap_tolerance) {
            throw std::runtime_error(
                "SLDG destination cells do not tile an affine triangle.");
        }
    }

    // Eight affine triangles contribute independently.  Near a limited zero,
    // the positive cell average and slope terms can nearly cancel, so scale
    // the roundoff guard by those terms before cancellation as well as by the
    // net source mass.  This is an error bound only; no deposited mass is
    // altered.
    const long double source_area =
        static_cast<long double>(source_r_max - source_r_min)
        * (source_w_max - source_w_min);
    const long double conditioned_source_mass_scale =
        source_area
        * (
            std::abs(static_cast<long double>(polynomial.cell_average))
            + std::abs(static_cast<long double>(polynomial.mode[0]))
            + std::abs(static_cast<long double>(polynomial.mode[1]))
            + std::abs(static_cast<long double>(polynomial.mode[2]))
            + std::abs(static_cast<long double>(polynomial.mode[3]))
            + std::abs(static_cast<long double>(polynomial.mode[4])));
    const long double geometric_mass_tolerance =
        16.0L
        * std::max<long double>(
            p.vp_sldg_geometry_tolerance,
            16384.0L * std::numeric_limits<long double>::epsilon())
        * std::max({
            std::numeric_limits<long double>::min(),
            std::abs(static_cast<long double>(source_mass)),
            std::abs(total_deposited),
            conditioned_source_mass_scale,
        });
    // source_mass is stored as double while the clipped integral above is
    // accumulated in long double.  For a subnormal source, the conversion
    // spacing is much larger than epsilon times that source.  Include the
    // double storage spacing explicitly in this comparison-only tolerance.
    const long double double_storage_tolerance =
        8.0L * std::numeric_limits<double>::epsilon()
        * std::max<long double>(
            std::numeric_limits<double>::min(),
            std::abs(static_cast<long double>(source_mass)));
    const long double mass_tolerance = std::max(
        geometric_mass_tolerance,
        double_storage_tolerance);
    if (!allow_signed_source && total_deposited
        > static_cast<long double>(source_mass) + mass_tolerance) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "SLDG affine-triangle remap deposited more than its source "
                   "mass: source="
                << source_mass
                << ", deposited=" << static_cast<double>(total_deposited)
                << ", excess="
                << static_cast<double>(
                       total_deposited
                       - static_cast<long double>(source_mass))
                << ", tolerance=" << static_cast<double>(mass_tolerance)
                << ", f_center=" << polynomial.cell_average
                << ", p2_modes=(" << polynomial.mode[0]
                << ',' << polynomial.mode[1]
                << ',' << polynomial.mode[2]
                << ',' << polynomial.mode[3]
                << ',' << polynomial.mode[4] << ')'
                << ", source_r=[" << source_r_min << ',' << source_r_max << ']'
                << ", source_w=[" << source_w_min << ',' << source_w_max << ']';
        throw std::runtime_error(message.str());
    }
    return static_cast<double>(total_deposited);
}

struct SLDGAxisCloud {
    std::array<int, 2> index{{-1, -1}};
    std::array<double, 2> weight{{0.0, 0.0}};
    int count = 0;
};

/// Builds a nonnegative two-point cloud that reproduces a coordinate between
/// neighboring cell centers.  Points between a domain face and its nearest
/// center remain in that boundary cell.
inline SLDGAxisCloud sldg_axis_cloud(
    double coordinate,
    const PhaseSpaceAxis& axis) {
    SLDGAxisCloud cloud;
    const double scale = std::max({
        1.0,
        std::abs(coordinate),
        std::abs(axis.edge.front()),
        std::abs(axis.edge.back()),
    });
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (coordinate < axis.edge.front() - tolerance
        || coordinate > axis.edge.back() + tolerance) {
        return cloud;
    }
    const double x = std::clamp(
        coordinate, axis.edge.front(), axis.edge.back());
    if (axis.center.size() == 1 || x <= axis.center.front()) {
        cloud.index[0] = 0;
        cloud.weight[0] = 1.0;
        cloud.count = 1;
        return cloud;
    }
    if (x >= axis.center.back()) {
        cloud.index[0] = static_cast<int>(axis.center.size()) - 1;
        cloud.weight[0] = 1.0;
        cloud.count = 1;
        return cloud;
    }
    const auto upper = std::upper_bound(
        axis.center.begin(), axis.center.end(), x);
    const int right = static_cast<int>(
        std::distance(axis.center.begin(), upper));
    const int left = right - 1;
    const double span = axis.center[static_cast<std::size_t>(right)]
        - axis.center[static_cast<std::size_t>(left)];
    const double right_weight = std::clamp(
        (x - axis.center[static_cast<std::size_t>(left)]) / span,
        0.0,
        1.0);
    cloud.index = {{left, right}};
    cloud.weight = {{1.0 - right_weight, right_weight}};
    cloud.count = 2;
    return cloud;
}

/// Deposits one finite mass packet with a compact, conservative cloud.
inline bool sldg_deposit_point_cloud(
    const SLDGPoint& arrival,
    int k,
    double reduced_mass,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    std::vector<double>& deposited_mass,
    std::vector<std::array<long double, 5>>* deposited_modal_moments = nullptr,
    bool allow_signed_mass = false) {
    if (arrival.hidden || !std::isfinite(reduced_mass)
        || (!allow_signed_mass && !(reduced_mass >= 0.0))) {
        return false;
    }
    const SLDGAxisCloud radial =
        sldg_axis_cloud(arrival.r, geometry.radial);
    const SLDGAxisCloud velocity =
        sldg_axis_cloud(arrival.w, geometry.radial_velocity);
    if (radial.count == 0 || velocity.count == 0) {
        return false;
    }
    for (int a = 0; a < radial.count; ++a) {
        for (int b = 0; b < velocity.count; ++b) {
            const int id = radial.index[static_cast<std::size_t>(a)];
            const int jd = velocity.index[static_cast<std::size_t>(b)];
            const double point_mass = reduced_mass
                * radial.weight[static_cast<std::size_t>(a)]
                * velocity.weight[static_cast<std::size_t>(b)];
            const std::size_t destination = idx(id, jd, k, p);
            deposited_mass[destination] += point_mass;
            if (deposited_modal_moments != nullptr) {
                const long double x =
                    (static_cast<long double>(arrival.r)
                     - geometry.radial.center[static_cast<std::size_t>(id)])
                    / geometry.radial.width[static_cast<std::size_t>(id)];
                const long double y =
                    (static_cast<long double>(arrival.w)
                     - geometry.radial_velocity.center[
                         static_cast<std::size_t>(jd)])
                    / geometry.radial_velocity.width[
                        static_cast<std::size_t>(jd)];
                const std::array<long double, 5> basis{{
                    x,
                    y,
                    x * x - 1.0L / 12.0L,
                    x * y,
                    y * y - 1.0L / 12.0L,
                }};
                for (std::size_t mode = 0; mode < basis.size(); ++mode) {
                    (*deposited_modal_moments)[destination][mode] +=
                        static_cast<long double>(point_mass) * basis[mode];
                }
            }
        }
    }
    return true;
}

struct SLDGDelayedCarrierSplit {
    bool contains_hidden = false;
    bool straddles = false;
    bool split_radially = true;
};

/// Classifies the visible/hidden transition on a 3x3 trace patch.
///
/// A returned point has touched_inner=true even when it is visible at the end
/// of the step.  Refining patches that contain such points keeps a narrow
/// hidden-time band from being skipped between two already-returned samples.
inline SLDGDelayedCarrierSplit sldg_delayed_carrier_split(
    const SLDGTracePatch& patch) {
    SLDGDelayedCarrierSplit result;
    int minimum_state = 2;
    int maximum_state = 0;
    int radial_variation = 0;
    int velocity_variation = 0;
    auto state = [](const SLDGPoint& point) {
        return point.hidden ? 2 : (point.touched_inner ? 1 : 0);
    };
    for (const SLDGPoint& point : patch.point) {
        const int point_state = state(point);
        result.contains_hidden = result.contains_hidden || point.hidden;
        minimum_state = std::min(minimum_state, point_state);
        maximum_state = std::max(maximum_state, point_state);
    }
    result.straddles = minimum_state != maximum_state;
    for (int ir = 0; ir < 2; ++ir) {
        for (int iw = 0; iw < 3; ++iw) {
            radial_variation += std::abs(
                state(patch.at(ir + 1, iw)) - state(patch.at(ir, iw)));
        }
    }
    for (int ir = 0; ir < 3; ++ir) {
        for (int iw = 0; iw < 2; ++iw) {
            velocity_variation += std::abs(
                state(patch.at(ir, iw + 1)) - state(patch.at(ir, iw)));
        }
    }
    result.split_radially = radial_variation >= velocity_variation;
    return result;
}

/// Resolves a source rectangle cut by the visible/hidden separatrix.  Tensor
/// Gauss quadrature partitions its reconstructed mass between the visible
/// grid and persistent carrier packets without renormalization.
inline void sldg_deposit_delayed_carrier_quadrature(
    double r0,
    double r1,
    double w0,
    double w1,
    const SLDGSourcePolynomial& polynomial,
    double Lk,
    int k,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    double diagnostic_measure,
    std::vector<double>& deposited_mass,
    std::vector<std::array<long double, 5>>* deposited_modal_moments,
    std::vector<DelayedInnerCarrierPacket>& captured_packets,
    SLDGStepDiagnostics& diagnostics,
    bool allow_signed_source = false) {
    const std::size_t packet_count_before = captured_packets.size();
    const auto& rule = phase_space_gauss_legendre_rule(
        p.vp_inner_carrier_quadrature_points);
    const double rc = 0.5 * (r0 + r1);
    const double wc = 0.5 * (w0 + w1);
    const double hr = 0.5 * (r1 - r0);
    const double hw = 0.5 * (w1 - w0);
    double source_mass = 0.0;
    double visible_mass = 0.0;
    double captured_mass = 0.0;
    double escaped_mass = 0.0;
    for (std::size_t qr = 0; qr < rule.first.size(); ++qr) {
        for (std::size_t qw = 0; qw < rule.first.size(); ++qw) {
            const double source_r = rc + hr * rule.first[qr];
            const double source_w = wc + hw * rule.first[qw];
            const double source_value =
                polynomial.value(source_r, source_w);
            if (!std::isfinite(source_value)
                || (!allow_signed_source && !(source_value >= 0.0))) {
                throw std::runtime_error(
                    "Delayed-return quadrature encountered an invalid reconstruction.");
            }
            const double point_mass = hr * hw
                * rule.second[qr] * rule.second[qw] * source_value;
            source_mass += point_mass;
            if (point_mass == 0.0) {
                continue;
            }
            const SLDGPoint arrival = sldg_trace_arrival_counted(
                source_r,
                source_w,
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
            if (arrival.hidden) {
                if (!(point_mass > 0.0)) {
                    throw std::runtime_error(
                        "A signed Hamiltonian residual reached the delayed-return carrier; "
                        "the orbit-reference closed-orbit mask is inconsistent.");
                }
                // Preserve the actual eta quadrature node and remaining
                // return time, so the hidden orbit resumes with the same L.
                captured_packets.push_back({
                    point_mass,
                    k,
                    0.5 * Lk * Lk,
                    arrival.boundary_speed,
                    arrival.remaining_hidden_time,
                });
                captured_mass += point_mass;
                continue;
            }
            if (sldg_deposit_point_cloud(
                    arrival,
                    k,
                    point_mass,
                    geometry,
                    p,
                    deposited_mass,
                    deposited_modal_moments,
                    allow_signed_source)) {
                visible_mass += point_mass;
            } else {
                escaped_mass += point_mass;
            }
        }
    }
    const double expected_source_mass =
        polynomial.rectangle_integral(r0, r1, w0, w1);
    const double quadrature_scale = std::max(1.0, expected_source_mass);
    if (std::abs(source_mass - expected_source_mass)
        > 256.0 * std::numeric_limits<double>::epsilon()
            * quadrature_scale) {
        throw std::runtime_error(
            "Delayed-return quadrature failed its source-mass identity.");
    }
    diagnostics.source_mass += diagnostic_measure * source_mass;
    diagnostics.deposited_mass += diagnostic_measure * visible_mass;
    diagnostics.carrier_captured_mass +=
        diagnostic_measure * captured_mass;
    diagnostics.escaped_mass += diagnostic_measure * escaped_mass;
    diagnostics.carrier_packets_captured += static_cast<long long>(
        captured_packets.size() - packet_count_before);
    ++diagnostics.transported_subcells;
}

// Reserve at least one visit for each child and divide extra work according
// to nonnegative represented mass so one branch cannot consume the full budget.
inline std::array<long long, 2> sldg_allocate_binary_refinement_work(
    long long available_work,
    double first_mass,
    double second_mass) {
    if (available_work < 2) {
        return {0, 0};
    }
    const long long extra = available_work - 2;
    const double nonnegative_first = std::max(0.0, first_mass);
    const double nonnegative_second = std::max(0.0, second_mass);
    const double total_mass = nonnegative_first + nonnegative_second;
    long long first_extra = extra / 2;
    if (total_mass > 0.0 && std::isfinite(total_mass)) {
        const long double weighted =
            static_cast<long double>(extra)
            * static_cast<long double>(nonnegative_first)
            / static_cast<long double>(total_mass);
        first_extra = std::clamp(
            static_cast<long long>(std::llround(weighted)),
            0LL,
            extra);
    }
    return {1 + first_extra, 1 + extra - first_extra};
}

// Separate folding/orientation, area change and non-affine bending tests;
// their largest normalized defect guides bounded refinement.
struct SLDGTracePatchAssessment {
    SLDGPolygon arrival_polygon{};
    std::array<SLDGAffineTriangle, 8> affine_triangles{};
    double mapped_area = 0.0;
    double area_defect = std::numeric_limits<double>::infinity();
    double map_defect = 0.0;
    double maximum_triangle_area_defect = 0.0;
    bool valid_basic_geometry = false;
    bool valid_triangle_geometry = false;
    bool valid_polygon = false;

    double refinement_score(const Params& p) const {
        if (valid_polygon) {
            return 0.0;
        }
        const double area_scale = std::max(
            p.vp_sldg_max_area_defect,
            std::numeric_limits<double>::min());
        const double map_scale = std::max(
            p.vp_sldg_max_map_defect,
            std::numeric_limits<double>::min());
        double score = std::max({
            area_defect / area_scale,
            maximum_triangle_area_defect / area_scale,
            map_defect / map_scale,
        });
        if (!valid_basic_geometry || !valid_triangle_geometry) {
            score = std::max(score, 2.0);
        }
        return std::isfinite(score)
            ? score
            : std::numeric_limits<double>::max();
    }
};

// Check the corner polygon and every affine triangle against the canonical
// area of its source patch, then apply the non-affine map-defect limit.
inline SLDGTracePatchAssessment sldg_assess_trace_patch(
    const SLDGTracePatch& trace_patch,
    double r0,
    double r1,
    double w0,
    double w1,
    const Params& p) {
    SLDGTracePatchAssessment result;
    const double source_area = (r1 - r0) * (w1 - w0);
    result.arrival_polygon = trace_patch.corner_polygon();
    result.affine_triangles =
        sldg_trace_patch_triangles(trace_patch, r0, r1, w0, w1);
    result.mapped_area = sldg_polygon_area(result.arrival_polygon);
    result.area_defect =
        std::abs(result.mapped_area - source_area) / source_area;
    result.valid_basic_geometry =
        result.mapped_area > p.vp_sldg_geometry_tolerance * source_area
        && result.area_defect <= p.vp_sldg_max_area_defect
        && sldg_is_finite_convex_ccw_quadrilateral(
            result.arrival_polygon,
            p.vp_sldg_geometry_tolerance * source_area);
    result.map_defect = result.valid_basic_geometry
        ? sldg_trace_patch_map_defect(
            trace_patch, r1 - r0, w1 - w0)
        : 0.0;

    result.valid_triangle_geometry = true;
    result.maximum_triangle_area_defect = 0.0;
    const long double source_triangle_area =
        static_cast<long double>(source_area) / 8.0L;
    for (const SLDGAffineTriangle& triangle : result.affine_triangles) {
        const long double mapped_twice_area =
            sldg_triangle_twice_signed_area(triangle.arrival);
        if (!(mapped_twice_area
              > 2.0L * p.vp_sldg_geometry_tolerance
                    * source_triangle_area)) {
            result.valid_triangle_geometry = false;
            continue;
        }
        const double triangle_area_defect = static_cast<double>(
            std::abs(0.5L * mapped_twice_area - source_triangle_area)
            / source_triangle_area);
        result.maximum_triangle_area_defect = std::max(
            result.maximum_triangle_area_defect,
            triangle_area_defect);
        if (triangle_area_defect > p.vp_sldg_max_area_defect) {
            result.valid_triangle_geometry = false;
        }
    }
    result.valid_polygon =
        result.valid_basic_geometry
        && result.valid_triangle_geometry
        && result.map_defect <= p.vp_sldg_max_map_defect;
    return result;
}

/// Deposits one source rectangle, recursively resolving non-affine flow maps.
///
/// A valid leaf uses exact polygon/rectangle overlap.  A folded or
/// insufficiently area-preserving image is split in r or w.  A rectangle cut
/// by the inner-wall impact separatrix is split in the coordinate that
/// contributes the larger separatrix variation.  Other invalid maps use the
/// measured directional curvature, with alternation only for a near tie.  Each
/// split reserves work for both children in proportion to their reconstructed
/// source mass.  Only after the configured refinement depth or branch budget
/// is exhausted is bounded Gauss-point deposition permitted, and the mass
/// carried by that fallback is ledgered.
inline long long sldg_deposit_source_rectangle(
    double r0,
    double r1,
    double w0,
    double w1,
    const SLDGSourcePolynomial& polynomial,
    double Lk,
    int k,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    double diagnostic_measure,
    const SLDGTracePatch* provided_trace_patch,
    const SLDGTracePatchAssessment* provided_assessment,
    int refinement,
    long long work_budget,
    std::vector<double>& deposited_mass,
    std::vector<std::array<long double, 5>>* deposited_modal_moments,
    std::vector<DelayedInnerCarrierPacket>& captured_packets,
    SLDGStepDiagnostics& diagnostics,
    bool allow_signed_source = false) {
    if (work_budget < 1) {
        throw std::runtime_error(
            "SLDG recursive refinement received an empty work budget.");
    }
    long long work_used = 1;
    const long long available_child_work = work_budget - work_used;
    ++diagnostics.traced_rectangles;
    const double rc = 0.5 * (r0 + r1);
    const double wc = 0.5 * (w0 + w1);
    const double source_area = (r1 - r0) * (w1 - w0);
    const double reconstructed = polynomial.value(rc, wc);
    if (!std::isfinite(reconstructed)
        || (!allow_signed_source && !(reconstructed >= 0.0))) {
        std::ostringstream message;
        message
            << "SLDG positivity limiter produced an invalid recursively "
            << "refined subcell average: k=" << k
            << ", refinement=" << refinement
            << ", r0=" << std::setprecision(17) << r0
            << ", r1=" << r1
            << ", w0=" << w0
            << ", w1=" << w1
            << ", cell_average=" << polynomial.cell_average
            << ", reconstructed=" << reconstructed << '.';
        throw std::runtime_error(message.str());
    }
    const double source_mass =
        polynomial.rectangle_integral(r0, r1, w0, w1);
    const double source_mass_tolerance =
        4096.0 * std::numeric_limits<double>::epsilon()
        * source_area
        * std::max({
            1.0,
            std::abs(polynomial.cell_average),
            std::abs(polynomial.mode[0]),
            std::abs(polynomial.mode[1]),
            std::abs(polynomial.mode[2]),
            std::abs(polynomial.mode[3]),
            std::abs(polynomial.mode[4]),
        });
    if (!std::isfinite(source_mass)
        || (!allow_signed_source
            && source_mass < -source_mass_tolerance)) {
        throw std::runtime_error(
            "SLDG positivity reconstruction produced a negative subcell mass.");
    }
    if (!allow_signed_source && !(source_mass > 0.0)) {
        return work_used;
    }
    auto child_source_mass = [&](
        bool split_radially,
        double split_coordinate,
        int half) {
        const double child_r0 =
            split_radially && half == 1 ? split_coordinate : r0;
        const double child_r1 =
            split_radially && half == 0 ? split_coordinate : r1;
        const double child_w0 =
            !split_radially && half == 1 ? split_coordinate : w0;
        const double child_w1 =
            !split_radially && half == 0 ? split_coordinate : w1;
        const double child_mass = polynomial.rectangle_integral(
            child_r0, child_r1, child_w0, child_w1);
        return allow_signed_source
            ? std::abs(child_mass)
            : std::max(0.0, child_mass);
    };

    const SLDGTracePatch local_trace_patch = provided_trace_patch
        ? SLDGTracePatch{}
        : sldg_trace_patch(
            r0,
            r1,
            w0,
            w1,
            Lk,
            x_r,
            r,
            field,
            dt,
            p,
            plan,
            diagnostics);
    const SLDGTracePatch& trace_patch = provided_trace_patch
        ? *provided_trace_patch
        : local_trace_patch;
    const SLDGDelayedCarrierSplit carrier_split =
        using_delayed_inner_return(p)
        ? sldg_delayed_carrier_split(trace_patch)
        : SLDGDelayedCarrierSplit{};
    auto deposit_refined_children = [&](bool split_radially) {
        const double split_fraction = 0.5;
        const double rm = r0 + split_fraction * (r1 - r0);
        const double wm = w0 + split_fraction * (w1 - w0);
        const double split_coordinate = split_radially ? rm : wm;
        const std::array<SLDGTracePatch, 2> child_trace_patch =
            sldg_split_trace_patch(
                trace_patch,
                r0,
                r1,
                w0,
                w1,
                Lk,
                split_radially,
                split_fraction,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                diagnostics);
        std::array<double, 2> child_mass{};
        std::array<double, 2> child_score{};
        std::array<SLDGTracePatchAssessment, 2> child_assessment{};
        for (int half = 0; half < 2; ++half) {
            const double child_r0 = split_radially && half == 1 ? rm : r0;
            const double child_r1 = split_radially && half == 0 ? rm : r1;
            const double child_w0 = !split_radially && half == 1 ? wm : w0;
            const double child_w1 = !split_radially && half == 0 ? wm : w1;
            child_mass[static_cast<std::size_t>(half)] =
                child_source_mass(split_radially, split_coordinate, half);
            child_assessment[static_cast<std::size_t>(half)] =
                sldg_assess_trace_patch(
                    child_trace_patch[static_cast<std::size_t>(half)],
                    child_r0,
                    child_r1,
                    child_w0,
                    child_w1,
                    p);
            const SLDGDelayedCarrierSplit child_carrier =
                using_delayed_inner_return(p)
                ? sldg_delayed_carrier_split(
                    child_trace_patch[static_cast<std::size_t>(half)])
                : SLDGDelayedCarrierSplit{};
            child_score[static_cast<std::size_t>(half)] =
                child_carrier.straddles
                ? std::numeric_limits<double>::max()
                : child_carrier.contains_hidden
                    ? 0.0
                    : child_assessment[static_cast<std::size_t>(half)]
                        .refinement_score(p);
        }

        // Visit the easier child first and give unused work to the second,
        // always reserving its initial visit.  Both inherit the same source
        // polynomial, so their source masses partition the parent integral.
        const std::array<long long, 2> nominal_budget =
            sldg_allocate_binary_refinement_work(
                available_child_work, child_mass[0], child_mass[1]);
        int first = 0;
        if (child_score[1] < child_score[0]
            || (child_score[1] == child_score[0]
                && child_mass[1] < child_mass[0])) {
            first = 1;
        }
        const int second = 1 - first;
        auto deposit_child = [&](int half, long long child_budget) {
            const double child_r0 = split_radially && half == 1 ? rm : r0;
            const double child_r1 = split_radially && half == 0 ? rm : r1;
            const double child_w0 = !split_radially && half == 1 ? wm : w0;
            const double child_w1 = !split_radially && half == 0 ? wm : w1;
            return sldg_deposit_source_rectangle(
                child_r0,
                child_r1,
                child_w0,
                child_w1,
                polynomial,
                Lk,
                k,
                x_r,
                r,
                field,
                geometry,
                dt,
                p,
                plan,
                diagnostic_measure,
                &child_trace_patch[static_cast<std::size_t>(half)],
                &child_assessment[static_cast<std::size_t>(half)],
                refinement + 1,
                child_budget,
                deposited_mass,
                deposited_modal_moments,
                captured_packets,
                diagnostics,
                allow_signed_source);
        };
        const long long first_work = deposit_child(
            first, nominal_budget[static_cast<std::size_t>(first)]);
        const long long second_budget = available_child_work - first_work;
        if (second_budget < 1) {
            throw std::runtime_error(
                "SLDG refinement failed to reserve work for its second child.");
        }
        const long long second_work = deposit_child(second, second_budget);
        return first_work + second_work;
    };
    const int carrier_maximum_refinement =
        p.vp_inner_carrier_max_refinement;
    if (carrier_split.straddles
        && refinement < carrier_maximum_refinement
        && available_child_work >= 2) {
        const bool split_radially = carrier_split.split_radially;
        work_used += deposit_refined_children(split_radially);
        return work_used;
    }
    // Visible/hidden transitions use their own quadrature and persistent
    // carrier partition after the dedicated boundary refinement limit.
    if (carrier_split.contains_hidden || carrier_split.straddles) {
        sldg_deposit_delayed_carrier_quadrature(
            r0,
            r1,
            w0,
            w1,
            polynomial,
            Lk,
            k,
            x_r,
            r,
            field,
            geometry,
            dt,
            p,
            plan,
            diagnostic_measure,
            deposited_mass,
            deposited_modal_moments,
            captured_packets,
            diagnostics,
            allow_signed_source);
        return work_used;
    }
    const SLDGTracePatchAssessment local_assessment = provided_assessment
        ? SLDGTracePatchAssessment{}
        : sldg_assess_trace_patch(trace_patch, r0, r1, w0, w1, p);
    const SLDGTracePatchAssessment& assessment = provided_assessment
        ? *provided_assessment
        : local_assessment;
    const bool valid_polygon = assessment.valid_polygon;

    if (!valid_polygon) {
        ++diagnostics.rejected_rectangles;
        diagnostics.maximum_rejected_area_defect = std::max(
            diagnostics.maximum_rejected_area_defect,
            std::max(
                assessment.area_defect,
                assessment.maximum_triangle_area_defect));
        diagnostics.maximum_rejected_map_defect = std::max(
            diagnostics.maximum_rejected_map_defect,
            assessment.map_defect);
    }
    const SLDGInnerReflectionSplit reflection_split = !valid_polygon
        ? sldg_inner_reflection_split(
            r0,
            r1,
            w0,
            w1,
            Lk,
            x_r,
            r,
            field,
            dt,
            p)
        : SLDGInnerReflectionSplit{};
    const int maximum_refinement =
        p.vp_sldg_max_refinement + (reflection_split.straddles ? 4 : 0);
    if (!valid_polygon
        && refinement < maximum_refinement
        && available_child_work >= 2) {
        const bool split_radially = reflection_split.straddles
            ? reflection_split.split_radially
            : sldg_trace_patch_prefers_radial_split(
                trace_patch,
                r1 - r0,
                w1 - w0,
                refinement % 2 == 0);
        work_used += deposit_refined_children(split_radially);
        return work_used;
    }

    // The geometric depth/work budget is exhausted.  Carry four positive
    // Gauss masses to destination cells and report their entire source mass
    // as fallback so the global accuracy budget can accept or reject it.
    if (!valid_polygon) {
        constexpr double gauss =
            0.577350269189625764509148780501957456;
        const std::array<SLDGPoint, 4> arrivals =
            sldg_trace_subcell_gauss_points(
                r0,
                r1,
                w0,
                w1,
                Lk,
                x_r,
                r,
                field,
                dt,
                p,
                plan,
                true,
                diagnostics);
        const double hr = 0.5 * (r1 - r0);
        const double hw = 0.5 * (w1 - w0);
        int q = 0;
        double fallback_deposited = 0.0;
        for (double xi : {-gauss, gauss}) {
            for (double zeta : {-gauss, gauss}) {
                const double source_value =
                    polynomial.value(rc + hr * xi, wc + hw * zeta);
                if (!std::isfinite(source_value)
                    || (!allow_signed_source && !(source_value >= 0.0))) {
                    throw std::runtime_error(
                        "SLDG fallback quadrature encountered an invalid "
                        "reconstructed distribution.");
                }
                const double point_mass = 0.25 * source_area * source_value;
                const SLDGPoint& arrival =
                    arrivals[static_cast<std::size_t>(q++)];
                if (!std::isfinite(arrival.r) || !std::isfinite(arrival.w)) {
                    throw std::runtime_error(
                        "SLDG fallback characteristic is non-finite.");
                }
                const int id =
                    sldg_cell_for_coordinate(geometry.radial.edge, arrival.r);
                const int jd = sldg_cell_for_coordinate(
                    geometry.radial_velocity.edge, arrival.w);
                if (id >= 0 && jd >= 0) {
                    const std::size_t destination = idx(id, jd, k, p);
                    deposited_mass[destination] += point_mass;
                    if (deposited_modal_moments != nullptr) {
                        const long double xd =
                            (static_cast<long double>(arrival.r)
                             - geometry.radial.center[
                                 static_cast<std::size_t>(id)])
                            / geometry.radial.width[
                                static_cast<std::size_t>(id)];
                        const long double yd =
                            (static_cast<long double>(arrival.w)
                             - geometry.radial_velocity.center[
                                 static_cast<std::size_t>(jd)])
                            / geometry.radial_velocity.width[
                                static_cast<std::size_t>(jd)];
                        const std::array<long double, 5> basis{{
                            xd,
                            yd,
                            xd * xd - 1.0L / 12.0L,
                            xd * yd,
                            yd * yd - 1.0L / 12.0L,
                        }};
                        for (std::size_t mode = 0;
                             mode < basis.size();
                             ++mode) {
                            (*deposited_modal_moments)[destination][mode] +=
                                static_cast<long double>(point_mass)
                                * basis[mode];
                        }
                    }
                    fallback_deposited += point_mass;
                }
            }
        }
        diagnostics.source_mass += diagnostic_measure * source_mass;
        diagnostics.deposited_mass +=
            diagnostic_measure * fallback_deposited;
        diagnostics.escaped_mass +=
            diagnostic_measure * (source_mass - fallback_deposited);
        diagnostics.quadrature_fallback_mass +=
            diagnostic_measure * source_mass;
        if (source_mass > diagnostics.largest_fallback_mass) {
            diagnostics.largest_fallback_mass = source_mass;
            diagnostics.largest_fallback_r = rc;
            diagnostics.largest_fallback_w = wc;
            diagnostics.largest_fallback_L = Lk;
            diagnostics.largest_fallback_refinement = refinement;
            diagnostics.largest_fallback_work = work_budget;
        }
        ++diagnostics.quadrature_fallback_subcells;
        ++diagnostics.transported_subcells;
        return work_used;
    }

    diagnostics.maximum_area_defect =
        std::max(
            diagnostics.maximum_area_defect,
            std::max(
                assessment.area_defect,
                assessment.maximum_triangle_area_defect));
    diagnostics.maximum_map_defect =
        std::max(diagnostics.maximum_map_defect, assessment.map_defect);
    const double deposited =
        sldg_deposit_affine_patch_with_quadratic_reconstruction(
            assessment.affine_triangles,
            source_mass,
            polynomial,
            k,
            geometry,
            p,
            deposited_mass,
            deposited_modal_moments,
            allow_signed_source);
    diagnostics.source_mass += diagnostic_measure * source_mass;
    diagnostics.deposited_mass += diagnostic_measure * deposited;
    diagnostics.escaped_mass +=
        diagnostic_measure * (source_mass - deposited);
    ++diagnostics.transported_subcells;
    return work_used;
}

/// Applies one frozen-field conservative remap.
inline std::pair<std::vector<double>, SLDGStepDiagnostics>
advect_vlasov_poisson_conservative_sldg_frozen_with_polynomials(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    const std::vector<SLDGSourcePolynomial>* supplied_polynomials,
    const char* reconstruction_name,
    DelayedInnerCarrier* carrier = nullptr,
    std::vector<std::array<double, 5>>* transported_modal_coefficients = nullptr,
    bool allow_signed_source = false,
    const std::vector<SLDGSourcePolynomial>* eta_node_polynomials = nullptr) {
    if (!using_any_sldg_transport(p)
        || !using_cell_average_eta_grid(p)) {
        throw std::runtime_error(
            std::string(reconstruction_name)
            + " requires a cell-integrated SLDG and cell-average eta configuration.");
    }
    if (f.size() != static_cast<std::size_t>(p.Nr) * p.Nw * p.NL) {
        throw std::runtime_error(
            std::string(reconstruction_name) + " distribution size mismatch.");
    }
    if (supplied_polynomials != nullptr
        && supplied_polynomials->size() != f.size()) {
        throw std::runtime_error(
            std::string(reconstruction_name)
            + " supplied-polynomial array size mismatch.");
    }
    if (eta_node_polynomials != nullptr
        && eta_node_polynomials->size()
            != static_cast<std::size_t>(p.vp_sldg_eta_quadrature_points)
                * f.size()) {
        throw std::runtime_error(
            std::string(reconstruction_name)
            + " eta-node polynomial array size mismatch.");
    }
    if (x_r.size() != static_cast<std::size_t>(p.Nr)) {
        throw std::runtime_error(
            "Conservative SLDG radial interpolation-coordinate size mismatch.");
    }
    if (!field.radial_edge.empty()) {
        const std::size_t nr = static_cast<std::size_t>(p.Nr);
        if (field.radial_edge.size() != nr + 1
            || field.source_mass_edge.size() != nr + 1
            || field.source_line_density.size() != nr
            || field.phi_edge.size() != nr + 1) {
            throw std::runtime_error(
                "Conservative SLDG finite-volume Poisson field size mismatch.");
        }
        const double endpoint_scale = std::max({
            1.0,
            std::abs(field.radial_edge.front()),
            std::abs(field.radial_edge.back()),
            std::abs(geometry.radial.edge.front()),
            std::abs(geometry.radial.edge.back()),
        });
        const double endpoint_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() * endpoint_scale;
        if (std::abs(
                field.radial_edge.front() - geometry.radial.edge.front())
                > endpoint_tolerance
            || std::abs(
                field.radial_edge.back() - geometry.radial.edge.back())
                > endpoint_tolerance) {
            throw std::runtime_error(
                "Conservative SLDG Poisson and transport radial faces disagree.");
        }
    } else if (field.phi.size() != static_cast<std::size_t>(p.Nr)
               || field.phi_r.size() != static_cast<std::size_t>(p.Nr)) {
        throw std::runtime_error(
            "Conservative SLDG nodal Poisson field size mismatch.");
    }
    validate_cell_average_eta_views(
        geometry,
        r,
        w,
        L,
        geometry.radial_velocity.width,
        geometry.legacy_Lq,
        p,
        "Conservative SLDG");
    if (allow_signed_source) {
        for (double value : f) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Conservative SLDG signed input contains a non-finite value.");
            }
        }
    } else {
        validate_finite_nonnegative_distribution(
            f,
            static_cast<std::size_t>(p.Nr) * p.Nw * p.NL,
            "Conservative SLDG input");
    }
    if (using_delayed_inner_return(p) && carrier == nullptr) {
        throw std::runtime_error(
            "inner_boundary=delayed_return requires a persistent carrier state.");
    }
    if (!using_delayed_inner_return(p)
        && carrier != nullptr
        && !carrier->packets.empty()) {
        throw std::runtime_error(
            "A nonempty delayed-return carrier cannot be used with another inner boundary.");
    }
    // Check geometry and signed/nonnegative representation before tracing.
    // The eta quadrature samples angular momentum while the storage plane
    // retains its cell-integrated eta identity.
    const CharacteristicIntegratorPlan plan = make_characteristic_integrator_plan(p);
    const int subdivisions = std::max(1, p.vp_sldg_subcells);
    const auto& eta_rule = phase_space_gauss_legendre_rule(
        p.vp_sldg_eta_quadrature_points);
    const int transport_threads = effective_threads(p, p.NL);
    std::vector<double> deposited_mass(f.size(), 0.0);
    const bool transport_modal_coefficients =
        supplied_polynomials != nullptr
        && transported_modal_coefficients != nullptr;
    std::vector<std::array<long double, 5>> deposited_modal_moments(
        transport_modal_coefficients ? f.size() : 0);
    std::vector<SLDGStepDiagnostics> partial(
        static_cast<std::size_t>(transport_threads));
    std::vector<std::vector<DelayedInnerCarrierPacket>> carrier_by_thread(
        static_cast<std::size_t>(transport_threads));
    std::vector<std::vector<DelayedInnerCarrierPacket>> old_carrier_by_eta(
        static_cast<std::size_t>(p.NL));
    if (carrier != nullptr) {
        for (DelayedInnerCarrierPacket packet : carrier->packets) {
            validate_delayed_inner_carrier_packet(
                packet, p.NL, "SLDG delayed-return carrier input");
            old_carrier_by_eta[static_cast<std::size_t>(packet.eta_index)]
                .push_back(packet);
        }
        carrier->packets.clear();
    }

    // A characteristic keeps eta fixed.  Giving each eta plane to exactly one
    // worker therefore permits direct, race-free deposition of both cell mass
    // and P2 moments.  Dynamic scheduling balances the more expensive low-L
    // planes while avoiding one full destination array per worker.
    parallel_for_dynamic_chunks(
        0,
        p.NL,
        transport_threads,
        1,
        [&](int k_begin, int k_end, int lane) {
                SLDGStepDiagnostics& diagnostics =
                    partial[static_cast<std::size_t>(lane)];
                std::vector<DelayedInnerCarrierPacket>& lane_carrier =
                    carrier_by_thread[static_cast<std::size_t>(lane)];
                constexpr double phase_factor = 8.0 * PI * PI;
                for (int k = k_begin; k < k_end; ++k) {
                for (DelayedInnerCarrierPacket packet :
                     old_carrier_by_eta[static_cast<std::size_t>(k)]) {
                    const double diagnostic_measure = phase_factor
                        * geometry.eta.width[
                            static_cast<std::size_t>(k)];
                    const double physical_mass =
                        diagnostic_measure * packet.reduced_mass;
                    diagnostics.carrier_mass_before += physical_mass;
                    ++diagnostics.carrier_packets_before;
                    // Hidden residence consumes physical time.  A packet
                    // that returns within this interval moves outside only
                    // for the time remaining after its return.
                    if (dt == 0.0 || packet.remaining_time > dt) {
                        packet.remaining_time = std::max(
                            0.0, packet.remaining_time - dt);
                        lane_carrier.push_back(packet);
                        continue;
                    }

                    const double visible_interval = std::max(
                        0.0, dt - packet.remaining_time);
                    const double Lk = std::sqrt(2.0 * packet.eta);
                    const SLDGPoint arrival = sldg_trace_arrival_counted(
                        p.r_min,
                        packet.boundary_speed,
                        Lk,
                        x_r,
                        r,
                        field,
                        visible_interval,
                        p,
                        plan,
                        true,
                        diagnostics);
                    diagnostics.carrier_released_mass += physical_mass;
                    ++diagnostics.carrier_packets_released;
                    if (arrival.hidden) {
                        lane_carrier.push_back({
                            packet.reduced_mass,
                            packet.eta_index,
                            packet.eta,
                            arrival.boundary_speed,
                            arrival.remaining_hidden_time,
                        });
                        diagnostics.carrier_captured_mass += physical_mass;
                        ++diagnostics.carrier_packets_captured;
                    } else if (sldg_deposit_point_cloud(
                                   arrival,
                                   packet.eta_index,
                                   packet.reduced_mass,
                                   geometry,
                                   p,
                                   deposited_mass,
                                   transport_modal_coefficients
                                       ? &deposited_modal_moments
                                       : nullptr)) {
                        diagnostics.deposited_mass += physical_mass;
                        diagnostics.carrier_release_deposited_mass +=
                            physical_mass;
                    } else {
                        diagnostics.escaped_mass += physical_mass;
                    }
                }
                const double diagnostic_measure = phase_factor
                    * geometry.eta.width[static_cast<std::size_t>(k)];
                const int plane_cells = p.Nr * p.Nw;
                std::vector<SLDGSourcePolynomial> plane_polynomial(
                    static_cast<std::size_t>(plane_cells));
                std::vector<unsigned char> plane_active(
                    static_cast<std::size_t>(plane_cells), 0);
                if (eta_node_polynomials == nullptr) {
                for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                        const std::size_t cell = idx(i, j, k, p);
                        const std::size_t plane_cell = static_cast<std::size_t>(
                            i * p.Nw + j);
                        SLDGSourcePolynomial polynomial;
                        if (supplied_polynomials != nullptr) {
                            polynomial = (*supplied_polynomials)[cell];
                        } else {
                            const std::array<double, 2> slope =
                                sldg_limited_slopes(f, i, j, k, geometry, p, field.background.H);
                            polynomial.cell_average = f[cell];
                            polynomial.center_r = geometry.radial.center[
                                static_cast<std::size_t>(i)];
                            polynomial.center_w =
                                geometry.radial_velocity.center[
                                    static_cast<std::size_t>(j)];
                            polynomial.scale_r = geometry.radial.width[
                                static_cast<std::size_t>(i)];
                            polynomial.scale_w =
                                geometry.radial_velocity.width[
                                    static_cast<std::size_t>(j)];
                            polynomial.mode[0] =
                                slope[0] * polynomial.scale_r;
                            polynomial.mode[1] =
                                slope[1] * polynomial.scale_w;
                        }
                        bool nonzero = polynomial.cell_average != 0.0;
                        for (double coefficient : polynomial.mode) {
                            nonzero = nonzero || coefficient != 0.0;
                        }
                        plane_polynomial[plane_cell] = polynomial;
                        plane_active[plane_cell] = nonzero ? 1 : 0;
                    }
                }
                }

                // Neighboring source subcells share trace points.  Clear
                // their validity flags at each actual eta quadrature node,
                // whose orbit map differs through the centrifugal force.
                const int trace_lattice_r_side =
                    2 * subdivisions * p.Nr + 1;
                const int trace_lattice_w_side =
                    2 * subdivisions * p.Nw + 1;
                std::vector<SLDGPoint> trace_lattice(
                    static_cast<std::size_t>(trace_lattice_r_side)
                    * static_cast<std::size_t>(trace_lattice_w_side));
                std::vector<unsigned char> trace_ready(
                    trace_lattice.size(), 0);
                const double eta_center = geometry.eta.center[
                    static_cast<std::size_t>(k)];
                const double eta_half_width = 0.5 * geometry.eta.width[
                    static_cast<std::size_t>(k)];
                for (std::size_t q_eta = 0;
                     q_eta < eta_rule.first.size();
                     ++q_eta) {
                    std::fill(trace_ready.begin(), trace_ready.end(), 0);
                    const double eta = eta_center
                        + eta_half_width * eta_rule.first[q_eta];
                    const double eta_average_weight =
                        0.5 * eta_rule.second[q_eta];
                    const double Lk = std::sqrt(2.0 * eta);
                    for (int i = 0; i < p.Nr; ++i) {
                    for (int j = 0; j < p.Nw; ++j) {
                    const std::size_t plane_cell = static_cast<std::size_t>(
                        i * p.Nw + j);
                    const std::size_t cell = idx(i, j, k, p);
                    SLDGSourcePolynomial polynomial =
                        eta_node_polynomials != nullptr
                        ? (*eta_node_polynomials)[q_eta * f.size() + cell]
                        : plane_polynomial[plane_cell];
                    bool nonzero = polynomial.cell_average != 0.0;
                    for (double coefficient : polynomial.mode) {
                        nonzero = nonzero || coefficient != 0.0;
                    }
                    if (!nonzero) {
                        continue;
                    }
                    const double fq = f[cell];
                    const double source_r0 =
                        geometry.radial.edge[static_cast<std::size_t>(i)];
                    const double source_r1 =
                        geometry.radial.edge[static_cast<std::size_t>(i + 1)];
                    const double dr_sub =
                        (source_r1 - source_r0)
                        / static_cast<double>(subdivisions);
                    const double source_w0 =
                        geometry.radial_velocity.edge[static_cast<std::size_t>(j)];
                    const double source_w1 =
                        geometry.radial_velocity.edge[static_cast<std::size_t>(j + 1)];
                    const double dw_sub =
                        (source_w1 - source_w0)
                        / static_cast<double>(subdivisions);
                    polynomial.cell_average *= eta_average_weight;
                    for (double& coefficient : polynomial.mode) {
                        coefficient *= eta_average_weight;
                    }
                    auto trace_lattice_point = [&](int ir, int iw) {
                        const int global_ir =
                            2 * subdivisions * i + ir;
                        const int global_iw =
                            2 * subdivisions * j + iw;
                        const std::size_t q =
                            static_cast<std::size_t>(global_ir)
                                * static_cast<std::size_t>(
                                    trace_lattice_w_side)
                            + static_cast<std::size_t>(global_iw);
                        if (trace_ready[q]) {
                            ++diagnostics.characteristic_trace_requests;
                            ++diagnostics.reused_characteristic_points;
                            return trace_lattice[q];
                        }
                        const double source_r = ir == 0
                            ? source_r0
                            : (ir == 2 * subdivisions
                                ? source_r1
                                : source_r0
                                    + 0.5 * static_cast<double>(ir) * dr_sub);
                        const double source_w = iw == 0
                            ? source_w0
                            : (iw == 2 * subdivisions
                                ? source_w1
                                : source_w0
                                    + 0.5 * static_cast<double>(iw) * dw_sub);
                        trace_lattice[q] = sldg_trace_arrival_counted(
                            source_r,
                            source_w,
                            Lk,
                            x_r,
                            r,
                            field,
                            dt,
                            p,
                            plan,
                            true,
                            diagnostics);
                        trace_ready[q] = 1;
                        return trace_lattice[q];
                    };
                    for (int sr = 0; sr < subdivisions; ++sr) {
                        const double r0 =
                            source_r0 + static_cast<double>(sr) * dr_sub;
                        const double r1 = r0 + dr_sub;
                        for (int sw = 0; sw < subdivisions; ++sw) {
                            const double w0 =
                                source_w0 + static_cast<double>(sw) * dw_sub;
                            const double w1 = w0 + dw_sub;
                            const double reconstructed_mass =
                                polynomial.rectangle_integral(r0, r1, w0, w1);
                            const double subcell_area =
                                (r1 - r0) * (w1 - w0);
                            const double reconstructed =
                                reconstructed_mass / subcell_area;
                            if (!std::isfinite(reconstructed_mass)
                                || (!allow_signed_source
                                    && !(reconstructed_mass >= 0.0))) {
                                std::ostringstream message;
                                message
                                    << "SLDG positivity limiter produced an "
                                    << "invalid source subcell average: i="
                                    << i << ", j=" << j << ", k=" << k
                                    << ", sr=" << sr << ", sw=" << sw
                                    << ", f_center=" << std::setprecision(17)
                                    << fq
                                    << ", r0=" << r0 << ", r1=" << r1
                                    << ", w0=" << w0 << ", w1=" << w1
                                    << ", reconstructed=" << reconstructed
                                    << ", minimum="
                                    << polynomial.minimum_on_parent_cell()
                                    << ", modes=(" << polynomial.mode[0]
                                    << ',' << polynomial.mode[1]
                                    << ',' << polynomial.mode[2]
                                    << ',' << polynomial.mode[3]
                                    << ',' << polynomial.mode[4] << ')'
                                    << '.';
                                throw std::runtime_error(message.str());
                            }
                            if ((!allow_signed_source
                                    && !(reconstructed > 0.0))
                                || (allow_signed_source
                                    && reconstructed_mass == 0.0)) {
                                continue;
                            }
                            SLDGTracePatch base_trace_patch;
                            const SLDGTracePatch* base_trace_patch_ptr = nullptr;
                            if (dt != 0.0) {
                                for (int pr = 0; pr < 3; ++pr) {
                                    for (int pw = 0; pw < 3; ++pw) {
                                        base_trace_patch.at(pr, pw) =
                                            trace_lattice_point(
                                                2 * sr + pr,
                                                2 * sw + pw);
                                    }
                                }
                                base_trace_patch_ptr = &base_trace_patch;
                            }
                            const long long work_counter =
                                sldg_deposit_source_rectangle(
                                    r0,
                                    r1,
                                    w0,
                                    w1,
                                    polynomial,
                                    Lk,
                                    k,
                                    x_r,
                                    r,
                                    field,
                                    geometry,
                                    dt,
                                    p,
                                    plan,
                                    diagnostic_measure,
                                    base_trace_patch_ptr,
                                    nullptr,
                                    0,
                                    p.vp_sldg_max_work_per_subcell,
                                    deposited_mass,
                                    transport_modal_coefficients
                                        ? &deposited_modal_moments
                                        : nullptr,
                                    lane_carrier,
                                    diagnostics,
                                    allow_signed_source);
                            diagnostics.maximum_source_work = std::max(
                                diagnostics.maximum_source_work,
                                work_counter);
                        }
                    }
                }
            }
            }
            }
        });

    std::vector<double> result(f.size(), 0.0);
    std::vector<std::array<double, 5>> result_modal_coefficients(
        transport_modal_coefficients ? f.size() : 0);
    const int result_threads = effective_threads(
        p, static_cast<int>(f.size()));
    std::vector<long long> projected_limited_by_thread(
        static_cast<std::size_t>(result_threads), 0);
    parallel_for_chunks(
        0,
        static_cast<int>(f.size()),
        result_threads,
        [&](int lo, int hi, int tid) {
            long long projected_limited = 0;
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int remainder = q - i * p.Nw * p.NL;
                const int j = remainder / p.NL;
                const long double area =
                    geometry.radial.width[static_cast<std::size_t>(i)]
                    * geometry.radial_velocity.width[static_cast<std::size_t>(j)];
                result[static_cast<std::size_t>(q)] =
                    static_cast<double>(
                        deposited_mass[static_cast<std::size_t>(q)] / area);
                if (transport_modal_coefficients) {
                    SLDGSourcePolynomial destination;
                    destination.cell_average = result[static_cast<std::size_t>(q)];
                    destination.center_r = geometry.radial.center[
                        static_cast<std::size_t>(i)];
                    destination.center_w = geometry.radial_velocity.center[
                        static_cast<std::size_t>(j)];
                    destination.scale_r = geometry.radial.width[
                        static_cast<std::size_t>(i)];
                    destination.scale_w = geometry.radial_velocity.width[
                        static_cast<std::size_t>(j)];
                    // Divide modal integrals by the orthogonal basis norms
                    // to recover destination coefficients with the same mean.
                    constexpr std::array<long double, 5> inverse_norm{{
                        12.0L, 12.0L, 180.0L, 144.0L, 180.0L}};
                    for (std::size_t mode = 0; mode < 5; ++mode) {
                        destination.mode[mode] = static_cast<double>(
                            inverse_norm[mode]
                            * deposited_modal_moments[
                                static_cast<std::size_t>(q)][mode]
                            / area);
                    }
                    if (!allow_signed_source) {
                        if (destination.enforce_positivity() < 1.0) {
                            ++projected_limited;
                        }
                    }
                    result_modal_coefficients[static_cast<std::size_t>(q)] =
                        destination.mode;
                }
            }
            projected_limited_by_thread[static_cast<std::size_t>(tid)] =
                projected_limited;
        });

    SLDGStepDiagnostics diagnostics;
    for (const SLDGStepDiagnostics& value : partial) {
        diagnostics.source_mass += value.source_mass;
        diagnostics.deposited_mass += value.deposited_mass;
        diagnostics.escaped_mass += value.escaped_mass;
        diagnostics.carrier_mass_before += value.carrier_mass_before;
        diagnostics.carrier_captured_mass +=
            value.carrier_captured_mass;
        diagnostics.carrier_released_mass +=
            value.carrier_released_mass;
        diagnostics.carrier_release_deposited_mass +=
            value.carrier_release_deposited_mass;
        diagnostics.quadrature_fallback_mass +=
            value.quadrature_fallback_mass;
        diagnostics.maximum_area_defect =
            std::max(diagnostics.maximum_area_defect, value.maximum_area_defect);
        diagnostics.maximum_map_defect =
            std::max(diagnostics.maximum_map_defect, value.maximum_map_defect);
        diagnostics.maximum_rejected_area_defect = std::max(
            diagnostics.maximum_rejected_area_defect,
            value.maximum_rejected_area_defect);
        diagnostics.maximum_rejected_map_defect = std::max(
            diagnostics.maximum_rejected_map_defect,
            value.maximum_rejected_map_defect);
        if (value.largest_fallback_mass > diagnostics.largest_fallback_mass) {
            diagnostics.largest_fallback_mass = value.largest_fallback_mass;
            diagnostics.largest_fallback_r = value.largest_fallback_r;
            diagnostics.largest_fallback_w = value.largest_fallback_w;
            diagnostics.largest_fallback_L = value.largest_fallback_L;
            diagnostics.largest_fallback_refinement =
                value.largest_fallback_refinement;
            diagnostics.largest_fallback_work =
                value.largest_fallback_work;
        }
        diagnostics.transported_subcells += value.transported_subcells;
        diagnostics.traced_rectangles += value.traced_rectangles;
        diagnostics.rejected_rectangles += value.rejected_rectangles;
        diagnostics.quadrature_fallback_subcells +=
            value.quadrature_fallback_subcells;
        diagnostics.maximum_source_work = std::max(
            diagnostics.maximum_source_work,
            value.maximum_source_work);
        diagnostics.characteristic_trace_requests +=
            value.characteristic_trace_requests;
        diagnostics.characteristic_traces += value.characteristic_traces;
        diagnostics.reused_characteristic_points +=
            value.reused_characteristic_points;
        diagnostics.carrier_packets_before +=
            value.carrier_packets_before;
        diagnostics.carrier_packets_captured +=
            value.carrier_packets_captured;
        diagnostics.carrier_packets_released +=
            value.carrier_packets_released;
    }
    for (long long count : projected_limited_by_thread) {
        diagnostics.hamiltonian_projected_limited_cells += count;
    }
    if (carrier != nullptr) {
        std::size_t packet_count = 0;
        for (const auto& packets : carrier_by_thread) {
            packet_count += packets.size();
        }
        if (packet_count > static_cast<std::size_t>(
                p.vp_inner_carrier_max_packets)) {
            throw std::runtime_error(
                "Delayed-return carrier exceeded vp_inner_carrier_max_packets; "
                "increase the limit only after checking boundary quadrature convergence.");
        }
        carrier->packets.reserve(packet_count);
        for (auto& packets : carrier_by_thread) {
            carrier->packets.insert(
                carrier->packets.end(),
                std::make_move_iterator(packets.begin()),
                std::make_move_iterator(packets.end()));
        }
        const DelayedInnerCarrierLedger after =
            evaluate_delayed_inner_carrier_ledger(
                *carrier, geometry, p);
        diagnostics.carrier_mass_after = after.physical_mass;
        diagnostics.carrier_packets_after =
            static_cast<long long>(after.packet_count);
    }
    // Close one mass budget across visible cells, persistent hidden packets
    // and escaped material before publishing the completed transport result.
    diagnostics.carrier_mass_residual =
        diagnostics.source_mass + diagnostics.carrier_mass_before
        - diagnostics.deposited_mass - diagnostics.carrier_mass_after
        - diagnostics.escaped_mass;
    const double carrier_budget_scale = std::max({
        1.0,
        diagnostics.source_mass + diagnostics.carrier_mass_before,
        diagnostics.deposited_mass + diagnostics.carrier_mass_after
            + diagnostics.escaped_mass,
    });
    if (std::abs(diagnostics.carrier_mass_residual)
        > p.vp_inner_carrier_mass_tolerance * carrier_budget_scale) {
        std::ostringstream message;
        message << "Delayed-return carrier mass budget failed: residual="
                << diagnostics.carrier_mass_residual
                << ", source=" << diagnostics.source_mass
                << ", hidden_before=" << diagnostics.carrier_mass_before
                << ", visible_after=" << diagnostics.deposited_mass
                << ", hidden_after=" << diagnostics.carrier_mass_after
                << ", escaped=" << diagnostics.escaped_mass;
        throw std::runtime_error(message.str());
    }
    // For a nonnegative DF this is a physical mass fraction.  The signed
    // residual path has separate caller-level reconstruction/error controls.
    const double fallback_fraction =
        (diagnostics.source_mass > 0.0)
        ? diagnostics.quadrature_fallback_mass / diagnostics.source_mass
        : 0.0;
    if (!allow_signed_source
        && fallback_fraction > p.vp_sldg_fallback_mass_tolerance) {
        std::ostringstream message;
        message
            << "SLDG non-affine quadrature fallback carried mass fraction "
            << fallback_fraction
            << ", exceeding vp_sldg_fallback_mass_tolerance="
            << p.vp_sldg_fallback_mass_tolerance
            << "; traced_rectangles=" << diagnostics.traced_rectangles
            << ", rejected_rectangles=" << diagnostics.rejected_rectangles
            << ", max_rejected_area_defect="
            << diagnostics.maximum_rejected_area_defect
            << ", max_rejected_map_defect="
            << diagnostics.maximum_rejected_map_defect
            << ", largest_fallback=(mass=" << diagnostics.largest_fallback_mass
            << ", r=" << diagnostics.largest_fallback_r
            << ", w=" << diagnostics.largest_fallback_w
            << ", L=" << diagnostics.largest_fallback_L
            << ", refinement=" << diagnostics.largest_fallback_refinement
            << ", work=" << diagnostics.largest_fallback_work << ')'
            << ". Increase (r,w,eta) resolution or reduce the transport interval.";
        throw std::runtime_error(message.str());
    }
    if (transported_modal_coefficients != nullptr) {
        if (transport_modal_coefficients) {
            *transported_modal_coefficients =
                std::move(result_modal_coefficients);
        } else {
            transported_modal_coefficients->clear();
        }
    }
    return {std::move(result), diagnostics};
}

/// Applies the grid-aligned limited linear reconstruction through the conservative map.
inline std::pair<std::vector<double>, SLDGStepDiagnostics>
advect_vlasov_poisson_conservative_sldg_frozen(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    DelayedInnerCarrier* carrier = nullptr) {
    if (!using_conservative_sldg_transport(p)) {
        throw std::runtime_error(
            "The grid-aligned conservative SLDG entry point requires "
            "vp_interpolation=conservative_sldg.");
    }
    return advect_vlasov_poisson_conservative_sldg_frozen_with_polynomials(
        f,
        x_r,
        r,
        w,
        L,
        field,
        geometry,
        dt,
        p,
        nullptr,
        "Conservative SLDG",
        carrier);
}

struct SLDGMidpointPrediction {
    Moments moments;
    double mass_before = 0.0;
    double mass_after = 0.0;
    double escaped_mass = 0.0;
    double hidden_mass_before = 0.0;
    double hidden_mass_after = 0.0;
    double hidden_mass_change = 0.0;
    long long limited_faces = 0;
};

/// Predicts the midpoint density from the spherical continuity equation.
///
/// Gravity at the middle of a Vlasov step only needs the density at that time.
/// The cell-integrated shell masses therefore obey
///
///   d M_i / dt = -(F_{i+1/2} - F_{i-1/2}),
///   F = 4 pi r^2 j_r.
///
/// A shared face flux gives an exactly conservative radial update.  If a large
/// trial interval would remove more mass than one donor cell contains, all
/// outgoing faces of that donor receive the same scale factor.  This preserves
/// nonnegative predicted density without changing the final distribution
/// remap.
inline SLDGMidpointPrediction predict_vlasov_poisson_midpoint_moments(
    const Moments& initial,
    const PhaseSpaceCellGeometry& geometry,
    double half_dt,
    const Params& p,
    double hidden_mass_before = 0.0) {
    if (initial.rho.size() != static_cast<std::size_t>(p.Nr)
        || initial.j.size() != static_cast<std::size_t>(p.Nr)) {
        throw std::runtime_error(
            "SLDG midpoint predictor received incompatible radial moments.");
    }
    if (!(half_dt >= 0.0) || !std::isfinite(half_dt)) {
        throw std::runtime_error(
            "SLDG midpoint predictor requires a finite nonnegative half-step.");
    }

    SLDGMidpointPrediction prediction;
    prediction.moments = initial;
    prediction.hidden_mass_before = hidden_mass_before;
    prediction.hidden_mass_after = hidden_mass_before;
    std::vector<double> shell_mass(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> face_flux(static_cast<std::size_t>(p.Nr + 1), 0.0);
    std::vector<double> outgoing_rate(static_cast<std::size_t>(p.Nr), 0.0);
    std::vector<double> donor_scale(static_cast<std::size_t>(p.Nr), 1.0);

    for (int i = 0; i < p.Nr; ++i) {
        const double ri = geometry.radial.center[static_cast<std::size_t>(i)];
        const double width =
            geometry.radial.width[static_cast<std::size_t>(i)];
        const double mass =
            4.0 * PI * ri * ri
            * initial.rho[static_cast<std::size_t>(i)] * width;
        if (!(mass >= 0.0) || !std::isfinite(mass)) {
            throw std::runtime_error(
                "SLDG midpoint predictor found invalid shell mass.");
        }
        shell_mass[static_cast<std::size_t>(i)] = mass;
        prediction.mass_before += mass;
    }

    for (int face = 1; face < p.Nr; ++face) {
        const double rf =
            geometry.radial.edge[static_cast<std::size_t>(face)];
        const double left_r =
            geometry.radial.center[static_cast<std::size_t>(face - 1)];
        const double right_r =
            geometry.radial.center[static_cast<std::size_t>(face)];
        const double fraction =
            std::clamp((rf - left_r) / (right_r - left_r), 0.0, 1.0);
        const double current =
            (1.0 - fraction)
                * initial.j[static_cast<std::size_t>(face - 1)]
            + fraction * initial.j[static_cast<std::size_t>(face)];
        face_flux[static_cast<std::size_t>(face)] =
            4.0 * PI * rf * rf * current;
    }

    if (using_delayed_inner_return(p)) {
        const double rf = geometry.radial.edge.front();
        face_flux.front() =
            4.0 * PI * rf * rf * initial.j.front();
        if (half_dt > 0.0 && face_flux.front() > 0.0) {
            face_flux.front() = std::min(
                face_flux.front(), hidden_mass_before / half_dt);
        }
    } else if (p.inner_boundary != "reflecting") {
        const double rf = geometry.radial.edge.front();
        face_flux.front() = std::min(
            0.0, 4.0 * PI * rf * rf * initial.j.front());
    }
    if (p.outer_boundary != "reflecting") {
        const double rf = geometry.radial.edge.back();
        face_flux.back() = std::max(
            0.0, 4.0 * PI * rf * rf * initial.j.back());
    }

    for (int face = 0; face <= p.Nr; ++face) {
        const double flux = face_flux[static_cast<std::size_t>(face)];
        const int donor = flux > 0.0
            ? face - 1
            : (flux < 0.0 ? face : -1);
        if (donor >= 0 && donor < p.Nr) {
            outgoing_rate[static_cast<std::size_t>(donor)] += std::abs(flux);
        }
    }
    if (half_dt > 0.0) {
        for (int i = 0; i < p.Nr; ++i) {
            const double requested =
                half_dt * outgoing_rate[static_cast<std::size_t>(i)];
            if (requested > shell_mass[static_cast<std::size_t>(i)]
                && requested > 0.0) {
                donor_scale[static_cast<std::size_t>(i)] =
                    shell_mass[static_cast<std::size_t>(i)] / requested;
            }
        }
        for (int face = 0; face <= p.Nr; ++face) {
            double& flux = face_flux[static_cast<std::size_t>(face)];
            const int donor = flux > 0.0
                ? face - 1
                : (flux < 0.0 ? face : -1);
            if (donor >= 0 && donor < p.Nr) {
                const double scale =
                    donor_scale[static_cast<std::size_t>(donor)];
                if (scale < 1.0) {
                    flux *= scale;
                    ++prediction.limited_faces;
                }
            }
        }
    }

    for (int i = 0; i < p.Nr; ++i) {
        const double updated_mass =
            shell_mass[static_cast<std::size_t>(i)]
            - half_dt * (
                face_flux[static_cast<std::size_t>(i + 1)]
                - face_flux[static_cast<std::size_t>(i)]);
        const double nonnegative_mass = std::max(0.0, updated_mass);
        const double ri = geometry.radial.center[static_cast<std::size_t>(i)];
        const double width =
            geometry.radial.width[static_cast<std::size_t>(i)];
        prediction.moments.rho[static_cast<std::size_t>(i)] =
            nonnegative_mass / (4.0 * PI * ri * ri * width);
        prediction.mass_after += nonnegative_mass;
    }
    if (using_delayed_inner_return(p)) {
        prediction.hidden_mass_change =
            -half_dt * face_flux.front();
        prediction.hidden_mass_after = std::max(
            0.0, hidden_mass_before + prediction.hidden_mass_change);
        prediction.escaped_mass = half_dt * face_flux.back();
    } else {
        prediction.escaped_mass = half_dt * (
            face_flux.back() - face_flux.front());
    }
    const double closure =
        prediction.mass_before + prediction.hidden_mass_before
        - prediction.mass_after - prediction.hidden_mass_after
        - prediction.escaped_mass;
    const double tolerance =
        256.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, prediction.mass_before);
    if (std::abs(closure) > tolerance) {
        throw std::runtime_error(
            "SLDG midpoint continuity predictor failed its mass budget.");
    }
    return prediction;
}

// Sum extensive budgets and counts across ranks; take maxima for local
// geometric/limiter defects and work bounds.
inline void sldg_reduce_step_diagnostics(
    SLDGStepDiagnostics& diagnostics,
    const MpiContext& mpi) {
    diagnostics.source_mass =
        mpi_allreduce_sum(diagnostics.source_mass, mpi);
    diagnostics.deposited_mass =
        mpi_allreduce_sum(diagnostics.deposited_mass, mpi);
    diagnostics.escaped_mass =
        mpi_allreduce_sum(diagnostics.escaped_mass, mpi);
    diagnostics.carrier_mass_before =
        mpi_allreduce_sum(diagnostics.carrier_mass_before, mpi);
    diagnostics.carrier_mass_after =
        mpi_allreduce_sum(diagnostics.carrier_mass_after, mpi);
    diagnostics.carrier_captured_mass =
        mpi_allreduce_sum(diagnostics.carrier_captured_mass, mpi);
    diagnostics.carrier_released_mass =
        mpi_allreduce_sum(diagnostics.carrier_released_mass, mpi);
    diagnostics.carrier_release_deposited_mass =
        mpi_allreduce_sum(
            diagnostics.carrier_release_deposited_mass, mpi);
    diagnostics.carrier_mass_residual =
        mpi_allreduce_sum(diagnostics.carrier_mass_residual, mpi);
    diagnostics.quadrature_fallback_mass =
        mpi_allreduce_sum(diagnostics.quadrature_fallback_mass, mpi);
    diagnostics.maximum_area_defect =
        mpi_allreduce_max(diagnostics.maximum_area_defect, mpi);
    diagnostics.maximum_map_defect =
        mpi_allreduce_max(diagnostics.maximum_map_defect, mpi);
    diagnostics.maximum_rejected_area_defect =
        mpi_allreduce_max(
            diagnostics.maximum_rejected_area_defect, mpi);
    diagnostics.maximum_rejected_map_defect =
        mpi_allreduce_max(
            diagnostics.maximum_rejected_map_defect, mpi);
    diagnostics.transported_subcells =
        mpi_allreduce_sum_ll(diagnostics.transported_subcells, mpi);
    diagnostics.traced_rectangles =
        mpi_allreduce_sum_ll(diagnostics.traced_rectangles, mpi);
    diagnostics.rejected_rectangles =
        mpi_allreduce_sum_ll(diagnostics.rejected_rectangles, mpi);
    diagnostics.quadrature_fallback_subcells =
        mpi_allreduce_sum_ll(
            diagnostics.quadrature_fallback_subcells, mpi);
    diagnostics.maximum_source_work =
        mpi_allreduce_max_ll(diagnostics.maximum_source_work, mpi);
    diagnostics.characteristic_trace_requests =
        mpi_allreduce_sum_ll(
            diagnostics.characteristic_trace_requests, mpi);
    diagnostics.characteristic_traces =
        mpi_allreduce_sum_ll(diagnostics.characteristic_traces, mpi);
    diagnostics.reused_characteristic_points =
        mpi_allreduce_sum_ll(
            diagnostics.reused_characteristic_points, mpi);
    diagnostics.carrier_packets_before =
        mpi_allreduce_sum_ll(diagnostics.carrier_packets_before, mpi);
    diagnostics.carrier_packets_after =
        mpi_allreduce_sum_ll(diagnostics.carrier_packets_after, mpi);
    diagnostics.carrier_packets_captured =
        mpi_allreduce_sum_ll(diagnostics.carrier_packets_captured, mpi);
    diagnostics.carrier_packets_released =
        mpi_allreduce_sum_ll(diagnostics.carrier_packets_released, mpi);
    diagnostics.hamiltonian_reconstructed_cells =
        mpi_allreduce_sum_ll(
            diagnostics.hamiltonian_reconstructed_cells, mpi);
    diagnostics.hamiltonian_limited_cells =
        mpi_allreduce_sum_ll(
            diagnostics.hamiltonian_limited_cells, mpi);
    diagnostics.hamiltonian_projected_limited_cells =
        mpi_allreduce_sum_ll(
            diagnostics.hamiltonian_projected_limited_cells, mpi);
    diagnostics.hamiltonian_rank_deficient_cells =
        mpi_allreduce_sum_ll(
            diagnostics.hamiltonian_rank_deficient_cells, mpi);
    diagnostics.hamiltonian_phase_degenerate_cells =
        mpi_allreduce_sum_ll(
            diagnostics.hamiltonian_phase_degenerate_cells, mpi);
    diagnostics.hamiltonian_eta_limited_cells =
        mpi_allreduce_sum_ll(
            diagnostics.hamiltonian_eta_limited_cells, mpi);
    diagnostics.hamiltonian_maximum_phase_gradient_fraction =
        mpi_allreduce_max(
            diagnostics.hamiltonian_maximum_phase_gradient_fraction, mpi);
}

// Assemble the transported distribution, field used for its map and the
// independently conservative midpoint predictor diagnostics.
inline SLDGStepResult finish_vlasov_poisson_sldg_step(
    std::pair<std::vector<double>, SLDGStepDiagnostics> final,
    VlasovPoissonField midpoint_field,
    const SLDGMidpointPrediction& predictor,
    const MpiContext& mpi) {
    SLDGStepResult result;
    result.f = std::move(final.first);
    result.midpoint_field = std::move(midpoint_field);
    result.diagnostics = final.second;
    result.diagnostics.midpoint_predictor_mass_before =
        predictor.mass_before;
    result.diagnostics.midpoint_predictor_mass_after =
        predictor.mass_after;
    result.diagnostics.midpoint_predictor_escaped_mass =
        predictor.escaped_mass;
    result.diagnostics.midpoint_predictor_hidden_mass_before =
        predictor.hidden_mass_before;
    result.diagnostics.midpoint_predictor_hidden_mass_after =
        predictor.hidden_mass_after;
    result.diagnostics.midpoint_predictor_hidden_mass_change =
        predictor.hidden_mass_change;
    result.diagnostics.midpoint_predictor_limited_faces =
        predictor.limited_faces;
    sldg_reduce_step_diagnostics(result.diagnostics, mpi);
    return result;
}

/// Uses a conservative continuity predictor to time-center self-gravity.
inline SLDGStepResult advect_vlasov_poisson_conservative_sldg(
    const std::vector<double>& f,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const VlasovPoissonField& initial_field,
    const Moments& initial_moments,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    double t,
    const Params& p,
    const MpiContext& mpi,
    DelayedInnerCarrier* carrier = nullptr) {
    (void)initial_field;
    const double local_hidden_mass = carrier != nullptr
        ? evaluate_delayed_inner_carrier_ledger(
            *carrier, geometry, p).physical_mass
        : 0.0;
    const double hidden_mass =
        mpi_allreduce_sum(local_hidden_mass, mpi);
    // Predict only the density needed for gravity, solve Poisson at t+dt/2,
    // then transport the original distribution for the full dt in that field.
    const SLDGMidpointPrediction predictor =
        predict_vlasov_poisson_midpoint_moments(
            initial_moments, geometry, 0.5 * dt, p, hidden_mass);
    VlasovPoissonField midpoint_field =
        solve_vlasov_poisson_field(
            predictor.moments,
            r,
            t + 0.5 * dt,
            p,
            predictor.hidden_mass_after);
    auto final = advect_vlasov_poisson_conservative_sldg_frozen(
        f, x_r, r, w, L, midpoint_field, geometry, dt, p, carrier);
    return finish_vlasov_poisson_sldg_step(
        std::move(final),
        std::move(midpoint_field),
        predictor,
        mpi);
}

}  // namespace ev
