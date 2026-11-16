#pragma once

// Weighted particle-in-cell Einstein--Vlasov evolution in uncharged,
// spherically symmetric ingoing Eddington--Finkelstein coordinates.
//
// Metric and marker conventions follow Appendix A/B of W. East,
// arXiv:2511.20567, after setting every Maxwell/charge term to zero:
//
//   ds^2 = -a b^2 dv^2 + 2 b dv dr + r^2 dOmega^2,
//   a = 1 - 2m/r,                         b(infinity) = 1,
//   d_r log b = 4 pi r T_rr,
//   b^{-1} d_r(mb) = 2 pi r^2 (T_rr + S).
//
// A marker stores invariant represented rest weight dm and the *specific*
// positive canonical covariant momentum p_r.  This is deliberately a distinct
// type from WeightedPICMarker: WeightedPICMarker::w remains an orthonormal
// Schwarzschild/polar-areal momentum and is never reinterpreted as p_r here.
// With chi=1+2 eta/r^2 and eta=L^2/2, an active marker deposits
//
//   T_rr = sum dm p_r / dV,       S = sum dm chi/(p_r dV),
// where dV=4*pi*integral(r^2 dr) is the coordinate shell volume.
//
// The characteristic equations contain no division by a, so they remain
// finite at an apparent horizon a=0. Post-horizon particles are retained down
// to a configurable fraction (default 0.1) of the apparent-horizon radius;
// excision also requires a<-0.01 and a trapped four-cell outward buffer.
// Excised markers are retained as marker
// records and in mass diagnostics; a fixed ADM boundary condition carries
// their gravitational mass in the inward mb solve.
// The pusher advances log(p_r), not p_r, so the future-directed p_r>0 branch
// is preserved at every Runge--Kutta stage.  Direct infalling Schwarzschild
// data use w<0.  Importing East-style outgoing data instead requires the
// explicit time reversal b_in=-b_out and (p_r)_in=-(p_r)_out.

#include "ev/weighted_vlasov_pic.hpp"

#include <array>
#include <optional>
#include <thread>

namespace ev {
namespace eddington_finkelstein {

enum class MarkerState : std::uint8_t {
    active = 0,
    excised = 1,
    escaped = 2,
};

// The null-slice particle state carries invariant rest weight, specific
// covariant p_r>0 and eta=L^2/2. These weights describe the particle population;
// the stress source also depends on its momentum through the null-slice measure.
struct Marker {
    double rest_weight = 0.0;
    double r = 0.0;
    double p_r = 0.0;
    double eta = 0.0;
    std::uint64_t id = 0;
    std::uint64_t source_id = 0;
    MarkerState state = MarkerState::active;
};

struct DepositedSources {
    std::vector<double> shell_rest_weight;
    std::vector<double> shell_t_rr_numerator;
    std::vector<double> shell_s_numerator;
    std::vector<double> t_rr;
    std::vector<double> s;
    std::vector<double> effective_marker_number;
    double active_rest_weight = 0.0;
    double deposited_rest_weight = 0.0;
    double excised_rest_weight = 0.0;
    double escaped_rest_weight = 0.0;
    double rest_weight_roundoff = 0.0;
};

struct Metric {
    std::vector<double> radius;
    std::vector<double> face_radius;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> a_r;
    std::vector<double> b_r;
    std::vector<double> m;
    std::vector<double> mb;
    std::vector<double> compactness;
    std::vector<double> face_a;
    std::vector<double> face_b;
    std::vector<double> face_a_r;
    std::vector<double> face_b_r;
    std::vector<double> face_m;
    std::vector<double> face_mb;
    std::vector<double> face_compactness;
    double adm_mass = 0.0;
    bool fixed_adm_mass_used = false;
    double minimum_a = 1.0;
    double maximum_compactness = 0.0;
    std::size_t maximum_compactness_cell = 0;
    bool apparent_horizon_found = false;
    double apparent_horizon_radius = 0.0;
    double apparent_horizon_mass = 0.0;
    double excision_radius = 0.0;
};

struct Field {
    DepositedSources sources;
    Metric metric;
    double total_rest_weight = 0.0;
};

// Select the integration constant of the radial mass equation explicitly.
// An outward solve sets mb at the inner face to zero. A fixed outer ADM value
// sets mb at the outer face instead; these are distinct boundary conditions.
struct ConstraintOptions {
    // Empty on the first slice: integrate mb outward from regular m(0)=0 and
    // infer ADM M.  Set thereafter: integrate inward from mb(infinity)=M.
    std::optional<double> fixed_adm_mass;
    double excision_a_threshold = -0.01;
    int excision_buffer_cells = 4;
    // Retain the interior down to this fraction of the outer apparent
    // horizon. A value of 1 permits a boundary up to the horizon radius.
    double excision_horizon_fraction = 0.1;
    // Final member of the positional aggregate initialization contract.
    bool enable_excision = true;
};

struct EvolutionOptions {
    WeightedPICParticleShape shape = WeightedPICParticleShape::tsc;
    unsigned int threads = std::max(1u, std::thread::hardware_concurrency());
    int characteristic_substeps = 1;
    std::optional<double> fixed_adm_mass;
    bool enable_excision = true;
    double excision_a_threshold = -0.01;
    int excision_buffer_cells = 4;
    double excision_horizon_fraction = 0.1;
    // A fixed isolated outer mass requires matter to remain within the
    // vacuum boundary. Library callers supplying a separate exterior model
    // can account for escaped matter separately by leaving this false.
    bool require_vacuum_outer_boundary = false;
};

class VacuumOuterBoundaryViolation : public std::runtime_error {
public:
    VacuumOuterBoundaryViolation()
        : std::runtime_error(
            "EF fixed-ADM vacuum outer boundary was crossed by positive-rest-weight "
            "matter; enlarge the domain or supply an evolving exterior boundary.") {}
};

inline void check_vacuum_outer_boundary(const std::vector<Marker>& markers) {
    for (const Marker& marker : markers) {
        if (marker.state == MarkerState::escaped && marker.rest_weight > 0.0) {
            throw VacuumOuterBoundaryViolation();
        }
    }
}

// The boundary is a selected cell center.  The metric sampler is linear
// between center/face knots, so checking every knot through the outward
// buffer establishes a<a_ex<0 and b>0 throughout that retained interval.
// An outer apparent horizon alone does not establish this local condition:
// a trapped shell can surround a regular untrapped central cavity.
// Both radial null characteristic directions point inward where a<0 and b>0.
// Require a finite trapped interval through the interpolation knots before
// removing particle sources behind this boundary.
inline bool causal_excision_boundary(
    const Metric& metric,
    double radius,
    double a_threshold,
    int buffer_cells) {
    const std::size_t cells = metric.radius.size();
    if (!(radius > 0.0) || !std::isfinite(radius)
        || !(a_threshold < 0.0) || !std::isfinite(a_threshold)
        || buffer_cells < 0 || cells == 0
        || metric.a.size() != cells || metric.b.size() != cells
        || metric.face_a.size() != cells + 1
        || metric.face_b.size() != cells + 1
        || !metric.apparent_horizon_found
        || !std::isfinite(metric.apparent_horizon_radius)
        || !(radius < metric.apparent_horizon_radius)) {
        return false;
    }
    const auto center = std::lower_bound(
        metric.radius.begin(), metric.radius.end(), radius);
    if (center == metric.radius.end() || *center != radius) {
        return false;
    }
    const std::size_t first = static_cast<std::size_t>(
        center - metric.radius.begin());
    const std::size_t buffer = static_cast<std::size_t>(buffer_cells);
    if (buffer >= cells - first) {
        return false;
    }
    auto trapped = [&](double a, double b) {
        return std::isfinite(a) && a < a_threshold
            && std::isfinite(b) && b > 0.0;
    };
    for (std::size_t i = first; i <= first + buffer; ++i) {
        if (!trapped(metric.a[i], metric.b[i])) {
            return false;
        }
        if (i > first && !trapped(metric.face_a[i], metric.face_b[i])) {
            return false;
        }
    }
    return true;
}

// Removing sources changes the constrained endpoint metric.  Validate any
// retained boundary against that recomputed metric before accepting a step.
inline void retain_causal_excision_boundary(
    Metric& metric,
    double applied_radius,
    const ConstraintOptions& options) {
    if (!(applied_radius >= 0.0) || !std::isfinite(applied_radius)
        || !(metric.excision_radius >= 0.0)
        || !std::isfinite(metric.excision_radius)) {
        throw std::runtime_error("EF excision boundary has an invalid radius.");
    }
    const double retained_radius = std::max(
        metric.excision_radius, applied_radius);
    const double cap = options.excision_horizon_fraction
        * metric.apparent_horizon_radius;
    const double radius_roundoff = 64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(cap));
    if (retained_radius > 0.0
        && (!causal_excision_boundary(
                metric, retained_radius, options.excision_a_threshold,
                options.excision_buffer_cells)
            || !std::isfinite(cap)
            || retained_radius > cap + radius_roundoff)) {
        throw std::runtime_error(
            "EF excision boundary violates its trapped-buffer or radius-cap "
            "condition in the recomputed endpoint metric; the step was not accepted.");
    }
    metric.excision_radius = retained_radius;
}

struct MetricSample {
    double a = 1.0;
    double b = 1.0;
    double a_r = 0.0;
    double b_r = 0.0;
    double m = 0.0;
};

struct CharacteristicState {
    double r = 0.0;
    double p_r = 1.0;
};

struct CharacteristicDerivative {
    double dr = 0.0;
    double dp_r = 0.0;
};

struct ExcisionResult {
    double radius = 0.0;
    double newly_excised_rest_weight = 0.0;
    std::size_t newly_excised_markers = 0;
};

struct StepResult {
    Field old_field;
    Field midpoint_field;
    Field endpoint_field;
    double accepted_dt = 0.0;
    double conserved_adm_mass = 0.0;
    double rest_weight_before = 0.0;
    double rest_weight_after = 0.0;
    double newly_excised_rest_weight = 0.0;
    std::size_t newly_excised_markers = 0;
    bool apparent_horizon_found = false;
};

namespace detail {

inline void validate_mesh(const WeightedPICFieldGrid& grid) {
    const std::size_t cells = grid.mesh.shell_volume.size();
    if (cells == 0 || grid.radius.size() != cells
        || grid.mesh.edge.size() != cells + 1
        || grid.mesh.volume_coordinate_center.size() != cells
        || grid.mesh.venus_center.size() != cells
        || grid.mesh.venus_width.size() != cells) {
        throw std::runtime_error(
            "Eddington--Finkelstein WPIC received an invalid radial mesh.");
    }
    for (std::size_t i = 0; i < cells; ++i) {
        if (!(grid.mesh.edge[i + 1] > grid.mesh.edge[i])
            || !(grid.mesh.shell_volume[i] > 0.0)
            || !(grid.radius[i] > grid.mesh.edge[i])
            || !(grid.radius[i] < grid.mesh.edge[i + 1])) {
            throw std::runtime_error(
                "Eddington--Finkelstein WPIC mesh is not strictly ordered.");
        }
    }
}

inline void validate_marker(const Marker& marker, const char* context) {
    const int state = static_cast<int>(marker.state);
    if (!(marker.rest_weight >= 0.0) || !std::isfinite(marker.rest_weight)
        || !(marker.r >= 0.0) || !std::isfinite(marker.r)
        || !(marker.p_r > 0.0) || !std::isfinite(marker.p_r)
        || !(marker.eta >= 0.0) || !std::isfinite(marker.eta)
        || (marker.r == 0.0 && marker.eta > 0.0)
        || state < static_cast<int>(MarkerState::active)
        || state > static_cast<int>(MarkerState::escaped)) {
        throw std::runtime_error(
            std::string(context) + " received an invalid EF marker.");
    }
}

struct MarkerDeposit {
    std::array<std::size_t, 3> cell{0, 0, 0};
    std::array<double, 3> weight{0.0, 0.0, 0.0};
    int count = 0;
    long double rest_weight = 0.0L;
    long double t_rr_numerator = 0.0L;
    long double s_numerator = 0.0L;
};

inline double b_inside_cell(
    double radius,
    double left_radius,
    double left_b,
    double t_rr) {
    return left_b * std::exp(
        2.0 * PI * t_rr
        * (radius * radius - left_radius * left_radius));
}

// Integrate d(mb)/dr=2*pi*r^2*b*(T_rr+S) over a cell segment. The source is
// constant in that cell and b retains its analytic exponential radial dependence.
inline double integrate_mb_source_segment(
    double left,
    double right,
    double cell_left,
    double cell_left_b,
    double t_rr,
    double s) {
    if (!(right > left) || t_rr + s == 0.0) {
        return 0.0;
    }
    // Three-point Gauss--Legendre is deterministic and resolves the smooth
    // b(r)=b_left exp[2 pi T_rr(r^2-r_left^2)] factor within a FV cell.
    constexpr std::array<double, 3> node{
        -0.774596669241483377035853079956,
        0.0,
        0.774596669241483377035853079956,
    };
    constexpr std::array<double, 3> weight{
        5.0 / 9.0,
        8.0 / 9.0,
        5.0 / 9.0,
    };
    const double midpoint = 0.5 * (left + right);
    const double half_width = 0.5 * (right - left);
    long double integral = 0.0L;
    for (std::size_t q = 0; q < node.size(); ++q) {
        const double r = midpoint + half_width * node[q];
        const double b = b_inside_cell(
            r, cell_left, cell_left_b, t_rr);
        integral += static_cast<long double>(weight[q])
            * (2.0L * static_cast<long double>(PI))
            * static_cast<long double>(t_rr + s)
            * static_cast<long double>(r) * r
            * static_cast<long double>(b);
    }
    return static_cast<double>(half_width * integral);
}

inline double interpolate_piecewise_center_face(
    double sampled_radius,
    std::size_t cell,
    const std::vector<double>& center_radius,
    const std::vector<double>& face_radius,
    const std::vector<double>& center_value,
    const std::vector<double>& face_value) {
    const double center = center_radius[cell];
    if (sampled_radius <= center) {
        const double width = center - face_radius[cell];
        const double fraction = width > 0.0
            ? (sampled_radius - face_radius[cell]) / width
            : 0.0;
        return (1.0 - fraction) * face_value[cell]
            + fraction * center_value[cell];
    }
    const double width = face_radius[cell + 1] - center;
    const double fraction = width > 0.0
        ? (sampled_radius - center) / width
        : 0.0;
    return (1.0 - fraction) * center_value[cell]
        + fraction * face_value[cell + 1];
}

inline double total_rest_weight(const std::vector<Marker>& markers) {
    long double result = 0.0L;
    for (const Marker& marker : markers) {
        validate_marker(marker, "EF WPIC rest-weight diagnostic");
        result += marker.rest_weight;
    }
    return static_cast<double>(result);
}

}  // namespace detail

/// Builds the same conservative q=r^3/3 radial field mesh used by WPIC.
inline WeightedPICFieldGrid make_field_grid(
    const Params& source_parameters,
    int radial_cells) {
    if (source_parameters.physics_model != "einstein_vlasov"
        || source_parameters.vp_universe != "static"
        || source_parameters.bp_collision_operator != "none"
        || !(source_parameters.inner_boundary == "origin_parity"
             || source_parameters.inner_boundary == "reflecting")
        || source_parameters.outer_boundary != "outflow_vacuum"
        || !using_cell_average_eta_grid(source_parameters)
        || radial_cells < source_parameters.Nr || radial_cells < 1
        || !(source_parameters.r_max > source_parameters.r_min)
        || source_parameters.r_min < 0.0) {
        throw std::runtime_error(
            "EF WPIC field grid requires isolated collisionless Einstein--Vlasov, "
            "vp_universe=static, cell-average eta, origin_parity or a finite "
            "inner mesh, outer_boundary=outflow_vacuum, a nonnegative radial "
            "domain, and no radial coarsening. EF characteristics reaching "
            "the inner face before excision cause a hard failure.");
    }
    // The solver contract, rather than the configured finite-wall r_min,
    // selects a true origin mesh.  Production origin_parity profiles retain a
    // positive r_min for phase-space/orbit controls, exactly as the existing
    // Schwarzschild WPIC grid builder does.
    if (using_origin_parity_inner(source_parameters)) {
        if (source_parameters.radial_grid != "linear") {
            throw std::runtime_error(
                "Origin-resolving EF WPIC currently requires a linear-r mesh.");
        }
        WeightedPICFieldGrid result;
        result.parameters = source_parameters;
        result.parameters.Nr = radial_cells;
        result.mesh.edge.resize(static_cast<std::size_t>(radial_cells + 1));
        result.mesh.volume_coordinate_center.resize(
            static_cast<std::size_t>(radial_cells));
        result.mesh.shell_volume.resize(static_cast<std::size_t>(radial_cells));
        result.mesh.venus_center.resize(static_cast<std::size_t>(radial_cells));
        result.mesh.venus_width.resize(static_cast<std::size_t>(radial_cells));
        result.radius.resize(static_cast<std::size_t>(radial_cells));
        for (int i = 0; i <= radial_cells; ++i) {
            result.mesh.edge[static_cast<std::size_t>(i)] =
                source_parameters.r_max * static_cast<double>(i)
                / static_cast<double>(radial_cells);
        }
        for (int i = 0; i < radial_cells; ++i) {
            const std::size_t cell = static_cast<std::size_t>(i);
            const double left = result.mesh.edge[cell];
            const double right = result.mesh.edge[cell + 1];
            const double q_left = left * left * left / 3.0;
            const double q_right = right * right * right / 3.0;
            result.mesh.volume_coordinate_center[cell] =
                0.5 * (q_left + q_right);
            result.mesh.shell_volume[cell] =
                4.0 * PI * (q_right - q_left);
            result.radius[cell] = std::cbrt(
                3.0 * result.mesh.volume_coordinate_center[cell]);
            result.mesh.venus_center[cell] = result.radius[cell];
            result.mesh.venus_width[cell] = right - left;
        }
        result.radial_coordinate =
            radial_coordinate_grid(result.radius, result.parameters);
        detail::validate_mesh(result);
        return result;
    }

    Params weak_field_parameters = source_parameters;
    weak_field_parameters.physics_model = "vlasov_poisson";
    WeightedPICFieldGrid result =
        make_weighted_pic_field_grid(weak_field_parameters, radial_cells);
    result.parameters = source_parameters;
    result.parameters.Nr = radial_cells;
    result.radius.resize(result.mesh.volume_coordinate_center.size());
    for (std::size_t i = 0; i < result.radius.size(); ++i) {
        result.radius[i] = std::cbrt(
            3.0 * result.mesh.volume_coordinate_center[i]);
    }
    result.mesh.venus_center = result.radius;
    result.radial_coordinate =
        radial_coordinate_grid(result.radius, result.parameters);
    detail::validate_mesh(result);
    return result;
}

/// Stable Schwarzschild-orthonormal w -> ingoing EF canonical p_r map.
/// This conversion is defined only outside the horizon (a>0).
// The exterior mass shell gives sqrt(a)*p_r=epsilon+w. For inward motion use
// chi/(epsilon-w) to preserve the positive small difference at large |w|.
inline double canonical_p_r_from_schwarzschild_w(
    double w,
    double eta,
    double radius,
    double a) {
    if (!(a > 0.0) || !(radius > 0.0) || !(eta >= 0.0)
        || !std::isfinite(w) || !std::isfinite(eta)
        || !std::isfinite(radius) || !std::isfinite(a)) {
        throw std::runtime_error(
            "Schwarzschild-to-EF momentum conversion requires r>0 and a>0.");
    }
    const double chi = eta == 0.0 ? 1.0
        : 1.0 + 2.0 * eta / (radius * radius);
    const double epsilon = std::sqrt(chi + w * w);
    const double numerator = w < 0.0
        ? chi / (epsilon - w)
        : epsilon + w;
    const double p_r = numerator / std::sqrt(a);
    if (!(p_r > 0.0) || !std::isfinite(p_r)) {
        throw std::runtime_error(
            "Schwarzschild-to-EF momentum conversion produced invalid p_r.");
    }
    return p_r;
}

/// Signed covariant momentum in the outgoing/retarded EF branch (b<0).
/// It is negative for a future-directed particle.  East's t->-t initial-data
/// operation maps this value to the positive ingoing momentum by changing
/// both b and p_r in sign while leaving areal radius and eta unchanged.
inline double outgoing_canonical_p_r_from_schwarzschild_w(
    double w,
    double eta,
    double radius,
    double a) {
    if (!(a > 0.0) || !(radius > 0.0) || !(eta >= 0.0)
        || !std::isfinite(w) || !std::isfinite(eta)
        || !std::isfinite(radius) || !std::isfinite(a)) {
        throw std::runtime_error(
            "Schwarzschild-to-outgoing-EF momentum conversion requires "
            "r>0 and a>0.");
    }
    const double chi = eta == 0.0 ? 1.0
        : 1.0 + 2.0 * eta / (radius * radius);
    const double epsilon = std::sqrt(chi + w * w);
    const double numerator = w > 0.0
        ? -chi / (epsilon + w)
        : w - epsilon;
    const double p_r = numerator / std::sqrt(a);
    if (!(p_r < 0.0) || !std::isfinite(p_r)) {
        throw std::runtime_error(
            "Schwarzschild-to-outgoing-EF conversion produced invalid p_r.");
    }
    return p_r;
}

inline double time_reverse_outgoing_canonical_p_r(double outgoing_p_r) {
    if (!(outgoing_p_r < 0.0) || !std::isfinite(outgoing_p_r)) {
        throw std::runtime_error(
            "East time reversal requires a future-directed outgoing p_r<0.");
    }
    return -outgoing_p_r;
}

/// Inverse ingoing EF canonical p_r -> Schwarzschild-orthonormal w map.
inline double schwarzschild_w_from_canonical_p_r(
    double p_r,
    double eta,
    double radius,
    double a) {
    if (!(p_r > 0.0) || !(a > 0.0) || !(radius > 0.0)
        || !(eta >= 0.0) || !std::isfinite(p_r)
        || !std::isfinite(eta) || !std::isfinite(radius)
        || !std::isfinite(a)) {
        throw std::runtime_error(
            "EF-to-Schwarzschild momentum conversion requires p_r,r,a>0.");
    }
    const double sqrt_a_p = std::sqrt(a) * p_r;
    const double chi = eta == 0.0 ? 1.0
        : 1.0 + 2.0 * eta / (radius * radius);
    return 0.5 * (sqrt_a_p - chi / sqrt_a_p);
}

inline Marker from_schwarzschild_marker(
    const WeightedPICMarker& source,
    double a) {
    validate_weighted_pic_marker(
        source, "Schwarzschild-to-EF marker conversion");
    if (source.internal_energy != 0.0) {
        throw std::runtime_error(
            "A collisionless Schwarzschild-to-EF marker conversion may not "
            "discard unresolved internal energy.");
    }
    if (source.state == WeightedPICMarkerState::delayed_inner) {
        throw std::runtime_error(
            "A delayed-inner Schwarzschild marker has no EF marker mapping.");
    }
    Marker result;
    result.rest_weight = source.mass;
    result.r = source.r;
    result.p_r = canonical_p_r_from_schwarzschild_w(
        source.w, source.eta, source.r, a);
    result.eta = source.eta;
    result.id = source.id;
    result.source_id = source.source_cell;
    result.state = source.state == WeightedPICMarkerState::escaped
        ? MarkerState::escaped
        : MarkerState::active;
    detail::validate_marker(result, "Schwarzschild-to-EF marker conversion");
    return result;
}

inline WeightedPICMarker to_schwarzschild_marker(
    const Marker& source,
    double a) {
    detail::validate_marker(source, "EF-to-Schwarzschild marker conversion");
    if (source.state == MarkerState::excised) {
        throw std::runtime_error(
            "An excised EF marker cannot be represented in the exterior "
            "Schwarzschild marker state model.");
    }
    WeightedPICMarker result;
    result.mass = source.rest_weight;
    result.r = source.r;
    result.w = schwarzschild_w_from_canonical_p_r(
        source.p_r, source.eta, source.r, a);
    result.eta = source.eta;
    result.id = source.id;
    result.source_cell = source.source_id;
    result.state = source.state == MarkerState::escaped
        ? WeightedPICMarkerState::escaped
        : WeightedPICMarkerState::active;
    return result;
}

/// Deterministic threaded deposition.  Workers construct one stencil record
/// per marker; the final long-double accumulation is always in marker-ID array
/// order, making results independent of the requested thread count.
// Construct each particle stencil independently, then accumulate in array
// order. Dividing the weighted numerators by coordinate shell volume supplies
// T_rr and S for the null constraints; the volume includes 4*pi*r^2 dr.
inline DepositedSources deposit_sources_parallel(
    const std::vector<Marker>& markers,
    const WeightedPICRadialMesh& mesh,
    WeightedPICParticleShape shape,
    unsigned int requested_threads) {
    const std::size_t cells = mesh.shell_volume.size();
    if (cells == 0 || mesh.edge.size() != cells + 1
        || mesh.volume_coordinate_center.size() != cells
        || requested_threads == 0
        || markers.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("EF WPIC deposition received invalid inputs.");
    }
    std::vector<detail::MarkerDeposit> contribution(markers.size());
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        static_cast<int>(requested_threads),
        [&](int first, int last, int) {
            for (int q = first; q < last; ++q) {
                const Marker& marker = markers[static_cast<std::size_t>(q)];
                detail::validate_marker(marker, "EF WPIC deposition");
                if (marker.state != MarkerState::active) {
                    continue;
                }
                if (marker.r < mesh.edge.front()
                    || marker.r > mesh.edge.back()) {
                    throw std::runtime_error(
                        "An active EF marker lies outside the field mesh.");
                }
                detail::MarkerDeposit& value =
                    contribution[static_cast<std::size_t>(q)];
                value.rest_weight = marker.rest_weight;
                value.t_rr_numerator =
                    static_cast<long double>(marker.rest_weight) * marker.p_r;
                const long double radius = std::max(
                    static_cast<long double>(marker.r), 1.0e-150L);
                const long double chi = 1.0L
                    + 2.0L * marker.eta / (radius * radius);
                value.s_numerator = static_cast<long double>(marker.rest_weight)
                    * chi / marker.p_r;
                if (shape == WeightedPICParticleShape::cic) {
                    const WeightedPICCICStencil stencil =
                        weighted_pic_cic_stencil(marker.r, mesh);
                    value.count = stencil.count;
                    for (int n = 0; n < stencil.count; ++n) {
                        value.cell[static_cast<std::size_t>(n)] =
                            stencil.cell[static_cast<std::size_t>(n)];
                        value.weight[static_cast<std::size_t>(n)] =
                            stencil.weight[static_cast<std::size_t>(n)];
                    }
                } else {
                    const WeightedPICTSCStencil stencil =
                        weighted_pic_tsc_stencil(marker.r, mesh);
                    value.count = stencil.count;
                    for (int n = 0; n < stencil.count; ++n) {
                        value.cell[static_cast<std::size_t>(n)] =
                            stencil.cell[static_cast<std::size_t>(n)];
                        value.weight[static_cast<std::size_t>(n)] =
                            stencil.weight[static_cast<std::size_t>(n)];
                    }
                }
            }
        });

    std::vector<long double> rest(cells, 0.0L);
    std::vector<long double> t_rr(cells, 0.0L);
    std::vector<long double> s(cells, 0.0L);
    std::vector<long double> rest_square(cells, 0.0L);
    long double active = 0.0L;
    long double excised = 0.0L;
    long double escaped = 0.0L;
    for (std::size_t q = 0; q < markers.size(); ++q) {
        const Marker& marker = markers[q];
        if (marker.state == MarkerState::excised) {
            excised += marker.rest_weight;
            continue;
        }
        if (marker.state == MarkerState::escaped) {
            escaped += marker.rest_weight;
            continue;
        }
        active += marker.rest_weight;
        const detail::MarkerDeposit& value = contribution[q];
        for (int n = 0; n < value.count; ++n) {
            const std::size_t slot = static_cast<std::size_t>(n);
            const std::size_t cell = value.cell[slot];
            const long double weight = value.weight[slot];
            const long double dm = value.rest_weight * weight;
            rest[cell] += dm;
            t_rr[cell] += value.t_rr_numerator * weight;
            s[cell] += value.s_numerator * weight;
            // The effective particle count uses deposited rest weights, N_eff=(sum W)^2
            // /sum W^2. It describes sampling and leaves the full stress source intact.
            rest_square[cell] += dm * dm;
        }
    }

    DepositedSources result;
    result.shell_rest_weight.resize(cells);
    result.shell_t_rr_numerator.resize(cells);
    result.shell_s_numerator.resize(cells);
    result.t_rr.resize(cells);
    result.s.resize(cells);
    result.effective_marker_number.resize(cells);
    long double deposited = 0.0L;
    for (std::size_t i = 0; i < cells; ++i) {
        result.shell_rest_weight[i] = static_cast<double>(rest[i]);
        result.shell_t_rr_numerator[i] = static_cast<double>(t_rr[i]);
        result.shell_s_numerator[i] = static_cast<double>(s[i]);
        result.t_rr[i] = static_cast<double>(
            t_rr[i] / mesh.shell_volume[i]);
        result.s[i] = static_cast<double>(s[i] / mesh.shell_volume[i]);
        result.effective_marker_number[i] = rest_square[i] > 0.0L
            ? static_cast<double>(rest[i] * rest[i] / rest_square[i])
            : 0.0;
        deposited += rest[i];
    }
    result.active_rest_weight = static_cast<double>(active);
    result.deposited_rest_weight = static_cast<double>(deposited);
    result.excised_rest_weight = static_cast<double>(excised);
    result.escaped_rest_weight = static_cast<double>(escaped);
    result.rest_weight_roundoff = static_cast<double>(deposited - active);
    return result;
}

inline DepositedSources deposit_sources(
    const std::vector<Marker>& markers,
    const WeightedPICRadialMesh& mesh,
    WeightedPICParticleShape shape = WeightedPICParticleShape::tsc) {
    return deposit_sources_parallel(markers, mesh, shape, 1);
}

inline Metric solve_metric(
    const DepositedSources& source,
    const WeightedPICFieldGrid& grid,
    const ConstraintOptions& options = {}) {
    detail::validate_mesh(grid);
    const std::size_t cells = grid.radius.size();
    if (source.t_rr.size() != cells || source.s.size() != cells
        || options.excision_buffer_cells < 0
        || !(options.excision_a_threshold < 0.0)
        || !std::isfinite(options.excision_a_threshold)
        || !(options.excision_horizon_fraction > 0.0)
        || !(options.excision_horizon_fraction <= 1.0)) {
        throw std::runtime_error("EF metric solve received invalid sources/options.");
    }
    if (options.fixed_adm_mass.has_value()
        && (!(options.fixed_adm_mass.value() >= 0.0)
            || !std::isfinite(options.fixed_adm_mass.value()))) {
        throw std::runtime_error("EF fixed ADM mass must be finite and nonnegative.");
    }

    Metric metric;
    metric.radius = grid.radius;
    metric.face_radius = grid.mesh.edge;
    metric.a.assign(cells, 1.0);
    metric.b.assign(cells, 1.0);
    metric.a_r.assign(cells, 0.0);
    metric.b_r.assign(cells, 0.0);
    metric.m.assign(cells, 0.0);
    metric.mb.assign(cells, 0.0);
    metric.compactness.assign(cells, 0.0);
    metric.face_a.assign(cells + 1, 1.0);
    metric.face_b.assign(cells + 1, 1.0);
    metric.face_a_r.assign(cells + 1, 0.0);
    metric.face_b_r.assign(cells + 1, 0.0);
    metric.face_m.assign(cells + 1, 0.0);
    metric.face_mb.assign(cells + 1, 0.0);
    metric.face_compactness.assign(cells + 1, 0.0);

    // Integrate d_r log b inward from b(r_max)=1.  With a cell-constant
    // T_rr this part is analytic and manifestly positive even inside a=0.
    metric.face_b.back() = 1.0;
    for (std::size_t i = cells; i-- > 0;) {
        const double left = metric.face_radius[i];
        const double right = metric.face_radius[i + 1];
        metric.face_b[i] = metric.face_b[i + 1] * std::exp(
            -2.0 * PI * source.t_rr[i]
            * (right * right - left * left));
        metric.b[i] = detail::b_inside_cell(
            metric.radius[i], left, metric.face_b[i], source.t_rr[i]);
    }

    std::vector<double> integral_left(cells, 0.0);
    std::vector<double> integral_right(cells, 0.0);
    for (std::size_t i = 0; i < cells; ++i) {
        integral_left[i] = detail::integrate_mb_source_segment(
            metric.face_radius[i],
            metric.radius[i],
            metric.face_radius[i],
            metric.face_b[i],
            source.t_rr[i],
            source.s[i]);
        integral_right[i] = detail::integrate_mb_source_segment(
            metric.radius[i],
            metric.face_radius[i + 1],
            metric.face_radius[i],
            metric.face_b[i],
            source.t_rr[i],
            source.s[i]);
    }

    // The two mass solves share b and every source integral. Their mb values differ
    // by a spatial constant fixed at a boundary, so their m values differ by C/b(r).
    // A prescribed outer ADM condition therefore carries its own central contribution.
    metric.fixed_adm_mass_used = options.fixed_adm_mass.has_value();
    if (options.fixed_adm_mass.has_value()) {
        metric.adm_mass = options.fixed_adm_mass.value();
        metric.face_mb.back() = metric.adm_mass;
        for (std::size_t i = cells; i-- > 0;) {
            metric.mb[i] = metric.face_mb[i + 1] - integral_right[i];
            metric.face_mb[i] = metric.mb[i] - integral_left[i];
        }
    } else {
        // Regular first slice.  A finite-r inner boundary is treated as an
        // empty regular interior; central point masses belong in the fixed-M
        // continuation, not in this initial outward solve.
        metric.face_mb.front() = 0.0;
        for (std::size_t i = 0; i < cells; ++i) {
            metric.mb[i] = metric.face_mb[i] + integral_left[i];
            metric.face_mb[i + 1] = metric.mb[i] + integral_right[i];
        }
        metric.adm_mass = metric.face_mb.back();
    }

    // Field derivatives follow the constraint equations at the same centres and
    // faces used by interpolation. The centre-to-face source interpolation is explicit.
    auto source_at_face = [&](const std::vector<double>& value, std::size_t face) {
        if (face == 0) {
            return value.front();
        }
        if (face == cells) {
            return value.back();
        }
        const double left_r = metric.radius[face - 1];
        const double right_r = metric.radius[face];
        const double fraction = (metric.face_radius[face] - left_r)
            / (right_r - left_r);
        return (1.0 - fraction) * value[face - 1]
            + fraction * value[face];
    };

    metric.minimum_a = std::numeric_limits<double>::infinity();
    metric.maximum_compactness = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < cells; ++i) {
        metric.m[i] = metric.mb[i] / metric.b[i];
        metric.compactness[i] = 2.0 * metric.m[i] / metric.radius[i];
        metric.a[i] = 1.0 - metric.compactness[i];
        metric.b_r[i] = 4.0 * PI * metric.radius[i]
            * metric.b[i] * source.t_rr[i];
        metric.a_r[i] = (1.0 - metric.a[i]) / metric.radius[i]
            - 4.0 * PI * metric.radius[i]
                * (metric.a[i] * source.t_rr[i] + source.s[i]);
        if (!std::isfinite(metric.a[i]) || !std::isfinite(metric.b[i])
            || !std::isfinite(metric.a_r[i])
            || !std::isfinite(metric.b_r[i]) || !(metric.b[i] > 0.0)) {
            throw std::runtime_error("EF metric constraint produced non-finite data.");
        }
        metric.minimum_a = std::min(metric.minimum_a, metric.a[i]);
        if (metric.compactness[i] > metric.maximum_compactness) {
            metric.maximum_compactness = metric.compactness[i];
            metric.maximum_compactness_cell = i;
        }
    }
    for (std::size_t face = 0; face <= cells; ++face) {
        const double r = metric.face_radius[face];
        metric.face_m[face] = metric.face_mb[face] / metric.face_b[face];
        if (r > 0.0) {
            metric.face_compactness[face] = 2.0 * metric.face_m[face] / r;
            metric.face_a[face] = 1.0 - metric.face_compactness[face];
            const double t_rr = source_at_face(source.t_rr, face);
            const double s = source_at_face(source.s, face);
            metric.face_b_r[face] = 4.0 * PI * r
                * metric.face_b[face] * t_rr;
            metric.face_a_r[face] = (1.0 - metric.face_a[face]) / r
                - 4.0 * PI * r * (metric.face_a[face] * t_rr + s);
        } else {
            // The singular black-hole interior is never sampled after
            // excision.  Keep the regular coordinate boundary value finite.
            metric.face_a[face] = 1.0;
            metric.face_compactness[face] = 0.0;
            metric.face_a_r[face] = 0.0;
            metric.face_b_r[face] = 0.0;
        }
    }

    // An apparent horizon is located from the outermost outward crossing of a=0.
    // The source-removal boundary additionally requires the causal checks below.
    // The outermost negative-to-positive zero is the apparent horizon.
    for (std::size_t i = 0; i < cells; ++i) {
        const double left_a = metric.face_a[i];
        const double right_a = metric.face_a[i + 1];
        if (left_a <= 0.0 && right_a >= 0.0) {
            const double left_r = metric.face_radius[i];
            const double right_r = metric.face_radius[i + 1];
            const double left_f = left_r * left_r * left_a;
            const double right_f = right_r * right_r * right_a;
            double horizon = left_r;
            if (right_f != left_f) {
                horizon = left_r
                    + (right_r - left_r) * (-left_f) / (right_f - left_f);
            }
            metric.apparent_horizon_found = true;
            metric.apparent_horizon_radius = std::clamp(
                horizon, left_r, right_r);
            metric.apparent_horizon_mass =
                0.5 * metric.apparent_horizon_radius;
        }
    }

    // Retain a substantial interior. Cap the threshold-and-buffer
    // boundary by a fraction of the apparent-horizon radius, then require
    // the boundary and its outward buffer to lie in one trapped interval.
    // In particular, never excise a regular inner cavity merely because an
    // outer trapped annulus exists. A qualifying cell sets the boundary.
    std::optional<std::size_t> outer_trigger;
    for (std::size_t i = 0; i < cells; ++i) {
        if (metric.a[i] < options.excision_a_threshold) {
            outer_trigger = i;
        }
    }
    if (options.enable_excision && metric.apparent_horizon_found && outer_trigger.has_value()
        && outer_trigger.value()
            >= static_cast<std::size_t>(options.excision_buffer_cells)) {
        const std::size_t buffer = static_cast<std::size_t>(options.excision_buffer_cells);
        const double cap = options.excision_horizon_fraction
            * metric.apparent_horizon_radius;
        const std::size_t last = outer_trigger.value() - buffer;
        for (std::size_t candidate = last + 1; candidate-- > 0;) {
            if (metric.radius[candidate] > cap) continue;
            if (causal_excision_boundary(
                    metric, metric.radius[candidate],
                    options.excision_a_threshold,
                    options.excision_buffer_cells)) {
                metric.excision_radius = metric.radius[candidate];
                break;
            }
        }
    }
    return metric;
}

inline Field solve_field(
    const DepositedSources& sources,
    const WeightedPICFieldGrid& grid,
    const ConstraintOptions& options = {}) {
    if (!options.enable_excision && sources.excised_rest_weight > 0.0) {
        throw std::runtime_error(
            "EF interior evolution requires data with every marker retained.");
    }
    Field result;
    result.sources = sources;
    result.metric = solve_metric(sources, grid, options);
    result.total_rest_weight = sources.active_rest_weight
        + sources.excised_rest_weight + sources.escaped_rest_weight;
    return result;
}

inline Field deposit_and_solve_field(
    const std::vector<Marker>& markers,
    const WeightedPICFieldGrid& grid,
    WeightedPICParticleShape shape = WeightedPICParticleShape::tsc,
    unsigned int requested_threads = 1,
    const ConstraintOptions& options = {}) {
    return solve_field(
        deposit_sources_parallel(
            markers, grid.mesh, shape, requested_threads),
        grid,
        options);
}

namespace detail {
inline MetricSample sample_metric_on_validated_mesh(
    double radius,
    const WeightedPICFieldGrid& grid,
    const Metric& metric) {
    const std::size_t cells = grid.radius.size();
    if (metric.radius.size() != cells
        || metric.a.size() != cells || metric.b.size() != cells
        || metric.a_r.size() != cells || metric.b_r.size() != cells
        || metric.m.size() != cells || metric.face_radius.size() != cells + 1
        || metric.face_m.size() != cells + 1
        || metric.face_a.size() != cells + 1
        || metric.face_b.size() != cells + 1
        || metric.face_a_r.size() != cells + 1
        || metric.face_b_r.size() != cells + 1
        || !std::isfinite(radius)) {
        throw std::runtime_error("EF metric sampling received invalid inputs.");
    }
    if (radius >= metric.face_radius.back()) {
        // Exact exterior Schwarzschild continuation (uncharged).
        const double r = std::max(radius, metric.face_radius.back());
        return {
            1.0 - 2.0 * metric.adm_mass / r,
            1.0,
            2.0 * metric.adm_mass / (r * r),
            0.0,
            metric.adm_mass,
        };
    }
    const double sampled_radius = std::clamp(
        radius, metric.face_radius.front(), metric.face_radius.back());
    const auto upper = std::upper_bound(
        metric.face_radius.begin(), metric.face_radius.end(), sampled_radius);
    const std::size_t cell = std::min(
        cells - 1,
        static_cast<std::size_t>(std::max<std::ptrdiff_t>(
            0, upper - metric.face_radius.begin() - 1)));
    auto interpolate = [&](const std::vector<double>& center,
                           const std::vector<double>& face) {
        return detail::interpolate_piecewise_center_face(
            sampled_radius,
            cell,
            metric.radius,
            metric.face_radius,
            center,
            face);
    };
    return {
        interpolate(metric.a, metric.face_a),
        interpolate(metric.b, metric.face_b),
        interpolate(metric.a_r, metric.face_a_r),
        interpolate(metric.b_r, metric.face_b_r),
        interpolate(metric.m, metric.face_m),
    };
}

}  // namespace detail

// A batch retains const references to one mesh/metric pair. Validate its mesh
// once, then reuse the unchanged interpolation for every particle and RK stage.
// The referenced arrays must remain unchanged for this sampler's lifetime.
class MetricSampler {
public:
    MetricSampler(const WeightedPICFieldGrid& grid, const Metric& metric)
        : grid_(grid), metric_(metric) {
        detail::validate_mesh(grid_);
        if (metric_.radius.size() != grid_.radius.size()
            || metric_.face_radius.size() != grid_.mesh.edge.size()) {
            throw std::runtime_error("EF metric sampling received inconsistent radial arrays.");
        }
        for (std::size_t i = 0; i < metric_.radius.size(); ++i) {
            if (!std::isfinite(metric_.radius[i])
                || metric_.radius[i] != grid_.radius[i]) {
                throw std::runtime_error("EF metric centers disagree with the field mesh.");
            }
        }
        for (std::size_t i = 0; i < metric_.face_radius.size(); ++i) {
            const double radius = metric_.face_radius[i];
            if (!std::isfinite(radius) || radius != grid_.mesh.edge[i]
                || (i > 0 && !(radius > metric_.face_radius[i - 1]))) {
                throw std::runtime_error("EF metric faces disagree with the ordered field mesh.");
            }
        }
        // This checks all arrays used by the interpolation, including faces.
        (void)detail::sample_metric_on_validated_mesh(
            grid_.radius.front(), grid_, metric_);
    }

    MetricSample operator()(double radius) const {
        return detail::sample_metric_on_validated_mesh(radius, grid_, metric_);
    }

private:
    const WeightedPICFieldGrid& grid_;
    const Metric& metric_;
};

inline MetricSample sample_metric(
    double radius,
    const WeightedPICFieldGrid& grid,
    const Metric& metric) {
    return MetricSampler(grid, metric)(radius);
}

// Hamilton equations advance the specific canonical momentum in advanced time v.
// Their finite expressions cross a=0 while retaining future orientation p_r>0.
inline CharacteristicDerivative characteristic_rhs(
    const CharacteristicState& state,
    double eta,
    const MetricSample& metric) {
    if (!(state.p_r > 0.0) || !(state.r >= 0.0) || !(eta >= 0.0)
        || !std::isfinite(state.p_r) || !std::isfinite(state.r)
        || !std::isfinite(eta) || !(metric.b > 0.0)
        || !std::isfinite(metric.a) || !std::isfinite(metric.b)
        || !std::isfinite(metric.a_r) || !std::isfinite(metric.b_r)
        || (state.r == 0.0 && eta > 0.0)) {
        throw std::runtime_error("EF characteristic RHS received invalid data.");
    }
    const double radius = std::max(
        state.r, 32.0 * std::numeric_limits<double>::min());
    const double chi = eta == 0.0 ? 1.0
        : 1.0 + 2.0 * eta / (radius * radius);
    CharacteristicDerivative result;
    result.dr = 0.5 * metric.b
        * (metric.a - chi / (state.p_r * state.p_r));
    // The exactly radial branch has no angular term, including at radii
    // whose square or cube underflows in binary64 arithmetic.
    const double angular_term = eta == 0.0 ? 0.0
        : 2.0 * metric.b * eta
            / (state.p_r * radius * radius * radius);
    result.dp_r = -0.5 * state.p_r
            * (metric.a * metric.b_r + metric.b * metric.a_r)
        - 0.5 * metric.b_r * chi / state.p_r
        + angular_term;
    if (!std::isfinite(result.dr) || !std::isfinite(result.dp_r)) {
        throw std::runtime_error("EF characteristic RHS became non-finite.");
    }
    return result;
}

// Energy measured by the radial unit timelike observer
// u^v=1/b, u^r=(a-1)/2, u^theta=u^phi=0. Its norm is -1 for every a,
// including a=0 and a<0; this observer is not generally geodesic.
// E=-p_mu u^mu=(p_r+(1+2 eta/r^2)/p_r)/2. Evaluating this positive
// expression avoids cancellation between covariant stress components.
// This prescribed orthonormal observer has a regular norm across a=0. Its
// energy is distinct from the conserved Killing energy of a static spacetime.
inline double infall_observer_energy(double radius, double p_r, double eta) {
    if (!(radius > 0.0) || !(p_r > 0.0) || !(eta >= 0.0)
        || !std::isfinite(radius) || !std::isfinite(p_r) || !std::isfinite(eta)) {
        throw std::runtime_error("EF infall observer received invalid momentum.");
    }
    const double chi = eta == 0.0 ? 1.0
        : 1.0 + 2.0 * eta / (radius * radius);
    const double energy = 0.5 * (p_r + chi / p_r);
    if (!std::isfinite(energy)) {
        throw std::runtime_error("EF infall observer energy is nonfinite.");
    }
    return energy;
}

inline CharacteristicDerivative characteristic_rhs(
    const CharacteristicState& state,
    double eta,
    const WeightedPICFieldGrid& grid,
    const Metric& metric) {
    return characteristic_rhs(state, eta, sample_metric(state.r, grid, metric));
}

namespace detail {
inline CharacteristicState rk4_advance_with_sampler(
    CharacteristicState state,
    double eta,
    double dt,
    int substeps,
    const MetricSampler& sample) {
    if (!(dt >= 0.0) || !std::isfinite(dt) || substeps < 1) {
        throw std::runtime_error("EF RK4 received invalid integration controls.");
    }
    if (!(state.r >= 0.0) || !(state.p_r > 0.0)
        || !std::isfinite(state.r) || !std::isfinite(state.p_r)
        || !(eta >= 0.0) || !std::isfinite(eta)
        || (state.r == 0.0 && eta > 0.0)) {
        throw std::runtime_error("EF RK4 received an invalid characteristic.");
    }

    // Future-directed characteristics have p_r>0 throughout this advanced-
    // null chart.  Evolving u=log(p_r), rather than p_r itself, makes that
    // invariant exact at every RK stage; a step can no longer cross the
    // singular p_r=0 boundary merely through an explicit stage overshoot.
    struct LogState {
        double r = 0.0;
        double log_p_r = 0.0;
    };
    struct LogDerivative {
        double dr = 0.0;
        double dlog_p_r = 0.0;
    };
    auto add = [](const LogState& value,
                  double scale,
                  const LogDerivative& derivative) {
        return LogState{
            value.r + scale * derivative.dr,
            value.log_p_r + scale * derivative.dlog_p_r,
        };
    };
    const double minimum_log_p =
        std::log(std::numeric_limits<double>::min());
    const double maximum_log_p =
        std::log(std::numeric_limits<double>::max());
    auto rhs = [&](const LogState& value) {
        if (!(value.r >= 0.0) || !std::isfinite(value.r)
            || !(value.log_p_r > minimum_log_p)
            || !(value.log_p_r < maximum_log_p)
            || !std::isfinite(value.log_p_r)
            || (value.r == 0.0 && eta > 0.0)) {
            throw std::runtime_error(
                "EF log-momentum RK4 stage left its characteristic patch.");
        }
        const double p_r = std::exp(value.log_p_r);
        const double radius = std::max(
            value.r, 32.0 * std::numeric_limits<double>::min());
        const MetricSample sampled = sample(value.r);
        const double inverse_p_squared = 1.0 / (p_r * p_r);
        const double chi = eta == 0.0 ? 1.0
            : 1.0 + 2.0 * eta / (radius * radius);
        const double angular_term = eta == 0.0 ? 0.0
            : 2.0 * sampled.b * eta * inverse_p_squared
                / (radius * radius * radius);
        LogDerivative result;
        result.dr = 0.5 * sampled.b
            * (sampled.a - chi * inverse_p_squared);
        result.dlog_p_r =
            -0.5 * (sampled.a * sampled.b_r
                + sampled.b * sampled.a_r)
            -0.5 * sampled.b_r * chi * inverse_p_squared
            + angular_term;
        if (!std::isfinite(result.dr)
            || !std::isfinite(result.dlog_p_r)) {
            throw std::runtime_error(
                "EF log-momentum characteristic RHS became non-finite.");
        }
        return result;
    };

    LogState log_state{state.r, std::log(state.p_r)};
    const double h = dt / static_cast<double>(substeps);
    for (int step = 0; step < substeps; ++step) {
        const LogDerivative k1 = rhs(log_state);
        const LogDerivative k2 = rhs(add(log_state, 0.5 * h, k1));
        const LogDerivative k3 = rhs(add(log_state, 0.5 * h, k2));
        const LogDerivative k4 = rhs(add(log_state, h, k3));
        log_state.r += h
            * (k1.dr + 2.0 * k2.dr + 2.0 * k3.dr + k4.dr) / 6.0;
        log_state.log_p_r += h
            * (k1.dlog_p_r + 2.0 * k2.dlog_p_r
                + 2.0 * k3.dlog_p_r + k4.dlog_p_r) / 6.0;
        if (!(log_state.r >= 0.0) || !std::isfinite(log_state.r)
            || !(log_state.log_p_r > minimum_log_p)
            || !(log_state.log_p_r < maximum_log_p)
            || !std::isfinite(log_state.log_p_r)) {
            throw std::runtime_error(
                "EF log-momentum RK4 left its characteristic patch.");
        }
    }
    state.r = log_state.r;
    state.p_r = std::exp(log_state.log_p_r);
    if (!(state.p_r > 0.0) || !std::isfinite(state.p_r)) {
        throw std::runtime_error(
            "EF log-momentum RK4 failed to reconstruct positive p_r.");
    }
    return state;
}

}  // namespace detail

inline CharacteristicState rk4_advance(
    CharacteristicState state,
    double eta,
    double dt,
    int substeps,
    const WeightedPICFieldGrid& grid,
    const Metric& metric) {
    const MetricSampler sample(grid, metric);
    return detail::rk4_advance_with_sampler(state, eta, dt, substeps, sample);
}

inline void advance_frozen_metric_parallel(
    std::vector<Marker>& markers,
    double dt,
    const WeightedPICFieldGrid& grid,
    const Metric& metric,
    unsigned int requested_threads,
    int characteristic_substeps = 1,
    bool apply_existing_excision = true) {
    if (!(dt >= 0.0) || !std::isfinite(dt) || requested_threads == 0
        || characteristic_substeps < 1
        || markers.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("EF frozen-metric push received invalid inputs.");
    }
    const MetricSampler sample(grid, metric);
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        static_cast<int>(requested_threads),
        [&](int first, int last, int) {
            for (int q = first; q < last; ++q) {
                Marker& marker = markers[static_cast<std::size_t>(q)];
                detail::validate_marker(marker, "EF frozen-metric push");
                if (marker.state != MarkerState::active || dt == 0.0) {
                    continue;
                }
                if (apply_existing_excision && metric.excision_radius > 0.0
                    && marker.r <= metric.excision_radius) {
                    marker.state = MarkerState::excised;
                    continue;
                }
                const double invariant_weight = marker.rest_weight;
                const double invariant_eta = marker.eta;
                const std::uint64_t invariant_id = marker.id;
                const std::uint64_t invariant_source = marker.source_id;
                const CharacteristicState advanced = detail::rk4_advance_with_sampler(
                    {marker.r, marker.p_r},
                    marker.eta,
                    dt,
                    characteristic_substeps,
                    sample);
                marker.r = advanced.r;
                marker.p_r = advanced.p_r;
                if (marker.r > grid.mesh.edge.back()) {
                    marker.state = MarkerState::escaped;
                } else if (apply_existing_excision
                           && metric.excision_radius > 0.0
                           && marker.r <= metric.excision_radius) {
                    marker.state = MarkerState::excised;
                } else if (marker.r < grid.mesh.edge.front()) {
                    throw std::runtime_error(
                        "An interior EF marker crossed the inner field boundary.");
                }
                if (marker.rest_weight != invariant_weight
                    || marker.eta != invariant_eta || marker.id != invariant_id
                    || marker.source_id != invariant_source) {
                    throw std::runtime_error("EF push violated a marker invariant.");
                }
            }
        });
}

inline void advance_frozen_metric(
    std::vector<Marker>& markers,
    double dt,
    const WeightedPICFieldGrid& grid,
    const Metric& metric,
    int characteristic_substeps = 1,
    bool apply_existing_excision = true) {
    advance_frozen_metric_parallel(
        markers,
        dt,
        grid,
        metric,
        1,
        characteristic_substeps,
        apply_existing_excision);
}

inline ExcisionResult excise_markers(
    std::vector<Marker>& markers,
    double excision_radius) {
    if (!(excision_radius >= 0.0) || !std::isfinite(excision_radius)) {
        throw std::runtime_error("EF particle excision received an invalid radius.");
    }
    ExcisionResult result;
    result.radius = excision_radius;
    if (excision_radius == 0.0) {
        return result;
    }
    long double removed = 0.0L;
    for (Marker& marker : markers) {
        detail::validate_marker(marker, "EF particle excision");
        if (marker.state == MarkerState::active
            && marker.r <= excision_radius) {
            marker.state = MarkerState::excised;
            removed += marker.rest_weight;
            ++result.newly_excised_markers;
        }
    }
    result.newly_excised_rest_weight = static_cast<double>(removed);
    return result;
}

inline StepResult advance_self_consistent(
    std::vector<Marker>& markers,
    double dt,
    const WeightedPICFieldGrid& grid,
    const EvolutionOptions& options = {}) {
    if (!(dt >= 0.0) || !std::isfinite(dt) || options.threads == 0
        || options.characteristic_substeps < 1
        || options.excision_buffer_cells < 0
        || !(options.excision_a_threshold < 0.0)
        || !std::isfinite(options.excision_a_threshold)
        || !(options.excision_horizon_fraction > 0.0)
        || !(options.excision_horizon_fraction <= 1.0)) {
        throw std::runtime_error("Self-consistent EF WPIC step has invalid controls.");
    }
    StepResult result;
    if (options.require_vacuum_outer_boundary) {
        check_vacuum_outer_boundary(markers);
    }
    result.rest_weight_before = detail::total_rest_weight(markers);
    ConstraintOptions initial_constraints;
    initial_constraints.fixed_adm_mass = options.fixed_adm_mass;
    initial_constraints.excision_a_threshold = options.excision_a_threshold;
    initial_constraints.excision_buffer_cells = options.excision_buffer_cells;
    initial_constraints.excision_horizon_fraction = options.excision_horizon_fraction;
    initial_constraints.enable_excision = options.enable_excision;
    result.old_field = deposit_and_solve_field(
        markers, grid, options.shape, options.threads, initial_constraints);
    result.conserved_adm_mass = options.fixed_adm_mass.has_value()
        ? options.fixed_adm_mass.value()
        : result.old_field.metric.adm_mass;

    ConstraintOptions fixed_constraints = initial_constraints;
    fixed_constraints.fixed_adm_mass = result.conserved_adm_mass;

    // Predict particles for half a step in the old field, then solve their
    // constraint sources. The full orbit step starts again from the original particles
    // in this midpoint field, giving the self-consistent second-order field coupling.
    std::vector<Marker> midpoint_markers = markers;
    advance_frozen_metric_parallel(
        midpoint_markers,
        0.5 * dt,
        grid,
        result.old_field.metric,
        options.threads,
        options.characteristic_substeps,
        options.enable_excision);
    if (options.require_vacuum_outer_boundary) {
        check_vacuum_outer_boundary(midpoint_markers);
    }
    result.midpoint_field = deposit_and_solve_field(
        midpoint_markers,
        grid,
        options.shape,
        options.threads,
        fixed_constraints);

    std::vector<Marker> endpoint_markers = markers;
    advance_frozen_metric_parallel(
        endpoint_markers,
        dt,
        grid,
        result.midpoint_field.metric,
        options.threads,
        options.characteristic_substeps,
        options.enable_excision);
    if (options.require_vacuum_outer_boundary) {
        check_vacuum_outer_boundary(endpoint_markers);
    }
    double applied_excision_radius = 0.0;
    for (std::size_t q = 0; q < markers.size(); ++q) {
        if (markers[q].state != MarkerState::excised
            && endpoint_markers[q].state == MarkerState::excised) {
            applied_excision_radius = result.midpoint_field.metric.excision_radius;
            break;
        }
    }
    result.endpoint_field = deposit_and_solve_field(
        endpoint_markers,
        grid,
        options.shape,
        options.threads,
        fixed_constraints);

    if (options.enable_excision) {
        const ExcisionResult excision = excise_markers(
            endpoint_markers, result.endpoint_field.metric.excision_radius);
        if (excision.newly_excised_markers > 0) {
            applied_excision_radius = std::max(
                applied_excision_radius, excision.radius);
            result.endpoint_field = deposit_and_solve_field(
                endpoint_markers,
                grid,
                options.shape,
                options.threads,
                fixed_constraints);
        }
        retain_causal_excision_boundary(
            result.endpoint_field.metric, applied_excision_radius,
            fixed_constraints);
    }

    // Count every active->excised transition, including markers removed by an
    // already established excision surface at the start of the frozen push.
    long double newly_excised_weight = 0.0L;
    for (std::size_t q = 0; q < markers.size(); ++q) {
        if (markers[q].state != MarkerState::excised
            && endpoint_markers[q].state == MarkerState::excised) {
            newly_excised_weight += endpoint_markers[q].rest_weight;
            ++result.newly_excised_markers;
        }
    }
    result.newly_excised_rest_weight =
        static_cast<double>(newly_excised_weight);

    // Commit only the completed endpoint after source deposition and causal
    // excision checks. Retained particle records close the total rest-weight budget.
    markers = std::move(endpoint_markers);
    result.accepted_dt = dt;
    result.rest_weight_after = detail::total_rest_weight(markers);
    result.apparent_horizon_found =
        result.endpoint_field.metric.apparent_horizon_found;
    return result;
}

// Every active particle bounds the common step by radial cell travel, inner
// chart distance and canonical momentum variation. This selector uses the full
// active population, including particles of small positive rest weight.
inline double choose_timestep(
    const std::vector<Marker>& markers,
    const WeightedPICFieldGrid& grid,
    const Metric& metric,
    double cfl,
    double dt_max) {
    if (!(cfl > 0.0) || !(cfl <= 1.0)
        || !(dt_max > 0.0) || !std::isfinite(dt_max)) {
        throw std::runtime_error(
            "EF WPIC timestep requires 0<cfl<=1 and finite dt_max>0.");
    }
    const MetricSampler sample(grid, metric);
    double result = dt_max;
    for (const Marker& marker : markers) {
        detail::validate_marker(marker, "EF WPIC timestep");
        if (marker.state != MarkerState::active) {
            continue;
        }
        const double sampled_radius = std::clamp(
            marker.r, grid.mesh.edge.front(), grid.mesh.edge.back());
        const auto upper = std::upper_bound(
            grid.mesh.edge.begin(), grid.mesh.edge.end(), sampled_radius);
        const std::size_t cell = std::min(
            grid.mesh.shell_volume.size() - 1,
            static_cast<std::size_t>(std::max<std::ptrdiff_t>(
                0, upper - grid.mesh.edge.begin() - 1)));
        const double dr = grid.mesh.edge[cell + 1] - grid.mesh.edge[cell];
        const CharacteristicDerivative rhs = characteristic_rhs(
            {marker.r, marker.p_r}, marker.eta, sample(marker.r));
        double candidate = dt_max;
        if (rhs.dr != 0.0) {
            candidate = std::min(candidate, cfl * dr / std::abs(rhs.dr));
        }
        if (rhs.dr < 0.0 && metric.excision_radius == 0.0) {
            // Before an applied excision, resolve the remaining chart domain
            // even inside the first field cell. This adds no origin model.
            const double distance = marker.r - grid.mesh.edge.front();
            candidate = std::min(candidate, cfl * distance / std::abs(rhs.dr));
        }
        if (rhs.dp_r != 0.0) {
            candidate = std::min(
                candidate, cfl * marker.p_r / std::abs(rhs.dp_r));
            candidate = std::min(
                candidate,
                std::sqrt(2.0 * cfl * dr / std::abs(rhs.dp_r)));
        }
        if (!(candidate > 0.0) || !std::isfinite(candidate)) {
            throw std::runtime_error("EF WPIC timestep estimate became invalid.");
        }
        result = std::min(result, candidate);
    }
    return result;
}

}  // namespace eddington_finkelstein
}  // namespace ev
