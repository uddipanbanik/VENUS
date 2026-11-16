#pragma once

// Advance the exactly soluble spherical drift between gravitational kicks.
// Inputs are physical radius r, peculiar radial velocity w, specific angular
// momentum L, the background time tau and a power-law index alpha, for which
// H=alpha/tau. Outputs replace r,w after a signed interval and any declared
// specular reflections at fixed physical walls. CPU and CUDA use this same
// formula, including the regular straight-through radial orbit at L=0.

#include <cmath>
#include <limits>

#ifdef __CUDACC__
#define VENUS_DRIFT_HD __host__ __device__
#else
#define VENUS_DRIFT_HD
#endif

namespace ev {
namespace central_drift_detail {

struct State {
    double r;
    double w; // Peculiar radial velocity; physical velocity is w + alpha*r/tau.
};

// Invert the transformed drift coordinate to recover log(tau_new/tau).
// At alpha=1/2 the expression has its analytic logarithmic limit.
VENUS_DRIFT_HD inline double log_time_ratio(double h, double alpha) {
    const double d = 1.0 - 2.0 * alpha;
    return d == 0.0 ? h : std::log1p(d * h) / d;
}

// Integrate the power-law background drift into h. log1p and expm1
// resolve small dt/tau and the continuous alpha=1/2 limit.
VENUS_DRIFT_HD inline double drift_coordinate(double dt, double tau, double alpha) {
    const double log_ratio = std::log1p(dt / tau);
    const double d = 1.0 - 2.0 * alpha;
    return d == 0.0 ? log_ratio : std::expm1(d * log_ratio) / d;
}

// h = [(tau_new/tau)^(1-2 alpha)-1]/(1-2 alpha), with the logarithmic
// limit at alpha=1/2. The comoving displacement is proportional to tau*h.
VENUS_DRIFT_HD inline State at_h(
    double r, double w, double L, double tau, double alpha, double h) {
    const double scale = std::exp(alpha * log_time_ratio(h, alpha));
    if (L == 0.0) {
        // A radial orbit is a straight signed coordinate through the origin.
        // At an exact origin endpoint choose the velocity after the crossing
        // in the direction of integration; h=0 preserves the supplied state.
        const double signed_radius = r + tau * h * w;
        const double radial_velocity = signed_radius > 0.0 ? w
            : (signed_radius < 0.0 ? -w
                : (h == 0.0 ? w : std::copysign(std::fabs(w), h)));
        return {scale * std::fabs(signed_radius), radial_velocity / scale};
    }
    // For L>0 the two Cartesian components form a straight drift in the
    // rescaled coordinates. Taking their norm recovers a positive spherical
    // radius; projection along that direction recovers its radial velocity.
    const double tangential = L / r;
    const double radial_axis = r + tau * h * w;
    const double tangential_axis = tau * h * tangential;
    const double norm = std::hypot(radial_axis, tangential_axis);
    return {scale * norm, norm > 0.0
        ? (w * radial_axis + tangential * tangential_axis) / (scale * norm)
        : 0.0};
}

// The sign of dr/dt is the sign of this quadratic in h:
// Q = alpha*r^2 + tau*r*w + (tau*r*w + tau^2*v^2)*h
//     + (1-alpha)*tau^2*v^2*h^2, v^2=w^2+(L/r)^2.
// Its at most two roots partition a drift into monotonic radial intervals.
// Inspecting every interval also detects an inner-wall crossing followed by
// an outward return during the same drift.
VENUS_DRIFT_HD inline int turning_points(
    double r, double w, double L, double tau, double alpha,
    double end_h, double* points) {
    const double dimensional_radial = tau * w;
    const double dimensional_tangential = L == 0.0 ? 0.0 : tau * L / r;
    // Scale the turning-point polynomial before evaluating its roots.
    // This preserves the root locations while limiting large squared values.
    const double normalization = std::fmax(r,
        std::fmax(std::fabs(dimensional_radial), std::fabs(dimensional_tangential)));
    if (!std::isfinite(normalization)) return -1;
    if (normalization == 0.0) return 0;
    r /= normalization;
    const double radial = dimensional_radial / normalization;
    const double tangential = dimensional_tangential / normalization;
    const double velocity2 = radial * radial + tangential * tangential;
    double a = (1.0 - alpha) * velocity2;
    double b = r * radial + velocity2;
    double c = alpha * r * r + r * radial;
    const double scale = std::fmax(std::fabs(a),
        std::fmax(std::fabs(b), std::fabs(c)));
    if (!std::isfinite(scale)) return -1;
    if (!(scale > 0.0)) return 0;
    a /= scale; b /= scale; c /= scale;
    double roots[2];
    int roots_count = 0;
    if (a == 0.0) {
        if (b != 0.0) roots[roots_count++] = -c / b;
    } else {
        const double discriminant = b * b - 4.0 * a * c;
        if (discriminant >= 0.0) {
            const double q = -0.5 * (b + std::copysign(std::sqrt(discriminant), b));
            if (q != 0.0) {
                roots[roots_count++] = q / a;
                roots[roots_count++] = c / q;
            } else {
                roots[roots_count++] = -b / (2.0 * a);
            }
        }
    }
    int count = 0;
    for (int i = 0; i < roots_count; ++i) {
        const double value = roots[i];
        if (std::isfinite(value) && value * end_h > 0.0
            && std::fabs(value) < std::fabs(end_h)) points[count++] = value;
    }
    if (count == 2 && std::fabs(points[1]) < std::fabs(points[0])) {
        const double temporary = points[0]; points[0] = points[1]; points[1] = temporary;
    }
    return count;
}

} // namespace central_drift_detail

// Advance the complete free angular drift between gravitational kicks.
// At an impact, reverse the physical normal velocity at that time and
// continue the unused interval. L and the tangential velocity at the wall
// remain unchanged. For alpha=0 this conserves kinetic energy exactly up to
// roundoff. Signed intervals support backward characteristic tracing.
// Returns false on an invalid state or more than 64 wall events.
VENUS_DRIFT_HD inline bool advance_central_drift_with_walls(
    double& r, double& w, double L, double dt, double tau, double alpha,
    double inner, bool reflecting_inner, double outer, bool reflecting_outer,
    int* event_count = nullptr) {
    using namespace central_drift_detail;
    if (!(r >= 0.0) || (r == 0.0 && (L != 0.0 || reflecting_inner))
        || !(L >= 0.0) || !(tau > 0.0)
        || !(tau + dt > 0.0) || !(alpha >= 0.0)
        || !std::isfinite(r) || !std::isfinite(w) || !std::isfinite(L)
        || !std::isfinite(dt) || !std::isfinite(tau) || !std::isfinite(alpha)
        || (reflecting_inner && (!(inner > 0.0) || r < inner))
        || (reflecting_outer && (!(outer > inner) || r > outer))) return false;
    // For a constant or linear scale factor, zero physical velocity and
    // L=0 give exactly constant physical radius. Preserve this solution at
    // a wall without interpreting roundoff in a*|x| as repeated impacts.
    if (L == 0.0 && (alpha == 0.0 || alpha == 1.0)
        && w + alpha * r / tau == 0.0) {
        w = -alpha * r / (tau + dt);
        if (event_count) *event_count = 0;
        return true;
    }
    int events = 0;
    double remaining = dt;
    while (remaining != 0.0) {
        const double direction = remaining > 0.0 ? 1.0 : -1.0;
        // A reflecting wall reverses dr/dt=w+H*r at the actual impact time.
        // Therefore the peculiar velocity changes as w_new=-w_old-2*H*r;
        // reversing w alone would move a physical wall with the background.
        const double physical_velocity = w + alpha * r / tau;
        if ((reflecting_inner && r == inner && direction * physical_velocity < 0.0)
            || (reflecting_outer && r == outer && direction * physical_velocity > 0.0)) {
            if (++events > 64) return false;
            w = -w - 2.0 * alpha * r / tau;
        }
        const double end_h = drift_coordinate(remaining, tau, alpha);
        if (!std::isfinite(end_h)) return false;
        double points[3];
        // Partition the interval at every radial turning point before searching
        // for a wall crossing. A trajectory can hit an inner wall and return
        // outward even when both ends of the whole drift lie inside the domain.
        const int turns = turning_points(r, w, L, tau, alpha, end_h, points);
        if (turns < 0) return false;
        points[turns] = end_h;
        double left = 0.0;
        bool hit = false;
        double wall = 0.0;
        double hit_h = 0.0;
        for (int i = 0; i <= turns; ++i) {
            const double right = points[i];
            const State endpoint = at_h(r, w, L, tau, alpha, right);
            if (!std::isfinite(endpoint.r)) return false;
            const double endpoint_tau = tau * std::exp(log_time_ratio(right, alpha));
            const double directed_speed = direction
                * (endpoint.w + alpha * endpoint.r / endpoint_tau);
            // A stationary point on the wall is already a valid endpoint.
            // An endpoint impact requires motion into the excluded region.
            const bool inner_hit = reflecting_inner && (endpoint.r < inner
                || (endpoint.r == inner && directed_speed < 0.0));
            const bool outer_hit = reflecting_outer && (endpoint.r > outer
                || (endpoint.r == outer && directed_speed > 0.0));
            if (inner_hit || outer_hit) {
                wall = inner_hit ? inner : outer;
                double lo = left, hi = right;
                // The radius is monotone on this partition, so bisection brackets the
                // first physical wall event without stepping across an unresolved turn.
                for (int iteration = 0; iteration < 80; ++iteration) {
                    const double mid = lo + 0.5 * (hi - lo);
                    if (mid == lo || mid == hi) break;
                    const double radius = at_h(r, w, L, tau, alpha, mid).r;
                    if (!std::isfinite(radius)) return false;
                    if (inner_hit ? radius > wall : radius < wall) lo = mid;
                    else hi = mid;
                }
                hit_h = lo + 0.5 * (hi - lo);
                hit = true;
                break;
            }
            left = right;
        }
        if (!hit) {
            const State endpoint = at_h(r, w, L, tau, alpha, end_h);
            if (!(endpoint.r >= 0.0) || !std::isfinite(endpoint.w)
                || (endpoint.r == 0.0 && (L != 0.0 || reflecting_inner))) return false;
            r = endpoint.r; w = endpoint.w;
            remaining = 0.0;
            break;
        }
        if (++events > 64) return false;
        const State impact = at_h(r, w, L, tau, alpha, hit_h);
        // Convert the impact coordinate back to physical elapsed time, reflect
        // at the wall, and continue the unused interval with the updated tau.
        const double elapsed = tau * std::expm1(log_time_ratio(hit_h, alpha));
        if (!std::isfinite(elapsed) || elapsed * remaining < 0.0) return false;
        tau += elapsed;
        // The root lies in this interval; cap its roundoff at the endpoint.
        remaining = std::fabs(elapsed) >= std::fabs(remaining) ? 0.0 : remaining - elapsed;
        r = wall;
        w = -impact.w - 2.0 * alpha * wall / tau;
        if (!std::isfinite(w)) return false;
    }
    if (event_count) *event_count = events;
    return true;
}

} // namespace ev

#undef VENUS_DRIFT_HD
