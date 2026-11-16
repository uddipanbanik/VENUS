#pragma once

// Resolve radial characteristics near pericentres and finite reflecting walls.
// The caller supplies the physical equations dr/dt and dw/dt in a frozen field,
// an initial (r,w), an interval and error controls. For a near-centre retry,
// rho=sqrt(r) and dt=r*d_tau reduce the rapid physical-time variation. A separate
// signed-distance coordinate resolves specular wall impacts. The returned state
// includes convergence and event information, allowing the caller to accept the
// orbit or refine it. Both constructions retain the supplied physical force.

#include "ev/common.hpp"

namespace ev {

struct VPOrbitRegularizationDiagnostics {
    long long retries = 0;
    long long accepted = 0;
    long long unresolved = 0;
    long long wall_retries = 0;
    long long wall_accepted = 0;
    long long wall_unresolved = 0;
    double maximum_ordinary_energy_error = 0.0;
    double maximum_final_energy_error = 0.0;
};

inline std::atomic<long long> vp_orbit_lc_retries{0};
inline std::atomic<long long> vp_orbit_lc_accepted{0};
inline std::atomic<long long> vp_orbit_lc_unresolved{0};
inline std::atomic<long long> vp_orbit_wall_retries{0};
inline std::atomic<long long> vp_orbit_wall_accepted{0};
inline std::atomic<long long> vp_orbit_wall_unresolved{0};
inline std::atomic<double> vp_orbit_maximum_ordinary_energy_error{0.0};
inline std::atomic<double> vp_orbit_maximum_final_energy_error{0.0};

inline void vp_orbit_atomic_max_finite(
    std::atomic<double>& destination,
    double value) {
    if (!std::isfinite(value) || !(value >= 0.0)) {
        return;
    }
    double previous = destination.load(std::memory_order_relaxed);
    while (previous < value
           && !destination.compare_exchange_weak(
               previous,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

inline VPOrbitRegularizationDiagnostics vp_orbit_regularization_diagnostics() {
    return {
        vp_orbit_lc_retries.load(std::memory_order_relaxed),
        vp_orbit_lc_accepted.load(std::memory_order_relaxed),
        vp_orbit_lc_unresolved.load(std::memory_order_relaxed),
        vp_orbit_wall_retries.load(std::memory_order_relaxed),
        vp_orbit_wall_accepted.load(std::memory_order_relaxed),
        vp_orbit_wall_unresolved.load(std::memory_order_relaxed),
        vp_orbit_maximum_ordinary_energy_error.load(
            std::memory_order_relaxed),
        vp_orbit_maximum_final_energy_error.load(
            std::memory_order_relaxed),
    };
}

/// A two-dimensional characteristic state in spherical phase space.
struct CharacteristicState {
    double r = 0.0;
    double w = 0.0;
};

/// A two-dimensional characteristic right-hand side.
struct CharacteristicDerivative {
    double dr = 0.0;
    double dw = 0.0;
};

struct RegularizedOrbitResult {
    CharacteristicState state;
    int attempted_steps = 0;
    int accepted_steps = 0;
    bool converged = false;
    int wall_crossings = 0;
    double first_wall_crossing_time =
        std::numeric_limits<double>::infinity();
    double first_wall_crossing_velocity = 0.0;
};

/// Signed-distance state for a finite specular radial wall.
///
/// The physical radius is r_min+|x|.  The unfolded velocity u changes
/// continuously through x=0, while the physical radial velocity is
/// sign(x)u.  A crossing of x=0 therefore represents the wall reflection
/// without locating and restarting at a separate impact event.
struct ReflectingWallState {
    double x = 0.0;
    double u = 0.0;
};

struct ReflectingWallDerivative {
    double dx = 0.0;
    double du = 0.0;
};

struct ReflectingWallEmbeddedStep {
    ReflectingWallState fifth_order;
    ReflectingWallState fourth_order;
    bool finite = false;
};

struct LeviCivitaState {
    double rho = 0.0;
    double w = 0.0;
    double t = 0.0;
};

struct LeviCivitaDerivative {
    double drho = 0.0;
    double dw = 0.0;
    double dt = 0.0;
};

inline LeviCivitaState lc_scaled_add(
    const LeviCivitaState& y,
    double scale,
    const LeviCivitaDerivative& k) {
    return {
        y.rho + scale * k.drho,
        y.w + scale * k.dw,
        y.t + scale * k.dt,
    };
}

template <class PhysicalRHS>
// Apply the chain rule: d_rho/d_tau=(rho/2)*dr/dt, dw/d_tau=r*dw/dt,
// and dt/d_tau=r. Physical time is integrated as a third dependent variable.
LeviCivitaDerivative levi_civita_rhs(
    const LeviCivitaState& y,
    PhysicalRHS&& physical_rhs) {
    const double r = y.rho * y.rho;
    const CharacteristicDerivative physical = physical_rhs({r, y.w});
    return {
        0.5 * y.rho * physical.dr,
        r * physical.dw,
        r,
    };
}

struct LeviCivitaEmbeddedStep {
    LeviCivitaState fifth_order;
    LeviCivitaState fourth_order;
    bool finite = false;
};

template <class PhysicalRHS>
// The shared seven stages produce embedded fifth- and fourth-order states.
// Their difference estimates the local error in radius, momentum and elapsed time.
LeviCivitaEmbeddedStep levi_civita_dopri54_step(
    const LeviCivitaState& y,
    double h,
    PhysicalRHS&& physical_rhs) {
    auto rhs = [&](const LeviCivitaState& z) {
        return levi_civita_rhs(z, physical_rhs);
    };
    auto combine = [&](std::initializer_list<std::pair<double, const LeviCivitaDerivative*>> terms) {
        LeviCivitaState out = y;
        for (const auto& term : terms) {
            out = lc_scaled_add(out, h * term.first, *term.second);
        }
        return out;
    };

    const LeviCivitaDerivative k1 = rhs(y);
    const LeviCivitaDerivative k2 = rhs(combine({
        {1.0 / 5.0, &k1},
    }));
    const LeviCivitaDerivative k3 = rhs(combine({
        {3.0 / 40.0, &k1}, {9.0 / 40.0, &k2},
    }));
    const LeviCivitaDerivative k4 = rhs(combine({
        {44.0 / 45.0, &k1}, {-56.0 / 15.0, &k2}, {32.0 / 9.0, &k3},
    }));
    const LeviCivitaDerivative k5 = rhs(combine({
        {19372.0 / 6561.0, &k1}, {-25360.0 / 2187.0, &k2},
        {64448.0 / 6561.0, &k3}, {-212.0 / 729.0, &k4},
    }));
    const LeviCivitaDerivative k6 = rhs(combine({
        {9017.0 / 3168.0, &k1}, {-355.0 / 33.0, &k2},
        {46732.0 / 5247.0, &k3}, {49.0 / 176.0, &k4},
        {-5103.0 / 18656.0, &k5},
    }));
    const LeviCivitaState fifth = combine({
        {35.0 / 384.0, &k1}, {500.0 / 1113.0, &k3},
        {125.0 / 192.0, &k4}, {-2187.0 / 6784.0, &k5},
        {11.0 / 84.0, &k6},
    });
    const LeviCivitaDerivative k7 = rhs(fifth);
    const LeviCivitaState fourth = combine({
        {5179.0 / 57600.0, &k1}, {7571.0 / 16695.0, &k3},
        {393.0 / 640.0, &k4}, {-92097.0 / 339200.0, &k5},
        {187.0 / 2100.0, &k6}, {1.0 / 40.0, &k7},
    });

    const bool finite =
        std::isfinite(fifth.rho) && std::isfinite(fifth.w) && std::isfinite(fifth.t)
        && std::isfinite(fourth.rho) && std::isfinite(fourth.w) && std::isfinite(fourth.t);
    return {fifth, fourth, finite};
}

// Scale each component by absolute plus relative tolerance before combining
// errors. The clock scale refers to the requested physical interval.
inline double levi_civita_error_norm(
    const LeviCivitaEmbeddedStep& step,
    const LeviCivitaState& start,
    double physical_dt,
    double relative_tolerance,
    double absolute_tolerance) {
    const double rho_scale = absolute_tolerance + relative_tolerance
        * std::max(std::abs(start.rho), std::abs(step.fifth_order.rho));
    const double w_scale = absolute_tolerance + relative_tolerance
        * std::max(std::abs(start.w), std::abs(step.fifth_order.w));
    const double t_scale = absolute_tolerance + relative_tolerance
        * std::max(std::abs(physical_dt), std::abs(step.fifth_order.t));
    const double erho =
        (step.fifth_order.rho - step.fourth_order.rho) / rho_scale;
    const double ew = (step.fifth_order.w - step.fourth_order.w) / w_scale;
    const double et = (step.fifth_order.t - step.fourth_order.t) / t_scale;
    return std::sqrt((erho * erho + ew * ew + et * et) / 3.0);
}

template <class PhysicalRHS>
RegularizedOrbitResult advance_radial_characteristic_levi_civita(
    CharacteristicState initial,
    double physical_dt,
    PhysicalRHS&& physical_rhs,
    double relative_tolerance,
    double absolute_tolerance,
    int max_steps) {
    RegularizedOrbitResult result;
    result.state = initial;
    if (physical_dt == 0.0) {
        result.converged = true;
        return result;
    }
    if (!(initial.r > 0.0) || !std::isfinite(initial.r) || !std::isfinite(initial.w)) {
        return result;
    }

    LeviCivitaState y{std::sqrt(initial.r), initial.w, 0.0};
    const double direction = (physical_dt > 0.0) ? 1.0 : -1.0;
    const double time_tolerance = std::max(
        32.0 * std::numeric_limits<double>::epsilon() * std::abs(physical_dt),
        0.1 * absolute_tolerance);
    double h = physical_dt / initial.r;

    for (int n = 0; n < max_steps; ++n) {
        const double remaining = physical_dt - y.t;
        if (direction * remaining <= time_tolerance) {
            result.state = {y.rho * y.rho, y.w};
            result.converged = true;
            return result;
        }

        const double r_now = std::max(y.rho * y.rho, 1.0e-300);
        const double target_h = remaining / r_now;
        if (!std::isfinite(h) || direction * h <= 0.0) {
            h = target_h;
        }
        if (std::abs(h) > 1.5 * std::abs(target_h)) {
            h = 1.5 * target_h;
        }
        const LeviCivitaDerivative local_rhs =
            levi_civita_rhs(y, physical_rhs);
        if (std::abs(local_rhs.drho) > 0.0) {
            const double rho_step_limit =
                0.01 * std::max(std::abs(y.rho), 1.0e-150)
                / std::abs(local_rhs.drho);
            if (std::abs(h) > rho_step_limit) {
                h = direction * rho_step_limit;
            }
        }

        ++result.attempted_steps;
        const LeviCivitaEmbeddedStep trial =
            levi_civita_dopri54_step(y, h, physical_rhs);
        double error = trial.finite
            ? levi_civita_error_norm(
                trial, y, physical_dt, relative_tolerance, absolute_tolerance)
            : std::numeric_limits<double>::infinity();

        if (trial.finite && error <= 1.0) {
            const bool reaches_target =
                direction * (trial.fifth_order.t - physical_dt) >= 0.0;
            // Locate the requested physical endpoint inside the accepted fictitious-time
            // step. Reintegrating partial steps preserves the physical clock used by the orbit.
            if (reaches_target) {
                double lo = 0.0;
                double hi = h;
                LeviCivitaState final_state = trial.fifth_order;
                for (int iteration = 0; iteration < 48; ++iteration) {
                    const double mid = 0.5 * (lo + hi);
                    const LeviCivitaEmbeddedStep partial =
                        levi_civita_dopri54_step(y, mid, physical_rhs);
                    if (!partial.finite) {
                        hi = mid;
                        continue;
                    }
                    final_state = partial.fifth_order;
                    const double delta = final_state.t - physical_dt;
                    if (std::abs(delta) <= time_tolerance) {
                        break;
                    }
                    if (direction * delta < 0.0) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                result.state = {
                    final_state.rho * final_state.rho,
                    final_state.w,
                };
                ++result.accepted_steps;
                result.converged = true;
                return result;
            }

            y = trial.fifth_order;
            ++result.accepted_steps;
            const double growth = (error > 0.0)
                ? 0.9 * std::pow(error, -0.2)
                : 5.0;
            h *= std::clamp(growth, 0.2, 5.0);
            continue;
        }

        const double shrink = std::isfinite(error) && error > 0.0
            ? 0.9 * std::pow(error, -0.2)
            : 0.2;
        h *= std::clamp(shrink, 0.1, 0.5);
        if (std::abs(h) <= std::numeric_limits<double>::min() * 1024.0) {
            break;
        }
    }

    result.state = {y.rho * y.rho, y.w};
    return result;
}

/// Frank/Tokayer partial Levi-Civita leapfrog retry.
///
/// This follows the algorithm cited in their Section 2.2: rho=sqrt(r),
/// dt=r d tau, KDK in fictitious time, midpoint accumulation of physical
/// time, and linear interpolation of the final overshooting tau step.  The
/// caller repeats the complete physical step with a larger
/// `minimum_tau_steps` until Frank & Tokayer's full-potential energy test
/// accepts it.  Keeping this separate from the generic Dormand-Prince LC
/// routine prevents changes to the VP and grid-BP solvers.
template <class PhysicalRHS>
// This distinct leapfrog advances kick--drift--kick in fictitious time. The
// caller controls its energy accuracy by repeating the whole interval at smaller
// delta_tau; the physical clock uses the midpoint value of rho squared.
RegularizedOrbitResult advance_radial_characteristic_levi_civita_leapfrog(
    CharacteristicState initial,
    double physical_dt,
    PhysicalRHS&& physical_rhs,
    int minimum_tau_steps,
    int max_steps) {
    RegularizedOrbitResult result;
    result.state = initial;
    if (physical_dt == 0.0) {
        result.converged = true;
        return result;
    }
    if (!(physical_dt > 0.0) || !(initial.r > 0.0)
        || !std::isfinite(initial.r) || !std::isfinite(initial.w)
        || minimum_tau_steps < 1 || max_steps < 1) {
        return result;
    }

    LeviCivitaState state{std::sqrt(initial.r), initial.w, 0.0};
    // This is the initial Delta-tau prescription used by the referenced
    // partial-LC leapfrog.  Doubling minimum_tau_steps halves Delta-tau.
    const double delta_tau = physical_dt
        / (2.0 * initial.r * static_cast<double>(minimum_tau_steps));
    if (!(delta_tau > 0.0) || !std::isfinite(delta_tau)) {
        return result;
    }
    const double time_tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * physical_dt;

    for (int step = 0; step < max_steps; ++step) {
        const double remaining = physical_dt - state.t;
        if (remaining <= time_tolerance) {
            result.state = {state.rho * state.rho, state.w};
            result.converged = true;
            return result;
        }

        ++result.attempted_steps;
        const LeviCivitaState before = state;
        const LeviCivitaDerivative first =
            levi_civita_rhs(before, physical_rhs);
        const double half_velocity =
            before.w + 0.5 * delta_tau * first.dw;
        const double rho_after = before.rho
            + delta_tau * 0.5 * before.rho * half_velocity;
        if (!std::isfinite(half_velocity) || !std::isfinite(rho_after)) {
            return result;
        }
        LeviCivitaState drifted{rho_after, half_velocity, before.t};
        const LeviCivitaDerivative second =
            levi_civita_rhs(drifted, physical_rhs);
        const double velocity_after =
            half_velocity + 0.5 * delta_tau * second.dw;
        const double rho_midpoint = 0.5 * (before.rho + rho_after);
        const double physical_increment =
            delta_tau * rho_midpoint * rho_midpoint;
        if (!std::isfinite(velocity_after)
            || !(physical_increment > 0.0)
            || !std::isfinite(physical_increment)) {
            return result;
        }

        const double time_after = before.t + physical_increment;
        ++result.accepted_steps;
        if (time_after >= physical_dt) {
            const double fraction = std::clamp(
                remaining / physical_increment,
                0.0,
                1.0);
            const double final_rho = before.rho
                + fraction * (rho_after - before.rho);
            const double final_velocity = before.w
                + fraction * (velocity_after - before.w);
            result.state = {
                final_rho * final_rho,
                final_velocity,
            };
            result.converged = std::isfinite(result.state.r)
                && result.state.r > 0.0
                && std::isfinite(result.state.w);
            return result;
        }

        state = {rho_after, velocity_after, time_after};
    }

    result.state = {state.rho * state.rho, state.w};
    return result;
}

inline ReflectingWallState reflecting_wall_scaled_add(
    const ReflectingWallState& y,
    double scale,
    const ReflectingWallDerivative& k) {
    return {
        y.x + scale * k.dx,
        y.u + scale * k.du,
    };
}

template <class UnfoldedRHS>
// Integrate the unfolded position and velocity with an embedded error estimate.
// The wall reflection appears when the signed position changes side.
ReflectingWallEmbeddedStep reflecting_wall_dopri54_step(
    const ReflectingWallState& y,
    double h,
    UnfoldedRHS&& rhs) {
    auto combine = [&](std::initializer_list<
                       std::pair<double, const ReflectingWallDerivative*>> terms) {
        ReflectingWallState out = y;
        for (const auto& term : terms) {
            out = reflecting_wall_scaled_add(
                out, h * term.first, *term.second);
        }
        return out;
    };

    const ReflectingWallDerivative k1 = rhs(y);
    const ReflectingWallDerivative k2 = rhs(combine({
        {1.0 / 5.0, &k1},
    }));
    const ReflectingWallDerivative k3 = rhs(combine({
        {3.0 / 40.0, &k1}, {9.0 / 40.0, &k2},
    }));
    const ReflectingWallDerivative k4 = rhs(combine({
        {44.0 / 45.0, &k1}, {-56.0 / 15.0, &k2},
        {32.0 / 9.0, &k3},
    }));
    const ReflectingWallDerivative k5 = rhs(combine({
        {19372.0 / 6561.0, &k1}, {-25360.0 / 2187.0, &k2},
        {64448.0 / 6561.0, &k3}, {-212.0 / 729.0, &k4},
    }));
    const ReflectingWallDerivative k6 = rhs(combine({
        {9017.0 / 3168.0, &k1}, {-355.0 / 33.0, &k2},
        {46732.0 / 5247.0, &k3}, {49.0 / 176.0, &k4},
        {-5103.0 / 18656.0, &k5},
    }));
    const ReflectingWallState fifth = combine({
        {35.0 / 384.0, &k1}, {500.0 / 1113.0, &k3},
        {125.0 / 192.0, &k4}, {-2187.0 / 6784.0, &k5},
        {11.0 / 84.0, &k6},
    });
    const ReflectingWallDerivative k7 = rhs(fifth);
    const ReflectingWallState fourth = combine({
        {5179.0 / 57600.0, &k1}, {7571.0 / 16695.0, &k3},
        {393.0 / 640.0, &k4}, {-92097.0 / 339200.0, &k5},
        {187.0 / 2100.0, &k6}, {1.0 / 40.0, &k7},
    });
    const bool finite =
        std::isfinite(fifth.x) && std::isfinite(fifth.u)
        && std::isfinite(fourth.x) && std::isfinite(fourth.u);
    return {fifth, fourth, finite};
}

inline double reflecting_wall_error_norm(
    const ReflectingWallEmbeddedStep& step,
    const ReflectingWallState& start,
    double relative_tolerance,
    double absolute_tolerance) {
    const double x_scale = absolute_tolerance
        + relative_tolerance
            * std::max(
                std::abs(start.x),
                std::abs(step.fifth_order.x));
    const double u_scale = absolute_tolerance
        + relative_tolerance
            * std::max(
                std::abs(start.u),
                std::abs(step.fifth_order.u));
    const double ex =
        (step.fifth_order.x - step.fourth_order.x) / x_scale;
    const double eu =
        (step.fifth_order.u - step.fourth_order.u) / u_scale;
    return std::sqrt(0.5 * (ex * ex + eu * eu));
}

/// Integrates a characteristic that may hit a finite reflecting face.
///
/// The signed-distance continuation turns a specular impact into an ordinary
/// crossing of x=0.  Dormand--Prince adaptivity resolves the force cusp at the
/// crossing and the centrifugal stiffness on either side.  No energy
/// projection or post-step correction is applied.  Event-driven callers can
/// request an immediate return at the first crossing; the returned state is
/// then located on r_min with the physical incoming radial velocity.
template <class PhysicalRHS>
RegularizedOrbitResult advance_radial_characteristic_reflecting_wall(
    CharacteristicState initial,
    double physical_dt,
    double r_min,
    PhysicalRHS&& physical_rhs,
    double relative_tolerance,
    double absolute_tolerance,
    int max_steps,
    bool stop_at_first_wall_crossing = false) {
    RegularizedOrbitResult result;
    result.state = initial;
    if (physical_dt == 0.0) {
        result.converged = true;
        return result;
    }
    const double boundary_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(r_min));
    if (!std::isfinite(initial.r) || !std::isfinite(initial.w)
        || initial.r < r_min - boundary_tolerance) {
        return result;
    }

    ReflectingWallState y{
        std::max(initial.r, r_min) - r_min,
        initial.w,
    };
    // For r=r_min+abs(x), physical w=sign(x)*u. Transform both derivatives with
    // the same sign so continuation through x=0 implements a specular reflection.
    auto unfolded_rhs = [&](const ReflectingWallState& z) {
        const double side = z.x < 0.0 ? -1.0 : 1.0;
        const CharacteristicState physical{
            r_min + std::abs(z.x),
            side * z.u,
        };
        const CharacteristicDerivative derivative = physical_rhs(physical);
        return ReflectingWallDerivative{
            side * derivative.dr,
            side * derivative.dw,
        };
    };
    auto physical_state = [&](const ReflectingWallState& z) {
        const double side = z.x < 0.0 ? -1.0 : 1.0;
        return CharacteristicState{
            r_min + std::abs(z.x),
            side * z.u,
        };
    };

    const double direction = physical_dt > 0.0 ? 1.0 : -1.0;
    const double time_tolerance =
        32.0 * std::numeric_limits<double>::epsilon()
        * std::abs(physical_dt);
    double elapsed = 0.0;
    double h = physical_dt;
    for (int n = 0; n < max_steps; ++n) {
        const double remaining = physical_dt - elapsed;
        if (direction * remaining <= time_tolerance) {
            result.state = physical_state(y);
            result.converged = true;
            return result;
        }
        if (!std::isfinite(h) || direction * h <= 0.0) {
            h = remaining;
        }
        if (std::abs(h) > std::abs(remaining)) {
            h = remaining;
        }
        // Keep a wall-directed trial from crossing the unfolded face and
        // returning to the same side inside one accepted RK step.  This makes
        // every impact visible to the event ledger below.
        if (y.x != 0.0 && y.u != 0.0 && y.x * (h * y.u) < 0.0) {
            const double crossing_scale = std::abs(y.x / y.u);
            if (std::abs(h) > 1.25 * crossing_scale) {
                h = direction * 1.25 * crossing_scale;
            }
        }
        // A trial that starts at the wall and moves outward can otherwise
        // include the outer turning point and a second wall impact in one
        // accepted step.  Estimate the return time from the local unfolded
        // acceleration and stop well before that impact.  Once the orbit is
        // wall-directed, the crossing cap above gives the event detector a
        // bracket.  The estimate controls only the step size; the accepted
        // state still comes from the adaptive Dormand--Prince integration.
        ReflectingWallDerivative local_derivative{0.0, 0.0};
        if (stop_at_first_wall_crossing) {
            local_derivative = unfolded_rhs(y);
        }
        const double path_velocity = direction * y.u;
        if (stop_at_first_wall_crossing
            && std::isfinite(local_derivative.du)
            && (y.x != 0.0 || path_velocity >= 0.0)) {
            const double side = y.x < 0.0 ? -1.0 : 1.0;
            const double distance = std::abs(y.x);
            const double distance_velocity = side * path_velocity;
            const double distance_acceleration =
                side * local_derivative.du;
            if (distance_velocity >= 0.0
                && distance_acceleration < 0.0) {
                const double inward_acceleration = -distance_acceleration;
                const double discriminant =
                    distance_velocity * distance_velocity
                    + 2.0 * inward_acceleration * distance;
                const double estimated_return_time =
                    (distance_velocity + std::sqrt(discriminant))
                    / inward_acceleration;
                if (estimated_return_time > 0.0
                    && std::isfinite(estimated_return_time)
                    && std::abs(h) > 0.5 * estimated_return_time) {
                    h = direction * 0.5 * estimated_return_time;
                }
            }
        }

        ++result.attempted_steps;
        const ReflectingWallEmbeddedStep trial =
            reflecting_wall_dopri54_step(y, h, unfolded_rhs);
        const bool skipped_return_from_face =
            stop_at_first_wall_crossing
            && trial.finite
            && y.x == 0.0
            && h * y.u > 0.0
            && trial.fifth_order.x < 0.0;
        const double error = trial.finite && !skipped_return_from_face
            ? reflecting_wall_error_norm(
                trial, y, relative_tolerance, absolute_tolerance)
            : std::numeric_limits<double>::infinity();
        if (trial.finite && error <= 1.0) {
            const double old_x = y.x;
            const double new_x = trial.fifth_order.x;
            // Count a wall event both for a bracketed crossing and for inward motion
            // starting exactly on the face. The first event can terminate an event-driven call.
            const bool starts_through_wall =
                old_x == 0.0
                && (h * y.u < 0.0
                    || (y.u == 0.0 && local_derivative.du < 0.0));
            const bool changes_side =
                (old_x > 0.0 && new_x < 0.0)
                || (old_x < 0.0 && new_x > 0.0);
            if (starts_through_wall || changes_side) {
                ++result.wall_crossings;
                if (!std::isfinite(result.first_wall_crossing_time)) {
                    double crossing_offset = 0.0;
                    double crossing_u = y.u;
                    if (!starts_through_wall) {
                        double lo = 0.0;
                        double hi = h;
                        const bool old_positive = old_x > 0.0;
                        // Bisect the accepted wall-crossing step, evaluating the same adaptive
                        // characteristic stages at each trial time to locate the impact velocity.
                        for (int iteration = 0; iteration < 56; ++iteration) {
                            const double mid = 0.5 * (lo + hi);
                            const ReflectingWallEmbeddedStep partial =
                                reflecting_wall_dopri54_step(
                                    y, mid, unfolded_rhs);
                            if (!partial.finite) {
                                hi = mid;
                                continue;
                            }
                            const bool same_side = old_positive
                                ? partial.fifth_order.x > 0.0
                                : partial.fifth_order.x < 0.0;
                            if (same_side) {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        crossing_offset = 0.5 * (lo + hi);
                        const ReflectingWallEmbeddedStep crossing =
                            reflecting_wall_dopri54_step(
                                y, crossing_offset, unfolded_rhs);
                        if (crossing.finite) {
                            crossing_u = crossing.fifth_order.u;
                        }
                    }
                    result.first_wall_crossing_time =
                        elapsed + crossing_offset;
                    const double crossing_side = old_x < 0.0 ? -1.0 : 1.0;
                    result.first_wall_crossing_velocity =
                        crossing_side * crossing_u;
                    if (stop_at_first_wall_crossing) {
                        result.state = {
                            r_min,
                            result.first_wall_crossing_velocity,
                        };
                        result.converged = true;
                        return result;
                    }
                }
            }
            y = trial.fifth_order;
            elapsed += h;
            ++result.accepted_steps;
            const double growth = error > 0.0
                ? 0.9 * std::pow(error, -0.2)
                : 5.0;
            h *= std::clamp(growth, 0.2, 5.0);
            continue;
        }

        const double shrink = std::isfinite(error) && error > 0.0
            ? 0.9 * std::pow(error, -0.2)
            : 0.2;
        h *= std::clamp(shrink, 0.1, 0.5);
        if (std::abs(h)
            <= std::numeric_limits<double>::min() * 1024.0) {
            break;
        }
    }

    result.state = physical_state(y);
    return result;
}

}  // namespace ev
