#pragma once

// Schedule field reconstruction and Boltzmann collision batches using
// separate elapsed coordinate-time ages. Each physical timescale becomes
// a deadline through its configured safety fraction. The remaining time
// before either deadline bounds the frozen-field orbital interval.
// Inputs are carried ages and newly evaluated clocks. Outputs are an event
// plan or an endpoint accept/retry decision. The caller owns state rollback,
// actual field reconstruction and collision execution.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ev {

enum class WeightedPICEventOrder {
    none,
    reconstruction_only,
    collision_then_reconstruction,
    reconstruction_then_collision,
};

struct WeightedPICEventClockState {
    double potential_age = 0.0;
    double collision_age = 0.0;
};

struct WeightedPICEventPlan {
    double orbit_interval = 0.0;
    double potential_interval = std::numeric_limits<double>::infinity();
    double collision_interval = std::numeric_limits<double>::infinity();
    bool potential_due = false;
    bool collision_due = false;
    WeightedPICEventOrder order = WeightedPICEventOrder::none;
};

/// Decision made after a provisional orbit step has exposed the endpoint
/// collision, dynamical, and potential-change clocks.
///
/// The step is safe only when the full collision age at its endpoint is no
/// larger than the deadline selected from those endpoint-known clocks.  A
/// rejected step must be rolled back before `retry_interval` is attempted.
/// If a still-safe deadline lies less than one representable coordinate-time
/// increment ahead, `apply_collision_now` asks the caller to roll back the
/// probe and apply the batch at the pre-step state.  `unresolved_overdue`
/// instead means that the pre-step age genuinely exceeds the deadline and
/// even the shorter probe has reached coordinate resolution; applying a
/// collision would then put the Boltzmann operator in the wrong temporal
/// order, so the caller must checkpoint and stop.
struct WeightedPICEndpointCadenceDecision {
    double controlling_timescale = std::numeric_limits<double>::infinity();
    double selected_interval = std::numeric_limits<double>::infinity();
    double endpoint_age = 0.0;
    double retry_interval = 0.0;
    double interval_over_controlling_timescale = 0.0;
    bool accept_step = false;
    bool apply_collision_now = false;
    bool unresolved_overdue = false;
};

struct WeightedPICFieldChangeClock {
    double timescale = std::numeric_limits<double>::infinity();
    double potential_relative_change = 0.0;
    double force_relative_change = 0.0;
    double relative_noise_floor = 0.0;
    bool potential_signal_resolved = false;
    bool force_signal_resolved = false;
};

/// Estimates t_phi from a finite field difference while rejecting changes
/// below the particle-mesh sampling floor.
///
/// A PIC field is reconstructed from a finite effective number N_eff of
/// weighted markers.  Field differences far below the nominal Monte-Carlo
/// scale N_eff^(-1/2) cannot be distinguished from marker-deposition noise.
/// Quiet sampling correlates neighboring fields and usually performs better
/// than that scale, so the detection floor uses one quarter of it.  This is a
/// deliberately permissive threshold: it removes dt->0 amplification of
/// deposition residuals without treating sub-noise changes as measured
/// physical derivatives.  Once a channel is resolved, its clock is
///
///   t_phi = Delta t / sqrt[(Delta X/X)^2-floor^2],
///
/// with X equal to the global potential or force norm.
inline WeightedPICFieldChangeClock
weighted_pic_noise_aware_field_change_clock(
    const std::vector<double>& previous_potential,
    const std::vector<double>& previous_force,
    const std::vector<double>& current_potential,
    const std::vector<double>& current_force,
    double observation_interval,
    double effective_marker_number) {
    if (!(observation_interval > 0.0)
        || !std::isfinite(observation_interval)
        || !(effective_marker_number > 0.0)
        || !std::isfinite(effective_marker_number)
        || previous_potential.size() != current_potential.size()
        || previous_force.size() != current_force.size()
        || previous_potential.size() != previous_force.size()) {
        throw std::runtime_error(
            "Invalid weighted PIC field-change clock inputs.");
    }
    // The fallback field clock uses global maximum norms of potential and
    // force. Its numerator and denominator both use the same reconstructed
    // field arrays and observation interval.
    double potential_norm = 0.0;
    double force_norm = 0.0;
    double potential_difference = 0.0;
    double force_difference = 0.0;
    for (std::size_t cell = 0; cell < current_potential.size(); ++cell) {
        potential_norm = std::max({
            potential_norm,
            std::abs(previous_potential[cell]),
            std::abs(current_potential[cell]),
        });
        force_norm = std::max({
            force_norm,
            std::abs(previous_force[cell]),
            std::abs(current_force[cell]),
        });
        potential_difference = std::max(
            potential_difference,
            std::abs(
                current_potential[cell] - previous_potential[cell]));
        force_difference = std::max(
            force_difference,
            std::abs(current_force[cell] - previous_force[cell]));
    }

    WeightedPICFieldChangeClock clock;
    // This is a configured sampling detection threshold, not an estimate
    // of a uniquely physical field derivative below the particle-mesh noise.
    // An unresolved channel returns a positive infinite variation time.
    clock.relative_noise_floor = std::max(
        4096.0 * std::numeric_limits<double>::epsilon(),
        0.25 / std::sqrt(effective_marker_number));
    if (potential_norm > 0.0) {
        clock.potential_relative_change =
            potential_difference / potential_norm;
    }
    if (force_norm > 0.0) {
        clock.force_relative_change = force_difference / force_norm;
    }
    auto resolved_timescale = [&](double relative_change, bool& resolved) {
        if (!(relative_change > clock.relative_noise_floor)) {
            return std::numeric_limits<double>::infinity();
        }
        resolved = true;
        const double resolved_change = std::sqrt(std::max(
            0.0,
            relative_change * relative_change
                - clock.relative_noise_floor * clock.relative_noise_floor));
        return resolved_change > 0.0
            ? observation_interval / resolved_change
            : std::numeric_limits<double>::infinity();
    };
    const double potential_timescale = resolved_timescale(
        clock.potential_relative_change,
        clock.potential_signal_resolved);
    const double force_timescale = resolved_timescale(
        clock.force_relative_change,
        clock.force_signal_resolved);
    clock.timescale = std::min(potential_timescale, force_timescale);
    return clock;
}

/// Selects the outer Boltzmann-operator interval.
///
/// The configured maximum is always a hard ceiling.  For a positive safety
/// fraction epsilon_op, the physical deadline is
///
///   Delta t_op = min(Delta t_max,
///                    epsilon_op min(t_coll,t_dyn,t_phi)).
///
/// Infinite clocks simply do not constrain the minimum.  A zero fraction
/// requests the fixed configured interval.
inline double weighted_pic_collision_selection_interval(
    double maximum_interval,
    double timescale_fraction,
    double collision_timescale,
    double dynamical_timescale,
    double potential_timescale) {
    if (!(maximum_interval > 0.0) || !std::isfinite(maximum_interval)
        || !(timescale_fraction >= 0.0)
        || !std::isfinite(timescale_fraction)) {
        throw std::runtime_error(
            "Invalid weighted PIC collision-cadence controls.");
    }
    auto valid_clock = [](double value) {
        return (value > 0.0 && std::isfinite(value))
            || (std::isinf(value) && value > 0.0);
    };
    if (!valid_clock(collision_timescale)
        || !valid_clock(dynamical_timescale)
        || !valid_clock(potential_timescale)) {
        throw std::runtime_error(
            "Invalid weighted PIC collision-cadence timescale.");
    }
    if (timescale_fraction == 0.0) {
        return maximum_interval;
    }
    // The outer collision interval is limited by the shortest available
    // collision, dynamical or measured field-variation time. Infinite clocks
    // leave the configured maximum as the finite ceiling.
    const double controlling = std::min({
        collision_timescale,
        dynamical_timescale,
        potential_timescale,
    });
    if (!std::isfinite(controlling)) {
        return maximum_interval;
    }
    const double selected = timescale_fraction * controlling;
    if (!(selected > 0.0) || !std::isfinite(selected)) {
        throw std::runtime_error(
            "Weighted PIC selected a non-positive collision deadline.");
    }
    return std::min(maximum_interval, selected);
}

/// Floating-point allowance for comparing an accumulated collision age with
/// its selected physical deadline.  This covers only a few ulps at the scale
/// of the two intervals; it must never hide a physically finite overshoot.
inline double weighted_pic_collision_deadline_roundoff_tolerance(
    double applied_interval,
    double selected_interval) {
    return 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({
            1.0,
            std::abs(applied_interval),
            std::abs(selected_interval),
        });
}

// This comparison permits only the declared floating-point allowance
// at the interval scale. It does not reset a carried age or excuse a
// resolved physical overshoot of the collision deadline.
inline bool weighted_pic_collision_deadline_is_roundoff_safe(
    double applied_interval,
    double selected_interval) {
    return applied_interval <= selected_interval
        + weighted_pic_collision_deadline_roundoff_tolerance(
            applied_interval, selected_interval);
}

// A pre-step decision consumes only the current accumulated age and a freshly
// selected physical deadline. Applying at the current state avoids committing
// an orbit increment below the existing coordinate-resolution scale. The
// collision batch still uses the accumulated age and the 64-epsilon safety test.
enum class WeightedPICPreStepCollisionStatus {
    advance,
    apply_collision_now,
    unsafe_overdue,
    selected_interval_unresolved,
    zero_elapsed_due,
};

struct WeightedPICPreStepCollisionDecision {
    WeightedPICPreStepCollisionStatus status =
        WeightedPICPreStepCollisionStatus::advance;
    double remaining_interval = 0.0;
    double coordinate_resolution = 0.0;
};

inline const char* weighted_pic_pre_step_collision_status_name(
    WeightedPICPreStepCollisionStatus status) {
    switch (status) {
    case WeightedPICPreStepCollisionStatus::advance: return "advance";
    case WeightedPICPreStepCollisionStatus::apply_collision_now:
        return "apply_collision_now";
    case WeightedPICPreStepCollisionStatus::unsafe_overdue:
        return "unsafe_overdue";
    case WeightedPICPreStepCollisionStatus::selected_interval_unresolved:
        return "selected_interval_unresolved";
    case WeightedPICPreStepCollisionStatus::zero_elapsed_due:
        return "zero_elapsed_due";
    }
    throw std::runtime_error("Invalid pre-step collision status.");
}

inline WeightedPICPreStepCollisionDecision
assess_weighted_pic_pre_step_collision_cadence(
    double collision_age,
    double selected_interval,
    double coordinate_time,
    double previous_coordinate_time) {
    if (!(collision_age >= 0.0) || !std::isfinite(collision_age)
        || !(selected_interval > 0.0) || !std::isfinite(selected_interval)
        || !std::isfinite(coordinate_time)
        || !std::isfinite(previous_coordinate_time)
        || coordinate_time < previous_coordinate_time) {
        throw std::runtime_error("Invalid pre-step collision cadence clock.");
    }
    WeightedPICPreStepCollisionDecision result;
    // Compare the remaining deadline with the coordinate resolution at the
    // current absolute time. A positive carried age can require a collision
    // now even when the remaining orbital increment is too small to resolve.
    result.remaining_interval = selected_interval - collision_age;
    result.coordinate_resolution =
        128.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(coordinate_time));
    // Reject a genuinely overdue age first. A deadline itself below time
    // resolution also stops; only an otherwise safe, positive-age batch can
    // be applied at the current pre-step state.
    if (!weighted_pic_collision_deadline_is_roundoff_safe(
            collision_age, selected_interval)) {
        result.status = WeightedPICPreStepCollisionStatus::unsafe_overdue;
    } else if (!(selected_interval > result.coordinate_resolution)) {
        result.status = collision_age == 0.0
            ? WeightedPICPreStepCollisionStatus::zero_elapsed_due
            : WeightedPICPreStepCollisionStatus::selected_interval_unresolved;
    } else if (result.remaining_interval <= result.coordinate_resolution) {
        // Positive elapsed age is required for every sampled operator batch.
        result.status = collision_age > 0.0
            ? WeightedPICPreStepCollisionStatus::apply_collision_now
            : WeightedPICPreStepCollisionStatus::zero_elapsed_due;
    }
    return result;
}

/// Audits one provisional collisionless step against the clocks measured at
/// its endpoint.  This is deliberately separate from the ordinary orbit CFL:
/// a KDK step may satisfy the force/orbit accuracy limit while still crossing
/// a newly shortened Boltzmann-operator deadline.
///
/// `coordinate_time` is used only to identify the smallest distinguishable
/// retry.  When a trial fails, the returned retry stays strictly inside the
/// currently measured deadline.  If that deadline already precedes
/// `collision_age`, repeated halving approaches the pre-step state; only the
/// final time-resolution failure is labelled irrecoverably overdue.
inline WeightedPICEndpointCadenceDecision
assess_weighted_pic_endpoint_collision_cadence(
    double collision_age,
    double proposed_interval,
    double coordinate_time,
    double maximum_interval,
    double timescale_fraction,
    double collision_timescale,
    double endpoint_dynamical_timescale,
    double endpoint_potential_timescale) {
    if (!(collision_age >= 0.0) || !std::isfinite(collision_age)
        || !(proposed_interval > 0.0) || !std::isfinite(proposed_interval)
        || !std::isfinite(coordinate_time)) {
        throw std::runtime_error(
            "Invalid weighted PIC provisional collision cadence.");
    }

    WeightedPICEndpointCadenceDecision decision;
    decision.controlling_timescale = std::min({
        collision_timescale,
        endpoint_dynamical_timescale,
        endpoint_potential_timescale,
    });
    decision.selected_interval = weighted_pic_collision_selection_interval(
        maximum_interval,
        timescale_fraction,
        collision_timescale,
        endpoint_dynamical_timescale,
        endpoint_potential_timescale);
    // A trial orbit accumulates collision age even when its force CFL is
    // safe. Re-evaluating the clocks at the returned endpoint detects a
    // physical deadline that shortened during that trial.
    decision.endpoint_age = collision_age + proposed_interval;
    if (!std::isfinite(decision.endpoint_age)) {
        throw std::runtime_error(
            "Weighted PIC collision age overflowed during a provisional step.");
    }
    if (std::isfinite(decision.controlling_timescale)) {
        decision.interval_over_controlling_timescale =
            decision.endpoint_age / decision.controlling_timescale;
    }

    // Acceptance concerns the complete accumulated collision age. The
    // caller must preserve the pre-trial particle/field/RNG state until this
    // endpoint check accepts the trial or requests an earlier batch.
    decision.accept_step =
        weighted_pic_collision_deadline_is_roundoff_safe(
            decision.endpoint_age, decision.selected_interval);
    if (decision.accept_step) {
        return decision;
    }

    const double coordinate_resolution =
        128.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(coordinate_time));
    // When a safe pre-step event lies within coordinate resolution, return
    // an explicit collision-now decision. Otherwise the retry interval
    // stays inside the newly measured endpoint deadline.
    const double time_left =
        decision.selected_interval - collision_age;
    if (time_left <= coordinate_resolution
        && weighted_pic_collision_deadline_is_roundoff_safe(
            collision_age, decision.selected_interval)) {
        // The event still belongs after the pre-step state, but no positive
        // KDK interval can land there reliably.  Applying it infinitesimally
        // early preserves temporal order and the hard cadence inequality.
        decision.apply_collision_now = true;
        return decision;
    }
    if (proposed_interval <= coordinate_resolution) {
        decision.unresolved_overdue = true;
        return decision;
    }

    if (time_left > coordinate_resolution) {
        // Stay inside the provisional endpoint deadline rather than landing
        // on it: the clock can shorten again at the retried endpoint.  The
        // caller applies the collision at the first accepted retried endpoint,
        // so there is no need to halve an otherwise safe orbit step and then
        // approach the same deadline through a long geometric sequence of
        // ever smaller committed KDK steps.  Retaining five per cent of the
        // currently measured time-left is a conservative bracket margin; the
        // 7/8 cap still makes progress when the endpoint clock barely moved.
        decision.retry_interval = std::min(
            0.875 * proposed_interval, 0.95 * time_left);
    } else {
        // The endpoint probe may have encountered a collapse that begins
        // later in the attempted step.  Halving localizes its onset before
        // declaring the already accumulated age irrecoverable.
        decision.retry_interval = 0.5 * proposed_interval;
    }
    if (!(decision.retry_interval > 0.0)
        || !std::isfinite(decision.retry_interval)) {
        decision.unresolved_overdue = true;
        decision.retry_interval = 0.0;
    }
    return decision;
}

inline double weighted_pic_scaled_event_interval(
    double timescale,
    double fraction,
    const char* name) {
    if (!(fraction > 0.0) || !std::isfinite(fraction)) {
        throw std::runtime_error(
            std::string("Invalid weighted PIC ") + name + " fraction.");
    }
    if (std::isinf(timescale) && timescale > 0.0) {
        return timescale;
    }
    if (!(timescale > 0.0) || !std::isfinite(timescale)) {
        throw std::runtime_error(
            std::string("Invalid weighted PIC ") + name + " timescale.");
    }
    return fraction * timescale;
}

/// Plans the next frozen-field orbit interval.
///
/// `potential_timescale` is the current minimum of |Phi/Phi_dot| and
/// |g/g_dot|.  `collision_timescale` is the current minimum local collision
/// time.  The two ages are retained across events, so repeated potential
/// refreshes cannot postpone a slower collision deadline and repeated
/// collisions cannot postpone a slower potential refresh.
inline WeightedPICEventPlan plan_weighted_pic_event(
    const WeightedPICEventClockState& clock,
    double potential_timescale,
    double collision_timescale,
    double potential_fraction,
    double collision_fraction,
    double maximum_orbit_interval =
        std::numeric_limits<double>::infinity()) {
    if (!(clock.potential_age >= 0.0) || !std::isfinite(clock.potential_age)
        || !(clock.collision_age >= 0.0) || !std::isfinite(clock.collision_age)
        || !(maximum_orbit_interval > 0.0)
        || std::isnan(maximum_orbit_interval)) {
        throw std::runtime_error("Invalid weighted PIC event-clock state.");
    }
    WeightedPICEventPlan plan;
    plan.potential_interval = weighted_pic_scaled_event_interval(
        potential_timescale, potential_fraction, "potential-refresh");
    plan.collision_interval = weighted_pic_scaled_event_interval(
        collision_timescale, collision_fraction, "collision");

    // Subtract each independent carried age from its own deadline.
    // Refreshing the field cannot erase elapsed collision time, and a
    // collision cannot erase the age of the field reconstruction.
    const double remaining_potential = std::max(
        0.0, plan.potential_interval - clock.potential_age);
    const double remaining_collision = std::max(
        0.0, plan.collision_interval - clock.collision_age);
    plan.orbit_interval = std::min({
        remaining_potential,
        remaining_collision,
        maximum_orbit_interval,
    });
    if (!std::isfinite(plan.orbit_interval)) {
        throw std::runtime_error(
            "Weighted PIC event planning needs a finite orbit cap when both physical deadlines are infinite.");
    }
    auto deadline_tolerance = [&](double interval, double age) {
        return 64.0 * std::numeric_limits<double>::epsilon()
            * std::max({
                1.0,
                std::isfinite(interval) ? std::abs(interval) : 0.0,
                std::abs(age),
                std::abs(plan.orbit_interval),
            });
    };
    const double potential_tolerance = deadline_tolerance(
        plan.potential_interval, clock.potential_age);
    const double collision_tolerance = deadline_tolerance(
        plan.collision_interval, clock.collision_age);
    plan.potential_due = remaining_potential
        <= plan.orbit_interval + potential_tolerance;
    plan.collision_due = remaining_collision
        <= plan.orbit_interval + collision_tolerance;

    // If both operations are due, compare their selected intervals to fix
    // the same deterministic ordering for every caller and restart.
    if (plan.potential_due && plan.collision_due) {
        plan.order = plan.collision_interval < plan.potential_interval
            ? WeightedPICEventOrder::collision_then_reconstruction
            : WeightedPICEventOrder::reconstruction_then_collision;
    } else if (plan.collision_due) {
        plan.order = WeightedPICEventOrder::collision_then_reconstruction;
    } else if (plan.potential_due) {
        plan.order = WeightedPICEventOrder::reconstruction_only;
    }
    return plan;
}

/// Advances both ages and resets only the clock whose event was completed.
/// Call this after carrying out `plan.order` successfully.
inline void commit_weighted_pic_event(
    WeightedPICEventClockState& clock,
    const WeightedPICEventPlan& plan) {
    if (!(plan.orbit_interval >= 0.0) || !std::isfinite(plan.orbit_interval)) {
        throw std::runtime_error("Invalid weighted PIC event interval.");
    }
    // Commit ages only after the planned operations succeed. Reset exactly
    // the clock whose operation was executed, retaining the other elapsed age.
    clock.potential_age += plan.orbit_interval;
    clock.collision_age += plan.orbit_interval;
    if (plan.potential_due) {
        clock.potential_age = 0.0;
    }
    if (plan.collision_due) {
        clock.collision_age = 0.0;
    }
}

/// Executes the endpoint operations in the order selected by the two clocks.
/// The three callbacks receive no arguments; callers capture the marker and
/// field state they own.  A capped orbit interval with no physical deadline
/// performs transport alone and retains both clock ages.
template <class Transport, class Reconstruct, class Collide>
inline void execute_weighted_pic_event(
    const WeightedPICEventPlan& plan,
    Transport&& transport,
    Reconstruct&& reconstruct,
    Collide&& collide) {
    if (plan.orbit_interval > 0.0) {
        transport();
    }
    // Transport precedes the endpoint operations. A cap-limited step with
    // no due event performs only transport; reconstruction/collision order
    // is exactly the one returned in the plan.
    switch (plan.order) {
        case WeightedPICEventOrder::none:
            break;
        case WeightedPICEventOrder::reconstruction_only:
            reconstruct();
            break;
        case WeightedPICEventOrder::collision_then_reconstruction:
            collide();
            reconstruct();
            break;
        case WeightedPICEventOrder::reconstruction_then_collision:
            reconstruct();
            collide();
            break;
    }
}

}  // namespace ev
