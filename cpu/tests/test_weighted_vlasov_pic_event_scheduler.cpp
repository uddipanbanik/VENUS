// Tests independent field/collision deadlines, deterministic ties, age
// accumulation/reset, and restart-equivalent weighted-PIC event plans.

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "ev/weighted_vlasov_pic_event_scheduler.hpp"

namespace {

bool close(double first, double second) {
    return std::abs(first - second) < 2.0e-14;
}

void test_collision_clock_is_faster() {
    ev::WeightedPICEventClockState clock;
    for (int event = 1; event <= 5; ++event) {
        const ev::WeightedPICEventPlan plan = ev::plan_weighted_pic_event(
            clock, 10.0, 2.0, 1.0, 1.0);
        assert(close(plan.orbit_interval, 2.0));
        assert(plan.collision_due);
        if (event < 5) {
            assert(!plan.potential_due);
            assert(plan.order
                   == ev::WeightedPICEventOrder::collision_then_reconstruction);
        } else {
            assert(plan.potential_due);
            assert(plan.order
                   == ev::WeightedPICEventOrder::collision_then_reconstruction);
        }
        ev::commit_weighted_pic_event(clock, plan);
    }
    assert(close(clock.potential_age, 0.0));
    assert(close(clock.collision_age, 0.0));
}

void test_potential_clock_is_faster() {
    ev::WeightedPICEventClockState clock;
    for (int event = 1; event <= 5; ++event) {
        const ev::WeightedPICEventPlan plan = ev::plan_weighted_pic_event(
            clock, 2.0, 10.0, 1.0, 1.0);
        assert(close(plan.orbit_interval, 2.0));
        assert(plan.potential_due);
        if (event < 5) {
            assert(!plan.collision_due);
            assert(plan.order
                   == ev::WeightedPICEventOrder::reconstruction_only);
        } else {
            assert(plan.collision_due);
            assert(plan.order
                   == ev::WeightedPICEventOrder::reconstruction_then_collision);
        }
        ev::commit_weighted_pic_event(clock, plan);
    }
    assert(close(clock.potential_age, 0.0));
    assert(close(clock.collision_age, 0.0));
}

void test_updated_timescale_keeps_elapsed_age() {
    ev::WeightedPICEventClockState clock;
    ev::WeightedPICEventPlan first = ev::plan_weighted_pic_event(
        clock, 2.0, 10.0, 1.0, 1.0);
    ev::commit_weighted_pic_event(clock, first);
    assert(close(clock.collision_age, 2.0));

    // The refreshed endpoint reveals a shorter collision time.  The two units
    // already accumulated count toward the new five-unit deadline.
    const ev::WeightedPICEventPlan second = ev::plan_weighted_pic_event(
        clock, 20.0, 5.0, 1.0, 1.0);
    assert(close(second.orbit_interval, 3.0));
    assert(second.collision_due);
    assert(!second.potential_due);
    assert(second.order
           == ev::WeightedPICEventOrder::collision_then_reconstruction);
}

void test_probe_can_reveal_immediate_field_event() {
    ev::WeightedPICEventClockState clock;
    const ev::WeightedPICEventPlan probe = ev::plan_weighted_pic_event(
        clock,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        1.0,
        1.0,
        0.25);
    assert(close(probe.orbit_interval, 0.25));
    assert(probe.order == ev::WeightedPICEventOrder::none);
    ev::commit_weighted_pic_event(clock, probe);

    // The endpoint probe now measures a 0.2 field deadline.  Since 0.25 has
    // already elapsed, the refresh is due immediately and the collision age
    // remains untouched.
    const ev::WeightedPICEventPlan refresh = ev::plan_weighted_pic_event(
        clock,
        0.2,
        std::numeric_limits<double>::infinity(),
        1.0,
        1.0,
        0.25);
    assert(close(refresh.orbit_interval, 0.0));
    assert(refresh.potential_due);
    assert(!refresh.collision_due);
    assert(refresh.order
           == ev::WeightedPICEventOrder::reconstruction_only);
    ev::commit_weighted_pic_event(clock, refresh);
    assert(close(clock.potential_age, 0.0));
    assert(close(clock.collision_age, 0.25));
}

void test_clock_tolerances_are_independent() {
    const ev::WeightedPICEventPlan plan = ev::plan_weighted_pic_event(
        {},
        0.4,
        1.5e17,
        0.1,
        1.0e-4,
        1.0e-6);
    assert(close(plan.orbit_interval, 1.0e-6));
    assert(!plan.potential_due);
    assert(!plan.collision_due);
    assert(plan.order == ev::WeightedPICEventOrder::none);
}

void test_endpoint_clock_collapse_rejects_and_shortens_kdk() {
    const auto rejected =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            0.02,   // age already accumulated since the last collision
            0.05,   // orbit-CFL step proposed from the pre-step clocks
            4.0,
            1.0,
            0.1,
            10.0,
            10.0,
            0.3);   // endpoint field reveals t_phi=0.3
    assert(!rejected.accept_step);
    assert(!rejected.apply_collision_now);
    assert(!rejected.unresolved_overdue);
    assert(close(rejected.selected_interval, 0.03));
    assert(rejected.endpoint_age > rejected.selected_interval);
    assert(rejected.retry_interval > 0.0);
    assert(rejected.retry_interval < 0.01);
    assert(rejected.retry_interval > 0.009);

    // With the same conservative endpoint clock, the returned retry remains
    // inside epsilon_op t_phi and may be committed.
    const auto accepted =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            0.02,
            rejected.retry_interval,
            4.0,
            1.0,
            0.1,
            10.0,
            10.0,
            0.3);
    assert(accepted.accept_step);
    assert(!accepted.apply_collision_now);
    assert(!accepted.unresolved_overdue);
    assert(
        accepted.interval_over_controlling_timescale
        <= 0.1 + 2.0e-14);
    // The driver may conservatively apply the collision at this accepted
    // post-retry endpoint.  It is close to, but still strictly below, the
    // endpoint-measured maximum interval.
    assert(accepted.endpoint_age > 0.098 * 0.3);
    assert(accepted.endpoint_age < accepted.selected_interval);
}

void test_retry_lands_near_deadline_without_geometric_cleanup() {
    const double collision_age = 0.105;
    const double deadline = 0.12;
    const double timescale_fraction = 0.05;
    const double controlling_timescale = deadline / timescale_fraction;

    const auto rejected =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            collision_age,
            0.016,
            447.0,
            0.5,
            timescale_fraction,
            controlling_timescale,
            10.0,
            20.0);
    assert(!rejected.accept_step);
    assert(!rejected.apply_collision_now);
    assert(!rejected.unresolved_overdue);
    assert(rejected.retry_interval > 0.0139);
    assert(rejected.retry_interval < deadline - collision_age);

    const auto accepted =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            collision_age,
            rejected.retry_interval,
            447.0,
            0.5,
            timescale_fraction,
            controlling_timescale,
            10.0,
            20.0);
    assert(accepted.accept_step);
    assert(!accepted.apply_collision_now);
    assert(!accepted.unresolved_overdue);
    assert(accepted.endpoint_age < accepted.selected_interval);
    assert(
        accepted.interval_over_controlling_timescale
        <= timescale_fraction + 2.0e-14);
    assert(
        accepted.interval_over_controlling_timescale
        > 0.99 * timescale_fraction);
}

void test_field_clock_rejects_sub_noise_cleanup_residual() {
    const std::vector<double> phi{1.0, 0.5};
    const std::vector<double> force{0.4, 0.2};
    std::vector<double> noisy_phi = phi;
    std::vector<double> noisy_force = force;
    noisy_phi[0] += 8.0e-8;
    noisy_force[0] -= 6.0e-8;
    const auto cleanup =
        ev::weighted_pic_noise_aware_field_change_clock(
            phi,
            force,
            noisy_phi,
            noisy_force,
            8.25e-11,
            1.0e6);
    assert(close(cleanup.relative_noise_floor, 2.5e-4));
    assert(!cleanup.potential_signal_resolved);
    assert(!cleanup.force_signal_resolved);
    assert(std::isinf(cleanup.timescale));

    // A coherent one-percent field change is far above the same finite-marker
    // floor and produces a genuine finite potential-change clock.
    std::vector<double> signal_phi = phi;
    std::vector<double> signal_force = force;
    signal_phi[0] += 0.01;
    signal_force[0] += 0.004;
    const auto signal =
        ev::weighted_pic_noise_aware_field_change_clock(
            phi,
            force,
            signal_phi,
            signal_force,
            0.002,
            1.0e6);
    assert(signal.potential_signal_resolved);
    assert(signal.force_signal_resolved);
    assert(std::isfinite(signal.timescale));
    assert(signal.timescale > 0.19);
    assert(signal.timescale < 0.21);
}

void test_unresolved_pre_step_overshoot_stops_instead_of_colliding() {
    const double coordinate_time = 10000.0;
    const double coordinate_resolution =
        128.0 * std::numeric_limits<double>::epsilon()
        * coordinate_time;
    const auto decision =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            0.04,
            0.5 * coordinate_resolution,
            coordinate_time,
            1.0,
            0.1,
            10.0,
            10.0,
            0.3);
    assert(!decision.accept_step);
    assert(!decision.apply_collision_now);
    assert(decision.unresolved_overdue);
    assert(decision.retry_interval == 0.0);
}

void test_unrepresentable_positive_remainder_applies_safely_early() {
    const double coordinate_time = 46.0;
    const double coordinate_resolution =
        128.0 * std::numeric_limits<double>::epsilon()
        * coordinate_time;
    const double deadline = 0.5;

    // The accumulated age is still below the deadline, but its remaining
    // positive interval is smaller than one distinguishable increment of the
    // absolute time coordinate.  A trial step would cross the deadline, so
    // the operator must be applied at the rolled-back pre-step state.
    const auto below =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            deadline - 0.5 * coordinate_resolution,
            coordinate_resolution,
            coordinate_time,
            1.0,
            0.1,
            5.0,
            10.0,
            10.0);
    assert(!below.accept_step);
    assert(below.apply_collision_now);
    assert(!below.unresolved_overdue);
    assert(below.retry_interval == 0.0);
    assert(below.endpoint_age > below.selected_interval);

    // A one-ulp excess is comparison roundoff, whereas a finite excess is a
    // real temporal-order violation and must retain the fail-closed result.
    const double one_ulp_over =
        std::nextafter(deadline, std::numeric_limits<double>::infinity());
    assert(ev::weighted_pic_collision_deadline_is_roundoff_safe(
        one_ulp_over, deadline));
    const auto roundoff_over =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            one_ulp_over,
            coordinate_resolution,
            coordinate_time,
            1.0,
            0.1,
            5.0,
            10.0,
            10.0);
    assert(roundoff_over.apply_collision_now);
    assert(!roundoff_over.unresolved_overdue);

    const double physical_overshoot = deadline + 1.0e-10;
    assert(!ev::weighted_pic_collision_deadline_is_roundoff_safe(
        physical_overshoot, deadline));
    const auto overdue =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            physical_overshoot,
            0.5 * coordinate_resolution,
            coordinate_time,
            1.0,
            0.1,
            5.0,
            10.0,
            10.0);
    assert(!overdue.accept_step);
    assert(!overdue.apply_collision_now);
    assert(overdue.unresolved_overdue);
}

void test_fixed_cadence_and_restart_age_are_deterministic() {
    const double fixed = ev::weighted_pic_collision_selection_interval(
        0.125,
        0.0,
        std::numeric_limits<double>::infinity(),
        2.0,
        0.01);
    assert(close(fixed, 0.125));

    // A restart restores this age and the three clock estimates exactly.  The
    // endpoint decision must therefore match an uninterrupted trajectory.
    const double saved_age = 0.01875;
    const auto uninterrupted =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            saved_age, 0.02, 7.0, 1.0, 0.1, 4.0, 2.0, 0.25);
    const double loaded_age = saved_age;
    const auto restarted =
        ev::assess_weighted_pic_endpoint_collision_cadence(
            loaded_age, 0.02, 7.0, 1.0, 0.1, 4.0, 2.0, 0.25);
    assert(uninterrupted.accept_step == restarted.accept_step);
    assert(
        uninterrupted.apply_collision_now
        == restarted.apply_collision_now);
    assert(
        uninterrupted.unresolved_overdue
        == restarted.unresolved_overdue);
    assert(uninterrupted.selected_interval == restarted.selected_interval);
    assert(uninterrupted.endpoint_age == restarted.endpoint_age);
    assert(uninterrupted.retry_interval == restarted.retry_interval);
}

void test_recorded_operation_order() {
    auto run = [](ev::WeightedPICEventOrder order, double interval = 1.0) {
        ev::WeightedPICEventPlan plan;
        plan.orbit_interval = interval;
        plan.order = order;
        std::vector<std::string> operations;
        ev::execute_weighted_pic_event(
            plan,
            [&] { operations.push_back("transport"); },
            [&] { operations.push_back("reconstruct"); },
            [&] { operations.push_back("collide"); });
        return operations;
    };
    assert((run(ev::WeightedPICEventOrder::none)
            == std::vector<std::string>{"transport"}));
    assert((run(ev::WeightedPICEventOrder::reconstruction_only)
            == std::vector<std::string>{"transport", "reconstruct"}));
    assert((run(ev::WeightedPICEventOrder::collision_then_reconstruction)
            == std::vector<std::string>{
                "transport", "collide", "reconstruct"}));
    assert((run(ev::WeightedPICEventOrder::reconstruction_then_collision)
            == std::vector<std::string>{
                "transport", "reconstruct", "collide"}));
    assert((run(ev::WeightedPICEventOrder::collision_then_reconstruction, 0.0)
            == std::vector<std::string>{"collide", "reconstruct"}));
}

}  // namespace

int main() {
    test_collision_clock_is_faster();
    test_potential_clock_is_faster();
    test_updated_timescale_keeps_elapsed_age();
    test_probe_can_reveal_immediate_field_event();
    test_clock_tolerances_are_independent();
    test_endpoint_clock_collapse_rejects_and_shortens_kdk();
    test_retry_lands_near_deadline_without_geometric_cleanup();
    test_field_clock_rejects_sub_noise_cleanup_residual();
    test_unresolved_pre_step_overshoot_stops_instead_of_colliding();
    test_unrepresentable_positive_remainder_applies_safely_early();
    test_fixed_cadence_and_restart_age_are_deterministic();
    test_recorded_operation_order();
    std::cout << "weighted PIC two-clock event-scheduler tests passed\n";
    return 0;
}
