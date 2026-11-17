#include "venus_gpu/restart.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void test_version6_metadata_and_marker_roundtrip() {
    const std::string path = "venus_gpu_restart_roundtrip_test.bin";
    venus_gpu::RestartState expected;
    expected.initialization_seed = UINT64_C(0x123456789abcdef0);
    expected.initialization_counter = 37;
    expected.collision_sampler.seed = UINT64_C(0x0fedcba987654321);
    expected.collision_sampler.next_counter = 91;
    expected.potential_event_age = 0.25;
    expected.collision_event_age = 0.375;
    expected.collision_timescale_estimate = 2.5;
    expected.potential_timescale_estimate = 3.75;
    expected.time = 50.125;
    expected.step = 4123;

    venus_gpu::Marker active;
    active.mass = 0.4;
    active.r = 1.25;
    active.w = -0.125;
    active.eta = 0.03;
    active.id = 10;
    active.source_cell = 2;
    active.internal_energy = 0.006;
    active.internal_velocity_dispersion = 0.1;
    active.inner_boundary_speed = 0.2;
    expected.markers.push_back(active);

    venus_gpu::Marker delayed = active;
    delayed.mass = 0.6;
    delayed.r = 0.25;
    delayed.w = 0.0;
    delayed.id = 11;
    delayed.source_cell = 3;
    delayed.state = venus_gpu::MarkerState::delayed_inner;
    delayed.internal_energy = 0.009;
    delayed.inner_boundary_speed = 0.35;
    delayed.remaining_inner_time = 0.75;
    expected.markers.push_back(delayed);

    venus_gpu::write_restart(path, expected);
    const venus_gpu::RestartState measured = venus_gpu::read_restart(path);
    std::remove(path.c_str());

    assert(measured.initialization_seed == expected.initialization_seed);
    assert(measured.initialization_counter == expected.initialization_counter);
    assert(measured.collision_sampler.seed == expected.collision_sampler.seed);
    assert(measured.collision_sampler.next_counter
        == expected.collision_sampler.next_counter);
    assert(measured.potential_event_age == expected.potential_event_age);
    assert(measured.collision_event_age == expected.collision_event_age);
    assert(measured.collision_timescale_estimate
        == expected.collision_timescale_estimate);
    assert(measured.potential_timescale_estimate
        == expected.potential_timescale_estimate);
    assert(measured.time == expected.time);
    assert(measured.step == expected.step);
    assert(measured.markers.size() == expected.markers.size());
    for (std::size_t i = 0; i < expected.markers.size(); ++i) {
        const venus_gpu::Marker& first = measured.markers[i];
        const venus_gpu::Marker& second = expected.markers[i];
        assert(first.mass == second.mass);
        assert(first.r == second.r);
        assert(first.w == second.w);
        assert(first.eta == second.eta);
        assert(first.id == second.id);
        assert(first.source_cell == second.source_cell);
        assert(first.state == second.state);
        assert(first.inner_boundary_speed == second.inner_boundary_speed);
        assert(first.remaining_inner_time == second.remaining_inner_time);
        assert(first.internal_energy == second.internal_energy);
        assert(first.internal_velocity_dispersion
            == second.internal_velocity_dispersion);
    }
}

venus_gpu::RestartState valid_state() {
    venus_gpu::RestartState state;
    venus_gpu::Marker marker;
    marker.mass = 1.0;
    marker.r = 1.0;
    marker.id = 1;
    state.markers.push_back(marker);
    return state;
}

template <class Action>
void expect_runtime_error(Action action) {
    bool threw = false;
    try {
        action();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_invalid_boundary_clocks_are_rejected() {
    const std::string path = "venus_gpu_restart_invalid_test.bin";
    venus_gpu::RestartState state = valid_state();
    state.markers[0].inner_boundary_speed =
        std::numeric_limits<double>::quiet_NaN();
    expect_runtime_error([&] { venus_gpu::write_restart(path, state); });

    state = valid_state();
    state.markers[0].remaining_inner_time = -1.0;
    expect_runtime_error([&] { venus_gpu::write_restart(path, state); });

    state = valid_state();
    state.markers[0].state = venus_gpu::MarkerState::delayed_inner;
    state.markers[0].remaining_inner_time = 0.0;
    expect_runtime_error([&] { venus_gpu::write_restart(path, state); });
    std::remove(path.c_str());
}

void test_payload_size_is_checked_before_marker_allocation() {
    const std::string path = "venus_gpu_restart_payload_test.bin";
    venus_gpu::write_restart(path, valid_state());
    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        output.put('\0');
    }
    expect_runtime_error([&] { (void)venus_gpu::read_restart(path); });
    std::remove(path.c_str());
}

}  // namespace

int main() {
    test_version6_metadata_and_marker_roundtrip();
    test_invalid_boundary_clocks_are_rejected();
    test_payload_size_is_checked_before_marker_allocation();
    std::cout << "VENUS GPU restart roundtrip passed.\n";
    return 0;
}
