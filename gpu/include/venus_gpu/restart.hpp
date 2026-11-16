#pragma once

// Represent a complete host-side GPU particle checkpoint.
// read_restart() takes a versioned binary file and returns ordered particles,
// time, step, random counters, event ages, and estimated event timescales.
// write_restart() serializes that state through an atomic file replacement.
// The CPU and GPU formats share the same field order, allowing a supported
// continuation to preserve both the physical population and collision clocks.

#include "venus_gpu/collision_common.hpp"
#include "venus_gpu/state.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace venus_gpu {

/// Bit-compatible host checkpoint shared with the VENUS CPU module.
///
/// The binary contract is intentionally implemented in the separate GPU tree:
/// production executables do not link the CPU solver.  A checkpoint written by
/// either code can nevertheless be continued by the other code.
struct RestartState {
    std::vector<Marker> markers;
    std::uint64_t initialization_seed = UINT64_C(0x6a09e667f3bcc909);
    std::uint64_t initialization_counter = 0;
    CollisionSamplerState collision_sampler{};
    // An age is elapsed physical time since that operator last ran. It is
    // carried into a continuation so a restart does not postpone a due event.
    double potential_event_age = 0.0;
    double collision_event_age = 0.0;
    double collision_timescale_estimate =
        std::numeric_limits<double>::infinity();
    double potential_timescale_estimate =
        std::numeric_limits<double>::infinity();
    // time and step identify the accepted particle state represented by
    // every payload field and by the stored operator ages.
    double time = 0.0;
    std::int64_t step = 0;
};

RestartState read_restart(const std::string& path);

/// Writes current version 6 through path.tmp followed by an atomic rename.
void write_restart(const std::string& path, const RestartState& state);

}  // namespace venus_gpu
