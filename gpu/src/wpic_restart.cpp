// Host-side reader/writer for the CPU/GPU-compatible WPIC restart contract.
// read_restart accepts versions 1--6 and initializes fields absent from a format;
// write_restart emits version 6 to path.tmp and atomically renames it.  The
// binary payload owns metadata plus an ordered array of host Marker records;
// callers explicitly download/upload WpicState around this I/O boundary.
// Magic, endian tag, scalar order, 81-byte marker layout, unique IDs, finite
// physical fields, event clocks, and absence of trailing bytes are contract
// invariants.  This file contains no CUDA calls and is also linked into the
// host-only checkpoint inspection utilities.

#include "venus_gpu/restart.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace venus_gpu {
namespace {

// Write scalars individually so the portable file layout has no compiler
// struct padding. Each operation checks the stream immediately.
template <class T>
void write_scalar(std::ofstream& output, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "restart scalar");
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("Failed while writing GPU WPIC restart.");
    }
}

template <class T>
T read_scalar(std::ifstream& input) {
    static_assert(std::is_trivially_copyable<T>::value, "restart scalar");
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("Unexpected end of GPU WPIC restart.");
    }
    return value;
}

// Ages are finite elapsed times. A positive infinite timescale is the
// explicit sentinel for a channel whose finite variation is not yet measured.
void validate_metadata(const RestartState& state) {
    if (!(state.time >= 0.0) || !std::isfinite(state.time) || state.step < 0
        || !(state.potential_event_age >= 0.0)
        || !std::isfinite(state.potential_event_age)
        || !(state.collision_event_age >= 0.0)
        || !std::isfinite(state.collision_event_age)
        || !(state.collision_timescale_estimate > 0.0)
        || std::isnan(state.collision_timescale_estimate)
        || !(state.potential_timescale_estimate > 0.0)
        || std::isnan(state.potential_timescale_estimate)) {
        throw std::runtime_error("GPU WPIC restart has invalid time metadata.");
    }
}

// Check physical ranges and boundary state before accepting or publishing
// a payload. IDs must be unique even when capture has shortened the sequence.
void validate_markers(const std::vector<Marker>& markers) {
    std::vector<std::uint64_t> ids;
    ids.reserve(markers.size());
    for (const Marker& marker : markers) {
        if (!(marker.mass > 0.0) || !std::isfinite(marker.mass)
            || !(marker.r >= 0.0) || !std::isfinite(marker.r)
            || !std::isfinite(marker.w) || !(marker.eta >= 0.0)
            || !std::isfinite(marker.eta)
            || !(marker.internal_energy >= 0.0)
            || !std::isfinite(marker.internal_energy)
            || !(marker.internal_velocity_dispersion >= 0.0)
            || !std::isfinite(marker.internal_velocity_dispersion)
            || !(marker.inner_boundary_speed >= 0.0)
            || !std::isfinite(marker.inner_boundary_speed)
            || !(marker.remaining_inner_time >= 0.0)
            || !std::isfinite(marker.remaining_inner_time)
            || static_cast<std::uint8_t>(marker.state)
                > static_cast<std::uint8_t>(MarkerState::escaped)) {
            throw std::runtime_error("GPU WPIC restart has an invalid marker.");
        }
        if (marker.state == MarkerState::delayed_inner
            && !(marker.remaining_inner_time > 0.0)) {
            throw std::runtime_error(
                "GPU WPIC restart has a delayed marker without a positive "
                "return time.");
        }
        ids.push_back(marker.id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::runtime_error("GPU WPIC restart has duplicate marker IDs.");
    }
}

}  // namespace

// Decode the version header before allocating particles. Optional version
// fields retain their RestartState defaults when an older file omits them.
RestartState read_restart(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open GPU WPIC restart: " + path);
    }
    char magic[16] = {};
    const char expected[16] = {
        'V', 'E', 'N', 'U', 'S', 'W', 'V', 'P', 'I', 'C', '1', '\0'};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid GPU WPIC restart magic: " + path);
    }
    const std::uint32_t version = read_scalar<std::uint32_t>(input);
    const std::uint32_t endian = read_scalar<std::uint32_t>(input);
    if (version < 1 || version > 6 || endian != UINT32_C(0x01020304)) {
        throw std::runtime_error("Unsupported GPU WPIC restart version.");
    }
    const std::uint64_t marker_count = read_scalar<std::uint64_t>(input);
    if (marker_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("GPU WPIC marker count overflows size_t.");
    }
    RestartState result;
    result.time = read_scalar<double>(input);
    result.step = read_scalar<std::int64_t>(input);
    result.initialization_seed = read_scalar<std::uint64_t>(input);
    result.initialization_counter = read_scalar<std::uint64_t>(input);
    if (version >= 2) {
        result.collision_sampler.seed = read_scalar<std::uint64_t>(input);
        result.collision_sampler.next_counter =
            read_scalar<std::uint64_t>(input);
    }
    if (version >= 3) {
        result.potential_event_age = read_scalar<double>(input);
        result.collision_event_age = read_scalar<double>(input);
    }
    if (version >= 6) {
        result.collision_timescale_estimate = read_scalar<double>(input);
        result.potential_timescale_estimate = read_scalar<double>(input);
    }
    // Reject truncated, trailing, or implausibly large payloads before resize.
    // Otherwise a corrupt marker count can request an enormous allocation even
    // when the file contains only a few bytes.  Marker records are deliberately
    // padding-free and their size is fixed by the restart version.
    const std::streampos payload_begin = input.tellg();
    input.seekg(0, std::ios::end);
    const std::streampos file_end = input.tellg();
    if (payload_begin < std::streampos(0) || file_end < payload_begin) {
        throw std::runtime_error("Could not determine GPU WPIC restart size.");
    }
    const std::uint64_t record_bytes = version >= 5 ? UINT64_C(81)
        : version >= 4 ? UINT64_C(73) : UINT64_C(65);
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(
        file_end - payload_begin);
    if (marker_count > UINT64_MAX / record_bytes
        || marker_count * record_bytes != payload_bytes) {
        throw std::runtime_error(
            "GPU WPIC restart marker count does not match its payload size.");
    }
    input.seekg(payload_begin);
    if (!input) {
        throw std::runtime_error("Could not seek within GPU WPIC restart.");
    }
    result.markers.resize(static_cast<std::size_t>(marker_count));
    for (Marker& marker : result.markers) {
        marker.mass = read_scalar<double>(input);
        marker.r = read_scalar<double>(input);
        marker.w = read_scalar<double>(input);
        marker.eta = read_scalar<double>(input);
            // Internal energy entered the older schema before dispersion. When
        // dispersion is absent, recover sigma from E_internal=3*m*sigma^2/2.
        if (version >= 4) {
            marker.internal_energy = read_scalar<double>(input);
        }
        if (version >= 5) {
            marker.internal_velocity_dispersion = read_scalar<double>(input);
        } else if (marker.internal_energy > 0.0) {
            marker.internal_velocity_dispersion = std::sqrt(
                2.0 * marker.internal_energy / (3.0 * marker.mass));
        }
        marker.id = read_scalar<std::uint64_t>(input);
        marker.source_cell = read_scalar<std::uint64_t>(input);
        const std::uint8_t state = read_scalar<std::uint8_t>(input);
        if (state > static_cast<std::uint8_t>(MarkerState::escaped)) {
            throw std::runtime_error("GPU WPIC restart marker state is invalid.");
        }
        marker.state = static_cast<MarkerState>(state);
        marker.inner_boundary_speed = read_scalar<double>(input);
        marker.remaining_inner_time = read_scalar<double>(input);
    }
    // Exact payload closure distinguishes a complete checkpoint from an
    // incompatible layout, trailing data, or an interrupted scalar read.
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("GPU WPIC restart has trailing data.");
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while closing GPU WPIC restart read.");
    }
    validate_metadata(result);
    validate_markers(result.markers);
    return result;
}

// Validate first, then serialize the current schema without changing the
// particle order or carried clocks supplied by the caller.
void write_restart(const std::string& path, const RestartState& state) {
    validate_metadata(state);
    validate_markers(state.markers);
    const std::string temporary = path + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not open GPU WPIC restart: " + temporary);
    }
    const char magic[16] = {
        'V', 'E', 'N', 'U', 'S', 'W', 'V', 'P', 'I', 'C', '1', '\0'};
    output.write(magic, sizeof(magic));
    write_scalar(output, std::uint32_t{6});
    write_scalar(output, UINT32_C(0x01020304));
    write_scalar(output, static_cast<std::uint64_t>(state.markers.size()));
    write_scalar(output, state.time);
    write_scalar(output, state.step);
    write_scalar(output, state.initialization_seed);
    write_scalar(output, state.initialization_counter);
    write_scalar(output, state.collision_sampler.seed);
    write_scalar(output, state.collision_sampler.next_counter);
    write_scalar(output, state.potential_event_age);
    write_scalar(output, state.collision_event_age);
    write_scalar(output, state.collision_timescale_estimate);
    write_scalar(output, state.potential_timescale_estimate);

    // Pack bounded blocks in the explicit scalar order. The on-disk record
    // size is independent of the host Marker struct's alignment or padding.
    constexpr std::size_t record_bytes =
        8 * sizeof(double) + 2 * sizeof(std::uint64_t) + sizeof(std::uint8_t);
    constexpr std::size_t block_markers = 16384;
    std::vector<char> buffer(record_bytes * block_markers);
    for (std::size_t begin = 0; begin < state.markers.size();
         begin += block_markers) {
        const std::size_t end = std::min(
            state.markers.size(), begin + block_markers);
        char* cursor = buffer.data();
        auto pack = [&](const auto& value) {
            std::memcpy(cursor, &value, sizeof(value));
            cursor += sizeof(value);
        };
        for (std::size_t i = begin; i < end; ++i) {
            const Marker& marker = state.markers[i];
            pack(marker.mass);
            pack(marker.r);
            pack(marker.w);
            pack(marker.eta);
            pack(marker.internal_energy);
            pack(marker.internal_velocity_dispersion);
            pack(marker.id);
            pack(marker.source_cell);
            const std::uint8_t marker_state =
                static_cast<std::uint8_t>(marker.state);
            pack(marker_state);
            pack(marker.inner_boundary_speed);
            pack(marker.remaining_inner_time);
        }
        output.write(buffer.data(), cursor - buffer.data());
        if (!output) {
            throw std::runtime_error("Failed writing GPU WPIC marker block.");
        }
    }
    // Finish and close the temporary file before replacing the public path.
    // Readers then open either the previous complete file or the new one.
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed flushing GPU WPIC restart.");
    }
    output.close();
    if (!output || std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("Could not install GPU WPIC restart: " + path);
    }
}

}  // namespace venus_gpu
