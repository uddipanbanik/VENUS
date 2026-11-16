#pragma once

// Measure the current active particle population without changing it.
// observe() takes the device state and previous/current interior fields and
// returns reduced receiver moments, timing, and an optional bounded sample.
// The observer requires active particles within the supplied outer radius.
// Device scratch storage persists between observations, and the caller owns
// the particle state throughout the measurement.

#include "receiver_sums_host.hpp"
#include "venus_gpu/state.hpp"
#include <memory>

namespace venus_gpu_receiver {
// Diagnostic samples retain IDs and actual particle radii so a hidden
// weight can be checked against the same deposition and interior field.
struct ReceiverSample {
    std::uint64_t index = 0, id = 0;
    double radius = 0, mass = 0, hidden_weight = 0;
    double previous_potential = 0, current_potential = 0;
    double previous_force = 0, current_force = 0;
};
// The full reduction supplies the clock; sampled records support only
// inspection and do not replace any part of the receiver quadrature.
struct DeviceReceiverResult {
    ReceiverSummary sum;
    double reduction_seconds = 0, sample_seconds = 0;
    std::vector<ReceiverSample> first_hidden_samples;
};
class DeviceReceivers {
public:
    DeviceReceivers();
    ~DeviceReceivers();
    DeviceReceivers(const DeviceReceivers&) = delete;
    DeviceReceivers& operator=(const DeviceReceivers&) = delete;
    // sample_limit controls only host diagnostic transfer. Every eligible
    // active particle participates in the reduction regardless of that limit.
    DeviceReceiverResult observe(const venus_gpu::WpicState& state,
        double outer, double qcenter0, InnerField previous, InnerField current,
        std::size_t sample_limit = 0);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace venus_gpu_receiver
