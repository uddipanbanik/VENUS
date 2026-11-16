#pragma once

// Store the spherical particle population on a CUDA device. Host Marker records
// are the input to upload(); download() returns records for output, checkpoints,
// or MPI redistribution. Physics modules share the same device arrays.
// Each record carries a physical weight and orbit coordinates, together with
// boundary and subhalo internal states. The rollback buffer saves the orbit
// coordinates needed to retry a kick-drift-kick step in a rebuilt field.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace venus_gpu {

constexpr double pi = 3.141592653589793238462643383279502884;

// A delayed particle is following its unresolved inner orbit; an escaped
// particle remains in the mass ledger while leaving the resolved force source.
enum class MarkerState : std::uint8_t {
    active = 0,
    delayed_inner = 1,
    escaped = 2,
};

// Here r is physical radius, w is radial peculiar velocity in an expanding
// background (radial velocity in a static one), and eta=L^2/2 for specific
// angular momentum L. Mass is the gravitational weight represented by a record.
struct Marker {
    double mass = 0.0;
    double r = 0.0;
    double w = 0.0;
    double eta = 0.0;
    // IDs follow particles through sorting and capture; source_cell records
    // their origin rather than their current radial position.
    std::uint64_t id = 0;
    std::uint64_t source_cell = 0;
    MarkerState state = MarkerState::active;
    // The delayed-inner state stores the speed and remaining travel time
    // needed to return through the unresolved central region.
    double inner_boundary_speed = 0.0;
    double remaining_inner_time = 0.0;
    // Subhalo encounters transfer orbital energy into internal energy and
    // update the one-dimensional internal velocity dispersion.
    double internal_energy = 0.0;
    double internal_velocity_dispersion = 0.0;
};

// This non-owning view gives kernels contiguous arrays for coalesced memory
// access. The owning WpicState must outlive every kernel using these pointers.
struct MarkerDeviceView {
    double* mass = nullptr;
    double* r = nullptr;
    double* w = nullptr;
    double* eta = nullptr;
    std::uint64_t* id = nullptr;
    std::uint64_t* source_cell = nullptr;
    std::uint8_t* state = nullptr;
    double* inner_boundary_speed = nullptr;
    double* remaining_inner_time = nullptr;
    double* internal_energy = nullptr;
    double* internal_velocity_dispersion = nullptr;
    std::size_t size = 0;
};

class WpicState {
public:
    WpicState();
    explicit WpicState(const std::vector<Marker>& markers);
    ~WpicState();
    WpicState(WpicState&&) noexcept;
    WpicState& operator=(WpicState&&) noexcept;
    WpicState(const WpicState&) = delete;
    WpicState& operator=(const WpicState&) = delete;

    // Upload replaces the complete population; download preserves its order
    // and every marker field so restart and collision identities stay aligned.
    void upload(const std::vector<Marker>& markers);
    std::vector<Marker> download() const;
    std::size_t size() const noexcept;
    MarkerDeviceView device_view() const noexcept;

private:
    // Capture marks one partner for removal. Stable compaction applies the
    // same selection to every array so mass, orbit, ID, and internal state agree.
    void compact_removed_device(
        const std::uint8_t* device_remove,
        std::size_t removed_count);
    friend class SubhaloCollisionModule;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class WpicOrbitRollbackBuffer {
public:
    WpicOrbitRollbackBuffer();
    ~WpicOrbitRollbackBuffer();
    WpicOrbitRollbackBuffer(WpicOrbitRollbackBuffer&&) noexcept;
    WpicOrbitRollbackBuffer& operator=(WpicOrbitRollbackBuffer&&) noexcept;
    WpicOrbitRollbackBuffer(const WpicOrbitRollbackBuffer&) = delete;
    WpicOrbitRollbackBuffer& operator=(const WpicOrbitRollbackBuffer&) = delete;

    // Capture precedes a trial orbit update. Restore recovers the incoming
    // orbit after a rejected endpoint, before another trial is attempted.
    void capture(const WpicState& state);
    void restore(WpicState& state) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Throws std::runtime_error when no CUDA device is visible.
void require_cuda_device();

}  // namespace venus_gpu
