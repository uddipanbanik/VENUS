// Keep spherical particle records in persistent GPU arrays.
// Inputs are validated host Marker records or a device removal mask after
// capture. Upload/download transpose between complete records and separate
// arrays for each quantity. Stable compaction keeps the same survivors in all
// arrays. The rollback buffer separately copies r and w for a trial KDK step,
// retaining mass, identity, eta, and internal state. Host calls are serialized,
// and the caller selects its CUDA device before allocating particle storage.

#include "venus_gpu/radial_mesh.hpp"
#include "venus_gpu/state.hpp"

#include "cuda_utils.cuh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace venus_gpu {

namespace {

struct KeepUnremoved {
    __host__ __device__ bool operator()(std::uint8_t removed) const {
        return removed == 0;
    }
};

}  // namespace

using detail::DeviceArray;

// The same index across these arrays identifies one particle. Kernels can
// read one physical quantity contiguously while retaining that correspondence.
struct WpicState::Impl {
    DeviceArray<double> mass;
    DeviceArray<double> r;
    DeviceArray<double> w;
    DeviceArray<double> eta;
    DeviceArray<std::uint64_t> id;
    DeviceArray<std::uint64_t> source_cell;
    DeviceArray<std::uint8_t> state;
    DeviceArray<double> inner_boundary_speed;
    DeviceArray<double> remaining_inner_time;
    DeviceArray<double> internal_energy;
    DeviceArray<double> internal_velocity_dispersion;
    std::size_t size = 0;
};

WpicState::WpicState() : impl_(new Impl) {}

WpicState::WpicState(const std::vector<Marker>& markers) : WpicState() {
    upload(markers);
}

WpicState::~WpicState() = default;
WpicState::WpicState(WpicState&&) noexcept = default;
WpicState& WpicState::operator=(WpicState&&) noexcept = default;

// Validate a complete host population, transpose it into arrays, then copy
// those arrays to the device. Physical and boundary-state checks precede copies.
void WpicState::upload(const std::vector<Marker>& markers) {
    const std::size_t count = markers.size();
    std::vector<double> mass(count), r(count), w(count), eta(count);
    std::vector<double> inner_speed(count), remaining(count), internal(count),
        dispersion(count);
    std::vector<std::uint64_t> id(count), source_cell(count);
    std::vector<std::uint8_t> state(count);
    // Validate the complete host population before mutating any device array.
    // Invalid enum values and delayed-boundary clocks are rejected before
    // device storage is changed; CUDA transfer errors are reported separately.
    for (std::size_t i = 0; i < count; ++i) {
        const Marker& marker = markers[i];
        const std::uint8_t marker_state =
            static_cast<std::uint8_t>(marker.state);
        if (!(marker.mass > 0.0) || !std::isfinite(marker.mass)
            || !(marker.r >= 0.0) || !std::isfinite(marker.r)
            || !std::isfinite(marker.w) || !(marker.eta >= 0.0)
            || !std::isfinite(marker.eta)
            || marker_state > static_cast<std::uint8_t>(MarkerState::escaped)
            || !(marker.internal_energy >= 0.0)
            || !std::isfinite(marker.internal_energy)
            || !(marker.internal_velocity_dispersion >= 0.0)
            || !std::isfinite(marker.internal_velocity_dispersion)
            || !(marker.inner_boundary_speed >= 0.0)
            || !std::isfinite(marker.inner_boundary_speed)
            || !(marker.remaining_inner_time >= 0.0)
            || !std::isfinite(marker.remaining_inner_time)
            || (marker.state == MarkerState::delayed_inner
                && !(marker.remaining_inner_time > 0.0))) {
            throw std::runtime_error("Invalid weighted PIC marker upload.");
        }
        mass[i] = marker.mass;
        r[i] = marker.r;
        w[i] = marker.w;
        eta[i] = marker.eta;
        id[i] = marker.id;
        source_cell[i] = marker.source_cell;
        state[i] = marker_state;
        inner_speed[i] = marker.inner_boundary_speed;
        remaining[i] = marker.remaining_inner_time;
        internal[i] = marker.internal_energy;
        dispersion[i] = marker.internal_velocity_dispersion;
    }
    impl_->mass.upload(mass.data(), count);
    impl_->r.upload(r.data(), count);
    impl_->w.upload(w.data(), count);
    impl_->eta.upload(eta.data(), count);
    impl_->id.upload(id.data(), count);
    impl_->source_cell.upload(source_cell.data(), count);
    impl_->state.upload(state.data(), count);
    impl_->inner_boundary_speed.upload(inner_speed.data(), count);
    impl_->remaining_inner_time.upload(remaining.data(), count);
    impl_->internal_energy.upload(internal.data(), count);
    impl_->internal_velocity_dispersion.upload(dispersion.data(), count);
    impl_->size = count;
}

// Reassemble host records in their stored order for checkpoint/output or
// redistribution. Every physical and identity field uses the same index.
std::vector<Marker> WpicState::download() const {
    const std::size_t count = impl_->size;
    std::vector<double> mass(count), r(count), w(count), eta(count);
    std::vector<double> inner_speed(count), remaining(count), internal(count),
        dispersion(count);
    std::vector<std::uint64_t> id(count), source_cell(count);
    std::vector<std::uint8_t> state(count);
    impl_->mass.download(mass.data(), count);
    impl_->r.download(r.data(), count);
    impl_->w.download(w.data(), count);
    impl_->eta.download(eta.data(), count);
    impl_->id.download(id.data(), count);
    impl_->source_cell.download(source_cell.data(), count);
    impl_->state.download(state.data(), count);
    impl_->inner_boundary_speed.download(inner_speed.data(), count);
    impl_->remaining_inner_time.download(remaining.data(), count);
    impl_->internal_energy.download(internal.data(), count);
    impl_->internal_velocity_dispersion.download(dispersion.data(), count);

    std::vector<Marker> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i].mass = mass[i];
        result[i].r = r[i];
        result[i].w = w[i];
        result[i].eta = eta[i];
        result[i].id = id[i];
        result[i].source_cell = source_cell[i];
        result[i].state = static_cast<MarkerState>(state[i]);
        result[i].inner_boundary_speed = inner_speed[i];
        result[i].remaining_inner_time = remaining[i];
        result[i].internal_energy = internal[i];
        result[i].internal_velocity_dispersion = dispersion[i];
    }
    return result;
}

std::size_t WpicState::size() const noexcept { return impl_->size; }

// Pass pointers and the common count to kernels without transferring data
// or transferring ownership out of WpicState.
MarkerDeviceView WpicState::device_view() const noexcept {
    MarkerDeviceView view;
    view.mass = impl_->mass.data();
    view.r = impl_->r.data();
    view.w = impl_->w.data();
    view.eta = impl_->eta.data();
    view.id = impl_->id.data();
    view.source_cell = impl_->source_cell.data();
    view.state = impl_->state.data();
    view.inner_boundary_speed = impl_->inner_boundary_speed.data();
    view.remaining_inner_time = impl_->remaining_inner_time.data();
    view.internal_energy = impl_->internal_energy.data();
    view.internal_velocity_dispersion =
        impl_->internal_velocity_dispersion.data();
    view.size = impl_->size;
    return view;
}

struct WpicOrbitRollbackBuffer::Impl {
    DeviceArray<double> radius;
    DeviceArray<double> radial_velocity;
    std::size_t size = 0;
};

WpicOrbitRollbackBuffer::WpicOrbitRollbackBuffer() : impl_(new Impl) {}
WpicOrbitRollbackBuffer::~WpicOrbitRollbackBuffer() = default;
WpicOrbitRollbackBuffer::WpicOrbitRollbackBuffer(
    WpicOrbitRollbackBuffer&&) noexcept = default;
WpicOrbitRollbackBuffer& WpicOrbitRollbackBuffer::operator=(
    WpicOrbitRollbackBuffer&&) noexcept = default;

// Save only r and w: a trial KDK endpoint changes these orbit coordinates.
// Collision updates and population changes occur outside that trial.
void WpicOrbitRollbackBuffer::capture(const WpicState& state) {
    const MarkerDeviceView view = state.device_view();
    impl_->radius.resize(view.size);
    impl_->radial_velocity.resize(view.size);
    if (view.size != 0) {
        VGPU_CUDA_CHECK(cudaMemcpy(
            impl_->radius.data(), view.r, view.size * sizeof(double),
            cudaMemcpyDeviceToDevice));
        VGPU_CUDA_CHECK(cudaMemcpy(
            impl_->radial_velocity.data(), view.w,
            view.size * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    impl_->size = view.size;
}

void WpicOrbitRollbackBuffer::restore(WpicState& state) const {
    const MarkerDeviceView view = state.device_view();
    // Restoring onto a changed population would mix particle identities;
    // reject that state before any orbit coordinate is overwritten.
    if (view.size != impl_->size) {
        throw std::runtime_error(
            "GPU orbit rollback marker count changed inside a KDK probe.");
    }
    if (view.size != 0) {
        VGPU_CUDA_CHECK(cudaMemcpy(
            view.r, impl_->radius.data(), view.size * sizeof(double),
            cudaMemcpyDeviceToDevice));
        VGPU_CUDA_CHECK(cudaMemcpy(
            view.w, impl_->radial_velocity.data(),
            view.size * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

void WpicState::compact_removed_device(
    const std::uint8_t* device_remove,
    std::size_t removed_count) {
    if (removed_count == 0) {
        return;
    }
    if (removed_count > impl_->size || device_remove == nullptr) {
        throw std::runtime_error("Invalid GPU marker compaction request.");
    }
    // Allocate every destination before replacing the old arrays. Both
    // stable passes below must select exactly the same retained population.
    const std::size_t retained = impl_->size - removed_count;
    DeviceArray<double> mass(retained), r(retained), w(retained), eta(retained);
    DeviceArray<std::uint64_t> id(retained), source_cell(retained);
    DeviceArray<std::uint8_t> state(retained);
    DeviceArray<double> inner_speed(retained), remaining(retained),
        internal(retained), dispersion(retained);

    // Thrust 1.12 shipped with CUDA 11.4 supports at most ten tuple fields.
    // Compact the SoA in two stable passes with the same stencil; both passes
    // therefore retain the identical marker sequence.
    const auto input_primary = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(impl_->mass.data()),
        thrust::device_pointer_cast(impl_->r.data()),
        thrust::device_pointer_cast(impl_->w.data()),
        thrust::device_pointer_cast(impl_->eta.data()),
        thrust::device_pointer_cast(impl_->id.data()),
        thrust::device_pointer_cast(impl_->source_cell.data())));
    const auto output_primary = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(mass.data()),
        thrust::device_pointer_cast(r.data()),
        thrust::device_pointer_cast(w.data()),
        thrust::device_pointer_cast(eta.data()),
        thrust::device_pointer_cast(id.data()),
        thrust::device_pointer_cast(source_cell.data())));
    const auto primary_end = thrust::copy_if(
        input_primary,
        input_primary + static_cast<std::ptrdiff_t>(impl_->size),
        thrust::device_pointer_cast(device_remove),
        output_primary,
        KeepUnremoved{});

    const auto input_secondary = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(impl_->state.data()),
        thrust::device_pointer_cast(impl_->inner_boundary_speed.data()),
        thrust::device_pointer_cast(impl_->remaining_inner_time.data()),
        thrust::device_pointer_cast(impl_->internal_energy.data()),
        thrust::device_pointer_cast(
            impl_->internal_velocity_dispersion.data())));
    const auto output_secondary = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(state.data()),
        thrust::device_pointer_cast(inner_speed.data()),
        thrust::device_pointer_cast(remaining.data()),
        thrust::device_pointer_cast(internal.data()),
        thrust::device_pointer_cast(dispersion.data())));
    const auto secondary_end = thrust::copy_if(
        input_secondary,
        input_secondary + static_cast<std::ptrdiff_t>(impl_->size),
        thrust::device_pointer_cast(device_remove),
        output_secondary,
        KeepUnremoved{});
    if (static_cast<std::size_t>(primary_end - output_primary) != retained
        || static_cast<std::size_t>(secondary_end - output_secondary)
            != retained) {
        throw std::runtime_error("GPU marker compaction count mismatch.");
    }
    // Publish the compacted arrays together after both selected counts
    // close, preserving mass/orbit/internal-state correspondence.
    impl_->mass = std::move(mass);
    impl_->r = std::move(r);
    impl_->w = std::move(w);
    impl_->eta = std::move(eta);
    impl_->id = std::move(id);
    impl_->source_cell = std::move(source_cell);
    impl_->state = std::move(state);
    impl_->inner_boundary_speed = std::move(inner_speed);
    impl_->remaining_inner_time = std::move(remaining);
    impl_->internal_energy = std::move(internal);
    impl_->internal_velocity_dispersion = std::move(dispersion);
    impl_->size = retained;
}

// Integrate dV=4*pi*r^2*dr through q=r^3/3. Deposition uses q midpoints;
// radial field output uses arithmetic radial midpoints on the same cells.
RadialMesh RadialMesh::from_edges(const std::vector<double>& radial_edges) {
    if (radial_edges.size() < 2) {
        throw std::runtime_error("A radial mesh requires at least one cell.");
    }
    RadialMesh mesh;
    mesh.edge = radial_edges;
    const std::size_t cells = radial_edges.size() - 1;
    mesh.q_center.resize(cells);
    mesh.radius_center.resize(cells);
    mesh.shell_volume.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        const double left = radial_edges[i];
        const double right = radial_edges[i + 1];
        if (!(left > 0.0) || !(right > left) || !std::isfinite(right)) {
            throw std::runtime_error(
                "Radial faces must be finite, positive, and increasing.");
        }
        const double q0 = left * left * left / 3.0;
        const double q1 = right * right * right / 3.0;
        mesh.q_center[i] = 0.5 * (q0 + q1);
        mesh.radius_center[i] = 0.5 * (left + right);
        mesh.shell_volume[i] = 4.0 * pi * (q1 - q0);
    }
    return mesh;
}

// Check visibility before creating any persistent numerical workspace.
void require_cuda_device() {
    int devices = 0;
    VGPU_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < 1) {
        throw std::runtime_error(
            "The VENUS GPU module requires a visible CUDA GPU.");
    }
}

}  // namespace venus_gpu
