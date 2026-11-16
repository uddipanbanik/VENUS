// Own GPU work arrays and report CUDA errors at explicit call boundaries.
// Inputs are element counts, host arrays for upload, and runtime status codes.
// DeviceArray returns a device pointer or downloaded host values; it releases
// its allocation when its owner is destroyed. Moving an array transfers that
// ownership and clears the old pointer. Resizing discards old contents, while
// explicit copies identify host/device synchronization points. The caller
// selects the CUDA device before creating these arrays.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace venus_gpu {
namespace detail {

// Convert a CUDA status into a host exception at the call boundary. Source
// context identifies the failed allocation, copy, or runtime operation.
inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':'
            << line << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

#define VGPU_CUDA_CHECK(expr) \
    ::venus_gpu::detail::cuda_check((expr), #expr, __FILE__, __LINE__)

// A single owner releases each allocation. Moving transfers ownership and
// clears the source pointer, preventing a second free of the same device data.
template <class T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t size) { resize(size); }
    ~DeviceArray() { reset(); }

    DeviceArray(DeviceArray&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Workspaces retain an allocation when their element count agrees. A
    // changed count discards old contents before allocating the new buffer.
    void resize(std::size_t size) {
        if (size == size_) {
            return;
        }
        reset();
        if (size != 0) {
            VGPU_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&data_), size * sizeof(T)));
            size_ = size;
        }
    }

    // Cleanup is noexcept, including during exception unwinding. Callers
    // check runtime failures at explicit work/transfer boundaries.
    void reset() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        size_ = 0;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    // Reset accumulated moments or flags before the next independent kernel
    // contribution; this operation does not resize the workspace.
    void zero() {
        if (size_ != 0) {
            VGPU_CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
        }
    }

    // Transfers occur explicitly at host/device boundaries. Upload sizes
    // storage to its input; download requires the exact allocated count.
    void upload(const T* source, std::size_t count) {
        if (count != size_) {
            resize(count);
        }
        if (count != 0) {
            VGPU_CUDA_CHECK(cudaMemcpy(
                data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void download(T* destination, std::size_t count) const {
        if (count != size_) {
            throw std::runtime_error("DeviceArray download size mismatch.");
        }
        if (count != 0) {
            VGPU_CUDA_CHECK(cudaMemcpy(
                destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

// Check launch status here. Synchronizing copies and other explicit CUDA
// boundaries subsequently expose errors from asynchronously executed work.
inline void check_kernel(const char* context) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << context << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

}  // namespace detail
}  // namespace venus_gpu
