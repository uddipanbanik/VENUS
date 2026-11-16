// Measure potential/force changes on the complete current GPU population.
// Inputs are particle arrays, inner-field values at two epochs, and the common
// mesh boundaries. A fixed device reduction returns block mass and quadratic
// field moments; host finishing verifies their population and mass closure.
// An optional bounded gather returns actual hidden-cloud records for inspection.
// All kernels here are read-only with respect to particles and random state.

#include "venus_gpu/receiver_sums.hpp"
#include <cuda_runtime.h>
#include <climits>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

namespace venus_gpu_receiver {
namespace {
constexpr unsigned width = 256;
constexpr unsigned long long absent = ~0ULL;

void cuda_check(cudaError_t status, const char* context) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

// Assign one particle to each lane. A fixed binary tree sums the mass and
// field-square terms within each block; the host finishes blocks in order.
__global__ void receiver_sums_kernel(venus_gpu::MarkerDeviceView markers,
        double outer, double qcenter0, InnerField previous, InnerField current,
        ReceiverTerms* partials, unsigned long long* first_hidden, int* invalid) {
    __shared__ ReceiverTerms shared[width];
    __shared__ unsigned long long minimum[width];
    const unsigned lane = threadIdx.x;
    const std::size_t index = std::size_t(blockIdx.x) * width + lane;
    ReceiverTerms value{};
    unsigned long long first = absent;
    if (blockDim.x != width) {
        if (lane == 0) atomicExch(invalid, 1);
        return;
    }
    // The complete measure requires every supplied particle to be active,
    // finite, positively weighted, and within the modeled outer radius.
    if (index < markers.size) {
        const double m = markers.mass[index], r = markers.r[index];
        if (markers.state[index] != 0 || !isfinite(m) || !(m > 0) ||
            !isfinite(m*m) || !isfinite(r) || r < 0 || r > outer) {
            atomicExch(invalid, 2);
        } else {
            value = terms(m, r, qcenter0, previous, current);
            if (!isfinite(value.hidden_mass) || !isfinite(value.phi0_square) ||
                !isfinite(value.phi1_square) || !isfinite(value.phi_difference_square) ||
                !isfinite(value.force0_square) || !isfinite(value.force1_square) ||
                !isfinite(value.force_difference_square)) atomicExch(invalid, 3);
            if (value.hidden_count) first = index;
        }
    }
    // Out-of-range lanes contribute zero. Every lane still participates in
    // the reduction barriers, including lanes in a partially filled block.
    shared[lane] = value;
    minimum[lane] = first;
    __syncthreads();
    // This prescribed reduction tree fixes the arithmetic order within a
    // block and also identifies its earliest hidden receiver.
    for (unsigned distance = width/2; distance > 0; distance /= 2) {
        if (lane < distance) {
            shared[lane] = add(shared[lane], shared[lane+distance]);
            minimum[lane] = min(minimum[lane], minimum[lane+distance]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        partials[blockIdx.x] = shared[0];
        first_hidden[blockIdx.x] = minimum[0];
    }
}

struct SampleBlock { unsigned long long block,offset; };

// A separate bounded diagnostic gather leaves the full reduction intact.
// Prefix counts give hidden receivers unique, input-ordered sample slots.
__global__ void gather_samples_kernel(venus_gpu::MarkerDeviceView markers,
        const SampleBlock* plan, std::size_t limit, double qcenter0,
        InnerField previous, InnerField current, ReceiverSample* samples) {
    __shared__ unsigned prefix[width];
    const unsigned lane = threadIdx.x;
    const std::size_t index = plan[blockIdx.x].block*width+lane;
    const double beta = index < markers.size
        ? hidden_fraction(markers.r[index],current.inner_radius,qcenter0) : 0.;
    const bool weighted = index < markers.size && markers.mass[index]*beta > 0;
    prefix[lane] = weighted ? 1 : 0;
    __syncthreads();
    // Read the previous prefix stage before any lane writes the next one;
    // the barriers preserve the scan dependency across the block.
    for (unsigned offset=1; offset<width; offset*=2) {
        const unsigned left = lane>=offset ? prefix[lane-offset] : 0;
        __syncthreads();
        prefix[lane] += left;
        __syncthreads();
    }
    if (!weighted) return;
    const auto destination=plan[blockIdx.x].offset+prefix[lane]-1;
    if (destination>=limit) return;
    ReceiverSample sample;
    sample.index = index;
    sample.id = markers.id[index];
    sample.radius = markers.r[index];
    sample.mass = markers.mass[index];
    sample.hidden_weight = beta;
    sample.previous_potential = potential(sample.radius,previous);
    sample.current_potential = potential(sample.radius,current);
    sample.previous_force = force(sample.radius,previous);
    sample.current_force = force(sample.radius,current);
    samples[destination] = sample;
}
} // namespace

// Reuse reduction buffers between observations. Diagnostic storage is
// capped at 64 records and therefore scales independently of particle count.
struct DeviceReceivers::Impl {
    ReceiverTerms* partials = nullptr;
    unsigned long long* first_hidden = nullptr;
    SampleBlock* sample_indices = nullptr;
    ReceiverSample* samples = nullptr;
    int* invalid = nullptr;
    std::size_t capacity = 0;
    Impl() {
        try {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&invalid),sizeof(int)),"receiver invalid allocation");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&sample_indices),64*sizeof(*sample_indices)),"receiver sample indices allocation");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&samples),64*sizeof(*samples)),"receiver samples allocation");
        } catch (...) {
            cudaFree(invalid); cudaFree(sample_indices); cudaFree(samples);
            throw;
        }
    }
    ~Impl() {
        cudaFree(partials); cudaFree(first_hidden); cudaFree(sample_indices);
        cudaFree(samples); cudaFree(invalid);
    }
    // Allocate the replacement pair before publishing it, retaining the old
    // capacity if either new allocation fails.
    void reserve(std::size_t blocks) {
        if (blocks <= capacity) return;
        ReceiverTerms* new_partials = nullptr;
        unsigned long long* new_first = nullptr;
        try {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&new_partials),blocks*sizeof(*partials)),"receiver block allocation");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&new_first),blocks*sizeof(*first_hidden)),"receiver index allocation");
        } catch (...) {
            cudaFree(new_partials); cudaFree(new_first);
            throw;
        }
        cudaFree(partials); cudaFree(first_hidden);
        partials = new_partials; first_hidden = new_first; capacity = blocks;
    }
};

DeviceReceivers::DeviceReceivers() : impl_(new Impl) {}
DeviceReceivers::~DeviceReceivers() = default;

// Validate the common uniform-interior field geometry, reduce all supplied
// particles, and optionally download the first bounded hidden-receiver sample.
DeviceReceiverResult DeviceReceivers::observe(const venus_gpu::WpicState& state,
        double outer, double qcenter0, InnerField previous, InnerField current,
        std::size_t sample_limit) {
    const auto reduction_start = std::chrono::steady_clock::now();
    require(state.size() > 0 && sample_limit <= 64,
        "Receiver observation requires active particles and at most64 samples.");
    const auto valid = [](const InnerField& f) {
        return std::isfinite(f.inner_radius) && f.inner_radius > 0
            && std::isfinite(f.inner_face_potential)
            && std::isfinite(f.inner_source_mass) && f.inner_source_mass >= 0
            && std::isfinite(f.gravity) && f.gravity > 0;
    };
    require(valid(previous) && valid(current)
        && previous.inner_radius == current.inner_radius
        && previous.gravity == current.gravity
        && std::isfinite(outer) && outer > current.inner_radius
        && std::isfinite(qcenter0)
        && qcenter0 > current.inner_radius*current.inner_radius*current.inner_radius/3.,
        "Unsupported or invalid uniform-inner receiver fields.");
    // The TSC transition occupies a positive representable volume interval;
    // its geometry must agree with the deposition hidden-weight calculation.
    const double face_q=current.inner_radius*current.inner_radius*current.inner_radius/3.;
    require(std::isfinite(face_q) && face_q>0
        && std::isfinite(2*(qcenter0-face_q)) && 2*(qcenter0-face_q)>0,
        "Receiver transition volume is unrepresentable.");
    const std::size_t blocks = (state.size()+width-1)/width;
    require(blocks <= INT_MAX, "Receiver grid exceeds CUDA grid bound.");
    impl_->reserve(blocks);
    cuda_check(cudaMemset(impl_->invalid,0,sizeof(int)),"receiver reset");
    receiver_sums_kernel<<<static_cast<unsigned>(blocks),width>>>(state.device_view(),
        outer,qcenter0,previous,current,impl_->partials,impl_->first_hidden,impl_->invalid);
    cuda_check(cudaGetLastError(),"receiver reduction launch");
    int invalid = 0;
    cuda_check(cudaMemcpy(&invalid,impl_->invalid,sizeof(int),cudaMemcpyDeviceToHost),"receiver validity download");
    require(invalid == 0, "Device receiver input or quadrature is invalid.");
    // Download only block partials for the full measure. Their active count
    // must close the complete input count before a field clock can use them.
    std::vector<ReceiverTerms> host(blocks);
    cuda_check(cudaMemcpy(host.data(),impl_->partials,blocks*sizeof(ReceiverTerms),cudaMemcpyDeviceToHost),"receiver block download");
    DeviceReceiverResult result;
    result.sum = finish_partials(host,state.size());
    result.reduction_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-reduction_start).count();
    const auto sample_start = std::chrono::steady_clock::now();
    if (sample_limit && result.sum.hidden_count) {
        // Build a small block plan from the full hidden counts. Sampling
        // ends at the requested limit and changes no particle or sum term.
        std::vector<SampleBlock> plan;
        unsigned long long found=0;
        for (std::size_t i=0;i<host.size() && found<sample_limit;++i) {
            if (host[i].hidden_count) {
                plan.push_back({static_cast<unsigned long long>(i),found});
                found+=host[i].hidden_count;
            }
        }
        const std::size_t count=std::min(sample_limit,static_cast<std::size_t>(result.sum.hidden_count));
        cuda_check(cudaMemcpy(impl_->sample_indices,plan.data(),plan.size()*sizeof(SampleBlock),cudaMemcpyHostToDevice),"receiver sample plan upload");
        gather_samples_kernel<<<static_cast<unsigned>(plan.size()),width>>>(state.device_view(),
            impl_->sample_indices,count,qcenter0,previous,current,impl_->samples);
        cuda_check(cudaGetLastError(),"receiver sample launch");
        result.first_hidden_samples.resize(count);
        cuda_check(cudaMemcpy(result.first_hidden_samples.data(),impl_->samples,
            count*sizeof(ReceiverSample),cudaMemcpyDeviceToHost),"receiver sample download");
    }
    result.sample_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-sample_start).count();
    return result;
}
} // namespace venus_gpu_receiver
