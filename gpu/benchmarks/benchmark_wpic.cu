// Measure the cost of GPU orbit evolution and elastic material encounters.
// Inputs are particle count, radial cell count, and timed KDK step count.
// An index-keyed generator creates a unit-mass volume-uniform population and
// uploads it once. The harness reports synchronized wall time, particle-step
// throughput, and collision trial/acceptance counts. Focused entry files choose
// one operator; the aggregate entry times both. Use one GPU and compare the
// same workload/physical controls when interpreting the reported throughput.

#include "venus_gpu/wpic.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#if defined(VENUS_BENCHMARK_VP_ONLY) && defined(VENUS_BENCHMARK_MATERIAL_ONLY)
#error "A focused benchmark must select exactly one workload."
#endif

namespace {

// Generate each sample from its index so identical workload sizes use the
// same physical particles regardless of GPU scheduling.
std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

double uniform(std::uint64_t value) {
    return (static_cast<double>(mix64(value) >> 11) + 0.5)
        * (1.0 / 9007199254740992.0);
}

}  // namespace

int main(int argc, char** argv) {
    venus_gpu::require_cuda_device();
    const std::size_t count = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
        : 1000000;
    const int cells = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 40;
    // The radial mesh is linear in radius. Sample uniformly in q=r^3/3
    // below so the initial particle density is uniform in physical volume.
    std::vector<double> edge(static_cast<std::size_t>(cells) + 1);
    for (int i = 0; i <= cells; ++i) {
        edge[static_cast<std::size_t>(i)] =
            0.05 + 19.95 * static_cast<double>(i) / cells;
    }
    const venus_gpu::RadialMesh mesh = venus_gpu::RadialMesh::from_edges(edge);
    const double q0 = std::pow(edge.front(), 3) / 3.0;
    const double q1 = std::pow(edge.back(), 3) / 3.0;
    // Equal weights sum to unit mass. Radial velocity and eta=L^2/2 are
    // sampled reproducibly to exercise both gravity and angular drift.
    std::vector<venus_gpu::Marker> markers(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double q = q0 + (q1 - q0) * uniform(i + 11);
        markers[i].mass = 1.0 / count;
        markers[i].r = std::cbrt(3.0 * q);
        markers[i].w = 0.1 * (2.0 * uniform(i + 23) - 1.0);
        markers[i].eta = 0.01 * uniform(i + 37);
        markers[i].id = i;
    }
    venus_gpu::WpicState state(markers);
#ifndef VENUS_BENCHMARK_MATERIAL_ONLY
    venus_gpu::VlasovPoissonModule vp(mesh);
    venus_gpu::VlasovPoissonOptions vp_options;
    vp_options.gravity = 1.0;
    // Warmup populates persistent buffers and executes the actual field and
    // orbit paths before timing. Synchronization closes all device work.
    for (int warmup = 0; warmup < 3; ++warmup) {
        vp.advance_kdk(state, warmup * 1.0e-3, 1.0e-3, vp_options);
    }
    cudaDeviceSynchronize();
    // Measure complete KDK steps, including their deposition and field
    // solves; the final synchronization includes asynchronously queued work.
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        vp.advance_kdk(state, step * 1.0e-3, 1.0e-3, vp_options);
    }
    cudaDeviceSynchronize();
    const auto stop = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(stop - start).count();
#endif

#ifndef VENUS_BENCHMARK_VP_ONLY
    // The material workload uses a constant cross section and local shells.
    // Its long supplied collision interval exercises internal CFL substeps.
    venus_gpu::CollisionOptions collision_options;
    collision_options.sigma0 = 0.5;
    collision_options.target_shell_markers = 4096;
    collision_options.minimum_shell_markers = 32;
    collision_options.collision_cfl = 0.2;
    venus_gpu::CollisionSamplerState sampler;
    venus_gpu::CollisionModule collisions(mesh);
    cudaDeviceSynchronize();
    const auto collision_start = std::chrono::steady_clock::now();
    const venus_gpu::CollisionDiagnostics collision_diagnostics =
        collisions.apply(state, 1000.0, collision_options, sampler);
    cudaDeviceSynchronize();
    const auto collision_stop = std::chrono::steady_clock::now();
    const double collision_seconds = std::chrono::duration<double>(
        collision_stop - collision_start).count();
#endif

    // Report the device and workload with the timing so particle-step and
    // encounter throughput can be compared at identical settings.
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::cout << std::setprecision(6)
              << "gpu=" << properties.name << '\n'
              << "markers=" << count << " radial_cells=" << cells;
#ifndef VENUS_BENCHMARK_MATERIAL_ONLY
    std::cout << " vp_steps=" << steps << '\n'
              << "vp_seconds=" << seconds
              << " vp_steps_per_second=" << steps / seconds
              << " marker_steps_per_second=" << count * steps / seconds << '\n';
#else
    std::cout << '\n';
#endif
#ifndef VENUS_BENCHMARK_VP_ONLY
    std::cout << "collision_seconds=" << collision_seconds
              << " collision_trials=" << collision_diagnostics.trial_pairs
              << " collision_accepted=" << collision_diagnostics.accepted_pairs
              << '\n';
#endif
    return 0;
}
