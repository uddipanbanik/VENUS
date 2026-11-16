// Measure collective spherical VP evolution across GPU particle owners.
// Inputs are global particle count, radial cell count, and timed KDK steps.
// Global IDs determine one reproducible physical sample, divided among ranks.
// Each rank advances its particles in the common field from summed radial
// moments. After warmup and a common start, the slowest-rank wall time sets
// the reported global throughput. Rank zero writes the result. Launch one
// process per GPU with the same shared mesh and physical options.

#include "venus_gpu/wpic_mpi.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// Global particle IDs determine the sample, so changing rank count
// redistributes one physical population rather than resampling it.
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
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Bind before constructing persistent GPU state. One MPI process owns
    // each visible GPU used by this benchmark.
    const int device =
        venus_gpu::bind_cuda_device_to_mpi_local_rank(MPI_COMM_WORLD);
    const std::size_t global_count = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
        : 2000000;
    const int cells = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 40;
    // Every rank builds the same radial mesh for collective moment sums.
    std::vector<double> edge(static_cast<std::size_t>(cells) + 1);
    for (int i = 0; i <= cells; ++i) {
        edge[static_cast<std::size_t>(i)] =
            0.05 + 19.95 * static_cast<double>(i) / cells;
    }
    const venus_gpu::RadialMesh mesh = venus_gpu::RadialMesh::from_edges(edge);
    const double q0 = std::pow(edge.front(), 3) / 3.0;
    const double q1 = std::pow(edge.back(), 3) / 3.0;
    // Striding global IDs assigns each particle to exactly one rank while
    // preserving unit global mass and the common volume-uniform sample.
    std::vector<venus_gpu::Marker> local;
    local.reserve((global_count + ranks - 1) / ranks);
    for (std::size_t global = static_cast<std::size_t>(rank);
         global < global_count;
         global += static_cast<std::size_t>(ranks)) {
        const double q = q0 + (q1 - q0) * uniform(global + 11);
        venus_gpu::Marker marker;
        marker.mass = 1.0 / global_count;
        marker.r = std::cbrt(3.0 * q);
        marker.w = 0.1 * (2.0 * uniform(global + 23) - 1.0);
        marker.eta = 0.01 * uniform(global + 37);
        marker.id = global;
        local.push_back(marker);
    }
    // The MPI VP module all-reduces deposited moments at both KDK field
    // evaluations, then moves only this rank's local particles.
    venus_gpu::WpicState state(local);
    venus_gpu::MpiVlasovPoissonModule vp(mesh, MPI_COMM_WORLD);
    venus_gpu::VlasovPoissonOptions options;
    options.gravity = 1.0;
    for (int warmup = 0; warmup < 3; ++warmup) {
        vp.advance_kdk(state, warmup * 1.0e-3, 1.0e-3, options);
    }
    cudaDeviceSynchronize();
    // Start timing after every rank has completed warmup. The slowest
    // rank determines the wall time for the collective calculation.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        vp.advance_kdk(state, step * 1.0e-3, 1.0e-3, options);
    }
    cudaDeviceSynchronize();
    const double local_seconds = MPI_Wtime() - start;
    double seconds = local_seconds;
    MPI_Allreduce(
        MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // A single rank publishes global throughput to avoid duplicate results.
    if (rank == 0) {
        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, device);
        std::cout << std::setprecision(6)
                  << "mpi_ranks=" << ranks << " gpu_per_rank="
                  << properties.name << '\n'
                  << "global_markers=" << global_count
                  << " radial_cells=" << cells << " vp_steps=" << steps
                  << '\n'
                  << "vp_seconds=" << seconds
                  << " global_marker_steps_per_second="
                  << global_count * steps / seconds << '\n';
    }
    MPI_Finalize();
    return 0;
}
