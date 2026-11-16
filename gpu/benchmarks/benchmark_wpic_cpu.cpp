// Measure the maintained CPU routines on the GPU benchmark's initial sample.
// Inputs are particle count, radial cell count, timed KDK steps, and CPU worker
// count. The index-keyed generator supplies the same particles as the GPU
// harness. Threaded deposition, the finite-volume Poisson solve, complete KDK,
// and material encounters produce wall times and trial/acceptance counts on
// stdout. Comparison requires the same physical/numerical controls, workload,
// build mode, and a stated host worker placement.

#include "ev/weighted_vlasov_pic_collisions.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// Use the same index-keyed generator as the GPU workload so its initial
// positions, velocities, angular momenta, and IDs agree.
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

// Build the CPU mesh fields explicitly from the GPU benchmark's radial
// faces. Shell volume is 4*pi*dq with q=r^3/3.
ev::WeightedPICRadialMesh make_mesh(int cells) {
    ev::WeightedPICRadialMesh mesh;
    mesh.edge.resize(static_cast<std::size_t>(cells) + 1);
    mesh.volume_coordinate_center.resize(cells);
    mesh.shell_volume.resize(cells);
    mesh.venus_center.resize(cells);
    mesh.venus_width.resize(cells);
    for (int i = 0; i <= cells; ++i) {
        mesh.edge[static_cast<std::size_t>(i)] =
            0.05 + 19.95 * static_cast<double>(i) / cells;
    }
    for (int i = 0; i < cells; ++i) {
        const double left = mesh.edge[static_cast<std::size_t>(i)];
        const double right = mesh.edge[static_cast<std::size_t>(i + 1)];
        const double q0 = left * left * left / 3.0;
        const double q1 = right * right * right / 3.0;
        mesh.volume_coordinate_center[static_cast<std::size_t>(i)] =
            0.5 * (q0 + q1);
        mesh.shell_volume[static_cast<std::size_t>(i)] =
            4.0 * ev::PI * (q1 - q0);
        mesh.venus_center[static_cast<std::size_t>(i)] = 0.5 * (left + right);
        mesh.venus_width[static_cast<std::size_t>(i)] = right - left;
    }
    return mesh;
}

// Select the CPU reconstruction and KDK characteristic path corresponding
// to the GPU benchmark's static, reflecting finite-volume problem.
ev::Params make_parameters(const ev::WeightedPICRadialMesh& mesh, int cells) {
    ev::Params parameters;
    parameters.physics_model = "vlasov_poisson";
    parameters.vp_universe = "static";
    parameters.vp_gravity = 1.0;
    parameters.vp_poisson_reconstruction = "constant_line_mass";
    parameters.vp_interpolation = "hamiltonian_conservative";
    parameters.kinetic_grid = "cell_average_eta";
    parameters.radial_grid = "linear";
    parameters.integrator = "semi_lagrangian_kick_drift_kick";
    parameters.inner_boundary = "reflecting";
    parameters.outer_boundary = "reflecting";
    parameters.Nr = cells;
    parameters.r_min = mesh.edge.front();
    parameters.r_max = mesh.edge.back();
    return parameters;
}

// Deposit and solve the incoming field, perform the first kick and drift,
// then deposit the moved particles for the endpoint field and second kick.
void advance(
    std::vector<ev::WeightedPICMarker>& markers,
    const ev::WeightedPICRadialMesh& mesh,
    const std::vector<double>& radius,
    const std::vector<double>& radial_coordinate,
    const ev::Params& parameters,
    double time,
    double dt,
    unsigned int threads) {
    const ev::WeightedPICDepositedMoments old_deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            markers, mesh, ev::WeightedPICParticleShape::cic, threads, false);
    const ev::VlasovPoissonField old_field =
        ev::solve_weighted_pic_poisson_field(
            old_deposit, mesh, time, parameters);
    ev::begin_weighted_pic_self_consistent_kdk_parallel(
        markers, time, dt, radial_coordinate, radius, old_field, parameters,
        threads);
    // The second force is evaluated from the drifted population; reusing the
    // incoming field here would remove the self-consistent KDK field update.
    const ev::WeightedPICDepositedMoments new_deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            markers, mesh, ev::WeightedPICParticleShape::cic, threads, false);
    const ev::VlasovPoissonField new_field =
        ev::solve_weighted_pic_poisson_field(
            new_deposit, mesh, time + dt, parameters);
    ev::finish_weighted_pic_self_consistent_kdk_parallel(
        markers, time + dt, dt, radial_coordinate, radius, new_field,
        parameters, threads);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t count = argc > 1
        ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
        : 1000000;
    const int cells = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 40;
    // The optional fourth argument sets CPU deposition/transport workers.
    // Include it in performance comparisons because it changes host resources.
    const unsigned int threads = argc > 4
        ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10))
        : 1;
    const ev::WeightedPICRadialMesh mesh = make_mesh(cells);
    const ev::Params parameters = make_parameters(mesh, cells);
    const std::vector<double> radius = mesh.venus_center;
    const std::vector<double> radial_coordinate = radius;
    // The volume-coordinate sample and each index key reproduce the GPU
    // initial population, with unit total mass.
    const double q0 = std::pow(mesh.edge.front(), 3) / 3.0;
    const double q1 = std::pow(mesh.edge.back(), 3) / 3.0;
    std::vector<ev::WeightedPICMarker> markers(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double q = q0 + (q1 - q0) * uniform(i + 11);
        markers[i].mass = 1.0 / count;
        markers[i].r = std::cbrt(3.0 * q);
        markers[i].w = 0.1 * (2.0 * uniform(i + 23) - 1.0);
        markers[i].eta = 0.01 * uniform(i + 37);
        markers[i].id = i;
    }
    // Exercise allocation and production routines before the timed steps.
    for (int warmup = 0; warmup < 3; ++warmup) {
        advance(
            markers, mesh, radius, radial_coordinate, parameters,
            warmup * 1.0e-3, 1.0e-3, threads);
    }
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        advance(
            markers, mesh, radius, radial_coordinate, parameters,
            step * 1.0e-3, 1.0e-3, threads);
    }
    const std::chrono::steady_clock::time_point stop =
        std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(stop - start).count();

    // Time the same constant-cross-section material interval after the
    // orbit workload, using the maintained CPU collision implementation.
    ev::Params collision_parameters = parameters;
    collision_parameters.bp_cross_section = "constant";
    collision_parameters.bp_boltzmann_kernel = "isotropic";
    collision_parameters.bp_collision_strength = 1.0;
    collision_parameters.bp_sigma0 = 0.5;
    ev::WeightedPICCollisionOptions collision_options;
    collision_options.target_shell_markers = 4096;
    collision_options.minimum_shell_markers = 32;
    collision_options.collision_cfl = 0.2;
    ev::WeightedPICCollisionSamplerState sampler;
    ev::WeightedPICCollisionWorkspace collision_workspace;
    const std::chrono::steady_clock::time_point collision_start =
        std::chrono::steady_clock::now();
    const ev::WeightedPICCollisionDiagnostics collision =
        ev::apply_weighted_pic_collision_interval(
            markers,
            mesh,
            1000.0,
            collision_parameters,
            collision_options,
            sampler,
            collision_workspace,
            threads);
    const std::chrono::steady_clock::time_point collision_stop =
        std::chrono::steady_clock::now();
    const double collision_seconds = std::chrono::duration<double>(
        collision_stop - collision_start).count();
    // Report workload sizes, worker count, and actual trial/accepted counts
    // beside the timings; stochastic work can differ at different settings.
    std::cout << std::setprecision(6)
              << "cpu_reference=VENUS"
              << " cpu_threads=" << threads
              << " markers=" << count
              << " radial_cells=" << cells
              << " vp_steps=" << steps << '\n'
              << "cpu_seconds=" << seconds
              << " cpu_marker_steps_per_second="
              << count * steps / seconds << '\n'
              << "cpu_collision_seconds=" << collision_seconds
              << " cpu_collision_trials=" << collision.trial_pairs
              << " cpu_collision_accepted=" << collision.accepted_pairs
              << '\n';
    return 0;
}
