#include "venus_gpu/wpic_mpi.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <climits>
#include <vector>

namespace {

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

venus_gpu::RadialMesh make_mesh() {
    std::vector<double> edge(129);
    for (std::size_t i = 0; i < edge.size(); ++i) {
        edge[i] = 0.25 + 4.0 * static_cast<double>(i) / 128.0;
    }
    return venus_gpu::RadialMesh::from_edges(edge);
}

double local_mass(const std::vector<venus_gpu::Marker>& markers) {
    double result = 0.0;
    for (std::size_t i = 0; i < markers.size(); ++i) {
        result += markers[i].mass;
    }
    return result;
}

venus_gpu::Marker make_subhalo_marker(
    std::size_t global,
    std::size_t global_count) {
    venus_gpu::Marker marker;
    marker.mass = (0.5 + uniform(global + 59)) / global_count;
    marker.r = 0.3 + 3.9 * uniform(global + 71);
    marker.w = 0.12 * (2.0 * uniform(global + 83) - 1.0);
    marker.eta = 0.005 + 0.025 * uniform(global + 97);
    marker.id = global;
    marker.source_cell = global % 128;
    marker.internal_velocity_dispersion =
        0.08 + 0.04 * uniform(global + 109);
    marker.internal_energy = 1.5 * marker.mass
        * marker.internal_velocity_dispersion
        * marker.internal_velocity_dispersion;
    return marker;
}

std::vector<venus_gpu::Marker> gather_markers(
    const std::vector<venus_gpu::Marker>& local,
    int rank,
    int ranks) {
    const std::size_t local_bytes_size = local.size() * sizeof(venus_gpu::Marker);
    assert(local_bytes_size <= static_cast<std::size_t>(INT_MAX));
    const int local_bytes = static_cast<int>(local_bytes_size);
    std::vector<int> counts(static_cast<std::size_t>(ranks), 0);
    MPI_Gather(
        &local_bytes, 1, MPI_INT,
        counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displacements(static_cast<std::size_t>(ranks), 0);
    int total_bytes = 0;
    if (rank == 0) {
        for (int source = 0; source < ranks; ++source) {
            displacements[static_cast<std::size_t>(source)] = total_bytes;
            total_bytes += counts[static_cast<std::size_t>(source)];
        }
        assert(total_bytes % static_cast<int>(sizeof(venus_gpu::Marker)) == 0);
    }
    std::vector<venus_gpu::Marker> gathered(
        rank == 0
            ? static_cast<std::size_t>(total_bytes) / sizeof(venus_gpu::Marker)
            : 0);
    MPI_Gatherv(
        local.empty() ? nullptr : const_cast<venus_gpu::Marker*>(local.data()),
        local_bytes,
        MPI_BYTE,
        gathered.empty() ? nullptr : gathered.data(),
        counts.data(),
        displacements.data(),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);
    if (rank == 0) {
        std::sort(
            gathered.begin(),
            gathered.end(),
            [](const venus_gpu::Marker& first,
               const venus_gpu::Marker& second) {
                return first.id < second.id;
            });
    }
    return gathered;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    assert(ranks >= 2);
    const int device =
        venus_gpu::bind_cuda_device_to_mpi_local_rank(MPI_COMM_WORLD);
    const venus_gpu::RadialMesh mesh = make_mesh();
    const std::size_t global_count = 120000;
    std::vector<venus_gpu::Marker> local;
    for (std::size_t global = static_cast<std::size_t>(rank);
         global < global_count;
         global += static_cast<std::size_t>(ranks)) {
        venus_gpu::Marker marker;
        marker.mass = 1.0 / global_count;
        marker.r = 0.3 + 3.9 * uniform(global + 17);
        marker.w = 0.08 * (2.0 * uniform(global + 31) - 1.0);
        marker.eta = 0.01 + 0.02 * uniform(global + 47);
        marker.id = global;
        local.push_back(marker);
    }
    venus_gpu::WpicState state(local);
    venus_gpu::MpiVlasovPoissonModule vp(mesh, MPI_COMM_WORLD);
    venus_gpu::VlasovPoissonOptions vp_options;
    vp_options.gravity = 0.0;
    const venus_gpu::DepositionDiagnostics ledger =
        vp.advance_kdk(state, 0.0, 0.005, vp_options);
    const venus_gpu::FieldHost field = vp.download_global_field();
    assert(std::abs(ledger.deposited_mesh_mass - 1.0) < 5.0e-12);
    assert(std::abs(field.source_mass_edge.back() - 1.0) < 5.0e-12);

    double mass_before = local_mass(state.download());
    MPI_Allreduce(
        MPI_IN_PLACE, &mass_before, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    venus_gpu::MpiCollisionModule collisions(mesh, MPI_COMM_WORLD);
    venus_gpu::CollisionOptions collision_options;
    collision_options.sigma0 = 0.2;
    collision_options.target_shell_markers = 2048;
    collision_options.minimum_shell_markers = 16;
    collision_options.collision_cfl = 0.2;
    venus_gpu::CollisionSamplerState sampler;
    const venus_gpu::CollisionDiagnostics collision = collisions.apply(
        state, 0.02, collision_options, sampler);
    double mass_after = local_mass(state.download());
    MPI_Allreduce(
        MPI_IN_PLACE, &mass_after, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    assert(std::abs(mass_after - mass_before) < 5.0e-12);
    assert(collision.active_markers == global_count);
    assert(collision.accepted_pairs > 0);

    // Rank-count invariance for the current unequal-mass pair-majorant
    // scattering/capture operator.  Rank 0 first computes the one-GPU
    // reference; all ranks then execute the radially distributed operator
    // with identical global cell RNG keys and adaptive substeps.
    const std::size_t subhalo_count = 120000;
    std::vector<venus_gpu::Marker> subhalo_local;
    for (std::size_t global = static_cast<std::size_t>(rank);
         global < subhalo_count;
         global += static_cast<std::size_t>(ranks)) {
        subhalo_local.push_back(make_subhalo_marker(global, subhalo_count));
    }
    venus_gpu::SubhaloCollisionOptions subhalo_options;
    subhalo_options.strength = 3.0;
    subhalo_options.velocity_scale = 0.1;
    subhalo_options.plummer_area_per_mass = 1.0;
    subhalo_options.low_speed_regularization = 0.05;
    subhalo_options.collision_cfl = 0.2;
    subhalo_options.maximum_substeps = 512;
    subhalo_options.inner_collision_cells = 1;
    subhalo_options.collision_lanes = 4;
    venus_gpu::CollisionSamplerState subhalo_reference_sampler;
    subhalo_reference_sampler.seed = UINT64_C(0x73756268616c6f31);
    venus_gpu::SubhaloCollisionDiagnostics reference_diagnostic;
    std::vector<venus_gpu::Marker> reference_markers;
    if (rank == 0) {
        std::vector<venus_gpu::Marker> complete;
        complete.reserve(subhalo_count);
        for (std::size_t global = 0; global < subhalo_count; ++global) {
            complete.push_back(make_subhalo_marker(global, subhalo_count));
        }
        venus_gpu::WpicState reference_state(complete);
        venus_gpu::SubhaloCollisionModule reference_module(mesh);
        reference_diagnostic = reference_module.apply(
            reference_state, 0.2, subhalo_options, subhalo_reference_sampler);
        reference_markers = reference_state.download();
        std::sort(
            reference_markers.begin(),
            reference_markers.end(),
            [](const venus_gpu::Marker& first,
               const venus_gpu::Marker& second) {
                return first.id < second.id;
            });
    }
    MPI_Barrier(MPI_COMM_WORLD);
    venus_gpu::WpicState distributed_state(subhalo_local);
    venus_gpu::MpiSubhaloCollisionModule distributed_module(
        mesh, MPI_COMM_WORLD);
    venus_gpu::CollisionSamplerState distributed_sampler;
    distributed_sampler.seed = UINT64_C(0x73756268616c6f31);
    const venus_gpu::SubhaloCollisionDiagnostics distributed_diagnostic =
        distributed_module.apply(
            distributed_state, 0.2, subhalo_options, distributed_sampler);
    const std::vector<venus_gpu::Marker> gathered = gather_markers(
        distributed_state.download(), rank, ranks);
    if (rank == 0) {
        assert(reference_diagnostic.trial_pairs
            == distributed_diagnostic.trial_pairs);
        assert(reference_diagnostic.accepted_pairs
            == distributed_diagnostic.accepted_pairs);
        assert(reference_diagnostic.scattering_events
            == distributed_diagnostic.scattering_events);
        assert(reference_diagnostic.capture_events
            == distributed_diagnostic.capture_events);
        assert(reference_diagnostic.substeps
            == distributed_diagnostic.substeps);
        assert(std::abs(
            reference_diagnostic.minimum_pair_velocity_scale
            - distributed_diagnostic.minimum_pair_velocity_scale) < 1.0e-14);
        assert(std::abs(
            reference_diagnostic.maximum_pair_velocity_scale
            - distributed_diagnostic.maximum_pair_velocity_scale) < 1.0e-14);
        assert(distributed_diagnostic.minimum_pair_velocity_scale > 0.0);
        assert(distributed_diagnostic.maximum_pair_velocity_scale
            >= distributed_diagnostic.minimum_pair_velocity_scale);
        assert(subhalo_reference_sampler.next_counter
            == distributed_sampler.next_counter);
        assert(reference_markers.size() == gathered.size());
        for (std::size_t i = 0; i < gathered.size(); ++i) {
            const venus_gpu::Marker& first = reference_markers[i];
            const venus_gpu::Marker& second = gathered[i];
            assert(first.id == second.id);
            assert(first.source_cell == second.source_cell);
            assert(first.state == second.state);
            assert(std::abs(first.mass - second.mass) < 1.0e-18);
            assert(std::abs(first.r - second.r) < 1.0e-14);
            assert(std::abs(first.w - second.w) < 1.0e-14);
            assert(std::abs(first.eta - second.eta) < 1.0e-14);
            assert(std::abs(first.internal_energy - second.internal_energy)
                < 1.0e-18);
            assert(std::abs(
                first.internal_velocity_dispersion
                - second.internal_velocity_dispersion) < 1.0e-14);
        }
        assert(distributed_diagnostic.accepted_pairs > 0);
    }
    if (rank == 0) {
        std::cout << "MPI ranks=" << ranks
                  << " rank0_device=" << device
                  << " global_mass=" << mass_after
                  << " collision_trials=" << collision.trial_pairs
                  << " accepted=" << collision.accepted_pairs << '\n'
                  << "subhalo_trials="
                  << distributed_diagnostic.trial_pairs
                  << " subhalo_accepted="
                  << distributed_diagnostic.accepted_pairs
                  << " subhalo_captures="
                  << distributed_diagnostic.capture_events << '\n'
                  << "Multi-GPU WPIC MPI tests passed.\n";
    }
    MPI_Finalize();
    return 0;
}
