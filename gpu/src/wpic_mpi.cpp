// Coordinate one spherical GPU calculation across MPI particle owners.
// Inputs are a communicator, local particle states, a common radial mesh,
// physical options, and elapsed time. VP sums extensive radial moments before
// each field solve, giving every rank the same gravity. Collision adapters
// migrate complete particle records to a unique radial owner, then reduce
// event counts, conserved-quantity budgets, and safety extrema globally.
// Outputs are updated local device populations and common field/diagnostics.
// Host staging supplies portable MPI exchange. Material collision streams
// depend on rank ownership; subhalo cells retain global cell keys and IDs.

#include "venus_gpu/mpi_material_sidm.hpp"
#include "venus_gpu/mpi_subhalo_collisions.hpp"
#include "venus_gpu/mpi_vlasov_poisson.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace venus_gpu {
namespace {

// Convert an MPI error into a host exception with the failed collective
// context. Every collective participant must follow the same call sequence.
void mpi_check(int status, const char* context) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(
        std::string(context) + ": " + std::string(message, length));
}

void cuda_check_mpi(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(status));
    }
}

// Sum extensive cell moments, which are additive over disjoint particle
// subsets. Ratios such as mean velocity or Neff are formed after this sum.
void allreduce_vector(std::vector<double>& values, MPI_Comm communicator) {
    if (values.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error("MPI radial moment array exceeds INT_MAX.");
    }
    if (!values.empty()) {
        mpi_check(
            MPI_Allreduce(
                MPI_IN_PLACE,
                values.data(),
                static_cast<int>(values.size()),
                MPI_DOUBLE,
                MPI_SUM,
                communicator),
            "MPI_Allreduce of WPIC radial moments failed");
    }
}

// Share local layout failures before an exchange so every rank rejects an
// invalid request at the same collective boundary.
void collective_require(
    bool local_condition,
    MPI_Comm communicator,
    const char* message) {
    int failed = local_condition ? 0 : 1;
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, communicator),
        "MPI collective precondition reduction failed");
    if (failed != 0) {
        throw std::runtime_error(message);
    }
}

// Combine the deposited source and all velocity moments, including squared
// particle weights and the resolved/hidden/escaped mass partition.
FieldHost globally_reduce_moments(
    FieldHost field, MPI_Comm communicator) {
    allreduce_vector(field.shell_mass, communicator);
    allreduce_vector(field.shell_radial_momentum, communicator);
    allreduce_vector(field.shell_radial_second_moment, communicator);
    allreduce_vector(field.shell_tangential_second_moment, communicator);
    allreduce_vector(field.shell_radial_energy_flux, communicator);
    allreduce_vector(field.shell_mass_square, communicator);
    double ledger[4] = {
        field.diagnostics.resolved_active_mass,
        field.diagnostics.deposited_mesh_mass,
        field.diagnostics.hidden_inner_mass,
        field.diagnostics.escaped_mass,
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE, ledger, 4, MPI_DOUBLE, MPI_SUM, communicator),
        "MPI_Allreduce of WPIC mass ledger failed");
    field.diagnostics.resolved_active_mass = ledger[0];
    field.diagnostics.deposited_mesh_mass = ledger[1];
    field.diagnostics.hidden_inner_mass = ledger[2];
    field.diagnostics.escaped_mass = ledger[3];
    field.diagnostics.mass_roundoff = ledger[1] - ledger[0];
    return field;
}

// Material sampling gives each rank a distinct deterministic seed derived
// from the portable seed. The common counter is synchronized afterward.
std::uint64_t mpi_mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

}  // namespace

// Obtain node-local rank through the shared-memory communicator and select
// local_rank modulo the visible device count before any device allocation.
int bind_cuda_device_to_mpi_local_rank(MPI_Comm communicator) {
    MPI_Comm local = MPI_COMM_NULL;
    mpi_check(
        MPI_Comm_split_type(
            communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local),
        "MPI shared-memory communicator creation failed");
    int local_rank = 0;
    mpi_check(MPI_Comm_rank(local, &local_rank), "MPI local rank failed");
    mpi_check(MPI_Comm_free(&local), "MPI local communicator free failed");
    int devices = 0;
    cuda_check_mpi(cudaGetDeviceCount(&devices), "CUDA device discovery failed");
    if (devices < 1) {
        throw std::runtime_error("No CUDA device is visible to this MPI rank.");
    }
    const int device = local_rank % devices;
    cuda_check_mpi(cudaSetDevice(device), "CUDA rank/device binding failed");
    return device;
}

// Duplicate the communicator for this module's lifetime and retain the
// local VP workspace. Release it only while MPI is still initialized.
struct MpiVlasovPoissonModule::Impl {
    Impl(const RadialMesh& mesh, MPI_Comm source)
        : module(mesh) {
        mpi_check(MPI_Comm_dup(source, &communicator), "MPI communicator duplication failed");
    }

    ~Impl() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized && communicator != MPI_COMM_NULL) {
            MPI_Comm_free(&communicator);
        }
    }

    MPI_Comm communicator = MPI_COMM_NULL;
    VlasovPoissonModule module;
    FieldHost global_field;
};

MpiVlasovPoissonModule::MpiVlasovPoissonModule(
    const RadialMesh& mesh, MPI_Comm communicator)
    : impl_(new Impl(mesh, communicator)) {}

MpiVlasovPoissonModule::~MpiVlasovPoissonModule() = default;
MpiVlasovPoissonModule::MpiVlasovPoissonModule(
    MpiVlasovPoissonModule&&) noexcept = default;
MpiVlasovPoissonModule& MpiVlasovPoissonModule::operator=(
    MpiVlasovPoissonModule&&) noexcept = default;

// A self-consistent KDK step needs two collective field constructions: one for
// the opening kick and one after drift for the closing kick.  Only O(Nr)
// extensive moments cross MPI; marker arrays remain on their owning devices.
DepositionDiagnostics MpiVlasovPoissonModule::advance_kdk(
    WpicState& local_state,
    double time,
    double dt,
    const VlasovPoissonOptions& options) {
    impl_->module.deposit_and_solve(local_state, time, options);
    impl_->global_field = globally_reduce_moments(
        impl_->module.download_field(), impl_->communicator);
    impl_->module.solve_from_global_moments(
        impl_->global_field, time, options);
    impl_->module.begin_kdk_with_current_field(
        local_state, time, dt, options);

    impl_->module.deposit_and_solve(local_state, time + dt, options);
    impl_->global_field = globally_reduce_moments(
        impl_->module.download_field(), impl_->communicator);
    impl_->module.solve_from_global_moments(
        impl_->global_field, time + dt, options);
    impl_->module.finish_kdk_with_current_field(
        local_state, time + dt, dt, options);
    return impl_->global_field.diagnostics;
}

// Observe local moments without transport, reduce the common source, and
// replace each device's field with the solution of that global source.
DepositionDiagnostics MpiVlasovPoissonModule::deposit_and_solve(
    const WpicState& local_state,
    double time,
    const VlasovPoissonOptions& options) {
    impl_->module.deposit_and_solve(local_state, time, options);
    impl_->global_field = globally_reduce_moments(
        impl_->module.download_field(), impl_->communicator);
    impl_->module.solve_from_global_moments(
        impl_->global_field, time, options);
    return impl_->global_field.diagnostics;
}

// This entry accepts a field-source moment set already made global by the
// caller. Each rank then performs only the common local Poisson solve.
void MpiVlasovPoissonModule::solve_from_global_moments(
    const FieldHost& global_moments,
    double time,
    const VlasovPoissonOptions& options) {
    impl_->global_field = global_moments;
    impl_->module.solve_from_global_moments(
        impl_->global_field, time, options);
}

// Split entry points let a caller place its collective source construction
// between the drift and endpoint kick while particles remain rank local.
void MpiVlasovPoissonModule::begin_kdk_with_current_field(
    WpicState& local_state,
    double time,
    double dt,
    const VlasovPoissonOptions& options) {
    impl_->module.begin_kdk_with_current_field(
        local_state, time, dt, options);
}

void MpiVlasovPoissonModule::finish_kdk_with_current_field(
    WpicState& local_state,
    double time_after_drift,
    double dt,
    const VlasovPoissonOptions& options) {
    impl_->module.finish_kdk_with_current_field(
        local_state, time_after_drift, dt, options);
}

FieldHost MpiVlasovPoissonModule::download_global_field() const {
    return impl_->module.download_field();
}

// Material DSMC owns contiguous radial-cell slabs.  Before every collision
// event an Alltoallv migration moves complete Marker records to the slab that
// owns their current radius; delayed-inner and escaped records stay on the end
// ranks.  Local meshes therefore contain whole collision shells and no pair is
// sampled on two ranks.
struct MpiCollisionModule::Impl {
    Impl(const RadialMesh& mesh, MPI_Comm source)
        : global_mesh(mesh) {
        static_assert(
            std::is_trivially_copyable<Marker>::value,
            "MPI marker migration requires a trivially copyable Marker.");
        mpi_check(MPI_Comm_dup(source, &communicator), "MPI communicator duplication failed");
        mpi_check(MPI_Comm_rank(communicator, &rank), "MPI rank failed");
        mpi_check(MPI_Comm_size(communicator, &ranks), "MPI size failed");
        if (ranks < 1 || static_cast<std::size_t>(ranks) > mesh.cells()) {
            throw std::runtime_error(
                "Multi-GPU collisions require at least one radial cell per rank.");
        }
        first_cell = static_cast<std::size_t>(rank) * mesh.cells() / ranks;
        last_cell = static_cast<std::size_t>(rank + 1) * mesh.cells() / ranks;
        std::vector<double> local_edges(
            mesh.edge.begin() + static_cast<std::ptrdiff_t>(first_cell),
            mesh.edge.begin() + static_cast<std::ptrdiff_t>(last_cell + 1));
        local_mesh = RadialMesh::from_edges(local_edges);
        collisions.reset(new CollisionModule(local_mesh));
    }

    ~Impl() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized && communicator != MPI_COMM_NULL) {
            MPI_Comm_free(&communicator);
        }
    }

    // Boundary carriers belong to the end ranks. An active radius maps to
    // the unique contiguous cell slab containing its current position.
    int owner(const Marker& marker) const {
        if (marker.state == MarkerState::delayed_inner) {
            return 0;
        }
        if (marker.state == MarkerState::escaped) {
            return ranks - 1;
        }
        const std::vector<double>::const_iterator upper = std::upper_bound(
            global_mesh.edge.begin(), global_mesh.edge.end(), marker.r);
        const std::size_t cell = static_cast<std::size_t>(std::max<std::ptrdiff_t>(
            0,
            std::min<std::ptrdiff_t>(
                static_cast<std::ptrdiff_t>(global_mesh.cells() - 1),
                upper - global_mesh.edge.begin() - 1)));
        return static_cast<int>(
            ((cell + 1) * static_cast<std::size_t>(ranks) - 1)
            / global_mesh.cells());
    }

    // Download complete records only for the communication boundary, group
    // by destination, exchange byte counts, then transfer the records once.
    void migrate(WpicState& state) {
        const std::vector<Marker> local = state.download();
        std::vector<std::vector<Marker>> bucket(static_cast<std::size_t>(ranks));
        for (std::size_t i = 0; i < local.size(); ++i) {
            bucket[static_cast<std::size_t>(owner(local[i]))].push_back(local[i]);
        }
        std::vector<int> send_count(static_cast<std::size_t>(ranks), 0);
        std::vector<int> receive_count(static_cast<std::size_t>(ranks), 0);
        std::vector<int> send_displacement(static_cast<std::size_t>(ranks), 0);
        std::vector<int> receive_displacement(static_cast<std::size_t>(ranks), 0);
        std::vector<Marker> send;
        // MPI_Alltoallv uses int byte counts/displacements. Validate both
        // each destination count and the complete packed send extent.
        std::uint64_t total_send_bytes = 0;
        bool send_layout_valid = true;
        for (int destination = 0; destination < ranks; ++destination) {
            const std::size_t markers =
                bucket[static_cast<std::size_t>(destination)].size();
            const bool count_valid = markers
                <= static_cast<std::size_t>(INT_MAX) / sizeof(Marker);
            const std::size_t bytes = count_valid
                ? markers * sizeof(Marker) : 0;
            send_layout_valid = send_layout_valid && count_valid
                && total_send_bytes
                    <= static_cast<std::uint64_t>(INT_MAX - bytes);
            send_count[static_cast<std::size_t>(destination)] =
                static_cast<int>(bytes);
            send_displacement[static_cast<std::size_t>(destination)] =
                total_send_bytes <= static_cast<std::uint64_t>(INT_MAX)
                ? static_cast<int>(total_send_bytes) : 0;
            total_send_bytes += bytes;
        }
        // All ranks decide this precondition together before entering the
        // count exchange, including ranks whose local particle set is empty.
        collective_require(
            send_layout_valid,
            communicator,
            "MPI collision send layout exceeds INT_MAX bytes.");
        send.reserve(local.size());
        for (int destination = 0; destination < ranks; ++destination) {
            send.insert(
                send.end(),
                bucket[static_cast<std::size_t>(destination)].begin(),
                bucket[static_cast<std::size_t>(destination)].end());
        }
        mpi_check(
            MPI_Alltoall(
                send_count.data(), 1, MPI_INT,
                receive_count.data(), 1, MPI_INT, communicator),
            "MPI collision marker-count exchange failed");
        // Build source displacements from the exchanged byte counts. The
        // received total must be representable and contain complete records.
        std::uint64_t received_bytes_u64 = 0;
        bool receive_layout_valid = true;
        for (int source = 0; source < ranks; ++source) {
            receive_displacement[static_cast<std::size_t>(source)] =
                received_bytes_u64 <= static_cast<std::uint64_t>(INT_MAX)
                ? static_cast<int>(received_bytes_u64) : 0;
            const int bytes = receive_count[static_cast<std::size_t>(source)];
            if (bytes < 0 || received_bytes_u64
                    > static_cast<std::uint64_t>(INT_MAX - bytes)) {
                receive_layout_valid = false;
            } else {
                received_bytes_u64 += static_cast<unsigned int>(bytes);
            }
        }
        receive_layout_valid = receive_layout_valid
            && received_bytes_u64 % sizeof(Marker) == 0;
        collective_require(
            receive_layout_valid,
            communicator,
            "MPI collision receive layout is invalid or exceeds INT_MAX.");
        // Allocate the exact receive population after the collective layout
        // check. This byte exchange assumes the same compiled Marker layout
        // across the communicator, as required by the trivial-copy contract.
        const int received_bytes = static_cast<int>(received_bytes_u64);
        std::vector<Marker> received(
            static_cast<std::size_t>(received_bytes) / sizeof(Marker));
        mpi_check(
            MPI_Alltoallv(
                send.empty() ? nullptr : static_cast<void*>(send.data()),
                send_count.data(), send_displacement.data(), MPI_BYTE,
                received.empty() ? nullptr : static_cast<void*>(received.data()),
                receive_count.data(), receive_displacement.data(), MPI_BYTE,
                communicator),
            "MPI radial marker migration failed");
        state.upload(received);
    }

    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int ranks = 1;
    RadialMesh global_mesh;
    RadialMesh local_mesh;
    std::size_t first_cell = 0;
    std::size_t last_cell = 0;
    std::unique_ptr<CollisionModule> collisions;
};

MpiCollisionModule::MpiCollisionModule(
    const RadialMesh& mesh, MPI_Comm communicator)
    : impl_(new Impl(mesh, communicator)) {}

MpiCollisionModule::~MpiCollisionModule() = default;
MpiCollisionModule::MpiCollisionModule(MpiCollisionModule&&) noexcept = default;
MpiCollisionModule& MpiCollisionModule::operator=(
    MpiCollisionModule&&) noexcept = default;

// Material collisions deliberately salt the counter stream by rank: global
// conservation and rate statistics are invariant, but stochastic trajectories
// need not be rank-count identical.  The portable counter advances by the
// maximum substep count so every rank begins the next collective event at the
// same epoch; additive errors and event counts are summed, safety extrema use
// max reductions.
CollisionDiagnostics MpiCollisionModule::apply(
    WpicState& state,
    double dt,
    const CollisionOptions& options,
    CollisionSamplerState& sampler) {
    // Material shells are rebuilt on their owning radial slabs. Rank-seeded
    // pair sampling preserves the global physical budgets while allowing
    // stochastic trajectories to vary when the rank decomposition changes.
    impl_->migrate(state);
    CollisionSamplerState local_sampler;
    local_sampler.seed = mpi_mix64(
        sampler.seed ^ static_cast<std::uint64_t>(impl_->rank));
    local_sampler.next_counter = sampler.next_counter;
    const CollisionDiagnostics local = impl_->collisions->apply(
        state, dt, options, local_sampler);
    CollisionDiagnostics global = local;
    unsigned long long sum_values[5] = {
        local.shell_visits,
        local.trial_pairs,
        local.accepted_pairs,
        static_cast<unsigned long long>(local.active_markers),
        static_cast<unsigned long long>(local.shells),
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE, sum_values, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
            impl_->communicator),
        "MPI collision diagnostic sum failed");
    global.shell_visits = sum_values[0];
    global.trial_pairs = sum_values[1];
    global.accepted_pairs = sum_values[2];
    global.active_markers = static_cast<std::size_t>(sum_values[3]);
    global.shells = static_cast<std::size_t>(sum_values[4]);
    // Each rank may need different local work. Advance the shared sampler
    // epoch by the maximum count so the next collective starts consistently.
    unsigned long long max_substeps = local.substeps;
    unsigned long long nonlocal = local.nonlocal_shells;
    mpi_check(MPI_Allreduce(
        MPI_IN_PLACE, &max_substeps, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
        impl_->communicator), "MPI collision substep reduction failed");
    mpi_check(MPI_Allreduce(
        MPI_IN_PLACE, &nonlocal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
        impl_->communicator), "MPI collision locality reduction failed");
    global.substeps = max_substeps;
    global.nonlocal_shells = static_cast<std::size_t>(nonlocal);
    // Extensive conservation residuals add over ranks; worst-pair and
    // locality errors are maxima, since averaging would hide a local failure.
    double sums[2] = {
        local.radial_momentum_error,
        local.kinetic_energy_error,
    };
    double maxima[3] = {
        local.maximum_shell_width_over_mean_free_path,
        local.max_pair_momentum_error,
        local.max_pair_kinetic_energy_error,
    };
    mpi_check(MPI_Allreduce(
        MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, impl_->communicator),
        "MPI collision invariant sum failed");
    mpi_check(MPI_Allreduce(
        MPI_IN_PLACE, maxima, 3, MPI_DOUBLE, MPI_MAX, impl_->communicator),
        "MPI collision invariant maximum failed");
    global.radial_momentum_error = sums[0];
    global.kinetic_energy_error = sums[1];
    global.maximum_shell_width_over_mean_free_path = maxima[0];
    global.max_pair_momentum_error = maxima[1];
    global.max_pair_kinetic_energy_error = maxima[2];
    sampler.next_counter += max_substeps;
    return global;
}

// Subhalo slabs use the same radial migration pattern, but restore
// marker-ID order and pass the global cell offset into the sampler.  Fixed
// reference mode uses the collective maximum-frequency callback for one
// mesh-wide clock.  Adaptive mode needs no per-local-step collective because
// complete, disjoint cells have unique owners inside the common outer batch.
struct MpiSubhaloCollisionModule::Impl {
    Impl(const RadialMesh& mesh, MPI_Comm source)
        : global_mesh(mesh) {
        static_assert(
            std::is_trivially_copyable<Marker>::value,
            "MPI marker migration requires a trivially copyable Marker.");
        mpi_check(
            MPI_Comm_dup(source, &communicator),
            "MPI subhalo communicator duplication failed");
        mpi_check(MPI_Comm_rank(communicator, &rank), "MPI rank failed");
        mpi_check(MPI_Comm_size(communicator, &ranks), "MPI size failed");
        if (ranks < 1 || static_cast<std::size_t>(ranks) > mesh.cells()) {
            throw std::runtime_error(
                "Multi-GPU subhalo collisions require one radial cell per rank.");
        }
        first_cell = static_cast<std::size_t>(rank) * mesh.cells() / ranks;
        last_cell = static_cast<std::size_t>(rank + 1) * mesh.cells() / ranks;
        std::vector<double> local_edges(
            mesh.edge.begin() + static_cast<std::ptrdiff_t>(first_cell),
            mesh.edge.begin() + static_cast<std::ptrdiff_t>(last_cell + 1));
        local_mesh = RadialMesh::from_edges(local_edges);
        collisions.reset(new SubhaloCollisionModule(local_mesh));
    }

    ~Impl() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized && communicator != MPI_COMM_NULL) {
            MPI_Comm_free(&communicator);
        }
    }

    // Boundary carriers belong to the end ranks. An active radius maps to
    // the unique contiguous cell slab containing its current position.
    int owner(const Marker& marker) const {
        if (marker.state == MarkerState::delayed_inner) {
            return 0;
        }
        if (marker.state == MarkerState::escaped) {
            return ranks - 1;
        }
        const std::vector<double>::const_iterator upper = std::upper_bound(
            global_mesh.edge.begin(), global_mesh.edge.end(), marker.r);
        const std::size_t cell = static_cast<std::size_t>(
            std::max<std::ptrdiff_t>(
                0,
                std::min<std::ptrdiff_t>(
                    static_cast<std::ptrdiff_t>(global_mesh.cells() - 1),
                    upper - global_mesh.edge.begin() - 1)));
        return static_cast<int>(
            ((cell + 1) * static_cast<std::size_t>(ranks) - 1)
            / global_mesh.cells());
    }

    // Download complete records only for the communication boundary, group
    // by destination, exchange byte counts, then transfer the records once.
    void migrate(WpicState& state) {
        const std::vector<Marker> local = state.download();
        std::vector<std::vector<Marker>> bucket(static_cast<std::size_t>(ranks));
        for (const Marker& marker : local) {
            bucket[static_cast<std::size_t>(owner(marker))].push_back(marker);
        }
        std::vector<int> send_count(static_cast<std::size_t>(ranks), 0);
        std::vector<int> receive_count(static_cast<std::size_t>(ranks), 0);
        std::vector<int> send_displacement(static_cast<std::size_t>(ranks), 0);
        std::vector<int> receive_displacement(static_cast<std::size_t>(ranks), 0);
        std::vector<Marker> send;
        // MPI_Alltoallv uses int byte counts/displacements. Validate both
        // each destination count and the complete packed send extent.
        std::uint64_t total_send_bytes = 0;
        bool send_layout_valid = true;
        for (int destination = 0; destination < ranks; ++destination) {
            const std::size_t markers =
                bucket[static_cast<std::size_t>(destination)].size();
            const bool count_valid = markers
                <= static_cast<std::size_t>(INT_MAX) / sizeof(Marker);
            const std::size_t bytes = count_valid
                ? markers * sizeof(Marker) : 0;
            send_layout_valid = send_layout_valid && count_valid
                && total_send_bytes
                    <= static_cast<std::uint64_t>(INT_MAX - bytes);
            send_count[static_cast<std::size_t>(destination)] =
                static_cast<int>(bytes);
            send_displacement[static_cast<std::size_t>(destination)] =
                total_send_bytes <= static_cast<std::uint64_t>(INT_MAX)
                ? static_cast<int>(total_send_bytes) : 0;
            total_send_bytes += bytes;
        }
        // All ranks decide this precondition together before entering the
        // count exchange, including ranks whose local particle set is empty.
        collective_require(
            send_layout_valid,
            communicator,
            "MPI subhalo send layout exceeds INT_MAX bytes.");
        send.reserve(local.size());
        for (int destination = 0; destination < ranks; ++destination) {
            send.insert(
                send.end(),
                bucket[static_cast<std::size_t>(destination)].begin(),
                bucket[static_cast<std::size_t>(destination)].end());
        }
        mpi_check(
            MPI_Alltoall(
                send_count.data(), 1, MPI_INT,
                receive_count.data(), 1, MPI_INT, communicator),
            "MPI subhalo marker-count exchange failed");
        // Build source displacements from the exchanged byte counts. The
        // received total must be representable and contain complete records.
        std::uint64_t received_bytes_u64 = 0;
        bool receive_layout_valid = true;
        for (int source = 0; source < ranks; ++source) {
            receive_displacement[static_cast<std::size_t>(source)] =
                received_bytes_u64 <= static_cast<std::uint64_t>(INT_MAX)
                ? static_cast<int>(received_bytes_u64) : 0;
            const int bytes = receive_count[static_cast<std::size_t>(source)];
            if (bytes < 0 || received_bytes_u64
                    > static_cast<std::uint64_t>(INT_MAX - bytes)) {
                receive_layout_valid = false;
            } else {
                received_bytes_u64 += static_cast<unsigned int>(bytes);
            }
        }
        receive_layout_valid = receive_layout_valid
            && received_bytes_u64 % sizeof(Marker) == 0;
        collective_require(
            receive_layout_valid,
            communicator,
            "MPI subhalo receive layout is invalid or exceeds INT_MAX.");
        // Allocate the exact receive population after the collective layout
        // check. This byte exchange assumes the same compiled Marker layout
        // across the communicator, as required by the trivial-copy contract.
        const int received_bytes = static_cast<int>(received_bytes_u64);
        std::vector<Marker> received(
            static_cast<std::size_t>(received_bytes) / sizeof(Marker));
        mpi_check(
            MPI_Alltoallv(
                send.empty() ? nullptr : static_cast<void*>(send.data()),
                send_count.data(), send_displacement.data(), MPI_BYTE,
                received.empty() ? nullptr
                                 : static_cast<void*>(received.data()),
                receive_count.data(), receive_displacement.data(), MPI_BYTE,
                communicator),
            "MPI radial subhalo marker migration failed");
        // The single-rank CPU/GPU checkpoint order is marker-id order.  MPI
        // receive concatenation is source-rank order, so restore the canonical
        // order before physical-cell pair diagnostics and device upload.
        std::sort(
            received.begin(),
            received.end(),
            [](const Marker& first, const Marker& second) {
                return first.id < second.id;
            });
        state.upload(received);
    }

    // Reference subhalo cadence uses the largest owner rate before choosing
    // a common step. The adaptive mode advances complete cells locally.
    static double reduce_maximum_frequency(double local, void* context) {
        Impl* self = static_cast<Impl*>(context);
        mpi_check(
            MPI_Allreduce(
                MPI_IN_PLACE,
                &local,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                self->communicator),
            "MPI subhalo collision-frequency reduction failed");
        return local;
    }

    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int ranks = 1;
    RadialMesh global_mesh;
    RadialMesh local_mesh;
    std::size_t first_cell = 0;
    std::size_t last_cell = 0;
    std::unique_ptr<SubhaloCollisionModule> collisions;
};

MpiSubhaloCollisionModule::MpiSubhaloCollisionModule(
    const RadialMesh& mesh,
    MPI_Comm communicator)
    : impl_(new Impl(mesh, communicator)) {}

MpiSubhaloCollisionModule::~MpiSubhaloCollisionModule() = default;
MpiSubhaloCollisionModule::MpiSubhaloCollisionModule(
    MpiSubhaloCollisionModule&&) noexcept = default;
MpiSubhaloCollisionModule& MpiSubhaloCollisionModule::operator=(
    MpiSubhaloCollisionModule&&) noexcept = default;

// apply() is collective at migration, the outer operator boundary, and the
// diagnostic reduction.  The callback synchronizes the fixed reference
// clock; the adaptive kernel advances uniquely owned cells independently.
// The returned record combines represented event masses, conservation
// ledgers, and locality/timescale extrema over the complete population.
SubhaloCollisionDiagnostics MpiSubhaloCollisionModule::apply(
    WpicState& state,
    double dt,
    const SubhaloCollisionOptions& options,
    CollisionSamplerState& sampler) {
    impl_->migrate(state);
    SubhaloCollisionOptions local_options = options;
    // Preserve each resolved cell's global index after radial decomposition
    // so the sampler and diagnostics refer to the same physical cell.
    local_options.resolved_cell_offset = impl_->first_cell;
    local_options.maximum_frequency_reducer =
        &Impl::reduce_maximum_frequency;
    local_options.maximum_frequency_context = impl_.get();
    CollisionSamplerState local_sampler = sampler;
    const SubhaloCollisionDiagnostics local = impl_->collisions->apply(
        state, dt, local_options, local_sampler);
    sampler = local_sampler;

    SubhaloCollisionDiagnostics global = local;
    // Counts and represented masses add over unique owners. Scheduling work
    // is a maximum over owners; locality and pair errors also use maxima.
    unsigned long long sums[12] = {
        local.cell_visits,
        local.trial_pairs,
        local.skipped_already_collided_trials,
        local.accepted_pairs,
        local.scattering_events,
        local.capture_events,
        local.inner_scattering_events,
        local.inner_capture_events,
        static_cast<unsigned long long>(local.active_markers_before),
        static_cast<unsigned long long>(local.active_markers_after),
        static_cast<unsigned long long>(local.removed_markers),
        static_cast<unsigned long long>(local.nonlocal_cells),
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE,
            sums,
            12,
            MPI_UNSIGNED_LONG_LONG,
            MPI_SUM,
            impl_->communicator),
        "MPI subhalo diagnostic sum failed");
    global.cell_visits = sums[0];
    global.trial_pairs = sums[1];
    global.skipped_already_collided_trials = sums[2];
    global.accepted_pairs = sums[3];
    global.scattering_events = sums[4];
    global.capture_events = sums[5];
    global.inner_scattering_events = sums[6];
    global.inner_capture_events = sums[7];
    global.active_markers_before = static_cast<std::size_t>(sums[8]);
    global.active_markers_after = static_cast<std::size_t>(sums[9]);
    global.removed_markers = static_cast<std::size_t>(sums[10]);
    global.nonlocal_cells = static_cast<std::size_t>(sums[11]);

    // Report the longest local scheduling path rather than adding parallel
    // kernel waves that occurred on different devices.
    unsigned long long scheduling_maxima[4] = {
        local.substeps,
        local.maximum_cell_substeps,
        local.kernel_batches,
        local.proposal_rebuilds,
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE,
            scheduling_maxima,
            4,
            MPI_UNSIGNED_LONG_LONG,
            MPI_MAX,
            impl_->communicator),
        "MPI subhalo scheduling-diagnostic reduction failed");
    global.substeps = scheduling_maxima[0];
    global.maximum_cell_substeps = scheduling_maxima[1];
    global.kernel_batches = scheduling_maxima[2];
    global.proposal_rebuilds = scheduling_maxima[3];
    double additive[4] = {
        local.represented_scattering_mass,
        local.represented_capture_mass,
        local.orbital_kinetic_energy_change,
        local.internal_energy_change,
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE,
            additive,
            4,
            MPI_DOUBLE,
            MPI_SUM,
            impl_->communicator),
        "MPI subhalo invariant sum failed");
    global.represented_scattering_mass = additive[0];
    global.represented_capture_mass = additive[1];
    global.orbital_kinetic_energy_change = additive[2];
    global.internal_energy_change = additive[3];
    double maxima[4] = {
        local.maximum_pair_momentum_error,
        local.maximum_pair_total_energy_error,
        local.maximum_cell_width_over_mean_free_path,
        local.maximum_pair_velocity_scale,
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE,
            maxima,
            4,
            MPI_DOUBLE,
            MPI_MAX,
            impl_->communicator),
        "MPI subhalo diagnostic maximum failed");
    global.maximum_pair_momentum_error = maxima[0];
    global.maximum_pair_total_energy_error = maxima[1];
    global.maximum_cell_width_over_mean_free_path = maxima[2];
    global.maximum_pair_velocity_scale = maxima[3];
    // A rank without a positive pair scale contributes +infinity to the
    // minimum, then an entirely absent scale is reported as the zero sentinel.
    const double local_pair_velocity_minimum =
        local.minimum_pair_velocity_scale > 0.0
        ? local.minimum_pair_velocity_scale
        : std::numeric_limits<double>::infinity();
    double minima[3] = {
        local.minimum_mean_free_path,
        local.minimum_collision_timescale,
        local_pair_velocity_minimum,
    };
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE,
            minima,
            3,
            MPI_DOUBLE,
            MPI_MIN,
            impl_->communicator),
        "MPI subhalo diagnostic minimum failed");
    global.minimum_mean_free_path = minima[0];
    global.minimum_collision_timescale = minima[1];
    global.minimum_pair_velocity_scale = std::isfinite(minima[2])
        ? minima[2]
        : 0.0;
    return global;
}

}  // namespace venus_gpu
