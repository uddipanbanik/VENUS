#pragma once

// Apply unequal-mass subhalo encounters collectively across GPU ranks.
// Inputs are disjoint local particle sets, a common radial mesh, elapsed time,
// collision controls, and sampler state. Radius ownership gathers each local
// encounter population onto one rank. apply() updates scattering/capture
// products and returns global event, energy, and locality budgets. Every rank
// participates, including ranks with an empty local encounter population.

#include "venus_gpu/mpi_common.hpp"
#include "venus_gpu/subhalo_collisions.hpp"

#include <memory>

namespace venus_gpu {

class MpiSubhaloCollisionModule {
public:
    MpiSubhaloCollisionModule(
        const RadialMesh& global_mesh, MPI_Comm communicator);
    ~MpiSubhaloCollisionModule();
    MpiSubhaloCollisionModule(MpiSubhaloCollisionModule&&) noexcept;
    MpiSubhaloCollisionModule& operator=(
        MpiSubhaloCollisionModule&&) noexcept;
    MpiSubhaloCollisionModule(const MpiSubhaloCollisionModule&) = delete;
    MpiSubhaloCollisionModule& operator=(
        const MpiSubhaloCollisionModule&) = delete;

    // Reference cadence uses a common maximum rate; adaptive cells cover
    // the supplied interval locally. IDs follow particles through migration.
    SubhaloCollisionDiagnostics apply(
        WpicState& state, double dt, const SubhaloCollisionOptions& options,
        CollisionSamplerState& sampler);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace venus_gpu
