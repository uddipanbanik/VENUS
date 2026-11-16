#pragma once

// Apply material dark matter collisions collectively across GPU ranks.
// Inputs are local particles, the common mesh, collision controls, elapsed
// time, and sampler state. Each radius-local collision shell has one owner;
// particles migrate to that owner for the single-GPU pair sampler. apply()
// returns globally reduced event, conservation, and locality diagnostics and
// advances the common sampler. All ranks call it in the same collective order.

#include "venus_gpu/material_sidm.hpp"
#include "venus_gpu/mpi_common.hpp"

#include <memory>

namespace venus_gpu {

class MpiCollisionModule {
public:
    MpiCollisionModule(const RadialMesh& global_mesh, MPI_Comm communicator);
    ~MpiCollisionModule();
    MpiCollisionModule(MpiCollisionModule&&) noexcept;
    MpiCollisionModule& operator=(MpiCollisionModule&&) noexcept;
    MpiCollisionModule(const MpiCollisionModule&) = delete;
    MpiCollisionModule& operator=(const MpiCollisionModule&) = delete;

    // Each owner advances its local shells over the supplied common dt.
    // Global reductions combine event budgets and synchronize sampler progress.
    CollisionDiagnostics apply(
        WpicState& state, double dt, const CollisionOptions& options,
        CollisionSamplerState& sampler);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace venus_gpu
