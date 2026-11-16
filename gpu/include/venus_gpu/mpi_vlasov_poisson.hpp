#pragma once

// Evolve disjoint GPU particle sets in one spherical gravitational field.
// Each rank deposits its local mass and velocity moments onto the common
// radial mesh. Collective sums supply the global moments before each Poisson
// solve, giving every rank the same field for its local kicks and drift.
// Inputs are local particle states and shared physical options; outputs are
// updated local particles and the common field used for profiles and clocks.

#include "venus_gpu/mpi_common.hpp"
#include "venus_gpu/vlasov_poisson.hpp"

#include <memory>

namespace venus_gpu {

class MpiVlasovPoissonModule {
public:
    MpiVlasovPoissonModule(const RadialMesh& mesh, MPI_Comm communicator);
    ~MpiVlasovPoissonModule();
    MpiVlasovPoissonModule(MpiVlasovPoissonModule&&) noexcept;
    MpiVlasovPoissonModule& operator=(MpiVlasovPoissonModule&&) noexcept;
    MpiVlasovPoissonModule(const MpiVlasovPoissonModule&) = delete;
    MpiVlasovPoissonModule& operator=(const MpiVlasovPoissonModule&) = delete;

    // Every rank executes the same deposition/solve sequence around KDK;
    // the field solve requires moments from all particle owners.
    DepositionDiagnostics advance_kdk(
        WpicState& local_state, double time, double dt,
        const VlasovPoissonOptions& options);
    DepositionDiagnostics deposit_and_solve(
        const WpicState& local_state, double time,
        const VlasovPoissonOptions& options);
    // Use already reduced moments when the caller has performed the
    // collective operation explicitly.
    void solve_from_global_moments(
        const FieldHost& global_moments, double time,
        const VlasovPoissonOptions& options);
    void begin_kdk_with_current_field(
        WpicState& local_state, double time, double dt,
        const VlasovPoissonOptions& options);
    void finish_kdk_with_current_field(
        WpicState& local_state, double time_after_drift, double dt,
        const VlasovPoissonOptions& options);
    // The returned field is common to the communicator and can be written
    // once by the output rank without summing it again.
    FieldHost download_global_field() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace venus_gpu
