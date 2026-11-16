#pragma once

// Own the mutable arrays and clocks for one CPU solver calculation.
// The state groups physical parameters, rank-local phase-space data, geometry,
// transport modes, collision workspaces and invariant diagnostics. Setup fills
// these fields; evolution updates them; output serializes the corresponding
// restart state. Streams are grouped separately because only the root MPI rank
// owns shared diagnostic files.

#include "ev/output.hpp"
#include "ev/phase_space_cells.hpp"
#include "ev/streamed_reversible_boltzmann.hpp"
#include "ev/timestep.hpp"

namespace ev {

/// Owns all mutable runtime state required by one C++ solver execution.
struct SolverState {
    // Global parameters describe the complete physical calculation. Local
    // parameters and the L/eta decomposition describe this rank's stored slab.
    Params global_p;
    Params local_p;
    MpiContext mpi;
    LDecomposition decomp;
    InitialDataRWL custom_ic;
    InitialDataRWL tracer_ic;
    bool binary_ic = false;
    bool has_tracer = false;
    bool has_cell_average_geometry = false;
    PhaseSpaceCellGeometry kinetic_geometry;

    // Physical grids and their integration weights accompany the distribution.
    // x_r is the radial interpolation coordinate, which may differ from r.
    std::vector<double> r;
    std::vector<double> x_r;
    std::vector<double> w;
    std::vector<double> L;
    std::vector<double> Lq;
    std::vector<double> rq;
    std::vector<double> wq;
    std::vector<double> f;
    std::vector<double> tracer_f;
    // High-order transport coefficients retain information beyond cell means.
    // Their validity flags identify whether the cached modes correspond to
    // the currently stored distribution.
    std::vector<std::array<double, 5>> hamiltonian_sldg_modes;
    std::vector<std::array<double, 5>> tracer_hamiltonian_sldg_modes;
    bool hamiltonian_sldg_modes_valid = false;
    bool tracer_hamiltonian_sldg_modes_valid = false;
    // Retain streamed projection/dense/operator graphs between collision
    // stages.  The workspace validates its geometry and operator hash before
    // each use, so adaptive velocity grids invalidate stale cache entries.
    BPStreamedCollisionWorkspace bp_streamed_collision_workspace;
    DelayedInnerCarrier inner_carrier;
    DelayedInnerCarrier tracer_inner_carrier;

    // These are numerical crossing scales, current coordinate time and
    // carried field-variation limits. They can influence the next interval
    // and belong to the solver temporal state.
    double radial_step = 0.0;
    double min_dw = 0.0;
    double t = 0.0;
    double last_dt = 0.0;
    double last_vp_potential_variation_timescale =
        std::numeric_limits<double>::infinity();
    double last_vp_potential_variation_dt_limit =
        std::numeric_limits<double>::infinity();
    double vp_sldg_transport_dt_cap =
        std::numeric_limits<double>::infinity();
    int step = 0;

    // Separate the latest-step boundary budget from its cumulative value
    // and from operator-increment diagnostics; each has a different physical
    // interpretation in the conservation history.
    EscapeStats last_escape;
    EscapeStats cumulative_escape;
    OperatorDiagnostics last_operator_diag;

    std::filesystem::path outdir;
    std::filesystem::path restart_dir;
    std::filesystem::path restart_tmp_dir;
    std::filesystem::path restart_path;
};

/// Groups CSV output streams opened only on the root MPI rank.
struct SolverOutputStreams {
    std::ofstream profiles;
    std::ofstream tracer_profiles;
    std::ofstream history;
    std::ofstream operator_mass_budget;
    std::ofstream w_grid_history;
    std::ofstream phase_snapshots;
    std::ofstream tracer_phase_snapshots;
};

}  // namespace ev
