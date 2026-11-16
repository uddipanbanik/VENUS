#pragma once

// Connect the CPU solver physical modules in their execution order.
// Arguments select the model, parameters and initial or restart state. The
// setup module constructs that state, the evolution module advances its
// transport and collision operators, and output records physical diagnostics.
// The entry layer also owns MPI lifetime and propagates a failed physical
// or numerical guard as the process exit status.

#include "ev/config.hpp"
#include "ev/solver_evolution.hpp"
#include "ev/solver_task_profile.hpp"
#include "ev/spherical_dsmc_solver.hpp"

namespace ev {

/// Executes one solver run after MPI has been initialized and before MPI finalization.
void run_solver_body(const Params& params, const MpiContext& mpi) {
    SolverState state = create_solver_state(params, mpi);

    if (is_root(state.mpi)) {
        std::cout << "Solver model=" << state.global_p.physics_model
                  << " parallel backend=" << parallel_backend_name()
                  << " threads_per_rank=" << state.global_p.threads
                  << " mpi_size=" << state.mpi.size
                  << " global_NL=" << state.decomp.global_NL << "\n";
    }

    if (is_root(state.mpi)) {
        protect_run_directories(state);
    }
    mpi_barrier(state.mpi);
    // Persistent DSMC particles own their transport/collision loop. A fresh
    // run first samples the configured distribution; its restart path instead
    // restores the carried particles and stochastic state.
    if (using_bp_spherical_dsmc(state.global_p)) {
        if (state.global_p.restart_from.empty()) {
            initialize_distribution(state);
        }
        run_spherical_dsmc_solver(state);
        return;
    }
    // The Eulerian path initializes the distribution, writes the physical
    // initial state and only then opens the time-evolution history streams.
    initialize_distribution(state);
    write_initial_outputs(state);
    SolverOutputStreams streams = open_output_streams(state);
    run_evolution_loop(state, streams);
    write_final_outputs(state);
}

/// Compatibility overload used by the original generic modular executable.
void run_solver_body(int argc, char** argv, const MpiContext& mpi) {
    run_solver_body(parse_args(argc, argv), mpi);
}

/// Shared MPI/error-handling shell for generic and fail-closed task entry points.
// A named executable supplies its physical task profile. Its help and
// validation describe that task while the generic entry accepts the
// full configured model selection.
int run_solver_with_optional_task_profile(
    int argc,
    char** argv,
    const SolverTaskProfile* profile) {
    if (profile != nullptr) {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                print_solver_task_profile_help(argv[0], *profile);
                return 0;
            }
        }
    }
    MpiContext mpi;
#ifdef EV_USE_MPI
    MPI_Init(&argc, &argv);
    mpi.enabled = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
#endif

    int exit_code = 0;
    try {
        // Validate the requested physical task before constructing its state
        // or creating output, so the executable identity and actual model agree.
        Params params = parse_args(argc, argv);
        if (profile != nullptr) {
            validate_solver_task_profile(*profile, params);
        }
        run_solver_body(params, mpi);
    // Report the failing rank and terminate the MPI calculation consistently.
    // A rank cannot continue into collective field operations after another
    // rank has left the physical evolution loop.
    } catch (const std::exception& ex) {
        std::cerr << "rank " << mpi.rank << " error: " << ex.what() << "\n";
        exit_code = 1;
#ifdef EV_USE_MPI
        if (mpi.enabled) {
            MPI_Abort(MPI_COMM_WORLD, exit_code);
        }
#endif
    }
#ifdef EV_USE_MPI
    if (mpi.enabled) {
        MPI_Finalize();
    }
#endif
    return exit_code;
}

/// Initializes MPI when requested and runs the compatibility modular dispatch.
int run_solver(int argc, char** argv) {
    return run_solver_with_optional_task_profile(argc, argv, nullptr);
}

/// Runs one named physics task and rejects every incompatible configuration.
int run_solver_for_task(
    int argc,
    char** argv,
    SolverTaskProfile profile) {
    return run_solver_with_optional_task_profile(argc, argv, &profile);
}

}  // namespace ev
