
// Inspect the sampled reversible Boltzmann activity plan for a configured
// distribution. Input is a normal solver configuration, with an optional
// restart override in VENUS_DIAGNOSTIC_RESTART_FROM. Output contains one
// row per radial shell with its inactive flag, initial sample count and
// activity bound, followed by aggregate locality and mean-free-path measures.
// This command builds a plan from the state; it advances no physical time.

#include "ev/config.hpp"
#include "ev/sampled_reversible_boltzmann.hpp"
#include "ev/solver_setup.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    // ---- Parse configuration and optional restart override ---------------
    if (argc != 2) {
        std::cerr
            << "usage: diagnose_sampled_boltzmann_shell_plan CONFIG\n";
        return 2;
    }
    try {
        ev::Params p;
        ev::load_config_file(argv[1], p);
        if (const char* restart =
                std::getenv("VENUS_DIAGNOSTIC_RESTART_FROM");
            restart != nullptr && restart[0] != '\0') {
            p.restart_from = restart;
        }

        // ---- Reconstruct the production phase-space distribution ----------
        ev::MpiContext mpi;
        ev::SolverState state = ev::create_solver_state(p, mpi);
        ev::initialize_distribution(state);
        const ev::PhaseSpaceCellGeometry geometry =
            ev::make_cell_average_eta_geometry(
                ev::make_positive_radial_cell_axis(state.global_p),
                state.global_p);
        // ---- Build the non-mutating shell activity and locality plan ------
        // Build the same shell activity estimates used by the sampled collision
        // backend. Inactive status is a numerical planning result with its own
        // weighted activity bound, rather than a removal of physical mass.
        const ev::BPSampledShellPlan plan =
            ev::bp_sampled_build_shell_plan(
                state.f,
                state.r,
                state.w,
                geometry.eta.center,
                geometry,
                state.local_p);

        // ---- Emit one auditable row per physical radial shell -------------
        std::cout
            << "shell,r,inactive,initial_samples,activity_upper_bound\n";
        std::cout << std::setprecision(17);
        int inactive_count = 0;
        for (int i = 0; i < state.global_p.Nr; ++i) {
            const bool inactive =
                plan.inactive[static_cast<std::size_t>(i)] != 0;
            inactive_count += inactive ? 1 : 0;
            std::cout
                << i << ','
                << state.r[static_cast<std::size_t>(i)] << ','
                << (inactive ? 1 : 0) << ','
                << plan.initial_samples[static_cast<std::size_t>(i)] << ','
                << plan.activity_upper_bound[static_cast<std::size_t>(i)]
                << '\n';
        }
        // The aggregate bounds expose omitted activity and the radial-cell
        // width relative to mean free path, which are needed to interpret a
        // homogeneous-shell collision approximation.
        std::cerr
            << "inactive_shells=" << inactive_count
            << " weighted_inactive_upper_bound="
            << plan.weighted_inactive_upper_bound
            << " weighted_activity_lower_bound="
            << plan.weighted_activity_lower_bound
            << " minimum_mean_free_path="
            << plan.minimum_mean_free_path
            << " maximum_Delta_r_over_lambda_rms="
            << plan.maximum_radial_cell_width_over_mean_free_path
            << " nonlocal_collision_cells="
            << plan.nonlocal_collision_cells << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
