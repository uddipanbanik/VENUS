
// Measure a finite collision update in one radial shell of a configured
// Boltzmann--Poisson distribution. Inputs select the shell, sample budget,
// collision interval and optional projection neighborhood. Outputs report the
// sampled reaction graph, invariant/entropy diagnostics and separate graph
// construction/update times. The full radial transport problem is held fixed.

#include "ev/config.hpp"
#include "ev/solver_setup.hpp"
#include "ev/sampled_reversible_boltzmann.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    // ---- Parse the physical shell and sampling experiment ----------------
    if (argc != 5 && argc != 6) {
        std::cerr
            << "usage: diagnose_implicit_reaction_shell "
               "CONFIG SHELL SAMPLES COLLISION_DT [CANDIDATE_RADIUS]\n";
        return 2;
    }
    try {
        ev::Params p;
        ev::load_config_file(argv[1], p);
        const int shell_index = std::stoi(argv[2]);
        const int samples = std::stoi(argv[3]);
        const double collision_dt = std::stod(argv[4]);
        // Set both sampling bounds to the requested count so this diagnostic
        // compares a fixed finite sampling workload. It reports unconverged
        // estimates explicitly instead of aborting before their errors are printed.
        p.bp_sampled_min_samples_per_batch = samples;
        p.bp_sampled_max_samples_per_batch = samples;
        p.bp_sampled_activity_boost_max = 1.0;
        p.bp_sampled_fail_on_unconverged = false;
        if (const char* thinning =
                std::getenv("VENUS_DIAGNOSTIC_NTC_PAIR_THINNING");
            thinning != nullptr && thinning[0] != '\0') {
            const std::string value(thinning);
            p.bp_sampled_ntc_pair_thinning =
                value == "1" || value == "true" || value == "yes";
        }
        if (argc == 6) {
            p.bp_reversible_candidate_radius =
                std::max(1, std::stoi(argv[5]));
        }
        // ---- Restore the configured distribution and isolate one shell ---
        ev::MpiContext mpi;
        ev::SolverState state = ev::create_solver_state(p, mpi);
        ev::initialize_distribution(state);
        if (shell_index < 0 || shell_index >= p.Nr) {
            throw std::runtime_error("shell index lies outside the radial grid.");
        }
        const ev::PhaseSpaceCellGeometry geometry =
            ev::make_cell_average_eta_geometry(
                ev::make_positive_radial_cell_axis(p), p);
        std::vector<double> shell_f(
            static_cast<std::size_t>(p.Nw * p.NL), 0.0);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                shell_f[static_cast<std::size_t>(
                    ev::bp_reversible_velocity_index(j, k, p))] =
                    state.f[ev::idx(shell_index, j, k, state.local_p)];
            }
        }
        // ---- Sample the collision graph and advance positive reactions ----
        // NTC means "no time counter": candidate pairs are sampled from a
        // rate majorant rather than assigned a deterministic collision time.
        const auto graph_start = std::chrono::steady_clock::now();
        ev::BPSampledShellGraph sampled =
            ev::bp_sampled_build_shell_graph(
                shell_f,
                state.r[static_cast<std::size_t>(shell_index)],
                state.w,
                geometry.eta.center,
                geometry,
                state.local_p,
                samples,
                static_cast<std::uint64_t>(shell_index));
        const auto graph_end = std::chrono::steady_clock::now();
        // Apply the positive finite-time reaction update to the sampled graph
        // and the same incoming shell distribution; the elapsed interval here
        // is a collision suboperator interval, not a whole self-gravitating step.
        const ev::BPReversibleShellResult advanced =
            ev::bp_reversible_advance_shell(
                shell_f, sampled.graph, collision_dt, state.local_p);
        const auto step_end = std::chrono::steady_clock::now();
        double implicit_step_seconds = 0.0;
        ev::BPReversibleShellResult implicit_advanced;
        if (const char* compare =
                std::getenv("VENUS_DIAGNOSTIC_COMPARE_IMPLICIT");
            compare != nullptr && compare[0] != '\0'
                && std::string(compare) != "0"
                && std::string(compare) != "false") {
            // The optional comparison disables only the explicit fast path and
            // reuses the identical initial shell and graph. Its timing therefore
            // compares the two reaction-update paths at fixed sampled physics.
            ev::Params implicit_p = state.local_p;
            implicit_p.bp_reversible_explicit_fast_path = false;
            const auto implicit_start =
                std::chrono::steady_clock::now();
            implicit_advanced =
                ev::bp_reversible_advance_shell(
                    shell_f,
                    sampled.graph,
                    collision_dt,
                    implicit_p);
            implicit_step_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - implicit_start).count();
        }
        const double graph_seconds =
            std::chrono::duration<double>(graph_end - graph_start).count();
        const double step_seconds =
            std::chrono::duration<double>(step_end - graph_end).count();
        // ---- Report sampling error, invariants, and computational cost ----
        std::cout << std::setprecision(17)
                  << "shell=" << shell_index
                  << " r="
                  << state.r[static_cast<std::size_t>(shell_index)]
                  << " samples=" << sampled.samples_per_batch
                  << " rhs_error=" << sampled.rhs_relative_error
                  << " rhs_gross_error=" << sampled.rhs_gross_error
                  << " heat_error=" << sampled.heat_relative_error
                  << " heat_gross_error=" << sampled.heat_gross_error
                  << " energy_invariant_error="
                  << sampled.energy_invariant_error
                  << " reactions=" << sampled.graph.reactions.size()
                  << " represented_activity="
                  << sampled.graph.represented_activity
                  << " omitted_domain_activity="
                  << sampled.graph.omitted_domain_activity
                  << " omitted_projection_activity="
                  << sampled.graph.omitted_projection_activity
                  << " omitted_fraction="
                  << advanced.omitted_fraction
                  << " maximum_projection_error="
                  << sampled.graph.maximum_projection_error
                  << " graph_seconds=" << graph_seconds
                  << " step_seconds=" << step_seconds
                  << " implicit_step_seconds=" << implicit_step_seconds
                  << " reaction_updates=" << advanced.reaction_updates
                  << " implicit_updates="
                  << advanced.implicit_reaction_updates
                  << " implicit_iterations="
                  << advanced.implicit_iterations
                  << " implicit_fraction="
                  << static_cast<double>(advanced.implicit_reaction_updates)
                      / std::max<std::uint64_t>(
                          1, advanced.reaction_updates)
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "diagnostic error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
