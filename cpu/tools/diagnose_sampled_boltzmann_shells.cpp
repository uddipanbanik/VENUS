
// Estimate sampled reversible collision errors in selected radial shells of
// a configured distribution. Inputs set the dense subset, sample bounds and
// optional importance proposal and shell range. Output reports independent
// operator/heat estimates, conservation residuals, positive-step limits,
// omitted-domain activity and cost for every shell. The loaded source state
// is held fixed; these estimates measure local sampling error.

#include "ev/config.hpp"
#include "ev/solver_setup.hpp"
#include "ev/sampled_reversible_boltzmann.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace {

struct ShellDiagnostic {
    double radius = 0.0;
    double density = 0.0;
    double attempted_activity = 0.0;
    double omitted_domain_fraction = 0.0;
    double omitted_w_domain_fraction = 0.0;
    double omitted_eta_domain_fraction = 0.0;
    double omitted_both_domain_fraction = 0.0;
    double omitted_projection_fraction = 0.0;
    int samples = 0;
    int adaptive_rounds = 0;
    bool converged = false;
    double rhs_error = 0.0;
    double rhs_gross_error = 0.0;
    double heat_error = 0.0;
    double heat_gross_error = 0.0;
    double energy_invariant_error = 0.0;
    double positive_dt_limit = 0.0;
    std::uint64_t dense_reactions = 0;
    std::uint64_t complement_reactions = 0;
    std::uint64_t complement_draws = 0;
    std::uint64_t retained_collision_draws = 0;
    bool ntc_fallback = false;
    double wall_seconds = 0.0;
};

void write_shell_diagnostic(
    int shell,
    const ShellDiagnostic& out) {
    std::cout
        << shell << ',' << out.radius << ',' << out.density << ','
        << out.attempted_activity << ','
        << out.omitted_domain_fraction << ','
        << out.omitted_w_domain_fraction << ','
        << out.omitted_eta_domain_fraction << ','
        << out.omitted_both_domain_fraction << ','
        << out.omitted_projection_fraction << ','
        << out.samples << ',' << out.adaptive_rounds << ','
        << (out.converged ? 1 : 0) << ','
        << out.rhs_error << ',' << out.rhs_gross_error << ','
        << out.heat_error << ','
        << out.heat_gross_error << ','
        << out.energy_invariant_error << ','
        << out.positive_dt_limit << ','
        << out.dense_reactions << ','
        << out.complement_reactions << ','
        << out.complement_draws << ','
        << out.retained_collision_draws << ','
        << (out.ntc_fallback ? 1 : 0) << ','
        << out.wall_seconds << '\n';
    std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
    // ---- Parse sampling, proposal, and shell-range controls --------------
    if (argc != 5 && argc != 7 && argc != 11 && argc != 13) {
        std::cerr
            << "usage: diagnose_sampled_boltzmann_shells "
               "CONFIG DENSE_CELLS MIN_SAMPLES MAX_SAMPLES "
               "[TAIL_FRACTION STRATIFIED_FRACTION UNIFORM_FRACTION "
               "TAIL_POWER ENERGY_STRATA L_STRATA] "
               "[FIRST_SHELL LAST_SHELL_EXCLUSIVE]\n";
        return 2;
    }
    try {
        ev::Params p;
        ev::load_config_file(argv[1], p);
        if (const char* restart = std::getenv("VENUS_DIAGNOSTIC_RESTART_FROM");
            restart != nullptr && restart[0] != '\0') {
            p.restart_from = restart;
        }
        if (const char* tolerance =
                std::getenv("VENUS_DIAGNOSTIC_RHS_TOLERANCE");
            tolerance != nullptr && tolerance[0] != '\0') {
            p.bp_sampled_rhs_tolerance = std::stod(tolerance);
        }
        if (const char* fraction =
                std::getenv("VENUS_DIAGNOSTIC_RHS_FLOOR_FRACTION");
            fraction != nullptr && fraction[0] != '\0') {
            p.bp_sampled_rhs_floor_fraction = std::stod(fraction);
        }
        if (const char* tolerance =
                std::getenv("VENUS_DIAGNOSTIC_HEAT_TOLERANCE");
            tolerance != nullptr && tolerance[0] != '\0') {
            p.bp_sampled_heat_tolerance = std::stod(tolerance);
        }
        if (const char* samples =
                std::getenv("VENUS_DIAGNOSTIC_OUTGOING_ANGLE_SAMPLES");
            samples != nullptr && samples[0] != '\0') {
            p.bp_sampled_outgoing_angle_samples =
                std::max(1, std::stoi(samples));
        }
        if (const char* seed = std::getenv("VENUS_DIAGNOSTIC_SEED");
            seed != nullptr && seed[0] != '\0') {
            p.bp_sampled_seed =
                static_cast<std::uint64_t>(std::stoull(seed));
        }
        if (const char* threads = std::getenv("VENUS_DIAGNOSTIC_THREADS");
            threads != nullptr && threads[0] != '\0') {
            p.threads = std::max(1, std::stoi(threads));
        }
        if (const char* replicates =
                std::getenv("VENUS_DIAGNOSTIC_REPLICATES_PER_ESTIMATE");
            replicates != nullptr && replicates[0] != '\0') {
            p.bp_sampled_replicates_per_estimate =
                std::max(1, std::stoi(replicates));
        }
        if (const char* radius =
                std::getenv("VENUS_DIAGNOSTIC_CANDIDATE_RADIUS");
            radius != nullptr && radius[0] != '\0') {
            p.bp_reversible_candidate_radius =
                std::max(1, std::stoi(radius));
        }
        if (const char* quantile =
                std::getenv("VENUS_DIAGNOSTIC_NTC_QUANTILE");
            quantile != nullptr && quantile[0] != '\0') {
            p.bp_sampled_ntc_quantile = std::stod(quantile);
        }
        if (const char* thinning =
                std::getenv("VENUS_DIAGNOSTIC_NTC_PAIR_THINNING");
            thinning != nullptr && thinning[0] != '\0') {
            const std::string value(thinning);
            p.bp_sampled_ntc_pair_thinning =
                value == "1" || value == "true" || value == "yes";
        }
        p.bp_sampled_dense_cells = std::max(0, std::stoi(argv[2]));
        p.bp_sampled_min_samples_per_batch =
            std::max(1, std::stoi(argv[3]));
        p.bp_sampled_max_samples_per_batch =
            std::max(p.bp_sampled_min_samples_per_batch, std::stoi(argv[4]));
        const bool has_proposal_override = argc == 11 || argc == 13;
        if (has_proposal_override) {
            p.bp_sampled_tail_fraction = std::stod(argv[5]);
            p.bp_sampled_stratified_fraction = std::stod(argv[6]);
            p.bp_sampled_uniform_fraction = std::stod(argv[7]);
            p.bp_sampled_tail_power = std::stod(argv[8]);
            p.bp_sampled_energy_strata = std::max(1, std::stoi(argv[9]));
            p.bp_sampled_L_strata = std::max(1, std::stoi(argv[10]));
            const double nonbulk_fraction =
                p.bp_sampled_tail_fraction
                + p.bp_sampled_stratified_fraction
                + p.bp_sampled_uniform_fraction;
            if (p.bp_sampled_tail_fraction < 0.0
                || p.bp_sampled_stratified_fraction < 0.0
                || !(p.bp_sampled_uniform_fraction > 0.0)
                || !(nonbulk_fraction < 1.0)
                || p.bp_sampled_tail_power < 0.0) {
                throw std::runtime_error(
                    "invalid sampled Boltzmann proposal override.");
            }
        }
        // Retain unconverged diagnostics so their actual error and sample count
        // can be inspected. This reporting choice is local to this tool and
        // does not declare an unconverged production update acceptable.
        p.bp_sampled_fail_on_unconverged = false;

        // ---- Restore the same distribution used by the production solver -
        ev::MpiContext mpi;
        ev::SolverState state = ev::create_solver_state(p, mpi);
        ev::initialize_distribution(state);
        int first_shell = 0;
        int last_shell = state.global_p.Nr;
        if (argc == 7) {
            first_shell = std::stoi(argv[5]);
            last_shell = std::stoi(argv[6]);
        } else if (argc == 13) {
            first_shell = std::stoi(argv[11]);
            last_shell = std::stoi(argv[12]);
        }
        if (first_shell < 0
            || last_shell <= first_shell
            || last_shell > state.global_p.Nr) {
            throw std::runtime_error(
                "sampled Boltzmann diagnostic shell range is invalid.");
        }

        // ---- Prepare physical velocity-cell measures and sample budgets ---
        const ev::PhaseSpaceCellGeometry geometry =
            ev::make_cell_average_eta_geometry(
                ev::make_positive_radial_cell_axis(state.global_p),
                state.global_p);
        const std::vector<double> eta = geometry.eta.center;
        const std::vector<int> initial_samples =
            ev::bp_sampled_shell_initial_samples(
                state.f,
                state.r,
                state.w,
                eta,
                geometry,
                state.local_p);
        std::vector<ShellDiagnostic> diagnostic(
            static_cast<std::size_t>(state.global_p.Nr));
        std::mutex output_mutex;

        std::cout
            << "shell,r,density,attempted_activity,omitted_domain_fraction,"
               "omitted_w_domain_fraction,omitted_eta_domain_fraction,"
               "omitted_both_domain_fraction,"
               "omitted_projection_fraction,samples_per_batch,adaptive_rounds,"
               "converged,rhs_error,rhs_gross_error,heat_error,heat_gross_error,"
               "energy_invariant_error,positive_dt_limit,dense_reactions,"
               "complement_reactions,complement_draws,"
               "retained_collision_draws,ntc_fallback,wall_seconds\n";
        std::cout << std::setprecision(17);
        std::cout.flush();

        // ---- Estimate each radial shell independently --------------------
        // Dynamic chunks balance dense central shells against cheap outer
        // shells; the mutex protects only ordered diagnostic output.
        ev::parallel_for_dynamic_chunks(
            first_shell,
            last_shell,
            ev::effective_threads(state.global_p, last_shell - first_shell),
            1,
            [&](int first_shell, int last_shell, int) {
                for (int i = first_shell; i < last_shell; ++i) {
                    const auto wall_start =
                        std::chrono::steady_clock::now();
                    std::vector<double> shell_f(
                        static_cast<std::size_t>(
                            state.global_p.Nw * state.global_p.NL),
                        0.0);
                    for (int j = 0; j < state.global_p.Nw; ++j) {
                        for (int k = 0; k < state.global_p.NL; ++k) {
                            shell_f[static_cast<std::size_t>(
                                ev::bp_reversible_velocity_index(
                                    j, k, state.global_p))] =
                                state.f[ev::idx(i, j, k, state.local_p)];
                        }
                    }
                    ev::BPSampledShellGraph sampled =
                        ev::bp_sampled_build_shell_graph(
                            shell_f,
                            state.r[static_cast<std::size_t>(i)],
                            state.w,
                            eta,
                            geometry,
                            state.local_p,
                            initial_samples[static_cast<std::size_t>(i)],
                            static_cast<std::uint64_t>(i));
                    ShellDiagnostic& out =
                        diagnostic[static_cast<std::size_t>(i)];
                    out.radius = state.r[static_cast<std::size_t>(i)];
                    for (std::size_t q = 0; q < shell_f.size(); ++q) {
                        out.density +=
                            sampled.graph.velocity_cell_volume[q] * shell_f[q];
                    }
                    // Attempted activity includes represented reactions plus channels
                    // omitted by the finite velocity domain or projection. Use that same
                    // denominator to compare distinct omission mechanisms.
                    out.attempted_activity =
                        sampled.graph.represented_activity
                        + sampled.graph.omitted_domain_activity
                        + sampled.graph.omitted_projection_activity;
                    if (out.attempted_activity > 0.0) {
                        out.omitted_domain_fraction =
                            sampled.graph.omitted_domain_activity
                            / out.attempted_activity;
                        double omitted_w = 0.0;
                        double omitted_eta = 0.0;
                        double omitted_both = 0.0;
                        // Separate outgoing support outside w, outside eta, or outside both.
                        // This identifies which physical velocity extent needs resolution
                        // without confusing finite-domain loss with sampling variance.
                        for (const ev::BPReversibleOmittedChannel& channel
                             : sampled.graph.omitted_domain_channels) {
                            const double activity =
                                ev::bp_reversible_channel_activity(
                                    shell_f,
                                    channel.incoming_first,
                                    channel.incoming_second,
                                    channel.coefficient);
                            if (channel.domain_mask
                                == ev::BP_REVERSIBLE_DOMAIN_RADIAL_VELOCITY) {
                                omitted_w += activity;
                            } else if (channel.domain_mask
                                       == ev::BP_REVERSIBLE_DOMAIN_ANGULAR_MOMENTUM) {
                                omitted_eta += activity;
                            } else {
                                omitted_both += activity;
                            }
                        }
                        out.omitted_w_domain_fraction =
                            omitted_w / out.attempted_activity;
                        out.omitted_eta_domain_fraction =
                            omitted_eta / out.attempted_activity;
                        out.omitted_both_domain_fraction =
                            omitted_both / out.attempted_activity;
                        out.omitted_projection_fraction =
                            sampled.graph.omitted_projection_activity
                            / out.attempted_activity;
                    }
                    out.samples = sampled.samples_per_batch;
                    out.adaptive_rounds = sampled.adaptive_rounds;
                    out.converged = sampled.converged;
                    out.rhs_error = sampled.rhs_relative_error;
                    out.rhs_gross_error = sampled.rhs_gross_error;
                    out.heat_error = sampled.heat_relative_error;
                    out.heat_gross_error = sampled.heat_gross_error;
                    out.energy_invariant_error =
                        sampled.energy_invariant_error;
                    // Evaluate the sampled instantaneous gain-minus-loss operator on
                    // the unchanged shell. Its negative components supply a positive
                    // explicit interval bound, distinct from a field/transport timestep.
                    const std::vector<double> rhs =
                        ev::bp_reversible_shell_rhs(
                            shell_f, sampled.graph, state.local_p);
                    out.positive_dt_limit =
                        ev::bp_reversible_positive_dt_limit(
                            shell_f,
                            rhs,
                            state.local_p.bp_collision_cfl);
                    out.dense_reactions = sampled.dense_reactions;
                    out.complement_reactions =
                        sampled.complement_reactions;
                    out.complement_draws = sampled.complement_draws;
                    out.retained_collision_draws =
                        sampled.retained_collision_draws;
                    out.ntc_fallback = sampled.ntc_fallback;
                    out.wall_seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - wall_start)
                            .count();
                    {
                        // Workers estimate independent shells concurrently. The mutex keeps
                        // each complete CSV row intact; consumers identify rows by shell index
                        // rather than inferring radial order from arrival time.
                        std::lock_guard<std::mutex> lock(output_mutex);
                        write_shell_diagnostic(i, out);
                    }
                }
            });
    } catch (const std::exception& error) {
        std::cerr << "diagnostic error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
