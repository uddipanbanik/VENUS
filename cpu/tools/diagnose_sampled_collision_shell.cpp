
// Inspect a sampled reversible collision operator in one radial shell.
// CONFIG and SHELL_INDEX identify the input distribution; key=value options
// change only this diagnostic experiment. Proposal-only mode reports sampling
// support. Streamed mode estimates the operator or applies one finite shell
// collision interval; stored-graph mode provides a comparison using retained
// reactions. Every branch reports convergence and its own work/error measures.

#include "ev/solver.hpp"

#include <chrono>

int main(int argc, char** argv) {
    // ---- Parse the shell experiment and non-persistent overrides ----------
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " CONFIG SHELL_INDEX [key=value ...]\n";
        return 2;
    }
    try {
        ev::Params p;
        ev::load_config_file(argv[1], p);
        p.bp_sampled_fail_on_unconverged = false;
        bool proposal_only = false;
        bool streamed = false;
        double advance_dt = 0.0;
        for (int arg = 3; arg < argc; ++arg) {
            const std::string assignment = argv[arg];
            const std::size_t equal = assignment.find('=');
            if (equal == std::string::npos) {
                throw std::runtime_error(
                    "Diagnostic overrides must use key=value syntax.");
            }
            const std::string key = assignment.substr(0, equal);
            const std::string value = assignment.substr(equal + 1);
            if (key == "proposal_only") {
                proposal_only = ev::parse_bool_value(value, key);
            } else if (key == "streamed") {
                streamed = ev::parse_bool_value(value, key);
            } else if (key == "advance_dt") {
                advance_dt = std::stod(value);
            } else {
                ev::apply_param(
                    p, key, value, "diagnostic command line");
            }
        }

        // ---- Restore the production distribution and isolate one shell ---
        ev::MpiContext mpi;
        ev::SolverState state = ev::create_solver_state(p, mpi);
        ev::initialize_distribution(state);
        const int shell_index = std::stoi(argv[2]);
        if (shell_index < 0 || shell_index >= state.local_p.Nr) {
            throw std::runtime_error("Shell index is outside the radial grid.");
        }
        const ev::PhaseSpaceCellGeometry geometry =
            ev::make_cell_average_eta_geometry(
                ev::make_positive_radial_cell_axis(state.local_p),
                state.local_p);
        const ev::BPSampledShellPlan shell_plan =
            ev::bp_sampled_build_shell_plan(
                state.f,
                state.r,
                state.w,
                geometry.eta.center,
                geometry,
                state.local_p);
        std::vector<double> shell_f(
            static_cast<std::size_t>(state.local_p.Nw * state.local_p.NL),
            0.0);
        for (int j = 0; j < state.local_p.Nw; ++j) {
            for (int k = 0; k < state.local_p.NL; ++k) {
                shell_f[static_cast<std::size_t>(
                    ev::bp_reversible_velocity_index(
                        j, k, state.local_p))] =
                    state.f[ev::idx(shell_index, j, k, state.local_p)];
            }
        }
        const double radius =
            state.r[static_cast<std::size_t>(shell_index)];
        // ---- Build the importance proposal and physical locality estimate -
        // Importance sampling changes which candidate channels are inspected;
        // inverse probability weights leave the Boltzmann rate unchanged.
        const ev::BPSampledProposal proposal =
            ev::bp_sampled_build_proposal(
                shell_f,
                radius,
                state.w,
                geometry.eta.center,
                geometry,
                state.local_p);
        const std::vector<int> dense_cells =
            ev::bp_sampled_dense_velocity_cells(
                proposal,
                radius,
                state.w,
                geometry.eta.center,
                state.local_p);
        const double complement_probability =
            ev::bp_sampled_complement_probability(
                proposal, dense_cells);
        double bulk_probability_squared = 0.0;
        for (double probability : proposal.bulk_probability) {
            bulk_probability_squared += probability * probability;
        }
        // The inverse sum of squared proposal probabilities is the effective
        // number of velocity cells sampled by the bulk proposal. It describes
        // sampling concentration rather than represented physical particle count.
        const double effective_bulk_cells = bulk_probability_squared > 0.0
            ? 1.0 / bulk_probability_squared
            : 0.0;
        std::cout << std::setprecision(17)
                  << "shell_index=" << shell_index << '\n'
                  << "radius=" << radius << '\n'
                  << "effective_bulk_velocity_cells="
                  << effective_bulk_cells << '\n'
                  << "selected_dense_velocity_cells="
                  << dense_cells.size() << '\n'
                  << "complement_probability="
                  << complement_probability << '\n'
                  << "inactive="
                  << static_cast<int>(shell_plan.inactive[
                         static_cast<std::size_t>(shell_index)]) << '\n'
                  << "initial_samples="
                  << shell_plan.initial_samples[
                         static_cast<std::size_t>(shell_index)] << '\n'
                  << "activity_upper_bound="
                  << shell_plan.activity_upper_bound[
                         static_cast<std::size_t>(shell_index)] << '\n'
                  << "mean_free_path="
                  << shell_plan.mean_free_path[
                         static_cast<std::size_t>(shell_index)] << '\n'
                  << "cell_width_over_mean_free_path="
                  << geometry.radial.width[
                         static_cast<std::size_t>(shell_index)]
                         / shell_plan.mean_free_path[
                               static_cast<std::size_t>(shell_index)]
                  << '\n';
        // Return after the support and locality inventory. No operator estimate
        // or finite collision interval is applied in this branch.
        if (proposal_only) {
            return 0;
        }
        if (streamed) {
            // ---- Streamed operator estimate or finite-time shell advance ---
            ev::BPStreamedProjectionCache cache;
            // A positive requested interval selects a finite collision-only shell
            // update. The returned distribution is compared with the unchanged
            // input using the same physical velocity-cell measure.
            if (advance_dt > 0.0) {
                const auto started = std::chrono::steady_clock::now();
                const ev::BPStreamedShellResult advanced =
                    ev::bp_streamed_advance_shell(
                        shell_f,
                        radius,
                        state.w,
                        geometry.eta.center,
                        geometry,
                        advance_dt,
                        state.local_p,
                        static_cast<std::uint64_t>(shell_index),
                        cache,
                        nullptr,
                        nullptr,
                        shell_plan.initial_samples[
                            static_cast<std::size_t>(shell_index)]);
                const double wall_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                double mass_before = 0.0;
                double mass_after = 0.0;
                double energy_before = 0.0;
                double energy_after = 0.0;
                double distribution_mass_l1 = 0.0;
                double radial_pressure_before = 0.0;
                double radial_pressure_after = 0.0;
                for (int q = 0;
                     q < state.local_p.Nw * state.local_p.NL;
                     ++q) {
                    const std::array<int, 2> indices =
                        ev::bp_reversible_velocity_indices(q, state.local_p);
                    // At fixed r, the canonical velocity measure is 2pi dw deta/r^2.
                    // Use it consistently for represented density, kinetic energy and the
                    // L1 distribution change of the returned shell.
                    const double volume = 2.0 * ev::PI
                        * geometry.radial_velocity.width[
                            static_cast<std::size_t>(indices[0])]
                        * geometry.eta.width[
                            static_cast<std::size_t>(indices[1])]
                        / (radius * radius);
                    const double energy = ev::bp_reversible_energy(
                        q,
                        radius,
                        state.w,
                        geometry.eta.center,
                        state.local_p);
                    mass_before += volume
                        * shell_f[static_cast<std::size_t>(q)];
                    mass_after += volume
                        * advanced.step.f[static_cast<std::size_t>(q)];
                    energy_before += volume * energy
                        * shell_f[static_cast<std::size_t>(q)];
                    energy_after += volume * energy
                        * advanced.step.f[static_cast<std::size_t>(q)];
                    distribution_mass_l1 += volume * std::abs(
                        advanced.step.f[static_cast<std::size_t>(q)]
                        - shell_f[static_cast<std::size_t>(q)]);
                    const double radial_velocity = state.w[
                        static_cast<std::size_t>(indices[0])];
                    radial_pressure_before += volume
                        * radial_velocity * radial_velocity
                        * shell_f[static_cast<std::size_t>(q)];
                    radial_pressure_after += volume
                        * radial_velocity * radial_velocity
                        * advanced.step.f[static_cast<std::size_t>(q)];
                }
                std::cout
                    << "streamed_advance=1\n"
                    << "advance_dt=" << advance_dt << '\n'
                    << "substeps=" << advanced.step.substeps << '\n'
                    << "samples_per_batch="
                    << advanced.samples_per_batch << '\n'
                    << "converged=" << (advanced.converged ? 1 : 0) << '\n'
                    << "mass_relative_error="
                    << ev::kinetic_relative_error(
                        mass_before, mass_after) << '\n'
                    << "energy_relative_error="
                    << ev::kinetic_relative_error(
                        energy_before, energy_after) << '\n'
                    << "distribution_mass_l1="
                    << distribution_mass_l1 << '\n'
                    << "radial_pressure_change="
                    << radial_pressure_after - radial_pressure_before << '\n'
                    << "projection_cache_hits="
                    << advanced.cache_hits << '\n'
                    << "projection_cache_misses="
                    << advanced.cache_misses << '\n'
                    << "attempted_draws="
                    << advanced.attempted_draws << '\n'
                    << "represented_channels="
                    << advanced.represented_channels << '\n'
                    << "wall_seconds=" << wall_seconds << '\n';
                return advanced.converged ? 0 : 1;
            }
            const ev::BPStreamedShellContext context =
                ev::bp_streamed_make_shell_context(
                    shell_f,
                    radius,
                    state.w,
                    geometry.eta.center,
                    geometry,
                    state.local_p);
            const auto started = std::chrono::steady_clock::now();
            const ev::BPStreamedOperatorEstimate estimate =
                ev::bp_streamed_estimate_shell_operator(
                    shell_f,
                    context,
                    state.w,
                    geometry.eta.center,
                    geometry,
                    state.local_p,
                    ev::bp_sampled_splitmix64(
                        state.local_p.bp_sampled_seed
                        ^ static_cast<std::uint64_t>(shell_index)),
                    ev::bp_sampled_splitmix64(
                        state.local_p.bp_sampled_seed
                        ^ static_cast<std::uint64_t>(shell_index)
                        ^ UINT64_C(0x9e3779b97f4a7c15)),
                    cache,
                    shell_plan.initial_samples[
                        static_cast<std::size_t>(shell_index)],
                    true);
            const double wall_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            double maximum_rhs = 0.0;
            for (double value : estimate.rhs) {
                maximum_rhs = std::max(maximum_rhs, std::abs(value));
            }
            // Normalize the two independent streamed batches with the same sample
            // count and add their deterministic dense part. Their differences
            // locate sampling fluctuations in the same physical operator.
            const std::vector<double> first_rhs =
                ev::bp_streamed_normalized_values(
                    estimate.first.rhs_sum,
                    estimate.samples_per_batch,
                    context.dense_rhs);
            const std::vector<double> second_rhs =
                ev::bp_streamed_normalized_values(
                    estimate.second.rhs_sum,
                    estimate.samples_per_batch,
                    context.dense_rhs);
            int maximum_difference_cell = 0;
            double maximum_batch_difference = 0.0;
            for (int q = 0; q < state.local_p.Nw * state.local_p.NL; ++q) {
                const double difference = std::abs(
                    first_rhs[static_cast<std::size_t>(q)]
                    - second_rhs[static_cast<std::size_t>(q)]);
                if (difference > maximum_batch_difference) {
                    maximum_batch_difference = difference;
                    maximum_difference_cell = q;
                }
            }
            const std::array<int, 2> maximum_difference_indices =
                ev::bp_reversible_velocity_indices(
                    maximum_difference_cell, state.local_p);
            const ev::BPStreamedProjectionKey& extreme_key =
                estimate.second.maximum_event_key;
            const int extreme_reverse_first =
                context.proposal.time_reverse_cell[
                    static_cast<std::size_t>(extreme_key.first)];
            const int extreme_reverse_second =
                context.proposal.time_reverse_cell[
                    static_cast<std::size_t>(extreme_key.second)];
            // Include both time-reversed incoming channels when evaluating this
            // proposal probability. This exposes the weighting of a rare large
            // operator contribution rather than treating its amplitude as bias.
            const double extreme_base_probability = (
                ev::bp_sampled_joint_proposal_probability(
                    context.proposal,
                    extreme_key.first,
                    extreme_key.second,
                    extreme_key.pair_phi)
                + ev::bp_sampled_joint_proposal_probability(
                    context.proposal,
                    extreme_reverse_first,
                    extreme_reverse_second,
                    extreme_key.pair_phi))
                / context.complement_probability;
            const auto extreme_projection = cache.record.find(extreme_key);
            const std::pair<double, double> extreme_activity =
                extreme_projection != cache.record.end()
                ? ev::bp_streamed_channel_activities(
                    shell_f,
                    extreme_key.first,
                    extreme_key.second,
                    extreme_projection->second.direct_species,
                    state.local_p.bp_reversible_entropy_floor)
                : std::pair<double, double>{0.0, 0.0};
            // The most restrictive negative component of the estimated RHS sets
            // the explicit positive-update interval. Sampling convergence and this
            // positivity condition answer different numerical questions.
            const double positivity_dt =
                ev::bp_reversible_positive_dt_limit(
                    shell_f,
                    estimate.rhs,
                    state.local_p.bp_collision_cfl);
            std::uint64_t production_draws =
                estimate.first.attempted_draws
                + estimate.second.attempted_draws;
            std::uint64_t retained_draws =
                estimate.first.retained_draws
                + estimate.second.retained_draws;
            std::uint64_t represented_channels =
                estimate.first.represented_channels
                + estimate.second.represented_channels;
            double represented_activity =
                estimate.first.represented_activity_sum
                + estimate.second.represented_activity_sum;
            double omitted_domain_activity =
                estimate.first.omitted_domain_activity_sum
                + estimate.second.omitted_domain_activity_sum;
            double omitted_projection_activity =
                estimate.first.omitted_projection_activity_sum
                + estimate.second.omitted_projection_activity_sum;
            for (const ev::BPStreamedBatch& batch : estimate.additional) {
                production_draws += batch.attempted_draws;
                retained_draws += batch.retained_draws;
                represented_channels += batch.represented_channels;
                represented_activity += batch.represented_activity_sum;
                omitted_domain_activity +=
                    batch.omitted_domain_activity_sum;
                omitted_projection_activity +=
                    batch.omitted_projection_activity_sum;
            }
            std::cout
                << "streamed=1\n"
                << "samples_per_batch=" << estimate.samples_per_batch << '\n'
                << "adaptive_rounds=" << estimate.adaptive_rounds << '\n'
                << "converged=" << (estimate.converged ? 1 : 0) << '\n'
                << "controller_draws="
                << estimate.controller_attempted_draws << '\n'
                << "production_draws="
                << production_draws << '\n'
                << "retained_collision_draws="
                << estimate.controller_retained_draws
                    + retained_draws << '\n'
                << "represented_channels="
                << represented_channels << '\n'
                << "represented_activity="
                << represented_activity << '\n'
                << "omitted_domain_activity="
                << omitted_domain_activity << '\n'
                << "omitted_projection_activity="
                << omitted_projection_activity << '\n'
                << "projection_cache_hits=" << cache.hits << '\n'
                << "projection_cache_misses=" << cache.misses << '\n'
                << "rhs_relative_error=" << estimate.error.rhs << '\n'
                << "rhs_gross_error=" << estimate.error.rhs_gross << '\n'
                << "heat_relative_error=" << estimate.error.heat << '\n'
                << "heat_gross_error=" << estimate.error.heat_gross << '\n'
                << "energy_invariant_error="
                << estimate.error.energy_invariant << '\n'
                << "maximum_abs_rhs=" << maximum_rhs << '\n'
                << "maximum_batch_difference="
                << maximum_batch_difference << '\n'
                << "maximum_difference_w_index="
                << maximum_difference_indices[0] << '\n'
                << "maximum_difference_L_index="
                << maximum_difference_indices[1] << '\n'
                << "maximum_difference_w="
                << state.w[static_cast<std::size_t>(
                    maximum_difference_indices[0])] << '\n'
                << "maximum_difference_eta="
                << geometry.eta.center[static_cast<std::size_t>(
                    maximum_difference_indices[1])] << '\n'
                << "first_maximum_event_rhs="
                << estimate.first.maximum_event_rhs << '\n'
                << "first_maximum_event_cell="
                << estimate.first.maximum_event_rhs_cell << '\n'
                << "first_maximum_event_pair="
                << estimate.first.maximum_event_key.first << ','
                << estimate.first.maximum_event_key.second << ','
                << estimate.first.maximum_event_key.pair_phi << ','
                << estimate.first.maximum_event_key.scatter_mu << ','
                << estimate.first.maximum_event_key.scatter_phi << ','
                << estimate.first.maximum_event_key.projection_variant << '\n'
                << "second_maximum_event_rhs="
                << estimate.second.maximum_event_rhs << '\n'
                << "second_maximum_event_cell="
                << estimate.second.maximum_event_rhs_cell << '\n'
                << "second_maximum_event_pair="
                << estimate.second.maximum_event_key.first << ','
                << estimate.second.maximum_event_key.second << ','
                << estimate.second.maximum_event_key.pair_phi << ','
                << estimate.second.maximum_event_key.scatter_mu << ','
                << estimate.second.maximum_event_key.scatter_phi << ','
                << estimate.second.maximum_event_key.projection_variant << '\n'
                << "extreme_base_pair_probability="
                << extreme_base_probability << '\n'
                << "extreme_incoming_f="
                << shell_f[static_cast<std::size_t>(extreme_key.first)]
                << ','
                << shell_f[static_cast<std::size_t>(extreme_key.second)]
                << '\n'
                << "extreme_forward_reverse_activity="
                << extreme_activity.first << ','
                << extreme_activity.second << '\n'
                << "positivity_dt=" << positivity_dt << '\n'
                << "wall_seconds=" << wall_seconds << '\n';
            return estimate.converged ? 0 : 1;
        }
        // ---- Stored-graph estimate for direct comparison ------------------
        const auto started = std::chrono::steady_clock::now();
        // The stored-graph branch reconstructs reactions from the same shell
        // and proposal. Comparing its errors/cost with streamed mode isolates
        // the storage/evaluation strategy at the chosen sampling controls.
        const ev::BPSampledShellGraph sampled =
            ev::bp_sampled_build_shell_graph(
                shell_f,
                radius,
                state.w,
                geometry.eta.center,
                geometry,
                state.local_p,
                state.local_p.bp_sampled_min_samples_per_batch,
                static_cast<std::uint64_t>(shell_index),
                &proposal);
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const std::vector<double> rhs = ev::bp_reversible_shell_rhs(
            shell_f, sampled.graph, state.local_p);
        double maximum_rhs = 0.0;
        for (double value : rhs) {
            maximum_rhs = std::max(maximum_rhs, std::abs(value));
        }
        std::cout << "dense_velocity_cells="
                  << sampled.dense_velocity_cells << '\n'
                  << "dense_reactions=" << sampled.dense_reactions << '\n'
                  << "complement_reactions="
                  << sampled.complement_reactions << '\n'
                  << "samples_per_batch="
                  << sampled.samples_per_batch << '\n'
                  << "adaptive_rounds=" << sampled.adaptive_rounds << '\n'
                  << "converged=" << (sampled.converged ? 1 : 0) << '\n'
                  << "ntc_fallback=" << (sampled.ntc_fallback ? 1 : 0)
                  << '\n'
                  << "complement_draws=" << sampled.complement_draws << '\n'
                  << "retained_collision_draws="
                  << sampled.retained_collision_draws << '\n'
                  << "rhs_relative_error="
                  << sampled.rhs_relative_error << '\n'
                  << "rhs_gross_error=" << sampled.rhs_gross_error << '\n'
                  << "heat_relative_error="
                  << sampled.heat_relative_error << '\n'
                  << "heat_gross_error=" << sampled.heat_gross_error << '\n'
                  << "energy_invariant_error="
                  << sampled.energy_invariant_error << '\n'
                  << "maximum_abs_rhs=" << maximum_rhs << '\n'
                  << "wall_seconds=" << wall_seconds << '\n';
        return sampled.converged ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
