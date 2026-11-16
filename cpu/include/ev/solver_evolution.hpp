#pragma once

// Time ordering for the spherical phase-grid evolution.
// At each accepted time, integrate f to obtain matter moments and solve the
// selected Poisson or Einstein constraints. Transport then advances f through
// the chosen frozen or midpoint field. Boltzmann--Poisson additionally applies
// its selected collision splitting and records each operator's separate budget.
// The state retains its actual accepted time, previous interval, unresolved inner
// carriers and any valid Hamiltonian modes. Output and restart occur at complete
// state boundaries so their profiles, clocks and distributions describe one epoch.

#include "ev/boltzmann_poisson.hpp"
#include "ev/hamiltonian_sldg_transport.hpp"
#include "ev/solver_setup.hpp"

namespace ev {

struct SolverVPTransportResult {
    std::vector<double> f;
    VlasovPoissonField field_used;
    SLDGStepDiagnostics sldg_diagnostics;
    bool has_sldg_diagnostics = false;
};

inline double solver_hidden_carrier_mass(const SolverState& state);

// A collision changes the within-cell velocity distribution. Its previous
// Hamiltonian polynomial representation must therefore be reconstructed before
// the next transport rather than reused as unchanged modal data.
inline void invalidate_solver_hamiltonian_modes_after_collision(
    SolverState& state) {
    state.hamiltonian_sldg_modes_valid = false;
    state.tracer_hamiltonian_sldg_modes_valid = false;
}

// Optional compression groups unresolved returning trajectories. Record
// changes in represented mass, boundary momentum, energy and absolute return-time
// moment so the effect of that representation remains measurable.
inline void compress_solver_delayed_inner_carrier(
    SolverState& state,
    DelayedInnerCarrier& carrier,
    double current_time,
    SLDGStepDiagnostics* diagnostics = nullptr) {
    if (!using_delayed_inner_return(state.local_p)
        || !(state.local_p.vp_inner_carrier_return_time_bin_width > 0.0)) {
        return;
    }
    const DelayedInnerCarrierCompressionLedger compression =
        compress_delayed_inner_carrier(
            carrier,
            state.kinetic_geometry,
            current_time,
            state.local_p);
    if (diagnostics == nullptr) {
        return;
    }
    diagnostics->carrier_compressions = mpi_allreduce_sum_ll(
        compression.applied ? 1 : 0, state.mpi);
    if (diagnostics->carrier_compressions == 0) {
        return;
    }
    diagnostics->carrier_compression_packets_before = mpi_allreduce_sum_ll(
        static_cast<long long>(compression.before.packet_count), state.mpi);
    diagnostics->carrier_compression_packets_after = mpi_allreduce_sum_ll(
        static_cast<long long>(compression.after.packet_count), state.mpi);
    diagnostics->carrier_packets_after =
        diagnostics->carrier_compression_packets_after;
    diagnostics->carrier_compression_mass_delta = mpi_allreduce_sum(
        compression.after.physical_mass
            - compression.before.physical_mass,
        state.mpi);
    diagnostics->carrier_compression_momentum_delta = mpi_allreduce_sum(
        compression.after.boundary_radial_momentum
            - compression.before.boundary_radial_momentum,
        state.mpi);
    diagnostics->carrier_compression_energy_delta = mpi_allreduce_sum(
        compression.after.boundary_kinetic_energy
            - compression.before.boundary_kinetic_energy,
        state.mpi);
    diagnostics->carrier_compression_return_time_moment_delta =
        mpi_allreduce_sum(
            compression.after.absolute_return_time_moment
                - compression.before.absolute_return_time_moment,
            state.mpi);
}

/// Dispatches the primary distribution through the selected VP transport.
// Conservative transports return their actual midpoint field and geometric
// budgets along with f. A passive VP component later uses this same selected
// field, preserving the total gravitational source.
inline SolverVPTransportResult advance_solver_vp_transport(
    SolverState& state,
    const std::vector<double>& source,
    const VlasovPoissonField& field,
    const Moments& moments,
    double dt,
    double t) {
    SolverVPTransportResult result;
    result.field_used = field;
    if (using_hamiltonian_sldg_transport(state.local_p)) {
        SLDGStepResult step =
            advect_vlasov_poisson_hamiltonian_sldg(
                source,
                state.x_r,
                state.r,
                state.w,
                state.L,
                field,
                moments,
                state.kinetic_geometry,
                dt,
                t,
                state.local_p,
                state.mpi,
                using_delayed_inner_return(state.local_p)
                    ? &state.inner_carrier
                    : nullptr,
                state.hamiltonian_sldg_modes_valid
                    ? &state.hamiltonian_sldg_modes
                    : nullptr);
        compress_solver_delayed_inner_carrier(
            state,
            state.inner_carrier,
            t + dt,
            &step.diagnostics);
        result.f = step.f;
        state.hamiltonian_sldg_modes =
            std::move(step.modal_coefficients);
        state.hamiltonian_sldg_modes_valid =
            step.has_modal_coefficients;
        result.field_used = step.midpoint_field;
        result.sldg_diagnostics = step.diagnostics;
        result.has_sldg_diagnostics = true;
    } else if (using_conservative_sldg_transport(state.local_p)) {
        SLDGStepResult step = advect_vlasov_poisson_conservative_sldg(
            source,
            state.x_r,
            state.r,
            state.w,
            state.L,
            field,
            moments,
            state.kinetic_geometry,
            dt,
                t,
                state.local_p,
                state.mpi,
                using_delayed_inner_return(state.local_p)
                    ? &state.inner_carrier
                    : nullptr);
        compress_solver_delayed_inner_carrier(
            state,
            state.inner_carrier,
            t + dt,
            &step.diagnostics);
        result.f = step.f;
        result.field_used = step.midpoint_field;
        result.sldg_diagnostics = step.diagnostics;
        result.has_sldg_diagnostics = true;
    } else if (state.global_p.integrator == "conservative_finite_volume") {
        result.f = advect_vlasov_poisson_finite_volume(
            source, state.r, state.w, state.L, field, dt, t, state.local_p);
    } else {
        result.f = advect_vlasov_poisson(
            source,
            state.x_r,
            state.r,
            state.w,
            state.L,
            field,
            dt,
            t,
            state.local_p);
    }
    return result;
}

/// Transports a passive distribution through the already selected midpoint field.
inline std::vector<double> advance_solver_passive_vp_transport(
    SolverState& state,
    const std::vector<double>& source,
    const VlasovPoissonField& field,
    double dt,
    double t) {
    if (using_hamiltonian_sldg_transport(state.local_p)) {
        std::vector<std::array<double, 5>> transported_modes;
        std::vector<double> transported =
            advect_vlasov_poisson_hamiltonian_sldg_frozen(
            source,
            state.x_r,
            state.r,
            state.w,
            state.L,
            field,
            state.kinetic_geometry,
            dt,
                state.local_p,
                using_delayed_inner_return(state.local_p)
                    ? &state.tracer_inner_carrier
                    : nullptr,
                state.tracer_hamiltonian_sldg_modes_valid
                    ? &state.tracer_hamiltonian_sldg_modes
                    : nullptr,
                &transported_modes).first;
        state.tracer_hamiltonian_sldg_modes = std::move(transported_modes);
        state.tracer_hamiltonian_sldg_modes_valid = true;
        compress_solver_delayed_inner_carrier(
            state, state.tracer_inner_carrier, t + dt);
        return transported;
    }
    if (using_conservative_sldg_transport(state.local_p)) {
        std::vector<double> transported =
            advect_vlasov_poisson_conservative_sldg_frozen(
            source,
            state.x_r,
            state.r,
            state.w,
            state.L,
            field,
            state.kinetic_geometry,
            dt,
            state.local_p,
            using_delayed_inner_return(state.local_p)
                ? &state.tracer_inner_carrier
                : nullptr).first;
        compress_solver_delayed_inner_carrier(
            state, state.tracer_inner_carrier, t + dt);
        return transported;
    }
    if (state.global_p.integrator == "conservative_finite_volume") {
        return advect_vlasov_poisson_finite_volume(
            source, state.r, state.w, state.L, field, dt, t, state.local_p);
    }
    return advect_vlasov_poisson(
        source,
        state.x_r,
        state.r,
        state.w,
        state.L,
        field,
        dt,
        t,
        state.local_p);
}

/// Selects the VP step from the dynamical/CFL and potential-variation limits.
inline double choose_solver_vp_timestep(
    SolverState& state,
    const Moments& moments,
    const VlasovPoissonField& field) {
    const double characteristic_dt = choose_dt_vlasov_poisson(
        state.f,
        state.r,
        state.w,
        state.L,
        field,
        state.radial_step,
        state.min_dw,
        state.local_p,
        state.mpi,
        &state.kinetic_geometry);
    VPPotentialVariationLimit variation =
        vp_potential_variation_limit(
            moments,
            field,
            state.r,
            state.t,
            state.local_p);
    const bool potential_macrostep =
        using_static_hamiltonian_potential_macrostep(state.local_p);
    const double step_limit = potential_macrostep
        ? std::min(
            state.local_p.dt_max,
            state.vp_sldg_transport_dt_cap)
        : characteristic_dt;
    double candidate = std::min(step_limit, variation.timestep);
    if (!potential_macrostep) {
        state.last_vp_potential_variation_timescale = variation.timescale;
        state.last_vp_potential_variation_dt_limit = variation.timestep;
        return candidate;
    }

    // The static field macrostep checks its potential clock also at a predicted
    // midpoint. Recompute the predicted field including any changing hidden mass
    // and shorten the candidate until the fresh variation limit is respected.
    const double hidden_mass = solver_hidden_carrier_mass(state);
    for (int correction = 0; correction < 3; ++correction) {
        if (!(candidate > 0.0) || !std::isfinite(candidate)) {
            break;
        }
        const SLDGMidpointPrediction predictor =
            predict_vlasov_poisson_midpoint_moments(
                moments,
                state.kinetic_geometry,
                0.5 * candidate,
                state.local_p,
                hidden_mass);
        const VlasovPoissonField midpoint_field =
            solve_vlasov_poisson_field(
                predictor.moments,
                state.r,
                state.t + 0.5 * candidate,
                state.global_p,
                predictor.hidden_mass_after);
        const VPPotentialVariationLimit midpoint_variation =
            vp_potential_variation_limit(
                predictor.moments,
                midpoint_field,
                state.r,
                state.t + 0.5 * candidate,
                state.local_p);
        variation.timescale = std::min(
            variation.timescale, midpoint_variation.timescale);
        variation.timestep = std::min(
            variation.timestep, midpoint_variation.timestep);
        const double corrected = std::min(candidate, variation.timestep);
        if (corrected >= candidate * (1.0 - 1.0e-10)) {
            candidate = corrected;
            break;
        }
        candidate = corrected;
    }
    state.last_vp_potential_variation_timescale = variation.timescale;
    state.last_vp_potential_variation_dt_limit = variation.timestep;
    return candidate;
}

inline void update_solver_sldg_transport_dt_cap(
    SolverState& state,
    double accepted_dt,
    const SolverVPTransportResult& transport) {
    if (!using_static_hamiltonian_potential_macrostep(state.local_p)
        || !transport.has_sldg_diagnostics
        || !(state.local_p.vp_sldg_initial_transport_dt > 0.0)) {
        return;
    }
    state.vp_sldg_transport_dt_cap = sldg_next_transport_dt_cap(
        accepted_dt, transport.sldg_diagnostics, state.local_p);
}

/// Evaluates the discrete phase-space mass used by transport and collision budgets.
// The complete represented mass includes both resolved f and delayed inner
// carriers. Their sum closes transport budgets across an unresolved inner passage.
double solver_distribution_mass(const SolverState& state) {
    const double visible_mass = distribution_mass(
        state.f,
        state.r,
        state.w,
        state.L,
        state.rq,
        state.wq,
        state.Lq,
        state.local_p,
        state.mpi);
    if (!using_delayed_inner_return(state.local_p)) {
        return visible_mass;
    }
    const double local_hidden = evaluate_delayed_inner_carrier_ledger(
        state.inner_carrier,
        state.kinetic_geometry,
        state.local_p).physical_mass;
    return visible_mass + mpi_allreduce_sum(local_hidden, state.mpi);
}

inline double solver_hidden_carrier_mass(const SolverState& state) {
    if (!using_delayed_inner_return(state.local_p)) {
        return 0.0;
    }
    const double local_hidden = evaluate_delayed_inner_carrier_ledger(
        state.inner_carrier,
        state.kinetic_geometry,
        state.local_p).physical_mass;
    return mpi_allreduce_sum(local_hidden, state.mpi);
}

inline long long solver_hidden_carrier_packet_count(
    const SolverState& state) {
    if (!using_delayed_inner_return(state.local_p)) {
        return 0;
    }
    return mpi_allreduce_sum_ll(
        static_cast<long long>(state.inner_carrier.packets.size()),
        state.mpi);
}

/// Evaluates collision invariants on the current distribution and quadrature.
DistributionCollisionInvariants solver_distribution_collision_invariants(const SolverState& state) {
    DistributionCollisionInvariants invariants = distribution_collision_invariants(
        state.f,
        state.r,
        state.w,
        state.L,
        state.rq,
        state.wq,
        state.Lq,
        state.local_p,
        state.mpi);
    invariants.mass += solver_hidden_carrier_mass(state);
    return invariants;
}

inline DistributionCollisionInvariants solver_distribution_collision_invariants(
    const SolverState& state,
    const std::vector<double>& distribution) {
    DistributionCollisionInvariants invariants =
        distribution_collision_invariants(
            distribution,
            state.r,
            state.w,
            state.L,
            state.rq,
            state.wq,
            state.Lq,
            state.local_p,
            state.mpi);
    invariants.mass += solver_hidden_carrier_mass(state);
    return invariants;
}

/// Writes explicit before/after collision-invariant budgets on the root rank.
// Measure before/after values on the same phase quadrature for each operator.
// Geometric remap, sampling, positivity and hidden-carrier diagnostics identify
// which approximation contributes to a measured mass or energy change.
void write_operator_mass_budget(
    SolverOutputStreams& streams,
    const SolverState& state,
    const char* operator_name,
    double operator_time,
    double operator_dt,
    const DistributionCollisionInvariants& before,
    const DistributionCollisionInvariants& after,
    const BPCollisionStepInfo* collision_info = nullptr,
    const SLDGStepDiagnostics* sldg_info = nullptr) {
    if (!is_root(state.mpi)) {
        return;
    }
    const double mass_delta = after.mass - before.mass;
    const double relative_mass_delta = (before.mass > 0.0) ? mass_delta / before.mass : 0.0;
    const double radial_momentum_delta = after.radial_momentum - before.radial_momentum;
    const double kinetic_energy_delta = after.kinetic_energy - before.kinetic_energy;
    const double relative_kinetic_energy_delta = (before.kinetic_energy > 0.0)
        ? kinetic_energy_delta / before.kinetic_energy
        : 0.0;
    const double remap_attempted =
        (collision_info != nullptr) ? collision_info->remap_attempted_event_density : 0.0;
    const double remap_skipped =
        (collision_info != nullptr) ? collision_info->remap_skipped_event_density : 0.0;
    const double remap_skipped_fraction =
        (remap_attempted > 0.0) ? remap_skipped / remap_attempted : 0.0;
    streams.operator_mass_budget
        << state.step << ',' << operator_time << ',' << operator_name << ',' << operator_dt << ','
        << before.mass << ',' << after.mass << ',' << mass_delta << ',' << relative_mass_delta << ','
        << before.radial_momentum << ',' << after.radial_momentum << ',' << radial_momentum_delta << ','
        << before.kinetic_energy << ',' << after.kinetic_energy << ',' << kinetic_energy_delta << ','
        << relative_kinetic_energy_delta << ','
        << remap_attempted << ',' << remap_skipped << ',' << remap_skipped_fraction << ','
        << ((collision_info != nullptr) ? collision_info->remap_skipped_w_event_density : 0.0) << ','
        << ((collision_info != nullptr) ? collision_info->remap_skipped_L_event_density : 0.0) << ','
        << ((collision_info != nullptr) ? collision_info->max_shell_remap_skipped_fraction : 0.0) << ','
        << ((collision_info != nullptr)
                ? collision_info->remap_skipped_domain_event_density
                : 0.0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->remap_skipped_projection_event_density
                : 0.0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->remap_skipped_inactive_event_density
                : 0.0)
        << ','
        << ((collision_info != nullptr) ? collision_info->maximum_projection_error : 0.0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_dense_reactions : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_complement_reactions : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_complement_draws : 0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->sampled_retained_collision_draws
                : 0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->sampled_ntc_fallback_shells
                : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_min_samples_per_batch : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_samples_per_batch : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_adaptive_rounds : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_unconverged_shells : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_inactive_shells : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_rhs_relative_error : 0.0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_rhs_gross_error : 0.0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_heat_relative_error : 0.0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_heat_gross_error : 0.0)
        << ','
        << ((collision_info != nullptr) ? collision_info->sampled_max_energy_invariant_error : 0.0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->minimum_mean_free_path
                : 0.0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->maximum_radial_cell_width_over_mean_free_path
                : 0.0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->nonlocal_collision_cells
                : 0)
        << ','
        << ((collision_info != nullptr) ? collision_info->reversible_reaction_updates : 0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->reversible_explicit_fast_path_fallbacks
                : 0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->reversible_implicit_reaction_updates
                : 0)
        << ','
        << ((collision_info != nullptr)
                ? collision_info->reversible_implicit_iterations
                : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->source_mass : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->deposited_mass : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->escaped_mass : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_mass_before : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_mass_after : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_captured_mass : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_released_mass : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_release_deposited_mass
                : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_mass_residual : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_packets_before : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_packets_after : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_packets_captured : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_packets_released : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->carrier_compressions : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_compression_packets_before
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_compression_packets_after
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_compression_mass_delta
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_compression_momentum_delta
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_compression_energy_delta
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->carrier_compression_return_time_moment_delta
                : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->quadrature_fallback_mass : 0.0)
        << ','
        << ((sldg_info != nullptr && sldg_info->source_mass > 0.0)
                ? sldg_info->quadrature_fallback_mass / sldg_info->source_mass
                : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->maximum_area_defect : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->maximum_map_defect : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->maximum_rejected_area_defect
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->maximum_rejected_map_defect
                : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->transported_subcells : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->traced_rectangles : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->rejected_rectangles : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->quadrature_fallback_subcells
                : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->maximum_source_work : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->largest_fallback_mass : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->largest_fallback_r : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->largest_fallback_w : 0.0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->largest_fallback_L : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->largest_fallback_refinement
                : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->largest_fallback_work : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->characteristic_trace_requests
                : 0)
        << ','
        << ((sldg_info != nullptr) ? sldg_info->characteristic_traces : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->reused_characteristic_points
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_mass_before
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_mass_after
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_escaped_mass
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_hidden_mass_before
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_hidden_mass_after
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_hidden_mass_change
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->midpoint_predictor_limited_faces
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_reconstructed_cells
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_limited_cells
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_projected_limited_cells
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_rank_deficient_cells
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_phase_degenerate_cells
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_eta_limited_cells
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_maximum_phase_gradient_fraction
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_reference_mass_fraction
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_residual_l1_fraction
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_reference_active_planes
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_minimum_positive_reference_scale
                : 1.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_positive_reference_scaled_planes
                : 0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_positivity_redistributed_mass
                : 0.0)
        << ','
        << ((sldg_info != nullptr)
                ? sldg_info->hamiltonian_positivity_redistributed_fraction
                : 0.0)
        << '\n';
    flush_output_checked(streams.operator_mass_budget, "operator_mass_budget.csv");
}

/// Advances one multirate Strang macro-step.
///
/// `bp_collision_apply_every` denotes a nominal number of Vlasov CFL steps,
/// not a number of calls to the collision routine.  The macro interval is
/// fixed from the CFL estimate at its beginning, collision is applied for one
/// half interval on each side, and the Vlasov flow is subcycled with freshly
/// solved fields until the interval is reached.  Restarts are left to the
/// caller after this function returns, so a checkpoint cannot contain only
/// one of the two collision half-steps.
// One collision half-step precedes a fixed physical macro interval, and the
// second half follows it. The intervening Vlasov substeps obtain fresh fields
// and local limits while consuming that exact common interval.
void advance_vp_multirate_strang_macro(
    SolverState& state,
    SolverOutputStreams& streams,
    const VlasovPoissonField& initial_field,
    const Moments& initial_moments) {
    const int cadence = std::max(1, state.global_p.bp_collision_apply_every);
    double seed_dt =
        choose_solver_vp_timestep(state, initial_moments, initial_field);
    seed_dt = std::min(seed_dt, state.global_p.t_end - state.t);
    if (!(seed_dt > 0.0) || !std::isfinite(seed_dt)) {
        throw std::runtime_error("Non-positive or non-finite multirate seed timestep.");
    }

    // Fix the macro endpoint from the starting field clock and requested nominal
    // cadence. Checkpoints are written by the caller only after both collision halves.
    const double macro_start = state.t;
    const int step_start = state.step;
    const double macro_end = std::min(
        state.global_p.t_end,
        macro_start + static_cast<double>(cadence) * seed_dt);
    const double macro_dt = macro_end - macro_start;

    // Cadence is handled here.  Force both collision calls to execute so they
    // form one actual Strang pair around the Vlasov subcycling interval.
    Params collision_p = state.local_p;
    collision_p.bp_collision_apply_every = 1;
    std::vector<double> collision_delta(state.f.size(), 0.0);
    std::vector<double> vlasov_delta(state.f.size(), 0.0);

    const std::vector<double> before_collision = state.f;
    const DistributionCollisionInvariants collision_before =
        solver_distribution_collision_invariants(state);
    const BPCollisionStepInfo collision_info_1 = apply_boltzmann_poisson_collision_step(
        state.f,
        state.has_tracer ? &state.tracer_f : nullptr,
        state.r,
        state.w,
        state.L,
        state.wq,
        state.Lq,
        0.5 * macro_dt,
        collision_p,
        state.mpi,
        &state.bp_streamed_collision_workspace);
    invalidate_solver_hamiltonian_modes_after_collision(state);
    const DistributionCollisionInvariants collision_after =
        solver_distribution_collision_invariants(state);
    write_operator_mass_budget(
        streams,
        state,
        "collision_macro_half_1",
        macro_start,
        0.5 * macro_dt,
        collision_before,
        collision_after,
        &collision_info_1);
    accumulate_operator_delta(collision_delta, before_collision, state.f, state.local_p);

    while (state.t < macro_end) {
        const Moments moments = compute_global_vp_moments(
            state.f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi);
        const VlasovPoissonField field =
            solve_vlasov_poisson_field(
                moments,
                state.r,
                state.t,
                state.global_p,
                solver_hidden_carrier_mass(state));
        const double requested_dt =
            choose_solver_vp_timestep(state, moments, field);
        const double remaining_macro_time = macro_end - state.t;
        const bool clipped_to_macro_boundary =
            requested_dt > remaining_macro_time;
        const double dt = std::min(requested_dt, remaining_macro_time);
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            throw std::runtime_error("Non-positive or non-finite Vlasov macro-substep.");
        }

        state.last_escape = estimate_escapes_vlasov_poisson(
            state.f,
            state.r,
            state.w,
            state.L,
            state.rq,
            state.wq,
            state.Lq,
            field,
            dt,
            state.local_p,
            state.mpi);
        state.cumulative_escape.outer_mass += state.last_escape.outer_mass;
        state.cumulative_escape.w_low_mass += state.last_escape.w_low_mass;
        state.cumulative_escape.w_high_mass += state.last_escape.w_high_mass;

        const std::vector<double> before_vlasov = state.f;
        const DistributionCollisionInvariants vlasov_before =
            solver_distribution_collision_invariants(state);
        SolverVPTransportResult transport =
            advance_solver_vp_transport(
                state, state.f, field, moments, dt, state.t);
        // A short remainder only aligns the Vlasov flow with the second
        // collision half-step.  It carries no information about the largest
        // acceptable transport interval, so leave the learned cap unchanged.
        if (!clipped_to_macro_boundary) {
            update_solver_sldg_transport_dt_cap(state, dt, transport);
        }
        std::vector<double> after_vlasov = std::move(transport.f);
        const DistributionCollisionInvariants vlasov_after =
            solver_distribution_collision_invariants(
                state, after_vlasov);
        write_operator_mass_budget(
            streams,
            state,
            "vlasov",
            state.t,
            dt,
            vlasov_before,
            vlasov_after,
            nullptr,
            transport.has_sldg_diagnostics
                ? &transport.sldg_diagnostics
                : nullptr);
        accumulate_operator_delta(vlasov_delta, before_vlasov, after_vlasov, state.local_p);
        state.f = std::move(after_vlasov);

        if (state.has_tracer) {
            state.tracer_f = advance_solver_passive_vp_transport(
                state, state.tracer_f, transport.field_used, dt, state.t);
        }

        state.t += dt;
        state.last_dt = dt;
        ++state.step;
        const bool adapted_w = adapt_w_grid_if_needed(
            state.f,
            state.w,
            state.wq,
            state.min_dw,
            state.r,
            state.L,
            state.rq,
            state.Lq,
            state.local_p,
            state.step,
            state.t,
            is_root(state.mpi) ? &streams.w_grid_history : nullptr,
            state.mpi);
        if (adapted_w) {
            invalidate_solver_hamiltonian_modes_after_collision(state);
        }
        state.global_p.w_max = state.local_p.w_max;
    }

    const std::vector<double> before_final_collision = state.f;
    const DistributionCollisionInvariants final_collision_before =
        solver_distribution_collision_invariants(state);
    const BPCollisionStepInfo collision_info_2 = apply_boltzmann_poisson_collision_step(
        state.f,
        state.has_tracer ? &state.tracer_f : nullptr,
        state.r,
        state.w,
        state.L,
        state.wq,
        state.Lq,
        0.5 * macro_dt,
        collision_p,
        state.mpi,
        &state.bp_streamed_collision_workspace);
    invalidate_solver_hamiltonian_modes_after_collision(state);
    const DistributionCollisionInvariants final_collision_after =
        solver_distribution_collision_invariants(state);
    write_operator_mass_budget(
        streams,
        state,
        "collision_macro_half_2",
        macro_end,
        0.5 * macro_dt,
        final_collision_before,
        final_collision_after,
        &collision_info_2);
    accumulate_operator_delta(
        collision_delta, before_final_collision, state.f, state.local_p);

    state.last_operator_diag = compute_operator_diagnostics(
        vlasov_delta,
        collision_delta,
        macro_dt,
        state.r,
        state.wq,
        state.L,
        state.rq,
        state.Lq,
        state.local_p,
        state.mpi);

    const int log_interval = std::max(1, 5 * state.global_p.save_every);
    if (is_root(state.mpi)
        && state.step / log_interval > step_start / log_interval) {
        std::cout << "macro_step=[" << step_start << ',' << state.step << "]"
                  << " t=[" << std::setprecision(8) << macro_start << ',' << state.t << ']'
                  << " dt_macro=" << std::scientific << std::setprecision(3) << macro_dt
                  << " vlasov_substeps=" << (state.step - step_start)
                  << std::defaultfloat << '\n';
    }
}

/// Runs the weak-field VP timestep loop with the same semi-Lagrangian algorithm as the EV solver.
// Profiles and history at the top of the loop describe the current completed
// state. The subsequently selected interval is clipped to the final physical
// time, then transport and collision operators build the next complete state.
void run_vlasov_poisson_evolution_loop(SolverState& state, SolverOutputStreams& streams) {
    int last_profile_step = state.step - std::max(1, state.global_p.save_every);
    int last_phase_step = state.step - std::max(1, state.global_p.phase_every);
    while (state.step <= state.global_p.max_steps) {
        const Moments moments = compute_global_vp_moments(
            state.f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi);
        const VlasovPoissonField field = solve_vlasov_poisson_field(
            moments,
            state.r,
            state.t,
            state.global_p,
            solver_hidden_carrier_mass(state));
        const Metric diagnostic_metric = vp_as_diagnostic_metric(field, state.global_p);
        const Moments tracer_moments = state.has_tracer
            ? compute_global_vp_moments(
                state.tracer_f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi)
            : Moments{};
        const double matter_mass = field.matter_mass;
        const double max_rho = *std::max_element(moments.rho.begin(), moments.rho.end());
        const double hidden_carrier_mass = field.hidden_carrier_mass;
        const double hidden_carrier_density = hidden_carrier_mass > 0.0
            ? 3.0 * hidden_carrier_mass
                / (4.0 * PI * std::pow(state.global_p.r_min, 3))
            : 0.0;
        const long long hidden_carrier_packets =
            solver_hidden_carrier_packet_count(state);

        if (is_root(state.mpi)
            && (state.step - last_profile_step >= std::max(1, state.global_p.save_every)
                || state.t >= state.global_p.t_end)) {
            write_profile_rows(streams.profiles, state.t, state.step, state.r, moments, diagnostic_metric, state.global_p);
            flush_output_checked(streams.profiles, "profiles.csv");
            last_profile_step = state.step;
            if (state.has_tracer) {
                write_moment_profile_rows(
                    streams.tracer_profiles, state.t, state.step, state.r, tracer_moments, state.global_p);
                flush_output_checked(streams.tracer_profiles, "tracer_profiles.csv");
            }
        }
        if (state.global_p.phase_every > 0
            && (state.step - last_phase_step >= state.global_p.phase_every
                || state.t >= state.global_p.t_end)) {
            std::vector<double> g = phase_projection_global(state.f, state.L, state.Lq, state.local_p, state.mpi);
            if (is_root(state.mpi)) {
                write_phase_snapshot_rows(streams.phase_snapshots, state.t, state.step, state.r, state.w, g, state.global_p);
                flush_output_checked(streams.phase_snapshots, "phase_space_snapshots.csv");
            }
            if (state.has_tracer) {
                std::vector<double> tracer_g =
                    phase_projection_global(state.tracer_f, state.L, state.Lq, state.local_p, state.mpi);
                if (is_root(state.mpi)) {
                    write_phase_snapshot_rows(
                        streams.tracer_phase_snapshots, state.t, state.step, state.r, state.w, tracer_g, state.global_p);
                    flush_output_checked(streams.tracer_phase_snapshots, "tracer_phase_space_snapshots.csv");
                }
            }
            last_phase_step = state.step;
        }
        if (is_root(state.mpi)) {
            streams.history << state.step << ',' << state.t << ',' << state.last_dt << ',' << matter_mass << ','
                            << max_rho << ',' << diagnostic_metric.min_gap << ','
                            << state.last_escape.outer_mass << ',' << state.last_escape.w_low_mass << ','
                            << state.last_escape.w_high_mass << ','
                            << state.cumulative_escape.outer_mass << ',' << state.cumulative_escape.w_low_mass << ','
                            << state.cumulative_escape.w_high_mass << ','
                            << state.last_escape.outer_cells << ',' << state.last_escape.w_low_cells << ','
                            << state.last_escape.w_high_cells << ','
                            << state.global_p.w_max << ',' << state.min_dw << ','
                            << hidden_carrier_mass << ','
                            << hidden_carrier_density << ','
                            << hidden_carrier_packets;
            write_operator_diagnostics_csv(streams.history, state.last_operator_diag);
            const VPOrbitRegularizationDiagnostics orbit_diag =
                vp_orbit_regularization_diagnostics();
            streams.history << ',' << orbit_diag.retries
                            << ',' << orbit_diag.accepted
                            << ',' << orbit_diag.unresolved
                            << ',' << orbit_diag.wall_retries
                            << ',' << orbit_diag.wall_accepted
                            << ',' << orbit_diag.wall_unresolved
                            << ',' << orbit_diag.maximum_ordinary_energy_error
                            << ',' << orbit_diag.maximum_final_energy_error
                            << ','
                            << state.last_vp_potential_variation_timescale
                            << ','
                            << state.last_vp_potential_variation_dt_limit;
            streams.history << '\n';
            flush_output_checked(streams.history, "history.csv");
        }

        if (state.t >= state.global_p.t_end) {
            break;
        }
        if (state.step >= state.global_p.max_steps) {
            if (is_root(state.mpi)) {
                std::cout << "Stopping: reached max_steps=" << state.global_p.max_steps << ".\n";
            }
            break;
        }

        // A cadence greater than one changes the Strang macro interval through the
        // dedicated routine. Ordinary single-step splitting continues below.
        const bool multirate_strang =
            state.global_p.bp_collision_splitting == "strang"
            && using_bp_collision_operator(state.local_p)
            && state.local_p.bp_collision_strength != 0.0
            && state.global_p.bp_collision_apply_every > 1;
        if (multirate_strang) {
            const int step_before_macro = state.step;
            advance_vp_multirate_strang_macro(
                state, streams, field, moments);
            if (state.global_p.restart_every > 0
                && (state.step / state.global_p.restart_every
                        > step_before_macro / state.global_p.restart_every
                    || state.t >= state.global_p.t_end)) {
                write_restart_state_distributed(
                    state.restart_path,
                    state.restart_tmp_dir,
                    state.global_p,
                    state.local_p,
                    state.decomp,
                    state.mpi,
                    state.t,
                    state.step,
                    state.f,
                    state.has_tracer ? &state.tracer_f : nullptr,
                    using_delayed_inner_return(state.local_p)
                        ? &state.inner_carrier
                        : nullptr,
                    using_delayed_inner_return(state.local_p) && state.has_tracer
                        ? &state.tracer_inner_carrier
                        : nullptr,
                    state.hamiltonian_sldg_modes_valid
                        ? &state.hamiltonian_sldg_modes
                        : nullptr,
                    state.tracer_hamiltonian_sldg_modes_valid
                        ? &state.tracer_hamiltonian_sldg_modes
                        : nullptr);
            }
            continue;
        }

        double dt = choose_solver_vp_timestep(state, moments, field);
        if (state.t + dt > state.global_p.t_end) {
            dt = state.global_p.t_end - state.t;
        }
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            throw std::runtime_error("Non-positive or non-finite timestep.");
        }

        state.last_escape = estimate_escapes_vlasov_poisson(
            state.f,
            state.r,
            state.w,
            state.L,
            state.rq,
            state.wq,
            state.Lq,
            field,
            dt,
            state.local_p,
            state.mpi);
        state.cumulative_escape.outer_mass += state.last_escape.outer_mass;
        state.cumulative_escape.w_low_mass += state.last_escape.w_low_mass;
        state.cumulative_escape.w_high_mass += state.last_escape.w_high_mass;

        // For Strang splitting apply half a collision before transport and half
        // after it. The alternative after-Vlasov ordering applies one full collision
        // interval after transport; both record their actual operator time and duration.
        BPCollisionStepInfo bp_collision;
        const bool strang_collision = state.global_p.bp_collision_splitting == "strang";
        const bool collision_active = using_bp_collision_operator(state.local_p)
            && state.local_p.bp_collision_strength != 0.0;
        std::vector<double> collision_delta(state.f.size(), 0.0);
        if (strang_collision) {
            const std::vector<double> before_collision = collision_active ? state.f : std::vector<double>{};
            const DistributionCollisionInvariants invariants_before_collision =
                solver_distribution_collision_invariants(state);
            const BPCollisionStepInfo half_collision = apply_boltzmann_poisson_collision_step(
                state.f,
                state.has_tracer ? &state.tracer_f : nullptr,
                state.r,
                state.w,
                state.L,
                state.wq,
                state.Lq,
                0.5 * dt,
                state.local_p,
                state.mpi,
                &state.bp_streamed_collision_workspace);
            if (collision_active) {
                invalidate_solver_hamiltonian_modes_after_collision(state);
            }
            const DistributionCollisionInvariants invariants_after_collision =
                solver_distribution_collision_invariants(state);
            write_operator_mass_budget(
                streams,
                state,
                "collision_half_1",
                state.t,
                0.5 * dt,
                invariants_before_collision,
                invariants_after_collision,
                &half_collision);
            merge_bp_collision_step_info(bp_collision, half_collision);
            if (collision_active) {
                accumulate_operator_delta(collision_delta, before_collision, state.f, state.local_p);
            }
        }

        const std::vector<double> before_vlasov = state.f;
        const DistributionCollisionInvariants invariants_before_vlasov =
            solver_distribution_collision_invariants(state);
        SolverVPTransportResult transport =
            advance_solver_vp_transport(
                state, state.f, field, moments, dt, state.t);
        update_solver_sldg_transport_dt_cap(state, dt, transport);
        std::vector<double> after_vlasov = std::move(transport.f);
        const DistributionCollisionInvariants invariants_after_vlasov =
            solver_distribution_collision_invariants(
                state, after_vlasov);
        write_operator_mass_budget(
            streams,
            state,
            "vlasov",
            state.t,
            dt,
            invariants_before_vlasov,
            invariants_after_vlasov,
            nullptr,
            transport.has_sldg_diagnostics
                ? &transport.sldg_diagnostics
                : nullptr);
        std::vector<double> vlasov_delta = operator_delta(before_vlasov, after_vlasov, state.local_p);
        state.f = std::move(after_vlasov);
        if (state.has_tracer) {
            state.tracer_f = advance_solver_passive_vp_transport(
                state, state.tracer_f, transport.field_used, dt, state.t);
        }
        const std::vector<double> before_final_collision = collision_active ? state.f : std::vector<double>{};
        const DistributionCollisionInvariants invariants_before_final_collision =
            solver_distribution_collision_invariants(state);
        const BPCollisionStepInfo final_collision = apply_boltzmann_poisson_collision_step(
            state.f,
            state.has_tracer ? &state.tracer_f : nullptr,
            state.r,
            state.w,
            state.L,
            state.wq,
            state.Lq,
            strang_collision ? 0.5 * dt : dt,
            state.local_p,
            state.mpi,
            &state.bp_streamed_collision_workspace);
        if (collision_active) {
            invalidate_solver_hamiltonian_modes_after_collision(state);
        }
        const DistributionCollisionInvariants invariants_after_final_collision =
            solver_distribution_collision_invariants(state);
        write_operator_mass_budget(
            streams,
            state,
            strang_collision ? "collision_half_2" : "collision_full",
            strang_collision ? state.t + dt : state.t,
            strang_collision ? 0.5 * dt : dt,
            invariants_before_final_collision,
            invariants_after_final_collision,
            &final_collision);
        merge_bp_collision_step_info(bp_collision, final_collision);
        if (collision_active) {
            accumulate_operator_delta(collision_delta, before_final_collision, state.f, state.local_p);
        }
        state.last_operator_diag = compute_operator_diagnostics(
            vlasov_delta,
            collision_delta,
            dt,
            state.r,
            state.wq,
            state.L,
            state.rq,
            state.Lq,
            state.local_p,
            state.mpi);
        state.t += dt;
        state.last_dt = dt;
        ++state.step;

        const bool adapted_w = adapt_w_grid_if_needed(
            state.f,
            state.w,
            state.wq,
            state.min_dw,
            state.r,
            state.L,
            state.rq,
            state.Lq,
            state.local_p,
            state.step,
            state.t,
            is_root(state.mpi) ? &streams.w_grid_history : nullptr,
            state.mpi);
        if (adapted_w) {
            invalidate_solver_hamiltonian_modes_after_collision(state);
        }
        state.global_p.w_max = state.local_p.w_max;

        if (state.global_p.restart_every > 0 && state.step % state.global_p.restart_every == 0) {
            write_restart_state_distributed(
                state.restart_path,
                state.restart_tmp_dir,
                state.global_p,
                state.local_p,
                state.decomp,
                state.mpi,
                state.t,
                state.step,
                state.f,
                state.has_tracer ? &state.tracer_f : nullptr,
                using_delayed_inner_return(state.local_p)
                    ? &state.inner_carrier
                    : nullptr,
                using_delayed_inner_return(state.local_p) && state.has_tracer
                    ? &state.tracer_inner_carrier
                    : nullptr,
                state.hamiltonian_sldg_modes_valid
                    ? &state.hamiltonian_sldg_modes
                    : nullptr,
                state.tracer_hamiltonian_sldg_modes_valid
                    ? &state.tracer_hamiltonian_sldg_modes
                    : nullptr);
        }

        if (is_root(state.mpi) && state.step % std::max(1, 5 * state.global_p.save_every) == 0) {
            std::cout << "step=" << std::setw(6) << state.step
                      << " t=" << std::setw(11) << std::setprecision(6) << state.t
                      << " dt=" << std::scientific << std::setprecision(3) << dt
                      << " mass=" << matter_mass
                      << " max_rho=" << max_rho
                      << " H=" << field.background.H;
            if (bp_collision.substeps > 0) {
                std::cout << " bp_substeps=" << bp_collision.substeps
                          << " bp_explicit=" << bp_collision.explicit_substeps
                          << " bp_sts_cycles=" << bp_collision.sts_cycles
                          << " maxDrr=" << bp_collision.max_D_rr
                          << " maxDrL=" << bp_collision.max_abs_D_rL
                          << " maxDLL=" << bp_collision.max_D_LL;
                if (bp_collision.collision_time_estimate > 0.0) {
                    std::cout << " tcoll=" << bp_collision.collision_time_estimate
                              << " dtcoeff=" << bp_collision.coeff_refresh_dt;
                }
            }
            std::cout << std::defaultfloat << "\n";
        }
    }
}

/// Runs the main timestep loop: moments, metric, diagnostics, advection, restart, and stopping checks.
// Dispatch Newtonian gravity to its own transport/collision loop. The
// Einstein--Vlasov grid path below reconstructs its polar metric before a
// frozen-metric pullback and stops at the configured compactness threshold.
void run_evolution_loop(SolverState& state, SolverOutputStreams& streams) {
    if (using_vlasov_poisson(state.global_p)) {
        run_vlasov_poisson_evolution_loop(state, streams);
        return;
    }
    while (state.step <= state.global_p.max_steps) {
        const Moments moments = compute_global_moments(
            state.f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi);
        const Metric metric = solve_metric(moments, state.r, state.global_p);
        const double matter_mass = metric.m.back() - state.global_p.central_mass;
        const double max_rho = *std::max_element(moments.rho.begin(), moments.rho.end());

        if (is_root(state.mpi)
            && (state.step % std::max(1, state.global_p.save_every) == 0 || state.t >= state.global_p.t_end)) {
            write_profile_rows(streams.profiles, state.t, state.step, state.r, moments, metric, state.global_p);
            flush_output_checked(streams.profiles, "profiles.csv");
        }
        if (state.global_p.phase_every > 0
            && (state.step % state.global_p.phase_every == 0 || state.t >= state.global_p.t_end)) {
            std::vector<double> g = phase_projection_global(state.f, state.L, state.Lq, state.local_p, state.mpi);
            if (is_root(state.mpi)) {
                write_phase_snapshot_rows(streams.phase_snapshots, state.t, state.step, state.r, state.w, g, state.global_p);
                flush_output_checked(streams.phase_snapshots, "phase_space_snapshots.csv");
            }
            if (state.has_tracer) {
                std::vector<double> tracer_g =
                    phase_projection_global(state.tracer_f, state.L, state.Lq, state.local_p, state.mpi);
                if (is_root(state.mpi)) {
                    write_phase_snapshot_rows(
                        streams.tracer_phase_snapshots, state.t, state.step, state.r, state.w, tracer_g, state.global_p);
                    flush_output_checked(streams.tracer_phase_snapshots, "tracer_phase_space_snapshots.csv");
                }
            }
        }
        if (is_root(state.mpi)) {
            streams.history << state.step << ',' << state.t << ',' << state.last_dt << ',' << matter_mass << ','
                            << max_rho << ',' << metric.min_gap << ','
                            << state.last_escape.outer_mass << ',' << state.last_escape.w_low_mass << ','
                            << state.last_escape.w_high_mass << ','
                            << state.cumulative_escape.outer_mass << ',' << state.cumulative_escape.w_low_mass << ','
                            << state.cumulative_escape.w_high_mass << ','
                            << state.last_escape.outer_cells << ',' << state.last_escape.w_low_cells << ','
                            << state.last_escape.w_high_cells << ','
                            << state.global_p.w_max << ',' << state.min_dw
                            << ",0,0,0";
            write_operator_diagnostics_csv(streams.history, state.last_operator_diag);
            const VPOrbitRegularizationDiagnostics orbit_diag =
                vp_orbit_regularization_diagnostics();
            streams.history << ',' << orbit_diag.retries
                            << ',' << orbit_diag.accepted
                            << ',' << orbit_diag.unresolved
                            << ',' << orbit_diag.wall_retries
                            << ',' << orbit_diag.wall_accepted
                            << ',' << orbit_diag.wall_unresolved
                            << ',' << orbit_diag.maximum_ordinary_energy_error
                            << ',' << orbit_diag.maximum_final_energy_error
                            << ','
                            << state.last_vp_potential_variation_timescale
                            << ','
                            << state.last_vp_potential_variation_dt_limit;
            streams.history << '\n';
            flush_output_checked(streams.history, "history.csv");
        }

        if (state.t >= state.global_p.t_end) {
            break;
        }
        // A small polar-areal compactness gap reaches the declared coordinate-domain
        // stop. Save this actual state and let the caller publish its final restart.
        if (metric.min_gap <= state.global_p.gap_stop) {
            if (is_root(state.mpi)) {
                std::cout << "Stopping: compactness gap reached " << metric.min_gap
                          << " at t=" << state.t << "\n";
            }
            break;
        }
        if (state.step >= state.global_p.max_steps) {
            if (is_root(state.mpi)) {
                std::cout << "Stopping: reached max_steps=" << state.global_p.max_steps << ".\n";
            }
            break;
        }

        double dt = choose_dt(
            state.f,
            state.r,
            state.w,
            state.L,
            metric,
            state.radial_step,
            state.min_dw,
            state.local_p,
            state.mpi);
        if (state.t + dt > state.global_p.t_end) {
            dt = state.global_p.t_end - state.t;
        }
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            throw std::runtime_error("Non-positive or non-finite timestep.");
        }

        state.last_escape = estimate_escapes(
            state.f,
            state.r,
            state.w,
            state.L,
            state.rq,
            state.wq,
            state.Lq,
            metric,
            dt,
            state.local_p,
            state.mpi);
        state.cumulative_escape.outer_mass += state.last_escape.outer_mass;
        state.cumulative_escape.w_low_mass += state.last_escape.w_low_mass;
        state.cumulative_escape.w_high_mass += state.last_escape.w_high_mass;

        const std::vector<double> before_vlasov = state.f;
        std::vector<double> after_vlasov = advect(state.f, state.x_r, state.r, state.w, state.L, metric, dt, state.local_p);
        const std::vector<double> vlasov_delta = operator_delta(before_vlasov, after_vlasov, state.local_p);
        const std::vector<double> collision_delta(state.f.size(), 0.0);
        state.last_operator_diag = compute_operator_diagnostics(
            vlasov_delta,
            collision_delta,
            dt,
            state.r,
            state.wq,
            state.L,
            state.rq,
            state.Lq,
            state.local_p,
            state.mpi);
        state.f = std::move(after_vlasov);
        state.t += dt;
        state.last_dt = dt;
        ++state.step;

        adapt_w_grid_if_needed(
            state.f,
            state.w,
            state.wq,
            state.min_dw,
            state.r,
            state.L,
            state.rq,
            state.Lq,
            state.local_p,
            state.step,
            state.t,
            is_root(state.mpi) ? &streams.w_grid_history : nullptr,
            state.mpi);
        state.global_p.w_max = state.local_p.w_max;

        if (state.global_p.restart_every > 0 && state.step % state.global_p.restart_every == 0) {
            write_restart_state_distributed(
                state.restart_path,
                state.restart_tmp_dir,
                state.global_p,
                state.local_p,
                state.decomp,
                state.mpi,
                state.t,
                state.step,
                state.f,
                state.has_tracer ? &state.tracer_f : nullptr,
                using_delayed_inner_return(state.local_p)
                    ? &state.inner_carrier
                    : nullptr,
                using_delayed_inner_return(state.local_p) && state.has_tracer
                    ? &state.tracer_inner_carrier
                    : nullptr,
                state.hamiltonian_sldg_modes_valid
                    ? &state.hamiltonian_sldg_modes
                    : nullptr,
                state.tracer_hamiltonian_sldg_modes_valid
                    ? &state.tracer_hamiltonian_sldg_modes
                    : nullptr);
        }

        if (is_root(state.mpi) && state.step % std::max(1, 5 * state.global_p.save_every) == 0) {
            std::cout << "step=" << std::setw(6) << state.step
                      << " t=" << std::setw(11) << std::setprecision(6) << state.t
                      << " dt=" << std::scientific << std::setprecision(3) << dt
                      << " mass=" << matter_mass
                      << " max_rho=" << max_rho
                      << " min_gap=" << metric.min_gap << std::defaultfloat << "\n";
        }
    }
}

/// Writes the final restart and final phase-space projection after the evolution loop ends.
// Publish the completed full state even for a normal step/time/domain stop.
// The final phase projection derives from the same f and MPI angular ownership.
void write_final_outputs(const SolverState& state) {
    write_restart_state_distributed(
        state.restart_path,
        state.restart_tmp_dir,
        state.global_p,
        state.local_p,
        state.decomp,
        state.mpi,
        state.t,
        state.step,
        state.f,
        state.has_tracer ? &state.tracer_f : nullptr,
        using_delayed_inner_return(state.local_p)
            ? &state.inner_carrier
            : nullptr,
        using_delayed_inner_return(state.local_p) && state.has_tracer
            ? &state.tracer_inner_carrier
            : nullptr,
        state.hamiltonian_sldg_modes_valid
            ? &state.hamiltonian_sldg_modes
            : nullptr,
        state.tracer_hamiltonian_sldg_modes_valid
            ? &state.tracer_hamiltonian_sldg_modes
            : nullptr);
    {
        std::vector<double> g = phase_projection_global(state.f, state.L, state.Lq, state.local_p, state.mpi);
        if (is_root(state.mpi)) {
            write_phase_space(state.outdir / "phase_space_final.csv", state.r, state.w, g, state.global_p);
        }
    }
    if (state.has_tracer) {
        std::vector<double> g = phase_projection_global(state.tracer_f, state.L, state.Lq, state.local_p, state.mpi);
        if (is_root(state.mpi)) {
            write_phase_space(state.outdir / "tracer_phase_space_final.csv", state.r, state.w, g, state.global_p);
        }
    }
    if (is_root(state.mpi)) {
        std::cout << "Wrote C++ solver data to " << std::filesystem::absolute(state.outdir) << "\n";
    }
}

}  // namespace ev
